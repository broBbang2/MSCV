"""아이폰 firstCV 앱이 저장한 프레임과 성능 기록을 가져와 학습 데이터에 추가한다.

앱은 켜져 있는 동안 0.5초마다 원본 프레임(박스 없는 사진)을 저장한다. 이 스크립트는
  1. 앱 Documents 폴더를 통째로 가져와서
  2. 새 프레임은 data/raw/에, 성능 기록(csv)은 data/app_logs/에 넣고
  3. 가져간 프레임 목록(pulled.txt)을 앱에 넣어 둔다. 앱을 다음에 켤 때 그 프레임들을 지운다
     (devicectl로는 기기 파일을 지울 수 없어서 앱이 대신 지운다)
  4. 04_auto_label.py를 다시 돌려 data/datasets/mine/에 반영한다. 학습은 하지 않는다.

앱 켤 때마다 저장되는 묶음(app날짜-시각)이 영상 하나처럼 취급된다.

실행 예:
    uv run scripts/08_pull_app.py                # 연결된 아이폰에서 가져와 데이터셋까지 반영
    uv run scripts/08_pull_app.py --no-label     # 가져오기만
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "data" / "app_logs"
BUNDLE_ID = "dev.breeadee.firstcv"


def devicectl(*args: str, tries: int = 3) -> subprocess.CompletedProcess:
    """아이폰이 잠겨 있거나 와이파이 연결이 잠깐 끊기면 가끔 실패해서, 몇 번 다시 시도한다."""
    for i in range(tries):
        try:
            return subprocess.run(["xcrun", "devicectl", *args], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            if i == tries - 1:
                raise SystemExit(f"아이폰과 통신하지 못했어요. 잠금을 풀고 다시 해보세요.\n{e.stderr.strip()[-300:]}") from e
            time.sleep(3)
    raise AssertionError("unreachable")


def find_iphone() -> str:
    """연결된(유선 또는 같은 와이파이) 실제 아이폰 하나의 UDID를 찾는다."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "devices.json"
        devicectl("list", "devices", "--json-output", str(out))
        devices = json.loads(out.read_text())["result"]["devices"]
    for d in devices:
        hw = d.get("hardwareProperties", {})
        if hw.get("deviceType") == "iPhone" and hw.get("reality") == "physical" \
                and d.get("connectionProperties", {}).get("tunnelState") != "unavailable":
            return hw["udid"]
    raise SystemExit("연결된 아이폰을 찾지 못했어요. 맥과 같은 와이파이에 있거나 케이블로 연결돼 있는지 확인하세요.")


def app_domain(device: str) -> list[str]:
    return ["--device", device, "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID]


def drop_near_duplicates(frames: list[Path], min_change: float) -> list[Path]:
    """같은 세션에서 직전에 남긴 사진과 거의 같은 사진은 버린다.

    앱을 켜 두고 가만히 있으면 0.5초마다 같은 장면이 쌓이는데, 이런 사진은 학습 시간만 늘린다.
    작게 줄인 흑백 사진끼리 픽셀 차이의 평균(0~255)이 min_change보다 작으면 같은 장면으로 본다.
    """
    kept, last_by_session = [], {}
    for f in frames:
        small = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (64, 36)).astype("float32")
        session = f.name.rsplit("_", 1)[0]
        prev = last_by_session.get(session)
        if prev is None or float(abs(small - prev).mean()) >= min_change:
            kept.append(f)
            last_by_session[session] = small
    return kept


def summarize(csv_path: Path) -> str:
    rows = list(csv.DictReader(csv_path.open()))[1:]  # 첫 줄은 측정이 덜 쌓인 값이라 뺀다
    if not rows:
        return "기록 없음"
    fps = statistics.median(float(r["fps"]) for r in rows)
    ms = statistics.median(float(r["inference_ms"]) for r in rows)
    return f"{len(rows) + 1}장, 화면 {fps:.0f} FPS, 추론 {ms:.1f} ms (중앙값)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--device", help="아이폰 UDID 또는 이름 (없으면 연결된 아이폰을 찾는다)")
    parser.add_argument("--no-label", action="store_true", help="가져오기만 하고 자동 라벨은 돌리지 않는다")
    parser.add_argument("--min-change", type=float, default=6.0,
                        help="직전 사진과 이만큼도 안 달라지면 버린다 (0이면 전부 남김)")
    args = parser.parse_args()
    pull(args.device or find_iphone(), args.min_change, label=not args.no_label)


def pull(device: str, min_change: float = 6.0, label: bool = True) -> int:
    """아이폰에서 가져와 data/raw에 넣고 (label이면) 자동 라벨까지 돌린다. 새로 넣은 사진 수를 돌려준다."""
    dropped_list = LOG_DIR / "dropped.txt"  # 비슷해서 버린 사진. 다음에 다시 가져와도 또 넣지 않는다
    dropped = set(dropped_list.read_text().split()) if dropped_list.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "Documents"
        devicectl("device", "copy", "from", *app_domain(device), "--source", "Documents", "--destination", str(docs))

        frames = sorted((docs / "frames").glob("*.jpg"))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        fresh = [f for f in frames if not (RAW_DIR / f.name).exists() and f.name not in dropped]
        new = drop_near_duplicates(fresh, min_change)
        for f in new:
            shutil.copy2(f, RAW_DIR / f.name)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        kept = {f.name for f in new}
        with dropped_list.open("a") as out:
            out.writelines(f"{f.name}\n" for f in fresh if f.name not in kept)
        for log in sorted(docs.glob("*app*.csv")):  # 시험용 묶음은 test_app...으로 시작한다
            shutil.copy2(log, LOG_DIR / log.name)  # 앱이 계속 덧붙이므로 매번 최신으로 덮어쓴다

        # 휴대폰에 있는 프레임은 전부 맥에 있으니 다음 실행 때 지워도 된다
        pulled = Path(tmp) / "pulled.txt"
        pulled.write_text("\n".join(f.name for f in frames))
        devicectl("device", "copy", "to", *app_domain(device), "--source", str(pulled),
                  "--destination", "Documents/pulled.txt")

    sessions = sorted({f.name.rsplit("_", 1)[0] for f in new})
    print(f"새 프레임 {len(new)}장 → {RAW_DIR}  "
          f"(휴대폰에 있던 {len(frames)}장 중 새것 {len(fresh)}장, 직전 사진과 거의 같은 {len(fresh) - len(new)}장 제외)")
    for s in sessions:
        log = LOG_DIR / f"{s}.csv"
        print(f"  {s}: {summarize(log) if log.exists() else '기록 없음'}")

    if new and label:
        print("자동 라벨로 data/datasets/mine/에 반영하는 중...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "04_auto_label.py")], check=True)
    return len(new)


if __name__ == "__main__":
    main()
