"""자동 라벨을 X-AnyLabeling으로 사람이 검수할 수 있게 열고, 고친 결과를 다시 YOLO 라벨로 되돌린다.

검수 순서:
    uv run scripts/05_review.py open             # YOLO 라벨 → JSON 변환 후 검수 앱 실행 (기본: val)
    (앱에서 박스를 지우고/고치고/추가한 뒤 저장)
    uv run scripts/05_review.py save             # JSON → YOLO 라벨로 되돌림
    uv run scripts/03_train.py --data mine --name mine3_n

    uv run scripts/05_review.py open --split train   # 학습용도 검수할 때

정답으로 남기기 (사진을 전부 검수한 뒤에만):
    uv run scripts/05_review.py save --gold      # data/gold/에도 저장 → 13_gold.py 채점의 정답이 된다

open은 이미 검수 중인 JSON이 있으면 덮어쓰지 않는다. 처음부터 다시 하려면 --reset.
04_auto_label.py를 다시 돌리면 mine/ 전체가 새로 만들어진다. --gold로 남긴 사진은 04가 정답을 다시 쓰지만,
--gold 없이 저장한 검수 내용은 사라진다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "datasets"
DATA_DIR = DATASETS_DIR / "mine"
GOLD_DIR = ROOT / "data" / "gold"  # 04_auto_label.py의 GOLD_DIR과 같은 곳
APP =["uvx", "--from", "x-anylabeling-cvhub[cpu]", "xanylabeling"]


def class_names() -> list[str]:
    names = yaml.safe_load((DATASETS_DIR / "mine.yaml").read_text())["names"]
    return [names[i] for i in sorted(names)]


def yolo_to_json(split: str, names: list[str], reset: bool) -> int:
    """YOLO txt를 X-AnyLabeling이 읽는 JSON(사진 옆에 같은 이름)으로 바꾼다."""
    n = 0
    for img in sorted((DATA_DIR / "images" / split).glob("*.jpg")):
        out = img.with_suffix(".json")
        if out.exists() and not reset:
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        shapes = []
        txt = DATA_DIR / "labels" / split / f"{img.stem}.txt"
        for line in txt.read_text().splitlines() if txt.exists() else []:
            c, cx, cy, bw, bh = line.split()
            x1, y1 = (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h
            x2, y2 = (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h
            shapes.append({
                "label": names[int(c)], "shape_type": "rectangle", "flags": {},
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None, "description": None, "difficult": False, "attributes": {},
            })
        out.write_text(json.dumps({
            "version": "4.2.0", "flags": {}, "checked": False, "shapes": shapes,
            "imagePath": img.name, "imageData": None, "imageHeight": h, "imageWidth": w,
        }, ensure_ascii=False, indent=2))
        n += 1
    return n


def same_labels(a: list[str], b: list[str]) -> bool:
    """YOLO 라벨 두 벌이 같은지. JSON을 거쳐 오면 소수점 끝자리가 달라질 수 있어 조금의 차이는 같다고 본다."""
    if len(a) != len(b):
        return False
    pa = sorted([float(v) for v in line.split()] for line in a)
    pb = sorted([float(v) for v in line.split()] for line in b)
    return all(abs(x - y) < 1e-3 for la, lb in zip(pa, pb) for x, y in zip(la, lb))


def json_to_yolo(split: str, names: list[str]) -> tuple[int, int, int]:
    """검수한 JSON을 YOLO txt로 되돌린다. 목록에 없는 이름의 박스는 알려주고 버린다.

    검수 전 라벨과 달라진 사진 수도 센다. 선생님 라벨이 얼마나 틀렸는지 감을 잡는 숫자다.
    """
    n_img = n_box = n_changed = 0
    for js in sorted((DATA_DIR / "images" / split).glob("*.json")):
        data = json.loads(js.read_text())
        w, h = data["imageWidth"], data["imageHeight"]
        lines = []
        for s in data["shapes"]:
            if s["shape_type"] != "rectangle":
                continue
            if s["label"] not in names:
                print(f"  모르는 이름 '{s['label']}' 무시: {js.name}")
                continue
            xs, ys = [p[0] for p in s["points"]], [p[1] for p in s["points"]]
            x1, x2 = max(min(xs), 0), min(max(xs), w)
            y1, y2 = max(min(ys), 0), min(max(ys), h)
            lines.append(f"{names.index(s['label'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        txt = DATA_DIR / "labels" / split / f"{js.stem}.txt"
        if not same_labels(txt.read_text().splitlines() if txt.exists() else [], lines):
            n_changed += 1
        txt.write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    return n_img, n_box, n_changed


def save_gold(split: str, names: list[str]) -> int:
    """검수한 JSON을 data/gold/에 정답으로 남긴다. 검수할 때 있던 클래스 목록을 함께 적어 둔다."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for js in sorted((DATA_DIR / "images" / split).glob("*.json")):
        data = json.loads(js.read_text())
        data["firstcv"] = {"classes": names, "reviewed": datetime.now().isoformat(timespec="minutes")}
        (GOLD_DIR / js.name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("action", choices=["open", "save"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--reset", action="store_true", help="검수 중인 JSON을 버리고 자동 라벨로 다시 시작")
    parser.add_argument("--gold", action="store_true",
                        help="save할 때 data/gold/에 정답으로도 남긴다. 그 split의 JSON이 전부 올라가므로 다 검수한 뒤에만")
    args = parser.parse_args()

    names = class_names()
    if args.action == "open":
        n = yolo_to_json(args.split, names, args.reset)
        print(f"JSON {n}장 새로 만듦. 검수 앱을 엽니다 (처음 실행은 설치 때문에 오래 걸릴 수 있음)")
        labels = DATA_DIR / "classes.txt"
        labels.write_text("\n".join(names))
        subprocess.run([*APP, "--filename", str(DATA_DIR / "images" / args.split),
                        "--labels", str(labels), "--validatelabel", "exact",
                        "--nodata", "--autosave", "--no-auto-update-check"], check=False)
    else:
        n_img, n_box, n_changed = json_to_yolo(args.split, names)
        print(f"{args.split}: 사진 {n_img}장, 박스 {n_box}개를 YOLO 라벨로 저장했어요 (검수로 바뀐 사진 {n_changed}장)")
        if args.gold:
            print(f"  data/gold/에 정답 {save_gold(args.split, names)}장을 남겼어요. 채점: uv run scripts/13_gold.py eval")


if __name__ == "__main__":
    main()
