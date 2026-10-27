"""프로젝트 전체 진행 상황을 한 화면에 요약한다: 로드맵, 데이터, 라벨 검수, 학습 결과.

파일만 읽으므로 학습 중에도 언제든 실행해도 된다. 한 run을 자세히 보려면 watch_train.py.

실행 예:
    uv run scripts/status.py
    uv run scripts/status.py --follow       # 5초마다 새로고침, 학습 중인 run의 진행 바도 표시
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = ROOT / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
# 로그가 이 시간 안에 바뀌었으면 학습 중으로 본다 (진행 바가 매 iteration마다 로그를 쓴다)
RUNNING_WINDOW_SEC = 120
# 실행 중인 작업의 로그가 이 시간 넘게 그대로면 멈춘 것으로 의심한다
STALL_SEC = 600


def count_files(folder: Path, exts: set[str]) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)


def roadmap() -> None:
    print("■ 로드맵 (README.md)")
    for line in (ROOT / "README.md").read_text().splitlines():
        m = re.match(r"- \[( |x)\] \*\*(.+?)\*\*", line)
        if m:
            print(f"  {'✔' if m[1] == 'x' else '·'} {m[2]}")
    print()


def data() -> None:
    print("■ 데이터")
    print(f"  영상 data/videos/   {count_files(DATA_DIR / 'videos', VIDEO_EXTS)}개")
    print(f"  사진 data/raw/      {count_files(DATA_DIR / 'raw', IMAGE_EXTS)}장")
    if not DATASETS_DIR.exists():
        print()
        return
    for ds in sorted(p for p in DATASETS_DIR.iterdir() if (p / "images").is_dir()):
        for split_dir in sorted((ds / "images").iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            n_img = count_files(split_dir, IMAGE_EXTS)
            label_dir = ds / "labels" / split
            labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
            n_box = sum(len(p.read_text().split("\n")) - 1 for p in labels)
            n_empty = n_img - sum(1 for p in labels if p.stat().st_size > 0)
            line = f"  {ds.name + '/' + split:<20} {n_img:>5}장  박스 {n_box:>6}  빈 사진 {n_empty:>4}"
            # 05_review.py open이 사진 옆에 만든 JSON = 검수 중
            n_json = len(list(split_dir.glob("*.json")))
            if n_json:
                line += f"  검수 JSON {n_json}"
            print(line)
    print()


def run_status(run_dir: Path, done: int, total: int) -> str:
    if done >= total:
        return "완료"
    logs = [run_dir.parent / f"{run_dir.name}.log", run_dir / "results.csv"]
    latest = max((p.stat().st_mtime for p in logs if p.exists()), default=0)
    return "학습 중" if time.time() - latest < RUNNING_WINDOW_SEC else "중단됨"


def last_log_line(log: Path) -> str:
    """로그 끝의 진행 바 한 줄. 진행 바는 \\r과 지우기 코드(ESC[K)로 같은 줄을 덮어쓴다."""
    if not log.exists():
        return ""
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - 4096))
        tail = f.read().decode(errors="ignore")
    parts = [p.strip() for p in re.split(r"\r|\n|\x1b\[K", tail)]
    return next((p for p in reversed(parts) if p), "")


def runs() -> None:
    print("■ 학습 (runs/)  ※ val 세트가 다르면 mAP를 서로 비교할 수 없다")
    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*/")) if RUNS_DIR.exists() else []:
        args_yaml = run_dir / "args.yaml"
        if not args_yaml.exists():
            continue  # *_val 폴더처럼 학습 run이 아닌 것
        args = yaml.safe_load(args_yaml.read_text())
        data_yaml = Path(args["data"])
        val = yaml.safe_load(data_yaml.read_text()).get("val", "?") if data_yaml.exists() else "?"
        val_name = f"{data_yaml.stem if val == 'images/val' else Path(str(val)).parts[0]}/val"
        model = Path(args["model"]).name
        if model == "last.pt":
            model = "(이어하기)"  # --resume으로 이어 학습하면 출발 모델이 last.pt로 바뀐다

        results = run_dir / "results.csv"
        epochs = []
        if results.exists():
            with results.open() as f:
                epochs = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        done = int(epochs[-1]["epoch"]) if epochs else 0
        best = max(epochs, key=lambda r: r["metrics/mAP50-95(B)"]) if epochs else None
        rows.append((run_dir.name, data_yaml.stem, val_name, model, done, args["epochs"], best,
                     run_status(run_dir, done, args["epochs"])))

    if not rows:
        print("  아직 학습한 run이 없다\n")
        return
    print(f"  {'run':<10} {'data':<15} {'val':<18} {'model':<11} {'epoch':>7}  "
          f"{'mAP50':>6} {'50-95':>6}  상태")
    for name, data_name, val_name, model, done, total, best, status in rows:
        m50 = f"{best['metrics/mAP50(B)']:.3f}" if best else "-"
        m95 = f"{best['metrics/mAP50-95(B)']:.3f}" if best else "-"
        print(f"  {name:<10} {data_name:<15} {val_name:<18} {model:<11} {f'{done}/{total}':>7}  "
              f"{m50:>6} {m95:>6}  {status}")
    print("  (mAP는 각 run의 best epoch 기준)\n")

    for name, *_, status in rows:
        if status == "학습 중":
            print(f"  ▶ {name}: {last_log_line(RUNS_DIR / f'{name}.log')}")
    print()


def notes() -> None:
    files = sorted((ROOT / "notes").glob("*.md"))
    print("■ 기록 (notes/)")
    for p in files:
        print(f"  {p.name}")
    if not files:
        print("  없음")


def stdout_file(pid: str) -> Path | None:
    """프로세스의 표준출력이 파일로 리다이렉트돼 있으면 그 경로 (예: > runs/x.log)."""
    # 프로세스가 그 사이 끝나면 lsof가 실패하므로 check=False
    out = subprocess.run(["lsof", "-a", "-p", pid, "-d", "1", "-Fn"],
                         check=False, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("n/") and Path(line[1:]).is_file():
            return Path(line[1:])
    return None


def train_progress(run_dir: Path) -> str:
    """학습 run의 완료 epoch, 남은 시간, best mAP 한 줄 요약."""
    args_yaml, results = run_dir / "args.yaml", run_dir / "results.csv"
    if not args_yaml.exists():
        return f"학습: {run_dir.name} 준비 중"
    total = yaml.safe_load(args_yaml.read_text())["epochs"]
    rows = []
    if results.exists():
        with results.open() as f:
            rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return f"학습: {run_dir.name} epoch 0/{total} 완료 (첫 epoch 진행 중)"
    done, elapsed = int(rows[-1]["epoch"]), rows[-1]["time"]
    eta = elapsed / done * (total - done)
    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    return (f"학습: {run_dir.name} epoch {done}/{total} 완료, 남은 시간 약 {int(eta // 3600)}h"
            f"{int(eta % 3600 // 60):02d}m, best mAP50={best['metrics/mAP50(B)']:.3f} "
            f"(epoch {int(best['epoch'])})")


def jobs() -> None:
    """학습뿐 아니라 라벨링·데이터 받기 등 scripts/의 번호 스크립트가 돌고 있는지 본다."""
    print("■ 실행 중인 작업")
    out = subprocess.run(["ps", "-ax", "-o", "pid=,etime=,command="],
                         check=True, capture_output=True, text=True)
    found = False
    for line in out.stdout.splitlines():
        pid, etime, cmd = line.split(None, 2)
        # uv run 래퍼 말고 실제 python 프로세스만
        if str(ROOT) not in cmd.split()[0] or not re.search(r"scripts/\d\d_\w+\.py", cmd):
            continue
        found = True
        script = re.search(r"scripts/\S+", cmd)[0]
        print(f"  {script}{cmd.split(script, 1)[1]}  (pid {pid}, 실행 {etime})")
        if script.endswith("03_train.py"):
            m = re.search(r"--name[ =](\S+)", cmd)
            print(f"    {train_progress(RUNS_DIR / (m[1] if m else 'home_n'))}")
        log = stdout_file(pid)
        if log is None:
            print("    로그: 터미널에 직접 출력 중 (파일로 저장 안 함)")
            continue
        idle = time.time() - log.stat().st_mtime
        warn = f"  ⚠ {int(idle // 60)}분째 로그 변화 없음 — 멈췄을 수 있음" if idle > STALL_SEC else ""
        print(f"    로그: {log.relative_to(ROOT) if log.is_relative_to(ROOT) else log}{warn}")
        print(f"    마지막: {last_log_line(log)[:120]}")
    if not found:
        print("  없음")
    print()


def report() -> None:
    roadmap()
    data()
    jobs()
    runs()
    notes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--follow", action="store_true", help="주기적으로 새로고침 (Ctrl+C로 종료)")
    parser.add_argument("--interval", type=int, default=5, help="--follow 새로고침 간격(초)")
    args = parser.parse_args()

    if not args.follow:
        report()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # 화면 지우기
            print(time.strftime("%H:%M:%S 기준\n"))
            report()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
