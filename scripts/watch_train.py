"""학습 진행 상황을 본다. ultralytics가 epoch마다 쓰는 runs/<이름>/results.csv를 읽어 요약한다.

학습을 건드리지 않고 파일만 읽으므로, 학습 중에 다른 터미널에서 언제든 실행해도 된다.

실행 예:
    uv run scripts/watch_train.py                  # runs/home_n 진행 상황 한 번 출력
    uv run scripts/watch_train.py --follow         # 30초마다 새로고침 (Ctrl+C로 종료)
    uv run scripts/watch_train.py --name home_s --last 5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


def total_epochs(run_dir: Path) -> int | None:
    """args.yaml에서 목표 epoch 수를 읽는다."""
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return None
    for line in args_yaml.read_text().splitlines():
        if line.startswith("epochs:"):
            return int(line.split(":", 1)[1])
    return None


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def report(run_dir: Path, last: int) -> None:
    results = run_dir / "results.csv"
    if not results.exists():
        print(f"{results} 없음 — 아직 첫 epoch이 끝나지 않았거나 이름이 다르다")
        return
    with results.open() as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        print("아직 기록된 epoch이 없다")
        return

    total = total_epochs(run_dir)
    cur = rows[-1]
    done = int(cur["epoch"])
    elapsed = cur["time"]
    line = f"[{run_dir.name}] epoch {done}/{total or '?'}  경과 {fmt_time(elapsed)}"
    if total:
        eta = elapsed / done * (total - done)
        line += f"  남은 시간 약 {fmt_time(eta)}"
    print(line)

    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    print(f"best: epoch {int(best['epoch'])}  mAP50={best['metrics/mAP50(B)']:.3f}  "
          f"mAP50-95={best['metrics/mAP50-95(B)']:.3f}\n")

    print(f"{'epoch':>5} {'box':>6} {'cls':>6} {'P':>6} {'R':>6} {'mAP50':>6} {'50-95':>6} {'v_box':>6}")
    for r in rows[-last:]:
        print(f"{int(r['epoch']):>5} {r['train/box_loss']:>6.3f} {r['train/cls_loss']:>6.3f} "
              f"{r['metrics/precision(B)']:>6.3f} {r['metrics/recall(B)']:>6.3f} "
              f"{r['metrics/mAP50(B)']:>6.3f} {r['metrics/mAP50-95(B)']:>6.3f} {r['val/box_loss']:>6.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", default="home_n", help="결과 폴더 이름 (runs/<name>)")
    parser.add_argument("--last", type=int, default=10, help="최근 몇 epoch을 표로 보여줄지")
    parser.add_argument("--follow", action="store_true", help="주기적으로 새로고침")
    parser.add_argument("--interval", type=int, default=30, help="--follow 새로고침 간격(초)")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.name
    if not args.follow:
        report(run_dir, args.last)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # 화면 지우기
            report(run_dir, args.last)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
