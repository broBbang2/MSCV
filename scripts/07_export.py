"""학습한 모델을 아이폰용 CoreML(.mlpackage)로 바꾸고 파일 크기를 비교한다.

16비트(half)로 저장하면 정확도는 거의 그대로이고 파일은 절반쯤으로 줄어든다.
결과는 runs/<이름>/weights/best_fp16.mlpackage (또는 best_int8)에 생기고, Xcode에서 열면
Performance 탭에서 연결한 아이폰으로 실제 추론 시간을 잴 수 있다.

coremltools 9.0은 프로젝트의 NumPy 2.x·torch 2.14와 맞지 않아 변환 중 오류가 난다.
그래서 변환만 coremltools가 시험된 버전으로 따로 띄운 환경에서 돌린다:
    uv run --isolated --no-project --python 3.12 --with ultralytics==8.4.184 --with torch==2.7.0 \\
        --with torchvision==0.22.0 --with "numpy<2" --with coremltools==9.0 \\
        python scripts/07_export.py                    # open2_n, open2_s를 16비트로 변환
    (같은 앞부분 + --with scikit-learn) python scripts/07_export.py --runs open2_s --int8   # 8비트로 더 줄이기
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


def size_mb(path: Path) -> float:
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(f.stat().st_size for f in files if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["open2_n", "open2_s"], help="변환할 runs/ 폴더 이름들")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="16비트 대신 8비트로 (더 작고, 정확도는 떨어질 수 있음)")
    parser.add_argument("--nms", action="store_true",
                        help="겹친 박스 정리(NMS)까지 모델에 넣는다. iOS Vision이 이름+박스로 바로 돌려줘서 앱이 단순해진다")
    args = parser.parse_args()

    rows = []
    for name in args.runs:
        pt = RUNS_DIR / name / "weights" / "best.pt"
        out = Path(YOLO(pt).export(format="coreml", imgsz=args.imgsz, half=not args.int8, int8=args.int8,
                                   nms=args.nms))
        # export는 항상 best.mlpackage로 저장하므로, 설정별 결과가 서로 덮어쓰지 않게 이름을 바꾼다
        final = out.with_name(f"best_{'int8' if args.int8 else 'fp16'}{'_nms' if args.nms else ''}.mlpackage")
        shutil.rmtree(final, ignore_errors=True)
        out.rename(final)
        rows.append((name, size_mb(pt), size_mb(final), final))

    print(f"\n{'모델':10s} {'.pt':>8s} {'CoreML':>8s}")
    for name, pt_mb, ml_mb, out in rows:
        print(f"{name:10s} {pt_mb:7.1f}MB {ml_mb:7.1f}MB  {out}")


if __name__ == "__main__":
    main()
