"""폰으로 찍은 영상에서 일정 간격으로 프레임을 뽑아 data/raw에 저장한다.

흔들려서 흐린 프레임은 라벨링해봐야 쓸모가 없으니 선명도(라플라시안 분산)로 걸러낸다.

실행 예:
    uv run scripts/02_extract_frames.py data/videos/board1.mov --every 0.5
    uv run scripts/02_extract_frames.py data/videos/board1.mov --min-sharpness 0   # 필터 끄기
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def sharpness(frame: cv2.typing.MatLike) -> float:
    """값이 클수록 선명하다. 흐린 사진은 경계가 뭉개져서 값이 작다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=float, default=0.5, help="몇 초마다 한 장 뽑을지")
    parser.add_argument("--min-sharpness", type=float, default=100.0, help="이보다 흐리면 버린다")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없어요: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps * args.every)), 1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    index = kept = blurry = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            if sharpness(frame) < args.min_sharpness:
                blurry += 1
            else:
                cv2.imwrite(str(RAW_DIR / f"{args.video.stem}_{index:06d}.jpg"), frame)
                kept += 1
        index += 1

    cap.release()
    print(f"저장 {kept}장, 흐려서 버림 {blurry}장 → {RAW_DIR}")


if __name__ == "__main__":
    main()
