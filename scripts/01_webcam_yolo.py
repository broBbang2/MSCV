"""1단계: 사전학습 YOLO를 카메라에 돌려본다. 's'를 누르면 원본 프레임을 data/raw에 저장한다.

실행 예:
    uv run scripts/01_webcam_yolo.py --list-cameras     # 카메라 번호 확인 (아이폰은 보통 1번)
    uv run scripts/01_webcam_yolo.py --camera 0          # 맥 내장 카메라로 탐지
    uv run scripts/01_webcam_yolo.py --camera 1 --no-detect   # 탐지 없이 촬영만 (데이터 수집용)

키: s = 프레임 저장, q = 종료
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def list_cameras(max_index: int = 5) -> None:
    """열 수 있는 카메라 번호와 해상도를 출력한다."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"[{i}] {w}x{h}")
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호")
    parser.add_argument("--model", default="yolo11n.pt", help="사전학습 모델 (처음 실행 시 자동 다운로드)")
    parser.add_argument("--conf", type=float, default=0.25, help="이 값보다 신뢰도가 낮은 박스는 버린다")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--no-detect", action="store_true", help="탐지 없이 촬영만")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = None if args.no_detect else YOLO(args.model)
    print(f"device={device}  model={'없음' if model is None else args.model}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요. --list-cameras로 번호를 확인하세요.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    prev = time.perf_counter()
    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame
        if model is not None:
            result = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
            view = result.plot()
            # 1초에 한 번, 지금 화면에서 무엇을 몇 개 찾았는지 터미널에 출력
            if time.perf_counter() - last_print > 1:
                names = [result.names[int(c)] for c in result.boxes.cls]
                print(dict(Counter(names)) or "탐지 없음")
                last_print = time.perf_counter()

        now = time.perf_counter()
        fps = 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(view, f"FPS {fps:4.1f}  saved {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("firstCV", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = RAW_DIR / f"cam_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(path), frame)  # 박스가 그려지지 않은 원본을 저장 (라벨링용)
            saved += 1
            print(f"저장: {path.name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
