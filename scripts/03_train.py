"""공개 데이터셋으로 사전학습 YOLO를 파인튜닝한다. 결과는 runs/<이름>/weights/best.pt에 저장된다.

처음부터 학습하지 않고, COCO로 이미 학습된 가중치에서 출발해 새 클래스에 맞춰 조금만 고친다.
그래서 데이터가 수천 장이어도 노트북에서 한두 시간이면 끝난다.

실행 예:
    uv run scripts/03_train.py                              # HomeObjects-3K, yolo26n, 50 epoch
    uv run scripts/03_train.py --epochs 3 --fraction 0.1    # 빨리 한 바퀴만 돌려보기
    uv run scripts/03_train.py --model yolo26s.pt --name home_s

학습이 끝나면 카메라로 확인:
    uv run scripts/01_webcam_yolo.py --camera 1 --model runs/home_n/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import safe_download

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "datasets"
RUNS_DIR = ROOT / "runs"

# 바로 받을 수 있는 공개 데이터셋. 클래스 목록은 데이터셋 라벨 순서와 같아야 한다.
DATASETS = {
    "homeobjects-3K": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip",
        "names": ["bed", "sofa", "chair", "table", "lamp", "tv", "laptop", "wardrobe",
                  "window", "door", "potted plant", "photo frame"],
    },
}


def prepare(name: str) -> Path:
    """데이터셋이 없으면 받아서 풀고, 절대경로가 적힌 yaml을 만들어 그 경로를 돌려준다."""
    spec = DATASETS[name]
    root = DATASETS_DIR / name
    if not (root / "images" / "train").exists():
        safe_download(spec["url"], dir=DATASETS_DIR, unzip=True, delete=True)

    yaml_path = DATASETS_DIR / f"{name}.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(spec["names"]))
    yaml_path.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", default="homeobjects-3K", choices=DATASETS)
    parser.add_argument("--model", default="yolo26n.pt", help="출발점이 될 사전학습 가중치")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="입력 해상도. 크면 작은 물체에 강하지만 느리다")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 중 쓸 비율 (빠른 시험용)")
    parser.add_argument("--name", default="home_n", help="결과 폴더 이름 (runs/<name>)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = prepare(args.data)
    print(f"device={device}  data={data}  model={args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    # best.pt로 검증 세트 점수를 다시 매겨 요약만 보여준다
    metrics = YOLO(RUNS_DIR / args.name / "weights" / "best.pt").val(data=str(data), device=device)
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
