"""공개 데이터셋으로 사전학습 YOLO를 파인튜닝한다. 결과는 runs/<이름>/weights/best.pt에 저장된다.

처음부터 학습하지 않고, COCO로 이미 학습된 가중치에서 출발해 새 클래스에 맞춰 조금만 고친다.
그래서 데이터가 수천 장이어도 노트북에서 한두 시간이면 끝난다.

실행 예:
    uv run scripts/03_train.py                              # HomeObjects-3K, yolo26n, 50 epoch
    uv run scripts/03_train.py --epochs 3 --fraction 0.1    # 빨리 한 바퀴만 돌려보기
    uv run scripts/03_train.py --model yolo26s.pt --name home_s
    uv run scripts/03_train.py --resume                     # 중단된 runs/home_n 이어서 학습

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

# 학습 하이퍼파라미터. 여기 없는 값은 ultralytics 기본값을 쓴다 (runs/<name>/args.yaml에 전체가 남는다).
# 예전에는 optimizer="auto"였는데, 이러면 데이터 크기·클래스 수에 따라 옵티마이저와 lr이 몰래 바뀌어
# 실험끼리 비교가 안 됐다. 그래서 auto가 실제로 골랐던 값(AdamW, lr 0.0004~0.0006)을 명시해 고정한다.
HYPERPARAMS = {
    # --- 옵티마이저: 손실을 줄이는 방향으로 가중치를 얼마나, 어떻게 고칠지 ---
    "optimizer": "AdamW",    # 파라미터마다 보폭을 따로 조절한다. 적은 데이터 파인튜닝에서 SGD보다 덜 까다롭다
    "lr0": 0.0005,           # 처음 학습률(보폭). 크면 사전학습 지식이 망가지고, 작으면 거의 안 배운다
    "momentum": 0.9,         # AdamW에선 beta1. 직전 이동 방향을 90% 유지해 진동을 줄인다
    "weight_decay": 0.0005,  # 가중치가 너무 커지지 않게 매 스텝 살짝 줄인다 → 과적합 완화
    # --- 학습률 스케줄: 학습 동안 lr을 어떻게 바꿀지 ---
    "warmup_epochs": 3,      # 처음 3 epoch은 lr을 0에서 lr0까지 천천히 올린다 (초반 큰 기울기로 망가지는 것 방지)
    "lrf": 0.01,             # 마지막 epoch의 lr = lr0 × lrf. 끝으로 갈수록 보폭을 줄여 미세 조정한다
    "cos_lr": False,         # False면 직선으로 감소, True면 코사인 곡선으로 감소
    # --- 멈춤 조건 ---
    "patience": 20,          # val 점수가 20 epoch 동안 안 오르면 조기 종료. best.pt는 가장 좋았던 시점으로 남는다
    # --- 데이터 증강: 같은 사진을 매번 조금씩 바꿔 보여줘 데이터가 많은 척한다 ---
    "mosaic": 1.0,           # 사진 4장을 한 장으로 이어 붙인다. 작은 물체·다양한 배경을 많이 보게 된다
    "close_mosaic": 10,      # 마지막 10 epoch은 mosaic을 끄고 실제 사진 모양에 맞춘다
    "fliplr": 0.5,           # 50% 확률로 좌우 반전
    "scale": 0.5,            # 크기를 ±50% 범위에서 무작위로 바꾼다 (가까이/멀리서 본 물체)
    "translate": 0.1,        # 위치를 ±10% 이동
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,  # 색조·채도·밝기 변화 (조명이 다른 방)
    "mixup": 0.0,            # 두 사진을 반투명하게 겹치기. 데이터가 적을 땐 오히려 헷갈려서 끈다
}


def prepare(name: str) -> Path:
    """데이터셋이 없으면 받아서 풀고, 절대경로가 적힌 yaml을 만들어 그 경로를 돌려준다.

    DATASETS에 없는 이름은 04_auto_label.py가 만든 data/datasets/<name>.yaml을 그대로 쓴다.
    """
    if name not in DATASETS:
        yaml_path = DATASETS_DIR / f"{name}.yaml"
        if not yaml_path.exists():
            raise SystemExit(f"{yaml_path}가 없어요. 04_auto_label.py를 먼저 실행하세요.")
        return yaml_path
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
    parser.add_argument("--data", default="homeobjects-3K",
                        help=f"공개 데이터셋({', '.join(DATASETS)}) 또는 직접 만든 데이터셋 이름 (예: mine)")
    parser.add_argument("--model", default="yolo26n.pt", help="출발점이 될 사전학습 가중치")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="입력 해상도. 크면 작은 물체에 강하지만 느리다")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 중 쓸 비율 (빠른 시험용)")
    parser.add_argument("--optimizer", default=HYPERPARAMS["optimizer"], help="AdamW, SGD, auto 등")
    parser.add_argument("--lr0", type=float, default=HYPERPARAMS["lr0"], help="처음 학습률. SGD라면 0.01 정도가 보통")
    parser.add_argument("--patience", type=int, default=HYPERPARAMS["patience"], help="조기 종료까지 기다릴 epoch 수")
    parser.add_argument("--name", default="home_n", help="결과 폴더 이름 (runs/<name>)")
    parser.add_argument("--resume", action="store_true",
                        help="중단된 학습을 runs/<name>/weights/last.pt에서 이어서 한다 (다른 학습 옵션은 무시)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = prepare(args.data)

    if args.resume:
        # 이어하기는 처음 학습할 때의 설정(args.yaml)을 그대로 쓴다
        last = RUNS_DIR / args.name / "weights" / "last.pt"
        print(f"resume from {last}")
        YOLO(last).train(resume=True)
    else:
        hyp = {**HYPERPARAMS, "optimizer": args.optimizer, "lr0": args.lr0, "patience": args.patience}
        print(f"device={device}  data={data}  model={args.model}")
        print(f"epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  "
              f"optimizer={hyp['optimizer']}  lr0={hyp['lr0']}  patience={hyp['patience']}")
        model = YOLO(args.model)
        model.train(
            **hyp,
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
    # project를 안 주면 ultralytics 전역 설정의 runs 폴더(다른 프로젝트일 수 있음)에 저장된다
    metrics = YOLO(RUNS_DIR / args.name / "weights" / "best.pt").val(
        data=str(data), device=device, project=str(RUNS_DIR), name=f"{args.name}_val", exist_ok=True
    )
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
