"""선생님 모델(YOLOE)에 물건 이름을 알려주고 data/raw 사진에 라벨 초안을 단다.

결과는 YOLO 학습 형식으로 data/datasets/mine/에 저장되고, 박스를 그린 미리보기는 mine/preview/에 저장된다.
미리보기를 보고 틀린 라벨만 고친 뒤 03_train.py로 학생 모델을 학습시킨다.

같은 물건도 부르는 이름에 따라 YOLOE 점수가 크게 달라서, 클래스마다 이름을 여러 개 주고 하나로 묶는다.
폰을 돌려 찍어 옆으로 누운 프레임도 있어서, 90도씩 돌려서도 찾아본 뒤 결과를 합친다.

실행 예:
    uv run scripts/04_auto_label.py                 # 기본 임계값 0.2
    uv run scripts/04_auto_label.py --conf 0.3      # 확실한 것만 (박스 수는 줄고 오답도 줄어든다)
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "datasets" / "mine"

# 학습할 클래스 → YOLOE에 줄 이름들. 이름을 바꾸거나 더하면 라벨 품질이 달라진다.
CLASSES = {
    "monitor": ["computer monitor", "screen"],
    "laptop": ["laptop"],
    "keyboard": ["keyboard"],
    "figure": ["figurine", "action figure", "small doll"],
    "toy car": ["toy car", "race car model", "lego car"],
    "tumbler": ["tumbler", "cup with straw"],
    "bottle": ["bottle"],
    "speaker": ["speaker"],
    "headphones": ["headphones"],
    "clothes": ["clothes", "t-shirt", "jersey"],
    "cap": ["cap"],
    "book": ["book"],
    "mouse": ["computer mouse"],
    "phone": ["cell phone", "smartphone"],
    "cup": ["cup", "mug", "plastic cup"],
    "shoes": ["shoes", "sneakers"],
    "backpack": ["backpack"],
    "chair": ["chair"],
    "sofa": ["sofa"],
}


def predict_rotated(model: YOLOE, frame, conf: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """원본·시계방향·반시계방향 90도로 돌려 찾고, 박스를 원본 좌표로 되돌려 모은다."""
    h, w = frame.shape[:2]
    boxes, scores, prompt_ids = [], [], []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        img = frame if rot is None else cv2.rotate(frame, rot)
        r = model.predict(img, conf=conf, verbose=False)[0].boxes
        b = r.xyxy.cpu().clone()
        if rot == cv2.ROTATE_90_CLOCKWISE:  # 원본 (x, y) → (h - y, x)
            b = torch.stack([b[:, 1], h - b[:, 2], b[:, 3], h - b[:, 0]], dim=1)
        elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:  # 원본 (x, y) → (y, w - x)
            b = torch.stack([w - b[:, 3], b[:, 0], w - b[:, 1], b[:, 2]], dim=1)
        boxes.append(b)
        scores.append(r.conf.cpu())
        prompt_ids.append(r.cls.cpu().long())
    return torch.cat(boxes), torch.cat(scores), torch.cat(prompt_ids)


def split_of(paths: list[Path]) -> dict[Path, str]:
    """영상 단위로 train/val을 나눈다. 영상이 하나뿐이면 뒤쪽 20% 구간을 val로 쓴다.

    같은 영상의 이웃 프레임은 거의 똑같아서, 무작위로 섞어 나누면 val 점수가 부풀려진다.
    """
    by_video = defaultdict(list)
    for p in sorted(paths):
        by_video[p.stem.rsplit("_", 1)[0]].append(p)
    videos = sorted(by_video)
    split = {}
    if len(videos) >= 2:
        val_videos = set(videos[-max(len(videos) // 5, 1):])  # 이름순 마지막 20% 영상
        for v, ps in by_video.items():
            split.update({p: "val" if v in val_videos else "train" for p in ps})
    else:
        ps = by_video[videos[0]]
        cut = int(len(ps) * 0.8)
        split.update({p: "train" if i < cut else "val" for i, p in enumerate(ps)})
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="선생님 모델 (클수록 정확하고 느리다)")
    parser.add_argument("--conf", type=float, default=0.2, help="이보다 확신이 낮은 박스는 라벨로 쓰지 않는다")
    args = parser.parse_args()

    names = list(CLASSES)
    prompts = [p for ps in CLASSES.values() for p in ps]
    prompt_to_class = torch.tensor([names.index(c) for c, ps in CLASSES.items() for _ in ps])

    model = YOLOE(args.model)
    model.set_classes(prompts)

    paths = sorted(RAW_DIR.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"{RAW_DIR}에 사진이 없어요. 02_extract_frames.py로 먼저 프레임을 뽑으세요.")
    split = split_of(paths)

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        (OUT_DIR / sub).mkdir(parents=True)

    counts = Counter()
    for p in paths:
        frame = cv2.imread(str(p))
        h, w = frame.shape[:2]
        boxes, scores, pids = predict_rotated(model, frame, args.conf)
        cls = prompt_to_class[pids]
        # 이름이 달라도 같은 클래스면, 같은 물체를 여러 번 찾은 박스를 하나만 남긴다
        keep = batched_nms(boxes, scores, cls, iou_threshold=0.5)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

        s = split[p]
        shutil.copy2(p, OUT_DIR / "images" / s / p.name)
        lines = []
        for (x1, y1, x2, y2), c in zip(boxes.tolist(), cls.tolist()):
            x1, x2 = max(x1, 0), min(x2, w)
            y1, y2 = max(y1, 0), min(y2, h)
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[names[c]] += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        for (x1, y1, _, _), c, sc in zip(boxes.tolist(), cls.tolist(), scores.tolist()):
            cv2.putText(frame, f"{names[c]} {sc:.2f}", (int(x1) + 4, int(y1) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        (OUT_DIR / "labels" / s / f"{p.stem}.txt").write_text("\n".join(lines))
        cv2.imwrite(str(OUT_DIR / "preview" / p.name), cv2.resize(frame, (960, int(960 * h / w))))

    yaml_names = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (OUT_DIR.parent / "mine.yaml").write_text(
        f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\nnames:\n{yaml_names}\n"
    )
    n_val = sum(v == "val" for v in split.values())
    print(f"사진 {len(paths)}장 (train {len(paths) - n_val} / val {n_val}), 박스 {sum(counts.values())}개")
    for n in names:
        print(f"  {n:12s} {counts[n]}")
    print(f"미리보기: {OUT_DIR / 'preview'}")


if __name__ == "__main__":
    main()
