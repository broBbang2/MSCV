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
from collections import Counter
from pathlib import Path

import cv2
import torch
from torchvision.ops import batched_nms, nms
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "datasets" / "mine"
# 10_prototypes.py가 채점해서 고른 클래스별 프로토타입. 있으면 그 클래스는 글자와 예시를 함께 준다
PROTOTYPES = ROOT / "data" / "prototypes" / "teacher.pt"

# 채점에만 쓰고 학습에는 절대 넣지 않는 영상 (이름이 test_로 시작하는 영상도 같은 취급)
VAL_VIDEOS = {"촬영본"}

# 학습할 클래스 → YOLOE에 줄 이름들. 이름을 바꾸거나 더하면 라벨 품질이 달라진다.
# 클래스를 추가하면 앱 화면용 한국어 이름도 ios/FirstCV/KoreanLabels.swift에 넣는다.
# 사람 라벨이 있는 COCO·Open Images 300장으로 채점해서 고른 이름이다 (평균 AP50 0.59 → 0.65).
# 특히 모니터는 tv/television을 넣자 0.22 → 0.83, 모자는 baseball cap/hat을 넣자 0.75 → 0.85.
CLASSES = {
    "monitor": ["computer monitor", "screen", "television", "tv", "display"],
    "laptop": ["laptop"],
    "keyboard": ["keyboard"],
    "figure": ["figurine", "action figure", "small doll"],
    "toy car": ["toy car", "race car model", "lego car"],
    "tumbler": ["tumbler", "cup with straw"],
    "bottle": ["bottle"],
    "speaker": ["speaker"],
    "headphones": ["headphones", "headset", "earphones", "over-ear headphones"],
    "clothes": ["clothes", "t-shirt", "jersey"],
    "cap": ["cap", "baseball cap", "hat"],
    "book": ["book"],
    "mouse": ["computer mouse"],
    "phone": ["cell phone", "smartphone", "mobile phone"],
    "cup": ["cup", "mug", "plastic cup"],
    "shoes": ["shoes", "sneakers"],
    "backpack": ["backpack", "bag"],
    "chair": ["chair"],
    "sofa": ["sofa", "couch"],
    # 처음엔 물건만 골랐는데, 사람이 없으면 공개 사진 속 수천 명이 "배경"으로 학습돼 사람을 적극적으로 무시하게 된다.
    # 맨 뒤에 붙여서 기존 클래스 번호는 그대로 둔다.
    "person": ["person"],
    # 보행 보조용 (2026-10-27). 걷는 길에서 부딪히거나 넘어질 수 있는 것들. 역시 맨 뒤에 붙인다.
    "door": ["door", "glass door"],
    "stairs": ["stairs", "staircase", "steps"],
    "bicycle": ["bicycle"],
    "car": ["car", "truck", "bus", "van"],
    "motorcycle": ["motorcycle", "motor scooter"],
    "pole": ["pole", "bollard", "street light", "lamp post", "utility pole"],
    "table": ["table", "desk", "dining table"],
    "trash can": ["trash can", "garbage bin", "waste basket"],
    "bench": ["bench"],
    "traffic cone": ["traffic cone"],
}

# 우리 클래스는 아니지만 자주 헷갈리는 물건. 선생님이 침대를 소파로, 리모컨을 휴대폰으로 억지로 맞추지 않게
# 같이 찾게 한 뒤 라벨에서는 버린다. 사람은 이제 진짜 클래스라 여기 넣지 않는다.
# (헷갈림용으로 넣었을 땐 피규어가 사람으로 흡수돼 AP50 0.77 → 0.33. 지금은 clean()의 겹침 정리가
# 같은 물체에 person과 figure가 함께 붙으면 점수 높은 쪽만 남긴다.)
# table·desk·door는 보행용 클래스가 되어 뺐다.
DISTRACTORS = ["bed", "pillow", "blanket", "remote control", "power strip", "cable", "box",
               "curtain", "plant", "lamp"]
SCREEN_CONTAINED = 0.85  # 모니터 박스 안에 이 비율 이상 들어간 박스는 화면 속 장면으로 보고 버린다


def load_prototypes() -> dict[str, torch.Tensor]:
    return torch.load(PROTOTYPES) if PROTOTYPES.exists() else {}


def set_teacher(model: YOLOE, classes: dict[str, list[str]], distractors: list[str],
                prototypes: dict[str, torch.Tensor]) -> torch.Tensor:
    """선생님에게 찾을 이름(과 프로토타입)을 넣고, 결과 번호 → CLASSES 번호 표를 돌려준다 (헷갈림용은 -1).

    classes는 CLASSES의 일부여도 된다 (06_open_data.py는 공개 데이터에 라벨이 없는 클래스만 채운다).
    """
    names = list(CLASSES)
    prompts = [p for ps in classes.values() for p in ps] + distractors
    prompt_to_class = [names.index(c) for c, ps in classes.items() for _ in ps] + [-1] * len(distractors)
    pe = model.get_text_pe(prompts)
    labels = list(prompts)
    extra = [c for c in classes if c in prototypes]
    if extra:
        pe = torch.cat([pe, torch.stack([prototypes[c] for c in extra])[None].to(pe.device, pe.dtype)], dim=1)
        labels += [f"<{c} 프로토타입>" for c in extra]
        prompt_to_class += [names.index(c) for c in extra]
    model.set_classes(labels, pe)
    return torch.tensor(prompt_to_class)


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
    """영상 단위로 train/val을 나눈다. VAL_VIDEOS와 이름이 test_로 시작하는 영상만 val이다.

    같은 영상의 이웃 프레임은 거의 똑같아서, 무작위로 섞어 나누면 val 점수가 부풀려진다.
    val을 고정해야 앱 데이터가 계속 쌓여도 모델끼리 같은 기준으로 비교할 수 있다.
    """
    split = {}
    for p in paths:
        video = p.stem.rsplit("_", 1)[0]
        split[p] = "val" if video in VAL_VIDEOS or video.startswith("test_") else "train"
    return split


def clean(boxes: torch.Tensor, scores: torch.Tensor, cls: torch.Tensor,
          monitor: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """선생님 결과를 라벨로 쓸 수 있게 정리한다.

    1. 같은 물체에 우리 클래스와 헷갈림용 이름이 함께 붙으면 점수가 높은 쪽만 남기고, 헷갈림용 박스는 버린다
    2. 이름이 달라도 같은 클래스면, 같은 물체를 여러 번 찾은 박스를 하나만 남긴다
    3. 모니터 화면 속 장면(영상 속 사람·소파 등)은 실제 물건이 아니므로 버린다
    """
    keep = nms(boxes, scores, iou_threshold=0.7)
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    real = cls >= 0
    boxes, scores, cls = boxes[real], scores[real], cls[real]

    keep = batched_nms(boxes, scores, cls, iou_threshold=0.5)
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    screens = boxes[cls == monitor]
    if len(screens) and len(boxes):
        # 각 박스가 모니터 박스와 겹치는 넓이 / 자기 넓이
        lt = torch.max(boxes[:, None, :2], screens[None, :, :2])
        rb = torch.min(boxes[:, None, 2:], screens[None, :, 2:])
        inter = (rb - lt).clamp(min=0).prod(dim=2)
        area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1).clamp(min=1)
        inside = (inter / area[:, None]).max(dim=1).values >= SCREEN_CONTAINED
        keep = ~inside | (cls == monitor)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    return boxes, scores, cls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="선생님 모델 (클수록 정확하고 느리다)")
    parser.add_argument("--conf", type=float, default=0.2, help="이보다 확신이 낮은 박스는 라벨로 쓰지 않는다")
    args = parser.parse_args()

    names = list(CLASSES)
    model = YOLOE(args.model)
    prototypes = load_prototypes()
    if prototypes:
        print(f"프로토타입 함께 사용: {', '.join(prototypes)}")
    # 헷갈림용 이름은 -1로 표시했다가 나중에 버린다
    prompt_to_class = set_teacher(model, CLASSES, DISTRACTORS, prototypes)

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
        boxes, scores, cls = clean(boxes, scores, prompt_to_class[pids], names.index("monitor"))

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
