"""공개 데이터의 사람 라벨로 클래스별 프로토타입(예시 박스 임베딩의 평균)을 만들고, 선생님에게 쓸지 채점해서 정한다.

선생님(YOLOE)은 글자 대신 "이런 모양" 예시를 받으면 글자로 설명하기 어려운 물건(계단, 기둥, 라바콘)을 더 잘 찾기도 한다.
2026-10-26 실험에서 피규어·장난감 자동차는 글자 + 프로토타입이 AP50 0.24 → 0.59, 0.10 → 0.54로 올랐다.

1. open/images/train의 사람 박스에서 클래스마다 최대 --shots장의 임베딩을 뽑아 평균 낸다
2. open/images/val에서 글자만 / 글자 + 프로토타입으로 각각 찾아 클래스별 AP50을 잰다
   (그 사진의 출처가 사람 라벨을 단 클래스만 채점한다. 다른 클래스는 정답을 몰라서 오답을 셀 수 없다)
3. 프로토타입을 더해 AP50이 --min-gain 이상 오른 클래스만 data/prototypes/teacher.pt에 저장한다
   04_auto_label.py와 06_open_data.py가 이 파일이 있으면 그 클래스에 프로토타입을 함께 쓴다

06_open_data.py --download-only로 사진을 먼저 받아 둔 뒤 돌린다.

실행 예:
    uv run scripts/10_prototypes.py
    uv run scripts/10_prototypes.py --shots 60 --min-gain 0.03
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict import YOLOEVPDetectPredictor
from ultralytics.utils.metrics import compute_ap

ROOT = Path(__file__).resolve().parents[1]
OPEN_DIR = ROOT / "data" / "datasets" / "open"
OUT = ROOT / "data" / "prototypes" / "teacher.pt"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


auto = load_script("04_auto_label.py")
open_data = load_script("06_open_data.py")


def human_boxes(seed: int, per_class: int) -> dict[str, dict]:
    """06과 같은 방법으로 고른 공개 사진의 사람 라벨. {사진 이름: {"covered": 클래스들, "boxes": [(클래스, xyxy 0~1)]}}"""
    coco = open_data.coco_all()
    lvis = open_data.lvis_items(coco, per_class, seed)
    items = lvis + open_data.coco_items(coco, per_class, seed, skip={it["key"] for it in lvis})
    items += open_data.oi_items(per_class, seed)
    return {it["key"]: {"covered": it["covered"], "boxes": [(b[0], b[1:]) for b in it["boxes"]]} for it in items}


def image_path(key: str, split: str) -> Path:
    return OPEN_DIR / "images" / split / f"{key}.jpg"


def make_prototypes(model: YOLOE, labels: dict[str, dict], shots: int, seed: int) -> dict[str, torch.Tensor]:
    vp = YOLOEVPDetectPredictor(overrides={"task": "detect", "mode": "predict", "imgsz": 640, "verbose": False})
    vp.setup_model(model.model)
    by_class = defaultdict(list)
    for key, it in labels.items():
        if image_path(key, "train").exists():
            for c in {c for c, _ in it["boxes"]}:
                by_class[c].append(key)

    rng = np.random.default_rng(seed)
    out = {}
    for c in sorted(by_class):
        keys = rng.permutation(sorted(by_class[c]))[:shots]
        vecs = []
        for key in keys:
            path = image_path(key, "train")
            h, w = cv2.imread(str(path)).shape[:2]
            boxes = [[x1 * w, y1 * h, x2 * w, y2 * h] for lab, (x1, y1, x2, y2) in labels[key]["boxes"] if lab == c]
            vp.set_prompts({"bboxes": np.array(boxes, dtype=np.float32), "cls": np.zeros(len(boxes), dtype=int)})
            vecs.append(vp.get_vpe(str(path))[0, 0])  # 같은 클래스 박스들은 하나로 묶여 나온다
        m = torch.stack(vecs).mean(0)
        out[c] = (m / m.norm()).cpu()
        print(f"  {c:14s} 예시 사진 {len(vecs)}장")
    return out


def evaluate(model: YOLOE, protos: dict[str, torch.Tensor], labels: dict[str, dict]) -> dict[str, tuple[float, int]]:
    """open/images/val에서 클래스별 (AP50, 정답 박스 수)."""
    names = list(auto.CLASSES)
    p2c = auto.set_teacher(model, auto.CLASSES, auto.DISTRACTORS, protos)
    res = defaultdict(lambda: {"conf": [], "tp": [], "n": 0})
    for key, it in labels.items():
        path = image_path(key, "val")
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        r = model.predict(img, conf=0.05, verbose=False)[0].boxes
        b, s, c = auto.clean(r.xyxy.cpu(), r.conf.cpu(), p2c[r.cls.cpu().long()], names.index("monitor"))
        for cname in it["covered"]:
            ci = names.index(cname)
            gts = torch.tensor([[x1 * w, y1 * h, x2 * w, y2 * h] for lab, (x1, y1, x2, y2) in it["boxes"]
                                if lab == cname] or np.zeros((0, 4)), dtype=torch.float32)
            res[cname]["n"] += len(gts)
            pb, ps = b[c == ci], s[c == ci]
            used = set()
            for i in ps.argsort(descending=True).tolist():
                hit = False
                if len(gts):
                    ious = box_iou(pb[i:i + 1], gts)[0]
                    for j in ious.argsort(descending=True).tolist():
                        if ious[j] < 0.5:
                            break
                        if j not in used:
                            used.add(j)
                            hit = True
                            break
                res[cname]["conf"].append(float(ps[i]))
                res[cname]["tp"].append(int(hit))
    out = {}
    for cname, v in res.items():
        if v["n"] == 0:
            continue
        o = np.argsort(-np.array(v["conf"]))
        tp = np.array(v["tp"], dtype=float)[o]
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        ap = compute_ap(ctp / v["n"], ctp / np.maximum(ctp + cfp, 1))[0] if len(tp) else 0.0
        out[cname] = (float(ap), v["n"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--teacher", default="yoloe-26l-seg.pt")
    parser.add_argument("--shots", type=int, default=40, help="클래스마다 프로토타입에 쓸 예시 사진 수")
    parser.add_argument("--min-gain", type=float, default=0.02, help="이만큼 AP50이 올라야 프로토타입을 쓴다")
    parser.add_argument("--min-boxes", type=int, default=10, help="채점 박스가 이보다 적으면 판단하지 않는다")
    parser.add_argument("--per-class", type=int, default=200, help="06_open_data.py와 같은 값")
    parser.add_argument("--seed", type=int, default=0, help="06_open_data.py와 같은 값")
    args = parser.parse_args()

    labels = human_boxes(args.seed, args.per_class)
    model = YOLOE(args.teacher)
    print("프로토타입 만드는 중")
    protos = make_prototypes(model, labels, args.shots, args.seed)

    print("채점 중 (open/images/val)")
    text = evaluate(model, {}, labels)
    both = evaluate(model, protos, labels)
    chosen = {}
    print(f"\n  {'클래스':14s} {'박스':>5s} {'글자':>6s} {'+프로토':>7s}")
    for c in sorted(text, key=lambda c: both[c][0] - text[c][0], reverse=True):
        n = text[c][1]
        gain = both[c][0] - text[c][0]
        use = n >= args.min_boxes and gain >= args.min_gain and c in protos
        if use:
            chosen[c] = protos[c]
        print(f"  {c:14s} {n:5d} {text[c][0]:6.2f} {both[c][0]:7.2f}  {'← 사용' if use else ''}")
    t_mean = np.mean([v[0] for v in text.values()])
    b_mean = np.mean([text[c][0] if c not in chosen else both[c][0] for c in text])
    print(f"\n  평균 AP50 글자만 {t_mean:.3f} → 고른 클래스만 프로토타입 {b_mean:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chosen, OUT)
    print(f"프로토타입 {len(chosen)}개 저장: {OUT}")


if __name__ == "__main__":
    main()
