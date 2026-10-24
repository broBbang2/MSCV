"""공개 데이터셋(COCO, Open Images)에서 우리 클래스가 있는 사진만 골라 받아 학습 데이터에 보탠다.

공개 데이터셋은 자기 클래스만 라벨이 달려 있다. 예를 들어 COCO 사진에 헤드폰이 찍혀 있어도 라벨이 없다.
그대로 섞으면 모델이 "헤드폰 = 배경"으로 배우므로, 그 데이터셋에 없는 클래스는 선생님 모델(YOLOE)이 채운다.

결과는 data/datasets/open/에, 내 영상 데이터와 합친 설정은 data/datasets/mine_open.yaml에 저장된다.
val은 내 영상(mine/images/val)만 쓴다. 우리가 잘해야 하는 건 공개 사진이 아니라 내 카메라 화면이기 때문이다.

실행 예:
    uv run scripts/06_open_data.py                  # 클래스당 최대 200장
    uv run scripts/06_open_data.py --per-class 400
    uv run scripts/03_train.py --data mine_open --name open_n --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "datasets"
CACHE = DATASETS_DIR / "_cache"
OUT_DIR = DATASETS_DIR / "open"

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OI_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv",
}
OI_CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"

# 공개 데이터셋 클래스 이름 → 우리 클래스 이름
COCO_MAP = {
    "tv": "monitor", "laptop": "laptop", "keyboard": "keyboard", "bottle": "bottle", "book": "book",
    "mouse": "mouse", "cell phone": "phone", "cup": "cup", "backpack": "backpack", "chair": "chair",
    "couch": "sofa",
}
OI_MAP = {
    "Computer monitor": "monitor", "Headphones": "headphones", "Footwear": "shoes", "Doll": "figure",
    "Hat": "cap", "Mug": "cup", "Coffee cup": "cup",
}
MAX_SIDE = 1024  # 디스크를 아끼려고 긴 변을 이 크기로 줄여서 저장한다
MIN_BOX = 0.01  # 가로나 세로가 사진의 1%보다 작은 박스는 버린다


def our_classes() -> dict[str, list[str]]:
    """04_auto_label.py의 CLASSES(클래스 → YOLOE 이름들)를 그대로 가져온다."""
    spec = importlib.util.spec_from_file_location("auto_label", ROOT / "scripts" / "04_auto_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CLASSES


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"받는 중: {url}")
        urllib.request.urlretrieve(url, path)
    return path


def coco_items(per_class: int, rng: random.Random) -> list[dict]:
    """COCO train2017에서 클래스마다 최대 per_class장을 골라 (url, 박스 목록)으로 돌려준다."""
    ann_path = CACHE / "annotations" / "instances_train2017.json"
    if not ann_path.exists():
        zip_path = fetch(COCO_ANN_URL, CACHE / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_train2017.json", CACHE)
        zip_path.unlink()
    coco = json.loads(ann_path.read_text())
    cat = {c["id"]: COCO_MAP[c["name"]] for c in coco["categories"] if c["name"] in COCO_MAP}
    images = {im["id"]: im for im in coco["images"]}

    boxes = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] not in cat or a["iscrowd"]:
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if w / im["width"] < MIN_BOX or h / im["height"] < MIN_BOX:
            continue
        boxes[a["image_id"]].append((cat[a["category_id"]], x / im["width"], y / im["height"],
                                     (x + w) / im["width"], (y + h) / im["height"]))
    return pick(
        {f"coco_{i}": {"url": images[i]["coco_url"], "boxes": b} for i, b in boxes.items()},
        set(COCO_MAP.values()), per_class, rng,
    )


def oi_items(per_class: int, rng: random.Random) -> list[dict]:
    """Open Images validation/test에서 클래스마다 최대 per_class장을 고른다. 그림·묶음 박스는 뺀다."""
    names = {r[0]: r[1] for r in csv.reader(fetch(OI_CLASSES_URL, CACHE / "oidv7-class-descriptions-boxable.csv").open())}
    items = {}
    for split, url in OI_URLS.items():
        for r in csv.DictReader(fetch(url, CACHE / Path(url).name).open()):
            ours = OI_MAP.get(names.get(r["LabelName"], ""))
            if ours is None or r["IsGroupOf"] == "1" or r["IsDepiction"] == "1":
                continue
            x1, x2, y1, y2 = (float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
            if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
                continue
            key = f"oi_{r['ImageID']}"
            items.setdefault(key, {"url": OI_IMAGE_URL.format(split=split, id=r["ImageID"]), "boxes": []})
            items[key]["boxes"].append((ours, x1, y1, x2, y2))
    return pick(items, set(OI_MAP.values()), per_class, rng)


def pick(items: dict[str, dict], classes: set[str], per_class: int, rng: random.Random) -> list[dict]:
    """클래스마다 그 클래스가 있는 사진을 무작위로 최대 per_class장 고른다 (사진 중복은 한 번만)."""
    by_class = defaultdict(list)
    for key, it in items.items():
        for c in {b[0] for b in it["boxes"]}:
            by_class[c].append(key)
    chosen = set()
    for c in sorted(classes):
        keys = sorted(by_class[c])
        rng.shuffle(keys)
        chosen.update(keys[:per_class])
    return [{"key": k, **items[k]} for k in sorted(chosen)]


def download(item: dict) -> Path | None:
    """사진을 받아 긴 변을 MAX_SIDE로 줄여 저장한다. 실패하면 None."""
    out = OUT_DIR / "images" / "train" / f"{item['key']}.jpg"
    if out.exists():
        return out
    try:
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(item["url"], tmp)
        img = cv2.imread(str(tmp))
        tmp.unlink()
        if img is None:
            return None
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return out
    except Exception as e:  # 네트워크 오류 하나로 전체를 멈추지 않는다
        print(f"  실패 {item['key']}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--per-class", type=int, default=200, help="공개 데이터셋 클래스마다 최대 몇 장")
    parser.add_argument("--teacher", default="yoloe-26l-seg.pt")
    parser.add_argument("--conf", type=float, default=0.3, help="선생님이 채우는 박스의 최소 확신도")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    classes = our_classes()
    names = list(classes)
    rng = random.Random(args.seed)
    sources = {"coco": (coco_items(args.per_class, rng), set(COCO_MAP.values())),
               "oi": (oi_items(args.per_class, rng), set(OI_MAP.values()))}

    shutil.rmtree(OUT_DIR / "labels", ignore_errors=True)
    for sub in ("images/train", "labels/train"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    teacher = YOLOE(args.teacher)
    counts, n_img = Counter(), 0
    for src, (items, covered) in sources.items():
        print(f"{src}: {len(items)}장 받는 중")
        with ThreadPoolExecutor(16) as pool:
            paths = list(pool.map(download, items))

        # 이 데이터셋에 라벨이 없는 클래스만 선생님에게 맡긴다
        fill = [c for c in names if c not in covered]
        prompts = [p for c in fill for p in classes[c]]
        prompt_to_class = [c for c in fill for _ in classes[c]]
        teacher.set_classes(prompts)

        for item, path in zip(items, paths):
            if path is None:
                continue
            lines = []
            for c, x1, y1, x2, y2 in item["boxes"]:
                lines.append(f"{names.index(c)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                counts[f"{c} ({src})"] += 1
            r = teacher.predict(str(path), conf=args.conf, verbose=False)[0].boxes
            for (cx, cy, w, h), p in zip(r.xywhn.tolist(), r.cls.int().tolist()):
                c = prompt_to_class[p]
                lines.append(f"{names.index(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                counts[f"{c} (선생님)"] += 1
            (OUT_DIR / "labels" / "train" / f"{path.stem}.txt").write_text("\n".join(lines))
            n_img += 1

    mine = yaml.safe_load((DATASETS_DIR / "mine.yaml").read_text())
    merged = {
        "path": str(DATASETS_DIR),
        "train": ["mine/images/train", "open/images/train"],
        "val": "mine/images/val",
        "names": mine["names"],
    }
    (DATASETS_DIR / "mine_open.yaml").write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False))
    print(f"공개 데이터 {n_img}장 준비 완료 → {OUT_DIR}")
    for k, v in sorted(counts.items()):
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
