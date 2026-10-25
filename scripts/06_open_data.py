"""공개 데이터셋(LVIS, COCO, Open Images)에서 우리 클래스가 있는 사진만 골라 받아 학습 데이터에 보탠다.

공개 데이터셋은 자기 클래스만 라벨이 달려 있다. 예를 들어 COCO 사진에 헤드폰이 찍혀 있어도 라벨이 없다.
그대로 섞으면 모델이 "헤드폰 = 배경"으로 배우므로, 그 데이터셋에 없는 클래스는 선생님 모델(YOLOE)이 채운다.

LVIS는 Hugging Face(finedet/lvis, 19.6GB)에 사진과 함께 올라와 있지만, 라벨 열만 원격으로 읽고
사진은 COCO 서버에서 필요한 것만 받는다. LVIS 사진은 원래 COCO 사진이라 COCO 라벨도 함께 붙인다.

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
import http.client
import importlib.util
import json
import random
import shutil
import socket
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem
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
LVIS_REPO = "datasets/finedet/lvis"

# 공개 데이터셋 클래스 이름 → 우리 클래스 이름
# book은 일부러 뺐다. COCO는 책장 속 책을 한 권씩 작게 달아서 우리 라벨 기준과 달라, 선생님이 같은 기준으로 채운다.
COCO_MAP = {
    "tv": "monitor", "laptop": "laptop", "keyboard": "keyboard", "bottle": "bottle",
    "mouse": "mouse", "cell phone": "phone", "cup": "cup", "backpack": "backpack", "chair": "chair",
    "couch": "sofa", "person": "person",
}
OI_MAP = {
    "Computer monitor": "monitor", "Headphones": "headphones", "Footwear": "shoes", "Doll": "figure",
    "Hat": "cap", "Mug": "cup", "Coffee cup": "cup",
}
# 셔츠·재킷은 넣지 않는다. 사람이 입은 옷까지 clothes로 배우면 사람만 비춰도 옷 박스가 생긴다.
LVIS_MAP = {
    "figurine": "figure", "speaker_(stero_equipment)": "speaker", "baseball_cap": "cap",
    "cap_(headwear)": "cap", "shoe": "shoes", "jersey": "clothes", "headset": "headphones",
    "earphone": "headphones",
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


def coco_all() -> dict[str, dict]:
    """COCO train2017에서 우리 클래스 박스가 있는 사진 전체를 {파일 이름: {url, boxes}}로 돌려준다."""
    ann_path = CACHE / "annotations" / "instances_train2017.json"
    if not ann_path.exists():
        zip_path = fetch(COCO_ANN_URL, CACHE / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_train2017.json", CACHE)
        zip_path.unlink()
    coco = json.loads(ann_path.read_text())
    cat = {c["id"]: COCO_MAP[c["name"]] for c in coco["categories"] if c["name"] in COCO_MAP}
    images = {im["id"]: im for im in coco["images"]}

    items = {}
    for a in coco["annotations"]:
        if a["category_id"] not in cat or a["iscrowd"]:
            continue
        im = images[a["image_id"]]
        box = to_box(cat[a["category_id"]], *a["bbox"], im["width"], im["height"])
        if box:
            items.setdefault(im["file_name"], {"url": im["coco_url"], "boxes": []})["boxes"].append(box)
    return items


def to_box(c: str, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple | None:
    """픽셀 xywh 박스를 0~1 xyxy로 바꾼다. 너무 작으면 None."""
    if w / img_w < MIN_BOX or h / img_h < MIN_BOX:
        return None
    return (c, x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h)


def lvis_items(coco: dict[str, dict], per_class: int, rng: random.Random) -> list[dict]:
    """Hugging Face의 LVIS에서 라벨 열만 읽어 고르고, 같은 사진의 COCO 라벨을 합친다."""
    fs = HfFileSystem()
    cat_path = fetch(f"https://huggingface.co/{LVIS_REPO}/resolve/main/categories.csv", CACHE / "lvis_categories.csv")
    cats = {int(r["id"]): LVIS_MAP[r["name"]] for r in csv.DictReader(cat_path.open()) if r["name"] in LVIS_MAP}

    def read(path: str) -> list[tuple]:
        # 사진 bytes는 건너뛰고 파일 이름과 라벨만 읽는다 (파일 하나에 1MB 남짓)
        with fs.open(path, block_size=2**20, cache_type="none") as f:
            t = pq.ParquetFile(f).read(columns=["image.path", "width", "height", "objects"])
        names = t.column("image").combine_chunks().field("path").to_pylist()
        return list(zip(names, t.column("width").to_pylist(), t.column("height").to_pylist(),
                        t.column("objects").to_pylist()))

    # Hugging Face 원격 읽기가 수십 분씩 걸릴 때가 있어서, 우리 클래스가 있는 행만 한 번 저장해 두고 재사용한다.
    # LVIS_MAP이 바뀌면 고르는 클래스가 달라지므로 캐시 이름에 클래스 번호를 넣는다.
    rows_path = CACHE / f"lvis_rows_{'-'.join(map(str, sorted(cats)))}.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    else:
        files = sorted(fs.glob(f"{LVIS_REPO}/data/train-*.parquet"))
        print(f"LVIS 라벨 읽는 중: 파일 {len(files)}개")
        with ThreadPoolExecutor(8) as pool:
            rows = [r for rs in pool.map(read, files) for r in rs if set(r[3]["category"]) & cats.keys()]
        rows_path.write_text(json.dumps(rows))

    items = {}
    for name, w, h, objs in rows:
        boxes = [b for b in (to_box(cats[c], *bb, w, h) for bb, c in zip(objs["bbox"], objs["category"]) if c in cats) if b]
        if boxes:
            base = coco.get(name, {"url": f"http://images.cocodataset.org/train2017/{name}", "boxes": []})
            items[f"coco_{Path(name).stem}"] = {"url": base["url"], "boxes": boxes + base["boxes"]}
    return pick(items, set(LVIS_MAP.values()), per_class, rng, covered=set(LVIS_MAP.values()) | set(COCO_MAP.values()))


def coco_items(coco: dict[str, dict], per_class: int, rng: random.Random, skip: set[str]) -> list[dict]:
    """COCO에서 클래스마다 최대 per_class장. LVIS에서 이미 고른 사진은 뺀다."""
    items = {f"coco_{Path(n).stem}": it for n, it in coco.items() if f"coco_{Path(n).stem}" not in skip}
    return pick(items, set(COCO_MAP.values()), per_class, rng, covered=set(COCO_MAP.values()))


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
    return pick(items, set(OI_MAP.values()), per_class, rng, covered=set(OI_MAP.values()))


def pick(items: dict[str, dict], classes: set[str], per_class: int, rng: random.Random,
         covered: set[str]) -> list[dict]:
    """클래스마다 그 클래스가 있는 사진을 무작위로 최대 per_class장 고른다 (사진 중복은 한 번만).

    covered는 이 사진들에서 사람이 라벨을 단 클래스. 나머지 클래스는 선생님이 채운다.
    """
    by_class = defaultdict(list)
    for key, it in items.items():
        for c in {b[0] for b in it["boxes"]}:
            by_class[c].append(key)
    chosen = set()
    for c in sorted(classes):
        keys = sorted(by_class[c])
        rng.shuffle(keys)
        chosen.update(keys[:per_class])
    return [{"key": k, "covered": covered, **items[k]} for k in sorted(chosen)]


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
    except (OSError, http.client.HTTPException) as e:  # 네트워크 오류 하나로 전체를 멈추지 않는다
        print(f"  실패 {item['key']}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--per-class", type=int, default=200, help="공개 데이터셋 클래스마다 최대 몇 장")
    parser.add_argument("--teacher", default="yoloe-26l-seg.pt")
    parser.add_argument("--conf", type=float, default=0.3, help="선생님이 채우는 박스의 최소 확신도")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mine-repeat", type=int, default=4,
                        help="학습 목록에 내 영상을 몇 번 넣을지. 공개 사진에 묻히지 않게 비중을 높인다")
    args = parser.parse_args()
    # 타임아웃이 없으면 서버가 응답 없이 연결만 붙잡고 있을 때 다운로드 스레드가 영원히 멈춘다
    socket.setdefaulttimeout(60)

    classes = our_classes()
    names = list(classes)
    rng = random.Random(args.seed)
    coco = coco_all()
    lvis = lvis_items(coco, args.per_class, rng)
    sources = {
        "lvis": lvis,
        "coco": coco_items(coco, args.per_class, rng, skip={it["key"] for it in lvis}),
        "oi": oi_items(args.per_class, rng),
    }

    # 예전 실행의 라벨과 사진을 지우고 새로 만든다 (사진은 이번에 고른 것만 남긴다)
    keep = {it["key"] for items in sources.values() for it in items}
    shutil.rmtree(OUT_DIR / "labels", ignore_errors=True)
    for sub in ("images/train", "labels/train"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    for old in (OUT_DIR / "images" / "train").glob("*.jpg"):
        if old.stem not in keep:
            old.unlink()

    teacher = YOLOE(args.teacher)
    counts, n_img = Counter(), 0
    for src, items in sources.items():
        print(f"{src}: {len(items)}장 받는 중")
        with ThreadPoolExecutor(16) as pool:
            paths = list(pool.map(download, items))

        # 이 데이터셋에서 사람이 라벨을 달지 않은 클래스만 선생님에게 맡긴다
        fill = [c for c in names if c not in items[0]["covered"]] if items else []
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
        "train": ["mine/images/train"] * args.mine_repeat + ["open/images/train"],
        "val": "mine/images/val",
        "names": mine["names"],
    }
    (DATASETS_DIR / "mine_open.yaml").write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False))
    print(f"공개 데이터 {n_img}장 준비 완료 → {OUT_DIR}")
    for k, v in sorted(counts.items()):
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
