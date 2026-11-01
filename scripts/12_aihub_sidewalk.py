"""AI Hub "인도보행 영상"(데이터셋 189)을 묶음 하나씩 받아, 필요한 사진만 줄여 남기고 원본은 지운다.

한국 인도에서 찍은 시각장애인 보행 보조용 데이터다. 바운딩박스 묶음 30개(각 약 10GB, 사진 약 2만 장)에
장애물 29종(볼라드, 킥보드, 가로수, 입간판, 신호등 ...)이 CVAT XML로 달려 있다.
내국인만 신청할 수 있고 다시 배포하면 안 되므로 data/aihub/는 git에 올리지 않는다.

받은 파일은 tar 안에 zip이 1GB 조각(.part<바이트 위치>)으로 들어 있다. 풀면 디스크가 두 배로 들어서,
tar 안의 조각들을 이어 붙인 하나의 파일처럼 읽어 zip을 바로 연다.

서피스마스킹 묶음 5개(각 10GB, 사진 약 1만 장)에는 바닥 영역(점자블록, 횡단보도, 맨홀, 계단 ...)이
폴리곤으로 달려 있다. 폴리곤 점과 그것을 감싸는 박스를 함께 적는다 (data/aihub/surface/).

남기는 것 (data/aihub/bbox/, surface/)
- 동영상 프레임이라 앞뒤 사진이 거의 같다. 폴더마다 --every장에 1장만 남긴다
- 드문 클래스(RARE)가 있는 사진은 전부 남긴다
- 긴 변 --size로 줄인 jpg와, 원래 이름 그대로의 박스(0~1 좌표)를 labels.jsonl에 적는다.
  우리 클래스로 바꾸는 일은 학습 데이터를 만들 때 한다 (어떤 클래스를 쓸지 아직 정하는 중이라)

API 키는 ~/.aihub_header("apikey:<키>", 본인만 읽기)에서 읽는다. 키는 화면에 찍지 않는다.

실행 예:
    uv run scripts/12_aihub_sidewalk.py list                  # 받을 수 있는 묶음과 filekey
    uv run scripts/12_aihub_sidewalk.py fetch Bbox_2          # 받기 → 줄여 남기기 → 원본 지우기
    uv run scripts/12_aihub_sidewalk.py extract data/aihub/_dl/Bbox_1.tar   # 이미 받은 tar만 처리
    uv run scripts/12_aihub_sidewalk.py stats                 # 지금까지 남긴 클래스별 박스 수
    uv run scripts/12_aihub_sidewalk.py build                 # YOLO 학습 형식으로 (data/datasets/aihub/)
    uv run scripts/12_aihub_sidewalk.py floor-data            # 점자블록·횡단보도 전용 작은 모델 데이터
    uv run scripts/12_aihub_sidewalk.py floor-label runs/floor_n/weights/best.pt   # 그 모델로 다른 사진에 채우기
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import io
import json
import re
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "aihub"
DL_DIR = OUT_DIR / "_dl"
DATASET_DIR = ROOT / "data" / "datasets" / "aihub"
HEADER = Path.home() / ".aihub_header"
DATASET = 189
API = "https://api.aihub.or.kr"
# AI Hub 이름 → 우리 클래스. 여기 있는 클래스는 사람이 단 라벨을 쓰고, 나머지는 선생님이 채운다
BBOX_MAP = {"person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle", "car": "car", "bus": "car",
            "truck": "car", "bench": "bench", "chair": "chair", "table": "table", "pole": "pole",
            "bollard": "bollard", "tree_trunk": "tree trunk"}
SURFACE_MAP = {"braille_guide_blocks/": "braille block", "roadway/crosswalk": "crosswalk",
               "alley/crosswalk": "crosswalk", "caution_zone/stairs": "stairs"}
# 인도 사진에 거의 없는 실내 물건. 선생님에게 맡기면 광고판 속 사람을 피규어로, 간판을 모니터로 잡는다
INDOOR = {"monitor", "laptop", "keyboard", "figure", "toy car", "tumbler", "speaker", "headphones", "book", "mouse",
          "sofa"}
VAL_EVERY = 10  # 폴더(한 번 찍은 영상) 10개 중 1개를 검증용으로. 같은 영상의 프레임이 학습·검증에 나뉘지 않게
# 바닥 영역 중 드물고 꼭 필요한 것 (점자블록, 횡단보도, 계단)
RARE_SURFACE = ("braille_guide_blocks/", "roadway/crosswalk", "alley/crosswalk", "caution_zone/stairs")
# 묶음에 거의 없어서 보이는 대로 다 남기는 클래스
RARE = {"scooter", "stroller", "wheelchair", "kiosk", "fire_hydrant", "barricade", "stop", "carrier", "parking_meter"}


class JoinedParts(io.RawIOBase):
    """tar 안의 <이름>.part<바이트 위치> 조각들을 이어 붙인 하나의 읽기 전용 파일."""

    def __init__(self, tar_path: Path):
        self.f = open(tar_path, "rb")  # noqa: SIM115 — close()에서 닫는다
        with tarfile.open(tar_path) as t:
            self.parts = sorted((int(m.name.rsplit(".part", 1)[1]), m.offset_data, m.size)
                                for m in t if ".part" in m.name)
        self.starts = [p[0] for p in self.parts]
        self.size = self.parts[-1][0] + self.parts[-1][2]
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        start, data_offset, size = self.parts[bisect.bisect_right(self.starts, self.pos) - 1]
        n = min(len(buf), start + size - self.pos)
        self.f.seek(data_offset + self.pos - start)
        got = self.f.readinto(memoryview(buf)[:n])
        self.pos += got
        return got

    def close(self) -> None:
        self.f.close()
        super().close()


def file_list() -> dict[str, tuple[str, str]]:
    """{묶음 이름: (크기, filekey)}. AI Hub 서버가 502를 자주 내서 몇 번 다시 묻는다."""
    for _ in range(5):
        text = subprocess.run(["curl", "-sS", f"{API}/info/{DATASET}.do"], check=False, capture_output=True,
                              text=True).stdout
        found = re.findall(r"─([^─\s]+)\.zip \| ([\d.]+ \w+) \| (\d+)", text)
        if found:
            return {name: (size, key) for name, size, key in found}
        time.sleep(5)
    raise SystemExit("AI Hub 파일 목록을 받지 못했어요. 잠시 뒤 다시 해 보세요.")


def download(name: str, filekey: str) -> Path:
    """끊기면 이어받는다. 키는 헤더 파일로 넘겨 명령줄에 남지 않게 한다."""
    if not HEADER.exists():
        raise SystemExit(f"{HEADER}가 없어요. API 키를 먼저 저장하세요.")
    DL_DIR.mkdir(parents=True, exist_ok=True)
    out = DL_DIR / f"{name.removesuffix('_new')}.tar"
    for attempt in range(20):
        r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(out), "-H", f"@{HEADER}", "-w", "%{http_code}",
                            f"{API}/down/0.6/{DATASET}.do?fileSn={filekey}"], check=False, capture_output=True,
                           text=True)
        # 연결이 중간에 끊겨도 HTTP 코드는 200으로 나온다. curl 종료 코드와 tar가 끝까지 있는지로 판단한다
        if r.stdout in ("200", "206", "416") and complete(out):  # 416: 이미 다 받았다
            return out
        if r.returncode == 33:  # 서버가 이어받기를 지원하지 않으면 처음부터
            out.unlink(missing_ok=True)
        size = out.stat().st_size / 1e9 if out.exists() else 0
        print(f"  다운로드 끊김 (HTTP {r.stdout}, curl {r.returncode}, {size:.1f}GB까지 받음), "
              f"20초 뒤 이어받기 ({attempt + 1}/20)")
        time.sleep(20)
    raise SystemExit(f"{name}을 받지 못했어요.")


def complete(tar_path: Path) -> bool:
    """tar의 마지막 파일까지 다 들어 있는지."""
    try:
        with tarfile.open(tar_path) as t:
            members = t.getmembers()
        return bool(members) and members[-1].offset_data + members[-1].size <= tar_path.stat().st_size
    except (tarfile.TarError, OSError):
        return False


def boxes_of(image: ET.Element) -> list[dict]:
    w, h = float(image.get("width")), float(image.get("height"))
    return [{"label": b.get("label"), "occluded": b.get("occluded") == "1",
             "box": [round(float(b.get(k)) / s, 5) for k, s in (("xtl", w), ("ytl", h), ("xbr", w), ("ybr", h))]}
            for b in image.iter("box")]


def polygons_of(image: ET.Element) -> list[dict]:
    """바닥 영역. label은 "종류/속성" (roadway/crosswalk, braille_guide_blocks/normal ...)."""
    w, h = float(image.get("width")), float(image.get("height"))
    out = []
    for p in image.iter("polygon"):
        attr = p.find("attribute")
        pts = [(float(x) / w, float(y) / h) for x, y in (q.split(",") for q in p.get("points").split(";"))]
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        out.append({"label": f"{p.get('label')}/{attr.text if attr is not None and attr.text else ''}",
                    "box": [round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)],
                    "points": [[round(x, 4), round(y, 4)] for x, y in pts]})
    return out


def is_rare(kind: str, labels: set[str]) -> bool:
    if kind == "bbox":
        return bool(labels & RARE)
    return any(lab.startswith(RARE_SURFACE) for lab in labels)


def extract(tar_path: Path, every: int, size: int) -> None:
    bundle = tar_path.stem  # Bbox_1, Surface_1
    kind = "bbox" if bundle.startswith("Bbox") else "surface"
    img_dir = OUT_DIR / kind / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_DIR / kind / "labels.jsonl"
    done = set()
    if labels_path.exists():
        done = {json.loads(line)["image"] for line in labels_path.open()}

    z = zipfile.ZipFile(io.BufferedReader(JoinedParts(tar_path), buffer_size=1 << 20))
    names = set(z.namelist())
    kept = Counter()
    n_images = n_kept = missing = 0
    with labels_path.open("a") as out:
        for xml_name in sorted(n for n in names if n.endswith(".xml")):
            folder = xml_name.split("/")[0]  # Bbox_0001, Surface_001
            images = sorted(ET.fromstring(z.read(xml_name)).iter("image"), key=lambda im: im.get("name"))
            for i, im in enumerate(images):
                n_images += 1
                items = boxes_of(im) if kind == "bbox" else polygons_of(im)
                if i % every and not is_rare(kind, {b["label"] for b in items}):
                    continue
                stem = f"{bundle.lower()}_{folder.split('_')[1]}_{Path(im.get('name')).stem}"
                src = f"{folder}/{im.get('name')}"
                if f"{stem}.jpg" in done:
                    continue
                if src not in names:
                    missing += 1
                    continue
                img = cv2.imdecode(np.frombuffer(z.read(src), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                scale = size / max(img.shape[:2])
                if scale < 1:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                out.write(json.dumps({"image": f"{stem}.jpg", "bundle": bundle, "folder": folder,
                                      "boxes": items}, ensure_ascii=False) + "\n")
                n_kept += 1
                kept.update(b["label"] for b in items)
            print(f"  {folder}: 지금까지 {n_kept}/{n_images}장 남김", end="\r")
    z.close()
    print(f"\n{bundle}: 사진 {n_images}장 중 {n_kept}장 남김 → {img_dir}"
          + (f"  (라벨은 있는데 사진이 없는 {missing}장 건너뜀)" if missing else ""))
    print("  " + ", ".join(f"{k} {v}" for k, v in kept.most_common()))


def stats() -> None:
    for kind in ("bbox", "surface"):
        labels_path = OUT_DIR / kind / "labels.jsonl"
        if not labels_path.exists():
            continue
        boxes, images, bundles = Counter(), Counter(), set()
        n = 0
        for line in labels_path.open():
            r = json.loads(line)
            n += 1
            bundles.add(r["bundle"])
            boxes.update(b["label"] for b in r["boxes"])
            images.update({b["label"] for b in r["boxes"]})
        print(f"[{kind}] 묶음 {sorted(bundles)}, 사진 {n}장")
        for k, v in boxes.most_common():
            print(f"  {k:30s} 개수 {v:6d}  사진 {images[k]:6d}")


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def our_class(kind: str, label: str) -> str | None:
    if kind == "bbox":
        return BBOX_MAP.get(label)
    return next((c for prefix, c in SURFACE_MAP.items() if label.startswith(prefix)), None)


def build(teacher_name: str, conf: float, rebuild: bool) -> None:
    """남겨 둔 사진을 YOLO 학습 형식(data/datasets/aihub/)으로. 사진은 복사하지 않고 링크한다."""
    from ultralytics import YOLOE

    spec = importlib.util.spec_from_file_location("auto_label", ROOT / "scripts" / "04_auto_label.py")
    auto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto)
    names = list(auto.CLASSES)
    teacher = YOLOE(teacher_name)
    prototypes = auto.load_prototypes()
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (DATASET_DIR / sub / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for kind, table in (("bbox", BBOX_MAP), ("surface", SURFACE_MAP)):
        labels_path = OUT_DIR / kind / "labels.jsonl"
        if not labels_path.exists():
            continue
        covered = set(table.values())
        fill = [c for c in names if c not in covered and c not in INDOOR]
        prompt_to_class = auto.set_teacher(teacher, {c: auto.CLASSES[c] for c in fill}, [], prototypes).tolist()
        rows = [json.loads(line) for line in labels_path.open()]
        for i, r in enumerate(rows):
            split = "val" if zlib.crc32(f"{r['bundle']}/{r['folder']}".encode()) % VAL_EVERY == 0 else "train"
            stem = Path(r["image"]).stem
            label_path = DATASET_DIR / "labels" / split / f"{stem}.txt"
            if label_path.exists() and not rebuild:
                continue
            img_path = OUT_DIR / kind / "images" / r["image"]
            lines = []
            for b in r["boxes"]:
                c = our_class(kind, b["label"])
                if c is None:
                    continue
                x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in b["box"])
                if x2 - x1 < 0.003 or y2 - y1 < 0.003:
                    continue
                lines.append(f"{names.index(c)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                counts[f"{c} (AI Hub)"] += 1
            res = teacher.predict(str(img_path), conf=conf, verbose=False)[0].boxes
            # AI Hub가 이미 이름을 붙인 물건(소화전, 입간판, 바리케이드 ...)과 겹치는 선생님 박스는 버린다.
            # 라바콘이라며 소화전·입간판을 잡는 일이 많았다
            human = [b["box"] for b in r["boxes"]] if kind == "bbox" else []
            for (cx, cy, w, h), p in zip(res.xywhn.tolist(), res.cls.int().tolist(), strict=True):
                c = names[prompt_to_class[p]]
                if any(iou((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), hb) > 0.5 for hb in human):
                    continue
                lines.append(f"{names.index(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                counts[f"{c} (선생님)"] += 1
            link = DATASET_DIR / "images" / split / r["image"]
            if not link.exists():
                link.symlink_to(img_path)
            label_path.write_text("\n".join(lines))
            if i % 200 == 0:
                print(f"  {kind}: {i}/{len(rows)}장", end="\r")
        print(f"  {kind}: {len(rows)}장 완료        ")
    n = {split: len(list((DATASET_DIR / "labels" / split).glob("*.txt"))) for split in ("train", "val")}
    print(f"AI Hub 학습 데이터 → {DATASET_DIR}  (학습 {n['train']}장, 검증 {n['val']}장)")
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v}")


FLOOR = ["braille block", "crosswalk"]
FLOOR_DIR = ROOT / "data" / "datasets" / "floor"


def floor_data() -> None:
    """점자블록·횡단보도만 보는 작은 모델용 데이터 (서피스 사진, 클래스 2개).

    선생님(YOLOE)이 이 둘을 못 찾아서, 이 모델을 먼저 학습해 다른 사진에 이 둘을 채운다 (floor-label).
    """
    import yaml

    names = class_names()
    ids = {names.index(c): i for i, c in enumerate(FLOOR)}
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (FLOOR_DIR / sub / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for label in (DATASET_DIR / "labels" / split).glob("surface_*.txt"):
            lines = [f"{ids[int(line.split()[0])]} {line.split(' ', 1)[1]}" for line in label.read_text().splitlines()
                     if int(line.split()[0]) in ids]
            (FLOOR_DIR / "labels" / split / label.name).write_text("\n".join(lines))
            link = FLOOR_DIR / "images" / split / f"{label.stem}.jpg"
            if not link.exists():
                link.symlink_to((DATASET_DIR / "images" / split / f"{label.stem}.jpg").resolve())
            n += 1
        print(f"  {split}: {n}장")
    (FLOOR_DIR.parent / "floor.yaml").write_text(yaml.safe_dump(
        {"path": str(FLOOR_DIR), "train": "images/train", "val": "images/val", "names": dict(enumerate(FLOOR))},
        allow_unicode=True, sort_keys=False))
    print(f"→ {FLOOR_DIR.parent / 'floor.yaml'}")


def floor_label(weights: str, conf: float) -> None:
    """바닥 모델로 사람 바닥 라벨이 없는 사진(AI Hub 바운딩박스, 공개 데이터, 내 영상)에 점자블록·횡단보도를 채운다.

    다시 돌려도 되게, 이 두 클래스의 예전 줄은 지우고 새로 쓴다. 서피스 사진(사람 라벨)은 건드리지 않는다.
    """
    from ultralytics import YOLO

    names = class_names()
    ours = [names.index(c) for c in FLOOR]
    model = YOLO(weights)
    counts = Counter()
    labels = [p for d in ("aihub", "open", "mine") for p in (ROOT / "data" / "datasets" / d / "labels").glob("*/*.txt")
              if not p.name.startswith("surface_")]
    for i, label in enumerate(labels):
        image = Path(str(label.parent).replace("/labels/", "/images/")) / f"{label.stem}.jpg"
        if not image.exists():
            continue
        keep = [line for line in label.read_text().splitlines() if line and int(line.split()[0]) not in ours]
        r = model.predict(str(image), conf=conf, verbose=False)[0].boxes
        for (cx, cy, w, h), c in zip(r.xywhn.tolist(), r.cls.int().tolist(), strict=True):
            keep.append(f"{ours[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[f"{FLOOR[c]} ({label.parts[-4]})"] += 1
        label.write_text("\n".join(keep))
        if i % 500 == 0:
            print(f"  {i}/{len(labels)}장", end="\r")
    print(f"바닥 모델로 {len(labels)}장 채움 (conf {conf})")
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v}")


def class_names() -> list[str]:
    spec = importlib.util.spec_from_file_location("auto_label", ROOT / "scripts" / "04_auto_label.py")
    auto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto)
    return list(auto.CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="받을 수 있는 묶음과 filekey")
    for name in ("fetch", "extract"):
        p = sub.add_parser(name)
        p.add_argument("target", help="fetch: 묶음 이름(Bbox_2) / extract: 받은 tar 경로")
        p.add_argument("--every", type=int, default=4, help="폴더마다 이만큼에 1장 남긴다 (드문 클래스는 전부)")
        p.add_argument("--size", type=int, default=1280, help="남길 사진의 긴 변")
        p.add_argument("--keep-tar", action="store_true", help="처리한 뒤 받은 원본을 지우지 않는다")
    sub.add_parser("stats", help="지금까지 남긴 클래스별 박스 수")
    p = sub.add_parser("build", help="남긴 사진을 YOLO 학습 형식으로 (없는 클래스는 선생님이 채움)")
    p.add_argument("--teacher", default="yoloe-26l-seg.pt")
    p.add_argument("--conf", type=float, default=0.3, help="선생님이 채우는 박스의 최소 확신도 (06과 같은 값)")
    p.add_argument("--rebuild", action="store_true", help="이미 만든 라벨도 다시 만든다")
    sub.add_parser("floor-data", help="점자블록·횡단보도만 보는 작은 모델용 데이터 (data/datasets/floor.yaml)")
    p = sub.add_parser("floor-label", help="바닥 모델로 다른 사진에 점자블록·횡단보도를 채운다")
    p.add_argument("weights", help="바닥 모델 가중치 (runs/floor_n/weights/best.pt)")
    p.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    if args.cmd == "list":
        for name, (size, key) in file_list().items():
            print(f"  {name:20s} {size:>7s}  filekey {key}")
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "floor-data":
        floor_data()
    elif args.cmd == "floor-label":
        floor_label(args.weights, args.conf)
    elif args.cmd == "build":
        build(args.teacher, args.conf, args.rebuild)
    else:
        if args.cmd == "fetch":
            files = file_list()
            name = next((n for n in files if n.removesuffix("_new") == args.target), None)
            if name is None:
                raise SystemExit(f"{args.target} 묶음이 없어요. list로 이름을 확인하세요.")
            print(f"{name} ({files[name][0]}) 받는 중...")
            tar_path = download(name, files[name][1])
        else:
            tar_path = Path(args.target)
        if not tar_path.name.startswith(("Bbox", "Surface")):
            raise SystemExit("바운딩박스(Bbox_)와 서피스마스킹(Surface_) 묶음만 처리해요.")
        extract(tar_path, args.every, args.size)
        if not args.keep_tar:
            tar_path.unlink()
            print(f"원본 {tar_path.name}을 지웠어요.")


if __name__ == "__main__":
    main()
