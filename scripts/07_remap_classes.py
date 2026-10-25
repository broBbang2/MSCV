"""19클래스 데이터셋을 10클래스로 바꾼 사본을 만든다. 원본(mine/, open/)은 건드리지 않는다.

- 남길 클래스(KEEP)만 남기고, tumbler처럼 비슷한 클래스는 합친다(MERGE).
- 사진은 복사하지 않고 원본을 가리키는 링크로 만든다. 디스크를 거의 쓰지 않는다.
- 내 영상(mine)은 남은 박스가 없어도 사진을 남긴다. "아무것도 없는 화면"도 배워야 하기 때문이다.
- 공개 데이터(open)는 빠진 클래스 때문에 받은 사진이라 박스가 다 사라지면 버린다.

검수(05_review.py)는 계속 mine/에서 하고, 검수가 끝날 때마다 이 스크립트를 다시 돌리면 된다.
결과 폴더는 매번 새로 만들어진다.

실행 예:
    uv run scripts/07_remap_classes.py                    # mine → mine10 (+ open이 있으면 open10)
    uv run scripts/07_remap_classes.py --sources mine     # 내 데이터만
    uv run scripts/03_train.py --data mine10 --name m10_n
    uv run scripts/03_train.py --data mine_open10 --name mo10_n
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "datasets"
SOURCE_YAML = DATASETS_DIR / "mine.yaml"  # mine과 open은 같은 19클래스 순서를 쓴다

KEEP = ["monitor", "laptop", "keyboard", "mouse", "phone",
        "cup", "bottle", "backpack", "chair", "book"]
MERGE = {"tumbler": "bottle"}  # 헷갈리는 짝은 버리지 않고 합친다
DROP_EMPTY = {"open"}  # 박스가 다 사라지면 사진째 버릴 데이터셋
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def source_names() -> list[str]:
    names = yaml.safe_load(SOURCE_YAML.read_text())["names"]
    return [names[i] for i in sorted(names)]


def build_mapping(names: list[str]) -> dict[int, int]:
    """원본 클래스 번호 → 새 클래스 번호. 매핑에 없는 클래스는 버린다."""
    mapping = {}
    for old_id, name in enumerate(names):
        target = MERGE.get(name, name)
        if target in KEEP:
            mapping[old_id] = KEEP.index(target)
    return mapping


def remap_dataset(src: str, mapping: dict[int, int]) -> Counter:
    src_dir = DATASETS_DIR / src
    out_dir = DATASETS_DIR / f"{src}10"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    stats: Counter = Counter()
    for split_dir in sorted((src_dir / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

        for img in sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label = src_dir / "labels" / split / f"{img.stem}.txt"
            lines = label.read_text().splitlines() if label.exists() else []
            kept = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                old_id = int(parts[0])
                if old_id in mapping:
                    kept.append(" ".join([str(mapping[old_id]), *parts[1:]]))
                    stats[f"{split} 박스 {KEEP[mapping[old_id]]}"] += 1
                else:
                    stats[f"{split} 버린 박스"] += 1

            if not kept and lines and src in DROP_EMPTY:
                stats[f"{split} 버린 사진"] += 1
                continue

            link = out_dir / "images" / split / img.name
            link.symlink_to(os.path.relpath(img, link.parent))
            (out_dir / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else ""))
            stats[f"{split} 사진"] += 1
            if not kept:
                stats[f"{split} 빈 사진"] += 1

    (out_dir / "classes.txt").write_text("\n".join(KEEP) + "\n")
    return stats


def write_yaml(name: str, train: str | list[str], val: str) -> Path:
    path = DATASETS_DIR / f"{name}.yaml"
    body = {"path": str(DATASETS_DIR), "train": train, "val": val,
            "names": dict(enumerate(KEEP))}
    path.write_text(yaml.safe_dump(body, allow_unicode=True, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["mine", "open"],
                        help="변환할 데이터셋 (data/datasets/<이름>)")
    args = parser.parse_args()

    names = source_names()
    mapping = build_mapping(names)
    dropped = [n for i, n in enumerate(names) if i not in mapping]
    merged = [f"{a}→{b}" for a, b in MERGE.items()]
    print(f"남김 {len(KEEP)}개: {', '.join(KEEP)}")
    print(f"합침: {', '.join(merged)}   버림: {', '.join(n for n in dropped)}\n")

    done = []
    for src in args.sources:
        if not (DATASETS_DIR / src / "images").is_dir():
            print(f"[{src}] 없음, 건너뜀")
            continue
        stats = remap_dataset(src, mapping)
        done.append(src)
        print(f"[{src} → {src}10]")
        for key in sorted(stats):
            print(f"  {key:<24}{stats[key]:>7}")
        print()

    if "mine" in done:
        print("→", write_yaml("mine10", "mine10/images/train", "mine10/images/val"))
    if "mine" in done and (DATASETS_DIR / "open10" / "images").is_dir():
        print("→", write_yaml("mine_open10", ["mine10/images/train", "open10/images/train"],
                              "mine10/images/val"))


if __name__ == "__main__":
    main()
