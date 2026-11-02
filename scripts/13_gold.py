"""사람이 검수한 정답(data/gold/)으로 채점한다. 선생님과의 일치도가 아니라 실제 정답률을 본다.

정답 만들기 (val 99장을 전부 검수):
    uv run scripts/05_review.py open             # 앱에서 박스를 고치고 저장
    uv run scripts/05_review.py save --gold      # data/gold/에 정답으로 남긴다

실행 예:
    uv run scripts/13_gold.py status                         # 정답이 몇 장, 클래스별 박스 몇 개인지
    uv run scripts/13_gold.py eval                           # 아이폰에 깔린 모델을 정답으로 채점
    uv run scripts/13_gold.py eval --runs walk34_s open2_s   # 여러 모델 비교
    uv run scripts/13_gold.py teacher                        # 선생님 라벨을 정답과 비교 (conf 0.1~0.3)
    uv run scripts/13_gold.py teacher --conf 0.2 --prompt "book=book,hardcover book,open book"

teacher는 학습 없이 선생님 라벨만 채점한다 (Griffin et al. 2025, arXiv 2506.02359의 방식).
그 논문에서는 라벨의 재현율(정답 중 찾은 비율)이 높을수록 학생이 잘 배웠고,
정밀도만 높은 설정(conf 0.8)은 오히려 학생을 가장 나쁘게 만들었다. 그래서 conf는 재현율과 F1을 함께 보고 고른다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "data" / "datasets" / "gold_eval"  # 채점할 때마다 모델 클래스 순서에 맞춰 새로 만든다
DEPLOYED = RUNS_DIR / "deployed.json"
IOU = 0.5  # 이만큼 겹쳐야 맞게 찾은 것으로 본다 (mAP50과 같은 기준)


def auto_label():
    """04_auto_label.py (정답 읽기, 선생님 설정·추론·정리 함수)를 그대로 가져온다."""
    spec = importlib.util.spec_from_file_location("auto_label", ROOT / "scripts" / "04_auto_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_all_gold(auto) -> tuple[dict[str, list], set[str]]:
    """사진 이름 → 정답 박스, 그리고 모든 정답 사진에서 검수한 클래스.

    어떤 사진에선 검수하지 않은 클래스가 섞이면 그 클래스는 공정하게 채점할 수 없어서, 모든 정답에 공통인 클래스만 쓴다.
    """
    gold, common = {}, None
    for path in sorted(auto.GOLD_DIR.glob("*.json")):
        if not (RAW_DIR / f"{path.stem}.jpg").exists():
            print(f"  원본 사진이 없어 건너뜀: {path.stem}.jpg")
            continue
        boxes, reviewed = auto.load_gold(path.stem)
        gold[path.stem] = boxes
        common = reviewed if common is None else common & reviewed
    if not gold:
        raise SystemExit("정답이 없어요. 05_review.py open → 검수 → 05_review.py save --gold 를 먼저 하세요.")
    return gold, common


def status(auto) -> None:
    gold, common = load_all_gold(auto)
    split = auto.split_of([RAW_DIR / f"{s}.jpg" for s in gold])
    n_val = sum(v == "val" for v in split.values())
    counts = Counter(n for boxes in gold.values() for n, *_ in boxes)
    print(f"정답 {len(gold)}장 (val {n_val} / train {len(gold) - n_val}), 박스 {sum(counts.values())}개")
    print(f"모든 정답에서 검수한 클래스 {len(common)}개")
    for n in auto.CLASSES:
        if n in common:
            print(f"  {n:14s} {counts[n]}")


def build_eval_set(gold: dict[str, list], common: set[str], names: list[str]) -> tuple[Path, Counter]:
    """정답을 모델의 클래스 순서에 맞춘 YOLO 데이터셋으로 만든다. 모델이 모르는 클래스의 박스는 뺀다."""
    shutil.rmtree(EVAL_DIR, ignore_errors=True)
    (EVAL_DIR / "images" / "val").mkdir(parents=True)
    (EVAL_DIR / "labels" / "val").mkdir(parents=True)
    counts = Counter()
    for stem, boxes in gold.items():
        img = RAW_DIR / f"{stem}.jpg"
        (EVAL_DIR / "images" / "val" / img.name).symlink_to(img)
        h, w = cv2.imread(str(img)).shape[:2]
        lines = []
        for n, x1, y1, x2, y2 in boxes:
            if n not in common or n not in names:
                continue
            x1, x2, y1, y2 = max(x1, 0), min(x2, w), max(y1, 0), min(y2, h)
            lines.append(f"{names.index(n)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[n] += 1
        (EVAL_DIR / "labels" / "val" / f"{stem}.txt").write_text("\n".join(lines))
    yaml_names = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    data = EVAL_DIR.parent / "gold_eval.yaml"
    data.write_text(f"path: {EVAL_DIR}\ntrain: images/val\nval: images/val\nnames:\n{yaml_names}\n")
    return data, counts


def evaluate(auto, runs: list[str]) -> None:
    from ultralytics import YOLO

    gold, common = load_all_gold(auto)
    results = {}  # run → (mAP50, mAP50-95, {클래스: AP50})
    gt_counts = Counter()
    for run in runs:
        model = YOLO(RUNS_DIR / run / "weights" / "best.pt")
        names = [model.names[i] for i in sorted(model.names)]
        data, counts = build_eval_set(gold, common, names)
        gt_counts |= counts
        m = model.val(data=str(data), device="mps", project=str(RUNS_DIR), name=f"{run}_gold", exist_ok=True,
                      plots=False, verbose=False)
        per_class = {names[c]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
        results[run] = (float(m.box.map50), float(m.box.map), per_class)

    print(f"\n사람 정답 {len(gold)}장으로 채점 (선생님 라벨이 아니라 실제 정답률)")
    header = f"{'클래스':14s} {'정답 박스':>8s}" + "".join(f" {r[:12]:>12s}" for r in runs)
    print(header)
    for n in sorted(gt_counts, key=lambda n: -gt_counts[n]):
        cells = "".join(f" {results[r][2][n]:12.2f}" if n in results[r][2] else f" {'-':>12s}" for r in runs)
        print(f"{n:14s} {gt_counts[n]:8d}{cells}")
    print(f"{'mAP50':14s} {'':8s}" + "".join(f" {results[r][0]:12.3f}" for r in runs))
    print(f"{'mAP50-95':14s} {'':8s}" + "".join(f" {results[r][1]:12.3f}" for r in runs))
    print("('-'는 그 모델이 모르는 클래스. 모델마다 아는 클래스가 다르면 mAP는 서로 다른 클래스의 평균이다)")


def parse_prompts(items: list[str], classes: dict[str, list[str]]) -> dict[str, list[str]]:
    classes = dict(classes)
    for item in items:
        name, _, prompts = item.partition("=")
        if name not in classes:
            raise SystemExit(f"'{name}'은 04_auto_label.py의 CLASSES에 없는 클래스예요")
        classes[name] = [p.strip() for p in prompts.split(",") if p.strip()]
    return classes


def count_matches(pred: torch.Tensor, pred_cls: torch.Tensor, gt: torch.Tensor, gt_cls: torch.Tensor,
                  classes: list[int], stats: dict[int, Counter]) -> None:
    """한 사진에서 클래스별로 맞게 찾은 것(tp), 헛것(fp), 놓친 것(fn)을 센다. pred는 점수 높은 순이어야 한다."""
    for c in classes:
        p, g = pred[pred_cls == c], gt[gt_cls == c]
        tp = 0
        if len(p) and len(g):
            iou = box_iou(p, g)
            used = torch.zeros(len(g), dtype=torch.bool)
            for i in range(len(p)):
                j = int(iou[i].masked_fill(used, -1).argmax())
                if iou[i, j] >= IOU:
                    used[j] = True
                    tp += 1
        stats[c]["tp"] += tp
        stats[c]["fp"] += len(p) - tp
        stats[c]["fn"] += len(g) - tp


def prf(s: Counter) -> tuple[float, float, float]:
    p = s["tp"] / max(s["tp"] + s["fp"], 1)
    r = s["tp"] / max(s["tp"] + s["fn"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def check_teacher(auto, confs: list[float], prompts: list[str], model_name: str) -> None:
    from ultralytics import YOLOE

    gold, common = load_all_gold(auto)
    names = list(auto.CLASSES)
    classes = parse_prompts(prompts, auto.CLASSES)
    # 선생님이 라벨을 다는 클래스 중 사람이 검수한 것만 채점한다
    scored = [names.index(n) for n in names if n in common and n not in auto.HUMAN_ONLY]

    model = YOLOE(model_name)
    prompt_to_class = auto.set_teacher(model, classes, auto.DISTRACTORS, auto.load_prototypes())
    stats = {conf: {c: Counter() for c in scored} for conf in confs}
    for i, (stem, boxes) in enumerate(gold.items(), 1):
        frame = cv2.imread(str(RAW_DIR / f"{stem}.jpg"))
        # 가장 낮은 conf로 한 번만 찾고, 높은 conf는 점수로 걸러 낸다 (conf를 높여 다시 찾는 것과 같은 결과)
        b_all, s_all, p_all = auto.predict_rotated(model, frame, min(confs))
        known = [(names.index(n), x1, y1, x2, y2) for n, x1, y1, x2, y2 in boxes if n in names]
        gt = torch.tensor([k[1:] for k in known], dtype=torch.float32).reshape(-1, 4)
        gt_cls = torch.tensor([k[0] for k in known], dtype=torch.long)
        for conf in confs:
            keep = s_all >= conf
            b, s, c = auto.clean(b_all[keep], s_all[keep], prompt_to_class[p_all[keep]], names.index("monitor"))
            order = s.argsort(descending=True)
            count_matches(b[order], c[order], gt, gt_cls, scored, stats[conf])
        print(f"\r  선생님 채점 중 {i}/{len(gold)}", end="", flush=True)
    print()

    changed = [n for n in classes if classes[n] != auto.CLASSES[n]]
    if changed:
        print("바꿔 본 이름: " + "; ".join(f"{n} = {', '.join(classes[n])}" for n in changed))
    print(f"\n선생님 라벨 vs 사람 정답 {len(gold)}장 (IoU {IOU} 이상이면 맞게 찾은 것)")
    print(f"{'클래스':14s} {'정답':>5s}" + "".join(f"   conf {conf:<4g} 정밀/재현" for conf in confs))
    for c in sorted(scored, key=lambda c: -(stats[confs[0]][c]["tp"] + stats[confs[0]][c]["fn"])):
        n_gt = stats[confs[0]][c]["tp"] + stats[confs[0]][c]["fn"]
        if n_gt == 0 and all(stats[conf][c]["fp"] == 0 for conf in confs):
            continue
        cells = "".join(f"   {prf(stats[conf][c])[0]:12.2f} / {prf(stats[conf][c])[1]:.2f}" for conf in confs)
        print(f"{names[c]:14s} {n_gt:5d}{cells}")
    print()
    for conf in confs:
        total = sum(stats[conf].values(), Counter())
        p, r, f1 = prf(total)
        print(f"conf {conf:<4g}: 정밀도 {p:.2f}  재현율 {r:.2f}  F1 {f1:.2f}  "
              f"(맞음 {total['tp']}, 헛것 {total['fp']}, 놓침 {total['fn']})")
    # 3단계 co-teaching의 버리는 비율(r̂)은 선생님 라벨 중 틀린 비율로 잡는다
    default = 0.2 if 0.2 in confs else confs[0]
    p, _, _ = prf(sum(stats[default].values(), Counter()))
    print(f"\nconf {default:g} 선생님 라벨 중 틀린 박스 비율: {1 - p:.0%} (co-teaching 버리는 비율 추정치)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("action", choices=["status", "eval", "teacher"])
    parser.add_argument("--runs", nargs="+", help="eval: 채점할 runs/ 아래 이름들 (기본: 아이폰에 깔린 모델)")
    parser.add_argument("--conf", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.3],
                        help="teacher: 비교할 선생님 확신 기준들")
    parser.add_argument("--prompt", action="append", default=[], metavar="클래스=이름1,이름2",
                        help="teacher: 이 클래스는 선생님에게 이 이름들로 묻는다 (여러 번 줄 수 있음)")
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="teacher: 선생님 모델 (04_auto_label.py와 같게)")
    args = parser.parse_args()

    auto = auto_label()
    if args.action == "status":
        status(auto)
    elif args.action == "eval":
        runs = args.runs or [json.loads(DEPLOYED.read_text())["run"]]
        evaluate(auto, runs)
    else:
        check_teacher(auto, sorted(args.conf), args.prompt, args.model)


if __name__ == "__main__":
    main()
