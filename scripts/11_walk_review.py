"""보행 모드 경고를 하나씩 보며 맞는 경고였는지 표시하고, 보행 보조 지표를 낸다.

앱은 보행 모드에서 경고할 때마다 그 순간 사진과 기록(data/walk_events/<세션>.csv)을 남긴다.
08_pull_app.py(또는 "데이터 가져와")가 가져온 뒤 이 스크립트를 돌린다.

검토 화면 키:  y 맞는 경고   n 헛경고(길에 아무것도 없거나 엉뚱한 것)   s 건너뛰기   b 이전   q 끝내기
검토 결과는 data/walk_events/review.csv에 저장되어, 다음에 돌리면 아직 안 본 것부터 이어서 한다.

지표 (같은 단계에서 4초마다 다시 말한 경고는 빼고, 처음 경고만 센다)
- 헛경고 비율: 검토한 경고 중 n 비율. 종류(장애물/낙차/다가오는 물체)별로
- 경고 거리: 맞는 경고가 처음 울렸을 때 장애물까지 거리 (경고 단계는 1.2m, 낙차 2m 안에서 울린다)
- 놓친 장애물: 앱의 "경고 놓침 기록" 버튼을 누른 수
- 10분당 경고 수: 앱이 저장한 프레임 수(0.5초마다 1장)로 보행 시간을 어림한다

실행 예:
    uv run scripts/11_walk_review.py            # 검토하고 끝나면 지표
    uv run scripts/11_walk_review.py --report   # 지표만
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DIR = ROOT / "data" / "walk_events"
LOG_DIR = ROOT / "data" / "app_logs"
REVIEW = EVENTS_DIR / "review.csv"
KINDS = {"obstacle": "장애물", "drop": "낙차", "approach": "다가오는 물체", "missed": "놓침"}
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SAVE_INTERVAL = 0.5  # 앱이 프레임을 저장하는 간격(초)


def load_events() -> list[dict]:
    rows = []
    for f in sorted(EVENTS_DIR.glob("*.csv")):
        if f == REVIEW:
            continue
        for r in csv.DictReader(f.open()):
            r["session"] = f.stem
            r["key"] = f"{r['image']}|{r['kind']}|{r['time']}"
            rows.append(r)
    return rows


def load_review() -> dict[str, str]:
    if not REVIEW.exists():
        return {}
    return {r["key"]: r["verdict"] for r in csv.DictReader(REVIEW.open())}


def save_review(review: dict[str, str]) -> None:
    with REVIEW.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "verdict"])
        w.writerows(sorted(review.items()))


def render(e: dict, index: int, total: int, verdict: str | None) -> np.ndarray:
    """사진에 경고 위치(동그라미)와 경고 내용을 그린다."""
    img = Image.open(EVENTS_DIR / e["image"]).convert("RGB")
    img.thumbnail((720, 960))
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 26, index=6)
    if e["x"] and e["y"]:
        x, y = float(e["x"]) * w, float(e["y"]) * h
        draw.ellipse([x - 28, y - 28, x + 28, y + 28], outline=(255, 60, 60), width=5)
    lines = [
        f"{index + 1}/{total}  {KINDS.get(e['kind'], e['kind'])} {e['level']}"
        + (" (반복)" if e["repeated"] == "1" else ""),
        (f"{float(e['distance']):.2f}m " if e["distance"] else "") + (e["label"] or "")
        + (f"  {float(e['speed']):.1f}m/s로 다가옴" if e["speed"] else "") + ("  걷는 중" if e["walking"] == "1" else ""),
        f"현재 판정: {verdict}" if verdict else "y 맞음  n 헛경고  s 건너뛰기  b 이전  q 끝",
    ]
    draw.rectangle([0, 0, w, 40 * len(lines) + 10], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((12, 8 + 40 * i), line, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def review_loop(events: list[dict], review: dict[str, str]) -> None:
    # 놓침은 사용자가 직접 누른 것이라 검토하지 않는다. 처음 경고만 본다 (반복은 같은 판정을 따른다고 본다)
    todo = [e for e in events if e["kind"] != "missed" and e["repeated"] != "1"]
    i = next((k for k, e in enumerate(todo) if e["key"] not in review), len(todo))
    if i == len(todo):
        print("새로 검토할 경고가 없어요.")
        return
    print(f"검토할 경고 {len(todo) - i}개 (전체 {len(todo)}개). 창을 클릭한 뒤 키를 누르세요.")
    while 0 <= i < len(todo):
        e = todo[i]
        cv2.imshow("walk review", render(e, i, len(todo), review.get(e["key"])))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            i = max(i - 1, 0)
            continue
        if key in (ord("y"), ord("n")):
            review[e["key"]] = "correct" if key == ord("y") else "false"
            save_review(review)
        i += 1
    cv2.destroyAllWindows()


def walking_minutes(sessions: set[str]) -> float:
    """경고가 있었던 세션의 앱 사용 시간(분). 저장한 프레임 수 × 0.5초로 어림한다 (저장을 멈춘 동안은 빠진다)."""
    total = 0
    for s in sessions:
        log = LOG_DIR / f"{s}.csv"
        if log.exists():
            total += sum(1 for _ in csv.DictReader(log.open()))
    return total * SAVE_INTERVAL / 60


def report(events: list[dict], review: dict[str, str]) -> None:
    first = [e for e in events if e["kind"] != "missed" and e["repeated"] != "1"]
    missed = [e for e in events if e["kind"] == "missed"]
    sessions = {e["session"] for e in events}
    minutes = walking_minutes(sessions)
    print(f"\n보행 기록: 세션 {len(sessions)}개, 약 {minutes:.1f}분, 처음 경고 {len(first)}개, 놓침 버튼 {len(missed)}번")
    print(f"\n  {'종류':10s} {'경고':>4s} {'검토':>4s} {'맞음':>4s} {'헛경고':>6s} {'헛경고율':>8s} {'경고 거리(중앙값)':>14s} {'10분당':>6s}")
    by_kind = defaultdict(list)
    for e in first:
        by_kind[e["kind"]].append(e)
    for kind in ("obstacle", "drop", "approach"):
        es = by_kind.get(kind, [])
        if not es:
            continue
        verdicts = [review[e["key"]] for e in es if e["key"] in review]
        ok = verdicts.count("correct")
        bad = verdicts.count("false")
        rate = f"{bad / len(verdicts):.0%}" if verdicts else "-"
        # 경고 거리는 경고(warn) 단계로 처음 울린 맞는 경고만 본다. 바로 멈춤(stop)이면 이미 늦게 알아챈 것이다
        dists = [float(e["distance"]) for e in es
                 if e["distance"] and review.get(e["key"]) == "correct" and e["level"] in ("warn", "")]
        dist = f"{statistics.median(dists):.2f}m (n={len(dists)})" if dists else "-"
        per10 = f"{len(es) / minutes * 10:.1f}" if minutes else "-"
        print(f"  {KINDS[kind]:10s} {len(es):4d} {len(verdicts):4d} {ok:4d} {bad:6d} {rate:>8s} {dist:>14s} {per10:>6s}")
    stops = [e for e in first if e["level"] == "stop" and review.get(e["key"]) == "correct"]
    if stops:
        print(f"\n  경고 없이 바로 멈춤 단계로 울린 맞는 경고 {len(stops)}개 (늦게 알아챈 장애물 후보)")
    if missed:
        print(f"  놓침 버튼 {len(missed)}번: " + ", ".join(f"{e['session']}/{e['image']}" for e in missed[:5])
              + (" ..." if len(missed) > 5 else ""))
    unreviewed = sum(1 for e in first if e["key"] not in review)
    if unreviewed:
        print(f"\n  아직 검토 안 한 경고 {unreviewed}개 (uv run scripts/11_walk_review.py)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--report", action="store_true", help="검토 없이 지표만 본다")
    args = parser.parse_args()

    events = load_events()
    if not events:
        raise SystemExit(f"{EVENTS_DIR}에 경고 기록이 없어요. 보행 모드로 걸은 뒤 08_pull_app.py로 가져오세요.")
    review = load_review()
    if not args.report:
        review_loop(events, review)
    report(events, review)


if __name__ == "__main__":
    main()
