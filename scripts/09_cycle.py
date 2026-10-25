"""앱 데이터 한 바퀴: 아이폰에서 가져오기 → 라벨링 → 학습 → 비교 → 더 나으면 아이폰 앱에 새 모델 설치.

사용자가 "데이터 가져와"라고 하면 이 스크립트를 돌린다.
  1. 08_pull_app.py: 앱이 저장한 프레임을 가져와 data/raw에 넣고 자동 라벨(04)까지 반영
  2. 03_train.py: 내 데이터 + 공개 데이터(mine_open)로 yolo26s를 처음부터 다시 파인튜닝
  3. 지금 아이폰에 깔린 모델과 같은 검증 세트(촬영본, test_ 영상)로 mAP50-95를 비교
  4. 새 모델이 더 나으면 CoreML(8비트 + NMS)로 바꿔 앱을 다시 빌드하고 아이폰에 설치

지금 깔린 모델은 runs/deployed.json에 기록한다.

실행 예:
    uv run scripts/09_cycle.py
    uv run scripts/09_cycle.py --epochs 10 --no-install    # 설치 없이 학습·비교만
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
IOS_DIR = ROOT / "ios"
DEPLOYED = RUNS_DIR / "deployed.json"
VAL_DATA = ROOT / "data" / "datasets" / "mine.yaml"

# coremltools가 프로젝트의 NumPy 2.x·torch 2.14와 맞지 않아, 변환만 시험된 버전으로 따로 띄운 환경에서 돌린다
EXPORT_ENV = ["uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", "ultralytics==8.4.184",
              "--with", "torch==2.7.0", "--with", "torchvision==0.22.0", "--with", "numpy<2",
              "--with", "coremltools==9.0", "--with", "scikit-learn", "python"]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(run: str) -> float:
    """검증 세트에서의 mAP50-95. 검증 세트가 바뀌었을 수 있어 깔린 모델도 매번 다시 잰다."""
    from ultralytics import YOLO

    m = YOLO(RUNS_DIR / run / "weights" / "best.pt").val(
        data=str(VAL_DATA), device="mps", project=str(RUNS_DIR), name=f"{run}_val", exist_ok=True,
        plots=False, verbose=False,
    )
    return float(m.box.map)


def same_classes(a: str, b: str) -> bool:
    from ultralytics import YOLO

    return YOLO(RUNS_DIR / a / "weights" / "best.pt").names == YOLO(RUNS_DIR / b / "weights" / "best.pt").names


def install(run: str, device: str) -> None:
    subprocess.run([*EXPORT_ENV, "scripts/07_export.py", "--runs", run, "--int8", "--nms"], cwd=ROOT, check=True)
    target = IOS_DIR / "FirstCV" / "Detector.mlpackage"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(RUNS_DIR / run / "weights" / "best_int8_nms.mlpackage", target)
    subprocess.run(["xcodegen"], cwd=IOS_DIR, check=True, capture_output=True)
    subprocess.run(["xcodebuild", "-project", "FirstCV.xcodeproj", "-scheme", "FirstCV",
                    "-destination", f"id={device}", "-derivedDataPath", "build",
                    "-allowProvisioningUpdates", "build"], cwd=IOS_DIR, check=True, capture_output=True)
    subprocess.run(["xcrun", "devicectl", "device", "install", "app", "--device", device,
                    "build/Build/Products/Debug-iphoneos/FirstCV.app"], cwd=IOS_DIR, check=True,
                   capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--epochs", type=int, default=15, help="지금까지 최고점이 10 epoch 근처라 15면 충분하다")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--no-install", action="store_true", help="더 나아도 아이폰에 설치하지 않는다")
    parser.add_argument("--force", action="store_true", help="새 데이터가 없어도 학습한다")
    args = parser.parse_args()

    pull_app = load_script("08_pull_app.py")
    device = pull_app.find_iphone()

    print("[1/4] 아이폰에서 데이터 가져오기")
    n_new = pull_app.pull(device)
    if n_new == 0 and not args.force:
        print("새 데이터가 없어서 학습하지 않아요.")
        return

    run = f"cycle_{datetime.now():%Y%m%d_%H%M}"
    print(f"[2/4] 학습: {run} ({args.model}, {args.epochs} epoch)")
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "03_train.py"), "--data", "mine_open",
                    "--model", args.model, "--name", run, "--epochs", str(args.epochs)], check=True)

    print("[3/4] 지금 깔린 모델과 비교")
    deployed = json.loads(DEPLOYED.read_text()) if DEPLOYED.exists() else {"run": "open2_s"}
    new = score(run)
    if same_classes(deployed["run"], run):
        old = score(deployed["run"])
        print(f"  깔린 모델 {deployed['run']}: mAP50-95 {old:.3f}")
        print(f"  새 모델   {run}: mAP50-95 {new:.3f}  ({new - old:+.3f})")
    else:
        # 클래스가 늘거나 바뀌면 깔린 모델은 새 클래스를 아예 못 찾아서 점수 비교가 공정하지 않다
        old = float("-inf")
        print(f"  클래스 구성이 바뀌어서 비교 없이 새 모델로 바꿔요. 새 모델 {run}: mAP50-95 {new:.3f}")

    if new <= old:
        print("[4/4] 새 모델이 낫지 않아서 아이폰은 그대로 둬요.")
        return
    if args.no_install:
        print("[4/4] 더 낫지만 --no-install이라 설치하지 않았어요.")
        return
    print("[4/4] 더 나아서 아이폰에 설치하는 중")
    install(run, device)
    DEPLOYED.write_text(json.dumps({"run": run, "map50_95": round(new, 4),
                                    "installed": datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(f"  설치 완료. 앱을 다시 켜면 {run} 모델로 인식해요.")


if __name__ == "__main__":
    main()
