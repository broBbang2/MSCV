"""지금 돌고 있는 작업(scripts/의 번호 스크립트)만 본다: 학습, 라벨링, 데이터 받기 등.

작업마다 실행 시간, 로그 파일, 로그 마지막 줄을 보여주고, 10분 넘게 로그가 그대로면 멈춤 경고를 띄운다.
학습이면 완료 epoch, 남은 시간, best mAP도 함께 보여준다. 프로젝트 전체 현황은 status.py.

실행 예:
    uv run scripts/jobs.py
    uv run scripts/jobs.py --follow       # 5초마다 새로고침 (Ctrl+C로 종료)
"""

from __future__ import annotations

import argparse
import time

from status import jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--follow", action="store_true", help="주기적으로 새로고침 (Ctrl+C로 종료)")
    parser.add_argument("--interval", type=int, default=5, help="--follow 새로고침 간격(초)")
    args = parser.parse_args()

    if not args.follow:
        jobs()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # 화면 지우기
            print(time.strftime("%H:%M:%S 기준\n"))
            jobs()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
