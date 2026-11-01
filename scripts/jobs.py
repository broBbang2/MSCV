"""지금 돌고 있는 작업(scripts/의 번호 스크립트)만 본다: 학습, 라벨링, 데이터 받기 등.

작업마다 실행 시간, 로그 파일, 로그 마지막 줄을 보여주고, 10분 넘게 로그가 그대로면 멈춤 경고를 띄운다.
학습이면 완료 epoch, 남은 시간, best mAP도 함께 보여준다. 프로젝트 전체 현황은 status.py.

실행 예:
    uv run scripts/jobs.py
    uv run scripts/jobs.py --follow                  # 실시간 (0.5초마다 진행 바까지 갱신, Ctrl+C로 종료)
    uv run scripts/jobs.py --follow --interval 2     # 갱신 간격 바꾸기
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time

from status import jobs


def render() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(time.strftime("%H:%M:%S 기준  (Ctrl+C로 종료)\n"))
        jobs()
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--follow", action="store_true", help="실시간으로 계속 갱신 (Ctrl+C로 종료)")
    parser.add_argument("--interval", type=float, default=0.5, help="--follow 갱신 간격(초)")
    args = parser.parse_args()

    if not args.follow:
        jobs()
        return
    # 화면 전체를 지우면 깜빡이므로, 커서를 맨 위로 옮겨 덮어쓰고 줄 끝·화면 아래 남은 글자만 지운다
    sys.stdout.write("\033[?25l\033[2J")  # 커서 숨기기, 처음 한 번만 화면 지우기
    try:
        while True:
            frame = render().replace("\n", "\033[K\n")
            sys.stdout.write(f"\033[H{frame}\033[J")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # 커서 되돌리기


if __name__ == "__main__":
    main()
