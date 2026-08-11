"""
bot_main.py — 본실험 실행 (A/B 50:50 무작위 배정)

로그는 main.db(이 폴더 안)에 쌓인다.

실행: python main/bot_main.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import run

NUM_USERS_MAIN = 706  # 본실험 인원 (파일럿 결과 보고 재계산 예정 — 일단 유지)

if __name__ == "__main__":
    run(n=NUM_USERS_MAIN, forced_group=None, stage="main")
