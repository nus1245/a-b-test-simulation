"""
bot_pilot.py — 파일럿 실행 (A안만 단독 노출, 베이스라인 선택률 추정용)

본실험(A/B 무작위 배정) 전에, A안 하나만 보여줬을 때 선택률이 어느 정도
나오는지 베이스라인을 먼저 재본다. 로그는 pilot.db(이 폴더 안)에 쌓인다.
본실험 표본수는 이 결과를 pilot_analysis.py에 넣어 재계산할 예정.

실행: python pilot/bot_pilot.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import run

NUM_USERS_PILOT = 150  # 파일럿 실행 인원

if __name__ == "__main__":
    run(n=NUM_USERS_PILOT, forced_group="A", stage="pilot")
