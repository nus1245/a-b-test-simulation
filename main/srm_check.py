# %%
"""
srm_check.py — SRM(Sample Ratio Mismatch) 점검

A/B 테스트 결과를 신뢰하기 전에 반드시 먼저 확인해야 하는 것: "배정 자체가
의도한 비율(여기서는 50:50)대로 이루어졌는가?"

배정 비율이 눈에 띄게 어긋나 있으면(SRM), 이후에 나오는 A/B 전환율 차이가
디자인 때문인지 배정 과정의 편향 때문인지 구분할 수 없게 된다. 그래서 본
분석(z-검정 등) 이전 단계의 게이트로 둔다.

scipy 없이도 검증 가능하도록 math.erfc만으로 카이제곱(df=1) 검정을 직접
구현하고, scipy.stats.chisquare 결과와 교차 검증한다. df=1 카이제곱은
표준정규분포 Z의 제곱과 같다는 성질을 이용해 p-value = erfc(sqrt(chi2/2))로
계산한다 (2측 z-검정과 동일한 결과).

대상: main.db (본실험, A/B 무작위 배정 데이터). 파일럿(pilot.db)은 A안만
단독 노출이라 애초에 배정 비율 검정 대상이 아니다.
"""

import os
import sys
import sqlite3
import math
import pandas as pd
from scipy.stats import chisquare

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글/이모지 출력 깨짐 방지

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_path

# %%
def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(get_db_path("main"))
    df = pd.read_sql("SELECT * FROM visit_log", conn)
    conn.close()
    return df

df = load_df()

# %%
# ── 1. 카이제곱 적합도 검정: 관측 배정 비율 vs 기대 배정 비율(50:50) ──
EXPECTED_RATIO = {"A": 0.5, "B": 0.5}  # bot_main.py의 random.choice(["A","B"]) 기준

obs_counts = df["group_name"].value_counts().to_dict()
n_total = sum(obs_counts.values())

chi2 = 0.0
for group, ratio in EXPECTED_RATIO.items():
    observed = obs_counts.get(group, 0)
    expected = n_total * ratio
    chi2 += (observed - expected) ** 2 / expected

# df=1 카이제곱의 양측 p-value (scipy.stats.chisquare와 동일한 값)
p_value = math.erfc(math.sqrt(chi2 / 2))

print("=" * 55)
print("  SRM(Sample Ratio Mismatch) 점검")
print("=" * 55)
print(f"  관측치: {obs_counts}  (총 {n_total}건)")
print(f"  기대치: 각 {n_total / 2:.0f}건 (50:50 가정)")
print(f"  chi2 = {chi2:.3f} (df=1),  p-value = {p_value:.5f}")

# 업계 관행: SRM 점검은 본 검정(α=0.05)보다 보수적인 α=0.01을 기준으로 삼는다.
# "차이가 있는지"를 찾는 게 아니라 "배정 절차 자체가 깨졌는지"를 걸러내는
# 게이트이기 때문에, 오탐(false alarm)을 줄이는 쪽보다 놓치지 않는 쪽을 우선한다.
if p_value < 0.01:
    print("  판정: ⚠️ SRM 의심됨 (p < 0.01) — 본 분석 전 원인 확인 필요")
else:
    print("  판정: 이상 없음 (p >= 0.01)")

# ── 1-1. scipy로 교차 검증 ──
# 위 math.erfc 수식을 손으로 유도한 게 맞는지, 통계 라이브러리 표준 구현과
# 대조해서 확인한다. 두 값이 일치해야 위 수동 계산을 신뢰할 수 있다.
scipy_chi2, scipy_p = chisquare(
    f_obs=[obs_counts.get(g, 0) for g in EXPECTED_RATIO],
    f_exp=[n_total * r for r in EXPECTED_RATIO.values()],
)
print()
print("[ scipy.stats.chisquare 교차 검증 ]")
print(f"  scipy: chi2 = {scipy_chi2:.3f}, p-value = {scipy_p:.5f}")
print(f"  수동:  chi2 = {chi2:.3f}, p-value = {p_value:.5f}")
assert math.isclose(chi2, scipy_chi2, rel_tol=1e-9), "수동 chi2 계산이 scipy와 불일치"
assert math.isclose(p_value, scipy_p, rel_tol=1e-6), "수동 p-value 계산이 scipy와 불일치"
print("  일치 확인 완료 ✅")

# %%
# ── 2. timestamp 연속성 확인: 여러 번의 실행이 섞여 있는지 ──
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_sorted = df.sort_values("timestamp").reset_index(drop=True)

gaps_sec = df_sorted["timestamp"].diff().dt.total_seconds()
GAP_THRESHOLD_SEC = 60  # 이보다 큰 공백이 있으면 별개 실행(batch)일 가능성

big_gaps = gaps_sec[gaps_sec > GAP_THRESHOLD_SEC]

print()
print("[ 실행 연속성 확인 ]")
print(f"  timestamp 범위: {df_sorted['timestamp'].min()} ~ {df_sorted['timestamp'].max()}")
print(f"  {GAP_THRESHOLD_SEC}초 초과 공백 개수: {len(big_gaps)}")
if len(big_gaps) > 0:
    print("  → 여러 번의 실행(batch)이 섞여 있을 가능성 있음. 아래 공백 지점 확인:")
    print(big_gaps)
else:
    print("  → 공백 없음: 단일 연속 실행으로 보임 (여러 회차 누적 때문은 아님)")

# %%
# ── 3. 실행 전반부 vs 후반부 배정 비율 비교 ──
# 특정 구간에서만 쏠렸는지, 실행 내내 일관되게 쏠렸는지 구분하기 위함.
# (예: 후반부에만 쏠렸다면 API 재시도/타임아웃 같은 시간 의존적 버그를 의심할 단서가 됨)
half = n_total // 2
first_half = df_sorted.iloc[:half]["group_name"].value_counts().to_dict()
second_half = df_sorted.iloc[half:]["group_name"].value_counts().to_dict()

print()
print("[ 전반부 vs 후반부 배정 분포 ]")
print(f"  전반부({half}건): {first_half}")
print(f"  후반부({n_total - half}건): {second_half}")
print("  → 양쪽 다 비슷한 방향으로 쏠려 있다면 특정 구간 버그보다는")
print("     단일 실행 전체에 걸친 우연한 편차(또는 배정 로직 자체 문제)에 가까움")

print()
print("=" * 55)
