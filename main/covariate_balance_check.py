# %%
"""
covariate_balance_check.py — 공변량 균형(Covariate Balance) 점검

SRM(srm_check.py)은 "A/B 표본 수 비율이 50:50에 가까운가"만 확인한다.
표본 수가 균형 잡혀 있어도, 그 안의 구성(연령대·채널·성별·시간대)이
한쪽으로 쏠려 있으면 관측된 전환율 차이가 디자인 효과가 아니라 구성 차이
때문일 수 있다 — 이걸 배제하려면 각 공변량이 A/B 간에 실제로 비슷하게
분포하는지 별도로 검정해야 한다.

방법: 공변량(범주형) × group_name 2xK 분할표에 대해 카이제곱 독립성 검정을
수행한다. 귀무가설은 "공변량 분포가 A/B 간에 같다"이며, p >= 0.05면
"이 공변량에서 A/B가 불균형하다는 근거가 없다"로 해석한다(균형을
"증명"하는 것이 아니라, 불균형의 증거가 없다는 뜻임에 유의).
"""

import os
import sys
import sqlite3
import pandas as pd
from scipy.stats import chi2_contingency

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글 출력 깨짐 방지

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
COVARIATES = ["channel", "age_group", "gender", "time_slot"]
ALPHA = 0.05

print("=" * 60)
print("  공변량 균형(Covariate Balance) 점검 — A vs B")
print("=" * 60)

results = []
for col in COVARIATES:
    contingency = pd.crosstab(df[col], df["group_name"])
    chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
    balanced = p_value >= ALPHA
    results.append(
        {"covariate": col, "chi2": chi2, "dof": dof, "p_value": p_value, "balanced": balanced}
    )

    print(f"\n[ {col} ]")
    print(contingency.to_string())
    print(f"  chi2 = {chi2:.3f}  (dof={dof}),  p-value = {p_value:.4f}")
    print(f"  판정: {'균형 (p >= 0.05)' if balanced else '⚠ 불균형 의심 (p < 0.05) — 원인 확인 필요'}")

# %%
# ── 요약 ──
print("\n" + "=" * 60)
print("  요약")
print("=" * 60)
summary = pd.DataFrame(results)
print(summary.to_string(index=False))

all_balanced = summary["balanced"].all()
print()
if all_balanced:
    print("모든 공변량에서 A/B 간 유의한 차이 없음 — 무작위 배정이 구성 면에서도")
    print("균형 잡혔다고 볼 수 있음. 관측된 전환율 차이를 공변량 쏠림으로 설명하기 어려움.")
else:
    unbalanced = summary[~summary["balanced"]]["covariate"].tolist()
    print(f"⚠ 불균형 의심 공변량: {unbalanced} — 본 분석 결과 해석 시 주의,")
    print("  필요하면 해당 공변량으로 층화(stratified) 분석 고려.")

print("=" * 60)
