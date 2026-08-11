# %%
import os
import sys
import sqlite3
import pandas as pd

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

# %%
df = load_df()

print(df.head())

print(df['gender'].value_counts())
# %%
print(df.columns)

# %%
category=['gender','age_group','channel','time_slot','group_name','wishlist','selected']

def summary_data(data):
    result = {}
    for x in category:
        result[x] = data[x].value_counts()
    return result

summary = summary_data(df)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(10, 12))

for ax, (name, counts) in zip(axes.flat, summary.items()):
    counts.plot(kind='bar', ax=ax)
    ax.bar_label(ax.containers[0])
    ax.set_title(name)
    ax.set_xlabel('')

for ax in axes.flat[len(summary):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

df[df['group_name']=='A']['selected'].value_counts()
df[df['group_name']=='B']['selected'].value_counts()

## 전부 봄
df['viewed'].value_counts()

##
df_a = df[df['group_name']=='A'].groupby('wishlist')['wishlist'].count()
df_b = df[df['group_name']=='B'].groupby('wishlist')['wishlist'].count()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, data, group_label in zip(axes, [df_a, df_b], ['A', 'B']):
    ax.pie(data.values, labels=['non_wish', 'wish'], autopct="%.1f%%")
    ax.set_title(f"{group_label} - wish_ratio")

plt.show()

def get_counts(df, group, column):
    return (df[df['group_name']==group]
            .groupby(column)[column]
            .count()
            .reindex([0,1], fill_value=0))

def make_autopct(values):
    def autopct_format(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count}cnt)"
    return autopct_format

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

configs = [(group, column) for group in ['A', 'B'] for column in ['wishlist', 'selected']]

for ax, (group_label, column) in zip(axes, configs):
    data = get_counts(df, group_label, column)
    labels = [f'non_{column}', column]
    ax.pie(data.values, labels=labels, autopct=make_autopct(data.values))
    ax.set_title(f"{group_label} - {column}_ratio")

plt.tight_layout()
plt.show()

# %%
# ── 5. 두 비율 z-검정: A/B 선택률 차이가 통계적으로 유의한가? ──
# 지금까지는 베이스라인·분포·변동성을 눈으로 확인하는 단계였고, 이제 핵심
# 질문을 검정할 차례: "A/B 선택률 차이가 우연이 아니라 진짜 디자인 효과인가?"
from scipy.stats import norm, chi2_contingency

n_A = int((df['group_name'] == 'A').sum())
n_B = int((df['group_name'] == 'B').sum())
x_A = int(df[df['group_name'] == 'A']['selected'].sum())
x_B = int(df[df['group_name'] == 'B']['selected'].sum())

p_A = x_A / n_A
p_B = x_B / n_B
diff = p_A - p_B

# 검정통계량(z)은 귀무가설(두 비율이 같다) 하의 합동(pooled) 비율로 계산하고,
# 신뢰구간은 각 군의 실제 비율(unpooled)로 계산한다 — 검정과 구간추정에서
# 분산 추정 방식이 다른 게 표준 관행이다.
p_pooled = (x_A + x_B) / (n_A + n_B)
print(p_pooled)
se_pooled = (p_pooled * (1 - p_pooled) * (1 / n_A + 1 / n_B)) ** 0.5
print(se_pooled)
z_stat = diff / se_pooled
p_value = 2 * norm.sf(abs(z_stat))

se_unpooled = (p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B) ** 0.5
z_crit = norm.ppf(0.975)  # 95% 신뢰구간
ci_low, ci_high = diff - z_crit * se_unpooled, diff + z_crit * se_unpooled

print("=" * 55)
print("  두 비율 z-검정: A vs B 선택률(전환율) 차이")
print("=" * 55)
print(f"  A군: {x_A}/{n_A} = {p_A * 100:.1f}%")
print(f"  B군: {x_B}/{n_B} = {p_B * 100:.1f}%")
print(f"  차이(A-B): {diff * 100:.1f}%p   95% CI: [{ci_low * 100:.1f}%p, {ci_high * 100:.1f}%p]")
print(f"  z = {z_stat:.3f},  p-value = {p_value:.2e}")
print(f"  판정(α=0.05): "
      f"{'유의함 — A/B 차이가 우연으로 보기 어려움' if p_value < 0.05 else '유의하지 않음'}")

# ── 카이제곱 독립성 검정으로 교차 검증 (2x2 분할표, df=1) ──
# 두 비율 z-검정과 (연속성 보정 없는) 카이제곱 독립성 검정은 수학적으로
# 동치(chi2 == z^2) — srm_check.py에서 썼던 것과 같은 교차 검증 패턴이다.
contingency = pd.crosstab(df['group_name'], df['selected'])
chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency, correction=False)
print()
print("[ 카이제곱 독립성 검정 교차 검증 ]")
print(f"  chi2 = {chi2_stat:.3f}  (z^2 = {z_stat ** 2:.3f}),  p-value = {chi2_p:.2e}")
print("=" * 55)

# %%
# ── 6. 효과크기(Effect Size): 차이가 "얼마나" 큰가 ──
# p-value는 "우연이 아니다"만 말해줄 뿐 크기는 말해주지 않는다 — 표본이 크면
# 아주 작은 차이도 유의하게 나올 수 있으므로, 실무적으로 의미 있는 크기인지는
# 효과크기로 따로 판단한다.
import math

def cohens_h(p1: float, p2: float) -> float:
    """두 비율 차이를 위한 표준 효과크기(arcsine 변환 기반).
    %p 절대 차이와 달리 베이스라인 수준에 따른 왜곡(0%~50% 구간과 50%~100%
    구간에서 같은 %p라도 체감 크기가 다른 문제)을 보정해준다."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

h = cohens_h(p_A, p_B)
h_abs = abs(h)
if h_abs < 0.2:
    h_label = "작음(small)"
elif h_abs < 0.5:
    h_label = "중간(medium)"
elif h_abs < 0.8:
    h_label = "큼(large)"
else:
    h_label = "매우 큼(very large)"

# 상대위험도(risk ratio)·오즈비(odds ratio) — 비즈니스 커뮤니케이션에 흔히 쓰는 지표
risk_ratio = p_A / p_B
odds_A = x_A / (n_A - x_A)
odds_B = x_B / (n_B - x_B)
odds_ratio = odds_A / odds_B

# 파이 계수(phi) — 카이제곱 기반 효과크기. 2x2 분할표에서는 Cramér's V와 동일하며
# Cohen의 상관계수 기준(.1/.3/.5 = 작음/중간/큼)으로 해석한다.
phi = math.sqrt(chi2_stat / (n_A + n_B))

print("=" * 55)
print("  효과크기(Effect Size)")
print("=" * 55)
print(f"  절대 차이(A-B):      {diff * 100:.1f}%p")
print(f"  Cohen's h:           {h:.3f}  ({h_label})")
print(f"  상대위험도(A/B):     {risk_ratio:.2f}배  (A가 B보다 {risk_ratio:.2f}배 더 많이 선택됨)")
print(f"  오즈비(odds ratio):  {odds_ratio:.2f}")
print(f"  파이 계수(phi):      {phi:.3f}")
print("=" * 55)