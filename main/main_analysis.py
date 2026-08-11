"""
main_analysis.py — 본실험(A/B) 통합 분석 파이프라인

파일럿 단계(pilot/pilot_analysis.py)에서 베이스라인 선택률과 필요 표본수를
먼저 추정한 뒤, 본실험(bot_main.py, A/B 50:50 무작위 배정)을 그 표본수만큼
돌려 쌓은 main.db 로그를 분석하는 통합 스크립트다.

기존에 흩어져 있던 4개 스크립트(srm_check.py, covariate_balance_check.py,
AB_TEST_SIMULATION.py, analyze.py)를 아래 순서 하나의 흐름으로 병합했다.
개별 원본 파일은 그대로 남아 있으며, 이 파일은 그 로직을 함수화해 순서대로
호출하는 통합 버전이다 — 개별 스크립트를 따로 돌리고 싶으면 원본을 그대로
실행해도 된다.

    1) SRM 게이트        (원본: srm_check.py)
       — "배정 비율 자체가 50:50대로 이루어졌는가?"부터 확인. 여기서 이상이
         있으면 이후 어떤 분석도 신뢰할 수 없으므로 가장 먼저 수행하는 게이트.
    2) 공변량 균형 게이트 (원본: covariate_balance_check.py)
       — 표본 수 비율은 맞아도 그 안의 구성(연령대/채널/성별/시간대)이
         한쪽으로 쏠려 있으면 전환율 차이가 디자인 효과가 아닐 수 있다.
    3) EDA 요약          (원본: analyze.py의 print_summary + 신규 시각화)
       — 군별 방문자 수·전환율, 세그먼트 crosstab, 체류시간, 찜→선택 전환율.
         plot_segment_conversion_rates()는 4개 원본 파일에는 없던 신규 함수 —
         "선택자 수" crosstab만으로는 군별 전체 선택률 차이(A > B) 때문에
         특정 세그먼트가 실제보다 쏠려 보이는 착시가 생길 수 있어(예: 40대
         선택자가 A=41명/B=8명), 세그먼트별 "선택률(%)"을 A/B 나란히 비교하는
         막대그래프를 추가했다. 통계적 유의성 검정은 아니고 참고용 시각화다.
    4) 두 비율 z-검정     (원본: AB_TEST_SIMULATION.py 5번 섹션 + 신규 시각화)
       — "A/B 선택률 차이가 우연이 아니라 진짜 디자인 효과인가?"를 검정.
         카이제곱 독립성 검정(2x2)으로 교차 검증. plot_conversion_rate_with_ci()도
         신규 함수 — A/B 선택률을 95% 신뢰구간과 함께 막대그래프로 보여준다.
    5) 효과크기          (원본: AB_TEST_SIMULATION.py 6번 섹션)
       — Cohen's h, 상대위험도, 오즈비, 파이 계수로 차이의 "크기"를 판단.
    6) CSV 추출          (원본: analyze.py의 export_csv)

주의: 1)·2)는 게이트 단계라 경고만 출력하고 실행을 막지는 않는다 — 이상이
발견되면 3) 이후 결과를 해석할 때 그 사실을 감안해야 한다는 뜻이다.

실행: python main/main_analysis.py
"""

from __future__ import annotations

import os
import sys
import math
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency, norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글/이모지 출력 깨짐 방지

# Windows 기본 matplotlib은 한글 폰트가 없어 그래프의 한글이 네모(□)로 깨진다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COVARIATES = ["channel", "age_group", "gender", "time_slot"]
SRM_ALPHA = 0.01   # SRM 게이트는 본 검정(0.05)보다 보수적 — 놓치지 않는 쪽 우선
BALANCE_ALPHA = 0.05
EXPECTED_RATIO = {"A": 0.5, "B": 0.5}  # bot_main.py의 random.choice(["A","B"]) 기준


# ============================================================
# 0. 데이터 로드 (원본: analyze.py / srm_check.py / covariate_balance_check.py 공통)
# ============================================================
def load_df() -> pd.DataFrame:
    """main.db(본실험, A/B 무작위 배정) visit_log 테이블 전체를 읽어온다."""
    conn = sqlite3.connect(get_db_path("main"))
    df = pd.read_sql("SELECT * FROM visit_log", conn)
    conn.close()
    return df


# ============================================================
# 1. SRM 게이트 — 배정 비율이 50:50대로 이루어졌는가 (원본: srm_check.py)
# ============================================================
def check_srm(df: pd.DataFrame) -> bool:
    """카이제곱 적합도 검정으로 관측 배정 비율 vs 기대 비율(50:50)을 비교한다.

    scipy 없이도 검증 가능하도록 math.erfc로 직접 구현한 값과
    scipy.stats.chisquare 결과를 교차 검증한다 (df=1 카이제곱 == 표준정규 Z^2).
    반환값 True = 이상 없음, False = SRM 의심(원인 확인 필요).
    """
    obs_counts = df["group_name"].value_counts().to_dict()
    n_total = sum(obs_counts.values())

    chi2 = 0.0
    for group, ratio in EXPECTED_RATIO.items():
        observed = obs_counts.get(group, 0)
        expected = n_total * ratio
        chi2 += (observed - expected) ** 2 / expected
    p_value = math.erfc(math.sqrt(chi2 / 2))

    scipy_chi2, scipy_p = chisquare(
        f_obs=[obs_counts.get(g, 0) for g in EXPECTED_RATIO],
        f_exp=[n_total * r for r in EXPECTED_RATIO.values()],
    )
    assert math.isclose(chi2, scipy_chi2, rel_tol=1e-9), "수동 chi2 계산이 scipy와 불일치"
    assert math.isclose(p_value, scipy_p, rel_tol=1e-6), "수동 p-value 계산이 scipy와 불일치"

    print("=" * 55)
    print("  [1/6] SRM(Sample Ratio Mismatch) 게이트")
    print("=" * 55)
    print(f"  관측치: {obs_counts}  (총 {n_total}건)")
    print(f"  기대치: 각 {n_total / 2:.0f}건 (50:50 가정)")
    print(f"  chi2 = {chi2:.3f} (df=1),  p-value = {p_value:.5f}  "
          f"(scipy 교차검증 일치 확인 완료)")

    passed = p_value >= SRM_ALPHA
    verdict = "이상 없음" if passed else "⚠️ SRM 의심됨 — 본 분석 전 원인 확인 필요"
    print(f"  판정(α={SRM_ALPHA}): {verdict}")
    print("=" * 55 + "\n")
    return passed


# ============================================================
# 2. 공변량 균형 게이트 — 구성비가 A/B 간에 쏠려있지 않은가 (원본: covariate_balance_check.py)
# ============================================================
def check_covariate_balance(df: pd.DataFrame) -> bool:
    """공변량(범주형) x group_name 분할표에 카이제곱 독립성 검정을 수행한다.

    귀무가설: "공변량 분포가 A/B 간에 같다". p >= 0.05면 "이 공변량에서
    A/B가 불균형하다는 근거가 없다"로 해석한다(균형을 증명하는 것이 아니라
    불균형의 증거가 없다는 뜻).
    """
    print("=" * 55)
    print("  [2/6] 공변량 균형(Covariate Balance) 게이트")
    print("=" * 55)

    results = []
    for col in COVARIATES:
        contingency = pd.crosstab(df[col], df["group_name"])
        chi2, p_value, dof, _ = chi2_contingency(contingency, correction=False)
        balanced = p_value >= BALANCE_ALPHA
        results.append({"covariate": col, "chi2": chi2, "dof": dof,
                         "p_value": p_value, "balanced": balanced})
        print(f"  [{col}] chi2={chi2:.3f} (dof={dof}), p={p_value:.4f} → "
              f"{'균형' if balanced else '⚠ 불균형 의심'}")

    summary = pd.DataFrame(results)
    all_balanced = bool(summary["balanced"].all())
    if all_balanced:
        print("\n  판정: 모든 공변량에서 A/B 간 유의한 차이 없음 — 관측된 전환율 차이를")
        print("        공변량 쏠림으로 설명하기 어려움.")
    else:
        unbalanced = summary[~summary["balanced"]]["covariate"].tolist()
        print(f"\n  판정: ⚠ 불균형 의심 공변량 {unbalanced} — 해석 시 주의, "
              "필요하면 층화 분석 고려.")
    print("=" * 55 + "\n")
    return all_balanced


# ============================================================
# 3. EDA 요약 — 군별 전환율/체류시간/찜 전환 (원본: analyze.py의 print_summary)
# ============================================================
def print_summary(df: pd.DataFrame) -> None:
    print("=" * 55)
    print("  [3/6] King-coffee 텀블러 A/B 테스트 EDA 요약")
    print("=" * 55)
    print(f"  총 방문자: {len(df)}명")
    print(f"  선택 완료: {df['selected'].sum()}명 ({df['selected'].mean()*100:.1f}%)")
    print(f"  찜하기:    {df['wishlist'].sum()}명\n")

    print("[ 군별 방문자 수 · 선택(전환) 비율 ]")
    grp = df.groupby('group_name').agg(visits=('selected', 'size'), selected=('selected', 'sum'))
    grp['rate'] = grp['selected'] / grp['visits'] * 100
    for name, row in grp.iterrows():
        bar = '#' * int(row['rate'] / 100 * 30)
        print(f"  {name}군: {int(row['visits']):>4}명 중 {int(row['selected']):>4}명 선택 "
              f"({row['rate']:5.1f}%)  {bar}")

    print("\n[ 연령대별 군 분포 (선택자 기준) ]")
    ct = pd.crosstab(df[df['selected'] == 1]['age_group'],
                      df[df['selected'] == 1]['group_name'], margins=True)
    print(ct.to_string())

    print("\n[ 유입 채널별 군 분포 (선택자 기준) ]")
    ct2 = pd.crosstab(df[df['selected'] == 1]['channel'],
                       df[df['selected'] == 1]['group_name'], margins=True)
    print(ct2.to_string())

    print("\n[ 군별 평균 체류 시간(초, 조회자 기준) ]")
    for name, sub in df.groupby('group_name'):
        viewers = sub[sub['time_on'] > 0]['time_on']
        avg = viewers.mean() if len(viewers) > 0 else 0
        print(f"  {name}군 평균: {avg:.1f}초  (n={len(viewers)})")

    print("\n[ 찜 → 선택 전환율 (군별) ]")
    for name, sub in df.groupby('group_name'):
        wished = sub[sub['wishlist'] == 1]
        converted = wished[wished['selected'] == 1]
        rate = len(converted) / len(wished) * 100 if len(wished) > 0 else 0
        print(f"  {name}군: 찜 {len(wished)}명 → 선택 {len(converted)}명 ({rate:.1f}%)")
    print("=" * 55 + "\n")


def plot_segment_conversion_rates(df: pd.DataFrame) -> None:
    """세그먼트별로 A군/B군 선택률(%)을 나란히 비교해 어느 구간에서 A/B 효과가
    두드러지는지 눈으로 확인한다 (통계적 유의성 검정 아님 — 참고용 시각화).

    위 [ 연령대별 군 분포(선택자 기준) ] 같은 "선택자 수" crosstab은 군별
    전체 선택률이 다르면(A 59.7% vs B 25.4%) 특정 세그먼트가 실제보다 더
    쏠려 보이는 착시가 생길 수 있다 — 그래서 인원수가 아니라 세그먼트 내
    "선택률"로 비교한다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, COVARIATES):
        rates = df.groupby([col, "group_name"])["selected"].mean().unstack() * 100
        rates.plot(kind="bar", ax=ax)
        ax.set_title(f"{col}별 A/B 선택률")
        ax.set_ylabel("선택률 (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="group")

    fig.suptitle("세그먼트별 A vs B 선택률 비교 (참고용, 통계검정 아님)", fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================
# 4. 두 비율 z-검정 — A/B 선택률 차이가 통계적으로 유의한가 (원본: AB_TEST_SIMULATION.py #5)
# ============================================================
def run_z_test(df: pd.DataFrame) -> dict:
    """A/B 선택률 차이를 두 비율 z-검정으로 검정하고, 카이제곱 독립성 검정으로 교차 검증한다.

    검정통계량(z)은 귀무가설(두 비율이 같다) 하의 합동(pooled) 비율로 계산하고,
    신뢰구간은 각 군의 실제 비율(unpooled)로 계산한다 — 표준 관행.
    """
    n_A = int((df['group_name'] == 'A').sum())
    n_B = int((df['group_name'] == 'B').sum())
    x_A = int(df[df['group_name'] == 'A']['selected'].sum())
    x_B = int(df[df['group_name'] == 'B']['selected'].sum())

    p_A = x_A / n_A
    p_B = x_B / n_B
    diff = p_A - p_B

    p_pooled = (x_A + x_B) / (n_A + n_B)
    se_pooled = (p_pooled * (1 - p_pooled) * (1 / n_A + 1 / n_B)) ** 0.5
    z_stat = diff / se_pooled
    p_value = 2 * norm.sf(abs(z_stat))

    se_A = (p_A * (1 - p_A) / n_A) ** 0.5
    se_B = (p_B * (1 - p_B) / n_B) ** 0.5
    se_unpooled = (se_A ** 2 + se_B ** 2) ** 0.5
    z_crit = norm.ppf(0.975)  # 95% 신뢰구간
    ci_low, ci_high = diff - z_crit * se_unpooled, diff + z_crit * se_unpooled

    print("=" * 55)
    print("  [4/6] 두 비율 z-검정: A vs B 선택률(전환율) 차이")
    print("=" * 55)
    print(f"  A군: {x_A}/{n_A} = {p_A * 100:.1f}%")
    print(f"  B군: {x_B}/{n_B} = {p_B * 100:.1f}%")
    print(f"  차이(A-B): {diff * 100:.1f}%p   95% CI: [{ci_low * 100:.1f}%p, {ci_high * 100:.1f}%p]")
    print(f"  z = {z_stat:.3f},  p-value = {p_value:.2e}")
    print(f"  판정(α=0.05): "
          f"{'유의함 — A/B 차이가 우연으로 보기 어려움' if p_value < 0.05 else '유의하지 않음'}")

    # 카이제곱 독립성 검정으로 교차 검증 (2x2, df=1) — z-검정과 수학적으로 동치(chi2 == z^2)
    contingency = pd.crosstab(df['group_name'], df['selected'])
    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency, correction=False)
    print(f"\n  [ 카이제곱 교차 검증 ] chi2={chi2_stat:.3f} (z^2={z_stat**2:.3f}), p={chi2_p:.2e}")
    print("=" * 55 + "\n")

    return {"n_A": n_A, "n_B": n_B, "x_A": x_A, "x_B": x_B,
            "p_A": p_A, "p_B": p_B, "se_A": se_A, "se_B": se_B,
            "z_stat": z_stat, "p_value": p_value, "z_crit": z_crit,
            "chi2_stat": chi2_stat}


def plot_conversion_rate_with_ci(z_test_result: dict) -> None:
    """A군/B군 선택률(전환율)을 95% 신뢰구간과 함께 막대그래프로 시각화한다.

    run_z_test()가 diff(A-B)에 대해 구했던 것과 같은 Wald 근사(z_crit ×
    표준오차)를 각 군에 개별 적용한 신뢰구간이다 — 두 막대의 오차범위가
    겹치지 않을수록 차이가 우연이 아닐 가능성이 높다는 걸 시각적으로 보여준다
    (다만 통계적 유의성 판정 자체는 [4/6]의 z-검정 p-value를 따른다).
    """
    p_A, p_B = z_test_result["p_A"], z_test_result["p_B"]
    se_A, se_B = z_test_result["se_A"], z_test_result["se_B"]
    z_crit = z_test_result["z_crit"]

    rates = [p_A * 100, p_B * 100]
    ci_halfwidths = [z_crit * se_A * 100, z_crit * se_B * 100]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(["A", "B"], rates, yerr=ci_halfwidths, capsize=10,
                   color=["#4C72B0", "#DD8452"])
    for bar, rate, half in zip(bars, rates, ci_halfwidths):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + half + 1.5,
                 f"{rate:.1f}%", ha="center", fontsize=11)

    ax.set_ylabel("선택률 (%)")
    ax.set_title("A vs B 선택률(전환율) — 95% 신뢰구간")
    ax.set_ylim(0, max(r + h for r, h in zip(rates, ci_halfwidths)) + 12)
    fig.tight_layout()
    plt.show()


# ============================================================
# 5. 효과크기 — 차이가 "얼마나" 큰가 (원본: AB_TEST_SIMULATION.py #6)
# ============================================================
def cohens_h(p1: float, p2: float) -> float:
    """두 비율 차이를 위한 표준 효과크기(arcsine 변환 기반).
    %p 절대 차이와 달리 베이스라인 수준에 따른 왜곡을 보정해준다."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def report_effect_size(z_test_result: dict) -> None:
    p_A, p_B = z_test_result["p_A"], z_test_result["p_B"]
    n_A, n_B = z_test_result["n_A"], z_test_result["n_B"]
    x_A, x_B = z_test_result["x_A"], z_test_result["x_B"]
    chi2_stat = z_test_result["chi2_stat"]

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

    risk_ratio = p_A / p_B

    print("=" * 55)
    print("  [5/6] 효과크기(Effect Size)")
    print("=" * 55)
    print(f"  절대 차이(A-B):      {(p_A - p_B) * 100:.1f}%p")
    print(f"  Cohen's h:           {h:.3f}  ({h_label})")
    print(f"  상대위험도(A/B):     {risk_ratio:.2f}배  (A가 B보다 {risk_ratio:.2f}배 더 많이 선택됨)")

    print("=" * 55 + "\n")


# ============================================================
# 6. CSV 추출 (원본: analyze.py의 export_csv)
# ============================================================
def export_csv(df: pd.DataFrame, path: str = "king_coffee_log.csv") -> None:
    print("=" * 55)
    print("  [6/6] CSV 추출")
    print("=" * 55)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  CSV 저장 완료: {path}  ({len(df)}행)")
    print("=" * 55)


# ============================================================
# main — 위 6단계를 순서대로 실행
# ============================================================
if __name__ == "__main__":
    df = load_df()
    if df.empty:
        print("데이터가 없습니다. 먼저 bot_main.py를 실행하세요.")
        sys.exit(1)

    check_srm(df)
    check_covariate_balance(df)
    print_summary(df)
    plot_segment_conversion_rates(df)
    z_result = run_z_test(df)
    plot_conversion_rate_with_ci(z_result)
    report_effect_size(z_result)
    export_csv(df)
