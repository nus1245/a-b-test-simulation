"""
pilot_analysis.py — 파일럿 EDA + 본실험 표본수/검정력 분석

King Coffee 텀블러 증정 이벤트 A/B 테스트를 위한 파일럿 단계 분석 스크립트.
LLM 페르소나 봇(bot_pilot.py)이 A안만 단독으로 노출된 상태에서 남긴 방문
로그(pilot.db)를 읽어, 다음 네 단계를 순서대로 수행한다.

    1) EDA          — 세그먼트(연령대/성별/유입경로/방문시간대)별 선택률을
                       살펴보고, 표본수 산식에 넣을 베이스라인이 특정 세그먼트에
                       치우쳐 있지 않은지 확인한다.
    2) 베이스라인 산출 — 전체 선택률(selected 평균)을 본실험 표본수 계산의
                       입력값(p1)으로 사용한다.
    3) 표본수 계산    — "본실험에서 이 정도 차이(MDE)는 반드시 탐지하고
                       싶다"는 목표를 넣으면, 두 비율 z-검정 공식으로 군당
                       필요한 최소 표본수를 역산한다.
    4) 검정력 곡선    — 표본수를 늘려갈수록 검정력이 어떻게 올라가는지를
                       MDE별로 시각화해, 표본수 산정 결과를 감(感)으로도
                       검증할 수 있게 한다.

주의: pilot.db에 데이터가 없으면(파일럿 미실행) 2번 단계에서 예외가 난다.
     먼저 `python pilot/bot_pilot.py`로 파일럿을 실행해 로그를 쌓아야 한다.

실행: python pilot/pilot_analysis.py
"""

from __future__ import annotations

import os
import sys
import sqlite3
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글/이모지 출력 깨짐 방지

# Windows 기본 matplotlib은 한글 폰트가 없어 그래프의 한글이 네모(□)로 깨진다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 선택률(selected)을 나눠 볼 세그먼트 컬럼. bot.py가 페르소나를 생성할 때
# 부여하는 속성들이며, DB 스키마는 database.py의 visit_log 테이블 정의를 따른다.
SEGMENT_COLUMNS = ["age_group", "gender", "channel", "time_slot"]


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_pilot_data() -> pd.DataFrame:
    """pilot.db(A안 단독 노출) visit_log 테이블 전체를 DataFrame으로 읽어온다."""
    conn = sqlite3.connect(get_db_path("pilot"))
    df = pd.read_sql("SELECT * FROM visit_log", conn)
    conn.close()
    if df.empty:
        raise ValueError("pilot.db에 데이터가 없습니다. 먼저 bot_pilot.py를 실행하세요.")
    return df


# ============================================================
# 2. EDA — 세그먼트별 선택률 탐색
# ============================================================
def describe_pilot(df: pd.DataFrame) -> None:
    """전체 요약 통계와 세그먼트별 선택률 breakdown을 콘솔에 출력한다."""
    n = len(df)
    print("=" * 55)
    print("  파일럿 EDA — 세그먼트별 선택률")
    print("=" * 55)
    print(f"  전체 표본수         : {n}")
    print(f"  조회율(viewed)      : {df['viewed'].mean() * 100:.1f}%")
    print(f"  위시리스트 담기율    : {df['wishlist'].mean() * 100:.1f}%")
    print(f"  최종 선택률(selected): {df['selected'].mean() * 100:.1f}%")

    for col in SEGMENT_COLUMNS:
        breakdown = (
            df.groupby(col)["selected"]
            .agg(표본수="count", 선택률=lambda s: round(s.mean() * 100, 1))
            .sort_values("선택률", ascending=False)
        )
        print(f"\n  [{col}]")
        print(breakdown.to_string())
    print("=" * 55)


def plot_segment_breakdown(df: pd.DataFrame, save_path: str | None = None) -> None:
    """연령대/성별/유입경로/방문시간대별 선택률을 2x2 막대그래프로 시각화한다.

    베이스라인(p1)이 특정 세그먼트에 쏠려 나온 값은 아닌지 눈으로 확인하기
    위한 그래프이며, 본실험 표본수 산출에 직접 쓰이지는 않는다.
    """
    overall_rate = df["selected"].mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, SEGMENT_COLUMNS):
        rates = df.groupby(col)["selected"].mean().sort_values(ascending=False) * 100
        ax.bar(rates.index.astype(str), rates.values, color="#4C72B0")
        ax.axhline(overall_rate, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{col}별 선택률")
        ax.set_ylabel("선택률 (%)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("파일럿 세그먼트별 선택률 (점선 = 전체 평균)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"세그먼트 breakdown 그래프 저장: {save_path}")
    else:
        plt.show()


# ============================================================
# 3. 베이스라인 산출
# ============================================================
def compute_baseline(df: pd.DataFrame) -> float:
    """A안 단독 노출 시 전체 selected 평균 = 본실험 표본수 계산용 베이스라인(p1)."""
    return df["selected"].mean()


# ============================================================
# 4. 표본수 계산 — 두 비율 z-검정
# ============================================================
def required_sample_size(
    p1: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = False,
) -> int:
    """두 비율 z-검정 기준, 군(A 또는 B)당 필요한 최소 표본수.

    p1       : 대조군(A) 베이스라인 선택률 (0~1), 파일럿에서 추정한 값
    mde      : 최소 탐지 효과(Minimum Detectable Effect).
               relative=False면 절대 %p 차이(예: 0.05 = 5%p 차이),
               relative=True면 p1 대비 상대 비율(예: 0.10 = p1의 10% 상대 증가)
    alpha    : 유의수준 (기본 0.05, 양측검정)
    power    : 검정력 (기본 0.8)
    반환     : 군당 필요 표본수 (올림)
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    p2 = p1 * (1 + mde) if relative else p1 + mde
    p2 = min(max(p2, 0.0), 1.0)

    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)


def report(
    p1: float,
    mde_list,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = False,
) -> None:
    """여러 MDE 후보에 대해 필요 표본수를 표로 출력."""
    print("=" * 55)
    print("  본실험(A/B) 표본수 산출 — 파일럿 베이스라인 기준")
    print("=" * 55)
    print(f"  베이스라인 선택률(A안 단독, 파일럿): {p1 * 100:.1f}%")
    print(f"  유의수준 α={alpha}, 검정력={power}, "
          f"MDE 기준={'상대(%)' if relative else '절대(%p)'}")
    print()
    print(f"  {'MDE':>10} | {'필요 표본수(군당)':>16} | {'총 필요(A+B)':>12}")
    for mde in mde_list:
        n = required_sample_size(p1, mde, alpha=alpha, power=power, relative=relative)
        print(f"  {mde:>10} | {n:>16,} | {n * 2:>12,}")
    print("=" * 55)


# ============================================================
# 5. 검정력 곡선
# ============================================================
def plot_power_curve(
    p1: float,
    mde_list,
    alpha: float = 0.05,
    relative: bool = False,
    n_min: int = 20,
    save_path: str | None = None,
) -> None:
    """군당 표본수(n)를 늘려갈 때 검정력이 어떻게 변하는지 MDE별로 곡선을 그린다.

    required_sample_size와 동일한 가정(두 비율 z-검정, alpha, 양측검정)에서의
    효과크기(Cohen's h)를 사용하므로, 곡선이 검정력 0.8을 넘는 지점의 n은
    required_sample_size의 계산 결과와 일치한다.
    """
    analysis = NormalIndPower()

    # x축 범위: 가장 작은 MDE(=가장 많은 표본이 필요한 케이스)를 기준으로 여유 있게 설정
    n_needed = [
        required_sample_size(p1, mde, alpha=alpha, power=0.8, relative=relative)
        for mde in mde_list
    ]
    n_max = int(max(n_needed) * 1.3)
    n_range = np.linspace(n_min, n_max, 200)

    plt.figure(figsize=(9, 6))
    for mde in mde_list:
        p2 = p1 * (1 + mde) if relative else p1 + mde
        p2 = min(max(p2, 0.0), 1.0)
        effect_size = proportion_effectsize(p1, p2)
        powers = analysis.power(effect_size=effect_size, nobs1=n_range, alpha=alpha, ratio=1.0)
        label = f"MDE={mde} ({'상대' if relative else '절대'})"
        plt.plot(n_range, powers, label=label)

    plt.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="목표 검정력 0.8")
    plt.xlabel("군당 표본수 (n)")
    plt.ylabel("검정력 (Power)")
    plt.title(f"검정력 곡선 (베이스라인 p1={p1 * 100:.1f}%, α={alpha})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"검정력 곡선 저장: {save_path}")
    else:
        plt.show()


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    pilot_df = load_pilot_data()

    # 1) EDA — 세그먼트 쏠림 여부 확인
    describe_pilot(pilot_df)
    plot_segment_breakdown(pilot_df)

    # 2~4) 베이스라인 → 표본수 → 검정력 곡선
    baseline = compute_baseline(pilot_df)
    mde_list = [0.03, 0.05, 0.10, 0.15]  # 절대 %p 기준 MDE 후보 — 파일럿 결과 보고 조정
    report(baseline, mde_list=mde_list, relative=False)
    plot_power_curve(baseline, mde_list=mde_list, relative=False)
