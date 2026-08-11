"""
sample_size_calc.py — 파일럿 베이스라인 + MDE 기반 본실험 표본수 산출

파일럿(bot_pilot.py, A안 단독 노출)에서 얻은 베이스라인 선택률을 바탕으로,
"본실험에서 A/B 간에 이 정도 차이(MDE)는 반드시 탐지하고 싶다"는 목표를
넣으면 군당 필요한 최소 표본수를 두 비율 z-검정 공식으로 계산한다.

주의: 파일럿 데이터가 아직 없으면 load_pilot_baseline()이 실패한다.
이 스크립트는 pilot.db에 데이터가 쌓인 뒤에 실행할 것 — 지금은 로직만
구현해둔 상태이며 실행하지 않는다.

실행(파일럿 완료 후): python pilot/sample_size_calc.py
"""

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

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트 (그래프 깨짐 방지)
plt.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글/이모지 출력 깨짐 방지

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_path


def load_pilot_baseline() -> float:
    """pilot.db(A안 단독 노출)에서 selected 평균 = 베이스라인 선택률을 구한다."""
    conn = sqlite3.connect(get_db_path("pilot"))
    df = pd.read_sql("SELECT selected FROM visit_log", conn)
    conn.close()
    if df.empty:
        raise ValueError("pilot.db에 데이터가 없습니다. 먼저 bot_pilot.py를 실행하세요.")
    return df["selected"].mean()


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
):
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


def plot_power_curve(
    p1: float,
    mde_list,
    alpha: float = 0.05,
    relative: bool = False,
    n_min: int = 20,
    save_path: str | None = None,
):
    """군당 표본수(n)를 늘려갈 때 검정력이 어떻게 변하는지 MDE별로 곡선을 그린다.

    두 비율 z-검정 기준 효과크기(Cohen's h)를 써서 required_sample_size와
    동일한 가정(alpha, 양측검정)으로 검정력을 계산한다.
    """
    analysis = NormalIndPower()

    # x축 범위: 가장 작은 MDE(=가장 많은 표본 필요)를 기준으로 여유 있게 설정
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


if __name__ == "__main__":
    baseline = load_pilot_baseline()
    # 절대 %p 기준 MDE 후보 — 파일럿 결과 보고 조정
    report(baseline, mde_list=[0.03, 0.05, 0.10, 0.15], relative=False)


