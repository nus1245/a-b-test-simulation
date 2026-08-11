"""
analyze.py — main.db(본실험 A/B) 로그 분석 + CSV 추출

실행: python main/analyze.py
"""

import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_path

def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(get_db_path("main"))
    df = pd.read_sql("SELECT * FROM visit_log", conn)
    conn.close()
    return df

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  King-coffee 텀블러 A/B 테스트 분석 리포트")
    print(f"{'='*55}")
    print(f"  총 방문자: {len(df)}명")
    print(f"  선택 완료: {df['selected'].sum()}명 "
          f"({df['selected'].mean()*100:.1f}%)")
    print(f"  찜하기:    {df['wishlist'].sum()}명\n")

    # ── 군(A/B)별 표본 수 · 전환율 ──────────────────────────
    print("[ 군별 방문자 수 · 선택(전환) 비율 ]")
    grp = df.groupby('group_name').agg(
        visits=('selected', 'size'),
        selected=('selected', 'sum'),
    )
    grp['rate'] = grp['selected'] / grp['visits'] * 100
    for name, row in grp.iterrows():
        bar = '#' * int(row['rate'] / 100 * 30)
        print(f"  {name}군: {int(row['visits']):>4}명 중 {int(row['selected']):>4}명 선택 "
              f"({row['rate']:5.1f}%)  {bar}")

    # ── 연령대 × 군 크로스탭 (선택자만) ────────────────────
    print("\n[ 연령대별 군 분포 (선택자 기준) ]")
    ct = pd.crosstab(df[df['selected'] == 1]['age_group'],
                      df[df['selected'] == 1]['group_name'], margins=True)
    print(ct.to_string())

    # ── 유입 채널 × 군 ──────────────────────────────────
    print("\n[ 유입 채널별 군 분포 (선택자 기준) ]")
    ct2 = pd.crosstab(df[df['selected'] == 1]['channel'],
                       df[df['selected'] == 1]['group_name'], margins=True)
    print(ct2.to_string())

    # ── 군별 평균 체류 시간 ──────────────────────────────
    print("\n[ 군별 평균 체류 시간(초, 조회자 기준) ]")
    for name, sub in df.groupby('group_name'):
        viewers = sub[sub['time_on'] > 0]['time_on']
        avg = viewers.mean() if len(viewers) > 0 else 0
        print(f"  {name}군 평균: {avg:.1f}초  (n={len(viewers)})")

    # ── 찜 → 선택 전환율 ──────────────────────────────────
    print("\n[ 찜 → 선택 전환율 (군별) ]")
    for name, sub in df.groupby('group_name'):
        wished = sub[sub['wishlist'] == 1]
        converted = wished[wished['selected'] == 1]
        rate = len(converted)/len(wished)*100 if len(wished) > 0 else 0
        print(f"  {name}군: 찜 {len(wished)}명 → 선택 {len(converted)}명 ({rate:.1f}%)")

    print(f"\n{'='*55}\n")

def export_csv(df: pd.DataFrame, path: str = "king_coffee_log.csv"):
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"CSV 저장 완료: {path}  ({len(df)}행)")

if __name__ == "__main__":
    df = load_df()
    if df.empty:
        print("데이터가 없습니다. 먼저 bot_main.py를 실행하세요.")
    else:
        print_summary(df)
        export_csv(df)
