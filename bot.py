"""
bot.py — Claude API 기반 가상 고객 시뮬레이터

흐름:
1. 페르소나 무작위 생성 (연령대/성별/유입채널/시간대)
2. Claude API에 페르소나 + 상품 설명 전달
3. Claude가 "이 고객이라면 어떻게 행동할까?" 스스로 판단해서 JSON 반환
4. 반환된 행동을 Flask /log 엔드포인트로 POST → SQLite 적재
"""

import sys
import anthropic
import requests
import random
import time
import json
import re
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 한글/이모지 출력 깨짐 방지

load_dotenv()  # .env 파일의 ANTHROPIC_API_KEY를 os.environ으로 로드 (없으면 무시)

# ── 설정 ───────────────────────────────────────────────────
# 이 파일은 공통 로직(페르소나 생성/Claude 호출/로그 전송)만 담당한다.
# 실제 실행은 bot_pilot.py 또는 bot_main.py에서 한다.
FLASK_URL  = "http://127.0.0.1:5000"
DELAY_SEC  = 0.5       # API 과부하 방지용 딜레이
MODEL_NAME = "claude-haiku-4-5-20251001"

# ── 페르소나 풀 ────────────────────────────────────────────
AGE_GROUPS = ["10s", "20s", "30s", "40s", "50s+"]
GENDERS    = ["M", "F"]
CHANNELS   = ["SNS", "direct", "friend_share", "search"]
TIME_SLOTS = ["morning", "lunch", "afternoon", "dinner", "late"]

# 채널별 연령대 가중치 (현실적인 치킨집 고객 분포)
CHANNEL_AGE_WEIGHT = {
    "SNS":          [0.20, 0.40, 0.25, 0.10, 0.05],
    "direct":       [0.05, 0.15, 0.30, 0.30, 0.20],
    "friend_share": [0.10, 0.30, 0.30, 0.20, 0.10],
    "search":       [0.05, 0.25, 0.35, 0.25, 0.10],
}

def make_persona(forced_group: str = None) -> dict:
    """forced_group이 주어지면 그 그룹으로 고정 배정(파일럿용),
    없으면 기존처럼 A/B 무작위 배정(본실험용, between-subject)."""
    channel   = random.choice(CHANNELS)
    age_group = random.choices(AGE_GROUPS, weights=CHANNEL_AGE_WEIGHT[channel])[0]
    gender    = random.choice(GENDERS)
    time_slot = random.choice(TIME_SLOTS)
    user_id   = f"bot_{int(time.time()*1000)}_{random.randint(100,999)}"
    group     = forced_group if forced_group else random.choice(["A", "B"])
    return dict(user_id=user_id, age_group=age_group,
                gender=gender, channel=channel, time_slot=time_slot, group=group)

# ── Claude API 호출 ────────────────────────────────────────
client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수(.env 또는 셸)에서 자동 로드

_DESIGN_LINES = {
    "A": "원형 이중 테두리 안에 브랜드 캐릭터(여자아이+커피잔 실루엣)만. 상호명 없음. 심플하고 미니멀한 구성.",
    "B": "하트형 이중 테두리 안에 캐릭터. 테두리 외곽을 따라 'KING-COFFEE' 상호명 + '2nd Anniversary' 텍스트. 장식적인 테두리 모양과 기념 문구가 특징.",
}

def _build_product_desc(group: str) -> str:
    """방문자는 자신이 배정된 안 하나만 본다 (between-subject).
    여러 안을 동시에 나열하지 않으므로 안 사이의 위치 편향(position bias) 문제 자체가 없다."""
    return f"""

King-coffee (치킨과 커피를 함께 파는 매장) 2주년 기념 무료 증정 텀블러:

- {_DESIGN_LINES[group]}
"""

def _build_example_json() -> str:
    """이 안 하나에 대한 반응(관심/찜/선택 여부)을 boolean으로 표현하는 예시.
    A/B 중 어느 쪽이 배정됐는지와 무관하게 형식만 보여주는 예시이므로 매번 무작위로 생성해
    Claude가 예시 값 자체를 정답으로 모방(anchoring)하지 않도록 한다."""
    viewed   = random.choice([True, True, True, False])  # 대부분은 봄, 일부는 이탈
    wishlist = viewed and random.choice([True, False])
    selected = wishlist and random.choice([True, False])
    time_on  = random.choice([6, 10, 15, 20]) if viewed else 0
    example = {
        "viewed": viewed,
        "time_on": time_on,
        "wishlist": wishlist,
        "selected": selected,
        "session_sec": time_on + random.randint(3, 8),
        "reason": "예시 문구일 뿐 실제 판단과 무관함",
    }
    return json.dumps(example, ensure_ascii=False)

def ask_claude(persona: dict) -> dict:
    example_json = _build_example_json()
    product_desc = _build_product_desc(persona["group"])
    prompt = f"""
당신은 King-coffee(치킨과 커피를 함께 파는 매장)를 자주 이용하는 단골 고객입니다.
이미 이 브랜드에 어느 정도 신뢰와 애착이 있는 상태이고, 2주년 기념으로 텀블러를
무료로 준다는 소식을 듣고 아래 페이지를 보게 되었습니다. 즉 낯선 사람을 설득해야
하는 일반 랜딩페이지가 아니라, 이미 관계가 있는 단골에게 공짜로 주는 이벤트이므로
일반적인 웹 방문자보다 관심을 가질 가능성이 자연스럽게 더 높습니다.

다만 모든 단골이 무조건 받아가는 것도 비현실적입니다. 예를 들어 "집에 이미 비슷한
텀블러가 많다", "이번 디자인이 내 취향은 아니다", "지금 바빠서 대충 훑고 넘어간다"
같은 자연스러운 이유로 관심이 없거나 이탈하는 경우도 있을 수 있습니다. 이런 관심도와
이탈 가능성은 아래 페르소나(연령대·유입경로·방문시간대)에 따라 그럴듯하게 달라져야
합니다.

이 고객은 아래 텀블러 디자인 "하나만" 보게 됩니다 (다른 대안은 존재하지 않습니다).

페르소나:
- 연령대: {persona['age_group']}
- 성별: {persona['gender']}
- 유입 경로: {persona['channel']} (SNS=소셜미디어 링크, direct=매장 방문 후 QR코드, friend_share=지인이 공유, search=직접 검색)
- 방문 시간대: {persona['time_slot']}

{product_desc}

이 고객이 페이지를 봤을 때의 행동을 자신의 취향과 상황에 맞게 결정하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이).
아래는 형식(키 이름, 타입, 구조)을 보여주기 위한 예시일 뿐이며, 실제 값은 이 예시와
무관하게 페르소나의 취향에 맞게 자유롭게 판단하세요:
{example_json}

규칙:
- viewed: 이 디자인을 실제로 살펴봤는지 (true/false)
- time_on: 체류 시간(초), 안 봤으면 0
- wishlist: 마음에 들어 찜했는지 (true/false, viewed=false면 항상 false)
- selected: 최종적으로 이 텀블러를 받기로 선택했는지 (true/false, viewed=false면 항상 false)
- session_sec: 총 체류 시간(초)
- reason: 선택/이탈 이유 한 문장
"""

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"파싱 실패: {raw}")

# ── 로그 전송 ──────────────────────────────────────────────
def send_log(persona: dict, decision: dict):
    payload = {
        "user_id":    persona["user_id"],
        "age_group":  persona["age_group"],
        "gender":     persona["gender"],
        "channel":    persona["channel"],
        "time_slot":  persona["time_slot"],
        "group_name": persona["group"],
        "stage":      persona.get("stage", "main"),
        "viewed":     1 if decision.get("viewed") else 0,
        "time_on":    decision.get("time_on", 0),
        "wishlist":   1 if decision.get("wishlist") else 0,
        "selected":   1 if decision.get("selected") else 0,
        "session_sec":decision.get("session_sec", 0),
        "reason":     decision.get("reason", ""),
    }
    resp = requests.post(f"{FLASK_URL}/log", json=payload, timeout=5)
    resp.raise_for_status()

# ── 메인 루프 ──────────────────────────────────────────────
def run(n: int, forced_group: str = None, stage: str = "main"):
    """n명 시뮬레이션 실행.
    forced_group을 주면(예: "A") 전원 그 그룹 고정 배정(파일럿),
    None이면 A/B 무작위 배정(본실험). stage는 로그에 남는 "pilot"/"main" 구분값."""
    print(f"\n{'='*55}")
    print(f"  King-coffee 가상 고객 시뮬레이션 ({stage} 모드)")
    print(f"  총 {n}명 · 모델: {MODEL_NAME}")
    print(f"{'='*55}\n")

    success = 0
    group_counts = {"A": 0, "B": 0}
    for i in range(1, n + 1):
        persona = make_persona(forced_group=forced_group)
        persona["stage"] = stage
        try:
            decision = ask_claude(persona)
            send_log(persona, decision)
            group_counts[persona["group"]] += 1
            sel = f"{persona['group']}안 선택" if decision.get("selected") else "이탈/비선택"
            reason = decision.get("reason", "")[:30]
            print(f"[{i:03d}/{n}] {persona['age_group']} {persona['gender']} "
                  f"({persona['channel']:12s}, {persona['group']}군) → {sel}  |  {reason}")
            success += 1
        except Exception as e:
            print(f"[{i:03d}/{n}] 오류: {e}")

        time.sleep(DELAY_SEC)

    print(f"\n완료: {success}/{n}명 처리 성공")
    if stage == "main":
        print(f"배정 결과 — A: {group_counts['A']}명, B: {group_counts['B']}명  (SRM 체크용)")
    print("분석하려면: python analyze.py")

if __name__ == "__main__":
    print("이 파일은 공통 로직 모듈입니다. 직접 실행하지 말고 아래 중 하나를 실행하세요:")
    print("  python bot_pilot.py   (파일럿: A안 단독 노출)")
    print("  python bot_main.py    (본실험: A/B 무작위 배정)")
