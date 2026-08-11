import os
import random
from flask import Flask, request, jsonify, session, Response
from database import init_all, get_conn
from datetime import datetime

VALID_STAGES = ("pilot", "main")

app = Flask(__name__)
# 세션 쿠키 서명용 — 로컬 프로토타입이라 프로세스 시작 시 무작위 생성 (재시작 시 세션 초기화됨)
app.secret_key = os.urandom(24)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# ── 메인 페이지: 방문자를 A/B 중 하나에 무작위 배정(세션 고정) 후 해당 안만 렌더링 ──
# Claude 디자인 툴이 내보낸 HTML은 자체 클라이언트 번들러가 {{ }} 문법을 쓰므로,
# Flask의 Jinja2 렌더러(render_template)를 거치면 서버사이드 템플릿 문법으로 오인되어
# 깨진다. 그래서 원본 그대로 읽어서 내려준다 (서버사이드 치환 없음).
@app.route("/")
def index():
    if "group" not in session:
        session["group"] = random.choice(["A", "B"])
    filename = "index_a.html" if session["group"] == "A" else "index_b.html"
    with open(os.path.join(TEMPLATES_DIR, filename), encoding="utf-8") as f:
        html = f.read()
    return Response(html, mimetype="text/html")

# ── 봇/실제 고객이 행동 로그를 POST로 보내면 DB에 저장 ──
@app.route("/log", methods=["POST"])
def log_action():
    data = request.get_json()
    stage = data.get("stage", "main")
    if stage not in VALID_STAGES:
        return jsonify({"status": "error", "message": f"알 수 없는 stage: {stage!r}"}), 400

    conn = get_conn(stage)
    conn.execute("""
        INSERT INTO visit_log (
            user_id, age_group, gender, channel, time_slot, group_name, stage,
            viewed, time_on, wishlist, selected, session_sec, reason, timestamp
        ) VALUES (
            :user_id, :age_group, :gender, :channel, :time_slot, :group_name, :stage,
            :viewed, :time_on, :wishlist, :selected, :session_sec, :reason, :timestamp
        )
    """, {
        "user_id":    data.get("user_id"),
        "age_group":  data.get("age_group"),
        "gender":     data.get("gender"),
        "channel":    data.get("channel"),
        "time_slot":  data.get("time_slot"),
        "group_name": data.get("group_name"),
        "stage":      stage,
        "viewed":     int(data.get("viewed", 0)),
        "time_on":    int(data.get("time_on", 0)),
        "wishlist":   int(data.get("wishlist", 0)),
        "selected":   int(data.get("selected", 0)),
        "session_sec":int(data.get("session_sec", 0)),
        "reason":     data.get("reason", ""),
        "timestamp":  datetime.now().isoformat(),
    })
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

# ── 현재 A/B 군별 전환(선택) 현황 확인용 (stage별로 분리 조회) ──
@app.route("/stats/<stage>")
def stats(stage):
    if stage not in VALID_STAGES:
        return jsonify({"status": "error", "message": f"알 수 없는 stage: {stage!r}"}), 400

    conn = get_conn(stage)
    rows = conn.execute("""
        SELECT group_name,
               COUNT(*) AS visits,
               SUM(selected) AS selected_cnt
        FROM visit_log
        GROUP BY group_name
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM visit_log").fetchone()[0]
    conn.close()
    result = {
        "stage": stage,
        "total": total,
        "groups": {
            r["group_name"]: {
                "visits": r["visits"],
                "selected": r["selected_cnt"],
                "rate": round(r["selected_cnt"] / r["visits"] * 100, 1) if r["visits"] else 0,
            }
            for r in rows
        },
    }
    return jsonify(result)

if __name__ == "__main__":
    init_all()
    print("[Flask] http://127.0.0.1:5000 에서 실행 중")
    app.run(debug=False, port=5000)
