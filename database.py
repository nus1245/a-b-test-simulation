import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일럿(A안 단독 베이스라인)과 본실험(A/B) 데이터를 물리적으로 다른 DB 파일에
# 저장한다. 분석 시 실수로 섞일 여지를 원천 차단하기 위함(stage 컬럼만으로
# 구분하던 이전 방식보다 안전).
DB_PATHS = {
    "pilot": os.path.join(BASE_DIR, "pilot", "pilot.db"),
    "main":  os.path.join(BASE_DIR, "main", "main.db"),
}

def get_db_path(stage: str) -> str:
    if stage not in DB_PATHS:
        raise ValueError(f"알 수 없는 stage: {stage!r} (\"pilot\" 또는 \"main\"만 허용)")
    return DB_PATHS[stage]

def get_conn(stage: str):
    conn = sqlite3.connect(get_db_path(stage))
    conn.row_factory = sqlite3.Row
    return conn

_VISIT_LOG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS visit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     TEXT,
        age_group   TEXT,
        gender      TEXT,
        channel     TEXT,
        time_slot   TEXT,
        group_name  TEXT,
        stage       TEXT,
        viewed      INTEGER DEFAULT 0,
        time_on     INTEGER DEFAULT 0,
        wishlist    INTEGER DEFAULT 0,
        selected    INTEGER DEFAULT 0,
        session_sec INTEGER,
        reason      TEXT,
        timestamp   TEXT
    )
"""

def init_db(stage: str):
    """stage("pilot"/"main")의 DB 파일/테이블이 없으면 생성한다.
    기존처럼 매번 DROP하지 않는다 — 파일럿과 본실험은 서로 다른 시점에
    실행되므로(예: 파일럿 끝낸 뒤 며칠 뒤 본실험), 서버 재시작 한 번에
    다른 단계의 데이터까지 날아가면 안 되기 때문."""
    db_path = get_db_path(stage)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_conn(stage)
    conn.execute(_VISIT_LOG_SCHEMA)
    conn.commit()
    conn.close()
    print(f"[DB] 준비 완료 ({stage}) →", db_path)

def init_all():
    for stage in DB_PATHS:
        init_db(stage)
