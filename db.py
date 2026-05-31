import sqlite3
import os
from datetime import datetime, timezone

import config

DB_PATH = os.path.join(config.WORK_DIR, "checks.db")

def _connect() -> sqlite3.Connection:
    os.makedirs(config.WORK_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at  TEXT    NOT NULL,
                filename    TEXT    NOT NULL,
                file_type   TEXT    NOT NULL,
                brand       TEXT,
                is_fake     INTEGER,
                similarity  REAL,
                cls_conf    REAL,
                threshold   REAL
            )
        """)
        conn.commit()

def log_check(filename: str, file_type: str,
              detections: list, threshold: float):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with _connect() as conn:
        if file_type == "image" and detections:
            for d in detections:
                conn.execute(
                    """INSERT INTO checks
                       (checked_at, filename, file_type,
                        brand, is_fake, similarity, cls_conf, threshold)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (now, filename, file_type,
                     d.get("brand"), int(d.get("is_fake", 0)),
                     d.get("similarity"), d.get("cls_conf"), threshold),
                )
        else:
            conn.execute(
                """INSERT INTO checks
                   (checked_at, filename, file_type, threshold)
                   VALUES (?,?,?,?)""",
                (now, filename, file_type, threshold),
            )
        conn.commit()

def get_history(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM checks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def get_stats() -> dict:
    with _connect() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM checks").fetchone()[0]
        total_fake = conn.execute(
            "SELECT COUNT(*) FROM checks WHERE is_fake = 1").fetchone()[0]
        total_real = conn.execute(
            "SELECT COUNT(*) FROM checks WHERE is_fake = 0").fetchone()[0]
        top_brands = conn.execute("""
            SELECT brand, COUNT(*) as cnt
            FROM checks
            WHERE brand IS NOT NULL
            GROUP BY brand
            ORDER BY cnt DESC
            LIMIT 5
        """).fetchall()

    return {
        "total":      total,
        "fake":       total_fake,
        "real":       total_real,
        "top_brands": [dict(r) for r in top_brands],
    }