"""KrishiSahay Database — SQLite for feedback and cache"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent.parent / "data" / "krishisahay.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            answer TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            rating INTEGER CHECK(rating IN (1, -1)),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def save_query(query_text: str, language: str, answer: str, sources: list) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (query_text, language, answer, sources) VALUES (?, ?, ?, ?)",
        (query_text, language, answer, json.dumps(sources))
    )
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id


def save_feedback(query_id: int, rating: int, comment: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (query_id, rating, comment) VALUES (?, ?, ?)",
        (query_id, rating, comment)
    )
    conn.commit()
    conn.close()


def get_recent_queries(limit: int = 10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM queries ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows
