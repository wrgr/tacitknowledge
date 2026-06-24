"""
SQLite database layer — users, themes, and model preferences.

Students: emma/liam/sofia/james/priya/tyler — all password Learn@2024
Admin:    admin — password admin123
"""

import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash

DB_FILE      = Path(__file__).parent / "assessments.db"
VALID_THEMES = {"light", "dark", "rustic", "ultra-light", "ultra-dark"}

# pbkdf2 relies only on hashlib.pbkdf2_hmac (present in every Python build).
# werkzeug's default of scrypt needs OpenSSL-with-scrypt, which the macOS
# system Python (linked against LibreSSL) lacks — so pin pbkdf2 for portability.
_HASH_METHOD = "pbkdf2:sha256"

_SEED_STUDENTS = [
    ("emma",  "Learn@2024", "Emma Clarke"),
    ("liam",  "Learn@2024", "Liam Patel"),
    ("sofia", "Learn@2024", "Sofia Nguyen"),
    ("james", "Learn@2024", "James Okafor"),
    ("priya", "Learn@2024", "Priya Singh"),
    ("tyler", "Learn@2024", "Tyler Brooke"),
]


def _conn():
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username           TEXT PRIMARY KEY,
                password_hash      TEXT NOT NULL,
                role               TEXT NOT NULL CHECK(role IN ('admin','student')),
                display_name       TEXT NOT NULL,
                theme              TEXT NOT NULL DEFAULT 'light',
                preferred_provider TEXT NOT NULL DEFAULT '',
                preferred_model    TEXT NOT NULL DEFAULT ''
            )
        """)
        c.commit()


def seed_default_users():
    """Populate the DB with admin + 6 demo students on first run."""
    init_db()
    with _conn() as c:
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            return
        rows = [
            ("admin", generate_password_hash("admin123", method=_HASH_METHOD), "admin",
             "Administrator", "light", "", ""),
        ]
        for uname, pwd, name in _SEED_STUDENTS:
            rows.append((uname, generate_password_hash(pwd, method=_HASH_METHOD), "student",
                         name, "light", "", ""))
        c.executemany(
            "INSERT OR IGNORE INTO users "
            "(username,password_hash,role,display_name,theme,preferred_provider,preferred_model) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        c.commit()
    print("  [db] Seeded 1 admin + 6 students.")
    print("       admin: admin / admin123")
    print("       students: emma liam sofia james priya tyler / Learn@2024")


# ── User CRUD ─────────────────────────────────────────────────────────────────

def get_user(username: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None


def all_users():
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM users ORDER BY role DESC, display_name"
        ).fetchall()
        return {r["username"]: dict(r) for r in rows}


# ── Preferences ───────────────────────────────────────────────────────────────

def set_theme(username: str, theme: str):
    if theme not in VALID_THEMES:
        return
    with _conn() as c:
        c.execute("UPDATE users SET theme=? WHERE username=?", (theme, username))
        c.commit()


def set_model_pref(username: str, provider: str, model: str):
    with _conn() as c:
        c.execute(
            "UPDATE users SET preferred_provider=?, preferred_model=? WHERE username=?",
            (provider or "", model or "", username),
        )
        c.commit()
