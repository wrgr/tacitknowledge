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
        c.execute("""
            CREATE TABLE IF NOT EXISTS llm_eval_cache (
                key        TEXT PRIMARY KEY,
                response   TEXT NOT NULL,
                created_at TEXT NOT NULL
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


def set_password(username: str, new_password: str) -> bool:
    """Replace a user's password hash. Returns True if a row was updated."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (generate_password_hash(new_password, method=_HASH_METHOD), username),
        )
        c.commit()
        return cur.rowcount > 0


def count_admins() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]


def update_user(old_username: str, new_username: str,
                display_name: str, role: str):
    """Update a user's username (PK), display name, and role.

    Returns (True, None) on success or (False, error_message) on failure.
    """
    if role not in ("admin", "student"):
        return False, "Invalid role."
    with _conn() as c:
        existing = c.execute(
            "SELECT role FROM users WHERE username=?", (old_username,)
        ).fetchone()
        if not existing:
            return False, "User not found."
        # Block removing the final admin (demotion or rename both count).
        if existing["role"] == "admin" and role != "admin":
            others = c.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND username!=?",
                (old_username,),
            ).fetchone()[0]
            if others == 0:
                return False, "Cannot demote the last remaining admin."
        if new_username != old_username:
            taken = c.execute(
                "SELECT 1 FROM users WHERE username=?", (new_username,)
            ).fetchone()
            if taken:
                return False, "That username is already taken."
        try:
            c.execute(
                "UPDATE users SET username=?, display_name=?, role=? WHERE username=?",
                (new_username, display_name, role, old_username),
            )
        except sqlite3.IntegrityError:
            return False, "Could not update user (constraint violation)."
        c.commit()
        return True, None


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


# ── LLM evaluative-call cache ──────────────────────────────────────────────────
# Determinism/testing aid only (see llm.cached_evaluative_call) -- not a
# cost-saving cache. Identical (model, base_url, prompt_version, prompt) input
# always returns the same stored response, so repeated test runs stay reproducible.

def eval_cache_get(key: str):
    init_db()  # no-op if already migrated; lets a fresh DB pick up the table
    with _conn() as c:
        row = c.execute("SELECT response FROM llm_eval_cache WHERE key=?", (key,)).fetchone()
        return row["response"] if row else None


def eval_cache_set(key: str, response: str):
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO llm_eval_cache (key, response, created_at) VALUES (?, ?, datetime('now'))",
            (key, response),
        )
        c.commit()
