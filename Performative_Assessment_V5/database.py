"""
SQLite database layer — users, themes, and model preferences.

Students: emma/liam/sofia/james/priya/tyler — all password Learn@2024
Admin:    admin — password admin123
"""

import json
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
        # Part C of the construct/exemplar brief: every accepted novel-equivalent FR match
        # is logged here for admin review -- promotion into a key point's exemplars list is
        # always an explicit human action, never automatic. Scoring never waits on this.
        c.execute("""
            CREATE TABLE IF NOT EXISTS novel_equivalent_review (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id          TEXT NOT NULL,
                key_point_id       TEXT NOT NULL,
                construct          TEXT NOT NULL,
                submission_excerpt TEXT NOT NULL,
                evidence_spans     TEXT NOT NULL,
                justification      TEXT NOT NULL,
                status             TEXT NOT NULL DEFAULT 'pending'
                                   CHECK(status IN ('pending','promoted','dismissed')),
                created_at         TEXT NOT NULL
            )
        """)
        # fr_hardening brief, Part D: every accepted FR match (exemplar or novel_equivalent)
        # is logged here so the novel-equivalent rate per key point has a real denominator --
        # novel_equivalent_review alone only ever contains novel-equivalent matches, so it
        # can't answer "novel-equivalent out of how many total matches" on its own.
        c.execute("""
            CREATE TABLE IF NOT EXISTS fr_match_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id    TEXT NOT NULL,
                key_point_id TEXT NOT NULL,
                construct    TEXT NOT NULL,
                match_type   TEXT NOT NULL CHECK(match_type IN ('exemplar','novel_equivalent')),
                created_at   TEXT NOT NULL
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


# ── Novel-equivalent review queue (FR construct/exemplar matching, Part C) ─────────────

def log_novel_equivalent(prompt_id: str, key_point_id: str, construct: str,
                         submission_excerpt: str, evidence_spans: list, justification: str):
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO novel_equivalent_review "
            "(prompt_id, key_point_id, construct, submission_excerpt, evidence_spans, "
            "justification, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', datetime('now'))",
            (prompt_id, key_point_id, construct, submission_excerpt,
             json.dumps(list(evidence_spans or [])), justification or ""),
        )
        c.commit()


def _row_to_review(row):
    d = dict(row)
    try:
        d["evidence_spans"] = json.loads(d["evidence_spans"])
    except (TypeError, ValueError):
        d["evidence_spans"] = []
    return d


def list_novel_equivalent_reviews(status: str = "pending"):
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM novel_equivalent_review WHERE status=? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        return [_row_to_review(r) for r in rows]


def get_novel_equivalent_review(review_id: int):
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM novel_equivalent_review WHERE id=?", (review_id,)
        ).fetchone()
        return _row_to_review(row) if row else None


def set_novel_equivalent_status(review_id: int, status: str) -> bool:
    if status not in ("pending", "promoted", "dismissed"):
        return False
    init_db()
    with _conn() as c:
        cur = c.execute(
            "UPDATE novel_equivalent_review SET status=? WHERE id=?", (status, review_id)
        )
        c.commit()
        return cur.rowcount > 0


# ── FR match log / novel-equivalent reliability metric (fr_hardening brief, Part D) ────

def log_fr_match(prompt_id: str, key_point_id: str, construct: str, match_type: str):
    """Record one accepted FR match (of either type) -- the denominator for the
    novel-equivalent rate. Purely additive bookkeeping; never read at grading time.
    """
    if match_type not in ("exemplar", "novel_equivalent"):
        return
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO fr_match_log (prompt_id, key_point_id, construct, match_type, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (prompt_id, key_point_id, construct, match_type),
        )
        c.commit()


def get_fr_match_stats():
    """Per-key-point reliability metric: total matches, novel-equivalent count/rate, and
    promote/dismiss counts among reviewed novel-equivalent entries -- all-time, across
    every prompt. A high rate for a key point is a signal to expand that point's
    exemplars, not evidence the grader is behaving unreliably.
    """
    init_db()
    with _conn() as c:
        totals = c.execute(
            "SELECT prompt_id, key_point_id, construct, "
            "  COUNT(*) AS total_matches, "
            "  SUM(CASE WHEN match_type='novel_equivalent' THEN 1 ELSE 0 END) AS novel_count "
            "FROM fr_match_log GROUP BY prompt_id, key_point_id, construct"
        ).fetchall()
        reviews = c.execute(
            "SELECT prompt_id, key_point_id, "
            "  SUM(CASE WHEN status='promoted' THEN 1 ELSE 0 END) AS promoted, "
            "  SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) AS dismissed, "
            "  SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending "
            "FROM novel_equivalent_review GROUP BY prompt_id, key_point_id"
        ).fetchall()
        review_by_kp = {(r["prompt_id"], r["key_point_id"]): dict(r) for r in reviews}

        stats = []
        for row in totals:
            key = (row["prompt_id"], row["key_point_id"])
            review = review_by_kp.pop(key, None)
            total = row["total_matches"] or 0
            novel = row["novel_count"] or 0
            stats.append({
                "prompt_id":        row["prompt_id"],
                "key_point_id":     row["key_point_id"],
                "construct":        row["construct"],
                "total_matches":    total,
                "novel_count":      novel,
                "novel_rate":       (novel / total) if total else 0.0,
                "promoted":         (review or {}).get("promoted", 0) or 0,
                "dismissed":        (review or {}).get("dismissed", 0) or 0,
                "pending_review":   (review or {}).get("pending", 0) or 0,
            })
        # key points with reviewed novel-equivalents but no match-log rows (pre-existing
        # data from before this table was added) -- surface them with total=novel so the
        # rate still reads as 100% rather than silently disappearing from the view.
        for key, review in review_by_kp.items():
            promoted, dismissed, pending = review.get("promoted", 0) or 0, review.get("dismissed", 0) or 0, review.get("pending", 0) or 0
            novel = promoted + dismissed + pending
            stats.append({
                "prompt_id":        key[0],
                "key_point_id":     key[1],
                "construct":        "",
                "total_matches":    novel,
                "novel_count":      novel,
                "novel_rate":       1.0 if novel else 0.0,
                "promoted":         promoted,
                "dismissed":        dismissed,
                "pending_review":   pending,
            })
        stats.sort(key=lambda s: s["novel_rate"], reverse=True)
        return stats
