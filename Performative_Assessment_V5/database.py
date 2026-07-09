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

# Canonical column list for the assessments results table. Mirrors the research
# export data dictionary (docs/research_export_data_dictionary.md) -- app.py's
# _RESEARCH_EXPORT_FIELDS is defined from this so the two can never drift.
ASSESSMENT_FIELDS = [
    "username",
    "display_name",
    "role",
    "report_file",
    "report_type",
    "task_title",
    "timestamp",
    "export_schema_version",
    "product_score_percent",
    "text_only_baseline_percent",
    "coverage_score_percent",
    "quality_score_percent",
    "matched_points",
    "missed_points",
    "strengths",
    "gaps",
    "word_count",
    "has_process_overlay",
    "process_quadrant",
    "effort_profile",
    "revision_toward_quality",
    "difficulty_point_count",
    "authenticity",
    "confidence_calibration",
    "closing_nudge_used",
    "process_caution",
    "process_review_priority",
    "process_review_reason",
    "thinking_solo",
    "ai_assistance_used",
    "ai_assistance_notes",
    "annotation_label",
    "annotation_notes",
    "annotation_reviewer",
    "annotation_updated_at",
]

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
                preferred_model    TEXT NOT NULL DEFAULT '',
                email              TEXT NOT NULL DEFAULT '',
                google_sub         TEXT NOT NULL DEFAULT '',
                auth_provider      TEXT NOT NULL DEFAULT 'password'
            )
        """)
        # CREATE TABLE IF NOT EXISTS doesn't alter existing databases, so add
        # the Google-OAuth columns to older DBs in place.
        existing = {r["name"] for r in c.execute("PRAGMA table_info(users)").fetchall()}
        for col, ddl in (("email",         "TEXT NOT NULL DEFAULT ''"),
                         ("google_sub",    "TEXT NOT NULL DEFAULT ''"),
                         ("auth_provider", "TEXT NOT NULL DEFAULT 'password'")):
            if col not in existing:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub "
                  "ON users(google_sub) WHERE google_sub != ''")
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
        # Structured assessment results -- one row per assessed task (FR reports
        # produce one row; scenario reports one row per scenario). Columns mirror
        # the research-export data dictionary exactly; rows are written at report
        # generation (and backfilled from existing report files at startup), so
        # the export and analytics can query this instead of re-parsing markdown.
        c.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols},
                UNIQUE(username, report_file, task_title)
            )
        """.format(cols=", ".join(f"{f} TEXT NOT NULL DEFAULT ''"
                                  for f in ASSESSMENT_FIELDS)))
        c.execute("CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments(username)")

        # Keep existing databases in sync with ASSESSMENT_FIELDS. CREATE TABLE IF NOT
        # EXISTS only helps fresh DBs; assessmentRework widens this table as new
        # report-facing evidence becomes exportable.
        existing_assessment_cols = {
            row["name"] for row in c.execute("PRAGMA table_info(assessments)").fetchall()
        }
        for field in ASSESSMENT_FIELDS:
            if field not in existing_assessment_cols:
                c.execute(f"ALTER TABLE assessments ADD COLUMN {field} TEXT NOT NULL DEFAULT ''")

        # Pooled key points (choose_n_of_m brief, Part B4): a novel-equivalent match on a
        # pool member needs to know which pool it belongs to, so the review UI can offer
        # "add as new exemplar" (existing behavior) vs. "add as a new pool member" (a
        # genuinely different technique the author didn't anticipate at all). No formal
        # migration framework here -- a guarded ADD COLUMN is the established idiom for
        # widening an existing table without disturbing existing rows.
        try:
            c.execute("ALTER TABLE novel_equivalent_review ADD COLUMN pool_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

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
            # Keep persisted assessment rows pointing at the renamed account
            # (app.py moves the reports/<username>/ folder to match).
            c.execute(
                "UPDATE assessments SET username=?, display_name=?, role=? WHERE username=?",
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


# Keep the determinism cache bounded — it grows one row per unique evaluative
# prompt and is never a cost-saving layer, so evicting the oldest rows only
# costs a re-run of the LLM call for very old inputs.
_EVAL_CACHE_MAX_ROWS = 2000


def eval_cache_set(key: str, response: str):
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO llm_eval_cache (key, response, created_at) VALUES (?, ?, datetime('now'))",
            (key, response),
        )
        excess = c.execute("SELECT COUNT(*) FROM llm_eval_cache").fetchone()[0] - _EVAL_CACHE_MAX_ROWS
        if excess > 0:
            c.execute(
                "DELETE FROM llm_eval_cache WHERE key IN ("
                "  SELECT key FROM llm_eval_cache ORDER BY created_at ASC, key ASC LIMIT ?)",
                (excess,),
            )
        c.commit()


# ── Novel-equivalent review queue (FR construct/exemplar matching, Part C) ─────────────

def log_novel_equivalent(prompt_id: str, key_point_id: str, construct: str,
                         submission_excerpt: str, evidence_spans: list, justification: str,
                         pool_id: str = None):
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO novel_equivalent_review "
            "(prompt_id, key_point_id, construct, submission_excerpt, evidence_spans, "
            "justification, pool_id, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', datetime('now'))",
            (prompt_id, key_point_id, construct, submission_excerpt,
             json.dumps(list(evidence_spans or [])), justification or "", pool_id),
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


# ── Assessment results (structured write-through of report data) ─────────────

def upsert_assessment_rows(rows):
    """Insert or replace assessment rows (dicts keyed by ASSESSMENT_FIELDS).

    Idempotent on (username, report_file, task_title), so re-generating a report
    or re-running the startup backfill never duplicates rows.
    """
    if not rows:
        return
    cols         = ", ".join(ASSESSMENT_FIELDS)
    placeholders = ", ".join("?" for _ in ASSESSMENT_FIELDS)
    with _conn() as c:
        c.executemany(
            f"INSERT OR REPLACE INTO assessments ({cols}) VALUES ({placeholders})",
            [tuple(str(r.get(f, "") or "") for f in ASSESSMENT_FIELDS) for r in rows],
        )
        c.commit()


def assessment_report_files():
    """Set of (username, report_file) pairs already persisted — used by the
    startup backfill to skip reports that are already in the table."""
    with _conn() as c:
        rows = c.execute("SELECT DISTINCT username, report_file FROM assessments").fetchall()
        return {(r["username"], r["report_file"]) for r in rows}


def assessment_report_export_versions():
    """Map each persisted report file to the export schema versions present in its rows."""
    with _conn() as c:
        rows = c.execute(
            "SELECT username, report_file, export_schema_version FROM assessments"
        ).fetchall()
        versions = {}
        for r in rows:
            key = (r["username"], r["report_file"])
            versions.setdefault(key, set()).add(r["export_schema_version"])
        return versions


def delete_assessment_rows(username: str, report_file: str):
    """Drop rows for a report file that no longer exists on disk."""
    with _conn() as c:
        c.execute("DELETE FROM assessments WHERE username=? AND report_file=?",
                  (username, report_file))
        c.commit()


def update_assessment_annotation(username: str, report_file: str,
                                 label: str, notes: str, reviewer: str, updated_at: str):
    """Mirror an instructor annotation onto the persisted rows for a report."""
    with _conn() as c:
        c.execute(
            "UPDATE assessments SET annotation_label=?, annotation_notes=?, "
            "annotation_reviewer=?, annotation_updated_at=? "
            "WHERE username=? AND report_file=?",
            (label, notes, reviewer, updated_at, username, report_file),
        )
        c.commit()


def all_assessment_rows():
    """Every persisted assessment row, ordered for the research export.

    Selects only ASSESSMENT_FIELDS explicitly (not SELECT *) so a database that
    still has a stale column from a prior schema (ALTER TABLE only adds columns,
    never drops them) doesn't leak an unexpected key into the row dicts.
    """
    cols = ", ".join(ASSESSMENT_FIELDS)
    with _conn() as c:
        rows = c.execute(
            f"SELECT {cols} FROM assessments "
            "ORDER BY username, timestamp, report_file, task_title"
        ).fetchall()
        return [dict(r) for r in rows]


def process_review_queue(limit: int = 10):
    """FR rows whose product/process pattern deserves human review first.

    The priority is an advisory triage signal computed at export time. It never
    changes the learner score; it just keeps likely divergence cases visible to
    instructors and researchers.
    """
    init_db()
    limit = max(1, min(int(limit or 10), 100))
    with _conn() as c:
        rows = c.execute(
            "SELECT username, report_file, task_title, timestamp, product_score_percent, "
            "       process_quadrant, process_review_priority, process_review_reason, "
            "       annotation_label "
            "FROM assessments "
            "WHERE report_type='free_response' "
            "  AND process_review_priority IN ('high','medium') "
            "ORDER BY CASE process_review_priority WHEN 'high' THEN 0 ELSE 1 END, "
            "         timestamp DESC, username ASC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def assessment_calibration_stats():
    """Aggregate instructor-annotation labels against LLM product scores.

    The annotation labels record an instructor's verdict on the LLM's grading
    of a report ('correct'/'partial'/'missing'/'needs_expert_review'), so the
    share labelled 'correct' is the LLM-vs-instructor agreement rate, and the
    average LLM score per label is the miscalibration signal (a high average
    score on 'missing'-labelled reports means the LLM over-credits).
    """
    labels = ("correct", "partial", "missing", "needs_expert_review")
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]

        label_rows = c.execute(
            "SELECT annotation_label AS label, COUNT(*) AS n, "
            "       AVG(CAST(NULLIF(product_score_percent,'') AS REAL)) AS avg_score "
            "FROM assessments WHERE annotation_label != '' GROUP BY annotation_label"
        ).fetchall()
        by_label   = {r["label"]: r["n"] for r in label_rows}
        avg_scores = {r["label"]: (round(r["avg_score"], 1) if r["avg_score"] is not None else None)
                      for r in label_rows}
        annotated  = sum(by_label.values())

        task_rows = c.execute(
            "SELECT task_title, report_type, COUNT(*) AS total, "
            "  SUM(annotation_label != '') AS annotated, "
            "  SUM(annotation_label = 'correct') AS correct, "
            "  SUM(annotation_label = 'partial') AS partial, "
            "  SUM(annotation_label = 'missing') AS missing, "
            "  SUM(annotation_label = 'needs_expert_review') AS needs_expert_review, "
            "  AVG(CAST(NULLIF(product_score_percent,'') AS REAL)) AS avg_score "
            "FROM assessments GROUP BY task_title, report_type"
        ).fetchall()
        by_task = []
        for r in task_rows:
            t = dict(r)
            t["avg_score"] = round(t["avg_score"], 1) if t["avg_score"] is not None else None
            t["agreement_rate"] = (t["correct"] / t["annotated"]) if t["annotated"] else None
            by_task.append(t)
        # most-disagreeing tasks first; un-annotated tasks sink to the bottom
        by_task.sort(key=lambda t: (t["agreement_rate"] is None,
                                    t["agreement_rate"] if t["agreement_rate"] is not None else 0))

        recent = [dict(r) for r in c.execute(
            "SELECT username, report_file, task_title, annotation_label, "
            "       product_score_percent, annotation_reviewer, annotation_updated_at "
            "FROM assessments WHERE annotation_label != '' "
            "ORDER BY annotation_updated_at DESC LIMIT 10"
        ).fetchall()]

    return {
        "total":          total,
        "annotated":      annotated,
        "labels":         {l: by_label.get(l, 0) for l in labels},
        "avg_score_by_label": {l: avg_scores.get(l) for l in labels},
        "agreement_rate": (by_label.get("correct", 0) / annotated) if annotated else None,
        "by_task":        by_task,
        "recent":         recent,
    }


# ── Google OAuth users ─────────────────────────────────────────────────────────

def get_user_by_google_sub(sub: str):
    if not sub:
        return None
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE google_sub=?", (sub,)).fetchone()
        return dict(row) if row else None


def _unique_username(c, base: str) -> str:
    """Derive a stable, unused username from an email local part."""
    import re as _re
    base = _re.sub(r'[^a-z0-9_\-]', '', base.lower())[:56] or "user"
    if len(base) < 3:
        base = (base + "user")[:56]
    candidate, n = base, 1
    while c.execute("SELECT 1 FROM users WHERE username=?", (candidate,)).fetchone():
        n += 1
        candidate = f"{base}{n}"
    return candidate


def get_or_create_google_user(sub: str, email: str, display_name: str, role: str):
    """Look up a Google account by its stable `sub`; provision it on first login.

    The generated username never changes afterwards (it keys reports/<username>/
    and the assessments rows), even if the Google display name or email does —
    those are refreshed on each login.
    """
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE google_sub=?", (sub,)).fetchone()
        if row:
            c.execute("UPDATE users SET email=?, display_name=? WHERE google_sub=?",
                      (email, display_name or row["display_name"], sub))
            c.commit()
            return get_user_by_google_sub(sub)

        username = _unique_username(c, email.split("@")[0])
        c.execute(
            "INSERT INTO users (username, password_hash, role, display_name, theme, "
            "preferred_provider, preferred_model, email, google_sub, auth_provider) "
            "VALUES (?, '', ?, ?, 'light', '', '', ?, ?, 'google')",
            (username, role if role in ("admin", "student") else "student",
             display_name or username, email, sub),
        )
        c.commit()
        return get_user_by_google_sub(sub)
