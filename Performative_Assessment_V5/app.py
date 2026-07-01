"""
Performative Assessment — web interface with secure student/admin gateway.
Run with:  python app.py
Open:      http://localhost:5001

Accounts (seeded on first run):
  admin  / admin123
  emma liam sofia james priya tyler  / Learn@2024

Security notes:
  - All login inputs are sanitised (null bytes stripped, length capped, whitespace trimmed)
  - CSRF token required on every login form submission
  - Rate limiting: 10 failed attempts per 60 s per IP (general)
  - Admin lockout: 3 failed attempts against an admin account → 15-minute hard lockout
  - Session cookies: HttpOnly, SameSite=Strict
  - Assessment sessions are bound to the creating user; other users cannot access them
  - Reports are stored in reports/<username>/ so students cannot read each other's files
"""

import hmac
import csv
import io
import json
import re
import secrets
import time
import uuid
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template,
                   request, Response, session, url_for)

import auth
import config
import database as db
import engine
import loaders
import report_parser
from llm import LLMError, LLMRateLimitError

# ── Persistent secret key ──────────────────────────────────────────────────────
_key_file = Path(__file__).parent / ".secret_key"
if _key_file.exists():
    _SECRET = _key_file.read_text(encoding="utf-8").strip()
else:
    _SECRET = secrets.token_hex(32)
    _key_file.write_text(_SECRET, encoding="utf-8")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = _SECRET
app.config.update(
    SESSION_COOKIE_HTTPONLY    = True,
    SESSION_COOKIE_SAMESITE    = "Strict",
    SESSION_COOKIE_SECURE      = False,   # set True when serving over HTTPS
    PERMANENT_SESSION_LIFETIME = 3600,
    MAX_CONTENT_LENGTH         = 16 * 1024 * 1024,   # reject oversized bodies (FR process_log)
)

auth.seed_default_users()


# ── Prevent browser caching of HTML pages ─────────────────────────────────────
# This ensures back/forward navigation always re-requests the page, so the
# session check fires and unauthenticated users are redirected to login.
@app.after_request
def no_cache_html(response):
    if "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"]        = "no-cache"
    return response


# ── LLM error handling ─────────────────────────────────────────────────────────
# llm.py raises LLMError / LLMRateLimitError with a clear, user-facing message.
# Without these handlers an LLM failure (rate limit, bad/blocked key, provider
# outage) surfaces as an opaque HTTP 500; here we return JSON the frontend's
# api() helper turns into a readable message instead of "Server error".
@app.errorhandler(LLMError)
def _handle_llm_error(e):
    status = 429 if isinstance(e, LLMRateLimitError) else 502
    return jsonify({"error": str(e)}), status


@app.errorhandler(ConnectionError)
def _handle_connection_error(e):
    return jsonify({"error": str(e)}), 503

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
PROMPTS_DIR   = Path(__file__).parent / "prompts"
REPORTS_BASE  = Path(__file__).parent / config.REPORTS_DIR

# Max characters accepted for a free-response submission. Bounds both the stored
# text and the writing-process trace size (keep in sync with the fr-textarea
# maxlength in index.html).
MAX_SUBMISSION_CHARS = 8000

scenarios            = engine.load_scenarios(SCENARIOS_DIR)
prompts              = engine.load_prompts(PROMPTS_DIR)
configured_providers = engine.get_configured_providers(config.PROVIDERS)

# Active scenario assessment sessions  { uuid: { runner, session, scenario, user_id, profile } }
_state: dict = {}

# Active free-response sessions  { uuid: { prompt, evaluation, profile, user_id, model, ... } }
_fr_state: dict = {}

# These in-memory stores are never persisted, so they must be bounded or they
# grow for the life of the process (an FR entry can hold a multi-MB process_log).
_SESSION_TTL_SECS    = 6 * 3600   # drop assessment state this long after creation
_MAX_ACTIVE_SESSIONS = 200        # hard per-store cap; evict oldest beyond this


def _register_session(store, sid, entry):
    """Insert an assessment-session entry, then evict stale/excess ones so the
    in-memory stores can't grow without bound."""
    now = time.monotonic()
    entry["_ts"] = now
    store[sid] = entry
    for k in [k for k, v in store.items() if now - v.get("_ts", now) > _SESSION_TTL_SECS]:
        store.pop(k, None)
    if len(store) > _MAX_ACTIVE_SESSIONS:
        oldest = sorted(store, key=lambda k: store[k].get("_ts", 0))
        for k in oldest[:len(store) - _MAX_ACTIVE_SESSIONS]:
            store.pop(k, None)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_csrf():
    tok = secrets.token_hex(32)
    session["csrf_token"] = tok
    return tok


def _user_theme():
    return session.get("theme", "light")


def _coerce_fr_rating(value):
    """Validate a rate/re-rate confidence rating (1-10 int); None if missing or out of range."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 10 else None


def _post_login_redirect():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("index"))


def _get_state(data):
    st = _state.get(data.get("session_id"))
    if not st:
        return None, (jsonify({"error": "session not found"}), 404)
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return None, (jsonify({"error": "forbidden"}), 403)
    st["_ts"] = time.monotonic()   # keep an in-progress session from being evicted
    return st, None


def _timestamp_from_report_filename(filename):
    m = re.search(r'(\d{8})_(\d{6})', filename)
    if not m:
        return ""
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


def _score_percent(score_text):
    if score_text is None:
        return ""
    text = str(score_text).strip()
    if not text:
        return ""
    if text.endswith("%"):
        text = text[:-1]
    try:
        return str(round(float(text)))
    except ValueError:
        return ""


def _join_export_list(items):
    return "; ".join(str(item) for item in (items or []) if item)


def _word_count(text):
    return len(text.split()) if text and text.strip() else 0


_RESEARCH_EXPORT_FIELDS = [
    "username",
    "display_name",
    "role",
    "report_file",
    "report_type",
    "task_title",
    "timestamp",
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
    "thinking_honey_mumford",
    "thinking_solo",
]


def _research_rows_for_report(username, user, filename, report):
    base = {
        "username": username,
        "display_name": user.get("display_name", username),
        "role": user.get("role", ""),
        "report_file": filename,
        "timestamp": _timestamp_from_report_filename(filename),
    }
    profile = report.get("thinking_profile") or {}
    hm = profile.get("honey_mumford") or {}
    solo = profile.get("solo") or {}
    base["thinking_honey_mumford"] = hm.get("style", "")
    base["thinking_solo"] = solo.get("level", "")

    if report.get("type") == "fr":
        ev = report.get("evaluation") or {}
        overlay = report.get("process_overlay") or {}
        score = _score_percent(ev.get("score") or (report.get("metadata") or {}).get("score"))
        return [{
            **base,
            "report_type": "free_response",
            "task_title": (report.get("metadata") or {}).get("prompt", ""),
            "product_score_percent": score,
            "text_only_baseline_percent": score,
            "coverage_score_percent": "",
            "quality_score_percent": "",
            "matched_points": _join_export_list(ev.get("matched_points")),
            "missed_points": _join_export_list(ev.get("missed_points")),
            "strengths": _join_export_list(ev.get("strengths")),
            "gaps": _join_export_list(ev.get("gaps")),
            "word_count": _word_count(report.get("submission", "")),
            "has_process_overlay": "yes" if overlay else "no",
            "process_quadrant": overlay.get("quadrant_label", "") if overlay else "",
            "effort_profile": overlay.get("effort_profile_text", "") if overlay else "",
            "revision_toward_quality": overlay.get("revision_rating", "") if overlay else "",
            "difficulty_point_count": len(overlay.get("difficulty_points", [])) if overlay else "",
            "authenticity": overlay.get("authenticity_text", "") if overlay else "",
            "confidence_calibration": overlay.get("confidence_calibration_text", "") if overlay else "",
        }]

    rows = []
    for scenario in report.get("scenarios", []):
        score = _score_percent(scenario.get("score"))
        transcript = scenario.get("transcript") or " ".join(
            p for p in [scenario.get("recall_transcript"), scenario.get("probe_transcript")] if p
        )
        rows.append({
            **base,
            "report_type": "scenario",
            "task_title": scenario.get("title", ""),
            "product_score_percent": score,
            "text_only_baseline_percent": score,
            "coverage_score_percent": _score_percent(scenario.get("coverage_score")),
            "quality_score_percent": _score_percent(scenario.get("quality_score")),
            "matched_points": _join_export_list(scenario.get("matched_points")),
            "missed_points": _join_export_list(scenario.get("missed_points")),
            "strengths": _join_export_list(scenario.get("strengths")),
            "gaps": _join_export_list(scenario.get("gaps")),
            "word_count": _word_count(transcript),
            "has_process_overlay": "no",
            "process_quadrant": "",
            "effort_profile": "",
            "revision_toward_quality": "",
            "difficulty_point_count": "",
            "authenticity": "",
            "confidence_calibration": "",
        })
    return rows


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    ip = request.remote_addr or "0.0.0.0"

    if "user_id" in session:
        return _post_login_redirect()

    if auth.is_admin_locked(ip):
        return redirect(url_for("locked"))

    if request.method == "GET":
        return render_template("login.html", error=None,
                               csrf_token=_new_csrf(), username_val="")

    form_tok = request.form.get("csrf_token", "")
    sess_tok = session.pop("csrf_token", "")
    if not sess_tok or not hmac.compare_digest(
            form_tok.encode("utf-8"), sess_tok.encode("utf-8")):
        return render_template("login.html",
                               error="Invalid request. Please try again.",
                               csrf_token=_new_csrf(), username_val="")

    if not auth.check_limit(ip):
        return render_template("login.html",
                               error="Too many login attempts. Please wait before trying again.",
                               csrf_token=_new_csrf(), username_val="")

    username = auth.sanitize_str(request.form.get("username", ""), max_len=64)
    password = auth.sanitize_str(request.form.get("password", ""), max_len=128)

    if not username or not password:
        return render_template("login.html",
                               error="Username and password are required.",
                               csrf_token=_new_csrf(), username_val=username)

    all_users      = auth.load_users()
    attempted_role = all_users.get(username, {}).get("role", "")
    is_admin_attempt = (attempted_role == "admin")

    user = auth.authenticate(username, password)

    if user is None:
        just_locked = auth.record_failed(ip, is_admin_attempt)
        if just_locked:
            return redirect(url_for("locked"))
        return render_template("login.html",
                               error="Invalid username or password.",
                               csrf_token=_new_csrf(), username_val=username)

    is_admin = (user["role"] == "admin")
    auth.record_success(ip, is_admin)
    session.clear()
    session["user_id"]             = username
    session["role"]                = user["role"]
    session["display_name"]        = user.get("display_name", username)
    session["theme"]               = user.get("theme", "light")
    session["preferred_provider"]  = user.get("preferred_provider", "")
    session["preferred_model"]     = user.get("preferred_model", "")

    return _post_login_redirect()


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/locked")
def locked():
    ip = request.remote_addr or "0.0.0.0"
    if not auth.is_admin_locked(ip) and "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("locked.html")


# ── Main assessment page ───────────────────────────────────────────────────────

@app.route("/")
@auth.login_required
def index():
    return render_template(
        "index.html",
        current_user=session.get("display_name", session.get("user_id", "")),
        is_admin=(session.get("role") == "admin"),
        user_theme=_user_theme(),
    )


# ── Admin dashboard ────────────────────────────────────────────────────────────

@app.route("/admin")
@auth.admin_required
def admin_dashboard():
    all_users = auth.load_users()
    user_list = [
        {
            "username":     uname,
            "display_name": udata.get("display_name", uname),
            "role":         udata["role"],
        }
        for uname, udata in all_users.items()
    ]

    report_tree = {}
    for u in user_list:
        udir = REPORTS_BASE / u["username"]
        if udir.is_dir():
            report_tree[u["username"]] = sorted(
                [f.name for f in udir.glob("*.md")], reverse=True
            )
        else:
            report_tree[u["username"]] = []

    return render_template(
        "admin.html",
        admin_name=session.get("display_name", "Admin"),
        users=user_list,
        report_tree=report_tree,
        user_theme=_user_theme(),
    )


@app.route("/admin/report/<username>/<filename>")
@auth.admin_required
def admin_view_report(username, filename):
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
        abort(400)
    if not re.match(r'^(?:fr_)?report_[\d_]+\.md$', filename):
        abort(400)

    report_path = REPORTS_BASE / username / filename
    if report_path.resolve().parent != (REPORTS_BASE / username).resolve():
        abort(400)
    if not report_path.exists():
        abort(404)

    content = report_path.read_text(encoding="utf-8")
    report = report_parser.parse_report_md(content)
    return render_template(
        "report_view.html",
        username=username,
        filename=filename,
        report=report,
        admin_name=session.get("display_name", "Admin"),
        user_theme=_user_theme(),
    )


@app.route("/admin/research-export.csv")
@auth.admin_required
def admin_research_export():
    all_users = auth.load_users()
    rows = []

    for username, user in all_users.items():
        user_dir = REPORTS_BASE / username
        if not user_dir.is_dir():
            continue

        for report_file in sorted(user_dir.glob("*.md")):
            try:
                content = report_file.read_text(encoding="utf-8")
                report = report_parser.parse_report_md(content)
            except Exception:
                continue
            rows.extend(_research_rows_for_report(username, user, report_file.name, report))

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=_RESEARCH_EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)

    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=research_export.csv",
            "Cache-Control": "no-store",
        },
    )


# ── Student report endpoints ───────────────────────────────────────────────────

@app.route("/my-reports")
@auth.login_required
def my_reports_page():
    """Dedicated page listing the logged-in student's own reports."""
    user_dir = REPORTS_BASE / session["user_id"]
    files = []
    if user_dir.is_dir():
        files = sorted([f.name for f in user_dir.glob("*.md")], reverse=True)
    return render_template(
        "my_reports.html",
        reports=files,
        current_user=session.get("display_name", session.get("user_id", "")),
        user_theme=_user_theme(),
    )


@app.route("/api/my-reports", methods=["POST"])
@auth.login_required
def api_my_reports():
    """List the calling user's own reports — students cannot access others'."""
    user_dir = REPORTS_BASE / session["user_id"]
    files = []
    if user_dir.is_dir():
        files = sorted([f.name for f in user_dir.glob("*.md")], reverse=True)
    return jsonify({"reports": files})


@app.route("/my-report/<filename>")
@auth.login_required
def student_view_report(filename):
    """Serve a report to the student who owns it. Strict path validation prevents traversal."""
    if not re.match(r'^(?:fr_)?report_[\d_]+\.md$', filename):
        abort(400)

    report_path = REPORTS_BASE / session["user_id"] / filename
    # Ensure the resolved path stays inside the user's own directory
    user_dir = (REPORTS_BASE / session["user_id"]).resolve()
    if report_path.resolve().parent != user_dir:
        abort(400)
    if not report_path.exists():
        abort(404)

    content = report_path.read_text(encoding="utf-8")
    report = report_parser.parse_report_md(content)
    return render_template(
        "student_report.html",
        filename=filename,
        report=report,
        current_user=session.get("display_name", session.get("user_id", "")),
        user_theme=_user_theme(),
    )


# ── Writing-process replay trace ────────────────────────────────────────────────

def _encode_process_log(process_log, cap_chars):
    """Delta-encode snapshots for compact on-disk storage.

    Snapshots are near-append-only full-text keyframes, so storing each in full
    is hugely redundant. We keep the first snapshot whole and store each later
    one as (common-prefix length `p`, changed suffix `s`) — reconstructed on
    read. Each snapshot's text is also capped at cap_chars as a safety bound.
    """
    pl = dict(process_log or {})
    snaps = pl.get("snapshots") or []
    deltas, prev = [], ""
    for i, s in enumerate(snaps):
        text = (s.get("text") or "")[:cap_chars]
        ts = s.get("timestamp_s", 0)
        if i == 0:
            deltas.append({"timestamp_s": ts, "text": text})
        else:
            p, m = 0, min(len(prev), len(text))
            while p < m and prev[p] == text[p]:
                p += 1
            deltas.append({"timestamp_s": ts, "p": p, "s": text[p:]})
        prev = text
    pl.pop("snapshots", None)
    pl["snapshots_delta"] = deltas
    return pl


def _decode_process_log(process_log):
    """Reconstruct full snapshots from a delta-encoded process_log.

    Passes through unchanged when there is no `snapshots_delta` key, so traces
    written before delta encoding (full-text snapshots) still serve correctly.
    """
    pl = dict(process_log or {})
    deltas = pl.pop("snapshots_delta", None)
    if deltas is None:
        return pl
    snaps, prev = [], ""
    for d in deltas:
        text = d["text"] if "text" in d else prev[:d.get("p", 0)] + d.get("s", "")
        snaps.append({"timestamp_s": d.get("timestamp_s", 0), "text": text})
        prev = text
    pl["snapshots"] = snaps
    return pl


def _serve_trace(username, filename):
    """Return the persisted writing-process trace JSON for a report, or 404.

    Same strict filename/path validation as the report routes; the trace lives
    beside its report as <report_stem>.trace.json.
    """
    if not re.match(r'^(?:fr_)?report_[\d_]+\.md$', filename):
        abort(400)
    user_dir   = (REPORTS_BASE / username).resolve()
    trace_path = REPORTS_BASE / username / (Path(filename).stem + ".trace.json")
    if trace_path.resolve().parent != user_dir:
        abort(400)
    if not trace_path.exists():
        return jsonify({"error": "no trace for this report"}), 404
    try:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
        raw["process_log"] = _decode_process_log(raw.get("process_log"))
        raw.pop("encoded", None)
        body = json.dumps(raw)
    except Exception:
        body = trace_path.read_text(encoding="utf-8")   # serve verbatim if anything is off
    return Response(body, mimetype="application/json",
                    headers={"Cache-Control": "no-store"})


@app.route("/my-report/<filename>/trace")
@auth.login_required
def student_report_trace(filename):
    return _serve_trace(session["user_id"], filename)


@app.route("/admin/report/<username>/<filename>/trace")
@auth.admin_required
def admin_report_trace(username, filename):
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
        abort(400)
    return _serve_trace(username, filename)


# ── Preference API endpoints ───────────────────────────────────────────────────

@app.route("/api/save-theme", methods=["POST"])
@auth.login_required
def api_save_theme():
    data  = request.get_json(silent=True) or {}
    theme = (data or {}).get("theme", "light")
    db.set_theme(session["user_id"], theme)
    session["theme"] = theme
    return jsonify({"ok": True})


@app.route("/api/save-model-pref", methods=["POST"])
@auth.login_required
def api_save_model_pref():
    data     = request.get_json(silent=True) or {}
    provider = data.get("provider", "")
    model    = data.get("model", "")
    db.set_model_pref(session["user_id"], provider, model)
    session["preferred_provider"] = provider
    session["preferred_model"]    = model
    return jsonify({"ok": True})


_USERNAME_RE = re.compile(r'^[a-z0-9_\-]{3,64}$')


@app.route("/admin/user/<username>", methods=["GET"])
@auth.admin_required
def admin_edit_user(username):
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
        abort(400)
    user = db.get_user(username)
    if not user:
        abort(404)
    return render_template(
        "admin_user_edit.html",
        admin_name=session.get("display_name", "Admin"),
        user=user,
        target=username,
        is_self=(username == session.get("user_id")),
        csrf_token=_new_csrf(),
        error=None,
        user_theme=_user_theme(),
    )


@app.route("/admin/user/<username>", methods=["POST"])
@auth.admin_required
def admin_update_user(username):
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
        abort(400)
    user = db.get_user(username)
    if not user:
        abort(404)

    form_tok = request.form.get("csrf_token", "")
    sess_tok = session.pop("csrf_token", "")
    if not form_tok or not hmac.compare_digest(form_tok, sess_tok):
        abort(400)

    new_username = auth.sanitize_str(request.form.get("username", ""), 64).lower()
    display_name = auth.sanitize_str(request.form.get("display_name", ""), 128)
    role         = auth.sanitize_str(request.form.get("role", ""), 16)
    new_password = request.form.get("new_password", "")

    def _re_render(error):
        merged = dict(user)
        merged.update(username=new_username or username,
                      display_name=display_name or user["display_name"],
                      role=role or user["role"])
        return render_template(
            "admin_user_edit.html",
            admin_name=session.get("display_name", "Admin"),
            user=merged, target=username, is_self=(username == session.get("user_id")),
            csrf_token=_new_csrf(), error=error, user_theme=_user_theme(),
        )

    if not _USERNAME_RE.match(new_username):
        return _re_render("Username must be 3–64 chars: lowercase letters, digits, '-' or '_'.")
    if not display_name:
        return _re_render("Display name cannot be empty.")
    if role not in ("admin", "student"):
        return _re_render("Role must be 'admin' or 'student'.")
    if new_password and len(new_password) < 8:
        return _re_render("Password must be at least 8 characters.")
    if len(new_password) > 256:
        return _re_render("Password is too long.")

    ok, err = db.update_user(username, new_username, display_name, role)
    if not ok:
        return _re_render(err)

    # Move the user's report directory if the username changed.
    if new_username != username:
        base    = REPORTS_BASE.resolve()
        old_dir = (REPORTS_BASE / username).resolve()
        new_dir = (REPORTS_BASE / new_username).resolve()
        # Defence-in-depth: ensure both paths stay inside REPORTS_BASE even
        # though the username regexes already forbid separators and dots.
        if old_dir.parent == base and new_dir.parent == base:
            if old_dir.is_dir() and not new_dir.exists():
                old_dir.rename(new_dir)

    if new_password:
        db.set_password(new_username, new_password)

    # Keep the live session coherent if an admin edited their own account.
    if username == session.get("user_id"):
        session["user_id"]      = new_username
        session["role"]         = role
        session["display_name"] = display_name

    return redirect(url_for("admin_dashboard"))


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/scenarios", methods=["POST"])
@auth.login_required
def api_scenarios():
    scenario_list = [
        {
            "index":       i,
            "id":          s["id"],
            "title":       s["title"],
            "description": s["description"],
            "has_probe_bank": bool(s.get("probe_bank")),
        }
        for i, s in enumerate(scenarios)
    ]

    all_providers = [
        {
            "name":          name,
            "model":         cfg["model"],
            "is_configured": engine.llm_is_available(cfg["api_key"]),
            "needs_key":     name != "Ollama",
        }
        for name, cfg in config.PROVIDERS.items()
    ]
    provider_names = [p["name"] for p in all_providers]
    default = (
        config.DEFAULT_PROVIDER if config.DEFAULT_PROVIDER in provider_names
        else (provider_names[0] if provider_names else None)
    )

    return jsonify({
        "scenarios":                scenario_list,
        "providers":                all_providers,
        "default_provider":         default,
        "user_preferred_provider":  session.get("preferred_provider", ""),
        "user_preferred_model":     session.get("preferred_model", ""),
    })


@app.route("/api/validate-key", methods=["POST"])
@auth.login_required
def api_validate_key():
    import llm as llm_module
    data          = request.get_json(silent=True) or {}
    provider_name = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg  = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key       = (data.get("api_key") or "").strip()
    model         = data.get("model") or provider_cfg["model"]
    base_url      = provider_cfg["base_url"]

    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})

    ok, error = llm_module.validate_api_key(provider_name, api_key, model, base_url)
    return jsonify({"valid": ok, "error": error})


@app.route("/api/models", methods=["POST"])
@auth.login_required
def api_models():
    data          = request.get_json(silent=True) or {}
    provider_name = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg  = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    models        = engine.get_available_models(provider_name, provider_cfg)
    default       = provider_cfg.get("model", models[0] if models else "")
    return jsonify({"models": models, "default": default})


@app.route("/api/start", methods=["POST"])
@auth.login_required
def api_start():
    data = request.get_json(silent=True) or {}
    try:
        scenario_index = int(data.get("index"))
        scenario = scenarios[scenario_index]
    except (TypeError, ValueError, IndexError):
        return jsonify({"error": "invalid scenario index"}), 400

    sid = str(uuid.uuid4())

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]
    base_url         = provider_cfg["base_url"]
    model            = data.get("model") or provider_cfg["model"]
    use_llm          = engine.llm_is_available(api_key)

    runner  = engine.ScenarioRunner(scenario, model=model, api_key=api_key, base_url=base_url)
    opening = runner.start()

    _register_session(_state, sid, {
        "runner":   runner,
        "session":  engine.Session(use_llm, model=model, api_key=api_key, base_url=base_url),
        "scenario": scenario,
        "profile":  None,
        "user_id":  session["user_id"],
    })

    result = {
        "session_id": sid,
        "opening":    opening,
        "phase":      "recall",
    }
    if session.get("role") == "admin" and scenario.get("expert_answers"):
        result["debug_expert_answer"] = scenario["expert_answers"][0].get("answer", "")
    return jsonify(result)


@app.route("/api/end-recall", methods=["POST"])
@auth.login_required
def api_end_recall():
    """Called when the learner clicks 'I'm Done' in the recall phase.
    Triggers gap analysis, builds the probe queue, transitions to probing phase,
    and returns the first probe question."""
    data = request.get_json(silent=True) or {}
    st, err = _get_state(data)
    if err:
        return err

    runner = st["runner"]
    if runner.phase != "recall":
        return jsonify({"error": "not in recall phase"}), 400

    # submit any final text typed before clicking I'm Done
    final_text = (data.get("final_text") or "").strip()
    if final_text:
        runner.respond(final_text, writing_metrics=data.get("writing_metrics"))

    first_probe, concluded = runner.end_recall()

    return jsonify({
        "first_probe":   first_probe,
        "concluded":     concluded,
        "phase":         runner.phase,
        "probe_count":   runner.probe_count(),
        "probe_number":  runner.current_probe_number(),
    })


@app.route("/api/respond", methods=["POST"])
@auth.login_required
def api_respond():
    data = request.get_json(silent=True) or {}
    st, err = _get_state(data)
    if err:
        return err
    runner = st["runner"]
    narration, concluded = runner.respond(data["user_input"], writing_metrics=data.get("writing_metrics"))
    return jsonify({
        "narration":    narration,
        "concluded":    concluded,
        "phase":        runner.phase,
        "probe_number": runner.current_probe_number() if runner.phase == "probing" else None,
        "probe_count":  runner.probe_count() if runner.phase == "probing" else None,
    })


@app.route("/api/evaluate", methods=["POST"])
@auth.login_required
def api_evaluate():
    data = request.get_json(silent=True) or {}
    st, err = _get_state(data)
    if err:
        return err

    runner = st["runner"]
    sess   = st["session"]
    try:
        evaluations = sess.evaluate(
            runner.scenario,
            transcript=runner.transcript(),
            recall_transcript=runner.recall_transcript,
            probe_transcript=runner.probe_transcript,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    eval_list = [
        {
            "score":           ev["score"],
            "coverage_score":  ev.get("coverage_score", ev["score"]),
            "quality_score":   ev.get("quality_score", 0.0),
            "feedback":        ev["feedback"],
            "strengths":       ev["strengths"],
            "gaps":            ev["gaps"],
            "matched_points":  ev["matched_points"],
            "missed_points":   ev["missed_points"],
            "quality_ratings": ev.get("quality_ratings", {}),
            "point_sources":   ev.get("point_sources", {}),
            # Per-phase scores (present only when recall/probe phases were scored separately)
            "recall_score":           ev.get("recall_score"),
            "recall_coverage_score":  ev.get("recall_coverage_score"),
            "recall_quality_score":   ev.get("recall_quality_score"),
            "recall_matched_points":  ev.get("recall_matched_points"),
            "recall_missed_points":   ev.get("recall_missed_points"),
            "probe_score":            ev.get("probe_score"),
            "probe_coverage_score":   ev.get("probe_coverage_score"),
            "probe_quality_score":    ev.get("probe_quality_score"),
            "probe_matched_points":   ev.get("probe_matched_points"),
            "probe_missed_points":    ev.get("probe_missed_points"),
        }
        for ev in evaluations
    ]

    return jsonify({
        "evaluations": eval_list,
        "summary": {
            "total_evaluations": sess.total_evaluations(),
            "average_score":     sess.average_score(),
        },
    })


@app.route("/api/thinking-profile", methods=["POST"])
@auth.login_required
def api_thinking_profile():
    data = request.get_json(silent=True) or {}
    st, err = _get_state(data)
    if err:
        return err

    sess     = st["session"]
    runner   = st["runner"]
    scenario = st["scenario"]

    if not sess.use_llm:
        return jsonify({"profile": None})

    try:
        profile = engine.analyse_thinking_profile(
            scenario, runner.transcript(),
            model=sess.model, api_key=sess.api_key, base_url=sess.base_url,
            writing_metrics=runner.writing_metrics,
            user_inputs=runner.user_inputs,
            recall_transcript=runner.recall_transcript,
            probe_transcript=runner.probe_transcript,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    st["profile"] = profile
    return jsonify({"profile": profile})


@app.route("/api/report", methods=["POST"])
@auth.login_required
def api_report():
    data = request.get_json(silent=True) or {}
    st, err = _get_state(data)
    if err:
        return err

    sess             = st["session"]
    user_reports_dir = REPORTS_BASE / session["user_id"]

    # Generate the thinking profile now if the async frontend call hasn't finished yet.
    thinking_profile = st.get("profile")
    if thinking_profile is None and sess.use_llm:
        thinking_profile = engine.analyse_thinking_profile(
            st["scenario"], st["runner"].transcript(),
            model=sess.model, api_key=sess.api_key, base_url=sess.base_url,
            writing_metrics=st["runner"].writing_metrics,
            user_inputs=st["runner"].user_inputs,
            recall_transcript=st["runner"].recall_transcript,
            probe_transcript=st["runner"].probe_transcript,
        )
        st["profile"] = thinking_profile

    try:
        path = engine.generate_report(
            sess,
            model=sess.model,
            api_key=sess.api_key,
            base_url=sess.base_url,
            output_dir=user_reports_dir,
            thinking_profile=thinking_profile,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"path": str(path)})


@app.route("/api/generate-scenario", methods=["POST"])
@auth.login_required
def api_generate_scenario():
    data        = request.get_json(silent=True) or {}
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]

    if not engine.llm_is_available(api_key):
        return jsonify({"error": "An API key is required for AI generation. "
                                  "Enter one in the key field or configure it in config.py."}), 400

    model = data.get("model") or provider_cfg["model"]
    draft = engine.generate_scenario_draft(
        description, model=model, api_key=api_key, base_url=provider_cfg["base_url"]
    )
    return jsonify({"scenario": draft})


@app.route("/api/save-scenario", methods=["POST"])
@auth.admin_required
def api_save_scenario():
    global scenarios
    data = request.get_json(silent=True) or {}

    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    scenario_id   = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    # normalize scoring weights
    sw = data.get("scoring_weights", {})
    cw = float(sw.get("coverage", 0.6))
    qw = float(sw.get("quality",  0.4))
    total_w = cw + qw
    if total_w > 0 and abs(total_w - 1.0) > 0.01:
        cw, qw = round(cw / total_w, 4), round(qw / total_w, 4)

    scenario_data = {
        "id":              scenario_id,
        "title":           title,
        "description":     data.get("description", ""),
        "situation":       data.get("situation", ""),
        "user_role":       data.get("user_role", "participant"),
        "constraints":     data.get("constraints", []),
        "decision_points": data.get("decision_points", []),
        "failure_modes":   data.get("failure_modes", []),
        "edge_cases":      data.get("edge_cases", []),
        "probe_bank":      data.get("probe_bank", []),
        "scoring_weights": {"coverage": cw, "quality": qw},
        "expert_answers": [
            {
                "id":         "expert_001",
                "answer":     data.get("expert_answer", ""),
                "key_points": data.get("key_points", []),
                "rubric":     data.get("rubric", {}),
            }
        ],
    }

    path = SCENARIOS_DIR / (scenario_id + ".json")
    path.write_text(json.dumps(scenario_data, indent=2), encoding="utf-8")
    scenarios = engine.load_scenarios(SCENARIOS_DIR)
    return jsonify({"path": str(path)})


@app.route("/api/generate-prompt", methods=["POST"])
@auth.login_required
def api_generate_prompt():
    data        = request.get_json(silent=True) or {}
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]

    if not engine.llm_is_available(api_key):
        return jsonify({"error": "An API key is required for AI generation. "
                                  "Enter one in the key field or configure it in config.py."}), 400

    model = data.get("model") or provider_cfg["model"]
    draft = engine.generate_prompt_draft(
        description, model=model, api_key=api_key, base_url=provider_cfg["base_url"]
    )
    return jsonify({"prompt": draft})


@app.route("/api/save-prompt", methods=["POST"])
@auth.admin_required
def api_save_prompt():
    global prompts
    data = request.get_json(silent=True) or {}

    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    # key_points arrive as {construct, exemplars, importance} objects from the admin UI
    # (construct/exemplar brief, Part E) -- ids are assigned by loaders.py's migration on
    # the next load, same as for a hand-edited or legacy prompt file.
    key_points = []
    for kp in (data.get("key_points") or []):
        if not isinstance(kp, dict):
            continue
        construct = (kp.get("construct") or "").strip()
        if not construct:
            continue
        importance = kp.get("importance")
        if importance not in loaders.FR_IMPORTANCE_LEVELS:
            importance = "MEDIUM"
        exemplars = [e.strip() for e in (kp.get("exemplars") or []) if isinstance(e, str) and e.strip()]
        key_points.append({"construct": construct, "exemplars": exemplars, "importance": importance})

    prompt_id   = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    prompt_data = {
        "id":          prompt_id,
        "title":       title,
        "description": data.get("description", ""),
        "prompt_text": data.get("prompt_text", ""),
        "word_limit":  int(data["word_limit"]) if data.get("word_limit") else None,
        "constraints": data.get("constraints", []),
        "expert_answers": [
            {
                "id":         "expert_001",
                "answer":     data.get("expert_answer", ""),
                "key_points": key_points,
            }
        ],
        "metadata": {},
    }
    general_guidance = data.get("general_guidance", "").strip()
    if general_guidance:
        prompt_data["general_guidance"] = general_guidance

    path = PROMPTS_DIR / (prompt_id + ".json")
    path.write_text(json.dumps(prompt_data, indent=2), encoding="utf-8")
    prompts = engine.load_prompts(PROMPTS_DIR)
    return jsonify({"path": str(path)})


# ── Free-Response API ──────────────────────────────────────────────────────────

@app.route("/api/fr/prompts", methods=["POST"])
@auth.login_required
def api_fr_prompts():
    is_admin = session.get("role") == "admin"
    prompt_list = []
    for i, p in enumerate(prompts):
        entry = {
            "index":            i,
            "id":               p["id"],
            "title":            p["title"],
            "description":      p["description"],
            "word_limit":       p.get("word_limit"),
            "prompt_text":      p.get("prompt_text", ""),
            "constraints":      p.get("constraints", []),
            "general_guidance": p.get("general_guidance", ""),
        }
        if is_admin and p.get("expert_answers"):
            entry["expert_answer"] = p["expert_answers"][0].get("answer", "")
        prompt_list.append(entry)
    return jsonify({"prompts": prompt_list})


# ── Novel-equivalent review queue (Part C, construct/exemplar brief) ────────────

@app.route("/api/admin/novel-equivalents", methods=["POST"])
@auth.admin_required
def api_admin_novel_equivalents():
    title_by_id = {p["id"]: p["title"] for p in prompts}
    reviews = db.list_novel_equivalent_reviews("pending")
    for r in reviews:
        r["prompt_title"] = title_by_id.get(r["prompt_id"], r["prompt_id"])
    return jsonify({"reviews": reviews})


@app.route("/api/admin/novel-equivalents/promote", methods=["POST"])
@auth.admin_required
def api_admin_novel_equivalent_promote():
    global prompts
    data      = request.get_json(silent=True) or {}
    review_id = data.get("review_id")
    exemplar  = (data.get("exemplar") or "").strip()
    if not exemplar:
        return jsonify({"error": "exemplar text is required"}), 400

    review = db.get_novel_equivalent_review(review_id)
    if not review or review["status"] != "pending":
        return jsonify({"error": "review not found"}), 404

    prompt_data = next((p for p in prompts if p["id"] == review["prompt_id"]), None)
    if not prompt_data:
        return jsonify({"error": "prompt not found"}), 404

    updated = False
    for ea in prompt_data.get("expert_answers", []):
        for kp in ea.get("key_points", []):
            if kp["id"] == review["key_point_id"]:
                if exemplar not in kp["exemplars"]:
                    kp["exemplars"].append(exemplar)
                updated = True

    if not updated:
        return jsonify({"error": "key point no longer exists on this prompt"}), 404

    path = PROMPTS_DIR / (prompt_data["id"] + ".json")
    path.write_text(json.dumps(prompt_data, indent=2), encoding="utf-8")
    prompts = engine.load_prompts(PROMPTS_DIR)

    db.set_novel_equivalent_status(review_id, "promoted")
    return jsonify({"status": "promoted"})


@app.route("/api/admin/novel-equivalents/dismiss", methods=["POST"])
@auth.admin_required
def api_admin_novel_equivalent_dismiss():
    data      = request.get_json(silent=True) or {}
    review_id = data.get("review_id")
    review    = db.get_novel_equivalent_review(review_id)
    if not review or review["status"] != "pending":
        return jsonify({"error": "review not found"}), 404

    db.set_novel_equivalent_status(review_id, "dismissed")
    return jsonify({"status": "dismissed"})


@app.route("/api/fr/submit", methods=["POST"])
@auth.login_required
def api_fr_submit():
    global _fr_state
    data      = request.get_json(silent=True) or {}
    prompt_id = data.get("prompt_id", "")
    text      = (data.get("text") or "").strip()[:MAX_SUBMISSION_CHARS]

    if not text:
        return jsonify({"error": "submission text is required"}), 400

    prompt_data = next((p for p in prompts if p["id"] == prompt_id), None)
    if not prompt_data:
        return jsonify({"error": "prompt not found"}), 404

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]
    base_url         = provider_cfg["base_url"]
    model            = data.get("model") or provider_cfg["model"]
    use_llm          = engine.llm_is_available(api_key)

    if use_llm:
        evaluation = engine.score_free_response_with_llm(model, api_key, base_url, prompt_data, text)
    else:
        evaluation = engine.score_free_response_with_keywords(prompt_data, text)

    # Part C of the construct/exemplar brief: log every accepted novel-equivalent match for
    # admin review. The learner's score above is already final -- this queue is for
    # improving the rubric going forward, not for gating this submission.
    for m in evaluation.get("novel_equivalent_matches", []):
        db.log_novel_equivalent(
            prompt_id=prompt_id,
            key_point_id=m["key_point_id"],
            construct=m["construct"],
            submission_excerpt=text[:2000],
            evidence_spans=m["evidence_spans"],
            justification=m.get("functional_justification") or "",
        )

    sid = str(uuid.uuid4())
    _register_session(_fr_state, sid, {
        "prompt":           prompt_data,
        "evaluation":       evaluation,
        "profile":          None,
        "process_overlay":  None,
        "writing_metrics":  data.get("writing_metrics"),
        "pre_rating":       _coerce_fr_rating(data.get("pre_rating")),
        "post_rating":      None,
        "user_id":          session["user_id"],
        "model":            model,
        "api_key":          api_key,
        "base_url":         base_url,
    })

    return jsonify({
        "session_id": sid,
        "evaluation": {
            "score":          evaluation["score"],
            "feedback":       evaluation["feedback"],
            "strengths":      evaluation["strengths"],
            "gaps":           evaluation["gaps"],
            "matched_points": evaluation["matched_points"],
            "missed_points":  evaluation["missed_points"],
        },
    })


@app.route("/api/fr/post-rating", methods=["POST"])
@auth.login_required
def api_fr_post_rating():
    """Store the post-write confidence rating — asked immediately after submission but
    before any score or feedback is shown to the learner (Part C of the rate/re-rate
    mechanic; see also Part D, feedback-before-score)."""
    data = request.get_json(silent=True) or {}
    sid  = data.get("session_id")
    st   = _fr_state.get(sid)

    if not st:
        return jsonify({"error": "session not found"}), 404
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    rating = _coerce_fr_rating(data.get("post_rating"))
    if rating is None:
        return jsonify({"error": "post_rating must be an integer 1-10"}), 400

    st["post_rating"] = rating
    return jsonify({"ok": True})


@app.route("/api/fr/thinking-profile", methods=["POST"])
@auth.login_required
def api_fr_thinking_profile():
    data = request.get_json(silent=True) or {}
    sid  = data.get("session_id")
    st   = _fr_state.get(sid)

    if not st:
        return jsonify({"error": "session not found"}), 404
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    if not engine.llm_is_available(st["api_key"]):
        return jsonify({"profile": None})

    try:
        profile = engine.analyse_thinking_profile(
            st["prompt"], st["evaluation"]["text"],
            model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
            writing_metrics=[st.get("writing_metrics")] if st.get("writing_metrics") else None,
            user_inputs=[st["evaluation"]["text"]],
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    st["profile"] = profile
    return jsonify({"profile": profile})


@app.route("/api/fr/report", methods=["POST"])
@auth.login_required
def api_fr_report():
    data = request.get_json(silent=True) or {}
    sid  = data.get("session_id")
    st   = _fr_state.get(sid)

    if not st:
        return jsonify({"error": "session not found"}), 404
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    user_reports_dir = REPORTS_BASE / session["user_id"]

    # Generate the thinking profile now if the async frontend call hasn't finished yet.
    thinking_profile = st.get("profile")
    if thinking_profile is None and engine.llm_is_available(st["api_key"]):
        thinking_profile = engine.analyse_thinking_profile(
            st["prompt"], st["evaluation"]["text"],
            model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
            writing_metrics=[st.get("writing_metrics")] if st.get("writing_metrics") else None,
            user_inputs=[st["evaluation"]["text"]],
        )
        st["profile"] = thinking_profile

    # Writing-process overlay: interpretive only, never blended into the product score.
    # Skipped entirely when the prompt disables it or no process_log was captured
    # (e.g. CLI submissions have no WritingTracker) — FR then scores product-only.
    process_overlay = st.get("process_overlay")
    process_log = (st.get("writing_metrics") or {}).get("process_log")
    if (process_overlay is None and st["prompt"].get("process_overlay_enabled", True)
            and process_log):
        process_overlay = engine.analyze_writing_process(
            process_log, st.get("writing_metrics"), st["evaluation"]["text"],
            product_score=st["evaluation"]["score"],
            model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
            use_llm=engine.llm_is_available(st["api_key"]),
        )

    # Confidence calibration (rate → explain → re-rate) is independent of process_log —
    # it can be present even when no writing process was captured at all, and vice versa.
    confidence_calibration = engine.compute_confidence_calibration(
        st.get("pre_rating"), st.get("post_rating")
    )
    if confidence_calibration:
        process_overlay = dict(process_overlay or {})
        process_overlay["confidence_calibration"] = confidence_calibration

    st["process_overlay"] = process_overlay

    try:
        path = engine.generate_fr_report(
            st["prompt"], st["evaluation"],
            model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
            output_dir=user_reports_dir,
            thinking_profile=thinking_profile,
            process_overlay=process_overlay,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Persist the raw writing-process trace next to the report so it can be
    # replayed later. Best-effort — never fail report generation over it.
    if process_log:
        try:
            trace_path = Path(path).parent / (Path(path).stem + ".trace.json")
            trace_path.write_text(json.dumps({
                "final_text":  (st["evaluation"].get("text", "") or "")[:MAX_SUBMISSION_CHARS],
                "process_log": _encode_process_log(process_log, MAX_SUBMISSION_CHARS),
                "encoded":     "delta-v1",
            }), encoding="utf-8")
        except Exception:
            pass

    return jsonify({"path": str(path)})


# ── Learning Profile API ───────────────────────────────────────────────────────

@app.route("/api/users", methods=["POST"])
@auth.admin_required
def api_users():
    """Return all non-admin users for the admin's profile viewer."""
    all_users = auth.load_users()
    user_list = [
        {"username": uname, "display_name": udata.get("display_name", uname)}
        for uname, udata in sorted(all_users.items())
        if udata.get("role") != "admin"
    ]
    return jsonify({"users": user_list})


@app.route("/api/learning-profile", methods=["POST"])
@auth.login_required
def api_learning_profile():
    """Return parsed report history for a user's learning profile dashboard."""
    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()

    if username and username != session["user_id"]:
        if session.get("role") != "admin":
            return jsonify({"error": "forbidden"}), 403
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
            return jsonify({"error": "invalid username"}), 400
    else:
        username = session["user_id"]

    all_users    = auth.load_users()
    display_name = all_users.get(username, {}).get("display_name", username)

    user_dir = REPORTS_BASE / username
    reports:      list = []
    hm_entries:   list = []
    solo_entries: list = []
    all_patterns: list = []
    all_gaps:     list = []

    if user_dir.is_dir():
        for f in sorted(user_dir.glob("*.md")):
            fname = f.name
            m = re.search(r'(\d{8})_(\d{6})', fname)
            if not m:
                continue
            d, t     = m.group(1), m.group(2)
            ts_str   = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}"

            try:
                content = f.read_text(encoding="utf-8")
                parsed  = report_parser.parse_report_md(content)
            except Exception:
                continue

            rtype     = parsed.get("type", "scenario")
            meta      = parsed.get("metadata", {})
            score_str = meta.get("score") if rtype == "fr" else meta.get("average_score")
            try:
                score = int((score_str or "0").rstrip('%'))
            except (ValueError, AttributeError):
                score = 0

            if rtype == "fr":
                title = meta.get("prompt", "Free Response")
            else:
                sc_list = parsed.get("scenarios", [])
                title   = sc_list[0]["title"] if sc_list else "Scenario"

            url = (
                f"/admin/report/{username}/{fname}"
                if session.get("role") == "admin"
                else f"/my-report/{fname}"
            )

            reports.append({
                "filename":  fname,
                "type":      rtype,
                "timestamp": ts_str,
                "date_str":  date_str,
                "score":     score,
                "title":     title,
                "url":       url,
            })

            # collect aggregate analysis data
            tp = parsed.get("thinking_profile")
            if tp:
                hm = tp.get("honey_mumford")
                if hm and hm.get("style"):
                    ev = hm.get("evidence", [])
                    hm_entries.append({
                        "style":      hm["style"],
                        "confidence": hm.get("confidence", ""),
                        "evidence":   ev if isinstance(ev, list) else ([ev] if ev else []),
                        "reasoning":  hm.get("reasoning", ""),
                        "date":       date_str,
                        "type":       rtype,
                        "score":      score,
                    })
                solo = tp.get("solo")
                if solo and solo.get("level"):
                    ev = solo.get("evidence", [])
                    solo_entries.append({
                        "level":      solo["level"],
                        "confidence": solo.get("confidence", ""),
                        "evidence":   ev if isinstance(ev, list) else ([ev] if ev else []),
                        "reasoning":  solo.get("reasoning", ""),
                        "date":       date_str,
                        "type":       rtype,
                        "score":      score,
                    })
                for p in (tp.get("patterns") or []):
                    if p not in all_patterns:
                        all_patterns.append(p)

            is_m = parsed.get("instructor_summary")
            if is_m:
                for g in (is_m.get("learning_gaps") or []):
                    if g not in all_gaps:
                        all_gaps.append(g)

    reports.sort(key=lambda r: r["timestamp"])

    scores   = [r["score"] for r in reports]
    sc_count = sum(1 for r in reports if r["type"] == "scenario")
    fr_count = sum(1 for r in reports if r["type"] == "fr")
    avg      = round(sum(scores) / len(scores)) if scores else 0
    best     = max(scores) if scores else 0

    if len(scores) >= 2:
        mid   = len(scores) // 2
        early = sum(scores[:mid]) / mid
        late  = sum(scores[mid:]) / (len(scores) - mid)
        trend = round(late - early)
    else:
        trend = 0

    return jsonify({
        "username":     username,
        "display_name": display_name,
        "reports":      reports,
        "summary": {
            "total":          len(reports),
            "scenario_count": sc_count,
            "fr_count":       fr_count,
            "average_score":  avg,
            "best_score":     best,
            "trend":          trend,
        },
        "aggregate": {
            "hm_entries":   hm_entries,
            "solo_entries": solo_entries,
            "all_patterns": all_patterns[:12],
            "all_gaps":     all_gaps[:12],
            "has_data":     bool(hm_entries or solo_entries or all_patterns or all_gaps),
        },
    })


@app.route("/api/learning-profile/analysis", methods=["POST"])
@auth.login_required
def api_learning_profile_analysis():
    """Generate an LLM-synthesised learning profile from all of a user's reports."""
    import llm as llm_module

    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()

    if username and username != session["user_id"]:
        if session.get("role") != "admin":
            return jsonify({"error": "forbidden"}), 403
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
            return jsonify({"error": "invalid username"}), 400
    else:
        username = session["user_id"]

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]
    base_url         = provider_cfg["base_url"]
    model            = data.get("model") or provider_cfg["model"]

    if not engine.llm_is_available(api_key):
        return jsonify({"error": "LLM not available — configure a provider with an API key."}), 400

    all_users    = auth.load_users()
    display_name = all_users.get(username, {}).get("display_name", username)
    user_dir     = REPORTS_BASE / username
    entries      = []

    if user_dir.is_dir():
        for f in sorted(user_dir.glob("*.md")):
            fname = f.name
            m = re.search(r'(\d{8})_(\d{6})', fname)
            if not m:
                continue
            d, t     = m.group(1), m.group(2)
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}"
            try:
                content = f.read_text(encoding="utf-8")
                parsed  = report_parser.parse_report_md(content)
            except Exception:
                continue

            rtype     = parsed.get("type", "scenario")
            meta      = parsed.get("metadata", {})
            score_str = meta.get("score") if rtype == "fr" else meta.get("average_score")
            try:
                score = int((score_str or "0").rstrip('%'))
            except (ValueError, AttributeError):
                score = 0

            entries.append({
                "date":    date_str,
                "type":    "Free Response" if rtype == "fr" else "Scenario",
                "score":   score,
                "summary": parsed.get("instructor_summary") or {},
                "profile": parsed.get("thinking_profile"),
            })

    if not entries:
        return jsonify({"error": "No reports found for this user."}), 404

    # Build the synthesis prompt
    lines = [
        f"You are analysing the learning profile of {display_name} across "
        f"{len(entries)} assessment(s).",
        "",
        "ASSESSMENT HISTORY (chronological):",
    ]
    for i, e in enumerate(entries, 1):
        lines.append(f"\nAssessment {i} — {e['date']} ({e['type']}, score: {e['score']}%)")
        s = e["summary"]
        if s.get("assessment"):
            lines.append(f"  Instructor assessment: {s['assessment']}")
        if s.get("learning_gaps"):
            lines.append(f"  Learning gaps: {', '.join(s['learning_gaps'])}")
        if s.get("recommendations"):
            lines.append(f"  Recommendations: {', '.join(s['recommendations'])}")
        tp = e["profile"]
        if tp:
            hm = tp.get("honey_mumford")
            if hm and hm.get("style"):
                conf = f" ({hm['confidence']} confidence)" if hm.get("confidence") else ""
                lines.append(f"  Learning style (Honey & Mumford): {hm['style']}{conf}")
                if hm.get("reasoning"):
                    lines.append(f"    Reasoning: {hm['reasoning']}")
            solo = tp.get("solo")
            if solo and solo.get("level"):
                conf = f" ({solo['confidence']} confidence)" if solo.get("confidence") else ""
                lines.append(f"  Cognitive level (SOLO Taxonomy): {solo['level']}{conf}")
                if solo.get("reasoning"):
                    lines.append(f"    Reasoning: {solo['reasoning']}")
            if tp.get("patterns"):
                lines.append(f"  Observed patterns: {'; '.join(tp['patterns'])}")
            if tp.get("instructor_note"):
                lines.append(f"  Instructor note: {tp['instructor_note']}")

    lines += [
        "",
        "Synthesise a comprehensive, evidence-based learning profile.",
        "Respond ONLY with valid JSON (no markdown, no extra text):",
        "{",
        '  "overall_narrative": "<2-3 sentence overarching summary of who this learner is>",',
        '  "learning_style_summary": "<paragraph on their Honey & Mumford style consistency, '
        'what it reveals about how they prefer to engage with tasks, and any evolution across assessments>",',
        '  "cognitive_development": "<paragraph on SOLO level progression and what it reveals '
        'about depth of understanding and conceptual integration>",',
        '  "consistent_strengths": ["<strength 1>", "<strength 2>"],',
        '  "development_areas": ["<area 1>", "<area 2>"],',
        '  "recommendations": ["<concrete actionable recommendation 1>", '
        '"<recommendation 2>", "<recommendation 3>"]',
        "}",
    ]

    system = (
        "You are an expert educational psychologist. "
        "Synthesise assessment data into a clear, evidence-based learning profile. "
        "Be specific — reference actual evidence from the assessments, not generic advice. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    try:
        raw    = llm_module.llm_chat(model, system, "\n".join(lines), api_key, base_url)
        result = llm_module._extract_json(raw)
        if not result:
            result = {
                "overall_narrative":     raw,
                "learning_style_summary": "",
                "cognitive_development":  "",
                "consistent_strengths":   [],
                "development_areas":      [],
                "recommendations":        [],
            }
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    return jsonify({"analysis": result, "report_count": len(entries)})


if __name__ == "__main__":
    print("\n  Performative Assessment  →  http://localhost:5001")
    print("  Admin:    admin / admin123")
    print("  Students: emma liam sofia james priya tyler / Learn@2024\n")
    app.run(host="localhost", port=5001, debug=False)
