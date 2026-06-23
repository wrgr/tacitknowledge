"""
Performative Assessment — web interface with secure student/admin gateway.
Run with:  python app.py
Open:      http://localhost:5000

Default credentials (change in users.json before deploying):
  admin   / admin123
  student / student123

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
import json
import re
import secrets
import uuid
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template,
                   request, session, url_for)

import auth
import config
import engine

# ── Persistent secret key ──────────────────────────────────────────────────────
# Stored in .secret_key so sessions survive server restarts.
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
    SESSION_COOKIE_HTTPONLY   = True,
    SESSION_COOKIE_SAMESITE   = "Strict",
    SESSION_COOKIE_SECURE     = False,   # set True when serving over HTTPS
    PERMANENT_SESSION_LIFETIME = 3600,   # 1-hour session lifetime
)

auth.seed_default_users()   # create users.json with defaults on first run

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
PROMPTS_DIR   = Path(__file__).parent / "prompts"
REPORTS_BASE  = Path(__file__).parent / config.REPORTS_DIR   # base reports folder

scenarios            = engine.load_scenarios(SCENARIOS_DIR)
prompts              = engine.load_prompts(PROMPTS_DIR)
configured_providers = engine.get_configured_providers(config.PROVIDERS)

# Active scenario assessment sessions keyed by UUID.
# Format: { uuid: { "runner": ..., "session": ..., "scenario": ...,
#                   "user_id": <str>, "profile": None } }
_state: dict = {}

# Active free-response sessions keyed by UUID.
# Format: { uuid: { "prompt": ..., "evaluation": ..., "profile": None,
#                   "user_id": <str>, "model": ..., "api_key": ..., "base_url": ... } }
_fr_state: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_csrf():
    """Generate a fresh CSRF token and store it in the Flask session."""
    tok = secrets.token_hex(32)
    session["csrf_token"] = tok
    return tok


def _post_login_redirect():
    """Send the freshly-authenticated user to the right page for their role."""
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("index"))


def _get_state(data):
    """
    Look up an assessment session by ID and authorise access.
    Returns (state_dict, None) on success or (None, error_response) on failure.
    """
    st = _state.get(data.get("session_id"))
    if not st:
        return None, (jsonify({"error": "session not found"}), 404)
    # Students may only access their own sessions; admins can access any
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return None, (jsonify({"error": "forbidden"}), 403)
    return st, None


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    ip = request.remote_addr or "0.0.0.0"

    # Already authenticated → skip the login page
    if "user_id" in session:
        return _post_login_redirect()

    # Hard-locked IP → send to lockout page
    if auth.is_admin_locked(ip):
        return redirect(url_for("locked"))

    # ── GET: show login form ───────────────────────────────────────────────────
    if request.method == "GET":
        return render_template("login.html", error=None,
                               csrf_token=_new_csrf(), username_val="")

    # ── POST: validate and process credentials ────────────────────────────────

    # CSRF check — compare using constant-time comparison to prevent timing attacks
    form_tok = request.form.get("csrf_token", "")
    sess_tok = session.pop("csrf_token", "")
    if not sess_tok or not hmac.compare_digest(
            form_tok.encode("utf-8"), sess_tok.encode("utf-8")):
        return render_template("login.html",
                               error="Invalid request. Please try again.",
                               csrf_token=_new_csrf(), username_val="")

    # General rate-limit check
    if not auth.check_limit(ip):
        return render_template("login.html",
                               error="Too many login attempts. Please wait before trying again.",
                               csrf_token=_new_csrf(), username_val="")

    # Sanitise inputs: strip null bytes, trim whitespace, cap length
    username = auth.sanitize_str(request.form.get("username", ""), max_len=64)
    password = auth.sanitize_str(request.form.get("password", ""), max_len=128)

    if not username or not password:
        return render_template("login.html",
                               error="Username and password are required.",
                               csrf_token=_new_csrf(), username_val=username)

    # Determine whether this attempt targets an admin account so the right
    # lockout counter is incremented (even if the username doesn't exist).
    all_users       = auth.load_users()
    attempted_role  = all_users.get(username, {}).get("role", "")
    is_admin_attempt = (attempted_role == "admin")

    user = auth.authenticate(username, password)

    if user is None:
        just_locked = auth.record_failed(ip, is_admin_attempt)
        if just_locked:
            # 3rd failed admin attempt — boot to lockout page immediately
            return redirect(url_for("locked"))
        # Use a generic message so attackers can't enumerate valid usernames
        return render_template("login.html",
                               error="Invalid username or password.",
                               csrf_token=_new_csrf(), username_val=username)

    # Successful login
    is_admin = (user["role"] == "admin")
    auth.record_success(ip, is_admin)
    session.clear()
    session["user_id"]      = username
    session["role"]         = user["role"]
    session["display_name"] = user.get("display_name", username)

    return _post_login_redirect()


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/locked")
def locked():
    ip = request.remote_addr or "0.0.0.0"
    # If the lock has expired and the user isn't logged in, let them try again
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

    # Build a dict mapping username → sorted list of report filenames
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
    )


@app.route("/admin/report/<username>/<filename>")
@auth.admin_required
def admin_view_report(username, filename):
    # Strict validation to prevent any path traversal
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', username):
        abort(400)
    if not re.match(r'^report_[\d_]+\.md$', filename):
        abort(400)

    report_path = REPORTS_BASE / username / filename
    if report_path.resolve().parent != (REPORTS_BASE / username).resolve():
        # Resolved path escapes the user directory — block path traversal
        abort(400)
    if not report_path.exists():
        abort(404)

    content = report_path.read_text(encoding="utf-8")
    return render_template(
        "report_view.html",
        username=username,
        filename=filename,
        content=content,
        admin_name=session.get("display_name", "Admin"),
    )


# ── API endpoints (all require an active login session) ───────────────────────

@app.route("/api/scenarios", methods=["POST"])
@auth.login_required
def api_scenarios():
    scenario_list = []
    for i, s in enumerate(scenarios):
        scenario_list.append({
            "index":       i,
            "id":          s["id"],
            "title":       s["title"],
            "description": s["description"],
            "max_turns":   s["max_turns"],
        })

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
        "scenarios":        scenario_list,
        "providers":        all_providers,
        "default_provider": default,
    })


@app.route("/api/models", methods=["POST"])
@auth.login_required
def api_models():
    data          = request.get_json()
    provider_name = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg  = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    models        = engine.get_available_models(provider_name, provider_cfg)
    default       = provider_cfg.get("model", models[0] if models else "")
    return jsonify({"models": models, "default": default})


@app.route("/api/start", methods=["POST"])
@auth.login_required
def api_start():
    data     = request.get_json()
    scenario = scenarios[int(data["index"])]
    sid      = str(uuid.uuid4())

    provider_name    = data.get("provider") or config.DEFAULT_PROVIDER
    provider_cfg     = config.PROVIDERS.get(provider_name) or config.PROVIDERS[config.DEFAULT_PROVIDER]
    api_key_override = (data.get("api_key") or "").strip()
    api_key          = api_key_override if api_key_override else provider_cfg["api_key"]
    base_url         = provider_cfg["base_url"]
    model            = data.get("model") or provider_cfg["model"]
    use_llm          = engine.llm_is_available(api_key)

    runner  = engine.ScenarioRunner(scenario, model=model, api_key=api_key, base_url=base_url)
    opening = runner.start()

    _state[sid] = {
        "runner":   runner,
        "session":  engine.Session(use_llm, model=model, api_key=api_key, base_url=base_url),
        "scenario": scenario,
        "profile":  None,
        "user_id":  session["user_id"],   # bind this assessment to the logged-in user
    }

    return jsonify({
        "session_id": sid,
        "opening":    opening,
        "max_turns":  scenario["max_turns"],
    })


@app.route("/api/respond", methods=["POST"])
@auth.login_required
def api_respond():
    data = request.get_json()
    st, err = _get_state(data)
    if err:
        return err

    narration, concluded = st["runner"].respond(data["user_input"])
    return jsonify({"narration": narration, "concluded": concluded})


@app.route("/api/evaluate", methods=["POST"])
@auth.login_required
def api_evaluate():
    data = request.get_json()
    st, err = _get_state(data)
    if err:
        return err

    runner  = st["runner"]
    sess    = st["session"]
    evaluations = sess.evaluate(runner.scenario, transcript=runner.transcript())

    eval_list = [
        {
            "score":          ev["score"],
            "feedback":       ev["feedback"],
            "strengths":      ev["strengths"],
            "gaps":           ev["gaps"],
            "matched_points": ev["matched_points"],
            "missed_points":  ev["missed_points"],
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
    data = request.get_json()
    st, err = _get_state(data)
    if err:
        return err

    sess     = st["session"]
    runner   = st["runner"]
    scenario = st["scenario"]

    if not sess.use_llm:
        return jsonify({"profile": None})

    profile = engine.analyse_thinking_profile(
        scenario, runner.transcript(),
        model=sess.model, api_key=sess.api_key, base_url=sess.base_url,
    )
    st["profile"] = profile
    return jsonify({"profile": profile})


@app.route("/api/report", methods=["POST"])
@auth.login_required
def api_report():
    data = request.get_json()
    st, err = _get_state(data)
    if err:
        return err

    sess             = st["session"]
    user_reports_dir = REPORTS_BASE / session["user_id"]   # student-specific subdirectory

    path = engine.generate_report(
        sess,
        model=sess.model,
        api_key=sess.api_key,
        base_url=sess.base_url,
        output_dir=user_reports_dir,
        thinking_profile=st.get("profile"),
    )
    return jsonify({"path": str(path)})


@app.route("/api/generate-scenario", methods=["POST"])
@auth.login_required
def api_generate_scenario():
    data        = request.get_json()
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
@auth.login_required
def api_save_scenario():
    global scenarios
    data = request.get_json()

    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    scenario_id   = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    scenario_data = {
        "id":          scenario_id,
        "title":       title,
        "description": data.get("description", ""),
        "situation":   data.get("situation", ""),
        "user_role":   data.get("user_role", "participant"),
        "max_turns":   int(data.get("max_turns", 8)),
        "constraints": data.get("constraints", []),
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


# ── Free-Response API endpoints ───────────────────────────────────────────────

@app.route("/api/fr/prompts", methods=["POST"])
@auth.login_required
def api_fr_prompts():
    prompt_list = [
        {
            "index":       i,
            "id":          p["id"],
            "title":       p["title"],
            "description": p["description"],
            "word_limit":  p.get("word_limit"),
        }
        for i, p in enumerate(prompts)
    ]
    return jsonify({"prompts": prompt_list})


@app.route("/api/fr/check", methods=["POST"])
@auth.login_required
def api_fr_check():
    # Fast keyword check used by the live sidebar — no LLM call.
    data      = request.get_json()
    prompt_id = data.get("prompt_id", "")
    text      = data.get("text", "")

    prompt_data = next((p for p in prompts if p["id"] == prompt_id), None)
    if not prompt_data:
        return jsonify({"error": "prompt not found"}), 404

    result = engine.check_fr_keywords(prompt_data, text)
    return jsonify(result)


@app.route("/api/fr/submit", methods=["POST"])
@auth.login_required
def api_fr_submit():
    global _fr_state
    data      = request.get_json()
    prompt_id = data.get("prompt_id", "")
    text      = (data.get("text") or "").strip()

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

    sid = str(uuid.uuid4())
    _fr_state[sid] = {
        "prompt":     prompt_data,
        "evaluation": evaluation,
        "profile":    None,
        "user_id":    session["user_id"],
        "model":      model,
        "api_key":    api_key,
        "base_url":   base_url,
    }

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


@app.route("/api/fr/thinking-profile", methods=["POST"])
@auth.login_required
def api_fr_thinking_profile():
    data = request.get_json()
    sid  = data.get("session_id")
    st   = _fr_state.get(sid)

    if not st:
        return jsonify({"error": "session not found"}), 404
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    if not engine.llm_is_available(st["api_key"]):
        return jsonify({"profile": None})

    profile = engine.analyse_thinking_profile(
        st["prompt"], st["evaluation"]["text"],
        model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
    )
    st["profile"] = profile
    return jsonify({"profile": profile})


@app.route("/api/fr/report", methods=["POST"])
@auth.login_required
def api_fr_report():
    data = request.get_json()
    sid  = data.get("session_id")
    st   = _fr_state.get(sid)

    if not st:
        return jsonify({"error": "session not found"}), 404
    if st["user_id"] != session.get("user_id") and session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    user_reports_dir = REPORTS_BASE / session["user_id"]

    path = engine.generate_fr_report(
        st["prompt"], st["evaluation"],
        model=st["model"], api_key=st["api_key"], base_url=st["base_url"],
        output_dir=user_reports_dir,
        thinking_profile=st.get("profile"),
    )
    return jsonify({"path": str(path)})


if __name__ == "__main__":
    print("\n  Performative Assessment  →  http://localhost:5000")
    print("  Default logins:  admin / admin123   |   student / student123")
    print("  Change default passwords in users.json before deploying.\n")
    app.run(host="localhost", port=5000, debug=False)
