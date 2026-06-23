"""
Authentication, rate limiting, and input sanitisation for the secure gateway.
"""

import json
import time
from functools import wraps
from pathlib import Path

from flask import abort, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

USERS_FILE = Path(__file__).parent / "users.json"

# ── In-memory rate-limit store — keyed by client IP ───────────────────────────
# Each entry holds:
#   count            — general failed-login count in the current window
#   window_start     — monotonic timestamp when the current window began
#   admin_attempts   — consecutive failed attempts against an admin account
#   admin_locked_until — monotonic timestamp until which admin login is blocked
_rate: dict = {}

WINDOW_SECS        = 60     # rolling window for general attempt counting
GENERAL_MAX        = 10     # max failed attempts (any account) per window
ADMIN_MAX          = 3      # failed admin-login attempts before hard lockout
ADMIN_LOCKOUT_SECS = 900    # 15-minute hard lockout after admin threshold


# ── Input sanitisation ─────────────────────────────────────────────────────────

def sanitize_str(value, max_len=128):
    """Coerce to str, strip null bytes and surrounding whitespace, truncate."""
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").strip()[:max_len]


# ── Rate limiting ──────────────────────────────────────────────────────────────

def _ip_state(ip):
    """Return (and lazily initialise) the rate-limit record for *ip*."""
    if ip not in _rate:
        _rate[ip] = {
            "count":             0,
            "window_start":      time.monotonic(),
            "admin_attempts":    0,
            "admin_locked_until": 0.0,
        }
    return _rate[ip]


def is_admin_locked(ip):
    """True when this IP is currently inside the admin hard-lockout window."""
    return time.monotonic() < _ip_state(ip)["admin_locked_until"]


def check_limit(ip):
    """Return True (allowed) when the IP has not exceeded the general attempt cap."""
    now = time.monotonic()
    s = _ip_state(ip)
    if now - s["window_start"] > WINDOW_SECS:
        s["count"] = 0
        s["window_start"] = now
    return s["count"] < GENERAL_MAX


def record_failed(ip, is_admin_attempt):
    """
    Increment failure counters for *ip*.
    Returns True if the IP has just been hard-locked (admin threshold reached).
    """
    now = time.monotonic()
    s = _ip_state(ip)

    # reset general window if expired
    if now - s["window_start"] > WINDOW_SECS:
        s["count"] = 0
        s["window_start"] = now
    s["count"] += 1

    if is_admin_attempt:
        s["admin_attempts"] += 1
        if s["admin_attempts"] >= ADMIN_MAX:
            s["admin_locked_until"] = now + ADMIN_LOCKOUT_SECS
            return True   # just triggered lockout

    return False


def record_success(ip, is_admin):
    """Reset attempt counters after a successful login."""
    s = _ip_state(ip)
    s["count"] = 0
    if is_admin:
        s["admin_attempts"] = 0


# ── User store ─────────────────────────────────────────────────────────────────

def load_users():
    """Read and return the users dict from users.json."""
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users):
    """Persist the users dict to users.json."""
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def seed_default_users():
    """Create users.json with a default admin and demo student on first run."""
    if USERS_FILE.exists():
        return
    users = {
        "admin": {
            "password_hash": generate_password_hash("admin123"),
            "role":          "admin",
            "display_name":  "Administrator",
        },
        "student": {
            "password_hash": generate_password_hash("student123"),
            "role":          "student",
            "display_name":  "Demo Student",
        },
    }
    save_users(users)
    print("  [auth] Created users.json — change default passwords before deploying.")


def authenticate(username, password):
    """
    Return the user record dict on success, or None on failure.
    Callers are responsible for rate-limit checks before calling this.
    """
    users = load_users()
    user = users.get(username)
    if not user:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return user


# ── Flask route decorators ────────────────────────────────────────────────────

def login_required(f):
    """Redirect to /login when no authenticated session is present."""
    @wraps(f)
    def _inner(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return _inner


def admin_required(f):
    """Require an active admin session; 403 for any other authenticated role."""
    @wraps(f)
    def _inner(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            abort(403)
        return f(*args, **kwargs)
    return _inner
