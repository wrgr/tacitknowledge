"""
Authentication, rate limiting, and input sanitisation for the secure gateway.
User store is now SQLite via database.py (replaces users.json).
"""

import time
from functools import wraps

from flask import abort, redirect, session, url_for
from werkzeug.security import check_password_hash

import database as db

# ── In-memory rate-limit store — keyed by client IP ───────────────────────────
_rate: dict = {}

WINDOW_SECS        = 60
GENERAL_MAX        = 10
ADMIN_MAX          = 3
ADMIN_LOCKOUT_SECS = 900


# ── Input sanitisation ─────────────────────────────────────────────────────────

def sanitize_str(value, max_len=128):
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").strip()[:max_len]


# ── Rate limiting ──────────────────────────────────────────────────────────────

def _ip_state(ip):
    if ip not in _rate:
        _rate[ip] = {
            "count":              0,
            "window_start":       time.monotonic(),
            "admin_attempts":     0,
            "admin_locked_until": 0.0,
        }
    return _rate[ip]


def is_admin_locked(ip):
    return time.monotonic() < _ip_state(ip)["admin_locked_until"]


def check_limit(ip):
    now = time.monotonic()
    s = _ip_state(ip)
    if now - s["window_start"] > WINDOW_SECS:
        s["count"] = 0
        s["window_start"] = now
    return s["count"] < GENERAL_MAX


def record_failed(ip, is_admin_attempt):
    now = time.monotonic()
    s = _ip_state(ip)
    if now - s["window_start"] > WINDOW_SECS:
        s["count"] = 0
        s["window_start"] = now
    s["count"] += 1
    if is_admin_attempt:
        s["admin_attempts"] += 1
        if s["admin_attempts"] >= ADMIN_MAX:
            s["admin_locked_until"] = now + ADMIN_LOCKOUT_SECS
            return True
    return False


def record_success(ip, is_admin):
    s = _ip_state(ip)
    s["count"] = 0
    if is_admin:
        s["admin_attempts"] = 0


# ── User store (SQLite-backed) ─────────────────────────────────────────────────

def seed_default_users():
    db.seed_default_users()


def load_users():
    return db.all_users()


def authenticate(username, password):
    user = db.get_user(username)
    if not user:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return user


# ── Flask route decorators ─────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def _inner(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return _inner


def admin_required(f):
    @wraps(f)
    def _inner(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            abort(403)
        return f(*args, **kwargs)
    return _inner
