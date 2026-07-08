"""
Provider configuration. Fill in your API key for each provider you want to use.
Providers with placeholder keys are hidden from the in-app dropdown automatically.
"""

PROVIDERS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o",
        "api_key":  "your-openai-key-here",
    },
    "Claude": {
        "base_url": "https://api.anthropic.com",
        "model":    "claude-opus-4-8",
        "api_key":  "your-anthropic-key-here",
    },
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model":    "gemini-2.5-flash",
        "api_key":  "your-google-key-here",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
        "api_key":  "your-groq-key-here",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model":    "mistral-small-latest",
        "api_key":  "your-mistral-key-here",
    },
    "GitHub Models": {
        # Free OpenAI-compatible endpoint for prototyping. Auth with a GitHub
        # personal access token that has the `models: read` permission.
        # Rate limited (free tier ~15 req/min, ~150 req/day) — light testing only.
        "base_url": "https://models.github.ai/inference",
        "model":    "openai/gpt-4o-mini",
        "api_key":  "your-github-pat-here",
    },
    "Ollama": {
        "base_url": "http://localhost:11434/v1",
        "model":    "llama3.2",
        "api_key":  "ollama",         # Ollama needs no real key; remove this entry if Ollama isn't installed
    },
}

DEFAULT_PROVIDER = "OpenAI"          # used by the CLI and as the pre-selected option in the web UI

REPORTS_DIR = "reports"              # folder (relative to this file) where generated Markdown reports are saved

# Self-consistency scoring: run the FINAL grading call (FR and scenario) N times
# and take a majority vote per key point, instead of trusting a single sample.
# Off by default -- costs SELF_CONSISTENCY_SAMPLES x the LLM calls and latency.
# Does not apply to evidence extraction, gap analysis, or thinking-profile
# classification -- those don't need this level of reliability investment.
SELF_CONSISTENCY_SCORING = False

# TUNABLE -- number of samples for self-consistency scoring, only used
# when SELF_CONSISTENCY_SCORING is enabled. Higher = more reliable,
# more cost/latency.
SELF_CONSISTENCY_SAMPLES = 3

# ── Google OAuth sign-in (optional) ───────────────────────────────────────────
# Enabled only when GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are set in the
# environment (create an OAuth 2.0 "Web application" client in Google Cloud
# Console; redirect URI: http://localhost:5001/auth/google/callback).
# Password login always keeps working alongside it.
import os as _os

GOOGLE_CLIENT_ID     = _os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = _os.environ.get("GOOGLE_CLIENT_SECRET", "")

# Access policy for Google sign-in:
#  - GOOGLE_ALLOWED_DOMAIN: only emails @this-domain may sign in ("" = any
#    verified Google account — fine for development, not for data collection).
#  - GOOGLE_ADMIN_EMAILS: comma-separated emails that get the admin role;
#    everyone else who passes the domain gate becomes a student.
GOOGLE_ALLOWED_DOMAIN = _os.environ.get("GOOGLE_ALLOWED_DOMAIN", "")
GOOGLE_ADMIN_EMAILS   = {e.strip().lower()
                         for e in _os.environ.get("GOOGLE_ADMIN_EMAILS", "").split(",")
                         if e.strip()}
