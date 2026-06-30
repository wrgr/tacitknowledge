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
    "Ollama": {
        "base_url": "http://localhost:11434/v1",
        "model":    "llama3.2",
        "api_key":  "ollama",         # Ollama needs no real key; remove this entry if Ollama isn't installed
    },
}

DEFAULT_PROVIDER = "OpenAI"          # used by the CLI and as the pre-selected option in the web UI

REPORTS_DIR = "reports"              # folder (relative to this file) where generated Markdown reports are saved
