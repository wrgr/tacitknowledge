"""
llm.py — LLM provider dispatch and response parsing.
"""

import json
import re
import urllib.request

_ANTHROPIC_HOST = "api.anthropic.com"

_PROVIDER_MODELS = {
    "OpenAI":  ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Claude":  ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "Gemini":  ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "Groq":    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "Mistral": ["mistral-large-latest", "mistral-small-latest", "mistral-nemo"],
}


def llm_is_available(api_key):
    return bool(api_key) and not api_key.startswith("your-")


def validate_api_key(provider_name, api_key, model, base_url):
    """Try a minimal API call. Returns (ok, error_message)."""
    import urllib.error

    if provider_name == "Ollama":
        try:
            host = base_url.rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3]
            req = urllib.request.Request(host + "/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
            return True, None
        except Exception:
            return False, "Cannot connect to Ollama"

    if _ANTHROPIC_HOST in base_url:
        try:
            import anthropic
        except ImportError:
            return False, "anthropic package not installed"
        try:
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model=model, max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, None
        except anthropic.AuthenticationError:
            return False, "Invalid API key"
        except Exception:
            return False, "Cannot connect to provider"

    # OpenAI-compatible providers (OpenAI, Groq, Mistral, Gemini, etc.)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    body = json.dumps({
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": "Hi"}],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, None
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "Invalid API key"
        if e.code == 429:
            return True, None  # rate-limited → key is valid
        return False, f"Provider error ({e.code})"
    except Exception:
        return False, "Cannot connect to provider"


def get_configured_providers(providers):
    return [{"name": name, "model": cfg["model"]}
            for name, cfg in providers.items()
            if llm_is_available(cfg["api_key"])]


def get_available_models(provider_name, provider_cfg):
    """Return the model list for a provider.
    Ollama: queried live from /api/tags so it reflects what the user actually has installed.
    Others: curated list from _PROVIDER_MODELS."""
    if provider_name == "Ollama":
        try:
            base = provider_cfg.get("base_url", "http://localhost:11434/v1").rstrip("/")
            host = base[:-3] if base.endswith("/v1") else base
            req = urllib.request.Request(host + "/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    return _PROVIDER_MODELS.get(provider_name, [])


def _fix_unescaped_quotes(s):
    """Escape double-quote characters that appear inside JSON string values.

    LLMs occasionally emit strings like:
        "reasoning": "the response ("quoted term") was..."
    where the inner quotes are not escaped.  This walks the text character-by-
    character and escapes any " that is not a structural delimiter.  A quote is
    treated as structural (i.e. genuinely closes the string) when the next
    non-whitespace character is one of  :  ,  }  ]  — the only chars that can
    legally follow a closed JSON string.  Any other successor means the quote
    is inside the value and needs a backslash prepended.
    """
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            if c == '\\':          # already-escaped sequence — copy both chars
                result.append(c)
                i += 1
                if i < len(s):
                    result.append(s[i])
                    i += 1
                continue
            if c == '"':
                # Peek at the next non-whitespace char to decide intent
                j = i + 1
                while j < len(s) and s[j] in ' \t\r\n':
                    j += 1
                next_c = s[j] if j < len(s) else ''
                if next_c in (':', ',', '}', ']'):
                    in_string = False   # legitimate closing quote
                    result.append(c)
                else:
                    result.append('\\')  # inner quote — escape it
                    result.append(c)
            else:
                result.append(c)
        else:
            if c == '"':
                in_string = True
            result.append(c)
        i += 1
    return ''.join(result)


def _extract_json(raw):
    """Extract the first valid JSON object from an LLM response.

    Handles three common failure modes:

    1. Greedy regex pollution — the old r"{.*}" (re.DOTALL) matched from the
       FIRST { to the LAST }, grabbing surrounding prose (e.g. "evaluation
       {of the transcript}: { ... }") and producing invalid JSON.  raw_decode()
       stops as soon as a complete object is found, so surrounding text is safe.

    2. Thinking-model leakage — reasoning models (DeepSeek R1, QWQ, Gemma 4)
       sometimes emit draft JSON inside <think>...</think> while they reason.
       Stripping those blocks before scanning means we never mistake a draft
       for the final answer.  llm_chat() also passes think=False to Ollama so
       thinking is suppressed at source; this strip is defense-in-depth for
       older Ollama builds that ignore that flag.

    3. Unescaped inner quotes — the LLM may write a double-quote inside a
       string value without escaping it (e.g. the word "quoted" inside a longer
       value).  If the strict pass fails, _fix_unescaped_quotes repairs these
       and a second parse attempt is made.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()

    # Pass 1: strict parse on the original text
    for m in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, m.start())
            return obj
        except json.JSONDecodeError:
            continue

    # Pass 2: repair unescaped inner quotes and retry
    repaired = _fix_unescaped_quotes(cleaned)
    for m in re.finditer(r"\{", repaired):
        try:
            obj, _ = decoder.raw_decode(repaired, m.start())
            return obj
        except json.JSONDecodeError:
            continue

    print("[llm] _extract_json: no valid JSON found in response. Raw output:\n" + raw[:500])
    return {}


def _raw_chat(model, api_key, base_url, max_tokens, system, user, think=None, json_mode=False):
    """Stdlib-only HTTP call. Tries OpenAI-compatible format first; falls back to Ollama's native API on 404."""
    import urllib.error

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})

    base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    is_ollama = not api_key or api_key.lower() == "ollama"
    if not is_ollama:
        headers["Authorization"] = "Bearer " + api_key

    # ── attempt 1: OpenAI-compatible /chat/completions ──
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if think is not None and is_ollama:
        body["think"] = think
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(body).encode()
    req = urllib.request.Request(base + "/chat/completions", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    # ── attempt 2: Ollama native /api/chat (older Ollama or base_url without /v1) ──
    host = base[:-3] if base.endswith("/v1") else base
    body = {"model": model, "messages": msgs, "stream": False, "options": {"num_predict": max_tokens}}
    if think is not None:
        body["think"] = think
    if json_mode:
        body["format"] = "json"
    payload = json.dumps(body).encode()
    req = urllib.request.Request(host + "/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"] or ""


def _call_llm(model, api_key, base_url, max_tokens, system, user, think=None, json_mode=False):
    """Dispatch to the right backend. No package is required at import time."""
    try:
        if _ANTHROPIC_HOST in base_url:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for the Claude provider. "
                    "Run: pip install anthropic"
                )
            client = anthropic.Anthropic(api_key=api_key)
            kwargs = {"model": model, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": user}]}
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            return next((b.text for b in response.content if b.type == "text"), "")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                msgs = []
                if system:
                    msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": user})
                is_ollama = not api_key or api_key.lower() == "ollama"
                extra = {"extra_body": {"think": think}} if think is not None and is_ollama else {}
                if json_mode:
                    extra["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=msgs, **extra)
                return response.choices[0].message.content or ""
            except ImportError:
                return _raw_chat(model, api_key, base_url, max_tokens, system, user, think=think, json_mode=json_mode)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ConnectionError("Cannot reach the LLM API. Check your API key, base URL, and network.") from e


def llm_generate(model, prompt, api_key, base_url):
    # One-sentence narration — thinking mode disabled so reasoning models respond directly.
    return _call_llm(model, api_key, base_url, 512, "", prompt, think=False)


def llm_chat(model, system, message, api_key, base_url):
    # Full structured response — used for evaluation and report generation.
    # think=False prevents thinking models from spending their token budget on
    # reasoning and truncating the JSON response.
    return _call_llm(model, api_key, base_url, 8192, system, message, think=False)


def llm_chat_json(model, system, message, api_key, base_url):
    # Like llm_chat but enables JSON mode (Ollama: format=json, OpenAI: response_format).
    # Uses a higher token budget (8192) because structured JSON responses with evidence
    # quotes and multi-field schemas are larger than typical prose completions.
    return _call_llm(model, api_key, base_url, 8192, system, message, think=False, json_mode=True)


def clip(text, max_chars=8000):
    """Truncate long text so the prompt fits within the model's context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for length]"
