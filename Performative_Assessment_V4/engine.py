"""
engine.py — all the program logic in one file.
Imported by cli.py (terminal) and app.py (web).
"""

import json            # reads/writes JSON data (scenario files, LLM responses)
import re              # used to search for JSON inside messy LLM output
import urllib.request  # stdlib HTTP — used as a zero-dependency fallback for OpenAI-compatible endpoints
from datetime import datetime  # used to timestamp report filenames
from pathlib import Path       # makes file paths work on any operating system

# SDK imports are intentionally lazy (inside _call_llm) so the program starts
# without requiring any extra packages — Ollama works out of the box via stdlib HTTP.

_ANTHROPIC_HOST = "api.anthropic.com"  # sentinel used to route Claude requests to the Anthropic SDK


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPERS
# Provider-agnostic interface. Configure providers in config.py.
# Claude uses the Anthropic SDK; every other provider uses the OpenAI-compatible SDK.
# ─────────────────────────────────────────────────────────────────────────────

def llm_is_available(api_key):
    # returns True if a real API key has been provided (not a placeholder)
    return bool(api_key) and not api_key.startswith("your-")


def get_configured_providers(providers):
    # return a list of {name, model} dicts for providers that have a real API key
    return [{"name": name, "model": cfg["model"]}
            for name, cfg in providers.items()
            if llm_is_available(cfg["api_key"])]


# Curated model lists for cloud providers — shown in the model dropdown
_PROVIDER_MODELS = {
    "OpenAI":  ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Claude":  ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "Gemini":  ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "Groq":    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "Mistral": ["mistral-large-latest", "mistral-small-latest", "mistral-nemo"],
}


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


def _raw_chat(model, api_key, base_url, max_tokens, system, user, think=None):
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
        body["think"] = think   # Ollama-specific: disable/enable thinking mode
    payload = json.dumps(body).encode()
    req = urllib.request.Request(base + "/chat/completions", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise   # unexpected error — let the caller handle it

    # ── attempt 2: Ollama native /api/chat (older Ollama or base_url without /v1) ──
    host = base[:-3] if base.endswith("/v1") else base   # strip /v1 to reach the root
    body = {"model": model, "messages": msgs, "stream": False, "options": {"num_predict": max_tokens}}
    if think is not None:
        body["think"] = think   # disables thinking for Gemma 4, QWQ, DeepSeek R1, etc.
    payload = json.dumps(body).encode()
    req = urllib.request.Request(host + "/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"] or ""


def _call_llm(model, api_key, base_url, max_tokens, system, user, think=None):
    """Dispatch to the right backend. No package is required at import time."""
    try:
        if _ANTHROPIC_HOST in base_url:
            # Claude's native API — requires: pip install anthropic
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
            # OpenAI-compatible endpoint (OpenAI, Groq, Mistral, Ollama, …)
            # Uses the openai SDK if installed, falls back to stdlib HTTP otherwise.
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                msgs = []
                if system:
                    msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": user})
                is_ollama = not api_key or api_key.lower() == "ollama"
                extra = {"extra_body": {"think": think}} if think is not None and is_ollama else {}
                response = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=msgs, **extra)
                return response.choices[0].message.content or ""
            except ImportError:
                return _raw_chat(model, api_key, base_url, max_tokens, system, user, think=think)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ConnectionError("Cannot reach the LLM API. Check your API key, base URL, and network.") from e


def llm_generate(model, prompt, api_key, base_url):
    # One-sentence narration — thinking mode disabled so reasoning models respond directly.
    return _call_llm(model, api_key, base_url, 512, "", prompt, think=False)


def llm_chat(model, system, message, api_key, base_url):
    # full structured response — used for evaluation and report generation.
    # think=False mirrors llm_generate: prevents thinking models from spending
    # their token budget on reasoning and truncating the JSON response.
    return _call_llm(model, api_key, base_url, 4096, system, message, think=False)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO LOADING
# Reads the JSON files in the scenarios/ folder.
# Scenarios are plain Python dicts — just use scenario["title"], scenario["situation"], etc.
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(path):
    # open and parse one scenario JSON file
    with open(path) as f:
        data = json.load(f)

    # fill in default values for optional fields so we don't have to check later
    if "description" not in data:
        data["description"] = ""
    if "user_role" not in data:
        data["user_role"] = "participant"
    if "max_turns" not in data:
        data["max_turns"] = 8
    if "constraints" not in data:
        data["constraints"] = []
    if "metadata" not in data:
        data["metadata"] = {}

    # do the same for each expert answer inside the scenario
    for ea in data.get("expert_answers", []):
        if "key_points" not in ea:
            ea["key_points"] = []
        if "rubric" not in ea:
            ea["rubric"] = {}

    return data  # the scenario is just a plain dict — no special class needed


def load_scenarios(folder):
    # load every .json file in the given folder and return them as a list
    folder = Path(folder)
    scenarios = []
    for path in sorted(folder.glob("*.json")):  # sorted so the menu order is predictable
        scenarios.append(load_scenario(path))
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# Two ways to score a transcript:
#   score_with_keywords — simple text matching, no Ollama needed
#   score_with_llm      — asks Ollama to read and judge the transcript
# Both return a plain dict with the same keys.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw):
    """Extract the first valid JSON object from an LLM response.

    Handles two common failure modes:

    1. Greedy regex pollution — the old r"\{.*\}" (re.DOTALL) matched from the
       FIRST { to the LAST }, grabbing surrounding prose (e.g. "evaluation
       {of the transcript}: { ... }") and producing invalid JSON.  raw_decode()
       stops as soon as a complete object is found, so surrounding text is safe.

    2. Thinking-model leakage — reasoning models (DeepSeek R1, QWQ, Gemma 4)
       sometimes emit draft JSON inside <think>...</think> while they reason.
       Stripping those blocks before scanning means we never mistake a draft
       for the final answer.  llm_chat() also passes think=False to Ollama so
       thinking is suppressed at source; this strip is defense-in-depth for
       older Ollama builds that ignore that flag.
    """
    # strip reasoning blocks before searching for JSON
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, m.start())
            return obj
        except json.JSONDecodeError:
            continue
    return {}


def score_with_keywords(scenario, transcript, expert_answer):
    # check which key phrases appear anywhere in the transcript (case-insensitive)
    text = transcript.lower()

    matched = []  # key points the learner mentioned
    missed = []   # key points the learner didn't mention

    for point in expert_answer["key_points"]:
        if point.lower() in text:
            matched.append(point)
        else:
            missed.append(point)

    # calculate a numeric score
    rubric = expert_answer["rubric"]
    if rubric:
        # weighted score: each key point has a point value in the rubric
        total = sum(rubric.values())   # total possible points
        earned = 0
        for point in matched:
            earned += rubric.get(point, 1.0)  # add that point's value (default 1 if not in rubric)
        if total > 0:
            score = earned / total
        else:
            score = 0.0
    elif expert_answer["key_points"]:
        # unweighted score: what fraction of key points did they mention?
        score = len(matched) / len(expert_answer["key_points"])
    else:
        score = 0.0  # no key points defined, can't score anything

    # build a simple feedback string
    feedback_parts = []
    if matched:
        feedback_parts.append("Covered : " + ", ".join(matched))
    if missed:
        feedback_parts.append("Missing : " + ", ".join(missed))

    if feedback_parts:
        feedback = "\n".join(feedback_parts)
    else:
        feedback = "No key points defined."

    # return a plain dict — same shape as score_with_llm so the rest of the code works either way
    return {
        "scenario_id":    scenario["id"],
        "transcript":     transcript,
        "expert_answer":  expert_answer,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          round(score, 4),
        "feedback":       feedback,
        "strengths":      [],  # keyword scoring doesn't produce these (only LLM scoring does)
        "gaps":           []
    }


def score_with_llm(model, api_key, base_url, scenario, transcript, expert_answer):
    key_points = expert_answer.get("key_points", [])
    rubric     = expert_answer.get("rubric", {})

    # Show each key point with its importance label so the LLM knows what matters most
    _weight_label = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}
    if key_points:
        kp_lines = []
        for p in key_points:
            w = rubric.get(p, 1)
            kp_lines.append("- " + p + " [" + _weight_label.get(w, "MEDIUM") + " — " + str(w) + "pt" + ("s" if w != 1 else "") + "]")
        key_points_text = "\n".join(kp_lines)
    else:
        key_points_text = "- (none specified)"

    if scenario["constraints"]:
        constraints_text = "\n".join("- " + c for c in scenario["constraints"])
    else:
        constraints_text = "- (none specified)"

    prompt = (
        "SCENARIO: " + scenario["description"] + "\n\n"
        "CONSTRAINTS (the response must satisfy every one of these):\n"
        + constraints_text + "\n\n"
        "EXPERT ANSWER (the complete ideal response — use this as your benchmark):\n"
        + expert_answer["answer"] + "\n\n"
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n"
        + key_points_text + "\n\n"
        "ASSESSMENT TRANSCRIPT:\n"
        + transcript + "\n\n"
        "GRADING INSTRUCTIONS:\n"
        "- Your job is to determine which key points the learner clearly demonstrated.\n"
        "- A key point is matched ONLY if the learner explicitly stated or clearly performed it.\n"
        "- Do NOT give credit for: vague references, implied knowledge, partial answers, or enthusiasm.\n"
        "- Do NOT give benefit of the doubt — if it is not clearly in the transcript, it is missed.\n"
        "- CRITICAL and HIGH points that are missed represent serious failures — reflect this in gaps and feedback.\n"
        "- Your feedback must be direct and honest, not encouraging. Name exactly what was missed and why it matters.\n\n"
        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "matched_points": [<copy the exact key point strings the learner clearly demonstrated>],\n'
        '  "missed_points":  [<copy the exact key point strings the learner omitted or got wrong>],\n'
        '  "strengths": [<1-3 specific things done correctly, as short phrases>],\n'
        '  "gaps": [<one sentence per gap explaining what was missing and why it matters>],\n'
        '  "feedback": "<2-3 sentences of direct, honest feedback for the learner>"\n'
        "}"
    )

    system = (
        "You are a strict, impartial grader for a performative assessment. "
        "Your role is to evaluate performance against explicit criteria — not to encourage or reassure the learner. "
        "Credit only what is clearly and explicitly demonstrated in the transcript. "
        "Be critical: missing a CRITICAL or HIGH point is a significant failure and must be named. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw = llm_chat(model, system, prompt, api_key, base_url)

    result = _extract_json(raw)

    matched = result.get("matched_points", [])
    missed  = result.get("missed_points",  [])

    # Compute the score from rubric weights rather than accepting whatever number the LLM returns.
    # The LLM's job is semantic matching; arithmetic is ours.
    matched_set = set(matched)
    if rubric and key_points:
        total  = sum(rubric.get(p, 1) for p in key_points)
        earned = sum(rubric.get(p, 1) for p in key_points if p in matched_set)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        score = len([p for p in key_points if p in matched_set]) / len(key_points)
    else:
        score = 0.5  # no criteria defined

    # Ensure missed_points covers every key point not in matched
    # (the LLM sometimes forgets to list some in missed)
    all_missed = [p for p in key_points if p not in matched_set]
    if all_missed and not missed:
        missed = all_missed

    return {
        "scenario_id":    scenario["id"],
        "transcript":     transcript,
        "expert_answer":  expert_answer,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          max(0.0, min(1.0, score)),
        "feedback":       result.get("feedback", ""),
        "strengths":      result.get("strengths", []),
        "gaps":           result.get("gaps", [])
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DRAFT GENERATION
# Takes a plain-English description and asks the LLM to produce a full
# scenario JSON draft that the instructor can review and edit in the web UI.
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario_draft(description, model, api_key, base_url):
    prompt = (
        "You are an instructional designer creating a performative assessment scenario.\n\n"
        'Based on this description: "' + description + '"\n\n'
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "description": "<one sentence summary>",\n'
        '  "situation": "<2-3 sentences describing the scene in second person, starting with You...>",\n'
        '  "user_role": "<the learner\'s role, e.g. nurse, driver, technician>",\n'
        '  "max_turns": 8,\n'
        '  "constraints": ["<must-satisfy rule 1>", "<rule 2>", "<rule 3>"],\n'
        '  "expert_answer": "<2-3 sentences describing the complete ideal step-by-step response>",\n'
        '  "key_points": ["<short phrase 1>", "<short phrase 2>", "<6-10 total>"],\n'
        '  "rubric": {"<key point phrase>": <weight 1-4>, "<next phrase>": <weight>}\n'
        "}\n\n"
        "Guidelines:\n"
        "- situation: write in second person (You are...)\n"
        "- key_points: 6-10 short phrases the learner must mention to score well\n"
        "- rubric weights: 1=low importance, 2=medium, 3=high, 4=critical\n"
        "- every key_point must have an entry in rubric\n"
        "Return only the JSON, no other text."
    )

    system = (
        "You are an expert instructional designer. "
        "Create clear, realistic performative assessment scenarios. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw = llm_chat(model, system, prompt, api_key, base_url)

    return _extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# THINKING PROFILE ANALYSIS
# Looks at HOW the learner responded — sequencing, confidence, level of detail,
# whether they anticipated issues — to classify their thinking and learning style.
# Runs as a separate call after scoring so the score appears without delay.
# Results accumulate across sessions to build a longitudinal profile.
# ─────────────────────────────────────────────────────────────────────────────

def analyse_thinking_profile(scenario, transcript, model, api_key, base_url, prior_profiles=None):
    system = (
        "You are an educational psychologist. "
        "Classify a learner's response using two established frameworks. "
        "Base your analysis only on HOW they responded — language, sequencing, depth — not on their score. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    prompt = (
        "Scenario: " + scenario["title"] + "\n\n"
        "Transcript:\n" + transcript + "\n\n"

        "## Framework 1 — Honey & Mumford Learning Style\n"
        "Choose exactly one:\n"
        "- Activist: dives straight in, action-first, minimal planning, energetic language\n"
        "- Reflector: considers options before acting, hedged language, weighs consequences\n"
        "- Theorist: explains the reasoning and underlying rules, logical and sequential\n"
        "- Pragmatist: practical and direct, skips theory, focuses on what works\n\n"

        "## Framework 2 — SOLO Taxonomy (depth of understanding)\n"
        "Choose exactly one:\n"
        "- Prestructural: misses the point, irrelevant or no response to the task\n"
        "- Unistructural: identifies one relevant element, nothing more\n"
        "- Multistructural: covers several relevant elements but treats them in isolation\n"
        "- Relational: integrates elements coherently, shows how they connect\n"
        "- Extended Abstract: generalises beyond the task, considers edge cases or broader principles\n\n"

        "Return this JSON (be concise — one sentence per evidence field):\n"
        "{\n"
        '  "honey_mumford_style":    "<Activist | Reflector | Theorist | Pragmatist>",\n'
        '  "honey_mumford_evidence": "<one sentence from the transcript that supports this>",\n'
        '  "solo_level":             "<Prestructural | Unistructural | Multistructural | Relational | Extended Abstract>",\n'
        '  "solo_evidence":          "<one sentence describing the depth of understanding shown>",\n'
        '  "observed_patterns":      [<2–3 short phrases: specific thinking behaviours noticed>],\n'
        '  "instructor_note":        "<one sentence on how to scaffold learning for this learner>"\n'
        "}"
    )

    raw = llm_chat(model, system, prompt, api_key, base_url)
    return _extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNER
# Manages the back-and-forth conversation for one scenario.
# Kept as a class because it needs to remember state between turns
# (which turn we're on, what the learner said, etc.).
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_PROMPTS = ["Keep going.", "What else would you do?"]  # used when the API can't generate narration
FALLBACK_CLOSING = "Share anything else you would do."         # used on the second-to-last turn


class ScenarioRunner:

    def __init__(self, scenario, model, api_key, base_url):
        self.scenario = scenario      # the scenario dict loaded from JSON
        self.model = model            # LLM model name
        self.api_key = api_key        # provider API key
        self.base_url = base_url      # provider base URL
        self.history = []             # list of {"role": ..., "content": ...} dicts, one per turn
        self.user_inputs = []         # just the learner's responses (used to check if they said anything)
        self.turn = 0                 # counts how many learner responses have been submitted

    def start(self):
        # reset everything so the runner can be reused for a fresh attempt
        self.history = []
        self.user_inputs = []
        self.turn = 0
        # return the opening line shown to the learner
        return "Examiner: " + self.scenario["situation"] + "\n\nWhat do you do?"

    def _get_narration(self, user_input, closing=False):
        if closing:
            # ask for a single closing sentence inviting the learner to add anything missed
            prompt = (
                "Scenario: " + self.scenario["situation"] + "\n\n"
                'The learner said: "' + user_input + '"\n\n'
                "Write one short sentence for the examiner inviting the learner to add "
                "anything else they might have missed. Write only the sentence, no extra text."
            )
            fallback = FALLBACK_CLOSING
        else:
            # ask for a single sentence acknowledging what the learner just did
            prompt = (
                "Scenario: " + self.scenario["situation"] + "\n\n"
                'The learner said: "' + user_input + '"\n\n'
                'Write one short sentence for the examiner that starts with "You\'ve" and '
                "acknowledges what the learner just did. Write only the sentence, no extra text."
            )
            fallback = FALLBACK_PROMPTS[(self.turn - 1) % len(FALLBACK_PROMPTS)]

        try:
            raw = llm_generate(self.model, prompt, self.api_key, self.base_url).strip()

            # remove <think>…</think> blocks produced by reasoning models (DeepSeek R1, QWQ, etc.)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # find the first non-empty line
            first_line = ""
            for line in raw.splitlines():
                if line.strip():
                    first_line = line.strip()
                    break

            # strip markdown formatting characters the model sometimes adds (**, *, _)
            first_line = first_line.replace("**", "").replace("*", "").replace("_", "")

            # take only the first sentence
            completion = ""
            for i, char in enumerate(first_line):
                if char in ".!?":
                    completion = first_line[:i + 1].strip()
                    break
            if not completion:
                completion = first_line  # no sentence boundary found — use the whole line

            if completion:
                return completion + " What next?"
            return fallback
        except Exception as e:
            print("[narration error] " + str(e))
            return fallback

    def respond(self, user_input):
        self.user_inputs.append(user_input)  # record what the learner said
        self.turn += 1                        # advance the turn counter

        is_concluded = self.turn >= self.scenario["max_turns"]       # True on the last turn
        is_closing = self.turn == self.scenario["max_turns"] - 1     # True one turn before the end

        if is_concluded:
            narration = ""  # no examiner response on the final turn
        elif is_closing:
            narration = self._get_narration(user_input, closing=True)
        else:
            narration = self._get_narration(user_input)

        # save this exchange to the history log
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": narration})

        return narration, is_concluded  # caller uses is_concluded to know when to stop

    def transcript(self):
        # build the full conversation as a readable string for evaluation and reports
        role = self.scenario["user_role"].title()  # e.g. "driver" → "Driver"
        lines = ["Examiner: " + self.scenario["situation"] + "\n\nWhat do you do?"]

        for msg in self.history:
            if msg["role"] == "user":
                lines.append(role + ": " + msg["content"])
            else:
                lines.append("Examiner: " + msg["content"])

        return "\n\n".join(lines)  # blank line between each turn


# ─────────────────────────────────────────────────────────────────────────────
# SESSION
# Tracks all evaluations across one sitting (possibly multiple scenarios).
# Kept as a class because it accumulates results over time.
# ─────────────────────────────────────────────────────────────────────────────

class Session:

    def __init__(self, use_llm, model, api_key, base_url):
        self.use_llm = use_llm    # True if a valid API key is configured for LLM scoring
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.results = []  # list of (scenario, [evaluation_dict, ...]) tuples

    def evaluate(self, scenario, transcript):
        # score the transcript against every expert answer in the scenario
        evaluations = []
        for expert_answer in scenario["expert_answers"]:
            if self.use_llm:
                ev = score_with_llm(self.model, self.api_key, self.base_url, scenario, transcript, expert_answer)
            else:
                ev = score_with_keywords(scenario, transcript, expert_answer)
            evaluations.append(ev)

        self.results.append((scenario, evaluations))  # save for the summary and report
        return evaluations

    def average_score(self):
        # compute the mean score across all evaluations in this session
        all_scores = []
        for scenario, evals in self.results:
            for ev in evals:
                all_scores.append(ev["score"])

        if not all_scores:
            return 0.0

        return round(sum(all_scores) / len(all_scores), 4)

    def total_evaluations(self):
        # count how many evaluations have been done (one per expert_answer per scenario)
        count = 0
        for scenario, evals in self.results:
            count += len(evals)
        return count


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# Writes a Markdown file to the reports/ folder with the full session results.
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(session, model, api_key, base_url, output_dir, thinking_profile=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)  # create the folder if it doesn't exist

    # build a filename like "report_20260618_162039.md"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / ("report_" + timestamp + ".md")

    # build one summary line per evaluation to send to Ollama for the instructor narrative
    summaries = []
    for scenario, evals in session.results:
        for ev in evals:
            if ev["gaps"]:
                gap_text = ", ".join(ev["gaps"])
            else:
                gap_text = "none"
            summaries.append(
                "Scenario '" + scenario["title"] + "': "
                "score " + f"{ev['score']:.0%}" + ". "
                "Gaps: " + gap_text + "."
            )

    summary_prompt = (
        "A learner completed " + str(len(session.results)) + " scenario(s) "
        "with an average score of " + f"{session.average_score():.0%}" + ".\n\n"
        "Per-scenario results:\n"
        + "\n".join(summaries) + "\n\n"
        "Return this JSON:\n"
        "{\n"
        '  "overall_assessment": "<2-3 sentence summary of the learner\'s performance>",\n'
        '  "learning_gaps": [<specific concepts or skills the learner struggled with>],\n'
        '  "recommendations": [<concrete steps the instructor can take to address the gaps>]\n'
        "}"
    )

    summary_system = (
        "You are an expert instructional designer reviewing assessment results. "
        "Write clear, actionable recommendations for the instructor. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
    instructor = _extract_json(raw)
    if not instructor:
        instructor = {"overall_assessment": raw, "learning_gaps": [], "recommendations": []}

    # build the Markdown report line by line
    lines = []
    lines.append("# Performative Assessment — Instructor Report")
    lines.append("")
    lines.append("**Date:** " + datetime.now().strftime("%Y-%m-%d %H:%M") + "  ")
    lines.append("**Model:** " + model + "  ")
    lines.append("**Scenarios completed:** " + str(len(session.results)) + "  ")
    lines.append("**Average score:** " + f"{session.average_score():.0%}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (scenario, evals) in enumerate(session.results, 1):
        lines.append("## Scenario " + str(i) + ": " + scenario["title"])
        lines.append("")
        if scenario["description"]:
            lines.append("_" + scenario["description"] + "_")
            lines.append("")
        if scenario["constraints"]:
            lines.append("**Constraints:**")
            for c in scenario["constraints"]:
                lines.append("  - " + c)
            lines.append("")

        for ev in evals:
            lines.append("**Score:** " + f"{ev['score']:.0%}")
            lines.append("")
            lines.append("**Conversation transcript:**")
            lines.append("")
            for line in ev["transcript"].splitlines():
                if line:
                    lines.append("> " + line)  # indent each line as a Markdown blockquote
                else:
                    lines.append(">")           # keep blank lines inside the blockquote
            lines.append("")
            lines.append("**Expert guidance:**")
            lines.append("> " + ev["expert_answer"]["answer"])
            lines.append("")

            if ev["strengths"]:
                lines.append("**Strengths:**")
                for s in ev["strengths"]:
                    lines.append("  - " + s)
                lines.append("")
            if ev["gaps"]:
                lines.append("**Gaps:**")
                for g in ev["gaps"]:
                    lines.append("  - " + g)
                lines.append("")
            if ev["matched_points"]:
                lines.append("**Key points covered:** " + ", ".join(ev["matched_points"]))
                lines.append("")
            if ev["missed_points"]:
                lines.append("**Key points missed:** " + ", ".join(ev["missed_points"]))
                lines.append("")
            if ev["feedback"]:
                lines.append("**Feedback for learner:** _" + ev["feedback"] + "_")
                lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Instructor Summary")
    lines.append("")
    lines.append(instructor.get("overall_assessment", ""))
    lines.append("")

    if instructor.get("learning_gaps"):
        lines.append("**Learning gaps identified:**")
        for gap in instructor["learning_gaps"]:
            lines.append("  - " + gap)
        lines.append("")

    if instructor.get("recommendations"):
        lines.append("**Recommendations:**")
        for rec in instructor["recommendations"]:
            lines.append("  - " + rec)
        lines.append("")

    if thinking_profile and (thinking_profile.get("honey_mumford_style") or thinking_profile.get("solo_level")):
        lines.append("## Learner Thinking Profile")
        lines.append("")
        hm = thinking_profile.get("honey_mumford_style", "")
        hm_ev = thinking_profile.get("honey_mumford_evidence", "")
        if hm:
            lines.append("**Honey & Mumford style:** " + hm)
            if hm_ev:
                lines.append("_" + hm_ev + "_")
            lines.append("")
        solo = thinking_profile.get("solo_level", "")
        solo_ev = thinking_profile.get("solo_evidence", "")
        if solo:
            lines.append("**SOLO level:** " + solo)
            if solo_ev:
                lines.append("_" + solo_ev + "_")
            lines.append("")
        patterns = thinking_profile.get("observed_patterns", [])
        if patterns:
            lines.append("**Observed patterns:**")
            for p in patterns:
                lines.append("  - " + p)
            lines.append("")
        note = thinking_profile.get("instructor_note", "")
        if note:
            lines.append("**Instructor note:** _" + note + "_")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path  # return the path so the caller can tell the user where the file was saved
