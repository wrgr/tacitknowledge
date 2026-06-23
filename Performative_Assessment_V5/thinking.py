"""
thinking.py — Honey & Mumford / SOLO taxonomy analysis of learner responses.

Looks at HOW the learner responded — sequencing, confidence, level of detail,
whether they anticipated issues — to classify their thinking and learning style.
Runs as a separate call after scoring so the score appears without delay.
Results accumulate across sessions to build a longitudinal profile.
"""

from llm import llm_chat, _extract_json


# ─────────────────────────────────────────────────────────────────────────────
# THINKING PROFILE ANALYSIS
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
