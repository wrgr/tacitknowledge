"""
thinking.py — Honey & Mumford / SOLO taxonomy analysis of learner responses.

Looks at HOW the learner responded — sequencing, confidence, level of detail,
whether they anticipated issues — to classify their thinking and learning style.
Runs as a separate call after scoring so the score appears without delay.
Results accumulate across sessions to build a longitudinal profile.
"""

import re

from llm import llm_chat, _extract_json


def _strip_md(val):
    """Strip markdown emphasis and stray $ from LLM-generated strings."""
    if isinstance(val, list):
        return [_strip_md(v) for v in val]
    if not isinstance(val, str):
        return val
    # bold/italic: **x**, *x*, __x__, _x_
    val = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', val)
    val = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', val)
    # stray $ signs
    val = val.replace('$', '')
    return val


# ─────────────────────────────────────────────────────────────────────────────
# THINKING PROFILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_thinking_profile(scenario, transcript, model, api_key, base_url, prior_profiles=None):
    system = (
        "You are an educational psychologist. "
        "Classify a learner's response using two established frameworks. "
        "Base your analysis only on HOW they responded — language, sequencing, depth — not on their score. "
        "You must back every classification with direct evidence from the transcript. "
        "When the transcript is too short or ambiguous to classify confidently, say so explicitly "
        "and give your most probable interpretation with the evidence that led you there. "
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

        "## Evidence and reasoning requirements\n"
        "- honey_mumford_evidence: list 2-3 direct quotes or close paraphrases from the transcript "
        "that support the style classification. If the transcript is very short, list everything usable.\n"
        "- honey_mumford_reasoning: explain the chain of logic — WHY does each piece of evidence "
        "point to this style and not an adjacent one (e.g. why Theorist and not Pragmatist).\n"
        "- honey_mumford_confidence: 'high' if ≥2 distinct evidence signals; 'medium' if only one "
        "signal or if another style was plausible; 'low' if the transcript barely has enough to judge.\n"
        "- solo_evidence: list 2-3 specific moments from the transcript that reveal the learner's "
        "depth of understanding.\n"
        "- solo_reasoning: explain WHY these moments place the learner at this SOLO level and not "
        "the level above or below.\n"
        "- solo_confidence: same scale as above.\n"
        "- insufficient_data_note: if the transcript is too short or ambiguous for a confident "
        "classification in EITHER framework, describe exactly what is missing, state your most "
        "probable interpretation, and cite the specific transcript text that drove that guess. "
        "Set to null when evidence is sufficient for both frameworks.\n"
        "- observed_patterns: 2-3 entries; each must name the behaviour AND quote the transcript "
        "moment that illustrates it (format: '<behaviour>: \"<quote>\"').\n\n"

        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "honey_mumford_style":     "<Activist | Reflector | Theorist | Pragmatist>",\n'
        '  "honey_mumford_evidence":  [<2-3 direct quotes or close paraphrases from the transcript>],\n'
        '  "honey_mumford_reasoning": "<explanation of why this evidence points to this style>",\n'
        '  "honey_mumford_confidence":"<high | medium | low>",\n'
        '  "solo_level":              "<Prestructural | Unistructural | Multistructural | Relational | Extended Abstract>",\n'
        '  "solo_evidence":           [<2-3 specific transcript moments showing depth of understanding>],\n'
        '  "solo_reasoning":          "<explanation of why this evidence places the learner at this SOLO level>",\n'
        '  "solo_confidence":         "<high | medium | low>",\n'
        '  "insufficient_data_note":  null | "<what is missing, most probable interpretation, and supporting quote>",\n'
        '  "observed_patterns":       [<2-3 strings: \'<behaviour>: "<quote from transcript>"\'>],\n'
        '  "instructor_note":         "<one sentence on how to scaffold learning for this learner>"\n'
        "}"
    )

    raw    = llm_chat(model, system, prompt, api_key, base_url)
    result = _extract_json(raw)

    _prose_fields = (
        "honey_mumford_evidence", "honey_mumford_reasoning",
        "solo_evidence", "solo_reasoning",
        "insufficient_data_note", "observed_patterns", "instructor_note",
    )
    for field in _prose_fields:
        if field in result:
            result[field] = _strip_md(result[field])

    return result
