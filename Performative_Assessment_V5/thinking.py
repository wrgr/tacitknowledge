"""
thinking.py — Honey & Mumford / SOLO taxonomy analysis of learner responses.

Looks at HOW the learner responded — sequencing, confidence, level of detail,
whether they anticipated issues — to classify their thinking and learning style.
Also incorporates writing process metrics (typing speed, deletions, word choice,
pauses) collected by the frontend to enrich the behavioural analysis.
Runs as a separate call after scoring so the score appears without delay.
Results accumulate across sessions to build a longitudinal profile.
"""

import re

from llm import llm_chat_json, _extract_json

# ─────────────────────────────────────────────────────────────────────────────
# WORD-CHOICE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

_HEDGING_WORDS = frozenset({
    'maybe', 'might', 'perhaps', 'possibly', 'could', 'seems', 'appear',
    'appears', 'think', 'believe', 'guess', 'probably', 'likely', 'somewhat',
    'sort', 'kind', 'unsure', 'uncertain', 'not sure', 'assume', 'suppose',
})


def _word_choice_metrics(text):
    """Return vocabulary richness and hedging statistics for a piece of text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return {}
    unique     = set(words)
    hedging    = sum(1 for w in words if w in _HEDGING_WORDS)
    return {
        'word_count':        len(words),
        'unique_word_ratio': round(len(unique) / len(words), 3),
        'avg_word_length':   round(sum(len(w) for w in words) / len(words), 1),
        'hedging_count':     hedging,
    }


def _format_process_section(writing_metrics, user_inputs):
    """Build a 'Writing Process Data' block for the LLM prompt.

    writing_metrics — list of per-turn dicts from the frontend tracker
    user_inputs     — list of final submitted strings, one per turn
    Both lists are aligned by index; missing entries are treated as empty.
    """
    if not writing_metrics and not user_inputs:
        return ""

    inputs  = user_inputs  or []
    metrics = writing_metrics or []
    n       = max(len(inputs), len(metrics))
    if n == 0:
        return ""

    lines = ["## Writing Process Data (behavioural signals — use to inform classification)"]

    for i in range(n):
        text = inputs[i]  if i < len(inputs)  else ""
        m    = metrics[i] if i < len(metrics) else {}
        if not m and not text:
            continue

        label = f"Turn {i + 1}" if n > 1 else "Submission"
        lines.append(f"\n{label}:")

        if m:
            if m.get("latency_s") is not None:
                lines.append(f"  - First-keystroke latency: {m['latency_s']}s after examiner message")
            if m.get("wpm") is not None:
                lines.append(f"  - Active typing speed: {m['wpm']} WPM (pauses >3 s excluded)")
            if m.get("deletion_count") is not None:
                pct = round((m.get("revision_ratio") or 0) * 100)
                lines.append(f"  - Deletions: {m['deletion_count']} keystrokes ({pct}% of total — revision ratio)")
            if m.get("paste_count"):
                lines.append(f"  - Copy-paste events: {m['paste_count']}")
            if m.get("pause_count"):
                lines.append(f"  - Mid-response pauses (>3 s): {m['pause_count']} (longest: {m.get('max_pause_s', 0)}s)")
            if m.get("total_time_s") is not None:
                lines.append(f"  - Total composition time: {m['total_time_s']}s")

        wc = _word_choice_metrics(text) if text else {}
        if wc:
            lines.append(f"  - Word count: {wc['word_count']}, unique-word ratio: {round(wc['unique_word_ratio']*100)}%")
            lines.append(f"  - Avg word length: {wc['avg_word_length']} chars (vocabulary sophistication proxy)")
            if wc['hedging_count']:
                lines.append(f"  - Hedging language: {wc['hedging_count']} instances (e.g. 'might', 'could', 'perhaps')")

    lines.append(
        "\nInterpretation guide (supporting signals only — always ground classification in the transcript):\n"
        "  - Low latency + high WPM → Activist tendency (acts before reflecting)\n"
        "  - High latency + pauses → Reflector tendency (thinks before committing)\n"
        "  - High revision ratio (>30%) → active self-monitoring or uncertainty\n"
        "  - Rich hedging language → possible Reflector or Theorist, or domain uncertainty\n"
        "  - Low unique-word ratio → narrow vocabulary or tightly focused reasoning\n"
        "  - Paste events → text may not reflect real-time thinking; flag in observed_patterns"
    )

    return "\n".join(lines)


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

def analyse_thinking_profile(scenario, transcript, model, api_key, base_url,
                             prior_profiles=None, writing_metrics=None, user_inputs=None):
    system = (
        "You are an educational psychologist. "
        "Classify a learner's response using two established frameworks. "
        "Base your analysis on HOW they responded — language, sequencing, depth — not on their score. "
        "When writing process data is provided, treat it as supporting behavioural evidence: "
        "hesitation, heavy revision, rapid typing, and hedging language are all interpretable signals. "
        "You must back every classification with direct evidence from the transcript. "
        "When the transcript is too short or ambiguous to classify confidently, say so explicitly "
        "and give your most probable interpretation with the evidence that led you there. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    process_section = _format_process_section(writing_metrics, user_inputs)

    prompt = (
        "Scenario: " + scenario["title"] + "\n\n"
        "Transcript:\n" + transcript + "\n\n"
        + (process_section + "\n\n" if process_section else "")
        + "## Framework 1 — Honey & Mumford Learning Style\n"
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
        "- observed_patterns: 2-3 entries; each must name the behaviour AND quote the transcript moment "
        "or process metric that illustrates it (format: '<behaviour>: \"<quote or metric>\"'). "
        "Writing process signals (e.g. heavy deletion, copy-paste, long pauses) are valid patterns "
        "when they are interpretable and relevant to the learning style classification.\n\n"

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

    raw    = llm_chat_json(model, system, prompt, api_key, base_url)
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
