"""
thinking.py — Honey & Mumford / SOLO taxonomy analysis of learner responses.

Now uses separate recall and probe transcripts to detect probe_phase_improvement:
whether the learner demonstrated notably richer or more specific knowledge under
structured probing than in free recall — a meaningful signal of whether depth
exists but wasn't spontaneously surfaced.
"""

import re

from llm import (
    llm_chat_json, _extract_json, clip,
    EVALUATIVE_TEMPERATURE, EVALUATIVE_SEED, cached_evaluative_call,
)

# Bump whenever the classification prompt's wording changes, so old cached
# results (see llm.cached_evaluative_call) don't silently apply to a changed prompt.
_PROMPT_VERSION_THINKING_PROFILE = "thinking_profile_v1"

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
    unique  = set(words)
    hedging = sum(1 for w in words if w in _HEDGING_WORDS)
    return {
        'word_count':        len(words),
        'unique_word_ratio': round(len(unique) / len(words), 3),
        'avg_word_length':   round(sum(len(w) for w in words) / len(words), 1),
        'hedging_count':     hedging,
    }


def _format_process_section(writing_metrics, user_inputs):
    """Build a 'Writing Process Data' block for the LLM prompt."""
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
    val = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', val)
    val = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', val)
    val = val.replace('$', '')
    return val


# ─────────────────────────────────────────────────────────────────────────────
# THINKING PROFILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_thinking_profile(scenario, transcript, model, api_key, base_url,
                             prior_profiles=None, writing_metrics=None, user_inputs=None,
                             recall_transcript="", probe_transcript="", bypass_cache=False):
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

    # Build transcript section — show phases separately when available
    if recall_transcript and probe_transcript:
        transcript_section = (
            "RECALL TRANSCRIPT (use for H&M style — shows how the learner spontaneously "
            "organises and expresses knowledge without any prompting):\n"
            + clip(recall_transcript) + "\n\n"
            "PROBING TRANSCRIPT (Socratic dialogue — examiner asked WHY, WHAT WOULD HAPPEN, "
            "and HOW THE LEARNER DECIDED; use for SOLO level and as additional H&M evidence):\n"
            + clip(probe_transcript)
        )
        probe_comparison_note = (
            "\n## Using Both Transcripts for Classification\n"
            "The probing phase was Socratic — the examiner asked about reasoning, not missing facts. "
            "This means probe responses are direct evidence of thinking depth:\n\n"
            "H&M style signals in probe responses:\n"
            "- Conditional or hedged answers ('it depends...', 'usually, but...') → Reflector\n"
            "- Explains mechanisms, goals, or underlying principles ('the purpose is...', "
            "'because otherwise...') → Theorist\n"
            "- Short, action-focused answers with no elaboration → Activist or Pragmatist\n"
            "- 'What works in practice' framing without theory → Pragmatist\n\n"
            "SOLO level — probe responses are especially diagnostic here:\n"
            "- Relational: explains WHY steps connect, what consequences follow, conditional reasoning\n"
            "- Extended Abstract: raises edge cases or principles unprompted in their probe answers\n"
            "- Multistructural ceiling: even when asked WHY, gives another list instead of reasoning\n\n"
            "Use recall as the primary H&M signal (spontaneous style). "
            "Use probe responses as primary SOLO evidence (reasoning depth under direct questioning). "
            "Set probe_phase_improvement: true if reasoning in probe responses was notably richer "
            "than what the recall transcript alone would have suggested.\n"
        )
    else:
        transcript_section = "Transcript:\n" + clip(transcript)
        probe_comparison_note = ""

    prompt = (
        "Scenario: " + scenario["title"] + "\n\n"
        + transcript_section + "\n\n"
        + (process_section + "\n\n" if process_section else "")
        + probe_comparison_note
        + "## Framework 1 — Honey & Mumford Learning Style\n"
        "Choose exactly one:\n"
        "- Activist: dives straight in, action-first, minimal planning, energetic language\n"
        "- Reflector: considers options before acting, hedged language, weighs consequences; "
        "may give richer answers when specifically asked (more under probing than in recall)\n"
        "- Theorist: explains the reasoning and underlying rules, logical and sequential; "
        "likely to explain WHY steps matter when prompted by rationale/decision probes\n"
        "- Pragmatist: practical and direct, skips theory, focuses on what works\n\n"

        "## Framework 2 — SOLO Taxonomy (depth of understanding)\n"
        "Choose exactly one. Base this primarily on the PROBING transcript, since the Socratic "
        "questions directly test reasoning depth ('why?', 'what would happen?', 'how do you decide?'):\n"
        "- Prestructural: misses the point, irrelevant or no response to the task\n"
        "- Unistructural: identifies one relevant element, nothing more\n"
        "- Multistructural: covers several relevant elements but treats them in isolation; "
        "flat step list without integrating how they connect — when asked WHY, gives another "
        "list rather than reasoning\n"
        "- Relational: integrates elements coherently; when asked WHY, explains goals, "
        "consequences, or conditions — 'I do X because otherwise Y happens', 'it depends on Z'\n"
        "- Extended Abstract: generalises beyond the task; raises edge cases, principles, or "
        "contraindications unprompted — even in probe responses\n\n"
        "NOTE: A learner who lists many steps fluently in recall but cannot explain the reasoning "
        "behind them when probed is Multistructural, not Relational.\n\n"

        "## Evidence and reasoning requirements\n"
        "- honey_mumford_evidence: list 2-3 direct quotes or close paraphrases from the transcript\n"
        "- honey_mumford_reasoning: explain WHY the evidence points to this style and not an adjacent one\n"
        "- honey_mumford_confidence: 'high' if ≥2 distinct signals; 'medium' if only one or adjacent style plausible; 'low' if barely enough\n"
        "- solo_evidence: list 2-3 specific transcript moments showing depth\n"
        "- solo_reasoning: explain WHY these place the learner at this SOLO level, not above or below\n"
        "- solo_confidence: same scale\n"
        "- insufficient_data_note: describe what is missing if either framework can't be classified confidently; null otherwise\n"
        "- observed_patterns: 2-3 entries; each must name the behaviour AND quote the transcript moment "
        "or process metric that illustrates it (format: '<behaviour>: \"<quote or metric>\"')\n"
        "- probe_phase_improvement: boolean — true if the learner's answers were notably richer "
        "(more specific, more conditional, more goal-linked) in the probing phase than in free recall\n"
        "- probe_phase_improvement_note: one sentence explaining the evidence for your probe_phase_improvement "
        "judgement (or null if no probe phase data)\n\n"

        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "honey_mumford_style":            "<Activist | Reflector | Theorist | Pragmatist>",\n'
        '  "honey_mumford_evidence":         [<2-3 direct quotes or close paraphrases>],\n'
        '  "honey_mumford_reasoning":        "<explanation>",\n'
        '  "honey_mumford_confidence":       "<high | medium | low>",\n'
        '  "solo_level":                     "<Prestructural | Unistructural | Multistructural | Relational | Extended Abstract>",\n'
        '  "solo_evidence":                  [<2-3 specific transcript moments>],\n'
        '  "solo_reasoning":                 "<explanation>",\n'
        '  "solo_confidence":                "<high | medium | low>",\n'
        '  "insufficient_data_note":         null | "<what is missing and most probable interpretation>",\n'
        '  "observed_patterns":              [<2-3 strings: \'<behaviour>: "<quote>"\'>],\n'
        '  "probe_phase_improvement":        true | false,\n'
        '  "probe_phase_improvement_note":   "<one sentence>" | null,\n'
        '  "instructor_note":                "<one sentence on how to scaffold learning for this learner>"\n'
        "}"
    )

    def _call():
        raw = llm_chat_json(model, system, prompt, api_key, base_url,
                            temperature=EVALUATIVE_TEMPERATURE, seed=EVALUATIVE_SEED)
        return _extract_json(raw)
    result = cached_evaluative_call(model, base_url, _PROMPT_VERSION_THINKING_PROFILE,
                                    system, prompt, _call, bypass_cache=bypass_cache)

    _prose_fields = (
        "honey_mumford_evidence", "honey_mumford_reasoning",
        "solo_evidence", "solo_reasoning",
        "insufficient_data_note", "observed_patterns", "instructor_note",
        "probe_phase_improvement_note",
    )
    for field in _prose_fields:
        if field in result:
            result[field] = _strip_md(result[field])

    return result
