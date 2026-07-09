"""
thinking.py — SOLO taxonomy analysis of learner responses.

analyse_thinking_profile() below is scenario mode's holistic LLM classifier: LLM-judged
SOLO level and probe_phase_improvement (whether the learner demonstrated notably richer
or more specific knowledge under structured probing than in free recall). It is
scenario-mode-only -- FR does not have a probe phase, and per the FR thinking-profile
fix, does not use this function at all.

derive_fr_solo_level() below is FR's replacement: a deterministic, LLM-free SOLO
derivation from data the FR scoring pipeline already computed. See its docstring for
why FR dropped the LLM SOLO judgment in favour of this.
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
        "  - High revision ratio (>30%) → active self-monitoring or uncertainty\n"
        "  - Rich hedging language → cautious phrasing or domain uncertainty\n"
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
        "Classify a learner's response using an established framework. "
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
            "RECALL TRANSCRIPT (shows how the learner spontaneously organises and "
            "expresses knowledge without any prompting; use as supporting context):\n"
            + clip(recall_transcript) + "\n\n"
            "PROBING TRANSCRIPT (Socratic dialogue — examiner asked WHY, WHAT WOULD HAPPEN, "
            "and HOW THE LEARNER DECIDED; use for SOLO level):\n"
            + clip(probe_transcript)
        )
        probe_comparison_note = (
            "\n## Using Both Transcripts for Classification\n"
            "The probing phase was Socratic — the examiner asked about reasoning, not missing facts. "
            "This means probe responses are direct evidence of thinking depth:\n\n"
            "SOLO level — probe responses are especially diagnostic here:\n"
            "- Relational: explains WHY steps connect, what consequences follow, conditional reasoning\n"
            "- Extended Abstract: raises edge cases or principles unprompted in their probe answers\n"
            "- Multistructural ceiling: even when asked WHY, gives another list instead of reasoning\n\n"
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
        + "## SOLO Taxonomy (depth of understanding)\n"
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
        "solo_evidence", "solo_reasoning",
        "insufficient_data_note", "observed_patterns", "instructor_note",
        "probe_phase_improvement_note",
    )
    for field in _prose_fields:
        if field in result:
            result[field] = _strip_md(result[field])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FR SOLO LEVEL — deterministic derivation (no LLM call)
# ─────────────────────────────────────────────────────────────────────────────
#
# FR thinking-profile fix: for FR specifically, the holistic LLM-judged SOLO level
# above is dropped in favour of this function. The LLM SOLO judgment duplicates, and
# can silently contradict, what Explanation Quality already measures via Chi's
# conditional/goal-linked/consequence-aware markers, with no evidence-span grounding
# or reconciliation against the Quality score it substantially overlaps with.
#
# This function replaces both for FR only, reading data the FR scoring pipeline has
# already computed and already grounded -- matched_points (standalone key points and
# credited pool members alike), each carrying a 0/1/2 quality_rating. Scenario mode
# continues to call analyse_thinking_profile() above, completely unchanged.
#
# SOLO's top level, Extended Abstract (generalising beyond the given task into new,
# hypothetical, or self-generated territory), is categorically outside what
# Coverage/Quality can capture -- both are scoped strictly to whether the *authored*
# key points were addressed within the *given* task. This function must never return
# "Extended Abstract"; that is a known, accepted scope limitation, not an oversight,
# and no LLM call is added to try to detect it either (that would reintroduce the
# exact ungrounded-holistic-judgment problem this function exists to remove).

# TUNABLE -- mean-quality cut point (0-2 scale) separating Multistructural from
# Relational; adjust after reviewing real FR submissions.
SOLO_RELATIONAL_QUALITY_THRESHOLD = 1.0


def derive_fr_solo_level(evaluation):
    """Deterministically derive a SOLO level for an FR evaluation.

    Returns matched_count and mean_quality alongside the level -- the actual inputs
    the rule used -- rather than an invented LLM-style confidence rating, since a
    deterministic rule over already-verified inputs doesn't need one.
    """
    matched = [m for m in (evaluation.get("matched_points") or []) if isinstance(m, dict)]
    matched_count = len(matched)
    mean_quality = (
        sum(m.get("quality_rating", 0) for m in matched) / matched_count
        if matched_count else 0.0
    )

    if matched_count == 0:
        level = "Prestructural"
    elif matched_count == 1:
        level = "Unistructural"
    elif mean_quality >= SOLO_RELATIONAL_QUALITY_THRESHOLD:
        level = "Relational"
    else:
        level = "Multistructural"

    return {
        "solo_level":    level,
        "matched_count": matched_count,
        "mean_quality":  round(mean_quality, 2),
    }
