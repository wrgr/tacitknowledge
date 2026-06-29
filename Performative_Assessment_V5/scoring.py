"""
scoring.py — keyword and LLM-based scoring for scenarios and free-response prompts.

Scenario scoring now has two dimensions:
  1. Coverage Score  — which key points were addressed (existing model, extended to full transcript)
  2. Quality Score   — how deeply each matched point was explained (Chi's discriminators: 0/1/2)

Combined score = coverage_weight × Coverage + quality_weight × Quality
(weights come from scenario["scoring_weights"], default 0.6/0.4)
"""

import re

from llm import llm_chat, _extract_json


_STOP = frozenset({
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
    'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'as', 'but', 'not', 'this', 'that', 'which', 'it', 'its',
    'beyond', 'than', 'their', 'they',
})


def _phrase_in_text(phrase, text_lower):
    """True if the key phrase is covered in the text.

    Handles four things the naïve approach misses:
    - Parenthetical elaborations are stripped: "(taste/cool temperature)" is ignored
    - "/" at the top level means OR: any one alternative matching is enough
    - Stop words are filtered so "from/to/of" don't count as required words
    - Bidirectional prefix: text "decompress" matches keyword "decompressional",
      and keyword "loosen" matches text "loosened" (the original behaviour)
    Requires ≥ 60% of significant words in an alternative to match.
    """
    def _sig_words(text):
        t = re.sub(r'\([^)]*\)', ' ', text)
        tokens = re.split(r'[\s]+', t.lower())
        words = [re.sub(r'[^a-z]', '', tok) for tok in tokens]
        return [w for w in words if w and w not in _STOP and len(w) > 2]

    def _word_found(w, text_lower):
        if re.search(r'\b' + re.escape(w), text_lower):
            return True
        if len(w) >= 5:
            for m in re.finditer(r'\b([a-z]{5,})', text_lower):
                if w.startswith(m.group(1)):
                    return True
        return False

    phrase_no_parens = re.sub(r'\([^)]*\)', ' ', phrase)
    alternatives = phrase_no_parens.split('/')

    for alt in alternatives:
        words = _sig_words(alt)
        if not words:
            continue
        matched = sum(1 for w in words if _word_found(w, text_lower))
        if matched >= max(1, len(words) * 0.6):
            return True
    return False


def _resolve_llm_matches(matched_from_llm, key_points):
    """Map LLM-returned matched_points back to the canonical key_points list."""
    resolved = set()
    for m in matched_from_llm:
        m_lower = m.lower().strip()
        for p in key_points:
            p_lower = p.lower()
            if p_lower == m_lower:
                resolved.add(p)
                break
            if p_lower in m_lower or m_lower in p_lower:
                resolved.add(p)
                break
            p_words = set(p_lower.split())
            m_words = set(m_lower.split())
            if p_words and p_words.issubset(m_words):
                resolved.add(p)
                break
    return resolved


def _compute_coverage_score(matched, key_points, rubric):
    """Weighted coverage score from matched key points."""
    if rubric and key_points:
        total  = sum(rubric.get(p, 1) for p in key_points)
        earned = sum(rubric.get(p, 1) for p in key_points if p in matched)
        return earned / total if total > 0 else 0.0
    elif key_points:
        return len([p for p in key_points if p in matched]) / len(key_points)
    return 0.5


def _compute_quality_score(quality_ratings, matched):
    """Mean quality rating across matched points, normalized to [0, 1]."""
    if not matched:
        return 0.0
    ratings = [quality_ratings.get(p, 0) for p in matched]
    if not ratings:
        return 0.0
    return sum(ratings) / (len(ratings) * 2)  # max rating is 2


def _compute_combined_score(coverage_score, quality_score, scoring_weights):
    cw = scoring_weights.get("coverage", 0.6)
    qw = scoring_weights.get("quality", 0.4)
    # normalize weights
    total = cw + qw
    if total > 0:
        cw, qw = cw / total, qw / total
    return max(0.0, min(1.0, cw * coverage_score + qw * quality_score))


def _determine_point_sources(recall_text, probe_text, matched_set):
    """Determine whether each matched point appeared in recall or only in probing."""
    sources = {}
    recall_lower = recall_text.lower() if recall_text else ""
    for p in matched_set:
        sources[p] = "recall" if _phrase_in_text(p, recall_lower) else "probe"
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# FREE-RESPONSE SCORING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def check_fr_keywords(prompt_data, text):
    """Fast keyword check for live sidebar — no LLM, returns matched/missed/word_count."""
    text_lower = text.lower()
    ea = prompt_data["expert_answers"][0] if prompt_data.get("expert_answers") else {}
    key_points = ea.get("key_points", [])
    matched = [p for p in key_points if _phrase_in_text(p, text_lower)]
    missed  = [p for p in key_points if not _phrase_in_text(p, text_lower)]
    word_count = len(text.split()) if text.strip() else 0
    return {"matched_points": matched, "missed_points": missed, "word_count": word_count}


def check_fr_with_llm(model, api_key, base_url, prompt_data, text):
    """LLM-based key point check for live sidebar — semantic matching, no scoring/feedback."""
    ea = prompt_data["expert_answers"][0] if prompt_data.get("expert_answers") else {}
    key_points = ea.get("key_points", [])

    if not key_points or not text.strip():
        return {"matched_points": [], "missed_points": list(key_points)}

    kp_text = "\n".join("- " + p for p in key_points)

    prompt = (
        "KEY POINTS:\n" + kp_text + "\n\n"
        "LEARNER'S TEXT (work in progress — may be incomplete):\n" + text + "\n\n"
        "For each key point, decide if the learner has clearly addressed it. "
        "Semantic equivalence counts — exact wording is not required.\n"
        "Return ONLY this JSON (no markdown, no extra text):\n"
        '{"matched_points": [<copy exact key point strings addressed so far>], '
        '"missed_points": [<copy exact key point strings not yet addressed>]}'
    )

    system = (
        "You check a learner's draft against key points. "
        "Credit semantic equivalence — exact wording not required. "
        "Respond with valid JSON only."
    )

    raw    = llm_chat(model, system, prompt, api_key, base_url)
    result = _extract_json(raw)

    matched     = result.get("matched_points", [])
    matched_set = _resolve_llm_matches(matched, key_points)

    return {
        "matched_points": [p for p in key_points if p in matched_set],
        "missed_points":  [p for p in key_points if p not in matched_set],
    }


def score_free_response_with_keywords(prompt_data, text):
    ea = prompt_data["expert_answers"][0] if prompt_data.get("expert_answers") else {}
    key_points = ea.get("key_points", [])
    rubric     = ea.get("rubric", {})
    text_lower = text.lower()

    matched = [p for p in key_points if _phrase_in_text(p, text_lower)]
    missed  = [p for p in key_points if not _phrase_in_text(p, text_lower)]

    if rubric:
        total  = sum(rubric.values())
        earned = sum(rubric.get(p, 1.0) for p in matched)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        score = len(matched) / len(key_points)
    else:
        score = 0.0

    parts = []
    if matched:
        parts.append("Covered: " + ", ".join(matched))
    if missed:
        parts.append("Missing: " + ", ".join(missed))

    return {
        "prompt_id":      prompt_data["id"],
        "text":           text,
        "expert_answer":  ea,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          score,
        "feedback":       "\n".join(parts) if parts else "No key points defined.",
        "strengths":      [],
        "gaps":           [],
    }


def score_free_response_with_llm(model, api_key, base_url, prompt_data, text):
    ea         = prompt_data["expert_answers"][0] if prompt_data.get("expert_answers") else {}
    key_points = ea.get("key_points", [])
    rubric     = ea.get("rubric", {})

    _weight_label = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}
    if key_points:
        kp_lines = ["- " + p + " [" + _weight_label.get(rubric.get(p, 1), "MEDIUM") + " — " + str(rubric.get(p, 1)) + "pt" + ("s" if rubric.get(p, 1) != 1 else "") + "]"
                    for p in key_points]
        key_points_text = "\n".join(kp_lines)
    else:
        key_points_text = "- (none specified)"

    constraints_text = (
        "\n".join("- " + c for c in prompt_data["constraints"])
        if prompt_data.get("constraints") else "- (none specified)"
    )

    prompt = (
        "WRITING TASK: " + prompt_data.get("prompt_text", prompt_data.get("description", "")) + "\n\n"
        "CONSTRAINTS:\n" + constraints_text + "\n\n"
        "EXPERT REFERENCE ANSWER:\n" + ea.get("answer", "(none)") + "\n\n"
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n" + key_points_text + "\n\n"
        "LEARNER'S SUBMISSION:\n" + text + "\n\n"
        "GRADING INSTRUCTIONS:\n"
        "- Determine which key points the learner clearly addressed in their written response.\n"
        "- Credit semantic equivalence — the learner does not need to use the exact phrasing.\n"
        "- Do NOT give credit for vague, off-topic, or entirely absent coverage.\n"
        "- Feedback must be direct and specific to the written submission.\n\n"
        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "matched_points": [<copy the exact key point strings the learner clearly addressed>],\n'
        '  "missed_points":  [<copy the exact key point strings omitted or inadequate>],\n'
        '  "strengths": [<1-3 specific things done well>],\n'
        '  "gaps": [<one sentence per gap explaining what was missing and why it matters>],\n'
        '  "feedback": "<2-3 sentences of direct feedback for the learner>"\n'
        "}"
    )

    system = (
        "You are an impartial grader evaluating a written response. "
        "Credit what is clearly demonstrated — exact wording is not required, semantic equivalence counts. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw    = llm_chat(model, system, prompt, api_key, base_url)
    result = _extract_json(raw)

    matched = result.get("matched_points", [])
    missed  = result.get("missed_points",  [])

    matched_set = _resolve_llm_matches(matched, key_points)

    if rubric and key_points:
        total  = sum(rubric.get(p, 1) for p in key_points)
        earned = sum(rubric.get(p, 1) for p in key_points if p in matched_set)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        score = len([p for p in key_points if p in matched_set]) / len(key_points)
    else:
        score = 0.5

    matched = [p for p in key_points if p in matched_set]
    missed  = [p for p in key_points if p not in matched_set]

    return {
        "prompt_id":      prompt_data["id"],
        "text":           text,
        "expert_answer":  ea,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          max(0.0, min(1.0, score)),
        "feedback":       result.get("feedback", ""),
        "strengths":      result.get("strengths", []),
        "gaps":           result.get("gaps", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SCORING  (two-dimensional)
# ─────────────────────────────────────────────────────────────────────────────

def score_with_keywords(scenario, transcript, expert_answer,
                        recall_transcript="", probe_transcript=""):
    """Keyword scoring — no quality dimension (fallback path)."""
    text = transcript.lower()

    matched = []
    missed  = []
    for point in expert_answer["key_points"]:
        if _phrase_in_text(point, text):
            matched.append(point)
        else:
            missed.append(point)

    rubric   = expert_answer["rubric"]
    coverage = _compute_coverage_score(set(matched), expert_answer["key_points"], rubric)
    weights  = scenario.get("scoring_weights", {"coverage": 0.6, "quality": 0.4})
    combined = _compute_combined_score(coverage, 0.0, weights)

    sources = _determine_point_sources(recall_transcript, probe_transcript, set(matched))

    feedback_parts = []
    if matched:
        feedback_parts.append("Covered : " + ", ".join(matched))
    if missed:
        feedback_parts.append("Missing : " + ", ".join(missed))

    return {
        "scenario_id":      scenario["id"],
        "transcript":       transcript,
        "recall_transcript": recall_transcript,
        "probe_transcript":  probe_transcript,
        "expert_answer":    expert_answer,
        "matched_points":   matched,
        "missed_points":    missed,
        "quality_ratings":  {p: 0 for p in matched},  # no quality dimension in keyword mode
        "point_sources":    sources,
        "coverage_score":   round(coverage, 4),
        "quality_score":    0.0,
        "score":            round(combined, 4),
        "feedback":         "\n".join(feedback_parts) if feedback_parts else "No key points defined.",
        "strengths":        [],
        "gaps":             [],
    }


def score_with_llm(model, api_key, base_url, scenario, transcript, expert_answer,
                   recall_transcript="", probe_transcript=""):
    """LLM scoring with two-dimensional output: coverage + explanation quality."""
    key_points = expert_answer.get("key_points", [])
    rubric     = expert_answer.get("rubric", {})

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

    # Show recall and probe transcripts separately when available
    transcript_section = ""
    if recall_transcript and probe_transcript:
        transcript_section = (
            "RECALL TRANSCRIPT (free recall — what the learner volunteered unprompted):\n"
            + recall_transcript + "\n\n"
            "PROBING TRANSCRIPT (responses to structured probes):\n"
            + probe_transcript
        )
    else:
        transcript_section = "ASSESSMENT TRANSCRIPT:\n" + transcript

    prompt = (
        "SCENARIO: " + scenario["description"] + "\n\n"
        "CONSTRAINTS (the response must satisfy every one of these):\n"
        + constraints_text + "\n\n"
        "EXPERT ANSWER (the complete ideal response — use this as your benchmark):\n"
        + expert_answer["answer"] + "\n\n"
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n"
        + key_points_text + "\n\n"
        + transcript_section + "\n\n"
        "GRADING INSTRUCTIONS:\n"
        "1. COVERAGE: Identify which key points the learner demonstrated across the full transcript "
        "(recall + probing combined). Credit semantic equivalence.\n"
        "2. EXPLANATION QUALITY: For each matched key point, rate the quality of the learner's "
        "explanation on a 0-2 scale:\n"
        "   0 = stated/named only (flat recitation, e.g. 'I would loosen the lug nuts')\n"
        "   1 = partial explanation (ONE of: conditional reasoning, goal-linked statement, "
        "or consequence awareness)\n"
        "   2 = full explanation (TWO OR MORE of: 'you do X because...', 'the purpose of X is...', "
        "'if you skip X then...')\n"
        "IMPORTANT: Fluency, confidence, and prose quality are NOT evidence of quality. "
        "Only conditional, goal-linked, and consequence-aware language counts.\n"
        "Do NOT give credit for vague references or enthusiasm without substance.\n\n"
        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "matched_points": [<copy the exact key point strings the learner clearly demonstrated>],\n'
        '  "missed_points":  [<copy the exact key point strings the learner omitted or got wrong>],\n'
        '  "quality_ratings": {<"key point string": 0|1|2, for each matched point>},\n'
        '  "quality_evidence": {<"key point string": "<quote that justifies the quality rating>">},\n'
        '  "strengths": [<1-3 specific things done correctly, as short phrases>],\n'
        '  "gaps": [<one sentence per gap explaining what was missing and why it matters>],\n'
        '  "feedback": "<2-3 sentences of direct, honest feedback for the learner>"\n'
        "}"
    )

    system = (
        "You are an impartial grader for a performative assessment. "
        "Your role is to evaluate performance against explicit criteria. "
        "Credit what is clearly demonstrated — exact wording is not required, semantic equivalence counts. "
        "Be critical: missing a CRITICAL or HIGH point is a significant failure and must be named. "
        "Quality scores must reflect genuine conditional/goal/consequence reasoning — NOT fluency or length. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw    = llm_chat(model, system, prompt, api_key, base_url)
    result = _extract_json(raw)

    matched = result.get("matched_points", [])
    missed  = result.get("missed_points",  [])

    matched_set = _resolve_llm_matches(matched, key_points)

    # arithmetic stays in Python — LLM only does semantic matching
    weights  = scenario.get("scoring_weights", {"coverage": 0.6, "quality": 0.4})
    coverage = _compute_coverage_score(matched_set, key_points, rubric)

    # quality ratings: resolve LLM keys back to canonical key_points
    raw_ratings = result.get("quality_ratings", {})
    quality_ratings = {}
    for llm_key, rating in raw_ratings.items():
        for p in matched_set:
            if p.lower() in llm_key.lower() or llm_key.lower() in p.lower():
                quality_ratings[p] = int(rating) if str(rating).isdigit() else 0
                break

    # fill missing quality ratings with 0
    for p in matched_set:
        if p not in quality_ratings:
            quality_ratings[p] = 0

    quality  = _compute_quality_score(quality_ratings, matched_set)
    combined = _compute_combined_score(coverage, quality, weights)

    sources = _determine_point_sources(recall_transcript, probe_transcript, matched_set)

    # Rebuild matched/missed from resolved set
    matched = [p for p in key_points if p in matched_set]
    missed  = [p for p in key_points if p not in matched_set]

    return {
        "scenario_id":      scenario["id"],
        "transcript":       transcript,
        "recall_transcript": recall_transcript,
        "probe_transcript":  probe_transcript,
        "expert_answer":    expert_answer,
        "matched_points":   matched,
        "missed_points":    missed,
        "quality_ratings":  quality_ratings,
        "quality_evidence": result.get("quality_evidence", {}),
        "point_sources":    sources,
        "coverage_score":   round(coverage, 4),
        "quality_score":    round(quality, 4),
        "score":            max(0.0, min(1.0, combined)),
        "feedback":         result.get("feedback", ""),
        "strengths":        result.get("strengths", []),
        "gaps":             result.get("gaps", []),
    }
