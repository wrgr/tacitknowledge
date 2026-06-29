"""
scoring.py — keyword and LLM-based scoring for scenarios and free-response prompts.

Scenario scoring now has two dimensions:
  1. Coverage Score  — which key points were addressed (existing model, extended to full transcript)
  2. Quality Score   — how deeply each matched point was explained (Chi's discriminators: 0/1/2)

Combined score = coverage_weight × Coverage + quality_weight × Quality
(weights come from scenario["scoring_weights"], default 0.6/0.4)
"""

import re

from llm import llm_chat, llm_chat_json, _extract_json, clip


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
# EVIDENCE EXTRACTION  (pre-pass compression before scoring)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_evidence(model, api_key, base_url, text, key_points):
    """Compress a transcript or submission to one evidence sentence per key point.

    Produces a compact block like:
        • loosen lug nuts: said "crack them first so they don't spin when jacked"
        • check torque: not addressed

    The scoring call then works from this summary instead of the full text,
    keeping the scoring prompt short regardless of how long the original was.
    Falls back to a hard clip if the LLM call fails.
    """
    if not key_points or not text.strip():
        return text

    kp_text = "\n".join("- " + p for p in key_points)
    prompt = (
        "KEY POINTS:\n" + kp_text + "\n\n"
        "TEXT:\n" + clip(text, 6000) + "\n\n"
        "For each key point, write ONE concise sentence summarising the most relevant thing "
        "the person said — use their own words where possible. "
        "If the key point was not addressed at all, write 'not addressed'.\n"
        "Format each line exactly as: '• <key point>: <sentence>'\n"
        "No extra text. One line per key point."
    )
    system = (
        "You extract concise per-topic evidence from text. "
        "One sentence per topic, using the person's own words. No extra commentary."
    )
    try:
        return llm_chat(model, system, prompt, api_key, base_url).strip() or clip(text, 3000)
    except Exception:
        return clip(text, 3000)


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

    raw    = llm_chat_json(model, system, prompt, api_key, base_url)
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

    # Pre-pass: compress the submission to per-key-point evidence before scoring.
    evidence = _extract_evidence(model, api_key, base_url, text, key_points)

    prompt = (
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n" + key_points_text + "\n\n"
        "LEARNER EVIDENCE (extracted from submission):\n" + evidence + "\n\n"
        "Grade the evidence against the key points. Credit semantic equivalence.\n\n"
        "Return this JSON — no markdown, no extra text:\n"
        '{"matched_points":[<exact key point strings addressed>],'
        '"missed_points":[<exact key point strings omitted>],'
        '"strengths":[<1-3 short phrases>],'
        '"gaps":[<one sentence per gap>],'
        '"feedback":"<2-3 sentences>"}'
    )

    system = (
        "You are an impartial grader. Score the learner's evidence against the key points. "
        "Credit semantic equivalence. Respond only with valid JSON."
    )

    raw    = llm_chat_json(model, system, prompt, api_key, base_url)
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

    # Pre-pass: compress each transcript to per-key-point evidence sentences.
    # The scoring prompt then works from this compact summary rather than the raw text,
    # keeping input size predictable regardless of how long the transcript was.
    if recall_transcript and probe_transcript:
        recall_evidence = _extract_evidence(model, api_key, base_url, recall_transcript, key_points)
        probe_evidence  = _extract_evidence(model, api_key, base_url, probe_transcript,  key_points)
        evidence_section = (
            "RECALL EVIDENCE (what the learner volunteered unprompted):\n" + recall_evidence + "\n\n"
            "PROBE EVIDENCE (what the learner said when questioned on their reasoning):\n" + probe_evidence
        )
    else:
        evidence_section = "LEARNER EVIDENCE:\n" + _extract_evidence(
            model, api_key, base_url, transcript, key_points
        )

    prompt = (
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n"
        + key_points_text + "\n\n"
        + evidence_section + "\n\n"
        "QUALITY SCALE (per matched point):\n"
        "  0 = stated only, no reasoning\n"
        "  1 = partial — ONE of: conditional, goal-linked, or consequence-aware\n"
        "  2 = full — TWO OR MORE of the above\n"
        "Fluency and length are NOT quality. Credit semantic equivalence for coverage.\n\n"
        "Return this JSON — no markdown, no extra text:\n"
        '{"matched_points":[<exact key point strings demonstrated>],'
        '"missed_points":[<exact key point strings omitted>],'
        '"quality_ratings":{"<key point>":0|1|2},'
        '"strengths":[<1-3 short phrases>],'
        '"gaps":[<one sentence per gap>],'
        '"feedback":"<2-3 sentences>"}'
    )

    system = (
        "You are an impartial grader. Score the learner's evidence against the key points. "
        "Credit semantic equivalence. Quality scores must reflect conditional/goal/consequence "
        "reasoning — not fluency. Respond only with valid JSON."
    )

    raw    = llm_chat_json(model, system, prompt, api_key, base_url)
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
        "quality_evidence": {},
        "point_sources":    sources,
        "coverage_score":   round(coverage, 4),
        "quality_score":    round(quality, 4),
        "score":            max(0.0, min(1.0, combined)),
        "feedback":         result.get("feedback", ""),
        "strengths":        result.get("strengths", []),
        "gaps":             result.get("gaps", []),
    }


def merge_phase_scores(recall_ev, probe_ev, scenario, expert_answer, full_transcript=""):
    """Combine per-phase scoring results into a single evaluation dict.

    recall_ev and probe_ev are the outputs of score_with_llm / score_with_keywords
    called with each phase transcript individually.  The combined score is derived
    in Python from the union of matched points — no extra LLM call needed.
    """
    key_points = expert_answer.get("key_points", [])
    rubric     = expert_answer.get("rubric", {})
    weights    = scenario.get("scoring_weights", {"coverage": 0.6, "quality": 0.4})

    recall_matched = set(recall_ev["matched_points"])
    probe_matched  = set(probe_ev["matched_points"])
    combined_matched = recall_matched | probe_matched

    # Take the highest quality rating demonstrated across either phase
    combined_quality = {}
    for p in combined_matched:
        r = recall_ev.get("quality_ratings", {}).get(p, 0)
        pr = probe_ev.get("quality_ratings", {}).get(p, 0)
        combined_quality[p] = max(r, pr)

    coverage = _compute_coverage_score(combined_matched, key_points, rubric)
    quality  = _compute_quality_score(combined_quality, combined_matched)
    combined = _compute_combined_score(coverage, quality, weights)

    matched = [p for p in key_points if p in combined_matched]
    missed  = [p for p in key_points if p not in combined_matched]

    # A point is credited to recall if shown there; otherwise it came from probing
    point_sources = {p: ("recall" if p in recall_matched else "probe") for p in combined_matched}

    return {
        "scenario_id":       scenario["id"],
        "transcript":        full_transcript or recall_ev["transcript"],
        "recall_transcript": recall_ev["transcript"],
        "probe_transcript":  probe_ev["transcript"],
        "expert_answer":     expert_answer,
        # Combined
        "matched_points":    matched,
        "missed_points":     missed,
        "quality_ratings":   combined_quality,
        "quality_evidence":  {**recall_ev.get("quality_evidence", {}), **probe_ev.get("quality_evidence", {})},
        "point_sources":     point_sources,
        "coverage_score":    round(coverage, 4),
        "quality_score":     round(quality, 4),
        "score":             max(0.0, min(1.0, combined)),
        "feedback":          probe_ev.get("feedback") or recall_ev.get("feedback", ""),
        "strengths":         probe_ev.get("strengths") or recall_ev.get("strengths", []),
        "gaps":              probe_ev.get("gaps") or recall_ev.get("gaps", []),
        # Per-phase
        "recall_score":           recall_ev["score"],
        "recall_coverage_score":  recall_ev["coverage_score"],
        "recall_quality_score":   recall_ev["quality_score"],
        "recall_matched_points":  recall_ev["matched_points"],
        "recall_missed_points":   recall_ev["missed_points"],
        "recall_quality_ratings": recall_ev.get("quality_ratings", {}),
        "probe_score":            probe_ev["score"],
        "probe_coverage_score":   probe_ev["coverage_score"],
        "probe_quality_score":    probe_ev["quality_score"],
        "probe_matched_points":   probe_ev["matched_points"],
        "probe_missed_points":    probe_ev["missed_points"],
        "probe_quality_ratings":  probe_ev.get("quality_ratings", {}),
    }
