"""
scoring.py — keyword and LLM-based scoring for scenarios and free-response prompts.

Two ways to score each type:
  keyword scoring — simple text matching, no LLM needed
  LLM scoring     — asks the LLM to read and judge the response
Both return a plain dict with the same keys so the rest of the code works either way.
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
        t = re.sub(r'\([^)]*\)', ' ', text)              # drop (...) elaborations
        tokens = re.split(r'[\s]+', t.lower())
        words = [re.sub(r'[^a-z]', '', tok) for tok in tokens]
        return [w for w in words if w and w not in _STOP and len(w) > 2]

    def _word_found(w, text_lower):
        # forward: \bword matches "wording", "loosened", etc.
        if re.search(r'\b' + re.escape(w), text_lower):
            return True
        # reverse: a text word (≥ 5 chars) is a prefix of the keyword
        # so "decompress" in text matches keyword "decompressional"
        if len(w) >= 5:
            for m in re.finditer(r'\b([a-z]{5,})', text_lower):
                if w.startswith(m.group(1)):
                    return True
        return False

    # strip parens first, then split on "/" for top-level OR alternatives
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
    """Map LLM-returned matched_points back to the canonical key_points list.

    The LLM is instructed to copy exact strings but sometimes capitalises differently
    or rephrases slightly.  We accept a match when:
      - case-insensitive exact match, OR
      - the canonical key_point is a substring of the LLM string (or vice-versa), OR
      - every word in the key_point appears in the LLM string (word-level overlap).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# FREE-RESPONSE SCORING
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

    # check which key phrases appear anywhere in the text (word-level, case-insensitive)
    matched = [p for p in key_points if _phrase_in_text(p, text_lower)]
    missed  = [p for p in key_points if not _phrase_in_text(p, text_lower)]

    # calculate a numeric score
    if rubric:
        # weighted score: each key point has a point value in the rubric
        total  = sum(rubric.values())
        earned = sum(rubric.get(p, 1.0) for p in matched)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        # unweighted score: what fraction of key points did they mention?
        score = len(matched) / len(key_points)
    else:
        score = 0.0  # no key points defined, can't score anything

    parts = []
    if matched:
        parts.append("Covered: " + ", ".join(matched))
    if missed:
        parts.append("Missing: " + ", ".join(missed))

    # return the same shape as score_free_response_with_llm so callers don't need to branch
    return {
        "prompt_id":      prompt_data["id"],
        "text":           text,
        "expert_answer":  ea,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          score,
        "feedback":       "\n".join(parts) if parts else "No key points defined.",
        "strengths":      [],   # keyword scoring doesn't produce these (only LLM scoring does)
        "gaps":           [],
    }


def score_free_response_with_llm(model, api_key, base_url, prompt_data, text):
    ea         = prompt_data["expert_answers"][0] if prompt_data.get("expert_answers") else {}
    key_points = ea.get("key_points", [])
    rubric     = ea.get("rubric", {})

    # show each key point with its importance label so the LLM knows what matters most
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

    # Map LLM-returned strings back to canonical key_points (handles capitalisation/rephrasing)
    matched_set = _resolve_llm_matches(matched, key_points)

    # compute the score from rubric weights rather than accepting whatever the LLM returns —
    # the LLM's job is semantic matching; arithmetic is ours
    if rubric and key_points:
        total  = sum(rubric.get(p, 1) for p in key_points)
        earned = sum(rubric.get(p, 1) for p in key_points if p in matched_set)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        score = len([p for p in key_points if p in matched_set]) / len(key_points)
    else:
        score = 0.5  # no criteria defined

    # Rebuild matched/missed from resolved set so the UI shows canonical strings
    matched = [p for p in key_points if p in matched_set]
    missed  = [p for p in key_points if p not in matched_set]

    return {
        "prompt_id":      prompt_data["id"],
        "text":           text,
        "expert_answer":  ea,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          max(0.0, min(1.0, score)),  # clamp to [0, 1] in case of floating-point drift
        "feedback":       result.get("feedback", ""),
        "strengths":      result.get("strengths", []),
        "gaps":           result.get("gaps", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_with_keywords(scenario, transcript, expert_answer):
    # check which key phrases appear anywhere in the transcript (word-level, case-insensitive)
    text = transcript.lower()

    matched = []  # key points the learner mentioned
    missed  = []  # key points the learner didn't mention

    for point in expert_answer["key_points"]:
        if _phrase_in_text(point, text):
            matched.append(point)
        else:
            missed.append(point)

    # calculate a numeric score
    rubric = expert_answer["rubric"]
    if rubric:
        # weighted score: each key point has a point value in the rubric
        total  = sum(rubric.values())   # total possible points
        earned = sum(rubric.get(p, 1.0) for p in matched)  # add each matched point's value (default 1 if not in rubric)
        score  = earned / total if total > 0 else 0.0
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

    # return a plain dict — same shape as score_with_llm so the rest of the code works either way
    return {
        "scenario_id":    scenario["id"],
        "transcript":     transcript,
        "expert_answer":  expert_answer,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          round(score, 4),
        "feedback":       "\n".join(feedback_parts) if feedback_parts else "No key points defined.",
        "strengths":      [],  # keyword scoring doesn't produce these (only LLM scoring does)
        "gaps":           [],
    }


def score_with_llm(model, api_key, base_url, scenario, transcript, expert_answer):
    key_points = expert_answer.get("key_points", [])
    rubric     = expert_answer.get("rubric", {})

    # show each key point with its importance label so the LLM knows what matters most
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
        "- Credit semantic equivalence — the learner does not need to use the exact phrasing.\n"
        "- Do NOT give credit for vague references, partial answers, or enthusiasm without substance.\n"
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
        "You are an impartial grader for a performative assessment. "
        "Your role is to evaluate performance against explicit criteria. "
        "Credit what is clearly demonstrated — exact wording is not required, semantic equivalence counts. "
        "Be critical: missing a CRITICAL or HIGH point is a significant failure and must be named. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw    = llm_chat(model, system, prompt, api_key, base_url)
    result = _extract_json(raw)

    matched = result.get("matched_points", [])
    missed  = result.get("missed_points",  [])

    # Map LLM-returned strings back to canonical key_points (handles capitalisation/rephrasing)
    matched_set = _resolve_llm_matches(matched, key_points)

    # compute the score from rubric weights rather than accepting whatever number the LLM returns —
    # the LLM's job is semantic matching; arithmetic is ours
    if rubric and key_points:
        total  = sum(rubric.get(p, 1) for p in key_points)
        earned = sum(rubric.get(p, 1) for p in key_points if p in matched_set)
        score  = earned / total if total > 0 else 0.0
    elif key_points:
        score = len([p for p in key_points if p in matched_set]) / len(key_points)
    else:
        score = 0.5  # no criteria defined

    # Rebuild matched/missed from resolved set so the UI shows canonical strings
    matched = [p for p in key_points if p in matched_set]
    missed  = [p for p in key_points if p not in matched_set]

    return {
        "scenario_id":    scenario["id"],
        "transcript":     transcript,
        "expert_answer":  expert_answer,
        "matched_points": matched,
        "missed_points":  missed,
        "score":          max(0.0, min(1.0, score)),  # clamp to [0, 1] in case of floating-point drift
        "feedback":       result.get("feedback", ""),
        "strengths":      result.get("strengths", []),
        "gaps":           result.get("gaps", []),
    }
