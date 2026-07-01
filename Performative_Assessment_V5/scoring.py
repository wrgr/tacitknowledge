"""
scoring.py — keyword and LLM-based scoring for scenarios and free-response prompts.

Scenario scoring now has two dimensions:
  1. Coverage Score  — which key points were addressed (existing model, extended to full transcript)
  2. Quality Score   — how deeply each matched point was explained (Chi's discriminators: 0/1/2)

Combined score = coverage_weight × Coverage + quality_weight × Quality
(weights come from scenario["scoring_weights"], default 0.6/0.4)
"""

import difflib
import logging
import re

import config
from llm import (
    llm_chat, llm_chat_json, _extract_json, clip,
    EVALUATIVE_TEMPERATURE, EVALUATIVE_SEED, cached_evaluative_call,
)

logger = logging.getLogger(__name__)

# Bump whenever the corresponding prompt's wording changes, so old cached
# results (see llm.cached_evaluative_call) don't silently apply to a changed prompt.
_PROMPT_VERSION_EXTRACT_EVIDENCE = "extract_evidence_v1"
_PROMPT_VERSION_FR_GRADE         = "fr_grade_v1"
_PROMPT_VERSION_SCENARIO_GRADE   = "scenario_grade_v1"


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


def _coerce_str(val):
    """Return val as a plain string; joins lists, converts anything else via str()."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return val if isinstance(val, str) else str(val)


def _coerce_str_list(val):
    """Return val as a flat list of strings; flattens one level of nesting."""
    if not isinstance(val, list):
        return [_coerce_str(val)] if val else []
    return [_coerce_str(item) for item in val]


def _match_point_for_string(m_lower, key_points):
    """Find the canonical key_point a raw LLM-returned string most likely refers to."""
    for p in key_points:
        p_lower = p.lower()
        if p_lower == m_lower:
            return p
        if p_lower in m_lower or m_lower in p_lower:
            return p
        p_words = set(p_lower.split())
        m_words = set(m_lower.split())
        if p_words and p_words.issubset(m_words):
            return p
    return None


def _resolve_llm_matches(matched_from_llm, key_points):
    """Map LLM-returned matched_points back to the canonical key_points list."""
    resolved = set()
    for m in matched_from_llm:
        if not isinstance(m, str):
            continue
        p = _match_point_for_string(m.lower().strip(), key_points)
        if p:
            resolved.add(p)
    return resolved


def _resolve_llm_matches_with_quotes(matched_entries, key_points):
    """Like _resolve_llm_matches, but for the evidence-grounded schema where each
    matched entry is {"key_point": ..., "supporting_quote": ...} (Part A of the
    ungrounded-matches fix). Bare strings are also accepted (supporting_quote=""),
    for backward compatibility with older cached results or a model that ignores
    the schema instruction.

    Returns (resolved_set, quotes_by_point).
    """
    resolved = set()
    quotes_by_point = {}
    for entry in matched_entries:
        if isinstance(entry, dict):
            m, quote = entry.get("key_point", ""), entry.get("supporting_quote", "")
        elif isinstance(entry, str):
            m, quote = entry, ""
        else:
            continue
        if not isinstance(m, str) or not m.strip():
            continue
        p = _match_point_for_string(m.lower().strip(), key_points)
        if p:
            resolved.add(p)
            if quote and isinstance(quote, str) and not quotes_by_point.get(p):
                quotes_by_point[p] = quote
    return resolved, quotes_by_point


def _normalize_for_quote_match(s):
    """Lowercase, strip punctuation, collapse whitespace -- for grounding a
    supporting_quote against the submission while tolerating minor
    whitespace/punctuation noise (not paraphrase-level looseness)."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _quote_supported(quote, submission_text):
    """Part A2: verify a supporting_quote actually appears in the submission text.

    Deterministic Python check, not another LLM call -- this is the actual fix
    for the ungrounded-match defect (a well-formed but factually false claim).
    Exact substring match after normalization handles verbatim/near-verbatim
    quotes; a high-threshold fuzzy fallback tolerates a stray word/typo but does
    NOT accept genuine paraphrases (ratio threshold is deliberately strict).
    """
    if not quote or not isinstance(quote, str):
        return False
    q = _normalize_for_quote_match(quote)
    t = _normalize_for_quote_match(submission_text)
    if not q or not t:
        return False
    if q in t:
        return True
    window = len(q)
    if window < 8:
        return False  # too short for fuzzy matching to be meaningful -- require exact
    step = max(1, window // 4)
    for i in range(0, max(1, len(t) - window + 1), step):
        segment = t[i:i + window]
        if difflib.SequenceMatcher(None, q, segment).ratio() >= 0.9:
            return True
    return False


_NEGATION_PATTERNS = re.compile(
    r"\b(failed to|did not|didn't|does not|doesn't|no discussion of|not addressed|"
    r"not covered|missing|absent|never (?:mention|address|discuss)(?:ed|es)?|lacks?|"
    r"without (?:mentioning|addressing|discussing)|nowhere (?:mention|address)(?:ed|es)?)\b",
    re.IGNORECASE,
)


def _find_narrative_contradictions(matched_set, narrative_texts):
    """Part B1: heuristic scan for direct contradictions between the structured
    matched_points verdict and the free-text feedback/gap narrative from the same
    grading pass -- e.g. a key point marked matched while the narrative states
    "failed to address <key point>". Sentence-scoped so a negation elsewhere in a
    long narrative doesn't falsely flag an unrelated key point. Not exhaustive --
    only needs to catch clear, direct contradictions.
    """
    contradictions = set()
    combined = " ".join(t for t in narrative_texts if t)
    if not combined or not matched_set:
        return contradictions
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    for p in matched_set:
        p_lower = p.lower()
        for sent in sentences:
            sent_lower = sent.lower()
            if p_lower in sent_lower and _NEGATION_PATTERNS.search(sent_lower):
                contradictions.add(p)
                break
    return contradictions


def _aggregate_grading_samples(samples, key_points):
    """Majority-vote matched_points across N self-consistency samples of a grading call.

    Ties default to "not matched" (the more conservative outcome). The aggregated
    matched-point set then flows through the SAME scoring arithmetic as a single
    sample -- we never average the raw numeric scores from each sample, since that
    could produce a valid-looking number built from inconsistent underlying judgments.
    Prose fields (feedback/strengths/gaps) are taken from the first sample and
    annotated to disclose that self-consistency was used.
    """
    n = len(samples)
    votes = {}
    quotes_by_point = {}
    for s in samples:
        matched_entries = s.get("matched_points", [])
        resolved, quotes = _resolve_llm_matches_with_quotes(matched_entries, key_points)
        for p in resolved:
            votes[p] = votes.get(p, 0) + 1
            if quotes.get(p) and not quotes_by_point.get(p):
                quotes_by_point[p] = quotes[p]

    majority_matched = [p for p in key_points if votes.get(p, 0) > n / 2]  # tie -> not matched

    aggregated = dict(samples[0])
    # Keep the evidence-grounded schema regardless of self-consistency mode, so
    # downstream A2 quote validation runs the same way either way.
    aggregated["matched_points"] = [
        {"key_point": p, "supporting_quote": quotes_by_point.get(p, "")} for p in majority_matched
    ]
    aggregated["missed_points"] = [p for p in key_points if p not in majority_matched]
    note = f"(graded via {n}-sample majority vote)"
    aggregated["feedback"] = (_coerce_str(aggregated.get("feedback", "")) + " " + note).strip()
    aggregated["self_consistency_samples"] = n
    return aggregated


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

def _extract_evidence(model, api_key, base_url, text, key_points, bypass_cache=False):
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
        "TRANSCRIPT:\n" + clip(text, 6000) + "\n\n"
        "The transcript contains turns labelled 'Examiner:' and learner turns (any other label). "
        "IMPORTANT: Only credit what the LEARNER said in their own turns. "
        "Do not attribute anything the Examiner says to the learner — even if the Examiner's "
        "question names or references a key point, credit only comes from the learner's response.\n\n"
        "For each key point, quote the LEARNER'S OWN WORDS verbatim (a short phrase or "
        "sentence copied exactly from their text, in quotation marks) that addresses it -- "
        "do not paraphrase or summarise. If the learner did not address the key point in "
        "their own words, write 'not addressed'.\n"
        "Format each line exactly as: '• <key point>: \"<verbatim quote>\"'\n"
        "No extra text. One line per key point."
    )
    system = (
        "You extract concise per-topic evidence from transcripts. "
        "Credit only what the learner said — never the examiner's questions or framing. "
        "One sentence per topic. No extra commentary."
    )
    try:
        def _call():
            return llm_chat(model, system, prompt, api_key, base_url,
                            temperature=EVALUATIVE_TEMPERATURE, seed=EVALUATIVE_SEED).strip()
        result = cached_evaluative_call(model, base_url, _PROMPT_VERSION_EXTRACT_EVIDENCE,
                                        system, prompt, _call, bypass_cache=bypass_cache)
        return result or clip(text, 3000)
    except Exception:
        return clip(text, 3000)


# ─────────────────────────────────────────────────────────────────────────────
# FREE-RESPONSE SCORING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def score_free_response_with_keywords(prompt_data, text):
    # Pure Python string/regex matching, no LLM call -- already deterministic,
    # so it's exempt from the determinism/caching/self-consistency work above.
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


def score_free_response_with_llm(model, api_key, base_url, prompt_data, text, bypass_cache=False):
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
    evidence = _extract_evidence(model, api_key, base_url, text, key_points, bypass_cache=bypass_cache)

    prompt = (
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n" + key_points_text + "\n\n"
        "LEARNER EVIDENCE (extracted from submission; each line's quote is verbatim from "
        "the submission):\n" + evidence + "\n\n"
        "Grade the evidence against the key points. Credit semantic equivalence for the "
        "SUBSTANCE of a key point, but every match must be evidence-grounded: a key point "
        "may only be marked matched if the LEARNER EVIDENCE above contains an actual quote "
        "supporting it. The surface presence of related vocabulary (e.g. the key point's own "
        "words appearing somewhere) is NOT sufficient grounds for a match. If no supporting "
        "quote exists for a key point, put it in missed_points instead.\n\n"
        "Return this JSON — no markdown, no extra text:\n"
        '{"matched_points":[{"key_point":"<exact key point string>",'
        '"supporting_quote":"<the verbatim quote from LEARNER EVIDENCE that supports it>"}],'
        '"missed_points":[<exact key point strings omitted>],'
        '"strengths":[<1-3 short phrases>],'
        '"gaps":[<one sentence per gap>],'
        '"feedback":"<2-3 sentences>"}'
    )

    system = (
        "You are an impartial grader. Score the learner's evidence against the key points. "
        "Credit semantic equivalence for substance, but every matched_points entry MUST include "
        "a supporting_quote copied verbatim from the LEARNER EVIDENCE -- never invent or "
        "paraphrase a quote, and never mark a key point matched without one. "
        "Respond only with valid JSON."
    )

    def _grade_once():
        raw = llm_chat_json(model, system, prompt, api_key, base_url,
                            temperature=EVALUATIVE_TEMPERATURE, seed=EVALUATIVE_SEED)
        return _extract_json(raw)

    def _grade():
        if config.SELF_CONSISTENCY_SCORING:
            samples = [_grade_once() for _ in range(config.SELF_CONSISTENCY_SAMPLES)]
            return _aggregate_grading_samples(samples, key_points)
        return _grade_once()

    result = cached_evaluative_call(model, base_url, _PROMPT_VERSION_FR_GRADE,
                                    system, prompt, _grade, bypass_cache=bypass_cache)

    matched_entries = result.get("matched_points", [])

    matched_set, quotes_by_point = _resolve_llm_matches_with_quotes(matched_entries, key_points)

    # Part A2: deterministic grounding check -- demote any match whose quote doesn't
    # actually appear in the submission. Not skippable; this is the real fix.
    ungrounded = []
    for p in list(matched_set):
        if not _quote_supported(quotes_by_point.get(p, ""), text):
            matched_set.discard(p)
            ungrounded.append(p)
    if ungrounded:
        logger.warning(
            "[scoring] FR grading: demoted ungrounded match(es) %s -- no supporting quote "
            "found in submission (prompt_id=%s)", ungrounded, prompt_data.get("id"),
        )

    # Part B: cross-check the surviving matched verdicts against the free-text
    # feedback/gaps narrative from the same grading pass; ties resolve conservatively.
    narrative_texts = [_coerce_str(result.get("feedback", ""))] + _coerce_str_list(result.get("gaps", []))
    contradictions = _find_narrative_contradictions(matched_set, narrative_texts)
    for p in contradictions:
        matched_set.discard(p)
        logger.warning(
            "[scoring] FR grading: demoted key point '%s' -- feedback/gaps narrative "
            "contradicts the matched verdict (prompt_id=%s)", p, prompt_data.get("id"),
        )

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
        "matched_point_quotes": {p: quotes_by_point.get(p, "") for p in matched},
        "score":          max(0.0, min(1.0, score)),
        "feedback":       _coerce_str(result.get("feedback", "")),
        "strengths":      _coerce_str_list(result.get("strengths", [])),
        "gaps":           _coerce_str_list(result.get("gaps", [])),
        "self_consistency_samples": result.get("self_consistency_samples", 0),
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
                   recall_transcript="", probe_transcript="", bypass_cache=False):
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
        recall_evidence = _extract_evidence(model, api_key, base_url, recall_transcript, key_points,
                                            bypass_cache=bypass_cache)
        probe_evidence  = _extract_evidence(model, api_key, base_url, probe_transcript,  key_points,
                                            bypass_cache=bypass_cache)
        evidence_section = (
            "RECALL EVIDENCE (what the learner volunteered unprompted):\n" + recall_evidence + "\n\n"
            "PROBE EVIDENCE (what the learner said when questioned on their reasoning):\n" + probe_evidence
        )
    else:
        evidence_section = "LEARNER EVIDENCE:\n" + _extract_evidence(
            model, api_key, base_url, transcript, key_points, bypass_cache=bypass_cache
        )

    prompt = (
        "KEY POINTS WITH IMPORTANCE WEIGHTS:\n"
        + key_points_text + "\n\n"
        + evidence_section + "\n\n"
        "QUALITY SCALE (per matched point):\n"
        "  0 = stated only, no reasoning\n"
        "  1 = partial — ONE of: conditional, goal-linked, or consequence-aware\n"
        "  2 = full — TWO OR MORE of the above\n"
        "Fluency and length are NOT quality. Credit semantic equivalence for coverage, but "
        "every match must be evidence-grounded: a key point may only be marked matched if the "
        "evidence above contains an actual quote supporting it. The surface presence of related "
        "vocabulary is NOT sufficient grounds for a match. If no supporting quote exists for a "
        "key point, put it in missed_points instead.\n\n"
        "Return this JSON — no markdown, no extra text:\n"
        '{"matched_points":[{"key_point":"<exact key point string>",'
        '"supporting_quote":"<the verbatim quote from the evidence above that supports it>"}],'
        '"missed_points":[<exact key point strings omitted>],'
        '"quality_ratings":{"<key point>":0|1|2},'
        '"strengths":[<1-3 short phrases>],'
        '"gaps":[<one sentence per gap>],'
        '"feedback":"<2-3 sentences>"}'
    )

    system = (
        "You are an impartial grader. Score the learner's evidence against the key points. "
        "Credit semantic equivalence. Quality scores must reflect conditional/goal/consequence "
        "reasoning — not fluency. Every matched_points entry MUST include a supporting_quote "
        "copied verbatim from the evidence -- never invent or paraphrase a quote, and never "
        "mark a key point matched without one. Respond only with valid JSON."
    )

    def _grade_once():
        raw = llm_chat_json(model, system, prompt, api_key, base_url,
                            temperature=EVALUATIVE_TEMPERATURE, seed=EVALUATIVE_SEED)
        return _extract_json(raw)

    def _grade():
        if config.SELF_CONSISTENCY_SCORING:
            samples = [_grade_once() for _ in range(config.SELF_CONSISTENCY_SAMPLES)]
            return _aggregate_grading_samples(samples, key_points)
        return _grade_once()

    result = cached_evaluative_call(model, base_url, _PROMPT_VERSION_SCENARIO_GRADE,
                                    system, prompt, _grade, bypass_cache=bypass_cache)

    matched_entries = result.get("matched_points", [])

    matched_set, quotes_by_point = _resolve_llm_matches_with_quotes(matched_entries, key_points)

    # Part A2: deterministic grounding check against the actual transcript(s) --
    # demote any match whose quote doesn't appear in what the learner actually said.
    validation_text = (
        (recall_transcript + "\n" + probe_transcript) if (recall_transcript and probe_transcript)
        else (transcript or recall_transcript or probe_transcript)
    )
    ungrounded = []
    for p in list(matched_set):
        if not _quote_supported(quotes_by_point.get(p, ""), validation_text):
            matched_set.discard(p)
            ungrounded.append(p)
    if ungrounded:
        logger.warning(
            "[scoring] scenario grading: demoted ungrounded match(es) %s -- no supporting "
            "quote found in transcript (scenario_id=%s)", ungrounded, scenario.get("id"),
        )

    # Part B: cross-check the surviving matched verdicts against the free-text
    # feedback/gaps narrative from the same grading pass; ties resolve conservatively.
    narrative_texts = [_coerce_str(result.get("feedback", ""))] + _coerce_str_list(result.get("gaps", []))
    contradictions = _find_narrative_contradictions(matched_set, narrative_texts)
    for p in contradictions:
        matched_set.discard(p)
        logger.warning(
            "[scoring] scenario grading: demoted key point '%s' -- feedback/gaps narrative "
            "contradicts the matched verdict (scenario_id=%s)", p, scenario.get("id"),
        )

    # arithmetic stays in Python — LLM only does semantic matching
    weights  = scenario.get("scoring_weights", {"coverage": 0.6, "quality": 0.4})
    coverage = _compute_coverage_score(matched_set, key_points, rubric)

    # quality ratings: resolve LLM keys back to canonical key_points
    raw_ratings = result.get("quality_ratings", {})
    quality_ratings = {}
    for llm_key, rating in raw_ratings.items():
        if not isinstance(llm_key, str):
            continue
        for p in matched_set:
            if not isinstance(p, str):
                continue
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
        "matched_point_quotes": {p: quotes_by_point.get(p, "") for p in matched},
        "quality_ratings":  quality_ratings,
        "quality_evidence": {},
        "point_sources":    sources,
        "coverage_score":   round(coverage, 4),
        "quality_score":    round(quality, 4),
        "score":            max(0.0, min(1.0, combined)),
        "feedback":         _coerce_str(result.get("feedback", "")),
        "strengths":        _coerce_str_list(result.get("strengths", [])),
        "gaps":             _coerce_str_list(result.get("gaps", [])),
        "self_consistency_samples": result.get("self_consistency_samples", 0),
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
        "matched_point_quotes": {
            **recall_ev.get("matched_point_quotes", {}), **probe_ev.get("matched_point_quotes", {})
        },
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
