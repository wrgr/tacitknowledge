"""
writing_process.py — Writing-process analysis for Free Response (FR) submissions.

Interprets *how* an essay was written (pauses, revisions, paste events, the build
sequence captured by the client-side WritingTracker) alongside the finished text,
per the FR writing-process assessment redesign. Deterministic pattern detection
(B1/B3/B4) is plain Python; the only LLM call (B2) judges whether revisions moved
the text toward Chi et al.'s markers of self-explanation (conditional, goal-linked,
consequence-aware statements) — a genuinely semantic judgment that pattern-matching
cannot make.

Process is always an interpretive overlay, never a silent adjustment to the product
score (see Part C of the design). Every signal here is a pattern with competing
hypotheses, not a verdict — see the "Interpretation caution" note in reports.py.
"""

from llm import (
    llm_chat_json, _extract_json,
    EVALUATIVE_TEMPERATURE, EVALUATIVE_SEED, cached_evaluative_call,
)

# Bump whenever the revision-judgment prompt's wording changes, so old cached
# results (see llm.cached_evaluative_call) don't silently apply to a changed prompt.
_PROMPT_VERSION_REVISION_TOWARD_QUALITY = "revision_toward_quality_v1"

# TUNABLE -- difficulty-point detection: how close a revision must be to a pause
# (in characters and in seconds) to count as "at" that pause, and how large a
# revision must be to count as "heavy" rather than a minor tweak.
DIFFICULTY_PROXIMITY_CHARS = 100
DIFFICULTY_PROXIMITY_S     = 60
DIFFICULTY_MIN_REVISION_CHARS = 15

# TUNABLE -- authenticity thresholds (bounded concern level from deterministic signals)
LARGE_PASTE_CHARS       = 200
ELEVATED_PASTED_FRACTION = 0.40
LOW_PASTED_FRACTION      = 0.10
ELEVATED_UNREVISED_PASTE_FRACTION = 0.15

# TUNABLE -- product score band used for the process x product quadrant
STRONG_PRODUCT_THRESHOLD = 0.7

# TUNABLE -- effort/frictionless classification thresholds
EFFORTFUL_REVISION_DENSITY   = 1.0   # revisions per 100 words
EFFORTFUL_PAUSE_RATIO        = 0.25  # pause time / total active time
FRICTIONLESS_REVISION_DENSITY = 0.3
FRICTIONLESS_PAUSE_RATIO      = 0.10

# Alternative interpretations, sourced verbatim from docs/fr_evidence_model.md — the
# "Does NOT rule out" column for each signal. Static because these alternatives are
# stable per signal type, not per instance (only revision-toward-quality needs a
# fresh, per-instance judgment from the LLM — see assess_revision_toward_quality).
ALT_DIFFICULTY_POINT = (
    "Does not rule out distraction, an unrelated interruption, or typing/technical friction."
)
ALT_AUTHENTICITY = (
    "Can reflect legitimate reuse of the learner's own prior notes, appropriate quotation, "
    "or simply confident single-pass writing, not necessarily reduced authorship."
)
ALT_REVISION_TOWARD_QUALITY = (
    "Could be coincidental phrasing improvement, or a later paste of better phrasing from elsewhere."
)
ALT_FRICTIONLESS_PROCESS = (
    "Could be genuine fluent competence, prior preparation, or unflagged reuse — process "
    "data alone cannot distinguish these."
)
ALT_CONFIDENCE_COLLAPSE = (
    "A learner who is simply a harsh self-rater in general, not specifically because of a "
    "gap this task exposed."
)
ALT_CONFIDENCE_RISE = (
    "General optimism after completing a task, independent of any specific clarification."
)

# TUNABLE -- confidence-collapse detection threshold, adjust after testing
CONFIDENCE_COLLAPSE_THRESHOLD = -2
CONFIDENCE_RISE_THRESHOLD     = 2


# ─────────────────────────────────────────────────────────────────────────────
# B1 — DETERMINISTIC PATTERN COMPUTATION (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _word_count(text):
    return len(text.split()) if text and text.strip() else 0


def _location_label(char_position, essay_len):
    """Coarse third-of-essay label so reports can say roughly where something fell."""
    if essay_len <= 0:
        return "unknown position"
    frac = max(0.0, min(1.0, char_position / essay_len))
    if frac < 1 / 3:
        return "early in the essay"
    if frac < 2 / 3:
        return "in the middle of the essay"
    return "late in the essay"


def compute_effort_profile(process_log, writing_metrics, essay_text):
    """Total active time, pause-to-writing ratio, revision density, longest pause + location."""
    process_log     = process_log or {}
    writing_metrics = writing_metrics or {}
    pauses    = process_log.get("pause_events") or []
    revisions = process_log.get("revision_events") or []
    words     = _word_count(essay_text)
    essay_len = len(essay_text or "")

    total_active_s = writing_metrics.get("total_time_s")
    if total_active_s is None:
        snapshots = process_log.get("snapshots") or []
        total_active_s = snapshots[-1]["timestamp_s"] if snapshots else 0.0
    total_active_s = total_active_s or 0.0

    pause_time_s = sum(p.get("duration_s", 0) for p in pauses)
    pause_to_writing_ratio = round(pause_time_s / total_active_s, 3) if total_active_s > 0 else 0.0

    revision_density = round((len(revisions) / words) * 100, 2) if words > 0 else 0.0

    longest_pause = max(pauses, key=lambda p: p.get("duration_s", 0), default=None)
    longest_pause_summary = None
    if longest_pause:
        longest_pause_summary = {
            "duration_s":    longest_pause.get("duration_s", 0),
            "char_position": longest_pause.get("char_position", 0),
            "location":      _location_label(longest_pause.get("char_position", 0), essay_len),
        }

    return {
        "total_active_time_s":     total_active_s,
        "pause_to_writing_ratio":  pause_to_writing_ratio,
        "revision_density":        revision_density,
        "revision_count":          len(revisions),
        "word_count":              words,
        "longest_pause":           longest_pause_summary,
    }


def find_difficulty_points(process_log, essay_text):
    """Pause + nearby heavy revision co-occurring at the same region — the IOED signature.

    Never conclude difficulty from a pause alone; a pause only becomes a candidate
    difficulty point when a substantive revision also happened nearby in place and time.
    """
    process_log = process_log or {}
    pauses      = process_log.get("pause_events") or []
    revisions   = process_log.get("revision_events") or []
    essay_len   = len(essay_text or "")

    points = []
    for pause in pauses:
        p_pos = pause.get("char_position", 0)
        p_t   = pause.get("timestamp_s", 0)
        for rev in revisions:
            changed = max(len(rev.get("removed_text", "")), len(rev.get("inserted_text", "")))
            if changed < DIFFICULTY_MIN_REVISION_CHARS:
                continue
            close_in_place = abs(rev.get("char_position", 0) - p_pos) <= DIFFICULTY_PROXIMITY_CHARS
            close_in_time  = abs(rev.get("timestamp_s", 0) - p_t) <= DIFFICULTY_PROXIMITY_S
            if close_in_place and close_in_time:
                points.append({
                    "char_position": p_pos,
                    "pause_s":       pause.get("duration_s", 0),
                    "note": (
                        f"a {pause.get('duration_s', 0)}s pause and a heavy rework "
                        f"{_location_label(p_pos, essay_len)}"
                    ),
                    "alternative_interpretation": ALT_DIFFICULTY_POINT,
                })
                break  # one candidate per pause is enough

    return points


def compute_authenticity(process_log, essay_text):
    """Bounded authenticity concern level from paste size/position/surrounding revision."""
    process_log = process_log or {}
    pastes      = process_log.get("paste_events") or []
    revisions   = process_log.get("revision_events") or []
    essay_len   = max(1, len(essay_text or ""))

    total_pasted_chars = sum(p.get("paste_length", 0) for p in pastes)
    pasted_fraction = round(total_pasted_chars / essay_len, 3)

    unrevised_large_pastes = 0
    for paste in pastes:
        if paste.get("paste_length", 0) < LARGE_PASTE_CHARS:
            continue
        p_pos = paste.get("char_position", 0)
        p_t   = paste.get("timestamp_s", 0)
        has_nearby_revision = any(
            abs(rev.get("char_position", 0) - p_pos) <= DIFFICULTY_PROXIMITY_CHARS
            and rev.get("timestamp_s", 0) >= p_t
            and rev.get("timestamp_s", 0) <= p_t + DIFFICULTY_PROXIMITY_S
            for rev in revisions
        )
        if not has_nearby_revision:
            unrevised_large_pastes += 1

    evidence = []
    if pastes:
        evidence.append(f"{len(pastes)} paste event(s) totalling {total_pasted_chars} characters "
                         f"({round(pasted_fraction * 100)}% of the final essay)")
    if unrevised_large_pastes:
        evidence.append(f"{unrevised_large_pastes} large paste(s) (≥{LARGE_PASTE_CHARS} chars) "
                         "with little surrounding revision")

    if pasted_fraction >= ELEVATED_PASTED_FRACTION or (
        unrevised_large_pastes and pasted_fraction >= ELEVATED_UNREVISED_PASTE_FRACTION
    ):
        level = "elevated"
    elif pastes and pasted_fraction >= LOW_PASTED_FRACTION:
        level = "low"
    elif pastes:
        level = "low"
    else:
        level = "none"

    return {
        "level":           level,
        "pasted_fraction": pasted_fraction,
        "evidence":        evidence,
        "alternative_interpretations": [ALT_AUTHENTICITY] if pastes else [],
    }


def compute_confidence_calibration(pre_rating, post_rating):
    """Rate → explain → re-rate (Rozenblit & Keil): confidence collapse after being forced
    to produce a detailed explanation is itself evidence of shallow understanding — the
    most directly validated mechanism in this whole system. Requires no LLM call.

    Independent of process_log — a submission can carry this signal even when no writing
    process was captured at all (e.g. process_overlay_enabled is off but ratings exist).
    """
    if pre_rating is None or post_rating is None:
        return None
    try:
        pre_rating  = int(pre_rating)
        post_rating = int(post_rating)
    except (TypeError, ValueError):
        return None

    delta = post_rating - pre_rating

    if delta <= CONFIDENCE_COLLAPSE_THRESHOLD:
        finding = "confidence_collapse"
        confidence = "moderate-high"
        note = (
            "Confidence dropped notably after writing the explanation — the classic "
            "illusion-of-explanatory-depth signature: being forced to produce a detailed "
            "explanation exposed a gap between perceived and actual understanding."
        )
        alternative = ALT_CONFIDENCE_COLLAPSE
    elif delta >= CONFIDENCE_RISE_THRESHOLD:
        finding = "confidence_rise"
        confidence = "low-moderate"
        note = (
            "Confidence rose after writing the explanation — the act of writing appears "
            "to have clarified or reinforced the learner's understanding. This is a "
            "positive process indicator, distinct from and not a substitute for the "
            "product score."
        )
        alternative = ALT_CONFIDENCE_RISE
    else:
        finding = "no_signal"
        confidence = "n/a"
        note = (
            "No notable change between pre- and post-write confidence — not itself "
            "informative of quality either way."
        )
        alternative = ""

    return {
        "pre_rating":       pre_rating,
        "post_rating":      post_rating,
        "confidence_delta": delta,
        "finding":          finding,
        "confidence":       confidence,
        "note":             note,
        "alternative_interpretation": alternative,
    }


def compute_trajectory(process_log, revision_density):
    """Compare first vs. last snapshot: write-forward (linear) vs. iterative rework."""
    process_log = process_log or {}
    snapshots   = process_log.get("snapshots") or []

    if len(snapshots) < 2:
        return "iterative" if revision_density >= EFFORTFUL_REVISION_DENSITY else "linear"

    first_text = snapshots[0].get("text", "")
    last_text  = snapshots[-1].get("text", "")
    if not first_text:
        return "iterative" if revision_density >= EFFORTFUL_REVISION_DENSITY else "linear"

    prefix_len = 0
    max_prefix = min(len(first_text), len(last_text))
    while prefix_len < max_prefix and first_text[prefix_len] == last_text[prefix_len]:
        prefix_len += 1

    prefix_ratio = prefix_len / len(first_text)
    return "linear" if prefix_ratio >= 0.85 else "iterative"


def classify_quadrant(product_score, effort_profile, authenticity, trajectory):
    """Derive the process x product quadrant — the core interpretive output (B3)."""
    is_frictionless = (
        authenticity.get("level") == "elevated"
        or (
            trajectory == "linear"
            and effort_profile.get("revision_density", 0) < FRICTIONLESS_REVISION_DENSITY
            and effort_profile.get("pause_to_writing_ratio", 0) < FRICTIONLESS_PAUSE_RATIO
        )
    )
    is_effortful = (not is_frictionless) and (
        trajectory == "iterative"
        or effort_profile.get("revision_density", 0) >= EFFORTFUL_REVISION_DENSITY
        or effort_profile.get("pause_to_writing_ratio", 0) >= EFFORTFUL_PAUSE_RATIO
    )

    product_strong = product_score >= STRONG_PRODUCT_THRESHOLD

    if product_strong and is_effortful:
        return {
            "label": "genuine_engaged_reasoning",
            "interpretation": (
                "Strong product paired with an effortful, iterative writing process — "
                "high-confidence evidence of genuine engaged reasoning."
            ),
            "alternative_interpretation": "",
        }
    if product_strong and is_frictionless:
        return {
            "label": "authenticity_review",
            "interpretation": (
                "Strong product paired with a frictionless or heavily pasted process. "
                "Per the evidence model, this pairing has no single confident claim — "
                "recommend the instructor confirm authorship with the learner if it "
                "matters for the assessment's purpose. This does not lower the "
                "competence score."
            ),
            "alternative_interpretation": ALT_FRICTIONLESS_PROCESS,
        }
    if (not product_strong) and is_effortful:
        return {
            "label": "engaged_under_knowledgeable",
            "interpretation": (
                "Weak product paired with an effortful process — a coaching signal. "
                "The learner appears to be trying and lacking the knowledge, not the effort."
            ),
            "alternative_interpretation": "",
        }
    if (not product_strong) and is_frictionless:
        return {
            "label": "disengaged_shallow_confident",
            "interpretation": (
                "Weak product paired with a frictionless, unrevised process. Per the "
                "evidence model, this pairing has no single confident claim on its own — "
                "consistent with disengagement or shallow confidence, but not proof of it."
            ),
            "alternative_interpretation": ALT_FRICTIONLESS_PROCESS,
        }

    # Neither clearly effortful nor clearly frictionless — not enough process signal to
    # place confidently in a quadrant; fall back to the product-only reading.
    label = "genuine_engaged_reasoning" if product_strong else "engaged_under_knowledgeable"
    return {
        "label": label,
        "interpretation": (
            "Process signals were mixed or sparse — this reading leans on the product "
            "score, with process treated as inconclusive supporting context."
        ),
        "alternative_interpretation": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# B2 — SEMANTIC JUDGMENT (LLM, bounded)
# ─────────────────────────────────────────────────────────────────────────────

def assess_revision_toward_quality(essay_text, process_log, model, api_key, base_url, bypass_cache=False):
    """Did substantive revisions move the text toward Chi's self-explanation markers?

    Fluency, length, and polish do NOT count — only movement toward conditional
    ("if/unless/because"), goal-linked ("in order to/the purpose is"), or
    consequence-aware ("which means/otherwise") phrasing. Fails gracefully to
    'not_assessed' rather than defaulting to zero.
    """
    revisions = [
        r for r in (process_log or {}).get("revision_events") or []
        if r.get("removed_text") or r.get("inserted_text")
    ]
    if not revisions:
        return {"rating": "not_assessed", "evidence": [], "alternative_explanation": ""}

    pairs_text = "\n".join(
        f"- before: {r.get('removed_text', '') or '(nothing — new text inserted)'}\n"
        f"  after:  {r.get('inserted_text', '') or '(deleted, nothing kept)'}\n"
        f"  context: ...{r.get('context_before', '')}"
        for r in revisions
    )

    system = (
        "You are an educational psychologist assessing revision quality in student writing. "
        "This process signal is supporting evidence, not proof, of understanding — treat it as "
        "one pattern to weigh, not a verdict. "
        "Judge ONLY whether edits moved the text toward conditional, goal-linked, or "
        "consequence-aware phrasing (Chi et al.'s markers of self-explanation). "
        "Fluency, length, and polish are NOT quality — do not credit them. "
        "Respond only with valid JSON — no markdown, no extra text."
    )
    prompt = (
        "FINISHED ESSAY:\n" + essay_text + "\n\n"
        "REVISION EVENTS (before → after, with surrounding context):\n" + pairs_text + "\n\n"
        "For the substantive revisions, did edits move the text toward:\n"
        "  - more conditional phrasing ('if', 'unless', 'because')\n"
        "  - more goal-linked phrasing ('in order to', 'the purpose is')\n"
        "  - more consequence-aware phrasing ('which means', 'otherwise')\n\n"
        "Return this JSON exactly — no markdown, no extra text:\n"
        "{\n"
        '  "rating": "none | some | clear",\n'
        '  "evidence": [ {"before": "<short quote>", "after": "<short quote>"} ],  // 1-2 pairs, or [] if rating is "none"\n'
        '  "alternative_explanation": "<one line: a plausible non-understanding explanation for the same edits, '
        'e.g. coincidental phrasing improvement or pasted-in phrasing — empty string if rating is \\"none\\">"\n'
        "}"
    )

    try:
        def _call():
            raw = llm_chat_json(model, system, prompt, api_key, base_url,
                                temperature=EVALUATIVE_TEMPERATURE, seed=EVALUATIVE_SEED)
            return _extract_json(raw)
        result = cached_evaluative_call(model, base_url, _PROMPT_VERSION_REVISION_TOWARD_QUALITY,
                                        system, prompt, _call, bypass_cache=bypass_cache)
    except Exception:
        return {"rating": "not_assessed", "evidence": [], "alternative_explanation": ""}

    rating = result.get("rating")
    if rating not in ("none", "some", "clear"):
        return {"rating": "not_assessed", "evidence": [], "alternative_explanation": ""}

    evidence = []
    for pair in (result.get("evidence") or [])[:2]:
        if isinstance(pair, dict) and (pair.get("before") or pair.get("after")):
            evidence.append({"before": str(pair.get("before", "")), "after": str(pair.get("after", ""))})

    alt = result.get("alternative_explanation")
    if not isinstance(alt, str) or not alt.strip():
        alt = ALT_REVISION_TOWARD_QUALITY if rating in ("some", "clear") else ""

    return {"rating": rating, "evidence": evidence, "alternative_explanation": alt}


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def analyze_writing_process(process_log, writing_metrics, essay_text, product_score,
                             model=None, api_key=None, base_url=None, use_llm=True):
    """Full writing-process interpretation (B1, B2, B3, B4).

    process_log may be sparse (few or no captured events) but should not be None —
    callers should skip the overlay entirely when no process_log was submitted at
    all (see process_overlay_enabled in Part C), rather than calling this with None.
    """
    process_log = process_log or {}

    effort_profile = compute_effort_profile(process_log, writing_metrics, essay_text)
    difficulty_points = find_difficulty_points(process_log, essay_text)
    authenticity = compute_authenticity(process_log, essay_text)
    trajectory = compute_trajectory(process_log, effort_profile["revision_density"])
    quadrant = classify_quadrant(product_score, effort_profile, authenticity, trajectory)

    if use_llm and model and api_key:
        revision_toward_quality = assess_revision_toward_quality(
            essay_text, process_log, model, api_key, base_url
        )
    else:
        revision_toward_quality = {"rating": "not_assessed", "evidence": [], "alternative_explanation": ""}

    return {
        "effort_profile":          effort_profile,
        "difficulty_points":       difficulty_points,
        "authenticity":            authenticity,
        "trajectory":              trajectory,
        "revision_toward_quality": revision_toward_quality,
        "quadrant":                quadrant,
    }
