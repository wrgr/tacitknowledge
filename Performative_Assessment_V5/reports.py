"""
reports.py — Markdown report generation for scenario sessions and free-response prompts.
"""

import logging
from datetime import datetime
from pathlib import Path

from llm import llm_chat, _extract_json

logger = logging.getLogger(__name__)

_INFERENCE_BOUNDARY = (
    "**Assessment scope:** This assessment evaluates procedural reasoning and declarative "
    "knowledge. It measures what the learner knows how to do and can explain — "
    "it does not verify physical execution or psychomotor skill."
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _append_thinking_profile(lines, thinking_profile):
    if not (thinking_profile.get("honey_mumford_style") or thinking_profile.get("solo_level")):
        return

    lines.append("## Learner Thinking Profile")
    lines.append("")

    insufficient = thinking_profile.get("insufficient_data_note")
    if insufficient:
        lines.append("> **Note — limited evidence:** " + insufficient)
        lines.append("")

    hm         = thinking_profile.get("honey_mumford_style", "")
    hm_conf    = thinking_profile.get("honey_mumford_confidence", "")
    hm_ev      = thinking_profile.get("honey_mumford_evidence", "")
    hm_reason  = thinking_profile.get("honey_mumford_reasoning", "")
    if hm:
        conf_tag = (" _(confidence: " + hm_conf + ")_") if hm_conf else ""
        lines.append("**Honey & Mumford style:** " + hm + conf_tag)
        if isinstance(hm_ev, list):
            for item in hm_ev:
                lines.append("  - _\"" + item + "\"_")
        elif hm_ev:
            lines.append("  - _" + hm_ev + "_")
        if hm_reason:
            lines.append("  > " + hm_reason)
        lines.append("")

    solo        = thinking_profile.get("solo_level", "")
    solo_conf   = thinking_profile.get("solo_confidence", "")
    solo_ev     = thinking_profile.get("solo_evidence", "")
    solo_reason = thinking_profile.get("solo_reasoning", "")
    if solo:
        conf_tag = (" _(confidence: " + solo_conf + ")_") if solo_conf else ""
        lines.append("**SOLO level:** " + solo + conf_tag)
        if isinstance(solo_ev, list):
            for item in solo_ev:
                lines.append("  - _\"" + item + "\"_")
        elif solo_ev:
            lines.append("  - _" + solo_ev + "_")
        if solo_reason:
            lines.append("  > " + solo_reason)
        lines.append("")

    # probe phase improvement signal
    ppi      = thinking_profile.get("probe_phase_improvement")
    ppi_note = thinking_profile.get("probe_phase_improvement_note")
    if ppi is not None:
        ppi_label = "Yes" if ppi else "No"
        lines.append("**Probe phase improvement:** " + ppi_label)
        if ppi_note:
            lines.append("  > " + ppi_note)
        lines.append("")

    patterns = thinking_profile.get("observed_patterns", [])
    if patterns:
        lines.append("**Observed patterns:**")
        for p in patterns:
            lines.append("  - " + p)
        lines.append("")

    note = thinking_profile.get("instructor_note", "")
    if note:
        lines.append("**Instructor note:** _" + note + "_")
        lines.append("")


def _append_scores(lines, ev):
    """Write Coverage, Quality, and Overall score sections."""
    coverage_pct = f"{ev.get('coverage_score', ev.get('score', 0)):.0%}"
    quality_pct  = f"{ev.get('quality_score', 0):.0%}"
    overall_pct  = f"{ev.get('score', 0):.0%}"

    lines.append("**Coverage Score:** " + coverage_pct + "  ")
    lines.append(
        "_Coverage measures which key steps were addressed across the full assessment "
        "(recall + probing combined)._"
    )
    lines.append("")
    lines.append("**Explanation Quality Score:** " + quality_pct + "  ")
    lines.append(
        "_Quality measures how deeply those steps were explained — whether the learner "
        "showed they understood WHY each step matters (conditional reasoning, goal-linked "
        "statements, consequence awareness), not just THAT it exists._"
    )
    lines.append("")
    lines.append("**Overall Score:** " + overall_pct)
    lines.append("")


_QUADRANT_LABELS = {
    "genuine_engaged_reasoning":    "Genuine engaged reasoning",
    "authenticity_review":         "Authenticity review",
    "engaged_under_knowledgeable": "Engaged but under-knowledgeable",
    "disengaged_shallow_confident": "Disengaged or shallow-confident",
}

_CONFIDENCE_FINDING_LABELS = {
    "confidence_collapse": "Confidence collapse (illusion-of-explanatory-depth signature)",
    "confidence_rise":     "Confidence rise",
    "no_signal":           "No notable change",
}

_PROCESS_INTERPRETATION_CAUTION = (
    "Process signals are indirect and ambiguous — interpreted as patterns, not verdicts. "
    "Competence is judged from the essay itself; the writing process is supporting context."
)


def _append_process_overlay(lines, overlay):
    """Write the Writing-Process interpretive overlay (Part D). Never touches the product score."""
    if not overlay:
        return

    lines.append("## Writing Process")
    lines.append("")

    quadrant = overlay.get("quadrant") or {}
    label = _QUADRANT_LABELS.get(quadrant.get("label"), quadrant.get("label", ""))
    if label:
        lines.append("**Process × Product:** " + label)
        if quadrant.get("interpretation"):
            lines.append("  > " + quadrant["interpretation"])
        if quadrant.get("alternative_interpretation"):
            lines.append("  > Alternative: " + quadrant["alternative_interpretation"])
        lines.append("")

    ep = overlay.get("effort_profile") or {}
    if ep:
        minutes = round((ep.get("total_active_time_s") or 0) / 60, 1)
        density = ep.get("revision_density") or 0
        if density > 0:
            per_words = round(100 / density)
            revision_phrase = f"roughly one substantial revision per {per_words} words"
        else:
            revision_phrase = "no substantial revisions"
        longest = ep.get("longest_pause")
        pause_phrase = (
            f"longest pause {longest['duration_s']}s ({longest['location']})"
            if longest else "no significant pauses"
        )
        lines.append(
            f"**Effort profile:** wrote for {minutes} min active time; {revision_phrase}; {pause_phrase}."
        )
        lines.append("")

    rtq = overlay.get("revision_toward_quality") or {}
    if rtq.get("rating") and rtq["rating"] != "not_assessed":
        lines.append("**Revision toward quality:** " + rtq["rating"])
        for pair in rtq.get("evidence", []):
            lines.append(f"  - \"{pair.get('before', '')}\" → \"{pair.get('after', '')}\"")
        if rtq.get("alternative_explanation"):
            lines.append("  > Alternative: " + rtq["alternative_explanation"])
        lines.append("")
    elif rtq.get("rating") == "not_assessed":
        lines.append("**Revision toward quality:** not assessed")
        lines.append("")

    difficulty_points = overlay.get("difficulty_points") or []
    if difficulty_points:
        lines.append("**Difficulty points:**")
        for dp in difficulty_points:
            note = f"  - {dp.get('note', '')} (around position {dp.get('char_position', 0)})"
            if dp.get("alternative_interpretation"):
                note += f" — alternative: {dp['alternative_interpretation']}"
            lines.append(note)
        lines.append("")

    authenticity = overlay.get("authenticity") or {}
    if authenticity.get("level") and authenticity["level"] != "none":
        ev_text = "; ".join(authenticity.get("evidence", []))
        detail = f" — {ev_text}" if ev_text else ""
        lines.append(f"**Authenticity:** {authenticity['level']}{detail}")
        for alt in authenticity.get("alternative_interpretations", []):
            lines.append(
                "  > Alternative: " + alt + " Recommend confirming with the learner "
                "if this matters for the assessment's purpose."
            )
        lines.append("")
    elif authenticity.get("level") == "none":
        lines.append("**Authenticity:** none — no pasted content detected.")
        lines.append("")

    cc = overlay.get("confidence_calibration")
    if cc:
        label = _CONFIDENCE_FINDING_LABELS.get(cc.get("finding"), cc.get("finding", ""))
        lines.append(
            f"**Confidence calibration:** {label} "
            f"(pre: {cc['pre_rating']}/10 → post: {cc['post_rating']}/10, "
            f"Δ={cc['confidence_delta']:+d})"
        )
        if cc.get("note"):
            lines.append("  > " + cc["note"])
        if cc.get("alternative_interpretation"):
            lines.append("  > Alternative: " + cc["alternative_interpretation"])
        lines.append("")

    lines.append("> " + _PROCESS_INTERPRETATION_CAUTION)
    lines.append("")


def _append_key_points_with_attribution(lines, ev):
    """Write key points with (volunteered) / (surfaced via probe) attribution."""
    matched = ev.get("matched_points", [])
    missed  = ev.get("missed_points",  [])
    sources = ev.get("point_sources",  {})
    quality = ev.get("quality_ratings",{})
    quotes  = ev.get("matched_point_quotes", {})  # absent for keyword-scoring / pre-existing reports

    _quality_label = {0: "stated only", 1: "partial explanation", 2: "full explanation"}

    if matched:
        lines.append("**Key points covered:**")
        for p in matched:
            source = sources.get(p, "recall")
            attr   = "(volunteered)" if source == "recall" else "(surfaced via probe)"
            qlabel = _quality_label.get(quality.get(p, 0), "stated only")
            lines.append(f"  - {p} — {attr} — _{qlabel}_")
            quote = quotes.get(p)
            if quote:
                lines.append(f"    > \"{quote}\"")
        lines.append("")

    if missed:
        lines.append("**Key points missed:** " + ", ".join(missed))
        lines.append("")


def _append_fr_key_points(lines, ev):
    """Write FR key points (construct/exemplar brief, Part D) -- one bullet per matched
    point showing HOW it was matched (known exemplar vs. novel equivalent pending admin
    review) and its full evidence. Multi-span evidence is shown together as one combined
    block, never implying the response was itemized when it wasn't -- both single- and
    multi-span matches are full matches, only match_type is a meaningful distinction here.
    """
    matched = ev.get("matched_points", [])
    missed  = ev.get("missed_points", [])

    if matched:
        lines.append("**Key points covered:**")
        for m in matched:
            if not isinstance(m, dict):
                lines.append(f"  - {m} — _matched_")  # legacy flat-string fallback
                continue
            construct = m.get("construct", "")
            if m.get("match_type") == "novel_equivalent":
                tag = "novel equivalent (pending admin review)"
            elif m.get("matched_exemplar"):
                tag = 'matched: known exemplar "' + m["matched_exemplar"] + '"'
            else:
                tag = "matched"
            lines.append(f"  - {construct} — _{tag}_")
            for span in m.get("evidence_spans", []):
                lines.append(f'    > "{span}"')
            if m.get("functional_justification"):
                lines.append("    _Justification:_ " + m["functional_justification"])
        lines.append("")

    if missed:
        labels = [m.get("construct", "") if isinstance(m, dict) else m for m in missed]
        lines.append("**Key points missed:** " + ", ".join(labels))
        lines.append("")


def _instructor_summary_fallback(overall_pct, coverage_pct, quality_pct, matched, missed):
    """Deterministic Instructor Summary built from data already computed in Python.

    Used when the LLM-generated summary fails to parse (or the call itself fails) --
    a report should never render a blank Instructor Summary section. Not a substitute
    for the LLM's analysis, just a graceful-degradation path, mirroring the fallback
    philosophy already used elsewhere (keyword scoring, "not_assessed" for
    revision-toward-quality).
    """
    parts = [f"Overall score: {overall_pct}."]
    if coverage_pct is not None:
        parts.append(f"Coverage: {coverage_pct}.")
    if quality_pct is not None:
        parts.append(f"Quality: {quality_pct}.")
    if matched:
        parts.append("Key points addressed: " + ", ".join(matched) + ".")
    if missed:
        parts.append("Key points missed: " + ", ".join(missed) + ".")
    parts.append("See detailed feedback above.")
    return {
        "overall_assessment": " ".join(parts),
        "learning_gaps": list(missed),
        "recommendations": [],
    }


def _generate_instructor_summary(model, api_key, base_url, summary_system, summary_prompt,
                                 fallback_builder, context_label):
    """Run the instructor-summary LLM call and normalise its output.

    On any failure -- the call itself raising, or _extract_json() failing to parse a
    malformed/truncated response -- falls back to fallback_builder() (a deterministic,
    templated summary) rather than letting a blank/garbled section reach the report.
    The failure is always logged (see _extract_json's own logging too), so silent
    failures stay visible even though the user-facing report degrades gracefully.
    """
    try:
        raw = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
    except Exception as e:
        logger.warning("[reports] instructor summary LLM call failed (%s): %s", context_label, e)
        return fallback_builder()

    instructor = _extract_json(raw)
    if not instructor or not instructor.get("overall_assessment"):
        logger.warning(
            "[reports] instructor summary parsing failed or incomplete (%s) -- raw response: %.200s",
            context_label, raw,
        )
        return fallback_builder()

    if isinstance(instructor.get("overall_assessment"), list):
        instructor["overall_assessment"] = " ".join(instructor["overall_assessment"])
    instructor["learning_gaps"]   = [str(g) for g in instructor.get("learning_gaps", []) if g]
    instructor["recommendations"] = [str(r) for r in instructor.get("recommendations", []) if r]
    return instructor


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(session, model, api_key, base_url, output_dir, thinking_profile=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = output_dir / ("report_" + timestamp + ".md")

    summaries = []
    for scenario, evals in session.results:
        for ev in evals:
            gap_text = ", ".join(ev["gaps"]) if ev["gaps"] else "none"
            summaries.append(
                "Scenario '" + scenario["title"] + "': "
                "overall " + f"{ev['score']:.0%}" + ", "
                "coverage " + f"{ev.get('coverage_score', ev['score']):.0%}" + ", "
                "quality " + f"{ev.get('quality_score', 0):.0%}" + ". "
                "Gaps: " + gap_text + "."
            )

    summary_prompt = (
        "A learner completed " + str(len(session.results)) + " scenario(s) "
        "with an average overall score of " + f"{session.average_score():.0%}" + ".\n\n"
        "Per-scenario results:\n"
        + "\n".join(summaries) + "\n\n"
        "Return this JSON:\n"
        "{\n"
        '  "overall_assessment": "<2-3 sentence summary of the learner\'s performance>",\n'
        '  "learning_gaps": [<specific concepts or skills the learner struggled with>],\n'
        '  "recommendations": [<concrete steps the instructor can take to address the gaps>]\n'
        "}"
    )

    summary_system = (
        "You are an expert instructional designer reviewing assessment results. "
        "Write clear, actionable recommendations for the instructor. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    def _fallback():
        all_evals = [ev for _, evals in session.results for ev in evals]
        n = len(all_evals)
        avg_coverage = sum(ev.get("coverage_score", ev.get("score", 0)) for ev in all_evals) / n if n else 0.0
        avg_quality  = sum(ev.get("quality_score", 0) for ev in all_evals) / n if n else 0.0
        matched_pool = sorted({p for ev in all_evals for p in ev.get("matched_points", [])})
        missed_pool  = sorted({p for ev in all_evals for p in ev.get("missed_points", [])})
        return _instructor_summary_fallback(
            f"{session.average_score():.0%}", f"{avg_coverage:.0%}", f"{avg_quality:.0%}",
            matched_pool, missed_pool,
        )

    instructor = _generate_instructor_summary(
        model, api_key, base_url, summary_system, summary_prompt, _fallback, "scenario session summary"
    )

    lines = []
    lines.append("# Performative Assessment — Instructor Report")
    lines.append("")
    lines.append("**Date:** " + datetime.now().strftime("%Y-%m-%d %H:%M") + "  ")
    lines.append("**Model:** " + model + "  ")
    lines.append("**Scenarios completed:** " + str(len(session.results)) + "  ")
    lines.append("**Average score:** " + f"{session.average_score():.0%}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (scenario, evals) in enumerate(session.results, 1):
        lines.append("## Scenario " + str(i) + ": " + scenario["title"])
        lines.append("")
        if scenario["description"]:
            lines.append("_" + scenario["description"] + "_")
            lines.append("")
        if scenario["constraints"]:
            lines.append("**Constraints:**")
            for c in scenario["constraints"]:
                lines.append("  - " + c)
            lines.append("")

        for ev in evals:
            # ── Scores (three-dimensional) ────────────────────────────────
            _append_scores(lines, ev)

            # ── Recall transcript ─────────────────────────────────────────
            recall_txt = ev.get("recall_transcript") or ev.get("transcript", "")
            if recall_txt:
                lines.append("**Recall transcript (free recall phase):**")
                lines.append("")
                for line in recall_txt.splitlines():
                    lines.append(("> " + line) if line else ">")
                lines.append("")

            # ── Probe transcript ──────────────────────────────────────────
            probe_txt = ev.get("probe_transcript", "")
            if probe_txt:
                lines.append("**Probing Phase transcript:**")
                lines.append("")
                for line in probe_txt.splitlines():
                    lines.append(("> " + line) if line else ">")
                lines.append("")

            lines.append("**Expert guidance:**")
            lines.append("> " + ev["expert_answer"]["answer"])
            lines.append("")

            if ev["strengths"]:
                lines.append("**Strengths:**")
                for s in ev["strengths"]:
                    lines.append("  - " + s)
                lines.append("")
            if ev["gaps"]:
                lines.append("**Gaps:**")
                for g in ev["gaps"]:
                    lines.append("  - " + g)
                lines.append("")

            _append_key_points_with_attribution(lines, ev)

            if ev["feedback"]:
                lines.append("**Feedback for learner:** _" + ev["feedback"] + "_")
                lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Instructor Summary")
    lines.append("")
    lines.append(instructor.get("overall_assessment", ""))
    lines.append("")

    if instructor.get("learning_gaps"):
        lines.append("**Learning gaps identified:**")
        for gap in instructor["learning_gaps"]:
            lines.append("  - " + gap)
        lines.append("")
    if instructor.get("recommendations"):
        lines.append("**Recommendations:**")
        for rec in instructor["recommendations"]:
            lines.append("  - " + rec)
        lines.append("")

    if thinking_profile:
        _append_thinking_profile(lines, thinking_profile)

    lines.append("---")
    lines.append("")
    lines.append(_INFERENCE_BOUNDARY)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_fr_report(prompt_data, evaluation, model, api_key, base_url, output_dir,
                        thinking_profile=None, process_overlay=None, ai_assistance=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = output_dir / ("fr_report_" + timestamp + ".md")

    ev        = evaluation
    score_pct = f"{ev['score']:.0%}"
    gap_text  = ", ".join(ev.get("gaps", [])) or "none"

    summary_prompt = (
        "A learner completed a free-response writing task: \"" + prompt_data["title"] + "\"\n\n"
        "Score: " + score_pct + "\n"
        "Gaps: " + gap_text + "\n\n"
        "Return this JSON:\n"
        "{\n"
        '  "overall_assessment": "<2-3 sentence summary of the learner\'s written performance>",\n'
        '  "learning_gaps": [<specific concepts or skills the learner struggled with>],\n'
        '  "recommendations": [<concrete steps the instructor can take to address the gaps>]\n'
        "}"
    )

    summary_system = (
        "You are an expert instructional designer reviewing a written assessment. "
        "Write clear, actionable recommendations for the instructor. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    def _fallback():
        matched_labels = [m.get("construct", "") if isinstance(m, dict) else m for m in ev.get("matched_points", [])]
        missed_labels  = [m.get("construct", "") if isinstance(m, dict) else m for m in ev.get("missed_points", [])]
        return _instructor_summary_fallback(
            score_pct, ev.get("coverage_score"), ev.get("quality_score"),
            matched_labels, missed_labels,
        )

    instructor = _generate_instructor_summary(
        model, api_key, base_url, summary_system, summary_prompt, _fallback, "FR summary"
    )

    lines = []
    lines.append("# Free Response Assessment — Instructor Report")
    lines.append("")
    lines.append("**Date:** " + datetime.now().strftime("%Y-%m-%d %H:%M") + "  ")
    lines.append("**Prompt:** " + prompt_data["title"] + "  ")
    lines.append("**Model:** " + model + "  ")
    lines.append("**Score:** " + score_pct)
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Prompt")
    lines.append("")
    lines.append(prompt_data.get("prompt_text", prompt_data.get("description", "")))
    lines.append("")

    if prompt_data.get("constraints"):
        lines.append("**Constraints:**")
        for c in prompt_data["constraints"]:
            lines.append("  - " + c)
        lines.append("")

    lines.append("## Learner's Submission")
    lines.append("")
    for line in ev["text"].splitlines():
        lines.append("> " + line if line else ">")
    lines.append("")

    ai_assistance = ai_assistance or {}
    if ai_assistance:
        used = "Yes" if ai_assistance.get("used") == "yes" else "No"
        lines.append("## AI Assistance Declaration")
        lines.append("")
        lines.append("**Used AI assistance:** " + used)
        notes = (ai_assistance.get("notes") or "").strip()
        if notes:
            lines.append("")
            lines.append("**Learner description:**")
            for line in notes.splitlines():
                lines.append("> " + line if line else ">")
        lines.append("")

    lines.append("## Evaluation")
    lines.append("")
    lines.append("**Score:** " + score_pct)
    lines.append("")

    if ev.get("feedback"):
        lines.append("**Feedback for learner:** _" + ev["feedback"] + "_")
        lines.append("")
    if ev.get("strengths"):
        lines.append("**Strengths:**")
        for s in ev["strengths"]:
            lines.append("  - " + s)
        lines.append("")
    if ev.get("gaps"):
        lines.append("**Gaps:**")
        for g in ev["gaps"]:
            lines.append("  - " + g)
        lines.append("")
    _append_fr_key_points(lines, ev)

    if ev.get("expert_answer", {}).get("answer"):
        lines.append("**Expert reference answer:**")
        lines.append("> " + ev["expert_answer"]["answer"])
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Instructor Summary")
    lines.append("")
    lines.append(instructor.get("overall_assessment", ""))
    lines.append("")

    if instructor.get("learning_gaps"):
        lines.append("**Learning gaps identified:**")
        for gap in instructor["learning_gaps"]:
            lines.append("  - " + gap)
        lines.append("")
    if instructor.get("recommendations"):
        lines.append("**Recommendations:**")
        for rec in instructor["recommendations"]:
            lines.append("  - " + rec)
        lines.append("")

    if process_overlay:
        _append_process_overlay(lines, process_overlay)

    if thinking_profile:
        _append_thinking_profile(lines, thinking_profile)

    lines.append("---")
    lines.append("")
    lines.append(_INFERENCE_BOUNDARY)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
