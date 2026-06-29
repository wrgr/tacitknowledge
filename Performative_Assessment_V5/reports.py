"""
reports.py — Markdown report generation for scenario sessions and free-response prompts.
"""

from datetime import datetime
from pathlib import Path

from llm import llm_chat, _extract_json

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


def _append_key_points_with_attribution(lines, ev):
    """Write key points with (volunteered) / (surfaced via probe) attribution."""
    matched = ev.get("matched_points", [])
    missed  = ev.get("missed_points",  [])
    sources = ev.get("point_sources",  {})
    quality = ev.get("quality_ratings",{})

    _quality_label = {0: "stated only", 1: "partial explanation", 2: "full explanation"}

    if matched:
        lines.append("**Key points covered:**")
        for p in matched:
            source = sources.get(p, "recall")
            attr   = "(volunteered)" if source == "recall" else "(surfaced via probe)"
            qlabel = _quality_label.get(quality.get(p, 0), "stated only")
            lines.append(f"  - {p} — {attr} — _{qlabel}_")
        lines.append("")

    if missed:
        lines.append("**Key points missed:** " + ", ".join(missed))
        lines.append("")


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

    try:
        raw        = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
        instructor = _extract_json(raw)
        if not instructor:
            instructor = {"overall_assessment": raw, "learning_gaps": [], "recommendations": []}
    except Exception:
        instructor = {"overall_assessment": "Summary unavailable — no LLM API key configured.",
                      "learning_gaps": [], "recommendations": []}

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


def generate_fr_report(prompt_data, evaluation, model, api_key, base_url, output_dir, thinking_profile=None):
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

    try:
        raw        = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
        instructor = _extract_json(raw)
        if not instructor:
            instructor = {"overall_assessment": raw, "learning_gaps": [], "recommendations": []}
    except Exception:
        instructor = {"overall_assessment": "Summary unavailable — no LLM API key configured.",
                      "learning_gaps": [], "recommendations": []}

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
    if ev.get("matched_points"):
        lines.append("**Key points covered:** " + ", ".join(ev["matched_points"]))
        lines.append("")
    if ev.get("missed_points"):
        lines.append("**Key points missed:** " + ", ".join(ev["missed_points"]))
        lines.append("")
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

    if thinking_profile:
        _append_thinking_profile(lines, thinking_profile)

    lines.append("---")
    lines.append("")
    lines.append(_INFERENCE_BOUNDARY)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
