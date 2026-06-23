"""
reports.py — Markdown report generation for scenario sessions and free-response prompts.
Writes a Markdown file to the reports/ folder with the full session results.
"""

from datetime import datetime
from pathlib import Path

from llm import llm_chat, _extract_json


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _append_thinking_profile(lines, thinking_profile):
    # skip the whole section if the LLM didn't return either framework result
    if not (thinking_profile.get("honey_mumford_style") or thinking_profile.get("solo_level")):
        return

    lines.append("## Learner Thinking Profile")
    lines.append("")

    hm    = thinking_profile.get("honey_mumford_style", "")
    hm_ev = thinking_profile.get("honey_mumford_evidence", "")
    if hm:
        lines.append("**Honey & Mumford style:** " + hm)
        if hm_ev:
            lines.append("_" + hm_ev + "_")
        lines.append("")

    solo    = thinking_profile.get("solo_level", "")
    solo_ev = thinking_profile.get("solo_evidence", "")
    if solo:
        lines.append("**SOLO level:** " + solo)
        if solo_ev:
            lines.append("_" + solo_ev + "_")
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


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(session, model, api_key, base_url, output_dir, thinking_profile=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)  # create the folder if it doesn't exist

    # build a filename like "report_20260618_162039.md"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = output_dir / ("report_" + timestamp + ".md")

    # build one summary line per evaluation to send to the LLM for the instructor narrative
    summaries = []
    for scenario, evals in session.results:
        for ev in evals:
            gap_text = ", ".join(ev["gaps"]) if ev["gaps"] else "none"
            summaries.append(
                "Scenario '" + scenario["title"] + "': "
                "score " + f"{ev['score']:.0%}" + ". "
                "Gaps: " + gap_text + "."
            )

    summary_prompt = (
        "A learner completed " + str(len(session.results)) + " scenario(s) "
        "with an average score of " + f"{session.average_score():.0%}" + ".\n\n"
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

    raw        = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
    instructor = _extract_json(raw)
    if not instructor:
        # if JSON parsing fails, use the raw text as the assessment so the report still saves
        instructor = {"overall_assessment": raw, "learning_gaps": [], "recommendations": []}

    # build the Markdown report line by line
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
            lines.append("**Score:** " + f"{ev['score']:.0%}")
            lines.append("")
            lines.append("**Conversation transcript:**")
            lines.append("")
            for line in ev["transcript"].splitlines():
                lines.append(("> " + line) if line else ">")  # indent each line as a Markdown blockquote
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
            if ev["matched_points"]:
                lines.append("**Key points covered:** " + ", ".join(ev["matched_points"]))
                lines.append("")
            if ev["missed_points"]:
                lines.append("**Key points missed:** " + ", ".join(ev["missed_points"]))
                lines.append("")
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

    path.write_text("\n".join(lines), encoding="utf-8")
    return path  # return the path so the caller can tell the user where the file was saved


def generate_fr_report(prompt_data, evaluation, model, api_key, base_url, output_dir, thinking_profile=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # build a filename like "fr_report_20260618_162039.md"
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

    raw        = llm_chat(model, summary_system, summary_prompt, api_key, base_url)
    instructor = _extract_json(raw)
    if not instructor:
        # if JSON parsing fails, use the raw text as the assessment so the report still saves
        instructor = {"overall_assessment": raw, "learning_gaps": [], "recommendations": []}

    # build the Markdown report line by line
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
        lines.append("> " + line if line else ">")  # indent each line as a Markdown blockquote
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

    path.write_text("\n".join(lines), encoding="utf-8")
    return path  # return the path so the caller can tell the user where the file was saved
