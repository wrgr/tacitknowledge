import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import reports
import report_parser


class CoverageCalibrationNoteTests(unittest.TestCase):
    """fr_recall_omission_fix brief, Part A: the Coverage Score calibration note must be
    visible right next to the score itself, not buried in a single end-of-report note."""

    def _generate(self, evaluation):
        prompt_data = {"id": "p1", "title": "Explain the concept", "prompt_text": "Explain it."}
        with tempfile.TemporaryDirectory() as tmp:
            # No network in tests -- force the deterministic fallback instructor-summary path.
            with patch.object(reports, "llm_chat", side_effect=RuntimeError("no network in tests")):
                path = reports.generate_fr_report(
                    prompt_data, evaluation, model="test-model", api_key=None,
                    base_url="http://example.invalid", output_dir=tmp,
                )
            return path.read_text(encoding="utf-8")

    def test_calibration_note_appears_immediately_after_score_line(self):
        evaluation = {
            "text": "My answer.", "score": 0.5, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
        }
        text = self._generate(evaluation)
        lines = text.splitlines()
        eval_idx = next(i for i, l in enumerate(lines) if l == "## Evaluation")
        score_idx = next(i for i in range(eval_idx, len(lines)) if lines[i].startswith("**Score:**"))
        self.assertIn(reports._FR_COVERAGE_CALIBRATION_NOTE, lines[score_idx + 1])

    def test_calibration_note_licenses_no_more_certainty_than_a_missed_point_warrants(self):
        # The note itself must not claim a missed point proves absent knowledge.
        self.assertIn("not conclusive evidence of a gap", reports._FR_COVERAGE_CALIBRATION_NOTE)

    def test_fallback_summary_never_renders_missing_coverage_as_none(self):
        evaluation = {
            "text": "My answer.", "score": 0.5, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": ["key idea"],
        }
        text = self._generate(evaluation)

        self.assertIn("Coverage: 50%.", text)
        self.assertNotIn("Coverage: None", text)


class ClosingNudgeReportRenderingTests(unittest.TestCase):
    """fr_recall_omission_fix brief, Part B5: closing_nudge_used is report-facing context,
    never a claim, so it renders plainly without an evidence-model hedge."""

    def test_renders_yes_when_nudge_was_used(self):
        lines = []
        reports._append_process_overlay(lines, {"closing_nudge_used": True})
        joined = "\n".join(lines)
        self.assertIn("**Closing nudge used:** Yes", joined)

    def test_renders_no_when_nudge_was_not_used(self):
        lines = []
        reports._append_process_overlay(lines, {"closing_nudge_used": False})
        joined = "\n".join(lines)
        self.assertIn("**Closing nudge used:** No", joined)

    def test_omitted_entirely_when_absent(self):
        lines = []
        reports._append_process_overlay(lines, {"quadrant": {}})
        joined = "\n".join(lines)
        self.assertNotIn("Closing nudge used", joined)

    def test_generated_report_parser_round_trips_closing_nudge_signal(self):
        prompt_data = {"id": "p1", "title": "Explain the concept", "prompt_text": "Explain it."}
        evaluation = {
            "text": "My answer.", "score": 0.5, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(reports, "llm_chat", side_effect=RuntimeError("no network in tests")):
                path = reports.generate_fr_report(
                    prompt_data, evaluation, model="test-model", api_key=None,
                    base_url="http://example.invalid", output_dir=tmp,
                    process_overlay={"closing_nudge_used": True},
                )
            parsed = report_parser.parse_report_md(path.read_text(encoding="utf-8"))

        self.assertTrue(parsed["process_overlay"]["closing_nudge_used"])
        self.assertIn("Yes", parsed["process_overlay"]["closing_nudge_text"])


class EvidenceModelDocTests(unittest.TestCase):
    """Part A1: the evidence-model table must carry a Coverage Score row before any
    report copy is allowed to render a Coverage Score claim (per the doc's own rule)."""

    def test_coverage_score_row_present(self):
        doc = (APP_DIR / "docs" / "fr_evidence_model.md").read_text(encoding="utf-8")
        self.assertIn("Coverage Score", doc)
        self.assertIn("recall-limited by design", doc)
        self.assertIn(
            "the same omission phenomenon (unprompted recall systematically "
            "underrepresents true knowledge)",
            doc,
        )


if __name__ == "__main__":
    unittest.main()
