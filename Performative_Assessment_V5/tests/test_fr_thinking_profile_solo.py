import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import thinking
import reports
import report_parser


class DeriveFrSoloLevelTests(unittest.TestCase):
    """Part B1: derive_fr_solo_level() is a pure, deterministic reader over
    matched_points/quality_rating -- no LLM call, no randomness."""

    def test_no_matched_points_is_prestructural(self):
        result = thinking.derive_fr_solo_level({"matched_points": []})
        self.assertEqual(result["solo_level"], "Prestructural")
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["mean_quality"], 0.0)

    def test_single_matched_point_is_unistructural_regardless_of_quality(self):
        result = thinking.derive_fr_solo_level(
            {"matched_points": [{"construct": "a", "quality_rating": 2}]}
        )
        self.assertEqual(result["solo_level"], "Unistructural")
        self.assertEqual(result["matched_count"], 1)

    def test_multiple_points_low_mean_quality_is_multistructural(self):
        result = thinking.derive_fr_solo_level({
            "matched_points": [
                {"construct": "a", "quality_rating": 0},
                {"construct": "b", "quality_rating": 0},
                {"construct": "c", "quality_rating": 1},
            ],
        })
        self.assertEqual(result["solo_level"], "Multistructural")
        self.assertAlmostEqual(result["mean_quality"], 0.33, places=2)

    def test_multiple_points_high_mean_quality_is_relational(self):
        result = thinking.derive_fr_solo_level({
            "matched_points": [
                {"construct": "a", "quality_rating": 2},
                {"construct": "b", "quality_rating": 1},
                {"construct": "c", "quality_rating": 1},
            ],
        })
        self.assertEqual(result["solo_level"], "Relational")
        self.assertAlmostEqual(result["mean_quality"], 1.33, places=2)

    def test_threshold_boundary_is_inclusive_of_relational(self):
        # mean_quality exactly at SOLO_RELATIONAL_QUALITY_THRESHOLD (1.0) is Relational.
        result = thinking.derive_fr_solo_level({
            "matched_points": [
                {"construct": "a", "quality_rating": 1},
                {"construct": "b", "quality_rating": 1},
            ],
        })
        self.assertEqual(result["solo_level"], "Relational")

    def test_missing_quality_rating_defaults_to_zero(self):
        # Keyword-only scoring (no LLM) never sets quality_rating at all.
        result = thinking.derive_fr_solo_level({
            "matched_points": [{"construct": "a"}, {"construct": "b"}],
        })
        self.assertEqual(result["mean_quality"], 0.0)
        self.assertEqual(result["solo_level"], "Multistructural")

    def test_pool_members_count_toward_matched_count_same_as_standalone(self):
        # matched_points already contains both standalone key points and credited pool
        # members (scoring._fr_flatten_matchable / _compute_fr_coverage) -- no separate
        # pools handling needed here.
        result = thinking.derive_fr_solo_level({
            "matched_points": [
                {"construct": "a", "quality_rating": 2, "pool_id": None},
                {"construct": "b", "quality_rating": 2, "pool_id": "listening_techniques"},
            ],
        })
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["solo_level"], "Relational")

    def test_never_returns_extended_abstract(self):
        # Extended Abstract is categorically outside what Coverage/Quality can capture --
        # this function must never return it, no matter how many points or how high the
        # quality.
        result = thinking.derive_fr_solo_level({
            "matched_points": [{"construct": f"p{i}", "quality_rating": 2} for i in range(20)],
        })
        self.assertNotEqual(result["solo_level"], "Extended Abstract")

    def test_source_never_mentions_extended_abstract_as_a_returnable_level(self):
        # Structural check: the function body has no code path that could produce it.
        source = inspect.getsource(thinking.derive_fr_solo_level)
        self.assertNotIn("Extended Abstract", source)


class ScenarioModeUnaffectedTests(unittest.TestCase):
    """Scope guard: scenario mode's holistic LLM classifier -- LLM SOLO and
    probe_phase_improvement -- must be completely untouched by the FR fix, and
    Honey & Mumford must not exist anywhere in the codebase (removed system-wide)."""

    def test_analyse_thinking_profile_has_no_honey_mumford_but_keeps_ppi(self):
        source = inspect.getsource(thinking.analyse_thinking_profile)
        self.assertNotIn("honey_mumford", source)
        self.assertNotIn("Honey", source)
        self.assertIn("probe_phase_improvement", source)
        self.assertIn("Extended Abstract", source)

    def test_generate_report_still_uses_original_thinking_profile_renderer(self):
        source = inspect.getsource(reports.generate_report)
        self.assertIn("_append_thinking_profile(lines, thinking_profile)", source)

    def test_original_append_thinking_profile_renders_solo_and_ppi_not_honey_mumford(self):
        lines = []
        reports._append_thinking_profile(lines, {
            "solo_level": "Relational",
            "solo_confidence": "medium",
            "probe_phase_improvement": True,
            "probe_phase_improvement_note": "richer under probing",
        })
        joined = "\n".join(lines)
        self.assertIn("**SOLO level:** Relational", joined)
        self.assertIn("**Probe phase improvement:** Yes", joined)
        self.assertNotIn("Honey", joined)
        self.assertNotIn("Mumford", joined)


class FrReportRenderingTests(unittest.TestCase):
    """Parts A/B/C/E: generate_fr_report() renders only the deterministic SOLO line --
    no Honey & Mumford, no probe phase improvement, no invented confidence tag."""

    def _generate(self, thinking_profile):
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
                    thinking_profile=thinking_profile,
                )
            return path.read_text(encoding="utf-8")

    def test_renders_solo_level_and_inputs_not_confidence(self):
        text = self._generate({"solo_level": "Multistructural", "matched_count": 3, "mean_quality": 0.67})
        self.assertIn("**SOLO level:** Multistructural", text)
        self.assertIn("matched_count: 3", text)
        self.assertIn("mean_quality: 0.67", text)
        self.assertNotIn("confidence:", text)

    def test_never_renders_honey_mumford(self):
        text = self._generate({"solo_level": "Relational", "matched_count": 2, "mean_quality": 1.5})
        self.assertNotIn("Honey", text)
        self.assertNotIn("Mumford", text)

    def test_never_renders_probe_phase_improvement(self):
        text = self._generate({"solo_level": "Relational", "matched_count": 2, "mean_quality": 1.5})
        self.assertNotIn("Probe phase improvement", text)

    def test_states_extended_abstract_scope_limitation(self):
        text = self._generate({"solo_level": "Relational", "matched_count": 2, "mean_quality": 1.5})
        self.assertIn("Extended Abstract", text)

    def test_omitted_entirely_when_no_thinking_profile(self):
        prompt_data = {"id": "p1", "title": "t", "prompt_text": "p"}
        evaluation = {
            "text": "x", "score": 0.5, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(reports, "llm_chat", side_effect=RuntimeError("no network in tests")):
                path = reports.generate_fr_report(
                    prompt_data, evaluation, model="test-model", api_key=None,
                    base_url="http://example.invalid", output_dir=tmp,
                )
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("Learner Thinking Profile", text)

    def test_uses_dedicated_fr_solo_renderer_not_shared_scenario_one(self):
        source = inspect.getsource(reports.generate_fr_report)
        self.assertIn("_append_fr_solo(lines, thinking_profile)", source)
        self.assertNotIn("_append_thinking_profile(lines, thinking_profile)", source)


class ReportParserRoundTripTests(unittest.TestCase):
    """Part E: newly generated FR reports round-trip through the parser with the new
    SOLO shape (matched_count/mean_quality, no confidence/evidence)."""

    def test_new_fr_solo_round_trips(self):
        prompt_data = {"id": "p1", "title": "t", "prompt_text": "p"}
        evaluation = {
            "text": "x", "score": 0.5, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
        }
        thinking_profile = {"solo_level": "Multistructural", "matched_count": 3, "mean_quality": 0.67}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(reports, "llm_chat", side_effect=RuntimeError("no network in tests")):
                path = reports.generate_fr_report(
                    prompt_data, evaluation, model="test-model", api_key=None,
                    base_url="http://example.invalid", output_dir=tmp,
                    thinking_profile=thinking_profile,
                )
            parsed = report_parser.parse_report_md(path.read_text(encoding="utf-8"))

        tp = parsed["thinking_profile"]
        self.assertNotIn("honey_mumford", tp)
        self.assertIsNone(tp["probe_phase_improvement"])
        self.assertEqual(tp["solo"]["level"], "Multistructural")
        self.assertEqual(tp["solo"]["matched_count"], 3)
        self.assertEqual(tp["solo"]["mean_quality"], 0.67)
        self.assertEqual(tp["solo"]["confidence"], "")  # never an invented confidence tag

    def test_old_format_fr_report_still_parses_without_error(self):
        # Backward compatibility: an already-generated report on disk from before Honey
        # & Mumford was removed system-wide still has a "Honey & Mumford style" line.
        # The parser must not crash on it, must silently ignore that section (the field
        # no longer exists anywhere), and must still parse the SOLO / probe-phase-
        # improvement lines that follow it correctly.
        old_report = (
            "# Free Response Assessment — Instructor Report\n\n"
            "**Date:** 2025-01-01 10:00  \n"
            "**Prompt:** Old prompt  \n"
            "**Model:** old-model  \n"
            "**Score:** 80%\n\n---\n\n"
            "## Evaluation\n\n**Score:** 80%\n\n---\n\n"
            "## Instructor Summary\n\nOld summary text.\n\n"
            "## Learner Thinking Profile\n\n"
            "**Honey & Mumford style:** Theorist _(confidence: high)_\n"
            '  - _"explains why steps matter"_\n'
            "  > Reasoning text for H&M.\n\n"
            "**SOLO level:** Relational _(confidence: medium)_\n"
            '  - _"integrates elements"_\n'
            "  > Reasoning text for SOLO.\n\n"
            "**Probe phase improvement:** Yes\n"
            "  > Probe note text.\n\n"
            "---\n\n**Assessment scope:** blah\n"
        )
        parsed = report_parser.parse_report_md(old_report)  # must not raise
        tp = parsed["thinking_profile"]
        self.assertNotIn("honey_mumford", tp)
        self.assertEqual(tp["solo"]["level"], "Relational")
        self.assertEqual(tp["solo"]["confidence"], "medium")
        self.assertTrue(tp["probe_phase_improvement"])
        self.assertEqual(tp["probe_phase_improvement_note"], "Probe note text.")
        # Old format has no matched_count/mean_quality -- key must be absent, not zeroed.
        self.assertNotIn("matched_count", tp["solo"])


class EvidenceModelDocTests(unittest.TestCase):
    """Part D: the evidence model must carry a row for the SOLO derivation, and no
    Honey & Mumford row anywhere (H&M was removed system-wide, not just from FR)."""

    def test_solo_row_present(self):
        doc = (APP_DIR / "docs" / "fr_evidence_model.md").read_text(encoding="utf-8")
        self.assertIn("SOLO level (derived from Coverage/Quality)", doc)
        self.assertIn("Extended-Abstract-level generalization", doc)

    def test_no_honey_mumford_table_row(self):
        doc = (APP_DIR / "docs" / "fr_evidence_model.md").read_text(encoding="utf-8")
        table_lines = [l for l in doc.splitlines() if l.startswith("|")]
        self.assertFalse(any("Honey" in l for l in table_lines))


if __name__ == "__main__":
    unittest.main()
