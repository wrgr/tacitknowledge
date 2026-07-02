import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import writing_process as wp


class WritingProcessCalibrationTests(unittest.TestCase):
    def test_pause_plus_nearby_heavy_revision_creates_difficulty_point(self):
        process_log = {
            "pause_events": [
                {"timestamp_s": 45, "duration_s": 12, "char_position": 80},
            ],
            "revision_events": [
                {
                    "timestamp_s": 70,
                    "char_position": 95,
                    "removed_text": "weak unsupported claim",
                    "inserted_text": "because the switch opens the current path",
                },
            ],
        }

        points = wp.find_difficulty_points(process_log, "x" * 240)

        self.assertEqual(len(points), 1)
        self.assertIn("heavy rework", points[0]["note"])
        self.assertIn("alternative_interpretation", points[0])

    def test_pause_without_nearby_heavy_revision_is_not_difficulty_point(self):
        process_log = {
            "pause_events": [
                {"timestamp_s": 45, "duration_s": 12, "char_position": 80},
            ],
            "revision_events": [
                {
                    "timestamp_s": 200,
                    "char_position": 220,
                    "removed_text": "typo",
                    "inserted_text": "type",
                },
            ],
        }

        self.assertEqual(wp.find_difficulty_points(process_log, "x" * 240), [])

    def test_large_unrevised_paste_creates_elevated_authenticity_signal(self):
        process_log = {
            "paste_events": [
                {"timestamp_s": 10, "char_position": 0, "paste_length": 260},
            ],
            "revision_events": [],
        }
        essay_text = "x" * 500

        authenticity = wp.compute_authenticity(process_log, essay_text)

        self.assertEqual(authenticity["level"], "elevated")
        self.assertGreater(authenticity["pasted_fraction"], 0.4)
        self.assertTrue(authenticity["alternative_interpretations"])

    def test_confidence_drop_creates_confidence_collapse_signal(self):
        calibration = wp.compute_confidence_calibration(8, 5)

        self.assertEqual(calibration["finding"], "confidence_collapse")
        self.assertEqual(calibration["confidence_delta"], -3)
        self.assertEqual(calibration["confidence"], "moderate-high")

    def test_quadrant_strong_product_effortful_process(self):
        effort_profile = {
            "revision_density": wp.EFFORTFUL_REVISION_DENSITY,
            "pause_to_writing_ratio": 0.0,
        }
        authenticity = {"level": "none"}

        quadrant = wp.classify_quadrant(
            product_score=0.85,
            effort_profile=effort_profile,
            authenticity=authenticity,
            trajectory="iterative",
        )

        self.assertEqual(quadrant["label"], "genuine_engaged_reasoning")

    def test_quadrant_strong_product_frictionless_process(self):
        effort_profile = {
            "revision_density": 0.0,
            "pause_to_writing_ratio": 0.0,
        }
        authenticity = {"level": "none"}

        quadrant = wp.classify_quadrant(
            product_score=0.85,
            effort_profile=effort_profile,
            authenticity=authenticity,
            trajectory="linear",
        )

        self.assertEqual(quadrant["label"], "authenticity_review")

    def test_full_overlay_without_llm_keeps_revision_judgment_not_assessed(self):
        process_log = {
            "snapshots": [
                {"timestamp_s": 1, "text": "The circuit works."},
                {"timestamp_s": 90, "text": "The circuit works because current has a closed path."},
            ],
            "pause_events": [
                {"timestamp_s": 30, "duration_s": 10, "char_position": 18},
            ],
            "revision_events": [
                {
                    "timestamp_s": 55,
                    "char_position": 18,
                    "removed_text": "works",
                    "inserted_text": "works because current has a closed path",
                },
            ],
            "paste_events": [],
        }

        overlay = wp.analyze_writing_process(
            process_log,
            {"total_time_s": 100},
            "The circuit works because current has a closed path.",
            product_score=0.8,
            use_llm=False,
        )

        self.assertEqual(overlay["revision_toward_quality"]["rating"], "not_assessed")
        self.assertEqual(len(overlay["difficulty_points"]), 1)
        self.assertEqual(overlay["authenticity"]["level"], "none")


if __name__ == "__main__":
    unittest.main()
