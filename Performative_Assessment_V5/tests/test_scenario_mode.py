import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import runner
import scoring
from session import Session

# _analyse_coverage's cached_evaluative_call (runner.py) has no bypass_cache
# parameter of its own (unlike scoring.score_with_llm/_extract_evidence, which
# we pass bypass_cache=True explicitly everywhere below) -- this environment
# variable is llm.py's documented alternate bypass, so probe-queue tests never
# touch the on-disk eval-response cache regardless.
os.environ.setdefault("DISABLE_EVAL_CACHE", "1")

SCENARIOS_DIR = APP_DIR / "scenarios"


def _load_scenario(filename):
    return json.loads((SCENARIOS_DIR / filename).read_text(encoding="utf-8"))


def _minimal_scenario(probe_bank, key_points=None, rubric=None, scoring_weights=None):
    """A small, fully self-contained scenario dict for probe-queue tests that need
    precise control over probe_bank composition, isolated from the real fixtures'
    incidental scoring/complexity."""
    scenario = {
        "id": "synthetic_scenario",
        "situation": "Test situation.",
        "user_role": "learner",
        "constraints": [],
        "expert_answers": [{
            "id": "expert_001",
            "answer": "expert answer",
            "key_points": key_points or [],
            "rubric": rubric or {},
        }],
        "probe_bank": probe_bank,
    }
    if scoring_weights:
        scenario["scoring_weights"] = scoring_weights
    return scenario


def _make_runner(scenario, recall_text):
    r = runner.ScenarioRunner(scenario, model="test-model", api_key=None, base_url="http://example.invalid")
    r.recall_history = [{"role": "user", "content": recall_text}]
    return r


def _coverage_response(statuses, disordered=False):
    """A canned runner._analyse_coverage LLM response. statuses is indexed like
    the probe_bank passed to _analyse_coverage / _build_probe_queue."""
    return json.dumps({
        "coverage": {str(i): s for i, s in enumerate(statuses)},
        "recall_disordered": disordered,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Part A: fixture shape sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioFixtureShapeTests(unittest.TestCase):
    """Part A: confirms the three real scenario fixtures still exercise the three
    documented probe-queue code paths this suite characterizes (authored bank,
    fully dynamic, partially authored). If one of these ever fails, the fixture
    files changed shape and every other test in this module may be exercising
    the wrong path."""

    def test_changing_tire_has_full_authored_context(self):
        s = _load_scenario("Changing_Tire.json")
        for key in ("id", "probe_bank", "decision_points", "failure_modes", "edge_cases", "scoring_weights"):
            self.assertIn(key, s)

    def test_example_has_no_authored_probe_bank_or_supplementary_context(self):
        s = _load_scenario("example.json")
        for key in ("probe_bank", "decision_points", "failure_modes", "edge_cases", "scoring_weights"):
            self.assertNotIn(key, s)

    def test_morning_dog_care_has_probe_bank_but_no_supplementary_context(self):
        s = _load_scenario("morning_dog_care_routine.json")
        self.assertIn("probe_bank", s)
        for key in ("decision_points", "failure_modes", "edge_cases", "scoring_weights"):
            self.assertNotIn(key, s)


# ─────────────────────────────────────────────────────────────────────────────
# Part B: probe queue builder
# ─────────────────────────────────────────────────────────────────────────────

class ProbeQueueCoverageClassificationTests(unittest.TestCase):
    """Part B1: covered/partial/missing classification. The LLM call itself is
    mocked -- these tests characterize how the code plumbs the LLM's verdict
    through to the coverage map, not whether an LLM would classify a given
    transcript correctly (that judgment is out of reach for a unit test)."""

    def test_coverage_map_reflects_covered_partial_missing_per_probe(self):
        scenario = _load_scenario("Changing_Tire.json")
        probe_bank = scenario["probe_bank"]  # 6 probes: sequencing, how, rationale, decision, error, edge_case
        r = _make_runner(scenario, "I would loosen the nuts and use the jack.")
        statuses = ["covered", "partial", "missing", "missing", "missing", "missing"]

        with patch.object(runner, "llm_chat_json", return_value=_coverage_response(statuses)):
            coverage_map, disordered = r._analyse_coverage(r.recall_transcript, probe_bank, [])

        probe_texts = [p["probe_text"] for p in probe_bank]
        self.assertEqual(coverage_map[probe_texts[0]], "covered")
        self.assertEqual(coverage_map[probe_texts[1]], "partial")
        self.assertEqual(coverage_map[probe_texts[2]], "missing")
        self.assertFalse(disordered)

    def test_missing_llm_verdict_for_an_index_defaults_to_missing(self):
        # _analyse_coverage falls back to "missing" for any probe index the LLM's
        # JSON coverage dict didn't include (cov.get(str(i), "missing")).
        scenario = _load_scenario("Changing_Tire.json")
        probe_bank = scenario["probe_bank"]
        r = _make_runner(scenario, "recall text")
        partial_response = json.dumps({"coverage": {"0": "covered"}, "recall_disordered": False})

        with patch.object(runner, "llm_chat_json", return_value=partial_response):
            coverage_map, _ = r._analyse_coverage(r.recall_transcript, probe_bank, [])

        probe_texts = [p["probe_text"] for p in probe_bank]
        self.assertEqual(coverage_map[probe_texts[0]], "covered")
        self.assertEqual(coverage_map[probe_texts[1]], "missing")


class ProbeQueueDisorderedGatingTests(unittest.TestCase):
    """Part B2: the sequencing probe is only ever considered when recall was
    flagged genuinely disordered -- gated in runner._build_probe_queue on the
    recall_disordered flag returned by _analyse_coverage."""

    def test_sequencing_probe_present_only_when_disordered(self):
        probe_bank = [
            {"probe_type": "sequencing", "target_key_point": "order", "probe_text": "seq?", "success_criteria": ""},
            {"probe_type": "how",        "target_key_point": "other", "probe_text": "how?", "success_criteria": ""},
        ]
        scenario = _minimal_scenario(probe_bank)

        r_disordered = _make_runner(scenario, "recall text")
        with patch.object(runner, "llm_chat_json",
                          return_value=_coverage_response(["missing", "missing"], disordered=True)):
            queue_disordered = r_disordered._build_probe_queue()

        r_ordered = _make_runner(scenario, "recall text")
        with patch.object(runner, "llm_chat_json",
                          return_value=_coverage_response(["missing", "missing"], disordered=False)):
            queue_ordered = r_ordered._build_probe_queue()

        self.assertIn("sequencing", [c["probe_type"] for c in queue_disordered])
        self.assertNotIn("sequencing", [c["probe_type"] for c in queue_ordered])


class ProbeQueuePriorityAndSortTests(unittest.TestCase):
    """Part B3: priority scoring uses the documented type-priority values plus a
    coverage-gap bonus (missing > partial > covered), and the final queue is
    sorted by that combined score, descending."""

    def test_documented_priority_constants(self):
        self.assertEqual(runner._PROBE_TYPE_PRIORITY, {
            "how": 0.90, "decision": 0.80, "error": 0.70,
            "edge_case": 0.65, "rationale": 0.55, "sequencing": 0.50,
        })

    def test_priority_scores_and_descending_sort_on_authored_bank(self):
        scenario = _load_scenario("Changing_Tire.json")  # 6 probes; also exercises the size cap (Part B6)
        r = _make_runner(scenario, "recall text")
        # index: 0 sequencing, 1 how, 2 rationale, 3 decision, 4 error, 5 edge_case
        statuses = ["missing", "partial", "missing", "covered", "missing", "partial"]

        with patch.object(runner, "llm_chat_json", return_value=_coverage_response(statuses, disordered=True)):
            queue = r._build_probe_queue()

        by_type = {c["probe_type"]: c for c in queue}
        self.assertAlmostEqual(by_type["how"]["priority_score"], 0.90 + 0.04, places=3)       # partial
        self.assertAlmostEqual(by_type["decision"]["priority_score"], 0.80 + 0.0, places=3)   # covered -- no bonus
        self.assertAlmostEqual(by_type["error"]["priority_score"], 0.70 + 0.08, places=3)     # missing
        self.assertAlmostEqual(by_type["edge_case"]["priority_score"], 0.65 + 0.04, places=3)  # partial

        scores = [c["priority_score"] for c in queue]
        self.assertEqual(scores, sorted(scores, reverse=True))


class ProbeQueuePerTypeCapTests(unittest.TestCase):
    """Part B4: error and edge_case probe types are capped at one occurrence
    each in a built queue."""

    def test_error_and_edge_case_capped_at_one_each(self):
        probe_bank = [
            {"probe_type": "how",       "target_key_point": "a", "probe_text": "h?",  "success_criteria": ""},
            {"probe_type": "decision",  "target_key_point": "b", "probe_text": "d?",  "success_criteria": ""},
            {"probe_type": "error",     "target_key_point": "c", "probe_text": "e1?", "success_criteria": ""},
            {"probe_type": "error",     "target_key_point": "d", "probe_text": "e2?", "success_criteria": ""},
            {"probe_type": "edge_case", "target_key_point": "e", "probe_text": "x1?", "success_criteria": ""},
            {"probe_type": "edge_case", "target_key_point": "f", "probe_text": "x2?", "success_criteria": ""},
        ]
        # After per-type dedup this leaves 4 unique-type candidates (how, decision,
        # error, edge_case) -- assumes MAX_PROBE_QUEUE_SIZE stays >= 4, true today
        # and expected to only increase (Phase 1.2).
        scenario = _minimal_scenario(probe_bank)
        r = _make_runner(scenario, "recall text")

        with patch.object(runner, "llm_chat_json", return_value=_coverage_response(["missing"] * 6)):
            queue = r._build_probe_queue()

        types = [c["probe_type"] for c in queue]
        self.assertLessEqual(types.count("error"), 1)
        self.assertLessEqual(types.count("edge_case"), 1)
        self.assertIn("error", types)      # the cap kept exactly one, not zero
        self.assertIn("edge_case", types)


class ProbeFollowupEligibilityTests(unittest.TestCase):
    """Part B4: only how/decision-type probes are ever marked eligible for a
    follow-up question, exercised via the real _probe_respond code path with
    _generate_next_probe_turn mocked out (it would otherwise require an LLM call
    unrelated to the eligibility logic under test)."""

    def _can_followup_for(self, probe_type):
        scenario = _minimal_scenario(probe_bank=[])
        r = runner.ScenarioRunner(scenario, model="test-model", api_key=None, base_url="http://example.invalid")
        r.phase = "probing"
        r.probe_queue = [{
            "probe_type": probe_type, "target_key_point": "x", "priority_score": 0.9,
            "probe_text": "q?", "success_criteria": "", "status": "pending",
            "exchange": [], "_followup_sent": False,
        }]
        r.probe_index = 0

        captured = {}

        def fake_turn(user_input, current_probe, can_followup, next_probe):
            captured["can_followup"] = can_followup
            return {"adequate": False, "action": "advance", "message": "closing"}

        with patch.object(r, "_generate_next_probe_turn", side_effect=fake_turn):
            r._probe_respond("my answer")
        return captured["can_followup"]

    def test_how_and_decision_are_eligible_for_followup(self):
        self.assertTrue(self._can_followup_for("how"))
        self.assertTrue(self._can_followup_for("decision"))

    def test_other_types_are_never_eligible_for_followup(self):
        for ptype in ("error", "edge_case", "rationale", "sequencing"):
            self.assertFalse(self._can_followup_for(ptype))


class ProbeQueueRationaleFoldingTests(unittest.TestCase):
    """Part B5: a rationale-type candidate is folded into an existing how probe
    targeting the same point (dropped from the candidate list entirely -- not
    textually merged into the how probe -- see runner._build_probe_queue's
    how_targets check), and only added standalone when no how probe shares its
    target."""

    def test_rationale_sharing_a_how_targets_point_is_folded_away(self):
        probe_bank = [
            {"probe_type": "how",       "target_key_point": "shared point",    "probe_text": "how?",           "success_criteria": ""},
            {"probe_type": "rationale", "target_key_point": "shared point",    "probe_text": "why-shared?",    "success_criteria": ""},
            {"probe_type": "rationale", "target_key_point": "different point", "probe_text": "why-different?", "success_criteria": ""},
        ]
        scenario = _minimal_scenario(probe_bank)
        r = _make_runner(scenario, "recall text")

        with patch.object(runner, "llm_chat_json", return_value=_coverage_response(["missing"] * 3)):
            queue = r._build_probe_queue()

        rationale_texts = [c["probe_text"] for c in queue if c["probe_type"] == "rationale"]
        self.assertEqual(rationale_texts, ["why-different?"])


class ProbeQueueSizeCapTests(unittest.TestCase):
    """Part B6: the queue is truncated to MAX_PROBE_QUEUE_SIZE -- referenced via
    the constant everywhere, never the current literal value, so this test
    doesn't need updating when Phase 1.2 changes it."""

    def test_queue_truncated_to_max_probe_queue_size(self):
        n = runner.MAX_PROBE_QUEUE_SIZE + 2
        probe_bank = [
            {"probe_type": "decision", "target_key_point": f"t{i}", "probe_text": f"q{i}?", "success_criteria": ""}
            for i in range(n)
        ]
        scenario = _minimal_scenario(probe_bank)
        r = _make_runner(scenario, "recall text")

        with patch.object(runner, "llm_chat_json", return_value=_coverage_response(["missing"] * n)):
            queue = r._build_probe_queue()

        self.assertEqual(len(queue), runner.MAX_PROBE_QUEUE_SIZE)


class ProbeQueueEmptyPathTests(unittest.TestCase):
    """Part B7: when nothing survives filtering, the queue comes back empty and
    end_recall() jumps straight to "concluded" without ever entering "probing".

    Note on scope: full coverage of an *authored* bank does NOT generally empty
    the queue -- runner._build_probe_queue only drops 'how'/'sequencing' probes
    when coverage == 'covered'; 'decision'/'error'/'edge_case' survive full
    coverage unconditionally, and 'rationale' only ever drops via target-folding,
    never via coverage. A genuinely empty queue therefore requires a bank made up
    entirely of how/sequencing probes, as constructed below -- this characterizes
    the current gating logic precisely, rather than assuming "recall covers
    everything" alone empties any authored bank's queue.
    """

    def test_empty_queue_when_only_how_and_sequencing_are_fully_covered(self):
        probe_bank = [
            {"probe_type": "how",        "target_key_point": "a", "probe_text": "how?", "success_criteria": ""},
            {"probe_type": "sequencing", "target_key_point": "b", "probe_text": "seq?", "success_criteria": ""},
        ]
        scenario = _minimal_scenario(probe_bank)
        r = _make_runner(scenario, "recall covering everything")

        with patch.object(runner, "llm_chat_json",
                          return_value=_coverage_response(["covered", "covered"], disordered=False)):
            result_text, concluded = r.end_recall()

        self.assertEqual(r.probe_queue, [])
        self.assertEqual(r.phase, "concluded")
        self.assertTrue(concluded)
        self.assertEqual(result_text, "")


class DynamicProbeGenerationTests(unittest.TestCase):
    """Part B8: example.json has no authored probe_bank -- confirm dynamic
    generation is invoked and the generated candidates flow through the exact
    same filter/priority/cap pipeline as an authored bank."""

    def test_dynamic_candidates_span_multiple_types_and_share_authored_bank_pipeline(self):
        scenario = _load_scenario("example.json")
        self.assertNotIn("probe_bank", scenario)  # confirms this exercises the dynamic path

        types = ["sequencing", "how", "rationale", "decision", "error", "edge_case", "rationale", "rationale"]
        dynamic_probes = [
            {"probe_type": t, "target_key_point": f"target_{i}", "probe_text": f"probe {i}?", "success_criteria": "c"}
            for i, t in enumerate(types)
        ]
        # A bare JSON array -- the exact shape _generate_dynamic_probes's own prompt
        # asks for ("Return only the JSON array, no markdown"). See
        # test_bare_json_array_response_is_parsed_correctly below: llm._extract_json
        # used to mishandle this shape (returning just the first element), fixed to
        # scan for either bracket so the outermost container is the one returned.
        dynamic_response = json.dumps(dynamic_probes)
        coverage_response = _coverage_response(["missing"] * len(dynamic_probes), disordered=True)

        r = _make_runner(scenario, "I would apologise and look up the order.")
        with patch.object(runner, "llm_chat_json",
                          side_effect=[dynamic_response, coverage_response]) as mock_call:
            queue = r._build_probe_queue()

        self.assertEqual(mock_call.call_count, 2)  # dynamic generation, then coverage analysis
        self.assertLessEqual(len(queue), runner.MAX_PROBE_QUEUE_SIZE)
        types_in_queue = [c["probe_type"] for c in queue]
        self.assertGreater(len(set(types_in_queue)), 1)          # spans multiple probe types
        self.assertLessEqual(types_in_queue.count("error"), 1)      # same per-type caps apply
        self.assertLessEqual(types_in_queue.count("edge_case"), 1)  # to dynamically generated candidates
        scores = [c["priority_score"] for c in queue]
        self.assertEqual(scores, sorted(scores, reverse=True))   # same descending sort

    def test_bare_json_array_response_is_parsed_correctly(self):
        # Pins the llm._extract_json fix: scanning for either bracket ("{" or "["),
        # leftmost first, means a bare top-level array's own opening "[" is now the
        # first candidate tried, so raw_decode parses the whole array -- not just
        # its first nested object, which is what the old {-only scan used to return.
        scenario = _load_scenario("example.json")
        r = _make_runner(scenario, "some recall text")
        bare_array_response = json.dumps([
            {"probe_type": "how", "target_key_point": "t", "probe_text": "q?", "success_criteria": "c"},
            {"probe_type": "decision", "target_key_point": "u", "probe_text": "q2?", "success_criteria": "c2"},
        ])

        with patch.object(runner, "llm_chat_json", return_value=bare_array_response):
            probes = r._generate_dynamic_probes("recall", "expert answer", [])

        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["probe_type"], "how")
        self.assertEqual(probes[1]["probe_type"], "decision")

    def test_dict_wrapped_array_response_still_also_works(self):
        # The dict-wrapping-a-list shape (_generate_dynamic_probes' own fallback
        # branch: "for v in result.values(): if isinstance(v, list): return v")
        # remains supported alongside the now-fixed bare-array shape above.
        scenario = _load_scenario("example.json")
        r = _make_runner(scenario, "some recall text")
        wrapped_response = json.dumps({"probes": [
            {"probe_type": "how", "target_key_point": "t", "probe_text": "q?", "success_criteria": "c"},
        ]})

        with patch.object(runner, "llm_chat_json", return_value=wrapped_response):
            probes = r._generate_dynamic_probes("recall", "expert answer", [])

        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["probe_type"], "how")


# ─────────────────────────────────────────────────────────────────────────────
# Part C: merge_phase_scores
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHETIC_KEY_POINTS = ["alpha", "beta", "gamma", "delta"]


def _synthetic_expert_answer(rubric=None):
    return {
        "id": "expert_001",
        "answer": "expert text",
        "key_points": _SYNTHETIC_KEY_POINTS,
        "rubric": rubric if rubric is not None else {p: 1 for p in _SYNTHETIC_KEY_POINTS},
    }


def _synthetic_scenario(scoring_weights=None):
    scenario = {
        "id": "synthetic_scenario",
        "situation": "s", "user_role": "learner", "constraints": [],
        "expert_answers": [_synthetic_expert_answer()],
    }
    if scoring_weights:
        scenario["scoring_weights"] = scoring_weights
    return scenario


def _phase_eval(matched, quality_ratings=None, transcript="t", score=0.0, feedback="", strengths=None, gaps=None):
    """A synthetic "already scored" scenario-mode evaluation dict, in the shape
    score_with_llm/score_with_keywords actually return, built directly rather
    than via an LLM call (per the brief, no LLM call is needed for these tests)."""
    return {
        "transcript": transcript,
        "matched_points": matched,
        "missed_points": [p for p in _SYNTHETIC_KEY_POINTS if p not in matched],
        "quality_ratings": quality_ratings or {},
        "coverage_score": 0.0,
        "quality_score": 0.0,
        "score": score,
        "feedback": feedback,
        "strengths": strengths or [],
        "gaps": gaps or [],
    }


class MergePhaseScoresUnionAndQualityTests(unittest.TestCase):
    """Part C1/C2: union of matched points across phases, and the higher quality
    rating wins when a point is matched in both phases with different ratings."""

    def test_union_of_matched_points_appears_exactly_once(self):
        recall_ev = _phase_eval(["alpha", "beta"])
        probe_ev = _phase_eval(["beta", "gamma"])
        scenario = _synthetic_scenario()
        expert_answer = scenario["expert_answers"][0]

        merged = scoring.merge_phase_scores(recall_ev, probe_ev, scenario, expert_answer)

        self.assertEqual(sorted(merged["matched_points"]), ["alpha", "beta", "gamma"])
        self.assertEqual(len(merged["matched_points"]), len(set(merged["matched_points"])))

    def test_max_quality_rating_kept_across_phases(self):
        recall_ev = _phase_eval(["alpha", "beta"], quality_ratings={"alpha": 0, "beta": 2})
        probe_ev = _phase_eval(["alpha", "beta"], quality_ratings={"alpha": 2, "beta": 0})
        scenario = _synthetic_scenario()
        expert_answer = scenario["expert_answers"][0]

        merged = scoring.merge_phase_scores(recall_ev, probe_ev, scenario, expert_answer)

        self.assertEqual(merged["quality_ratings"]["alpha"], 2)
        self.assertEqual(merged["quality_ratings"]["beta"], 2)


class MergePhaseScoresSourceTaggingTests(unittest.TestCase):
    """Part C3: a point matched in the recall phase at all is tagged "recall" --
    even if also touched again during probing -- and a point matched only in the
    probe phase is tagged "probe"."""

    def test_point_matched_in_both_phases_is_tagged_recall(self):
        recall_ev = _phase_eval(["beta"])
        probe_ev = _phase_eval(["beta", "gamma"])
        scenario = _synthetic_scenario()
        expert_answer = scenario["expert_answers"][0]

        merged = scoring.merge_phase_scores(recall_ev, probe_ev, scenario, expert_answer)

        self.assertEqual(merged["point_sources"]["beta"], "recall")
        self.assertEqual(merged["point_sources"]["gamma"], "probe")


class MergePhaseScoresArithmeticTests(unittest.TestCase):
    """Part C4: Coverage/Quality/Combined are recomputed in Python from the union
    of matched points -- not copied from either phase's own numbers."""

    def test_coverage_quality_combined_are_recomputed_not_copied(self):
        # recall matches only "alpha" (quality 0); probe matches "beta" (quality 2)
        # and "gamma" (quality 1) -- union is 3 of 4 equal-weight points.
        recall_ev = _phase_eval(["alpha"], quality_ratings={"alpha": 0}, score=0.11)
        probe_ev = _phase_eval(["beta", "gamma"], quality_ratings={"beta": 2, "gamma": 1}, score=0.22)
        scenario = _synthetic_scenario()  # default weights: coverage 0.6, quality 0.4
        expert_answer = scenario["expert_answers"][0]

        merged = scoring.merge_phase_scores(recall_ev, probe_ev, scenario, expert_answer)

        expected_coverage = 3 / 4          # 3 of 4 equal-weight key points matched
        expected_quality = (0 + 2 + 1) / (3 * 2)  # mean of the 3 matched points' ratings / max rating 2
        expected_combined = 0.6 * expected_coverage + 0.4 * expected_quality

        self.assertAlmostEqual(merged["coverage_score"], expected_coverage, places=4)
        self.assertAlmostEqual(merged["quality_score"], expected_quality, places=4)
        self.assertAlmostEqual(merged["score"], expected_combined, places=4)
        # Neither phase's own (deliberately wrong) dummy score leaked through.
        self.assertNotEqual(merged["score"], recall_ev["score"])
        self.assertNotEqual(merged["score"], probe_ev["score"])


class LegacyNoPhasesFallbackTests(unittest.TestCase):
    """Part C5: Session.evaluate()'s single-phase fallback when recall/probe
    transcripts aren't both provided -- exercises the branch that never calls
    merge_phase_scores at all (session.py)."""

    def test_single_transcript_call_bypasses_merge_phase_scores(self):
        scenario = _load_scenario("morning_dog_care_routine.json")
        transcript = "I would take her outside for a potty break, then feed her and refresh the water."
        session = Session(use_llm=False, model=None, api_key=None, base_url=None)

        with patch.object(scoring, "merge_phase_scores") as mock_merge:
            evals = session.evaluate(scenario, transcript)

        mock_merge.assert_not_called()
        self.assertEqual(len(evals), 1)
        ev = evals[0]
        # merge_phase_scores always adds these per-phase keys; a raw single-phase
        # score_with_keywords result never does -- a cheap, direct way to confirm
        # which code path actually produced this dict.
        self.assertNotIn("recall_score", ev)

        expected = scoring.score_with_keywords(scenario, transcript, scenario["expert_answers"][0])
        self.assertEqual(ev["score"], expected["score"])
        self.assertEqual(ev["matched_points"], expected["matched_points"])


# ─────────────────────────────────────────────────────────────────────────────
# Part D: score_with_llm / score_with_keywords (scenario mode)
# ─────────────────────────────────────────────────────────────────────────────

class ScoreWithKeywordsTests(unittest.TestCase):
    """Part D1: score_with_keywords is deterministic across repeated calls with
    identical input -- no LLM involved, so this guards a future regression, not
    present-day flakiness."""

    def test_repeated_calls_produce_identical_result(self):
        scenario = _load_scenario("Changing_Tire.json")
        expert_answer = scenario["expert_answers"][0]
        transcript = (
            "I would switch on hazard lights, apply the handbrake, and get everyone away "
            "from traffic. I'd loosen the nuts before jacking, find the correct jack point, "
            "and tighten the nuts in a diagonal pattern."
        )

        r1 = scoring.score_with_keywords(scenario, transcript, expert_answer)
        r2 = scoring.score_with_keywords(scenario, transcript, expert_answer)

        self.assertEqual(r1, r2)


class ScoreWithLlmEvidencePrepassTests(unittest.TestCase):
    """Part D2: the grading call receives the compressed per-phase evidence
    block, not the raw multi-turn transcript, and the evidence-extraction prompt
    itself instructs crediting only the learner's own turns."""

    def test_grading_prompt_contains_compressed_evidence_not_raw_transcript(self):
        scenario = _load_scenario("Changing_Tire.json")
        expert_answer = scenario["expert_answers"][0]
        long_transcript = "Examiner: " + ("filler " * 500) + "\n\nDriver: I would apply the handbrake."
        marker = "COMPRESSED_EVIDENCE_MARKER: handbrake applied"

        with patch.object(scoring, "_extract_evidence", return_value=marker) as mock_extract:
            with patch.object(scoring, "llm_chat_json", return_value=json.dumps({
                "matched_points": [], "missed_points": expert_answer["key_points"],
                "quality_ratings": {}, "strengths": [], "gaps": [], "feedback": "",
            })) as mock_grade:
                scoring.score_with_llm(
                    "test-model", None, "http://example.invalid",
                    scenario, long_transcript, expert_answer, bypass_cache=True,
                )

        mock_extract.assert_called_once()
        grading_prompt = mock_grade.call_args[0][2]  # llm_chat_json(model, system, message, ...)
        self.assertIn(marker, grading_prompt)
        self.assertNotIn("filler filler filler", grading_prompt)

    def test_extract_evidence_prompt_instructs_crediting_only_learner_turns(self):
        # Characterizes that the instruction is actually issued in the prompt sent
        # to the LLM. Whether a given model obeys it is a real-LLM judgment call
        # outside what a mocked unit test can verify -- see the pre-pass credit
        # check exercised end-to-end via a real grading call in the tagging test
        # below, which shows the plumbing that depends on this instruction working.
        with patch.object(scoring, "llm_chat", return_value="") as mock_chat:
            scoring._extract_evidence(
                "test-model", None, "http://example.invalid",
                "Examiner: what about X?\n\nDriver: I would do Y.",
                ["Y point"], bypass_cache=True,
            )

        mock_chat.assert_called_once()
        prompt_sent = mock_chat.call_args[0][2]  # llm_chat(model, system, message, ...)
        self.assertIn("Only credit what the LEARNER said", prompt_sent)
        self.assertIn("Do not attribute anything the Examiner says to the learner", prompt_sent)


class ScoreWithLlmGroundingTests(unittest.TestCase):
    """Part D3: a matched point whose supporting_quote does not actually appear
    in the transcript is demoted to missed -- mirrors the equivalent, already-
    tested FR behavior (scoring.py's shared _quote_supported helper)."""

    def test_ungrounded_quote_is_demoted_to_missed(self):
        scenario = _load_scenario("Changing_Tire.json")
        expert_answer = scenario["expert_answers"][0]
        transcript = "Driver: I would apply the handbrake and switch on hazard lights."

        canned = json.dumps({
            "matched_points": [
                {"key_point": "handbrake", "supporting_quote": "apply the handbrake"},
                {"key_point": "hazard lights",
                 "supporting_quote": "this quote does not appear anywhere in the transcript"},
            ],
            "missed_points": [p for p in expert_answer["key_points"] if p not in ("handbrake", "hazard lights")],
            "quality_ratings": {"handbrake": 1, "hazard lights": 1},
            "strengths": [], "gaps": [], "feedback": "",
        })

        with patch.object(scoring, "_extract_evidence", return_value=transcript):
            with patch.object(scoring, "llm_chat_json", return_value=canned):
                ev = scoring.score_with_llm(
                    "test-model", None, "http://example.invalid",
                    scenario, transcript, expert_answer, bypass_cache=True,
                )

        self.assertIn("handbrake", ev["matched_points"])
        self.assertNotIn("hazard lights", ev["matched_points"])
        self.assertIn("hazard lights", ev["missed_points"])


class ScoreWithLlmNarrativeContradictionTests(unittest.TestCase):
    """Part D4: a matched point is demoted when the free-text feedback/gaps
    narrative contains negation language about that same point, even though it
    was structurally matched and grounded."""

    def test_matched_point_demoted_by_contradicting_gaps_narrative(self):
        scenario = _load_scenario("Changing_Tire.json")
        expert_answer = scenario["expert_answers"][0]
        transcript = "Driver: I would apply the handbrake before doing anything else."

        canned = json.dumps({
            "matched_points": [
                {"key_point": "handbrake", "supporting_quote": "apply the handbrake"},
            ],
            "missed_points": [p for p in expert_answer["key_points"] if p != "handbrake"],
            "quality_ratings": {"handbrake": 1},
            "strengths": [],
            "gaps": ["The learner failed to address handbrake."],
            "feedback": "Overall reasonable, though incomplete.",
        })

        with patch.object(scoring, "_extract_evidence", return_value=transcript):
            with patch.object(scoring, "llm_chat_json", return_value=canned):
                ev = scoring.score_with_llm(
                    "test-model", None, "http://example.invalid",
                    scenario, transcript, expert_answer, bypass_cache=True,
                )

        self.assertNotIn("handbrake", ev["matched_points"])
        self.assertIn("handbrake", ev["missed_points"])


class ScoreWithLlmArithmeticTests(unittest.TestCase):
    """Part D5: Coverage/Quality/Combined are computed in Python from the
    verified match set, using a scenario's actual scoring_weights (or the
    0.6/0.4 default when unauthored) -- never taken from anything the LLM
    itself reports about its own scoring."""

    def _run(self, scenario, expert_answer, transcript, matched_with_quotes, quality_ratings, llm_extra=None):
        canned = {
            "matched_points": matched_with_quotes,
            "missed_points": [
                p for p in expert_answer["key_points"]
                if p not in [m["key_point"] for m in matched_with_quotes]
            ],
            "quality_ratings": quality_ratings,
            "strengths": [], "gaps": [], "feedback": "",
        }
        if llm_extra:
            canned.update(llm_extra)
        with patch.object(scoring, "_extract_evidence", return_value=transcript):
            with patch.object(scoring, "llm_chat_json", return_value=json.dumps(canned)):
                return scoring.score_with_llm(
                    "test-model", None, "http://example.invalid",
                    scenario, transcript, expert_answer, bypass_cache=True,
                )

    def test_arithmetic_uses_authored_scoring_weights_not_llm_reported_numbers(self):
        expert_answer = {
            "id": "e1", "answer": "a",
            "key_points": ["alpha", "beta"],
            "rubric": {"alpha": 1, "beta": 1},
        }
        scenario = {
            "id": "custom_weights_scenario", "situation": "s", "user_role": "learner",
            "constraints": [], "expert_answers": [expert_answer],
            "scoring_weights": {"coverage": 0.8, "quality": 0.2},
        }
        transcript = "I would do alpha and beta."
        matched = [{"key_point": "alpha", "supporting_quote": "do alpha"}]

        # The LLM response also includes bogus top-level score fields -- the real
        # schema never asks for these, but even if a model hallucinates them, they
        # must have zero effect on the returned numbers.
        ev = self._run(scenario, expert_answer, transcript, matched, {"alpha": 2},
                       llm_extra={"score": 0.01, "coverage_score": 0.01, "quality_score": 0.01})

        expected_coverage = 0.5   # 1 of 2 equal-weight points matched
        expected_quality = 1.0    # single matched point at rating 2 -> 2/2
        expected_combined = 0.8 * expected_coverage + 0.2 * expected_quality  # = 0.6

        self.assertEqual(ev["coverage_score"], expected_coverage)
        self.assertEqual(ev["quality_score"], expected_quality)
        self.assertAlmostEqual(ev["score"], expected_combined, places=4)
        self.assertNotEqual(ev["score"], 0.01)

    def test_default_weights_apply_when_scenario_has_no_scoring_weights_key(self):
        scenario = _load_scenario("example.json")  # no scoring_weights key (Part A)
        self.assertNotIn("scoring_weights", scenario)
        expert_answer = scenario["expert_answers"][0]
        transcript = "I would apologise and look up the order."
        matched = [{"key_point": "apologise", "supporting_quote": "would apologise"}]

        ev = self._run(scenario, expert_answer, transcript, matched, {"apologise": 0})

        total = sum(expert_answer["rubric"].get(p, 1) for p in expert_answer["key_points"])
        earned = expert_answer["rubric"].get("apologise", 1)
        expected_coverage = earned / total
        expected_quality = 0.0  # single matched point at rating 0
        expected_combined = 0.6 * expected_coverage + 0.4 * expected_quality  # default weights

        self.assertAlmostEqual(ev["coverage_score"], expected_coverage, places=4)
        self.assertEqual(ev["quality_score"], expected_quality)
        self.assertAlmostEqual(ev["score"], expected_combined, places=4)


class ScoreWithLlmPointSourceTaggingTests(unittest.TestCase):
    """Part D6: each matched point is tagged with the phase it was matched in
    (_determine_point_sources), exercised here via the single-call convention
    where recall_transcript/probe_transcript are supplied alongside a combined
    transcript -- the shape score_with_llm's own dual-evidence branch is keyed
    on. (Session.evaluate()'s current two-separate-calls convention never
    exercises this branch -- see exploration notes / merge_phase_scores tests,
    which cover point-source tagging for that convention instead.)"""

    def test_matched_points_tagged_by_which_phase_transcript_contains_them(self):
        expert_answer = {
            "id": "e1", "answer": "a",
            "key_points": ["mentions torque wrench", "mentions diagonal pattern"],
            "rubric": {"mentions torque wrench": 1, "mentions diagonal pattern": 1},
        }
        scenario = {
            "id": "tagging_scenario", "situation": "s", "user_role": "learner",
            "constraints": [], "expert_answers": [expert_answer],
        }
        recall_text = "Driver: I would use a torque wrench to finish tightening."
        probe_text = "Driver: I tighten in a diagonal pattern for even seating."
        combined = recall_text + "\n\n" + probe_text

        canned = json.dumps({
            "matched_points": [
                {"key_point": "mentions torque wrench", "supporting_quote": "use a torque wrench"},
                {"key_point": "mentions diagonal pattern", "supporting_quote": "tighten in a diagonal pattern"},
            ],
            "missed_points": [], "quality_ratings": {}, "strengths": [], "gaps": [], "feedback": "",
        })

        with patch.object(scoring, "_extract_evidence", return_value=combined):
            with patch.object(scoring, "llm_chat_json", return_value=canned):
                ev = scoring.score_with_llm(
                    "test-model", None, "http://example.invalid",
                    scenario, combined, expert_answer,
                    recall_transcript=recall_text, probe_transcript=probe_text,
                    bypass_cache=True,
                )

        self.assertEqual(ev["point_sources"]["mentions torque wrench"], "recall")
        self.assertEqual(ev["point_sources"]["mentions diagonal pattern"], "probe")


if __name__ == "__main__":
    unittest.main()
