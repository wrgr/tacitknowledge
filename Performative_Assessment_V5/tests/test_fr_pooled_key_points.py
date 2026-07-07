import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import loaders
import reports
import report_parser
import scoring


def _write_prompt(pools=None, key_points=None):
    """Build and load a minimal FR prompt with the given standalone key_points/pools,
    going through the real loaders.py migration path (id assignment, defaults, etc.)."""
    data = {
        "title": "t", "prompt_text": "p",
        "expert_answers": [{
            "answer": "a",
            "key_points": key_points or [],
            "pools": pools or [],
        }],
    }
    path = Path(tempfile.mktemp(suffix=".json"))
    path.write_text(json.dumps(data))
    return loaders.load_prompt(path)


# The brief's own motivating example: "describe at least two specific techniques".
# Exemplars deliberately avoid hyphenated phrases ("open-ended questions") -- an
# unrelated pre-existing quirk of _phrase_in_text's significant-word stripping means a
# hyphenated exemplar never matches via keyword-only scoring even verbatim, which would
# make these tests flaky for reasons that have nothing to do with pooled key points.
_TECHNIQUES_POOL = [{
    "pool_id": "listening_techniques",
    "required_count": 2,
    "importance": "HIGH",
    "members": [
        {"construct": "eye contact",             "exemplars": ["eye contact"]},
        {"construct": "paraphrasing",            "exemplars": ["paraphrasing"]},
        {"construct": "open questions",          "exemplars": ["open questions"]},
        {"construct": "avoiding interruptions",  "exemplars": ["avoiding interruptions"]},
    ],
}]

_DEFINITION_KP = [{"construct": "definition", "importance": "HIGH", "exemplars": ["definition of active listening"]}]


class PoolCoverageTests(unittest.TestCase):
    """Quality Criteria #1/#2: a task-compliant response reaches full Coverage credit,
    and matching more than required_count cannot earn more than the pool's full weight.
    """

    def test_task_compliant_response_reaches_full_coverage(self):
        prompt = _write_prompt(pools=_TECHNIQUES_POOL, key_points=_DEFINITION_KP)
        text = "This is the definition of active listening. I use eye contact and paraphrasing."
        ev = scoring.score_free_response_with_keywords(prompt, text)

        self.assertEqual(ev["score"], 1.0)
        self.assertEqual(ev["pools"][0]["matched_in_pool"], 2)
        self.assertEqual(ev["pools"][0]["credited_count"], 2)
        self.assertEqual(ev["pools"][0]["pool_coverage_fraction"], 1.0)

    def test_matching_more_than_required_does_not_exceed_full_credit(self):
        prompt = _write_prompt(pools=_TECHNIQUES_POOL, key_points=_DEFINITION_KP)
        text_2of4 = "This is the definition of active listening. I use eye contact and paraphrasing."
        text_4of4 = text_2of4 + " I also use open questions and practise avoiding interruptions."

        ev2 = scoring.score_free_response_with_keywords(prompt, text_2of4)
        ev4 = scoring.score_free_response_with_keywords(prompt, text_4of4)

        self.assertEqual(ev2["score"], 1.0)
        self.assertEqual(ev4["score"], ev2["score"])  # the min() cap holds
        self.assertEqual(ev4["pools"][0]["matched_in_pool"], 4)
        self.assertEqual(ev4["pools"][0]["credited_count"], 2)
        self.assertEqual(ev4["pools"][0]["pool_coverage_fraction"], 1.0)

    def test_partial_pool_match_is_partial_credit_not_zero_or_full(self):
        prompt = _write_prompt(pools=_TECHNIQUES_POOL, key_points=[])
        text_1of4 = "I use eye contact while listening."
        ev = scoring.score_free_response_with_keywords(prompt, text_1of4)

        self.assertEqual(ev["pools"][0]["matched_in_pool"], 1)
        self.assertEqual(ev["pools"][0]["credited_count"], 1)
        self.assertEqual(ev["pools"][0]["pool_coverage_fraction"], 0.5)
        self.assertEqual(ev["score"], 0.5)


class StandaloneUnaffectedTests(unittest.TestCase):
    """Quality Criteria #4/#5: standalone key points score identically to before this
    change, whether or not a pool is present alongside them, and a prompt with no pools
    at all is completely unaffected.
    """

    def test_standalone_arithmetic_identical_with_and_without_a_pool_present(self):
        standalone_kps = [{"id": "def", "importance": "HIGH"}]
        matched = [{"key_point_id": "def", "importance": "HIGH"}]

        total_no_pool, earned_no_pool, pools_no_pool = scoring._compute_fr_coverage(standalone_kps, [], matched)
        self.assertEqual(pools_no_pool, [])

        pool = {
            "pool_id": "p", "required_count": 2, "importance": "MEDIUM",
            "members": [{"id": "m1", "construct": "a"}, {"id": "m2", "construct": "b"}],
        }
        total_with_pool, earned_with_pool, pools_with_pool = scoring._compute_fr_coverage(
            standalone_kps, [pool], matched
        )

        # No pool member matched -> the pool contributes 0 to earned and its own weight
        # to total; subtracting that back out reproduces the exact pool-free numbers.
        pool_weight = scoring._FR_IMPORTANCE_WEIGHT["MEDIUM"]
        self.assertEqual(total_with_pool - pool_weight, total_no_pool)
        self.assertEqual(earned_with_pool - 0, earned_no_pool)

    def test_prompt_without_pools_key_defaults_to_empty_list(self):
        prompt = _write_prompt(pools=None, key_points=_DEFINITION_KP)
        self.assertEqual(prompt["expert_answers"][0]["pools"], [])

    def test_prompt_without_pools_scores_unaffected(self):
        prompt = _write_prompt(pools=[], key_points=_DEFINITION_KP)
        text = "This is the definition of active listening."
        ev = scoring.score_free_response_with_keywords(prompt, text)
        self.assertEqual(ev["score"], 1.0)
        self.assertEqual(ev["pools"], [])


class QualityMeanEquivalenceTests(unittest.TestCase):
    """Quality Criteria #3: pool members are matched and present in matched_points,
    uncapped, exactly as standalone points would be -- so a Quality mean computed over
    matched_points is identical whether the same items are pooled or standalone. Uses
    the LLM scoring path (mocked, bypass_cache=True so no DB/network is touched) since
    quality_rating only exists on that path.
    """

    def _canned_matches(self, ids):
        qualities = [2, 0, 1]
        exemplars = ["eye contact", "paraphrasing", "open-ended questions"]
        return json.dumps({
            "matches": [
                {"key_point_id": ids[i], "match_type": "exemplar", "matched_exemplar": exemplars[i],
                 "evidence_spans": [exemplars[i]], "quality_rating": qualities[i]}
                for i in range(3)
            ],
            "missed_points": [ids[3]],
            "strengths": [], "gaps": [], "feedback": "ok",
        })

    def test_pool_members_uncapped_and_quality_mean_identical_to_standalone_equivalent(self):
        prompt_pooled = _write_prompt(pools=_TECHNIQUES_POOL, key_points=[])
        member_ids = [m["id"] for m in prompt_pooled["expert_answers"][0]["pools"][0]["members"]]

        standalone_equiv_kps = [
            {"construct": m["construct"], "exemplars": m["exemplars"], "importance": "HIGH"}
            for m in _TECHNIQUES_POOL[0]["members"]
        ]
        prompt_standalone = _write_prompt(pools=[], key_points=standalone_equiv_kps)
        standalone_ids = [kp["id"] for kp in prompt_standalone["expert_answers"][0]["key_points"]]

        text = "I used eye contact, paraphrasing, and open-ended questions while listening."

        with patch.object(scoring, "llm_chat_json", return_value=self._canned_matches(member_ids)):
            ev_pooled = scoring.score_free_response_with_llm(
                "test-model", None, "http://example.invalid", prompt_pooled, text, bypass_cache=True,
            )
        with patch.object(scoring, "llm_chat_json", return_value=self._canned_matches(standalone_ids)):
            ev_standalone = scoring.score_free_response_with_llm(
                "test-model", None, "http://example.invalid", prompt_standalone, text, bypass_cache=True,
            )

        # Uncapped: all 3 matched pool members appear in matched_points even though
        # required_count is only 2 -- capping only ever touches Coverage arithmetic.
        self.assertEqual(len(ev_pooled["matched_points"]), 3)
        self.assertEqual(ev_pooled["pools"][0]["matched_in_pool"], 3)
        self.assertEqual(ev_pooled["pools"][0]["credited_count"], 2)  # Coverage IS capped

        def _quality_mean(ev):
            ratings = [m["quality_rating"] for m in ev["matched_points"]]
            return sum(ratings) / (len(ratings) * 2) if ratings else 0.0

        pooled_ratings     = sorted(m["quality_rating"] for m in ev_pooled["matched_points"])
        standalone_ratings = sorted(m["quality_rating"] for m in ev_standalone["matched_points"])
        self.assertEqual(pooled_ratings, standalone_ratings)
        self.assertEqual(_quality_mean(ev_pooled), _quality_mean(ev_standalone))

        # And Coverage genuinely differs between the two prompts (the whole point of the
        # feature) even though Quality does not.
        self.assertNotEqual(ev_pooled["score"], ev_standalone["score"])


class NoAuthoringTimeValidationTests(unittest.TestCase):
    """Quality Criteria #7: no validation/warning logic exists anywhere that checks
    required_count against the FR prompt's own task instructions text -- this was
    explicitly declined. A structural/no-crash check, since it's fundamentally a claim
    about the absence of code: a required_count that plainly can't be satisfied (higher
    than the pool has members) is accepted and left to a human to notice, not caught.
    """

    def test_required_count_higher_than_member_count_is_accepted_without_error(self):
        pool = [{
            "pool_id": "p", "required_count": 99, "importance": "MEDIUM",
            "members": [{"construct": "eye contact", "exemplars": ["eye contact"]},
                        {"construct": "paraphrasing", "exemplars": ["paraphrasing"]}],
        }]
        prompt = _write_prompt(pools=pool, key_points=[])
        self.assertEqual(prompt["expert_answers"][0]["pools"][0]["required_count"], 99)
        ev = scoring.score_free_response_with_keywords(prompt, "eye contact and paraphrasing")
        self.assertEqual(ev["pools"][0]["matched_in_pool"], 2)
        self.assertLess(ev["pools"][0]["pool_coverage_fraction"], 1.0)  # unwinnable, and that's fine

    def test_no_source_references_prompt_text_when_deriving_required_count(self):
        # A structural check that _migrate_fr_key_points never sees, and therefore can't
        # compare against, the FR prompt's own instructions text -- it only receives the
        # expert_answer dict, never prompt_text.
        source = inspect.getsource(loaders._migrate_fr_key_points)
        self.assertNotIn("prompt_text", source)
        params = list(inspect.signature(loaders._migrate_fr_key_points).parameters)
        self.assertEqual(params, ["ea"])


class ReportPoolSectionTests(unittest.TestCase):
    """Part C: the report's per-pool summary line is neutral -- states what was
    required and matched without editorializing about extra matches.
    """

    def _evaluation(self):
        return {
            "text": "x", "score": 1.0, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
            "pools": [{
                "pool_id": "listening_techniques", "importance": "HIGH", "required_count": 2,
                "member_count": 4, "matched_in_pool": 2, "credited_count": 2,
                "pool_coverage_fraction": 1.0,
                "matched_members": [
                    {"construct": "eye contact", "match_type": "exemplar", "matched_exemplar": "eye contact",
                     "evidence_spans": ["eye contact"], "functional_justification": None, "quality_rating": 1},
                ],
                "missed_members": [
                    {"key_point_id": "m2", "construct": "paraphrasing", "importance": "HIGH", "pool_id": "listening_techniques"},
                ],
            }],
        }

    def test_full_credit_phrasing_and_no_editorializing(self):
        lines = []
        reports._append_fr_pools(lines, self._evaluation())
        joined = "\n".join(lines)
        self.assertIn("## Pools", joined)
        self.assertIn("Matched 2 of 4", joined)
        self.assertIn("2 required", joined)
        self.assertIn("full credit for this section", joined)
        for phrase in ("wasted", "unnecessary", "discouraged", "extra credit"):
            self.assertNotIn(phrase, joined.lower())

    def test_omitted_entirely_when_no_pools(self):
        lines = []
        reports._append_fr_pools(lines, {"pools": []})
        self.assertEqual(lines, [])

    def test_flat_key_points_section_excludes_pool_members(self):
        ev = {
            "matched_points": [
                {"construct": "definition", "pool_id": None},
                {"construct": "eye contact", "pool_id": "listening_techniques"},
            ],
            "missed_points": [],
        }
        lines = []
        reports._append_fr_key_points(lines, ev)
        joined = "\n".join(lines)
        self.assertIn("definition", joined)
        self.assertNotIn("eye contact", joined)


class ReportParserRoundTripTests(unittest.TestCase):
    """Test #9: the gap this feature almost shipped with -- report_parser.py must parse
    the '## Pools' section back out, or the pool breakdown is invisible in the in-app
    report view even though it's correctly written to the .md file on disk.
    """

    def test_pools_survive_generate_then_parse_round_trip(self):
        prompt_data = {"id": "p1", "title": "Active Listening", "prompt_text": "Explain it."}
        evaluation = {
            "text": "x", "score": 1.0, "feedback": "", "strengths": [], "gaps": [],
            "matched_points": [], "missed_points": [],
            "expert_answer": {"answer": "expert text"},
            "pools": [{
                "pool_id": "listening_techniques", "importance": "HIGH", "required_count": 2,
                "member_count": 4, "matched_in_pool": 2, "credited_count": 2,
                "pool_coverage_fraction": 1.0,
                "matched_members": [
                    {"construct": "eye contact", "match_type": "exemplar", "matched_exemplar": "eye contact",
                     "evidence_spans": ["eye contact span"], "functional_justification": None, "quality_rating": 2},
                ],
                "missed_members": [
                    {"key_point_id": "m2", "construct": "paraphrasing", "importance": "HIGH", "pool_id": "listening_techniques"},
                    {"key_point_id": "m3", "construct": "open questions", "importance": "HIGH", "pool_id": "listening_techniques"},
                ],
            }],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(reports, "llm_chat", side_effect=RuntimeError("no network in tests")):
                path = reports.generate_fr_report(
                    prompt_data, evaluation, model="test-model", api_key=None,
                    base_url="http://example.invalid", output_dir=tmp,
                )
            content = path.read_text(encoding="utf-8")
            parsed = report_parser.parse_report_md(content)

        # Expert reference answer must still parse correctly -- the Pools section sits
        # after it, as its own top-level heading, and must not swallow it.
        self.assertEqual(parsed["evaluation"]["expert_answer"], "expert text")

        pools = parsed["evaluation"]["pools"]
        self.assertEqual(len(pools), 1)
        pool = pools[0]
        self.assertEqual(pool["label"], "Listening Techniques")
        self.assertEqual(pool["matched_in_pool"], 2)
        self.assertEqual(pool["member_count"], 4)
        self.assertEqual(pool["required_count"], 2)
        self.assertEqual(pool["credit_label"], "full")
        self.assertEqual(len(pool["matched_members"]), 1)
        self.assertEqual(pool["matched_members"][0]["construct"], "eye contact")
        self.assertEqual(pool["matched_members"][0]["quality_label"], "full explanation")
        self.assertEqual(pool["missed_members"], ["paraphrasing", "open questions"])


class DatabasePoolReviewTests(unittest.TestCase):
    """Quality Criteria #6: novel-equivalent review rows round-trip pool_id, and
    promote-as-new-member appends a member to the right pool. Skipped if this sandbox
    can't import database.py (it needs werkzeug, which isn't installed here) -- the
    existing test suite avoids that dependency for the same reason; this still runs
    for real in a properly provisioned dev/CI environment.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import database
        except ImportError as e:
            raise unittest.SkipTest(f"database.py not importable in this environment: {e}")
        cls.database = database

    def setUp(self):
        self._orig_db_file = self.database.DB_FILE
        self.database.DB_FILE = Path(tempfile.mktemp(suffix=".db"))
        self.database.init_db()

    def tearDown(self):
        self.database.DB_FILE = self._orig_db_file

    def test_pool_id_round_trips_through_log_and_get(self):
        self.database.log_novel_equivalent(
            prompt_id="demo", key_point_id="eye_contact", construct="eye contact",
            submission_excerpt="...", evidence_spans=["eye contact"],
            justification="justification text here", pool_id="listening_techniques",
        )
        reviews = self.database.list_novel_equivalent_reviews("pending")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["pool_id"], "listening_techniques")

    def test_standalone_review_has_no_pool_id(self):
        self.database.log_novel_equivalent(
            prompt_id="demo", key_point_id="definition", construct="definition",
            submission_excerpt="...", evidence_spans=["def"],
            justification="justification text here",
        )
        reviews = self.database.list_novel_equivalent_reviews("pending")
        self.assertIsNone(reviews[0]["pool_id"])

    def test_init_db_is_idempotent_with_the_new_column(self):
        self.database.init_db()
        self.database.init_db()  # must not raise on the second guarded ALTER TABLE


if __name__ == "__main__":
    unittest.main()
