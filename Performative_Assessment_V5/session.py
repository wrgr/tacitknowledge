"""
session.py — Session tracks all evaluations across one sitting.

Kept as a class because it accumulates results over time
(possibly across multiple scenarios in a single run).
"""

from scoring import score_with_llm, score_with_keywords


# ─────────────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────────────

class Session:

    def __init__(self, use_llm, model, api_key, base_url):
        self.use_llm  = use_llm    # True if a valid API key is configured for LLM scoring
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url
        self.results  = []  # list of (scenario, [evaluation_dict, ...]) tuples

    def evaluate(self, scenario, transcript):
        # score the transcript against every expert answer in the scenario
        evaluations = []
        for expert_answer in scenario["expert_answers"]:
            if self.use_llm:
                ev = score_with_llm(self.model, self.api_key, self.base_url, scenario, transcript, expert_answer)
            else:
                ev = score_with_keywords(scenario, transcript, expert_answer)
            evaluations.append(ev)

        self.results.append((scenario, evaluations))  # save for the summary and report
        return evaluations

    def average_score(self):
        # compute the mean score across all evaluations in this session
        all_scores = [ev["score"] for _, evals in self.results for ev in evals]
        return round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    def total_evaluations(self):
        # count how many evaluations have been done (one per expert_answer per scenario)
        return sum(len(evals) for _, evals in self.results)
