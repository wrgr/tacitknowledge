"""
session.py — Session tracks all evaluations across one sitting.
"""

from scoring import score_with_llm, score_with_keywords


class Session:

    def __init__(self, use_llm, model, api_key, base_url):
        self.use_llm  = use_llm
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url
        self.results  = []

    def evaluate(self, scenario, transcript,
                 recall_transcript="", probe_transcript=""):
        """Score the transcript against every expert answer in the scenario.

        recall_transcript and probe_transcript are provided by the runner when
        the phase-based assessment model is used. When absent (legacy path or
        CLI), full-transcript scoring works as before.
        """
        evaluations = []
        for expert_answer in scenario["expert_answers"]:
            if self.use_llm:
                ev = score_with_llm(
                    self.model, self.api_key, self.base_url,
                    scenario, transcript, expert_answer,
                    recall_transcript=recall_transcript,
                    probe_transcript=probe_transcript,
                )
            else:
                ev = score_with_keywords(
                    scenario, transcript, expert_answer,
                    recall_transcript=recall_transcript,
                    probe_transcript=probe_transcript,
                )
            evaluations.append(ev)

        self.results.append((scenario, evaluations))
        return evaluations

    def average_score(self):
        all_scores = [ev["score"] for _, evals in self.results for ev in evals]
        return round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    def total_evaluations(self):
        return sum(len(evals) for _, evals in self.results)
