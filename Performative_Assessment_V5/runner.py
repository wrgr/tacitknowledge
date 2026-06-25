"""
runner.py — ScenarioRunner and scenario draft generation.

ScenarioRunner manages the back-and-forth conversation for one scenario.
Kept as a class because it needs to remember state between turns
(which turn we're on, what the learner said, etc.).
"""

import re

from llm import llm_generate, llm_chat, _extract_json


FALLBACK_PROMPTS = ["Keep going.", "What else would you do?"]  # used when the API can't generate narration
FALLBACK_CLOSING = "Share anything else you would do."         # used on the second-to-last turn


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioRunner:

    def __init__(self, scenario, model, api_key, base_url):
        self.scenario        = scenario    # the scenario dict loaded from JSON
        self.model           = model       # LLM model name
        self.api_key         = api_key     # provider API key
        self.base_url        = base_url    # provider base URL
        self.history         = []          # list of {"role": ..., "content": ...} dicts, one per turn
        self.user_inputs     = []          # just the learner's responses (used to check if they said anything)
        self.writing_metrics = []          # per-turn behavioural metrics from the frontend
        self.turn            = 0           # counts how many learner responses have been submitted

    def start(self):
        # reset everything so the runner can be reused for a fresh attempt
        self.history         = []
        self.user_inputs     = []
        self.writing_metrics = []
        self.turn            = 0
        # return the opening line shown to the learner
        return "Examiner: " + self.scenario["situation"] + "\n\nWhat do you do?"

    def _get_narration(self, user_input, closing=False):
        if closing:
            # ask for a single closing sentence inviting the learner to add anything missed
            prompt = (
                "Scenario: " + self.scenario["situation"] + "\n\n"
                'The learner said: "' + user_input + '"\n\n'
                "Write one short sentence for the examiner inviting the learner to add anything else they might have missed. Write only the sentence, no extra text. Make sure to not lead the learner to any specific action or answer."
            )
            fallback = FALLBACK_CLOSING
        else:
            # ask for a single sentence acknowledging what the learner just did
            prompt = (
                "Scenario: " + self.scenario["situation"] + "\n\n"
                'The learner said: "' + user_input + '"\n\n'
                'Write one short sentence for the examiner that starts with "You\'ve" and '
                "acknowledges what the learner just did. Write only the sentence, no extra text. Make sure to not lead the learner to any specific action or answer."
            )
            fallback = FALLBACK_PROMPTS[(self.turn - 1) % len(FALLBACK_PROMPTS)]

        try:
            raw = llm_generate(self.model, prompt, self.api_key, self.base_url).strip()

            # remove <think>…</think> blocks produced by reasoning models (DeepSeek R1, QWQ, etc.)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # find the first non-empty line
            first_line = ""
            for line in raw.splitlines():
                if line.strip():
                    first_line = line.strip()
                    break

            # strip markdown formatting characters the model sometimes adds (**, *, _)
            first_line = first_line.replace("**", "").replace("*", "").replace("_", "")

            # take only the first sentence
            completion = ""
            for i, char in enumerate(first_line):
                if char in ".!?":
                    completion = first_line[:i + 1].strip()
                    break
            if not completion:
                completion = first_line  # no sentence boundary found — use the whole line

            if completion:
                return completion + " What next?"
            return fallback
        except Exception as e:
            print("[narration error] " + str(e))
            return fallback

    def respond(self, user_input, writing_metrics=None):
        self.user_inputs.append(user_input)                       # record what the learner said
        self.writing_metrics.append(writing_metrics or {})        # store behavioural metrics for this turn
        self.turn += 1                                            # advance the turn counter

        is_concluded = self.turn >= self.scenario["max_turns"]    # True on the last turn
        is_closing   = self.turn == self.scenario["max_turns"] - 1  # True one turn before the end

        if is_concluded:
            narration = ""  # no examiner response on the final turn
        elif is_closing:
            narration = self._get_narration(user_input, closing=True)
        else:
            narration = self._get_narration(user_input)

        # save this exchange to the history log
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": narration})

        return narration, is_concluded  # caller uses is_concluded to know when to stop

    def transcript(self):
        # build the full conversation as a readable string for evaluation and reports
        role  = self.scenario["user_role"].title()  # e.g. "driver" → "Driver"
        lines = ["Examiner: " + self.scenario["situation"] + "\n\nWhat do you do?"]

        for msg in self.history:
            if msg["role"] == "user":
                lines.append(role + ": " + msg["content"])
            else:
                lines.append("Examiner: " + msg["content"])

        return "\n\n".join(lines)  # blank line between each turn


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DRAFT GENERATION
# Takes a plain-English description and asks the LLM to produce a full
# scenario JSON draft that the instructor can review and edit in the web UI.
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario_draft(description, model, api_key, base_url):
    prompt = (
        "You are an instructional designer creating a performative assessment scenario.\n\n"
        'Based on this description: "' + description + '"\n\n'
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "description": "<one sentence summary>",\n'
        '  "situation": "<2-3 sentences describing the scene in second person, starting with You...>",\n'
        '  "user_role": "<the learner\'s role, e.g. nurse, driver, technician>",\n'
        '  "max_turns": 8,\n'
        '  "constraints": ["<must-satisfy rule 1>", "<rule 2>", "<rule 3>"],\n'
        '  "expert_answer": "<2-3 sentences describing the complete ideal step-by-step response>",\n'
        '  "key_points": ["<short phrase 1>", "<short phrase 2>", "<6-10 total>"],\n'
        '  "rubric": {"<key point phrase>": <weight 1-4>, "<next phrase>": <weight>}\n'
        "}\n\n"
        "Guidelines:\n"
        "- situation: write in second person (You are...)\n"
        "- key_points: 6-10 short phrases the learner must mention to score well\n"
        "- rubric weights: 1=low importance, 2=medium, 3=high, 4=critical\n"
        "- every key_point must have an entry in rubric\n"
        "Return only the JSON, no other text."
    )

    system = (
        "You are an expert instructional designer. "
        "Create clear, realistic performative assessment scenarios. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw = llm_chat(model, system, prompt, api_key, base_url)
    return _extract_json(raw)


def generate_prompt_draft(description, model, api_key, base_url):
    prompt = (
        "You are an instructional designer creating a free-response writing prompt for assessment.\n\n"
        'Based on this description: "' + description + '"\n\n'
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "description": "<one sentence summary shown in the prompt list>",\n'
        '  "prompt_text": "<2-4 sentences giving the learner a clear writing task, including length and format>",\n'
        '  "word_limit": <suggested word limit as an integer, e.g. 200>,\n'
        '  "constraints": ["<must-satisfy rule 1>", "<rule 2>", "<rule 3>"],\n'
        '  "expert_answer": "<3-5 sentences: the complete ideal response covering all key points>",\n'
        '  "key_points": ["<short phrase 1>", "<phrase 2>", "<6-10 total>"],\n'
        '  "rubric": {"<key point phrase>": <weight 1-4>, "<next phrase>": <weight>}\n'
        "}\n\n"
        "Guidelines:\n"
        "- prompt_text: clear, direct instruction ending with what the learner should produce\n"
        "- key_points: 6-10 short phrases the learner must mention to score well\n"
        "- rubric weights: 1=low importance, 2=medium, 3=high, 4=critical\n"
        "- every key_point must have an entry in rubric\n"
        "Return only the JSON, no other text."
    )

    system = (
        "You are an expert instructional designer. "
        "Create clear, rigorous free-response writing prompts for assessment. "
        "Respond only with valid JSON — no markdown, no extra text."
    )

    raw = llm_chat(model, system, prompt, api_key, base_url)
    return _extract_json(raw)
