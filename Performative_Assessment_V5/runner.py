"""
runner.py — ScenarioRunner and scenario draft generation.

ScenarioRunner is a five-phase state machine:
  recall   → learner types freely; examiner gives minimal acknowledgment
  probing  → examiner works through a probe queue targeting knowledge gaps
  concluded → all probes exhausted; session ready for scoring

Phase transitions:
  recall   → probing    via end_recall() (learner clicks "I'm Done")
  probing  → concluded  when probe_queue is exhausted
"""

import re

from llm import llm_generate, llm_chat, llm_chat_json, _extract_json


FALLBACK_RECALL_ACK = ["Go on.", "Continue.", "What next?", "Anything else?"]
FALLBACK_PROBE = "Can you tell me more about that?"


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNER  (phase-aware state machine)
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioRunner:

    def __init__(self, scenario, model, api_key, base_url):
        self.scenario        = scenario
        self.model           = model
        self.api_key         = api_key
        self.base_url        = base_url

        # shared history and behavioural data
        self.history         = []
        self.user_inputs     = []
        self.writing_metrics = []

        # phase state
        self.phase           = "recall"   # "recall" | "probing" | "concluded"
        self.recall_history  = []         # history entries from the recall phase
        self.probe_history   = []         # history entries from the probing phase

        # probe state
        self.probe_queue     = []         # ordered list of probe dicts
        self.probe_index     = 0          # position in probe_queue
        self.probe_results   = {}         # probe_index → {"satisfied": bool, "exchange": [...]}
        self._probe_followup_pending = False  # True after delivering a follow-up to same probe

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def is_concluded(self):
        return self.phase == "concluded"

    @property
    def recall_transcript(self):
        role = self.scenario["user_role"].title()
        lines = ["Examiner: " + self.scenario["situation"]]
        for msg in self.recall_history:
            if msg["role"] == "user":
                lines.append(role + ": " + msg["content"])
            else:
                lines.append("Examiner: " + msg["content"])
        return "\n\n".join(lines)

    @property
    def probe_transcript(self):
        if not self.probe_history:
            return ""
        role = self.scenario["user_role"].title()
        lines = []
        for msg in self.probe_history:
            if msg["role"] == "user":
                lines.append(role + ": " + msg["content"])
            else:
                lines.append("Examiner: " + msg["content"])
        return "\n\n".join(lines)

    def start(self):
        self.history         = []
        self.user_inputs     = []
        self.writing_metrics = []
        self.phase           = "recall"
        self.recall_history  = []
        self.probe_history   = []
        self.probe_queue     = []
        self.probe_index     = 0
        self.probe_results   = {}
        self._probe_followup_pending = False

        framing = (
            "Walk through what you would do — step by step, in as much detail as you can. "
            "When you feel you've said everything, click \"I'm Done\" to continue."
        )
        return "Examiner: " + self.scenario["situation"] + "\n\n" + framing

    def respond(self, user_input, writing_metrics=None):
        self.user_inputs.append(user_input)
        self.writing_metrics.append(writing_metrics or {})

        if self.phase == "recall":
            return self._recall_respond(user_input)
        elif self.phase == "probing":
            return self._probe_respond(user_input)
        else:
            return "", True  # already concluded

    def end_recall(self):
        """Called when learner clicks 'I'm Done'.
        Locks the recall transcript, runs gap analysis, builds probe queue,
        transitions phase to 'probing', and returns the first probe text."""
        self.phase = "probing"

        # build the probe queue
        self.probe_queue = self._build_probe_queue()
        self.probe_index = 0
        self._probe_followup_pending = False

        if not self.probe_queue:
            # nothing to probe — mark concluded
            self.phase = "concluded"
            return "", True

        first_probe = self._format_probe_delivery(self.probe_queue[0])
        self.probe_history.append({"role": "assistant", "content": first_probe})
        self.history.append({"role": "assistant", "content": first_probe})
        return first_probe, False

    def transcript(self):
        """Full combined transcript with phase labels."""
        role = self.scenario["user_role"].title()
        lines = ["Examiner: " + self.scenario["situation"]]
        for msg in self.recall_history:
            if msg["role"] == "user":
                lines.append(role + ": " + msg["content"])
            else:
                lines.append("Examiner: " + msg["content"])

        if self.probe_history:
            lines.append("\n--- Probing Phase ---\n")
            for msg in self.probe_history:
                if msg["role"] == "user":
                    lines.append(role + ": " + msg["content"])
                else:
                    lines.append("Examiner: " + msg["content"])

        return "\n\n".join(lines)

    def probe_count(self):
        return len(self.probe_queue)

    def current_probe_number(self):
        return min(self.probe_index + 1, len(self.probe_queue))

    # ─── Recall phase ────────────────────────────────────────────────────────

    def _recall_respond(self, user_input):
        self.recall_history.append({"role": "user", "content": user_input})
        self.history.append({"role": "user", "content": user_input})

        ack = self._get_recall_ack(user_input)

        self.recall_history.append({"role": "assistant", "content": ack})
        self.history.append({"role": "assistant", "content": ack})

        return ack, False  # recall phase never auto-concludes

    def _get_recall_ack(self, user_input):
        """Minimal, non-leading acknowledgment during recall."""
        prompt = (
            "Scenario: " + self.scenario["situation"] + "\n\n"
            'The learner said: "' + user_input + '"\n\n'
            "Write one very short acknowledgment (3-7 words). "
            "Do NOT ask leading questions, volunteer information, or suggest what was missed. "
            "Suitable responses: \"Go on.\", \"Continue.\", \"What next?\", \"Anything else?\". "
            "Write only the acknowledgment, no extra text."
        )
        try:
            raw = llm_generate(self.model, prompt, self.api_key, self.base_url).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            first_line = ""
            for line in raw.splitlines():
                if line.strip():
                    first_line = line.strip()
                    break
            first_line = first_line.replace("**", "").replace("*", "").replace("_", "")
            for i, ch in enumerate(first_line):
                if ch in ".!?":
                    return first_line[:i + 1].strip()
            return first_line or FALLBACK_RECALL_ACK[len(self.recall_history) % len(FALLBACK_RECALL_ACK)]
        except Exception as e:
            print("[recall ack error] " + str(e))
            return FALLBACK_RECALL_ACK[len(self.recall_history) % len(FALLBACK_RECALL_ACK)]

    # ─── Gap analysis & probe queue construction ──────────────────────────────

    def _build_probe_queue(self):
        """Analyze recall transcript against probe_bank to build ordered probe queue."""
        scenario   = self.scenario
        probe_bank = scenario.get("probe_bank", [])
        recall_txt = self.recall_transcript

        expert_answer = ""
        key_points    = []
        if scenario.get("expert_answers"):
            ea = scenario["expert_answers"][0]
            expert_answer = ea.get("answer", "")
            key_points    = ea.get("key_points", [])

        # normalise probe dicts: accept both {type,question} and {probe_type,probe_text}
        normalised = []
        for p in probe_bank:
            normalised.append({
                "probe_type":        p.get("probe_type") or p.get("type", "sequencing"),
                "probe_text":        p.get("probe_text") or p.get("question", ""),
                "target_key_point":  p.get("target_key_point", ""),
                "success_criteria":  p.get("success_criteria", ""),
            })
        probe_bank = normalised

        # if no probe_bank, generate probes dynamically
        if not probe_bank:
            probe_bank = self._generate_dynamic_probes(recall_txt, expert_answer, key_points)

        if not probe_bank:
            return []

        # gap analysis: determine which probes are still needed
        coverage_map = self._analyse_coverage(recall_txt, probe_bank, key_points)

        # build ordered queue per sequencing spec
        # order: sequencing → how → rationale → decision → error → edge_case
        TYPE_ORDER = ["sequencing", "how", "rationale", "decision", "error", "edge_case"]

        # separate probes by type; include probes where coverage is 'missing' or 'partial'
        buckets = {t: [] for t in TYPE_ORDER}
        for probe in probe_bank:
            ptype = probe.get("probe_type", "sequencing")
            if ptype not in buckets:
                ptype = "sequencing"
            coverage = coverage_map.get(probe.get("probe_text", ""), "missing")
            if coverage in ("missing", "partial"):
                buckets[ptype].append(probe)

        # guarantee at minimum one error and one edge_case probe
        for required_type in ("error", "edge_case"):
            if not buckets[required_type]:
                for probe in probe_bank:
                    if probe.get("probe_type") == required_type:
                        if probe not in buckets[required_type]:
                            buckets[required_type].append(probe)
                            break

        queue = []
        for t in TYPE_ORDER:
            queue.extend(buckets[t])

        # annotate each queued probe with runtime state
        for p in queue:
            p.setdefault("status", "pending")
            p.setdefault("exchange", [])
            p["_followup_sent"] = False

        return queue

    def _analyse_coverage(self, recall_txt, probe_bank, key_points):
        """LLM call to determine which probe targets are already covered in recall.
        Returns a dict mapping probe_text → 'covered' | 'partial' | 'missing'."""
        if not probe_bank:
            return {}

        probe_summaries = []
        for i, p in enumerate(probe_bank):
            probe_summaries.append(
                f'{i}: type={p.get("probe_type")}, target="{p.get("target_key_point","")}", '
                f'probe="{p.get("probe_text","")}"'
            )

        prompt = (
            "RECALL TRANSCRIPT:\n" + recall_txt + "\n\n"
            "PROBE BANK (numbered):\n" + "\n".join(probe_summaries) + "\n\n"
            "For each probe, assess whether the recall transcript ALREADY addresses it:\n"
            "  covered  = learner clearly addressed this in recall (no probe needed)\n"
            "  partial  = learner touched on it but incompletely\n"
            "  missing  = not addressed at all\n\n"
            "Return JSON:\n"
            '{"coverage": {"0": "covered|partial|missing", "1": "...", ...}}'
        )
        system = (
            "You assess how completely a learner's free recall covers specific knowledge targets. "
            "Be strict: 'covered' only when clearly and specifically addressed. "
            "Respond only with valid JSON."
        )
        try:
            raw    = llm_chat_json(self.model, system, prompt, self.api_key, self.base_url)
            result = _extract_json(raw)
            cov    = result.get("coverage", {})
            # map probe_text → status
            out = {}
            for i, p in enumerate(probe_bank):
                out[p.get("probe_text", "")] = cov.get(str(i), "missing")
            return out
        except Exception as e:
            print("[coverage analysis error] " + str(e))
            # fallback: assume all missing → probe everything
            return {p.get("probe_text", ""): "missing" for p in probe_bank}

    def _generate_dynamic_probes(self, recall_txt, expert_answer, key_points):
        """Generate probe_bank dynamically when none is authored in the scenario JSON."""
        scenario = self.scenario
        kp_text  = "\n".join("- " + p for p in key_points) if key_points else "(none specified)"
        dp_text  = "\n".join("- " + d for d in scenario.get("decision_points", [])) or "(none specified)"
        fm_text  = "\n".join("- " + f for f in scenario.get("failure_modes", []))   or "(none specified)"
        ec_text  = "\n".join("- " + e for e in scenario.get("edge_cases", []))      or "(none specified)"

        prompt = (
            "SCENARIO: " + scenario.get("description", scenario.get("title", "")) + "\n"
            "SITUATION: " + scenario.get("situation", "") + "\n\n"
            "EXPERT ANSWER:\n" + expert_answer + "\n\n"
            "KEY POINTS:\n" + kp_text + "\n\n"
            "DECISION POINTS:\n" + dp_text + "\n\n"
            "FAILURE MODES:\n" + fm_text + "\n\n"
            "EDGE CASES:\n" + ec_text + "\n\n"
            "LEARNER'S FREE RECALL:\n" + recall_txt + "\n\n"
            "Generate a structured probe bank covering all six probe types. "
            "Return JSON array — each item has:\n"
            '  "probe_type": one of sequencing|how|rationale|decision|error|edge_case\n'
            '  "target_key_point": the key point this probe targets (string)\n'
            '  "probe_text": the exact question the examiner should ask\n'
            '  "success_criteria": what a satisfactory answer looks like\n\n'
            "Include at least one probe of each type. Total 8-12 probes. "
            "Focus probes on gaps in the free recall above. "
            "Return only the JSON array, no markdown."
        )
        system = (
            "You are an expert CTA (Cognitive Task Analysis) practitioner. "
            "Generate structured diagnostic probes covering sequencing, how-to, rationale, "
            "decision points, error handling, and edge cases. "
            "Respond only with a valid JSON array."
        )
        try:
            raw    = llm_chat_json(self.model, system, prompt, self.api_key, self.base_url)
            result = _extract_json(raw)
            # result may be a list directly or wrapped in a key
            if isinstance(result, list):
                return result
            for v in result.values():
                if isinstance(v, list):
                    return v
            return []
        except Exception as e:
            print("[dynamic probe generation error] " + str(e))
            return []

    # ─── Probing phase ────────────────────────────────────────────────────────

    def _probe_respond(self, user_input):
        self.probe_history.append({"role": "user", "content": user_input})
        self.history.append({"role": "user", "content": user_input})

        if self.probe_index >= len(self.probe_queue):
            self.phase = "concluded"
            return "", True

        current_probe = self.probe_queue[self.probe_index]
        current_probe["exchange"].append({"role": "user", "content": user_input})

        adequate = self._evaluate_probe_response(user_input, current_probe)

        if adequate:
            current_probe["status"] = "satisfied"
            return self._advance_to_next_probe()
        else:
            if current_probe.get("_followup_sent"):
                # already attempted one follow-up → exhaust this probe
                current_probe["status"] = "exhausted"
                return self._advance_to_next_probe()
            else:
                # deliver one follow-up clarification
                current_probe["_followup_sent"] = True
                followup = self._get_probe_followup(user_input, current_probe)
                current_probe["exchange"].append({"role": "assistant", "content": followup})
                self.probe_history.append({"role": "assistant", "content": followup})
                self.history.append({"role": "assistant", "content": followup})
                return followup, False

    def _evaluate_probe_response(self, user_input, probe):
        """Use LLM to check whether the learner's response adequately satisfies the probe."""
        criteria = probe.get("success_criteria", "")
        if not criteria:
            # no criteria defined — treat as satisfied after any substantive response
            return len(user_input.split()) >= 5

        prompt = (
            f'PROBE: "{probe.get("probe_text", "")}"\n'
            f'SUCCESS CRITERIA: {criteria}\n'
            f'LEARNER RESPONSE: "{user_input}"\n\n'
            "Does the learner's response adequately satisfy the success criteria? "
            "Be strict: vague or evasive answers are NOT adequate. "
            'Return JSON: {"adequate": true|false, "reason": "<one sentence>"}'
        )
        system = (
            "You evaluate whether a learner's answer meets specific success criteria. "
            "Be strict — generic, partial, or off-topic answers are not adequate. "
            "Respond only with valid JSON."
        )
        try:
            raw    = llm_chat_json(self.model, system, prompt, self.api_key, self.base_url)
            result = _extract_json(raw)
            return bool(result.get("adequate", False))
        except Exception as e:
            print("[probe eval error] " + str(e))
            return len(user_input.split()) >= 10  # fallback heuristic

    def _advance_to_next_probe(self):
        """Move to the next probe in the queue and deliver it, or conclude."""
        self.probe_index += 1
        if self.probe_index >= len(self.probe_queue):
            self.phase = "concluded"
            closing = (
                "Thank you — that concludes the assessment. "
                "Your responses will now be evaluated."
            )
            self.probe_history.append({"role": "assistant", "content": closing})
            self.history.append({"role": "assistant", "content": closing})
            return closing, True

        next_probe = self.probe_queue[self.probe_index]
        next_probe["_followup_sent"] = False
        delivery = self._format_probe_delivery(next_probe)
        next_probe["exchange"].append({"role": "assistant", "content": delivery})
        self.probe_history.append({"role": "assistant", "content": delivery})
        self.history.append({"role": "assistant", "content": delivery})
        return delivery, False

    def _format_probe_delivery(self, probe):
        """Ask the LLM to deliver the probe naturally, referencing the learner's prior words."""
        probe_text     = probe.get("probe_text", "")
        probe_type     = probe.get("probe_type", "sequencing")
        target_kp      = probe.get("target_key_point", "")
        recall_summary = self.recall_transcript[-800:] if self.recall_transcript else ""

        type_guidance = {
            "sequencing":  "You are confirming/clarifying step order and completeness.",
            "how":         "You are probing HOW the learner would execute — specific actions, states, amounts.",
            "rationale":   "You are probing WHY — the purpose or goal behind the action.",
            "decision":    "You are probing a decision point or 'what if' — adaptive reasoning.",
            "error":       "You are probing error detection and recovery knowledge.",
            "edge_case":   "You are probing a boundary condition — when the standard approach does not apply.",
        }.get(probe_type, "")

        prompt = (
            "SCENARIO: " + self.scenario.get("situation", "") + "\n\n"
            "LEARNER'S RECALL (excerpt):\n" + recall_summary + "\n\n"
            "PROBE TYPE: " + probe_type + "\n"
            "GUIDANCE: " + type_guidance + "\n"
            "TARGET KEY POINT: " + target_kp + "\n"
            "PROBE QUESTION TO DELIVER: " + probe_text + "\n\n"
            "Rephrase this probe question naturally for the examiner to ask, "
            "referencing the learner's specific words from their recall where possible. "
            "Ask only ONE focused question. "
            "Tone: curious and collaborative, not interrogative. "
            "Do NOT volunteer correct answers or hint at what was missed. "
            "Write only the question, no preamble."
        )
        system = (
            "You are a disciplined CTA examiner delivering structured knowledge probes. "
            "Ask one focused question. Reference the learner's prior words. "
            "Never volunteer information or hint at correct answers. "
            "Write only the question."
        )
        try:
            raw = llm_chat(self.model, system, prompt, self.api_key, self.base_url).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # take first non-empty line
            for line in raw.splitlines():
                if line.strip():
                    return line.strip().replace("**", "").replace("*", "").replace("_", "")
            return probe_text
        except Exception as e:
            print("[probe delivery error] " + str(e))
            return probe_text

    def _get_probe_followup(self, user_input, probe):
        """Generate a single follow-up clarification for an inadequate response."""
        prompt = (
            f'PROBE ASKED: "{probe.get("probe_text", "")}"\n'
            f'LEARNER RESPONSE: "{user_input}"\n'
            f'SUCCESS CRITERIA: {probe.get("success_criteria", "")}\n\n'
            "The learner's response was incomplete or vague. "
            "Write ONE short follow-up question to elicit the missing detail. "
            "Do not ask multiple questions. Do not hint at the correct answer. "
            "Write only the follow-up question."
        )
        system = (
            "You are a CTA examiner. Ask one focused follow-up to get more specific information. "
            "Never volunteer correct answers. Write only the question."
        )
        try:
            raw = llm_chat(self.model, system, prompt, self.api_key, self.base_url).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            for line in raw.splitlines():
                if line.strip():
                    return line.strip()
            return FALLBACK_PROBE
        except Exception as e:
            print("[probe followup error] " + str(e))
            return FALLBACK_PROBE


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DRAFT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario_draft(description, model, api_key, base_url):
    prompt = (
        "You are an instructional designer creating a CTA-based performative assessment scenario.\n\n"
        'Based on this description: "' + description + '"\n\n'
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "description": "<one sentence summary>",\n'
        '  "situation": "<2-3 sentences in second person (You are...) that describe the scene WITHOUT '
        'asking the learner to do or fix anything — set the scene only>",\n'
        '  "user_role": "<the learner\'s role, e.g. nurse, driver, technician>",\n'
        '  "constraints": ["<must-satisfy rule 1>", "<rule 2>"],\n'
        '  "expert_answer": "<complete ideal step-by-step response>",\n'
        '  "key_points": ["<short phrase 1>", "<phrase 2>", "<6-10 total>"],\n'
        '  "rubric": {"<key point phrase>": <weight 1-4>},\n'
        '  "decision_points": ["<decision moment 1>", "<decision moment 2>"],\n'
        '  "failure_modes": ["<failure description 1>", "<failure description 2>"],\n'
        '  "edge_cases": ["<edge case 1>", "<edge case 2>"],\n'
        '  "scoring_weights": {"coverage": 0.6, "quality": 0.4},\n'
        '  "probe_bank": [\n'
        '    {\n'
        '      "probe_type": "sequencing",\n'
        '      "target_key_point": "<key point phrase>",\n'
        '      "probe_text": "<the exact question to ask>",\n'
        '      "success_criteria": "<what a satisfactory answer looks like>"\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- situation: second person, present tense; describe the scene only, not what to do\n"
        "- key_points: 6-10 short phrases\n"
        "- rubric weights: 1=low, 2=medium, 3=high, 4=critical\n"
        "- probe_bank: include at least one probe of each type: "
        "sequencing, how, rationale, decision, error, edge_case\n"
        "- probe_text: a complete question the examiner asks verbatim\n"
        "- success_criteria: describes what a satisfactory learner answer must include\n"
        "Return only the JSON, no other text."
    )

    system = (
        "You are an expert instructional designer specialising in Cognitive Task Analysis. "
        "Create clear, realistic performative assessment scenarios with structured probe banks. "
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
