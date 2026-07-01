"""
loaders.py — JSON loading for scenarios and free-response prompts.
Scenarios are plain Python dicts — just use scenario["title"], scenario["situation"], etc.
Prompts (free-response tasks) follow the same pattern.
"""

import json
import re
from pathlib import Path

# FR key points: importance replaces the old flat rubric-weight mapping (construct/exemplar
# brief, Part A). Same CRITICAL/HIGH/MEDIUM/LOW scale used elsewhere in this system.
FR_IMPORTANCE_LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
_LEGACY_WEIGHT_TO_IMPORTANCE = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}


def _slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug or "point"


def _migrate_fr_key_points(ea):
    """Part A2: wrap legacy flat-string key_points into the construct/exemplar shape.

    A flat string becomes {id, construct: <original string, unchanged>, exemplars: [],
    importance: <old rubric weight for that string, or MEDIUM>}. Already-migrated dict
    entries are normalised (missing fields filled in) so hand-authored prompts and
    AI-drafted prompts work the same way. IDs are de-duplicated so two key points that
    slugify to the same id don't collide.
    """
    raw_points = ea.get("key_points", [])
    legacy_rubric = ea.get("rubric", {}) if isinstance(ea.get("rubric"), dict) else {}
    seen_ids = set()
    migrated = []

    def _unique_id(base):
        candidate = base
        n = 2
        while candidate in seen_ids:
            candidate = f"{base}_{n}"
            n += 1
        seen_ids.add(candidate)
        return candidate

    for kp in raw_points:
        if isinstance(kp, str):
            construct   = kp
            weight      = legacy_rubric.get(kp, 2)
            importance  = _LEGACY_WEIGHT_TO_IMPORTANCE.get(weight, "MEDIUM")
            migrated.append({
                "id":         _unique_id(_slugify(construct)),
                "construct":  construct,
                "exemplars":  [],
                "importance": importance,
            })
        elif isinstance(kp, dict):
            construct  = kp.get("construct", "")
            importance = kp.get("importance")
            if importance not in FR_IMPORTANCE_LEVELS:
                importance = "MEDIUM"
            base_id = kp.get("id") or _slugify(construct)
            migrated.append({
                "id":         _unique_id(base_id),
                "construct":  construct,
                "exemplars":  [e for e in (kp.get("exemplars") or []) if isinstance(e, str) and e.strip()],
                "importance": importance,
            })
        # silently skip malformed entries (neither str nor dict)

    ea["key_points"] = migrated
    # rubric is superseded by per-point importance for FR prompts -- drop it so nothing
    # downstream mistakes it for the still-active scenario-mode rubric format.
    ea.pop("rubric", None)


# ─────────────────────────────────────────────────────────────────────────────
# FREE-RESPONSE PROMPT LOADING
# Reads the JSON files in the prompts/ folder.
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt(path):
    # open and parse one prompt JSON file
    with open(path) as f:
        data = json.load(f)

    if "id" not in data:
        data["id"] = Path(path).stem

    # fill in default values for optional fields so callers don't need to guard every key
    if "description" not in data:
        data["description"] = ""
    if "word_limit" not in data:
        data["word_limit"] = None
    if "constraints" not in data:
        data["constraints"] = []
    if "metadata" not in data:
        data["metadata"] = {}
    if "process_overlay_enabled" not in data:
        data["process_overlay_enabled"] = True
    if not data.get("general_guidance"):
        # Shown to the learner before writing (see general_guidance schema field). Authored
        # separately from key_points — never auto-derived, to avoid leaking scored points.
        data["general_guidance"] = (
            "Explain your reasoning clearly, including why each part matters and how it "
            "connects to the rest."
        )

    # do the same for each expert answer inside the prompt, then migrate key_points to the
    # construct/exemplar shape (Part A2 -- handles both legacy flat strings and already-
    # migrated dicts, so this is safe to call unconditionally)
    for ea in data.get("expert_answers", []):
        if "key_points" not in ea:
            ea["key_points"] = []
        _migrate_fr_key_points(ea)

    return data


def load_prompts(folder):
    # load every .json file in the given folder and return them as a list
    folder = Path(folder)
    # sorted so the menu order is predictable across runs
    return [load_prompt(path) for path in sorted(folder.glob("*.json"))]


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO LOADING
# Reads the JSON files in the scenarios/ folder.
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(path):
    # open and parse one scenario JSON file
    with open(path) as f:
        data = json.load(f)

    # inject id from filename if not present in the JSON
    if "id" not in data:
        data["id"] = Path(path).stem

    # fill in default values for optional fields so we don't have to check later
    if "description" not in data:
        data["description"] = ""
    if "user_role" not in data:
        data["user_role"] = "participant"
    if "max_turns" not in data:
        data["max_turns"] = 8
    if "constraints" not in data:
        data["constraints"] = []
    if "metadata" not in data:
        data["metadata"] = {}

    # new CTA probe-bank schema fields — fall back to empty lists so existing
    # scenarios continue to work without modification
    if "decision_points" not in data:
        data["decision_points"] = []
    if "failure_modes" not in data:
        data["failure_modes"] = []
    if "edge_cases" not in data:
        data["edge_cases"] = []
    if "probe_bank" not in data:
        # empty → probes generated dynamically at runtime from key_points + expert answer
        data["probe_bank"] = []
    if "scoring_weights" not in data:
        data["scoring_weights"] = {"coverage": 0.6, "quality": 0.4}

    # do the same for each expert answer inside the scenario
    for ea in data.get("expert_answers", []):
        if "key_points" not in ea:
            ea["key_points"] = []
        if "rubric" not in ea:
            ea["rubric"] = {}

    return data  # the scenario is just a plain dict — no special class needed


def load_scenarios(folder):
    # load every .json file in the given folder and return them as a list
    folder = Path(folder)
    return [load_scenario(path) for path in sorted(folder.glob("*.json"))]  # sorted so the menu order is predictable
