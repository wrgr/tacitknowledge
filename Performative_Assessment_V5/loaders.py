"""
loaders.py — JSON loading for scenarios and free-response prompts.
Scenarios are plain Python dicts — just use scenario["title"], scenario["situation"], etc.
Prompts (free-response tasks) follow the same pattern.
"""

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# FREE-RESPONSE PROMPT LOADING
# Reads the JSON files in the prompts/ folder.
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt(path):
    # open and parse one prompt JSON file
    with open(path) as f:
        data = json.load(f)

    # fill in default values for optional fields so callers don't need to guard every key
    if "description" not in data:
        data["description"] = ""
    if "word_limit" not in data:
        data["word_limit"] = None
    if "constraints" not in data:
        data["constraints"] = []
    if "metadata" not in data:
        data["metadata"] = {}

    # do the same for each expert answer inside the prompt
    for ea in data.get("expert_answers", []):
        if "key_points" not in ea:
            ea["key_points"] = []
        if "rubric" not in ea:
            ea["rubric"] = {}

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
