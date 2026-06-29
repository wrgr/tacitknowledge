"""
engine.py — re-exports all program logic so that `import engine` still works.
"""

from llm import (
    llm_is_available,
    get_configured_providers,
    get_available_models,
    llm_generate,
    llm_chat,
)
from loaders import (
    load_prompt,
    load_prompts,
    load_scenario,
    load_scenarios,
)
from scoring import (
    check_fr_keywords,
    check_fr_with_llm,
    score_free_response_with_keywords,
    score_free_response_with_llm,
    score_with_keywords,
    score_with_llm,
)
from runner import (
    ScenarioRunner,
    generate_scenario_draft,
    generate_prompt_draft,
    FALLBACK_RECALL_ACK,
    FALLBACK_PROBE,
)
from session import Session
from thinking import analyse_thinking_profile
from reports import generate_report, generate_fr_report
