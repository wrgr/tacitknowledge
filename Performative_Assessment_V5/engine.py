"""
engine.py — re-exports all program logic so that `import engine` still works.

Logic lives in:
  llm.py      — LLM provider dispatch and JSON extraction
  loaders.py  — JSON loading for scenarios and free-response prompts
  scoring.py  — keyword and LLM-based scoring
  runner.py   — ScenarioRunner and generate_scenario_draft
  session.py  — Session class
  thinking.py — Honey & Mumford / SOLO taxonomy analysis
  reports.py  — Markdown report generation
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
    score_free_response_with_keywords,
    score_free_response_with_llm,
    score_with_keywords,
    score_with_llm,
)
from runner import (
    ScenarioRunner,
    generate_scenario_draft,
    generate_prompt_draft,
    FALLBACK_PROMPTS,
    FALLBACK_CLOSING,
)
from session import Session
from thinking import analyse_thinking_profile
from reports import generate_report, generate_fr_report
