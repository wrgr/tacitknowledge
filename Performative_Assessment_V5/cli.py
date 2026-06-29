"""
Performative Assessment — terminal interface.
Run with:  python3 cli.py
"""

import sys           # used for sys.exit() if no scenarios are found
from pathlib import Path

import config        # loads PROVIDERS, DEFAULT_PROVIDER, REPORTS_DIR
import engine        # all the program logic lives in engine.py

SCENARIOS_DIR = Path(__file__).parent / "scenarios"  # path to the scenarios/ folder
WIDTH = 64           # width of the decorative divider lines
RECALL_DONE_WORDS = {"done", "end", "finished", "complete"}  # words that end recall phase
_provider_cfg = {}  # set in main() once the provider is resolved; used by run_assessment()


def divider(char="─"):
    print(char * WIDTH)  # print a full-width line, e.g. "────────────────────"


def pick_scenario(scenarios):
    # show a numbered list and ask the user to pick one
    print("\nAvailable scenarios:\n")
    for i, s in enumerate(scenarios, 1):  # enumerate starts numbering at 1
        print("  [" + str(i) + "] " + s["title"])
        if s["description"]:
            print("      " + s["description"])
    print()
    while True:  # keep asking until we get a valid number
        choice = input("Select a scenario (1–" + str(len(scenarios)) + "): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
            return scenarios[int(choice) - 1]  # -1 converts from 1-based display to 0-based list
        print("  Please enter a number between 1 and " + str(len(scenarios)) + ".")


def run_assessment(scenario):
    p = _provider_cfg
    runner = engine.ScenarioRunner(scenario, model=p["model"], api_key=p["api_key"], base_url=p["base_url"])

    print("\n  Walk through everything you would do, step by step.")
    print("  Type 'done' when you have said everything.\n")
    divider("=")
    print("\n" + runner.start() + "\n")
    divider()

    # ── Recall phase ──────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in RECALL_DONE_WORDS:
            break  # end recall, move to probing

        print("\n[...]\n", flush=True)

        try:
            narration, concluded = runner.respond(user_input)
        except (ConnectionError, KeyboardInterrupt):
            print("(interrupted)")
            break

        print(narration + "\n")
        divider()

        if concluded:
            if runner.recall_history or runner.probe_history:
                return runner
            return None

    # ── Transition to probing ──────────────────────
    if not runner.recall_history:
        return None  # nothing was said

    print("\n[...]\n", flush=True)
    try:
        first_probe, concluded = runner.end_recall()
    except (ConnectionError, KeyboardInterrupt):
        print("(interrupted)")
        return runner

    if concluded:
        return runner

    divider("=")
    print("  PROBING PHASE")
    print("  Answer each question as specifically as you can.\n")
    divider("=")
    print("\n" + first_probe + "\n")
    divider()

    # ── Probing phase ──────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        print("\n[...]\n", flush=True)

        try:
            narration, concluded = runner.respond(user_input)
        except (ConnectionError, KeyboardInterrupt):
            print("(interrupted)")
            break

        print(narration + "\n")
        divider()

        if concluded:
            break

    return runner


def show_evaluation(evaluations):
    for ev in evaluations:
        if ev.get("coverage_score") is not None and ev.get("quality_score") is not None:
            print(f"\n  Coverage:   {ev['coverage_score']:.0%}  (steps recalled)")
            print(f"  Quality:    {ev['quality_score']:.0%}  (depth of reasoning)")
            print(f"  Overall:    {ev['score']:.0%}\n")
        else:
            print("\nScore: " + f"{ev['score']:.0%}" + "\n")
        if ev["strengths"]:
            print("Strengths:")
            for s in ev["strengths"]:
                print("  + " + s)
            print()
        if ev["gaps"]:
            print("Gaps:")
            for g in ev["gaps"]:
                print("  - " + g)
            print()
        if ev["feedback"]:
            print("Feedback: " + ev["feedback"])


def main():
    print("=" * WIDTH)
    print("  Performative Assessment System")
    print("=" * WIDTH)

    scenarios = engine.load_scenarios(SCENARIOS_DIR)  # read all scenario JSON files
    if not scenarios:
        print("No scenario files found in " + str(SCENARIOS_DIR))
        sys.exit(1)  # exit with an error code so the shell knows something went wrong

    global _provider_cfg
    _provider_cfg = config.PROVIDERS.get(config.DEFAULT_PROVIDER, next(iter(config.PROVIDERS.values())))
    use_llm = engine.llm_is_available(_provider_cfg["api_key"])
    if use_llm:
        print("LLM configured — provider: '" + config.DEFAULT_PROVIDER + "', model: '" + _provider_cfg["model"] + "'")
    else:
        print("Warning: API key not configured — falling back to keyword scoring.")

    p = _provider_cfg
    session = engine.Session(use_llm, model=p["model"], api_key=p["api_key"], base_url=p["base_url"])

    while True:  # outer loop lets the user run multiple scenarios
        scenario = pick_scenario(scenarios)
        print("\n" + "=" * WIDTH + "\n  " + scenario["title"] + "\n" + "=" * WIDTH)

        runner = run_assessment(scenario)

        if runner:
            print("\n[Evaluating...]\n", flush=True)
            divider()
            evaluations = session.evaluate(
                scenario,
                transcript=runner.transcript(),
                recall_transcript=runner.recall_transcript,
                probe_transcript=runner.probe_transcript,
            )
            show_evaluation(evaluations)
            divider()

        answer = input("\nRun another scenario? [Y/n]: ").strip().lower()
        if answer == "n":
            break  # "n" exits; anything else continues

    # print session summary
    print("\n" + "=" * WIDTH)
    print(
        "  Session complete — " + str(session.total_evaluations()) + " evaluation(s), "
        "average score: " + f"{session.average_score():.0%}"
    )

    # offer to generate a report (only available with LLM, and only if something was evaluated)
    if session.results and use_llm:
        answer = input("\nGenerate instructor report? [Y/n]: ").strip().lower()
        if answer != "n":
            print("[Generating report...]", flush=True)
            path = engine.generate_report(
                session,
                model=session.model,
                api_key=session.api_key,
                base_url=session.base_url,
                output_dir=config.REPORTS_DIR
            )
            print("Report saved → " + str(path))

    print("=" * WIDTH)


if __name__ == "__main__":
    main()  # only run when this file is executed directly, not when imported
