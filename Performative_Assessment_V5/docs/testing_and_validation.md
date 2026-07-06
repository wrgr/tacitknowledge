# Testing and Validation

Run these checks after changing prompts, scoring, reports, or process-analysis code:

```powershell
python -m compileall Performative_Assessment_V5
python -m unittest discover -s Performative_Assessment_V5\tests
```

## What These Checks Cover

- `test_writing_process_calibration.py` protects the Phase 0 process-trace signals:
  difficulty points, paste/authenticity flags, confidence calibration, and quadrant
  classification.
- `test_prompt_inventory.py` protects the Phase 1a/1b prompt inventory by checking
  that prompt JSON files load, IDs are unique, expert answers include scoring
  evidence, rubric keys line up with key points, and circuit prompts carry Phase 1b
  research metadata.

These tests are intentionally local and deterministic. They do not require an LLM
API key, network access, a running Flask server, or a populated reports folder.
