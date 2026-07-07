# Research Export Data Dictionary

The admin research export is a CSV downloaded from `/admin/research-export.csv`.
Each row represents one scored assessment unit: one free-response report, or one
scenario inside a scenario report. The export is for analysis and review, not for
certification or readiness gating.

## Column Groups

### Identity and Task Context

| Column | Source | Meaning |
|---|---|---|
| `username` | account database | Login username for the learner whose report was exported. |
| `display_name` | account database | Learner display name. |
| `role` | account database | Account role at export time. |
| `report_file` | report filename | Markdown report file used for this row. |
| `report_type` | parsed report | `free_response` or `scenario`. |
| `task_title` | parsed report | Prompt title for free response, or scenario title. |
| `timestamp` | report filename | Timestamp inferred from the report filename, when available. |
| `export_schema_version` | export pipeline | Version of the structured export schema used to parse and persist this row. |

### Product-Only Assessment

These columns are the Phase 1a text-only baseline: they come from the submitted
answer and rubric, not from writing-process signals.

| Column | Source | Meaning |
|---|---|---|
| `product_score_percent` | parsed report | Overall score for the answer. |
| `text_only_baseline_percent` | parsed report | Current baseline score to compare against process-enriched interpretation. This intentionally mirrors `product_score_percent`. |
| `coverage_score_percent` | parsed scenario report | Scenario coverage score, if present. Empty for free-response reports. |
| `quality_score_percent` | parsed scenario report | Scenario explanation-quality score, if present. Empty for free-response reports. |
| `matched_points` | parsed report | Rubric/key-point constructs credited in the answer. |
| `missed_points` | parsed report | Rubric/key-point constructs not credited in the answer. |
| `strengths` | parsed report | Generated strengths from the report. |
| `gaps` | parsed report | Generated gaps or improvement areas from the report. |
| `word_count` | parsed report | Approximate word count from the learner submission or transcript. |

### Process-Derived Context

These columns are interpretive process signals. They are behavioral proxies, not
direct measures of cognition. Use them as supporting context and compare them
against product-only scores and human annotations.

| Column | Source | Meaning |
|---|---|---|
| `has_process_overlay` | parsed report | Whether a writing-process section was present. |
| `process_quadrant` | parsed writing process | Product/process quadrant label, such as engaged reasoning or authenticity review. |
| `effort_profile` | parsed writing process | Human-readable summary of active time, revision density, and pauses. |
| `revision_toward_quality` | parsed writing process | LLM-bounded judgment of whether revisions improved explanatory quality; may be `not assessed`. |
| `difficulty_point_count` | parsed writing process | Count of pause-plus-heavy-revision difficulty-point candidates. |
| `authenticity` | parsed writing process | Paste/revision-based authenticity signal. Ambiguous by design. |
| `confidence_calibration` | parsed writing process | Pre/post confidence change after explaining. |
| `closing_nudge_used` | parsed writing process | Whether the learner added content after the final generic recall checkpoint. Context only; not scored. |
| `process_caution` | parsed writing process | The report's standing caution that process signals are indirect supporting context, not verdicts. |

### Learner Self-Report

| Column | Source | Meaning |
|---|---|---|
| `ai_assistance_used` | learner declaration | Whether the learner declared AI assistance for a free-response submission. |
| `ai_assistance_notes` | learner declaration | Optional learner description or pasted summary of AI assistance. |

### Human Review

These columns are the annotation pipeline. They provide human labels that can be
used as comparison data for product-only and process-enriched interpretations.

| Column | Source | Meaning |
|---|---|---|
| `annotation_label` | admin sidecar annotation | Human review label: blank, `correct`, `partial`, `missing`, or `needs_expert_review`. |
| `annotation_notes` | admin sidecar annotation | Optional reviewer notes. |
| `annotation_reviewer` | admin sidecar annotation | Display name or username of the reviewer who last saved the annotation. |
| `annotation_updated_at` | admin sidecar annotation | Local server timestamp for the latest annotation save. |

### Thinking-Profile Context

| Column | Source | Meaning |
|---|---|---|
| `thinking_honey_mumford` | parsed report | Generated Honey & Mumford style label, if present. Scenario reports only — FR reports never populate this column. Older FR reports generated before the FR thinking-profile fix may still carry a value here; new FR reports leave it blank. See `docs/fr_evidence_model.md` for why FR dropped this signal. |
| `thinking_solo` | parsed report | Generated SOLO taxonomy label, if present. For scenario reports this is an LLM classification; for FR reports it is a deterministic derivation from Coverage/Quality data (see `docs/fr_evidence_model.md`) — the label values are comparable across both, but the two are not computed the same way and FR's derivation never returns `Extended Abstract`. |

## Interpretation Cautions

- `text_only_baseline_percent` is the comparison baseline. It should not be
  treated as a separate model score until a separate baseline scorer exists.
- Process columns should not be used alone to infer competence, authorship, or
  effort. The evidence model in `docs/fr_evidence_model.md` defines what each
  signal can and cannot support.
- AI-assistance fields are learner declarations. They are useful context, not
  proof that assistance was or was not used.
- Annotation labels are human judgments and should record reviewer uncertainty
  in `annotation_notes` when appropriate.
- Empty fields usually mean the source report did not contain that section, not
  that the behavior was absent.

## Recommended Analysis Use

1. Treat `text_only_baseline_percent` as the Phase 1a product-only baseline.
2. Compare process-derived fields against human `annotation_label` values.
3. Check whether process signals explain cases where similar product scores get
   different human annotations.
4. Keep AI-assistance declarations separate from writing-process signals when
   analyzing authorship or attribution questions.
