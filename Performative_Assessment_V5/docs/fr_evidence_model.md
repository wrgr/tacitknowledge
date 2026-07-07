# FR Writing-Process Evidence Model

This is the evidence model (Mislevy, Steinberg & Almond, 2003) for every writing-process
signal the Free Response (FR) mode computes: what claim it is allowed to support, how much
confidence that claim can bear, and what plausible alternative interpretations it does not
rule out. Bulut & Yildirim-Erbasli (2026) is the reason this table exists at all — process
indicators (response time, revisions, navigation) are behavioral *proxies*, not direct
measures of cognition, and they only strengthen an evidentiary claim when the claim they
support was stated *before* the data was treated as evidence. Undisciplined process mining
produces numbers that feel more precise than they are defensible.

**Rule: every future process signal added to this system must add a row to this table
before it is wired into scoring or reporting.** A signal with no row is not permitted to
render as a claim anywhere in the FR report — at most it may appear as a raw, unlabeled
number in a debug/instructor-only view.

| Signal | Claim it supports | Confidence | Does NOT rule out |
|---|---|---|---|
| Pause + revision co-occurrence at a text region (`difficulty_points`) | The writer encountered difficulty forming that part of the explanation | Low-moderate; positional correlation only | Distraction, an unrelated interruption, or typing/technical friction |
| Revision moves text toward conditional/goal-linked/consequence-aware phrasing (`revision_toward_quality`) | The writer engaged in real-time self-monitoring toward a more complete explanation | Moderate; requires the LLM judgment call (Part B of the writing-process brief) | Coincidental phrasing improvement, or a later paste of better phrasing from elsewhere |
| Large paste with little surrounding revision (`authenticity`) | The final text may not be fully the learner's own original composition | Low as a standalone signal; ambiguous per Koedinger & Aleven's Assistance Dilemma | Legitimate reuse of the learner's own prior notes, appropriate quotation, or simply confident single-pass writing |
| Frictionless/fast completion, low revision, strong product (`quadrant`: authenticity_review / disengaged_shallow_confident) | — no single confident claim — | N/A by design | Could be genuine fluent competence, prior preparation, or unflagged reuse; cannot be distinguished from process data alone |
| Confidence rating collapse, pre-write vs. post-write (`confidence_calibration`) | The writer's forced explanation exposed a gap between perceived and actual understanding | Moderate-high; this is the most directly validated mechanism in the whole system (Rozenblit & Keil, 2002) | A learner who is simply a harsh self-rater in general, not specifically because of a gap this task exposed |
| Coverage Score — key points addressed in unaided single-pass writing (`score`) | The learner demonstrated this knowledge spontaneously, without prompting | Moderate — grounded by evidence-span verification, but recall-limited by design | Absence of a key point may reflect production/recall failure under single-pass, unaided conditions rather than absent knowledge — the same omission phenomenon (unprompted recall systematically underrepresents true knowledge) that justifies the scenario mode's probing architecture applies here, uncorrected, since FR has no probing phase |
| SOLO level (derived from Coverage/Quality) | Structural complexity of the response — how many required constructs were addressed and how well-integrated the reasoning around them was | As strong as the underlying Coverage/Quality grounding it's derived from (no new evidence introduced) | A Multistructural label may reflect a deliberate stylistic choice (breadth over depth) rather than an inability to integrate reasoning; this method cannot detect Extended-Abstract-level generalization beyond the given task, and never claims to |

The Coverage Score row above was added after the fact — Coverage Score had been treated
as the ground-truth measure everything else in this table gets checked against, rather
than a signal needing its own audit. It gets no exemption from the rule above: a missed
key point is evidence of non-production under single-pass conditions, not proof of absent
knowledge.

FR previously rendered a Honey & Mumford (Activist/Reflector/Theorist/Pragmatist)
learning-style classification with no row in this table — a violation of the rule
above, and the reason it was removed rather than given one after the fact. Honey &
Mumford has weak validity even in its own validated (80-item self-report) instrument
(Coffield et al., 2004), and was being inferred here from a single short paragraph plus
a few keystroke metrics, a much thinner basis than the original instrument. Pashler,
McDaniel, Rohrer & Bjork (2008) additionally found no credible evidence that acting on a
learning-style label improves outcomes. It does not appear in this table because it does
not exist in FR at all, not because it was overlooked. Scenario mode still computes and
renders it (`thinking.analyse_thinking_profile`), unaffected — this table covers FR only.

## Language audit

Every sentence the FR report generates about a process signal must not claim more certainty
than the row above licenses, and must carry that row's "does not rule out" alternative in
the same sentence or the next one — not only in the report's standing caution note. This
is implemented as:

- `writing_process.py` — static `ALT_*` constants sourced verbatim from this table's
  "Does NOT rule out" column, attached to `difficulty_points`, `authenticity`, and the
  `quadrant` interpretation; the revision-toward-quality LLM call is asked to return its
  own one-line `alternative_explanation` per instance (that judgment is not stable across
  instances the way the others are, so it isn't a static lookup).
- `reports.py` — `_append_process_overlay()` renders the alternative next to each claim.
- `reports.py` — `_FR_COVERAGE_CALIBRATION_NOTE`, rendered directly under the Score line
  in `generate_fr_report()`, carries the Coverage Score row's alternative. It is a
  standing calibration note next to the number itself, not a single end-of-report
  disclaimer — this is the one row whose language rule lives outside
  `_append_process_overlay()`, because Coverage Score is the product score, not a
  process-overlay signal.

Audited and corrected against this table:
- The authenticity line no longer reads as an accusation ("authenticity concern: elevated").
  It now states the observed fact, the confidence level implied by the table (low as a
  standalone signal), and the alternative in the same sentence, e.g.: "A large portion of
  this response was pasted with limited surrounding revision — this can reflect reuse of
  the learner's own notes, direct quotation, or reduced authorship; recommend the
  instructor confirm with the learner if this matters for the assessment's purpose."
- The `authenticity_review` and `disengaged_shallow_confident` quadrant interpretations
  state plainly that a frictionless/pasted process has no single confident claim, per the
  table's "N/A by design" row, rather than asserting authorship reduction outright.
- The confidence-collapse finding is the only process signal allowed to be stated as a
  clear, named finding without heavy hedging, because it is the one row rated
  moderate-high — but it still carries its stated alternative (a generally harsh
  self-rater) next to it.
