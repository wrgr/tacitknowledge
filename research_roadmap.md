# Research Roadmap: Demonstrating Competence Through Process, Not Just Outcome

*Working document for the research team. Six trained undergraduate researchers, approximately seven weeks. This is a plan to break, not a script to follow: the central premise is a hypothesis we test early, not a foundation we build on.*

*We offer this as three pair-based tasks but can be arranged as you think best - it will flow more smoothly if you split it up!*


## The goal, in one paragraph

We are building a way to assess whether someone is competent at a diagnostic task by looking not only at whether they reached the right answer, but at how they got there and how well they can justify it. The long-term target is high-consequence settings such as technician readiness, where a human expert owns the final judgment and our tool supplies the evidence. We start in domains where the ground truth is formally checkable: writing, then circuit reasoning on paper, then circuits in a simulator, then a physical bench. The seven weeks run the cheap stages, verify the physical channel, and open a contingent window to collect real data.

## What is assumption and what is established (read this first)

The load-bearing claim, that the process trace carries competence signal beyond what the final answer shows, is a hypothesis. We test it in Week 2 on existing data. Until it clears, treat it as the thing we are trying to break, not the thing we are building on. If it fails, the project changes shape, and learning that in Week 2 is a result, not a setback.

Two other framings to hold honestly. The post-determination data collection is a stretch goal with two hard preconditions, not the main deliverable; the term is complete without it. And the participant-facing materials we will write, the worksheets and exercise instructions handed to people being assessed, are a separate kind of artifact from this roadmap: plain task language, neutral disclosure of what is recorded, and none of the research machinery in this document. Knowing where that line sits is itself part of doing this correctly.

## The arc and its gates

Each stage has a question that must clear before the next is worth the effort.

- **Phase 0, writing (warm-up).** Assess an essay from its writing process, not only the finished text. Cleanest possible rehearsal of the whole method with zero hardware risk.
- **Phase 1a, text-only baseline.** Assessment from what a learner writes or says, no instrumented trace. Establishes the ceiling of text-only assessment so later stages can show what the trace adds.
- **Phase 1b, descriptive circuit analysis on paper, no hardware.** Circuit reasoning tasks where the learner reads, predicts, and explains circuit behavior in writing, with their working captured. Bridges the writing warm-up to the circuit domain without yet introducing the simulator or the bench.
- **Phase 2, the simulator.** The instrumented digital logic fault-diagnosis exercise: logged simulator, captured trace, structured justification checked against that trace.
- **Phase 3, the physical bench.** Build and verify the instrumented bench. The checkout, can the parse recover known motions, is run this term. A real-learner collection on the bench is the stretch goal, gated below.

**The Week 2 gate.** On the existing DEEDS dataset, do stronger and weaker students produce distinguishable diagnostic traces? If yes, the trace-based premise holds. If no, we lean harder on the structured justification and less on the raw trace, and the roadmap absorbs the pivot.

**The collection gate (stretch goal).** A real-learner collection happens only if both are true: the exempt determination came back exempt, and the bench checkout passed. Data off an unverified bench is uninterpretable, because a learner-versus-record divergence could be the person or the instrument and you cannot tell which. Both preconditions, or no collection.

## Tracks and ownership

Six students, three pairs around buildable cores. The conceptually hard, expert-dependent work, the evidence model, the discrepancy taxonomies, the validity argument, is shared and faculty-led; students build tooling and first-pass crosswalks that faculty validate, rather than generating expert content themselves.

**Pair A: Instrumented environments and capture.** Owns Phase 2 and Phase 3 plumbing. Build or wrap a logic simulator that logs learner actions; implement the worksheet flow; capture traces in a clean documented format; stand up consent and data-handling plumbing. Then build the physical bench, instrument contact and timing, and run the motion battery. *Done when:* a learner can complete the simulator exercise end to end with full trace capture, and the bench provably recovers known motion sequences.

**Pair B: Trace and process representations.** Owns the Week 2 gate and the representations that run through every phase. Start immediately on the existing DEEDS dataset, no waiting on Pair A. Parse logs; represent the diagnostic process as action sequences and strategy signatures; run the gate analysis. This track is where the tacit idea lives concretely: the diagnostic process is the tacit knowledge, and representing it is the deliverable. *Done when:* there is a documented trace representation and a clear result on whether strong and weak traces separate.

**Pair C: Writing warm-up, text baseline, and annotation pipeline.** Owns Phase 0, Phase 1a, and the bridge into 1b. Build the essay-process exercise with a keystroke or revision log as the process channel and, where AI tools are in use, the AI-conversation trace as a distinct channel with its own attribution question. Build the text-only baseline and the pipeline that tags text with knowledge components and correctness. *Done when:* there is an essay-process capture, a text-only baseline number to beat, and a working annotation tool.

**Shared and faculty-led (everyone contributes a slice):**

- *Evidence model and crosswalks.* Competency definitions and rubrics come from faculty, anchored to standard outcomes and the existing concept inventory. Students build tooling and first-pass crosswalks from rubric line to specific trace-and-justification signature, which faculty validate. The load-bearing deliverable; the one students cannot own alone. Note: the writing-domain discrepancy taxonomy is harder to specify than the circuit one, not easier, because the space of legitimate writing processes is far wider than the space of legitimate diagnostic paths. Distinguishing idiosyncratic-but-legitimate process from process that signals the work is not the learner's is the real intellectual content of the warm-up.
- *Literature review.* Each pair owns the slice nearest its track; one student coordinates assembly. The output is a sharpened set of research questions, each section ending in the open question it leaves for our design, not a reading list.
- *Data governance, consent, and the IRB determination.* Pair A builds the plumbing; faculty owns the policy. File the exempt determination in Week 1; the determination itself takes about two days, but file early so the answer is settled long before it gates anything.
- *Validity and method-equivalence plan.* Faculty-owned design; students contribute measurement tooling. The eventual claim is modest: assessing against trace plus justification agrees with the established assessment where they overlap and adds signal where the old method was blind.

## Week by week

**Week 1.** File the exempt determination. Pair B pulls the DEEDS dataset and reproduces a load and parse. Pair A selects the simulator substrate and specs the bench. Pair C stands up the essay-process capture and the annotation pipeline. Faculty delivers the first competency anchor and seed tasks for writing, 1b, and the simulator.

**Week 2.** Pair B runs the gate analysis on existing data. Pair A has a simulator logging prototype on one task and bench parts in hand. Pair C runs a first essay-process capture and has annotation working on sample text. First evidence-model crosswalk drafted with faculty.

**Week 3.** Gate review: proceed or pivot. Phase 0 and 1a producing data. Stand up Phase 1b descriptive-circuit tasks on the existing capture tooling, since 1b needs no new instrument. Integrate worksheet plus logging end to end on one simulator task. Literature slices drafted.

**Week 4.** Bench checkout: run the fixed motion battery until the parse provably recovers known sequences. Internal dry run of the simulator exercise; capture real traces and justifications. Begin the justification-versus-trace comparison tooling.

**Week 5.** Analyze the dry run: can the justification-versus-trace divergence be scored against the crosswalk? Confirm checkout pass or document what failed. Confirm the determination is in hand. Both preconditions now known.

**Weeks 6 to 7, contingent.** If both preconditions cleared, run a small instructional collection against the verified bench and the simulator instrument, leaving real time at the end for analysis and write-up rather than collecting to the last day. If either failed, punt the collection as planned. Assemble the literature into the sharpened question set. Produce the analysis, a handoff document, and a short demo.

## Deliberately parked, and why

- *Video capture of tacit behavior.* Reintroduces hard sensing problems, precise contact and hand-object localization under occlusion, that are unnecessary while tasks live in writing, on paper, or in a simulator. 
- *Any readiness-gate or certification claim.* High-stakes and far beyond what a pilot supports. We test a method; we certify no one.

Parking these explicitly matters as much as the active list: it keeps a motivated team from wandering into the expensive problems before the cheap pieces are running.

## What success looks like in seven weeks

An essay-process exercise and a text-only baseline; an empirical answer to whether traces carry competence signal; descriptive-circuit and simulator instruments producing clean trace-plus-justification data; first-pass, expert-reviewed evidence models in at least the circuit domain; a verified physical bench with a passed motion-parse checkout; a filed exempt determination; a sharpened, literature-grounded question set; and, if both preconditions cleared, a first small real-learner collection. The term is complete and defensible even if the collection does not happen.

---

## Reading list

Grouped by the track each set serves most. Each pair starts with its own group; everyone reads the two cross-cutting anchors.

**Cross-cutting anchors (everyone):**

- Black and Wiliam (1998), *Inside the Black Box*. The foundational formative-versus-summative distinction the whole project rests on.
- The assessment-analytics framing of process data: process traces situated inside validity frameworks, asking what claims about competence the available evidence supports, and using sequence analysis where multiple solution paths mean a correct answer does not capture how it was reached. (MDPI *Encyclopedia*, "Assessment Analytics in Digital Assessments," 2026; find the current version.)

**Pair B, trace and process representations:**

- Scarlatos, Baker, and Lan (2025), "Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs" (LAK 2025; arXiv 2409.16490). Defines assessment from open-ended process rather than discrete items; note that it is post-hoc, not real-time, and tops out at modest accuracy.
- Hooshyar et al. (2026), neural-symbolic knowledge tracing / Responsible-DKT (arXiv 2604.08263). The case that an explicit, interpretable estimator should be kept separate from the language model, plus the cautions on why LLMs alone make unstable learner models.
- The DEEDS line of work on student session logs (Donzellini and Ponta for the environment; Vahdat et al. 2015 and Hussain et al. 2019 for mining the logs to predict student difficulty). This is the dataset the Week 2 gate runs on.

**Pair A, instrumented environments and the bench:**

- The SHERLOCK and HYDRIVE precedents: technician-troubleshooting tutors built on an evidence-centered design with an explicit student model and analysis of the troubleshooting path, not just the outcome. The direct ancestors of the bench.
- The DEEDS environment papers (shared with Pair B) for how a logged digital-design simulator is structured.
- Cognitive task analysis as the method for eliciting the expert path the bench scores against (Klein and Crandall, applied cognitive task analysis). The expert path is what the motion battery and the evidence model compare to.

**Pair C, writing warm-up, text baseline, annotation:**

- Evidence-centered design (Mislevy, Steinberg, and Almond, 2003) and stealth assessment (Shute, 2011; Shute and Rahimi, 2017): how to turn observed process into defensible competency claims, and how feedback interacts with assessment quality.
- The assistance dilemma (Koedinger and Aleven, 2007) and the help-seeking finding that learners are poor judges of when they need help: directly relevant to the AI-in-the-loop attribution problem in the writing warm-up.
- Multimodal learning analytics for procedural and communication skills, as orientation for the parked video thread and for how process channels get fused.

*Citations are working references to verify against the primary sources, not final bibliographic entries. Several arXiv identifiers should be confirmed before circulation.*

---

*Draft roadmap prepared with AI assistance. Working planning document for internal review and revision.*
