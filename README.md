HOW TO RUN
----------
Install dependencies (from inside the Performative_Assessment_V5/ folder):

    pip install -r requirements.txt

Web version:  python app.py  ->  open http://localhost:5001 in your browser
              (run from inside the Performative_Assessment_V5/ folder)
Terminal version: python cli.py

Run the test suite (requires pytest):

    python3 -m pytest tests/

Reports are saved in the reports/<username>/ folder (one sub-folder per user).


DEFAULT LOGIN CREDENTIALS
-------------------------
  Username: admin   Password: admin123     (role: admin / instructor)

  Username: emma    Password: Learn@2024   (role: student)
  Username: liam    Password: Learn@2024   (role: student)
  Username: sofia   Password: Learn@2024   (role: student)
  Username: james   Password: Learn@2024   (role: student)
  Username: priya   Password: Learn@2024   (role: student)
  Username: tyler   Password: Learn@2024   (role: student)

  Accounts are seeded automatically into the SQLite database on first run.
  Passwords are stored as secure hashes (never in plain text).

  Admins can change any account's name, username, role, or password from the
  Admin Dashboard: User Directory -> expand a user -> "Configure account".


ROLES
-----
  admin   -- logs in to the Admin Dashboard (/admin).
             Can view every student's reports, annotate the LLM's grading,
             monitor grading reliability, manage accounts, and download the
             research CSV export. Can also use the Assessment App.

  student -- logs in directly to the Assessment App (/).
             Can generate and view only their own reports.
             Cannot access the admin dashboard or other students' files.


ASSESSMENT MODES
----------------
  Scenario-Based -- multi-turn roleplay against an examiner: free recall first,
                    then structured probing, scored against expert answers.

  Free Response  -- a written submission scored against a prompt's key points
                    (construct/exemplar evidence model, with pooled key points).
                    While the learner writes, the browser captures a writing-
                    process trace (pauses, revisions, pastes, text snapshots).
                    The report then includes:
                      - a process overlay (effort, authenticity, trajectory,
                        rate->explain->re-rate confidence calibration)
                      - a Writing Process Replay: play back how the response
                        was written, with a revision heatmap
                    The raw trace is stored beside the report as a
                    .trace.json sidecar (delta-encoded).


INSTRUCTOR TOOLING (Admin Dashboard)
------------------------------------
  - Report annotation: on any report view, record whether the LLM's grading
    was correct / partial / missing / needs expert review, plus notes.
  - Grading Reliability panel: LLM-vs-instructor agreement rate, average LLM
    score per annotation label (over/under-crediting signal), per-task
    agreement, recent annotations, and novel-equivalent match reliability.
  - Process Review queue: surfaces free-response reports with high- or medium-
    priority product/process divergence signals for human review. The queue is
    advisory context only; it never changes the learner's score.
  - Novel-equivalent review queue: promote or dismiss novel FR matches;
    promotion into a key point's exemplar list is always a human action.
  - Research export: /admin/research-export.csv -- one row per assessed task,
    served from the assessments database table. Column definitions live in
    docs/research_export_data_dictionary.md.


THEMES
------
  Five themes are available and persist per user account across all pages:

    Ultra-Light  -- high-contrast bright white
    Light        -- default clean blue/white (default)
    Rustic       -- warm tan/brown tones
    Dark         -- dark grey, easy on eyes
    Ultra-Dark   -- black terminal with green accents

  The theme selector appears in the header on every page.
  Logged-in users have their theme saved to the database automatically.
  The login/locked pages remember theme via the browser (localStorage).


SECURITY FEATURES
-----------------
  - Login inputs are sanitised: null bytes stripped, whitespace trimmed, length capped.
  - CSRF token required on every login form submission and admin form.
  - Rate limiting: 10 failed login attempts per 60-second window per IP address.
  - Admin lockout: 3 failed attempts against an admin account triggers a 15-minute
    hard lockout. The browser is immediately redirected to a locked-out page.
  - Assessment sessions are bound to the user who created them; in-memory
    session state is TTL-bounded and capped.
  - Request bodies are size-limited; free-response submissions are length-capped.
  - Reports are stored in separate per-user sub-folders (reports/<username>/).
  - Session cookies are HttpOnly and SameSite=Strict.
    (Set SESSION_COOKIE_SECURE = True in app.py when serving over HTTPS.)
  - All HTML pages are served with Cache-Control: no-store so pressing the browser
    back button after logout cannot reveal cached pages -- the session is always
    re-checked on navigation.
  - Student report endpoints validate path ownership server-side; a student cannot
    access another student's report even by guessing the URL.
  - Prompt-injection defence on the grading call; LLM API errors surface as
    clear messages (rate limit / bad key / network) instead of generic failures.


FILE STRUCTURE
--------------
  config.py          -- provider endpoints, model names, API keys, tunables
  engine.py          -- re-exports the assessment logic below
  loaders.py         -- scenario/prompt JSON loading and validation
  runner.py          -- scenario conversation flow (recall -> probing)
  scoring.py         -- keyword + LLM scoring (construct/exemplar evidence model)
  writing_process.py -- writing-process (trace) analysis for FR submissions
  thinking.py        -- Honey & Mumford / SOLO thinking-profile analysis
  reports.py         -- report generation (scenario + free-response)
  report_parser.py   -- parses generated report markdown back into data
  llm.py             -- LLM provider dispatch, retries, JSON repair, eval cache
  auth.py            -- authentication, rate limiting, input sanitisation
  database.py        -- SQLite layer: users, assessments results table,
                        LLM eval cache, novel-equivalent review, match log
  app.py             -- web server, routes, session handling
  cli.py             -- terminal interface (no login required)
  assessments.db     -- SQLite database (auto-created on first run, do not commit)
  .secret_key        -- Flask session secret (auto-generated, do not delete)
  requirements.txt   -- Python dependencies

  static/
    themes.css        -- shared CSS variable overrides for all non-default themes
    index.js          -- main assessment-app logic (loaded by index.html)
    index-admin.js    -- admin-only debug tooling (auto-run, FR auto-fill)

  templates/
    index.html          -- main assessment app (students and admins)
    login.html          -- sign-in page
    locked.html         -- shown after 3 failed admin login attempts
    admin.html          -- admin dashboard (users, reports, grading reliability)
    admin_user_edit.html-- admin "Configure account" page
    report_view.html    -- admin report viewer (with annotation box)
    student_report.html -- student report viewer (students see only their own)
    my_reports.html     -- student's own-report list
    _process_replay.html-- shared writing-process replay widget

  tests/               -- pytest suite (scoring, calibration, prompt inventory)
  docs/                -- research export data dictionary, testing guide

  reports/<username>/
    report_YYYYMMDD_HHMMSS.md           -- scenario assessment reports
    fr_report_YYYYMMDD_HHMMSS.md        -- free-response assessment reports
    fr_report_YYYYMMDD_HHMMSS.trace.json-- raw writing-process trace (replay)
  reports/_annotations/                 -- instructor annotation sidecars

  The assessments database table mirrors every generated report (one row per
  assessed task) and is backfilled automatically from existing report files
  at startup -- it powers the research export and the reliability dashboard.


STUDENT REPORT ACCESS
---------------------
  After completing an assessment and clicking "Generate Report", the report is
  saved to reports/<username>/. Students can view their own past reports from the
  "My Reports" card on the home screen. Each report opens in a themed viewer page.
  Students cannot access reports belonging to other users.

  Admins can view all student reports from the Admin Dashboard (/admin).


LLM PROVIDERS
-------------
  The app supports multiple AI providers for scoring and report generation.
  Put keys in config.py, or paste a key into the in-app "API Key" field at
  runtime (keys typed in the UI are never written to disk). Supported:

    OpenAI        -- api.openai.com
    Claude        -- api.anthropic.com (requires the anthropic package)
    Gemini        -- Google Generative Language API via its OpenAI-compatible
                     endpoint (default model gemini-2.5-flash)
    Groq          -- api.groq.com
    Mistral       -- api.mistral.ai
    GitHub Models -- models.github.ai; free prototyping tier. Auth with a
                     GitHub personal access token that has models:read.
                     Model ids are publisher-prefixed (e.g. openai/gpt-4o-mini).
                     Output tokens are capped automatically for its free tier.
    Ollama        -- runs locally, no API key needed (ollama serve must be running)

  Each user's preferred provider and model are saved to their account in the
  database and restored automatically on next login.

  Without a valid API key the app falls back to keyword-matching scoring.
  Reports are still generated -- the LLM summary section will note that no
  API key was configured.


FILE REFERENCE -- what to edit for common changes
-------------------------------------------------
  config.py          -- provider endpoints/models/keys, self-consistency tunables
  database.py        -- user accounts, seed data, schema
  auth.py            -- rate-limit thresholds, lockout duration, sanitisation rules
  scoring.py         -- scoring logic and grading prompts
  writing_process.py -- process-signal thresholds (pauses, revisions, snapshots)
  reports.py         -- report layout and instructor summary
  llm.py             -- provider dispatch, retry/backoff, token caps
  app.py             -- web routes, session handling, report storage path
  static/index.js    -- frontend behaviour (views, WritingTracker, API calls)


SCENARIOS
---------
  Scenarios live in the scenarios/ folder as .json files.
  Add a new scenario by copying an existing .json file and editing the fields.
  Key fields:
    "title"          -- name shown in the menu
    "situation"      -- the scene text shown to the learner
    "max_turns"      -- how many responses the learner gets (default 8)
    "constraints"    -- rules shown in the report (do not affect scoring directly)
    "expert_answers" -- list of ideal answers; each has:
        "answer"     -- full ideal response text
        "key_points" -- short phrases the learner should mention
        "rubric"     -- point value for each key point (optional; equal weighting if omitted)

  Free-response prompts live in the prompts/ folder; the tests/ folder includes
  a prompt-inventory validation that checks their structure.


REQUIREMENTS
------------
  pip install -r requirements.txt        # Flask + Werkzeug (required)

  Optional provider SDKs:
    pip install openai      # official SDK path for OpenAI-compatible providers
    pip install anthropic   # required only for the Claude provider

  All OpenAI-compatible providers (OpenAI, Groq, Mistral, Gemini, GitHub
  Models, Ollama) also work without any SDK over the built-in HTTP client.
  Without any LLM key the app runs in keyword-matching mode automatically.

  For the test suite:  pip install pytest
