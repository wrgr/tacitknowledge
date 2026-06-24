HOW TO RUN
----------
Web version:  python app.py  ->  open http://localhost:5001 in your browser
              (run from inside the Performative_Assessment_V5/ folder)
Terminal version: python cli.py

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

  Accounts are seeded automatically from the database on first run.
  Passwords are stored as secure hashes in the SQLite database (never in plain text).

  To change a password, run this one-liner from inside Performative_Assessment_V5/:

      python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('yournewpassword'))"

  Then update the password_hash column in the assessments.db database using any
  SQLite browser, or by editing database.py and re-running the app.


ROLES
-----
  admin   -- logs in to the Admin Dashboard (/admin).
             Can view every student's reports and report files.
             Can also use the Assessment App.

  student -- logs in directly to the Assessment App (/).
             Can generate and view only their own reports.
             Cannot access the admin dashboard or other students' files.


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
  - CSRF token required on every login form submission.
  - Rate limiting: 10 failed login attempts per 60-second window per IP address.
  - Admin lockout: 3 failed attempts against an admin account triggers a 15-minute
    hard lockout. The browser is immediately redirected to a locked-out page.
  - Assessment sessions are bound to the user who created them.
  - Reports are stored in separate per-user sub-folders (reports/<username>/).
  - Session cookies are HttpOnly and SameSite=Strict.
  - All HTML pages are served with Cache-Control: no-store so pressing the browser
    back button after logout cannot reveal cached pages -- the session is always
    re-checked on navigation.
  - Student report endpoints validate path ownership server-side; a student cannot
    access another student's report even by guessing the URL.


FILE STRUCTURE
--------------
  config.py      -- settings (model name, API keys, reports folder)
  engine.py      -- all assessment logic (loading, scoring, conversation, reports)
  auth.py        -- authentication, rate limiting, input sanitisation
  database.py    -- SQLite layer; replaces the old users.json file
  app.py         -- web server, routes, login/logout/admin/student endpoints
  reports.py     -- report generation (scenario + free-response)
  scoring.py     -- keyword scoring and LLM-based scoring
  llm.py         -- LLM provider abstraction (OpenAI, Claude, Gemini, Groq, Mistral, Ollama)
  cli.py         -- terminal interface (no login required)
  assessments.db -- SQLite database (auto-created on first run, do not commit)
  .secret_key    -- Flask session secret (auto-generated on first run, do not delete)

  static/
    themes.css       -- shared CSS variable overrides for all non-default themes

  templates/
    index.html         -- main assessment app (students and admins)
    login.html         -- sign-in page
    locked.html        -- shown after 3 failed admin login attempts
    admin.html         -- admin dashboard (all users + their reports)
    report_view.html   -- admin report viewer (admin-only)
    student_report.html-- student report viewer (students see only their own)

  reports/<username>/
    report_YYYYMMDD_HHMMSS.md     -- scenario assessment reports
    fr_report_YYYYMMDD_HHMMSS.md  -- free-response assessment reports


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
  Configure API keys in config.py. Supported providers:

    OpenAI   -- requires OPENAI_API_KEY
    Claude   -- requires ANTHROPIC_API_KEY
    Gemini   -- requires GEMINI_API_KEY
    Groq     -- requires GROQ_API_KEY
    Mistral  -- requires MISTRAL_API_KEY
    Ollama   -- runs locally, no API key needed (ollama serve must be running)

  Each user's preferred provider and model are saved to their account in the
  database and restored automatically on next login.

  Without a valid API key the app falls back to keyword-matching scoring.
  Reports are still generated -- the LLM summary section will note that no
  API key was configured.


FILE REFERENCE -- what to edit for common changes
-------------------------------------------------
  config.py    -- AI model name, API keys, reports folder path
  database.py  -- user accounts, default seed data, schema
  auth.py      -- rate-limit thresholds, lockout duration, sanitisation rules
  engine.py    -- scoring logic, examiner prompts, report layout
  app.py       -- web routes, session handling, report storage path


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


REQUIREMENTS
------------
  pip install flask werkzeug

  For LLM scoring install the relevant SDK(s):
    pip install openai               # OpenAI + Groq + Mistral + Ollama
    pip install anthropic            # Claude
    pip install google-generativeai  # Gemini

  Without any LLM the app runs in keyword-matching mode automatically.
