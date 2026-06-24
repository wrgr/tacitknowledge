HOW TO RUN
----------
Terminal version:  python3 cli.py
Web version:       python3 app.py  →  open http://localhost:5001 in your browser
Reports are saved in the reports/<username>/ folder (one sub-folder per user).


DEFAULT LOGIN CREDENTIALS
-------------------------
  Username: admin     Password: admin123   (role: admin)
  Username: student   Password: student123 (role: student)

  IMPORTANT — change these before sharing the app with others.
  Passwords are stored as secure hashes in users.json (never in plain text).
  To change a password, edit users.json and replace the password_hash value with
  the output of this one-liner run in the same folder:

      python3 -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('yournewpassword'))"

  To add a new student account, copy an existing entry in users.json and set
  "role" to "student".  To add another admin, set "role" to "admin".


ROLES
-----
  admin   — logs in to the Admin Dashboard (/admin).
            Can view every student's reports. Can also use the Assessment App.

  student — logs in directly to the Assessment App (/).
            Can only see and generate their own reports. Cannot see other
            students' reports or access the admin dashboard.


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


FILE STRUCTURE
--------------
  config.py    — settings (model name, API keys, reports folder)
  engine.py    — all assessment logic (loading, scoring, conversation, reports)
  auth.py      — authentication, rate limiting, input sanitisation
  app.py       — web server, routes, login/logout/admin endpoints
  cli.py       — terminal interface (no login required)
  users.json   — user accounts with hashed passwords (auto-created on first run)
  .secret_key  — Flask session secret (auto-generated on first run, do not delete)

  templates/
    index.html      — main assessment app (students and admins)
    login.html      — sign-in page
    locked.html     — shown after 3 failed admin login attempts
    admin.html      — admin dashboard (all users + reports)
    report_view.html— admin report viewer

  reports/<username>/
    report_YYYYMMDD_HHMMSS.md  — generated reports, one sub-folder per user


FILE REFERENCE — what to edit for common changes
-------------------------------------------------
| File       | What to change                                                    |
|------------|-------------------------------------------------------------------|
| config.py  | AI model name, API keys, reports folder path                      |
| users.json | User accounts and hashed passwords                                |
| auth.py    | Rate-limit thresholds, lockout duration, sanitisation rules       |
| engine.py  | Scoring logic, examiner prompts, report layout                    |
| app.py     | Web routes, session handling, report storage path                 |
| cli.py     | Terminal menu flow and printed output                             |


COMMENTS
--------
Every line in every Python file has a comment explaining what it does.
Look for the  #  character — everything after it on the same line is a comment
and has no effect on the program. You can read, edit, or delete comments freely.


SCENARIOS
---------
Scenarios live in the scenarios/ folder as .json files.
Add a new scenario by copying an existing .json file and editing the fields.
Key fields:
  "title"          — name shown in the menu
  "situation"      — the scene text shown to the learner
  "max_turns"      — how many responses the learner gets (default 8)
  "constraints"    — rules shown in the report (do not affect scoring directly)
  "expert_answers" — list of ideal answers; each has:
      "answer"     — full ideal response text
      "key_points" — short phrases the learner should mention
      "rubric"     — point value for each key point (optional; equal weighting if omitted)


REQUIREMENTS
------------
  pip install flask werkzeug
  Ollama must be running locally (ollama serve) for LLM scoring and report generation.
  Without Ollama the program falls back to simple keyword matching.
  werkzeug is installed automatically as a Flask dependency.
