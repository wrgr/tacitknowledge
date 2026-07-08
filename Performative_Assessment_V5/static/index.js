// index.js — main assessment app logic (extracted from templates/index.html).
// Server-derived flags (IS_ADMIN) are set by an inline bootstrap before this loads.
// @ts-nocheck
// ── State ──────────────────────────────────────
const S  = { sessionId: null, phase: 'recall', probeCount: 0, probeNumber: 0, turn: 0, busy: false, provider: null, model: null, apiKey: '', providerIsConfigured: false };
const FR = {
  sessionId: null, promptData: null, preRating: null, postRating: null,
  // Closing nudge (fr_recall_omission_fix brief, Part B): capped at two displays total --
  // the second only happens if "Add more" was chosen the first time. closingNudgeUsed is
  // true only if the learner actually chose to add content, not merely because the nudge
  // was shown -- it's report-facing metadata about whether the pass was extended.
  closingNudgeShown: 0, closingNudgeUsed: false,
};

// ── Writing Tracker ────────────────────────────
// Captures per-turn behavioural metrics: typing speed, deletions, pauses, paste events.
// Call reset() when a new turn begins, collect(text) just before submitting.
//
// Content-aware capture (FR only): revision events, periodic full-text snapshots, and
// located pauses/pastes are additionally recorded into `process_log`, appended to the
// existing aggregate metrics payload. This capture is enabled for testing — review
// data-handling and disclosure requirements (see fields marked SENSITIVE_TESTING_ONLY)
// before any non-testing deployment.
//
// TUNABLE -- revision sensitivity, adjust after testing
const REVISION_MIN_CHARS = 8;
const REVISION_MIN_LOOKBACK = 15;

// TUNABLE -- snapshot cadence. Fine cadence keeps the writing-process replay
// accurate (each snapshot is a real keyframe of the text over time); the replay
// interpolates between them, so finer = closer to actual typing.
const SNAPSHOT_INTERVAL_S = 1.5;
const SNAPSHOT_INTERVAL_CHARS = 40;

// TUNABLE -- what counts as a significant pause
const PAUSE_THRESHOLD_S = 4;

// TUNABLE -- payload caps
const MAX_REVISION_EVENTS = 40;
const MAX_SNAPSHOTS = 500;
const MAX_PAUSE_EVENTS = 30;
const MAX_PASTE_EVENTS = 20;
const MAX_REMOVED_TEXT_CHARS = 300;   // truncate long removed/inserted text

// Keep the `max` most informative entries (by scoreFn) but preserve chronological order.
function _capByScore(arr, max, scoreFn) {
  if (arr.length <= max) return arr;
  const indexed = arr.map((item, i) => ({ item, i, score: scoreFn(item) }));
  indexed.sort((a, b) => b.score - a.score);
  const kept = indexed.slice(0, max);
  kept.sort((a, b) => a.i - b.i);
  return kept.map(k => k.item);
}

// Keep first, last, and an evenly-spaced sample of the middle.
function _capSnapshots(snaps, max) {
  if (snaps.length <= max) return snaps;
  const first = snaps[0], last = snaps[snaps.length - 1];
  const middle = snaps.slice(1, -1);
  const slots = max - 2;
  if (slots <= 0) return [first, last];
  const step = middle.length / slots;
  const sampled = [];
  for (let i = 0; i < slots; i++) {
    sampled.push(middle[Math.min(middle.length - 1, Math.floor(i * step))]);
  }
  return [first, ...sampled, last];
}

const WritingTracker = {
  _events:       [],   // [{type:'key'|'delete'|'paste', t:<ms>}, ...]
  _startTime:    null, // Date.now() when this turn began (set by reset)
  _firstKeyTime: null, // Date.now() of the first keystroke this turn

  // content-aware process log (FR only — populated via onInput/onPaste)
  _revisionEvents: [],
  _snapshots:      [],
  _pauseEvents:    [],
  _pasteEvents:    [],
  _pendingPaste:   null,
  _lastText:       '',
  _lastPos:        0,
  _lastEventTime:  null,
  _charsSinceSnapshot: 0,
  _lastSnapshotTime:   0,
  _captureActive:  false,  // true once onInput() has run this turn — gates process_log (FR only)

  reset() {
    this._events       = [];
    this._startTime    = Date.now();
    this._firstKeyTime = null;

    this._revisionEvents = [];
    this._snapshots      = [];
    this._pauseEvents    = [];
    this._pasteEvents    = [];
    this._pendingPaste   = null;
    this._lastText       = '';
    this._lastPos        = 0;
    this._lastEventTime  = null;
    this._charsSinceSnapshot = 0;
    this._lastSnapshotTime   = 0;
    this._captureActive  = false;
  },

  onKey(e) {
    const now = Date.now();
    if (this._firstKeyTime === null) this._firstKeyTime = now;
    const type = (e.key === 'Backspace' || e.key === 'Delete') ? 'delete' : 'key';
    this._events.push({ type, t: now });
  },

  // e: the ClipboardEvent; caretPos: selectionStart of the field *before* the paste lands
  onPaste(e, caretPos) {
    const now = Date.now();
    if (this._firstKeyTime === null) this._firstKeyTime = now;
    this._events.push({ type: 'paste', t: now });
    let length = 0;
    try { length = (e.clipboardData || window.clipboardData).getData('text').length; } catch (_) {}
    this._pendingPaste = { t: now, pos: caretPos, length };
  },

  // Called on every 'input' event of the FR textarea with the current value + caret index.
  // Drives located-pause detection, revision-event diffing, paste-span logging, and snapshots.
  onInput(text, caretPos) {
    const now  = Date.now();
    if (this._firstKeyTime === null) this._firstKeyTime = now;
    const nowS = +((now - this._startTime) / 1000).toFixed(1);
    this._captureActive = true;

    this._recordPause(now);

    if (this._pendingPaste) {
      const p = this._pendingPaste;
      this._pasteEvents.push({
        timestamp_s:   +((p.t - this._startTime) / 1000).toFixed(1),
        char_position: p.pos,
        paste_length:  p.length,
      });
      this._pendingPaste = null;
    } else {
      this._recordRevision(this._lastText, text, nowS);
    }

    this._maybeSnapshot(text, now, nowS);

    this._lastText      = text;
    this._lastPos       = caretPos;
    this._lastEventTime = now;
  },

  _recordPause(now) {
    if (this._lastEventTime === null) return;
    const gapMs = now - this._lastEventTime;
    if (gapMs <= PAUSE_THRESHOLD_S * 1000) return;
    const pos = this._lastPos;
    this._pauseEvents.push({
      timestamp_s:        +((this._lastEventTime - this._startTime) / 1000).toFixed(1),
      duration_s:         +(gapMs / 1000).toFixed(1),
      char_position:      pos,
      preceding_context:  this._lastText.slice(Math.max(0, pos - 80), pos),
    });
  },

  // An edit to already-committed text — not simple forward typing, and not backspacing
  // only the last few characters of the current word (see REVISION_MIN_* thresholds).
  _recordRevision(oldText, newText, nowS) {
    let prefixLen = 0;
    const maxPrefix = Math.min(oldText.length, newText.length);
    while (prefixLen < maxPrefix && oldText[prefixLen] === newText[prefixLen]) prefixLen++;

    let suffixLen = 0;
    const maxSuffix = Math.min(oldText.length, newText.length) - prefixLen;
    while (suffixLen < maxSuffix &&
           oldText[oldText.length - 1 - suffixLen] === newText[newText.length - 1 - suffixLen]) suffixLen++;

    const removed  = oldText.slice(prefixLen, oldText.length - suffixLen);
    const inserted = newText.slice(prefixLen, newText.length - suffixLen);
    if (!removed && !inserted) return;

    const lookback = oldText.length - prefixLen; // how far back from the previous end this edit reaches
    if (removed.length < REVISION_MIN_CHARS && lookback < REVISION_MIN_LOOKBACK) return;

    this._revisionEvents.push({
      timestamp_s:     nowS,
      char_position:   prefixLen,
      removed_text:    removed.slice(0, MAX_REMOVED_TEXT_CHARS),   // SENSITIVE_TESTING_ONLY
      inserted_text:   inserted.slice(0, MAX_REMOVED_TEXT_CHARS),
      context_before:  oldText.slice(Math.max(0, prefixLen - 80), prefixLen),
    });
  },

  _maybeSnapshot(text, now, nowS) {
    const charsDelta = text.length - this._lastText.length;
    if (charsDelta > 0) this._charsSinceSnapshot += charsDelta;
    const elapsed = now - this._lastSnapshotTime;
    if (this._snapshots.length === 0 ||
        elapsed >= SNAPSHOT_INTERVAL_S * 1000 ||
        this._charsSinceSnapshot >= SNAPSHOT_INTERVAL_CHARS) {
      this._snapshots.push({ timestamp_s: nowS, text });   // SENSITIVE_TESTING_ONLY
      this._lastSnapshotTime   = now;
      this._charsSinceSnapshot = 0;
    }
  },

  collect(finalText) {
    const now    = Date.now();
    const words  = finalText.trim() ? finalText.trim().split(/\s+/).length : 0;
    const keys   = this._events.filter(e => e.type === 'key').length;
    const dels   = this._events.filter(e => e.type === 'delete').length;
    const pastes = this._events.filter(e => e.type === 'paste').length;

    // Active typing duration: sum of inter-keystroke gaps <= 3 s; longer gaps are pauses
    const PAUSE_MS = 3000;
    let activeDuration = 0, pauseCount = 0, maxPauseMs = 0;
    for (let i = 1; i < this._events.length; i++) {
      const gap = this._events[i].t - this._events[i - 1].t;
      if (gap > PAUSE_MS) { pauseCount++; if (gap > maxPauseMs) maxPauseMs = gap; }
      else                { activeDuration += gap; }
    }

    const activeMin  = activeDuration / 60000;
    const wpm        = (activeMin > 0 && words > 0) ? Math.round(words / activeMin) : null;
    const totalKeys  = keys + dels;
    const latencyS   = this._firstKeyTime !== null
      ? +((this._firstKeyTime - this._startTime) / 1000).toFixed(1) : null;
    const totalTimeS = this._firstKeyTime !== null
      ? +((now - this._firstKeyTime) / 1000).toFixed(1) : null;

    const result = {
      wpm,
      latency_s:      latencyS,
      deletion_count: dels,
      revision_ratio: totalKeys > 0 ? +(dels / totalKeys).toFixed(3) : 0,
      paste_count:    pastes,
      pause_count:    pauseCount,
      max_pause_s:    maxPauseMs > 0 ? +(maxPauseMs / 1000).toFixed(1) : 0,
      total_time_s:   totalTimeS,
    };

    // process_log is FR-only content-aware capture — only populated when onInput() was
    // wired up (the FR textarea does this; the scenario textarea does not), so scenario
    // mode's payload is byte-for-byte unchanged.
    if (this._captureActive) {
      // Final snapshot near submission time, so trajectory reconstruction has an endpoint.
      const finalNowS = +((now - this._startTime) / 1000).toFixed(1);
      if (!this._snapshots.length || this._snapshots[this._snapshots.length - 1].text !== finalText) {
        this._snapshots.push({ timestamp_s: finalNowS, text: finalText });   // SENSITIVE_TESTING_ONLY
      }

      result.process_log = {
        revision_events: _capByScore(this._revisionEvents, MAX_REVISION_EVENTS,
                                      r => Math.max(r.removed_text.length, r.inserted_text.length)),
        snapshots:       _capSnapshots(this._snapshots, MAX_SNAPSHOTS),
        pause_events:    _capByScore(this._pauseEvents, MAX_PAUSE_EVENTS, p => p.duration_s),
        paste_events:    _capByScore(this._pasteEvents, MAX_PASTE_EVENTS, p => p.paste_length),
      };
    }

    return result;
  },
};

// ── Helpers ────────────────────────────────────
const $  = id => document.getElementById(id);
const show = id => $(id).classList.remove('hidden');
const hide = id => $(id).classList.add('hidden');

function llmAvailable() {
  return !!(S.apiKey || S.providerIsConfigured);
}

function updateFillAiButtons() {
  const available = llmAvailable();
  const tip = available ? '' : 'Add an API key to enable AI generation';
  ['btn-fill-ai', 'btn-fr-fill-ai'].forEach(id => {
    const btn = $(id);
    if (!btn) return;
    btn.disabled = !available;
    btn.title    = tip;
  });
}

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let d;
  try {
    d = await r.json();
  } catch (_) {
    if (r.status === 401 || r.status === 302) throw new Error('Session expired — please log in again.');
    throw new Error(`Server error (HTTP ${r.status})`);
  }
  if (d.error) throw new Error(d.error);
  return d;
}

// ── Mode navigation ────────────────────────────
function showHome() {
  ['view-dashboard','view-scenarios','view-assess','view-results','view-create',
   'view-fr-prompts','view-fr-write','view-fr-results','view-fr-create','view-fr-review','view-fr-reliability','view-reports',
   'view-profile'].forEach(hide);
  show('view-home');
  syncHomeSelectors();
  // show "← Dashboard" back link only for students
  const bd = $('back-to-dash');
  if (bd) bd.style.display = IS_ADMIN ? 'none' : 'inline';
}

async function showDashboard() {
  ['view-home','view-scenarios','view-assess','view-results','view-create',
   'view-fr-prompts','view-fr-write','view-fr-results','view-fr-create','view-fr-review','view-fr-reliability','view-reports',
   'view-profile'].forEach(hide);
  show('view-dashboard');
  await _loadDashboardStats();
}

async function _loadDashboardStats() {
  try {
    const data = await api('/api/learning-profile', {});
    const name = (data.display_name || data.username || '').split(' ')[0];
    $('dash-greeting').textContent = 'Welcome back, ' + name + '!';

    const reps = data.reports || [];
    if (!reps.length) {
      ['dash-stat-total','dash-stat-avg','dash-stat-best','dash-stat-trend'].forEach(id => {
        $(id).textContent = '0';
      });
      $('dash-no-data').classList.remove('hidden');
      return;
    }

    const scores = reps.map(r => r.score);
    const avg    = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const best   = Math.max(...scores);
    let trendStr = '—';
    let trendColor = '';
    if (scores.length >= 2) {
      const mid   = Math.floor(scores.length / 2);
      const early = scores.slice(0, mid).reduce((a, b) => a + b, 0) / mid;
      const late  = scores.slice(mid).reduce((a, b) => a + b, 0) / (scores.length - mid);
      const delta = Math.round(late - early);
      if (delta > 0)      { trendStr = '+' + delta + '%'; trendColor = 'var(--green)'; }
      else if (delta < 0) { trendStr = delta + '%';       trendColor = 'var(--red)';   }
      else                { trendStr = '→ 0%'; }
    }

    $('dash-stat-total').textContent = reps.length;
    $('dash-stat-avg').textContent   = avg + '%';
    $('dash-stat-best').textContent  = best + '%';
    $('dash-stat-trend').textContent = trendStr;
    if (trendColor) $('dash-stat-trend').style.color = trendColor;
  } catch (_) {
    // fail silently — stats not critical
  }
}

function showScenarioMode() {
  hide('view-home');
  show('view-scenarios');
}

function showFrMode() {
  hide('view-home');
  show('view-fr-prompts');
  if (!_frPromptsLoaded) loadFrPrompts();
}

$('mode-scenario').addEventListener('click', showScenarioMode);
$('mode-fr').addEventListener('click', showFrMode);

// ── My Reports ─────────────────────────────────
async function showReportsMode() {
  hide('view-home');
  show('view-reports');
  const list = $('reports-list');
  list.innerHTML = '<p style="color:var(--muted);font-size:.88rem">Loading…</p>';
  try {
    const data = await api('/api/my-reports', {});
    if (!data.reports.length) {
      list.innerHTML = '<p style="color:var(--muted);font-size:.88rem">No reports yet. Complete an assessment and click "Generate Report" to save one.</p>';
      return;
    }
    list.innerHTML = '';
    data.reports.forEach(filename => {
      const item = document.createElement('div');
      item.className = 'scenario-card';
      item.style.cssText = 'cursor:default';
      const icon = filename.startsWith('fr_') ? '📝' : '🎭';
      const label = filename.startsWith('fr_') ? 'Free Response' : 'Scenario';
      // parse timestamp from filename e.g. report_20260622_105552
      const m = filename.match(/(\d{8})_(\d{6})/);
      let dateStr = '';
      if (m) {
        const d = m[1], t = m[2];
        dateStr = d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6) + ' ' + t.slice(0,2)+':'+t.slice(2,4);
      }
      item.innerHTML = `
        <div class="icon">${icon}</div>
        <div class="info">
          <h3 style="font-size:.9rem">${filename}</h3>
          <p>${label}${dateStr ? ' &mdash; ' + dateStr : ''}</p>
        </div>
        <a href="/my-report/${encodeURIComponent(filename)}"
           style="font-size:.78rem;padding:5px 12px;border-radius:6px;border:1px solid var(--blue);color:var(--blue);text-decoration:none;white-space:nowrap;transition:background .12s"
           onmouseover="this.style.background='#eff6ff'"
           onmouseout="this.style.background=''"
           target="_blank" rel="noopener">View</a>`;
      list.appendChild(item);
    });
  } catch (err) {
    list.innerHTML = `<p style="color:var(--muted);font-size:.88rem">Failed to load: ${err.message}</p>`;
  }
}

// ── Chat ───────────────────────────────────────
function addBubble(role, text) {
  const area = $('chat-area');
  const wrap = document.createElement('div');
  wrap.className = `bubble-wrap ${role}`;

  const av = document.createElement('div');
  av.className = `avatar ${role}`;
  av.textContent = role === 'examiner' ? 'E' : 'You';

  const b = document.createElement('div');
  b.className = `bubble ${role}`;
  b.textContent = text;

  wrap.appendChild(av);
  wrap.appendChild(b);
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

let _typing = null;
function showTyping() {
  if (_typing) return;
  const area = $('chat-area');
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap examiner';
  const av = document.createElement('div');
  av.className = 'avatar examiner';
  av.textContent = 'E';
  const tb = document.createElement('div');
  tb.className = 'typing-bubble';
  tb.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  wrap.appendChild(av);
  wrap.appendChild(tb);
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
  _typing = wrap;
}
function hideTyping() {
  if (_typing) { _typing.remove(); _typing = null; }
}

function updateMeta() {
  if (S.phase === 'probing') {
    $('assess-meta').textContent = `Probe ${S.probeNumber} of ${S.probeCount}`;
    $('probe-progress').textContent = `(${S.probeNumber} / ${S.probeCount})`;
  } else {
    $('assess-meta').textContent = 'Recall Phase';
  }
}

function lock(on) {
  S.busy = on;
  $('user-input').disabled = on;
  $('btn-send').disabled   = on;
  $('btn-done').disabled   = on;
}

// ── Scenarios ──────────────────────────────────
const ICONS = ['🚗','🏥','🔥','⚡','🧪','🛠️','📋','🌊','🚨','🏗️'];

function updateBadge() {
  const model = $('model-select').value || S.model;
  $('model-badge').textContent = S.provider && model
    ? `${S.provider} — ${model}`
    : (S.provider || 'Keyword mode');
}

function syncHomeSelectors() {
  const hps = $('home-provider-select');
  const hms = $('home-model-select');
  if (!hps || !hms) return;
  const ps = $('provider-select');
  const ms = $('model-select');

  hps.innerHTML = '';
  Array.from(ps.options).forEach(opt => {
    const o = opt.cloneNode(true);
    o.selected = (opt.value === S.provider);
    hps.appendChild(o);
  });

  hms.innerHTML = '';
  Array.from(ms.options).forEach(opt => {
    const o = opt.cloneNode(true);
    o.selected = (opt.value === S.model);
    hms.appendChild(o);
  });

  if (ps.options.length > 0) show('home-provider-row');
}

async function fetchModels(providerName, preferredModel) {
  const sel = $('model-select');
  sel.innerHTML = '<option disabled>Loading…</option>';
  try {
    const data = await api('/api/models', { provider: providerName });
    sel.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      // prefer the user's saved model, otherwise fall back to provider default
      if (m === (preferredModel || data.default)) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!sel.value && data.models.length) sel.value = data.models[0];
    S.model = sel.value;
  } catch (_) {
    sel.innerHTML = '<option value="">—</option>';
    S.model = null;
  }
  updateBadge();
  syncHomeSelectors();
}

function saveModelPref(provider, model) {
  fetch('/api/save-model-pref', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model }),
  }).catch(() => {});
}

// keyed by provider name — populated when /api/scenarios loads
let _providerMeta = {};

function updateKeyField() {
  const meta         = _providerMeta[S.provider] || {};
  const keyInput     = $('api-key-input');
  const homeKeyInput = $('home-api-key-input');

  if (!meta.needs_key) {
    // Ollama — no API key needed
    keyInput.disabled    = true;
    keyInput.value       = '';
    keyInput.placeholder = 'No key required';
    if (homeKeyInput) { homeKeyInput.disabled = true; homeKeyInput.value = ''; homeKeyInput.placeholder = 'No key required'; }
  } else if (meta.is_configured) {
    keyInput.disabled    = false;
    keyInput.placeholder = 'Key on file — enter to override';
    if (homeKeyInput) { homeKeyInput.disabled = false; homeKeyInput.placeholder = 'Key on file — enter to override'; }
  } else {
    keyInput.disabled    = false;
    keyInput.placeholder = 'Enter API key';
    if (homeKeyInput) { homeKeyInput.disabled = false; homeKeyInput.placeholder = 'Enter API key'; }
  }

  S.apiKey               = keyInput.value.trim();
  S.providerIsConfigured = !meta.needs_key || meta.is_configured;

  // show the warning only when neither a stored key nor a typed key is available
  if (S.apiKey || S.providerIsConfigured) {
    hide('ollama-warning');
  } else {
    show('ollama-warning');
  }
  updateFillAiButtons();
}

async function loadScenarios() {
  const data = await api('/api/scenarios', {});

  // build provider dropdown from the full list (configured + unconfigured)
  if (data.providers && data.providers.length > 0) {
    _providerMeta = {};
    const sel = $('provider-select');
    sel.innerHTML = '';
    const preferredProvider = data.user_preferred_provider;
    const preferredModel    = data.user_preferred_model;
    data.providers.forEach(p => {
      _providerMeta[p.name] = p;
      const opt = document.createElement('option');
      opt.value       = p.name;
      opt.textContent = p.name;
      // prefer saved provider, then API default
      if (p.name === (preferredProvider || data.default_provider)) opt.selected = true;
      sel.appendChild(opt);
    });

    S.provider = sel.value;
    await fetchModels(S.provider, preferredProvider === S.provider ? preferredModel : '');
    updateKeyField();

    sel.addEventListener('change', async () => {
      S.provider = sel.value;
      $('api-key-input').value = '';
      const ks = $('key-status');
      ks.textContent = ''; ks.className = '';
      updateKeyField();
      await fetchModels(S.provider);
      saveModelPref(S.provider, $('model-select').value);
    });

    let _keyValidTimer = null;
    function _applyKeyStatus(text, cls) {
      const s = $('key-status'); s.textContent = text; s.className = cls;
      const hs = $('home-key-status'); hs.textContent = text; hs.className = cls;
    }
    $('api-key-input').addEventListener('input', function () {
      S.apiKey = this.value.trim();
      $('home-api-key-input').value = this.value;
      if (S.apiKey || S.providerIsConfigured) hide('ollama-warning');
      else show('ollama-warning');
      updateFillAiButtons();

      clearTimeout(_keyValidTimer);
      if (!S.apiKey) { _applyKeyStatus('', ''); return; }
      _applyKeyStatus('Checking…', 'key-checking');
      _keyValidTimer = setTimeout(async () => {
        try {
          const data = await api('/api/validate-key', {
            provider: S.provider,
            model: $('model-select').value,
            api_key: S.apiKey,
          });
          _applyKeyStatus(data.valid ? '✓ Valid' : ('✗ ' + (data.error || 'Invalid key')),
                          data.valid ? 'key-valid' : 'key-invalid');
        } catch (_) {
          _applyKeyStatus('✗ Check failed', 'key-invalid');
        }
      }, 800);
    });

    $('model-select').addEventListener('change', () => {
      S.model = $('model-select').value;
      updateBadge();
      saveModelPref(S.provider, S.model);
    });

    show('provider-row');

    $('home-provider-select').addEventListener('change', async function() {
      S.provider = this.value;
      $('provider-select').value = this.value;
      $('api-key-input').value = '';
      $('home-api-key-input').value = '';
      _applyKeyStatus('', '');
      updateKeyField();
      await fetchModels(S.provider);
      saveModelPref(S.provider, $('model-select').value);
    });

    $('home-model-select').addEventListener('change', function() {
      S.model = this.value;
      $('model-select').value = this.value;
      updateBadge();
      saveModelPref(S.provider, S.model);
    });

    $('home-api-key-input').addEventListener('input', function () {
      S.apiKey = this.value.trim();
      $('api-key-input').value = this.value;
      if (S.apiKey || S.providerIsConfigured) hide('ollama-warning');
      else show('ollama-warning');
      updateFillAiButtons();

      clearTimeout(_keyValidTimer);
      if (!S.apiKey) { _applyKeyStatus('', ''); return; }
      _applyKeyStatus('Checking…', 'key-checking');
      _keyValidTimer = setTimeout(async () => {
        try {
          const data = await api('/api/validate-key', {
            provider: S.provider,
            model: $('model-select').value,
            api_key: S.apiKey,
          });
          _applyKeyStatus(data.valid ? '✓ Valid' : ('✗ ' + (data.error || 'Invalid key')),
                          data.valid ? 'key-valid' : 'key-invalid');
        } catch (_) {
          _applyKeyStatus('✗ Check failed', 'key-invalid');
        }
      }, 800);
    });
  } else {
    show('ollama-warning');
    $('model-badge').textContent = 'Keyword mode';
  }

  const list = $('scenario-list');
  data.scenarios.forEach((s) => {
    const card = document.createElement('div');
    card.className = 'scenario-card';
    card.innerHTML = `
      <div class="icon">${ICONS[s.index % ICONS.length]}</div>
      <div class="info">
        <h3>${s.title}</h3>
        <p>${s.description}</p>
        ${s.has_probe_bank ? '<div class="meta" style="color:#10b981">&#10003; Probe bank</div>' : ''}
      </div>`;
    card.addEventListener('click', () => startScenario(s.index, s.title));
    list.appendChild(card);
  });
}

// ── Start assessment ───────────────────────────
async function startScenario(index, title) {
  $('chat-area').innerHTML = '';
  $('assess-title').textContent = title;
  S.turn = 0;
  S.phase = 'recall';
  S.probeCount = 0;
  S.probeNumber = 0;

  const data = await api('/api/start', { index, provider: S.provider, model: S.model, api_key: S.apiKey || undefined });
  S.sessionId = data.session_id;
  S.phase = data.phase || 'recall';
  S.debugExpertAnswer = data.debug_expert_answer || '';
  updateMeta();

  hide('view-scenarios');
  show('view-assess');
  hide('probing-banner');
  $('btn-done').style.display = '';
  if (IS_ADMIN) {
    const wrapper = $('autorun-wrapper');
    wrapper.style.display = S.debugExpertAnswer ? 'inline-block' : 'none';

    // TESTING PURPOSES ONLY: populate the admin "TEST CASES" dropdown with this
    // scenario's expert key points so testers have a quick scoring reference.
    // It's a plain display list (not a <select>) so clicking an item never
    // changes what the dropdown button shows.
    const testCasesWrapper = $('test-cases-wrapper');
    const testCasesList    = $('test-cases-list');
    if (testCasesWrapper && testCasesList) {
      testCasesList.innerHTML = '';
      const keyPoints = data.debug_key_points || [];
      keyPoints.forEach(kp => {
        const li = document.createElement('li');
        li.style.cssText = 'padding:4px 2px;border-bottom:1px solid rgba(245,158,11,.15)';
        li.textContent = kp.weight != null ? `${kp.point} (${kp.weight})` : kp.point;
        testCasesList.appendChild(li);
      });
      testCasesWrapper.style.display = keyPoints.length ? 'inline-block' : 'none';
      $('test-cases-menu').style.display = 'none';
    }
  }
  addBubble('examiner', data.opening);
  WritingTracker.reset();
  $('user-input').focus();
}

// ── Send a response ────────────────────────────
async function send() {
  const el   = $('user-input');
  const text = el.value.trim();
  if (!text || S.busy) return;

  const writingMetrics = WritingTracker.collect(text);

  addBubble('user', text);
  el.value = '';
  el.style.height = 'auto';
  lock(true);
  showTyping();
  WritingTracker.reset();

  try {
    const data = await api('/api/respond', { session_id: S.sessionId, user_input: text, writing_metrics: writingMetrics });
    hideTyping();
    S.turn++;
    if (data.phase) {
      S.phase = data.phase;
      if (data.probe_count)  S.probeCount  = data.probe_count;
      if (data.probe_number) S.probeNumber = data.probe_number;
      if (S.phase === 'probing') {
        show('probing-banner');
        hide('btn-done');
      }
    }
    updateMeta();
    if (data.narration) addBubble('examiner', data.narration);
    if (data.concluded) {
      await evaluate();
    } else {
      lock(false);
      $('user-input').focus();
    }
  } catch (err) {
    hideTyping();
    addBubble('examiner', `⚠ ${err.message}`);
    lock(false);
  }
}

// ── Evaluate ───────────────────────────────────
async function evaluate() {
  addBubble('examiner', 'Assessment complete — evaluating your responses…');
  showTyping();
  try {
    const data = await api('/api/evaluate', { session_id: S.sessionId });
    hideTyping();
    hide('view-assess');
    renderResults(data);
    show('view-results');
    // kick off profile analysis in the background — shows when ready
    fetchThinkingProfile();
  } catch (err) {
    hideTyping();
    addBubble('examiner', `⚠ Evaluation error: ${err.message}`);
    lock(false);
  }
}

// ── Thinking Profile ───────────────────────────
async function fetchThinkingProfile() {
  // show a subtle loading placeholder while the async call runs
  $('profile-content').innerHTML = '<p class="profile-loading"><span class="spinner"></span>Analysing thinking style…</p>';
  show('profile-section');
  try {
    const data = await api('/api/thinking-profile', { session_id: S.sessionId });
    if (data.profile) {
      renderProfile(data.profile);
    } else {
      hide('profile-section');
    }
  } catch (_) {
    hide('profile-section');  // silently hide if unavailable (no LLM key, etc.)
  }
}

function _buildProfileHTML(p) {
  // evidence may be a list (new schema) or a plain string (legacy)
  function evidenceHTML(ev) {
    if (!ev || (Array.isArray(ev) && !ev.length)) return '';
    const items = Array.isArray(ev) ? ev : [ev];
    return '<ul class="profile-evidence-list">' +
      items.map(e => `<li>"${e}"</li>`).join('') +
      '</ul>';
  }
  function confBadge(conf) {
    if (!conf) return '';
    return `<span class="profile-conf profile-conf-${conf}">${conf} confidence</span>`;
  }

  let html = '';

  if (p.insufficient_data_note) {
    html += `<div class="profile-data-warning"><strong>Limited evidence</strong> — ${p.insufficient_data_note}</div>`;
  }

  html += `<div class="profile-frameworks">
    <div class="profile-card">
      <div class="profile-card-label">Honey &amp; Mumford</div>
      <div class="profile-card-value">${p.honey_mumford_style || '—'} ${confBadge(p.honey_mumford_confidence)}</div>
      ${evidenceHTML(p.honey_mumford_evidence)}
      ${p.honey_mumford_reasoning ? `<div class="profile-card-reasoning">${p.honey_mumford_reasoning}</div>` : ''}
    </div>
    <div class="profile-card">
      <div class="profile-card-label">SOLO Level</div>
      <div class="profile-card-value">${p.solo_level || '—'} ${confBadge(p.solo_confidence)}</div>
      ${evidenceHTML(p.solo_evidence)}
      ${p.solo_reasoning ? `<div class="profile-card-reasoning">${p.solo_reasoning}</div>` : ''}
    </div>
  </div>`;

  const patterns = p.observed_patterns || [];
  if (patterns.length) {
    html += '<ul class="profile-patterns">' +
      patterns.map(pat => `<li>${pat}</li>`).join('') +
      '</ul>';
  }

  if (p.instructor_note) {
    html += `<p class="profile-note"><strong>For instructors:</strong> ${p.instructor_note}</p>`;
  }

  return html;
}

function renderProfile(p) {
  $('profile-session-label').textContent = 'This session';
  $('profile-content').innerHTML = _buildProfileHTML(p);
}

// FR thinking-profile fix: FR gets its own render function rather than reusing
// _buildProfileHTML above, which stays scenario-mode-only and unchanged. FR's
// profile is a single deterministic SOLO level (no LLM call) -- no Honey & Mumford
// (dropped entirely) and no invented confidence tag (the actual inputs -- matched
// point count and mean quality -- are shown instead).
function _buildFrProfileHTML(p) {
  return `<div class="profile-frameworks">
    <div class="profile-card">
      <div class="profile-card-label">SOLO Level</div>
      <div class="profile-card-value">${p.solo_level || '—'}</div>
      <div class="profile-card-evidence">Inputs: ${p.matched_count ?? 0} matched point(s), mean quality ${p.mean_quality ?? 0}/2 — see Key Points Covered above for the underlying evidence.</div>
      <div class="profile-card-reasoning">Deterministic reading of the Coverage/Quality data already scored above, not a separate judgment. Does not detect Extended Abstract (generalising beyond the given task).</div>
    </div>
  </div>`;
}

// ── End recall (I'm Done button) ───────────────
async function endRecall() {
  if (S.busy || S.phase !== 'recall') return;
  const text = $('user-input').value.trim();
  const writingMetrics = WritingTracker.collect(text);
  if (text) {
    addBubble('user', text);
    $('user-input').value = '';
    $('user-input').style.height = 'auto';
  }
  lock(true);
  showTyping();
  WritingTracker.reset();
  try {
    const data = await api('/api/end-recall', { session_id: S.sessionId, final_text: text || null, writing_metrics: writingMetrics });
    hideTyping();
    if (data.concluded) {
      await evaluate();
      return;
    }
    S.phase = data.phase || 'probing';
    S.probeCount  = data.probe_count  || 0;
    S.probeNumber = data.probe_number || 1;
    show('probing-banner');
    hide('btn-done');
    updateMeta();
    if (data.first_probe) addBubble('examiner', data.first_probe);
    lock(false);
    $('user-input').focus();
  } catch (err) {
    hideTyping();
    addBubble('examiner', `⚠ ${err.message}`);
    lock(false);
  }
}

$('btn-done').addEventListener('click', endRecall);


// ── Input wiring ───────────────────────────────
$('btn-send').addEventListener('click', send);

$('user-input').addEventListener('keydown', e => {
  WritingTracker.onKey(e);
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('user-input').addEventListener('paste', e => WritingTracker.onPaste(e, e.target.selectionStart));

$('user-input').addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ── Results ────────────────────────────────────
function scoreColor(s) {
  if (s >= 0.75) return 'var(--green)';
  if (s >= 0.5)  return 'var(--amber)';
  return 'var(--red)';
}

function renderResults(data) {
  const ev  = data.evaluations[0] || {};
  const pct = Math.round((ev.score || 0) * 100);
  const col = scoreColor(ev.score || 0);

  const hasTwoDim = ev.coverage_score != null && ev.quality_score != null;
  let scoreHtml = '';
  if (hasTwoDim) {
    const covPct  = Math.round((ev.coverage_score || 0) * 100);
    const qualPct = Math.round((ev.quality_score  || 0) * 100);
    scoreHtml = `
    <div class="score-card" style="padding-bottom:8px">
      <div style="display:flex;gap:24px;justify-content:center;margin-bottom:12px;flex-wrap:wrap">
        <div style="text-align:center">
          <div style="font-size:.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Coverage</div>
          <div style="font-size:2rem;font-weight:700;color:${scoreColor(ev.coverage_score||0)}">${covPct}%</div>
          <div style="font-size:.7rem;color:var(--muted)">steps recalled</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Explanation Quality</div>
          <div style="font-size:2rem;font-weight:700;color:${scoreColor(ev.quality_score||0)}">${qualPct}%</div>
          <div style="font-size:.7rem;color:var(--muted)">depth of reasoning</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Overall</div>
          <div class="score-number" style="color:${col};font-size:2.4rem">${pct}%</div>
          <div style="font-size:.7rem;color:var(--muted)">combined</div>
        </div>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" id="score-fill" style="background:${col}"></div>
      </div>
    </div>`;
  } else {
    scoreHtml = `
    <div class="score-card">
      <div class="score-number" style="color:${col}">${pct}%</div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" id="score-fill" style="background:${col}"></div>
      </div>
    </div>`;
  }
  let html = scoreHtml;

  if (ev.feedback) {
    html += `<div class="result-section">
      <h3>Feedback</h3>
      <p class="feedback-text">${ev.feedback}</p>
    </div>`;
  }

  if (ev.strengths?.length) {
    html += `<div class="result-section"><h3>Strengths</h3><ul class="result-list strengths">`;
    ev.strengths.forEach(s => {
      html += `<li><span class="bullet">✓</span>${s}</li>`;
    });
    html += `</ul></div>`;
  }

  if (ev.gaps?.length) {
    html += `<div class="result-section"><h3>Areas to improve</h3><ul class="result-list gaps">`;
    ev.gaps.forEach(g => {
      html += `<li><span class="bullet">△</span>${g}</li>`;
    });
    html += `</ul></div>`;
  }

  if (ev.matched_points?.length || ev.missed_points?.length) {
    html += `<div class="result-section"><h3>Key points</h3><div class="tag-list">`;
    (ev.matched_points || []).forEach(p => { html += `<span class="tag good">✓ ${p}</span>`; });
    (ev.missed_points  || []).forEach(p => { html += `<span class="tag miss">✗ ${p}</span>`; });
    html += `</div></div>`;
  }

  $('results-content').innerHTML = html;
  const canReport = llmAvailable();
  $('btn-report').disabled       = !canReport;
  $('btn-report').textContent    = 'Generate Report';
  $('report-note').textContent   = canReport ? '' : 'An LLM connection is required to generate reports. Configure a provider on the dashboard.';

  // Animate score bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { $('score-fill').style.width = pct + '%'; });
  });
}

$('btn-report').addEventListener('click', async () => {
  $('btn-report').disabled    = true;
  $('btn-report').innerHTML   = '<span class="spinner"></span>Generating…';
  $('report-note').textContent = '';
  try {
    const data = await api('/api/report', { session_id: S.sessionId });
    $('report-note').textContent = `Saved → ${data.path}`;
  } catch (err) {
    $('report-note').textContent = `Error: ${err.message}`;
  } finally {
    $('btn-report').disabled    = false;
    $('btn-report').textContent = 'Generate Report';
  }
});

$('btn-again').addEventListener('click', () => {
  hide('view-results');
  hide('profile-section');
  $('profile-content').innerHTML = '';
  IS_ADMIN ? showHome() : showDashboard();
});

// ── Free Response ──────────────────────────────
let _frPromptsLoaded = false;
const FR_ICONS = ['📝','💬','📚','✍️','🖊️','📄'];

async function loadFrPrompts() {
  try {
    const data = await api('/api/fr/prompts', {});
    const list = $('fr-prompt-list');
    list.innerHTML = '';

    if (!data.prompts.length) {
      list.innerHTML = '<p style="color:var(--muted);font-size:.88rem">No prompts available.</p>';
    } else {
      data.prompts.forEach((p, i) => {
        const card = document.createElement('div');
        card.className = 'scenario-card';
        card.innerHTML = `
          <div class="icon">${FR_ICONS[i % FR_ICONS.length]}</div>
          <div class="info">
            <h3>${p.title}</h3>
            <p>${p.description}</p>
            ${p.word_limit ? `<div class="meta">${p.word_limit}-word limit</div>` : ''}
          </div>`;
        card.addEventListener('click', () => startFrPrompt(i, data.prompts));
        list.appendChild(card);
      });
    }
    _frPromptsLoaded = true;
  } catch (err) {
    $('fr-prompt-list').textContent = `Failed to load: ${err.message}`;
  }
}

function askFrRating(question) {
  return new Promise(resolve => {
    $('fr-rating-question').textContent = question;
    const scale = $('fr-rating-scale');
    scale.innerHTML = '';
    for (let i = 1; i <= 10; i++) {
      const btn = document.createElement('button');
      btn.type      = 'button';
      btn.className = 'fr-rating-btn';
      btn.textContent = i;
      btn.addEventListener('click', () => {
        hide('fr-rating-overlay');
        resolve(i);
      });
      scale.appendChild(btn);
    }
    show('fr-rating-overlay');
  });
}

// Closing nudge (Part B): fixed, generic, content-blind text -- never references key
// points or varies by what's missing (same non-leakage principle as general_guidance
// and the api_fr_prompts key-point whitelist). Resolves true = "Add more", false =
// "Submit as final".
function askClosingNudge() {
  return new Promise(resolve => {
    const addBtn    = $('btn-fr-nudge-add');
    const submitBtn = $('btn-fr-nudge-submit');
    function onAdd()    { cleanup(); resolve(true); }
    function onSubmit() { cleanup(); resolve(false); }
    function cleanup() {
      hide('fr-closing-nudge-overlay');
      addBtn.removeEventListener('click', onAdd);
      submitBtn.removeEventListener('click', onSubmit);
    }
    addBtn.addEventListener('click', onAdd);
    submitBtn.addEventListener('click', onSubmit);
    show('fr-closing-nudge-overlay');
  });
}

// Final submit confirmation, including the AI-assistance declaration. Resolves null on
// cancel (submission aborted), or {used, notes} on confirm.
function askFrSubmitConfirm() {
  return new Promise(resolve => {
    const checkbox  = $('fr-ai-used-checkbox');
    const notes     = $('fr-ai-notes');
    const cancelBtn = $('btn-fr-confirm-cancel');
    const submitBtn = $('btn-fr-confirm-submit');
    checkbox.checked = false;
    notes.value = '';
    hide('fr-ai-notes');

    function onToggle() {
      if (checkbox.checked) show('fr-ai-notes');
      else { notes.value = ''; hide('fr-ai-notes'); }
    }
    function onCancel() { cleanup(); resolve(null); }
    function onSubmit() {
      cleanup();
      resolve({ used: checkbox.checked ? 'yes' : 'no', notes: checkbox.checked ? notes.value.trim() : '' });
    }
    function cleanup() {
      hide('fr-submit-confirm-overlay');
      checkbox.removeEventListener('change', onToggle);
      cancelBtn.removeEventListener('click', onCancel);
      submitBtn.removeEventListener('click', onSubmit);
    }
    checkbox.addEventListener('change', onToggle);
    cancelBtn.addEventListener('click', onCancel);
    submitBtn.addEventListener('click', onSubmit);
    show('fr-submit-confirm-overlay');
  });
}

function startFrPrompt(index, promptList) {
  const p = promptList[index];
  FR.promptData       = p;
  FR.preRating        = null;
  FR.postRating       = null;
  FR.closingNudgeShown = 0;
  FR.closingNudgeUsed  = false;

  $('fr-write-title').textContent = p.title;
  $('fr-prompt-text').textContent = p.prompt_text || p.description;
  $('fr-general-guidance').textContent = p.general_guidance || '';
  $('fr-textarea').value    = '';
  // disabled until the pre-write confidence rating is answered — this gate sits outside
  // the writing window itself, so it does not compromise passive process capture
  $('fr-textarea').disabled = true;
  $('btn-fr-submit').disabled  = true;
  $('btn-fr-submit').textContent = 'Submit';

  updateFrWordCount('', p.word_limit);

  // constraints sidebar
  if (p.constraints && p.constraints.length) {
    $('fr-constraints-list').innerHTML = p.constraints.map(c => `<li>${c}</li>`).join('');
    show('fr-constraints-section');
  } else {
    hide('fr-constraints-section');
  }

  hide('view-fr-prompts');
  show('view-fr-write');
  WritingTracker.reset();

  askFrRating('Before you begin — how well do you understand this topic well enough to explain it? (1–10)')
    .then(rating => {
      FR.preRating = rating;
      $('fr-textarea').disabled    = false;
      $('btn-fr-submit').disabled  = false;
      $('fr-textarea').focus();
    });
}

function updateFrWordCount(text, limit) {
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const lim   = limit || parseInt($('fr-word-count').dataset.limit) || 0;
  $('fr-word-count').dataset.limit = lim || '';
  const over  = lim && words > lim;
  $('fr-word-count').innerHTML =
    `<span style="font-weight:700;font-size:1.1rem;color:${over ? 'var(--red)' : 'var(--text)'}">${words}</span>` +
    (lim
      ? ` <span class="fr-word-limit${over ? ' fr-word-over' : ''}">/ ${lim} words</span>`
      : ` <span class="fr-word-limit">words</span>`);
}

$('fr-textarea').addEventListener('keydown', e => WritingTracker.onKey(e));
$('fr-textarea').addEventListener('paste', e => WritingTracker.onPaste(e, e.target.selectionStart));

$('fr-textarea').addEventListener('input', function () {
  const text = this.value;

  WritingTracker.onInput(text, this.selectionStart);
  updateFrWordCount(text);
});

$('btn-fr-back').addEventListener('click', () => {
  hide('view-fr-write');
  show('view-fr-prompts');
});

$('btn-fr-submit').addEventListener('click', async () => {
  const text = $('fr-textarea').value.trim();
  if (!text) { alert('Please write your response before submitting.'); return; }

  // Closing nudge (Part B1-B4): a single checkpoint before submission, capped at two
  // displays total -- the second happens only if "Add more" was chosen the first time,
  // and proceeds regardless of choice after that. Never becomes an open-ended loop, so
  // FR stays lightweight rather than turning into a probing conversation.
  if (FR.closingNudgeShown < 2) {
    FR.closingNudgeShown++;
    const wantsToAddMore = await askClosingNudge();
    if (wantsToAddMore) {
      FR.closingNudgeUsed = true;
      const ta = $('fr-textarea');
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
      return;
    }
  }

  const aiAssistance = await askFrSubmitConfirm();
  if (!aiAssistance) return;

  const writingMetrics = WritingTracker.collect(text);
  // Report-facing metadata only (Part B5/B6) -- merged into process_log, never sent
  // anywhere the grading call could see it; the grading call in api_fr_submit only ever
  // receives `text` and prompt_data, not writing_metrics.
  if (writingMetrics.process_log) {
    writingMetrics.process_log.closing_nudge_used = FR.closingNudgeUsed;
  }

  $('fr-textarea').disabled     = true;
  $('btn-fr-submit').disabled   = true;
  $('btn-fr-submit').innerHTML  = '<span class="spinner"></span>Evaluating…';

  try {
    const data = await api('/api/fr/submit', {
      prompt_id: FR.promptData.id,
      text,
      writing_metrics: writingMetrics,
      pre_rating: FR.preRating || undefined,
      ai_assistance: aiAssistance,
      provider: S.provider,
      model:    S.model,
      api_key:  S.apiKey || undefined,
    });
    FR.sessionId = data.session_id;

    // Post-write rating happens immediately after submission but BEFORE any score or
    // feedback is shown to the learner — the ordering that makes rate → re-rate a valid
    // illusion-of-explanatory-depth probe, not a self-fulfilling guess at the score.
    const postRating = await askFrRating("Now that you've explained it — how well do you feel you understood it? (1–10)");
    FR.postRating = postRating;
    api('/api/fr/post-rating', { session_id: data.session_id, post_rating: postRating }).catch(() => {});

    hide('view-fr-write');
    renderFrResults(data.evaluation);
    show('view-fr-results');
    fetchFrThinkingProfile();
  } catch (err) {
    $('fr-textarea').disabled    = false;
    $('btn-fr-submit').disabled  = false;
    $('btn-fr-submit').textContent = 'Submit';
    alert('Submission failed: ' + err.message);
  }
});

function renderFrResults(ev) {
  const pct = Math.round((ev.score || 0) * 100);
  const col = scoreColor(ev.score || 0);

  let html = `<div class="score-card">
    <div class="score-number" style="color:${col}">${pct}%</div>
    <div class="score-bar-bg">
      <div class="score-bar-fill" id="fr-score-fill" style="background:${col}"></div>
    </div>
  </div>`;

  if (ev.feedback) {
    html += `<div class="result-section"><h3>Feedback</h3><p class="feedback-text">${ev.feedback}</p></div>`;
  }
  if (ev.strengths?.length) {
    html += `<div class="result-section"><h3>Strengths</h3><ul class="result-list strengths">`;
    ev.strengths.forEach(s => { html += `<li><span class="bullet">✓</span>${s}</li>`; });
    html += `</ul></div>`;
  }
  if (ev.gaps?.length) {
    html += `<div class="result-section"><h3>Areas to improve</h3><ul class="result-list gaps">`;
    ev.gaps.forEach(g => { html += `<li><span class="bullet">△</span>${g}</li>`; });
    html += `</ul></div>`;
  }
  if (ev.matched_points?.length || ev.missed_points?.length) {
    html += `<div class="result-section"><h3>Key points</h3><div class="tag-list">`;
    (ev.matched_points || []).forEach(p => {
      const novel = p.match_type === 'novel_equivalent';
      const label = novel ? ' (novel match — pending review)' : '';
      html += `<span class="tag good">✓ ${p.construct}${label}</span>`;
    });
    (ev.missed_points || []).forEach(p => { html += `<span class="tag miss">✗ ${p.construct}</span>`; });
    html += `</div></div>`;
  }

  $('fr-results-content').innerHTML = html;
  const canFrReport = llmAvailable();
  $('btn-fr-report').disabled       = !canFrReport;
  $('btn-fr-report').textContent    = 'Generate Report';
  $('fr-report-note').textContent   = canFrReport ? '' : 'An LLM connection is required to generate reports. Configure a provider on the dashboard.';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const fill = $('fr-score-fill');
      if (fill) fill.style.width = pct + '%';
    });
  });
}

async function fetchFrThinkingProfile() {
  $('fr-profile-content').innerHTML = '<p><span class="spinner" style="border-top-color:var(--blue);border-color:var(--border);border-top-color:var(--blue)"></span> Analysing thinking style…</p>';
  show('fr-profile-section');
  try {
    const data = await api('/api/fr/thinking-profile', { session_id: FR.sessionId });
    if (data.profile) {
      renderFrProfile(data.profile);
    } else {
      hide('fr-profile-section');
    }
  } catch (_) {
    hide('fr-profile-section');
  }
}

function renderFrProfile(p) {
  $('fr-profile-content').innerHTML = _buildFrProfileHTML(p);
}

$('btn-fr-report').addEventListener('click', async () => {
  $('btn-fr-report').disabled   = true;
  $('btn-fr-report').innerHTML  = '<span class="spinner"></span>Generating…';
  $('fr-report-note').textContent = '';
  try {
    const data = await api('/api/fr/report', { session_id: FR.sessionId });
    $('fr-report-note').textContent = `Saved → ${data.path}`;
  } catch (err) {
    $('fr-report-note').textContent = `Error: ${err.message}`;
  } finally {
    $('btn-fr-report').disabled   = false;
    $('btn-fr-report').textContent = 'Generate Report';
  }
});

$('btn-fr-again').addEventListener('click', () => {
  hide('view-fr-results');
  hide('fr-profile-section');
  $('fr-profile-content').innerHTML = '';
  IS_ADMIN ? showHome() : showDashboard();
});

// ── Scenario Creator ───────────────────────────

function showCreate() {
  $('ai-description').value  = '';
  $('f-title').value         = '';
  $('f-description').value   = '';
  $('f-situation').value     = '';
  $('f-role').value          = 'participant';
  $('f-answer').value        = '';
  $('f-cov-weight').value    = '0.6';
  $('f-qual-weight').value   = '0.4';
  $('constraint-list').innerHTML = '';
  $('keypoint-list').innerHTML   = '';
  $('decision-list').innerHTML   = '';
  $('failure-list').innerHTML    = '';
  $('edge-list').innerHTML       = '';
  $('probe-list').innerHTML      = '';
  $('create-note').textContent   = '';
  hide('view-scenarios');
  show('view-create');
}

function addConstraintRow(value) {
  // add one editable row to the constraints list
  const list = $('constraint-list');
  const item = document.createElement('div');
  item.className = 'dynamic-item';
  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = 'e.g. Safety must be established first';
  input.value       = value || '';
  const remove = document.createElement('button');
  remove.className = 'btn-remove';
  remove.title     = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => item.remove());
  item.appendChild(input);
  item.appendChild(remove);
  list.appendChild(item);
}

function addKeyPointRow(value, weight) {
  // add one editable key point row with a weight dropdown
  const list = $('keypoint-list');
  const item = document.createElement('div');
  item.className = 'dynamic-item';

  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = 'e.g. hazard lights';
  input.value       = value || '';

  const sel = document.createElement('select');
  sel.title = 'Point weight';
  [['1','Low (1pt)'],['2','Medium (2pt)'],['3','High (3pt)'],['4','Critical (4pt)']].forEach(([v, label]) => {
    const opt = document.createElement('option');
    opt.value       = v;
    opt.textContent = label;
    if (parseInt(v) === (weight || 2)) opt.selected = true;
    sel.appendChild(opt);
  });

  const remove = document.createElement('button');
  remove.className   = 'btn-remove';
  remove.title       = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => item.remove());

  item.appendChild(input);
  item.appendChild(sel);
  item.appendChild(remove);
  list.appendChild(item);
}

function addSimpleRow(listId, placeholder, value) {
  const list = $(listId);
  const item = document.createElement('div');
  item.className = 'dynamic-item';
  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = placeholder;
  input.value       = value || '';
  const remove = document.createElement('button');
  remove.className   = 'btn-remove';
  remove.title       = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => item.remove());
  item.appendChild(input);
  item.appendChild(remove);
  list.appendChild(item);
}

const PROBE_TYPES = ['sequencing','how','rationale','decision','error','edge_case'];

function addProbeRow(probe) {
  const list = $('probe-list');
  const item = document.createElement('div');
  item.className = 'dynamic-item';

  const sel = document.createElement('select');
  sel.title = 'Probe type';
  sel.style.width = '140px';
  PROBE_TYPES.forEach(t => {
    const opt = document.createElement('option');
    opt.value       = t;
    opt.textContent = t;
    if (probe && probe.type === t) opt.selected = true;
    sel.appendChild(opt);
  });

  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = 'Probe question text…';
  input.value       = (probe && probe.question) || '';

  const remove = document.createElement('button');
  remove.className   = 'btn-remove';
  remove.title       = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => item.remove());

  item.appendChild(sel);
  item.appendChild(input);
  item.appendChild(remove);
  list.appendChild(item);
}

$('btn-add-constraint').addEventListener('click', () => addConstraintRow());
$('btn-add-keypoint').addEventListener('click',   () => addKeyPointRow());
$('btn-add-decision').addEventListener('click',   () => addSimpleRow('decision-list', 'e.g. Is the patient conscious?'));
$('btn-add-failure').addEventListener('click',    () => addSimpleRow('failure-list',  'e.g. Forgetting to check for hazards'));
$('btn-add-edge').addEventListener('click',       () => addSimpleRow('edge-list',     'e.g. Patient is pregnant'));
$('btn-add-probe').addEventListener('click',      () => addProbeRow(null));

$('btn-cancel-create').addEventListener('click', () => {
  hide('view-create');
  show('view-scenarios');
});

// Ask the AI to fill in the form fields from the description
$('btn-fill-ai').addEventListener('click', async () => {
  const desc = $('ai-description').value.trim();
  if (desc.length < 20) { $('ai-fill-note').textContent = 'Provide more detail'; return; }

  const btn = $('btn-fill-ai');
  btn.innerHTML = '<span class="spinner"></span>Generating…';
  btn.disabled  = true;
  $('create-note').textContent = '';

  try {
    const data = await api('/api/generate-scenario', { description: desc, provider: S.provider, model: S.model, api_key: S.apiKey || undefined });
    const s    = data.scenario;

    $('f-title').value       = s.title         || '';
    $('f-description').value = s.description   || '';
    $('f-situation').value   = s.situation     || '';
    $('f-role').value        = s.user_role     || 'participant';
    $('f-answer').value      = s.expert_answer || '';
    if (s.scoring_weights) {
      $('f-cov-weight').value  = s.scoring_weights.coverage || 0.6;
      $('f-qual-weight').value = s.scoring_weights.quality  || 0.4;
    }

    $('constraint-list').innerHTML = '';
    (s.constraints || []).forEach(c => addConstraintRow(c));

    $('keypoint-list').innerHTML = '';
    (s.key_points || []).forEach(p => addKeyPointRow(p, s.rubric ? s.rubric[p] : 2));

    $('decision-list').innerHTML = '';
    (s.decision_points || []).forEach(d => addSimpleRow('decision-list', 'e.g. Is the patient conscious?', d));

    $('failure-list').innerHTML = '';
    (s.failure_modes || []).forEach(f => addSimpleRow('failure-list', 'e.g. Forgetting to check for hazards', f));

    $('edge-list').innerHTML = '';
    (s.edge_cases || []).forEach(e => addSimpleRow('edge-list', 'e.g. Patient is pregnant', e));

    $('probe-list').innerHTML = '';
    (s.probe_bank || []).forEach(pb => addProbeRow(pb));

    $('create-note').textContent = 'AI draft loaded — review and edit each field before saving.';
  } catch (err) {
    $('create-note').textContent = 'Generation failed: ' + err.message;
  } finally {
    btn.innerHTML = 'Fill with AI';
    btn.disabled  = false;
  }
});

$('ai-description').addEventListener('input', () => {
  if ($('ai-fill-note').textContent) $('ai-fill-note').textContent = '';
});

// Save the form as a new scenario JSON file
$('btn-save-scenario').addEventListener('click', async () => {
  const title     = $('f-title').value.trim();
  const situation = $('f-situation').value.trim();
  const answer    = $('f-answer').value.trim();

  if (!title || !situation || !answer) {
    alert('Title, Situation, and Expert Answer are required.');
    return;
  }

  // collect key points and their weights from the dynamic rows
  const keyPoints = [];
  const rubric    = {};
  $('keypoint-list').querySelectorAll('.dynamic-item').forEach(item => {
    const text   = item.querySelector('input').value.trim();
    const weight = parseInt(item.querySelector('select').value);
    if (text) { keyPoints.push(text); rubric[text] = weight; }
  });

  const constraints = [];
  $('constraint-list').querySelectorAll('.dynamic-item input').forEach(input => {
    if (input.value.trim()) constraints.push(input.value.trim());
  });

  const decisionPoints = [];
  $('decision-list').querySelectorAll('.dynamic-item input').forEach(input => {
    if (input.value.trim()) decisionPoints.push(input.value.trim());
  });

  const failureModes = [];
  $('failure-list').querySelectorAll('.dynamic-item input').forEach(input => {
    if (input.value.trim()) failureModes.push(input.value.trim());
  });

  const edgeCases = [];
  $('edge-list').querySelectorAll('.dynamic-item input').forEach(input => {
    if (input.value.trim()) edgeCases.push(input.value.trim());
  });

  const probeBank = [];
  $('probe-list').querySelectorAll('.dynamic-item').forEach(item => {
    const probe_type = item.querySelector('select').value;
    const probe_text = item.querySelector('input').value.trim();
    if (probe_text) probeBank.push({ probe_type, probe_text });
  });

  const covWeight  = parseFloat($('f-cov-weight').value)  || 0.6;
  const qualWeight = parseFloat($('f-qual-weight').value) || 0.4;

  const btn = $('btn-save-scenario');
  btn.innerHTML = '<span class="spinner"></span>Saving…';
  btn.disabled  = true;

  try {
    const data = await api('/api/save-scenario', {
      title:            title,
      description:      $('f-description').value.trim(),
      situation:        situation,
      user_role:        $('f-role').value.trim() || 'participant',
      constraints:      constraints,
      expert_answer:    answer,
      key_points:       keyPoints,
      rubric:           rubric,
      decision_points:  decisionPoints,
      failure_modes:    failureModes,
      edge_cases:       edgeCases,
      probe_bank:       probeBank,
      scoring_weights:  { coverage: covWeight, quality: qualWeight },
    });

    $('create-note').textContent = 'Saved — returning to scenario list…';

    // reload the scenario list so the new scenario appears immediately
    $('scenario-list').innerHTML = '';
    await loadScenarios();

    // go back to the picker after a short pause
    setTimeout(() => { hide('view-create'); show('view-scenarios'); }, 1000);
  } catch (err) {
    $('create-note').textContent = 'Save failed: ' + err.message;
  } finally {
    btn.innerHTML = 'Save Scenario';
    btn.disabled  = false;
  }
});

// ── Novel-Equivalent Review Queue (Part C, construct/exemplar brief) ──────────

function _escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

async function showFrReview() {
  hide('view-fr-prompts');
  show('view-fr-review');
  await loadFrReviewQueue();
}

async function loadFrReviewQueue() {
  const list = $('fr-review-list');
  list.innerHTML = '<p style="color:var(--muted);font-size:.88rem">Loading…</p>';
  try {
    const data = await api('/api/admin/novel-equivalents', {});
    const reviews = data.reviews || [];
    if (!reviews.length) {
      list.innerHTML = '<p style="color:var(--muted);font-size:.88rem">No pending novel-equivalent matches.</p>';
      return;
    }
    list.innerHTML = '';
    reviews.forEach(r => list.appendChild(renderFrReviewCard(r)));
  } catch (err) {
    list.innerHTML = '<p style="color:var(--red);font-size:.88rem">Failed to load: ' + _escapeHtml(err.message) + '</p>';
  }
}

function renderFrReviewCard(r) {
  const card = document.createElement('div');
  card.className = 'scenario-card';
  card.style.cssText = 'cursor:default;text-align:left;margin-bottom:14px';

  const spansHtml = (r.evidence_spans || []).map(s => `<blockquote style="margin:4px 0;padding-left:10px;border-left:2px solid var(--border);font-size:.85rem">"${_escapeHtml(s)}"</blockquote>`).join('');
  const suggestedExemplar = (r.evidence_spans && r.evidence_spans[0]) || '';
  const isPool = !!r.pool_id;

  // Pooled key points (choose_n_of_m brief, Part B4): a pool-member review gets a second
  // promotion path -- "add as new pool member" -- alongside the existing exemplar route.
  // The system doesn't decide which is right; both are just available admin actions.
  const poolBadge = isPool
    ? `<div style="font-size:.72rem;color:var(--blue);margin-bottom:4px">Pool: ${_escapeHtml(r.pool_id)}</div>`
    : '';
  const newMemberSection = isPool ? `
    <div class="field" style="margin-top:10px">
      <label style="font-size:.78rem">New technique description (if this is a genuinely different technique, not a paraphrase of an existing member)</label>
      <input type="text" class="fr-review-member-input" value="${_escapeHtml(r.construct)}">
    </div>
  ` : '';

  card.innerHTML = `
    <div style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">${_escapeHtml(r.prompt_title)}</div>
    ${poolBadge}
    <div style="font-weight:600;margin-bottom:6px">${_escapeHtml(r.construct)}</div>
    <div style="font-size:.85rem;color:var(--muted);margin-bottom:6px">Evidence:</div>
    ${spansHtml}
    <div style="font-size:.85rem;margin:8px 0"><strong>Justification:</strong> ${_escapeHtml(r.justification)}</div>
    <div class="field" style="margin-top:10px">
      <label style="font-size:.78rem">Exemplar phrase to add (edit/generalize before promoting)</label>
      <input type="text" class="fr-review-exemplar-input" value="${_escapeHtml(suggestedExemplar)}">
    </div>
    ${newMemberSection}
    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
      <button class="btn btn-primary fr-review-promote">${isPool ? 'Add as new exemplar' : 'Promote'}</button>
      ${isPool ? '<button class="btn btn-primary fr-review-new-member">Add as new pool member</button>' : ''}
      <button class="btn btn-ghost fr-review-dismiss">Dismiss</button>
    </div>
    <div class="fr-review-note" style="font-size:.8rem;margin-top:6px"></div>
  `;

  const note = card.querySelector('.fr-review-note');

  card.querySelector('.fr-review-promote').addEventListener('click', async (e) => {
    const btn = e.target;
    const exemplar = card.querySelector('.fr-review-exemplar-input').value.trim();
    if (!exemplar) { note.textContent = 'Enter an exemplar phrase first.'; note.style.color = 'var(--red)'; return; }
    btn.disabled = true;
    try {
      await api('/api/admin/novel-equivalents/promote', { review_id: r.id, action: 'add_exemplar', exemplar });
      card.remove();
    } catch (err) {
      note.textContent = 'Promote failed: ' + err.message;
      note.style.color = 'var(--red)';
      btn.disabled = false;
    }
  });

  const newMemberBtn = card.querySelector('.fr-review-new-member');
  if (newMemberBtn) {
    newMemberBtn.addEventListener('click', async (e) => {
      const btn = e.target;
      const memberConstruct = card.querySelector('.fr-review-member-input').value.trim();
      const exemplar = card.querySelector('.fr-review-exemplar-input').value.trim();
      if (!memberConstruct) { note.textContent = 'Enter a description for the new technique first.'; note.style.color = 'var(--red)'; return; }
      btn.disabled = true;
      try {
        await api('/api/admin/novel-equivalents/promote', {
          review_id: r.id, action: 'new_member', member_construct: memberConstruct, exemplar,
        });
        card.remove();
      } catch (err) {
        note.textContent = 'Add as new member failed: ' + err.message;
        note.style.color = 'var(--red)';
        btn.disabled = false;
      }
    });
  }

  card.querySelector('.fr-review-dismiss').addEventListener('click', async (e) => {
    const btn = e.target;
    btn.disabled = true;
    try {
      await api('/api/admin/novel-equivalents/dismiss', { review_id: r.id });
      card.remove();
    } catch (err) {
      note.textContent = 'Dismiss failed: ' + err.message;
      note.style.color = 'var(--red)';
      btn.disabled = false;
    }
  });

  return card;
}

// ── Novel-Equivalent Reliability Metrics (fr_hardening brief, Part D) ─────────

async function showFrReliability() {
  hide('view-fr-prompts');
  show('view-fr-reliability');
  await loadFrReliability();
}

async function loadFrReliability() {
  const body  = $('fr-reliability-body');
  const empty = $('fr-reliability-empty');
  body.innerHTML = '<tr><td colspan="8" style="padding:6px 8px;color:var(--muted)">Loading…</td></tr>';
  empty.style.display = 'none';
  try {
    const data  = await api('/api/admin/fr-match-stats', {});
    const stats = data.stats || [];
    if (!stats.length) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    body.innerHTML = stats.map(s => `
      <tr style="border-bottom:1px solid var(--border)">
        <td style="padding:6px 8px">${_escapeHtml(s.prompt_title)}</td>
        <td style="padding:6px 8px">${_escapeHtml(s.construct || s.key_point_id)}</td>
        <td style="padding:6px 8px">${s.total_matches}</td>
        <td style="padding:6px 8px">${s.novel_count}</td>
        <td style="padding:6px 8px">${Math.round(s.novel_rate * 100)}%</td>
        <td style="padding:6px 8px">${s.promoted}</td>
        <td style="padding:6px 8px">${s.dismissed}</td>
        <td style="padding:6px 8px">${s.pending_review}</td>
      </tr>
    `).join('');
  } catch (err) {
    body.innerHTML = `<tr><td colspan="8" style="padding:6px 8px;color:var(--red)">Failed to load: ${_escapeHtml(err.message)}</td></tr>`;
  }
}

// ── Prompt Creator ─────────────────────────────

function showFrCreate() {
  $('fr-ai-description').value    = '';
  $('fp-title').value             = '';
  $('fp-description').value       = '';
  $('fp-prompt-text').value       = '';
  $('fp-word-limit').value        = '';
  $('fp-answer').value            = '';
  $('fp-general-guidance').value  = '';
  $('fp-constraint-list').innerHTML = '';
  $('fp-keypoint-list').innerHTML   = '';
  $('fp-pool-list').innerHTML       = '';
  $('fr-create-note').textContent   = '';
  hide('view-fr-prompts');
  show('view-fr-create');
}

function addFpConstraintRow(value) {
  const list = $('fp-constraint-list');
  const item = document.createElement('div');
  item.className = 'dynamic-item';
  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = 'e.g. Must include at least one specific example';
  input.value       = value || '';
  const remove = document.createElement('button');
  remove.className   = 'btn-remove';
  remove.title       = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => item.remove());
  item.appendChild(input);
  item.appendChild(remove);
  list.appendChild(item);
}

function addFpExemplarRow(exemplarList, value) {
  const row = document.createElement('div');
  row.className = 'fp-kp-exemplar-row';
  const input = document.createElement('input');
  input.type        = 'text';
  input.placeholder = 'e.g. eye contact';
  input.value       = value || '';
  const remove = document.createElement('button');
  remove.className   = 'btn-remove';
  remove.title       = 'Remove exemplar';
  remove.textContent = '×';
  remove.addEventListener('click', () => row.remove());
  row.appendChild(input);
  row.appendChild(remove);
  exemplarList.appendChild(row);
}

// kp: {construct, exemplars, importance} -- shape returned by /api/generate-prompt and by
// loaders.py's migration of legacy flat-string key points (construct/exemplar brief, Part E)
function addFpKeyPointRow(kp) {
  kp = kp || {};
  const list = $('fp-keypoint-list');
  const card = document.createElement('div');
  card.className = 'fp-kp-card';

  const top = document.createElement('div');
  top.className = 'fp-kp-card-top';

  const input = document.createElement('input');
  input.type        = 'text';
  input.className   = 'fp-kp-construct';
  input.placeholder = 'e.g. Names a technique for signaling engagement during listening';
  input.value        = kp.construct || '';

  const sel = document.createElement('select');
  sel.className = 'fp-kp-importance';
  sel.title     = 'Importance';
  [['LOW','Low'],['MEDIUM','Medium'],['HIGH','High'],['CRITICAL','Critical']].forEach(([v, label]) => {
    const opt = document.createElement('option');
    opt.value       = v;
    opt.textContent = label;
    if ((kp.importance || 'MEDIUM') === v) opt.selected = true;
    sel.appendChild(opt);
  });

  const removePoint = document.createElement('button');
  removePoint.className   = 'btn-remove';
  removePoint.title       = 'Remove key point';
  removePoint.textContent = '×';
  removePoint.addEventListener('click', () => card.remove());

  top.appendChild(input);
  top.appendChild(sel);
  top.appendChild(removePoint);
  card.appendChild(top);

  const exLabel = document.createElement('div');
  exLabel.className   = 'fp-kp-exemplars-label';
  exLabel.textContent  = 'Exemplars (concrete, distinct ways to satisfy the construct)';
  card.appendChild(exLabel);

  const exemplarList = document.createElement('div');
  exemplarList.className = 'fp-kp-exemplar-list';
  card.appendChild(exemplarList);

  const addExemplarBtn = document.createElement('button');
  addExemplarBtn.className   = 'btn-add';
  addExemplarBtn.textContent = '+ Add exemplar';
  addExemplarBtn.addEventListener('click', (e) => { e.preventDefault(); addFpExemplarRow(exemplarList); });
  card.appendChild(addExemplarBtn);

  (kp.exemplars && kp.exemplars.length ? kp.exemplars : ['']).forEach(ex => addFpExemplarRow(exemplarList, ex));

  list.appendChild(card);
}

// Pooled key points (choose_n_of_m brief, Part D). member: {construct, exemplars} --
// no importance field (importance is pool-level only, set on the pool card).
function addFpPoolMemberRow(list, member) {
  member = member || {};
  const card = document.createElement('div');
  card.className = 'fp-pool-member-card';

  const top = document.createElement('div');
  top.className = 'fp-pool-member-card-top';

  const input = document.createElement('input');
  input.type        = 'text';
  input.className   = 'fp-pm-construct';
  input.placeholder = 'e.g. Names a technique for confirming understanding of what was said';
  input.value        = member.construct || '';

  const removeMember = document.createElement('button');
  removeMember.className   = 'btn-remove';
  removeMember.title       = 'Remove member';
  removeMember.textContent = '×';
  removeMember.addEventListener('click', () => card.remove());

  top.appendChild(input);
  top.appendChild(removeMember);
  card.appendChild(top);

  const exLabel = document.createElement('div');
  exLabel.className  = 'fp-kp-exemplars-label';
  exLabel.textContent = 'Exemplars (concrete, distinct ways to satisfy the construct)';
  card.appendChild(exLabel);

  const exemplarList = document.createElement('div');
  exemplarList.className = 'fp-kp-exemplar-list';
  card.appendChild(exemplarList);

  const addExemplarBtn = document.createElement('button');
  addExemplarBtn.className   = 'btn-add';
  addExemplarBtn.textContent = '+ Add exemplar';
  addExemplarBtn.addEventListener('click', (e) => { e.preventDefault(); addFpExemplarRow(exemplarList); });
  card.appendChild(addExemplarBtn);

  (member.exemplars && member.exemplars.length ? member.exemplars : ['']).forEach(ex => addFpExemplarRow(exemplarList, ex));

  list.appendChild(card);
}

// pool: {pool_id, required_count, importance, members} -- pool_id is a human label here;
// the server slugifies it into the stable pool_id on save (loaders.py migration), same
// as how a standalone key point's construct text becomes its id.
function addFpPoolRow(pool) {
  pool = pool || {};
  const list = $('fp-pool-list');
  const card = document.createElement('div');
  card.className = 'fp-pool-card';

  const top = document.createElement('div');
  top.className = 'fp-pool-card-top';

  const label = document.createElement('input');
  label.type        = 'text';
  label.className   = 'fp-pool-label';
  label.placeholder = 'Pool label, e.g. Listening techniques';
  label.value        = pool.pool_id || '';

  const required = document.createElement('input');
  required.type        = 'number';
  required.className   = 'fp-pool-required';
  required.min          = '1';
  required.placeholder = 'Required';
  required.title       = 'How many members a response must match for full credit';
  required.value        = pool.required_count || '';

  const sel = document.createElement('select');
  sel.className = 'fp-pool-importance';
  sel.title     = 'Importance (applies to the whole pool)';
  [['LOW','Low'],['MEDIUM','Medium'],['HIGH','High'],['CRITICAL','Critical']].forEach(([v, lbl]) => {
    const opt = document.createElement('option');
    opt.value       = v;
    opt.textContent = lbl;
    if ((pool.importance || 'MEDIUM') === v) opt.selected = true;
    sel.appendChild(opt);
  });

  const removePool = document.createElement('button');
  removePool.className   = 'btn-remove';
  removePool.title       = 'Remove pool';
  removePool.textContent = '×';
  removePool.addEventListener('click', () => card.remove());

  top.appendChild(label);
  top.appendChild(required);
  top.appendChild(sel);
  top.appendChild(removePool);
  card.appendChild(top);

  const memberLabel = document.createElement('div');
  memberLabel.className  = 'fp-kp-exemplars-label';
  memberLabel.textContent = 'Members';
  card.appendChild(memberLabel);

  const memberList = document.createElement('div');
  memberList.className = 'fp-pool-member-list';
  card.appendChild(memberList);

  const addMemberBtn = document.createElement('button');
  addMemberBtn.className   = 'btn-add';
  addMemberBtn.textContent = '+ Add member';
  addMemberBtn.addEventListener('click', (e) => { e.preventDefault(); addFpPoolMemberRow(memberList); });
  card.appendChild(addMemberBtn);

  (pool.members && pool.members.length ? pool.members : [{}, {}]).forEach(m => addFpPoolMemberRow(memberList, m));

  list.appendChild(card);
}

$('btn-fp-add-constraint').addEventListener('click', () => addFpConstraintRow());
$('btn-fp-add-keypoint').addEventListener('click',   () => addFpKeyPointRow());
$('btn-fp-add-pool').addEventListener('click',       () => addFpPoolRow());

$('btn-cancel-fr-create').addEventListener('click', () => {
  hide('view-fr-create');
  show('view-fr-prompts');
});

$('btn-fr-fill-ai').addEventListener('click', async () => {
  const desc = $('fr-ai-description').value.trim();
  if (desc.length < 20) { $('fr-ai-fill-note').textContent = 'Provide more detail'; return; }

  const btn = $('btn-fr-fill-ai');
  btn.innerHTML = '<span class="spinner"></span>Generating…';
  btn.disabled  = true;
  $('fr-create-note').textContent = '';

  try {
    const data = await api('/api/generate-prompt', { description: desc, provider: S.provider, model: S.model, api_key: S.apiKey || undefined });
    const p    = data.prompt;

    $('fp-title').value       = p.title        || '';
    $('fp-description').value = p.description  || '';
    $('fp-prompt-text').value = p.prompt_text  || '';
    $('fp-word-limit').value  = p.word_limit   || '';
    $('fp-answer').value      = p.expert_answer|| '';
    $('fp-general-guidance').value = p.general_guidance || '';

    $('fp-constraint-list').innerHTML = '';
    (p.constraints || []).forEach(c => addFpConstraintRow(c));

    $('fp-keypoint-list').innerHTML = '';
    (p.key_points || []).forEach(kp => addFpKeyPointRow(kp));

    $('fr-create-note').textContent = 'AI draft loaded — review and edit each field before saving.';
  } catch (err) {
    $('fr-create-note').textContent = 'Generation failed: ' + err.message;
  } finally {
    btn.innerHTML = 'Fill with AI';
    btn.disabled  = false;
  }
});

$('fr-ai-description').addEventListener('input', () => {
  if ($('fr-ai-fill-note').textContent) $('fr-ai-fill-note').textContent = '';
});

$('btn-save-prompt').addEventListener('click', async () => {
  const title      = $('fp-title').value.trim();
  const promptText = $('fp-prompt-text').value.trim();
  const answer     = $('fp-answer').value.trim();

  if (!title || !promptText || !answer) {
    alert('Title, Prompt Text, and Expert Answer are required.');
    return;
  }

  const keyPoints = [];
  $('fp-keypoint-list').querySelectorAll('.fp-kp-card').forEach(card => {
    const construct = card.querySelector('.fp-kp-construct').value.trim();
    if (!construct) return;
    const importance = card.querySelector('.fp-kp-importance').value;
    const exemplars = [];
    card.querySelectorAll('.fp-kp-exemplar-row input').forEach(input => {
      if (input.value.trim()) exemplars.push(input.value.trim());
    });
    keyPoints.push({ construct, exemplars, importance });
  });

  // Pooled key points (choose_n_of_m brief, Part D) -- pool_id here is the raw label
  // text; the server slugifies it into a stable id on save (loaders.py migration).
  const pools = [];
  $('fp-pool-list').querySelectorAll('.fp-pool-card').forEach(card => {
    const poolLabel = card.querySelector('.fp-pool-label').value.trim();
    if (!poolLabel) return;
    const requiredCount = parseInt(card.querySelector('.fp-pool-required').value) || 0;
    const importance    = card.querySelector('.fp-pool-importance').value;
    const members = [];
    card.querySelectorAll('.fp-pool-member-card').forEach(mcard => {
      const construct = mcard.querySelector('.fp-pm-construct').value.trim();
      if (!construct) return;
      const exemplars = [];
      mcard.querySelectorAll('.fp-kp-exemplar-row input').forEach(input => {
        if (input.value.trim()) exemplars.push(input.value.trim());
      });
      members.push({ construct, exemplars });
    });
    if (!members.length) return;
    pools.push({ pool_id: poolLabel, required_count: requiredCount, importance, members });
  });

  const constraints = [];
  $('fp-constraint-list').querySelectorAll('.dynamic-item input').forEach(input => {
    if (input.value.trim()) constraints.push(input.value.trim());
  });

  const wordLimit = parseInt($('fp-word-limit').value) || null;

  const btn = $('btn-save-prompt');
  btn.innerHTML = '<span class="spinner"></span>Saving…';
  btn.disabled  = true;

  try {
    await api('/api/save-prompt', {
      title:             title,
      description:       $('fp-description').value.trim(),
      prompt_text:       promptText,
      word_limit:        wordLimit,
      constraints:       constraints,
      expert_answer:     answer,
      key_points:        keyPoints,
      pools:             pools,
      general_guidance:  $('fp-general-guidance').value.trim(),
    });

    $('fr-create-note').textContent = 'Saved — returning to prompt list…';
    _frPromptsLoaded = false;
    await loadFrPrompts();
    setTimeout(() => { hide('view-fr-create'); show('view-fr-prompts'); }, 1000);
  } catch (err) {
    $('fr-create-note').textContent = 'Save failed: ' + err.message;
  } finally {
    btn.innerHTML = 'Save Prompt';
    btn.disabled  = false;
  }
});

// ── Learning Profile ───────────────────────────
let _profileAllReports = [];
let _profileFilter     = 'all';

async function showProfileMode() {
  ['view-home','view-dashboard'].forEach(hide);
  show('view-profile');
  $('profile-heading').textContent = 'Learning Profile';

  if (IS_ADMIN) {
    show('profile-user-row');
    await _loadProfileUserList();
  } else {
    hide('profile-user-row');
    await _loadProfile(null);
  }
}

async function _loadProfileUserList() {
  try {
    const data = await api('/api/users', {});
    const sel  = $('profile-user-select');
    sel.innerHTML = '';
    data.users.forEach(u => {
      const opt       = document.createElement('option');
      opt.value       = u.username;
      opt.textContent = u.display_name;
      sel.appendChild(opt);
    });
    sel.onchange = () => _loadProfile(sel.value);
    if (sel.options.length) await _loadProfile(sel.value);
  } catch (err) {
    $('profile-report-list').innerHTML =
      `<p class="no-reports-msg">Could not load user list: ${err.message}</p>`;
  }
}

async function _loadProfile(username) {
  _profileAllReports = [];
  $('profile-report-list').innerHTML =
    '<p class="no-reports-msg">Loading…</p>';
  _clearChart();
  ['stat-total','stat-avg','stat-best','stat-trend'].forEach(id => {
    $(id).textContent = '—';
    $(id).style.color = '';
  });

  try {
    const body = username ? { username } : {};
    const data = await api('/api/learning-profile', body);
    _profileAllReports = data.reports;
    $('profile-heading').textContent =
      'Learning Profile — ' + (data.display_name || data.username);
    _renderAnalysis(data.aggregate);
    _renderProfile();
  } catch (err) {
    $('profile-report-list').innerHTML =
      `<p class="no-reports-msg">Failed to load: ${err.message}</p>`;
  }
}

function _filteredReports() {
  if (_profileFilter === 'all') return _profileAllReports;
  return _profileAllReports.filter(r => r.type === _profileFilter);
}

function _renderProfile() {
  const reps = _filteredReports();

  if (reps.length) {
    const scores = reps.map(r => r.score);
    const avg    = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const best   = Math.max(...scores);
    let trend    = 0;
    if (scores.length >= 2) {
      const mid   = Math.floor(scores.length / 2);
      const early = scores.slice(0, mid).reduce((a, b) => a + b, 0) / mid;
      const late  = scores.slice(mid).reduce((a, b) => a + b, 0) / (scores.length - mid);
      trend = Math.round(late - early);
    }
    $('stat-total').textContent = reps.length;
    $('stat-avg').textContent   = avg + '%';
    $('stat-best').textContent  = best + '%';
    const tEl = $('stat-trend');
    tEl.textContent = (trend > 0 ? '+' : '') + trend + '%';
    tEl.style.color = trend > 0 ? 'var(--green)' : trend < 0 ? 'var(--red)' : '';
  } else {
    $('stat-total').textContent = '0';
    ['stat-avg','stat-best','stat-trend'].forEach(id => {
      $(id).textContent = '—';
      $(id).style.color = '';
    });
  }

  _drawChart(reps);
  _renderProfileList([...reps].reverse());
}

function _renderProfileList(reps) {
  const el = $('profile-report-list');
  if (!reps.length) {
    el.innerHTML = '<p class="no-reports-msg">No assessments found for this filter.</p>';
    return;
  }
  el.innerHTML = '';
  reps.forEach(r => {
    const col       = r.score >= 75 ? 'var(--green)' : r.score >= 50 ? 'var(--amber)' : 'var(--red)';
    const typeLabel = r.type === 'fr' ? 'Free Response' : 'Scenario';
    const item      = document.createElement('div');
    item.className  = 'profile-report-item';
    item.innerHTML  = `
      <span class="pri-type ${r.type}">${typeLabel}</span>
      <div class="pri-info">
        <div class="pri-title">${r.title}</div>
        <div class="pri-date">${r.date_str}</div>
      </div>
      <div class="pri-score" style="color:${col}">${r.score}%</div>
      <a href="${r.url}" target="_blank" rel="noopener" class="btn-view-report">View</a>`;
    el.appendChild(item);
  });
}

// toggle buttons
document.querySelectorAll('#profile-toggles .toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#profile-toggles .toggle-btn')
      .forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    _profileFilter = btn.dataset.filter;
    _renderProfile();
  });
});

// ── Overarching analysis ────────────────────────
let _lastAggregate = null;

function _renderAnalysis(aggregate) {
  _lastAggregate = aggregate;
  const el     = $('analysis-aggregate');
  const noteEl = $('analysis-note');
  const btn    = $('btn-gen-analysis');
  hide('analysis-narrative');
  $('analysis-narrative').innerHTML = '';

  if (!aggregate || !aggregate.has_data) {
    el.innerHTML       = '<p class="no-analysis-msg">No thinking profile or instructor data found in reports. '
      + 'Complete assessments with an LLM-enabled session to populate this section.</p>';
    btn.disabled       = true;
    noteEl.textContent = '';
    return;
  }

  const canGenerate  = llmAvailable();
  btn.disabled       = !canGenerate;
  btn.textContent    = 'Generate AI Analysis';
  noteEl.textContent = canGenerate ? '' : 'Configure an LLM provider above to enable AI analysis.';

  // Side-by-side profile cards — mirrors the post-assessment Thinking Profile layout
  let html = '<div class="profile-frameworks">';
  html += _buildAggregateCard('Honey &amp; Mumford', aggregate.hm_entries || [], 'style');
  html += _buildAggregateCard('SOLO Taxonomy',       aggregate.solo_entries || [], 'level');
  html += '</div>';

  const patterns = aggregate.all_patterns || [];
  if (patterns.length) {
    html += '<div class="analysis-sub" style="margin-top:4px">Observed Patterns</div>';
    patterns.forEach(p => { html += `<div class="pattern-item">${p}</div>`; });
  }

  const gaps = aggregate.all_gaps || [];
  if (gaps.length) {
    html += '<div class="analysis-sub" style="margin-top:12px">Recurring Learning Gaps</div>';
    gaps.forEach(g => { html += `<div class="gap-item">${g}</div>`; });
  }

  el.innerHTML = html;
}

function _buildAggregateCard(label, entries, styleKey) {
  if (!entries.length) {
    return `<div class="profile-card">
      <div class="profile-card-label">${label}</div>
      <p class="no-analysis-msg" style="font-size:.78rem;margin-top:6px">No data in reports yet</p>
    </div>`;
  }

  // Tally occurrences to find the most common style/level
  const counts = {};
  entries.forEach(e => { const k = e[styleKey]; counts[k] = (counts[k] || 0) + 1; });
  const [primaryName, primaryCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  const pct = Math.round(primaryCount / entries.length * 100);

  function confBadge(conf) {
    if (!conf) return '';
    const lvl = conf.toLowerCase();
    const cls = ['high','medium','low'].includes(lvl) ? lvl : 'medium';
    return `<span class="profile-conf profile-conf-${cls}">${conf}</span>`;
  }

  let html = `<div class="profile-card">
    <div class="profile-card-label">${label}</div>
    <div class="profile-card-value">${primaryName}</div>`;

  if (entries.length > 1) {
    html += `<div style="font-size:.72rem;color:var(--muted);margin-bottom:10px">
      Most frequent — ${primaryCount} of ${entries.length} assessments (${pct}%)
    </div>`;
  }

  // Per-assessment history — one block per report entry
  entries.forEach((e, i) => {
    const name       = e[styleKey];
    const typeLabel  = e.type === 'fr' ? 'FR' : 'Scenario';
    const scoreCol   = e.score >= 75 ? 'var(--green)' : e.score >= 50 ? 'var(--amber)' : 'var(--red)';
    const evidence   = Array.isArray(e.evidence) ? e.evidence : (e.evidence ? [e.evidence] : []);

    html += `<div style="border-top:1px solid var(--border);padding-top:8px;margin-top:${i === 0 ? '0' : '8px'}">
      <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;flex-wrap:wrap">
        <span style="font-size:.7rem;color:var(--muted)">${e.date.slice(0,10)}</span>
        <span style="font-size:.65rem;color:var(--muted)">·</span>
        <span style="font-size:.7rem;color:var(--muted)">${typeLabel}</span>
        <span style="font-size:.65rem;color:var(--muted)">·</span>
        <span style="font-size:.7rem;font-weight:700;color:${scoreCol}">${e.score}%</span>
      </div>
      <div class="profile-card-value" style="font-size:.85rem;margin-bottom:2px">
        ${name}${e.confidence ? ' ' + confBadge(e.confidence) : ''}
      </div>`;

    if (evidence.length) {
      html += '<ul class="profile-evidence-list">';
      evidence.slice(0, 2).forEach(ev => { html += `<li>"${ev}"</li>`; });
      html += '</ul>';
    }
    if (e.reasoning) {
      html += `<div class="profile-card-reasoning">${e.reasoning}</div>`;
    }
    html += '</div>';
  });

  html += '</div>'; // close profile-card
  return html;
}

function _renderAiNarrative(a, reportCount) {
  const el = $('analysis-narrative');
  let html = `<div class="ai-narrative-label">AI Learning Profile Synthesis`;
  if (reportCount) html += ` <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:.72rem;color:var(--muted)">based on ${reportCount} report${reportCount > 1 ? 's' : ''}</span>`;
  html += '</div>';

  if (a.overall_narrative) {
    html += `<div class="narrative-overall">${a.overall_narrative}</div>`;
  }

  if (a.learning_style_summary) {
    html += `<div class="narrative-section">
      <div class="narrative-section-label">Learning Style</div>
      <div class="narrative-body">${a.learning_style_summary}</div>
    </div>`;
  }

  if (a.cognitive_development) {
    html += `<div class="narrative-section">
      <div class="narrative-section-label">Cognitive Development</div>
      <div class="narrative-body">${a.cognitive_development}</div>
    </div>`;
  }

  if (a.consistent_strengths?.length) {
    html += `<div class="narrative-section">
      <div class="narrative-section-label">Consistent Strengths</div>
      <ul class="narrative-list strengths">
        ${a.consistent_strengths.map(s => `<li>${s}</li>`).join('')}
      </ul>
    </div>`;
  }

  if (a.development_areas?.length) {
    html += `<div class="narrative-section">
      <div class="narrative-section-label">Development Areas</div>
      <ul class="narrative-list areas">
        ${a.development_areas.map(s => `<li>${s}</li>`).join('')}
      </ul>
    </div>`;
  }

  if (a.recommendations?.length) {
    html += `<div class="narrative-section">
      <div class="narrative-section-label">Recommendations</div>
      <ul class="narrative-list recs">
        ${a.recommendations.map(s => `<li>${s}</li>`).join('')}
      </ul>
    </div>`;
  }

  el.innerHTML = html;
}

$('btn-gen-analysis').addEventListener('click', async () => {
  const btn = $('btn-gen-analysis');
  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner"></span>Analysing…';
  hide('analysis-narrative');

  try {
    const username = IS_ADMIN && $('profile-user-select').value
      ? $('profile-user-select').value : null;
    const body = {
      provider: S.provider,
      model:    S.model,
      api_key:  S.apiKey || undefined,
    };
    if (username) body.username = username;

    const data = await api('/api/learning-profile/analysis', body);
    _renderAiNarrative(data.analysis, data.report_count);
    show('analysis-narrative');
  } catch (err) {
    const el = $('analysis-narrative');
    el.innerHTML = `<div class="narrative-overall" style="border-color:var(--red);color:var(--red)">Analysis failed: ${err.message}</div>`;
    show('analysis-narrative');
  } finally {
    btn.disabled  = false;
    btn.textContent = 'Regenerate Analysis';
  }
});

// ── Progress chart ──────────────────────────────
let _chartPts = [];

function _getChartColors() {
  const theme = document.documentElement.dataset.theme || '';
  const dark  = theme === 'dark' || theme === 'ultra-dark';
  const ultra = theme === 'ultra-dark';
  return {
    grid:     dark  ? '#334155' : '#e2e8f0',
    axis:     dark  ? '#94a3b8' : '#64748b',
    scenario: ultra ? '#00ffff' : '#3b82f6',
    fr:       ultra ? '#00ff41' : '#10b981',
    line:     dark  ? '#475569' : '#cbd5e1',
    nodata:   '#94a3b8',
  };
}

function _clearChart() {
  const canvas = $('profile-chart');
  if (!canvas) return;
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  _chartPts = [];
}

function _drawChart(reps) {
  const canvas = $('profile-chart');
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.parentElement.clientWidth - 32;
  const H   = 220;

  canvas.width        = W * dpr;
  canvas.height       = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const C = _getChartColors();

  if (!reps.length) {
    ctx.fillStyle    = C.nodata;
    ctx.font         = '13px system-ui, sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No assessments to display', W / 2, H / 2);
    _chartPts = [];
    return;
  }

  const PAD = { top: 20, right: 20, bottom: 38, left: 48 };
  const cW  = W - PAD.left - PAD.right;
  const cH  = H - PAD.top  - PAD.bottom;
  const n   = reps.length;
  const xFor = i => n === 1 ? PAD.left + cW / 2 : PAD.left + (i / (n - 1)) * cW;
  const yFor = s => PAD.top + cH - (Math.min(s, 100) / 100) * cH;

  // y-axis grid + labels
  ctx.setLineDash([4, 3]);
  for (const pct of [0, 25, 50, 75, 100]) {
    const yp = yFor(pct);
    ctx.strokeStyle  = C.grid;
    ctx.lineWidth    = 1;
    ctx.beginPath(); ctx.moveTo(PAD.left, yp); ctx.lineTo(PAD.left + cW, yp); ctx.stroke();
    ctx.fillStyle    = C.axis;
    ctx.font         = '10px system-ui, sans-serif';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(pct + '%', PAD.left - 6, yp);
  }
  ctx.setLineDash([]);

  // x-axis baseline
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.left, yFor(0)); ctx.lineTo(PAD.left + cW, yFor(0)); ctx.stroke();

  // x-axis date labels (up to 5 evenly spaced)
  const labelSet = new Set();
  const maxL = Math.min(n, 5);
  if (n === 1) { labelSet.add(0); }
  else { for (let k = 0; k < maxL; k++) labelSet.add(Math.round(k * (n - 1) / (maxL - 1))); }
  ctx.fillStyle = C.axis; ctx.font = '9px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  labelSet.forEach(i => ctx.fillText(reps[i].date_str.slice(0, 10), xFor(i), yFor(0) + 6));

  // build point coords
  const pts = reps.map((r, i) => ({ x: xFor(i), y: yFor(r.score), r }));

  // connecting line
  if (pts.length > 1) {
    ctx.strokeStyle = C.line; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();
  }

  // data points
  const radius = n > 20 ? 3.5 : 5;
  pts.forEach(p => {
    const col = p.r.type === 'fr' ? C.fr : C.scenario;
    ctx.fillStyle = col; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
  });

  // legend — only when both types are present and filter is "all"
  if (_profileFilter === 'all') {
    const hasScen = reps.some(r => r.type === 'scenario');
    const hasFr   = reps.some(r => r.type === 'fr');
    if (hasScen && hasFr) {
      const ly = PAD.top + 8;
      ctx.font = '10px system-ui, sans-serif';
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      let lx = PAD.left + cW - 148;
      ctx.fillStyle = C.scenario;
      ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = C.axis; ctx.fillText('Scenario', lx + 8, ly);
      lx += 68;
      ctx.fillStyle = C.fr;
      ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = C.axis; ctx.fillText('Free Response', lx + 8, ly);
    }
  }

  _chartPts = pts;
}

// chart hover tooltip
const _profileChart   = $('profile-chart');
const _chartTooltip   = $('chart-tooltip');

_profileChart.addEventListener('mousemove', e => {
  if (!_chartPts.length) return;
  const rect = _profileChart.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  let nearest = null, minDist = Infinity;
  _chartPts.forEach(p => {
    const d = Math.hypot(p.x - mx, p.y - my);
    if (d < minDist) { minDist = d; nearest = p; }
  });

  if (nearest && minDist < 30) {
    const cardRect = _profileChart.closest('.chart-card').getBoundingClientRect();
    let tx = e.clientX - cardRect.left + 14;
    let ty = e.clientY - cardRect.top  - 52;
    if (ty < 4) ty = e.clientY - cardRect.top + 14;
    if (tx + 220 > cardRect.width) tx = e.clientX - cardRect.left - 224;
    _chartTooltip.innerHTML     = `${nearest.r.date_str} &mdash; ${nearest.r.title}<br><strong>${nearest.r.score}%</strong>`;
    _chartTooltip.style.left    = tx + 'px';
    _chartTooltip.style.top     = ty + 'px';
    _chartTooltip.style.display = 'block';
  } else {
    _chartTooltip.style.display = 'none';
  }
});

_profileChart.addEventListener('mouseleave', () => {
  _chartTooltip.style.display = 'none';
});

// redraw chart on container resize
new ResizeObserver(() => {
  if (_profileAllReports.length) _drawChart(_filteredReports());
}).observe(_profileChart.parentElement);

// ── Theme ──────────────────────────────────────
function applyTheme(id) {
  document.documentElement.dataset.theme = id === 'light' ? '' : id;
  $('theme-select').value = id;
}

$('theme-select').addEventListener('change', function () {
  applyTheme(this.value);
  fetch('/api/save-theme', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ theme: this.value }),
  }).catch(() => {});
});

// ── Boot ───────────────────────────────────────
(function () {
  // Theme is server-rendered on <html data-theme> — just sync the selector.
  const t = document.documentElement.dataset.theme || 'light';
  $('theme-select').value = t;

  loadScenarios().catch(err => {
    $('scenario-list').textContent = `Failed to load: ${err.message}`;
  });

  // Route to the right starting view
  if (IS_ADMIN) {
    showHome();
  } else {
    showDashboard();
  }
}());

// Rustic theme: random flying logo
(function() {
  var _flyTimer = null;
  var RUST_IMG = '/static/login/rust_logo.png';
  function _isRustic() { return document.documentElement.dataset.theme === 'rustic'; }
  function _spawnFlyer() {
    if (!_isRustic()) return;
    var size = 100 + Math.random() * 900;
    var W = window.innerWidth, H = window.innerHeight;
    var edge = Math.floor(Math.random() * 4);
    var sx, sy;
    if      (edge === 0) { sx = Math.random() * W; sy = -size; }
    else if (edge === 1) { sx = W;                 sy = Math.random() * H; }
    else if (edge === 2) { sx = Math.random() * W; sy = H; }
    else                 { sx = -size;              sy = Math.random() * H; }
    var angleRanges = [[10,170],[100,260],[190,350],[280,440]];
    var r = angleRanges[edge];
    var angleDeg = r[0] + Math.random() * (r[1] - r[0]);
    var rad = angleDeg * Math.PI / 180;
    var speed = 1.5 + Math.random() * 4;
    var vx = Math.cos(rad) * speed, vy = Math.sin(rad) * speed;
    var rot = Math.random() * 360, rotSpeed = (Math.random() - 0.5) * 3;
    var el = document.createElement('div');
    el.style.cssText = 'position:fixed;pointer-events:none;z-index:9999;'
      + 'width:' + size + 'px;height:' + size + 'px;'
      + 'background:url(' + RUST_IMG + ') center/contain no-repeat;'
      + 'left:' + sx + 'px;top:' + sy + 'px;';
    document.body.appendChild(el);
    var px = sx, py = sy;
    (function tick() {
      px += vx; py += vy; rot += rotSpeed;
      el.style.left = px + 'px';
      el.style.top  = py + 'px';
      el.style.transform = 'rotate(' + rot + 'deg)';
      if (px < -(size+50) || px > W+size+50 || py < -(size+50) || py > H+size+50) { el.remove(); return; }
      requestAnimationFrame(tick);
    }());
  }
  function _scheduleNext() {
    var delay = 10000 + Math.random() * 20000;
    _flyTimer = setTimeout(function() { _spawnFlyer(); if (_isRustic()) _scheduleNext(); else _flyTimer = null; }, delay);
  }
  function _startFlyer() { if (_isRustic() && !_flyTimer) _scheduleNext(); }
  function _stopFlyer()  { if (_flyTimer) { clearTimeout(_flyTimer); _flyTimer = null; } }
  new MutationObserver(function() { if (_isRustic()) _startFlyer(); else _stopFlyer(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  _startFlyer();
})();
