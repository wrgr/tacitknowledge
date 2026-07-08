// index-admin.js — admin-only debug tooling (auto-run scenario, FR auto-fill).
// Loaded only for admin sessions; all elements it wires are admin-gated in the template.
// @ts-nocheck
// ── Debug: Auto-run scenario (admin only) ──────

// Toggle the mode-picker popup
$('btn-debug-autorun').addEventListener('click', (e) => {
  e.stopPropagation();
  if (S.busy || S.phase !== 'recall' || !S.debugExpertAnswer) return;
  const menu = $('autorun-menu');
  menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
});
// Hover highlight on options
document.querySelectorAll('.autorun-opt').forEach(btn => {
  btn.addEventListener('mouseenter', () => btn.style.background = 'rgba(245,158,11,.15)');
  btn.addEventListener('mouseleave', () => btn.style.background = 'transparent');
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    $('autorun-menu').style.display = 'none';
    autoRunScenario(btn.dataset.mode);
  });
});
// Close menu on outside click
document.addEventListener('click', () => { $('autorun-menu').style.display = 'none'; });

// ── TESTING PURPOSES ONLY: "TEST CASES" reference dropdown (admin only) ──
// Plain display list of the current scenario's expert key points, for manual
// testing reference. Purely informational — items are not clickable/selectable.
$('test-cases-toggle').addEventListener('click', (e) => {
  e.stopPropagation();
  const menu = $('test-cases-menu');
  menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
});
document.addEventListener('click', () => { $('test-cases-menu').style.display = 'none'; });

// Recall text for 'poor' and 'alright' is generated from the current scenario's
// own key points (S.debugKeyPoints, populated from debug_key_points in /api/start)
// rather than hardcoded per-scenario, so Auto-run adapts to any scenario JSON that
// follows the expert_answers[].key_points structure (e.g. Changing Tire, CPR) — not
// just the one it was originally written against.
function buildKeyPointRecall(mode) {
  const points = (S.debugKeyPoints || []).map(kp => kp.point).filter(Boolean);
  if (!points.length) return null;

  if (mode === 'poor') {
    const mentioned = points.slice(0, Math.max(1, Math.ceil(points.length * 0.25)));
    return `I would try to ${mentioned.join(' and ')}. Then I would just do whatever seems right ` +
           'at the time and hope for the best.';
  }
  if (mode === 'alright') {
    return `I would make sure to ${points.join(', ')}. ` +
           "I'd try to do each part carefully and in a reasonable order, though I might not get every detail exactly right.";
  }
  return null;
}

const AUTO_RUN_RESPONSES = {
  gibberish: {
    recall: 'The purple elephant carefully rotates seventeen times before the kitchen sink downloads a bicycle. ' +
            'Clouds taste like Thursday when the calculator forgets to sneeze. ' +
            'I would definitely consider the implications of this process going forward in a meaningful way.',
    probe:  'Yes, exactly, that is what I would do in this situation. The thing about it is that it depends ' +
            'on various factors which I have already considered at this point in time.',
  },
  poor: {
    recall: null,   // filled at runtime from the scenario's own key points
    probe:  "I'm not really sure about that specific part. I would just do what seems right at the time.",
  },
  alright: {
    recall: null,   // filled at runtime from the scenario's own key points
    probe:  'I would do that step carefully to make sure everything is safe and secure. ' +
            'It is important to follow the correct order so the outcome remains safe and correct throughout.',
  },
  ace: {
    recall: null,   // filled at runtime from S.debugExpertAnswer
    probe:  'I would carry out this step carefully because it is essential to the correct outcome. ' +
            'The reason this matters is that skipping it could cause problems or safety issues. ' +
            'The sequence is important because each step depends on the previous one being done correctly.',
  },
};

async function autoRunScenario(mode) {
  if (S.busy || S.phase !== 'recall' || !S.debugExpertAnswer) return;

  const responses = AUTO_RUN_RESPONSES[mode] || AUTO_RUN_RESPONSES.ace;
  const recallText = responses.recall || buildKeyPointRecall(mode) || S.debugExpertAnswer;
  const probeText  = responses.probe;

  $('autorun-wrapper').style.display = 'none';
  lock(true);

  try {
    // Phase 1: send the chosen recall response
    addBubble('user', recallText);
    showTyping();
    const recallResp = await api('/api/respond', {
      session_id: S.sessionId,
      user_input: recallText,
    });
    hideTyping();
    S.turn++;
    if (recallResp.narration) addBubble('examiner', recallResp.narration);

    // Phase 2: end recall and receive first probe (or go straight to evaluate)
    showTyping();
    const endResp = await api('/api/end-recall', {
      session_id: S.sessionId,
      final_text: null,
    });
    hideTyping();

    if (endResp.concluded) {
      lock(false);
      await evaluate();
      return;
    }

    S.phase       = endResp.phase       || 'probing';
    S.probeCount  = endResp.probe_count  || 0;
    S.probeNumber = endResp.probe_number || 1;
    show('probing-banner');
    hide('btn-done');
    updateMeta();
    if (endResp.first_probe) addBubble('examiner', endResp.first_probe);

    // Phase 3: answer every probe with the mode's canned response
    while (S.phase === 'probing') {
      await new Promise(r => setTimeout(r, 400));
      addBubble('user', probeText);
      showTyping();
      const probeResp = await api('/api/respond', {
        session_id: S.sessionId,
        user_input: probeText,
      });
      hideTyping();
      S.turn++;
      if (probeResp.phase) {
        S.phase = probeResp.phase;
        if (probeResp.probe_count)  S.probeCount  = probeResp.probe_count;
        if (probeResp.probe_number) S.probeNumber = probeResp.probe_number;
      }
      updateMeta();
      if (probeResp.narration) addBubble('examiner', probeResp.narration);
      if (probeResp.concluded) {
        lock(false);
        await evaluate();
        return;
      }
    }

    lock(false);
  } catch (err) {
    hideTyping();
    addBubble('examiner', `⚠ Auto-run error: ${err.message}`);
    lock(false);
    $('autorun-wrapper').style.display = 'inline-block';
  }
}

// ── Debug: Auto-submit free response (admin only) ─
$('btn-fr-debug-autofill').addEventListener('click', autoFillFr);

async function autoFillFr() {
  const answer = FR.promptData && FR.promptData.expert_answer;
  if (!answer) return;
  const ta = $('fr-textarea');
  ta.value = answer;
  ta.dispatchEvent(new Event('input'));
  // let word-count and key-point checks fire before submitting
  await new Promise(r => setTimeout(r, 900));
  $('btn-fr-submit').click();
}
