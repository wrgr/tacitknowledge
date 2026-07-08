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

const AUTO_RUN_RESPONSES = {
  gibberish: {
    recall: 'The purple elephant carefully rotates seventeen times before the kitchen sink downloads a bicycle. ' +
            'Clouds taste like Thursday when the calculator forgets to sneeze. ' +
            'I would definitely consider the implications of this process going forward in a meaningful way.',
    probe:  'Yes, exactly, that is what I would do in this situation. The thing about it is that it depends ' +
            'on various factors which I have already considered at this point in time.',
  },
  poor: {
    recall: 'I would pull over and stop the car. Then I would get the spare tyre out and put it on. ' +
            'After that I would drive away carefully and hope for the best.',
    probe:  "I'm not really sure about that specific part. I would just do what seems right at the time.",
  },
  alright: {
    recall: 'I would put on the hazard lights and pull onto the hard shoulder. Apply the handbrake. ' +
            'Get the spare tyre and jack from the boot. Jack up the car on a solid point, remove the flat, ' +
            'fit the spare and tighten the wheel nuts. Lower the car and drive off, ' +
            'making sure to check the tyre pressure when I can.',
    probe:  'I would do that step carefully to make sure everything is safe and secure. ' +
            'It is important to follow the correct order so the car stays stable throughout.',
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
  const recallText = responses.recall || S.debugExpertAnswer;
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
