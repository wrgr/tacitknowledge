"""
report_parser.py — Parse generated Markdown reports into structured dicts.
"""

import re


def parse_report_md(content):
    lines = content.split('\n')

    title_line = lines[0] if lines else ''
    report_type = 'fr' if 'Free Response' in title_line else 'scenario'

    sections = _split_sections(lines[1:])

    metadata = _parse_metadata(sections.get('_header', []))
    instructor_summary = _parse_instructor_summary(
        sections.get('Instructor Summary', [])
    )
    thinking_profile = _parse_thinking_profile(
        sections.get('Learner Thinking Profile', [])
    )

    result = {
        'type': report_type,
        'title': title_line.lstrip('# ').strip(),
        'metadata': metadata,
        'instructor_summary': instructor_summary,
        'thinking_profile': thinking_profile,
    }

    if report_type == 'fr':
        result['prompt'] = _parse_fr_prompt(sections.get('Prompt', []))
        result['submission'] = _parse_blockquote(
            sections.get("Learner's Submission", [])
        )
        result['ai_assistance'] = _parse_ai_assistance(
            sections.get('AI Assistance Declaration', [])
        )
        result['evaluation'] = _parse_fr_evaluation(
            sections.get('Evaluation', [])
        )
        result['evaluation']['pools'] = _parse_fr_pools(
            sections.get('Pools', [])
        )
        result['process_overlay'] = _parse_process_overlay(
            sections.get('Writing Process', [])
        )
    else:
        scenario_list = []
        for key, sec_lines in sections.items():
            if key.startswith('Scenario '):
                scenario_list.append(_parse_scenario(key, sec_lines))
        result['scenarios'] = sorted(scenario_list, key=lambda s: s['number'])

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_sections(lines):
    sections = {}
    current = '_header'
    buf = []
    for line in lines:
        if line.startswith('## '):
            sections[current] = buf
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = buf
    return sections


def _parse_metadata(lines):
    meta = {}
    for line in lines:
        m = re.match(r'^\*\*(.+?):\*\*\s*(.*?)(?:\s{2})?$', line.rstrip())
        if m:
            key = m.group(1).lower().replace(' ', '_')
            meta[key] = m.group(2).strip()
    return meta


def _parse_blockquote(lines):
    parts = []
    for line in lines:
        if line.startswith('> '):
            parts.append(line[2:])
        elif line == '>':
            parts.append('')
    return '\n'.join(parts).strip()


def _parse_fr_prompt(lines):
    prompt_text_lines = []
    constraints = []
    in_constraints = False
    for line in lines:
        stripped = line.strip()
        if stripped == '**Constraints:**':
            in_constraints = True
        elif in_constraints:
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                constraints.append(m.group(1).strip())
        elif stripped and stripped not in ('---',):
            prompt_text_lines.append(line)
    return {
        'text': '\n'.join(prompt_text_lines).strip(),
        'constraints': constraints,
    }


def _parse_ai_assistance(lines):
    if not any(line.strip() for line in lines):
        return None

    result = {"used": "", "notes": ""}
    notes = []
    in_notes = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**Used AI assistance:**"):
            value = stripped.split("**Used AI assistance:**", 1)[1].strip().lower()
            result["used"] = "yes" if value.startswith("yes") else "no"
            in_notes = False
        elif stripped == "**Learner description:**":
            in_notes = True
        elif in_notes and (line.startswith("> ") or line == ">"):
            notes.append(line[2:] if line.startswith("> ") else "")
    result["notes"] = "\n".join(notes).strip()
    return result


_FR_KP_BULLET = re.compile(r'^\s+-\s+(.*?)\s+—\s+_(.*)_\s*$')
_FR_KP_SPAN   = re.compile(r'^\s+>\s+"(.*)"\s*$')
_FR_KP_JUST   = re.compile(r'^\s+_Justification:_\s+(.*)$')
_FR_KP_QUALITY = re.compile(r'^\s+_Quality:_\s+(.*)$')
_FR_KP_EXEMPLAR = re.compile(r'^matched: known exemplar "(.*)"$')


def _parse_fr_evaluation(lines):
    """Parse the FR Evaluation section, including the construct/exemplar key-point
    bullets (Part D of the construct/exemplar brief): one bullet per matched point with
    a match-type tag, followed by its evidence span(s) and (for novel-equivalent matches)
    a justification line. 'matched_points'/'missed_points' stay flat construct-string
    lists for backward compatibility with older code paths (CSV export); the richer
    per-point breakdown lives in 'matched_points_detail'.

    Also tolerates the pre-brief flat "**Key points covered:** a, b, c" single-line
    format from reports generated before this change -- those reports have no per-point
    detail to recover, so matched_points_detail stays empty and callers fall back to the
    flat chip display.
    """
    ev = {
        'score': '',
        'feedback': '',
        'strengths': [],
        'gaps': [],
        'matched_points': [],
        'missed_points': [],
        'matched_points_detail': [],
        'expert_answer': '',
    }
    state = None
    expert_lines = []
    current_point = None

    def _flush_point():
        nonlocal current_point
        if current_point is not None:
            ev['matched_points_detail'].append(current_point)
            ev['matched_points'].append(current_point['construct'])
            current_point = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('**Score:**'):
            _flush_point()
            ev['score'] = stripped.split('**Score:**')[1].strip()
            state = None
        elif stripped.startswith('**Feedback for learner:**'):
            _flush_point()
            fb = re.sub(r'\*\*Feedback for learner:\*\*\s*', '', stripped)
            ev['feedback'] = fb.strip('_')
            state = None
        elif stripped == '**Strengths:**':
            _flush_point()
            state = 'strengths'
        elif stripped == '**Gaps:**':
            _flush_point()
            state = 'gaps'
        elif stripped.startswith('**Key points covered:**'):
            _flush_point()
            rest = stripped.split('**Key points covered:**', 1)[1].strip()
            if rest:
                # pre-brief flat comma-joined format -- no per-point detail available
                ev['matched_points'] = [p.strip() for p in rest.split(',') if p.strip()]
                state = None
            else:
                state = 'covered'
        elif stripped.startswith('**Key points missed:**'):
            _flush_point()
            val = stripped.split('**Key points missed:**')[1].strip()
            ev['missed_points'] = [p.strip() for p in val.split(',') if p.strip()]
            state = None
        elif stripped == '**Expert reference answer:**':
            _flush_point()
            state = 'expert'
        elif state == 'expert' and (line.startswith('> ') or line == '>'):
            expert_lines.append(line[2:] if line.startswith('> ') else '')
        elif state == 'strengths':
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                ev['strengths'].append(m.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'gaps':
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                ev['gaps'].append(m.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'covered':
            bullet = _FR_KP_BULLET.match(line)
            span   = _FR_KP_SPAN.match(line)
            just   = _FR_KP_JUST.match(line)
            if bullet:
                _flush_point()
                construct, tag = bullet.group(1).strip(), bullet.group(2).strip()
                m_ex = _FR_KP_EXEMPLAR.match(tag)
                if tag.startswith('novel equivalent'):
                    match_type, matched_exemplar = 'novel_equivalent', None
                elif m_ex:
                    match_type, matched_exemplar = 'exemplar', m_ex.group(1)
                else:
                    match_type, matched_exemplar = 'exemplar', None
                current_point = {
                    'construct': construct,
                    'match_type': match_type,
                    'matched_exemplar': matched_exemplar,
                    'evidence_spans': [],
                    'functional_justification': None,
                    'quality_label': '',
                }
            elif span and current_point is not None:
                current_point['evidence_spans'].append(span.group(1))
            elif just and current_point is not None:
                current_point['functional_justification'] = just.group(1).strip()
            elif current_point is not None and _FR_KP_QUALITY.match(line):
                current_point['quality_label'] = _FR_KP_QUALITY.match(line).group(1).strip()
            elif stripped.startswith('**'):
                _flush_point()
                state = None
    _flush_point()
    ev['expert_answer'] = '\n'.join(expert_lines).strip()
    return ev


_FR_POOL_LABEL   = re.compile(r'^\*\*(.+)\*\*$')
_FR_POOL_SUMMARY = re.compile(
    r'^Matched (\d+) of (\d+) — (\d+) required — (.+?) credit for this section\.$'
)
_FR_POOL_MISSED  = re.compile(r'^Not matched:\s*(.*)$')


def _parse_fr_pools(lines):
    """Pooled key points (choose_n_of_m brief, Part C): parse the '## Pools' section
    back into structured data. Reuses the same _FR_KP_* bullet regexes as
    _parse_fr_evaluation's flat key-points list, since reports._fr_match_bullet_lines
    renders matched members identically in both places -- only the surrounding pool
    label/summary/missed-list lines are new here.
    """
    pools = []
    current = None
    current_point = None

    def _flush_point():
        nonlocal current_point
        if current_point is not None and current is not None:
            current['matched_members'].append(current_point)
            current_point = None

    for line in lines:
        stripped = line.strip()
        label_m   = _FR_POOL_LABEL.match(stripped)
        summary_m = _FR_POOL_SUMMARY.match(stripped)
        missed_m  = _FR_POOL_MISSED.match(stripped)
        bullet    = _FR_KP_BULLET.match(line)
        span      = _FR_KP_SPAN.match(line)
        just      = _FR_KP_JUST.match(line)
        qual      = _FR_KP_QUALITY.match(line)

        if label_m:
            _flush_point()
            current = {
                'label':            label_m.group(1).strip(),
                'matched_in_pool':  0,
                'member_count':     0,
                'required_count':   0,
                'credit_label':     '',
                'matched_members':  [],
                'missed_members':   [],
            }
            pools.append(current)
        elif summary_m and current is not None:
            current['matched_in_pool'] = int(summary_m.group(1))
            current['member_count']    = int(summary_m.group(2))
            current['required_count']  = int(summary_m.group(3))
            current['credit_label']    = summary_m.group(4).strip()
        elif missed_m and current is not None:
            _flush_point()
            val = missed_m.group(1).strip()
            current['missed_members'] = [p.strip() for p in val.split(',') if p.strip()]
        elif bullet and current is not None:
            _flush_point()
            construct, tag = bullet.group(1).strip(), bullet.group(2).strip()
            m_ex = _FR_KP_EXEMPLAR.match(tag)
            if tag.startswith('novel equivalent'):
                match_type, matched_exemplar = 'novel_equivalent', None
            elif m_ex:
                match_type, matched_exemplar = 'exemplar', m_ex.group(1)
            else:
                match_type, matched_exemplar = 'exemplar', None
            current_point = {
                'construct':                construct,
                'match_type':               match_type,
                'matched_exemplar':         matched_exemplar,
                'evidence_spans':           [],
                'functional_justification': None,
                'quality_label':            '',
            }
        elif span and current_point is not None:
            current_point['evidence_spans'].append(span.group(1))
        elif just and current_point is not None:
            current_point['functional_justification'] = just.group(1).strip()
        elif qual and current_point is not None:
            current_point['quality_label'] = qual.group(1).strip()
    _flush_point()
    return pools


def _parse_process_overlay(lines):
    if not any(line.strip() for line in lines):
        return None

    overlay = {
        'quadrant_label': '',
        'quadrant_interpretation': '',
        'quadrant_alternative': '',
        'effort_profile_text': '',
        'revision_rating': '',
        'revision_evidence': [],
        'revision_alternative': '',
        'difficulty_points': [],
        'authenticity_text': '',
        'authenticity_alternatives': [],
        'confidence_calibration_text': '',
        'confidence_calibration_note': '',
        'confidence_calibration_alternative': '',
        'closing_nudge_used': None,
        'closing_nudge_text': '',
        'caution': '',
    }
    state = None
    for line in lines:
        stripped = line.strip()
        alt_match = re.match(r'^\s*>\s*Alternative:\s*(.*)', line)

        if stripped.startswith('**Process') and '×' in stripped and ':**' in stripped:
            overlay['quadrant_label'] = stripped.split(':**', 1)[1].strip()
            state = 'quadrant'
        elif stripped.startswith('**Effort profile:**'):
            overlay['effort_profile_text'] = stripped.split('**Effort profile:**', 1)[1].strip()
            state = None
        elif stripped.startswith('**Revision toward quality:**'):
            overlay['revision_rating'] = stripped.split('**Revision toward quality:**', 1)[1].strip()
            state = 'revision'
        elif stripped == '**Difficulty points:**':
            state = 'difficulty'
        elif stripped.startswith('**Authenticity:**'):
            overlay['authenticity_text'] = stripped.split('**Authenticity:**', 1)[1].strip()
            state = 'authenticity'
        elif stripped.startswith('**Confidence calibration:**'):
            overlay['confidence_calibration_text'] = stripped.split('**Confidence calibration:**', 1)[1].strip()
            state = 'confidence'
        elif stripped.startswith('**Closing nudge used:**'):
            text = stripped.split('**Closing nudge used:**', 1)[1].strip()
            overlay['closing_nudge_text'] = text
            overlay['closing_nudge_used'] = text.lower().startswith('yes')
            state = None
        elif alt_match and state == 'quadrant':
            overlay['quadrant_alternative'] = alt_match.group(1)
        elif alt_match and state == 'revision':
            overlay['revision_alternative'] = alt_match.group(1)
        elif alt_match and state == 'authenticity':
            overlay['authenticity_alternatives'].append(alt_match.group(1))
        elif alt_match and state == 'confidence':
            overlay['confidence_calibration_alternative'] = alt_match.group(1)
        elif stripped.startswith('> ') and state is None and stripped[2:] and not overlay['caution']:
            overlay['caution'] = stripped[2:]
        elif state == 'quadrant':
            m = re.match(r'^\s+>\s+(.*)', line)
            if m:
                overlay['quadrant_interpretation'] = m.group(1)
            elif stripped == '':
                state = None
        elif state == 'revision':
            m = re.match(r'^\s+-\s+"(.*)"\s*→\s*"(.*)"', line)
            if m:
                overlay['revision_evidence'].append({'before': m.group(1), 'after': m.group(2)})
            elif stripped.startswith('**'):
                state = None
        elif state == 'difficulty':
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                overlay['difficulty_points'].append(m.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'authenticity':
            if stripped == '':
                state = None
        elif state == 'confidence':
            m = re.match(r'^\s+>\s+(.*)', line)
            if m:
                overlay['confidence_calibration_note'] = m.group(1)
            elif stripped == '':
                state = None

    return overlay


def _parse_instructor_summary(lines):
    summary = {'assessment': '', 'learning_gaps': [], 'recommendations': []}
    state = None
    assessment_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '**Learning gaps identified:**':
            state = 'gaps'
        elif stripped == '**Recommendations:**':
            state = 'recs'
        elif state == 'gaps':
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                summary['learning_gaps'].append(m.group(1).strip())
        elif state == 'recs':
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                summary['recommendations'].append(m.group(1).strip())
        elif state is None and stripped not in ('---', ''):
            assessment_lines.append(line)
    summary['assessment'] = ' '.join(assessment_lines).strip()
    return summary


_FR_SOLO_INPUTS = re.compile(
    r'^\s+-\s+_matched_count:\s*(\d+),\s*mean_quality:\s*([\d.]+)_\s*$'
)


def _parse_thinking_profile(lines):
    if not any(line.strip() for line in lines):
        return None

    profile = {
        'insufficient_data_note': '',
        'honey_mumford': None,
        'solo': None,
        'patterns': [],
        'instructor_note': '',
        'probe_phase_improvement': None,
        'probe_phase_improvement_note': '',
    }
    state = None
    in_patterns = False
    hm   = {'style': '', 'confidence': '', 'evidence': [], 'reasoning': ''}
    solo = {'level': '', 'confidence': '', 'evidence': [], 'reasoning': ''}

    for line in lines:
        stripped = line.strip()
        if '**Note — limited evidence:**' in stripped:
            profile['insufficient_data_note'] = re.sub(
                r'.*\*\*Note — limited evidence:\*\*\s*', '', stripped
            )
        elif stripped.startswith('**Honey & Mumford style:**'):
            state = 'hm'; in_patterns = False
            rest = stripped[len('**Honey & Mumford style:**'):].strip()
            m = re.match(r'(.+?)\s*_\(confidence:\s*(.+?)\)_', rest)
            if m:
                hm['style'], hm['confidence'] = m.group(1).strip(), m.group(2).strip()
            else:
                hm['style'] = rest
        elif stripped.startswith('**SOLO level:**'):
            state = 'solo'; in_patterns = False
            rest = stripped[len('**SOLO level:**'):].strip()
            m = re.match(r'(.+?)\s*_\(confidence:\s*(.+?)\)_', rest)
            if m:
                solo['level'], solo['confidence'] = m.group(1).strip(), m.group(2).strip()
            else:
                solo['level'] = rest
        elif stripped.startswith('**Probe phase improvement:**'):
            state = 'ppi'; in_patterns = False
            val = stripped[len('**Probe phase improvement:**'):].strip()
            profile['probe_phase_improvement'] = (val.lower() == 'yes')
        elif stripped == '**Observed patterns:**':
            state = None; in_patterns = True
        elif stripped.startswith('**Instructor note:**'):
            state = None; in_patterns = False
            note = re.sub(r'\*\*Instructor note:\*\*\s*', '', stripped)
            profile['instructor_note'] = note.strip('_')
        elif state == 'hm':
            m_ev = re.match(r'^\s+-\s+_"(.+)"_', line)
            m_rs = re.match(r'^\s+>\s+(.*)', line)
            if m_ev:
                hm['evidence'].append(m_ev.group(1))
            elif m_rs:
                hm['reasoning'] = m_rs.group(1)
        elif state == 'ppi':
            m_rs = re.match(r'^\s+>\s+(.*)', line)
            if m_rs:
                profile['probe_phase_improvement_note'] = m_rs.group(1)
            elif stripped.startswith('**'):
                state = None
        elif state == 'solo':
            m_ev = re.match(r'^\s+-\s+_"(.+)"_', line)
            m_in = _FR_SOLO_INPUTS.match(line)
            m_rs = re.match(r'^\s+>\s+(.*)', line)
            if m_ev:
                solo['evidence'].append(m_ev.group(1))
            elif m_in:
                solo['matched_count'] = int(m_in.group(1))
                solo['mean_quality']  = float(m_in.group(2))
            elif m_rs:
                solo['reasoning'] = m_rs.group(1)
        elif in_patterns:
            m = re.match(r'^\s+-\s+(.*)', line)
            if m:
                profile['patterns'].append(m.group(1).strip())
            elif stripped.startswith('**') and not stripped.startswith('**Observed'):
                in_patterns = False

    if hm['style']:
        profile['honey_mumford'] = hm
    if solo['level']:
        profile['solo'] = solo

    if not profile['honey_mumford'] and not profile['solo']:
        return None

    return profile


def _parse_scenario(header, lines):
    m = re.match(r'Scenario (\d+):\s*(.*)', header)
    number = int(m.group(1)) if m else 0
    title  = m.group(2).strip() if m else header

    scenario = {
        'number': number,
        'title': title,
        'description': '',
        'constraints': [],
        'coverage_score': '',
        'quality_score': '',
        'score': '',
        'transcript': '',
        'recall_transcript': '',
        'probe_transcript': '',
        'expert_guidance': '',
        'strengths': [],
        'gaps': [],
        'matched_points': [],
        'missed_points': [],
        'feedback': '',
    }
    state = None
    transcript_lines = []
    recall_lines = []
    probe_lines  = []
    expert_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('_') and stripped.endswith('_') and not state and not scenario['description']:
            scenario['description'] = stripped.strip('_')
        elif stripped == '**Constraints:**':
            state = 'constraints'
        elif stripped.startswith('**Coverage Score:**'):
            scenario['coverage_score'] = stripped.split('**Coverage Score:**')[1].strip()
            state = None
        elif stripped.startswith('**Explanation Quality Score:**'):
            scenario['quality_score'] = stripped.split('**Explanation Quality Score:**')[1].strip()
            state = None
        elif stripped.startswith('**Overall Score:**'):
            scenario['score'] = stripped.split('**Overall Score:**')[1].strip()
            state = None
        # legacy single score line
        elif stripped.startswith('**Score:**') and not scenario['score']:
            scenario['score'] = stripped.split('**Score:**')[1].strip()
            state = None
        elif stripped == '**Recall transcript (free recall phase):**':
            state = 'recall'
        elif stripped == '**Probing Phase transcript:**':
            state = 'probe'
        elif stripped == '**Conversation transcript:**':
            state = 'transcript'
        elif stripped == '**Expert guidance:**':
            state = 'expert'
        elif stripped == '**Strengths:**':
            state = 'strengths'
        elif stripped == '**Gaps:**':
            state = 'gaps'
        elif stripped.startswith('**Key points covered:**'):
            state = 'matched'
        elif stripped.startswith('**Key points missed:**'):
            val = stripped.split('**Key points missed:**')[1].strip()
            scenario['missed_points'] = [p.strip() for p in val.split(',') if p.strip()]
            state = None
        elif stripped.startswith('**Feedback for learner:**'):
            fb = re.sub(r'\*\*Feedback for learner:\*\*\s*', '', stripped)
            scenario['feedback'] = fb.strip('_')
            state = None
        elif state == 'constraints':
            m2 = re.match(r'^\s+-\s+(.*)', line)
            if m2:
                scenario['constraints'].append(m2.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'recall':
            if line.startswith('> ') or line == '>':
                recall_lines.append(line[2:] if line.startswith('> ') else '')
            elif stripped.startswith('**'):
                state = None
        elif state == 'probe':
            if line.startswith('> ') or line == '>':
                probe_lines.append(line[2:] if line.startswith('> ') else '')
            elif stripped.startswith('**'):
                state = None
        elif state == 'transcript':
            if line.startswith('> ') or line == '>':
                transcript_lines.append(line[2:] if line.startswith('> ') else '')
            elif stripped.startswith('**'):
                state = None
        elif state == 'expert':
            if line.startswith('> ') or line == '>':
                expert_lines.append(line[2:] if line.startswith('> ') else '')
            elif stripped.startswith('**'):
                state = None
        elif state == 'strengths':
            m2 = re.match(r'^\s+-\s+(.*)', line)
            if m2:
                scenario['strengths'].append(m2.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'gaps':
            m2 = re.match(r'^\s+-\s+(.*)', line)
            if m2:
                scenario['gaps'].append(m2.group(1).strip())
            elif stripped.startswith('**'):
                state = None
        elif state == 'matched':
            m2 = re.match(r'^\s+-\s+(.*)', line)
            if m2:
                pt = m2.group(1).split('—')[0].strip()
                if pt:
                    scenario['matched_points'].append(pt)
            elif stripped.startswith('**'):
                state = None

    scenario['transcript']       = '\n'.join(transcript_lines).strip()
    scenario['recall_transcript']= '\n'.join(recall_lines).strip()
    scenario['probe_transcript'] = '\n'.join(probe_lines).strip()
    scenario['expert_guidance']  = '\n'.join(expert_lines).strip()

    # fallback: if only old-style score was set, put it in the right slot
    if not scenario['score'] and scenario.get('coverage_score'):
        scenario['score'] = scenario['coverage_score']

    return scenario
