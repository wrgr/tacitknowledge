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
        result['evaluation'] = _parse_fr_evaluation(
            sections.get('Evaluation', [])
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


def _parse_fr_evaluation(lines):
    ev = {
        'score': '',
        'feedback': '',
        'strengths': [],
        'gaps': [],
        'matched_points': [],
        'missed_points': [],
        'expert_answer': '',
    }
    state = None
    expert_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('**Score:**'):
            ev['score'] = stripped.split('**Score:**')[1].strip()
            state = None
        elif stripped.startswith('**Feedback for learner:**'):
            fb = re.sub(r'\*\*Feedback for learner:\*\*\s*', '', stripped)
            ev['feedback'] = fb.strip('_')
            state = None
        elif stripped == '**Strengths:**':
            state = 'strengths'
        elif stripped == '**Gaps:**':
            state = 'gaps'
        elif stripped.startswith('**Key points covered:**'):
            val = stripped.split('**Key points covered:**')[1].strip()
            ev['matched_points'] = [p.strip() for p in val.split(',') if p.strip()]
            state = None
        elif stripped.startswith('**Key points missed:**'):
            val = stripped.split('**Key points missed:**')[1].strip()
            ev['missed_points'] = [p.strip() for p in val.split(',') if p.strip()]
            state = None
        elif stripped == '**Expert reference answer:**':
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
    ev['expert_answer'] = '\n'.join(expert_lines).strip()
    return ev


def _parse_process_overlay(lines):
    if not any(line.strip() for line in lines):
        return None

    overlay = {
        'quadrant_label': '',
        'quadrant_interpretation': '',
        'effort_profile_text': '',
        'revision_rating': '',
        'revision_evidence': [],
        'difficulty_points': [],
        'authenticity_text': '',
        'caution': '',
    }
    state = None
    for line in lines:
        stripped = line.strip()
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
            state = None
        elif stripped.startswith('> ') and state is None and stripped[2:] and not overlay['caution']:
            overlay['caution'] = stripped[2:]
        elif state == 'quadrant':
            m = re.match(r'^\s+>\s+(.*)', line)
            if m:
                overlay['quadrant_interpretation'] = m.group(1)
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
            m_rs = re.match(r'^\s+>\s+(.*)', line)
            if m_ev:
                solo['evidence'].append(m_ev.group(1))
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
