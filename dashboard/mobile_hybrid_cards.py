"""Presentation-only adapters for the eight mobile reports. Never refresh or score data."""
import json
import math
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

SOURCES = {'signal-wall':'signals.json','velocity-decay':'velocity_decay_monitor.json','stuff-disruption':'stuff_disruption_feed.json','ivb-heat-map':'ivb_heat_map.json','apex-extraction':'apex-extraction/apex_extraction.json','mlb-extraction':'hidden-gems/mlb_extraction_ledger.json','waiver-wire':'waiver_wire.json','kinetic-drift':'admin/kinetic_drift_signals.json'}

TEMPLATES = Path(__file__).parent / 'templates' / 'mobile' / 'surface_reports'
ENV = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(default=True))

def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None

def fmt(value, unit='', signed=False, digits=1):
    n = number(value)
    return 'Not retained' if n is None else f'{n:+.{digits}f}{unit}' if signed else f'{n:.{digits}f}{unit}'

def evidence(label, value, context=''):
    return {'label': label, 'value': value, 'context': context}

def direction(value, label):
    n = number(value)
    if n is None:
        return None
    return f'{label} {"rose" if n > 0 else "fell" if n < 0 else "held steady"}'

def adapt(row, family):
    p = dict(row)
    name = p.get('player_name') or p.get('name') or 'Unknown player'
    if ',' in name:
        last, first = name.split(',', 1); name = f'{first.strip()} {last.strip()}'
    pid = str(p.get('player_id') or '')
    kind = p.get('player_type') or p.get('kind') or ('hitter' if p.get('role') == 'BAT' else 'pitcher')
    role = p.get('position') or p.get('role') or kind
    s = dict(name=name, team=p.get('team') or 'Team not supplied', role=role,
             player_id=pid, kind=kind, image=p.get('headshot_url') or (f'https://img.mlbstatic.com/mlb-photos/image/upload/w_360,q_90/v1/people/{pid}/headshot/67/current' if pid else ''),
             profile=p.get('profile_url') or (f'/scout/{pid}/' if pid else '#'),
             changed='The retained report does not include a quantified change.', why='', watch='', evidence=[],
             score_label='', score='', score_note='', tags=[], detail=[], context='')
    if family == 'signal-wall':
        hitter = kind == 'hitter'; badges = p.get('badges') or []
        s.update(score_label='EDGE SCORE', score=p.get('edge_score'), tags=badges,
                 score_note='High-priority signal' if (number(p.get('edge_score')) or 0) >= 80 else 'Signal strength')
        if hitter:
            ev = 'EV Burst' in badges; barrel = 'Barrel Jump' in badges
            s['changed'] = ('Exit velocity and barrel quality have both jumped.' if ev and barrel else
                            'Average exit velocity has risen.' if ev else 'Barrel-like contact has become more frequent.' if barrel else
                            'Recent contact quality stands out in the signal ranking.')
            s['why'] = 'Stronger contact quality can lead to more hard contact and extra-base results.'
            s['watch'] = 'Does the stronger contact quality persist over the next several games?'
            match = re.search(r'\(([+-][\d.]+) vs baseline\)', str(p.get('why') or ''))
            delta = f'{match.group(1)} mph vs earlier baseline' if match else 'Recent sample'
            s['evidence'] = [evidence('Avg exit velocity', fmt(p.get('metric_1'),' mph'),delta),
                             evidence('Barrel-like rate',fmt(p.get('metric_2'),'%'),'EV ≥98 mph; launch angle 26–30°'),
                             evidence('Apex Damage',fmt(p.get('metric_3'),' mph'),'Maximum exit velocity')]
        else:
            changes = [text for badge,text in [('Whiff Lift','swinging strikes are more frequent'),('Velo Jump','fastball velocity has risen'),('Extension Gain','release extension has increased')] if badge in badges]
            s['changed'] = ('; '.join(changes).capitalize()+'.') if changes else 'Recent bat-missing ability and fastball traits stand out in the signal ranking.'
            s['why'] = 'Changes in speed, release and bat-missing ability can affect strikeouts and contact quality.'
            s['watch'] = 'Do these pitch traits persist, with swinging strikes and contact results moving alongside them?'
            s['evidence'] = [evidence('Swinging-strike rate',fmt(p.get('metric_1'),'%'),'Share of all pitches; not whiffs per swing'),evidence('Fastball velocity',fmt(p.get('metric_2'),' mph'),'Recent sample'),evidence('Release extension',str(p.get('metric_3','Not retained')),'Feet from the rubber')]
        s['context'] = 'Recent 7-day sample; comparison uses the earlier portion of the 28-day window.'
    elif family == 'velocity-decay':
        d = number(p.get('velo_delta')); trend=p.get('trend_values') or []
        s.update(changed=(f'Fastball velocity is {abs(d):.1f} mph {"below" if d < 0 else "above"} the prior-appearance average.' if d is not None else 'The retained velocity comparison is unavailable.'),
                 why='Sustained velocity loss can change how the arsenal plays and may reflect workload or changing effectiveness.',
                 watch='Does velocity rebound, stabilize or continue falling in the next appearances?',
                 score_label='RISK SCORE',score=p.get('risk_score_label'),tags=[p.get('risk_tier'),p.get('primary_alert')],
                 context='Latest fastball appearance versus the average of up to four prior appearances in the 30-day window.')
        s['evidence']=[evidence('Latest velocity',fmt(trend[0] if trend else None,' mph'),'Latest retained appearance'),evidence('Velocity difference',fmt(d,' mph',True),'vs prior-appearance average'),evidence('Trend sample',str(len(trend))+' appearances','Sample size; not proof of persistent decline')]
        s['detail']=[evidence('Extension difference',p.get('extension_delta_label')),evidence('Perceived velocity proxy',p.get('perceived_delta_label')),evidence('Decay classification',p.get('decay_slope_label')),evidence('Velocity trace (latest first)',', '.join(fmt(v,' mph') for v in trend))]
    elif family == 'stuff-disruption':
        parts=[direction(p.get('ivb_delta'),'fastball vertical break'),direction(p.get('vaa_delta'),'approach angle')]
        s.update(changed='; '.join(x for x in parts if x).capitalize()+' versus the prior-appearance average.',
                 why='Pitch shape can strengthen or weaken before the change becomes obvious in strikeouts or contact results.',
                 watch='Does the shape change persist, and do swinging strikes and contact outcomes begin moving with it?',
                 score_label='DISRUPTION SCORE',score=p.get('disruption_score_label'),tags=[p.get('apex_tier'),p.get('primary_alert')],
                 context='Latest fastball appearance versus up to four prior appearances in the 30-day window. This report retains shape measurements, not a numeric Stuff+ rating or a ranked individual pitch.')
        s['evidence']=[evidence('Vertical break change',fmt(p.get('ivb_delta'),' in',True),'vs prior fastball appearances'),evidence('Approach angle change',fmt(p.get('vaa_delta'),'°',True),'vs prior fastball appearances'),evidence('Horizontal movement',fmt(p.get('movement_delta'),' in'),'Magnitude of change; not a direction')]
        s['detail']=[evidence('Active-spin proxy change',fmt(p.get('active_spin_delta'),digits=3)),evidence('IVB trace (latest first)',', '.join(fmt(v,' in') for v in p.get('trend_values',[])))]
    elif family == 'ivb-heat-map':
        value=str(p.get('ivb_vs_avg') or ''); match=re.search(r'[+-]?\d+(?:\.\d+)?',value); d=number(match.group()) if match else None
        s.update(changed=(f'Fastball vertical break is {abs(d):.1f} inches {"above" if d>=0 else "below"} the velocity-peer baseline.' if d is not None else 'Fastball shape is surfaced against its velocity-peer baseline; the difference is not retained.'),
                 why='An unusually distinct fastball shape can affect how the pitch plays against hitters.',watch='Does this fastball shape persist in subsequent appearances?',
                 tags=[p.get('band_label'),p.get('transition_badge')],
                 context='IVB is induced vertical break. This comparison is against velocity peers, not the player’s prior outing.')
        s['evidence']=[evidence('Current IVB',p.get('ivb_raw','Not retained'),'Fastball sample'),evidence('IVB vs velocity peers',p.get('ivb_vs_avg','Not retained'),'Velocity-bucket baseline'),evidence('Velocity context',p.get('velocity_bucket') or 'Not retained','Fastball speed band')]
        if not p.get('velocity_bucket'): s['evidence'] = s['evidence'][:2]
        s['detail']=[evidence('Velocity context',p.get('velocity_bucket') or 'Not retained'),evidence('Vertical approach angle',p.get('vaa')),evidence('Dead-zone status',p.get('dead_zone_label'))]
    elif family == 'apex-extraction':
        arm=p.get('signal_family')=='APEX ARM'; ms={m.get('code'):m for m in p.get('forensic_metrics',[])}
        triggered=[label for key,label in [('physical_shift','physical traits'),('vision_delta','pitch deception' if arm else 'contact/decision traits'),('market_latency','the model’s results-gap proxy')] if p.get(key)]
        s.update(changed=('This profile combines '+', '.join(triggered)+'.') if triggered else 'This profile combines the retained physical and supporting signal measurements.',
                 why='Several supporting measurements can make a player worth watching beyond any single statistic.',
                 watch='Does this combination persist and translate into sustained performance?',score_label='APEX SCORE',score=p.get('apex_score'),tags=[p.get('signal_family'),p.get('verdict')],
                 context='These are model components, which can share inputs. Results-gap and market-latency proxies do not establish measured ownership or market movement.')
        if arm:
            support=str(p.get('supporting_metric') or ''); m=re.search(r'iVB Delta ([+-]?[\d.]+)',support)
            if m:s['evidence'].append(evidence('Vertical break change',m.group(1)+' in','vs prior fastball appearances'))
            for code,label,ctx in [('VAA_DELTA','Approach angle change','vs prior fastball appearances'),('SSW_PROXY','Horizontal movement','Magnitude of change; causality unproven')]:
                if code in ms:s['evidence'].append(evidence(label,ms[code].get('value'),ctx))
        else:
            for code,label,ctx in [('DHH_PROXY','Maximum exit velocity','Peak contact speed'),('LA_CONSISTENCY','Launch angle','Recent mean, or model classification'),('HIGH_STAKES_DELTA','xBA minus AVG','Expected vs actual batting average; not a historical change')]:
                if code in ms:s['evidence'].append(evidence(label,ms[code].get('value'),ctx))
        s['detail']=[evidence('Physical component',p.get('physical_shift_score')),evidence('Vision/deception component',p.get('vision_delta_score')),evidence('Results-gap proxy component',p.get('market_latency_score')),evidence('Supporting measurements',p.get('supporting_metric'))]
    elif family == 'mlb-extraction':
        r=p.get('raw') or {}; hitter=kind=='hitter'
        if number(r.get('recent_ev' if hitter else 'recent_whiff_rate')) is not None:
            key='ev_delta' if hitter else 'whiff_delta'; d=number(r.get(key)); label='Average exit velocity' if hitter else 'Swinging-strike rate'
            s['changed']=(f'{label} {"rose" if d>0 else "fell" if d<0 else "held steady"} versus the earlier baseline.' if d is not None else f'{label} stands out in the recent sample.')
            if hitter:
                s['evidence']=[evidence('Avg exit velocity',fmt(r.get('recent_ev'),' mph'),'Recent sample'),evidence('Exit velocity change',fmt(d,' mph',True),'vs earlier baseline'),evidence('Barrel-like rate',fmt(number(r.get('recent_barrel_rate'))*100 if number(r.get('recent_barrel_rate')) is not None else None,'%'),'EV ≥98 mph; launch angle 26–30°')]
            else:
                s['evidence']=[evidence('Swinging-strike rate',fmt(number(r.get('recent_whiff_rate'))*100,'%'),'Share of all pitches'),evidence('Rate change',fmt(d*100 if d is not None else None,' pp',True),'Percentage points vs earlier baseline'),evidence('Fastball velocity',fmt(r.get('recent_fb_velo'),' mph'),'Recent sample')]
            s['context']='Recent 7-day sample versus the earlier portion of the source window. Market attention is unavailable unless separately supplied.'
        else:
            s['changed']='Underlying traits and surface results differ in the retained ledger ranking.'
            s['evidence']=[evidence(label,fmt(r.get(key),digits=2),'Model component; not a physical unit') for key,label in [('trait_score_raw','Underlying-trait score'),('surface_pressure_raw','Surface-pressure score'),('market_score_raw','Market-context score')] if number(r.get(key)) is not None]
            s['context']='Ledger-model scores summarize traits, surface results and available market context. They do not establish a change from the prior outing.'
        s.update(why='Underlying pitch or contact traits can help put recent results in context; the signal does not guarantee future production.',watch='Does the underlying trait persist, and do strikeout or contact results begin to reflect it?',score_label='EDGE SCORE',score=p.get('score'),tags=[p.get('diagnosis')])
        s['detail']=[evidence('Source',r.get('source_badge')),evidence('Sample',r.get('sample_note')),evidence('Market attention',r.get('metric_3'))]
    elif family == 'waiver-wire':
        s.update(changed=p.get('forensic_trigger') or 'The candidate feed has not supplied a performance or opportunity change.',
                 why='A supported skill or opportunity change may matter to a fantasy roster when league availability and playing time align.',
                 watch='Does playing time hold or grow, and is the player still available under your league’s rules?',
                 score_label='WAIVER SCORE',score=p.get('waiver_score'),tags=[p.get('market_status')],
                 context='Eligibility and role are context, not a guarantee of fantasy value. Confirm availability in your league.')
        s['evidence']=[evidence('Availability',p.get('ownership_gate'),'Retained eligibility gate'),evidence('Signal window',p.get('signal_window'),'Source window'),evidence('Opportunity status',p.get('deployment_label'),'Retained candidate status')]
    elif family == 'kinetic-drift':
        m=p.get('metrics') or {}; definitions=[('release_speed_delta','Fastball velocity',' mph'),('release_extension_delta','Release extension',' ft'),('ivb_delta','Vertical break',' in'),('release_pos_x_delta','Horizontal release',' ft'),('release_pos_z_delta','Vertical release',' ft'),('spin_delta','Spin',' rpm'),('hb_delta','Horizontal break',' in')]
        available=[(k,l,u) for k,l,u in definitions if number(m.get(k)) is not None]
        # Prefer supported physical measurements; all retained values remain in the depth panel.
        s['evidence']=[evidence(label,fmt(m[key],unit,True),'vs own prior-appearance baseline') for key,label,unit in available[:3]]
        phrases=[direction(m[key],label.lower()) for key,label,unit in available[:2]]
        s.update(changed=('; '.join(phrases).capitalize()+' versus his own baseline.') if phrases else 'The engine flagged delivery variation; component changes are not retained.',
                 why='Changes in release, speed or shape can alter how pitches play. The measurements alone do not establish fatigue, injury or another cause.',
                 watch='Does the delivery shift persist or reverse in the next appearances, with corresponding changes in pitch results?',
                 score_label='KDE SCORE',score=p.get('kde_score'),tags=[p.get('movement_state_label'),p.get('kde_band')],
                 context=f"Recent {p.get('recent_appearances','unretained')} appearances versus {p.get('baseline_appearances','unretained')} prior appearances.")
        s['detail']=[evidence(label,fmt(m[key],unit,True)) for key,label,unit in available]+[evidence('Risk component',p.get('kinetic_risk_score')),evidence('Emergence component',p.get('kinetic_emergence_score')),evidence('Instability component',p.get('kinetic_instability_score'))]
        if p.get('drift_trace'):s['detail'].append(evidence('Drift index trace (earliest first)', ' → '.join(str(x.get('game_date','')) + ': ' + str(x.get('drift_index','—')) for x in p['drift_trace'])))
    if family == 'apex-extraction':
        for item in s['evidence']:
            if item['label'] == 'Launch angle' and item['value'] == 'Surgical':
                item['value'] = '15–25° band'
                item['context'] = 'Range indicated by source classification'
        s['changed'] = ('Changes in fastball carry, approach angle and sideways movement are surfacing together.' if arm else 'Peak contact quality and expected-versus-actual hitting results stand out together.')
    s['evidence']=s['evidence'][:3]
    s['tags']=[t for t in s['tags'] if t]
    s['detail']=[m for m in s['detail'] if m['value'] is not None and m['value']!='']
    p['story']=s
    return p

def template_for(family):
    class MobileTemplate:
        def __init__(self, source):
            self.template=ENV.from_string('{% from "_hybrid_card.html" import hybrid_card %}'+source)
        def render(self, **context):
            context['players']=[adapt(row,family) for row in context.get('players',[])]
            source = SOURCES[family]
            payload = json.loads((Path(__file__).resolve().parents[1] / 'dist' / source).read_text())
            for player in context['players']:
                player['story']['source'] = source
                player['story']['generated_at'] = payload.get('source_updated_at') or payload.get('generated_at') or 'Not retained'
            return self.template.render(**context)
    return MobileTemplate
