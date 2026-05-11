# Kinetic Drift Engine V1

Branch: kinetic-drift-engine-v1  
Status: isolated preview branch, not production-wired.

## Current Scope
- Standalone builder: dashboard/build_kinetic_drift.py
- Standalone template: dashboard/templates/admin/kinetic_drift.html
- Local preview output: dist/admin/kinetic-drift/index.html
- JSON output: dist/admin/kinetic_drift_signals.json

## Current Engine Outputs
- KDE score
- KDE band
- Kinetic Risk Score
- Kinetic Emergence Score
- Kinetic Instability Score
- Confidence score
- Movement state
- Diagnosis
- Operator action
- Drift trace preview

## Production Isolation
This branch is intentionally not wired into:
- dashboard/build_all.py
- netlify.toml
- production routes
- live Signal Wall pages

## Next Refinements
- Real drift waveform rendering
- Release-point fragmentation visual
- Per-pitcher movement chronology
- Better diagnosis calibration for instability cases
- Risk/opportunity color refinement
- Player drilldown concept
- Later Signal Wall integration review

## V2 Calibration Pass
Merged to main: 6144f91

### What changed
- KDE band thresholds were raised:
  - Extreme: 90+
  - Major: 75+
  - Actionable: 60+
  - Early: 45+
- Risk, emergence, and instability formulas were softened to reduce over-amplification.
- Low-signal no-drift filter moved from 40 to 45.

### Result
- Cleaner distribution.
- Fewer false “extreme” labels.
- Better separation between true anomalies, actionable drift, and early movement signals.

### Still isolated
Kinetic Drift remains intentionally disconnected from:
- dashboard/build_all.py
- netlify.toml
- production functions
- live Signal Wall routing

## V3 Waveform UI Preview — kinetic-drift-v3-waveform-ui

### Current status
Kinetic Drift Engine now has a visual waveform proof module in the admin preview.

KDE remains isolated and experimental. It is not production-wired.

### V3 additions
- Added `drift_trace` JSON to each KDE signal.
- Added 3-outing SVG waveform renderer.
- Added latest-point pulse on waveform.
- Added waveform color system:
  - Red = decay / fatigue risk
  - Lime = emergence / power gain
  - Cyan = mechanical instability
- Added page-level explanation that color and slope mean different things:
  - Color = dominant signal family / diagnosis
  - Slope = recent 3-outing signal behavior
- Added forensic readout line under each waveform.
- Added proprietary DiamondSignals research surface notation.
- Added Kinetic Drift Engine © KDE branding language.

### Interpretation model
The waveform should not be read as a simple red/green/blue chart.

Color answers:
"What kind of kinetic signal is this?"

Slope answers:
"Is that signal accelerating, cooling, or holding across the last three outings?"

Examples:
- Red + rising = decay/fatigue risk intensifying
- Red + falling = prior decay event cooling, but still in the risk family
- Lime + rising = emergence/power-gain signal strengthening
- Lime + falling = emergence signal cooling
- Cyan + level = mechanical instability holding, not necessarily safe

### Branding note
The current page should use:
Kinetic Drift Engine © KDE

Avoid using "Hawking" in the visible product copy for now. The public-facing terminology should stay closer to "Synthetic Kinetic Biometrics," "inferred-movement intelligence," and "KDE."

### Still isolated
KDE remains intentionally disconnected from:
- dashboard/build_all.py
- netlify.toml
- production functions
- live Signal Wall routing

Production wiring comes later only after the waveform, diagnosis logic, UI hierarchy, and explanation layer are reviewed further.

## V4 Diagnosis Logic Preview — kinetic-drift-v4-diagnosis-logic

### Current status
KDE V4 adds diagnosis alignment logic on top of the V3 waveform preview.

KDE remains isolated and experimental. It is not production-wired.

### V4 additions
- Added `trace_behavior` classification:
  - ACCELERATING
  - COOLING
  - HOLDING
  - MIXED
  - CHOPPY / REVERSAL
  - CHOPPY / REBOUND
  - INSUFFICIENT TRACE
- Added `movement_state_label` for user-facing signal-family language:
  - BREAKDOWN_RISK → DECAY / FATIGUE RISK
  - EMERGENCE → EMERGENCE / POWER GAIN
  - INSTABILITY → MECHANICAL INSTABILITY
- Added `raw_diagnosis` to preserve the original diagnostic read.
- Added aligned user-facing `diagnosis` so card color, signal family, and forensic read agree.
- Fixed instability-dominant cards so they no longer display misleading emergence/decay language as the main diagnosis.
- Updated KDE preview UI to display:
  - signal family + trace behavior in waveform header
  - trace behavior in forensic readout
  - signal family + trace behavior in the action footer

### Interpretation example
A blue instability card may still have raw emergence pressure underneath, but the user-facing diagnosis should describe the dominant signal family.

Example:
- movement_state: INSTABILITY
- movement_state_label: MECHANICAL INSTABILITY
- trace_behavior: CHOPPY / REBOUND
- diagnosis: CHOPPY RELEASE / SHAPE VOLATILITY
- raw_diagnosis: EMERGING SHAPE / POWER GAIN

This preserves the underlying model read while preventing UX confusion.

### Still isolated
KDE remains intentionally disconnected from:
- dashboard/build_all.py
- netlify.toml
- production functions
- live Signal Wall routing

## V7 Final QA / Documentation Pass — kinetic-drift-v7-final-qa-docs

### Current branch
`kinetic-drift-v7-final-qa-docs`

### Current KDE stack
- V2: calibrated KDE score distribution
- V3: waveform preview UI
- V4: diagnosis alignment logic
- V5: mobile UX polish
- V6: operator command UX
- V7: final QA and documentation pass

### V7 objective
Prepare KDE as a complete experimental module candidate by documenting the current architecture, confirming production isolation, and creating a clear next-step checklist before any future integration.

### Current KDE output
KDE currently writes:
- `dist/admin/kinetic_drift_signals.json`
- `dist/admin/kinetic-drift/index.html`

### Current KDE fields
Key output fields include:
- `kde_score`
- `kde_band`
- `kinetic_risk_score`
- `kinetic_emergence_score`
- `kinetic_instability_score`
- `movement_state`
- `movement_state_label`
- `trace_behavior`
- `diagnosis`
- `raw_diagnosis`
- `operator_action`
- `operator_note`
- `confidence_score`
- `drift_trace`
- `metrics`

### Production isolation status
KDE remains intentionally isolated from:
- `dashboard/build_all.py`
- `netlify.toml`
- production Netlify functions
- live Signal Wall routing

### Remaining pre-production checklist
Before production wiring:
1. Confirm scoring behavior over several slates.
2. Review top 20 names for false positives.
3. Add stale-data / generated_at freshness guard.
4. Decide whether KDE belongs under `/admin/`, `/lab/`, or a future protected operator route.
5. Add auth protection before any public admin route exposure.
6. Decide whether KDE feeds Signal Wall cards, Player Forensics, Tracking Radar, or remains standalone.
7. Consider exporting a compact player-level KDE overlay for app-side use.
8. Add glossary/tooltips for KDE, KRS, KES, KIS, trace behavior, and operator commands.
9. Add a dedicated roadmap section for Hawking / inferred biomechanics descendants:
   - Release Window Integrity
   - Shape Mutation Report
   - Fatigue Fracture Report
   - Emergence Mechanics Report
   - Command Scatter / Release Fragmentation
   - Pitch Arsenal Drift
   - future Hitter Kinetic Timing

### V7 recommendation
Keep KDE experimental but “complete enough” after this pass. Next serious milestone should be integration planning, not more cosmetic polish.
