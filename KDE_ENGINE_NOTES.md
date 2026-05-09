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
