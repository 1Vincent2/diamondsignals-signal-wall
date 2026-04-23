# AAA Freshness Blocker

## Current state
- The Call-Up page now hides stale signal sections and relies on the fresher Movement Layer.
- `scripts/build_aaa_hitter_refresh.py` exports the current AAA hitter source into `dist/aaa_hitter_refresh.json`.
- The export includes machine-readable freshness metadata.

## Confirmed source reality
- Latest AAA snapshot: `2025-06-06`
- Unique AAA weeks available: `['2025-06-06']`
- `delta_ready = false`

## Blocking reason
Only one AAA snapshot is currently available, so a rolling AAA delta engine cannot run yet.

## What this means
We cannot build a truthful live AAA hitter surveillance layer from the current AAA source alone.

## Correct next workstream
Build or ingest a fresh AAA hitter source with multiple snapshots, then restore:
- live
hitter surveillance
- Apex 5
- surge / delta scoring

## Branch checkpoint
This branch now contains:
- page honesty improvements
- AAA hitter refresh export stub
- freshness status metadata


## 2026-04-23 checkpoint update
- Promotion Watch 14 DAY fetch bug was fixed in the live builder.
- AAA ingest function was migrated into the signal wall repo.
- 72 HR refresh artifacts are currently healthy and rebuild locally.
- Remaining blocker is no longer the fetch logic itself; it is the stale upstream `milb_aaa_weekly_signal_base` source for the 14 DAY layer.
