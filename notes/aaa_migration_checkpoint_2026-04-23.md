# AAA Migration Checkpoint — 2026-04-23

Completed:
- Fixed Promotion Watch 14 DAY fetch logic so distinct recent `week_start` values are used instead of duplicate rows from the same week.
- Migrated `netlify/functions/ingest-milb-aaa-weekly.mjs` into the signal wall repo.
- Verified local AAA refresh script:
  - `scripts/build_aaa_hitter_refresh.py`
  - outputs fresh `dist/aaa_hitter_refresh.json`
  - outputs fresh `dist/aaa_pitcher_refresh_probe.json`
- Rebuilt `dashboard/build_call_up_live.py` successfully with:
  - 72 HR pitching
  - 72 HR hitting
  - 14 DAY pitching
  - 14 DAY hitting
  - Recent MLB Arrivals

Known remaining issue:
- `milb_aaa_weekly_signal_base` is still stale upstream, so the 14 DAY board is structurally fixed but still limited by old weekly snapshots.

Repo split decision:
- AAA ingest / signal generation now belongs in `diamondsignals-signal-wall`
- old AAA newsletter-era functions were removed from `ds-deploy-root`

Relevant commits:
- signal wall: `46e634f` Fix Promotion Watch 14-day distinct week_start fetch
- signal wall: `7b961fc` Migrate AAA ingest function into signal wall repo
- ds-deploy-root: `0479168` Remove obsolete AAA newsletter-era functions
