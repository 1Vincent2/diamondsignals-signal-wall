# WORKFLOW

## Active source of truth
- `dashboard/`
- `dashboard/templates/`
- `src/js/player-search.js`

## Generated output
- `dist/`

## Archived / inactive staging work
- `archive/src-staging/`
- `archive/netlify-functions/`

## Build architecture
- Netlify build command: `python dashboard/build_all.py`
- `dashboard/build_all.py` is the single orchestrator for subdomain page generation
- Current build set includes:
  - `build_dashboard.py`
  - `build_typical_call_up.py`

## Scheduled rebuild architecture
- Scheduled function: `netlify/functions/trigger-rebuild.mjs`
- Current cron schedule: twice daily
- Current cron expression: `0 14,20 * * *`
- This currently targets:
  - 10:00 AM Eastern
  - 4:00 PM Eastern
- The scheduled function triggers a Netlify rebuild
- The Netlify rebuild runs `python dashboard/build_all.py`
- Therefore, all pages included in `build_all.py` are rebuilt on each scheduled run

## Rule for future subdomain pages
When adding a new subdomain page generator:
1. Create the generator script in `dashboard/`
2. Add that script to the `SCRIPTS` list in `dashboard/build_all.py`
3. Test locally with `python3 dashboard/build_all.py`
4. Deploy

If a new page is not added to `dashboard/build_all.py`, it will not be refreshed by the scheduled rebuilds.

## Normal local workflow
1. Edit files in:
   - `dashboard/`
   - `dashboard/templates/`
   - `src/js/player-search.js`
2. Rebuild locally
3. Inspect generated output in `dist/`
4. Commit source files and generated output when needed for deploy consistency

## Local build commands
### Rebuild all active subdomain pages
```bash
python3 dashboard/build_all.py