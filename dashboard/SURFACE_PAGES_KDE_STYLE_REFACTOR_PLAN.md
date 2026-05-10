# DiamondSignals Surface Pages — KDE-Style Refactor Plan

## Branch
surface-pages-kde-style-refactor

## Objective
Refactor each Signal Surface one at a time toward the cleaner KDE architecture:
- less fragile giant embedded HTML/CSS patching
- clearer template/component separation
- mobile-first report structure
- shared drawer navigation
- shared future Command Sort Rail
- stable card metadata contract
- local-preview auth bypass for simulator/Wi-Fi testing

## Why
KDE is easier to modify because it has cleaner template structure and explicit data attributes. Older report pages are harder to adjust because they rely on large Python string templates, accumulated CSS overrides, desktop-first layout assumptions, and inconsistent card metadata.

## Refactor Standard
Each surface should eventually have:

1. Builder/data layer
   - keep report calculations in Python
   - output normalized rows
   - add sortable metadata fields before rendering

2. Template layer
   - move large HTML/CSS blocks out of huge Python strings when practical
   - use templates similar to KDE where possible

3. Mobile shell
   - preserve global right-side drawer nav
   - keep mobile hero readable
   - preserve or recast desktop summary cards into mobile Signal Cards instead of blindly hiding them

4. Card metadata contract
   - data-player-id
   - data-player-name
   - data-player-team
   - data-source-tag
   - data-sort-score
   - data-rank
   - data-signal-type

5. Future Command Sort Rail
   - position: under hero / above report board
   - shared styling, not native iOS select if avoidable
   - sort by score, name, rank, risk/opportunity where relevant
   - driven by data attributes, not text scraping

6. Local preview auth bypass
   - allow localhost
   - allow 127.0.0.1
   - allow 192.168.*
   - production auth remains intact

## Page Order
1. MLB Extraction Ledger
2. Apex Extraction
3. Promotion Watch
4. IVB Heat Map
5. Stuff+ Disruption
6. Velocity Decay
7. Signal Wall revisit after shared pattern is proven

## Immediate Next Step
Start with MLB Extraction Ledger. Inspect builder structure, identify render boundaries, then either:
- create a dedicated Jinja template like KDE, or
- create reusable template fragments before deeper extraction.

Do not reintroduce sorting until the card metadata contract and stable mobile hero/signal-card structure are in place.
