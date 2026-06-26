# Signal Wall Mobile Usability QA

## Branch

feature/mobile-surface-report-overhaul

## Starting Checkpoint

27a984e — Document Signal Wall mobile approval after desktop repair

## Phase

Report 1 — Signal Wall  
Phase 5 — Mobile Usability QA

## Rule

Do not redesign the approved Signal Wall mobile layout.  
Only document and fix specific usability defects found during QA.

---

## Mobile Layout Preservation

- [x] Field Guide/Menu twin remains in the approved position.
- [x] LIVE status, report title, search, and card list remain usable.
- [x] Collapsed card layout matches the approved Simulator layout.
- [x] Expanded card layout matches the approved Simulator layout.
- [x] No desktop layout changes are introduced.

---

## Card Interaction QA

- [x] Tapping a collapsed card opens the player details.
- [x] Opening one card closes the previously open card.
- [x] Close Metrics closes the open card.
- [x] Card open/close state does not trap scrolling.
- [x] Expanded card content remains readable on iPhone-width screens.

---

## Field Guide/Menu Interaction QA

- [x] Field Guide opens from the mobile twin control.
- [x] Menu opens from the mobile twin control.
- [x] Field Guide does not cover or misplace the Menu control.
- [x] Menu does not cover or misplace the Field Guide control.
- [x] Field Guide overlay does not leave cards stuck in an unusable state.

---

## Tracking Action QA

- [x] INITIATE TRACKING button is visible in the expanded card.
- [x] INITIATE TRACKING is tappable on mobile.
- [x] Tracking action does not accidentally collapse the card.
- [x] Tracking state feedback is understandable enough for this phase.
- [x] Watchlist state polish can remain a later hardening item if needed.

---

## Scroll / Touch QA

- [x] Page scroll remains smooth with no major jumpiness.
- [x] Expanded card can be scrolled/read without losing context.
- [x] Touch targets feel large enough.
- [x] No important content is hidden behind the bottom browser bar.
- [x] No horizontal scrolling appears.

---

## QA Result

Status: PASS.

## Issues Found

Resolved during Phase 5 QA:

- Field Guide pill border/glow was weaker than the Menu pill.
- Mobile Menu label needed to be rebranded from `Menu` to `MODULE / FEEDS`.
- First label patch caused `MODULE / FEEDS` to contaminate the Field Guide twin on reinstall.
- Metric cards can move slightly within the open frame when touched; noted as non-blocking unless it becomes a real usability defect.

## Fixes Applied

- `SIGNAL_WALL_MOBILE_TWIN_PILL_STYLE_POLISH_V18`
- `SIGNAL_WALL_MOBILE_MENU_LABEL_MODULE_FEEDS_V18`
- `SIGNAL_WALL_MOBILE_FIELD_GUIDE_LABEL_RESET_V19`

Phone preview approved after the V19 reset.

## Next Step

Proceed to Report 1 — Phase 6: Performance QA.
