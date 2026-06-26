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

- [ ] Field Guide/Menu twin remains in the approved position.
- [ ] LIVE status, report title, search, and card list remain usable.
- [ ] Collapsed card layout matches the approved Simulator layout.
- [ ] Expanded card layout matches the approved Simulator layout.
- [ ] No desktop layout changes are introduced.

---

## Card Interaction QA

- [ ] Tapping a collapsed card opens the player details.
- [ ] Opening one card closes the previously open card.
- [ ] Close Metrics closes the open card.
- [ ] Card open/close state does not trap scrolling.
- [ ] Expanded card content remains readable on iPhone-width screens.

---

## Field Guide/Menu Interaction QA

- [ ] Field Guide opens from the mobile twin control.
- [ ] Menu opens from the mobile twin control.
- [ ] Field Guide does not cover or misplace the Menu control.
- [ ] Menu does not cover or misplace the Field Guide control.
- [ ] Field Guide overlay does not leave cards stuck in an unusable state.

---

## Tracking Action QA

- [ ] INITIATE TRACKING button is visible in the expanded card.
- [ ] INITIATE TRACKING is tappable on mobile.
- [ ] Tracking action does not accidentally collapse the card.
- [ ] Tracking state feedback is understandable enough for this phase.
- [ ] Watchlist state polish can remain a later hardening item if needed.

---

## Scroll / Touch QA

- [ ] Page scroll remains smooth with no major jumpiness.
- [ ] Expanded card can be scrolled/read without losing context.
- [ ] Touch targets feel large enough.
- [ ] No important content is hidden behind the bottom browser bar.
- [ ] No horizontal scrolling appears.

---

## QA Result

Status: Pending.

## Issues Found

None yet.

## Next Step

Complete Simulator review, then either:
1. Mark PASS and commit this QA checklist, or
2. Document specific defects and fix only those defects.
