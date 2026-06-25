# DiamondSignals Mobile Isolation Contract

## Purpose

The mobile surface report overhaul must be built in quarantine so the existing desktop reports remain protected.

## Non-Negotiable Rules

1. Mobile work must not alter desktop layout, desktop navigation, desktop shells, or shared desktop styles.
2. Mobile templates must live under:

   dashboard/templates/mobile/

3. Mobile CSS and JS must live under:

   dashboard/static/mobile/

4. Report work must happen report-by-report. No global sweep.
5. Each report requires a Desktop Live Review Gate before mobile implementation.
6. Each mobile report requires a data map:
   - Collapsed Header = Master Signal
   - Expanded Tray = Logic Details
7. No duplicated metrics between collapsed header and expanded tray.
8. Mobile state must be isolated:
   - expand/collapse state
   - active-card state
   - drawer state
   - Field Guide state
   - tracking state feedback

   None of these may trigger hidden desktop layout handlers or desktop container behavior.

9. Touch targets must be 44px to 48px minimum where practical.
10. Expand/collapse and drawers must feel smooth on iPhone.
11. No mobile-only controls may appear on desktop.
12. Desktop fixes are deferred until after mobile signoff unless explicitly approved.

## Required Per-Report Workflow

1. Open desktop report live.
2. Review and document existing desktop UI issues.
3. Decide whether those issues affect mobile design.
4. Define collapsed Master Signal fields.
5. Define expanded Logic Details fields.
6. Build mobile report in quarantined files only.
7. Test on iPhone.
8. Test in simulator.
9. Confirm desktop is unchanged.
10. Sign off before moving to the next report.
