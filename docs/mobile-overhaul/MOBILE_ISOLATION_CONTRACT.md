# DiamondSignals Mobile Command Experience — Isolation Contract

## Purpose
Build a first-class mobile presentation in quarantine while preserving the accepted desktop Signal Wall and the hardened tracking/authentication contract.

## Golden base
- Signal Wall production baseline: `b8d2bafec4c1fd44917cad07d613e628ff488551`
- Working branch: `mobile-command-experience-v1`

## Non-negotiable rules
1. Mobile presentation must not alter desktop layout, desktop navigation, desktop shells, or shared desktop styles.
2. Mobile templates live under `dashboard/templates/mobile/`.
3. Mobile CSS/JS live under `dashboard/static/mobile/`.
4. All mobile selectors and behavior must be scoped beneath `.ds-mobile-report-view` or another explicit `ds-mobile-` wrapper.
5. Work report-by-report. No global visual sweep.
6. Shared canonical data and business actions may be reused; desktop presentation is not the mobile design source of truth.
7. Mobile interaction state must not trigger desktop handlers except canonical shared actions such as tracking.
8. The hardened INITIATE TRACKING contract remains canonical; mobile presentation must preserve the required player/action attributes.
9. Touch targets should be at least 44px where practical.
10. Mobile-only controls must never appear on desktop.
11. Desktop non-regression is a release gate.
12. Production/Netlify auto-publishing remains locked until explicit acceptance.

## Mobile interaction grammar
- DISCOVER: immersive swipeable signal deck.
- ALL PLAYERS: dense ranked scanner for rapid player lookup.
- SEARCH: direct player access.
- FILTER: narrow the active signal set.
- TAP: open player intelligence.
- INFO: contextual Field Guide explanation.
- TRACK: invoke the canonical hardened tracking action.

## Signal Wall v1 principle
The first actionable player signal should appear immediately. Operational metadata is compressed into telemetry rather than consuming the first viewport.

## Workflow
1. Verify desktop golden baseline.
2. Build only in quarantined mobile files.
3. Run isolation audit.
4. Build/test mobile preview.
5. Test iPhone Simulator and real mobile Safari when appropriate.
6. Confirm desktop remains unchanged.
7. Sign off before expanding to another report.
