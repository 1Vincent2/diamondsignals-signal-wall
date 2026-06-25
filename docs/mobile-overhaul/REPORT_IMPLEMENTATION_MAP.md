# DiamondSignals Mobile Surface Report Implementation Map

## Purpose

This document defines the required report-by-report mobile overhaul sequence.

Each surface report must pass through:

1. Desktop Live Review Gate
2. Mobile Data Map
3. Quarantined Mobile Build
4. Interaction QA
5. Mobile Usability QA
6. Performance QA
7. Desktop Non-Regression Gate

No report may be skipped.
No global sweep is allowed.

---

## Shared Mobile Rules

- Collapsed card = Master Signal.
- Expanded tray = Logic Details.
- No duplicated metrics between header and tray.
- Mobile state must not trigger desktop state.
- Touch targets should be 44px to 48px minimum where practical.
- Expand/collapse should feel smooth on iPhone.
- Field Guide/Menu twin behavior should be used where applicable.
- Desktop fixes are deferred until after mobile signoff unless explicitly approved.

---

# Report 1 — Signal Wall

## Desktop Live Review Gate

Review:

- Nav bar
- Page margins
- Top shell alignment
- Search placement
- Field Guide placement
- Card spacing

Document issues before mobile work.

## Mobile Master Signal

- Player name
- Hitter/pitcher role
- Rank or signal classification
- Primary Edge Score
- One strongest signal label

## Mobile Expanded Tray

- Full metric grid
- SEAGER
- BABIP
- BB%
- K%
- BB/K or K/BB
- Season context
- Sparkline
- Why-this-signal explanation
- Tracking action

---

# Report 2 — Waiver Wire

## Desktop Live Review Gate

Review:

- Hero/header alignment
- Field Guide placement
- Cards/table spacing
- Desktop margins
- Action buttons

## Mobile Master Signal

- Player name
- Team/position
- Waiver priority score
- Availability or roster status
- One-line pickup reason

## Mobile Expanded Tray

- Recent usage trend
- Playing time context
- Core performance metrics
- Category impact
- Risk label
- Schedule/context note
- Recommended action: add, watch, stash, ignore

---

# Report 3 — Watch List

## Desktop Live Review Gate

Review:

- Tracked-state indicators
- Remove/keep action feedback
- Card spacing
- Nav/margins

## Mobile Master Signal

- Player name
- Watch status: Tracking, Rising, Cooling, Triggered
- Watch reason
- One key movement indicator

## Mobile Expanded Tray

- Original watch thesis
- Current supporting metrics
- What changed since added
- Trigger conditions
- Next review signal/date
- Remove or keep tracking action

---

# Report 4 — Apex Extraction

## Desktop Live Review Gate

Review:

- Header/hero
- Field Guide placement
- Desktop alignment
- Signal cards
- Color/heat-state readability

## Mobile Master Signal

- Player name
- Apex score
- Extraction type
- Primary breakout indicator

## Mobile Expanded Tray

- Underlying skill components
- Trend direction
- Baseline comparison
- Market timing context
- Opportunity window
- Why signal is actionable
- Risk or false-positive warning

---

# Report 5 — MLB Extraction

## Desktop Live Review Gate

Review:

- Desktop shell
- Field Guide controls
- Card/table alignment
- Spacing and margins

## Mobile Master Signal

- Player name
- Extraction score
- MLB readiness status
- Primary MLB-level indicator

## Mobile Expanded Tray

- Performance evidence
- Role/path context
- Recent trend
- Promotion or roster relevance
- Statcast or production support
- Action recommendation

---

# Report 6 — Typical Call-Up

## Desktop Live Review Gate

Review:

- Promotion cards
- ETA presentation
- Desktop Field Guide
- Margins/nav

## Mobile Master Signal

- Player name
- Call-up probability
- ETA bucket
- Primary promotion reason

## Mobile Expanded Tray

- 40-man/roster context
- Team need
- Performance readiness
- Playing time path
- Recent context if available
- Fantasy action: add, stash, monitor

---

# Report 7 — Velocity Decay Monitor

## Desktop Live Review Gate

Review:

- Pitcher risk display
- Severity labels
- Chart/metric readability
- Margins/nav

## Mobile Master Signal

- Pitcher name
- Velocity delta
- Severity level
- Affected pitch or arsenal piece

## Mobile Expanded Tray

- Velocity trend chart
- Baseline vs recent velocity
- Pitch mix context
- Command/performance impact
- Injury-risk context
- Action recommendation: fade, monitor, hold

---

# Report 8 — Stuff+ Disruption Feed

## Desktop Live Review Gate

Review:

- Stuff+ deltas
- Pitch-family display
- Desktop card spacing
- Field Guide behavior

## Mobile Master Signal

- Pitcher name
- Stuff+ delta
- Disruption score
- Primary pitch or arsenal change

## Mobile Expanded Tray

- Stuff+ components
- Pitch-level movement
- Whiff/chase/command context
- Recent usage changes
- Why arsenal changed
- Action recommendation

---

# Report 9 — IVB Heat Map

## Desktop Live Review Gate

Review:

- Heat-state labels
- Field Guide pill
- Modal/drawer close behavior
- Chart/card alignment
- Desktop margins

## Mobile Master Signal

- Pitcher name
- IVB score or delta
- Heat label: climber, dead zone, apex rise
- Primary fastball-shape signal

## Mobile Expanded Tray

- IVB raw
- IVB vs average
- Apex rise
- Dead zone status
- Whiff probability
- Pitch-shape explanation
- Risk/sustainability context

---

# Completion Rule

A report is complete only when:

- Desktop was reviewed live first.
- Desktop notes were captured.
- Mobile master/tray data map was defined.
- Mobile was built in quarantined files only.
- No duplicated metrics exist.
- State isolation passed.
- Touch targets passed.
- Performance passed.
- Desktop non-regression passed.
