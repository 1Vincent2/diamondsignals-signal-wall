# DiamondSignals Mobile Hardening Checklist

Date: 2026-06-10
Scope: All report routes, mobile shells, Field Guide drawers, menu systems, tracking controls, and report hero/header layouts.

## Purpose

This checklist prevents a repeat of the Apex mobile corruption pattern. Any future report modification must preserve the mobile shell, menu behavior, Field Guide drawer behavior, tracking controls, and route/domain/report contracts before visual polish is accepted.

## Core Rule

No report is considered hardened unless it passes the green baseline audit runner:

./scripts/run_green_baseline_audit.sh

## Required Checks

1. Mobile menu must not block title, hero, Field Guide trigger, tracking buttons, or card actions.
2. Field Guide drawers must preserve overlay, drawer/panel, open/close class, aria-hidden state, body/document open-state class, visible mobile title/header/close control, and z-index above the mobile menu when open.
3. Tracking actions must remain wired and pass scripts/test_tracking_actions.py.
4. Route/domain contract must stay clean and pass scripts/test_route_domain_contract.py.
5. Report data-mode truth must pass dashboard/audit_report_data_modes.py.
6. Mobile header/menu obstruction must pass scripts/audit_mobile_header_menu_contract.py.
7. Before every commit, run ./scripts/run_green_baseline_audit.sh.

## Apex Corruption Lessons

- Do not patch mobile CSS from desktop assumptions.
- Do not add drawer/menu/Field Guide rules without checking z-index stacking.
- Do not let fixed mobile headers overlap report titles.
- Do not introduce duplicate mobile menu systems.
- Do not allow report Field Guides to hide behind global menu layers.
- Do not do cosmetic work before tracking, routing, status, and mobile contracts pass.
- Do not trust visual inspection alone.

## Closeout Standard

A report/mobile change is complete only when:

- git status --short is clean except intentional files
- FINAL_STATUS: PASS_MOBILE_HEADER_MENU_AUDIT
- FINAL_STATUS: PASS_INSPECTION_ONLY
- Tracking regression audit passed
- GREEN BASELINE AUDIT PASSED
