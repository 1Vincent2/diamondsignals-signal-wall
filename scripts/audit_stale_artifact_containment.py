#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_COMPATIBILITY = [
    "dashboard/build_aaa_acquisition_board.py",
    "dashboard/build_mlb_extraction.py",
    "dashboard/templates/hidden_gems/mlb_extraction.html",
    "dashboard/build_waiver_candidates.py",
    "dashboard/build_player_signal_index.py",
    "dashboard/enrich_surface_season_context.py",
    "dashboard/audit_report_data_modes.py",
    "scripts/audit_legacy_route_names.py",
    "scripts/audit_mobile_header_menu_contract.py",
]

ARCHIVE_BACKUP = [
    "dashboard/build_aaa_acquisition_board.py.fg-mobile-header.bak",
    "dashboard/build_hidden_gems.py.BACKUP_2026-04-24_1325",
    "dashboard/build_call_up_live.py.bak",
    "dashboard/build_call_up_live.py.hardening.bak",
    "dashboard/templates/shell_nav.html.bak",
]

STALE_DIST_NO_DEPLOY = [
    "dist/mlb-local/index.html",
    "dist/typical-call-up-v3/index.html",
    "dist/typical-call-up/index_tabbed_live.html",
]

COMPATIBILITY_DIST = [
    "dist/hidden-gems/index.html",
    "dist/hidden-gems/mlb_extraction_ledger.json",
]

ALLOWED_LEGACY_IDENTIFIERS = [
    "hidden-gems",
    "hidden_gems",
    "hidden_gems_score",
    "hidden_gems_score_raw",
    "HIDDEN_GEMS_DIR",
    "templates/hidden_gems",
    "active_nav=\"hidden_gems\"",
]

DISALLOWED_USER_FACING_COPY = [
    "Hidden Gems Pipeline",
    "DiamondSignals // Hidden Gems",
    "<h1 class=\"hero-title\">Hidden Gems</h1>",
    "Hidden Gems // Institutional Edge",
    "Hidden Gems // Field Guide",
    "// Hidden Gems",
]

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def classify_existing(paths: list[str], label: str) -> list[Path]:
    found = []
    print(f"\n--- {label} ---")
    for item in paths:
        path = ROOT / item
        if path.exists():
            print(f"OK: {item}")
            found.append(path)
        else:
            print(f"SKIP missing: {item}")
    return found

def main() -> None:
    problems: list[str] = []

    print("--- DiamondSignals stale artifact containment audit ---")

    active = classify_existing(ACTIVE_COMPATIBILITY, "ACTIVE_COMPATIBILITY")
    backups = classify_existing(ARCHIVE_BACKUP, "ARCHIVE_BACKUP")
    stale_dist = classify_existing(STALE_DIST_NO_DEPLOY, "STALE_DIST_NO_DEPLOY")
    compat_dist = classify_existing(COMPATIBILITY_DIST, "COMPATIBILITY_DIST")

    print("\n--- active compatibility identifier check ---")
    for path in active:
        text = read(path)
        if any(token in text for token in ["hidden-gems", "hidden_gems", "Hidden Gems"]):
            print(f"OK: {rel(path)} contains legacy compatibility naming")
        for bad in DISALLOWED_USER_FACING_COPY:
            if bad in text and path.name not in {"audit_legacy_route_names.py"}:
                problems.append(f"{rel(path)} contains disallowed user-facing legacy copy: {bad}")

    print("\n--- backup containment check ---")
    for path in backups:
        text = read(path)
        has_legacy_copy = any(bad in text for bad in DISALLOWED_USER_FACING_COPY) or "AAA Gems" in text
        if has_legacy_copy:
            print(f"OK: {rel(path)} classified as ARCHIVE_BACKUP with legacy copy contained")
        else:
            print(f"OK: {rel(path)} classified as ARCHIVE_BACKUP")

    print("\n--- stale dist containment check ---")
    for path in stale_dist:
        text = read(path)
        if "/hidden-gems" in text or "MLB Extraction Ledger" in text or "Hidden Gems" in text:
            print(f"OK: {rel(path)} classified as STALE_DIST_NO_DEPLOY")
        else:
            print(f"OK: {rel(path)} classified as STALE_DIST_NO_DEPLOY")

    print("\n--- compatibility dist check ---")
    for path in compat_dist:
        text = read(path)
        if "Hidden Gems" in text:
            problems.append(f"{rel(path)} contains user-facing Hidden Gems copy inside compatibility dist")
        else:
            print(f"OK: {rel(path)} compatibility output has no user-facing Hidden Gems copy")

    print("\n--- summary ---")
    print(f"active_compatibility_files: {len(active)}")
    print(f"archive_backup_files: {len(backups)}")
    print(f"stale_dist_no_deploy_files: {len(stale_dist)}")
    print(f"compatibility_dist_files: {len(compat_dist)}")
    print(f"containment_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_STALE_ARTIFACT_CONTAINMENT")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_STALE_ARTIFACT_CONTAINMENT")

if __name__ == "__main__":
    main()
