#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

SCAN_FILES = [
    ROOT / "dashboard/build_aaa_acquisition_board.py",
    ROOT / "dashboard/build_mlb_extraction.py",
    ROOT / "dashboard/templates/shell_nav.html",
    ROOT / "dashboard/templates/hidden_gems/mlb_extraction.html",
    ROOT / "dist/hidden-gems/index.html",
    ROOT / "dist/mlb-local/index.html",
    ROOT / "dist/typical-call-up/index_tabbed_live.html",
]

USER_FACING_HIDDEN_GEMS_PATTERNS = [
    r"<title>[^<]*Hidden Gems[^<]*</title>",
    r"<h1[^>]*>[^<]*Hidden Gems[^<]*</h1>",
    r"brand-title[^\\n]*Hidden Gems",
    r"drawer-kicker[^\\n]*Hidden Gems",
    r"Hidden Gems Pipeline",
    r"Hidden Gems //",
    r"// Hidden Gems",
]

ALLOWED_LEGACY_PATTERNS = [
    r"dist/hidden-gems",
    r"/hidden-gems",
    r"hidden_gems_score",
    r"hidden_gems_score_raw",
    r"HIDDEN_GEMS_DIR",
    r"hidden_gems",
    r"load_hidden_gems",
    r"build_hidden_gems",
    r"templates/hidden_gems",
    r"active_nav=\"hidden_gems\"",
]

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def main() -> None:
    problems: list[str] = []

    print("--- DiamondSignals legacy route/name contract audit ---")

    checked = 0
    for path in SCAN_FILES:
        if not path.exists():
            print(f"SKIP: {path.relative_to(ROOT)} does not exist")
            continue

        checked += 1
        text = read(path)
        rel = str(path.relative_to(ROOT))

        for pattern in USER_FACING_HIDDEN_GEMS_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.I):
                line = text.count("\n", 0, m.start()) + 1
                snippet = " ".join(m.group(0).split())[:220]
                problems.append(f"{rel}:{line}: user-facing legacy name still present: {snippet}")

        # Legacy paths are allowed, but only if the file contains no user-facing branding hit.
        # This keeps compatibility from becoming product language again.
        if "hidden-gems" in text or "hidden_gems" in text:
            print(f"OK: {rel} may contain legacy compatibility identifiers")

    print("\n--- summary ---")
    print(f"files_checked: {checked}")
    print(f"user_facing_legacy_name_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_LEGACY_ROUTE_NAME_CONTRACT")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_LEGACY_ROUTE_NAME_CONTRACT")

if __name__ == "__main__":
    main()
