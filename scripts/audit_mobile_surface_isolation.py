#!/usr/bin/env python3
"""DiamondSignals Mobile Command Experience isolation audit."""
from pathlib import Path
import subprocess
import sys

ALLOWED_PREFIXES = (
    "dashboard/templates/mobile/",
    "dashboard/static/mobile/",
    "docs/mobile-overhaul/",
    "scripts/audit_mobile_surface_isolation.py",
)

HIGH_RISK_PREFIXES = (
    "dashboard/templates/components/",
    "dashboard/templates/ledger_styles.css",
    "dashboard/templates/shell_styles.css",
    "dashboard/templates/report_styles.css",
    "dashboard/build_dashboard.py",
    "dashboard/build_all.py",
    "src/js/player-card-actions.js",
    "dist/",
)

REQUIRED = (
    Path("docs/mobile-overhaul/MOBILE_ISOLATION_CONTRACT.md"),
    Path("dashboard/templates/mobile/surface_reports/_mobile_shell.html"),
    Path("dashboard/static/mobile/mobile_surface_base.css"),
    Path("dashboard/static/mobile/mobile_command_experience.js"),
)

def changed_paths():
    out = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        text=True, capture_output=True, check=True
    ).stdout.splitlines()
    return [(line, line[3:].strip()) for line in out if len(line) >= 4]

def main():
    issues = []
    for path in REQUIRED:
        if not path.exists():
            issues.append(f"MISSING_REQUIRED_MOBILE_FILE: {path}")

    for line, path in changed_paths():
        if not path.startswith(ALLOWED_PREFIXES):
            issues.append(f"OUTSIDE_MOBILE_QUARANTINE: {line}")
        if path.startswith(HIGH_RISK_PREFIXES):
            issues.append(f"HIGH_RISK_DESKTOP_OR_TRACKING_PATH: {line}")

    css = Path("dashboard/static/mobile/mobile_surface_base.css")
    if css.exists():
        text = css.read_text(encoding="utf-8")
        if ".ds-mobile-report-view" not in text:
            issues.append("MOBILE_CSS_NAMESPACE_MISSING")

    js = Path("dashboard/static/mobile/mobile_command_experience.js")
    if js.exists():
        text = js.read_text(encoding="utf-8")
        if ".ds-mobile-report-view" not in text:
            issues.append("MOBILE_JS_NAMESPACE_MISSING")

    if issues:
        for issue in issues:
            print(issue)
        print("FINAL_STATUS: FAIL_MOBILE_COMMAND_ISOLATION")
        return 1

    print("MOBILE_COMMAND_ISOLATION: PASS")
    print("FINAL_STATUS: PASS_MOBILE_COMMAND_ISOLATION")
    return 0

if __name__ == "__main__":
    sys.exit(main())
