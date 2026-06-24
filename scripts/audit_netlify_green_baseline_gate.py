#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

issues = []

def require_file(rel):
    path = ROOT / rel
    if not path.exists():
        issues.append(f"missing file: {rel}")
        return ""
    return path.read_text(encoding="utf-8")

netlify = require_file("netlify.toml")
wrapper = require_file("scripts/run_admin_green_baseline_from_build.py")
baseline = require_file("scripts/run_green_baseline_audit.sh")

def require(label, condition):
    if not condition:
        issues.append(label)

require(
    "netlify command must gate build_all with green baseline wrapper using &&",
    "python3 scripts/run_admin_green_baseline_from_build.py && python3 dashboard/build_all.py" in netlify,
)

require(
    "netlify command must not allow green baseline wrapper bypass with semicolon before build_all",
    "python3 scripts/run_admin_green_baseline_from_build.py; python3 dashboard/build_all.py" not in netlify,
)

require(
    "wrapper must run green baseline audit even without admin job id",
    "running green baseline audit without admin job wrapper" in wrapper and "result = run_audit()" in wrapper,
)

require(
    "wrapper must return nonzero exit code when baseline fails without admin job wrapper",
    "return result[\"exit_code\"]" in wrapper,
)

require(
    "wrapper must stream green baseline output live instead of hiding failing audit output",
    "subprocess.Popen" in wrapper
    and "print(line, end=\"\", flush=True)" in wrapper
    and "stderr=subprocess.STDOUT" in wrapper
)

require(
    "admin job wrapper must still report started state when job id exists",
    "Green baseline audit started in Netlify build runtime." in wrapper,
)

require(
    "admin job wrapper must still report passed/failed final status",
    "Green baseline audit passed." in wrapper and "Green baseline audit failed." in wrapper,
)

require(
    "green baseline runner must still end with passed marker",
    "--- GREEN BASELINE AUDIT PASSED ---" in baseline,
)

if issues:
    print("--- DiamondSignals Netlify green baseline gate audit ---")
    print(f"netlify_green_baseline_gate_issues: {len(issues)}")
    for issue in issues:
        print(f" - {issue}")
    print("\nFINAL_STATUS: FAIL_NETLIFY_GREEN_BASELINE_GATE")
    sys.exit(1)

print("--- DiamondSignals Netlify green baseline gate audit ---")
print("netlify_green_baseline_gate_issues: 0")
print("normal_netlify_deploys_run_green_baseline: true")
print("admin_job_wrapped_deploys_still_report_to_supabase: true")
print("\nFINAL_STATUS: PASS_NETLIFY_GREEN_BASELINE_GATE")
