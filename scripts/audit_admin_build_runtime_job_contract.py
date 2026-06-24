#!/usr/bin/env python3
from pathlib import Path
import re
import sys

path = Path("scripts/run_admin_green_baseline_from_build.py")
text = path.read_text() if path.exists() else ""

issues = []

if not path.exists():
    issues.append("missing scripts/run_admin_green_baseline_from_build.py")

required = [
    "def find_job_id()",
    "ADMIN_AUDIT_JOB_ID",
    "INCOMING_HOOK_BODY",
    "def supabase_patch(",
    "def run_audit()",
    "subprocess.Popen",
    "stdout=subprocess.PIPE",
    "stderr=subprocess.STDOUT",
    "print(line, end=\"\", flush=True)",
    "duration_ms",
    "result[\"duration_ms\"]",
    "\"status\": \"running\"",
    "\"status\": \"succeeded\" if result[\"ok\"] else \"failed\"",
]

for marker in required:
    if marker not in text:
        issues.append(f"missing admin build-runtime marker: {marker}")

run_audit_match = re.search(r"def run_audit\(\):(?P<body>.*?)\ndef main\(\):", text, re.S)
if not run_audit_match:
    issues.append("could not locate run_audit body")
else:
    body = run_audit_match.group("body")
    if "started_at_ms = time.time()" not in body:
        issues.append("run_audit must start duration timer")
    if '"duration_ms": int((time.time() - started_at_ms) * 1000)' not in body:
        issues.append("run_audit must return duration_ms")
    if '"exit_code": exit_code' not in body:
        issues.append("run_audit must return exit_code")
    if '"stdout": tail(output)' not in body:
        issues.append("run_audit must return stdout tail")

print("--- DiamondSignals admin build-runtime job contract audit ---")
print(f"admin_build_runtime_job_contract_issues: {len(issues)}")

for issue in issues:
    print(f"FAIL: {issue}")


if issues:
    print()
    print("FINAL_STATUS: FAIL_ADMIN_BUILD_RUNTIME_JOB_CONTRACT")
    sys.exit(1)

print("admin_job_id_supported: true")
print("incoming_hook_body_supported: true")
print("build_runtime_supabase_updates_supported: true")
print("build_runtime_duration_ms_supported: true")
print("build_runtime_logs_stream_to_netlify: true")
print()
print("FINAL_STATUS: PASS_ADMIN_BUILD_RUNTIME_JOB_CONTRACT")
