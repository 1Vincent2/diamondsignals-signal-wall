#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

issues = []

SOURCE_SCAN_ROOTS = [
    Path("dashboard"),
    Path("netlify"),
    Path("scripts"),
    Path("public"),
]

TEXT_EXTS = {
    ".py", ".js", ".mjs", ".ts", ".tsx", ".html", ".css", ".json",
    ".toml", ".md", ".txt", ".yml", ".yaml", ".sh"
}

AUDIT_DEFINITION_ALLOWLIST_FILES = {
    # Audit/seal files intentionally contain marker strings as assertions.
    Path("scripts/audit_signal_wall_env_boundary.py"),
    Path("scripts/audit_strict_environment_preflight.py"),
    Path("scripts/audit_admin_build_runtime_job_contract.py"),
    Path("scripts/audit_netlify_green_baseline_gate.py"),
    Path("netlify.toml"),
    Path("ADMIN_UI_INTEGRATION_SEAL.md"),
    Path("PRE_MOBILE_READINESS_SEAL.md"),
    Path("MOBILE_DEFERRED_UNTIL_ITEM21.md"),
}

PRIVATE_MARKER_ALLOWLIST_FILES = AUDIT_DEFINITION_ALLOWLIST_FILES | {
    # Approved server/build-only env consumers.
    Path("dashboard/build_aaa_acquisition_board.py"),
    Path("dashboard/sync_prospect_intelligence.py"),
    Path("dashboard/build_call_up_v2.py"),
    Path("dashboard/build_ivb_heat_map.py"),
    Path("dashboard/build_dashboard.py"),
    Path("dashboard/build_call_up_v3_preview.py"),
    Path("dashboard/build_aaa_weekly_signal_base.py"),
    Path("dashboard/build_call_up_live.py"),
    Path("dashboard/build_market_eligibility.py"),
    Path("dashboard/lib/build_operational_status_feed.py"),
    Path("dashboard/archive_call_up_builders/build_typical_call_up.py"),
    Path("dashboard/archive_call_up_builders/build_typical_call_up_current_live_variant.py"),
    Path("netlify/functions/ingest-milb-aaa-weekly.mjs"),
    Path("netlify/functions/front-door-capture.js"),
    Path("netlify/functions/trigger-rebuild.mjs"),
    Path("netlify/functions/admin-audit-runner.mjs"),
    Path("netlify/functions/admin-audit-worker-background.mjs"),
    Path("netlify/functions/admin-runner.mjs"),
    Path("scripts/verify_aaa_weekly_ingest.sh"),
    Path("scripts/run_admin_green_baseline_from_build.py"),
    Path("scripts/run_aaa_weekly_ingest.sh"),
}

PRIVATE_SERVICE_MARKERS = [
    "SUPABASE_SERVICE_ROLE_KEY",
    "SERVICE_ROLE",
    "ADMIN_RUN_TOKEN",
    "NETLIFY_BUILD_HOOK_URL",
    "NETLIFY_GREEN_BASELINE_AUDIT_BUILD_HOOK_URL",
    "YAHOO_CLIENT_SECRET",
    "YAHOO_REFRESH_TOKEN",
    "YAHOO_ACCESS_TOKEN",
]

DEV_DEBUG_MARKERS = [
    "DEBUG=true",
    "NODE_ENV=development",
    "FLASK_ENV=development",
    "NEXT_PUBLIC_DEBUG",
    "VITE_DEBUG",
    "debugger;",
]

LOCAL_TEST_CONFIG_MARKERS = [
    ".env.local",
    ".env.development",
    "local_test_config",
    "test_config",
]

PERMISSIVE_LOGGING_MARKERS = [
    "print(os.environ",
    "console.log(process.env",
    "JSON.stringify(process.env",
    "dump_env",
    "env_dump",
    "log_env",
]

NON_PRODUCTION_CLIENT_MARKERS = [
    "NEXT_PUBLIC_APP_ENV=development",
    "NEXT_PUBLIC_ENV=development",
    "PUBLIC_ENV=development",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY=test",
    "NEXT_PUBLIC_SUPABASE_URL=http://",
]

def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix not in TEXT_EXTS:
        return False
    if ".git" in path.parts or "node_modules" in path.parts or ".netlify" in path.parts or "dist" in path.parts:
        return False
    return True

def scan_roots(roots, markers, label, allowlist_files=None):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not should_scan(path):
                continue
            if allowlist_files and path in allowlist_files:
                continue
            text = read_text(path)
            for marker in markers:
                if marker in text:
                    issues.append(f"{label} marker found in source/config {path}: {marker}")

print("--- Running underlying Signal Wall env-boundary audit ---")
env_audit = subprocess.run(
    [sys.executable, "scripts/audit_signal_wall_env_boundary.py"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(env_audit.stdout)
if env_audit.returncode != 0:
    issues.append("audit_signal_wall_env_boundary.py failed")

netlify = Path("netlify.toml")
netlify_text = read_text(netlify)

required_env = "python3 scripts/audit_signal_wall_env_boundary.py"
required_strict = "python3 scripts/audit_strict_environment_preflight.py"
build_entry = "python3 dashboard/build_all.py"

if required_env not in netlify_text:
    issues.append("netlify build command must run audit_signal_wall_env_boundary.py")
if required_strict not in netlify_text:
    issues.append("netlify build command must run audit_strict_environment_preflight.py")
if build_entry not in netlify_text:
    issues.append("netlify build command must run dashboard/build_all.py")
if all(x in netlify_text for x in [required_env, required_strict, build_entry]):
    if not (netlify_text.find(required_env) < netlify_text.find(required_strict) < netlify_text.find(build_entry)):
        issues.append("environment preflight audits must run before dashboard/build_all.py")

runner = Path("scripts/run_green_baseline_audit.sh")
runner_text = read_text(runner)
if required_strict not in runner_text:
    issues.append("green baseline must include audit_strict_environment_preflight.py")

scan_roots(SOURCE_SCAN_ROOTS, PRIVATE_SERVICE_MARKERS, "private service credential/env-name", PRIVATE_MARKER_ALLOWLIST_FILES)
scan_roots(SOURCE_SCAN_ROOTS, DEV_DEBUG_MARKERS, "development/debug", AUDIT_DEFINITION_ALLOWLIST_FILES)
scan_roots(SOURCE_SCAN_ROOTS, LOCAL_TEST_CONFIG_MARKERS, "local test config", AUDIT_DEFINITION_ALLOWLIST_FILES)
scan_roots(SOURCE_SCAN_ROOTS, PERMISSIVE_LOGGING_MARKERS, "permissive logging", AUDIT_DEFINITION_ALLOWLIST_FILES)
scan_roots(SOURCE_SCAN_ROOTS, NON_PRODUCTION_CLIENT_MARKERS, "non-production client bundle", AUDIT_DEFINITION_ALLOWLIST_FILES)

print("--- DiamondSignals strict environment preflight audit ---")
print(f"strict_environment_preflight_issues: {len(issues)}")

for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_STRICT_ENVIRONMENT_PREFLIGHT")
    sys.exit(1)

print("development_debug_flags_blocked: true")
print("local_test_configs_blocked: true")
print("permissive_logging_blocked: true")
print("service_credentials_exposure_blocked: true")
print("non_production_client_bundle_settings_blocked: true")
print("netlify_env_preflight_before_build: true")
print("preflight_scans_source_and_config_not_stale_dist: true")
print()
print("FINAL_STATUS: PASS_STRICT_ENVIRONMENT_PREFLIGHT")
