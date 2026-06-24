#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess
import sys

issues = []

PUBLIC_DIRS = [
    Path("dist"),
    Path("dashboard/templates"),
    Path("src"),
    Path("netlify/request-gates"),
]

PRIVATE_ENV_MARKERS = [
    "SUPABASE_SERVICE_ROLE_KEY",
    "ADMIN_RUN_TOKEN",
    "NETLIFY_BUILD_HOOK_URL",
    "NETLIFY_GREEN_BASELINE_AUDIT_BUILD_HOOK_URL",
    "YAHOO_CLIENT_ID",
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
    "console.debug(",
    "debugger;",
]

LOCAL_TEST_CONFIG_MARKERS = [
    "localhost:",
    "127.0.0.1",
    "0.0.0.0",
    ".env.local",
    ".env.development",
    "test_config",
    "local_test",
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
    "NEXT_PUBLIC_SUPABASE_URL=http://",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY=test",
    "NEXT_PUBLIC_APP_ENV=development",
    "NEXT_PUBLIC_ENV=development",
    "PUBLIC_ENV=development",
]

SAFE_TEXT_EXTS = {
    ".py", ".js", ".mjs", ".ts", ".tsx", ".html", ".css", ".json",
    ".toml", ".md", ".txt", ".yml", ".yaml", ".sh"
}

ALLOWLIST_FILES = {
    Path("scripts/audit_signal_wall_env_boundary.py"),
    Path("scripts/audit_strict_environment_preflight.py"),
    Path("scripts/audit_admin_build_runtime_job_contract.py"),
    Path("scripts/audit_netlify_green_baseline_gate.py"),
    Path("netlify.toml"),
}

def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def scan_paths(markers, label):
    for root in PUBLIC_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SAFE_TEXT_EXTS:
                continue
            if path in ALLOWLIST_FILES:
                continue
            text = read_text(path)
            for marker in markers:
                if marker in text:
                    issues.append(f"{label} marker leaked in {path}: {marker}")

def scan_repo_source(markers, label):
    for path in Path(".").rglob("*"):
        if ".git" in path.parts or "node_modules" in path.parts or ".netlify" in path.parts:
            continue
        if not path.is_file() or path.suffix not in SAFE_TEXT_EXTS:
            continue
        if path in ALLOWLIST_FILES:
            continue
        text = read_text(path)
        for marker in markers:
            if marker in text:
                issues.append(f"{label} marker found in source {path}: {marker}")

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

required_preflight = "python3 scripts/audit_signal_wall_env_boundary.py"
strict_preflight = "python3 scripts/audit_strict_environment_preflight.py"
build_entry = "python3 dashboard/build_all.py"

if required_preflight not in netlify_text:
    issues.append("netlify build command must run audit_signal_wall_env_boundary.py before build")
if strict_preflight not in netlify_text:
    issues.append("netlify build command must run audit_strict_environment_preflight.py before build")
if build_entry not in netlify_text:
    issues.append("netlify build command must include dashboard/build_all.py")

if all(x in netlify_text for x in [required_preflight, strict_preflight, build_entry]):
    if not (netlify_text.find(required_preflight) < netlify_text.find(strict_preflight) < netlify_text.find(build_entry)):
        issues.append("environment preflights must run before dashboard/build_all.py")

scan_paths(PRIVATE_ENV_MARKERS, "private service credential/env-name")
scan_paths(DEV_DEBUG_MARKERS, "development/debug")
scan_paths(LOCAL_TEST_CONFIG_MARKERS, "local test config")
scan_paths(PERMISSIVE_LOGGING_MARKERS, "permissive logging")
scan_paths(NON_PRODUCTION_CLIENT_MARKERS, "non-production client bundle")

scan_repo_source(PERMISSIVE_LOGGING_MARKERS, "permissive logging")
scan_repo_source(NON_PRODUCTION_CLIENT_MARKERS, "non-production client setting")

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
print()
print("FINAL_STATUS: PASS_STRICT_ENVIRONMENT_PREFLIGHT")
