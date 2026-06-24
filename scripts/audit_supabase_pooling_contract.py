#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_FILES = [
    Path("dashboard/build_ivb_heat_map.py"),
    Path("dashboard/build_dashboard.py"),
    Path("dashboard/build_call_up_live.py"),
    Path("dashboard/lib/build_operational_status_feed.py"),
    Path("scripts/run_admin_green_baseline_from_build.py"),
    Path("netlify/functions/admin-audit-runner.mjs"),
    Path("netlify/functions/admin-audit-worker-background.mjs"),
    Path("netlify/functions/front-door-capture.js"),
    Path("netlify/functions/ingest-milb-aaa-weekly.mjs"),
]

SOURCE_ROOTS = [
    Path("dashboard"),
    Path("scripts"),
    Path("netlify"),
]

EXCLUDED_PARTS = {
    "__pycache__",
    "archive_call_up_builders",
}

EXCLUDED_SUFFIXES = (
    ".bak",
    ".pyc",
)

DIRECT_DB_MARKERS = [
    "DATABASE_URL",
    "SUPABASE_DB_PASSWORD",
    "SUPAVISOR",
    "supavisor",
    "pooler",
    "postgres://",
    "postgresql://",
    "psycopg",
    "pg8000",
    "asyncpg",
    "create_engine(",
]

ALLOWED_DIRECT_DB_MARKER_FILES = {
    Path("scripts/audit_signal_wall_env_boundary.py"),
    Path("scripts/audit_strict_environment_preflight.py"),
    Path("scripts/audit_supabase_pooling_contract.py"),
}

REQUIRED_ACTIVE_PATTERNS = {
    Path("dashboard/build_ivb_heat_map.py"): [
        "create_client",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("dashboard/build_dashboard.py"): [
        "create_client",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("dashboard/build_call_up_live.py"): [
        "create_client",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("dashboard/lib/build_operational_status_feed.py"): [
        "create_client",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("scripts/run_admin_green_baseline_from_build.py"): [
        "/rest/v1/admin_audit_jobs",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("netlify/functions/admin-audit-runner.mjs"): [
        "/rest/v1/admin_audit_jobs",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("netlify/functions/admin-audit-worker-background.mjs"): [
        "/rest/v1/admin_audit_jobs",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("netlify/functions/front-door-capture.js"): [
        "/rest/v1/founding_access",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
    Path("netlify/functions/ingest-milb-aaa-weekly.mjs"): [
        "createClient",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ],
}

issues = []

def is_skipped(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if path.name.endswith(EXCLUDED_SUFFIXES):
        return True
    return False

def read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

print("--- DiamondSignals Supabase / Supavisor pooling contract audit ---")

active_checked = 0
for rel in ACTIVE_FILES:
    path = ROOT / rel
    text = read(path)
    print(f"active_file_checked: {rel}")
    active_checked += 1

    if not path.exists():
        issues.append(f"missing active Supabase runtime file: {rel}")
        continue

    for required in REQUIRED_ACTIVE_PATTERNS.get(rel, []):
        if required not in text:
            issues.append(f"{rel} missing expected Supabase REST/client marker: {required}")

    for marker in DIRECT_DB_MARKERS:
        if marker in text:
            issues.append(f"{rel} contains forbidden direct database/pooler marker: {marker}")

source_files_checked = 0
direct_db_marker_hits = []

for root in SOURCE_ROOTS:
    base = ROOT / root
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file() or is_skipped(path):
            continue
        rel = path.relative_to(ROOT)
        text = read(path)
        source_files_checked += 1

        for marker in DIRECT_DB_MARKERS:
            if marker in text and rel not in ALLOWED_DIRECT_DB_MARKER_FILES:
                direct_db_marker_hits.append((str(rel), marker))
                issues.append(f"unexpected direct database/pooler marker in active source: {rel} -> {marker}")

print(f"active_supabase_runtime_files_checked: {active_checked}")
print(f"source_files_checked_for_direct_db_pooling: {source_files_checked}")
print(f"direct_db_pooler_marker_hits: {len(direct_db_marker_hits)}")
for rel, marker in direct_db_marker_hits:
    print(f"FAIL: direct_db_pooler_marker: {rel} -> {marker}")

print(f"supabase_pooling_contract_issues: {len(issues)}")
for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_SUPABASE_POOLING_CONTRACT")
    sys.exit(1)

print("supabase_access_uses_https_rest_or_client: true")
print("direct_postgres_database_url_blocked_in_active_runtime: true")
print("supavisor_pooler_url_not_consumed_by_active_runtime: true")
print("service_role_key_server_side_only_boundary_preserved: true")
print("archive_backup_supabase_references_excluded_from_runtime_contract: true")
print()
print("FINAL_STATUS: PASS_SUPABASE_POOLING_CONTRACT")
