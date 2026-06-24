from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
issues = []

def read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        issues.append(f"missing required file: {rel}")
        return ""
    return path.read_text(errors="ignore")

def require_includes(rel: str, label: str, fragments: list[str]) -> None:
    text = read(rel)
    for fragment in fragments:
        if fragment not in text:
            issues.append(f"{label} missing fragment in {rel}: {fragment}")

require_includes("scripts/run_green_baseline_audit.sh", "green baseline audit runner", [
    "Promotion Watch integrity audit",
    "build output freshness audit",
    "status contract audit",
    "admin audit manifest builder",
    "admin audit manifest contract audit",
    "tracking regression audit",
])

require_includes("scripts/run_admin_green_baseline_from_build.py", "Netlify build audit observability", [
    "Admin audit job detected",
    "Green baseline audit started in Netlify build runtime.",
    "Green baseline audit passed.",
    "Green baseline audit failed. Review stdout/stderr payload.",
    "signals_netlify_build_runtime",
    "admin_audit_jobs",
])

require_includes("scripts/build_admin_audit_manifest.py", "admin audit manifest builder", [
    "admin_audit_manifest_v1",
    "audit_invocations",
    "audit_registry_ready",
    "FINAL_STATUS: PASS_BUILD_ADMIN_AUDIT_MANIFEST",
])

require_includes("scripts/audit_admin_audit_manifest.py", "admin audit manifest contract", [
    "FINAL_STATUS: PASS_ADMIN_AUDIT_MANIFEST",
    "FAIL_ADMIN_AUDIT_MANIFEST",
    "audit_invocation_count",
])

require_includes("scripts/audit_build_output_freshness.py", "build failure/freshness observability", [
    "FINAL_STATUS: PASS_BUILD_OUTPUT_FRESHNESS",
    "FAIL_BUILD_OUTPUT_FRESHNESS",
    "build_output_freshness_issues",
])

require_includes("scripts/audit_status_contract.py", "status payload observability", [
    "build_success",
    "build_started_at",
    "build_finished_at",
    "FINAL_STATUS: PASS_STATUS_CONTRACT",
    "FAIL_STATUS_CONTRACT",
])

require_includes("scripts/audit_report_payload_null_safety.py", "API/payload failure observability", [
    "FINAL_STATUS: PASS_PAYLOAD_NULL_SAFETY",
    "FAIL_PAYLOAD_NULL_SAFETY",
])

require_includes("scripts/audit_signal_wall_tracking_regression_lock.py", "tracking write handoff observability", [
    "FINAL_STATUS: PASS_SIGNAL_WALL_TRACKING_REGRESSION_LOCK",
    "FAIL",
    "APP_WATCHLIST_STATUS_URL",
])

require_includes("src/js/player-card-actions.js", "tracking status client observability", [
    "APP_WATCHLIST_STATUS_URL",
    "catch",
    "tracking-active",
    "data-tracking-state",
])

require_includes("scripts/audit_signal_wall_env_boundary.py", "environment boundary observability", [
    "FINAL_STATUS: PASS_SIGNAL_WALL_ENV_BOUNDARY",
    "FAIL_SIGNAL_WALL_ENV_BOUNDARY",
])

require_includes("scripts/audit_signal_wall_seo_boundary.py", "SEO boundary observability", [
    "FINAL_STATUS: PASS_SIGNAL_WALL_SEO_BOUNDARY",
    "FAIL_SIGNAL_WALL_SEO_BOUNDARY",
])

if issues:
    print("--- DiamondSignals Signal Wall observability contract audit ---")
    print(f"signal_wall_observability_issues: {len(issues)}")
    for issue in issues:
        print(f" - {issue}")
    print("\nFINAL_STATUS: FAIL_SIGNAL_WALL_OBSERVABILITY_CONTRACT")
    sys.exit(1)

print("--- DiamondSignals Signal Wall observability contract audit ---")
print("signal_wall_observability_issues: 0")
print("\nFINAL_STATUS: PASS_SIGNAL_WALL_OBSERVABILITY_CONTRACT")
