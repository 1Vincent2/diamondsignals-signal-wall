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

def require_exists(rel: str, label: str) -> None:
    if not (ROOT / rel).exists():
        issues.append(f"{label} missing required file: {rel}")

require_includes("MOBILE_DEFERRED_UNTIL_ITEM21.md", "mobile deferred and report bottling policy", [
    "Mobile redesign is intentionally deferred until after Item 21.",
    "The core report surfaces are bottled.",
    "report builders, report data contracts, status payloads, tracking identity markup",
    "Desktop-only report-surface refinements must remain scoped",
    "Broad mobile layout changes are not allowed during Items 12–21",
    "protected by a mobile-specific audit",
    "After Item 21 is complete, the planned mobile redesign can begin only as a separate tracked phase",
    "mobile redesign should be additive or isolated where possible",
    "should not rewrite core report generation, data contracts, desktop report surfaces, status payloads, or tracking identity behavior",
])

require_includes("scripts/run_green_baseline_audit.sh", "green baseline mobile/deferred coverage", [
    "scripts/audit_mobile_header_menu_contract.py",
    "scripts/audit_desktop_mobile_deferred_contract.py",
])

require_includes("scripts/audit_mobile_header_menu_contract.py", "mobile header/menu guard", [
    "PASS_MOBILE_HEADER_MENU_AUDIT",
    "FAIL_MOBILE_HEADER_MENU_AUDIT",
    "mobile_header_menu_issues",
])

require_includes("scripts/test_ivb_heat_map_mobile_field_guide.py", "specific existing mobile exception audit", [
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_BOTTOM_PILL_V1",
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_CLOSE_X_V1",
    "OK: IVB mobile Field Guide bottom sticky audit passed",
])

desktop_audits = [
    ("scripts/test_velocity_decay_desktop_nav_parity.py", "Velocity Decay desktop nav parity audit"),
    ("scripts/test_velocity_decay_desktop_ui_polish.py", "Velocity Decay desktop UI polish audit"),
    ("scripts/test_mlb_extraction_desktop_nav_parity.py", "MLB Extraction desktop nav parity audit"),
    ("scripts/test_ivb_heat_map_desktop_nav_parity.py", "IVB desktop nav parity audit"),
    ("scripts/test_ivb_heat_map_desktop_ui_polish.py", "IVB desktop UI polish audit"),
]

for rel, label in desktop_audits:
    require_exists(rel, label)

require_includes("dashboard/templates/shell_styles.css", "shared shell desktop/mobile boundary", [
    "DS_PRO_DESKTOP_NAV_V2_NAV_ONLY",
    "@media screen and (min-width: 761px)",
    "@media screen and (max-width: 760px)",
    "topnav.ds-shell-nav",
])

require_includes("dashboard/templates/shell_nav.html", "mobile/shared shell nav template", [
    'topnav ds-shell-nav',
    'Tracking Radar',
    'Roster Terminal',
])

require_exists("dashboard/templates/shell_nav_v2.html", "desktop pro nav template")

require_includes("dashboard/build_ivb_heat_map.py", "known scoped mobile exception remains named", [
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_BOTTOM_PILL_V1",
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_CLOSE_X_V1",
    "IVB_HEAT_MAP_DESKTOP_UI_POLISH_V1",
    "Mobile layout, mobile menu, Field Guide modal behavior, and tracking remain untouched.",
])

if issues:
    print("--- DiamondSignals desktop/mobile deferred contract audit ---")
    print(f"desktop_mobile_deferred_issues: {len(issues)}")
    for issue in issues:
        print(f" - {issue}")
    print("\nFINAL_STATUS: FAIL_DESKTOP_MOBILE_DEFERRED_CONTRACT")
    sys.exit(1)

print("--- DiamondSignals desktop/mobile deferred contract audit ---")
print("mobile_redesign_deferred_until: after Item 21")
print("desktop_mobile_deferred_issues: 0")
print("\nFINAL_STATUS: PASS_DESKTOP_MOBILE_DEFERRED_CONTRACT")
