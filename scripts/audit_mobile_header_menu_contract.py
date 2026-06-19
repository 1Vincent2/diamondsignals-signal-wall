#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(".")
DIST = ROOT / "dist"

ROUTE_FILES = [
    DIST / "index.html",
    DIST / "live/index.html",
    DIST / "waiver-wire/index.html",
    DIST / "watch-list/index.html",
    DIST / "apex-extraction/index.html",
    DIST / "mlb-extraction/index.html",
    DIST / "typical-call-up/index.html",
    DIST / "velocity-decay-monitor/index.html",
    DIST / "stuff-disruption-feed/index.html",
    DIST / "ivb-heat-map/index.html",
    DIST / "hidden-gems/index.html",
    DIST / "admin/kinetic-drift/index.html",
]

OPTIONAL_COMPATIBILITY_ROUTE_FILES = {
    DIST / "hidden-gems/index.html",
}

MOBILE_BREAKPOINT_PATTERNS = [
    r"@media\s*\([^)]*max-width\s*:\s*768px",
    r"@media\s*\([^)]*max-width\s*:\s*760px",
    r"@media\s*\([^)]*max-width\s*:\s*720px",
    r"@media\s*\([^)]*max-width\s*:\s*640px",
    r"@media\s*\([^)]*max-width\s*:\s*600px",
]

HEADER_TERMS = [
    "mobile-menu",
    "hamburger",
    "ds-mobile-menu",
    "topnav",
    "mobile-nav",
    "menu-toggle",
    "field-guide",
    "drawer",
]

OBSTRUCTION_RISK_PATTERNS = [
    (r"position\s*:\s*fixed", "fixed positioning"),
    (r"position\s*:\s*absolute", "absolute positioning"),
    (r"z-index\s*:\s*[0-9]+", "explicit z-index"),
    (r"overflow\s*:\s*hidden", "overflow hidden"),
    (r"height\s*:\s*[0-9.]+px", "fixed pixel height"),
    (r"transform\s*:\s*translate", "translated drawer/menu"),
    (r"pointer-events\s*:\s*none", "pointer-events none"),
    (r"pointer-events\s*:\s*auto", "pointer-events auto"),
]

REQUIRED_SAFE_PATTERNS = [
    ("viewport", r'<meta\s+name=["\']viewport["\']'),
    ("mobile breakpoint", "|".join(MOBILE_BREAKPOINT_PATTERNS)),
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_no(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def compact(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def route_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit_route(path: Path) -> list[str]:
    problems = []

    if not path.exists():
        return [f"{route_label(path)}: missing route file"]

    html = read(path)
    rel = route_label(path)

    for label, pattern in REQUIRED_SAFE_PATTERNS:
        if not re.search(pattern, html, flags=re.I | re.S):
            problems.append(f"{rel}: missing {label}")

    has_any_header_term = any(term in html.lower() for term in HEADER_TERMS)
    if not has_any_header_term:
        # Landing / watch-list routes may intentionally have no full report header.
        if rel not in {"dist/index.html", "dist/watch-list/index.html"}:
            problems.append(f"{rel}: no recognizable mobile/header/menu terms found")

    # Report routes with mobile drawer/menu should have button semantics or clear toggles.
    if "ds-mobile-menu" in html or "mobile-menu" in html or "hamburger" in html:
        if not re.search(r"(aria-expanded|aria-controls|menu-toggle|hamburger|ds-mobile-menu-button)", html, flags=re.I):
            problems.append(f"{rel}: mobile menu exists but lacks obvious toggle/accessibility contract")

    # Detect field guide drawer routes with risky overlay mechanics but no document/body open-state.
    # Accept surface-specific implementations:
    # - document.body.classList.add/remove/toggle("field-guide-open")
    # - document.body.classList.add/remove/toggle("waiver-field-guide-open")
    # - document.body.classList.add/remove/toggle("pw-field-guide-open")
    # - document.body.classList.add/remove/toggle("kde-field-guide-open")
    # - any equivalent *-field-guide-open body/document class toggle
    if "field-guide" in html.lower() or "drawer" in html.lower():
        if "field-guide-open" in html:
            has_field_guide_open_toggle = re.search(
                r"(document\.body|body|document\.documentElement)?\s*\.?\s*classList\s*\.\s*(add|remove|toggle)\s*\(\s*['\"][a-z0-9_-]*field-guide-open['\"]",
                html,
                flags=re.I,
            )
            if not has_field_guide_open_toggle:
                problems.append(f"{rel}: field-guide-open class exists but no obvious document/body *-field-guide-open class toggle found")

    # Mobile obstruction risk summary: not failure by itself, but fail if many risks and no mobile breakpoint.
    risk_hits = []
    for pattern, label in OBSTRUCTION_RISK_PATTERNS:
        for m in re.finditer(pattern, html, flags=re.I):
            risk_hits.append((label, line_no(html, m.start()), compact(m.group(0))))

    has_mobile_breakpoint = any(re.search(p, html, flags=re.I) for p in MOBILE_BREAKPOINT_PATTERNS)

    if len(risk_hits) >= 12 and not has_mobile_breakpoint:
        sample = "; ".join(f"L{ln}:{label}" for label, ln, _ in risk_hits[:8])
        problems.append(f"{rel}: many overlay/layout risk rules but no recognized mobile breakpoint. sample: {sample}")

    # Specific title/header collision checks.
    has_report_title = re.search(r"(hero-title|section-title|report-title|signal-title|surface-title|page-title)", html, flags=re.I)
    has_fixed_header = re.search(r"(topnav|mobile-menu|site-header|ds-pro-header|header)[\s\S]{0,600}?position\s*:\s*fixed", html, flags=re.I)
    has_mobile_padding_guard = re.search(r"(padding-top|margin-top|scroll-padding-top|safe-area-inset-top)", html, flags=re.I)

    if has_report_title and has_fixed_header and not has_mobile_padding_guard:
        problems.append(f"{rel}: possible fixed-header/title collision; no obvious top padding/safe-area guard")

    return problems


def main() -> None:
    all_problems = []

    print("--- DiamondSignals mobile header/menu obstruction audit ---")

    checked = 0
    for path in ROUTE_FILES:
        if not path.exists():
            if path in OPTIONAL_COMPATIBILITY_ROUTE_FILES:
                print(f"SKIP: optional compatibility route not present: {route_label(path)}")
                continue

            all_problems.append(f"{route_label(path)}: missing expected route")
            continue

        checked += 1
        problems = audit_route(path)

        if problems:
            print(f"\n--- {route_label(path)} ---")
            for problem in problems:
                print("FAIL:", problem)
            all_problems.extend(problems)
        else:
            print(f"OK: {route_label(path)} mobile header/menu contract has no obvious obstruction flags")

    print("\n--- summary ---")
    print(f"routes_checked: {checked}")
    print(f"mobile_header_menu_issues: {len(all_problems)}")

    if all_problems:
        print("\nFINAL_STATUS: FAIL_MOBILE_HEADER_MENU_AUDIT")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_MOBILE_HEADER_MENU_AUDIT")


if __name__ == "__main__":
    main()
