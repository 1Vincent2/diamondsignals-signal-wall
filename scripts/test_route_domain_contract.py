#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SIGNALS_HOST = "signals.diamondsignals.ai"
ROOT_HOST = "diamondsignals.ai"
APP_HOST = "app.diamondsignals.ai"

REPORT_ROUTE_FILES = [
    "dist/live/index.html",
    "dist/waiver-wire/index.html",
    "dist/watch-list/index.html",
    "dist/apex-extraction/index.html",
    "dist/mlb-extraction/index.html",
    "dist/typical-call-up/index.html",
    "dist/velocity-decay-monitor/index.html",
    "dist/stuff-disruption-feed/index.html",
    "dist/ivb-heat-map/index.html",
    "dist/admin/kinetic-drift/index.html",
]

# Root domain is allowed only as an explicit marketing-site nav link.
ALLOWED_ROOT_DOMAIN_SNIPPETS = {
    'href="https://diamondsignals.ai"',
    "href='https://diamondsignals.ai'",
    'href="https://diamondsignals.ai/"',
    "href='https://diamondsignals.ai/'",
}

FORBIDDEN_ROOT_DOMAIN_CONTEXTS = [
    r"fetch\(\s*['\"]https://diamondsignals\.ai",
    r"requests\.get\(\s*['\"]https://diamondsignals\.ai",
    r"urlopen\(\s*['\"]https://diamondsignals\.ai",
    r"curl\s+['\"]?https://diamondsignals\.ai",
    r"base_url\s*=.*https://diamondsignals\.ai",
    r"BASE_URL\s*=.*https://diamondsignals\.ai",
    r"PUBLIC_URL\s*=.*https://diamondsignals\.ai",
    r"SITE_URL\s*=.*https://diamondsignals\.ai",
    r"CANONICAL.*https://diamondsignals\.ai",
    r"canonical.*https://diamondsignals\.ai",
    r"og:url.*https://diamondsignals\.ai",
]

SOURCE_GLOBS = [
    "dashboard/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "*.sh",
    "*.toml",
    "*.json",
    "*.yml",
    "*.yaml",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_no(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def is_allowed_root_link(fragment: str) -> bool:
    return any(snippet in fragment for snippet in ALLOWED_ROOT_DOMAIN_SNIPPETS)


def audit_report_routes() -> list[str]:
    problems: list[str] = []

    for rel in REPORT_ROUTE_FILES:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: missing route HTML")
            continue

        html = read(path)

        # Report routes may link out to the marketing site, but should not use root
        # diamondsignals.ai for canonical/report fetch/verification behavior.
        for pattern in FORBIDDEN_ROOT_DOMAIN_CONTEXTS:
            match = re.search(pattern, html, flags=re.I)
            if match:
                problems.append(
                    f"{rel}:{line_no(html, match.start())}: forbidden root-domain functional reference: {match.group(0)[:120]}"
                )

        for match in re.finditer(r"https?://diamondsignals\.ai[^\"'\s<)]*", html, flags=re.I):
            start = max(0, match.start() - 80)
            end = min(len(html), match.end() + 80)
            fragment = html[start:end]

            if is_allowed_root_link(fragment):
                continue

            # app.diamondsignals.ai is not matched by this regex, but keep this defensive.
            if APP_HOST in fragment:
                continue

            problems.append(
                f"{rel}:{line_no(html, match.start())}: root-domain reference outside allowed Main Site nav link: {match.group(0)}"
            )

    return problems


def audit_source_verification_targets() -> list[str]:
    problems: list[str] = []

    files: set[Path] = set()
    for glob in SOURCE_GLOBS:
        files.update(ROOT.glob(glob))

    for path in sorted(files):
        if not path.is_file():
            continue
        if ".git" in path.parts or "node_modules" in path.parts:
            continue

        text = read(path)
        rel = path.relative_to(ROOT)

        for pattern in FORBIDDEN_ROOT_DOMAIN_CONTEXTS:
            for match in re.finditer(pattern, text, flags=re.I):
                problems.append(
                    f"{rel}:{line_no(text, match.start())}: forbidden root-domain verification/fetch reference: {match.group(0)[:140]}"
                )

    return problems


def main() -> None:
    problems = []
    problems.extend(audit_report_routes())
    problems.extend(audit_source_verification_targets())

    if problems:
        print("--- Route/domain contract failures ---")
        for problem in problems:
            print(problem)
        fail(f"{len(problems)} route/domain contract issue(s) found")

    print("OK: route/domain contract is clean. Reports do not use root diamondsignals.ai for verification, canonical, or functional report fetch targets.")


if __name__ == "__main__":
    main()
