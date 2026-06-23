#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

SIGNAL_DOMAIN = "https://signals.diamondsignals.ai"

ROBOTS = DIST / "robots.txt"
SITEMAP = DIST / "sitemap.xml"

REQUIRED_ROBOTS_LINES = [
    "User-agent: *",
    "Allow: /",
    "Disallow: /admin/",
    "Disallow: /watch-list/",
    "Disallow: /watch-list",
    "Disallow: /status/",
    f"Sitemap: {SIGNAL_DOMAIN}/sitemap.xml",
]

REQUIRED_SITEMAP_URLS = {
    f"{SIGNAL_DOMAIN}/",
    f"{SIGNAL_DOMAIN}/live/",
    f"{SIGNAL_DOMAIN}/live-v2/",
    f"{SIGNAL_DOMAIN}/waiver-wire/",
    f"{SIGNAL_DOMAIN}/apex-extraction/",
    f"{SIGNAL_DOMAIN}/mlb-extraction/",
    f"{SIGNAL_DOMAIN}/hidden-gems/",
    f"{SIGNAL_DOMAIN}/typical-call-up/",
    f"{SIGNAL_DOMAIN}/velocity-decay-monitor/",
    f"{SIGNAL_DOMAIN}/stuff-disruption-feed/",
    f"{SIGNAL_DOMAIN}/ivb-heat-map/",
}

FORBIDDEN_SITEMAP_FRAGMENTS = [
    "/admin/",
    "/watch-list",
    "/api/",
    "/auth",
    "/terminal",
    "/watchlist",
    "app.diamondsignals.ai",
]


def main() -> int:
    print("--- DiamondSignals crawl/index boundary audit ---")
    issues: list[str] = []

    if not ROBOTS.exists():
        issues.append("missing dist/robots.txt")
        robots_text = ""
    else:
        robots_text = ROBOTS.read_text(encoding="utf-8", errors="ignore")

    for line in REQUIRED_ROBOTS_LINES:
        if line not in robots_text:
            issues.append(f"robots.txt missing required line: {line}")

    if not SITEMAP.exists():
        issues.append("missing dist/sitemap.xml")
        sitemap_urls = set()
    else:
        try:
            root = ET.parse(SITEMAP).getroot()
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            sitemap_urls = {
                (loc.text or "").strip()
                for loc in root.findall(".//sm:loc", ns)
                if (loc.text or "").strip()
            }
        except Exception as exc:
            issues.append(f"sitemap.xml parse error: {exc}")
            sitemap_urls = set()

    print(f"sitemap_urls: {len(sitemap_urls)}")

    missing = sorted(REQUIRED_SITEMAP_URLS - sitemap_urls)
    extra_private = sorted(
        url
        for url in sitemap_urls
        if any(fragment in url for fragment in FORBIDDEN_SITEMAP_FRAGMENTS)
    )

    for url in missing:
        issues.append(f"sitemap missing public URL: {url}")

    for url in extra_private:
        issues.append(f"sitemap contains forbidden private/internal URL: {url}")

    print("\n--- summary ---")
    print(f"crawl_index_boundary_issues: {len(issues)}")

    if issues:
        for issue in issues:
            print("FAIL:", issue)
        print("\nFINAL_STATUS: FAIL_CRAWL_INDEX_BOUNDARIES")
        return 1

    print("\nFINAL_STATUS: PASS_CRAWL_INDEX_BOUNDARIES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
