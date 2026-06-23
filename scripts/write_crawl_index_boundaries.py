#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

SIGNAL_DOMAIN = "https://signals.diamondsignals.ai"

PUBLIC_SITEMAP_PATHS = [
    "/",
    "/live/",
    "/live-v2/",
    "/waiver-wire/",
    "/apex-extraction/",
    "/mlb-extraction/",
    "/hidden-gems/",
    "/typical-call-up/",
    "/velocity-decay-monitor/",
    "/stuff-disruption-feed/",
    "/ivb-heat-map/",
]

ROBOTS_TEXT = f"""User-agent: *
Allow: /

Disallow: /admin/
Disallow: /watch-list/
Disallow: /watch-list
Disallow: /status/
Disallow: /_headers

Sitemap: {SIGNAL_DOMAIN}/sitemap.xml
"""


def write_robots() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST / "robots.txt").write_text(ROBOTS_TEXT, encoding="utf-8")


def write_sitemap() -> None:
    DIST.mkdir(parents=True, exist_ok=True)

    urlset = ET.Element(
        "urlset",
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9",
    )

    for path in PUBLIC_SITEMAP_PATHS:
        url = ET.SubElement(urlset, "url")
        loc = ET.SubElement(url, "loc")
        loc.text = f"{SIGNAL_DOMAIN}{path}"

    tree = ET.ElementTree(urlset)
    ET.indent(tree, space="  ", level=0)
    tree.write(DIST / "sitemap.xml", encoding="utf-8", xml_declaration=True)


def main() -> None:
    write_robots()
    write_sitemap()
    print("OK: wrote dist/robots.txt")
    print("OK: wrote dist/sitemap.xml")
    print(f"sitemap_public_urls: {len(PUBLIC_SITEMAP_PATHS)}")
    print("FINAL_STATUS: PASS_WRITE_CRAWL_INDEX_BOUNDARIES")


if __name__ == "__main__":
    main()
