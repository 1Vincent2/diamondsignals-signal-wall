#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DOMAIN = "https://signals.diamondsignals.ai"

PUBLIC_PAGES = {
    "index.html": "/",
    "live/index.html": "/live/",
    "waiver-wire/index.html": "/waiver-wire/",
    "apex-extraction/index.html": "/apex-extraction/",
    "mlb-extraction/index.html": "/mlb-extraction/",
    "typical-call-up/index.html": "/typical-call-up/",
    "velocity-decay-monitor/index.html": "/velocity-decay-monitor/",
    "stuff-disruption-feed/index.html": "/stuff-disruption-feed/",
    "ivb-heat-map/index.html": "/ivb-heat-map/",
}

REQUIRED_HTML_TOKENS = [
    '<meta name="robots" content="index,follow">',
    '<meta name="description"',
    '<meta property="og:site_name" content="DiamondSignals">',
    '<meta property="og:type" content="website">',
    '<meta property="og:title"',
    '<meta property="og:description"',
    '<meta property="og:url"',
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta name="twitter:title"',
    '<meta name="twitter:description"',
    '<script type="application/ld+json">',
    '"@context": "https://schema.org"',
    '"publisher": {',
    '"@type": "Organization"',
]

FORBIDDEN_HTML_TOKENS = [
    "noindex",
    "nofollow",
    "https://diamondsignals.ai",
    "https://app.diamondsignals.ai",
]

def extract_json_ld(text: str) -> list[dict]:
    payloads = []
    pattern = re.compile(
        r'<script\s+type=["\']application/ld\+json["\']\s*>(.*?)</script>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        raw = match.group(1).strip()
        try:
            payloads.append(json.loads(raw))
        except Exception as exc:
            payloads.append({"__parse_error__": str(exc)})
    return payloads

def main() -> int:
    print("--- DiamondSignals all-domain SEO / structured data contract audit ---")
    issues: list[str] = []

    robots = DIST / "robots.txt"
    sitemap = DIST / "sitemap.xml"

    if not robots.exists():
        issues.append("missing dist/robots.txt")
        robots_text = ""
    else:
        robots_text = robots.read_text(encoding="utf-8", errors="ignore")

    for token in [
        "User-agent: *",
        "Allow: /",
        "Disallow: /admin/",
        "Disallow: /watch-list/",
        "Disallow: /status/",
        f"Sitemap: {DOMAIN}/sitemap.xml",
    ]:
        if token not in robots_text:
            issues.append(f"robots.txt missing required token: {token}")

    if not sitemap.exists():
        issues.append("missing dist/sitemap.xml")
        sitemap_urls = set()
    else:
        try:
            root = ET.parse(sitemap).getroot()
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            sitemap_urls = {
                (loc.text or "").strip()
                for loc in root.findall(".//sm:loc", ns)
                if (loc.text or "").strip()
            }
        except Exception as exc:
            issues.append(f"sitemap.xml parse error: {exc}")
            sitemap_urls = set()

    for rel, route in PUBLIC_PAGES.items():
        expected_url = f"{DOMAIN}{route}"
        path = DIST / rel

        if not path.exists():
            issues.append(f"missing public page: dist/{rel}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        if expected_url not in sitemap_urls:
            issues.append(f"sitemap missing public URL: {expected_url}")

        canonical = f'<link rel="canonical" href="{expected_url}">'
        if canonical not in text:
            issues.append(f"missing canonical URL in dist/{rel}: {expected_url}")

        head_match = re.search(r"<head[^>]*>(.*?)</head>", text, re.IGNORECASE | re.DOTALL)
        head = head_match.group(1) if head_match else ""
        if not head:
            issues.append(f"missing HTML head in dist/{rel}")

        for token in REQUIRED_HTML_TOKENS:
            if token not in head:
                issues.append(f"missing SEO token in dist/{rel}: {token}")

        for token in FORBIDDEN_HTML_TOKENS:
            if token in head:
                issues.append(f"forbidden SEO/index token in dist/{rel}: {token}")

        json_ld = extract_json_ld(head)
        if not json_ld:
            issues.append(f"missing parseable JSON-LD in dist/{rel}")
        else:
            valid_payload = False
            for payload in json_ld:
                if "__parse_error__" in payload:
                    issues.append(f"invalid JSON-LD in dist/{rel}: {payload['__parse_error__']}")
                    continue
                if (
                    payload.get("@context") == "https://schema.org"
                    and payload.get("url") == expected_url
                    and isinstance(payload.get("publisher"), dict)
                    and payload["publisher"].get("@type") == "Organization"
                ):
                    valid_payload = True
            if not valid_payload:
                issues.append(f"JSON-LD missing schema.org/url/publisher contract in dist/{rel}")

    forbidden_sitemap_fragments = [
        "/admin/",
        "/watch-list",
        "/status/",
        "/api/",
        "/auth",
        "/watchlist",
        "app.diamondsignals.ai",
    ]

    for url in sorted(sitemap_urls):
        if any(fragment in url for fragment in forbidden_sitemap_fragments):
            issues.append(f"sitemap exposes forbidden URL: {url}")

    print(f"seo_public_pages_checked: {len(PUBLIC_PAGES)}")
    print(f"sitemap_urls_checked: {len(sitemap_urls)}")
    print(f"seo_contract_issues: {len(issues)}")

    for issue in issues:
        print(f"FAIL: {issue}")

    if issues:
        print()
        print("FINAL_STATUS: FAIL_SIGNAL_WALL_ALL_DOMAIN_SEO_CONTRACT")
        return 1

    print("robots_txt_index_boundary_preserved: true")
    print("sitemap_public_routes_preserved: true")
    print("canonical_links_present: true")
    print("open_graph_metadata_present: true")
    print("twitter_metadata_present: true")
    print("schema_org_json_ld_present: true")
    print()
    print("FINAL_STATUS: PASS_SIGNAL_WALL_ALL_DOMAIN_SEO_CONTRACT")
    return 0

if __name__ == "__main__":
    sys.exit(main())
