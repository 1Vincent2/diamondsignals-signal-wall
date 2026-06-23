from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
robots_path = ROOT / "dist" / "robots.txt"
sitemap_path = ROOT / "dist" / "sitemap.xml"

failures = []

if not robots_path.exists():
    failures.append("dist/robots.txt missing")
else:
    robots = robots_path.read_text(encoding="utf-8")
    required_robot_lines = [
        "User-agent: *",
        "Allow: /",
        "Disallow: /admin/",
        "Disallow: /watch-list/",
        "Disallow: /watch-list",
        "Disallow: /status/",
        "Sitemap: https://signals.diamondsignals.ai/sitemap.xml",
    ]
    for line in required_robot_lines:
        if line not in robots:
            failures.append(f"robots.txt missing required line: {line}")

    forbidden_robot_lines = [
        "Disallow: /live/",
        "Disallow: /waiver-wire/",
        "Disallow: /apex-extraction/",
        "Disallow: /hidden-gems/",
        "Disallow: /ivb-heat-map/",
    ]
    for line in forbidden_robot_lines:
        if line in robots:
            failures.append(f"robots.txt wrongly blocks public report path: {line}")

if not sitemap_path.exists():
    failures.append("dist/sitemap.xml missing")
else:
    sitemap = sitemap_path.read_text(encoding="utf-8")
    required_urls = [
        "https://signals.diamondsignals.ai/",
        "https://signals.diamondsignals.ai/live/",
        "https://signals.diamondsignals.ai/live-v2/",
        "https://signals.diamondsignals.ai/waiver-wire/",
        "https://signals.diamondsignals.ai/apex-extraction/",
        "https://signals.diamondsignals.ai/mlb-extraction/",
        "https://signals.diamondsignals.ai/hidden-gems/",
        "https://signals.diamondsignals.ai/typical-call-up/",
        "https://signals.diamondsignals.ai/velocity-decay-monitor/",
        "https://signals.diamondsignals.ai/stuff-disruption-feed/",
        "https://signals.diamondsignals.ai/ivb-heat-map/",
    ]
    for url in required_urls:
        if f"<loc>{url}</loc>" not in sitemap:
            failures.append(f"sitemap.xml missing public URL: {url}")

    forbidden_urls = [
        "https://signals.diamondsignals.ai/admin/",
        "https://signals.diamondsignals.ai/watch-list/",
        "https://signals.diamondsignals.ai/status/",
        "https://app.diamondsignals.ai",
    ]
    for url in forbidden_urls:
        if url in sitemap:
            failures.append(f"sitemap.xml exposes private/disallowed URL: {url}")

    try:
        ET.fromstring(sitemap)
    except ET.ParseError as exc:
        failures.append(f"sitemap.xml is not valid XML: {exc}")

if failures:
    print("FINAL_STATUS: FAIL_SIGNAL_WALL_SEO_BOUNDARY")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("FINAL_STATUS: PASS_SIGNAL_WALL_SEO_BOUNDARY")
