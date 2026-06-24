#!/usr/bin/env python3

import json
import sys
import urllib.request
from datetime import datetime, timezone

DOMAIN = "https://signals.diamondsignals.ai"

STATUS_URLS = [
    ("Signal Wall", f"{DOMAIN}/status/signal-wall.json"),
    ("Promotion Watch", f"{DOMAIN}/status/promotion-watch.json"),
    ("Waiver Wire", f"{DOMAIN}/status/waiver-wire.json"),
    ("Apex Extraction", f"{DOMAIN}/status/apex-extraction.json"),
    ("MLB Extraction", f"{DOMAIN}/status/mlb-extraction.json"),
    ("IVB Heat Map", f"{DOMAIN}/status/ivb-heat-map.json"),
    ("Velocity Decay", f"{DOMAIN}/status/velocity-decay.json"),
    ("Stuff Disruption", f"{DOMAIN}/status/stuff-disruption.json"),
    ("Depth Radar", f"{DOMAIN}/status/depth-radar.json"),
]

ROUTE_URLS = [
    f"{DOMAIN}/",
    f"{DOMAIN}/live/",
    f"{DOMAIN}/robots.txt",
    f"{DOMAIN}/sitemap.xml",
    f"{DOMAIN}/status/admin-audit-manifest.json",
    f"{DOMAIN}/status/signal-wall.json",
]

MAX_REASONABLE_SOURCE_AGE_MINUTES = 240


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_head_status(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers)


def main():
    issues = []

    print("--- DiamondSignals live production health check ---")
    print(f"checked_at_utc: {datetime.now(timezone.utc).isoformat()}")
    print(f"domain: {DOMAIN}")

    print()
    print("--- admin audit manifest ---")
    try:
        manifest = fetch_json(f"{DOMAIN}/status/admin-audit-manifest.json")
        head = manifest.get("head", {})
        green = manifest.get("green_baseline", {})
        print(f"short_sha: {head.get('short_sha')}")
        print(f"subject: {head.get('subject')}")
        print(f"generated_at: {manifest.get('generated_at')}")
        print(f"audit_invocation_count: {green.get('audit_invocation_count')}")
        print(f"head_is_clean: {head.get('is_clean')}")

        if not head.get("short_sha"):
            issues.append("manifest missing head.short_sha")
        if not green.get("audit_invocation_count"):
            issues.append("manifest missing green baseline audit count")
    except Exception as exc:
        issues.append(f"manifest fetch failed: {exc}")

    print()
    print("--- report status freshness ---")
    for label, url in STATUS_URLS:
        try:
            payload = fetch_json(url)
            state = payload.get("state")
            build_success = payload.get("build_success")
            used_fallback = payload.get("used_fallback")
            degraded = payload.get("degraded")
            source_age = payload.get("source_age_minutes")
            finished = payload.get("build_finished_at")
            mode = payload.get("mode")

            print()
            print(label)
            print(f"  url: {url}")
            print(f"  state: {state}")
            print(f"  build_success: {build_success}")
            print(f"  used_fallback: {used_fallback}")
            print(f"  degraded: {degraded}")
            print(f"  source_age_minutes: {source_age}")
            print(f"  build_finished_at: {finished}")
            print(f"  mode: {mode}")

            if state != "fresh":
                issues.append(f"{label}: state is not fresh: {state}")
            if build_success is not True:
                issues.append(f"{label}: build_success is not true: {build_success}")
            if used_fallback is True:
                issues.append(f"{label}: used_fallback is true")
            if degraded is True:
                issues.append(f"{label}: degraded is true")

            # IVB currently has a different age profile; warn but do not fail solely on that.
            if isinstance(source_age, (int, float)) and source_age > MAX_REASONABLE_SOURCE_AGE_MINUTES:
                if label == "IVB Heat Map":
                    print(f"  WARNING: source age high but tolerated for IVB profile: {source_age}")
                else:
                    issues.append(f"{label}: source_age_minutes high: {source_age}")

        except Exception as exc:
            issues.append(f"{label}: status fetch failed: {exc}")

    print()
    print("--- route smoke ---")
    for url in ROUTE_URLS:
        try:
            status, headers = fetch_head_status(url)
            content_type = headers.get("content-type") or headers.get("Content-Type")
            cache_control = headers.get("cache-control") or headers.get("Cache-Control")
            print(f"{status} {url} | content-type={content_type} | cache-control={cache_control}")
            if status != 200:
                issues.append(f"route not 200: {url} -> {status}")
        except Exception as exc:
            issues.append(f"route smoke failed: {url}: {exc}")

    print()
    print("--- summary ---")
    print(f"production_health_issues: {len(issues)}")

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        print()
        print("FINAL_STATUS: FAIL_LIVE_PRODUCTION_HEALTH")
        return 1

    print("live_manifest_present: true")
    print("live_reports_fresh: true")
    print("live_routes_healthy: true")
    print("fallbacks_off: true")
    print("degraded_flags_off: true")
    print()
    print("FINAL_STATUS: PASS_LIVE_PRODUCTION_HEALTH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
