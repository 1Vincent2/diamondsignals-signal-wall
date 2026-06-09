#!/usr/bin/env python3
import json
from pathlib import Path

STATUS_PATH = Path("dist/status/promotion-watch.json")
PAYLOAD_PATH = Path("dist/typical-call-up/promotion_watch.json")
HTML_PATH = Path("dist/typical-call-up/index.html")

SECTION_KEYS = [
    "pitchers_72hr",
    "hitters_72hr",
    "pitchers_14day",
    "hitters_14day",
    "recent_arrivals",
    "depth_radar",
]

PLACEHOLDER_BY_SECTION = {
    "pitchers_72hr": "No 72 HR pitching prospect signals available.",
    "hitters_72hr": "No 72 HR hitting prospect signals available.",
    "pitchers_14day": "No 14 day pitching prospect signals available.",
    "hitters_14day": "No 14 day hitting prospect signals available.",
    "recent_arrivals": "No prospect-relevant MLB arrivals in the last 14 days.",
}

def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"FAIL: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def list_count(value) -> int:
    return len(value) if isinstance(value, list) else 0

def main() -> None:
    status = load_json(STATUS_PATH)
    payload = load_json(PAYLOAD_PATH)

    if not HTML_PATH.exists():
        raise SystemExit(f"FAIL: missing {HTML_PATH}")
    html = HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    if status.get("report_id") != "promotion_watch":
        raise SystemExit(f"FAIL: wrong report_id in status: {status.get('report_id')}")

    if status.get("state") != "fresh":
        raise SystemExit(f"FAIL: Promotion Watch status is not fresh: {status.get('state')}")

    if status.get("degraded"):
        raise SystemExit("FAIL: Promotion Watch is degraded")

    if status.get("used_fallback"):
        raise SystemExit("FAIL: Promotion Watch used fallback")

    status_counts = status.get("section_counts") or {}
    payload_counts = payload.get("section_counts") or {}
    top_signals = payload.get("top_signals") or {}

    failures = []

    for key in SECTION_KEYS:
        status_count = int(status_counts.get(key, 0) or 0)
        payload_status_count = int(payload_counts.get(key, 0) or 0)

        direct_count = list_count(payload.get(key))
        nested_count = list_count(top_signals.get(key)) if isinstance(top_signals, dict) else 0

        if status_count != payload_status_count:
            failures.append(
                f"{key}: status section_count {status_count} != payload section_count {payload_status_count}"
            )

        if status_count > 0 and direct_count == 0:
            failures.append(f"{key}: status populated but direct payload export is empty")

        if status_count > 0 and nested_count == 0:
            failures.append(f"{key}: status populated but nested top_signals export is empty")

        placeholder = PLACEHOLDER_BY_SECTION.get(key)
        if status_count > 0 and placeholder and placeholder in html:
            failures.append(f"{key}: populated status but empty-state placeholder rendered")

    if status_counts.get("depth_radar", 0) and "AAA GEMS" not in html:
        failures.append("depth_radar populated but AAA GEMS label not rendered")

    if "Depth Radar" in html or "DEPTH RADAR" in html:
        failures.append("legacy Depth Radar label leaked into rendered Promotion Watch page")

    if failures:
        raise SystemExit("FAIL: Promotion Watch payload contract audit failed:\\n- " + "\\n- ".join(failures))

    print("OK: Promotion Watch status, payload, nested export, direct export, and rendered HTML are aligned.")

if __name__ == "__main__":
    main()
