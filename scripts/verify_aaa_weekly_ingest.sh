#!/usr/bin/env bash
set -euo pipefail

WEEK_START="${1:-}"

if [[ -z "$WEEK_START" ]]; then
  echo "Usage: scripts/verify_aaa_weekly_ingest.sh YYYY-MM-DD"
  echo "Example: scripts/verify_aaa_weekly_ingest.sh 2026-05-25"
  exit 2
fi

if [[ ! "$WEEK_START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "ERROR: WEEK_START must be YYYY-MM-DD. Got: $WEEK_START"
  exit 2
fi

if [[ -z "${SUPABASE_URL:-}" || -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
  echo "ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in this shell."
  exit 2
fi

echo "--- AAA WEEKLY VERIFY RUNNER ---"
echo "week_start=$WEEK_START"
echo

echo "--- verify raw weekly rows landed ---"
python3 - <<PY
import os
import pandas as pd
from supabase import create_client

WEEK_START = "${WEEK_START}"

EXPECTED_ORGS = {
    "Atlanta Braves",
    "Boston Red Sox",
    "Chicago Cubs",
    "Detroit Tigers",
    "Houston Astros",
    "Los Angeles Dodgers",
    "New York Mets",
    "New York Yankees",
    "San Francisco Giants",
    "Seattle Mariners",
    "Texas Rangers",
}

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

resp = (
    sb.table("milb_raw_weekly")
    .select("week_start,level,org_mlb_team,player_id,player_name,position_group,pa,bf,iso,so_p,bb_allowed,updated_at")
    .eq("level", "AAA")
    .eq("week_start", WEEK_START)
    .limit(5000)
    .execute()
)

rows = resp.data or []
print("raw_rows:", len(rows))

if not rows:
    raise SystemExit("FAIL: no raw weekly rows found")

df = pd.DataFrame(rows)

org_counts = df.groupby("org_mlb_team", dropna=False).size().sort_values(ascending=False)
print("\\norg_counts:")
print(org_counts.to_string())

found_orgs = set(df["org_mlb_team"].dropna().astype(str).unique())
missing_orgs = sorted(EXPECTED_ORGS - found_orgs)
unexpected_orgs = sorted(found_orgs - EXPECTED_ORGS)

hitters = int(df["pa"].notna().sum()) if "pa" in df.columns else 0
pitchers = int(df["bf"].notna().sum()) if "bf" in df.columns else 0

print("\\nhitters:", hitters)
print("pitchers:", pitchers)
print("missing_orgs:", missing_orgs)
print("unexpected_orgs:", unexpected_orgs)

if missing_orgs:
    raise SystemExit(f"FAIL: missing expected orgs: {missing_orgs}")

if unexpected_orgs:
    raise SystemExit(f"FAIL: unexpected orgs present: {unexpected_orgs}")

if hitters < 40:
    raise SystemExit(f"FAIL: hitter stat coverage too low: {hitters}")

if pitchers < 5:
    raise SystemExit(f"FAIL: pitcher stat coverage too low: {pitchers}")

print("\\nPASS: raw weekly ingest coverage looks healthy.")
PY

echo
echo "--- rebuild AAA weekly signal-base preview ---"
PYTHONPATH=. python3 dashboard/build_aaa_weekly_signal_base.py

echo
echo "--- inspect signal-base preview for target week ---"
python3 - <<PY
import pandas as pd
from pathlib import Path

WEEK_START = "${WEEK_START}"
p = Path("dist/aaa_weekly_signal_base_preview.csv")

if not p.exists():
    raise SystemExit("FAIL: missing dist/aaa_weekly_signal_base_preview.csv")

df = pd.read_csv(p)
target = df[df["week_start"].astype(str) == WEEK_START].copy()

print("preview_rows_total:", len(df))
print("target_week_rows:", len(target))

if target.empty:
    raise SystemExit(f"FAIL: no signal-base preview rows for {WEEK_START}")

hitters = int(target["pa"].notna().sum()) if "pa" in target.columns else 0
pitchers = int(target["bf"].notna().sum()) if "bf" in target.columns else 0

print("target_hitters:", hitters)
print("target_pitchers:", pitchers)

if hitters < 40:
    raise SystemExit(f"FAIL: signal-base hitter rows too low: {hitters}")

if pitchers < 5:
    raise SystemExit(f"FAIL: signal-base pitcher rows too low: {pitchers}")

print("\\nweek_counts:")
print(df.groupby("week_start").size().sort_index(ascending=False).head(10).to_string())

print("\\nPASS: signal-base preview contains target week.")
PY

echo
echo "--- rebuild Promotion Watch live payload ---"
PYTHONPATH=. python3 dashboard/build_call_up_live.py

echo
echo "--- validate Promotion Watch payload freshness and stale-date guard ---"
python3 - <<'PY'
import json
from pathlib import Path

payload_path = Path("dist/typical-call-up/promotion_watch.json")
status_path = Path("dist/status/promotion-watch.json")

payload = json.loads(payload_path.read_text(encoding="utf-8"))
status = json.loads(status_path.read_text(encoding="utf-8"))

top = payload.get("top_signals") or {}
sections = payload.get("section_counts") or {}
layers = payload.get("pipeline_layers") or []
notes = payload.get("hardening_notes") or []

print("payload_generated_at:", payload.get("generated_at"))
print("payload_status:", payload.get("status"))
print("payload_mode:", payload.get("mode"))
print("status_state:", status.get("state"))
print("section_counts:", sections)
print("pipeline_layers:", layers)

required_sections = [
    "pitchers_72hr",
    "hitters_72hr",
    "pitchers_14day",
    "hitters_14day",
    "recent_arrivals",
    "depth_radar",
]

missing = [s for s in required_sections if s not in top]
print("missing_sections:", missing)

if missing:
    raise SystemExit(f"FAIL: missing required sections: {missing}")

for section in required_sections:
    rows = top.get(section) or []
    print(section, "rows:", len(rows))

for section in ["pitchers_14day", "hitters_14day"]:
    dates = sorted({str(r.get("week_start")) for r in top.get(section, []) if r.get("week_start")}, reverse=True)
    print(section, "week_start_values:", dates)

bad_dates = ["2025-06-06", "2025-06-02", "2026-04-20"]
text = payload_path.read_text(encoding="utf-8")
bad_found = [d for d in bad_dates if d in text]
print("bad_dates_found:", bad_found)

if payload.get("status") != "fresh":
    raise SystemExit("FAIL: payload status is not fresh")

if "aaa_weekly_window_age_guard" not in layers:
    raise SystemExit("FAIL: aaa_weekly_window_age_guard missing from pipeline_layers")

if not any("older than 28 days" in str(note) for note in notes):
    raise SystemExit("FAIL: 28-day freshness guard note missing")

if bad_found:
    raise SystemExit(f"FAIL: stale dates found in payload: {bad_found}")

if len(top.get("pitchers_14day") or []) < 8:
    raise SystemExit("FAIL: pitchers_14day unexpectedly thin")

if len(top.get("hitters_14day") or []) < 8:
    raise SystemExit("FAIL: hitters_14day unexpectedly thin")

print("PASS: Promotion Watch payload QA clean.")
PY

echo
echo "--- final repo status ---"
git status --short
