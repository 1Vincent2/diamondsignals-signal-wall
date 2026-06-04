#!/usr/bin/env bash
set -euo pipefail

WEEK_START="${1:-}"
SEASON="${2:-2026}"
BASE_URL="${BASE_URL:-http://localhost:8888}"

if [[ -z "$WEEK_START" ]]; then
  echo "Usage: scripts/run_aaa_weekly_ingest.sh YYYY-MM-DD [SEASON]"
  echo "Example: scripts/run_aaa_weekly_ingest.sh 2026-05-25 2026"
  exit 2
fi

if [[ -z "${ADMIN_RUN_TOKEN:-}" ]]; then
  echo "ERROR: ADMIN_RUN_TOKEN is not set in this shell."
  exit 2
fi

if [[ ! "$WEEK_START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "ERROR: WEEK_START must be YYYY-MM-DD. Got: $WEEK_START"
  exit 2
fi

LOG_DIR="/tmp/diamondsignals_aaa_weekly_ingest_${WEEK_START}"
mkdir -p "$LOG_DIR"

# Current approved AAA org/team map:
# 431  | Atlanta Braves       | Gwinnett Stripers
# 533  | Boston Red Sox       | Worcester Red Sox
# 451  | Chicago Cubs         | Iowa Cubs
# 512  | Detroit Tigers       | Toledo Mud Hens
# 5434 | Houston Astros       | Sugar Land Space Cowboys
# 238  | Los Angeles Dodgers  | Oklahoma City Comets
# 552  | New York Mets        | Syracuse Mets
# 531  | New York Yankees     | Scranton/Wilkes-Barre RailRiders
# 105  | San Francisco Giants | Sacramento River Cats
# 529  | Seattle Mariners     | Tacoma Rainiers
# 102  | Texas Rangers        | Round Rock Express

TEAM_IDS=(431 533 451 512 5434 238 552 531 105 529 102)

echo "--- AAA WEEKLY INGEST RUNNER ---"
echo "week_start=$WEEK_START"
echo "season=$SEASON"
echo "base_url=$BASE_URL"
echo "log_dir=$LOG_DIR"
echo "team_count=${#TEAM_IDS[@]}"
echo

for TEAM_ID in "${TEAM_IDS[@]}"; do
  echo "--- ingest team_id=$TEAM_ID week=$WEEK_START ---"

  OUT_FILE="${LOG_DIR}/aaa_weekly_ingest_${WEEK_START}_${TEAM_ID}.txt"

  curl -sS \
    "${BASE_URL}/.netlify/functions/ingest-milb-aaa-weekly?team_id=${TEAM_ID}&week_start=${WEEK_START}&season=${SEASON}&token=${ADMIN_RUN_TOKEN}" \
    | tee "$OUT_FILE"

  echo
  if grep -q "TimeoutError\|Failed to connect\|Missing/invalid\|Unauthorized\|upsert_errors=[1-9]" "$OUT_FILE"; then
    echo "ERROR: ingest appears to have failed for team_id=$TEAM_ID"
    echo "See: $OUT_FILE"
    exit 1
  fi

  sleep 2
done

echo "--- ingest complete for all teams ---"
echo "logs: $LOG_DIR"
