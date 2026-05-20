import re
import json
import urllib.request
from datetime import datetime

BASE = "https://signals.diamondsignals.ai"
URL = f"{BASE}/hidden-gems/?v={int(datetime.now().timestamp())}"

html = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8", errors="replace")

ids = re.findall(r'data-profile-url="/scout/([0-9]+)/"', html)
ids = list(dict.fromkeys(ids))

print("--- MLB Extraction parsed JSON audit-link hardening check ---")
print("mlb_extraction_url:", URL)
print("audit_link_count:", len(ids))
print("first_10_ids:", ids[:10])

failures = []
nav_artifacts = []
missing_support = []
missing_scout = []
profile_counts = {}

for pid in ids:
    scout_url = f"{BASE}/scout/{pid}/?v={int(datetime.now().timestamp())}"

    try:
        page = urllib.request.urlopen(scout_url, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e:
        failures.append((pid, "FETCH_FAILED", str(e)))
        continue

    if "supportMetricsCard" not in page or "renderSupportMetrics" not in page:
        failures.append((pid, "MISSING_SUPPORT_RENDER", scout_url))

    if "COMMAND NAV" in page or "playerSearch" in page:
        nav_artifacts.append((pid, scout_url))

    m = re.search(r"window\.__DS_SCOUT_PLAYER__ = (.*?);", page, re.S)
    if not m:
        failures.append((pid, "NO_PLAYER_JSON", scout_url))
        continue

    try:
        player = json.loads(m.group(1))
    except Exception as e:
        failures.append((pid, "PLAYER_JSON_PARSE_FAILED", str(e)))
        continue

    support = player.get("support_metrics")
    scout = player.get("scout_metrics")

    if not support:
        missing_support.append((pid, player.get("player_name"), player.get("position"), scout_url))
    else:
        profile = support.get("profile") or "NO_PROFILE_LABEL"
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

    if not scout:
        missing_scout.append((pid, player.get("player_name"), player.get("position"), scout_url))

print("\n--- profile counts ---")
print(json.dumps(profile_counts, indent=2, sort_keys=True))

print("\n--- failures ---")
print("count:", len(failures))
for item in failures[:20]:
    print(item)

print("\n--- nav artifacts ---")
print("count:", len(nav_artifacts))
for item in nav_artifacts[:20]:
    print(item)

print("\n--- missing support_metrics ---")
print("count:", len(missing_support))
for item in missing_support[:20]:
    print(item)

print("\n--- missing scout_metrics ---")
print("count:", len(missing_scout))
for item in missing_scout[:20]:
    print(item)

status = (
    len(ids) > 0
    and len(failures) == 0
    and len(nav_artifacts) == 0
    and len(missing_support) == 0
)

print("\nFINAL_STATUS:", "PASS" if status else "CHECK")

if not status:
    raise SystemExit(1)
