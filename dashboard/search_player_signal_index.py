#!/usr/bin/env python3
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "dist" / "admin" / "player_signal_index.json"

def key(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def operator_readout(player):
    reports = set(player.get("reports_triggered", []))
    metrics_text = " ".join(
        str(v).lower()
        for s in player.get("signals", [])
        for v in s.get("metrics", {}).values()
    )

    reads = []

    if "Velocity Decay" in reports:
        reads.append("DECAY RISK: velocity/perceived-velo degradation is present. Be careful chasing surface results.")

    if "Stuff+ Disruption" in reports:
        reads.append("SHAPE DISRUPTION: movement/shape indicators are active. This may be a real skills-change signal.")

    if "IVB Heat Map" in reports:
        reads.append("IVB WATCH: pitch-shape/vertical-break context is active. Check whether this supports or contradicts the box score.")

    if "Promotion Watch" in reports:
        reads.append("PROMOTION PRESSURE: player is appearing in call-up/proximity logic. Useful for stash or role-change discussion.")

    if "MLB Extraction Ledger" in reports or "Signal Wall" in reports:
        reads.append("EXTRACTION WINDOW: player has surfaced in DiamondSignals opportunity logic. Market may not be fully priced yet.")

    if "whiff" in metrics_text:
        reads.append("WHIFF SUPPORT: whiff language is present, which strengthens the case that performance is physics-backed.")

    if not reads:
        reads.append("NO STRONG READOUT: player appears in the index, but no major cross-report interpretation fired yet.")

    return reads[:5]


def reddit_talking_point(player):
    name = player.get("player_name", "This player")
    reports = set(player.get("reports_triggered", []))
    reads = operator_readout(player)

    if "Velocity Decay" in reports:
        return f"{name} is a spot where I’d be careful chasing the surface line. The underlying DiamondSignals read is flagging velocity/perceived-velo decay, which can turn a good box score into a lagging indicator pretty quickly."

    if "Stuff+ Disruption" in reports and "Promotion Watch" in reports:
        return f"{name} is interesting because this isn’t just box-score noise. There’s movement/shape disruption showing up, and he’s also appearing in promotion/role-pressure logic. That combination is worth tracking before the broader market fully reacts."

    if "Stuff+ Disruption" in reports or "IVB Heat Map" in reports:
        return f"{name} has some physics-backed movement in the profile. I’d focus less on the last box score and more on whether the pitch-shape/whiff indicators keep confirming the change."

    if "MLB Extraction Ledger" in reports or "Signal Wall" in reports:
        return f"{name} has surfaced in opportunity logic, which usually means the underlying indicators are moving before the market has fully priced it. I’d treat him as a watch/add candidate depending on league depth."

    return f"{name} is in the lookup, but I don’t see enough cross-report confirmation yet. I’d avoid overreacting unless the next signal confirms the box-score story."


def reddit_talking_point(player):
    name = player.get("player_name", "This player")
    reports = set(player.get("reports_triggered", []))
    reads = operator_readout(player)

    if "Velocity Decay" in reports:
        return f"{name} is a spot where I’d be careful chasing the surface line. The underlying DiamondSignals read is flagging velocity/perceived-velo decay, which can turn a good box score into a lagging indicator pretty quickly."

    if "Stuff+ Disruption" in reports and "Promotion Watch" in reports:
        return f"{name} is interesting because this isn’t just box-score noise. There’s movement/shape disruption showing up, and he’s also appearing in promotion/role-pressure logic. That combination is worth tracking before the broader market fully reacts."

    if "Stuff+ Disruption" in reports or "IVB Heat Map" in reports:
        return f"{name} has some physics-backed movement in the profile. I’d focus less on the last box score and more on whether the pitch-shape/whiff indicators keep confirming the change."

    if "MLB Extraction Ledger" in reports or "Signal Wall" in reports:
        return f"{name} has surfaced in opportunity logic, which usually means the underlying indicators are moving before the market has fully priced it. I’d treat him as a watch/add candidate depending on league depth."

    return f"{name} is in the lookup, but I don’t see enough cross-report confirmation yet. I’d avoid overreacting unless the next signal confirms the box-score story."

q = " ".join(sys.argv[1:]).strip()
if not q:
    raise SystemExit("Usage: python3 dashboard/search_player_signal_index.py player name")

data = json.loads(INDEX.read_text(encoding="utf-8"))
needle = key(q)

matches = [
    p for p in data["players"]
    if needle in key(p.get("player_name")) or needle in key(p.get("search_name"))
]

print(f"Query: {q}")
print(f"Matches: {len(matches)}")

for p in matches[:10]:
    print("\n" + "=" * 72)
    print(f"{p.get('player_name')} | {p.get('team') or 'TEAM ?'} | {p.get('position') or 'POS ?'}")
    print("Reports:", ", ".join(p.get("reports_triggered", [])))

    print("\nOperator Readout:")
    for read in operator_readout(p):
        print(f"  - {read}")

    print("\nReddit Talking Point:")
    print("  " + reddit_talking_point(p))

    print("\nSignals:")
    for s in p.get("signals", []):
        print(f"\n- {s.get('report_label')} / {s.get('section')}")
        metrics = s.get("metrics", {})
        for k, v in list(metrics.items())[:12]:
            print(f"  {k}: {v}")
