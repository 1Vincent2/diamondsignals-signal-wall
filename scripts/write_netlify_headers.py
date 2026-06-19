#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(parents=True, exist_ok=True)

headers = """# DiamondSignals publish headers

/dossier_canon.json
  Access-Control-Allow-Origin: https://app.diamondsignals.ai
  Access-Control-Allow-Methods: GET, HEAD, OPTIONS
  Access-Control-Allow-Headers: Content-Type

/scout_metrics.json
  Access-Control-Allow-Origin: https://app.diamondsignals.ai
  Access-Control-Allow-Methods: GET, HEAD, OPTIONS
  Access-Control-Allow-Headers: Content-Type

/player_index.json
  Access-Control-Allow-Origin: https://app.diamondsignals.ai
  Access-Control-Allow-Methods: GET, HEAD, OPTIONS
  Access-Control-Allow-Headers: Content-Type

/admin/player_signal_index.json
  Access-Control-Allow-Origin: https://app.diamondsignals.ai
  Access-Control-Allow-Methods: GET, HEAD, OPTIONS
  Access-Control-Allow-Headers: Content-Type
"""

out = DIST / "_headers"
out.write_text(headers, encoding="utf-8")
print(f"Wrote {out}")
