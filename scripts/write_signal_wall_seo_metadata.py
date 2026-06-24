#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DOMAIN = "https://signals.diamondsignals.ai"

PAGES = {
    "index.html": {
        "path": "/",
        "title": "DiamondSignals // Signal Intelligence",
        "description": "DiamondSignals surfaces baseball signal intelligence, player movement, physics indicators, promotion pressure, and tracking-ready player reports.",
        "type": "WebSite",
    },
    "live/index.html": {
        "path": "/live/",
        "title": "DiamondSignals // Signal Wall",
        "description": "Live Signal Wall for baseball player signals, Statcast-backed movement, canonical dossier routing, and Tracking Radar handoff.",
        "type": "CollectionPage",
    },
    "waiver-wire/index.html": {
        "path": "/waiver-wire/",
        "title": "DiamondSignals // Waiver Wire",
        "description": "Waiver Wire signal surface for actionable player monitoring, roster pressure, and tracking-ready baseball intelligence.",
        "type": "CollectionPage",
    },
    "apex-extraction/index.html": {
        "path": "/apex-extraction/",
        "title": "DiamondSignals // Apex Extraction",
        "description": "Apex Extraction identifies high-conviction player movement, contact-quality, and signal acceleration patterns.",
        "type": "CollectionPage",
    },
    "mlb-extraction/index.html": {
        "path": "/mlb-extraction/",
        "title": "DiamondSignals // MLB Extraction",
        "description": "MLB Extraction connects Statcast-backed player indicators, canonical identity routing, and signal-led scouting context.",
        "type": "CollectionPage",
    },
    "typical-call-up/index.html": {
        "path": "/typical-call-up/",
        "title": "DiamondSignals // Promotion Watch",
        "description": "Promotion Watch tracks AAA-to-MLB movement, call-up pressure, recent arrivals, and player acceleration signals.",
        "type": "CollectionPage",
    },
    "velocity-decay-monitor/index.html": {
        "path": "/velocity-decay-monitor/",
        "title": "DiamondSignals // Velocity Decay Monitor",
        "description": "Velocity Decay Monitor tracks pitcher velocity loss, fatigue flags, movement decay, and risk escalation.",
        "type": "CollectionPage",
    },
    "stuff-disruption-feed/index.html": {
        "path": "/stuff-disruption-feed/",
        "title": "DiamondSignals // Stuff Disruption Feed",
        "description": "Stuff Disruption Feed surfaces pitcher movement, pitch-shape, and disruption signals from current baseball data.",
        "type": "CollectionPage",
    },
    "ivb-heat-map/index.html": {
        "path": "/ivb-heat-map/",
        "title": "DiamondSignals // IVB Heat Map",
        "description": "IVB Heat Map tracks induced vertical break, fastball carry, shape changes, and pitcher movement indicators.",
        "type": "CollectionPage",
    },
}

START = "<!-- DS SEO METADATA START -->"
END = "<!-- DS SEO METADATA END -->"

def esc(value: str) -> str:
    return html.escape(value, quote=True)

def canonical(path: str) -> str:
    return f"{DOMAIN}{path}"

def seo_block(meta: dict[str, str]) -> str:
    url = canonical(meta["path"])
    title = meta["title"]
    description = meta["description"]
    schema_type = meta["type"]

    schema = {
        "@context": "https://schema.org",
        "@type": schema_type,
        "name": title,
        "description": description,
        "url": url,
        "isPartOf": {
            "@type": "WebSite",
            "name": "DiamondSignals",
            "url": DOMAIN + "/",
        },
        "publisher": {
            "@type": "Organization",
            "name": "DiamondSignals",
            "url": DOMAIN + "/",
        },
    }

    if meta["path"] == "/":
        schema["potentialAction"] = {
            "@type": "SearchAction",
            "target": DOMAIN + "/?q={search_term_string}",
            "query-input": "required name=search_term_string",
        }

    return f"""{START}
<link rel="canonical" href="{esc(url)}">
<meta name="robots" content="index,follow">
<meta name="description" content="{esc(description)}">
<meta property="og:site_name" content="DiamondSignals">
<meta property="og:type" content="website">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{esc(url)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}">
<script type="application/ld+json">
{json.dumps(schema, ensure_ascii=False, indent=2)}
</script>
{END}"""

def inject(path: Path, meta: dict[str, str]) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    block = seo_block(meta)

    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END) + r"\n?",
        flags=re.DOTALL,
    )
    text = pattern.sub("", text)

    if "</head>" not in text:
        raise RuntimeError(f"missing </head>: {path.relative_to(ROOT)}")

    text = text.replace("</head>", block + "\n</head>", 1)
    path.write_text(text, encoding="utf-8")
    return True

def main() -> None:
    updated = 0
    missing = []

    for rel, meta in PAGES.items():
        path = DIST / rel
        if not path.exists():
            missing.append(rel)
            continue
        inject(path, meta)
        updated += 1
        print(f"OK: injected SEO metadata: dist/{rel}")

    print(f"seo_pages_updated: {updated}")
    print(f"seo_pages_missing: {len(missing)}")
    for rel in missing:
        print(f"WARN: missing SEO target: dist/{rel}")

    print("FINAL_STATUS: PASS_WRITE_SIGNAL_WALL_SEO_METADATA")

if __name__ == "__main__":
    main()
