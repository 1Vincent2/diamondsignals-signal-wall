from pathlib import Path
from jinja2 import Template

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
WATCH_LIST_DIR = DIST_DIR / "watch-list"
TEMPLATES_DIR = REPO_ROOT / "dashboard" / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // Tracking Radar</title>
  <style>
{{ shell_styles | safe }}

    :root {
      --bg: #080808;
      --surface: #121212;
      --card-radial: radial-gradient(circle at top left, #171717 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #7c7c84;
      --blue: #6aa6ff;
      --red: #ef4444;
      --gold: #fbbf24;
      --shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
      --radius: 18px;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); }
    body {
      background:
        radial-gradient(circle at top left, rgba(106,166,255,0.06), transparent 24%),
        radial-gradient(circle at top right, rgba(239,68,68,0.04), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }

    .app {
      width: min(1180px, calc(100% - 24px));
      margin: 0 auto;
      padding: 18px 0 36px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-bottom: 16px;
    }

    .hero-card,
    .summary-card,
    .panel {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      background: var(--card-radial);
      box-shadow: var(--shadow);
    }

    .hero-card {
      padding: 18px;
    }

    .eyebrow {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--blue);
      margin-bottom: 10px;
    }

    .hero-title {
      margin: 0 0 10px 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 0.96;
      letter-spacing: -0.04em;
      font-weight: 900;
      text-transform: uppercase;
    }

    .hero-sub {
      margin: 0;
      color: var(--soft);
      font-size: 14px;
      line-height: 1.55;
      max-width: 64ch;
    }

    .summary-card {
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-content: start;
    }

    .summary-label {
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--tiny);
    }

    .summary-value {
      font-size: 28px;
      line-height: 1;
      font-weight: 800;
    }

    .panel {
      padding: 16px;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 14px;
    }

    .panel-kicker {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--blue);
      margin-bottom: 4px;
    }

    .panel-title {
      margin: 0;
      font-size: 24px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -0.03em;
      text-transform: uppercase;
    }

    .panel-badge {
      min-height: 32px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--soft);
      white-space: nowrap;
    }

    .placeholder {
      border: 1px dashed rgba(255,255,255,0.12);
      border-radius: 14px;
      padding: 18px;
      color: var(--soft);
      font-size: 14px;
      line-height: 1.6;
      background: rgba(255,255,255,0.02);
    }

    @media (max-width: 900px) {
      .hero {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  {{ nav_html | safe }}
  {{ search_html | safe }}

  <div class="app">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Roster Intelligence // Staging Layer</div>
        <h1 class="hero-title">Tracking Radar</h1>
        <p class="hero-sub">
          Central surveillance queue for players tracked from Promotion Watch, Signal Wall, and future report surfaces.
          This page is the staging layer before players are assigned into one or more rosters inside the Roster Terminal.
        </p>
      </div>

      <div class="summary-card">
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value">LOCAL</div>
        </div>
        <div>
          <div class="summary-label">Saved Players</div>
          <div class="summary-value" id="watchlist-count">0</div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-kicker">Tracking Queue</div>
          <h2 class="panel-title">Tracked Players</h2>
        </div>
        <div class="panel-badge" id="watchlist-status">Awaiting Local Data</div>
      </div>

      <div id="watchlist-root" class="placeholder">
        No players loaded yet. Initiate tracking on players from Promotion Watch or Signal Wall, then return here.
      </div>
    </section>

    {{ footer_html | safe }}
  </div>

  <script>
    (function () {
      const STORAGE_KEY = "diamondsignals_watch_list_v1";

      function getWatchList() {
        try {
          const raw = window.localStorage.getItem(STORAGE_KEY);
          const parsed = raw ? JSON.parse(raw) : [];
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      }

      function escapeHtml(value) {
        return String(value || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function removeWatchListPlayer(playerKey) {
        const rows = getWatchList();
        const next = rows.filter((player) => {
          if (player.playerId) return String(player.playerId) !== String(playerKey);
          return String(player.playerName || "") !== String(playerKey);
        });

        try {
          window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {}

        renderWatchList();
      }

      function renderWatchList() {
        const root = document.getElementById("watchlist-root");
        const countEl = document.getElementById("watchlist-count");
        const statusEl = document.getElementById("watchlist-status");
        const rows = getWatchList();

        if (countEl) countEl.textContent = String(rows.length);
        if (statusEl) statusEl.textContent = rows.length ? `LOCAL CACHE • ${rows.length}` : "AWAITING LOCAL DATA";

        if (!root) return;

        if (!rows.length) {
          root.className = "placeholder";
          root.innerHTML = "No players loaded yet. Initiate tracking on players from Promotion Watch or Signal Wall, then return here.";
          return;
        }

        const cards = rows
          .slice()
          .sort((a, b) => String(b.savedAt || "").localeCompare(String(a.savedAt || "")))
          .map((row) => {
            const name = escapeHtml(row.playerName || "Unknown");
            const type = escapeHtml((row.playerType || "unknown").toUpperCase());
            const team = escapeHtml(row.team || "—");
            const source = escapeHtml(row.sourceTag || "FOLLOW");
            const savedAt = escapeHtml(row.savedAt || "—");
            const profileUrl = row.profileUrl ? escapeHtml(row.profileUrl) : "#";

            return `
              <article style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px; background:rgba(255,255,255,0.02);">
                <div style="display:flex; justify-content:space-between; gap:12px; align-items:start;">
                  <div>
                    <div style="font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--blue); margin-bottom:6px;">Watch Asset</div>
                    <div style="font-size:24px; font-weight:800; letter-spacing:-0.03em; line-height:1;">${name}</div>
                    <div style="margin-top:8px; color:var(--soft); font-size:13px; line-height:1.5;">${type} // ${team} // ${source}</div>
                  </div>
                  <div style="display:flex; flex-direction:column; gap:8px; align-items:flex-end;">
                    <div style="min-height:32px; padding:0 12px; border-radius:999px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.03); display:inline-flex; align-items:center; justify-content:center; font-family:var(--mono); font-size:10px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); white-space:nowrap;">${source}</div>
                    <a href="${profileUrl}" style="min-height:34px; padding:0 12px; border-radius:10px; border:1px solid rgba(96,165,250,0.32); background:rgba(37,99,235,0.95); color:white; text-decoration:none; display:inline-flex; align-items:center; justify-content:center; font-family:var(--mono); font-size:10px; font-weight:800; letter-spacing:.05em; text-transform:uppercase; white-space:nowrap;">Open Dossier</a>
                  </div>
                </div>
                <div style="margin-top:12px; display:flex; align-items:center; justify-content:space-between; gap:12px;">
                  <div style="font-family:var(--mono); font-size:11px; color:var(--tiny); letter-spacing:.04em;">Saved: ${savedAt}</div>
                  <button
                    type="button"
                    data-remove-player="${escapeHtml(row.playerId || row.playerName || "")}"
                    style="min-height:32px; width:148px; min-width:148px; max-width:148px; padding:0 10px; border-radius:4px; border:1px solid rgba(255,255,255,0.14); background:rgba(255,255,255,0.04); color:#ffffff; font-family:var(--mono); font-size:10px; font-weight:800; letter-spacing:.05em; text-transform:uppercase; cursor:pointer;"
                  >REMOVE_ASSET</button>
                </div>
              </article>
            `;
          })
          .join("");

        root.className = "";
        root.innerHTML = `<div style="display:grid; grid-template-columns:1fr; gap:12px;">${cards}</div>`;

        root.querySelectorAll("[data-remove-player]").forEach((button) => {
          button.addEventListener("click", function () {
            removeWatchListPlayer(button.getAttribute("data-remove-player") || "");
          });
        });
      }

      function syncWatchListView() {
        renderWatchList();
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", syncWatchListView);
      } else {
        syncWatchListView();
      }

      window.addEventListener("focus", syncWatchListView);

      document.addEventListener("visibilitychange", function () {
        if (!document.hidden) syncWatchListView();
      });

      window.addEventListener("storage", function (event) {
        if (!event || event.key === STORAGE_KEY) {
          syncWatchListView();
        }
      });
    })();
  </script>
</body>
</html>
"""
)

def render_html() -> str:
    return HTML_TEMPLATE.render(
        nav_html=Template(NAV_TEMPLATE).render(active_nav="watch_list"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
    )

def main() -> None:
    WATCH_LIST_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WATCH_LIST_DIR / "index.html"
    output_path.write_text(render_html(), encoding="utf-8")
    print(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
