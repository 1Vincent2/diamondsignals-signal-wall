function initPlayerSearch() {
  const searchRoot = document.getElementById("playerSearch");
  const input = document.getElementById("playerSearchInput");
  const resultsEl = document.getElementById("playerSearchResults");

  if (!searchRoot || !input || !resultsEl) {
    return;
  }

  let debounceTimer = null;
  let activeIndex = -1;
  let currentResults = [];
  let playerIndex = [];
  let playerIndexLoaded = false;
  let playerIndexLoading = false;
  let playerIndexPromise = null;

  function closeResults() {
    resultsEl.hidden = true;
    resultsEl.innerHTML = "";
    activeIndex = -1;
    currentResults = [];
  }

  function setLoading(message = "Searching players...") {
    resultsEl.hidden = false;
    resultsEl.innerHTML = `<div class="player-search-loading">${escapeHtml(message)}</div>`;
    activeIndex = -1;
    currentResults = [];
  }

  function setEmpty() {
    resultsEl.hidden = false;
    resultsEl.innerHTML = `<div class="player-search-empty">No players found</div>`;
    activeIndex = -1;
    currentResults = [];
  }

  function setError() {
    resultsEl.hidden = false;
    resultsEl.innerHTML = `<div class="player-search-error">Search temporarily unavailable</div>`;
    activeIndex = -1;
    currentResults = [];
  }

  function renderResults(results) {
    currentResults = results.slice();
    activeIndex = -1;

    if (!results.length) {
      setEmpty();
      return;
    }

    resultsEl.hidden = false;
    resultsEl.innerHTML = results
      .map((player, index) => {
        const sub = [player.team, player.position].filter(Boolean).join(" • ");
        const headshotUrl = player.headshot_url || "";
        return `
          <a class="player-search-result" href="${escapeHtml(player.profile_url)}" data-index="${index}">
            ${
              headshotUrl
                ? `<img
                    class="player-search-avatar"
                    src="${escapeHtml(headshotUrl)}"
                    alt="${escapeHtml(player.full_name)}"
                    loading="lazy"
                  />`
                : `<div class="player-search-avatar" aria-hidden="true"></div>`
            }
            <div class="player-search-meta">
              <div class="player-search-name">${escapeHtml(player.full_name)}</div>
              <div class="player-search-sub">${escapeHtml(sub || "PLAYER")}</div>
            </div>
          </a>
        `;
      })
      .join("");
  }

  function updateActiveResult() {
    const nodes = Array.from(resultsEl.querySelectorAll(".player-search-result"));
    nodes.forEach((node, index) => {
      node.classList.toggle("active", index === activeIndex);
    });
  }

  function normalizePlayer(player) {
    const playerId = String(player?.player_id || "").trim();
    return {
      player_id: playerId,
      full_name: String(player?.full_name || "Unknown Player"),
      first_name: String(player?.first_name || ""),
      last_name: String(player?.last_name || ""),
      team: String(player?.team || ""),
      team_name: String(player?.team_name || ""),
      position: String(player?.position || ""),
      headshot_url: String(player?.headshot_url || ""),
      profile_url: playerId ? `/scout/${playerId}/` : "/scout/",
    };
  }

  async function loadPlayerIndex() {
    if (playerIndexLoaded) {
      return playerIndex;
    }

    if (playerIndexLoading && playerIndexPromise) {
      return playerIndexPromise;
    }

    playerIndexLoading = true;
    playerIndexPromise = fetch("/player_index.json", { cache: "force-cache" })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Could not load player_index.json (${res.status})`);
        }
        const payload = await res.json();
        const players = Array.isArray(payload?.players) ? payload.players : [];
        playerIndex = players.map(normalizePlayer);
        playerIndexLoaded = true;
        return playerIndex;
      })
      .finally(() => {
        playerIndexLoading = false;
      });

    return playerIndexPromise;
  }

  function matchesQuery(player, q) {
    const fullName = player.full_name.toLowerCase();
    const firstName = player.first_name.toLowerCase();
    const lastName = player.last_name.toLowerCase();
    const team = player.team.toLowerCase();
    const teamName = player.team_name.toLowerCase();
    const position = player.position.toLowerCase();
    const playerId = player.player_id.toLowerCase();

    return (
      fullName.includes(q) ||
      firstName.includes(q) ||
      lastName.includes(q) ||
      team.includes(q) ||
      teamName.includes(q) ||
      position.includes(q) ||
      playerId.includes(q)
    );
  }

  function rankResults(results, q) {
    const startsWith = [];
    const wordStarts = [];
    const contains = [];

    for (const player of results) {
      const name = player.full_name.toLowerCase();
      const first = player.first_name.toLowerCase();
      const last = player.last_name.toLowerCase();

      if (
        name.startsWith(q) ||
        first.startsWith(q) ||
        last.startsWith(q)
      ) {
        startsWith.push(player);
      } else if (
        name.split(/\s+/).some((part) => part.startsWith(q))
      ) {
        wordStarts.push(player);
      } else {
        contains.push(player);
      }
    }

    return [...startsWith, ...wordStarts, ...contains].slice(0, 8);
  }

  async function runSearch(query) {
    if (query.length < 2) {
      closeResults();
      return;
    }

    setLoading(playerIndexLoaded ? "Searching players..." : "Loading player index...");

    try {
      const players = await loadPlayerIndex();
      const q = query.toLowerCase();

      const results = rankResults(
        players.filter((player) => matchesQuery(player, q)),
        q
      );

      renderResults(results);
    } catch (error) {
      setError();
    }
  }

  input.addEventListener("input", () => {
    const query = input.value.trim();

    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      runSearch(query);
    }, 120);
  });

  input.addEventListener("keydown", (event) => {
    const nodes = Array.from(resultsEl.querySelectorAll(".player-search-result"));

    if (event.key === "Escape") {
      closeResults();
      return;
    }

    if (!nodes.length) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = activeIndex < nodes.length - 1 ? activeIndex + 1 : 0;
      updateActiveResult();
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = activeIndex > 0 ? activeIndex - 1 : nodes.length - 1;
      updateActiveResult();
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      const targetNode = activeIndex >= 0 ? nodes[activeIndex] : nodes[0];
      if (targetNode) {
        window.location.href = targetNode.getAttribute("href");
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!searchRoot.contains(event.target)) {
      closeResults();
    }
  });

  input.addEventListener("focus", async () => {
    if (currentResults.length) {
      resultsEl.hidden = false;
      return;
    }

    if (input.value.trim().length >= 2) {
      await runSearch(input.value.trim());
      return;
    }

    if (!playerIndexLoaded) {
      loadPlayerIndex().catch(() => {});
    }
  });

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initPlayerSearch);
} else {
  initPlayerSearch();
}