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

  function closeResults() {
    resultsEl.hidden = true;
    resultsEl.innerHTML = "";
    activeIndex = -1;
    currentResults = [];
  }

  function setLoading() {
    resultsEl.hidden = false;
    resultsEl.innerHTML = `<div class="player-search-loading">Searching players...</div>`;
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
        return `
          <a class="player-search-result" href="${player.profile_url}" data-index="${index}">
            <img
              class="player-search-avatar"
              src="${player.headshot_url}"
              alt="${escapeHtml(player.full_name)}"
              loading="lazy"
            />
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

  async function runSearch(query) {
    if (query.length < 2) {
      closeResults();
      return;
    }

    setLoading();

    try {
      const res = await fetch(`/.netlify/functions/player-search?q=${encodeURIComponent(query)}`);
      const payload = await res.json();

      if (!res.ok) {
        setError();
        return;
      }

      renderResults(Array.isArray(payload.results) ? payload.results : []);
    } catch (error) {
      setError();
    }
  }

  input.addEventListener("input", () => {
    const query = input.value.trim();

    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      runSearch(query);
    }, 250);
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

  input.addEventListener("focus", () => {
    if (currentResults.length) {
      resultsEl.hidden = false;
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