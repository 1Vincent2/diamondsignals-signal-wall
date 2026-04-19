(function () {
  const CARD_SELECTOR = ".js-player-card";
  const WATCH_BUTTON_SELECTOR = ".js-add-to-roster";
  const STORAGE_KEY = "diamondsignals_watch_list_v1";
  const LEGACY_STORAGE_KEY = "diamondsignals_roster_v1";

  function readStorage(key) {
    try {
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function getWatchList() {
    const current = readStorage(STORAGE_KEY);
    if (current.length) return current;

    const legacy = readStorage(LEGACY_STORAGE_KEY);
    if (!legacy.length) return [];

    const migrated = legacy.map((p) => ({
      playerId: p.playerId || "",
      playerName: p.playerName || "",
      playerType: p.playerType || "",
      team: p.team || "",
      profileUrl: p.profileUrl || "",
      sourceTag: p.sourceTag || "FOLLOW",
      savedAt: p.savedAt || new Date().toISOString(),
    }));

    setWatchList(migrated);
    return migrated;
  }

  function setWatchList(watchList) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(watchList));
    } catch {
      // no-op
    }
  }

  function inferSourceTag(card, player) {
    const text = (card.innerText || "").toUpperCase();

    if ((player.playerType || "").toLowerCase() === "pitcher") return "LIVE_SIGNAL";
    if ((player.playerType || "").toLowerCase() === "hitter") return "LIVE_SIGNAL";
    if (text.includes("MOVEMENT_AUDIT") || text.includes("RECENT_ARRIVAL") || text.includes("DEBUT") || text.includes("RECALL")) {
      return "ARRIVAL";
    }
    return "FOLLOW";
  }

  function upsertWatchListPlayer(player) {
    const watchList = getWatchList();
    const existingIndex = watchList.findIndex((p) => {
      if (player.playerId && p.playerId) return String(p.playerId) === String(player.playerId);
      return String(p.playerName || "").toLowerCase() === String(player.playerName || "").toLowerCase();
    });

    if (existingIndex >= 0) {
      watchList[existingIndex] = { ...watchList[existingIndex], ...player, savedAt: new Date().toISOString() };
    } else {
      watchList.push({ ...player, savedAt: new Date().toISOString() });
    }

    setWatchList(watchList);
    return watchList;
  }

  function ensureToast() {
    let toast = document.getElementById("dsPlayerActionToast");
    if (toast) return toast;

    toast = document.createElement("div");
    toast.id = "dsPlayerActionToast";
    toast.setAttribute("aria-live", "polite");
    toast.style.position = "fixed";
    toast.style.right = "16px";
    toast.style.bottom = "16px";
    toast.style.zIndex = "9999";
    toast.style.padding = "10px 14px";
    toast.style.borderRadius = "12px";
    toast.style.border = "1px solid rgba(255,255,255,0.10)";
    toast.style.background = "linear-gradient(180deg, #121212 0%, #080808 100%)";
    toast.style.color = "#f0f0f0";
    toast.style.fontFamily = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    toast.style.fontSize = "13px";
    toast.style.boxShadow = "0 14px 34px rgba(0,0,0,0.34)";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(8px)";
    toast.style.transition = "opacity 180ms ease, transform 180ms ease";
    toast.style.pointerEvents = "none";
    document.body.appendChild(toast);
    return toast;
  }

  let toastTimer = null;

  function showToast(message) {
    const toast = ensureToast();
    toast.textContent = message;
    toast.style.opacity = "1";
    toast.style.transform = "translateY(0)";

    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
    }, 1800);
  }

  function openProfileFromCard(card) {
    const profileUrl = card.getAttribute("data-profile-url");
    if (!profileUrl) return;
    window.location.href = profileUrl;
  }

  function getPlayerFromCard(card) {
    const base = {
      playerId: card.getAttribute("data-player-id") || "",
      playerName: card.getAttribute("data-player-name") || "",
      playerType: card.getAttribute("data-player-type") || "",
      team: card.getAttribute("data-player-team") || "",
      profileUrl: card.getAttribute("data-profile-url") || "",
    };

    return {
      ...base,
      sourceTag: inferSourceTag(card, base),
    };
  }

  function bindCard(card) {
    if (!card || card.dataset.playerCardBound === "true") return;
    card.dataset.playerCardBound = "true";

    card.style.cursor = "pointer";

    card.addEventListener("click", function (event) {
      const watchButton = event.target.closest(WATCH_BUTTON_SELECTOR);
      if (watchButton) return;
      openProfileFromCard(card);
    });

    card.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        const watchButton = event.target.closest(WATCH_BUTTON_SELECTOR);
        if (watchButton) return;
        event.preventDefault();
        openProfileFromCard(card);
      }
    });

    if (!card.hasAttribute("tabindex")) {
      card.setAttribute("tabindex", "0");
    }

    const watchButton = card.querySelector(WATCH_BUTTON_SELECTOR);
    if (watchButton) {
      watchButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        const player = getPlayerFromCard(card);
        upsertWatchListPlayer(player);
        showToast(`${player.playerName || "Player"} added to Watch List`);
      });
    }
  }

  function initPlayerCardActions() {
    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));
    cards.forEach(bindCard);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPlayerCardActions);
  } else {
    initPlayerCardActions();
  }

  window.DiamondSignalsWatchList = {
    getWatchList,
    setWatchList,
  };
})();
