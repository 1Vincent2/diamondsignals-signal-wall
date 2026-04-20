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
    const explicit = (card.getAttribute("data-source-tag") || "").trim().toUpperCase();
    if (explicit) return explicit;

    const text = (card.innerText || "").toUpperCase();
    if (text.includes("MOVEMENT_AUDIT") || text.includes("RECENT_ARRIVAL") || text.includes("DEBUT") || text.includes("RECALL")) {
      return "ARRIVAL";
    }
    if ((player.playerType || "").toLowerCase() === "pitcher") return "LIVE_SIGNAL";
    if ((player.playerType || "").toLowerCase() === "hitter") return "LIVE_SIGNAL";
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

  function isPlayerProvisioned(player) {
    const watchList = getWatchList();
    return watchList.some((p) => {
      if (player.playerId && p.playerId) return String(p.playerId) === String(player.playerId);
      return String(p.playerName || "").toLowerCase() === String(player.playerName || "").toLowerCase();
    });
  }

  function getDefaultProvisionLabel(button) {
    const explicit = (button?.getAttribute("data-default-label") || "").trim();
    return explicit || "PROVISION";
  }

  function applyProvisionedState(button, isProvisioned) {
    if (!button) return;
    button.textContent = isProvisioned ? "PROVISIONED" : getDefaultProvisionLabel(button);
    button.setAttribute("data-provisioned", isProvisioned ? "true" : "false");
    button.classList.toggle("is-provisioned", !!isProvisioned);
    button.setAttribute("aria-pressed", isProvisioned ? "true" : "false");
    button.disabled = !!isProvisioned;
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

  function syncCardProvisionStates() {
    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));
    cards.forEach((card) => {
      const watchButton = card.querySelector(WATCH_BUTTON_SELECTOR);
      if (!watchButton) return;
      const player = getPlayerFromCard(card);
      applyProvisionedState(watchButton, isPlayerProvisioned(player));
    });
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
      const initialPlayer = getPlayerFromCard(card);
      applyProvisionedState(watchButton, isPlayerProvisioned(initialPlayer));

      watchButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (watchButton.disabled || watchButton.getAttribute("data-provisioned") === "true") {
          return;
        }

        const player = getPlayerFromCard(card);
        const playerId = String(player.playerId || "").trim();

        if (!playerId) {
          showToast(`Unable to provision ${player.playerName || "player"}: missing player id`);
          return;
        }

        const authUrl = new URL("https://app.diamondsignals.ai/auth");
        authUrl.searchParams.set("next", "/watch-list");
        authUrl.searchParams.set("add_player_id", playerId);

        window.location.href = authUrl.toString();
      });
    }
  }

  function initPlayerCardActions() {
    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));
    cards.forEach(bindCard);
    syncCardProvisionStates();
  }

  function syncProvisionUI() {
    syncCardProvisionStates();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPlayerCardActions);
  } else {
    initPlayerCardActions();
  }

  window.addEventListener("focus", syncProvisionUI);

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) syncProvisionUI();
  });

  window.addEventListener("storage", function (event) {
    if (!event || event.key === STORAGE_KEY || event.key === LEGACY_STORAGE_KEY) {
      syncProvisionUI();
    }
  });

  window.DiamondSignalsWatchList = {
    getWatchList,
    setWatchList,
    syncProvisionUI,
  };
})();
