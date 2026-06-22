(function () {
  const CARD_SELECTOR = ".js-player-card";
  const WATCH_BUTTON_SELECTOR = ".js-add-to-roster";
  const STORAGE_KEY = "diamondsignals_watch_list_v1";
  const LEGACY_STORAGE_KEY = "diamondsignals_roster_v1";
  const APP_AUTH_URL = "https://app.diamondsignals.ai/auth";
  const APP_WATCHLIST_PATH = "/watchlist";
  const APP_WATCHLIST_STATUS_URL = "https://app.diamondsignals.ai/api/watchlist/status";
  const appWatchlistStatusCache = new Map();
  const appWatchlistStatusPending = new Set();

  function readStorage(key) {
    try {
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function retireLegacyWatchListStorage() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
      window.localStorage.removeItem(LEGACY_STORAGE_KEY);
    } catch {
      // no-op
    }
  }

  function getWatchList() {
    retireLegacyWatchListStorage();
    return [];
  }

  function setWatchList(_watchList) {
    retireLegacyWatchListStorage();
    return [];
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

  function upsertWatchListPlayer(_player) {
    retireLegacyWatchListStorage();
    return [];
  }

  function isPlayerProvisioned(_player) {
    retireLegacyWatchListStorage();
    return false;
  }

  function getDefaultProvisionLabel(button) {
    const explicit = (button?.getAttribute("data-default-label") || "").trim();
    return explicit || "INITIATE TRACKING";
  }

  function applyProvisionedState(button, isProvisioned) {
    if (!button) return;

    if (button.getAttribute("data-opening-passport") === "true") {
      return;
    }

    button.textContent = isProvisioned ? "TRACKING ACTIVE" : getDefaultProvisionLabel(button);
    button.setAttribute("data-provisioned", isProvisioned ? "true" : "false");
    button.classList.toggle("is-provisioned", !!isProvisioned);
    button.setAttribute("aria-pressed", isProvisioned ? "true" : "false");
    button.disabled = !!isProvisioned;
  }

  function getPlayerStatusKey(player) {
    const playerId = String(player?.playerId || "").trim();
    return playerId || "";
  }

  function applyCardProvisionState(card, button, isProvisioned) {
    applyProvisionedState(button, isProvisioned);

    card.classList.toggle("is-provisioned", !!isProvisioned);
    card.classList.toggle("tracking-active", !!isProvisioned);
    card.setAttribute("data-provisioned", isProvisioned ? "true" : "false");
    card.setAttribute("data-tracking-state", isProvisioned ? "active" : "idle");
  }

  function fetchAppWatchlistStatus(player) {
    const playerId = getPlayerStatusKey(player);
    if (!playerId || appWatchlistStatusPending.has(playerId)) return;

    appWatchlistStatusPending.add(playerId);

    const url = new URL(APP_WATCHLIST_STATUS_URL);
    url.searchParams.set("player_id", playerId);

    fetch(url.toString(), {
      method: "GET",
      credentials: "include",
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    })
      .then((response) => {
        if (!response.ok) return null;
        return response.json();
      })
      .then((payload) => {
        if (!payload || payload.ok !== true) return;

        const isTracked = Boolean(payload.tracked || payload.in_watchlist || payload.exists);
        appWatchlistStatusCache.set(playerId, isTracked);

        Array.from(document.querySelectorAll(WATCH_BUTTON_SELECTOR)).forEach((button) => {
          const card = button.closest(CARD_SELECTOR) || button.closest(".asset-card") || button.closest("article") || button.parentElement;
          if (!card) return;

          const cardPlayer = getPlayerFromCard(card, button);
          if (getPlayerStatusKey(cardPlayer) !== playerId) return;

          applyCardProvisionState(card, button, isTracked);
        });
      })
      .catch(() => {
        // If the app session is unavailable or the request fails, leave the Signal Wall button in its safe default state.
      })
      .finally(() => {
        appWatchlistStatusPending.delete(playerId);
      });
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
    const profileUrl = (card.getAttribute("data-profile-url") || "").trim();
    if (!profileUrl || profileUrl === "#") {
      return;
    }
    window.location.href = profileUrl;
  }

  function getPlayerFromCard(card, button) {
    const source = button || card;

    const base = {
      playerId: source.getAttribute("data-player-id") || card.getAttribute("data-player-id") || "",
      playerName: source.getAttribute("data-player-name") || card.getAttribute("data-player-name") || "",
      playerType: source.getAttribute("data-player-type") || card.getAttribute("data-player-type") || card.getAttribute("data-player-role") || "",
      team: source.getAttribute("data-player-team") || card.getAttribute("data-player-team") || "",
      profileUrl: source.getAttribute("data-profile-url") || card.getAttribute("data-profile-url") || "",
    };

    return {
      ...base,
      sourceTag: inferSourceTag(card, base),
    };
  }

  function buildAppTrackingUrl(player) {
    const playerId = String(player.playerId || "").trim();
    const params = new URLSearchParams();

    params.set("add_player_id", playerId);

    if (player.playerName) {
      params.set("player_name", player.playerName);
    }

    if (player.team) {
      params.set("player_team", player.team);
    }

    params.set("signal_source", player.sourceTag || "Signal Wall");

    const nextPath = `${APP_WATCHLIST_PATH}?${params.toString()}`;
    const authUrl = new URL(APP_AUTH_URL);
    authUrl.searchParams.set("next", nextPath);
    return authUrl.toString();
  }

  function syncCardProvisionStates() {
    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));

    Array.from(document.querySelectorAll(WATCH_BUTTON_SELECTOR)).forEach((button) => {
      const card = button.closest(CARD_SELECTOR) || button.closest(".asset-card") || button.closest("article") || button.parentElement;
      if (card && !cards.includes(card)) cards.push(card);
    });

    cards.forEach((card) => {
      const watchButton = card.querySelector(WATCH_BUTTON_SELECTOR);
      if (!watchButton) return;

      const player = getPlayerFromCard(card, watchButton);
      const playerStatusKey = getPlayerStatusKey(player);
      const cachedProvisioned = playerStatusKey ? appWatchlistStatusCache.get(playerStatusKey) === true : false;

      applyCardProvisionState(card, watchButton, cachedProvisioned);
      fetchAppWatchlistStatus(player);
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
      const initialPlayer = getPlayerFromCard(card, watchButton);
      applyProvisionedState(watchButton, isPlayerProvisioned(initialPlayer));

      watchButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (watchButton.disabled || watchButton.getAttribute("data-provisioned") === "true") {
          return;
        }

        const player = getPlayerFromCard(card, watchButton);
        const playerId = String(player.playerId || "").trim();

        if (!playerId) {
          showToast(`Unable to initiate tracking for ${player.playerName || "player"}: missing player id`);
          return;
        }

        retireLegacyWatchListStorage();
        watchButton.setAttribute("data-opening-passport", "true");
        watchButton.disabled = true;
        watchButton.textContent = "OPENING PASSPORT";
        showToast(`${player.playerName || "Player"} queued for Passport Tracking`);

        window.setTimeout(() => {
          window.location.href = buildAppTrackingUrl(player);
        }, 450);
      });
    }
  }

  function initPlayerCardActions() {
    retireLegacyWatchListStorage();

    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));

    Array.from(document.querySelectorAll(WATCH_BUTTON_SELECTOR)).forEach((button) => {
      const card = button.closest(CARD_SELECTOR) || button.closest("article") || button.parentElement;
      if (card && !cards.includes(card)) cards.push(card);
    });

    cards.forEach(bindCard);
    syncCardProvisionStates();
  }

  function syncProvisionUI() {
    syncCardProvisionStates();
  }

  function scheduleProvisionSync() {
    syncProvisionUI();
    window.setTimeout(syncProvisionUI, 50);
    window.setTimeout(syncProvisionUI, 250);
    window.setTimeout(syncProvisionUI, 750);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      initPlayerCardActions();
      scheduleProvisionSync();
    });
  } else {
    initPlayerCardActions();
    scheduleProvisionSync();
  }

  window.addEventListener("focus", scheduleProvisionSync);
  window.addEventListener("pageshow", scheduleProvisionSync);
  window.addEventListener("resize", scheduleProvisionSync);
  window.addEventListener("orientationchange", scheduleProvisionSync);

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) scheduleProvisionSync();
  });

  window.addEventListener("storage", function (event) {
    if (!event || event.key === STORAGE_KEY || event.key === LEGACY_STORAGE_KEY) {
      scheduleProvisionSync();
    }
  });

  window.DiamondSignalsTrackingRadar = {
    getWatchList,
    setWatchList,
    syncProvisionUI,
  };
})();
