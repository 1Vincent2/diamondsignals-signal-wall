(function () {
  const CARD_SELECTOR = ".js-player-card";
  const WATCH_BUTTON_SELECTOR = ".js-add-to-roster";
  const STORAGE_KEY = "diamondsignals_watch_list_v1";
  const LEGACY_STORAGE_KEY = "diamondsignals_roster_v1";
  const APP_AUTH_URL = "https://app.diamondsignals.ai/auth";
  const APP_WATCHLIST_PATH = "/watchlist";
  const APP_WATCHLIST_STATUS_URL = "https://app.diamondsignals.ai/api/watchlist/status";

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

  function setWatchList() {
    retireLegacyWatchListStorage();
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

  function getDefaultProvisionLabel(button) {
    const explicit = (button?.getAttribute("data-default-label") || "").trim();
    return explicit || "INITIATE TRACKING";
  }

  function applyProvisionedState(button, isProvisioned) {
    if (!button) return;
    button.textContent = isProvisioned ? "TRACKING ACTIVE" : getDefaultProvisionLabel(button);
    button.setAttribute("data-provisioned", isProvisioned ? "true" : "false");
    button.classList.toggle("is-provisioned", !!isProvisioned);
    button.setAttribute("aria-pressed", isProvisioned ? "true" : "false");
    button.disabled = !!isProvisioned;
  }

  function applyPendingState(button) {
    if (!button) return;
    button.textContent = "OPENING PASSPORT";
    button.setAttribute("data-provisioned", "pending");
    button.setAttribute("aria-pressed", "true");
    button.disabled = true;
    button.classList.add("is-provisioned");
  }

  function applyCardTrackingState(card, isProvisioned) {
    if (!card) return;
    card.classList.toggle("is-provisioned", !!isProvisioned);
    card.classList.toggle("tracking-active", !!isProvisioned);
    card.setAttribute("data-provisioned", isProvisioned ? "true" : "false");
    card.setAttribute("data-tracking-state", isProvisioned ? "active" : "idle");
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

  async function fetchAppWatchlistStatus(player) {
    const playerId = String(player.playerId || "").trim();
    if (!playerId) return null;

    try {
      const url = new URL(APP_WATCHLIST_STATUS_URL);
      url.searchParams.set("player_id", playerId);

      const response = await fetch(url.toString(), {
        method: "GET",
        credentials: "include",
        headers: {
          "Accept": "application/json",
        },
      });

      if (!response.ok) return null;

      const payload = await response.json();

      if (typeof payload.tracked === "boolean") return payload.tracked;
      if (typeof payload.in_watchlist === "boolean") return payload.in_watchlist;
      if (typeof payload.exists === "boolean") return payload.exists;

      return null;
    } catch {
      return null;
    }
  }

  async function resolveProvisionedState(player) {
    const appStatus = await fetchAppWatchlistStatus(player);
    return typeof appStatus === "boolean" ? appStatus : false;
  }

  async function refreshCardTrackingState(card, watchButton) {
    if (!card || !watchButton) return;
    const player = getPlayerFromCard(card, watchButton);
    const provisioned = await resolveProvisionedState(player);
    applyProvisionedState(watchButton, provisioned);
    applyCardTrackingState(card, provisioned);
  }

  async function syncCardProvisionStates() {
    retireLegacyWatchListStorage();

    const cards = Array.from(document.querySelectorAll(CARD_SELECTOR));

    Array.from(document.querySelectorAll(WATCH_BUTTON_SELECTOR)).forEach((button) => {
      const card = button.closest(CARD_SELECTOR) || button.closest(".asset-card") || button.closest("article") || button.parentElement;
      if (card && !cards.includes(card)) cards.push(card);
    });

    await Promise.all(cards.map(async (card) => {
      const watchButton = card.querySelector(WATCH_BUTTON_SELECTOR);
      if (!watchButton) return;
      await refreshCardTrackingState(card, watchButton);
    }));
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
      applyProvisionedState(watchButton, false);
      applyCardTrackingState(card, false);
      refreshCardTrackingState(card, watchButton);

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
        applyPendingState(watchButton);
        applyCardTrackingState(card, true);
        showToast(`${player.playerName || "Player"} opening Passport Watchlist`);

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

  window.addEventListener("storage", function () {
    retireLegacyWatchListStorage();
    scheduleProvisionSync();
  });

  window.DiamondSignalsTrackingRadar = {
    getWatchList,
    setWatchList,
    syncProvisionUI,
    fetchAppWatchlistStatus,
    resolveProvisionedState,
  };
})();
