(function () {
  const CARD_SELECTOR = ".js-player-card";
  const ROSTER_BUTTON_SELECTOR = ".js-add-to-roster";
  const APP_BASE_URL = "https://app.diamondsignals.ai";

  function buildAuthUrl(playerId) {
    const authUrl = new URL(`${APP_BASE_URL}/auth`);
    authUrl.searchParams.set("next", "/terminal");
    if (playerId) {
      authUrl.searchParams.set("add_player_id", String(playerId));
    }
    return authUrl.toString();
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
    return {
      playerId: card.getAttribute("data-player-id") || "",
      playerName: card.getAttribute("data-player-name") || "",
      playerType: card.getAttribute("data-player-type") || "",
      team: card.getAttribute("data-player-team") || "",
      profileUrl: card.getAttribute("data-profile-url") || "",
    };
  }

  function bindCard(card) {
    if (!card || card.dataset.playerCardBound === "true") return;
    card.dataset.playerCardBound = "true";

    card.style.cursor = "pointer";

    card.addEventListener("click", function (event) {
      const rosterButton = event.target.closest(ROSTER_BUTTON_SELECTOR);
      if (rosterButton) return;
      openProfileFromCard(card);
    });

    card.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        const rosterButton = event.target.closest(ROSTER_BUTTON_SELECTOR);
        if (rosterButton) return;
        event.preventDefault();
        openProfileFromCard(card);
      }
    });

    if (!card.hasAttribute("tabindex")) {
      card.setAttribute("tabindex", "0");
    }

    const rosterButton = card.querySelector(ROSTER_BUTTON_SELECTOR);
    if (rosterButton) {
      rosterButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        const player = getPlayerFromCard(card);
        showToast(`Sending ${player.playerName || "player"} to Roster Terminal...`);
        window.location.href = buildAuthUrl(player.playerId);
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

  window.DiamondSignalsRoster = {
    buildAuthUrl,
  };
})();