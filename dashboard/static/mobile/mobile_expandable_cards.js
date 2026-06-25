/*
  DiamondSignals Mobile Expandable Cards

  State isolation rule:
  This script may only bind inside .ds-mobile-report-view.
  It must not bind to desktop report containers.
*/

(function () {
  function initMobileExpandableCards(root) {
    if (!root) return;

    const cards = root.querySelectorAll(".ds-mobile-card");

    cards.forEach((card) => {
      const button = card.querySelector(".ds-mobile-card-master");
      const tray = card.querySelector(".ds-mobile-card-tray");

      if (!button || !tray || button.dataset.dsMobileBound === "true") return;

      button.dataset.dsMobileBound = "true";

      button.addEventListener("click", () => {
        const isOpen = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!isOpen));
        tray.hidden = isOpen;
        card.classList.toggle("is-expanded", !isOpen);
      });
    });
  }

  window.DiamondSignalsMobileCards = {
    init: function () {
      document.querySelectorAll(".ds-mobile-report-view").forEach(initMobileExpandableCards);
    }
  };

  document.addEventListener("DOMContentLoaded", function () {
    window.DiamondSignalsMobileCards.init();
  });
})();
