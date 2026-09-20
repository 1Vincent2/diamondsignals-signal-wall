/*
  DiamondSignals Mobile Command Experience bootstrap.
  State and event binding must remain inside .ds-mobile-report-view.
*/
(function () {
  function initMobileCommandExperience(root) {
    if (!root || root.dataset.dsMobileCommandBound === "true") return;
    root.dataset.dsMobileCommandBound = "true";
    root.dispatchEvent(new CustomEvent("ds-mobile-command-ready", { bubbles: false }));
  }

  function init() {
    document.querySelectorAll(".ds-mobile-report-view").forEach(initMobileCommandExperience);
  }

  window.DiamondSignalsMobileCommand = { init };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
