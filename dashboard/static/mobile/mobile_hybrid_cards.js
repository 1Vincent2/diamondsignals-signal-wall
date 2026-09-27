/* Keep nested controls independent of legacy whole-card dossier navigation. */
(function () {
  function init() {
    document.querySelectorAll('.ds-hybrid-card').forEach(card => {
      const link = card.querySelector('.ds-mobile-intel-link');
      if (link) card.dataset.profileUrl = link.getAttribute('href');
      card.querySelectorAll('details, .ds-mobile-intel-link').forEach(control => {
        ['click', 'keydown'].forEach(type => control.addEventListener(type, event => event.stopPropagation()));
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true});
  else init();
})();
