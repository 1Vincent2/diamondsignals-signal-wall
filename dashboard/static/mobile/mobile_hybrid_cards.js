/* Shared inline exploration; the existing action script still owns tracking. */
(function () {
  function init() {
    document.querySelectorAll('.ds-hybrid-card').forEach(card => {
      if (card.dataset.hybridBound) return;
      card.dataset.hybridBound = 'true';
      // A card is a reading surface. Full dossier navigation is an explicit link.
      card.dataset.profileUrl = '#';
      card.removeAttribute('tabindex');
      card.style.cursor = 'default';
      const toggle = card.querySelector('.ds-hybrid-toggle');
      const panel = card.querySelector('.ds-hybrid-depth');
      const close = card.querySelector('.ds-hybrid-close');
      function setExpanded(expanded) {
        panel.hidden = !expanded;
        toggle.setAttribute('aria-expanded', String(expanded));
        toggle.textContent = expanded ? 'HIDE INTELLIGENCE' : 'VIEW INTELLIGENCE';
      }
      toggle.addEventListener('click', () => setExpanded(panel.hidden));
      close.addEventListener('click', () => {
        setExpanded(false);
        toggle.focus({preventScroll:true});
        const bounds = toggle.getBoundingClientRect();
        if (bounds.top < 0 || bounds.bottom > window.innerHeight) toggle.scrollIntoView({block:'nearest',behavior:'instant'});
      });
      card.querySelectorAll('details, .ds-hybrid-toggle, .ds-hybrid-close, .ds-hybrid-depth').forEach(control => {
        ['click', 'keydown'].forEach(type => control.addEventListener(type, event => event.stopPropagation()));
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true});
  else init();
})();
