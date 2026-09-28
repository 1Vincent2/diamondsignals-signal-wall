/* Presentation controls only. Existing shared code owns tracking and auth. */
(function () {
  function init() {
    document.querySelectorAll('.ds-hybrid-card').forEach(card => {
      if (card.dataset.hybridBound) return;
      card.dataset.hybridBound = 'true';
      card.dataset.profileUrl = '#';
      card.removeAttribute('tabindex');
      card.style.cursor = 'default';
      const toggle = card.querySelector('.ds-hybrid-toggle');
      const panel = card.querySelector('.ds-hybrid-depth');
      toggle.addEventListener('click', () => {
        const expanded = panel.hidden;
        panel.hidden = !expanded;
        toggle.setAttribute('aria-expanded', String(expanded));
        toggle.textContent = expanded ? 'HIDE INTELLIGENCE' : 'DEEPER INTELLIGENCE';
      });
      const tabs = [...card.querySelectorAll('[role="tab"]')];
      function selectTab(tab, focus) {
        tabs.forEach(item => {
          const selected = item === tab;
          item.setAttribute('aria-selected', String(selected));
          item.tabIndex = selected ? 0 : -1;
          document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
        });
        if (focus) tab.focus({preventScroll:true});
      }
      tabs.forEach((tab,index) => {
        tab.addEventListener('click', () => selectTab(tab,false));
        tab.addEventListener('keydown', event => {
          let next;
          if (event.key === 'ArrowRight') next = (index+1)%tabs.length;
          if (event.key === 'ArrowLeft') next = (index+tabs.length-1)%tabs.length;
          if (event.key === 'Home') next = 0;
          if (event.key === 'End') next = tabs.length-1;
          if (next !== undefined) {event.preventDefault();selectTab(tabs[next],true);}
        });
      });
      card.querySelectorAll('details, .ds-hybrid-toggle, .ds-hybrid-depth').forEach(control => {
        ['click', 'keydown'].forEach(type => control.addEventListener(type, event => event.stopPropagation()));
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true});
  else init();
})();
