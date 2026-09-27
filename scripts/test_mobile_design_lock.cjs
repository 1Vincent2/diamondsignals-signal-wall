// Verify generated mobile DOM, real shared action code, and retained source contracts.
const assert=require('node:assert/strict'),fs=require('node:fs'),{JSDOM}=require('jsdom');
const pairs=[['live','signal-wall'],['velocity-decay','velocity-decay'],['stuff-disruption','stuff-disruption'],['ivb-heat-map','ivb-heat-map'],['apex-extraction','apex-extraction'],['mlb-extraction','mlb-extraction'],['waiver-wire','waiver-wire'],['kinetic-drift','kinetic-drift']];
const actions=fs.readFileSync('src/js/player-card-actions.js','utf8').replace('window.location.href = buildAppTrackingUrl(player);','window.__trackingTarget = buildAppTrackingUrl(player);');
(async()=>{
 for(const [route,family] of pairs){
  const html=fs.readFileSync(`dist/mobile-${route}-canary/index.html`,'utf8');
  const d=new JSDOM(html,{url:`https://diamondsignals-mobile-preview.netlify.app/mobile-${route}-canary/`,runScripts:'dangerously',beforeParse(w){w.fetch=async()=>({ok:true,json:async()=>({ok:true,tracked:false})});w.HTMLElement.prototype.scrollIntoView=function(){};}});
  await new Promise(r=>d.window.addEventListener('load',r));d.window.eval(actions);
  await new Promise(r=>setTimeout(r,0));const doc=d.window.document,cards=[...doc.querySelectorAll('.ds-hybrid-card')];
  assert(html.includes('<script src="/player-card-actions.js">'));
  if(family==='waiver-wire'&&!cards.length){assert.match(doc.body.textContent,/feed is unavailable/);assert.equal(doc.querySelector('.js-add-to-roster'),null);}
  else assert(cards.length>0,family);
  for(const card of cards){
   assert(card.querySelectorAll('.ds-hybrid-evidence dl>div').length<=3);
   const headings=[...card.querySelectorAll('.ds-mobile-card-body > section > h3')].map(x=>x.textContent);
   assert.deepEqual(headings.map(x=>x.split(' /')[0]),['WHAT CHANGED','WHY IT MATTERS','KEY EVIDENCE','WHAT TO WATCH NEXT']);
   assert.match(card.dataset.playerId,/^\d+$/);assert(card.querySelector('time').textContent.includes('Source updated:'));
   assert.equal(new URL(card.querySelector('.ds-mobile-intel-link').href).searchParams.get('mobile_origin'),family);
   assert.equal(card.dataset.profileUrl,'#');
   const toggle=card.querySelector('.ds-hybrid-toggle'),panel=card.querySelector('.ds-hybrid-depth');
   assert.equal(panel.hidden,true);assert.equal(toggle.getAttribute('aria-controls'),panel.id);
   const originalUrl=d.window.location.href; toggle.click();
   assert.equal(panel.hidden,false);assert.equal(toggle.getAttribute('aria-expanded'),'true');
   assert.equal(toggle.textContent,'HIDE INTELLIGENCE');assert.equal(d.window.location.href,originalUrl);
   card.querySelector('.ds-hybrid-close').click();assert.equal(panel.hidden,true);
   assert.equal(toggle.getAttribute('aria-expanded'),'false');assert.equal(doc.activeElement,toggle);
   const details=card.querySelector('details');if(details){let bubbled=false;card.addEventListener('click',()=>bubbled=true);details.querySelector('summary').click();assert.equal(bubbled,false);assert(details.open);}
   const button=card.querySelector('.js-add-to-roster');assert.equal(button.disabled,false);button.click();
   const target=new URL(d.window.__trackingTarget),next=new URL(target.searchParams.get('next'),target.origin);
   assert.equal(target.origin,'https://app.diamondsignals.ai');assert.equal(next.pathname,'/watchlist');assert.equal(next.searchParams.get('add_player_id'),card.dataset.playerId);assert.equal(next.searchParams.get('player_name'),card.dataset.playerName);
  }
  doc.querySelector('[data-ds-mobile-menu-open]').click();assert.equal(doc.querySelector('[data-ds-mobile-menu-drawer]').getAttribute('aria-hidden'),'false');assert.equal(doc.querySelectorAll('[data-ds-mobile-menu-drawer] nav a').length,10);
  doc.querySelector('[data-ds-field-guide-open]').click();assert.equal(doc.querySelector('[data-ds-field-guide]').hidden,false);
  if(cards.length){doc.querySelector('[data-ds-open-scanner]').click();assert.equal(doc.querySelector('[data-ds-mobile-panel="players"]').hidden,false);let search=doc.querySelector('[data-ds-player-search]');search.value='zz-no-match';search.dispatchEvent(new d.window.Event('input'));assert([...doc.querySelectorAll('[data-ds-player-row]')].every(r=>r.hidden));}
  console.log(`${family}: ${cards.length} cards; hierarchy, evidence, identity, tracking handoff, dossier origin, nested controls, menu and guide PASS`);d.window.close();
 }
 // Existing tracking state must suppress duplicate actions without changing auth code.
 const d=new JSDOM(fs.readFileSync('dist/mobile-live-canary/index.html','utf8'),{url:'https://diamondsignals-mobile-preview.netlify.app/mobile-live-canary/',runScripts:'dangerously',beforeParse(w){w.fetch=async()=>({ok:true,json:async()=>({ok:true,tracked:true})});}});
 await new Promise(r=>d.window.addEventListener('load',r));d.window.eval(actions);await new Promise(r=>setTimeout(r,0));assert([...d.window.document.querySelectorAll('.js-add-to-roster')].every(b=>b.disabled&&b.textContent==='ASSET TRACKED'));d.window.close();console.log('Existing tracked state / duplicate action suppression: PASS');
})().catch(e=>{console.error(e);process.exitCode=1});
