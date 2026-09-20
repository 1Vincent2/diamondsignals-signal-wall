(function(){
const COPY={
"EDGE SCORE":"Composite DiamondSignals signal-strength score. Higher values indicate stronger underlying movement and conviction.",
"SEAGER":"Hitter decision-quality signal: attacks in-zone pitches while refusing chase.",
"BABIP":"Batting average on balls in play. Use with skill signals to separate surface results from underlying movement.",
"K%":"Strikeout rate.","BB%":"Walk rate.","K/BB":"Strikeout-to-walk ratio.","BB/K":"Walk-to-strikeout ratio."
};
function init(root){
 if(!root||root.dataset.mobileSignalWallBound==="true")return; root.dataset.mobileSignalWallBound="true";
 const deck=root.querySelector("[data-ds-mobile-deck]"), cards=[...root.querySelectorAll(".ds-mobile-signal-card")];
 const modes=[...root.querySelectorAll("[data-ds-mobile-mode]")], panels=[...root.querySelectorAll("[data-ds-mobile-panel]")], count=root.querySelector("[data-ds-current-index]");
 function mode(name){modes.forEach(b=>{const on=b.dataset.dsMobileMode===name;b.classList.toggle("is-active",on);b.setAttribute("aria-selected",String(on));});panels.forEach(p=>p.hidden=p.dataset.dsMobilePanel!==name);}
 modes.forEach(b=>b.addEventListener("click",()=>mode(b.dataset.dsMobileMode)));
 root.querySelectorAll("[data-ds-open-scanner]").forEach(b=>b.addEventListener("click",()=>mode("players")));
 function go(i){if(!cards.length)return;const n=Math.max(0,Math.min(cards.length-1,i));cards[n].scrollIntoView({behavior:"smooth",inline:"center",block:"nearest"});if(count)count.textContent=String(n+1);}
 root.querySelector("[data-ds-deck-prev]")?.addEventListener("click",()=>go((Number(count?.textContent)||1)-2));
 root.querySelector("[data-ds-deck-next]")?.addEventListener("click",()=>go(Number(count?.textContent)||1));
 if(deck&&cards.length){let raf=0;deck.addEventListener("scroll",()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{const x=deck.scrollLeft+deck.clientWidth/2;let best=0,d=Infinity;cards.forEach((c,i)=>{const cd=Math.abs(c.offsetLeft+c.offsetWidth/2-x);if(cd<d){d=cd;best=i;}});if(count)count.textContent=String(best+1);});},{passive:true});}
 const search=root.querySelector("[data-ds-player-search]"), rows=[...root.querySelectorAll("[data-ds-player-row]")];let filter="all";
 function apply(){const q=(search?.value||"").trim().toLowerCase();rows.forEach(r=>{r.hidden=!((filter==="all"||r.dataset.playerType===filter)&&(!q||r.dataset.playerSearch.includes(q)));});}
 search?.addEventListener("input",apply);root.querySelectorAll("[data-ds-filter]").forEach(b=>b.addEventListener("click",()=>{filter=b.dataset.dsFilter;root.querySelectorAll("[data-ds-filter]").forEach(x=>x.classList.toggle("is-active",x===b));apply();}));
 rows.forEach(r=>r.addEventListener("click",()=>{mode("discover");setTimeout(()=>go(Number(r.dataset.cardIndex)||0),20);}));
 const sheet=root.querySelector("[data-ds-explainer]"), title=root.querySelector("[data-ds-explainer-title]"), copy=root.querySelector("[data-ds-explainer-copy]");
 root.querySelectorAll("[data-ds-metric-info]").forEach(b=>b.addEventListener("click",e=>{e.stopPropagation();const key=(b.dataset.dsMetricInfo||"METRIC").toUpperCase();if(title)title.textContent=key;if(copy)copy.textContent=COPY[key]||"DiamondSignals context for this live metric. Full Field Guide detail will be connected in the next refinement.";if(sheet)sheet.hidden=false;}));
 root.querySelector("[data-ds-explainer-close]")?.addEventListener("click",()=>{if(sheet)sheet.hidden=true;});
}
function boot(){document.querySelectorAll(".ds-mobile-signal-wall").forEach(init);}if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot,{once:true});else boot();
})();