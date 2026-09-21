(function(){
const MOBILE_REPORTS=[
["SIGNALS","/mobile-live-canary/","signal-wall"],["VELOCITY DECAY","/mobile-velocity-decay-canary/","velocity-decay"],["STUFF+ DISRUPTION","/mobile-stuff-disruption-canary/","stuff-disruption"],["IVB HEAT MAP","/mobile-ivb-heat-map-canary/","ivb-heat-map"],["APEX EXTRACTION","/mobile-apex-extraction-canary/","apex-extraction"],["MLB EXTRACTION","/mobile-mlb-extraction-canary/","mlb-extraction"],["WAIVER WIRE","/mobile-waiver-wire-canary/","waiver-wire"],["KINETIC DRIFT","/mobile-kinetic-drift-canary/","kinetic-drift"]
];
const COPY={
"BLAST PATH":"Recent average exit velocity on batted balls, in mph.",
"BLAST RATE":"Recent barrel rate as a percentage of batted balls.",
"APEX DAMAGE":"Maximum exit velocity in the recent sample, in mph.",
"MISS ENGINE":"Swinging strikes and blocked swinging strikes as a percentage of all recent pitches; this is not whiffs per swing.",
"VELOCITY FUEL":"Recent average fastball velocity, in mph.",
"RELEASE DECEPTION":"Recent average release extension, in feet.",
"PHYSICS CORE":"MLB Extraction underlying-trait measure. The ledger model supplies a trait score; Statcast fallback profiles use exit velocity for hitters or whiff rate for pitchers. Read the displayed units and source context.",
"MARKET GAP":"MLB Extraction surface-pressure measure. The ledger model supplies a surface-pressure score; Statcast fallback profiles use exit-velocity change for hitters or velocity change for pitchers. It is not a price or ownership percentage.",
"MARKET ATTENTION FEED":"MLB Extraction market measure: a market score in ledger profiles or rostered percentage when an ownership feed is supplied. FEED OFFLINE means that feed is unavailable.",
"MARKET ATTENTION":"MLB Extraction market measure: a market score in ledger profiles or rostered percentage when an ownership feed is supplied. A dash means no value was provided.",

"EDGE SCORE":"Composite DiamondSignals signal-strength score. Higher values indicate stronger underlying movement and conviction.",
"SEAGER":"Hitter decision-quality signal: attacks in-zone pitches while refusing chase.",
"BABIP":"Batting average on balls in play. Use with skill signals to separate surface results from underlying movement.",
"K%":"Strikeout rate.","BB%":"Walk rate.","K/BB":"Strikeout-to-walk ratio.","BB/K":"Walk-to-strikeout ratio.",
"RISK SCORE":"Composite Velocity Decay risk score. Higher values indicate stronger evidence of fastball deterioration versus the pitcher’s recent baseline.",
"VELOCITY DELTA":"Change in recent fastball velocity versus the comparison baseline.",
"EXTENSION DELTA":"Change in release extension versus baseline; lost extension can reduce perceived velocity and alter pitch shape.",
"PERCEIVED VELOCITY":"DiamondSignals proxy for how velocity and extension combine to affect the hitter’s effective reaction window.",
"DISRUPTION SCORE":"Composite Stuff+ Disruption score measuring the strength of recent pitch-shape change.",
"IVB DELTA":"Change in induced vertical break versus the pitcher’s baseline.",
"VAA DELTA":"Change in vertical approach angle versus baseline.",
"MOVEMENT DELTA":"Recent change in pitch movement profile versus baseline.",
"IVB VS AVG":"Induced vertical break compared with the relevant velocity-band baseline.",
"IVB RAW":"Measured induced vertical break for the fastball sample.",
"VAA":"Vertical approach angle: the angle at which the pitch enters the hitting zone.",
"DEAD ZONE":"Flags fastball shape that falls into the engine’s low-distinction movement band.",
"APEX SCORE":"Composite Apex Extraction score for an underlying physical/skill shift that may be ahead of market recognition.",
"PHYSICAL SHIFT":"Strength of the underlying physical-performance change detected by Apex.",
"VISION DELTA":"Change in hitter decision/recognition quality captured by the Apex signal layer.",
"MARKET LATENCY":"Estimated gap between the underlying player signal and what the market currently reflects.",
"WAIVER SCORE":"Composite waiver command score for a verified, market-eligible player.",
"OWNERSHIP GATE":"Market-eligibility check used before DiamondSignals will surface a waiver recommendation.",
"SIGNAL WINDOW":"Recency window supporting the current waiver signal.",
"DEPLOYMENT":"Current command state indicating whether the asset is actionable, surveillance-only, or locked.",
"KDE SCORE":"Kinetic Drift Engine headline score: the strongest of KRS, KES, and KIS.",
"KRS":"Kinetic Risk Score: deterioration and fatigue signals versus the pitcher’s own recent baseline.",
"KES":"Kinetic Emergence Score: improving delivery or pitch-shape signals versus the pitcher’s baseline.",
"KIS":"Kinetic Instability Score: unusual mechanical or pitch-shape variability across recent appearances."
};
function menuMarkup(active){
 return '<div class="ds-mobile-drawer-head"><div><span>DIAMONDSIGNALS</span><strong>COMMAND MENU</strong></div><button type="button" data-ds-mobile-menu-close>×</button></div><nav>'+MOBILE_REPORTS.map(([label,href,report])=>'<a href="'+href+'"'+(report===active?' class="is-active"':'')+'>'+label+' <span>›</span></a>').join("")+'<a href="https://app.diamondsignals.ai/auth?next=/watchlist">TRACKING RADAR <span>›</span></a><a href="https://app.diamondsignals.ai/auth?next=/terminal">ROSTER TERMINAL <span>›</span></a></nav>';
}
function ensureMenu(root){
 let backdrop=root.querySelector(".ds-mobile-command-backdrop");
 if(!backdrop){backdrop=document.createElement("div");backdrop.className="ds-mobile-command-backdrop";backdrop.hidden=true;backdrop.setAttribute("data-ds-mobile-menu-close","");root.append(backdrop);}
 let drawer=root.querySelector("[data-ds-mobile-menu-drawer]");
 if(!drawer){drawer=document.createElement("aside");drawer.className="ds-mobile-command-drawer";drawer.setAttribute("data-ds-mobile-menu-drawer","");drawer.setAttribute("aria-hidden","true");root.append(drawer);}
 if(drawer.dataset.dsSharedMenu!=="true"){drawer.innerHTML=menuMarkup(root.dataset.mobileReport||"");drawer.dataset.dsSharedMenu="true";}
 return {drawer,backdrop};
}
function init(root){
 if(!root||root.dataset.mobileSignalWallBound==="true")return; root.dataset.mobileSignalWallBound="true";
 const deck=root.querySelector("[data-ds-mobile-deck]"), cards=[...root.querySelectorAll(".ds-mobile-signal-card")];
 const parkedPromotion=root.dataset.mobileReport==="promotion-watch";
 // Preserve native dossier navigation; only attach a known mobile origin.
 const originReport=MOBILE_REPORTS.find(([,path,key])=>key===root.dataset.mobileReport && window.location.pathname.replace(/\/$/,"")===path.replace(/\/$/,""));
 if(!parkedPromotion&&originReport)root.querySelectorAll(".ds-mobile-intel-link").forEach(link=>{
   const target=new URL(link.getAttribute("href")||"",window.location.href);
   if(target.origin===window.location.origin&&/^\/scout\/\d+\/$/.test(target.pathname)){
     target.searchParams.set("mobile_origin",originReport[2]);
     link.setAttribute("href",target.pathname+target.search+target.hash);
   }
 });
 if(parkedPromotion&&!root.querySelector("[data-ds-mobile-menu-drawer]")){
   const active=root.dataset.mobileReport||"";
   const backdrop=document.createElement("div");backdrop.className="ds-mobile-command-backdrop";backdrop.hidden=true;backdrop.setAttribute("data-ds-mobile-menu-close","");
   const drawer=document.createElement("aside");drawer.className="ds-mobile-command-drawer";drawer.setAttribute("data-ds-mobile-menu-drawer","");drawer.setAttribute("aria-hidden","true");
   drawer.innerHTML='<div class="ds-mobile-drawer-head"><div><span>DIAMONDSIGNALS</span><strong>COMMAND MENU</strong></div><button type="button" data-ds-mobile-menu-close>×</button></div><nav>'+MOBILE_REPORTS.map(([label,href])=>'<a href="'+href+'"'+(href.includes(active)&&active?' class="is-active"':'')+'>'+label+' <span>›</span></a>').join("")+'<a href="https://app.diamondsignals.ai/auth?next=/watchlist">TRACKING RADAR <span>›</span></a><a href="https://app.diamondsignals.ai/auth?next=/terminal">ROSTER TERMINAL <span>›</span></a></nav>';
   root.append(backdrop,drawer);
 }
 const menuBtn=root.querySelector("[data-ds-mobile-menu-open]");
 if(parkedPromotion){
   const drawer=root.querySelector("[data-ds-mobile-menu-drawer]"),backdrop=root.querySelector(".ds-mobile-command-backdrop");
   function menu(open){if(drawer){drawer.classList.toggle("is-open",open);drawer.setAttribute("aria-hidden",String(!open));}if(menuBtn)menuBtn.setAttribute("aria-expanded",String(open));if(backdrop){backdrop.hidden=!open;backdrop.classList.toggle("is-open",open);}}
   menuBtn?.addEventListener("click",()=>menu(true));root.querySelectorAll("[data-ds-mobile-menu-close]").forEach(el=>el.addEventListener("click",()=>menu(false)));
 }else{
   ensureMenu(root);
   function menu(open){const {drawer,backdrop}=ensureMenu(root);drawer.classList.toggle("is-open",open);drawer.setAttribute("aria-hidden",String(!open));if(menuBtn)menuBtn.setAttribute("aria-expanded",String(open));backdrop.hidden=!open;backdrop.classList.toggle("is-open",open);}
   root.addEventListener("click",e=>{const target=e.target?.closest?.("[data-ds-mobile-menu-open],[data-ds-mobile-menu-close]");if(!target||!root.contains(target))return;if(target.matches("[data-ds-mobile-menu-open]"))menu(true);else menu(false);});
 }
 const guide=root.querySelector("[data-ds-field-guide]");root.querySelector("[data-ds-field-guide-open]")?.addEventListener("click",()=>{if(guide)guide.hidden=false;});root.querySelector("[data-ds-field-guide-close]")?.addEventListener("click",()=>{if(guide)guide.hidden=true;});
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
 if(parkedPromotion)root.querySelectorAll(".ds-mobile-intel-link").forEach(link=>link.addEventListener("click",e=>{
   const href=link.getAttribute("href");if(!href||href==="#")return;e.preventDefault();
   let intel=root.querySelector("[data-ds-mobile-intelligence]");
   if(!intel){intel=document.createElement("section");intel.className="ds-mobile-intelligence-sheet";intel.setAttribute("data-ds-mobile-intelligence","");intel.innerHTML='<div class="ds-mobile-intelligence-head"><div><span>PLAYER INTELLIGENCE</span><strong>SCOUT DOSSIER</strong></div><button type="button" aria-label="Close intelligence">×</button></div><iframe title="Player intelligence"></iframe>';root.append(intel);intel.querySelector("button").addEventListener("click",()=>{intel.classList.remove("is-open");document.body.classList.remove("ds-intel-open");});}
   const frame=intel.querySelector("iframe");if(frame)frame.src=href;intel.classList.add("is-open");document.body.classList.add("ds-intel-open");
 }));
}
function boot(){if(!document.getElementById("ds-mobile-global-interaction-style")){const s=document.createElement("style");s.id="ds-mobile-global-interaction-style";s.textContent=".ds-mobile-intelligence-sheet{position:fixed;z-index:110;left:0;right:0;bottom:0;height:min(88vh,900px);background:#060a10;border-top:1px solid rgba(182,255,0,.35);border-radius:22px 22px 0 0;box-shadow:0 -30px 90px #000;transform:translateY(105%);transition:transform .24s ease;overflow:hidden}.ds-mobile-intelligence-sheet.is-open{transform:translateY(0)}.ds-mobile-intelligence-head{height:62px;box-sizing:border-box;display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.09);background:#09111a}.ds-mobile-intelligence-head span{display:block;color:#b6ff00;font:900 8px ui-monospace,monospace;letter-spacing:.12em}.ds-mobile-intelligence-head strong{display:block;margin-top:3px;font:900 14px ui-monospace,monospace;color:#fff}.ds-mobile-intelligence-head button{width:40px;height:40px;border-radius:50%;border:1px solid rgba(255,255,255,.14);background:#111822;color:#fff;font-size:22px}.ds-mobile-intelligence-sheet iframe{display:block;width:100%;height:calc(100% - 62px);border:0;background:#060a10}.ds-intel-open{overflow:hidden}";document.head.appendChild(s);}document.querySelectorAll(".ds-mobile-signal-wall").forEach(init);}if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot,{once:true});else boot();
})();
