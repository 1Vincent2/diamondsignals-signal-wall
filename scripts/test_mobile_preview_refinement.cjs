// Run from repo root with jsdom and esbuild available to Node. Uses only local build fixtures.
const assert=require('node:assert/strict'),fs=require('node:fs'),{JSDOM}=require('jsdom'),esbuild=require('esbuild');
const {execFileSync}=require('node:child_process');
const temp=fs.mkdtempSync(require('node:path').join(require('node:os').tmpdir(),'ds-mobile-refinement-'));
const origin='https://diamondsignals-mobile-preview.netlify.app';
(async()=>{
for(const builder of ['build_mobile_apex_extraction.py','build_mobile_command_signal_wall.py','build_mobile_mlb_extraction.py'])execFileSync('python3',['dashboard/'+builder]);
const scoutHtml=execFileSync('python3',['-c',`import ast
from pathlib import Path
tree=ast.parse(Path('dashboard/build_dashboard.py').read_text())
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='scout_shell_html')
ns={'SHELL_STYLES_TEMPLATE':''}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'scout','exec'),ns)
print(ns['scout_shell_html']().replace('__SCOUT_PLAYER_JSON__','{"player_id":"668723","player_name":"Test","canonical_score":null}'))
`],{encoding:'utf8'});

for(const [route,key] of [['mobile-apex-extraction-canary','apex-extraction'],['mobile-live-canary','signal-wall'],['mobile-mlb-extraction-canary','mlb-extraction']]){
 const d=new JSDOM(fs.readFileSync(`dist/${route}/index.html`,'utf8'),{url:origin+'/'+route+'/',runScripts:'dangerously'}); await new Promise(r=>d.window.addEventListener('load',r));const doc=d.window.document;
 for(const a of doc.querySelectorAll('.ds-mobile-intel-link')) if(a.pathname.startsWith('/scout/'))assert.equal(new URL(a.href).searchParams.get('mobile_origin'),key);
 const missing=[];for(const b of doc.querySelectorAll('[data-ds-metric-info]')){b.click();assert.equal(doc.querySelector('[data-ds-explainer]').hidden,false);const copy=doc.querySelector('[data-ds-explainer-copy]').textContent;if(copy.includes('next refinement'))missing.push(b.dataset.dsMetricInfo);doc.querySelector('[data-ds-explainer-close]').click();assert.equal(doc.querySelector('[data-ds-explainer]').hidden,true);}assert.deepEqual([...new Set(missing)],[]);
 doc.querySelector('[data-ds-mobile-menu-open]').click();assert.equal(doc.querySelector('[data-ds-mobile-menu-drawer]').getAttribute('aria-hidden'),'false');assert.equal(doc.querySelectorAll('[data-ds-mobile-menu-drawer] nav a').length,10);d.window.close();console.log(route+': generated links, metric sheets, menu PASS');
}
const shared=fs.readFileSync('dashboard/static/mobile/mobile_signal_wall_command.js','utf8');
const copyKeys=new Set([...shared.matchAll(/^"([^"\n]+)":/gm)].map(m=>m[1]));
for(const file of fs.readdirSync('dashboard/templates/mobile/surface_reports')){
 if(!file.endsWith('_command.html')||file==='promotion_watch_command.html')continue;
 const template=fs.readFileSync('dashboard/templates/mobile/surface_reports/'+file,'utf8');
 for(const m of template.matchAll(/data-ds-metric-info="([^"{]+)"/g))assert(copyKeys.has(m[1]),file+': missing copy '+m[1]);
}
console.log('All completed report fixed metric mappings: PASS');
const html=scoutHtml;
for(const [key,path,label] of [['kinetic-drift','/mobile-kinetic-drift-canary/','BACK TO KINETIC DRIFT'],['ivb-heat-map','/mobile-ivb-heat-map-canary/','BACK TO IVB HEAT MAP'],['','/','Back to Signal Wall'],['https://evil.example','/','Back to Signal Wall'],['__proto__','/','Back to Signal Wall'],['promotion-watch','/','Back to Signal Wall']]){
 const d=new JSDOM(html,{url:origin+'/scout/668723/?mobile_origin='+encodeURIComponent(key),runScripts:'dangerously'});const a=d.window.document.getElementById('scoutReturnLink');assert.equal(a.getAttribute('href'),path);assert.equal(a.textContent,label);assert.equal(d.window.document.getElementById('scoutSignalPill').textContent,'Signal Score --');d.window.close();
}console.log('Generated dossier allowlist, default return and absent score: PASS');
await esbuild.build({entryPoints:['netlify/request-gates/signals-gate.ts'],bundle:true,platform:'node',format:'cjs',outfile:temp+'/gate.cjs'});
global.Netlify={env:{get:()=>undefined}};const gate=require(temp+'/gate.cjs').default;let next=0;const ctx={next:()=>{next++;return new Response('ORIGINAL')}};
for(const path of ['/','/index.html']){const r=await gate(new Request(origin+path),ctx);assert.equal(r.status,200);const doc=new JSDOM(await r.text()).window.document;assert.equal(doc.querySelectorAll('nav a').length,8);assert.equal(doc.querySelector('form'),null);assert.equal(next,0);}
for(const host of ['signals.diamondsignals.ai','diamondsignals.ai','app.diamondsignals.ai','other.netlify.app','diamondsignals-mobile-preview.netlify.app.evil.example']){const r=await gate(new Request('https://'+host+'/'),ctx);assert.equal(await r.text(),'ORIGINAL');}
const r=await gate(new Request(origin+'/live/'),ctx);assert.equal(r.status,503);console.log('Exact-host launcher, production root passthrough, protected routes fail closed: PASS');
})().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>fs.rmSync(temp,{recursive:true,force:true}));
