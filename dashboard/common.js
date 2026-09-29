/* Shared helpers for index.html (Session analysis) and load.html (Load monitoring).
   Data: window.DASH_DATA (outputs/dashboard_data.js) and, on the session page,
   window.SESSION_BINS (outputs/session_bins.js). Everything is pseudonymised (P-codes); the public demo uses synthetic data. */

/* ---------- page start (public demo: no gate) ---------- */
function gate(onEnter){
  const g=document.getElementById('gate'), app=document.getElementById('app');
  if(g) g.remove();
  app.style.display='block'; onEnter();
}

/* ---------- data ---------- */
const D = window.DASH_DATA || null;
const Q = D && D.squad;
const INFO = (Q && Q.positions) || {};
const FIX  = (Q && Q.fixtures)  || {};
const gradeOf = d => (FIX[d]||{}).grade || '';
const fixLabel = d => { const f=FIX[d]; return f ? `${f.grade} v ${f.opp}` : ''; };

/* grades each athlete has played (dual-grade players appear in both) */
const GR = {};
if(Q){ Q.sessions.a.forEach((ai,i)=>{ if(Q.sessions.t[i]!=='M') return; const g=gradeOf(Q.sessions.d[i]);
  if(g)(GR[Q.athletes[ai]]=GR[Q.athletes[ai]]||new Set()).add(g); }); }

/* metrics: key -> label/colour; `col` is the column name in DASH_DATA.squad.sessions */
const METRICS = {
  td: {label:'Total distance',        short:'TD',     unit:'m',   col:'dist', color:'#1f2430'},
  hsr:{label:'High-speed running',    short:'HSR',    unit:'m',   col:'hsr',  color:'#e4571f', note:'>5.5 m/s'},
  spr:{label:'Sprint distance',       short:'Sprint', unit:'m',   col:'spr',  color:'#7c3aed', note:'>7 m/s'},
  tal:{label:'Total acceleration load',short:'TAL',   unit:'m·s⁻¹', col:'tal',  color:'#0e86c7', note:'Σ|Δv|'}
};
const LINES=['Goalkeeper','Full-Back','Half-Back','Midfield','Half-Forward','Full-Forward'];
const GROUPS=['Goalkeeper','Inside defender','Middle 8','Inside forward'];

/* ---------- formatting ---------- */
const DAY=86400000;
const iso=t=>new Date(t).toISOString().slice(0,10);
const addDays=(s,n)=>iso(Date.parse(s)+n*DAY);
const fmtD =s=>new Date(s+'T00:00:00').toLocaleDateString('en-GB',{day:'numeric',month:'short'});
const fmtDL=s=>new Date(s+'T00:00:00').toLocaleDateString('en-GB',{weekday:'short',day:'numeric',month:'short'});
const n0=v=>v==null||isNaN(v)?'—':Math.round(v).toLocaleString('en-GB');
const n1=v=>v==null||isNaN(v)?'—':(Math.round(v*10)/10).toLocaleString('en-GB',{minimumFractionDigits:1,maximumFractionDigits:1});
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
const clockS=s=>{s=Math.round(s);return `${String(Math.floor(s/3600)%24).padStart(2,'0')}:${String(Math.floor(s%3600/60)).padStart(2,'0')}`;};
const pct=(a,b)=>b?Math.round((a/b-1)*100):null;
const signed=v=>v==null?'—':(v>0?'+':'')+v+'%';

/* ---------- SVG kit ---------- */
const NS='http://www.w3.org/2000/svg';
function S_(tag,at,txt){const e=document.createElementNS(NS,tag);for(const k in(at||{}))e.setAttribute(k,at[k]);if(txt!=null)e.textContent=txt;return e;}
function svgIn(el,w,h,label){if(typeof el==='string')el=document.getElementById(el);el.innerHTML='';const s=S_('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':label});el.appendChild(s);return s;}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)*(r1-r0)/((d1-d0)||1);
function niceMax(mx){if(!(mx>0))return 1;const p=Math.pow(10,Math.floor(Math.log10(mx)));return [1,1.2,1.5,2,2.5,3,4,5,6,8,10].map(k=>k*p).find(k=>k>=mx*0.999);}
const fmtK=v=>Math.abs(v)>=1000?(v/1000).toLocaleString('en-GB',{maximumFractionDigits:1})+'k':String(Math.round(v*10)/10);
const FONT='Inter, system-ui, sans-serif';
function txt(svg,x,y,s,at){svg.appendChild(S_('text',Object.assign({x,y,'font-size':11,'font-family':FONT,fill:'#9a9ea6'},at||{}),s));}
function hline(svg,x0,x1,y,at){svg.appendChild(S_('line',Object.assign({x1:x0,x2:x1,y1:y,y2:y,stroke:'#eeece6'},at||{})));}
function pathOf(vals,x,y){let d='',pen=false;vals.forEach((v,i)=>{if(v==null||isNaN(v)){pen=false;return;}d+=(pen?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)+' ';pen=true;});return d;}

/* filled area under a series (one polygon per unbroken run, so gaps stay gaps) */
function areaOf(vals,x,y,y0){ let d='',run=[];
  const flush=()=>{ if(run.length>1){ d+='M'+x(run[0][0]).toFixed(1)+' '+y0+' '+run.map(([i,v])=>'L'+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ')+' L'+x(run[run.length-1][0]).toFixed(1)+' '+y0+' Z '; } run=[]; };
  vals.forEach((v,i)=>{ if(v==null||isNaN(v)) flush(); else run.push([i,v]); }); flush(); return d; }

/* ---------- tooltip ---------- */
let TIP=null;
function tipShow(e,html){ if(!TIP)TIP=document.getElementById('tip'); TIP.innerHTML=html; TIP.style.opacity=1;
  const r=TIP.getBoundingClientRect(); let x=e.clientX+14,y=e.clientY-r.height-10;
  if(x+r.width>innerWidth-8)x=e.clientX-r.width-14; if(y<8)y=e.clientY+16; TIP.style.left=x+'px'; TIP.style.top=y+'px'; }
function tipHide(){ if(!TIP)TIP=document.getElementById('tip'); TIP.style.opacity=0; }
function hover(el,html){ el.addEventListener('mousemove',e=>tipShow(e,typeof html==='function'?html():html)); el.addEventListener('mouseleave',tipHide); }

/* ---------- small UI builders ---------- */
function seg(id,opts,val,onPick){ const el=document.getElementById(id);
  el.innerHTML=opts.map(o=>`<button data-v="${o.v}" ${o.disabled?'disabled':''} title="${o.title||''}">${o.l}</button>`).join('');
  el.querySelectorAll('button').forEach(b=>{ b.setAttribute('aria-pressed',b.dataset.v===String(val)); b.onclick=()=>onPick(b.dataset.v); }); }
function sel(id,opts,val,onPick){ const el=document.getElementById(id);
  el.innerHTML=opts.map(o=>o.group!==undefined?`<optgroup label="${o.group}">${o.items.map(i=>`<option value="${i.v}">${i.l}</option>`).join('')}</optgroup>`:`<option value="${o.v}" ${o.disabled?'disabled':''}>${o.l}</option>`).join('');
  el.value=val; el.onchange=()=>onPick(el.value); }

/* ---------- URL hash state (bookmarkable, shared between pages for a/g) ---------- */
function readHash(){ return Object.fromEntries(new URLSearchParams(location.hash.slice(1))); }
function writeHash(obj){ history.replaceState(null,'','#'+new URLSearchParams(obj).toString());
  const shared=new URLSearchParams({a:obj.a||'squad',g:obj.g||'all'}).toString();
  document.querySelectorAll('.tabs a').forEach(a=>{a.href=a.getAttribute('data-page')+'#'+shared;}); }

/* lettered section header: secHead('A','Performance summary','optional description') */
function secHead(letter,title,desc){ return `<div class="sec" role="heading" aria-level="2"><span class="l">${letter}</span><span class="t">${title}</span>${desc?`<span class="d">${desc}</span>`:''}<span class="rule"></span></div>`; }
