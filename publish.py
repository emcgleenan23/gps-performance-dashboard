"""Build a shareable copy of the dashboard for GitHub Pages.

    python publish.py --public   # synthetic/demo data only: plain self-contained pages, no password
    python publish.py            # real data: encrypted pages, asks for the password (never stored)
    DASH_PASSWORD=... python publish.py

What it does
  1. Inlines each page (dashboard/index.html, dashboard/load.html) into ONE self-contained file:
     CSS, shared JS and the pipeline data (outputs/*.js) are embedded, nothing is fetched.
  2. gzips it and encrypts it with AES-256-GCM. The key is derived from your password with
     PBKDF2-SHA256 (600,000 iterations, random salt per build) - the same scheme StatiCrypt uses.
  3. Writes docs/<page>.html: a small unlock page that decrypts in the browser (Web Crypto API).
     Without the password the files are just random bytes, so it is safe for the repo/Pages
     site to be public. Nothing unencrypted is written to docs/.

Share the Pages URL and the password separately (e.g. link by email, password by text).
Change the password = run this again with a new one and push; old links stop opening.
"""
import base64
import getpass
import gzip
import json
import os
import pathlib
import re
import secrets
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = pathlib.Path(__file__).parent
SRC, DATA, OUT = ROOT / "dashboard", ROOT / "outputs", ROOT / "docs"
PAGES = {"index.html": "Session analysis", "load.html": "Load monitoring"}
ITER = 600_000
MIN_LEN = 12


def inline(page: str) -> str:
    html = (SRC / page).read_text()

    def js(path):  # keep "</script>" inside data from closing the tag
        return path.read_text().replace("</script", "<\\/script")

    def sub_script(m):
        src = m.group(1)
        p = (SRC / src).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{page}: {src} not found - run `python run.py` first")
        return f"<script>/* {p.name} */\n{js(p)}\n</script>"

    html = re.sub(r'<script src="([^"]+)"></script>', sub_script, html)
    html = html.replace('<link rel="stylesheet" href="common.css">', f"<style>\n{(SRC / 'common.css').read_text()}\n</style>")
    # the in-page password gate is cosmetic; the encryption replaces it
    html = html.replace("<head>", "<head><script>window.__PUBLISHED=true</script>", 1)
    return html


def encrypt(plain: bytes, password: str, salt: bytes) -> dict:
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    iv = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(iv, gzip.compress(plain, 9), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "iter": ITER, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


UNLOCK = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow"><title>__TITLE__</title>
<style>
  :root{--bg:#15171c;--card:#20232b;--ink:#f6f4ef;--mut:#a7abb4;--acc:#e4571f}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;display:flex;align-items:center;justify-content:center;padding:24px}
  .c{width:100%;max-width:380px} .m{width:64px;height:5px;background:var(--acc);margin-bottom:22px}
  h1{font-size:25px;line-height:1.2;margin:0 0 6px} p{color:var(--mut);font-size:14px;line-height:1.5;margin:0 0 20px}
  form{display:flex;gap:8px} input[type=password]{flex:1;font-size:15px;padding:12px 14px;border-radius:10px;border:1px solid #33363f;background:var(--card);color:var(--ink)}
  input:focus,button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
  button{font-size:15px;font-weight:600;padding:12px 18px;border:0;border-radius:10px;background:var(--acc);color:#fff;cursor:pointer}
  button:disabled{opacity:.6;cursor:progress}
  label{display:flex;gap:8px;align-items:center;margin-top:14px;font-size:13px;color:var(--mut)}
  #e{min-height:18px;margin-top:12px;font-size:13px;color:#f0a58a}
  .n{margin-top:26px;font-size:12px;color:#6c7079;line-height:1.5}
</style></head>
<body><div class="c"><div class="m"></div><h1>Performance dashboard</h1><p>__TITLE__ · private. Enter the password to open.</p>
<form id="f"><input id="pw" type="password" placeholder="Password" autocomplete="current-password" aria-label="Password" autofocus required><button id="b">Open</button></form>
<label><input type="checkbox" id="rem"> Remember on this device</label>
<div id="e" role="alert"></div>
<div class="n">Encrypted with AES-256. Contains pseudonymised athlete data — please don't forward.</div></div>
<script>
const P=__PAYLOAD__;
const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const KEY='gpsdash_'+P.salt.slice(0,12);                        // same salt on every page of one build
async function keyFrom(pw){const base=await crypto.subtle.importKey('raw',new TextEncoder().encode(pw),'PBKDF2',false,['deriveBits']);
  return new Uint8Array(await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt:b64(P.salt),iterations:P.iter},base,256));}
async function decrypt(raw){
  const k=await crypto.subtle.importKey('raw',raw,'AES-GCM',false,['decrypt']);
  const gz=await crypto.subtle.decrypt({name:'AES-GCM',iv:b64(P.iv)},k,b64(P.ct));      // throws on a wrong password
  return await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
}
function show(html){ document.open(); document.write(html); document.close(); }
const store=(raw,persist)=>{const s=btoa(String.fromCharCode(...raw)); try{sessionStorage.setItem(KEY,s); if(persist)localStorage.setItem(KEY,s);}catch(e){}};
(async()=>{
  if(location.search.includes('logout')){try{sessionStorage.removeItem(KEY);localStorage.removeItem(KEY);}catch(e){} history.replaceState(null,'',location.pathname+location.hash);}
  let saved=null; try{saved=sessionStorage.getItem(KEY)||localStorage.getItem(KEY);}catch(e){}
  if(saved){ try{ show(await decrypt(b64(saved))); return; }catch(e){ try{sessionStorage.removeItem(KEY);localStorage.removeItem(KEY);}catch(_){} } }
})();
document.getElementById('f').addEventListener('submit',async ev=>{ev.preventDefault();
  const b=document.getElementById('b'), err=document.getElementById('e'); b.disabled=true; err.textContent='';
  try{ const raw=await keyFrom(document.getElementById('pw').value); const html=await decrypt(raw); store(raw,document.getElementById('rem').checked); show(html); }
  catch(e){ err.textContent='That password didn\'t open the dashboard.'; b.disabled=false; document.getElementById('pw').select(); }});
</script></body></html>
"""


def public():
    """Unencrypted build - ONLY for the synthetic sample squad."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.config import load_config
    root = pathlib.Path(load_config(str(ROOT / "config.yaml"))["data"]["root"]).expanduser()
    if root.resolve() != (ROOT / "sample_data" / "raw").resolve():
        sys.exit(f"--public refuses to run: data.root is {root}, not the synthetic sample_data/raw.\n"
                 "Use the encrypted build (python publish.py) for real athlete data.")
    OUT.mkdir(exist_ok=True)
    for page in PAGES:
        html = (inline(page).replace('<meta name="robots" content="noindex, nofollow">', "")
                .replace("Pipeline output</span>", "Synthetic demo data</span>"))
        (OUT / page).write_text(html)
        print(f"[ok] docs/{page}: {len(html) / 1e6:.1f} MB (public, unencrypted)")
    (OUT / ".nojekyll").write_text("")
    rb = OUT / "robots.txt"
    if rb.exists():
        rb.unlink()
    print("[ok] docs/ holds a public demo. Never use --public with real athlete data.")


def main():
    if "--public" in sys.argv:
        return public()
    pw = os.environ.get("DASH_PASSWORD") or getpass.getpass("Dashboard password: ")
    if len(pw) < MIN_LEN:
        sys.exit(f"Use a password of at least {MIN_LEN} characters (a short phrase is ideal).")
    if not os.environ.get("DASH_PASSWORD") and getpass.getpass("Repeat password: ") != pw:
        sys.exit("Passwords didn't match.")
    OUT.mkdir(exist_ok=True)
    salt = secrets.token_bytes(16)          # one salt per build -> unlock once, both pages open
    for page, title in PAGES.items():
        html = inline(page)
        payload = encrypt(html.encode(), pw, salt)
        (OUT / page).write_text(UNLOCK.replace("__TITLE__", title).replace("__PAYLOAD__", json.dumps(payload)))
        print(f"[ok] docs/{page}: {len(html) / 1e6:.1f} MB page -> {(OUT / page).stat().st_size / 1e6:.1f} MB encrypted")
    (OUT / ".nojekyll").write_text("")
    (OUT / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print("[ok] docs/ is ready to commit. Nothing in it is readable without the password.")


if __name__ == "__main__":
    main()
