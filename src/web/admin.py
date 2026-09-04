"""Admin surface: login, status, and controls.

Reachable publicly at /admin, so every handler here assumes an untrusted
caller. Read-only status is deliberately generous; anything that changes the
running service requires a valid session AND a POST.
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import time
from pathlib import Path

COOKIE = "tr_admin"


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def gather_status() -> dict:
    """Everything the status panel shows. Never raises."""
    st: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    st["service"] = _run(["systemctl", "--user", "is-active", "translate.service"]) or "unknown"
    st["engine"] = "unified streaming (parakeet-unified-en-0.6b)"

    if shutil.which("nvidia-smi"):
        gpu = _run(["nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"])
        parts = [p.strip() for p in gpu.split(",")] if gpu else []
        if len(parts) == 4:
            st["gpu"] = {"temp_c": parts[0], "util_pct": parts[1],
                         "mem_used_mb": parts[2], "mem_total_mb": parts[3]}

    # Scheduler state: which window we are in, and whether overridden.
    flag = Path.home() / "translate-manual.flag"
    st["manual_override"] = flag.exists()

    log = Path.home() / "translate.log"
    if log.exists():
        st["log_age_sec"] = int(time.time() - log.stat().st_mtime)
        try:
            with open(log, "rb") as fh:
                fh.seek(max(0, fh.seek(0, 2) - 60000))
                tail = fh.read().decode("utf-8", "replace").splitlines()
            import re
            ansi = re.compile(r"\x1b\[[0-9;]*m")
            recent = [ansi.sub("", ln) for ln in tail]
            st["sentences_seen"] = sum(1 for ln in recent if "mode=streaming/sentence" in ln)
            st["errors_seen"] = sum(1 for ln in recent if "| ERROR " in ln)
            st["recent"] = [ln[:160] for ln in recent
                            if "[EN]" in ln or "| ERROR " in ln][-8:]
        except Exception:
            pass

    wlog = Path.home() / "sermons" / "logs" / "translate-window.log"
    if wlog.exists():
        try:
            st["scheduler"] = wlog.read_text(errors="replace").splitlines()[-5:]
        except Exception:
            pass
    return st


def do_action(name: str) -> tuple[bool, str]:
    """State-changing operations. Deliberately few and explicit."""
    if name == "restart":
        subprocess.Popen(["systemctl", "--user", "restart", "translate.service"])
        return True, "Restarting translation — this page will reconnect shortly."
    if name == "stop":
        subprocess.Popen(["systemctl", "--user", "stop", "translate.service"])
        return True, "Translation stopped. The scheduler will restart it in its next window."
    if name == "clear_override":
        try:
            (Path.home() / "translate-manual.flag").unlink()
            return True, "Manual override cleared — the schedule is live again."
        except FileNotFoundError:
            return True, "No override was set."
        except Exception as e:
            return False, f"Could not clear override: {e}"
    return False, "Unknown action."


LOGIN_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translation Admin</title><style>
:root{--bg:#F5F4F0;--card:#fff;--ink:#23211C;--muted:#8A8578;--line:#DDD9CF;--accent:#175E54}
@media(prefers-color-scheme:dark){:root{--bg:#14130F;--card:#1E1C17;--ink:#EAE7DF;--muted:#928C7D;--line:#33302A;--accent:#5FB3A5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;display:grid;place-items:center;min-height:100dvh;padding:20px}
form{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:24px;width:100%;max-width:22rem;display:grid;gap:12px}
h1{font-size:19px;margin:0 0 4px}label{font-size:13px;color:var(--muted)}
input{font:inherit;padding:9px 11px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);width:100%}
button{font:inherit;font-weight:600;padding:10px;border:0;border-radius:7px;background:var(--accent);color:#fff;cursor:pointer}
.err{color:#97372C;font-size:14px}
@media(prefers-color-scheme:dark){.err{color:#DE8878}}
</style></head><body><form method="POST" action="/admin/login">
<h1>Translation Admin</h1>__ERR__
<div><label for="u">Username</label><input id="u" name="username" autocomplete="username" autocapitalize="none" required></div>
<div><label for="p">Password</label><input id="p" name="password" type="password" autocomplete="current-password" required></div>
<button type="submit">Sign in</button></form></body></html>"""


def login_page(error: str = "") -> bytes:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return LOGIN_PAGE.replace("__ERR__", err).encode()


ADMIN_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translation Admin</title><style>
:root{--bg:#F5F4F0;--card:#fff;--ink:#23211C;--muted:#8A8578;--line:#DDD9CF;
--accent:#175E54;--good:#1F6F4A;--bad:#97372C;--warn:#91621A;--code:#EEEBE3}
@media(prefers-color-scheme:dark){:root{--bg:#14130F;--card:#1E1C17;--ink:#EAE7DF;--muted:#928C7D;
--line:#33302A;--accent:#5FB3A5;--good:#6BC195;--bad:#DE8878;--warn:#DCAF63;--code:#100F0C}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:52rem;margin:0 auto;padding:18px 16px 60px}
header{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
h1{font-size:20px;margin:0 auto 0 0}
a.out{font-size:14px;color:var(--muted)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.tile .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.tile .v{font-size:20px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.v.good{color:var(--good)}.v.bad{color:var(--bad)}.v.warn{color:var(--warn)}
section{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px;margin-bottom:14px}
h2{font-size:15px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
pre{margin:0;background:var(--code);border-radius:6px;padding:10px;font-size:12.5px;
overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.actions{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;font-size:14px;padding:9px 14px;border-radius:7px;border:1px solid var(--line);
background:var(--bg);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
#msg{font-size:14px;color:var(--good);min-height:1.2em}
</style></head><body><main>
<header><h1>Translation Admin</h1><span id="msg"></span>
<a class="out" href="/admin/logout">Sign out</a></header>
<div class="grid" id="tiles"></div>
<section><h2>Controls</h2><div class="actions">
<button class="primary" data-act="restart">Restart translation</button>
<button data-act="stop">Stop translation</button>
<button data-act="clear_override">Clear manual override</button>
</div></section>
<section><h2>Recent activity</h2><pre id="recent">…</pre></section>
<section><h2>Scheduler</h2><pre id="sched">…</pre></section>
</main><script>
function tile(k,v,cls){return '<div class="tile"><div class="k">'+k+'</div><div class="v '+(cls||'')+'">'+v+'</div></div>';}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function refresh(){
  fetch('/admin/api/status',{credentials:'same-origin'}).then(function(r){
    if(r.status===401){location.href='/admin';return null;} return r.json();
  }).then(function(d){
    if(!d)return;
    var t='';
    t+=tile('Service',esc(d.service),d.service==='active'?'good':'warn');
    if(d.gpu){t+=tile('GPU',esc(d.gpu.temp_c)+'&deg;C',Number(d.gpu.temp_c)>=80?'warn':'good');
              t+=tile('GPU memory',esc(d.gpu.mem_used_mb)+' MB');}
    if(d.log_age_sec!==undefined)t+=tile('Log age',esc(d.log_age_sec)+'s',d.log_age_sec>180?'warn':'good');
    if(d.sentences_seen!==undefined)t+=tile('Sentences',esc(d.sentences_seen));
    if(d.errors_seen!==undefined)t+=tile('Errors',esc(d.errors_seen),d.errors_seen>0?'bad':'good');
    t+=tile('Override',d.manual_override?'SET':'clear',d.manual_override?'warn':'good');
    document.getElementById('tiles').innerHTML=t;
    document.getElementById('recent').textContent=(d.recent||['(nothing yet)']).join('\\n');
    document.getElementById('sched').textContent=(d.scheduler||['(no entries)']).join('\\n');
  }).catch(function(){});
}
document.querySelector('.actions').addEventListener('click',function(e){
  var b=e.target.closest('button[data-act]'); if(!b)return;
  var act=b.dataset.act;
  if(act!=='clear_override'&&!confirm('Really '+act.replace('_',' ')+'? This affects the live service.'))return;
  fetch('/admin/api/action',{method:'POST',credentials:'same-origin',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({action:act})})
   .then(function(r){return r.json();}).then(function(d){
     document.getElementById('msg').textContent=d.message||'';
     setTimeout(refresh,1500);
   }).catch(function(){});
});
refresh(); setInterval(refresh,5000);
</script></body></html>"""
