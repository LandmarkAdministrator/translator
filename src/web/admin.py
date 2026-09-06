"""Admin surface: login, status, and controls.

Reachable publicly at /admin, so every handler here assumes an untrusted
caller. Read-only status is deliberately generous; anything that changes the
running service requires a valid session AND a POST.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

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
    st["engine"] = running_program()

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


OVERRIDE_FLAG = Path.home() / "translate-manual.flag"
UNIT = Path.home() / ".config" / "systemd" / "user" / "translate.service"
WANT_EXEC = str(Path.home() / "bin" / "start-translate-unified")


def running_program() -> str:
    """The program the service would actually launch, read from the unit.

    Reported rather than assumed: the panel used to hard-code "unified
    streaming" and would have claimed it while the legacy translate.py was
    running, which is exactly the situation an operator needs to see.
    """
    exe = _run(["systemctl", "--user", "show", "translate.service",
                "-p", "ExecStart", "--value"])
    m = re.search(r"path=([^ ;]+)", exe or "")
    path = m.group(1) if m else ""
    if path.endswith("start-translate-unified"):
        return "unified streaming (parakeet-unified-en-0.6b)"
    if path:
        # The legacy program was retired 2026-09-06. Anything else pointing
        # here is a misconfiguration, not a supported mode.
        return f"UNEXPECTED [{Path(path).name}] — should be start-translate-unified"
    return "unknown"


def _ensure_program() -> Optional[str]:
    """Point the unit at the unified launcher before starting it.

    The scheduler enforces this every five minutes, but a button press is not
    the scheduler: without this, Start would happily launch whatever the unit
    last pointed at."""
    exe = _run(["systemctl", "--user", "show", "translate.service",
                "-p", "ExecStart", "--value"])
    if WANT_EXEC in (exe or ""):
        return None
    try:
        text = UNIT.read_text()
        fixed = re.sub(r"^ExecStart=.*$", "ExecStart=%h/bin/start-translate-unified",
                       text, count=1, flags=re.M)
        if fixed != text:
            UNIT.write_text(fixed)
            subprocess.run(["systemctl", "--user", "daemon-reload"], timeout=10)
            return "corrected the service to the unified program first"
    except Exception as e:
        return f"could not correct the program ({e})"
    return None


def do_action(name: str) -> tuple[bool, str]:
    """State-changing operations. Deliberately few and explicit.

    Start and stop both pause the schedule first. Without that the window
    checker — which runs every five minutes — would simply undo them: it stops
    translation outside a service window and starts it inside one, so a bare
    'start' on a Friday would die within five minutes and look like a crash.
    The cost is that the schedule stays paused until it is explicitly resumed,
    which is why the page warns about it.
    """
    if name == "start":
        note = _ensure_program()
        try:
            OVERRIDE_FLAG.touch()
        except Exception as e:
            return False, f"Could not pause the schedule: {e}"
        subprocess.Popen(["systemctl", "--user", "start", "translate.service"])
        msg = ("Starting translation (takes ~40s to load models). The "
               "automatic schedule is PAUSED until you resume it.")
        return True, (msg + " — " + note) if note else msg
    if name == "restart":
        note = _ensure_program()
        subprocess.Popen(["systemctl", "--user", "restart", "translate.service"])
        msg = "Restarting translation — this page will reconnect shortly."
        return True, (msg + " — " + note) if note else msg
    if name == "stop":
        try:
            OVERRIDE_FLAG.touch()
        except Exception as e:
            return False, f"Could not pause the schedule: {e}"
        subprocess.Popen(["systemctl", "--user", "stop", "translate.service"])
        return True, ("Translation stopped. The automatic schedule is PAUSED "
                      "until you resume it.")
    if name == "clear_override":
        try:
            OVERRIDE_FLAG.unlink()
            return True, ("Schedule resumed — services will start and stop "
                          "automatically again.")
        except FileNotFoundError:
            return True, "The schedule was already running automatically."
        except Exception as e:
            return False, f"Could not resume the schedule: {e}"
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
table{width:100%;border-collapse:collapse;font-size:14px;margin-bottom:10px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);font-weight:700;padding:4px 6px;border-bottom:1px solid var(--line)}
td{padding:5px 6px;border-bottom:1px solid var(--line)}
select,input[type=time],input[type=number]{font:inherit;font-size:14px;padding:5px 7px;
border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input[type=number]{width:5rem}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.inl{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center}
button.del{padding:4px 10px;font-size:13px;color:var(--bad);border-color:var(--line)}
.err{color:var(--bad)}
.hint{font-size:12.5px;color:var(--muted);margin:10px 0 0}
#warn{background:var(--warn);color:#fff;border-radius:9px;padding:12px 14px;
margin-bottom:14px;font-size:14px;font-weight:600}
#warn[hidden]{display:none}
</style></head><body><main>
<header><h1>Translation Admin</h1><span id="msg"></span>
<a class="out" href="/admin/logout">Sign out</a></header>
<div class="grid" id="tiles"></div>
<div id="warn" hidden></div>
<section><h2>Controls</h2><div class="actions">
<button class="primary" data-act="start">Start translation</button>
<button data-act="stop">Stop translation</button>
<button data-act="restart">Restart</button>
<button data-act="clear_override">Resume automatic schedule</button>
</div>
<p class="hint">Start and Stop pause the automatic schedule so the window
checker cannot undo them. Use <b>Resume automatic schedule</b> when you are
done, or services will not start on their own.</p></section>
<section><h2>Service schedule</h2>
<table id="wins"><thead><tr><th>Day</th><th>Translation starts</th><th>Ends</th><th></th></tr></thead>
<tbody></tbody></table>
<div class="row"><button id="addwin" type="button">Add a window</button>
<label class="inl">Stop archive work <input id="drain" type="number" min="0" max="180" step="5"> min before</label>
<button class="primary" id="savesched" type="button">Save schedule</button></div>
<p class="hint">Translation starts at the window time, not the service time — it
needs about a minute to load and the room is quiet beforehand. Archive jobs have
run up to 40 minutes, so a drain shorter than that can leave one competing for
the GPU during a service.</p></section>

<section><h2>Audio routing</h2>
<div class="row"><label class="inl">Input <select id="indev"></select></label></div>
<table id="outs"><thead><tr><th>Language</th><th>Output</th><th>Channel</th><th>On</th></tr></thead>
<tbody></tbody></table>
<div class="row"><button class="primary" id="saveaudio" type="button">Save audio routing</button></div>
<p class="hint">Devices are matched by name, so they survive the card renumbering
a reboot can cause. A device already in use by translation may not appear in this
list — its configured name is kept and shown regardless.</p></section>

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
    t+=tile('Schedule',d.manual_override?'PAUSED':'automatic',d.manual_override?'warn':'good');
    var wrongprog=(d.engine||'').indexOf('UNEXPECTED')>=0;
    t+=tile('Program',wrongprog?'WRONG':'unified',wrongprog?'bad':'good');
    document.getElementById('tiles').innerHTML=t;
    var w=document.getElementById('warn');
    var warn='';
    if(wrongprog)warn='⚠ The service is pointed at an unexpected program. Press '+
      'Restart to put it back on the unified stack.';
    else if(d.manual_override)warn='⚠ The automatic schedule is PAUSED. Translation will NOT '+
      'start by itself for the next service. Press "Resume automatic schedule" when done.';
    w.hidden=!warn; if(warn)w.textContent=warn;
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
// ---- settings: schedule and audio routing --------------------------------
var DAYS=['','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
var CFG=null;
function opt(v,label,sel){return '<option value="'+esc(v)+'"'+(sel?' selected':'')+'>'+esc(label)+'</option>';}

function drawSchedule(){
  var tb=document.querySelector('#wins tbody'), h='';
  CFG.schedule.windows.forEach(function(w,i){
    var d=''; for(var n=1;n<=7;n++) d+=opt(n,DAYS[n],n===w.day);
    h+='<tr data-i="'+i+'"><td><select class="wday">'+d+'</select></td>'+
       '<td><input class="wstart" type="time" value="'+esc(w.start)+'"></td>'+
       '<td><input class="wend" type="time" value="'+esc(w.end)+'"></td>'+
       '<td><button class="del" type="button">Remove</button></td></tr>';
  });
  tb.innerHTML=h||'<tr><td colspan="4">No windows — services will never start.</td></tr>';
  document.getElementById('drain').value=CFG.schedule.drain_min;
}
function readSchedule(){
  var ws=[];
  document.querySelectorAll('#wins tbody tr[data-i]').forEach(function(tr){
    ws.push({day:parseInt(tr.querySelector('.wday').value,10),
             start:tr.querySelector('.wstart').value,
             end:tr.querySelector('.wend').value});
  });
  return {drain_min:parseInt(document.getElementById('drain').value,10)||0, windows:ws};
}
function drawAudio(){
  var outs=CFG.devices.outputs||[], ins=CFG.devices.inputs||[];
  // A device in use by translation vanishes from enumeration; keep the
  // configured name in the list so saving does not silently change it.
  function withCurrent(list,cur){
    var names=list.map(function(d){return d.name;});
    if(cur && names.indexOf(cur)<0) return [{name:cur,missing:true}].concat(list);
    return list;
  }
  var iv=CFG.audio.input_device, ih='';
  withCurrent(ins,iv).forEach(function(d){
    ih+=opt(d.name,d.name+(d.missing?'  (in use / not detected)':''),d.name===iv);});
  document.getElementById('indev').innerHTML=ih;
  var tb=document.querySelector('#outs tbody'), h='';
  (CFG.audio.languages||[]).forEach(function(l,i){
    var oh=''; withCurrent(outs,l.output_device).forEach(function(d){
      oh+=opt(d.name,d.name+(d.missing?'  (in use / not detected)':''),d.name===l.output_device);});
    var ch=l.output_channel, cs=opt('','Both',ch===null||ch===undefined)+opt('0','Left',ch===0)+opt('1','Right',ch===1);
    h+='<tr data-code="'+esc(l.code)+'"><td>'+esc(l.name||l.code)+'</td>'+
       '<td><select class="odev">'+oh+'</select></td>'+
       '<td><select class="och">'+cs+'</select></td>'+
       '<td><input class="oen" type="checkbox"'+(l.enabled?' checked':'')+'></td></tr>';
  });
  tb.innerHTML=h;
}
function readAudio(){
  var ls=[];
  document.querySelectorAll('#outs tbody tr[data-code]').forEach(function(tr){
    var c=tr.querySelector('.och').value;
    ls.push({code:tr.dataset.code, output_device:tr.querySelector('.odev').value,
             output_channel:c===''?null:parseInt(c,10),
             enabled:tr.querySelector('.oen').checked});
  });
  return {input_device:document.getElementById('indev').value, languages:ls};
}
function loadConfig(){
  fetch('/admin/api/config',{credentials:'same-origin'}).then(function(r){
    if(r.status===401){location.href='/admin';return null;} return r.json();
  }).then(function(d){ if(!d)return; CFG=d; drawSchedule(); drawAudio(); }).catch(function(){});
}
function saveConfig(kind,data,btn){
  var m=document.getElementById('msg'); m.className=''; m.textContent='Saving…';
  fetch('/admin/api/config',{method:'POST',credentials:'same-origin',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:kind,data:data})})
   .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
   .then(function(res){
     m.textContent=res.j.message||''; m.className=res.ok?'':'err';
     if(res.ok) loadConfig();
   }).catch(function(){ m.textContent='Could not reach the server.'; m.className='err'; });
}
document.getElementById('addwin').addEventListener('click',function(){
  CFG.schedule=readSchedule();
  CFG.schedule.windows.push({day:7,start:'09:15',end:'12:45'}); drawSchedule();
});
document.querySelector('#wins').addEventListener('click',function(e){
  if(!e.target.classList.contains('del'))return;
  var tr=e.target.closest('tr'); CFG.schedule=readSchedule();
  CFG.schedule.windows.splice(parseInt(tr.dataset.i,10),1); drawSchedule();
});
document.getElementById('savesched').addEventListener('click',function(){
  saveConfig('schedule',readSchedule(),this);});
document.getElementById('saveaudio').addEventListener('click',function(){
  saveConfig('audio',readAudio(),this);});

refresh(); setInterval(refresh,5000); loadConfig();
</script></body></html>"""
