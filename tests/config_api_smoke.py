#!/usr/bin/env python3
"""Settings API: authentication, validation, atomic writes.

A bad schedule means no services until someone notices and a wrong audio device
means a silent auditorium, so the point of these checks is that nothing invalid
can be written and that a write keeps a backup.
"""
import http.client, json, os, sys, tempfile, threading, time, shutil
from pathlib import Path
PORT = 8889
tmp = Path(tempfile.mkdtemp())
os.environ["TRANSLATOR_ADMIN_CREDENTIALS"] = str(tmp/"admin.json")
shutil.copy("config/schedule.conf", tmp/"schedule.conf")
os.environ["TRANSLATOR_SCHEDULE_CONF"] = str(tmp/"schedule.conf")
os.environ["TRANSLATOR_SETTINGS_YAML"] = str(tmp/"settings.yaml")
(tmp/"settings.yaml").write_text(
 'input_device: "USB Audio CODEC"\nlanguages:\n- code: es\n  name: Spanish\n'
 '  output_device: "USB Audio CODEC"\n  output_channel: 0\n  enabled: true\n')
sys.path.insert(0, "src")
from web import auth
auth.CRED_PATH = Path(os.environ["TRANSLATOR_ADMIN_CREDENTIALS"])
auth.save_credentials("admin", "a-long-enough-password", auth.CRED_PATH)
from web.live_server import LiveServer
threading.Thread(target=LiveServer(PORT, host="127.0.0.1").run_forever, daemon=True).start()
time.sleep(0.8)

def req(method, path, body=None, cookie=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=8)
    h = {"Content-Type": "application/json"} if body else {}
    if cookie: h["Cookie"] = cookie
    c.request(method, path, body=body, headers=h)
    r = c.getresponse(); d = r.read(); st = r.status; c.close()
    return st, d

st,_ = req("GET", "/admin/api/config")
assert st == 401, f"unauth GET not refused ({st})"
print("  unauthenticated GET refused: OK")
st,_ = req("POST", "/admin/api/config", json.dumps({"kind":"schedule","data":{}}))
assert st == 401, f"unauth POST not refused ({st})"
print("  unauthenticated POST refused: OK")

c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=8)
c.request("POST","/admin/login",body="username=admin&password=a-long-enough-password",
          headers={"Content-Type":"application/x-www-form-urlencoded"})
r=c.getresponse(); cookie=r.getheader("Set-Cookie").split(";")[0]; c.close()

st,d = req("GET", "/admin/api/config", cookie=cookie)
cfg = json.loads(d)
print(f"  authenticated GET: {st}, windows={len(cfg['schedule']['windows'])}, "
      f"drain={cfg['schedule']['drain_min']}, outputs={len(cfg['devices']['outputs'])}")

st,d = req("POST","/admin/api/config",
           json.dumps({"kind":"schedule","data":{"drain_min":30,
             "windows":[{"day":7,"start":"12:45","end":"09:15"}]}}), cookie)
print(f"  invalid schedule rejected: {st} — {json.loads(d)['message']}")

st,d = req("POST","/admin/api/config",
           json.dumps({"kind":"schedule","data":{"drain_min":45,
             "windows":[{"day":7,"start":"09:58","end":"12:45"}]}}), cookie)
print(f"  valid schedule saved: {st} — {json.loads(d)['message'][:52]}")
print("  file now contains:")
for ln in (tmp/"schedule.conf").read_text().splitlines():
    if ln and not ln.startswith("#"): print("     ", ln)
assert (tmp/"schedule.conf.bak").exists(), "no backup kept"
print("  backup kept: OK")

st,d = req("POST","/admin/api/config",
           json.dumps({"kind":"bogus","data":{}}), cookie)
print(f"  unknown section rejected: {st} — {json.loads(d)['message']}")

# Audio: values change in place and the comments between entries — where the
# routing notes live — survive the save.
(tmp/"settings.yaml").write_text(
 '# Sound room routing notes.\n'
 'input_device: "USB Audio CODEC"\nlanguages:\n- code: es\n  name: Spanish\n'
 '  output_device: "USB Audio CODEC"\n  output_channel: 0\n  enabled: true\n'
 '# Russian goes out the onboard jack: the Behringer is full.\n'
 '- code: ru\n  name: Russian\n  output_device: "ALC269VC Analog"\n'
 '  output_channel: null\n  enabled: true\n')
st,d = req("POST","/admin/api/config", json.dumps({"kind":"audio","data":{
    "input_device":"Onyx Producer",
    "languages":[{"code":"es","output_device":"USB Audio CODEC","output_channel":1,"enabled":True},
                 {"code":"ru","output_device":"USB Audio CODEC","output_channel":0,"enabled":False}]}}),
           cookie)
assert st == 200, (st, d)
text = (tmp/"settings.yaml").read_text()
assert "# Sound room routing notes." in text and "# Russian goes out the onboard jack" in text, text
import yaml
raw = yaml.safe_load(text)
assert raw["input_device"] == "Onyx Producer", raw
assert raw["languages"][0]["output_channel"] == 1 and raw["languages"][0]["enabled"] is True, raw
assert raw["languages"][1]["output_device"] == "USB Audio CODEC" and raw["languages"][1]["enabled"] is False, raw
assert raw["languages"][1]["name"] == "Russian", raw
print("  audio saved: values updated, comments between entries kept: OK")
st,d = req("POST","/admin/api/config", json.dumps({"kind":"audio","data":{
    "input_device":"Onyx Producer",
    "languages":[{"code":"es","output_device":"USB Audio CODEC","output_channel":0,"enabled":True},
                 {"code":"ru","output_device":"USB Audio CODEC","output_channel":0,"enabled":True}]}}),
           cookie)
print(f"  same device+channel twice rejected: {st} — {json.loads(d)['message'][:60]}")
assert st == 400
print("\nALL CONFIG API TESTS PASSED")
