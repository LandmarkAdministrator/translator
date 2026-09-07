#!/usr/bin/env python3
"""Hard numbers for a day's services, from ~/translate.log on the Translate PC.

Run it ON the host (it reads journald for the service's start/stop events):

    ssh administrator@10.1.170.184 'python3 -' < tests/service_tally.py

Per bucket (morning service, afternoon tests, evening service): sentences per
minute, how many end in a sentence mark / start with a stray one / hold two
sentences, words per segment, the segments cut by the cap or a timeout,
10-minute bins with first-word-to-audio latency, per-language latency and
queue depth, speed-ups, errors, drops, restarts.

Written 2026-09-06 for the first live service of the punctuation-boundary
flush policy; the bucket boundaries (13:00, 18:00) suit Sunday's schedule and
are trivial to change.
"""
"""
Runs come from journald (Started / stopped events of translate.service today)
— the only reliable start markers: the coordinator's own banner is
block-buffered and lands late in the file, and the NeMo server's banner has
no timestamp. Lines are today's tail of the log (walked back until the
time-of-day stops decreasing, i.e. the previous day). Buckets: morning
(start < 13:00, the old policy), afternoon tests (13:00-18:00), evening
service (>= 18:00).
"""
import os, re, statistics, collections, subprocess

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^(\d\d:\d\d:\d\d) \|")
EN = re.compile(r"^(\d\d:\d\d:\d\d) \|.*?\[EN\] (.*?) \| mode=streaming/sentence")
TR = re.compile(r"^(\d\d:\d\d:\d\d) \|.*?\[(ES|HT|RU)\] (.*?) \| e2e=([\d.]+)s \| translate=([\d.]+)s "
                r"\| tts=([\d.]+)s \| audio=([\d.]+)s \| queue_was=(\d+)")
SPEED = re.compile(r"^(\d\d:\d\d:\d\d) \|.*?\[(ES|HT|RU)\] queue_depth=(\d+) → speaking at ([\d.]+)x")
END = re.compile(r"[.?!][\"')\]]?\s*$")
TWO = re.compile(r"[.?!][\"')\]]?\s+\S")
LEAD = re.compile(r"^\s*[.?!,;:]")

# ---- runs from journald ---------------------------------------------------
J = re.compile(r"^\w{3} \d\d (\d\d:\d\d:\d\d) .*?: (.*)$")
runs, cur = [], None
out = subprocess.run(["journalctl", "--user", "-u", "translate.service", "--since", "today",
                      "--no-pager", "-o", "short"], capture_output=True, text=True).stdout
for l in out.splitlines():
    m = J.match(l)
    if not m:
        continue
    t, msg = m.groups()
    if msg.startswith("Started"):
        cur = {"start": t, "end": "23:59:59", "how": ""}
        runs.append(cur)
    elif cur and ("Deactivated" in msg or "Main process exited" in msg or msg.startswith("Stopping")):
        if cur["end"] == "23:59:59":
            cur["end"] = t
            cur["how"] = msg[:70]

# ---- today's tail of the log -----------------------------------------------
lines = [ANSI.sub("", l.rstrip("\n")) for l in open(os.path.expanduser("~/translate.log"), errors="replace")]
last_t, cut = "99:99:99", 0
for i in range(len(lines) - 1, -1, -1):
    m = TS.match(lines[i])
    if m:
        if m.group(1) > last_t:      # time went UP going backwards: previous day
            cut = i + 1
            break
        last_t = m.group(1)
lines = lines[cut:]

def in_runs(t, group):
    return any(r["start"] <= t < r["end"] for r in group)

def pct(v):
    return f"{100 * v:.1f}%"

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")

def report(label, group):
    print(f"\n=== {label}: {len(group)} start(s): "
          + ", ".join(f"{r['start']}-{r['end'][:5]}" + (f" ({r['how']})" if r['how'] and 'Deactivated' not in r['how'] else "") for r in group)
          + (f"   [{len(group) - 1} restart(s)]" if len(group) > 1 else ""))
    if not group:
        return
    sents = [(m.group(1), m.group(2).strip()) for m in map(EN.match, lines) if m and in_runs(m.group(1), group)]
    trs = [m.groups() for m in map(TR.match, lines) if m and in_runs(m.group(1), group)]
    speeds = [m.groups() for m in map(SPEED.match, lines) if m and in_runs(m.group(1), group)]
    stamped = [(TS.match(l).group(1), l) for l in lines if TS.match(l)]
    seg = [l for t, l in stamped if in_runs(t, group)]
    errors = sum(1 for l in seg if "| ERROR " in l or "Playback error" in l)
    drops = sum(1 for l in seg if "DROP |" in l)
    frags = sum(1 for l in seg if "[EN-frag]" in l)
    if not sents:
        print("    no sentences")
        return
    first, last = sents[0][0], sents[-1][0]
    h = lambda t: int(t[:2]) * 60 + int(t[3:5]) + int(t[6:8]) / 60
    mins = max(1e-9, h(last) - h(first))
    texts = [t for _, t in sents]
    words = [len(t.split()) for t in texts]
    print(f"    {first} -> {last} ({mins:.0f} min): {len(sents)} sentences from {frags} fragments "
          f"({len(sents) / mins:.1f}/min) | errors {errors} | drops {drops}")
    print(f"    ends with .?!   {pct(sum(bool(END.search(t)) for t in texts) / len(texts))}")
    print(f"    leads with mark {pct(sum(bool(LEAD.match(t)) for t in texts) / len(texts))}")
    print(f"    two sentences   {pct(sum(bool(TWO.search(t)) for t in texts) / len(texts))}")
    print(f"    words/segment   median {statistics.median(words):.0f}  p90 {q(words, .9)}  max {max(words)}"
          f"  | >=60 words (cap-sized) {sum(w >= 60 for w in words)}")
    unp = [w for t, w in zip(texts, words) if not END.search(t)]
    if unp:
        print(f"    unpunctuated segments (cut by cap/timeout/silence): {len(unp)} — words median {statistics.median(unp):.0f} p90 {q(unp, .9)}")
    bins = collections.OrderedDict()
    for ts, t in sents:
        bins.setdefault(ts[:4] + "0", []).append(bool(END.search(t)))
    e2e_bins = collections.OrderedDict()
    for r in trs:
        if r[1] == "ES":
            e2e_bins.setdefault(r[0][:4] + "0", []).append(float(r[3]))
    print("    10-min bins (sentences | ending with .?! | cut unpunctuated | ES latency first word -> audio, median / p95):")
    for b, v in bins.items():
        e = e2e_bins.get(b, [])
        lat = f"{statistics.median(e):4.1f}s / {q(e, .95):4.1f}s" if e else "   -"
        print(f"      {b}  {len(v):3d}  {100 * sum(v) / len(v):3.0f}%  {len(v) - sum(v):3d}   {lat}")
    if trs:
        print("    per language (translation events):")
        for lang in ("ES", "HT", "RU"):
            rows = [r for r in trs if r[1] == lang]
            if not rows:
                continue
            e2e = [float(r[3]) for r in rows]; tr = [float(r[4]) for r in rows]
            tts = [float(r[5]) for r in rows]; au = [float(r[6]) for r in rows]; qd = [int(r[7]) for r in rows]
            print(f"      {lang}: n={len(rows)}  e2e median {statistics.median(e2e):.1f}s p95 {q(e2e, .95):.1f}s max {max(e2e):.1f}s"
                  f" | translate median {statistics.median(tr):.2f}s | tts median {statistics.median(tts):.2f}s"
                  f" | audio median {statistics.median(au):.1f}s | queue>0 {pct(sum(x > 0 for x in qd) / len(qd))} max {max(qd)}")
    if speeds:
        c = collections.Counter((l, f) for _, l, _, f in speeds)
        print("    speed-ups: " + ", ".join(f"{l} {f}x ×{k}" for (l, f), k in sorted(c.items())))
    else:
        print("    speed-ups: none (queues never deep enough)")

print(f"today's tail of translate.log: {len(lines)} lines; translate.service runs today: {len(runs)}")
report("MORNING service (old policy)", [r for r in runs if r["start"] < "13:00:00"])
report("AFTERNOON tests (new policy)", [r for r in runs if "13:00:00" <= r["start"] < "18:00:00"])
report("EVENING service (new policy)", [r for r in runs if r["start"] >= "18:00:00"])
