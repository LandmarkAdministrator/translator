# 2026-09-06 (evening) — code review follow-up (deployed 2026-09-07)

A full read of the production path after the day's work, the decisions taken
on each finding, and what changed. **Deployed to the production host on
Monday 2026-09-07, 16:08–16:15 EDT** (a day early, outside any window; the
backlog worker happened to be idle). §5 is the checklist that was followed,
each step verified; the dry run through the unit showed `protocol 2`, a
heartbeat 24 s after start, biasing on, the right environment, no errors.

The review itself (what the system does, how, and the findings) was delivered
in conversation; the findings are restated here with their outcome.

---

## 1. Findings and decisions

| # | Finding | Decision | Outcome |
|---|---------|----------|---------|
| A1 | Scheduler force-stops the backlog with a bare `pkill -f`, which matches any process whose *arguments* mention the pattern — the trap its own `worker_running()` avoids | verify, then fix | Verified (a decoy `sleep` was matched); fixed — kills only comm-validated PIDs |
| A2 | ASR pipe `readline()` has no deadline; a wedged NeMo server leaves the service silent until the 900 s watchdog | fix, but silence must never count as failure | Fixed with a 30 s reply deadline; silence is unaffected because the server answers every push (empty when nothing was said). A dead backend now exits the service for systemd to restart |
| A3 | No backpressure on the input deque if ASR fell behind | accepted as-is (any modern CPU keeps up; older hardware would not work anyway) | No change |
| B1 | A thermal trip sets the manual flag and cancels all later services | not worth fixing — guard is being retired when the fan part arrives | No change |
| B2 | ASR server re-sends the whole token stream every 1.5 s | explained; fix | Server now sends only what is new (protocol 2) — see §2 |
| B3 | `_internal_split` split at the *last* sentence mark, so "A. B. C" went out as "A. B." | double-check; fix if no reason | Confirmed by test, no reason — fixed to the first mark; `tick()` now drains a complete sentence too |
| B4 | No unit tests on the core pipeline | build them if useful | Two added: pipeline configuration (would have caught the Russian crash), ASR client with fake servers |
| B5 | Certificate renewal restarts the web server; must not land in a service window | early morning only | Timer changed to 03:20 daily, no catch-up on boot |
| C | Hygiene: dead Whisper path, coupling into it, stale launcher defaults, site value in source, comment-destroying YAML writes, unbounded lockout table, committed scratch paths, no NeMo manifest | only after checking nothing breaks | All done; checked by the full test set plus end-to-end runs (§4) |
| — | Found by the evening tally: the hang watchdog restarted a healthy service at 20:30 after 15 min of post-service silence | (user's A2 rule: silence is never a failure) | `HEARTBEAT` every 60 s from the audio path; `STALL_SECONDS` 900 → 300 |
| — | The ASR module logged through stdlib `logging`, which nothing routes: "Parakeet loaded" and the stall-kill message never reached the service log | — | Module moved to loguru; the server's protocol is announced on start |

Also fixed on the way, because the end-to-end test found it: `start()` opened
the audio input *before* setting `_running`, and the callback drops chunks
until it is set. A paced input never noticed (first chunk after 1.5 s); a
`--no-realtime` file run lost the entire file. `_running` is now set first.

---

## 2. What changed, by area

### ASR subprocess link (`parakeet_asr.py`, `unified_asr_server.py`)

- **Reply deadline.** `_RemoteUnifiedModel` reads the pipe's raw descriptor
  with `select()` against `UNIFIED_REPLY_TIMEOUT` (default 30 s; decode is
  ~165 ms per chunk). No reply in time → the server is killed and the call
  raises. Silence cannot trip this: the server replies to *every* push, with
  an empty delta when nothing was said (`tests/test_asr_client.py` checks
  exactly that).
- **Protocol 2.** Each reply carries only tokens new since the last one plus
  `count`, the running total, so the client can detect a missed reply. The
  cumulative reply it replaces re-serialised the whole stream every chunk
  (~30k tokens by the third hour). The client accepts either protocol —
  `count` absent means cumulative — so a mismatched deploy degrades to
  "works", not "silent". `capture_fragments.py` and `drive_unified_server.py`
  handle both too.
- **Exit for restart.** `ParakeetASRBuffer.alive()` reports whether the
  backend is gone for good. The coordinator's audio callback sets `_fatal`
  when it is; `run()` takes the Ctrl+C drain path (queued audio still plays)
  and then raises `SystemExit` → exit 1 → `Restart=always` brings the service
  back in ~45 s. A transient decode error, or a quiet room, never reaches
  this: the backend is still alive.

### Sentence buffer (`sentence_buffer.py`)

- `_internal_split` returns the **first** valid mark, not the last.
- `tick()` also emits a complete sentence sitting at the head of the buffer,
  so a fragment carrying two marks drains one sentence per chunk instead of
  holding the second until a timeout releases it glued to whatever followed.
- A remainder never gets a fresh clock: from `feed()` it takes the fragment's
  arrival time (as before); from `tick()` it keeps the most recent arrival.
  Cases 11–13 in `tests/test_sentence_buffer.py` pin this down.
- `tick()` now records `last_reason` for its cap/timeout emits.

### Scheduler (`scripts/ops/translate-window-check.sh`)

- `worker_pids()` returns comm-validated PIDs; `worker_running()` and the
  force-stop both use it. The unused `hm10` is gone.

### Retired code

- `src/pipeline/asr.py` (720 lines) and `asr_process.py` (230) deleted; the
  Whisper batch branches in the coordinator with them (94 lines). `run.py`
  still accepts `--parakeet` and ignores it. `scripts/setup.py`'s run entry
  had been passing arguments the constructor did not accept for some time;
  it now matches.
- The one thing translation kept using from `asr.py` — the repetition-loop
  filter — is `src/pipeline/text_filters.py::looks_like_loop`, moved
  verbatim.
- `MultiTargetTranslator`, `MultiLanguageTTS` (unused) deleted.
  `faster-whisper` and `ctranslate2` dropped from `requirements.txt` and from
  the list `install.sh` writes; the installer no longer downloads Whisper.
- The Opus-MT branch in `translation.py` **stays**: it is a working lighter
  backend (no `NLLB_MODEL`), not dead code.

### Web (`auth.py`, `relay.py`, `config_api.py`, `translate-web.service`)

- The trusted-proxy list comes from `TRANSLATOR_TRUSTED_PROXIES`; the code
  default is loopback only, and `systemd/translate-web.service` sets the
  Caddy address. **Deploying the code without the unit means every visitor
  behind Caddy shares one lockout counter** (login only; the page is
  unaffected) — a startup warning says so, and §5 has the step.
- `Sessions.record_failure` forgets an address 24 h after its last failure;
  escalation still holds across any lockout shorter than that.
- `write_audio` edits values in place so comments between entries survive
  (the production file keeps its routing notes there). The patched text must
  load back to exactly the intended document or it falls back to a clean
  dump. `config_api_smoke.py` covers it.
- The relay's drop-policy comment now says what the code does (oldest item,
  regardless of kind).

### Launcher and environment

- `scripts/run_production.sh` defaults are the production sentence-buffer
  values (30 / 15 / 60 words / punctuation boundary); until now they were the
  old policy and only the launcher's overrides made the service behave. Its
  header describes the current stack. `--parakeet` no longer passed.
- The launcher `start-translate-unified`, and the `translate`,
  `translate-window` and `gpu-thermal-guard` units, are **now in the repo**
  (`scripts/ops/`, `systemd/`) — they had existed only on the host, like the
  scheduler did this morning.
- `requirements-nemo.txt`: `pip freeze` of the production NeMo venv (Python
  3.11.15, nemo-toolkit 3.0.0, torch 2.11.0+cu128).

### Heartbeat and the stall watchdog (`coordinator.py`, `translate-window-check.sh`)

- `_on_audio_chunk_streaming` logs `HEARTBEAT | chunks=… | fragments=… |
  asr_alive=… | queues=[…]` every `PIPELINE_HEARTBEAT_SEC` (60 s). It runs on
  every chunk whether or not anyone is speaking, and stops only if the audio
  thread is stuck — so the log's mtime now tracks the process, not the room.
- `STALL_SECONDS` 900 → 300: five missed heartbeats is a hang; before, 180 s
  restart-looped on pre-service silence (morning) and 900 s still restarted a
  healthy service fifteen minutes after the evening's preaching ended.

### ASR module logging (`parakeet_asr.py`)

- Moved from stdlib `logging` (never routed anywhere, so its INFO lines were
  invisible and its ERROR lines reached stderr bare) to loguru like the rest.
- `start()` logs `unified ASR server ready: protocol 2 (…)` — the deploy check.

### Service stop (`systemd/translate.service`)

- Found by the post-deploy dry run: `systemctl stop` sent SIGTERM, which
  Python treats as an immediate exit, so the pipeline never took its drain
  path — the last sentence was cut mid-word at every window end and no
  `SESSION_END` was logged. Now `KillSignal=SIGINT` (the Ctrl+C path: input
  closes, final flush, queued audio plays, stats logged), `KillMode=mixed`
  (the NeMo child outlives the main process just long enough to hand over
  the last tokens), `TimeoutStopSec=45`.

### Certificate renewal

- `systemd/translate-cert-renew.timer`: `OnCalendar=*-*-* 03:20:00` (was
  03:20 and 15:20) with `RandomizedDelaySec=20m` and **no** `Persistent=true`
  — a run missed while the machine was off must not fire at the next boot,
  which could be Sunday morning. Six-day certificates renewed at three days
  left make a missed night harmless. The service unit and both scripts are in
  the repo (`scripts/ops/translate-cert-*.sh`).

### Tests

- `tests/test_pipeline_config.py` — for every language in `settings.yaml`
  (enabled) and `site.json` (audio), constructs a `TranslationService` and a
  `TTSService` under `run_production.sh`'s exports without loading a model;
  checks the coordinator accepts them; validates routing and schedule. This
  is the check that would have caught the Russian crash-loop in a second.
- `tests/test_asr_client.py` — fake ASR servers, no NeMo: both protocols
  decode identically, empty replies (silence) are not errors, a stalling
  server is detected in the deadline and reported dead, a crashing one too.
- `test_sentence_buffer.py` cases 11–13; `config_api_smoke.py` audio
  round-trip; `web_smoke.py` updated for the status event the server has sent
  since 2026-09-04 (it had been failing at HEAD for that reason).
- Committed experiment scripts no longer hard-code session scratch paths.

### Documentation

`README.md` rewritten for the stack as it is (it described Whisper as the
default); `docs/SETUP.md` and `docs/DEPLOYMENT.md` corrected where they said
the same.

---

## 3. Explaining B2 (the question asked)

The growth was **not** on the web server. The web server keeps a bounded
ring of recent events for late joiners (200 entries) and drops per-client
when a phone falls behind.

It was the **NeMo subprocess**: it kept every token of the service in a list
and JSON-encoded the whole list in reply to every 1.5 s chunk, while the
client kept only the new tail. Ninety minutes in that is ~15k tokens per
reply, forty times a minute; three hours, ~30k. Sending only the new tokens
removes the growth entirely, which is cleaner than dropping after some
minutes: nothing old is ever needed again once it has been sent. (NeMo's own
decoder state still holds the hypothesis for the stream — that is inside the
library and it is small.)

---

## 4. Validation

Everything below ran on the development laptop (ROCm) against this tree.

| Check | Result |
|-------|--------|
| `test_sentence_buffer.py` (13 cases) | pass |
| `test_asr_client.py` (fake servers: protocols, silence, stall, crash) | pass; stall detected in 2 s at a 2 s deadline |
| `test_pipeline_config.py` | pass — es → Kokoro, ht → MMS, ru → MMS, coordinator, routing, schedule |
| `config_api_smoke.py`, `admin_smoke.py`, `web_smoke.py`, `relay_smoke.py`, `tls_smoke.py` | pass |
| Protocol 2 vs the earlier protocol-1 capture, 90 s sermon clip, real NeMo server, biasing on | **identical** fragment sequence and transcript (82 fragments) |
| End-to-end `run.py --input-file` (90 s, Spanish, file input) | 55 chunks, 56 fragments, 17 sentences, 0 drops, clean drain, exit 0 |
| Fatal path: `kill -9` the NeMo child mid-stream | `FATAL` logged, drain, `SESSION_END`, exit 1 seven seconds after the kill |

And on the **production host** itself (RTX 3060, CUDA), after the evening
service, from a copy of this tree at `~/translator-review` sharing the venvs
and models by symlink — the production checkout untouched:

| Check | Result |
|-------|--------|
| `test_pipeline_config.py`, `test_asr_client.py`, `test_sentence_buffer.py` on the host, against its real `settings.yaml` | pass — es → Kokoro, ht → MMS, ru → MMS |
| 90 s clip, paced, three languages to the real outputs (Behringer L/R, onboard jack) | 56 fragments, 17 sentences, 17 × ES/HT/RU, 2 heartbeats, 0 errors, exit 0 |
| `kill -9` the NeMo child mid-stream | `FATAL`, drain, exit 1 six seconds later; GPU fully released |
| 10-min sermon, unpaced | 386 fragments → 100 sentences in 95 s, `protocol 2` announced, GPU peak 8.2 GB (= production), ASR 86 ms/chunk, 0 errors |
| its transcript vs the reference | **WER 2.64 %** — identical to the laptop capture's figure; 99.0 % end with a mark, 4.0 % two-sentence |
| **after deployment**, `systemctl --user start translate.service` on the real unit | heartbeat at 24 s, `protocol 2`, biasing on, env correct, 0 errors; stop clean |

Not exercised: the scheduler's force-stop against a real backlog worker
(verified with a decoy process only), the cert timer (unit file change
only), and the heartbeat against the *deployed* watchdog (the heartbeat lines
were seen in every host run; the 300 s threshold ships with the scheduler).

---

## 5. Deployment checklist — done 2026-09-07 (kept as the procedure)

The production checkout is at `cff8b00` (2026-09-02); everything since was
copied in by hand, so it shows 8 modified tracked files and 21 untracked
(`src/web/`, `config/`, …). A `git pull` would refuse to overwrite the
untracked ones. Checked by checksum on 2026-09-06: **no file on the host is
ahead of the repo** — the three that differ (`coordinator.py`,
`unified_asr_server.py`, `drive_unified_server.py`) are older copies — so a
reset to the pushed branch overwrites nothing of value. It leaves untracked
files alone: `settings.yaml`, the `.bak` copies, `staging-models/`, the stale
`src/pipeline/bus.py`.

The review commits are local to the laptop — push them first.

```bash
# laptop
git push origin master

# host
ssh administrator@10.1.170.184
cd ~/translator
git fetch origin
git reset --hard origin/master
git log -1 --oneline          # the last review commit (660ffb5 or later)

# the test copy used on 2026-09-06 evening; nothing depends on it
rm -rf ~/translator-review ~/review-audio

# 1. scheduler + launcher (+ guard, unchanged today) into ~/bin — the units point there
cp scripts/ops/translate-window-check.sh scripts/ops/start-translate-unified scripts/ops/gpu-thermal-guard.sh ~/bin/
diff ~/bin/translate-window-check.sh scripts/ops/translate-window-check.sh && echo same
#    (this also takes STALL_SECONDS from 900 to 300 — safe only together with
#     the pipeline code above, which writes the heartbeat it now expects)

# 2. web unit gains TRANSLATOR_TRUSTED_PROXIES; restart is a ~2 s page reconnect
cp systemd/translate-web.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart translate-web.service
journalctl --user -u translate-web -n 5 | grep -c "TRUSTED_PROXIES is not set"   # must print 0

# 3. cert timer: daily at 03:20 only (system unit)
sudo cp systemd/translate-cert-renew.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart translate-cert-renew.timer
systemctl list-timers translate-cert-renew.timer      # next run 03:20–03:40

# 4. the pre-service check, now that ru is in settings.yaml here
./venv/bin/python tests/test_pipeline_config.py

# 5. a dry run of the pipeline on the host (mute the outputs, or run before anyone is in the room)
~/bin/start-translate-unified & sleep 120; kill %1
grep -a "protocol 2\|HEARTBEAT" ~/translate.log | tail -3     # both must appear
```

`translate.service` itself is not running on a Tuesday; the next scheduled
start picks the new code up. The admin panel's Start does the same.

**Rollback:** `git checkout <previous commit>` in `~/translator` and copy the
old scheduler back from `~/bin/translate-window-check.sh.bak-0906`. The
protocol change needs no coordination: either side works with the other.

---

## 6. Carried forward

1. Russian short phrases still unreviewed by a Russian speaker (user is
   taking this over).
2. Thermal guard to be retired when the fan part arrives (B1 becomes moot).
3. `drain_min` 30 vs a measured 40-minute worst-case archive job — the fix
   belongs in the worker (`~/Multi-Bitrate-Sermons`).
4. One-script install for other churches: the NeMo venv, the units and the
   launcher are now all in the repo, which is the material it needs.
5. `scripts/download_models.py` no longer lists Whisper, but it still fetches
   only the Opus-MT / Piper fallbacks; the production models (NLLB, Kokoro,
   MMS, Parakeet) download themselves on first start. Part of item 4.
6. **Names that are also words get translated.** Seen on the host run:
   "Thank you, Brother Oar" → Russian "Спасибо, брат Весло" (a rowing oar);
   Spanish presumably "Hermano Remo". Context biasing fixed the *recognition*
   ("Orr" → "Oar") and NLLB then translated the word. The fix is a
   name-protection list — the short-phrase dictionary already sits in front
   of NLLB and is the natural place — reviewed by the speakers.
7. A "hard numbers after a service" habit: `tests/service_tally.py` over
   `~/translate.log` on the host.
