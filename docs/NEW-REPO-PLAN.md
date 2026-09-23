# One project for both systems — the plan

**Written 2026-09-23.** Supersedes `docs/MERGE-PLAN.md`, which assumed the
sermon archive would move *into* this repository. It will not: both move into a
**new repository**, stripped of everything that is ours alone, with a setup
flow that stands the whole thing up for a church that is not us. This document
is the plan for that, and for the piece that comes first — replacing Google
Sheets.

---

## 0. What exists today

**`~/Projects/translator`** (public on GitHub) — the live service. Parakeet
streaming ASR, NLLB-1.3B, Kokoro / MMS voices, three languages, the phone page
and its `/admin`, a scheduler that runs the service windows unattended. About
130 code files. Runs on the Translate PC; develops on the laptop.

**`~/Projects/Multi-Bitrate-Sermons`** (private on GitHub) — the post-service
pipeline. Ingest, Whisper ASR, translation, dubbing, multi-bitrate encode, HLS
+ subtitles, `index.json`, a WordPress plugin that renders the library, Meili
search, backups. About 130 code files and 3,772 published sermons.

They already share: the GPU on one machine (`docs/BACKLOG-CONTRACT.md`), the
same translation models, the same glossary problems, the same NAS, the same
two operators. They do not share a line of code.

---

## 1. Decisions

My recommendation first, reasoning after. These are the things I want confirmed
before Phase 2; **D3 is needed now**, because it is the piece you asked to
build first.

| | Decision | Recommendation |
|---|---|---|
| D1 | New repo history | **Clean initial commit. No history import.** |
| D2 | New repo visibility | **Private until the scrub is verified, then public.** |
| D3 | Sheets replacement | **SQLite owned by one small HTTP service, LAN clients.** |
| D4 | ASR engines | **One engine for speech (Parakeet), Whisper kept for singing.** |
| D5 | WordPress plugin | **Keep it in the repo, renamed generic.** |
| D6 | Layout | **One repo, one installable package, two entry points.** |

**D1 — clean history.** `translator` is public and `Multi-Bitrate-Sermons` is
private. Any import of the private history into a public repo publishes it
permanently, and that history contains NAS paths, host addresses, the sheet ID,
member names in speaker tables, and at least one `.env.example` that has held
real keys. A scrub of the *worktree* is easy; a scrub of 5 years of history is
not, and one missed blob is not recallable. So: the new repo starts at commit
one, with both old repositories kept private, read-only, as the historical
record. We lose `git blame` across the seam; we keep the ability to ever make
this public.

**D2 — private first.** The scrub is verifiable (Phase 1 produces a check
script that fails the build on a match). Flip to public when that script passes
on a clean clone and a fresh-machine install has gone end to end.

**D4, D5, D6** are argued in §6, §3 and §4.

---

## 2. Replacing Google Sheets — do this first

### 2.1 What the sheet actually does

Four different jobs are riding in one spreadsheet, and only one of them is a
spreadsheet job:

1. **Editorial record** — one row per sermon: date, service, prefix, speaker,
   series, title, scripture. Typed by a human, often from a phone.
2. **Derived names** — `upload_filename`, `thumbnail_filename`,
   `series_image_filename`, `speaker_filename`, built by sheet *formulas* from
   the editorial fields.
3. **Job ledger** — three stages each own a five-column block on `Main`:
   encode `I–M`, ASR `T–X`, dub `Y–AC` (status / worker token / started /
   heartbeat / finished). This is what `scripts/pipeline/claims.py` arbitrates.
4. **Dashboard** — what a human looks at to see what is stuck.

Job 1 wants a form. Job 2 wants code. Job 3 wants a transaction. Job 4 wants a
page. Sheets is a poor fit for three of the four, and the seams show: a 30-second
`CLAIM_SETTLE_SEC` read-after-write wait, heartbeat staleness parsed out of
status strings, legacy `failed: reason` attempt-counting rules, 26 call sites
holding Google credentials, and a 3,772-row sheet that is slow to open.

### 2.2 The design

**One service owns the data; everything else is a client.**

- **Store**: SQLite in WAL mode, on the machine that already runs the web
  service (the Translate PC). Not on the NAS — SMB file locking is exactly what
  `manifest.py` already works around with an `O_EXCL` sentinel, and SQLite over
  SMB is a corruption story waiting to happen.
- **Service**: a handful of JSON endpoints served by the *existing* hand-rolled
  asyncio HTTP server in `src/web/` (no Flask, no FastAPI, no new dependency),
  authenticated with the token machinery already in `src/web/auth.py`.
- **Clients**: every worker and script talks to it over the LAN through one
  small client module, with retry and backoff. Nothing else touches the DB file.

**Schema** (four tables):

```
sermons   slug PK, date, service, prefix, speaker, series, title,
          scripture, notes, source_path, created_at, updated_at
jobs      (slug, stage) PK, state, worker_token, host, attempts,
          started_at, heartbeat_at, finished_at, last_error
settings  key PK, value            -- church name, paths, stage tuning
events    id PK, at, actor, slug, field, old, new   -- audit trail
```

Derived filenames are *not* stored as formulas; they are computed by
`pipeline/naming.py`, which already exists, and written alongside the row so the
record of what a file was called survives a naming-rule change.

**API**:

```
GET  /api/sermons?since=        list / sync
GET  /api/sermons/<slug>
POST /api/sermons               upsert (editorial), writes an event row
POST /api/jobs/claim            {stage, token, host} -> next row or 204
POST /api/jobs/<slug>/<stage>/heartbeat
POST /api/jobs/<slug>/<stage>/finish   {ok|fail, error}
GET  /api/jobs?stage=&state=    dashboards, alert_check
```

`claim` is one SQL transaction: pick the oldest eligible row, mark it claimed,
stamp the token, return it. That single change deletes the 30-second settle
wait, the stale-heartbeat guesswork and the status-string attempt parsing —
`claims.py` (250 lines of careful hedging) becomes a client call, and the
fencing token is checked on every subsequent write, so a resurrected worker
cannot stomp a row that was re-claimed while it was away.

**The human side** — a small `/archive` console in the same web service, behind
the same password as `/admin`:

- this week's quick-add form (the thing that is actually typed on a phone),
- the sermon list with search and an edit form,
- a job board: what is running, what failed, a retry button,
- nightly CSV export to the NAS, so a human-readable copy always exists and
  disaster recovery never depends on the service being up.

### 2.3 What we give up, and the cover for it

| Lost with Sheets | Cover |
|---|---|
| Editing from the Google Sheets phone app | The `/archive` console is a web page; it works on a phone on the LAN or over the existing TLS gateway |
| Revision history / "who changed this" | The `events` table, which is better: it records the actor and both values |
| Formulas | `naming.py`, already the authority elsewhere |
| Google's hosting (nothing to run) | One more service on a box that is already always on; DB is one file, backed up nightly to the NAS |
| Offline-by-Google availability | Workers already block and wait when the NAS is absent; the client does the same when the store is absent — no job starts, so no job fails |

### 2.4 Migration

1. Build the store, service, client and console. Nothing deployed.
2. One-shot importer reads the live sheet and seeds SQLite; a diff tool proves
   the two agree row for row.
3. **Dual-read week**: workers read from the store, and a checker compares
   store and sheet every hour and reports any divergence. Sheet stays
   authoritative for editorial input.
4. Flip editorial input to the console. Sheet becomes read-only, kept.
5. Delete the Google credentials from the 26 call sites and drop
   `google-api-python-client`.

**Scope**: 16 files import `pipeline/sheet.py`; about ten write back. I will
keep `sheet.py`'s public shape (`load_snapshot`, `SheetSnapshot`, `SermonRow`)
and re-point it at the store, so most of those 16 change one import line and
nothing else. Roughly: store + service + client 2 days, port 1–2 days, console
1–2 days, then the dual-read week.

---

## 3. What is ours alone

Measured, not guessed — `git grep` for our name, initials, hosts, paths and
keys across both worktrees:

**`translator`: 45 hits.** Nearly clean already.

| Where | What it is | Becomes |
|---|---|---|
| `scripts/ops/*.sh`, `scripts/run_bench_safely.sh` | host addresses, user, NAS paths | `config/site.yaml` + install answers |
| `scripts/ops/translate-cert-*.sh` | our domain, internal CA | config keys; cert step optional |
| `config/site.json` | church name, page strings | `config/site.example.json` + wizard |
| `config/bias_phrases.txt` | member and place names | `config/bias_phrases.example.txt`, ours moves out of the repo |
| `config/schedule.conf` | our service times | wizard answers |
| `systemd/*.service`, `manifest.webmanifest` | our naming | templated at install |

**`Multi-Bitrate-Sermons`: 1,299 hits.** The bulk is one thing:

| Where | What it is | Becomes |
|---|---|---|
| `plugin/<our-slug>-sermons/**` (~600 hits) | plugin slug, PHP class prefix, CSS class prefix, option and table names, author | mechanical rename to a generic slug, plus a migration step for the live site's options/tables |
| `scripts/normalize_speakers*.py` (71) | our speakers' names and spellings | a data file, `config/speakers.yaml`, shipped empty |
| `scripts/caddy_*.py`, `setup_caddy_*.sh` (36) | our domains and download host | config keys |
| `scripts/lbc-usb-backup-weekly.sh` (18) | our drive labels and mount points | config keys |
| `*.md` (≈100) | our history, hosts, decisions | stays with the old repo; the new repo gets fresh docs |

Phase 1 ends with `scripts/check_no_local_data.sh`, which greps a clean clone
for all of the above and exits non-zero on a hit. That script is what makes D2
safe.

---

## 4. The new repository

One repo, one installable package, two entry points — not a monorepo of two
independent projects, because the whole reason to do this is that the shared
half stops being written twice.

```
<newrepo>/
  core/         models, ASR, translation, TTS, text filters, number guard,
                glossary, config loading, logging          [shared]
  store/        schema, service, client, migrations, console  [shared]
  live/         real-time: capture, streaming ASR, coordinator, scheduler
  archive/      post-service: ingest, stages, workers, manifest, publish
  web/          the asyncio server, auth, /admin, the live page, /archive
  plugin/       the WordPress plugin (generic slug)
  setup/        install.sh, doctor, first-run wizard, model prefetch
  config/       *.example.* only — no church data, ever
  docs/  tests/
```

What actually merges, rather than sits side by side:

- **translation** — one model loader, one glossary, one number guard, one
  name-protection list. Today the archive has `translate_glossary.py` with
  proven entries that the live path has never seen (TODO item 2).
- **ASR** — one engine wrapper with a streaming mode and a batch mode (§6).
- **TTS** — one voice registry; the archive's dubbing and the live path pick
  from the same list.
- **store** — both sides get a ledger; the live side's service tally and the
  archive's job board are the same table.
- **GPU arbitration** — `BACKLOG-CONTRACT.md` stops being a contract between
  two programs and becomes one scheduler.
- **install** — one script, one doctor, one wizard.

### 4.1 Setting it up at a church

`./setup.sh` — detects the GPU (CUDA / ROCm / CPU / Apple), asks what to
install (live only, archive only, both), then hands off to a first-run web
wizard for the things a non-programmer should answer: church name and branding,
service times, languages and voices, where storage lives, where the website is,
admin password. It writes `config/site.yaml`, fetches the models, installs the
systemd units, runs the doctor, and prints the URLs. Re-runnable; the wizard is
the same page as `/admin` afterwards.

This is also TODO item 3 ("installer on a fresh machine") — a clean Debian
install has still never gone end to end, and this plan is the reason to finally
do it.

---

## 5. Phases

Each phase ends with both stacks still running on the laptop; that is the gate.
Nothing deploys to the Translate PC until Phase 5, and never on a Saturday.

| | Phase | Contents | Rough |
|---|---|---|---|
| 0 | **Sheets replacement** | §2, built in this repo, flagged off, not deployed | ~1 week + dual-read week |
| 1 | **Scrub** | §3 inventory turned into config + `check_no_local_data.sh` | 2 days |
| 2 | **New repo, live side** | skeleton, `core/`, live stack runs from the new layout on the laptop | 2–3 days |
| 3 | **Archive side** | pipeline moves in, ports to the store client, Sheets code deleted | 3–4 days |
| 4 | **Setup** | wizard, doctor, docs, fresh-machine install end to end | 2–3 days |
| 5 | **Cutover** | PC runs the new layout; old repos go read-only | 1 day + one Sunday of watching |

Phase 0 happens in `translator` deliberately: it is useful on its own, it is
the riskiest piece (live data, 3,772 rows, three workers), and proving it before
the move means the move is only a move.

---

## 6. ASR: one engine or two

You asked whether the archive's Whisper is better than the live Parakeet. Our
own measurements, on our own material:

| Material | Parakeet (unified) | Whisper | Winner |
|---|---|---|---|
| Sermon, static file | **2.42 %** (2.21 % with phrase biasing) | 5.12 % batch | Parakeet, by half |
| Sermon, live | **2.69 %** | 3.34 % (the old live program) | Parakeet |
| Pristine oratory (JFK) | 1.47 % live / 0.59 % static | **0.59 % static** | Whisper, narrowly |
| Congregational singing | 45–49 % | **~27 %** | Whisper, clearly |

Licences are not the deciding factor — Whisper is MIT, Parakeet CC-BY-4.0, both
fine commercially. Accuracy by domain is.

**My opinion:** the belief that the archive's Whisper is better does not hold
for preaching — it is twice the error rate on exactly the material that matters
most, and the archive's subtitles are permanent in a way the live output is not.
Singing is the real exception, and it is a big one for a full-service recording.

So: **one engine wrapper, Parakeet for speech in both apps, Whisper kept as a
selectable engine for music-heavy items and as the fallback for languages
Parakeet does not cover.** A segment router (speech → Parakeet, music →
Whisper) is the eventual right answer and is a natural fit for the archive's
existing per-segment structure; it is not Phase 3 work.

**Do not switch the archive on my numbers alone.** Before Phase 3 flips the
default, run three complete services both ways and spot-check the singing
sections by hand. That is a half-day and it is cheap insurance against
regenerating 3,772 sermons' worth of regret.

---

## 7. Risks

- **A live service every week.** Phase 5 cutover lands on a Monday, with the
  old layout one `systemctl` away for the following Sunday.
- **3,772 sermons of state.** The dual-read week and the hourly divergence
  check exist for this; the sheet is kept read-only afterwards, not deleted.
- **The WordPress plugin is live on the public site.** The rename touches
  option and table names; it ships with a migration step and is tested on a
  staging site before the live one.
- **Public/private.** D1 and the Phase 1 check script are the whole mitigation.
  No public push before that script passes on a clean clone.
- **Two repos for a while.** Phases 0–1 keep both alive; that is deliberate, so
  a bad week never blocks a Sunday.
