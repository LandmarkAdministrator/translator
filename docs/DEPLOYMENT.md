# Deployment

How a serving machine is built, updated, backed up and diagnosed. The
reference installation is the production host described at the end; the
same two scripts build any other.

- [What a serving machine consists of](#what-a-serving-machine-consists-of)
- [Fresh machine](#fresh-machine)
- [Updating a running site](#updating-a-running-site)
- [Rollback](#rollback)
- [Backup and recovery](#backup-and-recovery)
- [Diagnosis](#diagnosis)
- [Sharing the GPU](#sharing-the-gpu)
- [The reference host](#the-reference-host)

---

## What a serving machine consists of

Two layers, two scripts:

| Layer | Script | Puts in place |
|---|---|---|
| Operating system | `./install.sh --cuda` or `--rocm` | Debian repositories, GPU driver (the NVIDIA procedure in `docs/SETUP.md`), system packages, the main venv (`venv/`, Python 3.13, torch 2.11), base models; calls `install_site.sh` at the end |
| Site | `scripts/install_site.sh` | the NeMo venv for the streaming ASR (`~/nemo-venv`, Python 3.11 via uv, `requirements-nemo.txt`), a default `config/settings.yaml`, the admin password (`~/.config/translator/admin.json`, scrypt), the scheduler / launcher / thermal guard in `~/bin`, every systemd unit, linger, optionally TLS from an internal CA and a full model prefetch, then a verification |

`install_site.sh` is idempotent — run it after every `git pull` and it changes
only what differs, reporting each step as ok / changed / skipped / FAILED.
`--check` reports without changing anything.

The units it installs (sources in `systemd/`, scripts in `scripts/ops/`):

| Unit | Role |
|---|---|
| `translate.service` | the pipeline; started and stopped by the scheduler inside service windows, never enabled at boot; `Restart=always` brings it back if its ASR backend dies |
| `translate-web.service` | the page, WebSocket stream and `/admin`; runs all the time |
| `translate-window.timer` | every five minutes and a minute after boot: `~/bin/translate-window-check.sh` opens and closes the windows in `config/schedule.conf`, restarts a hung pipeline, and manages the archive worker (see [Sharing the GPU](#sharing-the-gpu)) |
| `translate-tally.timer` | 23:30 nightly: `tests/service_tally.py` writes the day's numbers for the admin panel |
| `gpu-thermal-guard.service` | warns at 75 °C, stops translation at 85 °C (`--no-thermal-guard` to omit) |
| `translate-cert-renew.timer` | system unit, 03:20 daily, only with `--tls` |

Site-specific files, all editable afterwards from the admin panel or by hand:

| File | Holds | In git? |
|---|---|---|
| `config/site.json` | church name, service times, languages offered on the page and every string the page shows | yes |
| `config/schedule.conf` | service windows, drain lead time | yes |
| `config/bias_phrases.txt` | phrases boosted inside the ASR decoder (Bible books, local names) | yes |
| `config/settings.yaml` | audio input, per-language output device and channel | **no** (machine-specific) |
| `~/.config/translator/admin.json` | admin password hash | no |
| `/etc/lego/` | ACME settings, CA certificate, issued certificate | no |

---

## Fresh machine

Debian 13, a supported GPU, the audio interface plugged in, the user in
`sudo`. Secure Boot off for NVIDIA (the DKMS module is unsigned).

```bash
cd ~
git clone https://github.com/LandmarkAdministrator/translator.git translator
cd translator
./install.sh --cuda            # or --rocm; NVIDIA needs one reboot in the middle — re-run afterwards
./scripts/install_site.sh --web-host 0.0.0.0        # LAN page; add --trusted-proxies <ip> behind a reverse proxy
./scripts/install_site.sh --prefetch                # every model now (~8 GB), so the first service never downloads
scripts/gpu_doctor.sh                               # must end with "GPU stack is healthy."
```

Then:

1. Sign in to `http://<host>:8080/admin`, pick the audio input and each
   language's output, save; set the service windows.
2. Edit `config/site.json`: church name, service times, the languages and
   their page wording.
3. `./venv/bin/python tests/test_pipeline_config.py` — the pre-service check.
4. A dry run outside a window: `~/bin/start-translate-unified & sleep 120; kill %1`,
   then `grep -a "protocol 2\|HEARTBEAT" ~/translate.log` must show both.
5. Public access, if wanted: a reverse proxy on another machine forwards to
   port 8080 (WebSocket upgrades pass through a plain `reverse_proxy`), or
   `install_site.sh --tls …` for a certificate from an internal CA.

Things the fresh-machine path has **not** yet been through: it has only run
on the two existing machines (item 3 in `TODO.md`); on ROCm hosts the NeMo
venv is still made by hand (`docs/SETUP.md`).

---

## Updating a running site

The production checkout is a plain clone tracking `origin/master`. Outside a
service window:

```bash
cd ~/translator
git pull --ff-only
./scripts/install_site.sh --yes          # copies changed scripts/units, restarts translate-web only if its unit changed
./venv/bin/python tests/test_pipeline_config.py
```

What needs what:

| Changed | Takes effect |
|---|---|
| the page (`src/web/static/`), `config/site.json` | on the next request — nothing to restart |
| `src/web/*.py` (server, admin) | `systemctl --user restart translate-web.service` (a two-second page reconnect) |
| pipeline code, `scripts/run_production.sh` | at the next scheduled start; an admin-panel Start does the same |
| `scripts/ops/*`, `systemd/*` | `install_site.sh` copies them and reloads; the scheduler picks its new copy up on its next tick |
| `requirements*.txt` | `./install.sh` (main venv) or delete and recreate `~/nemo-venv` via `install_site.sh` |

`translate.service` is never restarted by an update; the scheduler also
restores its `ExecStart` every five minutes, so a different program cannot be
started there by accident.

---

## Rollback

```bash
cd ~/translator
git log --oneline -10                     # find the last good commit
git checkout <commit>
./scripts/install_site.sh --yes           # puts that revision's scripts and units back
```

`git checkout master && git pull --ff-only` returns to the tip. The ASR
protocol change of 2026-09-06 needs no coordination — either side of the
pipe works with the other.

---

## Backup and recovery

Everything that is code or shared configuration is in git. What is not:

| What | Where | Recovery |
|---|---|---|
| audio device settings | `config/settings.yaml` | re-pick in `/admin` (two minutes), or restore the file |
| admin password | `~/.config/translator/admin.json` | `./venv/bin/python scripts/set_admin_password.py` |
| TLS | `/etc/lego/` | `install_site.sh --tls …` re-issues |
| models (~8 GB) | `~/.cache/huggingface`, `models/` | `install_site.sh --prefetch` re-downloads |
| logs and tallies | `~/translate.log`, `~/translator/logs/`, `~/sermons/logs/` | not needed to run |

A copy of the three small files is enough to rebuild a machine from git in
under an hour plus download time. The sermon archive on the same host has its
own backups (a Synology NAS and two USB drives, weekly on Friday 03:00) and is
not part of this project.

Full recovery on new hardware is the [Fresh machine](#fresh-machine) procedure
with those files restored before the first `install_site.sh`.

---

## Diagnosis

| Question | Where to look |
|---|---|
| Is the GPU stack sound? | `scripts/gpu_doctor.sh` — one check per known failure, each with its fix |
| Did the service run, and how well? | `/admin` → the nightly tally, or `ssh host 'python3 -' < tests/service_tally.py` for today |
| Why did the scheduler do that? | `~/sermons/logs/translate-window.log` (every decision), `schedule.log` (launch failures) |
| Is the pipeline alive? | `~/translate.log`: a `HEARTBEAT` line every 60 s, `ERROR` lines for real failures; `journalctl --user -u translate.service` |
| The page? | `journalctl --user -u translate-web.service`; `tests/web_smoke.py` against a running server |
| Before a service | `./venv/bin/python tests/test_pipeline_config.py` |

Silence is not a fault: the scheduler judges liveness by the heartbeat, not by
recognized speech, so an empty room does not trigger a restart.

---

## Sharing the GPU

The reference host also runs the sermon-archive backlog worker between
services. `docs/BACKLOG-CONTRACT.md` records how the scheduler drains it
before every window, force-stops it at window open, relaunches it afterwards,
and honours the thermal guard's hold. On a host without
`~/Multi-Bitrate-Sermons` the scheduler skips all of that.

---

## The reference host

The production machine ("Translate", a mini PC in the sound room): Ryzen AI 9
HX 370 with an RTX 3060 12 GB over OCuLink, Debian 13, kernel 7.1.8 from
backports with headers, Secure Boot off, NVIDIA driver 610.57.04 as the open
DKMS module from NVIDIA's Debian 13 repository, torch 2.11.0+cu128 in both
venvs, NeMo 3.0.0 in `~/nemo-venv`. Audio: a Behringer USB interface (Spanish
left, Creole right) and the onboard jack for Russian. The page is reached
through a reverse proxy on another VLAN; `TRANSLATOR_TRUSTED_PROXIES` in the
web unit names it so the login lockout counts real client addresses.

The development laptop (Radeon 890M, ROCm 7.2.2, gfx1150 with
`HSA_OVERRIDE_GFX_VERSION=11.0.0`, Secure Boot on — harmless for the in-tree
driver) runs the same stack a little slower; it is the ROCm reference.

The onnx-asr Parakeet TDT model (`./install.sh --parakeet`,
`scripts/install_parakeet.sh`) remains as the ASR fallback for a machine
without the NeMo venv; on ROCm 7.2 its runtime silently uses the CPU (the
published wheel links the ROCm 6 ABI), which is fast enough for streaming
but is not the production path.
