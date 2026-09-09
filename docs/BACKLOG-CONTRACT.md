# Sharing the GPU with the sermon archive — the contract as implemented

The Translate PC does two jobs with one 12 GB card: live translation during
services (~7 GB) and, between them, the sermon-archive backlog worker from the
`Multi-Bitrate-Sermons` project (up to ~10 GB, Whisper large-v3). They must
never overlap, and the archive side sent a contract on 2026-09-08 saying how
its worker must be treated. This is how the live-translation side honours it.

**Who runs the schedule.** `translate-window.timer` → `~/bin/translate-window-check.sh`
(source: `scripts/ops/translate-window-check.sh`) every five minutes, and a
minute after boot. The admin panel does **not** replace it: the panel edits
`config/schedule.conf` (windows, drain lead time) and the script reads that
file. There is no plan to retire the script.

| Contract point | How it is met |
|---|---|
| Never run the worker during or near live translation | `drain_min` before every window the worker is asked to stop; inside a window a worker still alive is force-stopped |
| Drain, don't kill | `touch ~/Multi-Bitrate-Sermons/stop.flag` at window − `drain_min`; the worker finishes its job and exits |
| Drain at least 30 min, 45 safer | `drain_min 45` in `config/schedule.conf` since 2026-09-08 (jobs have run up to 45 min) |
| Force-kill acceptable at window open | `worker_pids()` — `pgrep -f` **plus** a `/proc/PID/comm` check, so an admin's shell that mentions the pattern is never killed — and `kill` on those PIDs only. The stale-claim reaper recovers the row |
| Relaunch outside windows in its own systemd unit, never as a scheduler child | `systemd-run --user --collect --unit=lbc-backlog-worker …` (the oneshot-cgroup bug of 2026-09-06 is what made this a rule) |
| Check the NAS first | `mountpoint -q /mnt/synology/web`, else "backlog start skipped" |
| Skip if already running; liveness without self-match | the fixed unit name refuses duplicates, and `worker_pids()` matches `Multi-Bitrate-Sermons/.venv/bin/python.*scripts/unified_worker.py` with the comm check |
| Leave the worker running whenever translation is not scheduled | the free-time branch starts it if it is not running, every tick |
| Boot recovery (weekly reboot Sun 04:00) | `OnStartupSec=60` on the timer, then every five minutes: the 04:05 tick relaunches the worker |
| Respect the thermal guard's hold | since 2026-09-08: if `stop.flag` **and** `~/.gpu-guard-stopped-backlog` both exist while the guard unit is running, the scheduler leaves the flag alone and does not relaunch; the guard lifts its own stop once the card is under 70 °C. (The guard was retired on the production host on 2026-09-09 when the card got a proper fan; the rule stays for hosts that run it) |
| A "hands off everything" switch | `~/translate-manual.flag` — unchanged. While it exists the script does nothing at all. The admin panel's Start and Stop set it; "Resume automatic schedule" clears it |
| Manual translation start must not collide with the worker | since 2026-09-08 the admin panel's Start sets `stop.flag` and stops the worker (unit first, then any comm-validated PID) before starting translation; the worker returns when the schedule is resumed |
| Greppable state-change log | `~/sermons/logs/translate-window.log` (every decision) and `schedule.log` (launch failures) — unchanged |
| If the old scheduler is ever retired: disable the timer, keep the backup | not planned; every version of the script is in git history (`scripts/ops/`), and the pre-git copies are in `~/archive/test-campaign-2026-09/bin/` |

Two things worth knowing on the archive side:

- On a host without `~/Multi-Bitrate-Sermons` the scheduler skips the worker
  logic entirely (other churches have no backlog).
- Where the guard runs, it also sets `~/translate-manual.flag` when it has to
  stop a *live* service at 85 °C; that hold is deliberate and needs a person.
  The production host no longer runs it (fan fitted, guard retired
  2026-09-09), so the card's own thermal protection is the only limit there.
