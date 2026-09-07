#!/usr/bin/env bash
# Translate-PC scheduler. One box, two jobs, never at the same time:
#   * inside a service window  -> live translation (new stack, unified streaming)
#   * outside a service window -> sermon-archive backlog worker
#
# The GPU cannot hold both (live ~7.4 GiB, backlog ~10 GiB), and a backlog job
# running during a service measurably degrades live translation, so the backlog
# is drained BEFORE a window opens rather than being killed as it opens —
# live translation must never start late.
#
# Service windows:
#   Sun 09:15-12:45, Sun 18:15-21:00, Wed 17:45-21:15
# Backlog stops DRAIN_MIN minutes before each window and resumes after it.
#
# Also: restarts live translation if it hangs (log silent) or loses its audio
# output. Manual override: `touch ~/translate-manual.flag` stops this script
# doing anything at all (for testing); remove it to resume.
set -u
LOG="$HOME/sermons/logs/translate-window.log"
UNIT="$HOME/.config/systemd/user/translate.service"
WANT_EXEC="$HOME/bin/start-translate-unified"

# Which program runs is decided by one line in one file, and nothing else
# guards it. On 2026-09-06 that line was changed back to the legacy
# translate.py between services; the unified stack's fixes silently did not
# run and the congregation page showed "no service in progress" while audio
# played. The legacy program was retired that day, so there is no longer a
# supported reason for this to differ -- if it does, something is wrong and
# the log should say so.
ensure_program() {
  cur=$(systemctl --user show translate.service -p ExecStart --value 2>/dev/null \
        | grep -o 'path=[^ ;]*' | head -1 | cut -d= -f2)
  [ "$cur" = "$WANT_EXEC" ] && return 0
  log "WRONG PROGRAM: ExecStart=$cur -> restoring $WANT_EXEC"
  sed -i "s|^ExecStart=.*|ExecStart=%h/bin/start-translate-unified|" "$UNIT"
  systemctl --user daemon-reload
  systemctl --user is-active --quiet translate.service && {
    systemctl --user restart translate.service
    log "restarted onto the correct program"
  }
}
# Seconds since translate.service became active; 0 when inactive. Used so the
# hang watchdog cannot kill a service that is still loading its models.
svc_uptime() {
  local ts
  ts=$(systemctl --user show translate.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
  [ -z "$ts" ] || [ "$ts" = "0" ] && { echo 0; return; }
  local now; now=$(awk '{printf "%d", $1*1000000}' /proc/uptime)
  echo $(( (now - ts) / 1000000 ))
}

TRANSLATE_LOG="$HOME/translate.log"
WORKER_START="$HOME/Multi-Bitrate-Sermons/scripts/lbc-start-unified-worker.sh"
STOP_FLAG="$HOME/Multi-Bitrate-Sermons/stop.flag"
WORKER_PAT="Multi-Bitrate-Sermons/.venv/bin/python.*worker"
STALL_SECONDS=300          # live log silent this long in-window = hung.
                           # The pipeline writes a HEARTBEAT line every 60 s
                           # from its audio path whether or not anyone is
                           # speaking, so this measures the process, not the
                           # room: five missed heartbeats is a hang. Before
                           # the heartbeat, silence and a hang were the same
                           # thing to this check — 180 s restart-looped on
                           # pre-service silence and even 900 s restarted a
                           # healthy service after the evening service ended
                           # (2026-09-06, 09:20 and 20:30).
MIN_UPTIME=300             # never restart a service still loading models
PLAYBACK_ERR_WINDOW=120
PLAYBACK_ERR_MIN=2
DRAIN_MIN=30               # default; overridden by config/schedule.conf
SCHEDULE_CONF="$HOME/translator/config/schedule.conf"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }


ensure_program   # must run after log() exists

[ -e "$HOME/translate-manual.flag" ] && exit 0

dow=$(date +%u)

# minutes-of-day helpers so DRAIN_MIN arithmetic is simple
now_min=$(( 10#$(date +%H) * 60 + 10#$(date +%M) ))
in_live=0; in_drain=0
consider() {  # $1 day, $2 start_min, $3 end_min
  [ "$dow" -ne "$1" ] && return
  [ "$now_min" -ge "$2" ] && [ "$now_min" -lt "$3" ] && in_live=1
  [ "$now_min" -ge $(( $2 - DRAIN_MIN )) ] && [ "$now_min" -lt "$3" ] && in_drain=1
}
# Windows come from config/schedule.conf so they can be edited from /admin
# without touching this script. If the file is missing or unreadable the
# built-in schedule below applies: losing the file must not mean losing every
# service until someone notices.
hm2min() { echo $(( 10#${1%%:*} * 60 + 10#${1##*:} )); }
loaded_windows=0
if [ -r "$SCHEDULE_CONF" ]; then
  # drain_min first: consider() uses it, so reading it in the same pass would
  # apply a stale value to any window listed above it in the file.
  while read -r kind a _rest; do
    [ "$kind" = "drain_min" ] || continue
    case "$a" in ''|*[!0-9]*) ;; *) DRAIN_MIN="$a" ;; esac
  done < "$SCHEDULE_CONF"
  while read -r kind a b c _rest; do
    case "$kind" in
      window)
        case "$a" in [1-7]) ;; *) continue ;; esac
        case "$b" in [0-9][0-9]:[0-9][0-9]) ;; *) continue ;; esac
        case "$c" in [0-9][0-9]:[0-9][0-9]) ;; *) continue ;; esac
        consider "$a" "$(hm2min "$b")" "$(hm2min "$c")"
        loaded_windows=$(( loaded_windows + 1 ))
        ;;
    esac
  done < "$SCHEDULE_CONF"
fi
if [ "$loaded_windows" -eq 0 ]; then
  log "schedule.conf missing or empty -> using built-in windows"
  consider 7 $((  9*60+15 )) $(( 12*60+45 ))
  consider 7 $(( 18*60+15 )) $(( 21*60+ 0 ))
  consider 3 $(( 17*60+45 )) $(( 21*60+15 ))
fi

# Only count real python worker processes: a plain `pgrep -f` also matches any
# shell whose command line merely mentions the pattern (an admin ssh command,
# this script's own invocation), which would make the backlog look "running"
# forever and never start.
worker_pids() {
  local pid comm
  for pid in $(pgrep -f "$WORKER_PAT" 2>/dev/null); do
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    case "$comm" in python*|Python*) echo "$pid" ;; esac
  done
}
worker_running() { [ -n "$(worker_pids)" ]; }

# ---------- backlog side ----------
if [ "$in_drain" -eq 1 ]; then
  # Approaching or inside a window: no archive work.
  [ -e "$STOP_FLAG" ] || { touch "$STOP_FLAG"; log "draining backlog before service window"; }
  if [ "$in_live" -eq 1 ] && worker_running; then
    # Window is open and it still hasn't exited — live translation wins.
    # Kill only the PIDs worker_pids() validated by comm. A bare `pkill -f`
    # here would match any process whose ARGUMENTS mention the pattern — an
    # admin's ssh command, a grep — exactly the trap the comment above
    # warns about, and one that killed live sessions on 2026-09-06.
    for pid in $(worker_pids); do kill "$pid" 2>/dev/null; done
    log "backlog still running at window open -> force-stopped (stale claim reaper will recover the row)"
  fi
else
  # Free time: let the archive work.
  [ -e "$STOP_FLAG" ] && { rm -f "$STOP_FLAG"; log "outside service windows -> backlog allowed"; }
  if ! worker_running; then
    # systemd-run: the worker gets its OWN transient unit.  A plain setsid/
    # nohup child lives in THIS oneshot service's cgroup and is killed the
    # moment the check script exits (bug found 2026-09-06: worker died at
    # birth on every timer tick since 2026-09-04).
    if ! mountpoint -q /mnt/synology/web; then
      log "NAS not mounted — backlog start skipped"
    elif systemd-run --user --collect --unit=lbc-backlog-worker \
        -p WorkingDirectory="$HOME/Multi-Bitrate-Sermons" \
        -p StandardOutput=append:"$HOME/sermons/logs/unified-worker-nohup-$(hostname).log" \
        -p StandardError=append:"$HOME/sermons/logs/unified-worker-nohup-$(hostname).log" \
        "$HOME/Multi-Bitrate-Sermons/.venv/bin/python" -u scripts/unified_worker.py 2>>"$HOME/sermons/logs/schedule.log"; then
      log "started sermon-archive backlog worker (unit lbc-backlog-worker)"
    else
      log "backlog worker launch FAILED (systemd-run) — see schedule.log"
    fi
  fi
fi

# ---------- live translation side ----------
active=0
systemctl --user is-active --quiet translate.service && active=1

recent_errors() {
  local cutoff
  cutoff=$(date -d "-${PLAYBACK_ERR_WINDOW} seconds" '+%Y-%m-%d %H:%M:%S')
  tail -n 400 "$TRANSLATE_LOG" 2>/dev/null \
    | grep -E "Playback error|\| ERROR " \
    | awk -v c="$cutoff" '($1 " " $2) >= c' \
    | wc -l
}

if [ "$in_live" -eq 1 ]; then
  if [ "$active" -eq 0 ]; then
    systemctl --user start translate.service
    log "in window, was not running -> started"
  elif [ -f "$TRANSLATE_LOG" ] && [ -z "$(find "$TRANSLATE_LOG" -newermt "-$STALL_SECONDS seconds")" ] \
       && [ "$(svc_uptime)" -ge "$MIN_UPTIME" ]; then
    systemctl --user restart translate.service
    log "in window, log stalled ${STALL_SECONDS}s -> restarted (hang watchdog)"
  else
    n=$(recent_errors)
    if [ "${n:-0}" -ge "$PLAYBACK_ERR_MIN" ]; then
      systemctl --user restart translate.service
      log "in window, $n errors in last ${PLAYBACK_ERR_WINDOW}s -> restarted (dead output watchdog)"
    fi
  fi
else
  if [ "$active" -eq 1 ]; then
    systemctl --user stop translate.service
    log "outside window -> stopped"
  fi
fi
