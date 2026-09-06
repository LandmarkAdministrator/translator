#!/usr/bin/env bash
# GPU thermal guard for the passively-cooled RTX 3060 (OEM fan replaced with a
# case fan, so headroom is smaller than stock). Runs as a user service, polls
# every 10 s.
#
# Two consumers share this card and they are NOT equally important:
#
#   live translation  — stopping it interrupts a service, so a trip also drops
#                       ~/translate-manual.flag and a human must clear it.
#   archive backlog   — entirely deferrable, so a trip just stops it and the
#                       guard resumes it by itself once the card has cooled.
#
# The earlier version only ever looked at translate.service, so a backlog job
# could take the card to 90 C with the guard silent — it never even logged,
# because the whole trip body sat behind "is translate.service active".
set -u
WARN_C=${WARN_C:-75}
TRIP_C=${TRIP_C:-85}
RESUME_C=${RESUME_C:-70}          # backlog may restart once we are back under this
LOG="$HOME/sermons/logs/gpu-thermal-guard.log"
FLAG="$HOME/translate-manual.flag"
BACKLOG_STOP="$HOME/Multi-Bitrate-Sermons/stop.flag"
# Marks a backlog stop as ours. The scheduler also uses BACKLOG_STOP to drain
# before a service window; clearing that one would start archive work during a
# service, so we only ever remove a stop we placed ourselves.
OURS="$HOME/.gpu-guard-stopped-backlog"

mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

warned=0
log "guard started (warn ${WARN_C}C, trip ${TRIP_C}C, backlog resumes under ${RESUME_C}C)"
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ "$t" =~ ^[0-9]+$ ]]; then
    if [ "$t" -ge "$TRIP_C" ]; then
      p=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | head -1)

      # Deferrable work first: this is usually what is heating the card, and
      # stopping it costs nothing but throughput.
      if [ ! -e "$BACKLOG_STOP" ]; then
        mkdir -p "$(dirname "$BACKLOG_STOP")" 2>/dev/null
        touch "$BACKLOG_STOP" && touch "$OURS"
        log "TRIP: GPU ${t}C >= ${TRIP_C}C (power $p) -> archive backlog stopped"
      fi

      # A running service is different: interrupting one needs a human to look.
      if systemctl --user is-active --quiet translate.service; then
        systemctl --user stop translate.service
        touch "$FLAG"
        log "TRIP: GPU ${t}C -> translate.service stopped, $FLAG set (remove it to allow restart)"
      fi
      warned=1

    elif [ "$t" -lt "$RESUME_C" ]; then
      # Cooled off. Release only a backlog stop that we placed.
      if [ -e "$OURS" ]; then
        rm -f "$BACKLOG_STOP" "$OURS"
        log "cooled to ${t}C -> archive backlog allowed again"
      fi
      warned=0

    elif [ "$t" -ge "$WARN_C" ]; then
      if [ "$warned" -eq 0 ]; then
        log "WARN: GPU ${t}C (power $(nvidia-smi --query-gpu=power.draw --format=csv,noheader))"
        warned=1
      fi
    fi
  fi
  sleep 10
done
