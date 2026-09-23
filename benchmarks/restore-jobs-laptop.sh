#!/bin/bash
# Put the laptop's nightly ASR repair wave back after the benchmark.
systemctl --user enable --now repair-night-start.timer repair-day-stop.timer 2>&1 | tail -1
echo "[$(date '+%F %T')] repair timers restored: $(systemctl --user is-enabled repair-night-start.timer) / $(systemctl --user is-enabled repair-day-stop.timer)" \
  >> /home/administrator/Projects/translator/benchmarks/translate/laptop-run.log
