#!/usr/bin/env bash
# Stall-watcher for RCCL build. Catches BOTH executing-phase wedges AND
# pre-execution wedges (mode-C hang during spec-stability LLM re-synth while a
# feature is still 'ready'/'pending'). Fires when bob run is alive but:
#   claude=0 AND cputime frozen across 25s AND build.log mtime >300s AND
#   no workspace writes in 4min. Multi-signal => won't false-kill a live compile
#   (which moves cputime + writes files). Never touches DB (supervisor resets).
cd /home/yelkhamr/dark-factory/rocm-systems
while true; do
  sleep 60
  P=$(pgrep -f "bob run --all" | head -1); [ -z "$P" ] && continue
  LOG_AGE=$(( $(date +%s) - $(stat -c %Y build.log 2>/dev/null || echo $(date +%s)) ))
  [ "$LOG_AGE" -lt 300 ] && continue                       # log fresh: alive
  [ "$(pgrep -f claude | wc -l)" -gt 0 ] && continue       # subagents alive
  [ "$(pgrep -c -f "amdclang|hipcc|make -j|all_reduce_perf|cmake" 2>/dev/null)" -gt 0 ] && continue  # build/bench alive
  WR=$(find . -type f -mmin -4 2>/dev/null | grep -vE '/\.git/|build\.log|/\.bob/|stall_watcher|supervisor\.log' | wc -l)
  [ "$WR" -gt 0 ] && continue                              # recent writes: alive
  t1=$(ps -o cputime= -p $P 2>/dev/null); sleep 25; t2=$(ps -o cputime= -p $P 2>/dev/null)
  [ "$t1" != "$t2" ] && continue                           # cputime advancing: alive
  echo "$(date +%T) STALL-WATCHER: killing wedged bob run pid=$P (log_age=${LOG_AGE}s claude=0 writes4m=0 cputime=$t1 frozen)"
  kill -9 $P 2>/dev/null; pkill -9 -f claude 2>/dev/null
done
