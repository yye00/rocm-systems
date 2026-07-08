#!/usr/bin/env bash
# Stall-watcher for RCCL build. Fires when a bob run is wedged (mode-C hang),
# including the alive-but-FROZEN subagent case (claude process up but cputime
# not advancing). Multi-signal, won't false-kill a live compile/benchmark.
cd /home/yelkhamr/dark-factory/rocm-systems
while true; do
  sleep 60
  P=$(pgrep -f "bob run --all" | head -1); [ -z "$P" ] && continue
  LOG_AGE=$(( $(date +%s) - $(stat -c %Y build.log 2>/dev/null || echo $(date +%s)) ))
  [ "$LOG_AGE" -lt 600 ] && continue                       # log fresh (<10min): alive
  # real build/benchmark work = alive
  [ "$(pgrep -c -f 'amdclang|hipcc|make -j|cmake|clang++' 2>/dev/null)" -gt 0 ] && continue
  [ "$(pgrep -af 'all_reduce_perf|reduce_scatter_perf|all_gather_perf|alltoall_perf' 2>/dev/null | grep -v claude | wc -l)" -gt 0 ] && continue
  # source/test writes (exclude build-output settling) = alive
  WR=$(find . -type f -mmin -3 2>/dev/null | grep -vE '/\.git/|build\.log|/\.bob/|/build/|/_deps/|CMakeFiles|stall_watcher|supervisor\.log|/\.remember/|bob\.db' | wc -l)
  [ "$WR" -gt 0 ] && continue
  # cputime advance check on BOTH bob run AND any subagent — frozen BOTH => wedged
  CL=$(pgrep -f 'claude --output-format' | head -1)
  b1=$(ps -o cputime= -p $P 2>/dev/null); c1=$(ps -o cputime= -p $CL 2>/dev/null)
  sleep 25
  b2=$(ps -o cputime= -p $P 2>/dev/null); c2=$(ps -o cputime= -p $CL 2>/dev/null)
  # convert HH:MM:SS cputime to seconds; near-frozen = delta < 3s (hung agents creep ~1s)
  _sec(){ awk -F: "{n=NF; s=0; for(i=1;i<=n;i++) s=s*60+\$i; print s}" <<< "${1:-0:0}"; }
  bd=$(( $(_sec "$b2") - $(_sec "$b1") )); cd=$(( $(_sec "$c2") - $(_sec "$c1") ))
  if [ "$bd" -lt 3 ] && [ "$cd" -lt 3 ]; then
    echo "$(date +%T) STALL-WATCHER: killing wedged bob run pid=$P (log_age=${LOG_AGE}s no-compile no-perf no-src-writes bobrun-cputime=$b1 FROZEN subagent-cputime=$c1 FROZEN)"
    kill -9 $P 2>/dev/null; pkill -9 -f claude 2>/dev/null
  fi
done
