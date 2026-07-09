#!/usr/bin/env bash
# Multi-GPU engagement verification: prove RCCL AllReduce runs use DISTINCT GPUs,
# not one. For np in {2,4,8}: start a per-GPU rocm-smi sampler, run all_reduce_perf_mpi,
# then report which GPU indices showed activity + the busbw + #wrong.
set -uo pipefail
RCCL_ROOT="${RCCL_ROOT:-$HOME/dark-factory/rocm-systems/projects/rccl}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/dark-factory/rocm-systems/projects/rccl-tests}"
PERF="$TESTS_ROOT/build/all_reduce_perf_mpi"
LIB="$RCCL_ROOT/build/release"
OUT="${OUT:-$(dirname "$0")/verify_mgpu_out}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"

echo "=== multi-GPU engagement check (date $(date -u +%Y-%m-%dT%H:%M:%SZ), host $(hostname)) ==="
for np in 2 4 8; do
  mon="$OUT/gpumon_np${np}.log"
  : > "$mon"
  # background per-GPU sampler every 0.5s for the run window
  ( for i in $(seq 1 240); do
      echo "$(rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use .%.: [0-9]+' | grep -oE '[0-9]+$' | paste -sd,)" >> "$mon"
      sleep 0.5
    done ) &
  MON=$!
  echo "--- np=$np: running AllReduce 64MiB (gate ON, DDA), validation -c 1 ---"
  RCCL_DDA_NRANKS_RELAX=1 timeout 150 mpirun --allow-run-as-root -np "$np" \
    "$PERF" -b 64M -e 64M -f 2 -g 1 -d float -c 1 -n 80 -w 20 \
    > "$OUT/ar_np${np}.txt" 2>&1
  kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
  busbw=$(grep -E '# Avg bus bandwidth' "$OUT/ar_np${np}.txt" | grep -oE '[0-9.]+' | tail -1)
  oob=$(grep -c '# Out of bounds values : 0 OK' "$OUT/ar_np${np}.txt")
  # which GPU indices ever went nonzero during the run
  active=$(awk -F, '{for(i=1;i<=NF;i++) if($i+0>0) seen[i]=1} END{n=0;s="";for(i=1;i<=8;i++) if(seen[i]){s=s (s?",":"") (i-1);n++}; print n" GPUs active: ["s"]"}' "$mon")
  # peak per-GPU
  peak=$(awk -F, '{for(i=1;i<=8;i++) if($i+0>mx[i])mx[i]=$i} END{s="";for(i=1;i<=8;i++)s=s sprintf("g%d=%d ",i-1,mx[i]); print s}' "$mon")
  echo "    busbw=${busbw:-ERR} GB/s  #wrong0_OK=$oob  ->  $active"
  echo "    peak per-GPU: $peak"
done
echo ""
echo "EXPECTATION: np=2 -> 2 distinct GPUs active; np=4 -> 4; np=8 -> 8."
echo "If any run shows only 1 GPU active, that is a RED FLAG (not really collective)."
echo "Logs: $OUT/"
