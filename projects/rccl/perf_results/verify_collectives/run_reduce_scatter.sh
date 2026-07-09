#!/bin/bash
set -u
LIB=~/dark-factory/rocm-systems/projects/rccl/build/release
BIN=~/dark-factory/rocm-systems/projects/rccl-tests/build/reduce_scatter_perf_mpi
OUT=~/dark-factory/rocm-systems/projects/rccl/perf_results/verify_collectives
cd "$OUT"

for NP in 2 4 8; do
  for R in 1 2 3; do
    MON="$OUT/reduce_scatter_np${NP}_run${R}.gpumon"
    LOG="$OUT/reduce_scatter_np${NP}_run${R}.txt"
    # start sampler
    ( for i in $(seq 1 400); do rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use .%.: [0-9]+' | grep -oE '[0-9]+$' | paste -sd, ; sleep 0.5; done ) > "$MON" &
    MONPID=$!
    sleep 1
    LD_LIBRARY_PATH=$LIB RCCL_DDA_NRANKS_RELAX=1 mpirun --allow-run-as-root -np $NP $BIN -b 64M -e 64M -f 2 -g 1 -d float -c 1 -n 80 -w 20 > "$LOG" 2>&1
    RC=$?
    kill $MONPID 2>/dev/null
    wait $MONPID 2>/dev/null
    echo "np=$NP run=$R rc=$RC done"
  done
done
echo "ALL DONE"
