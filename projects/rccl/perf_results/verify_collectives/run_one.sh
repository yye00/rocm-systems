#!/bin/bash
# args: NP RUN
NP=$1
RUN=$2
OUTDIR=~/dark-factory/rocm-systems/projects/rccl/perf_results/verify_collectives
BIN=~/dark-factory/rocm-systems/projects/rccl-tests/build/all_reduce_perf_mpi
LIB=~/dark-factory/rocm-systems/projects/rccl/build/release
OUT=$OUTDIR/all_reduce_np${NP}_run${RUN}.txt
MON=$OUTDIR/all_reduce_np${NP}_run${RUN}.gpumon

# start monitor
( for i in $(seq 1 400); do rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use .%.: [0-9]+' | grep -oE '[0-9]+$' | paste -sd, ; sleep 0.5; done ) > "$MON" &
MONPID=$!

LD_LIBRARY_PATH=$LIB RCCL_DDA_NRANKS_RELAX=1 mpirun --allow-run-as-root -np $NP $BIN -b 64M -e 64M -f 2 -g 1 -d float -c 1 -n 80 -w 20 > "$OUT" 2>&1

kill $MONPID 2>/dev/null
wait $MONPID 2>/dev/null
echo "DONE np=$NP run=$RUN"
