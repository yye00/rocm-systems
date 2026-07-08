#!/bin/bash
set -u
ROOT=/home/yelkhamr/dark-factory/rocm-systems
BL=$ROOT/projects/rccl/perf_results/baseline
BIN=$ROOT/projects/rccl-tests/build
export LD_LIBRARY_PATH=$ROOT/projects/rccl/build/release:${LD_LIBRARY_PATH:-}
run() {
  local out=$1; shift
  echo ">>> $out : $*"
  "$@" -b 8 -e 2G -f 2 -g 8 -c 1 > "$BL/$out" 2>&1
  echo "    exit=$? wrong=$(grep -oE 'Out of bounds values : [0-9]+' "$BL/$out" | tail -1)"
}
run allreduce_half.txt      $BIN/all_reduce_perf     -d half
run allreduce_bfloat16.txt  $BIN/all_reduce_perf     -d bfloat16
run reducescatter_half.txt  $BIN/reduce_scatter_perf -d half
run allgather_half.txt      $BIN/all_gather_perf     -d half
run alltoall_half.txt       $BIN/alltoall_perf       -d half
echo "ALL SWEEPS DONE"
