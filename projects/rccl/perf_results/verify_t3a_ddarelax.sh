#!/usr/bin/env bash
# ============================================================================
# INDEPENDENT VERIFICATION — RCCL T3a  RCCL_DDA_NRANKS_RELAX  AllReduce win
# ============================================================================
# Purpose: reproduce, from scratch and auditably, the claim that
#   RCCL_DDA_NRANKS_RELAX=1 makes 2/4-rank AllReduce FASTER than the default
#   ring, while remaining BIT-EXACT (#wrong==0), on 8x MI355X (gfx950).
#
# It runs, for each (rank-count, size): gate OFF (baseline ring) vs gate ON
# (DDA path), with rccl-tests validation ON (-c 1) so #wrong is checked, and
# captures NCCL_DEBUG engagement proof that the DDA path actually initializes
# ONLY when the gate is on (the anti-misattribution check).
#
# Anyone can run this on a gfx950 8-GPU node and get the same verdict. No
# trust in prior numbers required — this rebuilds nothing and measures live.
#
# Usage:   bash verify_t3a_ddarelax.sh            (writes logs to ./verify_out/)
# Requires: librccl.so (this branch), rccl-tests all_reduce_perf_mpi, mpirun,
#           ROCm 7.x, 8x gfx950. Set RCCL_ROOT/TESTS_ROOT below if paths differ.
# ============================================================================
set -uo pipefail

RCCL_ROOT="${RCCL_ROOT:-$HOME/dark-factory/rocm-systems/projects/rccl}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/dark-factory/rocm-systems/projects/rccl-tests}"
PERF="$TESTS_ROOT/build/all_reduce_perf_mpi"
LIB="$RCCL_ROOT/build/release"
OUT="${OUT:-$(dirname "$0")/verify_out}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"

echo "=== ENVIRONMENT ==="
echo "date(UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "librccl: $(ls -la "$LIB"/librccl.so.1.0 2>/dev/null | awk '{print $5" bytes "$6" "$7" "$8}')"
echo "gate symbol embedded: $(strings "$LIB"/librccl.so.1.0 2>/dev/null | grep -c DDA_NRANKS_RELAX) (expect >=1)"
echo "all_reduce_perf_mpi: $([ -x "$PERF" ] && echo present || echo MISSING)"
echo "rocm: $(ls -d /opt/rocm-* 2>/dev/null | tr '\n' ' ')"
echo "gpus(gfx950): $(rocminfo 2>/dev/null | grep -c gfx950)"
echo "perf binary links librccl: $(ldd "$PERF" 2>/dev/null | grep -i rccl)"
echo ""

run() {  # run <ranks> <size> <gate> <tag>
  local ranks="$1" size="$2" gate="$3" tag="$4"
  local f="$OUT/ar_np${ranks}_${size}_relax${gate}.txt"
  RCCL_DDA_NRANKS_RELAX="$gate" timeout 180 mpirun --allow-run-as-root -np "$ranks" \
    "$PERF" -b "$size" -e "$size" -f 2 -g 1 -d float -c 1 -n 50 -w 10 \
    > "$f" 2>&1
  local busbw wrong oob
  busbw=$(grep -E '# Avg bus bandwidth' "$f" | grep -oE '[0-9.]+' | tail -1)
  oob=$(grep -c '# Out of bounds values : 0 OK' "$f")
  wrong=$(grep -E "^ *${size//M/}|^ *67108864|^ *[0-9]" "$f" | grep -oE '[0-9]+ *$' | tail -1)
  printf "  np=%s size=%s relax=%s -> busbw=%s GB/s  wrong0_OK=%s  raw=%s\n" \
    "$ranks" "$size" "$gate" "${busbw:-ERR}" "${oob:-0}" "$(basename "$f")"
}

echo "=== T3a A/B: gate OFF (ring) vs gate ON (DDA), validation -c 1 (#wrong checked) ==="
for ranks in 2 4; do
  for size in 16M 64M 256M; do
    run "$ranks" "$size" 0 "off"
    run "$ranks" "$size" 1 "on"
  done
done
echo ""

echo "=== ENGAGEMENT PROOF (anti-misattribution): DDA inits ONLY when gate=1 ==="
# With gate ON, all ranks must log ncclDdaIpcCommInit; with gate OFF, zero do.
RCCL_DDA_NRANKS_RELAX=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT timeout 120 \
  mpirun --allow-run-as-root -np 4 -x RCCL_DDA_NRANKS_RELAX -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  "$PERF" -b 64M -e 64M -f 2 -g 1 -d float -c 1 -n 5 -w 2 \
  > "$OUT/engage_relax_on.txt" 2>&1
RCCL_DDA_NRANKS_RELAX=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT timeout 120 \
  mpirun --allow-run-as-root -np 4 -x RCCL_DDA_NRANKS_RELAX -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  "$PERF" -b 64M -e 64M -f 2 -g 1 -d float -c 1 -n 5 -w 2 \
  > "$OUT/engage_relax_off.txt" 2>&1
echo "  DDA init lines with gate ON : $(grep -c 'ncclDdaIpcCommInit\|DdaIpc' "$OUT/engage_relax_on.txt") (expect >0, ideally 1/rank)"
echo "  DDA init lines with gate OFF: $(grep -c 'ncclDdaIpcCommInit\|DdaIpc' "$OUT/engage_relax_off.txt") (expect 0 — the gate is the only difference)"
echo ""

echo "=== SUMMARY TABLE (busbw GB/s; higher=better; #wrong must be 0) ==="
printf "%-8s %-8s %-14s %-14s %-10s\n" ranks size ring_OFF dda_ON speedup
for ranks in 2 4; do
  for size in 16M 64M 256M; do
    o=$(grep -E '# Avg bus bandwidth' "$OUT/ar_np${ranks}_${size}_relax0.txt" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)
    n=$(grep -E '# Avg bus bandwidth' "$OUT/ar_np${ranks}_${size}_relax1.txt" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)
    sp=$(awk "BEGIN{if($o>0)printf \"%.2fx\", $n/$o; else print \"n/a\"}" 2>/dev/null)
    printf "%-8s %-8s %-14s %-14s %-10s\n" "$ranks" "$size" "${o:-ERR}" "${n:-ERR}" "$sp"
  done
done
echo ""
echo "=== CORRECTNESS: every run must report 'Out of bounds values : 0 OK' ==="
echo "  runs with #wrong==0: $(grep -l '# Out of bounds values : 0 OK' "$OUT"/ar_np*.txt 2>/dev/null | wc -l) / $(ls "$OUT"/ar_np*.txt 2>/dev/null | wc -l)"
echo ""
echo "VERDICT RULE: T3a is CONFIRMED iff (speedup>1.0 at 2 and 4 ranks) AND (all runs #wrong==0)"
echo "             AND (DDA init lines: gate-ON>0, gate-OFF==0)."
echo "Logs in: $OUT/"
