#!/usr/bin/env bash
# Test suite for ab_bench.sh — the RCCL A/B benchmark harness.
# Run: bash projects/rccl/perf_results/test_ab_bench.sh
#
# These tests exercise the acceptance criteria without needing GPUs:
#   1. File exists + --help works
#   2. self-consistency: comparing a baseline file to itself -> <=2% delta
#   3. output columns: size, baseline busbw, after busbw, uplift %, #wrong (or rel-err) + PASS/FAIL verdict
#   4. parser correctness against a known rccl-tests fixture
#   5. lossy (rel-err) gate mode

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB="$HERE/ab_bench.sh"
BASE="$HERE/baseline/allreduce_half.txt"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== AC1: file exists and --help =="
check "ab_bench.sh exists"        "[ -f '$AB' ]"
check "--help exits 0"            "bash '$AB' --help >/dev/null 2>&1"
check "--help mentions compare"   "bash '$AB' --help 2>&1 | grep -qi compare"

echo "== parser: fixture =="
# Build a tiny synthetic rccl-tests output with known busbw values.
cat > "$TMP/fix.txt" <<'EOF'
# rccl-tests version 2.18.3
# Collective test starting: all_reduce_perf
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
        1024           512  bfloat16     sum      -1    45.80   10.00   20.00       0    43.80   10.10   20.10       0
        2048          1024  bfloat16     sum      -1    45.37   30.00   40.00       0    43.91   31.00   41.00       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 30.0
EOF
PARSED="$(bash "$AB" parse "$TMP/fix.txt")"
check "parser emits size 1024"    "echo \"\$PARSED\" | grep -q '^1024 '"
check "parser busbw 20.00 @1024"  "echo \"\$PARSED\" | awk '\$1==1024{exit !(\$3==20 || \$3==20.00)}'"
check "parser busbw 40.00 @2048"  "echo \"\$PARSED\" | awk '\$1==2048{exit !(\$3==40 || \$3==40.00)}'"

echo "== AC4: compare output columns + verdict =="
OUT="$(bash "$AB" compare --baseline "$BASE" --after "$BASE")"
check "col header size"           "echo \"\$OUT\" | grep -qi 'size'"
check "col baseline busbw"        "echo \"\$OUT\" | grep -qi 'baseline'"
check "col after busbw"           "echo \"\$OUT\" | grep -qi 'after'"
check "col uplift %"              "echo \"\$OUT\" | grep -qi 'uplift'"
check "col #wrong"                "echo \"\$OUT\" | grep -qi '#wrong'"
check "verdict line present"      "echo \"\$OUT\" | grep -qiE 'VERDICT:\s*(PASS|FAIL)'"

echo "== AC3: self-consistency base-vs-base <=2% =="
check "self-compare verdict PASS" "echo \"\$OUT\" | grep -qi 'VERDICT: PASS'"
# max abs uplift reported should be ~0 (<=2%)
MAXUP="$(echo "$OUT" | grep -i 'max .*regression' )"
check "max regression reported"   "[ -n \"\$MAXUP\" ]"
check "crossover reported"        "echo \"\$OUT\" | grep -qi 'crossover'"
# Programmatic self-consistency: selftest subcommand
check "selftest passes on base"   "bash '$AB' selftest --baseline '$BASE' >/dev/null 2>&1"

echo "== uplift math: synthetic after with +10% =="
# Make an 'after' file where busbw is 10% higher than baseline fixture.
cat > "$TMP/after.txt" <<'EOF'
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
        1024           512  bfloat16     sum      -1    41.00   11.00   22.00       0    40.00   11.10   22.10       0
        2048          1024  bfloat16     sum      -1    41.00   33.00   44.00       0    40.00   34.00   45.00       0
EOF
CMP="$(bash "$AB" compare --baseline "$TMP/fix.txt" --after "$TMP/after.txt")"
# 20 -> 22 = +10%
check "uplift ~+10% @1024"        "echo \"\$CMP\" | awk '/^\| *1024/{gsub(/[|%]/,\"\"); for(i=1;i<=NF;i++) if(\$i+0>9.9 && \$i+0<10.1){found=1}} END{exit !found}'"
check "positive-uplift verdict"   "echo \"\$CMP\" | grep -qi 'VERDICT: PASS'"

echo "== correctness gate: #wrong>0 -> FAIL =="
cat > "$TMP/bad.txt" <<'EOF'
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
        1024           512  bfloat16     sum      -1    41.00   11.00   22.00       3    40.00   11.10   22.10       3
EOF
BADOUT="$(bash "$AB" compare --baseline "$TMP/fix.txt" --after "$TMP/bad.txt")"
check "wrong>0 -> VERDICT FAIL"   "echo \"\$BADOUT\" | grep -qi 'VERDICT: FAIL'"

echo "== lossy mode: rel-err column instead of #wrong =="
cat > "$TMP/relerr.txt" <<'EOF'
1024 0.004
2048 0.006
EOF
LOSSY="$(bash "$AB" compare --baseline "$TMP/fix.txt" --after "$TMP/after.txt" --relerr-file "$TMP/relerr.txt" --relerr-max 0.01)"
check "rel-err column shown"       "echo \"\$LOSSY\" | grep -qi 'rel-err'"
check "rel-err within max -> PASS" "echo \"\$LOSSY\" | grep -qi 'VERDICT: PASS'"
cat > "$TMP/relerr_bad.txt" <<'EOF'
1024 0.05
2048 0.006
EOF
LOSSYBAD="$(bash "$AB" compare --baseline "$TMP/fix.txt" --after "$TMP/after.txt" --relerr-file "$TMP/relerr_bad.txt" --relerr-max 0.01)"
check "rel-err over max -> FAIL"   "echo \"\$LOSSYBAD\" | grep -qi 'VERDICT: FAIL'"

echo
echo "==== $pass passed, $fail failed ===="
[ "$fail" -eq 0 ]
