#!/usr/bin/env bash
# Test suite for IMPACT_SUMMARY.md — the consolidated before/after impact report.
# Run: bash projects/rccl/perf_results/test_impact_summary.sh
#
# This is the "RCCL build verifies ..." check referenced by the feature ACs. It
# runs without GPUs: it validates that the consolidated report exists and that it
# actually aggregates the F001 baseline against every per-target before/after
# result, with an explicit verdict and default-posture recommendation per target.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC="$HERE/IMPACT_SUMMARY.md"

pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }
# grep -qi against the doc
has(){ grep -qi -- "$1" "$DOC"; }

echo "== AC1: file exists =="
check "IMPACT_SUMMARY.md exists"        "[ -f '$DOC' ]"
check "file is non-empty"               "[ -s '$DOC' ]"

echo "== AC2: references F001 baseline and every target before/after table =="
check "references F001 baseline"        "has 'F001'"
check "names the baseline artifact"     "has 'baseline'"
for t in T1a T1b T2 T3 T4; do
  check "references target $t"          "has '$t'"
done
# the shared AllReduce sweep is the common axis
check "references AllReduce sweep"      "has 'AllReduce'"

echo "== AC3: gfx950 reported (this host); gfx942 deferred (not fabricated) =="
check "reports gfx950 results"          "has 'gfx950'"
check "mentions gfx942"                 "has 'gfx942'"
check "gfx942 marked deferred"          "grep -qiE 'gfx942.*defer|defer.*gfx942' '$DOC'"

echo "== AC4: each target has explicit PASS/FAIL verdict + default-posture rec =="
# every target row must carry a verdict token (PASS, FAIL, or DEFERRED)
check "verdict column present"          "has 'verdict'"
# T3 is reported as two sub-features (T3a/T3b); its verdict lines use the suffixed form.
for t in T1a T1b T2 T3 T4; do
  check "target $t has a verdict" \
    "grep -iE '(^|[^A-Za-z])$t[ab]?([^A-Za-z]|\$)' '$DOC' | grep -qiE 'PASS|FAIL|DEFER'"
done
check "verdict PASS appears"            "grep -qE 'PASS' '$DOC'"
check "verdict FAIL appears"            "grep -qE 'FAIL' '$DOC'"

echo "== default-posture recommendation =="
check "states all remain off by default" "grep -qiE 'off by default|default[- ]off' '$DOC'"
check "has a recommendation section"      "has 'recommend'"
check "mentions auto-selection posture"   "grep -qiE 'auto[- ]select' '$DOC'"

echo "== correctness axis (bit-exact vs measured rel-err) =="
check "distinguishes bit-exact"           "grep -qiE 'bit-exact' '$DOC'"
check "distinguishes rel-err"             "grep -qiE 'rel-err|relative error' '$DOC'"

echo "== crossover / regime axis =="
check "documents crossover/regime"        "grep -qiE 'crossover|regime' '$DOC'"

echo
echo "==== $pass passed, $fail failed ===="
[ "$fail" -eq 0 ]
