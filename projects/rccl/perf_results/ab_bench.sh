#!/usr/bin/env bash
# ab_bench.sh — RCCL A/B benchmark harness (before/after delta).
#
# Runs a named rccl-tests sweep, parses per-size busbw/algbw/latency/#wrong out
# of rccl-tests stdout, and emits a markdown before/after table with per-size
# busbw uplift %, the crossover size where the "after" run first beats the
# baseline, the max regression, and a PASS/FAIL verdict.
#
# For lossy targets (e.g. compressed collectives) the #wrong correctness gate is
# replaced by a measured mean-relative-error column via --relerr-file/--relerr-max.
#
# Subcommands:
#   run       Run an rccl-tests sweep and write its stdout to an output file.
#   parse     Parse an rccl-tests output file -> "size algbw busbw latency wrong".
#   compare   Emit a markdown before/after table + verdict from two output files.
#   selftest  Compare a baseline file against itself; assert <=2% delta (exit 0/1).
#
# This harness is the shared "AFTER" evidence generator for every RCCL target
# feature. It has no GPU dependency for parse/compare/selftest, so it can be
# unit-tested on any host; only `run` needs rccl-tests + GPUs.

set -euo pipefail

SELF_CONSISTENCY_PCT="2.0"   # AC: base-vs-base must report <=2% delta.
DEFAULT_RELERR_MAX="0.01"

# ---------------------------------------------------------------------------
usage() {
  cat <<'USAGE'
ab_bench.sh — RCCL A/B benchmark harness (before/after delta)

USAGE:
  ab_bench.sh run     --name NAME --collective COLL [options]   Run an rccl-tests sweep.
  ab_bench.sh parse   FILE                                      Parse rccl-tests output.
  ab_bench.sh compare --baseline BEFORE --after AFTER [options] Emit before/after table.
  ab_bench.sh selftest --baseline FILE                          Self-consistency check.
  ab_bench.sh --help                                            Show this help.

run options:
  --name NAME            Label for the sweep (used in the output filename).
  --collective COLL      rccl-tests binary stem: all_reduce, all_gather,
                         reduce_scatter, alltoall, broadcast, ...
  --dtype DTYPE          half | bfloat16 | float | ... (default: half)
  --ngpus N              Number of GPUs / ranks (default: 8)
  --minbytes SIZE        Min message size, e.g. 8       (default: 8)
  --maxbytes SIZE        Max message size, e.g. 2G      (default: 2G)
  --step FACTOR          Size step factor               (default: 2)
  --op REDOP             Reduction op (sum, prod, ...)  (default: sum)
  --out FILE             Where to write rccl-tests stdout (default: ./NAME.txt)
  --bindir DIR           Directory holding *_perf binaries
                         (default: $RCCL_TESTS_DIR or ../../rccl-tests/build)
  --env "K=V K=V"        Extra env vars exported for the run (space-separated).

compare options:
  --baseline FILE        BEFORE rccl-tests output (gate-OFF / prior build).
  --after FILE           AFTER rccl-tests output  (gate-ON / new build).
  --relerr-file FILE     Lossy mode: "size relerr" pairs replacing the #wrong gate.
  --relerr-max FLOAT     Max allowed mean relative error (default: 0.01).
  --min-uplift FLOAT     Min mean busbw uplift %% for PASS (default: 0 => no regression).
  --max-regression FLOAT Max tolerated per-size busbw regression %% (default: 2.0).

Examples:
  ab_bench.sh compare --baseline baseline/allreduce_half.txt --after after.txt
  ab_bench.sh selftest --baseline baseline/allreduce_half.txt
USAGE
}

die() { echo "ab_bench.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# parse FILE
#   Emit one line per data row: "size algbw busbw latency_us wrong"
#   Uses the out-of-place columns (time algbw busbw #wrong) which are the
#   canonical rccl-tests reporting columns. Rows with N/A #wrong are treated
#   as 0 (rccl-tests prints N/A for in-place duplicates of zero-size rows).
parse_file() {
  local f="$1"
  [ -f "$f" ] || die "parse: file not found: $f"
  awk '
    # Skip comments/blank/header lines.
    /^[[:space:]]*#/  { next }
    /^[[:space:]]*$/  { next }
    {
      # Data row layout (out-of-place block first):
      #  1:size 2:count 3:type 4:redop 5:root 6:time 7:algbw 8:busbw 9:#wrong ...
      size=$1; time=$6; algbw=$7; busbw=$8; wrong=$9
      if (size !~ /^[0-9]+$/) next          # not a numeric data row
      if (wrong == "N/A" || wrong == "")  wrong=0
      # Aggregate duplicate zero-size warmup rows: keep the last per size.
      key=size
      A_size[key]=size; A_algbw[key]=algbw; A_busbw[key]=busbw
      A_time[key]=time; A_wrong[key]=wrong
      order[key]=NR
    }
    END {
      # Emit sorted by numeric size.
      n=0
      for (k in A_size) { keys[n++]=k }
      # simple insertion sort by numeric value
      for (i=1;i<n;i++){ x=keys[i]; j=i-1; while(j>=0 && (keys[j]+0)>(x+0)){keys[j+1]=keys[j];j--}; keys[j+1]=x }
      for (i=0;i<n;i++){ k=keys[i];
        printf "%d %s %s %s %d\n", A_size[k], A_algbw[k], A_busbw[k], A_time[k], A_wrong[k]
      }
    }
  ' "$f"
}

# ---------------------------------------------------------------------------
# run: execute an rccl-tests sweep
cmd_run() {
  local name="" collective="" dtype="half" ngpus="8"
  local minb="8" maxb="2G" step="2" op="sum" out="" bindir="" extraenv=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --name)       name="$2"; shift 2;;
      --collective) collective="$2"; shift 2;;
      --dtype)      dtype="$2"; shift 2;;
      --ngpus)      ngpus="$2"; shift 2;;
      --minbytes)   minb="$2"; shift 2;;
      --maxbytes)   maxb="$2"; shift 2;;
      --step)       step="$2"; shift 2;;
      --op)         op="$2"; shift 2;;
      --out)        out="$2"; shift 2;;
      --bindir)     bindir="$2"; shift 2;;
      --env)        extraenv="$2"; shift 2;;
      *) die "run: unknown option $1";;
    esac
  done
  [ -n "$name" ]       || die "run: --name required"
  [ -n "$collective" ] || die "run: --collective required"
  [ -n "$bindir" ]     || bindir="${RCCL_TESTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../rccl-tests/build" 2>/dev/null && pwd || true)}"
  [ -n "$out" ]        || out="./${name}.txt"

  local bin="$bindir/${collective}_perf"
  [ -x "$bin" ] || die "run: rccl-tests binary not found/executable: $bin"

  if [ -n "$extraenv" ]; then
    # shellcheck disable=SC2086
    export $extraenv
  fi
  echo "ab_bench: running $bin -b $minb -e $maxb -f $step -g $ngpus -d $dtype -o $op -c 1" >&2
  "$bin" -b "$minb" -e "$maxb" -f "$step" -g "$ngpus" -d "$dtype" -o "$op" -c 1 | tee "$out"
  echo "ab_bench: wrote $out" >&2
}

# ---------------------------------------------------------------------------
# compare: markdown before/after table + verdict
cmd_compare() {
  local base="" after="" relerr_file="" relerr_max="$DEFAULT_RELERR_MAX"
  local min_uplift="0" max_regression="2.0"
  while [ $# -gt 0 ]; do
    case "$1" in
      --baseline)      base="$2"; shift 2;;
      --after)         after="$2"; shift 2;;
      --relerr-file)   relerr_file="$2"; shift 2;;
      --relerr-max)    relerr_max="$2"; shift 2;;
      --min-uplift)    min_uplift="$2"; shift 2;;
      --max-regression) max_regression="$2"; shift 2;;
      *) die "compare: unknown option $1";;
    esac
  done
  [ -f "$base" ]  || die "compare: --baseline file not found: $base"
  [ -f "$after" ] || die "compare: --after file not found: $after"

  local lossy=0
  [ -n "$relerr_file" ] && { lossy=1; [ -f "$relerr_file" ] || die "compare: --relerr-file not found: $relerr_file"; }

  local btmp atmp
  btmp="$(mktemp)"; atmp="$(mktemp)"
  parse_file "$base"  > "$btmp"
  parse_file "$after" > "$atmp"

  awk -v lossy="$lossy" -v relf="$relerr_file" -v relmax="$relerr_max" \
      -v minup="$min_uplift" -v maxreg="$max_regression" -v scpct="$SELF_CONSISTENCY_PCT" \
      -v afile="$atmp" '
    function abs(x){ return x<0?-x:x }
    BEGIN {
      # Load after-run into arrays keyed by size.
      while ((getline line < afile) > 0) {
        nf=split(line, p, " ")
        if (nf<5) continue
        s=p[1]; a_busbw[s]=p[3]; a_algbw[s]=p[2]; a_time[s]=p[4]; a_wrong[s]=p[5]
        a_have[s]=1
      }
      close(afile)
      # Load rel-err file if lossy.
      if (lossy) {
        while ((getline rl < relf) > 0) {
          if (rl ~ /^[[:space:]]*#/ || rl ~ /^[[:space:]]*$/) continue
          split(rl, rp, " "); relerr[rp[1]]=rp[2]
        }
        close(relf)
      }
      # Header.
      if (lossy)
        print "| size (B) | baseline busbw (GB/s) | after busbw (GB/s) | uplift % | rel-err |"
      else
        print "| size (B) | baseline busbw (GB/s) | after busbw (GB/s) | uplift % | #wrong |"
      print "|---:|---:|---:|---:|---:|"
    }
    # Baseline rows on stdin.
    {
      s=$1; b_busbw=$3
      if (!(s in a_have)) next
      ab=a_busbw[s]+0; bb=b_busbw+0
      up = (bb>0) ? (ab-bb)/bb*100.0 : 0.0
      # track stats
      sum_up += up; nrows++
      if (up > best_up || nrows==1) { best_up=up }
      if (up < worst_up || nrows==1) { worst_up=up }
      # crossover: first size (ascending) where after strictly beats baseline
      if (crossover=="" && ab > bb) crossover=s
      # gate
      gatecol=""
      if (lossy) {
        re = (s in relerr) ? relerr[s]+0 : 0
        if (re > relmax+0) gate_fail=1
        gatecol=sprintf("%.4g", re)
      } else {
        w=a_wrong[s]+0
        if (w > 0) gate_fail=1
        gatecol=sprintf("%d", w)
      }
      printf "| %d | %.2f | %.2f | %+.2f | %s |\n", s, bb, ab, up, gatecol
    }
    END {
      mean_up = (nrows>0) ? sum_up/nrows : 0
      # max gate-OFF regression = most negative uplift (report as positive magnitude)
      max_reg = (worst_up < 0) ? -worst_up : 0
      print ""
      printf "mean busbw uplift: %+.2f%%\n", mean_up
      printf "max busbw uplift:  %+.2f%%\n", best_up
      printf "max regression:    %.2f%% (most negative per-size delta)\n", max_reg
      if (crossover != "")
        printf "crossover size:    %d B (first size where after beats baseline)\n", crossover
      else
        print  "crossover size:    none (after never strictly beats baseline)"

      # Verdict.
      verdict="PASS"; reason=""
      if (gate_fail) { verdict="FAIL"; reason="correctness gate violated" }
      else if (max_reg > maxreg+0) { verdict="FAIL"; reason=sprintf("regression %.2f%% > %.2f%% budget", max_reg, maxreg+0) }
      else if (mean_up < minup+0)  { verdict="FAIL"; reason=sprintf("mean uplift %.2f%% < %.2f%% required", mean_up, minup+0) }
      else { reason=sprintf("no correctness loss, regression within %.2f%%", maxreg+0) }
      printf "VERDICT: %s (%s)\n", verdict, reason
    }
  ' "$btmp"

  rm -f "$btmp" "$atmp"
}

# ---------------------------------------------------------------------------
# selftest: base vs base must report <=SELF_CONSISTENCY_PCT delta.
cmd_selftest() {
  local base=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --baseline) base="$2"; shift 2;;
      *) die "selftest: unknown option $1";;
    esac
  done
  [ -f "$base" ] || die "selftest: --baseline file not found: $base"
  local out
  out="$(cmd_compare --baseline "$base" --after "$base")"
  echo "$out"
  # Extract max regression and max uplift; both must be within tolerance.
  local maxreg maxup
  maxreg="$(echo "$out" | awk -F'[: %]+' '/^max regression:/{print $3}')"
  maxup="$(echo "$out"  | awk '/^max busbw uplift:/{v=$4; gsub(/[+%]/,"",v); if(v<0)v=-v; print v}')"
  awk -v r="${maxreg:-0}" -v u="${maxup:-0}" -v lim="$SELF_CONSISTENCY_PCT" '
    BEGIN{
      if (r+0 <= lim+0 && u+0 <= lim+0) { exit 0 }
      printf "selftest: self-consistency FAILED (reg=%.4f up=%.4f > %.2f%%)\n", r, u, lim > "/dev/stderr"
      exit 1
    }'
}

# ---------------------------------------------------------------------------
main() {
  [ $# -eq 0 ] && { usage; exit 0; }
  case "$1" in
    -h|--help|help) usage; exit 0;;
    run)      shift; cmd_run "$@";;
    parse)    shift; [ $# -ge 1 ] || die "parse: FILE required"; parse_file "$1";;
    compare)  shift; cmd_compare "$@";;
    selftest) shift; cmd_selftest "$@";;
    *) die "unknown subcommand '$1' (try --help)";;
  esac
}
main "$@"
