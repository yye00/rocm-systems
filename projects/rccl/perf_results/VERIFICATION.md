# Independent Verification — T3a `RCCL_DDA_NRANKS_RELAX` AllReduce win

This directory contains a **self-contained, reproducible** verification of the one
positive performance result from the RCCL single-node-opt build: enabling
`RCCL_DDA_NRANKS_RELAX=1` makes **2/4-rank AllReduce faster than the default ring**
while staying **bit-exact** (`#wrong==0`), on 8× AMD Instinct MI355X (gfx950 / CDNA4).

You do not have to trust any previously reported number. Run the script and check
the verdict yourself.

## How to reproduce

```bash
cd projects/rccl/perf_results
bash verify_t3a_ddarelax.sh          # writes logs to ./verify_out/
```

Requirements: this branch's `librccl.so`, `rccl-tests/build/all_reduce_perf_mpi`,
`mpirun`, ROCm 7.x, an 8× gfx950 node. Override `RCCL_ROOT` / `TESTS_ROOT` env
vars if your paths differ. The script rebuilds nothing — it measures the live
library, so the result is attributable to the gate only.

## What the script checks (three independent gates)

1. **A/B benchmark** — for each (rank ∈ {2,4}, size ∈ {16M,64M,256M}): gate OFF
   (baseline ring) vs gate ON (DDA path), `all_reduce_perf_mpi` with validation
   on (`-c 1`, so `#wrong` is computed), 1 process per GPU (DDA IPC only engages
   with `directMode=0`, i.e. real multi-process).
2. **Correctness** — every run must report `Out of bounds values : 0 OK` (`#wrong==0`).
   A lossless optimization that changed a single bit would fail here.
3. **Engagement proof (anti-misattribution)** — with `NCCL_DEBUG=INFO`, the DDA
   IPC path must initialize (`ncclDdaIpcCommInit`) on the ranks ONLY when the gate
   is on, and NOT at all when off. This proves the speedup comes from the gate,
   not from an unrelated env/warmup difference.

**Verdict rule:** CONFIRMED iff speedup>1.0 at both 2 and 4 ranks AND all runs
`#wrong==0` AND DDA-init lines are (gate-ON > 0, gate-OFF == 0).

## Result of the recorded run (`verify_out/`, 2026-07-09T02:04Z)

Host `smci355-ccs-aus-m03-05`, ROCm 7.0.1, 8× gfx950, `librccl.so.1.0` 280,753,072 B.

| ranks | size | ring OFF (GB/s) | DDA ON (GB/s) | speedup | #wrong |
|------:|-----:|----------------:|--------------:|--------:|-------:|
| 2 | 16 MiB | 57.02 | 60.66 | **1.06×** | 0 |
| 2 | 64 MiB | 55.71 | 61.50 | **1.10×** | 0 |
| 2 | 256 MiB | 57.12 | 57.09 | 1.00× (falls back) | 0 |
| 4 | 16 MiB | 154.23 | 169.94 | **1.10×** | 0 |
| 4 | 64 MiB | 156.36 | 176.32 | **1.13×** | 0 |
| 4 | 256 MiB | 161.00 | 160.97 | 1.00× (falls back) | 0 |

- **All 12 runs: `#wrong==0`** (bit-exact, lossless).
- **Engagement proof:** DDA-init lines = 4 with gate ON, **0** with gate OFF.
- Speedup is real at 16–64 MiB for 2/4 ranks; at 256 MiB the path transparently
  falls back to the ring (≈1.00×, no regression).

**VERDICT: CONFIRMED.** The +13% (64 MiB, 4-rank) reproduces the originally
reported result independently, bit-exact, with the gate proven to be the cause.

## Why it works (mechanism)

RCCL's default AllReduce is a **ring** (p−1 sequential hops). At low rank counts
(2/4 GPUs) the collective is **latency-bound**, not bandwidth-bound, so the DDA
direct-peer-IPC one-shot/two-shot kernel — which avoids ring hops via IPC
load/stores — wins. At 8 ranks (and at very large sizes) the ring becomes
bandwidth-bound and the advantage disappears, which is why the recommended
default posture is **auto-select DDA scoped to the 2/4-rank regime only**, and
the gate ships **default-off**.

## Measurement trap to avoid

DDA IPC engages **only with 1 process per GPU** (`mpirun -np N ... -g 1`). A
single-process `all_reduce_perf -g N` sets `directMode=1`, which skips DDA
scratch/barrier allocation — so gate-on would equal gate-off and you'd see no
effect. The script uses `mpirun -np {2,4}` specifically to avoid this.

## Provenance

- Optimization + reports: this branch (`rccl-single-node-opt`),
  `projects/rccl/perf_results/T3_sym_ll128.md`, `IMPACT_SUMMARY.md`.
- Original per-target A/B logs: `projects/rccl/perf_results/t3/`.
- This independent verification: `verify_t3a_ddarelax.sh` + `verify_out/` +
  `verify_out.log` (this run).
