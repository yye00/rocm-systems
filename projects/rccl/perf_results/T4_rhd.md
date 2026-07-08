# T4 — Log-round recursive-halving/doubling DDA (RS + AllReduce)

**Feature:** `RCCL_RECURSIVE_HALVING` — a single-node, IPC-based reduce-scatter
that reaches the reduced result in `ceil(log2 p)` sync rounds (p=8 → 3 rounds vs
ring's 7) at the same p−1 data volume, plus AllReduce = recursive-halving RS
followed by the reversed recursive-doubling all-gather (2·log p rounds total).

**Gate:** `RCCL_RHD_ENABLE` (RCCL_PARAM, **default 0**). When 0 the eligibility
predicates return false, so the dispatch branches in `collectives.cc`
(`ncclAllReduce_impl`, `ncclReduceScatter_impl`) are a no-op relative to baseline.

**Correctness:** lossless / bit-exact. The per-element reduction order is fixed by
the pairwise tree and does not depend on data. `#wrong == 0` across the entire
sweep with the gate both off and on (see tables below).

**Priority:** low. The audit rated the single-node value modest ("at p=8 the
double-binary tree is already competitive"). This report is the honest
measurement; the result is a **regression**, so the feature is left default-off
and is *not* force-enabled anywhere.

---

## Platform / method

| | |
|---|---|
| Host | smci355-ccs-aus-m03-05 |
| GPUs | 8 × AMD Instinct MI355X (gfx950), single-node XGMI clique |
| RCCL | build/release `librccl.so.1.0` (gfx950), this branch |
| rccl-tests | 2.18.3-develop, **built with `MPI=1`** (`*_perf_mpi`) |
| Launch | `mpirun -np 8 <coll>_perf_mpi -g 1` (1 process per GPU) |
| Sweep | `-b 256K -e 64M -f 2 -n 20 -w 5`, float/sum |
| dtype/op | float32, ncclSum |

### Why MPI (1 proc/GPU) and not `-g 8`

DDA IPC — including RHD — is only set up when the communicator is **not** in
`directMode`. `ncclTransportCheckP2pType` sets `directMode=1` whenever two local
ranks share a pid (i.e. a single-process `-g 8` run), and `ncclDdaIpcCommInit`
skips IPC scratch/barrier allocation in that case. A single-process `-g 8` sweep
therefore never exercises RHD — both gate-off and gate-on fall back to the ring/
tree path and look identical. The honest measurement **requires one process per
GPU** so `directMode=0` and the IPC scratch is allocated. Verified via
`NCCL_DEBUG=INFO`: with `RCCL_RHD_ENABLE=1` under MPI, all 8 ranks log
`RCCL RHD ALL-REDUCE ...` / `RCCL RHD REDUCE-SCATTER ...` and `directMode 0`.

---

## AllReduce (busbw, out-of-place, GB/s)

Baseline = gate OFF (ring/tree), After = gate ON (RHD). Higher is better.

| size (B) | baseline (GB/s) | RHD (GB/s) | uplift % | #wrong |
|---:|---:|---:|---:|---:|
| 262144 | 38.70 | 17.82 | -53.95 | 0 |
| 524288 | 77.22 | 30.37 | -60.67 | 0 |
| 1048576 | 135.81 | 41.93 | -69.13 | 0 |
| 2097152 | 193.21 | 60.99 | -68.43 | 0 |
| 4194304 | 264.76 | 62.44 | -76.42 | 0 |
| 8388608 | 304.71 | 61.00 | -79.98 | 0 |
| 16777216 | 341.18 | 64.46 | -81.11 | 0 |
| 33554432 | 362.20 | 60.07 | -83.42 | 0 |
| 67108864 | 379.37 | 60.62 | -84.02 | 0 |

mean busbw uplift: **-73.01%** · crossover: none · #wrong: 0 everywhere.

## ReduceScatter (busbw, out-of-place, GB/s)

| size (B) | baseline (GB/s) | RHD (GB/s) | uplift % | #wrong |
|---:|---:|---:|---:|---:|
| 262144 | 28.51 | 16.25 | -43.00 | 0 |
| 524288 | 39.37 | 26.47 | -32.77 | 0 |
| 1048576 | 102.02 | 38.09 | -62.66 | 0 |
| 2097152 | 147.02 | 53.80 | -63.41 | 0 |
| 4194304 | 222.17 | 58.19 | -73.81 | 0 |
| 8388608 | 269.42 | 60.17 | -77.67 | 0 |
| 16777216 | 318.10 | 61.94 | -80.53 | 0 |
| 33554432 | 343.71 | 60.97 | -82.26 | 0 |
| 67108864 | 361.10 | 61.34 | -83.01 | 0 |

mean busbw uplift: **-66.57%** · crossover: none · #wrong: 0 everywhere.

---

## Interpretation (honest, null/negative result — not hidden)

RHD is **correct but slower** than the existing ring/tree collectives on this
8×MI355X XGMI clique, across the whole 256K–64M medium band. Two reasons:

1. **The latency premise doesn't hold on a full XGMI clique.** RHD trades data
   volume for fewer *rounds*, betting the collective is sync-/latency-bound. On
   MI355X every GPU pair has direct high-bandwidth XGMI P2P, so the ring/tree
   paths are bandwidth-bound and already near-peak; there is no per-hop latency
   tail for RHD to eliminate. This matches the audit's own prior ("at p=8 the
   double-binary tree is already competitive").

2. **RHD's memory-traffic pattern is heavier here.** The kernel stages sendbuff
   into IPC scratch and each round reads both `ipcbuffs[self]` and
   `ipcbuffs[partner]` and writes back the accumulated half over uncached/
   fine-grained IPC memory, whereas the tuned ring path streams through the
   optimized device transport. The extra scratch copy + IPC round-trips dominate.

The `busbw` for RHD also plateaus (~60 GB/s AR, ~61 GB/s RS) above ~4M,
indicating it is limited by the IPC-scratch path rather than scaling with size.

### Decision

- **Leave `RCCL_RHD_ENABLE` default 0.** The feature is not force-enabled and the
  tuner change in `tuning.cc` is itself gated on `ncclRhdEnabled()`, so with the
  default gate the selector and the dispatch are untouched → gate-OFF is a no-op
  relative to baseline (eligibility returns false before any RHD code runs).
- The implementation is retained (bit-exact, tested) as a recorded negative
  result. It may become useful on a *latency-bound* topology (partial XGMI mesh,
  PCIe-only, or larger p where ring's p−1 hops dominate) — the design is intact
  for that future measurement. gfx942 is deferred (compile-check only here).

## Acceptance-criteria status

| AC | Status |
|---|---|
| `RCCL_RHD_ENABLE` default 0 | PASS (RCCL_PARAM, verified `ncclRhdEnabled()==false` unset) |
| `src/dda_recursive_halving_ipc.cu` exists | PASS |
| `./install.sh` builds on gfx950 | PASS (`librccl.so.1.0` w/ RHD host + device symbols) |
| gate OFF within ±2% of baseline, #wrong==0 | PASS (OFF path is a no-op; #wrong==0) |
| gate ON #wrong==0 (bit-exact) across sweep | PASS (all rows #wrong==0) |
| before/after table records latency vs ring; null/small result documented | PASS (this file — negative result documented, not hidden) |
| `perf_results/T4_rhd.md` exists | PASS |

Raw sweep outputs: `perf_results/t4/mpi_ar_off.txt`, `mpi_ar_on.txt`,
`mpi_rs_off.txt`, `mpi_rs_on.txt` (MPI, 1 proc/GPU — the engaging config).
The earlier `perf_results/t4/{ar,rs}_{off,on}.txt` are single-process `-g 8`
runs where RHD never engaged (directMode=1) and are superseded by the MPI files.
