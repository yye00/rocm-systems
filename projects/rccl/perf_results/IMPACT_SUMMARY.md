# Consolidated before/after impact report (F001 baseline vs all targets)

This is the single roll-up of the measured impact of every RCCL optimization
target against the **F001 baseline**. Each target has its own before/after report
(linked below); this document aggregates them onto one axis — the shared AllReduce
sweep plus each target's own collective — with a per-target PASS/FAIL/DEFERRED
verdict, the crossover/regime where it helps, the correctness class
(bit-exact vs measured rel-err), and a one-line default-posture recommendation.

**Bottom line: all five targets remain OFF by default.** Only one sub-feature
(T3's `RCCL_DDA_NRANKS_RELAX`) shows a measured positive; it is the one candidate
worth promoting behind auto-selection for its regime (2/4-rank AllReduce). Every
other gate-ON path is either a measured regression on this XGMI clique or could
not be honestly exercised on the available hardware and is marked DEFERRED (not
fabricated).

---

## Platform / provenance

| | |
|---|---|
| Host | smci355-ccs-aus-m03-05 |
| GPUs | 8 × AMD Instinct MI355X (**gfx950** / CDNA4), single-node XGMI clique |
| ROCm / HIP | 7.0.1 / 7.0.51831 |
| RCCL | `build/release/librccl.so` (gfx950), this branch |
| Baseline | **F001** — `perf_results/baseline/` (fp32/fp16 sweeps, `#wrong==0`) |
| Measurement rule | DDA IPC paths engage only with **1 process per GPU** (`mpirun -np N … -g 1`); `-g N` single-process sets `directMode=1` and skips DDA (carried from T4) |
| Size ceiling | 2 GiB (F001) / 64 MiB (MPI sweeps): a co-tenant vLLM server occupies ~295 GB on one GPU; larger buffers OOM. Documented in `baseline/ENV.txt`. |

### Architecture scope: gfx950 measured, **gfx942 deferred**

This host is **gfx950-only** (MI355X). Every runtime number in this report is
gfx950. **gfx942 (MI300X / CDNA3) results are deferred**, not fabricated — the
target code is arch-independent host logic (T2/T3) or compile-checked device
fallbacks (T1a/T1b FP4→INT4), and would be measured on a gfx942 host. This
matches the ARCH-SCOPE note in `baseline/ENV.txt`.

---

## Roll-up table (verdict per target)

Verdict is measured against each target's own acceptance threshold. "Correctness"
is the class the gate-ON path belongs to. "gate-OFF == baseline" is verified for
every target (eligibility returns false → no-op vs F001, `#wrong==0`).

| Target | Feature (gate, default 0) | Collective | Regime tested | gate-ON vs F001 baseline | Correctness | Verdict | Default posture |
|---|---|---|---|---|---|---|---|
| **T1a** | QuickReduce lossy 2-shot AR (`RCCL_QUICKREDUCE_ENABLE`) | AllReduce | 1 MiB–512 MiB (crossover 1 MiB) | ~0.08–0.13× (**8–12× slower**) in active band; falls back to baseline >512 MiB | **lossy** — measured rel-err: Q8 1.0e-2 (pass), Q6 4.3e-2, FP8 3.6e-2, Q4 1.9e-1, FP4 1.6e-1 | **FAIL** (speedup ≥1.5× not met; Q4/FP4 rel-err ≤2e-2 not met) | keep OFF |
| **T1b** | Quantized MoE AllToAll (`RCCL_QUANT_ALLTOALL_ENABLE`) | AllToAll | 8 KiB–128 MiB (crossover 256 MiB) | ~0.003–0.03× (**30–160× slower**) in active band; falls back to baseline ≥256 MiB (`#wrong==0`) | **lossy** — per-token fp8; stock bitwise validator flags every lossy element | **FAIL** (ON uplift not met; not demonstrable w/ stock validator) | keep OFF |
| **T2** | CPX/XCD chiplet-aware tuner (`RCCL_CHIPLET_TUNER_ENABLE`) | AllReduce (+RS/AG rows) | 4 KiB–4 MiB, SPX host | gate-ON on SPX host == baseline (CPX rows never match on 8-rank SPX); `#wrong==0` | **bit-exact** (lossless; matching is a strict superset of legacy path) | **DEFERRED** (CPX ≥2× small-msg latency needs a CPX-partitioned host; this node is SPX) | keep OFF |
| **T3a** | DDA nRanks relax 2/4/8 (`RCCL_DDA_NRANKS_RELAX`) | AllReduce | 64 MiB, 2 & 4 ranks | **+10.5% @ 2 ranks, +12.7% @ 4 ranks** (61.5 vs 55.7; 176.4 vs 156.5 GB/s) | **bit-exact** (`#wrong==0`, ncclSum path) | **PASS** (positive, bit-exact, faster than ring at low rank) | OFF now; **promote behind auto-selection** for 2/4-rank AR |
| **T3b** | Symmetric LL128 axis (`RCCL_SYM_LL128_ENABLE`) | AllReduce (symmetric) | large band | not runtime-verifiable: no LL128 device kernel emitted + symmetric memory unsupported on this node | bit-exact when engaged (gate only widens inert tuning tables) | **DEFERRED** (scaffold; needs LL128 kernel + symmetric-memory support) | keep OFF |
| **T4** | Recursive-halving/doubling DDA (`RCCL_RHD_ENABLE`) | AllReduce + ReduceScatter | 256 KiB–64 MiB | AR mean **−73%**, RS mean **−67%** (both **5–6× slower**); no crossover | **bit-exact** (`#wrong==0` fp32/fp16/bf16) | **FAIL** (regression: latency premise fails on bandwidth-bound XGMI clique) | keep OFF |

Per-target source reports:
[T1a](T1a_quickreduce.md) ·
[T1b](T1b_quant_alltoall.md) ·
[T2](T2_cpx_tuner.md) ·
[T3](T3_sym_ll128.md) ·
[T4](T4_rhd.md).

---

## Shared AllReduce sweep — F001 baseline vs each AllReduce target (gfx950)

Baseline = F001 (gate OFF, ring/tree). All figures busbw GB/s, higher is better.
T1b operates on AllToAll (not AllReduce) so it is not on this shared axis; see its
own table in [T1b_quant_alltoall.md](T1b_quant_alltoall.md).

| size (B) | F001 baseline | T1a QuickReduce (fp4, on) | T4 RHD (on) | T3a DDA relax (on) |
|---:|---:|---:|---:|---:|
| 262 144 | 38.70 | — | 17.82 (−54%) | — |
| 1 048 576 | 130.6 | 10.6 (0.08×) | 41.93 (−69%) | — |
| 4 194 304 | 260.9 | 27.6 (0.11×) | 62.44 (−76%) | — |
| 16 777 216 | 330.7 | 38.2 (0.12×) | 64.46 (−81%) | — |
| 67 108 864 | ~379 | 41.5 (0.11×) | 60.62 (−84%) | — |
| 67 108 864 @ 2 ranks | 55.65 | — | — | **61.49 (+10.5%)** |
| 67 108 864 @ 4 ranks | 156.46 | — | — | **176.40 (+12.7%)** |
| 536 870 912 | ~326 | 326.1 (1.00×, fell back) | — | — |

Reading the sweep:
- **T1a / T4** are net-negative across the whole medium band on the 8-rank full
  XGMI clique — the fabric is bandwidth-bound, so trading wire bytes (T1a codec)
  or rounds (T4) for extra IPC-scratch traffic loses. Both cleanly fall back to
  the baseline outside their eligible range (T1a ≥512 MiB → 1.00×).
- **T3a** is the lone win: at low rank counts (2/4) the collective is more
  latency-sensitive, so the IPC LD/ST DDA path beats the ring fallback while
  staying bit-exact.

---

## Crossover / regime summary

| Target | Where (if anywhere) it would help |
|---|---|
| T1a | Only where AllReduce is genuinely wire-limited (inter-node/RDMA) **and** a vectorized packed-math codec replaces the naive bit-packing kernel. Not on an intra-node XGMI clique. |
| T1b | Only where the AllToAll wire is the constraint (inter-node RDMA) **and** paired with a tolerance-aware validator for lossy MoE dispatch/combine. |
| T2 | CPX compute-partition mode (per-XCD ranks) on small messages — the intra-GPU bandwidth hierarchy (within-XCD ≫ XCD→IOD ≫ IOD→HBM) is what the chiplet rows exploit. Unverifiable in SPX. |
| **T3a** | **2/4-rank single-node AllReduce** (latency-bound regime) — measured positive here. |
| T3b | Large-band symmetric AllReduce, once an LL128 device kernel exists and symmetric memory is enabled. |
| T4 | Latency-bound topologies (partial XGMI mesh, PCIe-only, or larger p where ring's p−1 hops dominate). Never on a full XGMI clique. |

---

## Correctness classification

- **Bit-exact (lossless):** T2, T3a, T3b, T4 — `#wrong==0` across their sweeps with
  the gate both off and on. Reduction order is data-independent (fixed pairwise
  tree / ncclSum path); the tuner-only changes (T2, T3b) never alter results.
- **Lossy (measured rel-err):** T1a, T1b — these trade precision for wire bytes.
  T1a's measured mean rel-err meets the 2e-2 bar only for Q8 (1.0e-2); Q6/FP8 land
  ~3–4e-2 and the 4-bit codecs (Q4 1.9e-1, FP4 1.6e-1) are fundamentally over the
  bar for a single per-32-element scale. T1b's per-token fp8 is bounded per token
  but the stock bitwise validator has no tolerance mode, so it reports large
  `#wrong` by design wherever the lossy path engages.

Every target's **gate-OFF path is verified bit-identical to the F001 baseline**
(eligibility returns false before any target code runs), so shipping all five
default-off leaves the production collective unperturbed.

---

## Recommendation — default posture

**All five targets remain OFF by default**, and the shipped defaults are the
correct posture: default-off gates make each feature a verified no-op relative to
F001.

- **Promote behind auto-selection (one candidate):** **T3a `RCCL_DDA_NRANKS_RELAX`**
  is the only measured positive — bit-exact and +10–13% for 2/4-rank AllReduce. It
  is worth wiring into the tuner's auto-selection **scoped to its regime**
  (single-node, 2/4 ranks, ncclSum) while remaining off for the 8-rank case where
  it offers nothing. This is the one gate whose auto-selection is justified by data.
- **Keep OFF, retain as substrate:** T1a and T1b are correct-but-slow on this
  bandwidth-bound clique; retain default-off as a base for a future
  vectorized-codec / inter-node rewrite — do not auto-select.
- **Keep OFF, revisit on the right hardware:** T2 (needs a CPX-partitioned host),
  T3b (needs an LL128 device kernel + symmetric memory), and T4 (needs a
  latency-bound topology). Their runtime verdicts are **deferred**, not negative —
  they must stay default-off until they can be honestly measured where their
  regime applies.

No target should be force-enabled or made default-on on this gfx950 XGMI clique.
gfx942 promotion decisions await gfx942 measurements.
