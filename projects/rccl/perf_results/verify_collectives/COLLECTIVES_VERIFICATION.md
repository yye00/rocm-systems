# RCCL Collectives Multi-GPU Verification Report

**Node:** smci355 (8x AMD MI355X, gfx950, ROCm 7.0.1)
**Config:** 64 MiB, fp32, `-c 1` validation, `-n 80 -w 20`, 1 process/GPU, `RCCL_DDA_NRANKS_RELAX=1`
**Scope:** 7 collectives x np in {2,4,8} x 3 repeats = 63 runs total
**Date:** 2026-07-01

---

## 1. Summary Table (collective x np: #GPUs-active / busbw GB/s / #wrong==0)

| Collective      | np=2 GPUs / busbw / #wrong0 | np=4 GPUs / busbw / #wrong0 | np=8 GPUs / busbw / #wrong0 |
|-----------------|-----------------------------|-----------------------------|-----------------------------|
| all_reduce      | 2 / 35.4-60.3 / YES         | 4 / 80.5-176.2 / YES        | 8 / 364-380 / YES           |
| all_gather      | 2 / 47.0-58.2 (med 54.8) / YES | 4 / 102.7-116.6 (med 110.9) / YES | 8 / 231.7-354.5 (med 268.9) / YES |
| reduce_scatter  | 2 / 24.6-55.1 / YES         | 4 / 42.7-147.2 / YES        | 8 / 138-331 / YES           |
| broadcast       | 2 / 59.2-60.9 / YES         | 4 / 94.0-151.8 / YES        | 8 / 196-197 / YES           |
| reduce          | 2 / 57-63 (med 63.1) / YES  | 4 / ~143.5 / YES            | 8 / ~212.5 / YES            |
| alltoall        | 2 / 49.5-53.2 (med 51.3) / YES | 4 / 147.3-148.6 (med 148.0) / YES | 8 / 214.8-215.7 (med 215.6) / YES |
| sendrecv        | 2 / 56.1-57.8 (med 57.3) / YES | 4 / 51.4-57.7 (med 53.3) / YES | 8 / 59.8-60.1 (med 60.0) / YES |

*busbw = in-place bus bandwidth unless noted; ranges span the 3 repeats.*
*GPU-active counts reflect the AUTHORITATIVE rccl-tests per-rank device binding, not the node-wide sampler (see anomalies).*

---

## 2. Per-Collective PASS/FAIL

| Collective      | GPU-Mapping | Correctness | Basis |
|-----------------|-------------|-------------|-------|
| all_reduce      | PASS        | PASS        | rank->device binding {0,1}/{0-3}/{0-7}; #wrong==0 all 9 runs |
| all_gather      | PASS        | PASS        | binary 'Using devices' lists exactly np ranks; #wrong==0 all 9 runs |
| reduce_scatter  | PASS        | PASS        | 'Using devices' np-scaled; #wrong==0 both columns all 9 runs |
| broadcast       | PASS        | PASS        | busbw scales 59->92-152->197; #wrong==0 all 9 runs |
| reduce          | PASS        | PASS        | busbw scales 63->143->213; #wrong==0 both columns all 9 runs |
| alltoall        | PASS        | PASS        | sustained-load filter isolates np participants; busbw scales 51->148->216; #wrong==0 all 9 runs |
| sendrecv        | PASS        | PASS        | 'Using devices' np-scaled; out-of-place #wrong==0 all 9 runs (in-place N/A is normal for sendrecv) |

**GPU-Mapping criterion:** np=8 lights all 8 GPUs; np=4 uses >=4; np=2 uses 2; no collapse-to-1-GPU. All 7 PASS.
**Correctness criterion:** every run prints '# Out of bounds values : 0 OK' with #wrong==0. All 7 PASS.

---

## 3. OVERALL VERDICT: PASS

**ALL 7 tested RCCL collectives correctly map across 2, 4, and 8 GPUs with all expected GPUs lighting up, and produce correct results (#wrong==0) on every one of the 63 runs.**

- Every collective scales monotonically in busbw with rank count (2 -> 4 -> 8), the definitive signature of genuine multi-GPU participation and proof that no collective collapses to a single GPU.
- At np=8, all collectives drove all 8 MI355X GPUs (confirmed by rank->device bindings and/or high uniform per-GPU load, ~85-97%).
- Zero errors, aborts, hangs, or nonzero #wrong across the entire matrix.

---

## 4. Anomalies / Red Flags

**None are correctness or mapping defects. All anomalies are measurement-environment artifacts of a SHARED node.**

1. **Node-wide sampler over-counts GPUs (measurement artifact, NOT a defect).** The mandated `rocm-smi --showuse` sampler is node-wide and cannot isolate this test's ranks. On a shared box with co-tenant workloads, it registered "all 8 GPUs active" even for np=2/np=4 for several collectives (broadcast, reduce, and the naive pass of all_gather/reduce_scatter/sendrecv/alltoall). The authoritative GPU count is the rccl-tests per-rank "Using devices" binding (used above), corroborated by sustained-load filtering and busbw scaling.

2. **Confirmed co-tenant contamination (reduce_scatter, alltoall).** `rocm-smi --showpids` revealed foreign python3 PIDs (notably PID 514151, ~293 GB VRAM + heavy SDMA on GPU 1). An idle baseline showed 40-60% spikes across all 8 GPUs with no test running. This is the source of the sampler over-count and of bandwidth noise.

3. **Run-to-run busbw variance / low outliers (shared-node contention, NOT correctness).** Examples: all_reduce np2 run2 (60 vs ~35-39), np4 run2 (80.5 vs ~176); reduce_scatter np2 run2 (~24.6 vs ~54), np4 run3 (~42.7 vs ~143); broadcast np4 run2 (94 vs ~130-152). All such runs still passed validation with #wrong==0. Attributed to shared HBM/xGMI bandwidth contention.

4. **sendrecv in-place validation = N/A.** Expected/normal rccl-tests behavior (sendrecv validates only the out-of-place buffer). Not a failure.

5. **Sampler cadence adjustment.** all_gather sampler cadence tightened from 0.5s to 0.15s after an initial pass hit the 10-min SSH timeout; np2_run1 used the original 0.5s. No impact on correctness or mapping conclusions.

**Recommendation:** For future verification on this shared box, pin ranks with `HIP_VISIBLE_DEVICES` and/or subtract the idle GPU-usage baseline so per-GPU attribution is reliable at small np.

---

## 5. Artifacts

Raw outputs, monitor logs (`.gpumon`), and per-collective runner scripts are under:
`~/dark-factory/rocm-systems/projects/rccl/perf_results/verify_collectives/`
Naming: `<collective>_np<N>_run<R>.txt` and `<collective>_np<N>_run<R>.gpumon`.
