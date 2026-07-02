# RCCL Single-Node Optimization — PEAS Agent Specification (Targets 1–4)

**What this is.** A PEAS (Performance measure · Environment · Actuators · Sensors) task specification for an autonomous coding agent ("dark factory") implementing the top single-node performance gaps from
[`RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md`](./RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md). Every target is gated behind an env var (default OFF) and **must be measured with rccl-tests before and after the change** (§6 is mandatory, not optional).

**Scope of "targets 1–4".** Per the audit that split original Rank 1 into 1a/1b, targets 1–4 are:

| Target | Gap (rank in the gaps doc) | Collective | Lossy? | Default posture |
|---|---|---|---|---|
| **T1a** | Inline block-quantized two-shot AllReduce — QuickReduce (Rank 1a) | AllReduce | **Yes** | off; never auto-selected on exact path |
| **T1b** | Quantized MoE AllToAll dispatch/combine — MoRI/RCCLX-LP (Rank 1b) | AllToAll | **Yes** | off |
| **T2** | CPX/XCD/IOD chiplet-hierarchy-aware algo+protocol selection (Rank 2) | AR/AG/RS | No | off initially |
| **T3** | Generalize DDA/symmetric AR past 8-rank/ncclSum/3-dtype + add LL128 (Rank 3) | AR/RS/AG | No | off |
| **T4** | Log-round recursive-halving/circulant RS/AllReduce (Rank 4) | RS, AR | No | off — **low priority** (audit: mostly an inter-node lever; at p=8 the double-binary tree already fills the valley — implement/measure but do not expect a large win) |

**Recommended implementation order (risk-adjusted, from the audit): T2 → T3 → T1a → T1b → T4.** T2 is lossless with the tuner scaffold already present; the lossy quantized paths (T1a/T1b) land after the safe wins and stay off by default.

---

## 1. PEAS summary

| Element | Definition |
|---|---|
| **Performance measure** | Per target: **rccl-tests bus-bandwidth (GB/s) and latency (µs) delta, gate-ON vs the pre-change baseline**, over the target's size sweep and rank counts; **zero regression** with the gate OFF; **correctness** (bit-exact for lossless T2–T4; bounded relative error for lossy T1a/T1b); clean build on gfx942 **and** gfx950. A target "passes" only with a recorded before/after table (§6) showing net improvement in its regime and no regression elsewhere. |
| **Environment** | RCCL source at `projects/rccl` (develop ≈ 2.30.4); one AMD node = 8× MI300X (gfx942/CDNA3) or 8× MI355X (gfx950/CDNA4), fully-connected XGMI clique (+ XCD/IOD chiplet & CPX partition modes). ROCm/HIP toolchain; rccl-tests at `projects/rccl-tests`. |
| **Actuators** | Edit RCCL source (device kernels, `src/include/algorithms/*`, `src/collectives.cc`, `src/graph/tuning.cc`, tuner CSVs, `src/device/generate.py`, enums), add `RCCL_PARAM` env gates, build `librccl.so`, build & run rccl-tests, run `tools/topo_expl`, run `rocprof`. |
| **Sensors** | rccl-tests stdout (algbw/busbw/latency per size + correctness `#wrong`), compiler/linker output, `tools/topo_expl` topology dump (XGMI clique / XCD enumeration), `rocprofiler-systems`/`rocprof` kernel traces, `git diff`. |

---

## 2. Performance measure (detail)

For each target, success is defined by **all** of:

1. **Impact (gate ON vs baseline).** In the target's stated regime, median **busbw uplift ≥ threshold** (or latency reduction), measured by the §6 before/after protocol. Per-target thresholds are in the task cards (§7). A target that does not beat its own baseline in its regime is a **negative result** — record it and stop, do not ship it enabled.
2. **No regression (gate OFF).** With the env gate OFF, the full sweep must match the pre-change baseline within noise (**±2%** busbw). This proves the change is inert when disabled.
3. **Correctness.**
   - Lossless targets (T2, T3, T4): rccl-tests reports `#wrong == 0` (bit-exact) across the sweep.
   - Lossy targets (T1a, T1b): quantized output must stay within a **bounded relative error** vs an fp32 reference reduction (default tolerance: mean-rel-err ≤ 2e-2 for Q4/FP4, ≤ 5e-3 for Q8/FP8; the agent records the measured error, does not silently relax the bound). Lossy paths must **never** be selected on the default exact-reduction path.
4. **Portability.** Builds and passes on **both** gfx942 and gfx950. Arch-specific code (e.g. FP4 on gfx950 only) is `#if`-guarded; the other arch falls back cleanly.
5. **Evidence.** A committed results block (§6 template) with the exact commands, ROCm version, GPU arch, and the before/after numbers.

---

## 3. Environment (detail)

- **Code:** `projects/rccl`. Key files by target are in §7. Cost model: `src/graph/tuning.cc:1148` (`*time = lat*latCount + nBytes/(1000*bw)`). Env-gate pattern: `RCCL_PARAM(Name, "ENV_NAME", default)` (see `src/collectives.cc:127-128`).
- **Hardware/arch:** detect via `comm->archName` / `IsArchMatch(..., "gfx942"|"gfx950")`. Native FP4 (`v_cvt_scalef32_pk_fp4_f16`) is **gfx950-only**; gfx942 uses INT4/6/8 + FP8 codecs. FP8 on gfx942 is FNUZ (E4M3FNUZ) — scale differs from gfx950/OCP FP8.
- **Build RCCL:** `cd projects/rccl && ./install.sh` (or the project's CMake flow); produces `librccl.so`. Rebuild after each change.
- **rccl-tests:** `cd projects/rccl-tests && ./install.sh` (links against the RCCL under test). Binaries: `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, `alltoall_perf`, `broadcast_perf`, `reduce_perf` (in the rccl-tests `build/`).
- **Topology check:** `tools/topo_expl` confirms the 8-GPU XGMI clique (and XCD/CPX enumeration for T2).

---

## 4. Actuators (what the agent may change)

- Add device kernels under `src/device/` and register instantiations via `src/device/generate.py` + `rccl_metadata.h`.
- Add algorithm headers under `src/include/algorithms/<collective>/`.
- Add dispatch/eligibility in `src/collectives.cc`; cost-model/selection in `src/graph/tuning.cc`.
- Add/extend tuner CSVs in `tuner/` and the CSV schema in `src/plugin/tuner/csv_tuner.cc`.
- Register `RCCL_PARAM` env gates (default 0/off).
- Build `librccl.so`, build/run rccl-tests, run `tools/topo_expl`, run `rocprof`.
- **Must not:** enable any target by default; alter the exact-`ncclSum` default path to route through a lossy kernel; touch inter-node/network paths; remove existing tests.

---

## 5. Sensors (how the agent observes)

- **rccl-tests stdout:** columns include size, count, type, redop, then out-of-place & in-place `time / algbw / busbw / #wrong`. **busbw** is the hardware-utilization metric to track; **#wrong** is the correctness sensor.
- **Build logs:** compile/link success per arch; treat warnings on new kernels as signals.
- **topo_expl:** confirms clique/XCD topology so a measured change isn't confounded by mis-detected topology.
- **rocprof / rocprofiler-systems:** per-kernel time and XGMI activity to attribute a delta to the new kernel.
- **git diff:** scope check — a target's diff should touch only its declared files + its env gate.

---

## 6. MANDATORY benchmark protocol — rccl-tests BEFORE and AFTER (this is the deliverable's core)

Run this for **every** target. Nothing ships without a completed before/after table.

### 6.1 Capture the BASELINE (before any code change)
```bash
# 0. Clean tree at the target's start point; build RCCL + rccl-tests as-is.
cd projects/rccl        && ./install.sh
cd ../rccl-tests        && ./install.sh
# 1. Record environment
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null); arch: $(rocminfo | grep -m1 gfx)"
# 2. Run the target's sweep (example = AllReduce, 8 GPUs, fp16 & bf16), save to baseline files
for dt in half bfloat16; do
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -d $dt -c 1 | tee baseline_allreduce_${dt}.txt
done
```
`-c 1` keeps correctness checking on. Use the **collective and size range of the target** (task cards give the exact sweep and rank counts, e.g. `-g 2/4/8` for T3, `alltoall_perf` for T1b, `-b 4K -e 4M` small-message band for T2).

### 6.2 Implement the change behind its env gate, rebuild.

### 6.3 Capture AFTER — two runs: gate OFF (regression) then gate ON (impact)
```bash
cd projects/rccl && ./install.sh && cd ../rccl-tests && ./install.sh   # rebuild with the change
# (a) gate OFF -> must match baseline within +/-2% busbw, #wrong==0
for dt in half bfloat16; do
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -d $dt -c 1 | tee after_gateOFF_allreduce_${dt}.txt
done
# (b) gate ON -> the impact measurement (example gate for T1a)
for dt in half bfloat16; do
  RCCL_QUICKREDUCE_ENABLE=1 RCCL_QUICKREDUCE_CODEC=q4 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -d $dt -c 1 | tee after_gateON_q4_allreduce_${dt}.txt
done
```

### 6.4 Compute deltas and record (commit this block with the change)
```
Target: T1a (QuickReduce Q4 AllReduce)   Arch: gfx950 MI355X x8   ROCm: <ver>
Sweep: all_reduce_perf -b 8 -e 8G -f 2 -g 8 -d half

 size     baseline busbw   gateOFF busbw   gateON busbw   uplift   #wrong(ON)   rel-err
 1 MiB       X.X GB/s         X.X (±)         Y.Y GB/s     +Z%         0*         <e>
 16 MiB      ...              ...             ...          ...         ...        ...
 1 GiB       ...              ...             ...          +Z%         ...        ...
Crossover (gateON beats baseline) at: ~N MiB
Regression (gateOFF vs baseline): max |Δ| = __% (must be ≤2%)
Verdict: PASS/FAIL  (+ one-line reason)
```
`* #wrong` is expected non-zero for lossy targets under rccl-tests' exact checker — for T1a/T1b replace the `#wrong` gate with the **rel-err vs fp32 reference** from §2.3 and note it here.

### 6.5 Repeat on the other arch (gfx942 ↔ gfx950). A target passes only when both arches are recorded.

---

## 7. Target task cards

Each card lists the objective, files to touch (verified in the gaps doc §3), the env gate, the §6 sweep to use, and the pass threshold. **Do §6.1 baseline capture before writing code.**

### T1a — Inline block-quantized two-shot AllReduce (QuickReduce)  ·  lossy
- **Objective:** two-shot RS→reduce→AG where XGMI transfers carry block-32 quantized data; codecs Q4/Q6/Q8/FP8 (gfx942) + native FP4 (gfx950). Relax DDA's exactly-8-rank gate to 2/4/8.
- **Files:** new `src/include/algorithms/all_reduce/quick_reduce_codec.h`, new kernel `src/dda_quick_reduce_ipc.cu` + `.../all_reduce/quick_reduce_dda.h` (model on `all_reduce_dda.h:56 ddaAllReduceTreeIpc`); reuse `src/include/ipc_gpu_barrier.h`, `CollCommon.h`; dispatch in `src/collectives.cc` near the DDA call (`:440-442`); FP4 builtin co-located with FP8 in `src/include/rccl_float8.h`; cost model `src/graph/tuning.cc:1148` scales effective `nBytes` by compression ratio.
- **Env gate:** `RCCL_QUICKREDUCE_ENABLE=0`, `RCCL_QUICKREDUCE_CODEC={q4,q6,q8,fp8,fp4}`, `RCCL_QUICKREDUCE_MIN_BYTES≈1MiB`.
- **Sweep:** `all_reduce_perf -b 8 -e 8G -f 2 -g 8 -d half` and `-d bfloat16`; also `-g 2` and `-g 4`.
- **Pass:** gate-ON busbw ≥ **1.5×** baseline above the ~1 MiB crossover on gfx950 FP4 (record the crossover); gate-OFF within ±2%; **rel-err ≤ 2e-2 (Q4/FP4)**. Lossy → stays off the exact default path.

### T1b — Quantized MoE AllToAll dispatch/combine  ·  lossy  ·  separate kernel
- **Objective:** quantized intra-node AllToAll (permute/shuffle, **no reduction**), per-token scales, asymmetric precision (MXFP4 dispatch / FP8 combine). Do **not** reuse the T1a RS/AG/reduce machinery.
- **Files:** new kernel under `src/device/` + `src/include/algorithms/alltoall/`; dispatch in `src/collectives.cc` alongside the existing Pivot A2A path; codec shared with T1a where symmetric, extended for per-token scales.
- **Env gate:** `RCCL_QUANT_ALLTOALL_ENABLE=0`, codec select.
- **Sweep:** `alltoall_perf -b 8K -e 1G -f 2 -g 8 -d half` (MoE token-sized messages).
- **Pass:** gate-ON busbw uplift in the MoE message band; gate-OFF ±2%; rel-err within bound; **no** dependence on `ncclSum` associativity.

### T2 — CPX/XCD/IOD chiplet-hierarchy-aware selection  ·  lossless  ·  RECOMMENDED LEAD
- **Objective:** detect SPX vs CPX and XCD/IOD grouping; pick algo+protocol per message-size zone (small→Tree+LL, mid→Tree+LL128, large→Ring+Simple).
- **Files:** partition/XCD detector in `src/graph/topo.*` (consume `s_partition_id` already parsed at `src/misc/alt_rsmi.cc:133,151`); extend CSV schema `src/plugin/tuner/csv_tuner.cc:44` with `partitionMode,localGroup` (backward-compatible) + match loop `:587-602`; ship `tuner/rccl_tuner_gfx942.csv` and add CPX rows to `tuner/rccl_tuner_gfx950.csv`; optional IOD-hop term in `src/graph/tuning.cc:1148`.
- **Env gate:** `RCCL_CHIPLET_TUNER_ENABLE=0`.
- **Sweep:** small-message band `all_reduce_perf -b 4K -e 4M -f 2 -g 1` under `mpirun -np 64 --bind-to numa` (CPX) and `-g 8` (SPX).
- **Pass:** **≥2×** small-message latency reduction vs default in CPX; `#wrong==0`; gate-OFF ±2%.

### T3 — Generalize DDA/symmetric AR + add symmetric LL128  ·  lossless
- **Objective:** add LL128 to the symmetric protocol axis (`rcclSymkProto` currently `_LL`/`_Simple` only, `src/sym_kernels.cc:152`); relax DDA/symmetric AR eligibility beyond 8 ranks (`dda_all_reduce_ipc.cu:136 kDdaNranks`) to 2/4/8 and beyond ncclSum-only for associative ops on the LD/ST path.
- **Files:** `src/sym_kernels.cc` (proto axis + tuning entries), `src/device/symmetric/generate.py` + `kernel.cuh` (LL128 variant), `CollCommon.h` (op generalization), `dda_all_reduce_ipc.cu` (rank gate).
- **Env gate:** `RCCL_SYM_LL128_ENABLE=0`, `RCCL_DDA_NRANKS_RELAX=0`.
- **Sweep:** `all_reduce_perf`/`reduce_scatter_perf -b 1M -e 256M -f 2 -g 4` and `-g 2`.
- **Pass:** LL128 selected and **≥ ~95% of peak** busbw in the large band; **8-rank ncclSum path bit-identical to baseline** (regression); `#wrong==0`. **Caveat:** verify 128 B atomic writes on XGMI for gfx942/gfx950 before enabling.

### T4 — Log-round recursive-halving/circulant RS/AllReduce  ·  lossless  ·  LOW PRIORITY
- **Objective:** ⌈log₂p⌉-round RS (p=8 → 3 rounds vs ring's 7) at p−1 volume; AR = RS + reversed AG. New RCCL-local algo id (extend the `RCCL_DIRECT_ALLGATHER = NCCL_NUM_ALGORITHMS` pattern in `src/include/rccl_common.h:63`).
- **Files:** new `src/dda_recursive_halving_ipc.cu` + `.../reduce_scatter/rhd_dda.h`; dispatch `src/collectives.cc` (RS `:576`, AR `:440`); cost model latCount term `src/graph/tuning.cc:1147`.
- **Env gate:** `RCCL_RHD_ENABLE=0`.
- **Sweep:** `reduce_scatter_perf`/`all_reduce_perf -b 256K -e 64M -f 2 -g 8`.
- **Pass:** lower latency than ring in the **sync-bound medium band**; `#wrong==0`. **Honest expectation:** the audit rates single-node value modest (double-binary tree already competitive at p=8) — a flat or small result here is an acceptable, recordable outcome; do not force-enable.

---

## 8. Definition of done (global gates)

A target is **done** when: (1) code compiles and links on gfx942 **and** gfx950; (2) the §6 before/after table is committed showing net improvement in-regime and ≤2% gate-OFF regression; (3) correctness gate met (bit-exact lossless / bounded rel-err lossy); (4) the diff touches only the declared files + its env gate; (5) the target remains **off by default** and lossy paths never intercept the exact-reduction path. All four+ targets done = a summary table comparing baseline vs each gate-ON target across the shared AllReduce sweep, committed alongside the code.

*Prepared as the execution spec for targets 1–4 of `RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md`. Env-gate names are proposals from the gaps doc; the agent may rename but must keep them default-off. rccl-tests before/after (§6) is required for every target.*
