# RCCL Collective-Algorithm Survey — Run 2026-07

**Scope:** `projects/rccl` at the `develop` tip (RCCL 2.30.x, an AMD/ROCm fork of NVIDIA NCCL).
**Method:** produced with the re-runnable survey funnel in [`tools/algo_survey/`](tools/algo_survey/README.md) — code-inventory fan-out → completeness gate → literature sweep → candidate-scoring gate → adversarial code cross-check → synthesis.
**Relationship to prior docs:** this is a *dated re-run*, not a replacement. It (1) re-verifies the algorithm inventory against the current tree and **corrects line anchors that drifted** since the earlier passes, (2) folds in **2025–2026 AMD-measured evidence** that materially sharpens three of the gaps, and (3) reconciles with the three existing reports rather than duplicating their detail:
> - `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` — the multi-node structural survey (Rabenseifner, hierarchical, Bruck/A2A, MSCCL++, Swing, fault-tolerance).
> - `RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md` — the intra-node ranked report (QuickReduce, chiplet-hierarchy, generalized DDA, log-round RS, …).
> - `COLLECTIVE_COMM_STATE_OF_THE_ART_2026.md` — the literature landscape briefing.

> This is an engineering analysis, not a code change to the shipping library. Every `file:line` below was re-read on this branch during this run.

---

## Part 0 — Executive summary

RCCL implements the standard NCCL algorithm set (Ring, double-binary Tree, CollNet Direct/Chain, NVLS / NVLS-Tree, PAT) plus AMD-specific intra-node fast paths (DDA IPC/fabric one-/two-shot for AllReduce/ReduceScatter/AllGather/AllToAll, Pivot AllToAll, Hierarchical AllGather, RocSHMEM GDA, and a GPU-initiated-networking / **GIN** transport). Algorithm selection is a latency + size/bandwidth cost model (`tuning.cc`) over a topology graph that explicitly models the PCIe/xGMI/NIC hierarchy and exposes NVLink/xGMI-domain structure to an external tuner plugin.

**Verification note — the source has not changed, only our line numbers.** No `src/` code changed between the survey base (`bece9cc9`) and this run; the prior passes' *findings* still hold, but their `file:line` anchors had drifted. The corrected anchors are in Part 1.

This run confirms the earlier gap set and **upgrades the confidence of three gaps to "demonstrated on AMD"** on the strength of 2025–2026 AMD-measured evidence:

| # | Gap (missing / under-exploited in RCCL) | Strongest AMD-measured evidence | Regime | Confidence (this run) |
|---|------------------------------------------|---------------------------------|--------|-----------------------|
| 1 | **Inline block-quantized AllReduce** (compress → reduce → decompress in flight) | **QuickReduce FP4/INT3 on MI300X/MI355** — up to **2.25× vs RCCL**, ~4–5× at TP=2 for 1 GB (AMD ROCm blog) | Intra-node, bandwidth-bound TP AllReduce | **High** (AMD's own lib, not integrated) |
| 2 | **Chiplet-hierarchy-aware algorithm/protocol selection** (XCD/CPX/IOD) | **AMD RCCL-Tuner blog** — tree-hierarchy cuts a 126-step Ring to **12 steps, 3.1×** over Ring+Simple on MI300X CPX | Intra-node MI300X, esp. CPX/partitioned mode | **High** (achievable via *existing* tuner API — under-exploited, not absent) |
| 3 | **Staged / aggregated inter-node AllToAll for MoE** (incast-aware; FP8) | **FLASH** 1.18–4.48× vs RCCL FanOut on 4×8 MI300X; **DeepEP** (ROCm fork) FP8 dispatch/combine with GIN overlap | Inter-node MoE / expert-parallel | **Medium-High** |
| 4 | **Explicit multi-level hierarchical AllReduce** (small inter-node phase) | **PCCL / "Big Send-off"** on 2,048 GCDs Frontier MI250X (~10× AR, 40–60% end-to-end); **HiCCL** 1.55× vs RCCL | Multi-node MI250X/MI300X | Medium-High |
| 5 | **Programmable execution-plan interpreter** (restore MSCCL++) — RCCL had it and **removed it** (`CHANGELOG.md:85`) | **MSCCL++** 3.8× small / 2.2× large AllReduce on MI300X (ASPLOS'25) | Delivery vehicle for 1–4 as data | Medium-High |
| 6 | **Bandwidth-optimal log-step AllReduce** (Rabenseifner RHD / Swing) | MPICH/NSDI'24 (topology-gated; competes with the double-binary tree) | Medium msgs, many ranks; rail/torus for Swing | Medium (benchmark candidate) |
| 7 | **Fault-tolerant / elastic collectives** | NCCLX FTAR, R2CCL, Mycroft (SOSP'25) — library-agnostic techniques to port | 10k–100k-GPU survivability | High strategic (at scale) |

The two items that jumped to **High** this run — **inline quantized AllReduce (#1)** and **chiplet-aware selection (#2)** — are the ones with AMD's *own* published measurements, and both are *intra-node* wins that need no new network fabric. #2 is special: the capability to exploit it **already exists** through the tuner plugin's `nvlDomainInfo`; the gap is that RCCL does not select chiplet-aware plans *automatically*.

---

## Part 1 — What algorithms are in RCCL today (re-verified anchors)

### 1.1 Algorithm / protocol enumeration — `src/include/plugin/nccl_tuner.h`

```c
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6
#define NCCL_NUM_ALGORITHMS NCCL_NUM_ALGORITHMS_V5   // 7
// protocols: NCCL_PROTO_LL 0 / LL128 1 / SIMPLE 2
// datatypes now include rccl_float8 / rccl_bfloat8 (FP8)   <-- relevant to gap #1
```

### 1.2 Core algorithms (NCCL-derived)

| Algorithm | Enum | Key anchor (this run) | Pattern | Collectives |
|-----------|------|-----------------------|---------|-------------|
| **Ring** | `RING=1` | `src/device/all_reduce.h:15` (`runRing`) | reduce-scatter → all-gather; `2(p-1)` steps | All |
| **Tree** (double-binary) | `TREE=0` | `src/device/all_reduce.h:106` (`runTreeUpDown`), `:172` (`runTreeSplit`) | reduce up / bcast down; split variant overlaps both directions | AllReduce |
| **CollNet Direct / Chain** | `2` / `3` | `src/device/all_reduce.h` (Direct/Chain runners) | in-network reduction through a SHARP-capable NIC/switch | AllReduce (+AG/RS for Direct) |
| **NVLS / NVLS-Tree** | `4` / `5` | `src/device/all_reduce.h` (NVLS runners) | NVLink-SHARP multimem — **NVIDIA-only, inert on AMD** | AllReduce/AG/RS |
| **PAT** | `6` | `src/device/all_gather.h`, `reduce_scatter.h` | recursive-doubling over aggregated trees, `log(p)`; upstream NCCL, 1 GPU/node | AllGather, ReduceScatter |

**Cost model** — `src/graph/tuning.cc:1627`:

```c
*time = lat * latCount + nBytes / (1000 * bw);
```

Tree gets a half-bandwidth modeling factor while Ring/NVLS scale by ranks/steps — `src/graph/tuning.cc:1324-1325`:

```c
if (a == NCCL_ALGO_RING || a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= (1.0 * nRanks) / nsteps;
else ratio *= .5;                                   // TREE / COLLNET*
```

`bw`/`lat` come from per-arch tables in `tuning.cc`, scaled by 27-bucket size-indexed correction factors. Topology graph: `src/graph/topo.cc`; paths `paths.cc`; ring/tree search `search.cc`; double-binary tree `trees.cc`; AMD pre-baked orderings `rome_models.cc`.

### 1.3 AMD-specific algorithms (beyond stock NCCL)

| Feature | Anchor | Mechanism | Gating (verified) |
|---------|--------|-----------|-------------------|
| **DDA** one-/two-shot (AR + RS/AG/AllToAll) | `src/dda_*_{ipc,fabric}.cu`, `src/include/algorithms/**`, `src/collectives.cc:175` | IPC/fabric peer pointers + GPU barrier; flat one-shot small, tree two-shot large | gfx942/gfx950, single node, **exactly 8 ranks** (`kDdaNranks`); `RCCL_PARAM(DdaThreshold, …, 64 MB)` (`collectives.cc:175`); `ncclSum` only for reducing collectives |
| **Pivot AllToAll** | `src/device/alltoall_pivot.h` | multiple **bidirectional ring pairs** on a specific Rome topology | env opt-in (`RCCL_ALL_TO_ALL_PIVOT_ENABLE=0` default); per-rank ≥ 744 KB |
| **Hierarchical AllGather** | `src/device/hierarchical_ag_shuffle.h` | inter-node AG → intra-node AG → local shuffle | `RCCL_HIERARCHICAL_ALLGATHER` (on) **and `nNodes ≥ 8`** |
| **RocSHMEM GDA AllToAll(v)** | `src/device/alltoall_gda.h`, `alltoallv_gda.h` | GPU-initiated one-sided A2A over RocSHMEM | `ENABLE_ROCSHMEM`, size ≤ threshold |
| **GIN (GPU-initiated networking)** | `src/gin/` (`gin_plugin_rocshmem_gda.cc`, `gin_plugin_anvil_sdma.cc`, `gin_host_proxy.cc`) | device-initiated network puts (SDMA / RocSHMEM-GDA backends) | plugin-gated; the AMD analog of NVSHMEM that a DeepEP-style MoE path would build on (gap #3) |

### 1.3a Paths outside the `NCCL_ALGO_*` enum

- **Symmetric-memory kernel family** — `src/include/sym_kernels.h:36` (`enum ncclSymkKernelId`) + `src/device/symmetric/`: AllReduce/AllGather/ReduceScatter via LL, LD/ST, **multicast (MC)**, **TMA**, **Rail-optimized** variants. MC/TMA depend on NVIDIA multimem/TMA and are **largely inert on AMD**.
- **Direct / one-shot ReduceScatter** — `src/device/reduce_scatter.h` (`enableDirectReduceScatter`): batched `reduceCopy` instead of the ring.
- **`oneRankReduce`** — `src/device/onerank.cu`: single-rank special case.
- **SendRecv (P2P) engine** — `src/device/sendrecv.h`: underpins naive cross-node AllToAll and all point-to-point.

### 1.4 Capability that was removed

`CHANGELOG.md:85`: *"Removed MSCCL and MSCCL++ collective integration; legacy `mscclLoadAlgo`, `mscclRunAlgo`, `mscclUnloadAlgo` APIs remain as no-ops."* Verified: **zero** `mscclpp` files in `src/`. Released RCCL (~2.26–2.28) shipped MSCCL++ dispatch on gfx942 gated by `RCCL_MSCCLPP_THRESHOLD`; this branch does not. This is why gap #5 is *restore a proven path*, not *build new*.

### 1.5 Chiplet / compute-partition awareness (what exists vs what doesn't)

RCCL is **topology-aware of CPX/chiplet mode** — `src/graph/xml.cc:445` overrides PCIe function IDs in CPX mode, `src/graph/xml.h:23` bumps `MAX_SUBS` to 512 for CPX, `src/graph/connect.cc:1205` sizes channel arrays for CPX at 64 ranks — and it groups GPUs into xGMI/NVLink domains (`src/graph/paths.cc:1385`, `nvlDomain[]`), exposing `nNvlDomains` to the tuner plugin (`src/include/plugin/tuner/tuner_v5.h:14`). **What is missing** is *automatic* selection of chiplet-hierarchy-optimal (algo, protocol) per collective/size in core; today that intelligence lives in an external tuner plugin (AMD's blog builds exactly this). See gap #2.

---

## Part 2 — Gap analysis (ranked; strengthened with 2025–2026 AMD evidence)

For each gap: what it is, where RCCL falls short (with anchor), the AMD-measured evidence, and the honest caveat that survived the adversarial cross-check.

### Gap 1 — Inline block-quantized AllReduce (compress → reduce → decompress in flight) · **High**

**What it is.** For bandwidth-bound tensor-parallel AllReduce, compress each chunk to FP4/FP6/FP8/INT-k *before* it crosses xGMI/PCIe, reduce in the compressed (or a cheap dequant→reduce→requant) domain, and decompress at the destination. Moves `¼–½` the bytes on the fabric that actually limits the collective.

**Where RCCL falls short.** RCCL can *reduce* FP8 buffers — `rccl_float8`/`rccl_bfloat8` are recognized floating-point types in `src/device/reduce_kernel.h:17-31` — but there is **no inline-compression collective**: a grep of `src/device/` for block-quant / dequant / codebook / compress-chunk logic returns nothing. Quantization, if any, happens in the framework *before* the RCCL call, so RCCL still moves full-width bytes.

**AMD-measured evidence.** **QuickReduce** — AMD's own high-performance all-reduce with inline compression — reports (ROCm blogs) **QR FP4 up to 2.25× vs RCCL** on 2×/4×MI300X, ~**4.14× at TP=2** for 1 GB, and **QR INT3 ~5× at TP=2**. It is a *separate library*, not an RCCL algorithm. **Flash Communication** (arXiv:2412.04964) makes the same case for low-bit one-shot AllReduce as a TP inference bottleneck-breaker.

**Cross-check caveat (survived refute).** The win is real but **regime-bounded**: it needs an accuracy-tolerant consumer (inference/TP AllReduce, not bitwise-exact training reductions) and shrinks at high TP where latency, not bandwidth, dominates (2.25× at TP=2 → 1.52× at TP=8 for FP4). Ship it behind an opt-in datatype/threshold gate, never as a silent default. Detailed integration plan: `RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md` Rank 1.

### Gap 2 — Chiplet-hierarchy-aware (XCD/CPX/IOD) algo+protocol selection · **High** (under-exploited, not absent)

**What it is.** MI300X is 8 XCDs across multiple IODs; in CPX / partitioned mode each XCD is a separate device. Treating the on-package hierarchy as a real communication level — reduce within an XCD/IOD first with a low-latency protocol, then cross IODs once — turns a long intra-node Ring into a shallow tree.

**Where RCCL falls short.** RCCL *sees* the hierarchy (`paths.cc:1385` domains; CPX topology handling `xml.cc:445` / `xml.h:23` / `connect.cc:1205`) and exposes `nNvlDomains` to the tuner, but its **default cost-model selection does not automatically pick chiplet-hierarchy-optimal (algo, protocol)** per size — it leans on Ring+Simple, which is 126 steps at 64 ranks.

**AMD-measured evidence.** AMD's ROCm blog *"Optimizing MI300X Inter-Chiplet Communication via the RCCL Tuner API"*: a tree-hierarchical plan cuts the **126-step Ring to 12 steps**, and with LL's low per-message latency yields **up to 3.1× over the default Ring+Simple** for MI300X inter-chiplet AllReduce — built entirely through the **existing tuner plugin API**.

**Cross-check caveat (survived refute).** This is the *cheapest* high-value item because the mechanism already ships: the residual work is to **promote chiplet-aware selection from an external tuner recipe into the default cost model** (or ship a default tuner profile), so users get it without hand-tuning. Detailed plan: `RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md` Rank 2.

### Gap 3 — Staged / aggregated inter-node AllToAll for MoE (incast-aware, FP8) · **Medium-High**

**What it is.** Aggregate every local GPU's partial destined for a remote node into one node-to-node message, exchange node-to-node, then scatter locally — collapsing `O(p²)` tiny network messages to `O(nodes²)` large ones and load-balancing incast on the fast intra-server fabric before crossing the slow inter-server links. Overlap dispatch with expert compute via GPU-initiated networking.

**Where RCCL falls short.** Pivot A2A (`alltoall_pivot.h`) and RocSHMEM GDA cover the intra-XGMI-clique case, but **cross-node AllToAll falls back to a naive per-peer send/recv loop** (`sendrecv.h` engine). MoE dispatch/combine — small/medium messages across many nodes, increasingly FP8 — is exactly the regime that hurts.

**AMD-measured evidence.** **FLASH** (arXiv:2505.09764), implemented on ROCm/RCCL/MSCCL and measured on **4 nodes × 8 MI300X**, beats **RCCL FanOut A2A by 1.18–4.48×** on Megatron-LM MoE via an incast-aware, Birkhoff-decomposed schedule computed in ~32 µs. **DeepEP** (with a **`ROCm/DeepEP` fork**) provides FP8 dispatch/combine kernels that overlap communication with compute via NVSHMEM/GIN — directly buildable on RCCL's existing **GIN** transport (`src/gin/`).

**Cross-check caveat (survived refute).** Plain single-level Bruck's intra-node multi-GPU benefit is muted (ICHPC-Asia'24); the win is the **inter-node aggregation + incast-awareness**, not Bruck per se. Layer onto Pivot/GDA/GIN rather than replacing them. Detailed plan: `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` §2.4/§3.4.

### Gap 4 — Explicit multi-level hierarchical AllReduce · **Medium-High**

Factor AllReduce so the slow inter-node phase touches only `1/∏Nᵢ` of the buffer; each level uses its fabric's best algorithm/protocol; pipeline by chunk across levels. RCCL's Ring *implicitly* does a one-level split and `hierarchical_ag_shuffle.h` is a hand-rolled 2-level AllGather, but there is **no general, composable, bandwidth-minimizing AllReduce decomposition** (the hierarchical split-comms are wired to AllGather only). Evidence: **PCCL / "Big Send-off"** (SC'25) on 2,048 GCDs of Frontier MI250X (~10× AR, 40–60% end-to-end GPT-3-scale); **HiCCL** (IPDPS'25) 1.55× vs RCCL. Caveat: gains over an *already-tuned* RCCL Ring are incremental and regime-specific — A/B before default-on. Detailed plan: `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` §2.2/§3.2.

### Gap 5 — Restore a programmable execution-plan interpreter (MSCCL++) · **Medium-High**

RCCL removed MSCCL/MSCCL++ (`CHANGELOG.md:85`); the tuner plugin can only *re-rank* built-in algorithms, not express a new pattern. Restoring the MSCCL++ executor (the no-op API stubs are a natural re-entry point) is the **delivery vehicle** that lets gaps 1/3/4/6 — and externally-synthesized TE-CCL/TACCL/ForestColl plans — ship as *data* rather than code. Evidence: **MSCCL++** 3.8× small / 2.2× large AllReduce on MI300X (ASPLOS'25, arXiv:2504.09014); **TE-CCL** 3.18× vs RCCL (SIGCOMM'24). Detailed plan: `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` §2.5/§3.5.

### Gap 6 — Bandwidth-optimal log-step AllReduce (Rabenseifner RHD / Swing) · **Medium (benchmark candidate)**

No recursive halving–doubling and no Swing anywhere in the tree. RHD fills a medium-message / many-rank valley at `2·log₂(p)` steps and `~2n` bytes; Swing (NSDI'24) targets bandwidth-limited rail/torus fabrics. **Honest caveat that survived cross-check:** RCCL's Tree is a genuine **double-binary tree** (`trees.cc`), so it already covers much of RHD's theoretical valley — RHD/Swing are *benchmark candidates*, adopted only if measured to beat the tuned tree in their regime; Swing is a no-op inside a fully-connected xGMI clique or on a non-blocking fat tree. Detailed plan: `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` §2.1/§2.6.

### Gap 7 — Fault-tolerant / elastic collectives · **High strategic (at scale)**

RCCL inherits NCCL's no-native-fault-tolerance limitation: a NIC-port-down or straggler stalls the whole communicator. Library-agnostic techniques exist to port — **NCCLX FTAR**, **R2CCL**, **OptCC**, **Mycroft** (SOSP'25). About job *survivability*, not raw speed; higher priority at 10k–100k GPUs than any bandwidth item, lower for single-node users. Detail: `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` §2.7.

### On the MVAPICH / D.K. Panda "bidirectional PCIe tree" direction (the motivating question)

Re-confirmed this run: **RCCL already implements the core of that line of work.** Each physical link *direction* is a separate `ncclTopoLink` with its own independent `bw` budget (`topo.cc`), so PCIe/xGMI is modeled **full-duplex** by default; `followPath` (`search.cc`) charges reverse bandwidth **only** for hardware whose directions are physically coupled (pre-Ampere NVSwitch, POWER9 NVLink). Both directions are already driven at the schedule level via multi-channel mirrored rings + the double-binary tree. The **only** genuine residual — *reduce-within-PCIe-switch-first* staging on **PCIe-only (non-xGMI)** boxes — is best pursued as a PCIe-switch *level* inside the hierarchical decomposition (Gap 4), **not** as a `search.cc` change. On mainstream xGMI clusters there is essentially nothing to add here. (This corrects an earlier draft's "cost-model defect" framing, which was wrong.)

---

## Part 3 — Recommended sequencing (this run)

Ordered by (AMD-measured confidence × tractability), intra-node wins first:

1. **Gap 2 — chiplet-aware selection.** Cheapest: mechanism already ships via the tuner (`nvlDomainInfo`); promote it into the default cost model / a default tuner profile. AMD-measured 3.1×. Ship a default MI300X CPX profile.
2. **Gap 1 — inline quantized AllReduce.** AMD's own QuickReduce shows 2.25× vs RCCL; integrate as an opt-in datatype/threshold-gated AllReduce algorithm. Accuracy-gated, inference-first.
3. **Gap 3 — staged inter-node MoE AllToAll.** Build on GIN; borrow DeepEP's dispatch/combine + FLASH's incast-aware schedule. Clear MoE win, independent of the AllReduce work.
4. **Gap 4 — hierarchical AllReduce.** Highest multi-node payoff; A/B vs tuned Ring/Tree before default-on. Reuses `hierarchical_ag_shuffle.h` for the AG half.
5. **Gap 5 — restore MSCCL++ interpreter.** Force-multiplier: delivers 1/3/4/6 and synthesized plans as data.
6. **Gap 6 — Rabenseifner/Swing.** Prototype as benchmark candidates; adopt only where they beat the tuned tree.
7. **Gap 7 — fault tolerance.** Prioritize when large-cluster survivability is the target.

Every item is independently shippable behind its own `RCCL_*` env gate, validated with `rccl-tests` + `tools/topo_expl/`. **No item ships default-on on theory** — each competes with an already-tuned RCCL path and needs an A/B in its target regime first.

---

## Part 4 — What changed vs the prior passes

| | Prior docs | This run (2026-07) |
|---|---|---|
| **Line anchors** | drifted (cost model `tuning.cc:1148`, tree ratio `:858`) | corrected (`:1627`, `:1324-1325`); DDA threshold now a unified 64 MB `RCCL_PARAM` (`collectives.cc:175`) |
| **Gap 1 (quantized)** | flagged, general | **upgraded to High** with QuickReduce FP4/INT3 **AMD-measured 2.25× vs RCCL**; confirmed no inline-compress path in `src/device/` |
| **Gap 2 (chiplet)** | flagged | **upgraded to High**; pinned to AMD's own RCCL-Tuner blog (3.1×) and the exact CPX/`nvlDomain` code that makes it *already-exploitable via the tuner* |
| **Gap 3 (MoE A2A)** | FLASH cited | added **DeepEP (ROCm fork)** + the concrete **GIN** (`src/gin/`) build target |
| **PCIe/Panda** | corrected across passes | re-confirmed on current tree |
| **Method** | prose audits | now backed by a **re-runnable funnel** (`tools/algo_survey/`) with explicit decision gates + adversarial cross-check |

---

## References (new / load-bearing this run)

- QuickReduce FP4 & INT3 on MI355 — AMD ROCm Blogs (`rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2` and `.../quick-reduce-3`). Up to 2.25× vs RCCL; ~4–5× at TP=2 for 1 GB.
- *Optimizing MI300X Inter-Chiplet Communication via the RCCL Tuner API* — AMD ROCm Blogs (`rocm.blogs.amd.com/software-tools-optimization/cpx-rccl-tuner`). 126→12 steps, 3.1×.
- FLASH — arXiv:2505.09764. 1.18–4.48× vs RCCL FanOut A2A on 4×8 MI300X.
- DeepEP — `github.com/deepseek-ai/DeepEP`, `github.com/ROCm/DeepEP`. FP8 MoE dispatch/combine, GIN/NVSHMEM overlap.
- MSCCL++ — arXiv:2504.09014 (ASPLOS'25). 3.8× small / 2.2× large AllReduce on MI300X.
- The Big Send-off / PCCL — arXiv:2504.18658 (SC'25). Frontier MI250X, 2,048 GCDs.
- HiCCL — arXiv:2408.05962 (IPDPS'25). 1.55× vs RCCL.
- Swing — arXiv:2401.09356 (NSDI'24). Rabenseifner — ICCS 2004; Thakur/Rabenseifner/Gropp — IJHPCA 2005.
- Flash Communication — arXiv:2412.04964. Inter-APU on MI300A / Infinity Fabric — arXiv:2508.11298. NUMA-aware GPU mapping — arXiv:2511.02132.
- Full reference lists in the three companion reports.

---

## Appendix — Verification & method

- **Inventory** re-read on this branch; every `file:line` in Part 1 was opened during this run. No `src/` change since the survey base `bece9cc9` — prior *findings* stand; only *anchors* were corrected.
- **Literature** swept across the funnel's directions; each candidate captured with baseline + platform + regime, and **AMD-measured evidence preferred** over NVIDIA-measured claims for applicability.
- **Anti-hype gate applied:** "N× over GPU-aware MPI" and NVIDIA-multimem-only paths were *not* counted as RCCL wins.
- **Adversarial cross-check:** each surviving gap was argued *against* before *for*; the caveats in Part 2 are what survived. The PCIe/Panda "defect" framing from an early draft was refuted and remains removed.
- **Reproduce / re-run:** [`tools/algo_survey/`](tools/algo_survey/README.md) — the funnel, gates, and cross-check protocol that produced this report, runnable as a sub-agent fleet or by hand.

*Prepared by re-surveying the `develop` tip; anchors corrected against current source; literature reconciled to 2026-07 with AMD-measured evidence prioritized. Line numbers may drift as the branch advances — re-run the funnel to refresh.*
