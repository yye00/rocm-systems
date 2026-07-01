# RCCL Collective-Algorithm Survey and High-Impact Improvement Proposals

**Scope:** `projects/rccl` on the `develop` branch (RCCL 2.30.x, an AMD/ROCm fork of NVIDIA NCCL).
**Goal:** (1) inventory the collective-communication algorithms actually implemented in RCCL, (2) identify published, peer-reviewed algorithms that have a strong, evidence-backed chance of improving performance and are *missing or under-exploited* in RCCL, and (3) give concrete, file-level implementation guidance.

> This is an engineering analysis document, not a change to the shipping library. All code references are `file:line` against the surveyed tree and were verified during the survey.

---

## Part 0 — Executive summary

RCCL implements the standard NCCL algorithm set (Ring, double-binary Tree, CollNet Direct/Chain, NVLS / NVLS-Tree, PAT) plus a set of AMD-specific intra-node fast paths (DDA IPC one-/two-shot, Pivot AllToAll, Hierarchical AllGather, RocSHMEM GDA). The selection machinery is a latency + size/bandwidth cost model (`tuning.cc`) over a topology graph that explicitly models the PCIe/XGMI/NIC hierarchy (`topo.cc`, `paths.cc`, `search.cc`).

Five gaps stand out where the literature shows **large, repeatable** wins that RCCL does not currently capture:

| # | Missing / weak capability | Published basis | Where it wins | Confidence |
|---|---------------------------|-----------------|---------------|------------|
| 1 | **Bandwidth-optimal recursive halving–doubling AllReduce** (Rabenseifner) as a first-class algorithm | Rabenseifner 2004; Thakur et al. 2005 | Medium messages, large rank counts, multi-node | Medium |
| 2 | **Explicit multi-level (hierarchical) reduce-scatter → inter-node → all-gather decomposition** | BlueConnect (MLSys'19); HiCCL (IPDPS'25) | Multi-node MI300/MI200; inter-node phase carries only 1/L of the data | Medium-High |
| 3 | **PCIe-switch-locality-aware hierarchical reduction** for PCIe-attached (non-XGMI) GPU platforms — *narrow residual of the MVAPICH / D.K. Panda direction* | Panda group GPU-aware MPI; Faraji & Afsahi 2018 | PCIe-only boxes (e.g. MI210/PCIe); niche on XGMI clusters | Low–Medium (see 2.3 — mostly already covered) |
| 4 | **Staged / aggregated AllToAll** (Bruck for small, hierarchical aggregation for large) | Bruck 1997; Panda group hierarchical A2A (NSF'21) | MoE / expert-parallel small + medium inter-node AllToAll | Medium |
| 5 | **Programmable execution-plan interpreter** (MSCCL/MSCCL++-style), now *removed* from RCCL | MSCCL/TACCL (ASPLOS'21, NSDI'22); MSCCL++ | Lets synthesized, topology-optimal schedules ship without a code change | Medium |
| 6 | **Swing** bandwidth-optimal AllReduce for rail/torus inter-node fabrics | De Sensi et al., NSDI'24 | Large-scale torus / dragonfly / rail-optimized Ethernet | Medium (topology-gated) |

> **Correction after code cross-check (see the verification note at the end):** an earlier draft of this document claimed item #3 was backed by a "cost-model defect" in which RCCL ignores the reverse direction of PCIe links. **That claim was wrong and has been removed.** On re-reading the code, RCCL already models each link direction as an *independent, full-bandwidth* resource (full-duplex), and already drives both directions via multi-channel mirrored rings and the double-binary tree. The core premise of the Panda "use the idle reverse PCIe direction" work is therefore *already largely satisfied* in RCCL; only a narrow residual remains (Part 2.3). The strongest genuinely-missing items are **#2 (hierarchical decomposition)** and, as a fresh algorithm, **#1 (Rabenseifner)** — with the caveats noted in their sections.

---

## Part 1 — What algorithms are in RCCL today

### 1.1 Algorithm/protocol enumeration

The canonical list lives in `src/include/plugin/nccl_tuner.h:27-41`:

```c
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6
...
#define NCCL_PROTO_LL 0      // 16-byte lines, 8B data + 8B flag
#define NCCL_PROTO_LL128 1   // 128-byte lines, higher BW, low latency
#define NCCL_PROTO_SIMPLE 2  // FIFO head/tail, max throughput
```

### 1.2 Core algorithms (NCCL-derived)

| Algorithm | Enum | Key file(s) | Pattern | Collectives |
|-----------|------|-------------|---------|-------------|
| **Ring** | `RING=1` | `src/device/all_reduce.h:15`, `all_gather.h`, `reduce_scatter.h`, `broadcast.h`, `reduce.h`, `sendrecv.h` | Logical ring; reduce-scatter then all-gather pipeline; `2(p-1)` steps | All |
| **Tree** (double-binary, up/down + split) | `TREE=0` | `src/device/all_reduce.h:110` (`runTreeUpDown`), `:183` (`runTreeSplit`) | Reduce up the tree, broadcast down; split-thread variant overlaps the two directions to use bidirectional NIC BW | AllReduce |
| **CollNet Direct** | `COLLNET_DIRECT=2` | `src/device/all_reduce.h:329` | In-network reduction (scatter/gather/reduce/bcast through a SHARP-capable switch/NIC) | AllReduce, AllGather, ReduceScatter |
| **CollNet Chain** | `COLLNET_CHAIN=3` | `src/device/all_reduce.h:706` | Linear chain through the collective NIC | AllReduce |
| **NVLS / NVLS-Tree** | `NVLS=4`, `NVLS_TREE=5` | `src/device/all_reduce.h:465`, `:598` | NVLink-SHARP fabric-memory multimem reduction; **NVIDIA-only**, inert on AMD HW | AllReduce, AllGather, ReduceScatter |
| **PAT** (Parallel Aggregated Trees) | `PAT=6` | `src/device/all_gather.h:157`, `reduce_scatter.h` | Bruck-derived recursive-doubling tree; `log(p)` scaling for AllGather/ReduceScatter | AllGather, ReduceScatter |

**Cost model** (the function that ranks all of the above), `src/graph/tuning.cc:1121-1148`:

```c
// time = latency * pipeline_stages  +  message_bytes / bandwidth
*time = lat * latCount + nBytes / (1000 * bw);
```

`bw` and `lat` come from per-architecture tables (`tuning.cc:135-197`), multiplied by message-size-indexed correction factors (`treeCorrectionFactor` / `ringCorrectionFactor`, 27 buckets). The topology graph that feeds `bw` is built in `src/graph/topo.cc`, paths in `src/graph/paths.cc`, ring/tree orderings searched in `src/graph/search.cc`, and AMD systems can match a pre-baked ordering from `src/graph/rome_models.cc`.

### 1.3 AMD-specific algorithms (beyond stock NCCL)

| Feature | File(s) | Mechanism | Gating |
|---------|---------|-----------|--------|
| **DDA** (Direct Data Access) one-shot/two-shot | `src/include/algorithms/*/​*_dda.h`, `CollCommon.h`, `src/dda_*_ipc.cu` | IPC peer pointers + GPU barrier; *flat* (one-shot all-to-all reduce) under a threshold, *tree* (two-shot RS+AG) above it | gfx942/gfx950, single node, `nRanks≥8`, `ncclSum`, fp32/fp16/bf16; `RCCL_DDA_*` |
| **Pivot AllToAll** | `src/device/alltoall_pivot.h` | Multiple **bidirectional ring pairs** across an all-to-all-connected XGMI clique | `pivotA2AEnabled`, Rome model, >~750 KB/rank |
| **Hierarchical AllGather** | `src/device/hierarchical_ag_shuffle.h`, `collectives.cc:171` | inter-node AG → intra-node AG → local shuffle kernel | `RCCL_HIERARCHICAL_ALLGATHER`, multi-node |
| **RocSHMEM GDA AllToAll(v)** | `src/device/alltoall_gda.h`, `alltoallv_gda.h` | GPU-initiated one-sided AllToAll over RocSHMEM | `ENABLE_ROCSHMEM`, size ≤ threshold |

### 1.4 Notable capability that was **removed**

`docs/how-to/rccl-usage-tips.rst:19-20`:

> *"MSCCL and MSCCL++ integration has been removed from RCCL. The legacy API symbols `mscclLoadAlgo`, `mscclRunAlgo`, and `mscclUnloadAlgo` remain as no-ops for link compatibility."*

There is no MSCCL interpreter directory in `src/` any longer. This matters for the gap analysis (Part 2.5): RCCL can no longer execute an externally *synthesized* schedule; the tuner plugin (`src/include/plugin/tuner/tuner_v6.h`) can only re-rank the built-in algorithms, not introduce a new communication pattern.

---

## Part 2 — Gap analysis: high-impact algorithms missing or under-exploited

### 2.1 Bandwidth-optimal recursive halving–doubling AllReduce (Rabenseifner)

**What it is.** AllReduce = reduce-scatter (recursive *halving*: at step *s* a rank exchanges with a partner at distance `2^s`, sending `n/2^{s+1}` bytes) followed by all-gather (recursive *doubling*, mirror image). Total `2·log₂(p)` steps, exactly `2n·(p-1)/p ≈ 2n` bytes moved — the bandwidth lower bound — but with **logarithmic step count** instead of the ring's `2(p-1)`.

**Why it may help and where RCCL falls short.** RCCL has two general AllReduce shapes: **Ring** (bandwidth-optimal data volume but `2(p-1)` serial dependency — latency grows linearly with rank count) and **Tree** (latency `~2·log(p)`). The RCCL cost model applies a `ratio *= .5` factor to Tree's bus bandwidth (`tuning.cc:858-861`), so on paper there is a medium-message / many-rank valley that Rabenseifner's `2·log₂(p)` steps at `~2n` (bandwidth-optimal) bytes would fill. Thakur, Rabenseifner & Gropp (2005) and Rabenseifner (2004) report exactly this as the method of choice for that regime in MPICH.

> **Cross-check caveat (important).** RCCL's "Tree" is NVIDIA's **double-binary tree**, which in practice reaches **~95% of ring bandwidth** — it is *not* really a half-bandwidth algorithm; the `ratio *= .5` is a modeling convention for a single tree, and two complementary trees run together. So the theoretical "valley" that Rabenseifner fills is, in a well-tuned RCCL, **already substantially covered by the double-binary tree**. Rabenseifner's marginal benefit over a tuned tree is therefore *uncertain*, and it carries known GPU-side downsides (awkward non-power-of-2 handling; many small partner messages at late reduce-scatter steps). It is a genuine *missing algorithm* (no `NCCL_ALGO_*` exists), but it should be treated as a **candidate to benchmark**, not a guaranteed win.

**AMD-specific nuance.** Within a single fully-connected XGMI clique (8× MI300, all-to-all links), Ring is already near-optimal and DDA covers the small/medium case — so RHD's intra-node value is modest. Any win is **inter-node / at the rail level**, and as a building block of the hierarchical decomposition in 2.2.

**Verdict:** Medium confidence. Clear *gap* (no enum for it) but competes head-to-head with the existing double-binary tree; adopt only if measured to beat it in the target regime.

### 2.2 Explicit multi-level hierarchical decomposition (BlueConnect / HiCCL / multi-lane)

**What it is.** Treat AllReduce as a *factorized* sequence over the L levels of the hierarchy (intra-GPU-clique XGMI → intra-node → inter-node NIC → rail):

```
AllReduce =  ReduceScatter(level 0) ∘ ReduceScatter(level 1) ∘ … 
             ∘ AllReduce(top level on 1/∏Nᵢ of the data)
             ∘ AllGather(… ∘ AllGather(level 1) ∘ AllGather(level 0)
```

Each level uses the algorithm/protocol best suited to *its* fabric, and — critically — **the slow inter-node phase only ever touches `1/(N₀·N₁·…)` of the buffer.** BlueConnect (Cho et al., MLSys'19) reported up to **87% reduction** in synchronization overhead on 192 GPUs for ResNet-50 *versus the then-current baseline*; HiCCL (Hidayetoglu et al., IPDPS'25) reports an average **17× over GPU-aware MPI** and — importantly — only **parity-to-competitive with the vendor libraries (NCCL / RCCL / oneCCL)**, not a clear win over them, by composing these primitives with striping + pipelining.

> **Attribution caveat (added after cross-check).** The "17×" is over GPU-aware *MPI*, **not** over RCCL. The MVAPICH "multi-lane" result (arXiv 2508.13397, 2025) that measured **1.59–2.45×** used **multiple MPI processes per GPU** — an MPI-runtime technique that does not map directly onto RCCL's single-process/multi-GPU model — so it is *motivating context*, not a like-for-like promise of RCCL speedup. The defensible claim is narrower: the *decomposition shape* (small inter-node phase) is sound and underlies these results, but the realized gain over an already-tuned RCCL Ring/Tree is unproven and must be measured.

**Where RCCL falls short.** RCCL's Ring *implicitly* does a one-level intra/inter split (the ring threads a node's GPUs then hops the NIC), and the Hierarchical AllGather path (`hierarchical_ag_shuffle.h`) is a hand-rolled 2-level AllGather. But there is **no general, composable, bandwidth-minimizing decomposition for AllReduce** that guarantees the inter-node phase carries only `1/L` of the data, and no mechanism to *order* levels by fabric bandwidth. This is the most promising structural improvement for multi-node MI300X/MI325 clusters — but note RCCL's ring already recovers much of the benefit, so the upside is *incremental*, not the raw factors above.

**Verdict:** Medium-High confidence. Most promising multi-node structural change; reuses 2.1 as its top-level kernel. Requires A/B measurement against tuned Ring/Tree before adoption.

### 2.3 The MVAPICH / D.K. Panda "bidirectional PCIe tree" direction you raised — *mostly already covered*

This is the item that motivated your question. After cross-checking the code, the honest finding is that **RCCL already implements the core of what this line of work advocates**, so the earlier draft's "cost-model defect" framing was incorrect and has been removed. The detail matters, so here is the evidence.

**Correcting the record.** The claim in the earlier draft — that PCIe is treated as *unidirectional* and the reverse direction is "wasted" — is **false**. In `src/graph/topo.cc:172-197` (`ncclTopoConnectNodes`), **each physical link direction is a separate `ncclTopoLink` object with its own independent `bw` budget** (a call adds `A→B`; the reverse `B→A` is a separate object created by a separate call). In `followPath` (`src/graph/search.cc:84-122`), reserving a path in the forward direction decrements **only** the forward link's budget:

```c
float revBw = 0;                                    // search.cc:103  (starts at ZERO)
if (link->remNode->type == DEV && ... < 80 && ...)  revBw += fwBw/8;  // :106  pre-Ampere NVSwitch
if (link->remNode->type == CPU && ...POWER && NVL)  revBw += fwBw;    // :110  POWER9 NVLink
...
SUB_ROUND(link->bw, fwBw);                           // :116  forward budget only
if (revBw) SUB_ROUND(revLink->bw, revBw);            // :117  reverse charged ONLY when coupled
```

`revBw` is therefore **not** "the reverse bandwidth we forgot to use" — it is a **coupling penalty** applied *only* to hardware whose two directions are *not* independent (pre-Ampere NVSwitch, POWER9 NVLink). For PCIe/XGMI, `revBw` stays 0 precisely because those links **are full-duplex**: the two directions are modeled as independent, simultaneously-usable resources. In other words, **RCCL already assumes full-duplex PCIe** — the opposite of the earlier claim.

**RCCL also already drives both directions at the schedule level.** A single unidirectional ring channel uses each link in one direction, but RCCL lays down **multiple channels including reverse-ordered rings**, and the **double-binary tree** (`all_reduce.h` Tree path) is the textbook construction for balancing traffic in *both* directions of every link (each rank is a leaf in one tree and internal in the other). NVIDIA/AMD's double-binary tree reaches ~95% of ring bandwidth for this reason. So the "pair a forward partial with a reverse partial so the link runs full-duplex" idea from the Panda work is, in substance, **already present**.

**What genuinely remains (narrow).** The Panda group's specific target — GPUs hanging off **PCIe switches with no XGMI/NVLink between them** (e.g. PCIe-only MI210 boxes, or older 4-GPU-per-switch servers) — benefits from *reduce-within-switch-first* staging so that a switch's shared uplink carries reduced (smaller) data rather than every GPU's full buffer. RCCL's topology search already *prefers* intra-switch paths and models shared-uplink contention (each path through a `PCI` switch→root link decrements that shared link's budget in `followPath`), so even this is partially covered. The residual opportunity is: make the *reduction order* explicitly switch-locality-aware on PCIe-only platforms — which is best pursued as a **special case of the hierarchical decomposition in 2.2**, not as a separate cost-model change.

**Where it plugs in (if pursued).** Add a PCIe-switch level to the level vector in 2.2 so the intra-switch reduce-scatter happens before crossing the switch uplink. No `search.cc` change is warranted — the reverse-bandwidth accounting is already correct.

**Verdict:** Low–Medium confidence, and **only** on PCIe-attached (non-XGMI) platforms. On XGMI clusters (the mainstream MI250/MI300/MI325 training target) there is essentially nothing to gain here beyond what RCCL already does. This is the most important correction surfaced by the cross-check.

### 2.4 Staged / aggregated AllToAll (Bruck + hierarchical aggregation)

**What it is.** RCCL's general AllToAll is the naive direct scheme: every rank posts `p-1` sends — `O(p)` messages, fine for large payloads but latency-bound for small/medium ones. Two published replacements:
- **Bruck (1997):** `log₂(p)` rounds of doubling-distance exchanges with local rotation/packing — `O(log p)` messages, dramatically fewer for small messages.
- **Hierarchical aggregation (Panda group, "Adaptive and Hierarchical Large Message AllToAll," NSF'21):** aggregate all intra-node partials destined for a remote node into **one** inter-node message per node-pair, exchange, then locally scatter — collapses `O(p²)` tiny network messages to `O(nodes²)` large ones, exploiting bidirectional links on both the intra- and inter-node legs.

**Where RCCL falls short.** RCCL already has the *all-to-all-connected XGMI* case covered by Pivot A2A (`alltoall_pivot.h`) and a RocSHMEM GDA path, but the **cross-node** AllToAll falls back to direct/ring. MoE / expert-parallel training (the dominant AllToAll consumer today) is exactly small-to-medium messages across many nodes — the regime Bruck and hierarchical aggregation target. Note the honest caveat from the literature (ICHPC-Asia'24, *Bruck Performance Analysis*): plain Bruck's intra-node multi-GPU benefit is muted; the **inter-node** and **aggregation** variants are where the win is.

**Verdict:** Medium. Real gap for inter-node MoE AllToAll; layer onto the existing Pivot/GDA paths rather than replacing them. Temper expectations with the ICHPC-Asia'24 caveat above — measure before enabling by default.

### 2.5 Restore a programmable execution-plan interpreter (MSCCL/MSCCL++-style)

**What it is.** MSCCL (ASPLOS'21 / TACCL NSDI'22) and MSCCL++ execute a *data-driven schedule* — a per-(topology, size) program of send/recv/reduce steps synthesized offline (often optimally, via constraint solvers) — on a generic GPU interpreter kernel. This lets a vendor ship a *new* topology-tailored algorithm as a data file, with no library rebuild.

**Where RCCL falls short.** This was explicitly **removed** (`rccl-usage-tips.rst:19`). The remaining extension point, the tuner plugin (`tuner_v6.h`), can only *choose among* Ring/Tree/CollNet/PAT and tweak channels/chunk size — it cannot express a new pattern. So every improvement in 2.1–2.4 currently requires a C++/HIP code change. Re-introducing an interpreter (or an MSCCL++-style executor) would let synthesized schedules — including Rabenseifner and hierarchical plans — be delivered and A/B-tested as data.

**Verdict:** Medium. A force-multiplier rather than a point fix; reduces the cost of shipping 2.1–2.4 and future research.

### 2.6 Swing — bandwidth-optimal AllReduce for rail/torus inter-node fabrics

**What it is.** Swing (De Sensi et al., NSDI'24) keeps communicating partners *close* on bandwidth-limited topologies by "swinging" direction each step: peer `π(r,s) = r ± ρ(s) mod p` with `ρ(s)=(1-(-2)^{s+1})/3`, distance `≤2^s` and strictly below recursive-doubling's for `s>1`. `2·log₂(p)` steps, exactly `2n` bytes; on D-dim tori it splits data into `2D` lanes to use every port. Reported **up to 3×** over recursive-doubling/ring on 2D/3D tori and HyperX (1k–4k nodes).

**Important applicability gate (don't over-sell it).** The paper is explicit: on a **full-bandwidth, non-blocking fat tree, Swing equals recursive doubling** — its advantage exists only where bisection is *limited* and hop-distance creates link contention (torus, dragonfly, rail-optimized Ethernet). So Swing is *not* a win inside a fully-connected XGMI node; it is a forward-looking option for **large-scale AMD clusters on Ultra-Ethernet / rail or torus inter-node fabrics**.

**Verdict:** Medium, topology-gated. Highest value only at scale on contention-limited networks; implement as a selectable inter-node AllReduce, guarded by a topology predicate.

---

## Part 3 — Implementation guidance

General integration rules that apply to every new *algorithm* (2.1, 2.2, 2.6):

1. **Add the enum** in `src/include/plugin/nccl_tuner.h` (bump `NCCL_NUM_ALGORITHMS` and the `_V*` compat shims carefully — the tuner ABI versions on this constant).
2. **Add a topology graph pattern** (`struct ncclTopoGraph.pattern`, `src/include/graph.h`) and search support in `src/graph/search.cc` so channels/rings can be laid out for it.
3. **Add cost-model entries** (`hwLat`, `bwRatio`, correction factors) in `src/graph/tuning.cc:135-197` and the `busBw`/`ratio` logic near `tuning.cc:805-865` so `ncclTopoGetAlgoTime` (`:1121`) can rank it.
4. **Add the device kernel** under `src/device/`, register it in `src/device/generate.py` (it generates the `RunWorkColl` template instantiations) and `rccl_metadata.h`.
5. **Connect host orchestration** in `src/graph/connect.cc` and the enqueue path.
6. **Gate with an env var** (follow the `RCCL_*` pattern in `src/param/`) defaulting to off, then promote after validation. Validate with `rccl-tests` and the topology emulator in `tools/topo_expl/`.

### 3.1 Rabenseifner recursive halving–doubling AllReduce

- **New algo** `NCCL_ALGO_RHD`. Reuse RCCL's existing `ReduceScatter` and `AllGather` ring/PAT device code as building blocks, but drive them with **recursive-doubling partner offsets** (`partner = rank XOR (1<<s)` for power-of-two; use Rabenseifner's 3-step pre/post-processing to fold non-power-of-two ranks down to the nearest lower power of two).
- **Cost-model shape:** latency term `2·log₂(p)·hwLat`; bandwidth term `2n·(p-1)/p / busBw` — i.e. a *low* latency multiplier with *full* bandwidth (no `×0.5` tree penalty). This is what makes it win the medium-message valley; encode it so the model naturally selects it there.
- **Scope:** enable for inter-node / rank-counts where `2(p-1) ≫ 2log₂(p)`; keep Ring/DDA for the single fully-connected XGMI clique.
- **Effort:** ~2–3 kEng-weeks; lowest-risk because the primitives already exist.

### 3.2 Hierarchical (multi-level) AllReduce

- **Represent the level vector** `[N₀(XGMI clique), N₁(node), N₂(rail), N₃(global)]` from the topology graph. RCCL already discovers clique/NUMA/NIC structure (`paths.cc`, `rome_models.cc`, `topo.h:187-188`) — derive the factor list there.
- **Pipeline:** intra-clique `ReduceScatter` (XGMI, Simple/LL128) → node-level `ReduceScatter` → inter-node `AllReduce` on `n/∏Nᵢ` bytes (use 3.1 or CollNet) → mirror `AllGather`s. Crucially, **pipeline by chunk** across levels (HiCCL's striping) so levels overlap rather than barrier — RCCL's channel/chunk machinery (`prims_simple.h`, multi-channel duplication in `connect.cc:83-181`) already supports the chunking.
- **Cost model:** sum per-level `lat + bytesₗ/bwₗ`; the inter-node term shrinks by `∏Nᵢ`, which is the whole point — make sure the model reflects the reduced inter-node volume so it gets selected for multi-node.
- **Reuse:** generalize the existing `hierarchical_ag_shuffle.h` AllGather into the AG half of this pipeline.
- **Effort:** ~4–6 kEng-weeks; highest multi-node payoff.

### 3.3 PCIe-switch-locality-aware reduction (narrow residual of the Panda direction)

> **Scope note:** as established in Part 2.3, RCCL already models PCIe as full-duplex and already uses both link directions (multi-channel mirrored rings + double-binary tree). **Do not** add reverse-bandwidth accounting to `followPath` — that would be wrong; the two directions are already independent budgets. There is *no* cost-model defect to fix.

- **Only applies to PCIe-attached (non-XGMI) GPU platforms.** On XGMI clusters, skip this entirely.
- **The change, if pursued:** treat the PCIe switch as an explicit level in the 3.2 level vector, so an intra-switch reduce-scatter runs *before* any GPU under the switch drives its buffer across the shared switch→root uplink. This shrinks the data crossing the (potentially oversubscribed) uplink to `1/(GPUs-per-switch)`.
- **Validation:** `tools/topo_expl/` can synthesize PCIe-switch tree models; compare bus bandwidth on a PCIe-attached MI210 box with and without switch-level staging. Confirm no regression on XGMI systems (it should be a no-op there).
- **Effort:** ~2 kEng-weeks *as an extension of 3.2* (not a standalone item).

### 3.4 Staged / aggregated inter-node AllToAll

- **New host path** in `collectives.cc` for cross-node AllToAll: when `nNodes>1` and per-pair message is small/medium, **aggregate** every local GPU's partial destined for remote node *j* into one node-to-node buffer, exchange node-to-node (bidirectional, one message per ordered node-pair), then locally scatter. This mirrors `hierarchical_ag_shuffle.h` but for AllToAll.
- **Add a Bruck device kernel** for the small-message single-level case (`log₂(p)` rounds, local rotation + pack). Keep Pivot A2A for the intra-XGMI-clique case; select by `nNodes`, message size, and topology.
- **Effort:** ~3–4 kEng-weeks; high value for MoE.

### 3.5 Execution-plan interpreter (MSCCL++-style)

- Reintroduce a **schedule executor** kernel + host loader (the removed MSCCL API stubs in `nccl.h.in:1044+` are still present as no-ops — a natural re-entry point). Prefer the MSCCL++ executor model (point-to-point + signal primitives) over the older interpreter for maintainability.
- Wire it as a *peer* of the tuner: a loaded plan overrides algorithm selection for matching `(collective, size, topology)` tuples. This is the delivery vehicle that makes 3.1–3.4 (and future synthesized plans) shippable as data.
- **Effort:** ~6–8 kEng-weeks; strategic rather than tactical.

### 3.6 Swing (selectable inter-node AllReduce)

- Implement as an inter-node AllReduce variant with the partner function `π(r,s)=r±ρ(s) mod p` and `2D`-lane data split on torus dims. **Guard it behind a topology predicate** that only enables it when bisection is contention-limited (torus/dragonfly/rail) — never inside a fully-connected XGMI clique or on a non-blocking fat tree, where the paper shows no gain.
- **Effort:** ~3–4 kEng-weeks; defer until AMD rail/torus inter-node fabrics are the target.

---

## Part 4 — Recommended sequencing

1. **3.2 Hierarchical (multi-level) AllReduce** — the most promising genuinely-missing structural change for multi-node clusters. Build it first and *measure it against tuned Ring/Tree* to establish the real (incremental) upside before investing further.
2. **3.4 Staged / aggregated inter-node AllToAll** — clearest independent win, targets MoE, doesn't overlap the AllReduce work.
3. **3.1 Rabenseifner RHD AllReduce** — implement as a *benchmark candidate* (it competes with the existing double-binary tree; adopt only if it wins in the target regime). Reuses existing primitives, so cheap to prototype.
4. **3.3 PCIe-switch-locality staging** — only if PCIe-attached (non-XGMI) platforms are a target; fold into 3.2.
5. **3.5 plan interpreter** — strategic; unlocks data-driven delivery of everything above.
6. **3.6 Swing** — when rail/torus inter-node fabrics become a priority.

Each item is independently shippable behind its own `RCCL_*` env gate, validated with `rccl-tests` + `tools/topo_expl/`. **No item should be adopted on theory alone** — every one above competes with an already-tuned RCCL path, so each needs an A/B benchmark in its target regime before it ships enabled-by-default.

---

## References

- R. Rabenseifner. *Optimization of Collective Reduction Operations.* ICCS 2004.
- R. Thakur, R. Rabenseifner, W. Gropp. *Optimization of Collective Communication Operations in MPICH.* IJHPCA 2005.
- M. Cho et al. *BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy.* MLSys 2019 / IBM J. R&D 2019.
- M. Hidayetoglu et al. *HiCCL: A Hierarchical Collective Communication Library.* IPDPS 2025. arXiv:2408.05962.
- D. De Sensi, T. Bonato, D. Saam, T. Hoefler. *Swing: Short-cutting Rings for Higher Bandwidth Allreduce.* NSDI 2024. arXiv:2401.09356.
- J. Bruck et al. *Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems.* IEEE TPDS 1997.
- *Optimizing Allreduce Operations for Modern Heterogeneous Architectures with Multiple Processes per GPU* (MVAPICH multi-lane, AMD MI300A results). arXiv:2508.13397, 2025.
- I. Faraji, A. Afsahi. *Design considerations for GPU-aware collective communications in MPI.* Concurrency & Computation, 2018.
- *Adaptive and Hierarchical Large Message All-to-All Communication Algorithms for Large-scale Dense GPU Systems* (Ohio State / MVAPICH). NSF par.nsf.gov/10300197, 2021.
- *Bruck Algorithm Performance Analysis for Multi-GPU All-to-All Communication.* HPC Asia (ICHPC) 2024.
- M. Cowan et al. *MSCCL / Synthesizing Optimized Collective Communication Algorithms* (ASPLOS 2021); *TACCL* (NSDI 2022); MSCCL++ (Microsoft).

---

## Appendix — Verification note (code cross-check)

Every load-bearing code claim in this document was re-read against the source on the surveyed checkout. Results:

**Verified correct (Part 1 inventory):**
- Algorithm/protocol enum values — `src/include/plugin/nccl_tuner.h:27-41` ✓
- Cost-model formula `time = lat*latCount + nBytes/(1000*bw)` — `src/graph/tuning.cc:1148` ✓
- Tree bus-bandwidth factor `ratio *= .5` (non-ring/NVLS) — `src/graph/tuning.cc:858-861` ✓
- Device-kernel anchors — Ring `all_reduce.h:15`, `runTreeUpDown:110`, `runTreeSplit:183`, CollNet-Direct `:329`, NVLS `:465`, NVLS-Tree `:598`, CollNet-Chain `:706`; PAT `all_gather.h:157` / `reduce_scatter.h:189` ✓
- DDA gating — `nRanks < 8` disabled, gfx942/gfx950 only, disabled in-group / under symmetric support, 64 MB default threshold — `src/collectives.cc:134-144`, `:128` ✓
- MSCCL/MSCCL++ removed (API symbols now no-ops) — `docs/how-to/rccl-usage-tips.rst:19-20`, stubs in `nccl.h.in:1044+`; no `msccl` directory under `src/` ✓

**Corrected (was wrong in the first draft):**
- **The claimed "PCIe reverse-bandwidth cost-model defect" in `search.cc` does not exist.** `ncclTopoConnectNodes` (`topo.cc:172-197`) stores each link direction as an independent `ncclTopoLink` with its own `bw`; `followPath` (`search.cc:84-122`) decrements only the forward budget and charges reverse (`revBw`) **only** for coupled-direction hardware (pre-Ampere NVSwitch `:106`, POWER9 NVLink `:110`). PCIe/XGMI are therefore already modeled as **full-duplex**. Item #3 and its implementation guidance (Part 2.3, Part 3.3) were rewritten accordingly, and the priority ordering (Part 4) no longer leads with a non-existent fix.
- Confidence levels and literature attributions were recalibrated: HiCCL's "17×" is over GPU-aware **MPI** (parity with RCCL/NCCL, not a win over them); the MVAPICH "1.59–2.45×" uses **multiple MPI processes per GPU** (not a like-for-like RCCL result); and RCCL's double-binary tree already reaches ~95% of ring bandwidth, which tempers the Rabenseifner (item #1) value proposition. All affected verdicts now say "benchmark before adopting."

*Prepared by survey of the `develop` branch tree and corrected after a line-by-line code cross-check. Line numbers may drift as the branch advances.*
