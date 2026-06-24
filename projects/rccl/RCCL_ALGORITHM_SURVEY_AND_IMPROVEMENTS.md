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
| 1 | **Bandwidth-optimal recursive halving–doubling AllReduce** (Rabenseifner) as a first-class algorithm | Rabenseifner 2004; Thakur et al. 2005 | Medium messages, large rank counts, multi-node | High |
| 2 | **Explicit multi-level (hierarchical) reduce-scatter → inter-node → all-gather decomposition** | BlueConnect (MLSys'19); HiCCL (IPDPS'25); MVAPICH multi-lane (2025) | Multi-node MI300/MI200; inter-node phase carries only 1/L of the data | High |
| 3 | **Bidirectional, PCIe-tree-locality-aware scheduling** (the MVAPICH / D.K. Panda style) — drive *both* directions of each PCIe link concurrently | Panda group GPU-aware MPI; Faraji & Afsahi 2018 | PCIe-attached GPUs, GPU↔NIC paths, non-XGMI platforms | High |
| 4 | **Staged / aggregated AllToAll** (Bruck for small, hierarchical aggregation for large) | Bruck 1997; Panda group hierarchical A2A (NSF'21) | MoE / expert-parallel small + medium AllToAll | Medium-High |
| 5 | **Programmable execution-plan interpreter** (MSCCL/MSCCL++-style), now *removed* from RCCL | MSCCL/TACCL (ASPLOS'21, NSDI'22); MSCCL++ | Lets synthesized, topology-optimal schedules ship without a code change | Medium |
| 6 | **Swing** bandwidth-optimal AllReduce for rail/torus inter-node fabrics | De Sensi et al., NSDI'24 | Large-scale torus / dragonfly / rail-optimized Ethernet | Medium (topology-gated) |

The single most concrete and immediately actionable finding for the question you raised — *bidirectional PCIe tree exploitation, à la the former DK Panda / MVAPICH team* — is item **#3**, and it is backed by a specific defect in the current cost model documented in Part 2.3.

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

**Why it helps and where RCCL falls short.** RCCL has only two general AllReduce shapes: **Ring** (bandwidth-optimal data volume but `2(p-1)` serial dependency — latency grows linearly with rank count) and **Tree** (latency-optimal `~2·log(p)` but the well-known ~50% bandwidth penalty baked into the model at `tuning.cc` via the `ratio *= .5` tree factor). The medium-message / many-rank regime — exactly where large-model training and multi-node inference live — is a valley between the two: too big for Tree's bandwidth penalty, too many ranks for Ring's latency. Rabenseifner fills that valley. Thakur, Rabenseifner & Gropp (2005) and the original Rabenseifner (2004) report it as the method of choice for that regime in MPICH; modern GPU re-derivations (e.g. arXiv 2508.13397, 2025) confirm `2·log p` step-count benefits hold on GPU fabrics.

**AMD-specific nuance.** Within a single fully-connected XGMI clique (8× MI300, all-to-all links), Ring is already near-optimal and DDA covers the small/medium case — so RHD's intra-node value is modest. The win is **inter-node and at the rail level**, and as the building block of the hierarchical decomposition in 2.2.

**Verdict:** High confidence, clear gap (no `NCCL_ALGO_*` for it), well-trodden implementation path.

### 2.2 Explicit multi-level hierarchical decomposition (BlueConnect / HiCCL / multi-lane)

**What it is.** Treat AllReduce as a *factorized* sequence over the L levels of the hierarchy (intra-GPU-clique XGMI → intra-node → inter-node NIC → rail):

```
AllReduce =  ReduceScatter(level 0) ∘ ReduceScatter(level 1) ∘ … 
             ∘ AllReduce(top level on 1/∏Nᵢ of the data)
             ∘ AllGather(… ∘ AllGather(level 1) ∘ AllGather(level 0)
```

Each level uses the algorithm/protocol best suited to *its* fabric, and — critically — **the slow inter-node phase only ever touches `1/(N₀·N₁·…)` of the buffer.** BlueConnect (Cho et al., MLSys'19) reported up to **87% reduction** in synchronization overhead on 192 GPUs for ResNet-50; HiCCL (Hidayetoglu et al., IPDPS'25) reports an average **17×** over GPU-aware MPI and parity-to-better than NCCL/RCCL by composing exactly these primitives with striping + pipelining; the MVAPICH "multi-lane" study (arXiv 2508.13397, 2025) measured **1.59–2.45×** on AMD MI300A (Tuolumne) and NVIDIA nodes for large buffers using precisely this intra-RS / inter-AR / intra-AG split.

**Where RCCL falls short.** RCCL's Ring *implicitly* does a one-level intra/inter split (the ring threads a node's GPUs then hops the NIC), and the Hierarchical AllGather path (`hierarchical_ag_shuffle.h`) is a hand-rolled 2-level AllGather. But there is **no general, composable, bandwidth-minimizing decomposition for AllReduce** that guarantees the inter-node phase carries only `1/L` of the data, and no mechanism to *order* levels by fabric bandwidth. This is the highest-leverage structural improvement for multi-node MI300X/MI325 clusters.

**Verdict:** High confidence. Largest practical multi-node win; reuses 2.1 as its top-level kernel.

### 2.3 Bidirectional, PCIe-tree-locality-aware scheduling — the MVAPICH / D.K. Panda direction you raised

This is the item most directly matching your example, and the survey found a **concrete, fixable defect** in the cost model.

**The defect.** PCIe (and in general every non-NVLink/XGMI link) is treated as essentially **unidirectional** during path reservation. In `src/graph/search.cc:84-117` (`followPath`), reverse bandwidth `revBw` is only charged — and therefore only *modeled as usable* — for two narrow cases:

```c
// NVSwitch DEV node on pre-Ampere:           revBw += fwBw/8;   (search.cc:106)
// POWER9 CPU NVLink:                          revBw += fwBw;     (search.cc:110)
// everything else (incl. all PCIe): revBw stays 0  -> reverse direction is "free"/ignored
```

Consequently the search never *credits* a schedule for using the return direction of a PCIe link, and the algorithm builders never deliberately construct schedules that push distinct data **simultaneously up and down** the same PCIe switch port. Modern PCIe (Gen4/Gen5) and Infinity Fabric host links are **full-duplex** — a Gen5 x16 link sustains ~64 GB/s *each way concurrently*. Leaving the return path idle on the GPU↔switch and GPU↔NIC segments wastes up to half the available bisection on PCIe-attached platforms (e.g. MI210/PCIe boxes, and the GPU↔NIC leg on every platform).

**What the Panda/MVAPICH line of work does.** Their GPU-aware MPI collectives (Faraji & Afsahi, *Concurrency & Computation* 2018; the Ohio State MVAPICH2-GDR designs; their hierarchical large-message AllToAll, NSF'21) build the intra-node schedule from the **measured PCIe/NVLink tree**: GPUs under the same PCIe switch reduce locally first; each physical link is then driven in **both directions at once** by pairing a "forward" partial with a "reverse" partial (a bidirectional-ring / mirror-tree construction), so a Gen-N x16 link delivers its full bidirectional figure rather than half. The result is the staged "reduce within switch → exchange across switches → broadcast within switch" pattern, with every link bidirectionally saturated.

**Where this plugs into RCCL.**
- The topology already represents PCIe switches as `PCI` nodes and even flattens BCM two-level switches (`topo.cc:199-271`), so the **tree structure is available** — it just isn't exploited bidirectionally.
- `search.cc` would need a `revBw` accounting path for `LINK_PCI`/`LINK_SYS` when the platform reports full-duplex, and the ring/tree builders (`connect.cc`, `trees.cc`, `rings.cc`) would need to emit **mirror pairs** that the device kernels run concurrently (the Tree split-thread mechanism in `all_reduce.h:183` and the Pivot bidirectional-ring kernel in `alltoall_pivot.h` are existing precedents for two-direction device code).

**Verdict:** High confidence, *directly* your example, with a pinpointed code defect. Biggest gains on PCIe-attached and GPU↔NIC paths; smaller inside a fully-connected XGMI clique (where XGMI is already aggregated bidirectionally at `topo.cc:172-197`).

### 2.4 Staged / aggregated AllToAll (Bruck + hierarchical aggregation)

**What it is.** RCCL's general AllToAll is the naive direct scheme: every rank posts `p-1` sends — `O(p)` messages, fine for large payloads but latency-bound for small/medium ones. Two published replacements:
- **Bruck (1997):** `log₂(p)` rounds of doubling-distance exchanges with local rotation/packing — `O(log p)` messages, dramatically fewer for small messages.
- **Hierarchical aggregation (Panda group, "Adaptive and Hierarchical Large Message AllToAll," NSF'21):** aggregate all intra-node partials destined for a remote node into **one** inter-node message per node-pair, exchange, then locally scatter — collapses `O(p²)` tiny network messages to `O(nodes²)` large ones, exploiting bidirectional links on both the intra- and inter-node legs.

**Where RCCL falls short.** RCCL already has the *all-to-all-connected XGMI* case covered by Pivot A2A (`alltoall_pivot.h`) and a RocSHMEM GDA path, but the **cross-node** AllToAll falls back to direct/ring. MoE / expert-parallel training (the dominant AllToAll consumer today) is exactly small-to-medium messages across many nodes — the regime Bruck and hierarchical aggregation target. Note the honest caveat from the literature (ICHPC-Asia'24, *Bruck Performance Analysis*): plain Bruck's intra-node multi-GPU benefit is muted; the **inter-node** and **aggregation** variants are where the win is.

**Verdict:** Medium-high. Clear gap for inter-node MoE AllToAll; layer onto the existing Pivot/GDA paths rather than replacing them.

### 2.5 Restore a programmable execution-plan interpreter (MSCCL/MSCCL++-style)

**What it is.** MSCCL (ASPLOS'21 / TACCL NSDI'22) and MSCCL++ execute a *data-driven schedule* — a per-(topology, size) program of send/recv/reduce steps synthesized offline (often optimally, via constraint solvers) — on a generic GPU interpreter kernel. This lets a vendor ship a *new* topology-tailored algorithm as a data file, with no library rebuild.

**Where RCCL falls short.** This was explicitly **removed** (`rccl-usage-tips.rst:19`). The remaining extension point, the tuner plugin (`tuner_v6.h`), can only *choose among* Ring/Tree/CollNet/PAT and tweak channels/chunk size — it cannot express a new pattern. So every improvement in 2.1–2.4 currently requires a C++/HIP code change. Re-introducing an interpreter (or an MSCCL++-style executor) would let synthesized schedules — including Rabenseifner, hierarchical, and bidirectional-PCIe plans — be delivered and A/B-tested as data.

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

### 3.3 Bidirectional PCIe-tree scheduling (your example)

This can ship in two independently-valuable stages:

- **Stage A — model the return path (small, surgical).** In `src/graph/search.cc:84-117`, extend `followPath` to charge `revBw` for `LINK_PCI` and `LINK_SYS` when the platform advertises full-duplex (add a `fullDuplex` bit to the link during discovery in `topo.cc:172-197`, set from PCIe Gen and Infinity-Fabric capability). This alone lets the existing search *find* schedules that reserve both directions, improving channel counts on PCIe-attached and GPU↔NIC paths.
- **Stage B — emit mirror schedules (the algorithmic part).** In the ring/tree builders (`rings.cc`, `trees.cc`, `connect.cc`), construct **bidirectional pairs**: for each PCIe-local group, build a forward ring and its mirror so each physical link carries a distinct partial in each direction. Run them concurrently on the device using the existing two-direction precedents — the Tree split-thread kernel (`all_reduce.h:183`) and the Pivot bidirectional-ring kernel (`alltoall_pivot.h`). Prioritize the **GPU↔NIC PCIe leg** first: it exists on *every* platform and currently runs half-duplex for collectives that don't overlap send/recv.
- **Validation:** `tools/topo_expl/` can synthesize PCIe-switch tree models; compare reserved bisection before/after Stage A, then bus-bandwidth before/after Stage B on a PCIe-attached MI210 box and on the GPU↔NIC path of an MI300X node.
- **Effort:** Stage A ~1 kEng-week; Stage B ~3–4 kEng-weeks.

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

1. **3.3 Stage A** (model PCIe return path) — tiny, surgical, directly addresses the bidirectional-PCIe question, and benefits *all* algorithms by improving channel search. Do this first.
2. **3.1 Rabenseifner RHD AllReduce** — low risk, reuses existing primitives, fills the medium-message valley.
3. **3.2 Hierarchical AllReduce** — biggest multi-node win; uses 3.1 as its top level.
4. **3.3 Stage B** (mirror bidirectional schedules) and **3.4 staged AllToAll** in parallel — both are device-kernel work with the Tree-split/Pivot precedents to copy from.
5. **3.5 plan interpreter** — strategic; unlocks data-driven delivery of everything above.
6. **3.6 Swing** — when rail/torus inter-node fabrics become a priority.

Each item is independently shippable behind its own `RCCL_*` env gate, validated with `rccl-tests` + `tools/topo_expl/`.

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

*Prepared by automated survey of the `develop` branch tree. Code citations verified against the surveyed checkout; line numbers may drift as the branch advances.*
