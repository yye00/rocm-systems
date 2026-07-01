# Single-Node Intra-Node RCCL Performance Gaps — Implementation-Ready Report

## 1. Scope & Summary

**Scope.** This report covers **single-node / intra-node performance only** for RCCL (`/home/user/rocm-systems/projects/rccl`, develop ≈ 2.30.4). Target hardware is one AMD node — 8× MI300X/MI325 (gfx942/CDNA3) or 8× MI355X (gfx950/CDNA4) in a fully-connected XGMI clique, plus the intra-GPU XCD/IOD chiplet and CPX partition-mode hierarchy. Collectives in scope: intra-node AllReduce, AllGather, ReduceScatter, AllToAll, Broadcast, Reduce. **Everything inter-node / network / rail / torus, plus anything requiring NVIDIA multimem / TMA / NVSwitch, and all resilience / spec-adherence items, are excluded.** Only latency/bandwidth wins over the XGMI clique count.

The dominant, high-confidence, AMD-native opportunity is **inline low-precision (quantized/compressed) AllReduce over XGMI** — RCCL's existing DDA IPC path (`src/dda_all_reduce_ipc.cu`, `src/include/algorithms/all_reduce/all_reduce_dda.h`) moves **full-precision** fp32/fp16/bf16, gated to **exactly 8 ranks, ncclSum only**, so a quantized reduce-scatter+all-gather (QuickReduce-class) is the single biggest missing lever. The many QuickReduce/Flash/EQuARX/MoRI candidates collapse into **one implementation cluster** (block-wise inline quant + two-shot AR), which I merge for ranking. Secondary levers: **CPX/XCD chiplet-aware tuner conditioning** (the tuner mechanism exists but has no XCD/IOD dimension), **portable LD/ST one-shot symmetric kernels generalized past the 8-rank/ncclSum DDA limits**, and **log-round (recursive-halving/circulant) RS/AR** to cut ring's p−1 steps. Schedule-synthesis candidates (ForestColl/TE-CCL/TACCL) and full MSCCL++ DSL are down-ranked: they need a removed schedule executor and large new engines for uncertain single-node (already-near-optimal-ring) gains.

### Summary Ranking Table

| Rank | Gap | Collective | Impact | Conf | Code Verdict | Key Citation |
|---|---|---|---|---|---|---|
| 1 | Inline block-quantized two-shot AllReduce (QuickReduce cluster: INT4/6/8/FP8 gfx942, native FP4/MXFP gfx950; incl. Flash Comm, MoRI-A2A, RCCLX-LP, EQuARX as sub-variants) | AllReduce (+A2A/AG/RS ext.) | H | H | ABSENT | ROCm Blogs 2025/2026 (QuickReduce); MSCCL++ ASPLOS'26 (peer-rev.) |
| 2 | CPX/XCD/IOD chiplet-hierarchy-aware algo+proto selection (Tuner API dimension) | AR/AG/RS | H | H | PARTIAL | ROCm Blog 2025 (CPX RCCL Tuner) |
| 3 | Generalized one-shot LD/ST + two-shot symmetric AR beyond DDA 8-rank/ncclSum/3-dtype limits | AR/RS/AG | M–H | H | PARTIAL | Demystifying NCCL, arXiv 2025 |
| 4 | Log-round bandwidth-optimal RS/AllReduce (recursive-halving / circulant skips) | RS, AR | M | M | ABSENT | Träff, arXiv 2024 |
| 5 | MSCCL++ 1PA/2PA/2PR one-sided-async intra-node collectives (non-multimem) | AR/AG/RS | M | H | PARTIAL | MSCCL++ ASPLOS'26 (peer-rev.) |
| 6 | NCCLZ decoupled quantization + entropy-coded collectives | AR/AG/RS | L–M | L | ABSENT | NCCLZ arXiv 2026 (unverified) |
| 7 | Multi-lane multi-process-per-GPU CPX-die-parallel AllReduce | AR | L–M | M | ABSENT | arXiv 2025 (MI300A) |
| 8 | ForestColl / TE-CCL / TACCL synthesized bandwidth-optimal single-node schedules | AG/A2A/AR | L | M | ABSENT | ForestColl NSDI'26; TE-CCL SIGCOMM'24; TACCL NSDI'23 (peer-rev.) |

**Excluded after review:** NVLS/NVLS_TREE multimem, symmetric MC/TMA/STMC/LDMC paths, MSCCL++ SwitchChannel/2PA-Switch, MoRI inter-node RDMA path, contrib/nccl_ep Hopper/Blackwell-specific FP8 dispatch (marked NOT SUPPORTED, TMA/warp-specialized). All are NVIDIA-multimem/TMA-dependent or inter-node.

---

## 2. Ranking Rationale

Rank = **single-node perf impact × research confidence × AMD viability**, honestly discounted:

- **Rank 1** wins on all three axes: measured 1.5–4.14× vs RCCL on a single MI300X/MI355X node (named baseline: stock RCCL/PyNCCL AllReduce, FP16/BF16), AMD-native (CDNA3/CDNA4, no NVIDIA primitives), and directly extends an existing RCCL structure. The ~10 separate quantized-AR candidates are the **same mechanism** (block-wise inline quant + two-shot RS/AG) and are merged.
- **Rank 2** is high impact/confidence (2–3.1× small-message latency, ROCm-blog measured vs RCCL default) and the tuner scaffold already exists — only the XCD/IOD dimension is missing, so it is low-risk.
- **Rank 3** is a portable, already-present kernel family that just needs its gates widened — high confidence, moderate incremental gain.
- **Rank 4** is algorithmically proven (log p rounds vs ring's p−1) but preprint (medium confidence) and gains over an already-good ring at p=8 are modest.
- **Rank 5** is peer-reviewed with strong MI300X numbers, but its distinguishing one-sided/async/ring-overlap machinery is a large new abstraction on top of what DDA already covers; medium single-node marginal value.
- **Ranks 6–8** are down-ranked for **low confidence / unverified citation (NCCLZ)**, **APU-only baseline-inflated numbers (multi-lane)**, or **requiring a removed schedule executor + a large new synthesis engine for uncertain single-node ring-beating gains (ForestColl/TE-CCL/TACCL)**.

---

## 3. Gap Details

### Rank 1 — Inline block-quantized two-shot AllReduce (QuickReduce cluster)

**(a) What it is + citation.** A ROCm-native two-shot AllReduce (reduce-scatter to a designated reducer per shard, then all-gather broadcast) where **every XGMI transfer carries block-quantized data** instead of fp16/bf16. Block size 32, per-block scale factor; codecs FP8/Q8(int8)/Q6/Q4 on gfx942, plus **native FP4** on gfx950 via `__builtin_amdgcn_cvt_scalef32_pk_fp4_f16` (scale computed in FP16, deviating from OCP MXFP4 E8M0). Quant/dequant is interleaved with the transfer. This cluster merges: **QuickReduce** (ROCm Blogs 2025 + FP4 follow-up 2026, `rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html`, `github.com/mk1-project/quickreduce`, not peer-reviewed); **Flash Communication** (arXiv 2412.04964, 2024, not peer-reviewed) — the "quantize once before RS, dequant once after AG, 2 passes not N" framing; **MoRI** hybrid MXFP4-dispatch/FP8-combine A2A (LMSYS 2026, `github.com/ROCm/mori`); **RCCLX LP collectives** extending quant to A2A/AG/RS (Meta Eng. 2026); **EQuARX** block-wise INT8 pipelined-with-reduce (arXiv 2506.17615, 2025); **FlashCommunication V2** any-bit bit-splitting/spike-reserving (arXiv 2508.03760, 2025). MSCCL++ (ASPLOS 2026, **peer-reviewed**, arXiv 2504.09014) anchors the two-shot structure.

**Measured perf vs baseline.** QuickReduce vs **stock RCCL AllReduce (FP16/BF16)** on a single node: gfx942 MI300X up to ~3× at TP=2 (2.25× on 2×/4× MI300X), TTFT >1.2× with Q4; gfx950 MI355X at 1 GB: FP4 **4.14× (TP=2), 3.43× (TP=4), 1.52× (TP=8)**. Crossover ~1 MB (TP2/4), ~4 MB (TP8); below ~512 KB–2 MB the un-quantized custom/RCCL AR wins (compression overhead dominates). Flash Comm: >3× intra-node comm, ~2× TTFT vs NCCL BF16 ring (L40). All are **lossy** (accuracy-tolerant workloads only).

**(b) Code evidence (ABSENT).**
- `QuickReduce` appears **only** in the survey doc `projects/rccl/COLLECTIVE_COMM_STATE_OF_THE_ART_2026.md:18,56` as an external technique; **zero** hits under `src/`.
- No FP4 builtin: grep `cvt_scalef32_pk_fp4` across `src/` → none. The only `__builtin_amdgcn_cvt_scalef32_pk_*` uses are in `src/include/rccl_float8.h:71,76,93,98,115,120,144,149` and are **FP8/BF8 conversions with a hardcoded scale of `1.f`** (datatype conversion feeding `v_pk_add_f16`), **not** a block-quantization codec. `rccl_float8.h:381` "quantize" = FP8 saturation, not payload compression.
- `ncclDataType_t` (`src/nccl.h.in:593,595`) stops at `ncclFloat8e4m3=10`, `ncclNumTypes=12` — no FP4/INT4 sub-byte comm datatype.
- The two "two-shot" AllReduce paths carry full precision: `src/include/algorithms/all_reduce/all_reduce_dda.h:21 ddaAllReduceFlatIpc` (one-shot, `reduceScatter` pattern=2, line 43) and `:56 ddaAllReduceTreeIpc` (two-shot `reduceScatter`+`allGather`, lines 76/90), both templated on full-width `T` via `uint4` (`CollCommon.h:83 vecElementAdd`). No scale/quant/fp4 tokens in the header.
- Gating: `src/dda_all_reduce_ipc.cu:133 nNodes==1`, `:136 nRanks==kDdaNranks`(8), `:139 op==ncclSum`, fp32/fp16/bf16 only; `:31 kDdaFlatTreeThresholdBytes=1<<18`.

**(c) Implementation plan (RCCL files).**
1. **New codec header** `src/include/algorithms/all_reduce/quick_reduce_codec.h`: block-32 symmetric quant/dequant. gfx942: Q4/Q6/Q8/FP8 pack/unpack via packed-int SIMD + per-block FP16 scale (`group_abs_max`). gfx950: add FP4 path guarded by `#if defined(__gfx950__)` using `__builtin_amdgcn_cvt_scalef32_pk_fp4_f16` (co-locate with the existing FP8 builtins in `src/include/rccl_float8.h`).
2. **New kernel** `src/dda_quick_reduce_ipc.cu` + header `.../all_reduce/quick_reduce_dda.h` modeled on `ddaAllReduceTreeIpc`: quantize shard → IPC store → `IpcGpuBarrier.syncOnSameBlockIdx` → dequant+reduce in FP16/FP32 → requantize → all-gather → dequant into recvbuff. Reuse `src/include/ipc_gpu_barrier.h` (weak-fence path) and `CollCommon.h`.
3. **Dispatch:** in `src/collectives.cc` near the existing `ncclAllReduceDdaIpc` call (`:440–442`), add a tier: if `RCCL_QUICKREDUCE_ENABLE`, dtype∈{fp16,bf16}, size ≥ crossover, op==ncclSum → route to quick-reduce. Add eligibility `ncclAllReduceQuickReduceEligible` mirroring `dda_all_reduce_ipc.cu:133–163` but **relaxing the exactly-8-rank gate** to 2/4/8.
4. **Cost model:** in `src/graph/tuning.cc` (cost `:1148 *time = lat*latCount + nBytes/(1000*bw)`) scale effective `nBytes` by the compression ratio (~0.28 for Q4) for the quantized variant so the selector picks it above crossover.
5. **Env gate:** `RCCL_QUICKREDUCE_ENABLE` (default 0), `RCCL_QUICKREDUCE_CODEC` (q4/q6/q8/fp8/fp4), `RCCL_QUICKREDUCE_MIN_BYTES` (default ~1 MB). Register alongside existing DDA params.
6. **Validate:** `rccl-tests/all_reduce_perf -b 512K -e 1G -f 2 -g 8 -d half` with/without the env gate on one node; confirm bus-BW uplift ≥ crossover and check numerical error vs a reference sum. Use `tools/topo_expl` to confirm the 8-GPU XGMI clique topology is detected.

**Effort:** Large (2–4 wk: codec + kernel + FP4 ISA + accuracy validation). **Risk:** Medium–High — lossy (needs accuracy gate; NOT drop-in for exact ncclSum); FP4 path is gfx950-only and ISA-sensitive; must guard against enabling on tiny messages where it loses.

---

### Rank 2 — CPX/XCD/IOD chiplet-hierarchy-aware algo+protocol selection

**(a) What it is + citation.** Exploit the MI300X intra-GPU 3-tier bandwidth hierarchy (within-XCD L2 ~51.6 TB/s ≫ XCD→IOD ~17.2 TB/s ≫ IOD→HBM ~5.3 TB/s) and CPX partition mode (each XCD a logical GPU, cross-IOD XGMI the bottleneck). Detect CPX vs SPX and pick algo+protocol per message-size zone. *Optimizing MI300X Inter-Chiplet Communication via the RCCL Tuner API* (AMD ROCm Blog 2025, `rocm.blogs.amd.com/software-tools-optimization/cpx-rccl-tuner/README.html`, not peer-reviewed).

**Measured perf vs baseline.** vs **RCCL default Ring+Simple**: Tree+LL gives 2–3× lower latency for small messages (peak ~3.1× at 4–16 KB). Zones: 0–262 KB Tree+LL, 262 KB–4 MB Tree+LL128 (~20 GB/s bus BW), >4 MB Ring+Simple (~26 GB/s).

**(b) Code evidence (PARTIAL).** The generic CSV tuner exists but has **no chiplet dimension**: schema `src/plugin/tuner/csv_tuner.cc:44` = `colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff` — no CPX/XCD/IOD/partition-mode field; match loop keys only on those fields. Only shipped CSV `tuner/rccl_tuner_gfx950.csv` has 4 rules, all `nNodes=1,nRanks=8`, no CPX/64-rank/XCD rows. CPX in `src/` is only topology-parser accommodation: `src/graph/xml.cc:434` (PCIe func-ID override), `src/graph/xml.h:22` (`MAX_SUBS` 128→512). No 3-tier bandwidth model in `src/graph/topo.h:37-41` (single per-arch XGMI width) or `src/graph/tuning.cc`. `src/misc/alt_rsmi.cc:133,151` parses `s_partition_id` from PCI location_id but only logs it.

**(c) Implementation plan.**
1. Add a partition-mode/XCD detector: extend `src/graph/topo.*` to record XCD count and IOD grouping (consume the `s_partition_id` bitfield already parsed in `src/misc/alt_rsmi.cc:133,151`), and expose `comm->partitionMode` / `comm->nXcd`.
2. Extend the CSV schema in `src/plugin/tuner/csv_tuner.cc:44` with optional trailing fields `partitionMode,localGroup` (backward-compatible parsing, default wildcard); add them to the match loop (`:587-602`).
3. Ship `tuner/rccl_tuner_gfx942.csv` and add CPX (nRanks=64) rows to `tuner/rccl_tuner_gfx950.csv` implementing the 3-zone Tree+LL / Tree+LL128 / Ring+Simple recommendation.
4. Optionally add an IOD-distance term to the cost model `src/graph/tuning.cc:1148` (add a die-to-die hop penalty when peers are cross-IOD).
5. **Env gate:** `RCCL_CHIPLET_TUNER_ENABLE` (default 0 initially).
6. **Validate:** run `all_reduce_perf -b 4K -e 4M -g 1` under `mpirun -np 64 ... --bind-to numa` (CPX) and `-g 8` (SPX); confirm the small-message latency drop vs default. Use `tools/topo_expl` to confirm XCD enumeration.

**Effort:** Medium (1–2 wk). **Risk:** Low–Medium — tuner scaffold exists; main risk is correct XCD/IOD detection across driver versions and CPX-mode device enumeration.

---

### Rank 3 — Generalized one-shot LD/ST + two-shot symmetric AR beyond DDA limits

**(a) What it is + citation.** Portable load/store one-shot symmetric kernel (each rank reads all peers' symmetric buffers and reduces locally) and two-shot RS→AG, generalized past DDA's exactly-8-rank/ncclSum/3-dtype gates, with LL128 as the large-buffer bandwidth path (~95% of peak). *Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms* (arXiv 2507.04786, 2025, not peer-reviewed).

**Measured perf vs baseline.** LL128 sustains **~95% of peak intra-node bandwidth, within ~5% of Simple** (baseline: NCCL A100/H100 NVLink); multimem advantage concentrated at small sizes only (inert on AMD).

**(b) Code evidence (PARTIAL).** Symmetric kernel family present: `src/device/symmetric/kernel.cuh:9-42` declares AllReduce `AGxLL_R`/`RSxLD_AGxST`, ReduceScatter `LD`, etc.; `src/device/symmetric/all_reduce.cuh` implements the two-shot RSxLD_AGxST core; analytic cost model in `src/sym_kernels.cc:429,460` (two-shot busBytes `2*nBytes*(nRanks-1)/nRanks`, one-shot `(nRanks-1)*nBytes`). **Missing:** LL128 is not in the symmetric protocol axis — `src/sym_kernels.cc:152 rcclSymkProto` has only `_LL` and `_Simple` (`_Count=2`); grep `LL128` in `src/device/symmetric/`, `sym_kernels.cc`, `symmetric_sched.cc` → none. LL128 lives only in the legacy prims path (`src/device/prims_ll128.h`). The DDA one-shot/two-shot (`all_reduce_dda.h`) is gated to 8 ranks/ncclSum/3 dtypes (`dda_all_reduce_ipc.cu:133-163`). MC/TMA paths are NVIDIA-only (`src/device/symmetric/generate.py:106` "no NVLS/multimem on ROCm").

**(c) Implementation plan.**
1. Add an LL128 protocol entry to the symmetric axis: extend `rcclSymkProto` in `src/sym_kernels.cc:152` to `_LL128`, wire tuning entries (`baseLat`/`smBw`/`withinPeakFactor`), and add a symmetric LL128 kernel variant via `src/device/symmetric/generate.py` + `kernel.cuh`.
2. Widen the DDA/symmetric AR eligibility beyond 8 ranks: parameterize `kDdaNranks` (currently `dda_all_reduce_ipc.cu:136`) to accept 2/4/8, and relax the ncclSum-only gate for LD/ST one-shot where reduce ops are associative (add ncclMax/Min/Prod paths in `CollCommon.h vecElementAdd` generalization).
3. **Env gate:** `RCCL_SYM_LL128_ENABLE`, `RCCL_DDA_NRANKS_RELAX`.
4. **Validate:** `all_reduce_perf`/`reduce_scatter_perf -b 1M -e 256M -g 4` and `-g 2`, confirm LL128 selection and ≥ ~95%-of-peak bus BW; regression-test 8-rank ncclSum path unchanged.

**Effort:** Medium (1–2 wk). **Risk:** Medium — LL128 needs guaranteed 128B atomic writes on XGMI (verify on gfx942/gfx950; NCCL disables LL128 where atomicity isn't guaranteed).

---

### Rank 4 — Log-round bandwidth-optimal RS/AllReduce (recursive-halving / circulant skips)

**(a) What it is + citation.** Round- and bandwidth-optimal non-pipelined schedule for fully-connected systems: rank r exchanges with (r ± s_k) mod p over ⌈log2 p⌉ rounds while still moving p−1 blocks/rank (BW lower bound). For p=8: **3 RS rounds vs ring's 7** at equal volume. AllReduce = RS + reversed AG in 2⌈log2 p⌉ rounds. *Optimal, Non-pipelined Reduce-scatter and Allreduce Algorithms* (Träff, arXiv 2410.14234, 2024, preprint).

**Measured perf vs baseline.** Proven optimal in rounds (⌈log2 p⌉) and volume (p−1 blocks); matches/beats ring and standard MPI RS/AR in message-passing benchmarks. **No single-node GPU speedup number** (algorithmic/MPI result) — hence medium confidence.

**(b) Code evidence (ABSENT).** Algorithm enum fixed at 7 (`src/init.cc:101` `{Tree,Ring,CollNetDirect,CollNetChain,NVLS,NVLSTree,PAT}`); no recursive-halving/circulant/butterfly variant. Ring RS is a linear p−1 loop `src/device/reduce_scatter.h:128-129`. Grep `circulant|recursive-halv|rabenseifner|butterfly` across the tree (excl. `.md`) → none.

**(c) Implementation plan.**
1. Add a new algorithm id (extend the `RCCL_DIRECT_ALLGATHER = NCCL_NUM_ALGORITHMS` pattern in `src/include/rccl_common.h:63`) — an RCCL-local algo enum beyond the fixed 7, avoiding perturbing the NCCL enum.
2. New IPC kernel `src/dda_recursive_halving_ipc.cu` + `.../reduce_scatter/rhd_dda.h`: for each of ⌈log2 p⌉ rounds, IPC-exchange the halving-derived partner slice and reduce, reusing `IpcGpuBarrier` and `CollCommon.h` peer-pointer copy/reduce.
3. Dispatch in `src/collectives.cc` (RS `:576`, AR `:440`) behind an env gate for p∈{2,4,8}.
4. Cost model `src/graph/tuning.cc:1147` uses `latCount = numPipeOps` for ring; add a log-p latCount term so the selector prefers RHD at sync-bound medium sizes.
5. **Env gate:** `RCCL_RHD_ENABLE`.
6. **Validate:** `reduce_scatter_perf`/`all_reduce_perf -b 256K -e 64M -g 8`; confirm fewer sync rounds → lower latency in the medium band vs ring; verify bit-exact reduction (this is lossless, unlike Rank 1).

**Effort:** Medium (1–2 wk). **Risk:** Medium — gains over an already-good XGMI ring at p=8 may be modest; correctness of halving partner derivation for non-power-of-two p needs care.

---

### Rank 5 — MSCCL++ 1PA/2PA/2PR one-sided-async intra-node collectives (non-multimem)

**(a) What it is + citation.** Thread-block DSL synthesizing one-sided/async intra-node collectives: **1PA** (LL-packet all-pairs push, relaxed sync, small msgs), **2PA** (rotating-buffer RS+AG), **2PR** (ring RS+AG with reduction overlapped against DMA copy, large msgs). Portable MemoryChannel (P2P load/store) / PortChannel (DMA). *MSCCL++: Rethinking GPU Communication Abstractions for AI Inference* (ASPLOS 2026, **peer-reviewed**, arXiv 2504.09014).

**Measured perf vs baseline.** On MI300X vs **RCCL 2.20.5**: up to **3.8× (small)** and **2.2× (large)** AllReduce; geomean 2.08× vs RCCL, up to 5.4×. (Exclude the SwitchChannel/2PA-Switch NVSwitch-multimem variant — inert on AMD.)

**(b) Code evidence (PARTIAL).** MSCCL++ was **removed** on this branch (`CHANGELOG.md:45`); `mscclLoadAlgo/RunAlgo/UnloadAlgo` are no-op WARN stubs (`src/misc/api_trace.cc:194-236`). Grep `mscclpp|MemoryChannel|PortChannel|LLPacket` in `src/` → none. The functional overlap is DDA: flat one-shot <256 KB + tree two-shot ≥256 KB (`dda_all_reduce_ipc.cu:31,55`), but **barrier-synchronized** (`IpcGpuBarrier.syncOnSameBlockIdx`) not one-sided/relaxed-LL, gated to 8 ranks/ncclSum, **no ring (2PR) variant**, no compute-overlap. Symmetric family provides LL/LD-ST but is NCCL-upstream (`ncclLsaBarrier`), not the MSCCL++ DSL.

**(c) Implementation plan.**
1. Add a **2PR ring RS+AG with reduction/DMA overlap** as a new IPC kernel `src/dda_ring_all_reduce_ipc.cu` + `.../all_reduce/ring2p_dda.h` (the genuinely missing variant vs DDA's flat/tree), overlapping the reduce of chunk k with the copy of chunk k+1.
2. Add a relaxed-LL 1PA one-shot path variant reusing the weak-fence flags in `src/include/ipc_gpu_barrier.h:63-79` (`setFlagNoMemFence`) to approximate one-sided semantics without a full DSL.
3. Dispatch/tiering in `src/collectives.cc` by size: 1PA <~32 KB, DDA-tree/2PA 32 KB–1 MB, 2PR ≥1 MB.
4. **Env gate:** `RCCL_RING2P_ENABLE`.
5. **Validate:** `all_reduce_perf -b 1K -e 1G -g 8`, compare against DDA-only and default RCCL.

**Effort:** Large (2–3 wk). **Risk:** Medium — do NOT port SwitchChannel; ring-overlap correctness and the 8-rank generalization need care. Marginal single-node value over Rank 1 + existing DDA is the main uncertainty.

---

### Rank 6 — NCCLZ decoupled quantization + entropy-coded collectives

**(a) What it is + citation.** Adds a compression layer that **decouples quantization from entropy coding** (lossless entropy coding on top of quantized values), pipelined separately, to shrink XGMI bytes/hop further than fixed block-quant. *NCCLZ: Compression-Enabled GPU Collectives with Decoupled Quantization and Entropy Coding* (arXiv 2605.12396, 2026 — **citation could not be verified; ID is anomalous**).

**Measured perf vs baseline.** Reports speedups from reduced bytes vs uncompressed NCCL collectives; **exact numbers not confirmed** (abstract-level only).

**(b) Code evidence (ABSENT).** Grep `NCCLZ|entropy|huffman|rANS|tANS|arithmetic.cod|codebook|bitpack` in `src/` → none. All `compress` hits are code-object/fatbin build flags (`CMakeLists.txt:856-857`, `src/device/Makefile:27`) or NIC CQE compression. `src/include/algorithms/` has only DDA subdirs.

**(c) Implementation plan.** Build on Rank 1's codec: after block-quant, add an optional entropy stage (e.g. rANS) as a device pass in `src/include/algorithms/all_reduce/quick_reduce_codec.h`, pipelined via a second kernel phase. Gate `RCCL_QUICKREDUCE_ENTROPY`. Validate byte-reduction and net bus-BW on `all_reduce_perf` ≥16 MB. **Effort:** Large. **Risk:** High — entropy-coding on GPU has data-dependent throughput that can erase the byte savings; **lowest priority pending citation verification.**

---

### Rank 7 — Multi-lane multi-process-per-GPU CPX-die-parallel AllReduce

**(a) What it is + citation.** Assign multiple processes per GPU, each driving a separate comm "lane" over buffer partition s/PPG; in CPX mode each XCD is a logical GPU, exploiting die-locality; structured RS→lane-AR→AGv. *Optimizing Allreduce ... with Multiple Processes per GPU* (arXiv 2508.13397, 2025, preprint).

**Measured perf vs baseline.** Up to **33× on MI300A CPX** (multi-lane + 2 PPG) vs **Cray MPICH MPI_Allreduce** — but baseline **lacks IPC** so is heavily inflated; 1.17× on MI300A SPX; measured on **MI300A APU, not MI300X** (XGMI-clique numbers would differ).

**(b) Code evidence (ABSENT).** CPX in `src/` is only topology parsing (`xml.h:22`, `xml.cc:434`). Grep `PPG|nProcsPerGpu|ranksPerGpu` → none. `isMultiRankGpu` (`src/include/comm.h:682`) is a permission gate that disables NVLS (`init.cc:1521-1533`); no lane/partition pipeline. Every `lane` token is a hardware warp lane.

**(c) Implementation plan.** Largely a launcher/host-orchestration concern (MPI + numa binding runs standard ranks). In-RCCL work: an s/PPG buffer-partition RS→AR→AGv path keyed off `isMultiRankGpu` in `src/collectives.cc`. **Env gate:** `RCCL_MULTILANE_PPG`. Validate `all_reduce_perf -g 1` under `mpirun -np 16` CPX with IPC enabled (fair baseline). **Effort:** Medium. **Risk:** Medium — the headline number is baseline-inflated and APU-specific; real MI300X gain vs IPC-capable DDA is unproven.

---

### Rank 8 — ForestColl / TE-CCL / TACCL synthesized single-node schedules

**(a) What it is + citation.** Offline synthesis of bandwidth-optimal P2P-send schedules for a fully-connected node. **ForestColl** (spanning-tree packing, polynomial-time, NSDI 2026, peer-reviewed, arXiv 2402.06787; tested on AMD MI250). **TE-CCL** (multi-commodity-flow MILP, SIGCOMM 2024, peer-reviewed). **TACCL** (sketch-guided ILP, NSDI 2023, peer-reviewed).

**Measured perf vs baseline.** ForestColl: claims always-optimal algorithmic BW, beats NCCL/RCCL (exact single-node factors not cleanly extractable). TE-CCL: **3.18× vs RCCL** algorithm BW on a GPU testbed (headline emphasizes larger fabrics). TACCL: 12–35% (1KB–1MB), 61%–3.4× (>1MB) vs NCCL. **Single-node gains over an already-near-optimal ring may be small.**

**(b) Code evidence (ABSENT).** Enum fixed at 7 (`src/init.cc:101`); no synthesis engine. Grep `forest|spanning.?tree|multi-commodity|taccl|MILP|solver|gurobi` in `src/` → none. `src/graph/trees.cc` is the classic NCCL double-binary btree. The one schedule-executor path (MSCCL) is removed/no-op (`src/misc/api_trace.cc:194-236`). `src/scheduler/` is runtime kernel-plan batching, not a schedule interpreter.

**(c) Implementation plan.** (1) Offline: run the open-source synthesizer (ForestColl/TE-CCL) against a `tools/topo_expl`-dumped XGMI topology to emit a P2P send schedule. (2) In-RCCL: build a **schedule executor** (the removed MSCCL interpreter role) — a new `src/scheduler/synth_sched.cc` that parses a JSON send-step program and drives IPC P2P sends. **Env gate:** `RCCL_SYNTH_SCHEDULE_FILE`. Validate `all_gather_perf`/`alltoall_perf -g 8` with a synthesized schedule vs ring. **Effort:** Very Large (executor + integration). **Risk:** High — requires resurrecting a removed subsystem for uncertain single-node ring-beating gains; **lowest structural priority.**

---

## 4. Confidence & Citations Appendix

| Gap | Confidence | Why | Citation verification |
|---|---|---|---|
| Rank 1 QuickReduce cluster | **High** | Multiple ROCm-blog + open-source (mk1-project) measured single-node MI300X/MI355X results vs named RCCL baseline; MSCCL++ peer-reviewed anchor. AMD-native, no NVIDIA primitives. | ROCm blogs = engineering (not peer-reviewed) but reproducible/open-source. Sub-variants Flash Comm / EQuARX / FlashComm V2 are **NVIDIA-measured preprints** (down-weighted within cluster; ideas portable, need HIP port). MoRI intra-node path in-scope; RDMA path excluded. |
| Rank 2 CPX/XCD tuner | **High** | ROCm-blog measured 2–3.1× small-msg latency vs RCCL default; tuner scaffold already in `src/`. | Engineering blog, not peer-reviewed; bandwidth-tier numbers are AMD-published. |
| Rank 3 Symmetric LD/ST + LL128 | **High** (code) / Med (perf transfer) | Kernels present in `src/device/symmetric/`; gap is LL128-in-symmetric + gate relaxation. | Perf numbers are **NVIDIA NVLink** (A100/H100); LL128 XGMI atomicity must be verified on gfx942/gfx950. |
| Rank 4 Recursive-halving/circulant | **Medium** | Proven optimal in rounds/volume; **no GPU single-node number**. | **Preprint** (Träff arXiv 2024); author has related peer-reviewed work. |
| Rank 5 MSCCL++ 1PA/2PA/2PR | **High** | Peer-reviewed (ASPLOS 2026), strong MI300X-vs-RCCL numbers; DDA covers a subset. | Verified peer-reviewed. Exclude SwitchChannel (NVSwitch multimem). |
| Rank 6 NCCLZ | **Low** | Concept plausible but entropy-coding GPU throughput is data-dependent. | **Citation could not be verified** — arXiv ID 2605.12396 is anomalous (future-dated, non-standard); treat as unverified. Down-ranked accordingly. |
| Rank 7 Multi-lane PPG | **Medium** | Real CPX mechanism, but 33× is baseline-inflated (MPICH lacks IPC) and **MI300A APU**, not MI300X. | Preprint; baseline not IPC-capable, so absolute numbers pessimistic/non-comparable. |
| Rank 8 ForestColl/TE-CCL/TACCL | **Medium** (research) / Low (single-node ROI) | All peer-reviewed; ForestColl tested on AMD MI250. But single-node ring is already near-optimal and executor was removed. | Verified peer-reviewed (NSDI'26 / SIGCOMM'24 / NSDI'23). Single-node factor for ForestColl not cleanly extractable from PDF. |

**Flagged / could-not-verify:** NCCLZ citation (Rank 6) — anomalous arXiv ID, abstract-level only; do not schedule before confirming the paper exists and its single-node/intra-node applicability. Flash Communication, EQuARX, FlashComm V2 (within Rank 1) — measured on NVIDIA only; treated as portable *ideas* feeding the AMD-native QuickReduce codec, not standalone AMD-measured results.

**File references in this report were verified by reading the tree:** `src/dda_all_reduce_ipc.cu:31,55,133-163`, `src/include/algorithms/all_reduce/all_reduce_dda.h:21,43,56,76,90`, `src/include/algorithms/CollCommon.h:49,83,116`, `src/init.cc:101`, `src/graph/tuning.cc:858-861,1147-1148`, `src/collectives.cc:246,309,440-442,576`, `src/include/rccl_float8.h:71-149,381`, `src/nccl.h.in:593-596`, `src/sym_kernels.cc:152,429,460,486`, `src/plugin/tuner/csv_tuner.cc:44,57-59`, `src/include/rccl_common.h:63`, `src/device/symmetric/generate.py:106`, `tuner/rccl_tuner_gfx950.csv`, `src/misc/api_trace.cc:194-236` (all confirmed present).