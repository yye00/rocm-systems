# State of the Art in GPU Collective Communication (2023–2026)

**A companion briefing to `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md`.**
Scope: the newest algorithmic, systems, network-co-design, and observability advances in GPU collective communication, with **AMD/ROCm/RCCL applicability** and **confidence/vendor-hype flags** per item.

> **Two caveats to read first.**
> 1. **Version skew.** Public write-ups often say "MSCCL++ / NPKit are in RCCL." That is true of *released* RCCL (≈2.26–2.28) but **not of the current `develop` branch (2.30.4)**, whose CHANGELOG has a `Removed` section for both MSCCL/MSCCL++ (`CHANGELOG.md:45`) and NPKit, and whose `src/` contains **zero** `mscclpp` files (verified). Where this doc says a feature "is in RCCL," read it as "is in shipping RCCL releases," and see the branch note.
> 2. **Vendor numbers.** Multipliers from vendor blogs (SHARP "9×", TRT-LLM "3×", ZeRO++ "4×", AMD Pollara "+25%") are best-case, component-level, or single-source. They are labeled as such; peer-reviewed and on-hardware results are called out separately.

---

## Top-line ranking for an AMD/RCCL operator in 2026

| # | Advance | Category | Venue / status | AMD/RCCL status | Confidence |
|---|---------|----------|----------------|-----------------|------------|
| 1 | **MSCCL++** GPU-driven programmable collectives | Runtime | ASPLOS'25 + prod | In released RCCL (removed on `develop`); 3.8× small-msg on MI300X | High |
| 2 | **"Big Send-off" (PCCL)** hierarchical + learned selection | Algorithm/lib | SC'25 (peer-reviewed) | **Proven on Frontier MI250X**: ~10× AR, 40–60% E2E training vs RCCL | High |
| 3 | **QuickReduce / custom one-shot AllReduce** | Inference | AMD blog + vLLM/SGLang | **AMD-native**, ~2.25–3× vs RCCL small-msg | High |
| 4 | **FLASH** incast-aware all-to-all | Algorithm | arXiv'25 | **Measured on MI300X**, 1.18–4.48× vs RCCL FanOut (MoE) | Med-High |
| 5 | **HiCCL** compositional hierarchical library | Algorithm/lib | IPDPS'25 (peer-reviewed) | Portable NV/AMD/Intel; **1.55× RCCL** | High |
| 6 | **RCCLX** (Meta) DDA + FP8 collectives | Runtime | Meta eng, 2026 | **AMD-native** (Torchcomms); ~9–10% TP-inference latency cut | High |
| 7 | **TE-CCL** schedule synthesis (MILP flow) | Algorithm | SIGCOMM'24 (peer-reviewed) | Emits RCCL-beating schedules: **3.18× vs RCCL** | High |
| 8 | **DeepEP / COMET** MoE all-to-all | MoE | code + MLSys'25 | ROCm/rocSHMEM DeepEP port exists, *experimental* | High perf / Med ROCm |
| 9 | **Rina** ring-AllReduce + P4 in-network aggregation | Network | ICNP'24 (peer-reviewed) | The **SHARP substitute** for AMD (SHARP is IB/NVIDIA-locked) | High |
| 10 | **Fault-tolerant collectives** (FTAR / R2CCL / Mycroft) | Resilience | 2025 (mixed) | **Gap** — RCCL has no native equivalent | High (gap is real) |
| 11 | **UEC 1.0 + AMD Pensando Pollara 400** | Network | Ratified spec + product | **AMD-native scale-out path**; NIC-side in-place AllReduce | High spec / Med claims |
| 12 | **Swing / Bine Trees / Trivance** | Algorithm | NSDI'24 / SC'25 / arXiv'26 | Topology-gated (torus/dragonfly) — **no win on XGMI cliques** | High / High / Med |

---

## 1. Pure algorithms (beyond Ring / double-binary Tree / Rabenseifner / Bruck)

- **Swing** — NSDI'24 (De Sensi et al.), [arXiv:2401.09356](https://arxiv.org/abs/2401.09356). "Swings" partners between torus directions to keep hop-distance low: `2·log₂(p)` steps, `2n` bytes (bandwidth-optimal), up to **3×** on 2D/3D tori & HyperX. **Explicitly equals recursive-doubling on a non-blocking fat tree** → *no benefit inside a fully-connected XGMI clique*. AMD relevance only if the inter-node fabric is a torus. **Peer-reviewed, high confidence.**
- **Bine Trees** — SC'25, [arXiv:2508.17311](https://arxiv.org/abs/2508.17311). Binomial *negabinary* trees pick partners via base-(−2) to keep traffic on local links; up to **5×** and **33% less global-link traffic** on four real supercomputers (Dragonfly/Dragonfly+/fat-tree/torus). Applies to hierarchical/oversubscribed AMD clusters. **Peer-reviewed.**
- **Trivance** — arXiv'26 [2602.17254] (*preprint*). `⌈log₃ n⌉`-step multiport AllReduce using both ring ports with joint reductions; matches Bruck's step bound but **3× less congestion**; 5–30% CT for ≤8–128 MiB. Torus/multiport; power-of-3 ideal. **Preprint, med confidence.**
- **"Short-circuiting rings for low-latency AllReduce"** — arXiv'25 [2510.03491] (*distinct from Swing despite the title*). Uses **photonic circuit switching** to short-cut the physical ring for recursive-doubling. Hardware-gated (needs reconfigurable optics). **Preprint.**
- **Recursive-multiplying / multi-radix** — Ruefenacht et al. (*Parallel Computing*) generalizes recursive-doubling off power-of-2 (8–40% on Cray XC30); Jocksch et al. **IJHPCA'26** ([doi](https://journals.sagepub.com/doi/10.1177/10943420251363423)) adds "cyclic copy-in reduction," ~½-order-of-magnitude vs MPICH/OpenMPI on **dual-socket AMD EPYC**, matches NCCL for long messages. General-purpose, AMD-tested. **Peer-reviewed / high.**
- **FLASH** — arXiv'25 [2505.09764]. Incast-aware all-to-all: shuffle/load-balance over fast intra-server fabric before crossing slow inter-server links (Birkhoff decomposition, 32 µs to schedule vs TACCL's ~1 hr). **Implemented on ROCm/RCCL/MSCCL, measured on 4×8 MI300X: 1.18–4.48× vs RCCL FanOut** on Megatron-LM MoE. **Strongest direct-AMD all-to-all result. Med-high.**
- **PAT (Parallel Aggregated Trees)** — arXiv'25 [2506.20252] (NCCL lead author). Tree AllGather/ReduceScatter for *any* rank count, log transfers for small messages. **Already the `NCCL_ALGO_PAT` path in RCCL** (see survey §1.2). Portable. **High (algorithm) / med (peer-review).**
- **Lower bounds:** the AllReduce bandwidth term `2·T_B*(N)` (Patarasuk & Yuan, JPDC'09) is unbeatable; Swing/Trivance/short-circuiting attack the *latency/congestion* term. **Open problem (flagged):** no tight provable bound captures the realistic *two-tier* (fast XGMI + slow RDMA) model — practice still uses BlueConnect-style hierarchical decomposition.

## 2. Hierarchical libraries (most AMD-relevant algorithm class)

- **"The Big Send-off" (PCCL)** — SC'25, [arXiv:2504.18658](https://arxiv.org/abs/2504.18658). Hierarchical library with **learning-based adaptive algorithm selection**, evaluated on **2,048 GCDs of Frontier (AMD MI250X)**: ~**10× all-reduce and 40–60% end-to-end GPT-3-scale training vs RCCL** (its "168× reduce-scatter" is a small-message corner case, not the headline). **Strongest empirical AMD evidence in this briefing. Peer-reviewed, high.**
- **HiCCL** — IPDPS'25, [arXiv:2408.05962](https://arxiv.org/abs/2408.05962). Compositional API (multicast/reduction/fence primitives factorized per network level) + striping/pipelining; **17× over GPU-aware MPI, 1.15× NCCL, 1.55× RCCL, 12.1× oneCCL**, portable NV/AMD/Intel. **Peer-reviewed, high.**
- **BlueConnect lineage** — MLSys'19; the intellectual basis for decomposing AllReduce into per-fabric reduce-scatter/all-gather. Directly underpins survey §2.2.

## 3. LLM training & inference collectives

**Compute–comm overlap:** **FLUX** (ByteDance, [arXiv:2406.06858](https://arxiv.org/abs/2406.06858)) fuses AllGather/ReduceScatter into GEMM tiles, 1.24–1.66× — *CUDA/CUTLASS/NVSHMEM-only*; **Domino** (MS DeepSpeed, [2409.15241](https://arxiv.org/abs/2409.15241)) 1.3× via chunk pipelining; **T3** (ASPLOS'24) is a *hardware* proposal (near-memory reduce), research-only. Concepts portable to RCCL; implementations are not.

**Compression:** **THC** (NSDI'24, [2302.08545](https://arxiv.org/abs/2302.08545)) — homomorphic compression that sums *without* decompression, enabling in-network aggregation; 1.40–1.47× to target accuracy, peer-reviewed. **ZeRO++** (2306.10209) quantized weights/grads, "4× less communication" (*volume*, not time). **fp8 AllReduce** needs stochastic rounding + scaling to avoid underflow.

**MoE all-to-all:** **DeepEP** (DeepSeek) — NVSHMEM upstream; **[ROCm/DeepEP](https://github.com/ROCm/DeepEP)** rocSHMEM port for MI300X exists but is *experimental*. **COMET** (MLSys'25, [2502.19811](https://arxiv.org/abs/2502.19811)) 1.96× layer / 1.71× E2E, deployed at 10k-GPU. **Tutel** (MLSys'23) adaptive parallelism + 2D hierarchical all-to-all.

**Inference low-latency AllReduce (AMD at parity here):** **QuickReduce** (AMD-native, inline INT4/6/8, [ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html)) up to 2.25–3× vs RCCL; vLLM **CustomAllreduce** wins < ~512 KB; **TRT-LLM MultiShot** ~3× (NVSwitch-only, hype-flagged); **NCCL 2.27 symmetric memory** + SHARP (NVIDIA-only). **KV disaggregation** (DistServe OSDI'24, Mooncake, NIXL/Dynamo) moves KV-cache P2P over RDMA — portable via UCX.

## 4. Programmable / synthesized runtimes

- **MSCCL++** — ASPLOS'25, [arXiv:2504.09014](https://arxiv.org/abs/2504.09014). Async put/signal primitives + DSL + GPU-side executor; geomean **1.7× (up to 5.4×)** collectives, **3.8× small / 2.2× large on MI300X**. **In released RCCL** (dispatch above `RCCL_MSCCLPP_THRESHOLD`) — but **removed on this `develop` branch** (see caveat). The single most actionable "restore" item for RCCL (survey §2.5).
- **Schedule synthesis:** **TE-CCL** (SIGCOMM'24, [doi](https://dl.acm.org/doi/10.1145/3651890.3672249)) casts scheduling as multi-commodity-flow MILP — **3.18× vs RCCL, 2.14× vs TACCL**. **TACCL** (NSDI'23, ILP + sketches). **TACOS** topology-aware. All emit onto MSCCL/MSCCL++ executors for AMD topologies. **Peer-reviewed.**

## 5. In-network aggregation & network co-design

- **NVIDIA SHARP** — in-switch reduction on InfiniBand/Quantum (and Spectrum-X Ethernet); vendor ~5×/9× MPI, ~2× AR bandwidth. **NVIDIA/IB-locked; unusable by AMD** — it is the named cause of the RCCL↔NCCL gap (SemiAnalysis).
- **Rina** — ICNP'24, [arXiv:2407.19721](https://arxiv.org/abs/2407.19721). Fuses P4/Tofino in-network aggregation into *ring*-AllReduce (rack = one ring node); up to 6× vs ring, linear per-switch gains. **GPU-agnostic over RDMA → the practical SHARP substitute for AMD. Peer-reviewed.** Lineage: SwitchML (NSDI'21, 5.5×), ATP (NSDI'21, multi-tenant), NetReduce (ASPLOS'23).
- **Ultra Ethernet 1.0** (June 2025) + **In-Network Collectives (INC)** — AMD is a member; open SHARP-over-Ethernet path. **Broadcom Tomahawk Ultra** (51.2 Tbps, 250 ns, in-switch AllReduce) is merchant silicon AMD could sit behind — *needs RCCL to emit the INC protocol.*
- **AMD Pensando Pollara 400 / Vulcano 800** — first UEC-ready AI NIC; **SSDK+RCCL do NIC-side in-place AllReduce** (NIC-offload, *not* switch in-network reduction) + UEC MRC transport. Vendor: +25% vs RoCEv2, ~10% vs CX7 (single-source). **AMD's live offload path.**
- **UALink 200G 1.0** (April 2025) — open scale-up to 1,024 accelerators; **AMD founding member**; AMD's answer to NVLink. Silicon forthcoming.
- **Optical/OCS** — TopoOpt (NSDI'23, 3.4×), Google Lightwave (SIGCOMM'23), TPU v4 (ISCA'23), PCCL-photonic (2509.15450), SWOT (CoNEXT'26), RECCL (JOCN'25). Co-design principle transfers; hardware assumption (reconfigurable optics) does not apply to today's XGMI/RDMA. **Research/TPU-specific.**

## 6. Scale & resilience

- **NCCLX** — Meta, [arXiv:2510.20171](https://arxiv.org/abs/2510.20171). The 100k-GPU blueprint: **CTran** zero-copy SM-free transport, **DQPLB** distance-aware QP load balancing, **FTAR** fault-tolerant AllReduce; 12% step-latency cut, 11× faster init at 96k, 15–80% decode gains. NVIDIA/NVSHMEM-specific but the reference gap-list.
- **RCCLX** — Meta, 2026 ([eng blog](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/)). AMD counterpart to NCCLX (Torchcomms backend): DDA peer-memory O(1) small AllReduce + FP8; ~9–10% TP-inference latency cut. **AMD-native.**
- **Fault tolerance (a gap RCCL shares):** **R2CCL** (12.18× lower overhead vs AdapCC), **OptCC** ("Don't Let a Few Network Failures Slow the Entire AllReduce," 2–6% overhead), **Mycroft** (SOSP'25, localizes failing rank/link). NCCL/RCCL have **no native fault tolerance** — a single port-down stalls the communicator. **High strategic priority at scale.**
- **Production infra lessons:** MegaScale (NSDI'24), Meta RoCE (SIGCOMM'24: congestion control belongs in the collective library at 400G+), ByteDance straggler study (2505.05713: **most slowdowns are compute, not network**).

## 7. Benchmarking & observability

- **AMD-native:** **rccl-tests** (algbw/busbw), **TransferBench** (per-xGMI-link bandwidth — reveals ~75% link efficiency and slowest-link-limits-collective), **rocprofiler-systems / -compute / rocprof**, **RCCL Tuner API** (CPX per-size selection). Note **NPKit was removed on this `develop` branch** (use the profiler plugin API instead).
- **Cross-vendor / portable:** **mscclpp-benchmarks**, PyTorch Profiler/Kineto (works on ROCm via roctracer), **Holistic Trace Analysis** (comm-compute overlap %, trace-driven). NCCL Profiler Plugin API / NCCL Inspector / ncclsee are NVIDIA-first but concept-portable.
- **Production tracing at scale:** ARGUS (Tencent, <2% overhead, kernel-level), PerfTracker/EROICA (Alibaba), VCCL (µs-granularity RDMA telemetry). Recurring finding: **compute stragglers masquerade as network faults through collective dependencies** — a caveat when diagnosing RCCL.

## 8. What this means for RCCL (ties to the survey)

1. **Restore MSCCL++ (survey §2.5).** RCCL *had* it and just removed it; it's proven on MI300X (3.8×) and is the delivery vehicle for TE-CCL/TACCL-synthesized schedules. Highest-leverage, lowest-novelty.
2. **Hierarchical decomposition (§2.2) is validated on AMD** by Big Send-off/PCCL (Frontier) and HiCCL — pursue it, but benchmark against tuned Ring/Tree since gains are regime-specific.
3. **Incast-aware all-to-all (§2.4) is demonstrated on MI300X** by FLASH (1.18–4.48× vs RCCL FanOut) — the clearest independent MoE win.
4. **Add fault-tolerant collectives (§2.7)** — a survivability gap, not a speed gap; arguably top priority for 10k+-GPU AMD clusters.
5. **In-network reduction path for AMD is Rina-style P4-INA / UEC-INC**, since SHARP is off-limits.
6. **The full-duplex-PCIe finding (§2.3) stands** — nothing in this research contradicts it.

## Confidence & hype flags (summary)

- **Peer-reviewed anchors:** Swing (NSDI'24), Bine Trees (SC'25), Big Send-off (SC'25), HiCCL (IPDPS'25), TE-CCL (SIGCOMM'24), COMET (MLSys'25), MSCCL++ (ASPLOS'25), Rina (ICNP'24), SwitchML/ATP (NSDI'21), THC (NSDI'24), Mycroft (SOSP'25), TopoOpt (NSDI'23), TPU v4 (ISCA'23).
- **Preprints (author numbers only):** Trivance, short-circuiting rings, PAT, FLASH, RCCLX, NCCLX, R2CCL, OptCC, ARGUS, PerfTracker.
- **Vendor/best-case (discount accordingly):** SHARP 5×/9×, TRT-LLM MultiShot 3×, ZeRO++ "4×", FLUX "96% overlap", NCCL 2.27 "9×", AMD Pollara +10–25%, PCCL "168×" (small-message corner).

---

*Compiled 2026-07-01 from a multi-agent literature sweep (algorithms, LLM-scale, systems/infra, network co-design, benchmarking), reconciled against the RCCL `develop` (2.30.4) source tree. Preprint IDs dated 2025–2026 reflect this environment's clock. All performance figures are attributed to their stated baseline; confidence and peer-review status are stated per item.*
