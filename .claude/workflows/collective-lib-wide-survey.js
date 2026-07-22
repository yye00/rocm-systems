export const meta = {
  name: 'collective-lib-wide-survey',
  description: 'Wide-search funnel over a collective-comm library: code inventory + literature sweep, screening gate, per-candidate code check, adversarial verification, completeness critic',
  phases: [
    { title: 'Survey', detail: 'parallel code-inventory and themed literature agents' },
    { title: 'Screen', detail: 'Gate 1: dedup + priority ranking' },
    { title: 'CodeCheck', detail: 'Gate 2: per-candidate in-code presence check' },
    { title: 'Verify', detail: 'Gate 3: adversarial 2-lens refutation' },
    { title: 'Critic', detail: 'completeness critic over the funnel output' },
  ],
}

const cfg = Object.assign({
  repoPath: '/home/user/rocm-systems/projects/rccl',
  component: 'RCCL (AMD ROCm fork of NVIDIA NCCL), rocm-systems develop branch, RCCL 2.30.4 for ROCm 7.14.0 (NCCL 2.28.9/2.29.7/2.30.4 compatibility)',
  priorSurvey: '/home/user/rocm-systems/projects/rccl/RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md',
  sotaDoc: '/home/user/rocm-systems/projects/rccl/COLLECTIVE_COMM_STATE_OF_THE_ART_2026.md',
  maxCandidates: 14,
  agentModel: 'opus',
}, (args && typeof args === 'object') ? args : {})

// Pin every subagent to this model so the funnel's adversarial code cross-check
// and literature verification run on the intended tier, regardless of what the
// session model resolved to at launch time. Override via args.agentModel.
const MODEL = cfg.agentModel

const HW_CONTEXT = `Target hardware context: AMD MI300X/MI325X/MI355X nodes (8 GPUs, fully-connected XGMI clique intra-node), inter-node via RDMA NICs (InfiniBand or RoCE/Ultra-Ethernet, often rail-optimized), plus PCIe-only platforms (MI210). NVIDIA-only features (NVLS multimem, TMA, C2C) are inert on AMD.`

const PRIOR_COVERED = `Items ALREADY covered by the prior survey (do not re-propose as new; only report material NEW developments about them): Rabenseifner recursive halving-doubling; hierarchical multi-level AllReduce (BlueConnect, HiCCL, PCCL/Big Send-off, MVAPICH multi-lane); PCIe-switch-locality staging (Panda direction — verified mostly covered by RCCL full-duplex link model); staged/aggregated AllToAll incl. Bruck and FLASH; restore MSCCL/MSCCL++ programmable executor (+ TACCL, TE-CCL); Swing (NSDI'24); fault-tolerant collectives (NCCLX FTAR, R2CCL, OptCC, Mycroft).`

const INV_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: '<=300 word prose summary of what this area implements' },
    algorithms: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, files: { type: 'string', description: 'file:line anchors' },
      mechanism: { type: 'string' }, gating: { type: 'string', description: 'when it is used / env gates / hw gates' },
      amd_active: { type: 'string', description: 'active on AMD hw, inert, or unverified' },
    }, required: ['name', 'files', 'mechanism'] } },
    surprises: { type: 'string', description: 'anything that contradicts the prior survey doc, or is new since it' },
  },
  required: ['summary', 'algorithms'],
}

const CAND_SCHEMA = {
  type: 'object',
  properties: {
    candidates: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      source: { type: 'string', description: 'paper/system, venue, year, arXiv id if known' },
      claim: { type: 'string', description: 'measured gain, vs WHAT baseline, on WHAT hardware — be precise' },
      mechanism: { type: 'string', description: '2-4 sentences on how it works' },
      regime: { type: 'string', description: 'message sizes / scale / topology where it wins' },
      amd_relevance: { type: 'string', description: 'why it would (not) transfer to AMD MI300-class systems' },
      already_known: { type: 'boolean', description: 'true if in the prior-covered list' },
    }, required: ['name', 'source', 'claim', 'mechanism', 'regime', 'already_known'] } },
    notes: { type: 'string' },
  },
  required: ['candidates'],
}

const SCREEN_SCHEMA = {
  type: 'object',
  properties: {
    screened: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, source: { type: 'string' }, claim: { type: 'string' },
      mechanism: { type: 'string' }, regime: { type: 'string' },
      priority: { type: 'number', description: '1 (highest) .. 5' },
      novelty: { type: 'string', enum: ['new', 'update-to-prior', 'prior'] },
      rationale: { type: 'string' },
    }, required: ['name', 'source', 'claim', 'mechanism', 'priority', 'novelty', 'rationale'] } },
    dropped: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, reason: { type: 'string' } }, required: ['name', 'reason'] } },
  },
  required: ['screened', 'dropped'],
}

const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['present', 'partial', 'missing'] },
    evidence: { type: 'string', description: 'file:line citations proving the status' },
    notes: { type: 'string', description: 'nearest existing mechanism, extension points, gates' },
  },
  required: ['status', 'evidence'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reasoning: { type: 'string', description: 'concrete evidence for the verdict' },
    corrections: { type: 'string', description: 'any corrections to the claim (numbers, baseline, hw)' },
  },
  required: ['refuted', 'confidence', 'reasoning'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    missing_areas: { type: 'array', items: { type: 'string' } },
    weak_claims: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string' },
  },
  required: ['missing_areas', 'weak_claims', 'overall'],
}

// ---------- Phase 1+2: concurrent code inventory + literature sweep ----------
phase('Survey')

const invAreas = [
  { key: 'kernels', prompt: `Inventory the collective ALGORITHMS implemented in ${cfg.component} at ${cfg.repoPath}. Your area: device kernels and algorithm enumeration. Read src/include/plugin/nccl_tuner.h (NCCL_ALGO_*/NCCL_PROTO_*), src/device/*.h (all_reduce, all_gather, reduce_scatter, broadcast, reduce, sendrecv, alltoall_pivot, alltoall_gda, alltoallv_gda, hierarchical_ag_shuffle), src/device/symmetric/ and src/include/sym_kernels.h, src/device/onerank.cu, src/device/generate.py. For each distinct algorithm/kernel family record name, file:line, mechanism, gating, and whether it is active on AMD hardware. Cross-check against the prior survey at ${cfg.priorSurvey} — independently verify, do not trust it; report any drift or omissions in 'surprises'. ${HW_CONTEXT}` },
  { key: 'graph', prompt: `Inventory the algorithm-SELECTION machinery in ${cfg.component} at ${cfg.repoPath}. Your area: topology modeling, search, and the cost model. Read src/graph/topo.cc, paths.cc, search.cc, tuning.cc, trees.cc, rings.cc, connect.cc, rome_models.cc, and src/include/graph.h. Record: how topology (XGMI/PCIe/NIC/rail) is modeled incl. link direction handling; how rings/trees are laid out; the cost model that ranks algorithms; PXN / rail-aligned routing support and whether it is active on AMD; any hierarchy awareness. Also check the tuner plugin interface (src/include/plugin/tuner/*): what CAN and CANNOT an external tuner change? Cross-check the prior survey at ${cfg.priorSurvey}; report drift in 'surprises'. ${HW_CONTEXT}` },
  { key: 'amd', prompt: `Inventory the AMD-SPECIFIC and recently-synced algorithm paths in ${cfg.component} at ${cfg.repoPath}. Your area: (a) DDA IPC one/two-shot collectives (src/include/algorithms/, src/dda_*_ipc.cu, src/collectives.cc gating), (b) the NEW DDA FABRIC path (src/dda_all_reduce_fabric.cu, src/dda_all_gather_fabric.cu, src/dda_alltoall_fabric.cu, src/dda_reduce_scatter_fabric.cu, src/fabric_init.cu, src/fabric_gpu_barrier.cu, src/fabric_mem_handler.cc) — determine what fabric (MNNVL? multi-node XGMI fabric?) it targets and its AMD status, (c) Pivot AllToAll (src/device/alltoall_pivot.h), (d) hierarchical AllGather (src/device/hierarchical_ag_shuffle.h, rccl_wrap.cc), (e) rocSHMEM GDA AllToAll(v), (f) WarpSpeed warp-specialization traffic shaping, (g) the subsystems synced from NCCL 2.28-2.30: src/gin/ (GPU-initiated networking), src/rma/ (one-sided RMA), src/nccl_device/ (device API: lsa_barrier, gin_barrier, ll_a2a), src/scheduler/ (allgatherv_sched, symmetric_sched), src/ce_coll.cc (copy-engine collectives), src/mnnvl.cc, src/sym_kernels.cc, and the new proxytrace profiler (RCCL_PROXYTRACE). For each: name, file:line, mechanism, gating, AMD status (active/inert/unverified — see CHANGELOG.md 'Known issues'). The prior survey at ${cfg.priorSurvey} predates some of these — flag anything it does not cover in 'surprises'. ${HW_CONTEXT}` },
  { key: 'delta', prompt: `Inventory REMOVED capabilities and EXTENSION POINTS in ${cfg.component} at ${cfg.repoPath}. Read CHANGELOG.md (all Unreleased sections), docs/how-to/rccl-usage-tips.rst, src/misc/api_trace.cc (msccl stubs), src/include/plugin/ (tuner, net, profiler plugin interfaces), bindings/nccl4py if present, and grep for mscclpp/MSCCL remnants. Record: what algorithm-relevant capabilities were removed recently (MSCCL/MSCCL++, NPKit, COLLTRACE) and what stubs remain; what plugin interfaces exist and precisely what each can/cannot control; env-var surface relevant to algorithm choice (RCCL_* in src/param/ or via grep). In 'surprises', report anything contradicting the prior survey at ${cfg.priorSurvey}. ${HW_CONTEXT}` },
]

const litThemes = [
  { key: 'hier', prompt: `Literature sweep — theme: HIERARCHICAL and TOPOLOGY-AWARE collective algorithms (AllReduce/AllGather/ReduceScatter). Use WebSearch/WebFetch. Cover: MVAPICH / D.K. Panda group work on PCIe-tree-hierarchy-aware and bidirectional-link-exploiting collectives (all years, incl. 2024-2026 MVAPICH-Plus releases); BlueConnect; HiCCL; PCCL/Big Send-off; multi-lane allreduce; 2D/hierarchical ring schemes (Google 2D torus, tofu); Cluster/rack-scale hierarchy. Prioritize papers with MEASURED gains vs NCCL/RCCL/MPI on GPU clusters. ${PRIOR_COVERED} Report precise claims (gain, baseline, hardware). ${HW_CONTEXT}` },
  { key: 'synth', prompt: `Literature sweep — theme: SYNTHESIZED/PROGRAMMABLE collective schedules and executors. Use WebSearch/WebFetch. Cover: MSCCL, MSCCL++ (incl. current ROCm/MI300X support status in 2026), TACCL, TE-CCL, SCCL, Blink, ForestColl, AutoCCL, TCCL, LCCL, any 2025-2026 successors; also compiler-style approaches (e.g. CoCoNet, Centauri) and tuner-search systems (e.g. nccl-tuner autotuning, Meta's NCCLX tuner). Prioritize measured gains vs NCCL/RCCL on GPU clusters, especially AMD. ${PRIOR_COVERED} Report precise claims. ${HW_CONTEXT}` },
  { key: 'a2a', prompt: `Literature sweep — theme: ALLTOALL and MoE/expert-parallel communication. Use WebSearch/WebFetch. Cover: DeepSeek DeepEP (GPU-initiated NVSHMEM-based MoE dispatch/combine — and any ROCm ports e.g. on rocSHMEM), FLASH, Bruck variants, hierarchical/aggregated AllToAll (Panda group), incast/congestion-aware A2A, MoE comm-compute overlap systems (Tutel, Comet, FlexEP, MegaScale-MoE 2025-2026), all-to-all for inference disaggregation. Prioritize measured gains vs NCCL/RCCL, especially on AMD. ${PRIOR_COVERED} Report precise claims. ${HW_CONTEXT}` },
  { key: 'bwopt', prompt: `Literature sweep — theme: BANDWIDTH/LATENCY-OPTIMAL allreduce algorithms and GPU-side fast paths. Use WebSearch/WebFetch. Cover: Swing (NSDI'24) and follow-ups (e.g. Bine/binomial-negabinary trees, De Sensi group 2025-2026); recursive halving-doubling on GPUs; one-shot/two-shot fused allreduce kernels (vLLM custom allreduce, TensorRT-LLM, TokenWeave, flux); symmetric-memory/multimem collectives and their AMD equivalents; allreduce for inference latency (small message). Prioritize measured gains vs NCCL/RCCL. ${PRIOR_COVERED} Report precise claims. ${HW_CONTEXT}` },
  { key: 'net', prompt: `Literature sweep — theme: IN-NETWORK COMPUTING, COMPRESSION, and SPARSITY for collectives. Use WebSearch/WebFetch. Cover: SHARP-style switch reduction and any AMD/Broadcom/Ultra-Ethernet Consortium in-network-collective work (UEC INC, 2024-2026); programmable-switch aggregation (ATP, SwitchML); gradient compression collectives (THC SC'23, quantized allreduce in NCCL-like libs, ZeRO++ qgZ); sparse collectives (OmniReduce, SparCML); RDMA/transport-level improvements relevant to collectives (multi-QP striping, adaptive routing awareness). Prioritize measured gains and AMD-ecosystem feasibility. ${PRIOR_COVERED} Report precise claims. ${HW_CONTEXT}` },
  { key: 'fresh', prompt: `Literature sweep — theme: FRESH 2025-2026 developments in GPU collective communication NOT covered by other themes. Use WebSearch/WebFetch. Search for: papers/blogs H2-2025 through mid-2026 on collective communication for LLM training/inference (SC'25, NSDI'26, MLSys'26, ASPLOS'26, EuroSys'26, arXiv); NCCL 2.28-2.31 new features (device API, symmetric memory, GIN/GPU-initiated networking, one-sided RMA, CE collectives) and how vendors exploit them; Triton-distributed / Iris (AMD) GPU-driven communication; anything specifically evaluating or beating RCCL on MI300X/MI325X/MI355X. ${PRIOR_COVERED} Report precise claims. ${HW_CONTEXT}` },
]

const surveyResults = await parallel([
  ...invAreas.map(a => () => agent(a.prompt, { label: `inv:${a.key}`, phase: 'Survey', schema: INV_SCHEMA, model: MODEL })),
  ...litThemes.map(t => () => agent(t.prompt, { label: `lit:${t.key}`, phase: 'Survey', schema: CAND_SCHEMA, model: MODEL })),
])

const inv = {}
invAreas.forEach((a, i) => { inv[a.key] = surveyResults[i] })
const litRaw = litThemes.map((t, i) => ({ theme: t.key, result: surveyResults[invAreas.length + i] })).filter(x => x.result)
const allCandidates = litRaw.flatMap(x => (x.result.candidates || []).map(c => ({ ...c, theme: x.theme })))
log(`Survey complete: ${allCandidates.length} raw candidates from ${litRaw.length} literature themes; ${Object.values(inv).filter(Boolean).length}/4 inventory areas returned`)

// ---------- Gate 1: screening (needs ALL candidates + inventory — barrier justified) ----------
phase('Screen')
const invSummary = Object.entries(inv).filter(([, v]) => v).map(([k, v]) => `### ${k}\n${v.summary}\nSurprises: ${v.surprises || 'none'}`).join('\n\n')

const screen = await agent(`You are the SCREENING GATE of a wide-search funnel for ${cfg.component}. Below are (A) summaries of what the library already implements, and (B) raw literature candidates for missing high-impact algorithms.

Deduplicate the candidates (merge same paper/technique reported by multiple themes). Then rank by: strength of published evidence (measured, peer-reviewed, on-GPU, ideally on-AMD) x expected performance impact x AMD-hardware relevance x plausibility of being missing from the library. Drop candidates that are obviously already implemented, NVIDIA-hardware-bound with no AMD analogue, or have only simulated/theoretical evidence. Mark novelty: 'new' (not in the prior-covered list), 'update-to-prior' (new evidence about a prior item), 'prior'. Keep at most ${cfg.maxCandidates} in 'screened', priority 1 = highest. Favor 'new' and 'update-to-prior' items; include a 'prior' item only if major new evidence emerged.

${PRIOR_COVERED}

(A) LIBRARY INVENTORY SUMMARIES:
${invSummary}

(B) RAW CANDIDATES (JSON):
${JSON.stringify(allCandidates)}`, { label: 'gate1:screen', phase: 'Screen', schema: SCREEN_SCHEMA, model: MODEL })

const screened = (screen && screen.screened ? screen.screened : []).sort((a, b) => a.priority - b.priority).slice(0, cfg.maxCandidates)
log(`Gate 1: ${screened.length} candidates advance (${(screen && screen.dropped || []).length} dropped)`)

// ---------- Gate 2 (code check) + Gate 3 (adversarial verify), pipelined per candidate ----------
const checked = await pipeline(
  screened,
  (c) => agent(`CODE-CHECK GATE. Candidate technique: "${c.name}" (${c.source}). Mechanism: ${c.mechanism}. Claim: ${c.claim}.

Determine whether ${cfg.component} at ${cfg.repoPath} ALREADY implements this technique (or a functional equivalent), partially implements it, or lacks it. Search the source (src/device, src/graph, src/include/algorithms, src/gin, src/rma, src/nccl_device, src/scheduler, src/, CHANGELOG.md) and cite file:line evidence for whatever you conclude. 'partial' means a related mechanism exists but misses the core of the technique — explain exactly what is missing. In 'notes', name the nearest existing mechanism and the natural extension/integration point. Be adversarial toward the assumption that it is missing: hunt for equivalents under different names. ${HW_CONTEXT}`, { label: `check:${c.name.slice(0, 28)}`, phase: 'CodeCheck', schema: CHECK_SCHEMA, model: MODEL })
    .then(chk => ({ ...c, check: chk })),
  (c) => {
    if (!c || !c.check) return null
    if (c.check.status === 'present') { return { ...c, verdict: { survives: false, reason: 'already implemented' } } }
    return parallel([
      () => agent(`ADVERSARIAL VERIFIER — literature-validity lens. Try to REFUTE this claim using WebSearch/WebFetch against the actual paper/source: "${c.name}" (${c.source}): ${c.claim}. Mechanism: ${c.mechanism}. Check: do the numbers match the paper? Is the baseline really NCCL/RCCL (not MPI)? Is the hardware relevant (GPU cluster, ideally AMD)? Is it peer-reviewed or at least reproducibly documented? Set refuted=true if the claim materially misstates the source or the evidence is simulation-only/vaporware. Record any number/baseline corrections in 'corrections'.`, { label: `verify-lit:${c.name.slice(0, 22)}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: MODEL }),
      () => agent(`ADVERSARIAL VERIFIER — applicability lens. Try to REFUTE that "${c.name}" (${c.mechanism}) would deliver a real performance win if added to ${cfg.component}. Known in-code status: ${c.check.status}; evidence: ${c.check.evidence}; nearest mechanism: ${c.check.notes || 'n/a'}. ${HW_CONTEXT} Consider: does an existing RCCL path already capture most of the benefit (e.g. double-binary tree, DDA, multi-channel full-duplex rings)? Does the technique depend on NVIDIA-only hardware (multimem/NVLS, TMA, NVSwitch) with no AMD analogue? Is its winning regime one AMD customers actually hit? You may read the code at ${cfg.repoPath}. Set refuted=true only with a concrete argument.`, { label: `verify-app:${c.name.slice(0, 22)}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: MODEL }),
    ]).then(vs => {
      const [lit, app] = vs
      const refutes = vs.filter(Boolean).filter(v => v.refuted).length
      return { ...c, litVerdict: lit, appVerdict: app, verdict: { survives: refutes === 0, reason: refutes ? 'refuted by ' + (lit && lit.refuted ? 'literature-lens ' : '') + (app && app.refuted ? 'applicability-lens' : '') : 'survived both lenses' } }
    })
  },
)

const results = checked.filter(Boolean)
const confirmed = results.filter(c => c.verdict && c.verdict.survives)
const rejected = results.filter(c => c.verdict && !c.verdict.survives)
log(`Gates 2+3: ${confirmed.length} confirmed, ${rejected.length} rejected of ${results.length} checked`)

// ---------- Completeness critic ----------
phase('Critic')
const critic = await agent(`COMPLETENESS CRITIC for a wide-search funnel over ${cfg.component}. The funnel produced the confirmed gap list below. Name (a) research areas or hardware-exploitation angles plausibly MISSED by the sweep (themes: hierarchical, synthesis/executors, alltoall/MoE, bw-optimal, in-network/compression, fresh-2026), (b) confirmed claims that still look weak (evidence quality, AMD transferability). Be specific and brief.

CONFIRMED: ${JSON.stringify(confirmed.map(c => ({ name: c.name, source: c.source, claim: c.claim, status: c.check && c.check.status, litConf: c.litVerdict && c.litVerdict.confidence, corrections: c.litVerdict && c.litVerdict.corrections })))}
REJECTED (with reasons): ${JSON.stringify(rejected.map(c => ({ name: c.name, reason: c.verdict.reason })))}
INVENTORY SURPRISES: ${JSON.stringify(Object.entries(inv).map(([k, v]) => v && v.surprises))}`, { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA, model: MODEL })

return { inventory: inv, screenedCount: screened.length, confirmed, rejected, dropped: screen && screen.dropped, critic }