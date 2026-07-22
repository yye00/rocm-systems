// RCCL Collective-Algorithm Survey & Gap-Analysis Workflow
// ---------------------------------------------------------
// A re-runnable "search-wide" funnel that (1) inventories the collective
// algorithms actually implemented in RCCL, (2) sweeps published literature for
// algorithmic and hardware-exploiting improvements, (3) passes candidates
// through decision gates, and (4) adversarially cross-checks every surviving
// gap against the real source before it is reported.
//
// Run with the Claude Code `Workflow` tool:  Workflow({ scriptPath: <this file> })
// The funnel, gates, and cross-check protocol are documented in README.md.
//
// Stages:
//   1. Inventory      (fan-out over code slices)          -> what EXISTS
//   2. Inventory-Gate (completeness critic + merge)       -> DECISION GATE
//   3. Literature     (fan-out over research directions)  -> what COULD exist
//   4. Candidate-Gate (score impact x applicability x     -> DECISION GATE
//                      implementability, dedup, filter)
//   5. Cross-check    (per candidate: code-presence check -> ADVERSARIAL GATE
//                      then skeptic tries to refute)
//   6. Synthesis      (structured, ranked gap list)

export const meta = {
  name: 'rccl-algo-survey',
  description: 'Survey RCCL collective algorithms, sweep literature, cross-check gaps',
  whenToUse: 'Search-wide funnel to find published algorithmic / hardware-exploiting collective-communication improvements missing from RCCL, verified against source.',
  phases: [
    { title: 'Inventory',      detail: 'fan-out code readers over RCCL subsystems' },
    { title: 'Inventory-Gate', detail: 'completeness critic + merge (decision gate)' },
    { title: 'Literature',     detail: 'fan-out web/arXiv readers over research directions' },
    { title: 'Candidate-Gate', detail: 'score + dedup + filter candidates (decision gate)' },
    { title: 'Cross-check',    detail: 'per-candidate code-presence + adversarial refute' },
    { title: 'Synthesis',      detail: 'ranked, verified gap list' },
  ],
}

// Repo root of the RCCL subcomponent; override via args.rcclPath when re-running.
const RCCL = (args && args.rcclPath) || '/home/user/rocm-systems/projects/rccl'

// ---- schemas ---------------------------------------------------------------

const INVENTORY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    slice: { type: 'string' },
    algorithms: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          name: { type: 'string' },
          enum_or_id: { type: 'string', description: 'NCCL_ALGO_* value, kernel id, or "n/a"' },
          files: { type: 'array', items: { type: 'string' }, description: 'file:line anchors' },
          pattern: { type: 'string', description: 'communication pattern / step structure' },
          collectives: { type: 'array', items: { type: 'string' } },
          hardware_assumptions: { type: 'string' },
          gating: { type: 'string', description: 'arch / rank-count / size / env gates' },
          notes: { type: 'string' },
        },
        required: ['name', 'files', 'pattern'],
      },
    },
    cost_model_notes: { type: 'string' },
  },
  required: ['slice', 'algorithms'],
}

const COMPLETENESS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    coverage_assessment: { type: 'string' },
    missing_paths: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { name: { type: 'string' }, where: { type: 'string' }, why_it_matters: { type: 'string' } },
        required: ['name', 'where'],
      },
    },
  },
  required: ['coverage_assessment'],
}

const LIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    direction: { type: 'string' },
    candidates: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          name: { type: 'string' },
          citation: { type: 'string', description: 'authors, venue, year, arXiv id if any' },
          hardware_exploited: { type: 'string', description: 'what HW property the algorithm leverages' },
          mechanism: { type: 'string' },
          claimed_gain: { type: 'string', description: 'reported speedup + baseline + platform' },
          target_regime: { type: 'string', description: 'message size / topology / scale where it wins' },
          amd_evidence: { type: 'string', description: 'measured on AMD/ROCm? which GPU?' },
          maturity: { type: 'string', enum: ['proven-on-amd', 'proven-elsewhere', 'theoretical', 'unknown'] },
        },
        required: ['name', 'mechanism', 'claimed_gain'],
      },
    },
  },
  required: ['direction', 'candidates'],
}

const SCORING_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    ranked: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          name: { type: 'string' },
          impact: { type: 'integer', description: '1-5 expected perf/resilience upside' },
          applicability: { type: 'integer', description: '1-5 fit to AMD/RCCL hardware & model' },
          implementability: { type: 'integer', description: '1-5 tractability given RCCL internals' },
          score: { type: 'number' },
          already_in_rccl: { type: 'string', enum: ['no', 'partial', 'yes', 'unknown'] },
          rationale: { type: 'string' },
          decision: { type: 'string', enum: ['advance', 'drop'] },
        },
        required: ['name', 'score', 'decision'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['ranked'],
}

const CODECHECK_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    name: { type: 'string' },
    present_in_rccl: { type: 'string', enum: ['absent', 'partial', 'present'] },
    evidence: { type: 'string', description: 'file:line anchors proving present/partial/absent' },
    what_is_missing: { type: 'string' },
  },
  required: ['name', 'present_in_rccl', 'evidence'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    name: { type: 'string' },
    is_real_gap: { type: 'boolean' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    refutation_attempt: { type: 'string', description: 'the strongest case that this is NOT a real/beneficial gap' },
    surviving_claim: { type: 'string', description: 'what remains true after refutation' },
    recommended_action: { type: 'string' },
  },
  required: ['name', 'is_real_gap', 'confidence'],
}

// ---- stage 1: inventory (fan-out over code slices) -------------------------

const CODE_SLICES = [
  { key: 'core-algos', hint: `Core NCCL-derived algorithms. Read ${RCCL}/src/device/{all_reduce,all_gather,reduce_scatter,broadcast,reduce,sendrecv}.h and the enum in ${RCCL}/src/include/plugin/nccl_tuner.h. Enumerate Ring, double-binary Tree (runTreeUpDown/runTreeSplit), CollNet Direct/Chain, NVLS/NVLS-Tree, PAT with exact file:line and which are inert on AMD.` },
  { key: 'amd-dda', hint: `AMD DDA / IPC / fabric one-shot & two-shot paths. Read ${RCCL}/src/dda_*.cu, ${RCCL}/src/include/algorithms/**, ${RCCL}/src/collectives.cc. Capture kDdaNranks, arch gates (gfx942/gfx950), size thresholds, RCCL_DDA_THRESHOLD, ncclSum restrictions, which collectives covered.` },
  { key: 'alltoall', hint: `AllToAll family. Read ${RCCL}/src/device/{alltoall_pivot,alltoall_gda,alltoallv_gda,sendrecv,hierarchical_ag_shuffle}.h and the a2a paths in ${RCCL}/src/enqueue.cc / collectives.cc. Capture Pivot A2A bidirectional rings, RocSHMEM GDA, and the naive cross-node fallback.` },
  { key: 'symmetric', hint: `Symmetric-memory kernel family. Read ${RCCL}/src/include/sym_kernels.h and ${RCCL}/src/device/symmetric/*. Enumerate the ncclSymkKernelId kernels (LL/LD-ST/MC/TMA/Rail variants) and note which depend on NVIDIA multimem/TMA and are inert on AMD.` },
  { key: 'topo-costmodel', hint: `Topology + cost model. Read ${RCCL}/src/graph/{topo,paths,search,tuning,trees,rings,rome_models}.cc. Capture: the time = lat*latCount + nBytes/(1000*bw) model, tree ratio*=.5, per-arch bw/lat tables, correction factors, double-binary tree construction, how link directions are modeled (full-duplex? followPath revBw coupling), switch-uplink contention.` },
  { key: 'protocols', hint: `Protocols & primitives. Read ${RCCL}/src/device/{prims_simple,prims_ll,prims_ll128,primitives,common_kernel}.h. Capture LL / LL128 / Simple protocol structure, chunking/pipelining, multi-channel duplication.` },
  { key: 'plugin-scheduler', hint: `Tuner plugin, scheduler, MSCCL removal state. Read ${RCCL}/src/include/plugin/tuner/*, ${RCCL}/src/scheduler/*, ${RCCL}/CHANGELOG.md, ${RCCL}/src/misc/api_trace.cc. Confirm whether MSCCL/MSCCL++ is present or removed (grep for mscclpp, RCCL_MSCCLPP_THRESHOLD), and what the tuner ABI can and cannot express.` },
  { key: 'transports', hint: `Transports & fabrics. Read ${RCCL}/src/transport/* and ${RCCL}/src/{gin,rma,proxy.cc}. Capture P2P/SHM/NET/IB paths, GPU-initiated (GIN) support, xGMI/PCIe usage, and any in-network / collective-NIC (SHARP-like) support.` },
]

phase('Inventory')
const inventoryRaw = await parallel(CODE_SLICES.map((s) => () =>
  agent(
    `You are surveying the RCCL source (an AMD/ROCm fork of NCCL) to inventory the collective-communication algorithms ACTUALLY IMPLEMENTED in this tree. Use Read/Grep/Glob on real files — do not rely on memory. Slice: "${s.key}". ${s.hint}\n\nReturn every distinct algorithm/kernel path you can substantiate with a file:line anchor. Be precise about hardware assumptions and gating. If something is present but inert on AMD (NVIDIA-only), say so explicitly.`,
    { label: `inv:${s.key}`, phase: 'Inventory', schema: INVENTORY_SCHEMA }
  )
)).filter(Boolean)

// ---- stage 2: inventory decision gate (completeness critic) ----------------

phase('Inventory-Gate')
const inventoryDigest = JSON.stringify(
  inventoryRaw.map((r) => ({ slice: r.slice, algorithms: (r.algorithms || []).map((a) => ({ name: a.name, files: a.files, gating: a.gating })) })),
).slice(0, 12000)

const completeness = await agent(
  `Here is a merged inventory of algorithm paths another set of agents found in RCCL (${RCCL}):\n\n${inventoryDigest}\n\nAct as a completeness critic. Search the tree for algorithm/kernel dispatch paths that are MISSING from this list — e.g. one-rank special cases, direct/one-shot reduce-scatter, batched reduceCopy paths, quantized paths, any RunWorkColl instantiation in ${RCCL}/src/device/generate.py or rccl_metadata.h not represented above. Report what was missed and where.`,
  { label: 'inv:completeness', phase: 'Inventory-Gate', schema: COMPLETENESS_SCHEMA }
)

// ---- stage 3: literature (fan-out over research directions) -----------------

const LIT_DIRECTIONS = [
  { key: 'bw-optimal-allreduce', q: 'Bandwidth-optimal AllReduce: Rabenseifner recursive halving-doubling, Swing (NSDI24), 2D/torus/multi-lane ring, ForestColl. Focus on step-count vs bandwidth tradeoffs and topology gating.' },
  { key: 'hierarchical-decomp', q: 'Hierarchical / multi-level AllReduce decomposition: BlueConnect (MLSys19), HiCCL (IPDPS25), PCCL / "Big Send-off" (SC25 on Frontier MI250X), MVAPICH multi-lane / D.K. Panda PCIe-tree bidirectional work. Note which have AMD evidence.' },
  { key: 'moe-alltoall', q: 'AllToAll for MoE / expert parallel: Bruck (1997), FLASH (2025 on MI300X), hierarchical/aggregated incast-aware AllToAll, DeepEP. Focus on inter-node small/medium message regime.' },
  { key: 'in-network', q: 'In-network / switch aggregation: NVIDIA SHARP, NVLS multimem, in-switch reduction, and any AMD/Ultra-Ethernet analog. What is exploitable on AMD fabrics?' },
  { key: 'compression', q: 'Compression / quantization collectives: QuickReduce, THC, NCCLZ, block-quantized two-shot AllReduce, entropy-coded gradients. Focus on intra-node bandwidth-bound AllReduce on MI300X/MI355X.' },
  { key: 'programmable', q: 'Programmable / synthesized collective runtimes: MSCCL (ASPLOS21), TACCL (NSDI22), MSCCL++ (ASPLOS25 on MI300X), TE-CCL (SIGCOMM24), Blink. Focus on shipping synthesized schedules as data.' },
  { key: 'fault-tolerant', q: 'Fault-tolerant / elastic collectives: NCCLX FTAR, R2CCL, OptCC, Mycroft (SOSP25), AdapCC. Route around NIC/rank failure at 10k-100k GPU scale.' },
  { key: 'amd-hw-exploit', q: 'AMD-hardware-exploiting collectives: CDNA3/CDNA4 chiplet (XCD/CPX/IOD) hierarchy-aware selection, Infinity Fabric / xGMI bidirectional exploitation, MI300X/MI325X/MI355X specific optimizations, LDS/scratch use in reduction kernels.' },
  { key: 'llm-inference', q: 'Low-latency collectives for LLM inference: one-shot custom AllReduce, TensorRT-LLM / vLLM custom allreduce, latency-optimal small-message AllReduce, tensor-parallel overlap.' },
  { key: 'pcie-bidirectional', q: 'PCIe-tree and bidirectional-link exploitation specifically (the D.K. Panda / MVAPICH direction): reduce-within-switch-first staging, exploiting full-duplex PCIe, GPU-aware MPI collective design (Faraji & Afsahi 2018). What is genuinely distinct from what NCCL/RCCL already do?' },
]

phase('Literature')
const litRaw = await parallel(LIT_DIRECTIONS.map((d) => () =>
  agent(
    `Research published, peer-reviewed (or strong-preprint) collective-communication algorithms in this direction: ${d.q}\n\nUse web search and the alphaXiv MCP tools (find them via ToolSearch: query "alphaxiv" or "select:mcp__alphaXiv__discover_papers,mcp__alphaXiv__answer_pdf_queries"; and WebSearch/WebFetch). Prefer real papers with citations over memory. For each candidate algorithm capture: what hardware property it exploits, its mechanism, the CLAIMED speedup with baseline and platform, the target regime (message size / topology / scale), and whether it has been measured on AMD/ROCm hardware specifically. Be honest about maturity and over-claiming (e.g. gains over GPU-aware MPI are not gains over RCCL).`,
    { label: `lit:${d.key}`, phase: 'Literature', schema: LIT_SCHEMA }
  )
)).filter(Boolean)

// ---- stage 4: candidate decision gate (score, dedup, filter) ---------------

phase('Candidate-Gate')
const allCandidates = litRaw.flatMap((r) => (r.candidates || []).map((c) => ({ ...c, direction: r.direction })))
const inventoryNames = inventoryRaw.flatMap((r) => (r.algorithms || []).map((a) => a.name))
const scoreInput = JSON.stringify({
  candidates: allCandidates.map((c) => ({ name: c.name, mechanism: c.mechanism, claimed_gain: c.claimed_gain, target_regime: c.target_regime, amd_evidence: c.amd_evidence, maturity: c.maturity })),
  rccl_already_has: inventoryNames,
}).slice(0, 22000)

const scored = await agent(
  `You are the candidate decision gate for an RCCL improvement survey. Below is a de-duplicated list of literature candidates plus the list of algorithms RCCL already implements.\n\n${scoreInput}\n\nFor each DISTINCT candidate (merge duplicates/aliases): score impact (1-5), applicability to AMD/RCCL hardware & single-process-multi-GPU model (1-5), and implementability given RCCL internals (1-5); compute score = impact*applicability*implementability; mark already_in_rccl (no/partial/yes/unknown); and decide "advance" or "drop". DROP anything already fully in RCCL, NVIDIA-multimem-only with no AMD analog, or purely theoretical with no path to AMD. ADVANCE the strongest genuinely-missing, hardware-exploiting or algorithmic wins. Aim to advance the top ~10-14.`,
  { label: 'gate:scoring', phase: 'Candidate-Gate', schema: SCORING_SCHEMA, effort: 'high' }
)

const advancing = (scored.ranked || []).filter((c) => c.decision === 'advance')

// ---- stage 5: adversarial cross-check (per surviving candidate) -------------
// pipeline: code-presence check -> skeptic tries to refute it is a real gap.

phase('Cross-check')
const crosschecked = await pipeline(
  advancing,
  (cand) => agent(
    `Cross-check whether "${cand.name}" (${cand.rationale || ''}) is actually present in the RCCL source at ${RCCL}. Read/Grep the real code. Decide present / partial / absent and cite file:line evidence. If partial, say exactly what is missing versus the published algorithm.`,
    { label: `xc:code:${cand.name}`.slice(0, 48), phase: 'Cross-check', schema: CODECHECK_SCHEMA }
  ).then((cc) => ({ cand, cc })),
  ({ cand, cc }) => agent(
    `Adversarially evaluate this proposed RCCL gap. Candidate: "${cand.name}". Code-presence finding: ${JSON.stringify(cc)}.\n\nYour job is to REFUTE that this is a real, beneficial, implementable gap for RCCL. Consider: is it already covered by an existing RCCL path (Ring/Tree/DDA/Pivot/symmetric)? Does its claimed gain hold on AMD hardware and RCCL's process model, or only vs MPI / only on NVIDIA? Is the target regime relevant to AMD deployments? Default to is_real_gap=false unless the gap survives your strongest refutation. State what survives and the recommended action.`,
    { label: `xc:refute:${cand.name}`.slice(0, 48), phase: 'Cross-check', schema: VERDICT_SCHEMA, effort: 'high' }
  ).then((v) => ({ name: cand.name, score: cand.score, direction: cand.direction, codecheck: cc, verdict: v }))
)

const confirmed = crosschecked.filter(Boolean).filter((x) => x.verdict && x.verdict.is_real_gap)

// ---- stage 6: synthesis ----------------------------------------------------

phase('Synthesis')
log(`inventory slices: ${inventoryRaw.length}; literature candidates: ${allCandidates.length}; advanced: ${advancing.length}; confirmed gaps: ${confirmed.length}`)

return {
  inventory: inventoryRaw,
  inventory_completeness: completeness,
  literature_candidates: allCandidates,
  scored: scored.ranked,
  crosschecked,
  confirmed_gaps: confirmed,
  summary: {
    slices: inventoryRaw.length,
    candidates: allCandidates.length,
    advanced: advancing.length,
    confirmed: confirmed.length,
  },
}
