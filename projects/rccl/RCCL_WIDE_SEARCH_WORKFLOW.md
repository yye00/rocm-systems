# Reusable Wide-Search Workflow for Collective-Algorithm Discovery

**What this is.** A re-runnable, multi-agent workflow that executes the *search-wide* stage of the
development funnel for a collective-communication library: inventory what the code implements,
sweep the literature for published algorithms with strong measured wins, and pass every candidate
through explicit decision gates with adversarial cross-checking before it is allowed into the
final report. The executable definition lives at
[`.claude/workflows/collective-lib-wide-survey.js`](../../.claude/workflows/collective-lib-wide-survey.js)
and can be invoked by name (`collective-lib-wide-survey`) from any Claude Code session in this
repository, parameterized via `args` (`repoPath`, `component`, `priorSurvey`, `maxCandidates`).

It was used to produce `RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md` (first, hand-orchestrated run)
and `RCCL_WIDE_SEARCH_REFRESH_2026-07.md` (first codified run). Because the funnel takes the prior
survey as an input and instructs the literature agents to hunt for what is *not* already covered,
re-running it periodically yields an incremental "what's new" delta rather than a rediscovery of
the same items.

## Funnel shape

```
Phase 1  SURVEY (10 agents, fully parallel — code and literature are independent)
  ├─ 4 code-inventory agents, one per area:
  │    kernels  — device kernels + NCCL_ALGO/PROTO enums + symmetric-kernel family
  │    graph    — topology model, ring/tree search, cost model, tuner-plugin limits
  │    amd      — AMD-specific paths (DDA, Pivot, hierarchical AG, GDA, WarpSpeed)
  │               + newly-synced subsystems (gin/, rma/, nccl_device/, scheduler/, CE)
  │    delta    — CHANGELOG removals, plugin extension points, env-var surface
  └─ 6 literature agents, one per theme:
       hier     — hierarchical / topology-aware collectives (incl. MVAPICH/Panda line)
       synth    — synthesized schedules + programmable executors (MSCCL++, TACCL, …)
       a2a      — AllToAll / MoE expert-parallel communication
       bwopt    — bandwidth/latency-optimal allreduce + GPU-side fast paths
       net      — in-network computing, compression, sparsity
       fresh    — open-ended sweep of the last ~12 months

Gate 1  SCREEN (1 agent; the only intentional barrier — needs ALL candidates at once)
  dedup across themes → rank by (evidence strength × impact × AMD relevance ×
  plausibility-of-missing) → keep top N (default 14) → favor items NOT already in
  the prior survey.  Output: screened list + explicit dropped list with reasons.

Gate 2  CODE-CHECK (1 agent per candidate, pipelined — no barrier)
  "Is this actually missing?"  Each agent searches the tree adversarially for the
  technique under any name and returns status ∈ {present, partial, missing} with
  file:line evidence plus the nearest existing mechanism / integration point.
  status = present ⇒ candidate exits the funnel here.

Gate 3  ADVERSARIAL VERIFY (2 agents per surviving candidate, parallel lenses)
  literature-validity lens — re-derive the claim from the actual paper: numbers,
    baseline (NCCL/RCCL vs MPI!), hardware, peer review.  Refute on mismatch.
  applicability lens — try to refute that the technique would win on AMD hardware
    given what the code already does (double-binary tree, full-duplex link model,
    DDA, …) and given NVIDIA-only hardware dependencies.
  A candidate must survive BOTH lenses; refutations carry written reasoning and
  correction notes that flow into the report.

Phase 5  CRITIC (1 agent)
  Completeness critic over the whole funnel output: which research angles did the
  sweep miss, and which surviving claims still look weak.  Its findings seed the
  next run (or an immediate supplementary round).

SYNTHESIS (main session, not an agent)
  The orchestrating session writes the report from the structured gate outputs —
  confirmed gaps with corrected claims, per-gap implementation guidance anchored
  to the integration points surfaced by Gate 2, and the rejected/dropped lists as
  an audit trail.
```

## Design principles (why it is shaped this way)

1. **Wide before deep.** Ten independent survey agents cast a deliberately broad net; the
   funnel exists so breadth at the top does not cost accuracy at the bottom. Every claim in the
   final report has passed at least three independent checks (screen, code-check, 2-lens verify).
2. **Code and literature never trust each other.** Literature agents do not decide what is
   missing; code agents do not decide what is valuable. The screen and code-check gates are the
   only places those two evidence streams meet, and both record their reasoning.
3. **Adversarial by default.** The first hand-run of this survey produced a wrong load-bearing
   claim (a "PCIe reverse-bandwidth cost-model defect" that does not exist) that only fell to an
   adversarial re-audit. Gates 2 and 3 bake that re-audit in: verifiers are instructed to
   *refute*, not confirm, and "already implemented under another name" is treated as the default
   hypothesis to kill.
4. **The baseline question is a first-class check.** The single most common literature error
   found across runs is gains quoted against MPI being read as gains against NCCL/RCCL. The
   literature-validity lens asks this explicitly.
5. **Priors are input, not gospel.** The prior survey document is fed to every agent as
   context with the instruction to verify independently and report drift ("surprises"). This is
   what makes the workflow *incremental*: reruns surface new subsystems (e.g. `src/gin/`,
   `src/rma/` appearing in 2.30) and new papers instead of re-proving old ones.
6. **Rejections are deliverables.** The dropped and refuted lists, with reasons, are kept in
   the report. In a search-wide stage, a well-evidenced "no" prevents the next person from
   re-opening the same dead end.
7. **Barriers only where semantics demand.** Only Gate 1 blocks on all upstream results
   (dedup/ranking is a cross-item operation). Everything else is pipelined per candidate so a
   slow literature verify on one item never stalls the code check of another.

## How to re-run

From a Claude Code session at the repo root, ask for the workflow by name, e.g.:

> Run the `collective-lib-wide-survey` workflow over projects/rccl and write the refresh report.

Optional `args` overrides: `{ "repoPath": "...", "component": "...", "priorSurvey": "...",
"maxCandidates": 14 }`. Point `priorSurvey` at the most recent survey/refresh report so the run
searches for the delta. After the run, the orchestrating session should:

1. Read the returned structure (`confirmed`, `rejected`, `dropped`, `critic`, `inventory`).
2. Chase anything the critic flags (usually one supplementary agent round).
3. Write/refresh the report; keep the audit trail (rejected + dropped + corrections).
4. Update `priorSurvey`'s "prior-covered" list inside the workflow script if items graduated
   from "candidate" to "covered".

## Scope notes and known limits

- The funnel evaluates *algorithmic/structural* gaps, not micro-optimizations (kernel occupancy,
  protocol thresholds) — those belong to the tuning stage of the funnel, downstream of this one.
- Literature agents can only find what is published or publicly documented; proprietary vendor
  schedules (e.g. unpublished NCCLX internals) appear only via secondary sources.
- Gate 3's applicability lens argues from topology and code structure, not measurement. Its
  output is a *confidence*, not a benchmark: nothing that exits this funnel should ship without
  the A/B measurement the reports themselves prescribe.
