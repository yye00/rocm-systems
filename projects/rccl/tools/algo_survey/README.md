# RCCL Collective-Algorithm Survey Workflow

A **re-runnable, sub-agent "search-wide" funnel** for finding published algorithmic
and hardware-exploiting collective-communication improvements that are *missing or
under-exploited* in RCCL, and verifying each candidate against the real source before
it is reported.

This directory is the reusable machinery behind the survey reports at the repo root
(`RCCL_ALGORITHM_SURVEY_AND_IMPROVEMENTS.md`, `RCCL_ALGO_SURVEY_RUN_2026-07.md`,
`COLLECTIVE_COMM_STATE_OF_THE_ART_2026.md`,
`RCCL_SINGLE_NODE_PERF_GAPS_IMPLEMENTATION.md`). Re-run it when the branch advances,
when new hardware ships, or when new literature appears, and it will regenerate a
fresh, cross-checked gap list.

- `rccl_algo_survey_workflow.mjs` — the orchestration script (run with the Claude Code `Workflow` tool).
- `README.md` — this playbook (the funnel, the decision gates, the cross-check protocol, and how to run it by hand if you don't want to spawn a fleet).

---

## Why a funnel (and not one big prompt)

Finding a *real* gap has three failure modes, and each stage exists to kill one:

1. **You miss what's already there.** A single reader skims and declares an algorithm
   "missing" when it's implemented under a different name. → *Inventory* fan-out +
   *completeness critic* gate.
2. **You get seduced by a headline number.** "17× faster" turns out to be over
   GPU-aware MPI, or NVIDIA-multimem-only, or a small-message corner case. →
   *Literature* fan-out captures baseline/platform/regime explicitly, and the
   *candidate gate* scores applicability, not hype.
3. **You report a plausible-but-wrong gap.** It survives skim review but not a
   line-by-line read. → *Adversarial cross-check* re-reads the code and a skeptic
   tries to refute every surviving candidate.

The stages are wired so wrong candidates die early and cheap, and only claims that
survive an adversarial read reach the report.

---

## The funnel

```
                       ┌─────────────────────────────────────────────┐
   Stage 1  INVENTORY  │  ~8 code readers, one per RCCL subsystem     │  what EXISTS
   (fan-out)           │  core-algos · amd-dda · alltoall · symmetric │
                       │  topo-costmodel · protocols · plugin · net   │
                       └───────────────────────┬─────────────────────┘
                                               ▼
   Stage 2  GATE ▸ completeness critic: "what dispatch paths did we miss?"
            (merge inventory; flag one-rank / direct-RS / quantized / generated kernels)
                                               ▼
                       ┌─────────────────────────────────────────────┐
   Stage 3  LITERATURE │  ~10 web/arXiv readers, one per direction:   │  what COULD exist
   (fan-out)           │  bw-optimal · hierarchical · moe-a2a ·       │
                       │  in-network · compression · programmable ·   │
                       │  fault-tolerant · amd-hw · llm-inference ·   │
                       │  pcie-bidirectional                          │
                       └───────────────────────┬─────────────────────┘
                                               ▼
   Stage 4  GATE ▸ score impact × applicability × implementability, dedup,
            drop {already-in-RCCL, NVIDIA-only, purely-theoretical}, ADVANCE top ~10-14
                                               ▼
   Stage 5  CROSS-CHECK (per surviving candidate, pipelined)
            (a) code-presence read → absent / partial / present + file:line evidence
            (b) adversarial skeptic → REFUTE it is a real, beneficial, AMD-relevant gap
                (default is_real_gap=false; keep only what survives)
                                               ▼
   Stage 6  SYNTHESIS ▸ ranked, verified gap list → the human writes the report
```

### Decision gates (the two places candidates die)

- **Inventory gate (Stage 2):** a completeness critic re-scans for dispatch paths the
  fan-out missed (one-rank special cases, direct/one-shot reduce-scatter, generated
  `RunWorkColl` instantiations, quantized paths). Prevents false "missing" claims.
- **Candidate gate (Stage 4):** every literature candidate is scored on
  `impact × applicability × implementability` and marked `already_in_rccl`. Anything
  already fully in RCCL, NVIDIA-multimem-only with no AMD analog, or purely theoretical
  with no AMD path is **dropped** here. Only the strongest ~10–14 advance.

### Cross-check (the adversarial stage)

Stage 5 is a two-step **pipeline** per candidate so each finding is verified the moment
its code-read completes (no barrier):

1. **Code-presence read** — an agent greps/reads the actual tree and returns
   `absent | partial | present` with `file:line` evidence and, if partial, exactly
   what differs from the published algorithm.
2. **Adversarial refute** — a second agent is told to *refute* that this is a real,
   beneficial, implementable gap (already covered? gain only vs MPI / only on NVIDIA?
   regime irrelevant to AMD?). It **defaults to `is_real_gap = false`** and keeps a
   candidate only if the gap survives its strongest attack.

Only candidates with `is_real_gap = true` reach `confirmed_gaps` in the output.

---

## Running it

### With the Workflow tool (the fleet)

```
Workflow({ scriptPath: "projects/rccl/tools/algo_survey/rccl_algo_survey_workflow.mjs",
           args: { rcclPath: "/abs/path/to/projects/rccl" } })
```

It runs in the background (~30–40 sub-agents; code readers use `Read`/`Grep`/`Glob`,
literature readers use `WebSearch`/`WebFetch` and the alphaXiv MCP tools) and returns a
structured object:

```jsonc
{
  "inventory":              [ /* per-slice algorithm lists with file:line */ ],
  "inventory_completeness": { "coverage_assessment": "...", "missing_paths": [...] },
  "literature_candidates":  [ /* every candidate with citation + regime + amd_evidence */ ],
  "scored":                 [ /* impact/applicability/implementability + advance|drop */ ],
  "crosschecked":           [ /* per candidate: codecheck + adversarial verdict */ ],
  "confirmed_gaps":         [ /* only gaps that survived refutation */ ],
  "summary":                { "slices": N, "candidates": N, "advanced": N, "confirmed": N }
}
```

Take `confirmed_gaps` + `inventory` and write the report. **The workflow finds and
verifies; a human ranks, sequences, and writes** — do not ship its raw output as the
report.

### By hand (no fleet)

The same funnel works sequentially if you don't want to spawn agents (this is how the
2026-07 run was produced after the fleet step was declined):

1. **Inventory** — walk the subsystems in `CODE_SLICES` (top of the script); record
   every algorithm with a `file:line` anchor. Re-verify anchors — **line numbers drift**
   (e.g. the cost model moved `tuning.cc:1148 → :1627` between survey passes).
2. **Completeness pass** — grep `src/device/generate.py`, `rccl_metadata.h`,
   `onerank.cu`, `reduce_scatter.h` for paths outside the `NCCL_ALGO_*` enum.
3. **Literature** — search each direction in `LIT_DIRECTIONS`; for every hit record
   *baseline*, *platform*, *regime*. Flag "vs MPI" and "NVIDIA-only" immediately.
4. **Score & filter** — apply the Stage-4 rubric; drop already-present / NVIDIA-only /
   theoretical.
5. **Cross-check** — for each survivor, read the code to confirm absent/partial, then
   argue the *against* case before you argue the *for* case.
6. **Write** — rank by (impact × applicability × implementability), sequence by
   dependency, and give `file:line` integration points.

---

## Cross-checking / anti-hype rules (baked into the prompts)

- A speedup is meaningless without its **baseline** and **platform**. "N× over
  GPU-aware MPI" is *not* N× over RCCL. Capture both or drop the number.
- NVIDIA **multimem / TMA / NVLS** paths are typically **inert on AMD** — treat as a
  gap only if there is a real AMD fabric analog.
- Prefer **AMD-measured** evidence (ROCm blogs, MI250X/MI300X/MI355X papers) over
  NVIDIA-measured claims when judging applicability.
- RCCL is **single-process / multi-GPU**; "multiple processes per GPU" MPI results do
  not map directly.
- Re-verify every `file:line` on each run; **never trust a prior report's anchors**.

---

## Extending

- **New hardware** (e.g. next-gen CDNA): add an `amd-hw-exploit` literature direction
  and a matching inventory slice if new dispatch paths appear.
- **New research area:** append to `LIT_DIRECTIONS`.
- **Deeper verification:** raise the refute step to a 3-voter panel
  (`parallel([...]).then(majority)`) for higher-confidence confirmation, mirroring the
  adversarial-verify pattern.
- **Tighter budget:** trim `CODE_SLICES` / `LIT_DIRECTIONS` and set `effort: 'low'` on
  the fan-out stages; keep `effort: 'high'` on the two gates and the refute step.
```
