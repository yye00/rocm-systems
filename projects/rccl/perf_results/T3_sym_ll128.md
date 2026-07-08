# T3 — Generalize DDA/symmetric AR beyond 8-rank/ncclSum + symmetric LL128 axis

Feature: `76a1477e-d635-4c22-aa14-504aa27ca33c`
Date: 2026-07-08
Hardware: 8× AMD Instinct MI355X (gfx950), single-node XGMI clique.
Build: incremental `make -j64 rccl` in `build/release` (exit 0). librccl.so = gfx950, 273 MB.
Gates (both default 0): `RCCL_SYM_LL128_ENABLE`, `RCCL_DDA_NRANKS_RELAX`.

> Measurement discipline (carried over from T4): DDA IPC paths only engage with
> **one process per GPU** (`mpirun -np N ... -g 1`). A single-process `-g N` run sets
> `directMode=1`, which makes `ncclDdaIpcCommInit` skip scratch/barrier allocation, so
> DDA never runs and gate-on == gate-off. All numbers below are MPI (1 proc/GPU).
> Binary: `projects/rccl-tests/build/all_reduce_perf_mpi`. Raw sweeps in `perf_results/t3/mpi_*.txt`.

---

## Summary

| Sub-feature | Status | Verdict |
|---|---|---|
| `RCCL_DDA_NRANKS_RELAX` (2/4/8-rank DDA AllReduce) | **Working & verified** | Positive — bit-exact and *faster* than ring at 2/4 ranks |
| `RCCL_SYM_LL128_ENABLE` (LL128 protocol axis) | **Scaffold only, gate inert** | Not runtime-verifiable on this node (see below) |
| Both gates default-off == baseline | **Verified** | Bit-identical, busbw within ±2% of F001 baseline |
| MC/TMA paths | **Untouched** | Remain NVIDIA-only, excluded in `generate.py` |

---

## 1. Both gates OFF (default) == baseline — VERIFIED

8-rank AllReduce, fp32, both gates 0:

| size | busbw (GB/s) | #wrong |
|---|---|---|
| 64 MiB | 378.2 / 378.6 | 0 |

F001 baseline reference (`perf_results/baseline`, T4 report) reported ~379 GB/s AR fp32 at 64 MiB.
Delta ≈ **−0.1%**, inside the ±2% acceptance band. All sizes 8 B..64 MiB: `#wrong == 0`,
`Out of bounds values : 0 OK`. Raw: `t3/mpi_ar_g8_default_off.txt`.

> Size ceiling 64 MiB (not 8 GiB): GPU3 hosts an external vLLM tenant (~294 GB VRAM occupied,
> confirmed via `rocm-smi --showmeminfo vram`); larger buffers OOM. Same constraint documented in
> F001 `ENV.txt` and T4.

## 2. `RCCL_DDA_NRANKS_RELAX=1` — 2/4-rank DDA AllReduce — VERIFIED (positive)

Engagement confirmed via `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT`:
- relax=1, `-np 4`: **all 4 ranks** log `ncclDdaIpcCommInit: scratch 67108864 bytes, IpcGpuBarrier nBlocks=24` and `directMode 0`.
- relax=0, `-np 4`: **0 ranks** init DDA (clean control — the gate is the only difference).

AllReduce fp32 at 64 MiB, DDA relaxed path vs ring fallback:

| ranks | relax OFF (ring) busbw | relax ON (DDA) busbw | Δ | #wrong |
|---|---|---|---|---|
| 2 | 55.65 GB/s | **61.49 GB/s** | +10.5% | 0 |
| 4 | 156.46 GB/s | **176.40 GB/s** | +12.7% | 0 |

Unlike T4/RHD (which was 5–6× slower), the generalized one-shot/two-shot DDA AllReduce is
**both bit-exact and faster** than the ring fallback at low rank counts, because at 2/4 ranks
the collective is more latency-sensitive and the IPC LD/ST path wins. All sizes: `#wrong == 0`.
Raw: `t3/mpi_ar_g{2,4}_relax_{on,off}.txt`.

Correctness of the relax path rests on three in-tree invariants (all covered by the build):
- `DeviceMailbox` flag buffer is always sized/strided for `NRANKS` (`getFlagIdx = block*NRANKS+rank`),
  so the layout is participant-count independent; only `nRanks_ (<= NRANKS)` peers are signalled/waited on.
  Using `NRANKS` in the wait loop would deadlock for `nRanks_ < NRANKS` (absent peers never set their flag).
- Peer IPC table is allocated for `kDdaNranks` but only `comm->nRanks` entries are populated/copied.
- `ddaNranksSupported()` keeps counts `∈ {2,4,8}` (power-of-two `<= kDdaNranks`), matching the
  template instantiations `ncclAllReduceDdaIpcLaunch<T,{2,4,8}>`.

`ncclSum`-only remains enforced for the relaxed AllReduce eligibility (associative-op relaxation
beyond sum is not exercised here; the LD/ST reduce still routes through the sum path).

## 3. `RCCL_SYM_LL128_ENABLE=1` — LL128 protocol axis — SCAFFOLD, NOT RUNTIME-VERIFIABLE HERE

Honest negative, two independent blockers on this node:

1. **No device kernel maps to the LL128 protocol.** The change adds `rcclSymkProto_LL128`
   to the protocol enum and a matching tuning column (`sym_kernels.cc`), gated behind
   `ncclSymLL128Enabled()`. `src/device/symmetric/generate.py` was **not** extended to emit an
   LL128 device kernel, so selection can never resolve to `rcclSymkProto_LL128` — the tuning
   entries are inert by construction (documented in the code comment at the enum). This keeps the
   gate safe (off == baseline) but means the "selects LL128, ≥95% peak" AC has no kernel to select.

2. **Symmetric memory is unsupported on this platform.** `comm->symmetricSupport` requires
   `ncclCuMemEnable()` (`init.cc:2213`). With defaults the node logs
   `Symmetric memory is not supported. cuMemEnable 0` on all ranks, so the entire symmetric-kernel
   family (where LL128 would live) never engages. Force-enabling `NCCL_CUMEM_ENABLE=1
   NCCL_WIN_ENABLE=1` switches P2P transport to CUMEM but rccl-tests then fails validation
   (buffers aren't window-registered) — i.e. the symmetric AR path is not reachable through the
   standard perf harness on this system.

Consequently the LL128 large-band busbw table is **deferred** — it cannot be honestly produced
on this hardware. With the gate ON, correctness is unaffected: `#wrong == 0`,
`Out of bounds values : 0 OK` (the gate only widens tuning tables that no kernel consults).

**128B atomic-write verification on XGMI (gfx942/gfx950): NOT performed** — it is moot until a
real LL128 device kernel exists and symmetric memory is available. The gate must stay default-off
until both are true; the init-time INFO log warns of this requirement.

## 4. MC / TMA paths — untouched

`generate.py` still excludes multicast (`*MC`) and TMA algos on ROCm (no NVLS/multimem);
no MC/TMA kernel was added or enabled. Confirmed unchanged in the diff.

---

## gfx942

Deferred. Only gfx950 hardware is present on this node. The relax and LL128-gate code is
arch-independent host logic and compiles for both; gfx942 runtime numbers require gfx942 silicon.

## Reproduce

```bash
RCCL=projects/rccl/build/release
export LD_LIBRARY_PATH=$RCCL:$LD_LIBRARY_PATH
BIN=projects/rccl-tests/build/all_reduce_perf_mpi

# default-off baseline (both gates 0)
mpirun --allow-run-as-root -np 8 $BIN -b 8 -e 64M -f 2 -g 1

# DDA relax, 2/4 ranks (verify engagement)
RCCL_DDA_NRANKS_RELAX=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  mpirun --allow-run-as-root -np 4 -x RCCL_DDA_NRANKS_RELAX -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $BIN -b 8 -e 64M -f 2 -g 1     # expect "ncclDdaIpcCommInit: scratch" on all ranks, directMode 0
```
