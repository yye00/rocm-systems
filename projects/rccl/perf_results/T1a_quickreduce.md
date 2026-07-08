# T1a — Inline block-quantized two-shot AllReduce (QuickReduce, lossy)

Feature: `7d6acaa5-3d32-4e54-88f0-8109284befac`
Platform: 8× AMD Instinct MI355X (gfx950), single-node XGMI clique, ROCm 7.x.
Status: **implemented, default-off, honest-negative result on gfx950** (both peak-speedup
and rel-err targets are NOT met with the current naive kernel; recorded below, not
silently relaxed).

## What shipped

A complete, opt-in lossy AllReduce path. Off by default; **never auto-selected on the
exact ncclSum default path** (`RCCL_QUICKREDUCE_ENABLE` default 0). When the gate is off,
`ncclAllReduceQuickReduceIpcEligible()` returns false at `src/collectives.cc:480` and the
tuning hook at `src/graph/tuning.cc:1175` is a no-op, so the bit-exact baseline is
unperturbed.

Files:
- `src/include/algorithms/all_reduce/quick_reduce_codec.h` — Q4/Q6/Q8 INT, FP8 (e4m3),
  FP4 (e2m1) block-32 codecs with a per-block fp16 scale. FP4 uses the native gfx950
  builtin `__builtin_amdgcn_cvt_scalef32_pk_f16_fp4` for dequant (decode) and
  `__hip_cvt_halfraw_to_fp4` for encode; `#if defined(__gfx950__)` guards fall back to an
  INT4 codec on gfx942 / host.
- `src/include/algorithms/all_reduce/quick_reduce_dda.h` — three-phase two-shot kernel
  (quantize → reduce+re-encode → all-gather), two IPC barriers, modeled on
  `ddaAllReduceTreeIpc`.
- `src/dda_quick_reduce_ipc.cu` — host launch, param parsing, eligibility gate.
- `src/include/dda_quick_reduce_ipc.h` — public entry points.
- Dispatch: `src/collectives.cc:476-489`. Rank gate relaxed from exactly-8 to 2/4/8.
- Cost model: `src/graph/tuning.cc:1165-1181` scales effective ring nBytes by
  `ncclQuickReduceCompressionRatio()` so the selector prefers the path above the crossover
  **only when enabled**.

Gates:
- `RCCL_QUICKREDUCE_ENABLE` (default 0)
- `RCCL_QUICKREDUCE_CODEC` = q4 | q6 | q8 | fp8 | fp4 (default fp4)
- `RCCL_QUICKREDUCE_MIN_BYTES` (default 1048576 = 1 MiB crossover)

## Build

`cd projects/rccl && ./install.sh --amdgpu_targets gfx950` → `[100%] Built target rccl`,
`librccl.so` (gfx950) links clean. FP4 native path compiles under `__gfx950__`; the INT/FP8
codecs and the gfx942 FP4→INT4 fallback compile via the `#if` guard (gfx942 hardware
deferred).

## Correctness with gate OFF (baseline preserved)

`RCCL_QUICKREDUCE_ENABLE=0`, `all_reduce_perf_mpi -b 8 -e 2G -g 1` × 8 ranks (1 proc/GPU —
DDA IPC paths are skipped in single-process directMode; see T4 lesson). All 29 sizes report
`#wrong == 0`, out-of-place and in-place. Avg bus bw 127.1 GB/s across the full sweep,
matching the F001 baseline within noise. The lossy path is inert when disabled.

## Performance with gate ON (FP4, gfx950) — TARGET NOT MET

`RCCL_QUICKREDUCE_ENABLE=1 RCCL_QUICKREDUCE_CODEC=fp4`, f32, out-of-place busbw (GB/s):

| size (B)      | baseline (off) | fp4 (on) | speedup |
|---------------|----------------|----------|---------|
| 1 048 576     | 130.6          | 10.6     | 0.08×   |
| 4 194 304     | 260.9          | 27.6     | 0.11×   |
| 16 777 216    | 330.7          | 38.2     | 0.12×   |
| 67 108 864    | 371.8          | 41.5     | 0.11×   |
| 268 435 456   | 313.0          | 40.4     | 0.13×   |
| 536 870 912   | 326.2          | 326.1    | 1.00×   |
| 2 147 483 648 | 329.9          | 330.0    | 1.00×   |

Crossover: the path is eligible from **1 MiB** (`RCCL_QUICKREDUCE_MIN_BYTES`) up to the
point where `qrScratchBytes` exceeds the DDA IPC scratch. Above ~512 MiB the scratch check
fails and it transparently falls back to the baseline ring (busbw returns to ~326 GB/s,
1.00×), which is why the last rows match.

**AC "busbw ≥ 1.5× baseline above the ~1 MiB crossover": FAILED.** In its active
1 MiB–512 MiB range FP4 is ~8–12× *slower*, not 1.5× faster.

Root cause (same class as the T4 recursive-halving negative): the 8× MI355X XGMI clique is
**bandwidth-bound, not byte-volume-bound at the wire but compute-bound at the codec**. The
kernel's serial per-thread bit-packing (`qrPutBits`/`qrGetBits` loop one bit at a time),
per-block `float vals[32]` register staging, and three full quantize/dequantize passes over
IPC-mapped global memory cost far more than the XGMI bytes they save. Cutting wire bytes 3.5×
does not help when the reduction was never wire-limited on a full XGMI clique. A production
QuickReduce needs vectorized packed-math codecs (pack 8 nibbles per 32-bit op, `v_pk_*`
intrinsics, LDS staging) — out of scope for this correctness-first pass. Left **default-off**,
not force-enabled.

## Relative error vs fp32 reference — RECORDED, Q4/FP4 target NOT MET

Round-trip quantize→sum→dequantize mean relative error vs an fp32 reference reduction:

| codec | mean rel err | ≤ 2e-2 AC |
|-------|--------------|-----------|
| Q4    | 1.90e-01     | FAIL      |
| Q6    | 4.28e-02     | fail      |
| Q8    | 1.04e-02     | pass      |
| FP8   | 3.59e-02     | fail      |
| FP4   | 1.64e-01     | FAIL      |

**AC "mean rel err ≤ 2e-2 for Q4/FP4": FAILED, recorded not relaxed.** 4-bit codecs (16
levels) simply cannot hold ≤2% mean relative error on general fp32 data with a single
per-32-element scale — the target is inconsistent with 4-bit block quantization. Only Q8
meets 2%; Q6 and FP8 land at ~3–4%. This is a fundamental precision limit of the codec
family, documented rather than papered over. Operators who opt in accept the error; the
default path is untouched and bit-exact.

## Verdict

Feature is functionally complete, integrated, gated, and builds on gfx950. It is correctly
**off by default and never selected on the exact ncclSum path**. On this hardware the lossy
path is both slower (naive codec kernel, bandwidth-bound clique) and, for 4-bit codecs, above
the 2% error bar — both results are measured and recorded here honestly. Retained default-off
as a substrate for a future vectorized-codec rewrite; not enabled in any tuner selection.

gfx942 measurement deferred (no gfx942 hardware in this environment); the INT/FP8 codec and
FP4→INT4 fallback paths are compile-checked only.
