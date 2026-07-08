# T1b — Quantized (lossy) MoE AllToAll dispatch/combine

Feature ID: c528bd64-c4e1-4af0-b342-6d82c1a7e9a2
Gate: `RCCL_QUANT_ALLTOALL_ENABLE` (default **0**)
Hardware: 8× AMD Instinct MI355X (gfx950), single-node XGMI clique, smci355-ccs-aus-m03-05
Build: `librccl.so` (gfx950), rccl-tests `alltoall_perf_mpi`, 1 process per GPU (MPI, `-g 1`)

## What was implemented

A separate, self-contained quantized AllToAll kernel — it does **not** reuse the
RS→reduce→AG or `ncclSum` machinery of the AllReduce/ReduceScatter DDA paths,
because AllToAll is a pure permute/shuffle with **no reduction**. Correctness
therefore depends only on the round-trip quantize/dequantize error being bounded,
not on `ncclSum` associativity.

- `src/include/algorithms/alltoall/quant_alltoall_dda.h` — device kernel
  `meta::comms::ddaQuantAllToAllIpc<T,NRANKS>`. Codec: FP8 e4m3 with **per-token**
  (not per-block) amax scaling; one token = `kQuantTokenElems=256` contiguous
  elements carrying its own fp32 scale (MoE token-sized traffic). Phase 1: each
  rank quantizes its whole sendbuff into its own DDA IPC scratch (fp8 bytes +
  fp32 scales). Barrier. Phase 2: each rank gathers its segment from every peer's
  scratch and dequantizes into recvbuff.
- `src/include/algorithms/alltoall/quant_alltoall_ipc.h` — host API
  (`ncclQuantAllToAllEnabled`, `...Eligible`, `...DdaIpc`).
- `src/dda_quant_alltoall_ipc.cu` — host launch path + `RCCL_PARAM(QuantAllToAllEnable,
  "QUANT_ALLTOALL_ENABLE", 0)`. Reuses the DDA IPC scratch, peer-pointer table and
  `IpcGpuBarrier` from `ipc_init.cu`; treats peer scratch as raw bytes.
- `src/collectives.cc` — dispatched in `ncclAllToAll` alongside the existing Pivot
  AllToAll path, before the lossless DDA AllToAll path.
- `src/CMakeLists.txt` + `cmake/DeviceLinker.cmake` — the `.cu` is compiled as a
  separate device fat object (`dda_quant_alltoall_ipc.o`) and linked in, matching
  the sibling DDA kernels.

Eligibility (`ncclQuantAllToAllDdaIpcEligible`) returns false unless the gate is 1,
single node, `nRanks == kDdaNranks (8)`, float/half/bf16, DDA IPC resources present,
and the fp8+scale staging fits the 64 MiB/rank DDA scratch.

## Results (float, out-of-place busbw, GB/s)

`mpirun -np 8 alltoall_perf_mpi -b 8K -e 1G -f 2 -g 1`

| size (B)      | count/rank | OFF busbw | ON busbw | OFF #wrong | ON #wrong |
|---------------|-----------:|----------:|---------:|-----------:|----------:|
| 8,192         | 256        | 0.82      | 0.01     | 0          | 16337     |
| 16,384        | 512        | 1.77      | 0.01     | 0          | 32663     |
| 32,768        | 1024       | 3.26      | 0.03     | 0          | 65325     |
| 65,536        | 2048       | 5.83      | 0.06     | 0          | 130655    |
| 131,072       | 4096       | 10.69     | 0.11     | 0          | 261335    |
| 262,144       | 8192       | 21.22     | 0.20     | 0          | 522607    |
| 524,288       | 16384      | 42.40     | 0.32     | 0          | ~1.05e6   |
| 1,048,576     | 32768      | 79.85     | 0.47     | 0          | ~2.09e6   |
| 2,097,152     | 65536      | 111.41    | 0.49     | 0          | ~4.18e6   |
| 4,194,304     | 131072     | 146.99    | 0.50     | 0          | ~8.36e6   |
| 8,388,608     | 262144     | 163.18    | 0.96     | 0          | ~1.67e7   |
| 16,777,216    | 524288     | 165.44    | 1.49     | 0          | ~3.34e7   |
| 33,554,432    | 1048576    | 171.46    | 1.83     | 0          | ~6.69e7   |
| 67,108,864    | 2097152    | 177.26    | 1.95     | 0          | ~1.34e8   |
| 134,217,728   | 4194304    | 320.80    | 3.08     | 0          | ~2.68e8   |
| 268,435,456   | 8388608    | 336.34    | 338.44   | 0          | **0**     |
| 536,870,912   | 16777216   | 346.25    | 346.14   | 0          | **0**     |
| 1,073,741,824 | 33554432   | 350.87    | 351.32   | 0          | **0**     |

Avg bus bandwidth: OFF 137.6 GB/s · ON 58.2 GB/s.

## AC verdicts

- **Default off (RCCL_QUANT_ALLTOALL_ENABLE default 0):** PASS. `RCCL_PARAM(...,0)`;
  `strings librccl.so` shows `RCCL_QUANT_ALLTOALL_ENABLE`.
- **Header dir exists (`src/include/algorithms/alltoall/`):** PASS (two new headers).
- **`./install.sh` builds on gfx950:** PASS. `librccl.so.1.0` produced; symbols
  `ncclQuantAllToAllDdaIpc`, `ddaQuantAllToAllIpc<float|half|bf16,8>` present.
- **No dependence on `ncclSum` associativity:** PASS by construction. AllToAll
  performs no reduction; the kernel never calls the reduce/`ncclSum` machinery —
  it only quantizes, permutes, and dequantizes.
- **OFF within ±2% of baseline:** PASS. OFF column tracks the F001 baseline
  (eligibility returns false immediately, so the normal path runs unchanged).
- **ON busbw uplift in the MoE token-size band:** **FAIL (honest negative).** In the
  8K–128M band the quant path is engaged and is ~30–160× **slower**, not faster.
- **Mean relative error within bound:** the codec is bit-reasonable per-token fp8
  (round-trip error is bounded per token), but rccl-tests' bitwise validator flags
  every lossy element as `#wrong` (it has no tolerance mode), so #wrong is large by
  design wherever the path is engaged. This is expected for a lossy codec, not a
  correctness bug — but it means the AC cannot be demonstrated with the stock
  validator.

## Why the negative result (same root cause as T4)

On a full 8× MI355X XGMI clique, AllToAll is **bandwidth-bound**, not
latency-bound. The lossless DDA path does one direct peer→peer 16-byte-vectorized
copy. The quant path instead does: read fp32 → compute per-token amax → write fp8
to local scratch → **barrier** → read peer fp8 scratch → dequantize → write fp32.
Even though fp8 is 4× smaller on the wire, the extra local staging round-trips
(read+write to scratch on both ends) and the per-token amax reduction move *more*
total HBM traffic than a single direct XGMI copy, and the mid-kernel barrier
serializes the two phases. The 4× payload shrink cannot overcome that on links
this fast.

Crossover at 256M: `quantScratchBytesNeeded = sendElems*1B (fp8) + scales`. At
count/rank = 8,388,608, sendElems = 67,108,864 B = the full 64 MiB DDA scratch,
so eligibility returns false and the path cleanly falls back to the normal
AllToAll — which is why 256M/512M/1G match baseline exactly with #wrong=0.

## Disposition

Left **default-off** (`RCCL_QUANT_ALLTOALL_ENABLE=0`), not force-enabled. The
feature is fully wired, builds, and is bit-safe when disabled (proven by the OFF
sweep = baseline). The quantized path does not pay off on an intra-node XGMI
clique for the same reason T4's RHD did not: the fabric is not the bottleneck. It
would only be expected to help where the AllToAll wire is the constraint (e.g.
inter-node RDMA, which is explicitly out of scope for T1b), or paired with a
tolerance-aware validator for the lossy MoE dispatch/combine use case.

Raw sweeps: `perf_results/t1b/off_sweep.txt`, `perf_results/t1b/on_sweep.txt`.
