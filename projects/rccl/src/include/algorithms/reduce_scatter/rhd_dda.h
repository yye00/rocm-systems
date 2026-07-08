/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Log-round recursive-halving/doubling DDA kernels (single node, IPC).
 *
 * ReduceScatter: ceil(log2 p) rounds of recursive halving.  At round k each
 * rank pairs with partner = self ^ (1 << (logp-1-k)) and accumulates the
 * half of the segment space it will keep.  After logp rounds rank r holds the
 * fully reduced segment r in its own scratch buffer (p-1 total volume, the
 * same as ring, but in logp barriers instead of p-1).
 *
 * AllReduce = recursive-halving RS followed by the reversed recursive-doubling
 * AllGather, so the whole collective completes in 2*logp rounds.
 *
 * Bit-exact / lossless: the per-element reduction order is fixed by the
 * pairwise tree and does not depend on data.  Includes use *.h names so RCCL
 * hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

// ceil(log2(NRANKS)) for a power-of-two rank count, evaluated at compile time.
template <int NRANKS>
struct RhdLog2 {
  static constexpr int value = 1 + RhdLog2<(NRANKS >> 1)>::value;
};
template <>
struct RhdLog2<1> {
  static constexpr int value = 0;
};

// Recursive-halving reduce-scatter.  `count` is the per-rank (per-segment)
// element count; the scratch buffers hold NRANKS contiguous segments each of
// `count` elements.  The host copies this rank's sendbuff into
// ipcbuffs[selfRank] before launch, so the first barrier only needs acquire.
template <typename T, int NRANKS>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaReduceScatterRhdIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    int selfRank,
    IpcGpuBarrier barrier) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  static_assert((NRANKS & (NRANKS - 1)) == 0, "rhd: NRANKS must be a power of 2");
  constexpr int kLogP = RhdLog2<NRANKS>::value;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);

  const size_t gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t eStart = gtIdx * countPerThread;
  const size_t eStride = static_cast<size_t>(gridDim.x) * blockDim.x * countPerThread;

  // Ensure every rank has published its sendbuff into scratch.
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

#pragma unroll
  for (int k = 0; k < kLogP; ++k) {
    if (k > 0) {
      barrier.syncOnSameBlockIdx<
          true /* hasPreviousMemAccess */,
          true /* hasSubsequentMemAccess */>();
    }
    const int shift = kLogP - 1 - k;
    const int blk = 1 << shift;
    const int partner = selfRank ^ (1 << shift);
    const int segStart = (selfRank >> shift) << shift;

    for (int s = segStart; s < segStart + blk; ++s) {
      const size_t base = static_cast<size_t>(s) * count;
      for (size_t e = eStart; e < count; e += eStride) {
        const uint4 a =
            reinterpret_cast<const uint4*>(&ipcbuffs[selfRank][base + e])[0];
        const uint4 b =
            reinterpret_cast<const uint4*>(&ipcbuffs[partner][base + e])[0];
        const uint4 sum = vecElementAdd<T>(a, b);
        *reinterpret_cast<uint4*>(&ipcbuffs[selfRank][base + e]) = sum;
      }
    }
  }

  // Final reduced segment for this rank now lives in ipcbuffs[selfRank].
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();

  const size_t base = static_cast<size_t>(selfRank) * count;
  for (size_t e = eStart; e < count; e += eStride) {
    *reinterpret_cast<uint4*>(&recvbuff[e]) =
        reinterpret_cast<const uint4*>(&ipcbuffs[selfRank][base + e])[0];
  }
}

// Recursive-halving RS followed by the reversed recursive-doubling AllGather.
// `count` is the per-rank (per-segment) element count; total elements are
// NRANKS * count.  On exit recvbuff holds all NRANKS reduced segments.
template <typename T, int NRANKS>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceRhdIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    int selfRank,
    IpcGpuBarrier barrier) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  static_assert((NRANKS & (NRANKS - 1)) == 0, "rhd: NRANKS must be a power of 2");
  constexpr int kLogP = RhdLog2<NRANKS>::value;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);

  const size_t gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t eStart = gtIdx * countPerThread;
  const size_t eStride = static_cast<size_t>(gridDim.x) * blockDim.x * countPerThread;

  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // ---- Recursive-halving reduce-scatter ----
#pragma unroll
  for (int k = 0; k < kLogP; ++k) {
    if (k > 0) {
      barrier.syncOnSameBlockIdx<
          true /* hasPreviousMemAccess */,
          true /* hasSubsequentMemAccess */>();
    }
    const int shift = kLogP - 1 - k;
    const int blk = 1 << shift;
    const int partner = selfRank ^ (1 << shift);
    const int segStart = (selfRank >> shift) << shift;

    for (int s = segStart; s < segStart + blk; ++s) {
      const size_t base = static_cast<size_t>(s) * count;
      for (size_t e = eStart; e < count; e += eStride) {
        const uint4 a =
            reinterpret_cast<const uint4*>(&ipcbuffs[selfRank][base + e])[0];
        const uint4 b =
            reinterpret_cast<const uint4*>(&ipcbuffs[partner][base + e])[0];
        const uint4 sum = vecElementAdd<T>(a, b);
        *reinterpret_cast<uint4*>(&ipcbuffs[selfRank][base + e]) = sum;
      }
    }
  }

  // ---- Reversed recursive-doubling all-gather ----
#pragma unroll
  for (int j = 0; j < kLogP; ++j) {
    barrier.syncOnSameBlockIdx<
        true /* hasPreviousMemAccess */,
        true /* hasSubsequentMemAccess */>();
    const int shift = j;
    const int blk = 1 << shift;
    const int partner = selfRank ^ (1 << shift);
    const int segStart = (partner >> shift) << shift;

    for (int s = segStart; s < segStart + blk; ++s) {
      const size_t base = static_cast<size_t>(s) * count;
      for (size_t e = eStart; e < count; e += eStride) {
        *reinterpret_cast<uint4*>(&ipcbuffs[selfRank][base + e]) =
            reinterpret_cast<const uint4*>(&ipcbuffs[partner][base + e])[0];
      }
    }
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();

  for (int s = 0; s < NRANKS; ++s) {
    const size_t base = static_cast<size_t>(s) * count;
    for (size_t e = eStart; e < count; e += eStride) {
      *reinterpret_cast<uint4*>(&recvbuff[base + e]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[selfRank][base + e])[0];
    }
  }
}

} // namespace meta::comms
