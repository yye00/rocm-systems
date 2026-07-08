/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Quantized (lossy) MoE AllToAll dispatch/combine device kernels.
 *
 * AllToAll is a pure permute/shuffle: segment r of every rank's sendbuff is
 * routed to rank r.  There is NO reduction, so this path deliberately does
 * NOT touch the RS->reduce->AG or ncclSum machinery used by the AllReduce /
 * ReduceScatter DDA paths -- correctness depends only on the round-trip
 * quantize/dequantize error being bounded, not on ncclSum associativity.
 *
 * Codec: FP8 e4m3 with per-token (not per-block) amax scaling.  Each
 * contiguous group of kQuantTokenElems elements is one "token" and carries
 * its own fp32 scale, matching MoE token-sized traffic.  The quantized bytes
 * plus the per-token scales are staged in the DDA IPC scratch buffers.
 *
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves
 * correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"
#include "rccl_float8.h"

#include <cstdint>

namespace meta::comms {

// One token = this many contiguous elements sharing a single fp32 scale.
// MoE token-sized messages: 256 keeps the scale table tiny while giving each
// token an independent dynamic range.
constexpr int kQuantTokenElems = 256;

// OCP e4m3 max representable magnitude (gfx950 uses non-fnuz e4m3).
constexpr float kFp8E4m3Max = 448.0f;

// Number of tokens covering `count` elements (round up so a partial tail
// token still gets a scale).
__host__ __device__ inline size_t quantNumTokens(size_t count) {
  return (count + kQuantTokenElems - 1) / kQuantTokenElems;
}

// Scratch byte layout, per rank, for `sendElems` total send elements:
//   [ fp8 data : sendElems bytes ][ pad to 4B ][ scales : nTokens * 4 bytes ]
__host__ __device__ inline size_t quantScaleOffsetBytes(size_t sendElems) {
  return (sendElems + 3u) & ~static_cast<size_t>(3u);
}

__host__ __device__ inline size_t
quantScratchBytes(size_t sendElems, size_t nTokens) {
  return quantScaleOffsetBytes(sendElems) + nTokens * sizeof(float);
}

// Quantize `count` source elements (segment for one destination rank) into
// fp8 with a per-token scale.  One thread per token.
template <typename T>
static inline __device__ void quantizeTokens(
    const T* __restrict__ src,
    uint8_t* __restrict__ qdst,
    float* __restrict__ scales,
    size_t count,
    size_t gtIdx,
    size_t stride) {
  const size_t nTokens = quantNumTokens(count);
  for (size_t tok = gtIdx; tok < nTokens; tok += stride) {
    const size_t base = tok * kQuantTokenElems;
    const size_t end =
        base + kQuantTokenElems < count ? base + kQuantTokenElems : count;

    float amax = 0.0f;
    for (size_t i = base; i < end; ++i) {
      float v = static_cast<float>(src[i]);
      float a = v < 0.0f ? -v : v;
      amax = a > amax ? a : amax;
    }
    // scale maps the token's dynamic range into e4m3.  Guard the all-zero
    // token so we never divide by zero.
    float scale = amax > 0.0f ? amax / kFp8E4m3Max : 1.0f;
    scales[tok] = scale;

    float inv = 1.0f / scale;
    for (size_t i = base; i < end; ++i) {
      float v = static_cast<float>(src[i]) * inv;
      rccl_float8 q = static_cast<rccl_float8>(v);
      qdst[i] = *reinterpret_cast<const uint8_t*>(&q);
    }
  }
}

// Dequantize `count` fp8 elements (one source rank's segment for me) back into
// destination type using the per-token scales.  One thread per token.
template <typename T>
static inline __device__ void dequantizeTokens(
    const uint8_t* __restrict__ qsrc,
    const float* __restrict__ scales,
    T* __restrict__ dst,
    size_t count,
    size_t gtIdx,
    size_t stride) {
  const size_t nTokens = quantNumTokens(count);
  for (size_t tok = gtIdx; tok < nTokens; tok += stride) {
    const size_t base = tok * kQuantTokenElems;
    const size_t end =
        base + kQuantTokenElems < count ? base + kQuantTokenElems : count;
    const float scale = scales[tok];
    for (size_t i = base; i < end; ++i) {
      uint8_t byte = qsrc[i];
      rccl_float8 q = *reinterpret_cast<const rccl_float8*>(&byte);
      dst[i] = static_cast<T>(static_cast<float>(q) * scale);
    }
  }
}

// Single-kernel quantized AllToAll.
//
// ipcRaw[r]  : base of rank r's DDA IPC scratch (raw bytes).
// recvbuff   : my output, count elements per source rank (count * NRANKS).
// sendbuff   : my input,  count elements per destination rank (count * NRANKS).
// count      : elements per rank-pair segment.
// selfRank   : my rank.
//
// Phase 1: I quantize my whole sendbuff into my own scratch (fp8 + scales).
// Phase 2 (after barrier): for each source rank r I read the segment r sent to
// me (its segment index == selfRank) and dequantize it into recvbuff[r].
template <typename T, int NRANKS>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaQuantAllToAllIpc(
        uint8_t* const* __restrict__ ipcRaw,
        T* __restrict__ recvbuff,
        size_t count,
        const T* __restrict__ sendbuff,
        int selfRank,
        IpcGpuBarrier barrier) {
  const size_t sendElems = count * NRANKS;
  const size_t nTokensPerSeg = quantNumTokens(count);
  const size_t scaleOff = quantScaleOffsetBytes(sendElems);

  const size_t gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  uint8_t* myScratch = ipcRaw[selfRank];
  uint8_t* myQ = myScratch;
  float* myScales = reinterpret_cast<float*>(myScratch + scaleOff);

  // Phase 1: quantize every destination segment of my sendbuff.
#pragma unroll NRANKS
  for (int dst = 0; dst < NRANKS; ++dst) {
    quantizeTokens<T>(
        sendbuff + dst * count,
        myQ + dst * count,
        myScales + dst * nTokensPerSeg,
        count,
        gtIdx,
        stride);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // Phase 2: gather my segment from every source rank and dequantize.
#pragma unroll NRANKS
  for (int r = 0; r < NRANKS; ++r) {
    uint8_t* srcScratch = ipcRaw[r];
    const uint8_t* srcQ = srcScratch + selfRank * count;
    const float* srcScales =
        reinterpret_cast<const float*>(srcScratch + scaleOff) +
        selfRank * nTokensPerSeg;
    dequantizeTokens<T>(
        srcQ, srcScales, recvbuff + r * count, count, gtIdx, stride);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
