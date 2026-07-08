/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * QuickReduce: lossy block-quantized two-shot AllReduce DDA kernel
 * (single node, IPC).  Modeled on ddaAllReduceTreeIpc
 * (all_reduce_dda.h:56) but every XGMI transfer carries block-32 quantized
 * payload plus a per-block fp16 scale (see quick_reduce_codec.h), so the
 * reduce-scatter and all-gather rounds move a fraction of the full-precision
 * byte volume.
 *
 * Three phases, two IPC barriers between them:
 *   1. quantize: each rank encodes its NRANKS send segments into its own
 *      IPC scratch (payload region A + scale region B).
 *   2. reduce:   rank r reads segment r from every peer, dequantizes into
 *      fp32, sums, and re-encodes the reduced segment into its own result
 *      region (C + D).
 *   3. allgather: every rank reads every peer's reduced result segment,
 *      dequantizes, and writes it into recvbuff at the segment offset.
 *
 * LOSSY: correctness is bounded by the round-trip quantize/dequantize error,
 * not by ncclSum associativity.  Off by default; never selected on the exact
 * ncclSum default path.  Includes use *.h names so RCCL hipify output resolves.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"
#include "algorithms/all_reduce/quick_reduce_codec.h"

namespace meta::comms {

// Element <-> fp32 conversions for the codec's fp32 working precision.
template <typename T>
__device__ inline float qrLoadFloat(const T* p) {
  if constexpr (std::is_same<T, float>::value) {
    return *p;
  } else if constexpr (std::is_same<T, half>::value) {
    return __half2float(*p);
  } else {
    return __bfloat162float(*p);
  }
}

template <typename T>
__device__ inline void qrStoreFloat(T* p, float v) {
  if constexpr (std::is_same<T, float>::value) {
    *p = v;
  } else if constexpr (std::is_same<T, half>::value) {
    *p = __float2half(v);
  } else {
    *p = __float2bfloat16(v);
  }
}

// Byte offsets of the four scratch regions for one rank.  Must match the
// layout assumed by qrScratchBytes() in quick_reduce_codec.h.
struct QrScratchLayout {
  size_t offPayload;  // A: NRANKS send segments
  size_t offScale;    // B: NRANKS send scale tables
  size_t offResultP;  // C: 1 reduced-result segment payload
  size_t offResultS;  // D: 1 reduced-result segment scale table
  size_t segPayload;  // payload bytes per segment
  size_t segScale;    // scale bytes per segment
};

__host__ __device__ inline QrScratchLayout
qrScratchLayout(size_t countPerRank, int nranks, QrCodec c) {
  QrScratchLayout l{};
  l.segPayload = qrSegPayloadBytes(countPerRank, c);
  l.segScale = qrSegScaleBytes(countPerRank);
  l.offPayload = 0;
  l.offScale = qrAlignUp((size_t)nranks * l.segPayload, 16);
  l.offResultP = qrAlignUp(l.offScale + (size_t)nranks * l.segScale, 16);
  l.offResultS = qrAlignUp(l.offResultP + l.segPayload, 16);
  return l;
}

// Two-shot block-quantized AllReduce.  `countPerRank` is the per-segment
// element count; total elements are NRANKS * countPerRank.  ipcbuffs[r] points
// at rank r's IPC scratch, sized >= qrScratchBytes(countPerRank, NRANKS, C).
// On exit recvbuff holds all NRANKS reduced segments.
template <typename T, int NRANKS, QrCodec C>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaQuickReduceAllReduceIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t countPerRank,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    QrScratchLayout layout) {
  static_assert(is_supported_type_v<T>, "qr: unsupported element type");
  using Codec = QuickReduceCodec<C>;
  constexpr int kBB = qrCodecBlockBytes(C);

  const size_t numBlk = qrNumBlocks(countPerRank);
  const size_t gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  uint8_t* selfBase = reinterpret_cast<uint8_t*>(ipcbuffs[selfRank]);

  // ---- Phase 1: quantize this rank's NRANKS send segments ----
  const size_t totalBlk = (size_t)NRANKS * numBlk;
  for (size_t bi = gtIdx; bi < totalBlk; bi += stride) {
    const int seg = (int)(bi / numBlk);
    const size_t blk = bi % numBlk;
    const size_t e0 = blk * kQrBlockElems;
    const int n = (int)min((size_t)kQrBlockElems, countPerRank - e0);

    float vals[kQrBlockElems];
    const T* src = sendbuff + (size_t)seg * countPerRank + e0;
    for (int i = 0; i < n; ++i) vals[i] = qrLoadFloat<T>(src + i);

    uint8_t* qout =
        selfBase + layout.offPayload + (size_t)seg * layout.segPayload + blk * kBB;
    __half* sout = reinterpret_cast<__half*>(
        selfBase + layout.offScale + (size_t)seg * layout.segScale +
        blk * sizeof(uint16_t));
    Codec::encode(vals, n, qout, sout);
  }

  barrier.syncOnSameBlockIdx<true, true>();

  // ---- Phase 2: reduce segment `selfRank` across all peers, re-encode ----
  for (size_t blk = gtIdx; blk < numBlk; blk += stride) {
    const size_t e0 = blk * kQrBlockElems;
    const int n = (int)min((size_t)kQrBlockElems, countPerRank - e0);

    float acc[kQrBlockElems];
    for (int i = 0; i < n; ++i) acc[i] = 0.0f;

    for (int p = 0; p < NRANKS; ++p) {
      uint8_t* peer = reinterpret_cast<uint8_t*>(ipcbuffs[p]);
      const uint8_t* qin = peer + layout.offPayload +
          (size_t)selfRank * layout.segPayload + blk * kBB;
      const __half scale = *reinterpret_cast<const __half*>(
          peer + layout.offScale + (size_t)selfRank * layout.segScale +
          blk * sizeof(uint16_t));
      float tmp[kQrBlockElems];
      Codec::decode(qin, scale, n, tmp);
      for (int i = 0; i < n; ++i) acc[i] += tmp[i];
    }

    uint8_t* qout = selfBase + layout.offResultP + blk * kBB;
    __half* sout = reinterpret_cast<__half*>(
        selfBase + layout.offResultS + blk * sizeof(uint16_t));
    Codec::encode(acc, n, qout, sout);
  }

  barrier.syncOnSameBlockIdx<true, true>();

  // ---- Phase 3: all-gather every peer's reduced result segment ----
  for (size_t bi = gtIdx; bi < totalBlk; bi += stride) {
    const int seg = (int)(bi / numBlk);
    const size_t blk = bi % numBlk;
    const size_t e0 = blk * kQrBlockElems;
    const int n = (int)min((size_t)kQrBlockElems, countPerRank - e0);

    uint8_t* peer = reinterpret_cast<uint8_t*>(ipcbuffs[seg]);
    const uint8_t* qin = peer + layout.offResultP + blk * kBB;
    const __half scale = *reinterpret_cast<const __half*>(
        peer + layout.offResultS + blk * sizeof(uint16_t));
    float out[kQrBlockElems];
    Codec::decode(qin, scale, n, out);

    T* dst = recvbuff + (size_t)seg * countPerRank + e0;
    for (int i = 0; i < n; ++i) qrStoreFloat<T>(dst + i, out[i]);
  }

  barrier.syncOnSameBlockIdx<true, false>();
}

} // namespace meta::comms
