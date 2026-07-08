/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * QuickReduce block-quantized codecs for the lossy two-shot AllReduce
 * (RS -> reduce -> AG) DDA path.  Every XGMI transfer in that path carries
 * block-32 quantized data plus a per-block FP16 scale, so the reduction pays
 * a fraction of the full-precision XGMI byte volume.
 *
 * Codecs (compile-time selected):
 *   Q4  : 4-bit signed integer, per-block fp16 scale   (16 B / 32 elems)
 *   Q6  : 6-bit signed integer, per-block fp16 scale   (24 B / 32 elems)
 *   Q8  : 8-bit signed integer, per-block fp16 scale   (32 B / 32 elems)
 *   FP8 : OCP e4m3,             per-block fp16 scale   (32 B / 32 elems)
 *   FP4 : OCP e2m1,             per-block fp16 scale   (16 B / 32 elems)
 *
 * FP4 uses the native gfx950 conversion builtins
 * (__builtin_amdgcn_cvt_scalef32_pk_fp4_f16 / ..._pk_f16_fp4).  On gfx942 the
 * FP4 codec is #if-guarded to fall back to the Q4 INT codec, so a single fat
 * object compiles for both arches.  INT and FP8 codecs are arch-independent.
 *
 * LOSSY: this path is off by default (RCCL_QUICKREDUCE_ENABLE=0) and is never
 * selected on the exact ncclSum default path.  Correctness is bounded by the
 * round-trip quantize/dequantize error, not by ncclSum associativity.
 *
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves
 * correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "rccl_float8.h"

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/amd_detail/amd_hip_fp4.h>
#endif

namespace meta::comms {

// One quantization block = this many contiguous elements sharing a single
// fp16 scale.  32 keeps the scale table small (1/16 byte per element) while
// giving each block an independent dynamic range.
constexpr int kQrBlockElems = 32;

enum class QrCodec : int {
  Q4 = 0,
  Q6 = 1,
  Q8 = 2,
  FP8 = 3,
  FP4 = 4,
};

// Bytes of quantized payload one block occupies for a given codec (excludes
// the 2-byte fp16 scale, which is stored in a separate scale table).
__host__ __device__ inline constexpr int qrCodecBlockBytes(QrCodec c) {
  return c == QrCodec::Q4   ? (kQrBlockElems * 4) / 8   // 16
      : c == QrCodec::Q6    ? (kQrBlockElems * 6) / 8   // 24
      : c == QrCodec::Q8    ? (kQrBlockElems * 8) / 8   // 32
      : c == QrCodec::FP8   ? kQrBlockElems             // 32
                            : (kQrBlockElems * 4) / 8;  // FP4 -> 16
}

// Number of blocks covering `count` elements (round up so a partial tail block
// still gets a scale).
__host__ __device__ inline size_t qrNumBlocks(size_t count) {
  return (count + kQrBlockElems - 1) / kQrBlockElems;
}

__host__ __device__ inline size_t qrAlignUp(size_t x, size_t a) {
  return (x + (a - 1)) & ~(a - 1);
}

// Quantized payload bytes for one `count`-element segment under codec `c`.
__host__ __device__ inline size_t qrSegPayloadBytes(size_t count, QrCodec c) {
  return qrNumBlocks(count) * (size_t)qrCodecBlockBytes(c);
}

// Per-block fp16 scale table bytes for one `count`-element segment.
__host__ __device__ inline size_t qrSegScaleBytes(size_t count) {
  return qrNumBlocks(count) * sizeof(uint16_t);
}

// Total per-rank DDA-IPC scratch bytes QuickReduce needs to stage NRANKS
// quantized send segments plus one quantized reduced-result segment.
//   [ send payload : NRANKS*segPayload ][ pad ]
//   [ send scales  : NRANKS*segScale   ][ pad ]
//   [ result payload : segPayload ][ pad ]
//   [ result scales  : segScale ]
__host__ __device__ inline size_t
qrScratchBytes(size_t count, int nranks, QrCodec c) {
  const size_t segP = qrSegPayloadBytes(count, c);
  const size_t segS = qrSegScaleBytes(count);
  size_t off = 0;
  off = qrAlignUp(off + (size_t)nranks * segP, 16); // after send payload
  off = qrAlignUp(off + (size_t)nranks * segS, 16); // after send scales
  off = qrAlignUp(off + segP, 16);                  // after result payload
  off = off + segS;                                 // after result scales
  return off;
}

// OCP maxima (gfx950 non-fnuz encodings).
constexpr float kQrFp8E4m3Max = 448.0f;
constexpr float kQrFp4E2m1Max = 6.0f;

// ------------------------------------------------------------------------
// Little-endian bit field pack/unpack over a per-block byte buffer.  Used by
// the INT codecs (Q4/Q6/Q8).  `buf` must be zero-initialized before packing.
// ------------------------------------------------------------------------
__device__ inline void qrPutBits(
    uint8_t* __restrict__ buf, int bitpos, uint32_t val, int nbits) {
  for (int i = 0; i < nbits; ++i) {
    const int b = bitpos + i;
    const uint8_t bit = (val >> i) & 1u;
    buf[b >> 3] = (uint8_t)((buf[b >> 3] & ~(1u << (b & 7))) | (bit << (b & 7)));
  }
}

__device__ inline uint32_t
qrGetBits(const uint8_t* __restrict__ buf, int bitpos, int nbits) {
  uint32_t v = 0;
  for (int i = 0; i < nbits; ++i) {
    const int b = bitpos + i;
    v |= ((uint32_t)((buf[b >> 3] >> (b & 7)) & 1u)) << i;
  }
  return v;
}

// ------------------------------------------------------------------------
// FP4 (e2m1) single-element conversions.  Native builtins on gfx950; the INT4
// path is used for host compilation and as the gfx942 fallback.
// ------------------------------------------------------------------------
__device__ inline uint8_t qrFp4FromFloat(float v) {
#if defined(__gfx950__)
  __half h = __float2half(v);
  __half_raw hr = *reinterpret_cast<__half_raw*>(&h);
  return (uint8_t)__hip_cvt_halfraw_to_fp4(
      hr, __HIP_E2M1, hipRoundNearest);
#else
  // gfx942 / host fallback: 4-bit signed integer (INT4) codec on a unit range.
  int q = (int)rintf(v * (7.0f / kQrFp4E2m1Max));
  if (q > 7) q = 7;
  if (q < -8) q = -8;
  return (uint8_t)(q & 0xF);
#endif
}

__device__ inline float qrFp4ToFloat(uint8_t nib) {
#if defined(__gfx950__)
  // NOTE: do NOT use __hip_cvt_fp4_to_halfraw here. In ROCm 7.0.1 that header
  // wrapper unpacks with scale=0, which zeroes every result (verified: all
  // decoded values came back 0.0). Call the unpack builtin directly with
  // scale=1.0f -- the per-block fp16 scale is applied separately in decode(),
  // so this conversion must be an identity-scale dequant.
  auto packed =
      __builtin_amdgcn_cvt_scalef32_pk_f16_fp4((unsigned)(nib & 0xF), 1.0f, 0);
  __half h = *reinterpret_cast<__half*>(&packed);
  return __half2float(h);
#else
  int q = nib & 0xF;
  if (q > 7) q -= 16;
  return (float)q * (kQrFp4E2m1Max / 7.0f);
#endif
}

// ------------------------------------------------------------------------
// Codec traits: block encode/decode.  One thread owns one block.
// ------------------------------------------------------------------------
template <QrCodec C>
struct QuickReduceCodec {
  static __host__ __device__ constexpr int blockBytes() {
    return qrCodecBlockBytes(C);
  }

  // Encode up to kQrBlockElems values (n <= kQrBlockElems) into qout, emitting
  // one fp16 scale.  vals are read in fp32.
  static __device__ void
  encode(const float* __restrict__ vals, int n, uint8_t* __restrict__ qout, __half* __restrict__ scaleOut) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
      float a = vals[i] < 0.0f ? -vals[i] : vals[i];
      amax = a > amax ? a : amax;
    }

    if constexpr (C == QrCodec::Q4 || C == QrCodec::Q6 || C == QrCodec::Q8) {
      constexpr int kBits = C == QrCodec::Q4 ? 4 : (C == QrCodec::Q6 ? 6 : 8);
      constexpr int kMaxLevel = (1 << (kBits - 1)) - 1; // 7 / 31 / 127
      float scale = amax > 0.0f ? amax / (float)kMaxLevel : 1.0f;
      *scaleOut = __float2half(scale);
      float inv = 1.0f / scale;
      for (int b = 0; b < blockBytes(); ++b) qout[b] = 0;
      for (int i = 0; i < n; ++i) {
        int q = (int)rintf(vals[i] * inv);
        if (q > kMaxLevel) q = kMaxLevel;
        if (q < -(kMaxLevel + 1)) q = -(kMaxLevel + 1);
        qrPutBits(qout, i * kBits, (uint32_t)(q & ((1 << kBits) - 1)), kBits);
      }
    } else if constexpr (C == QrCodec::FP8) {
      float scale = amax > 0.0f ? amax / kQrFp8E4m3Max : 1.0f;
      *scaleOut = __float2half(scale);
      float inv = 1.0f / scale;
      for (int i = 0; i < n; ++i) {
        rccl_float8 q = static_cast<rccl_float8>(vals[i] * inv);
        qout[i] = *reinterpret_cast<const uint8_t*>(&q);
      }
      for (int i = n; i < kQrBlockElems; ++i) qout[i] = 0;
    } else { // FP4
      float scale = amax > 0.0f ? amax / kQrFp4E2m1Max : 1.0f;
      *scaleOut = __float2half(scale);
      float inv = 1.0f / scale;
      for (int b = 0; b < blockBytes(); ++b) qout[b] = 0;
      for (int i = 0; i < n; ++i) {
        uint8_t nib = qrFp4FromFloat(vals[i] * inv) & 0xF;
        // two nibbles per byte, low nibble first.
        qout[i >> 1] |= (i & 1) ? (uint8_t)(nib << 4) : nib;
      }
    }
  }

  // Decode n values from qin using the fp16 scale.
  static __device__ void
  decode(const uint8_t* __restrict__ qin, __half scale, int n, float* __restrict__ out) {
    float s = __half2float(scale);
    if constexpr (C == QrCodec::Q4 || C == QrCodec::Q6 || C == QrCodec::Q8) {
      constexpr int kBits = C == QrCodec::Q4 ? 4 : (C == QrCodec::Q6 ? 6 : 8);
      for (int i = 0; i < n; ++i) {
        uint32_t raw = qrGetBits(qin, i * kBits, kBits);
        int q = (int)raw;
        if (q >= (1 << (kBits - 1))) q -= (1 << kBits); // sign-extend
        out[i] = (float)q * s;
      }
    } else if constexpr (C == QrCodec::FP8) {
      for (int i = 0; i < n; ++i) {
        uint8_t byte = qin[i];
        rccl_float8 q = *reinterpret_cast<const rccl_float8*>(&byte);
        out[i] = static_cast<float>(q) * s;
      }
    } else { // FP4
      for (int i = 0; i < n; ++i) {
        uint8_t nib = (i & 1) ? (uint8_t)(qin[i >> 1] >> 4) : (uint8_t)(qin[i >> 1] & 0xF);
        out[i] = qrFp4ToFloat(nib) * s;
      }
    }
  }
};

} // namespace meta::comms
