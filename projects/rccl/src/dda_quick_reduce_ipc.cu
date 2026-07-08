/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch the QuickReduce lossy block-quantized two-shot AllReduce
 * DDA kernel (meta::comms::ddaQuickReduceAllReduceIpc).  Reuses the DDA IPC
 * scratch buffers, peer-pointer table and IpcGpuBarrier set up by ipc_init.cu.
 *
 * LOSSY: gated by RCCL_QUICKREDUCE_ENABLE (default 0).  Never engaged on the
 * exact ncclSum default path unless the operator explicitly opts in, so the
 * bit-exact baseline is unaffected when the gate is off.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_quick_reduce_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/all_reduce/quick_reduce_codec.h"
#include "algorithms/all_reduce/quick_reduce_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <strings.h>

RCCL_PARAM(QuickReduceEnable, "QUICKREDUCE_ENABLE", 0);
// ~1 MiB: below this size the fixed per-block quantize/dequantize overhead and
// the extra IPC scratch round-trip outweigh the reduced XGMI byte volume.
RCCL_PARAM(QuickReduceMinBytes, "QUICKREDUCE_MIN_BYTES", 1048576);

using meta::comms::QrCodec;

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

// Parse RCCL_QUICKREDUCE_CODEC once.  Default FP4: highest peak speedup, native
// on gfx950 and INT4-fallback on gfx942 (see quick_reduce_codec.h).
static QrCodec quickReduceCodec() {
  const char* s = ncclGetEnv("RCCL_QUICKREDUCE_CODEC");
  if (s == nullptr) {
    return QrCodec::FP4;
  }
  if (strcasecmp(s, "q4") == 0) return QrCodec::Q4;
  if (strcasecmp(s, "q6") == 0) return QrCodec::Q6;
  if (strcasecmp(s, "q8") == 0) return QrCodec::Q8;
  if (strcasecmp(s, "fp8") == 0) return QrCodec::FP8;
  if (strcasecmp(s, "fp4") == 0) return QrCodec::FP4;
  WARN("RCCL_QUICKREDUCE_CODEC=%s not recognized; using fp4", s);
  return QrCodec::FP4;
}

static bool quickReduceCommUsable(const ncclComm* comm) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  // Relaxed from the exactly-8 DDA gate: QuickReduce serves 2/4/8 ranks.
  if (comm->nRanks != 2 && comm->nRanks != 4 && comm->nRanks != 8) {
    return false;
  }
  return true;
}

static bool quickReduceTypeOpOk(ncclDataType_t datatype, ncclRedOp_t op) {
  if (op != ncclSum) {
    return false;
  }
  return datatype == ncclFloat32 || datatype == ncclFloat16 ||
      datatype == ncclBfloat16;
}

template <typename T, int NRANKS, QrCodec C>
static void launchQuickReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t countPerRank,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock =
      meta::comms::getGridAndBlockDims(countPerRank, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  T** d_ipcbuffs = reinterpret_cast<T**>(comm->ddaIpcPeerPtrsDev);
  auto layout = meta::comms::qrScratchLayout(countPerRank, NRANKS, C);

  meta::comms::ddaQuickReduceAllReduceIpc<T, NRANKS, C>
      <<<grid, block, 0, stream>>>(
          d_ipcbuffs,
          static_cast<T*>(recvbuff),
          countPerRank,
          static_cast<const T*>(sendbuff),
          comm->rank,
          barrierHost,
          layout);
}

template <typename T, int NRANKS>
static ncclResult_t launchByCodec(
    const void* sendbuff,
    void* recvbuff,
    size_t countPerRank,
    QrCodec codec,
    ncclComm* comm,
    cudaStream_t stream) {
  switch (codec) {
  case QrCodec::Q4:
    launchQuickReduce<T, NRANKS, QrCodec::Q4>(
        sendbuff, recvbuff, countPerRank, comm, stream);
    break;
  case QrCodec::Q6:
    launchQuickReduce<T, NRANKS, QrCodec::Q6>(
        sendbuff, recvbuff, countPerRank, comm, stream);
    break;
  case QrCodec::Q8:
    launchQuickReduce<T, NRANKS, QrCodec::Q8>(
        sendbuff, recvbuff, countPerRank, comm, stream);
    break;
  case QrCodec::FP8:
    launchQuickReduce<T, NRANKS, QrCodec::FP8>(
        sendbuff, recvbuff, countPerRank, comm, stream);
    break;
  case QrCodec::FP4:
    launchQuickReduce<T, NRANKS, QrCodec::FP4>(
        sendbuff, recvbuff, countPerRank, comm, stream);
    break;
  default:
    return ncclInvalidArgument;
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t launchByRanks(
    const void* sendbuff,
    void* recvbuff,
    size_t countPerRank,
    int nranks,
    QrCodec codec,
    ncclComm* comm,
    cudaStream_t stream) {
  switch (nranks) {
  case 2:
    return launchByCodec<T, 2>(
        sendbuff, recvbuff, countPerRank, codec, comm, stream);
  case 4:
    return launchByCodec<T, 4>(
        sendbuff, recvbuff, countPerRank, codec, comm, stream);
  case 8:
    return launchByCodec<T, 8>(
        sendbuff, recvbuff, countPerRank, codec, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}

} // namespace

bool ncclQuickReduceEnabled() {
  return rcclParamQuickReduceEnable() != 0;
}

float ncclQuickReduceCompressionRatio() {
  if (!ncclQuickReduceEnabled()) {
    return 1.0f;
  }
  const QrCodec c = quickReduceCodec();
  // Effective quantized bytes per element: codec payload + fp16 scale amortized
  // across a 32-element block.  Ratio is versus fp16 wire precision (the
  // smallest full-precision type QuickReduce serves); >= 1.
  const float payloadPerElem =
      (float)meta::comms::qrCodecBlockBytes(c) / (float)meta::comms::kQrBlockElems;
  const float scalePerElem = 2.0f / (float)meta::comms::kQrBlockElems;
  const float qBytes = payloadPerElem + scalePerElem;
  const float ratio = 2.0f / qBytes; // vs 2-byte fp16
  return ratio > 1.0f ? ratio : 1.0f;
}

bool ncclAllReduceQuickReduceIpcEligible(
    ncclComm* comm,
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (!ncclQuickReduceEnabled()) {
    return false;
  }
  if (!quickReduceCommUsable(comm)) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (!quickReduceTypeOpOk(datatype, op)) {
    return false;
  }

  const size_t typeSize = ncclTypeSize(datatype);
  // Below the crossover the quantize overhead dominates; leave to baseline.
  if (count * typeSize < (size_t)rcclParamQuickReduceMinBytes()) {
    return false;
  }
  // Two-shot RS/AG scatters the buffer into nRanks equal segments.
  if (count % comm->nRanks) {
    return false;
  }
  const size_t countPerRank = count / comm->nRanks;
  const QrCodec codec = quickReduceCodec();
  const size_t needed =
      meta::comms::qrScratchBytes(countPerRank, comm->nRanks, codec);
  if (needed > comm->ddaIpcScratchBytes) {
    return false;
  }
  return true;
}

ncclResult_t ncclAllReduceQuickReduceIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  const QrCodec codec = quickReduceCodec();
  const size_t countPerRank = count / comm->nRanks;
  INFO(
      NCCL_COLL,
      "RCCL QUICKREDUCE ALL-REDUCE (LOSSY) count=%zu datatype=%d codec=%d rank=%d nRanks=%d",
      count,
      (int)datatype,
      (int)codec,
      comm->rank,
      comm->nRanks);
  switch (datatype) {
  case ncclFloat32:
    return launchByRanks<float>(
        sendbuff, recvbuff, countPerRank, comm->nRanks, codec, comm, stream);
  case ncclFloat16:
    return launchByRanks<half>(
        sendbuff, recvbuff, countPerRank, comm->nRanks, codec, comm, stream);
  case ncclBfloat16:
    return launchByRanks<bf16>(
        sendbuff, recvbuff, countPerRank, comm->nRanks, codec, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
