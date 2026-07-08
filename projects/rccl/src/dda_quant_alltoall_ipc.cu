/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaQuantAllToAllIpc, the lossy (fp8, per-token
 * scale) MoE AllToAll dispatch/combine kernel.  Reuses the DDA IPC scratch
 * buffers, peer-pointer table and IpcGpuBarrier set up by ipc_init.cu.
 * Gated by RCCL_QUANT_ALLTOALL_ENABLE (default 0).
 *
 * AllToAll is a pure permute/shuffle with NO reduction, so this path does not
 * touch the RS->reduce->AG or ncclSum machinery used by the AllReduce /
 * ReduceScatter DDA paths -- correctness depends only on the round-trip
 * quantize/dequantize error being bounded, not on ncclSum associativity.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/alltoall/quant_alltoall_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/alltoall/quant_alltoall_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

RCCL_PARAM(QuantAllToAllEnable, "QUANT_ALLTOALL_ENABLE", 0);

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

// Total scratch bytes one rank needs to stage its fp8 payload + per-token
// scales for a `count`-per-peer-segment AllToAll across `nRanks` peers.
static size_t quantScratchBytesNeeded(size_t count, int nRanks) {
  const size_t sendElems = count * static_cast<size_t>(nRanks);
  const size_t nTokensTotal =
      meta::comms::quantNumTokens(count) * static_cast<size_t>(nRanks);
  return meta::comms::quantScratchBytes(sendElems, nTokensTotal);
}

template <typename T>
static ncclResult_t ncclQuantAllToAllDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t needBytes = quantScratchBytesNeeded(count, comm->nRanks);
  if (needBytes > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC quant alltoall: fp8+scale staging needs %zu bytes; comm scratch is %zu bytes",
        needBytes,
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock =
      meta::comms::getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  // The DDA peer-pointer table holds each rank's scratch base; the quant kernel
  // treats those buffers as raw bytes (fp8 data followed by fp32 scales).
  uint8_t** d_ipcRaw =
      reinterpret_cast<uint8_t**>(comm->ddaIpcPeerPtrsDev);

  meta::comms::ddaQuantAllToAllIpc<T, kDdaNranks><<<grid, block, 0, stream>>>(
      d_ipcRaw,
      static_cast<T*>(recvbuff),
      count,
      static_cast<const T*>(sendbuff),
      comm->rank,
      barrierHost);
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclQuantAllToAllEnabled() {
  return rcclParamQuantAllToAllEnable() != 0;
}

bool ncclQuantAllToAllDdaIpcEligible(
    ncclComm* comm,
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t count,
    ncclDataType_t datatype) {
  if (!ncclQuantAllToAllEnabled()) {
    return false;
  }
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  // Lossy fp8 dispatch/combine is defined for floating-point element types only.
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }
  // Both the fp8 payload and per-token scales must fit the shared DDA scratch.
  if (quantScratchBytesNeeded(count, comm->nRanks) > comm->ddaIpcScratchBytes) {
    return false;
  }
  return true;
}

ncclResult_t ncclQuantAllToAllDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  INFO(
      NCCL_COLL,
      "RCCL QUANT ALLTOALL count=%zu datatype=%d rank=%d nRanks=%d",
      count,
      (int)datatype,
      comm->rank,
      comm->nRanks);
  switch (datatype) {
  case ncclFloat32:
    return ncclQuantAllToAllDdaIpcTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclQuantAllToAllDdaIpcTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclQuantAllToAllDdaIpcTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
