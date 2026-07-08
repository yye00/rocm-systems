/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch the log-round recursive-halving/doubling DDA kernels
 * (meta::comms::ddaReduceScatterRhdIpc / ddaAllReduceRhdIpc).  Reuses the
 * DDA IPC scratch buffers, peer-pointer table and IpcGpuBarrier set up by
 * ipc_init.cu.  Gated by RCCL_RHD_ENABLE (default 0).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_recursive_halving_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/reduce_scatter/rhd_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>

RCCL_PARAM(RhdEnable, "RHD_ENABLE", 0);

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

// Common IPC-resource / power-of-two-rank guards shared by RS and AR paths.
static bool rhdCommUsable(const ncclComm* comm) {
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
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  // Recursive halving requires a power-of-two rank count.
  if ((comm->nRanks & (comm->nRanks - 1)) != 0) {
    return false;
  }
  return true;
}

static bool rhdTypeOpOk(ncclDataType_t datatype, ncclRedOp_t op) {
  if (op != ncclSum) {
    return false;
  }
  return datatype == ncclFloat32 || datatype == ncclFloat16 ||
      datatype == ncclBfloat16;
}

template <typename T>
static ncclResult_t reduceScatterRhdTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclComm* comm,
    cudaStream_t stream) {
  const size_t totalCount = recvcount * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC RHD reduce-scatter: total element count %zu needs %zu bytes; comm scratch is %zu bytes",
        totalCount,
        totalCount * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock = meta::comms::getGridAndBlockDims(recvcount, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  T** d_ipcbuffs = reinterpret_cast<T**>(comm->ddaIpcPeerPtrsDev);

  CUDACHECK(cudaMemcpyAsync(
      comm->ddaIpcScratch,
      sendbuff,
      totalCount * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream));

  meta::comms::ddaReduceScatterRhdIpc<T, kDdaNranks>
      <<<grid, block, 0, stream>>>(
          d_ipcbuffs,
          static_cast<T*>(recvbuff),
          recvcount,
          comm->rank,
          barrierHost);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t allReduceRhdTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (count * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC RHD all-reduce: element count %zu needs %zu bytes; comm scratch is %zu bytes",
        count,
        count * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const size_t countPerRank = count / comm->nRanks;
  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock =
      meta::comms::getGridAndBlockDims(countPerRank, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  T** d_ipcbuffs = reinterpret_cast<T**>(comm->ddaIpcPeerPtrsDev);

  CUDACHECK(cudaMemcpyAsync(
      comm->ddaIpcScratch,
      sendbuff,
      count * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream));

  meta::comms::ddaAllReduceRhdIpc<T, kDdaNranks>
      <<<grid, block, 0, stream>>>(
          d_ipcbuffs,
          static_cast<T*>(recvbuff),
          countPerRank,
          comm->rank,
          barrierHost);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclRhdEnabled() {
  return rcclParamRhdEnable() != 0;
}

bool ncclReduceScatterRhdIpcEligible(
    ncclComm* comm,
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (!ncclRhdEnabled()) {
    return false;
  }
  if (!rhdCommUsable(comm)) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (!rhdTypeOpOk(datatype, op)) {
    return false;
  }

  const size_t typeSize = ncclTypeSize(datatype);
  const size_t totalCount = recvcount * comm->nRanks;
  if (totalCount * typeSize > comm->ddaIpcScratchBytes) {
    return false;
  }
  // 16-byte loads: each per-rank segment must be 16-byte aligned.
  if ((recvcount * typeSize) % 16) {
    return false;
  }
  return true;
}

ncclResult_t ncclReduceScatterRhdIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  INFO(
      NCCL_COLL,
      "RCCL RHD REDUCE-SCATTER recvcount=%zu datatype=%d rank=%d nRanks=%d",
      recvcount,
      (int)datatype,
      comm->rank,
      comm->nRanks);
  switch (datatype) {
  case ncclFloat32:
    return reduceScatterRhdTyped<float>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return reduceScatterRhdTyped<half>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return reduceScatterRhdTyped<bf16>(
        sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}

bool ncclAllReduceRhdIpcEligible(
    ncclComm* comm,
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (!ncclRhdEnabled()) {
    return false;
  }
  if (!rhdCommUsable(comm)) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (!rhdTypeOpOk(datatype, op)) {
    return false;
  }

  const size_t typeSize = ncclTypeSize(datatype);
  if (count * typeSize > comm->ddaIpcScratchBytes) {
    return false;
  }
  // Recursive halving scatters the buffer into nRanks equal segments.
  if (count % comm->nRanks) {
    return false;
  }
  // 16-byte loads: each per-rank segment must be 16-byte aligned.
  const size_t countPerRank = count / comm->nRanks;
  if ((countPerRank * typeSize) % 16) {
    return false;
  }
  return true;
}

ncclResult_t ncclAllReduceRhdIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  INFO(
      NCCL_COLL,
      "RCCL RHD ALL-REDUCE count=%zu datatype=%d rank=%d nRanks=%d",
      count,
      (int)datatype,
      comm->rank,
      comm->nRanks);
  switch (datatype) {
  case ncclFloat32:
    return allReduceRhdTyped<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return allReduceRhdTyped<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return allReduceRhdTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
