/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/all_reduce/all_reduce_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

// Relax the DDA/symmetric AllReduce eligibility beyond exactly 8 ranks. When 0
// (default) the classic 8-rank-only gate is enforced and behaviour is bit- and
// perf-identical to baseline. When 1, comms of 2/4/8 ranks are eligible. Only
// power-of-two participant counts <= kDdaNranks are supported because the DDA
// kernels are template-instantiated per participant count.
RCCL_PARAM(DdaNranksRelax, "DDA_NRANKS_RELAX", 0);

bool ncclDdaNranksRelaxEnabled() {
  return rcclParamDdaNranksRelax() != 0;
}

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

/** Flat below this size; tree above (see ddaAllReduceFlatIpc / ddaAllReduceTreeIpc). */
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

/** True when nRanks is a supported DDA participant count. */
static bool ddaNranksSupported(int nRanks) {
  if (nRanks == kDdaNranks) {
    return true;
  }
  if (!ncclDdaNranksRelaxEnabled()) {
    return false;
  }
  return nRanks == 2 || nRanks == 4 || nRanks == 8;
}

template <typename T, int NRANKS>
static ncclResult_t ncclAllReduceDdaIpcLaunch(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const size_t sizeBytes = count * sizeof(T);
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk =
      wantTree && (count % static_cast<size_t>(NRANKS) == 0);

  if (wantTree && !treeOk) {
    INFO(
        NCCL_ALL,
        "DDA IPC: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel",
        sizeBytes,
        count,
        NRANKS);
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock = meta::comms::getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaIpcPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(
        comm->ddaIpcScratch,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));
    meta::comms::ddaAllReduceTreeIpc<T, NRANKS, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  } else {
    meta::comms::ddaAllReduceFlatIpc<T, NRANKS, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  }

  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes",
        count,
        count * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  // Dispatch to the template instantiation for the active participant count.
  // Only power-of-two counts <= kDdaNranks are instantiated; ncclAllReduceDdaIpcEligible
  // guarantees comm->nRanks is one of these before we get here.
  switch (comm->nRanks) {
  case 8:
    return ncclAllReduceDdaIpcLaunch<T, 8>(
        sendbuff, recvbuff, count, comm, stream);
  case 4:
    return ncclAllReduceDdaIpcLaunch<T, 4>(
        sendbuff, recvbuff, count, comm, stream);
  case 2:
    return ncclAllReduceDdaIpcLaunch<T, 2>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    WARN("DDA IPC allreduce: unsupported nRanks %d", comm->nRanks);
    return ncclInvalidUsage;
  }
}

} // namespace

bool ncclAllReduceDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
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
  if (!ddaNranksSupported(comm->nRanks)) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }
  size_t need = count * 4;
  if (datatype == ncclFloat16 || datatype == ncclBfloat16) {
    need = count * 2;
  }
  if (need > comm->ddaIpcScratchBytes) {
    return false;
  }
  if ((count *  ncclTypeSize(datatype)) % 16) {
    // 16 byte alignment as we do 16-byte loads in DDA kernel
    return false;
  }
  if ((count *  ncclTypeSize(datatype)) > kDdaFlatTreeThresholdBytes) {
    if (count % comm->nRanks || ((count / comm->nRanks * ncclTypeSize(datatype)) % 16)) {
      // In two-shot algo, each rank is reduces count/nRanks_ elements so we
      // need to make sure that is 16-byte aligned
      return false;
    }	
  }

  return true;
}

ncclResult_t ncclAllReduceDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
