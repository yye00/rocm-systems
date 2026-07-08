/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaQuantAllToAllIpc, the lossy (fp8, per-token
 * scale) MoE AllToAll dispatch/combine kernel.  Reuses the DDA IPC scratch
 * buffers, peer-pointer table and IpcGpuBarrier set up by ipc_init.cu.
 * Gated by RCCL_QUANT_ALLTOALL_ENABLE (default 0).
 *
 * AllToAll is a pure permute/shuffle with NO reduction, so this path does not
 * touch the RS->reduce->AG or ncclSum machinery: correctness depends only on
 * the round-trip quantize/dequantize error being bounded.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef QUANT_ALLTOALL_IPC_H_
#define QUANT_ALLTOALL_IPC_H_

#include "nccl.h"

struct ncclComm;

/** True when the quantized AllToAll feature is enabled (RCCL_QUANT_ALLTOALL_ENABLE=1). */
bool ncclQuantAllToAllEnabled();

/**
 * Check if the quantized DDA alltoall is eligible for the given parameters.
 * Returns false unless RCCL_QUANT_ALLTOALL_ENABLE=1.
 */
bool ncclQuantAllToAllDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype);

/**
 * Execute the quantized (lossy fp8) DDA alltoall operation using IPC.
 */
ncclResult_t ncclQuantAllToAllDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream);

#endif // QUANT_ALLTOALL_IPC_H_
