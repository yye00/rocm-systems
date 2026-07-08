/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path for the QuickReduce lossy block-quantized two-shot AllReduce DDA
 * collective.  Gated by RCCL_QUICKREDUCE_ENABLE (default 0);
 * see dda_quick_reduce_ipc.cu.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_QUICK_REDUCE_IPC_H_
#define DDA_QUICK_REDUCE_IPC_H_

#include "nccl.h"

struct ncclComm;

/** True when the QuickReduce feature is enabled (RCCL_QUICKREDUCE_ENABLE=1). */
bool ncclQuickReduceEnabled();

/** Effective compression ratio (>=1) of the selected codec, for the cost
 *  model.  Returns 1 when the feature is disabled. */
float ncclQuickReduceCompressionRatio();

/** Eligibility for the QuickReduce lossy all-reduce fast path.  Always false
 *  unless RCCL_QUICKREDUCE_ENABLE=1, so baseline behavior is unperturbed. */
bool ncclAllReduceQuickReduceIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op);

/** Execute the QuickReduce lossy all-reduce using IPC scratch buffers. */
ncclResult_t ncclAllReduceQuickReduceIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

#endif
