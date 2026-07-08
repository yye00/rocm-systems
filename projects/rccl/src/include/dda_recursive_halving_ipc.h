/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path for the log-round recursive-halving/doubling DDA collectives.
 * Gated by RCCL_RHD_ENABLE (default 0); see dda_recursive_halving_ipc.cu.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_RECURSIVE_HALVING_IPC_H_
#define DDA_RECURSIVE_HALVING_IPC_H_

#include "nccl.h"

struct ncclComm;

/** True when the recursive-halving feature is enabled (RCCL_RHD_ENABLE=1). */
bool ncclRhdEnabled();

/** Eligibility for the recursive-halving reduce-scatter fast path. */
bool ncclReduceScatterRhdIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op);

/** Execute recursive-halving reduce-scatter using IPC scratch buffers. */
ncclResult_t ncclReduceScatterRhdIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

/** Eligibility for the recursive-halving all-reduce fast path. */
bool ncclAllReduceRhdIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op);

/** Execute recursive-halving all-reduce (RS + reversed AG) using IPC. */
ncclResult_t ncclAllReduceRhdIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

#endif
