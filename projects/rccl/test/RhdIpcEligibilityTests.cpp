/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Unit tests for the log-round recursive-halving/doubling DDA collectives
 * (T4, RCCL_RHD_ENABLE).  Covers:
 *   - the RCCL_RHD_ENABLE gate default (0) and its effect on eligibility,
 *   - the reduce-scatter / all-reduce eligibility guard branches, and
 *   - the invalid-datatype dispatch fall-through.
 *
 * The eligibility gate reads the cached RCCL_RHD_ENABLE param, so tests that
 * flip the gate run in isolated child processes (RUN_ISOLATED_TEST_WITH_ENV)
 * to get a fresh param cache per value.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "dda_recursive_halving_ipc.h"
#include "gtest/gtest.h"
#include "ipc_init_detail.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

// The RCCL-local algorithm id must sit past the NCCL enum and before the count
// sentinel, so it never perturbs the upstream NCCL_ALGO_* numbering.
TEST(RhdAlgoId, RecursiveHalvingIsRcclLocalAndDistinct)
{
    EXPECT_GE(static_cast<int>(RCCL_RECURSIVE_HALVING), NCCL_NUM_ALGORITHMS);
    EXPECT_LT(static_cast<int>(RCCL_RECURSIVE_HALVING),
              static_cast<int>(RCCL_ALGO_COUNT));
    EXPECT_NE(static_cast<int>(RCCL_RECURSIVE_HALVING),
              static_cast<int>(RCCL_DIRECT_ALLGATHER));
    EXPECT_NE(static_cast<int>(RCCL_RECURSIVE_HALVING),
              static_cast<int>(RCCL_SYMMETRIC));
}

// AC: RCCL_RHD_ENABLE defaults to 0 (feature off unless explicitly enabled).
TEST(RhdGate, DefaultDisabled)
{
    RUN_ISOLATED_TEST(
        "RhdGate_DefaultDisabled",
        []()
        {
            unsetenv("RCCL_RHD_ENABLE");
            EXPECT_FALSE(ncclRhdEnabled());
        });
}

TEST(RhdGate, EnabledWhenSetToOne)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdGate_EnabledWhenSetToOne",
        []() { EXPECT_TRUE(ncclRhdEnabled()); },
        {{"RCCL_RHD_ENABLE", "1"}});
}

TEST(RhdGate, DisabledWhenSetToZero)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdGate_DisabledWhenSetToZero",
        []() { EXPECT_FALSE(ncclRhdEnabled()); },
        {{"RCCL_RHD_ENABLE", "0"}});
}

// When the gate is off, eligibility must be false regardless of an otherwise
// valid comm/args — this is what makes the collectives.cc dispatch a no-op
// relative to baseline.
TEST(RhdEligibilityGated, ReduceScatterFalseWhenDisabled)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdEligibilityGated_RS_Disabled",
        []()
        {
            DdaIpcMockComm mock;
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));
        },
        {{"RCCL_RHD_ENABLE", "0"}});
}

TEST(RhdEligibilityGated, AllReduceFalseWhenDisabled)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdEligibilityGated_AR_Disabled",
        []()
        {
            DdaIpcMockComm mock;
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);
            // count must be a multiple of nRanks (8) for AllReduce eligibility.
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 32, ncclFloat32, ncclSum));
        },
        {{"RCCL_RHD_ENABLE", "0"}});
}

// With the gate enabled, walk the reduce-scatter eligibility guard branches.
TEST(RhdEligibilityEnabled, ReduceScatterGuards)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdEligibilityEnabled_RS_Guards",
        []()
        {
            DdaIpcMockComm mock;
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);

            // Happy path: 4 float32 per rank = 16 bytes/segment (16B aligned).
            EXPECT_TRUE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));

            // Null comm.
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                nullptr, sendbuff, recvbuff, 4, ncclFloat32, ncclSum));

            // Missing IPC resources.
            mock.setIpcResourcesPresent(false);
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));
            mock.setIpcResourcesPresent(true);

            // Zero count.
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 0, ncclFloat32, ncclSum));

            // Multi-node.
            mock.comm.nNodes = 2;
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));
            mock.comm.nNodes = 1;

            // Wrong rank count.
            mock.comm.nRanks = 4;
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));
            mock.comm.nRanks = nccl_dda_ipc_detail::kDdaNranks;

            // Unsupported op.
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclMax));

            // Unsupported datatype.
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclInt32, ncclSum));

            // Per-segment not 16-byte aligned (2 float32 = 8 bytes).
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 2, ncclFloat32, ncclSum));

            // Scratch too small.
            mock.comm.ddaIpcScratchBytes = 8;
            EXPECT_FALSE(ncclReduceScatterRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 4, ncclFloat32, ncclSum));
        },
        {{"RCCL_RHD_ENABLE", "1"}});
}

// With the gate enabled, walk the all-reduce eligibility guard branches.
TEST(RhdEligibilityEnabled, AllReduceGuards)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RhdEligibilityEnabled_AR_Guards",
        []()
        {
            DdaIpcMockComm mock;
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);

            // Happy path: count=32 (multiple of 8), per-rank 4 float32 = 16B.
            EXPECT_TRUE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 32, ncclFloat32, ncclSum));

            // Zero count.
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 0, ncclFloat32, ncclSum));

            // Unsupported op.
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 32, ncclFloat32, ncclMax));

            // count not a multiple of nRanks (33 % 8 != 0).
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 33, ncclFloat32, ncclSum));

            // Per-rank segment not 16-byte aligned: count=16 -> 2 float32/rank
            // = 8 bytes.
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 16, ncclFloat32, ncclSum));

            // Scratch too small.
            mock.comm.ddaIpcScratchBytes = 8;
            EXPECT_FALSE(ncclAllReduceRhdIpcEligible(
                mock.get(), sendbuff, recvbuff, 32, ncclFloat32, ncclSum));
        },
        {{"RCCL_RHD_ENABLE", "1"}});
}

// Invalid datatype must fall through the dispatch switch to ncclInvalidArgument
// (independent of the gate — exercises the default case directly).
TEST(RhdDispatch, ReduceScatterInvalidDatatype)
{
    DdaIpcMockComm mock;
    void*          sendbuff = reinterpret_cast<void*>(0x10);
    void*          recvbuff = reinterpret_cast<void*>(0x20);
    EXPECT_EQ(ncclReduceScatterRhdIpc(
                  sendbuff, recvbuff, 4, ncclInt32, ncclSum, mock.get(), nullptr),
              ncclInvalidArgument);
}

TEST(RhdDispatch, AllReduceInvalidDatatype)
{
    DdaIpcMockComm mock;
    void*          sendbuff = reinterpret_cast<void*>(0x10);
    void*          recvbuff = reinterpret_cast<void*>(0x20);
    EXPECT_EQ(ncclAllReduceRhdIpc(
                  sendbuff, recvbuff, 32, ncclInt32, ncclSum, mock.get(), nullptr),
              ncclInvalidArgument);
}

} // namespace RcclUnitTesting
