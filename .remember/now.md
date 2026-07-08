
## F001 a105413b baseline BEFORE capture — COMPLETE (2026-07-07)
- librccl.so rebuilt (gfx950, 29MB) via install.sh --amdgpu_targets gfx950; earlier build dir had no .so (OOM-killed rebuild).
- LESSON: never run two install.sh in the same rccl build/ tree — they race on fmt FetchContent and break CMake configure ("Build step for fmt failed"). Kill duplicates, then a single clean build reuses cached device .o files.
- topo_expl -m 59 confirms 8xgfx950 XGMI clique (nranks=8, 56 max channels). topo_expl_impl.cpp fix (zero cpu host_hash) required for gfx950 fixture.
- All 5 sweeps (allreduce half/bf16, reducescatter, allgather, alltoall) captured 8B..2GiB, exit=0, #wrong==0, saved to projects/rccl/perf_results/baseline/.
- Size ceiling 2G (not 8G AC): GPU3 has external vLLM tenant occupying ~293GB; -e 4G/8G OOM. Documented in ENV.txt.
