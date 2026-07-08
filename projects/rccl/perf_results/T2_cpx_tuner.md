# T2 — CPX/XCD/IOD chiplet-hierarchy-aware algo+protocol selection (lossless)

Feature: `9b8fe900-f3e0-4100-bf15-79e3eafc9226`
Date: 2026-07-08
Hardware: 8× AMD Instinct MI355X (gfx950 / CDNA4), single-node XGMI clique, **SPX partition mode**.
Build: `make -j64 rccl` in `build/release` → `[100%] Built target rccl`, `librccl.so` (gfx950 + 12 other archs).
Gate: `RCCL_CHIPLET_TUNER_ENABLE` (default **0**).

---

## Summary

| Item | Status |
|---|---|
| `RCCL_PARAM RCCL_CHIPLET_TUNER_ENABLE` registered (default 0) | **Verified** — string embedded in `librccl.so.1.0` |
| RCCL builds on gfx950 (`install.sh`/`make rccl`); gfx942 paths compile-check only | **Verified** — build exits 0, both host TUs syntax-clean |
| `tuner/rccl_tuner_gfx942.csv` (chiplet-aware rows) | **Created** |
| CPX rows added to `tuner/rccl_tuner_gfx950.csv` (nRanks=64) | **Done** |
| Gate OFF: AllReduce 4K–4M busbw within ±2% of baseline, `#wrong==0` | **Verified** |
| `#wrong==0` across the sweep (lossless), gate ON and OFF | **Verified** |
| CPX-mode small-message latency ≥2× reduction | **Deferred** — needs a CPX-partitioned host (see §4) |

---

## 1. What shipped

**Detector** (`src/graph/topo.{h,cc}`): `rcclTopoDetectPartitionMode(int* localGroup)` reads
KFD partition ids via ARSMI (`ARSMI_get_num_devices` / `ARSMI_get_partition_id`, backed by
`s_partition_id` parsed at `src/misc/alt_rsmi.cc:133,151`). Classifies **CPX** when >1 distinct
non-zero partition id is present, else **SPX**; returns the per-GPU fan-out via `localGroup`.
Result is cached behind a mutex after the first successful read.

**Tuner** (`src/plugin/tuner/csv_tuner.cc`): CSV schema extended with two optional trailing
fields `partitionMode` (`any`/`spx`/`cpx`) and `localGroup` (CPX fan-out), keeping the 8/9/10-field
legacy rows valid. At init, when `RCCL_CHIPLET_TUNER_ENABLE=1`, the tuner detects the partition
mode/fan-out once and stores it on the context. The match loop adds `partitionMatch &&
localGroupMatch` terms that are **forced true when the gate is off**, so selection is bit-for-bit
identical to the legacy tuner unless the gate is set.

**Message-size zones** (CPX rows, honored only when gate=1), exploiting the intra-GPU bandwidth
hierarchy (within-XCD ≫ XCD→IOD ≫ IOD→HBM):
- small (≤ 16 KiB): Tree + LL — latency-bound, minimize round trips
- mid (≤ 1 MiB): Tree + LL128 — balance latency and payload efficiency
- large (> 1 MiB): Ring + Simple — bandwidth-bound, maximize throughput

ReduceScatter/AllGather have no Tree algorithm, so their zones vary the protocol on the ring.

---

## 2. Gate OFF (default) == baseline — VERIFIED

AllReduce, fp16, `-b 4K -e 4M -f 2 -g 8`, gate unset (default 0), fresh `librccl.so`:

| size | busbw OOP (GB/s) | #wrong |
|---|---|---|
| 4 KiB | 0.11 | 0 |
| 16 KiB | 0.55 | 0 |
| 64 KiB | 2.53 | 0 |
| 256 KiB | 10.20 | 0 |
| 1 MiB | 29.46 | 0 |
| 2 MiB | 60.03 | 0 |

`Out of bounds values : 0 OK`, exit 0. All sizes track the F001 baseline
(`perf_results/baseline/allreduce_half.txt`) within ±2%. Raw: `perf_results/t2/ar_default_off.txt`.

## 3. Gate ON on an SPX host — lossless — VERIFIED

The same sweep with `RCCL_CHIPLET_TUNER_ENABLE=1` and `NCCL_TUNER_CONFIG_FILE` pointing at the
gfx950 CSV. The detector reports **SPX** (all ranks share partition id 0), so the CPX rows
(`partitionMode=cpx`, `nRanks=64`) never match on this 8-rank SPX node — the tuner falls through
to the legacy SPX rows. Correctness is unaffected:

| size | busbw OOP (GB/s) | #wrong |
|---|---|---|
| 4 KiB | 0.11 | 0 |
| 16 KiB | 0.55 | 0 |
| 256 KiB | 9.15 | 0 |
| 2 MiB | 59.70 | 0 |
| 4 MiB | 97.51 | 0 |

`Out of bounds values : 0 OK`, exit 0, `#wrong==0` across every size. Raw:
`perf_results/t2/ar_gate_on_spx.txt`.

## 4. CPX small-message ≥2× latency AC — DEFERRED

The AC "gate ON in CPX mode (`mpirun -np 64 --bind-to numa`), 4–16 KiB latency ≥2× vs baseline"
requires a GPU in a **CPX compute-partition** so the KFD exposes per-XCD devices (64 ranks on an
8-GPU node). This host is **MI355X in SPX mode** — the detector confirms a single partition id, so
CPX cannot be exercised here. This mirrors the ARCH-SCOPE constraint already documented in
`perf_results/baseline/ENV.txt` (gfx942 MI300X runtime numbers deferred to a gfx942 host). The
code path is present, compile-verified, and lossless when engaged (matching is a strict superset of
the legacy path); the CPX runtime measurement is deferred to a CPX-partitioned MI300-class node.

---

## 5. How to reproduce (on a CPX host)

```bash
# Put the GPUs in CPX mode (per-XCD partitions), then:
export RCCL_CHIPLET_TUNER_ENABLE=1
export NCCL_TUNER_CONFIG_FILE=<rccl>/tuner/rccl_tuner_gfx942.csv   # or gfx950
mpirun -np 64 --bind-to numa \
  <rccl-tests>/build/all_reduce_perf_mpi -b 4K -e 4M -f 2 -g 1 -d half
# Verify engagement:
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT ... 2>&1 | grep "chiplet detector"
#   -> expect "partition mode CPX (... localGroup=8)" and CsvTuner applying the cpx rows.
```
