# Kimi K3.1 Pro TP16 prefill wave stalls, 2026-11-19

Question: which rank is late when 2,048-row TP16 prefill waves stall? On
2026-11-18 (run KP2, `../prefill-20261118/`) 58 of 129 waves of a cold
262,144-token prompt lost 0.5-2.3 s (median 0.98 s) waiting on peers.

## Runs

All 16 Sparks, mesh lane 2 of `k31_pro_rp`, release `a242ef38e` with the
device library of #1945, the hub fleet hold taken for each run, MemAvailable
checked, the release adapter put back after. KP2's prompts: 4,096 tokens at
corpus offset 15,031, then 262,144 at 170,000.

| Run | Build and lane | 262,144-token prefill |
| --- | --- | --- |
| KD1 | The release adapter rebuilt from the release tree with one change: the stage runner prints its arrival and timing lines (`STAGE-STEP-ARRIVAL`, `STAGE-STEP-TIMING`) every step instead of every 64 (sha16 `ce2386bf097162c9`) | 611.3 s (KP2 592.1 s) |
| KD2 | As KD1 with `kv_snapshot_directory` removed from every node of the lane's `deployment.json` (the KV snapshot store off), restored after | KD2_RESULT |

## Method

In every collective round each rank's arrival kernel records when each
peer's data lands, measured from its own wait start
(`model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh`), and
the stage runner logs the sums per step. For each wave, a rank is late by the
median, over the other 15 ranks, of how long they waited for its data, minus
that of the median rank (`stall_map.py`, `late_rank.py`). The hub's fleet
telemetry (`~/fleet-telemetry`: GPU clocks and throttle counters at 1 Hz,
CPU and memory at 1 Hz, ping at 1 Hz, dmesg) covers the same window.

## Findings

- **No single late rank.** In KD1, 41 of 129 waves had a rank whose data
  reached its peers more than 300 ms after the median rank's. Eight ranks
  were late in such waves: 4 (24 waves), 6 (20), 10 (16), 11 (13), 15 (12),
  0 (11), 5 (10), 2 (9). Ranks 7, 8, 9 and 12 never were; 1, 3, 13 and 14 at
  most twice.
- **The stalls follow the 16,384-token KV checkpoint.** 31 of the 41 waves
  are the checkpoint wave or the one after it (`SPARK_K3_KV_CHECKPOINT_TOKENS`
  16,384, every 8 waves): there the batch engine keeps a resume state, and
  each rank saves its recurrent state and KV pages through the KV snapshot
  store. Each rank wrote 3.2 GB in 12,474 files to its snapshot store during
  KD1 (`kv snapshot store close ... used_bytes=3235520512 files=12474` on
  spark4).
- **The late nodes are Gigabyte boards.** Every rank late in more than two
  waves is a Gigabyte AI TOP ATOM (spark0, 2, 4, 5, 6, a, b, f). The Lenovo
  (spark7, 8, 9), MSI (spark1) and Dell (sparkc) nodes never were; three
  Gigabytes (spark3, d, e) rarely. By chance all five non-Gigabyte nodes
  would fall in the calm half about 1.3% of the time. Kernels do not split
  the same way: spark1 and sparkc run the same 6.17 kernel as most
  Gigabytes.
- **Not the GPU, the host or memory.** Over KD1 no node spent any time power
  capped or thermally slowed, SM clocks held 2,171-2,190 MHz on every node
  (the coolest nodes were the late ones), every node answered every 1 Hz ping
  within 1.8 ms, no node logged a kernel message during the prefill, and
  free memory (7-8.4 GiB) does not track lateness (correlation -0.16).
- **INFERENCE: the snapshot writes slow the NIC on those boards.** SMMU0
  translates all four CX-7 functions and the root NVMe (and the Ceph USB
  disk) through one command queue (`docs/CX7_DMA_INVESTIGATION_2026-10.md`
  section 2.3); a burst of NVMe writes at a checkpoint queues map, unmap and
  invalidation commands ahead of the NIC's translations. Run KD2 tests this
  by turning the snapshot store off.
- **The fix in flight.** PR #1949 writes a prefix to disk only once another
  request reads it, so a cold prefill makes none of these writes.

## Files

- `stall_map-KD1.txt`, `stall_map-KD2.txt`: the late ranks wave by wave,
  with each node's vendor.
- `late_rank-KD1.txt`, `late_rank-KD2.txt`: per wave, the latest rank, the
  rank that waited least, and the wave's service time.
- `telemetry-KD1.txt`: GPU throttle counters, clocks, temperatures, memory,
  load and ping per node over KD1.
- `stall_map.py`, `late_rank.py`, `telemetry_window.py`: the analyses (run on the hub).
