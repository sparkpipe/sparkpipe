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
| KD2 | As KD1 with `kv_snapshot_directory` removed from every node of the lane's `deployment.json`, restored after | did not start: the KV binding requires the snapshot store (`k31_stage kv binding refused: the deployment names no kv_snapshot_directory`) |
| KD3 | As KD1 with PR #1949 (a prefix reaches disk only after another request reads it) applied to the release tree (sha16 `3fa8436b11e80954`) | 520.1 s, no wave with a late rank (the 4,096-token prompt 8.0 s, KD1 15.1 s) |

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
  16,384, every 8 waves). At each checkpoint the release writes the chain the
  checkpoint closes to the KV snapshot store in one burst: on spark4, 256 KV
  pages and the recurrent state, 64 MB in 257 files within 0.13-0.47 s, 16
  times in the 262,144-token prompt. Over KD1 spark4's store grew by 1.09 GB
  in 4,180 files (`used_bytes=2144526336 files=8294` at open,
  `used_bytes=3235520512 files=12474` at close; both totals count earlier
  runs' files).
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
- **PR #1949 removes the stalls.** In KD3 no wave had a late rank
  (`stall_map-KD3.txt`), and the 262,144 tokens took 520.1 s against 611.3 s
  in KD1 and 592.1 s in KP2. The median wave barely moved (4.06 s against
  4.13 s); the tail went: the 90th percentile fell from 6.46 s to 4.19 s, the
  longest wave from 9.55 s to 4.37 s, and the time the waves spent above the
  median from 88.3 s to 5.7 s. The checkpoint waves and the waves after them
  took a median 6.07 s in KD1 against 4.11 s for the rest; in KD3 4.10 s
  against 4.05 s. What lateness is left is small and off the checkpoints:
  spark0 was 100-300 ms late in 8 waves, none a checkpoint wave or the one
  after it, and sparkc in 2.
- **It changes when the store is written, not how much.** KD3 wrote the same
  1.09 GB in 4,180 files to spark4's store during the prompt as KD1: under
  #1949 the cold prompt's pages still count as read (why is not traced
  here). But they leave about 32 a wave as the prompt runs, so at a
  checkpoint only the recurrent state is left to write, one 28 MB file. That
  file stalled no wave; 257 files within half a second made Gigabyte ranks
  late in KD1.
- **INFERENCE: the burst of writes slows the NIC on those boards.** SMMU0
  translates all four CX-7 functions and the root NVMe (and the Ceph USB
  disk) through one command queue (`docs/CX7_DMA_INVESTIGATION_2026-10.md`
  section 2.3); 257 NVMe writes within half a second queue map, unmap and
  invalidation commands ahead of the NIC's translations. KD3 fits: one large
  write at the checkpoint did not stall a wave, a burst of many did. The
  snapshot store cannot be turned off to test this directly (run KD2).
- **One run each way.** KD1 and KD3 ran 40 minutes apart on the same lane,
  release, prompts and nodes, with the same telemetry (`telemetry-KD1.txt`,
  `telemetry-KD3.txt`), and the stalls of KD1 and of KP2 the day before sit
  on the checkpoint waves, the waves #1949 changes. A rerun of KD1's build
  after KD3 would rule out a fleet change between the runs; the fleet went
  to the weightd ABI 17 release first, so that rerun needs both builds on
  the new release's tree.

## Files

- `stall_map-KD1.txt`, `stall_map-KD3.txt`: the late ranks wave by wave,
  with each node's vendor.
- `late_rank-KD1.txt`, `late_rank-KD3.txt`: per wave, the latest rank, the
  rank that waited least, and the wave's service time.
- `telemetry-KD1.txt`, `telemetry-KD3.txt`: GPU throttle counters, clocks,
  temperatures, memory, load and ping per node over each run's long prompt.
- `stall_map.py`, `late_rank.py`, `telemetry_window.py`: the analyses (run on the hub).
