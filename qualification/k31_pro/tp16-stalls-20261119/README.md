# Kimi K3.1 Pro TP16 prefill wave stalls, 2026-11-19 and 20

Question: which rank is late when 2,048-row TP16 prefill waves stall? On
2026-11-18 (run KP2, `../prefill-20261118/`) 58 of 129 waves of a cold
262,144-token prompt lost 0.5-2.3 s (median 0.98 s) waiting on peers.

## Runs

All 16 Sparks, mesh lane 2 of `k31_pro_rp`, the hub fleet hold taken for
each run, MemAvailable checked, the release adapter put back after. KP2's
prompts: 4,096 tokens at corpus offset 15,031, then 262,144 at 170,000.
KD1-KD3 ran on release `a242ef38e` with the device library of #1945. KD4
and KD5 ran back to back on 2026-11-20 (03:27-03:48Z) on release
`c2b714343` (weightd IPC ABI 17 with four mesh slots a rank, page cache ABI
16 with #1949), with the swap controller idle and no model active. Before
them the test lane's residentd, `weightd_warm`, transport, device library and
adapter, and the hub API's binaries, adapter and device library, were copied
from the release's own K3.1 lane and API.

| Run | Build and lane | 262,144-token prefill |
| --- | --- | --- |
| KD1 | The release adapter rebuilt from the release tree with one change: the stage runner prints its arrival and timing lines (`STAGE-STEP-ARRIVAL`, `STAGE-STEP-TIMING`) every step instead of every 64 (sha16 `ce2386bf097162c9`) | 611.3 s (KP2 592.1 s) |
| KD2 | As KD1 with `kv_snapshot_directory` removed from every node of the lane's `deployment.json`, restored after | did not start: the KV binding requires the snapshot store (`k31_stage kv binding refused: the deployment names no kv_snapshot_directory`) |
| KD3 | As KD1 with PR #1949 (a prefix reaches disk only after another request reads it) applied to the release tree (sha16 `3fa8436b11e80954`) | 520.1 s, no wave with a late rank (the 4,096-token prompt 8.0 s, KD1 15.1 s) |
| KD4 | Release `c2b714343`'s tree, which contains #1949, with the same print change (sha16 `f274c8236630e9eb`) | 558.0 s, 41 waves with a late rank |
| KD5 | As KD4 with #1949 reverted in the five source files it changed, page cache ABI 15 (sha16 `c409c1141ed53a16`) | 565.6 s, 45 waves with a late rank |

Where the time went: each wave's service time above the run's median wave,
summed over the checkpoint waves with the waves after them (16 + 16) and
over the other 96; "calm" waves are those where no rank was over 100 ms
late in the wave or the one before.

| Run | Release | #1949 | 262,144 tokens | Waves with a rank over 300 ms late | Above the median, checkpoint waves and the next | Above the median, other waves | Calm waves |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD1 | `a242ef38e` | no | 611.3 s | 41 | 74.1 s | 14.2 s | 59 |
| KD3 | `a242ef38e` | yes | 520.1 s | 0 | 1.7 s | 4.0 s | 108 |
| KD4 | `c2b714343` | yes | 558.0 s | 41 | 6.1 s | 21.3 s | 35 |
| KD5 | `c2b714343` | no | 565.6 s | 45 | 18.7 s | 18.6 s | 35 |

A calm wave costs the same on both releases: 3.88-3.90 s plus 1.1-1.3 ms
per 1K of context in all four runs.

## Method

In every collective round each rank's arrival kernel records when each
peer's data lands, measured from its own wait start
(`model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh`), and
the stage runner logs the sums per step. For each wave, a rank is late by the
median, over the other 15 ranks, of how long they waited for its data, minus
that of the median rank (`stall_map.py`, `late_rank.py`). The hub's fleet
telemetry (`~/fleet-telemetry`: GPU clocks and throttle counters at 1 Hz,
CPU and memory at 1 Hz, ping at 1 Hz, dmesg) covers the same window. A
wave's wall-clock window is rebuilt from the probe's start and the API's
per-wave service times (`wave_clock.py`), which places each node's
recurrent-state writes (`state_align.py`) and its periodic host jobs from
the system journal (`event_align.py`) in a wave.

## Findings

- **No single late rank.** In KD1, 41 of 129 waves had a rank whose data
  reached its peers more than 300 ms after the median rank's. Eight ranks
  were late in such waves: 4 (24 waves), 6 (20), 10 (16), 11 (13), 15 (12),
  0 (11), 5 (10), 2 (9). In KD4: 0 (12), 15 (12), 12 (7), 11 (6), 2 (5),
  10 (5), 4 and 6 (2). In KD5: 12 (18), 14 (10), 6 and 11 (7), 0 (5),
  3 and 15 (4), 5 (3). The Lenovo nodes (spark7, 8, 9) were never late in
  any run.
- **On `a242ef38e` the stalls follow the 16,384-token KV checkpoint.** In KD1,
  31 of the 41 waves are the checkpoint wave or the one after it
  (`SPARK_K3_KV_CHECKPOINT_TOKENS` 16,384, every 8 waves). At each checkpoint
  the release writes the chain the checkpoint closes to the KV snapshot store
  in one burst: on spark4, 256 KV pages and the recurrent state, 64 MB in 257
  files within 0.13-0.47 s, 16 times in the 262,144-token prompt. Over KD1
  spark4's store grew by 1.09 GB in 4,180 files (`used_bytes=2144526336
  files=8294` at open, `used_bytes=3235520512 files=12474` at close; both
  totals count earlier runs' files). The late ranks were late in the wave
  their own node wrote its burst (`parts-KD1.txt`: rank 4 a median 649 ms,
  ranks 0 and 2 over 200 ms).
- **There the late nodes are Gigabyte boards.** In KD1 every rank late in
  more than two waves is a Gigabyte AI TOP ATOM (spark0, 2, 4, 5, 6, a, b,
  f). The Lenovo (spark7, 8, 9), MSI (spark1) and Dell (sparkc) nodes never
  were; three Gigabytes (spark3, d, e) rarely. By chance all five
  non-Gigabyte nodes would fall in the calm half about 1.3% of the time.
  Kernels do not split the same way: spark1 and sparkc run the same 6.17
  kernel as most Gigabytes. On `c2b714343` the Dell node is late too (KD5's
  latest rank, 18 waves).
- **Not the GPU, the host or memory.** In no run did a node spend any time
  power capped or thermally slowed. SM clocks held 2,171-2,190 MHz on every
  node (the coolest nodes were the late ones), every node answered every
  1 Hz ping within 4.3 ms, and no node logged a kernel message during a
  prefill. Free memory (5.9-8.9 GiB) does not track lateness (KD1
  correlation -0.16).
- **PR #1949 removes the checkpoint stalls on both releases.** The time above
  the median in the checkpoint waves and the waves after them fell from
  74.1 s to 1.7 s on `a242ef38e` (KD1 to KD3) and from 18.7 s to 6.1 s on
  `c2b714343` (KD5 to KD4). With #1949 the checkpoint waves take what the
  others take: KD3 4.10 s against 4.05 s, KD4 4.16 s against 4.22 s; without
  it KD1 6.07 s against 4.11 s and KD5 4.55 s against 4.16 s. KD5's late
  waves bunch at the checkpoint (29 of 45 within one wave of it), KD4's do
  not (13 of 41, where 15 would be chance).
- **It changes when the store is written, not how much.** Every run wrote the
  same 1.09 GB in 4,180 files to spark4's store during the prompt: under
  #1949 the cold prompt's pages still count as read (why is not traced
  here). With #1949 they leave about 32 a wave as the prompt runs, so at a
  checkpoint only the recurrent state is left to write, one 28 MB file;
  without it (KD1, KD5) the 257-file burst comes back at each checkpoint
  (`snapshot_writes-spark4-KD5.txt`). A rank is not late in the wave its own
  node writes its state file (`parts-KD3.txt`, `parts-KD4.txt`).
- **On `c2b714343` a second source costs about 20 s a prompt, with or without
  #1949.** Away from the checkpoints the waves spent 21.3 s (KD4) and 18.6 s
  (KD5) above the median, against 4.0-14.2 s on `a242ef38e`, and only 35 of
  128 waves were calm, against 108 in KD3. In KD4 spark0 and sparkf went late
  together at steps 93, 100, 108, 115-116 and 123, about every 33 s on the
  wall clock, a period that drifts across the 8-wave checkpoint cycle, so it
  is not counted in tokens (`wave_clock-KD4-KD5.txt`). It does not line up
  with any periodic job the nodes' journals show: the 17-second uplink
  monitor (every node but sparkf), the 2-minute Ceph OSD guard scan and
  spark0's Ceph CLI calls, or `sysstat` (`host_jobs.txt`: each job's hit
  rate equals the node's base rate). The
  late rank's launch, graph and copy times and its weightd wait stay within
  20 ms of its peers', its combine within 40 ms (`parts-KD4.txt`,
  `parts-KD5.txt`): its data lands late, so the node's GPU or NIC is slow in
  that wave. KD3 had a trace of it: spark0 100
  to 300 ms late about every 32 s. The release and the day changed together
  between KD3 and KD4, so these runs cannot say whether the release or the
  fleet brought it.
- **INFERENCE: on `a242ef38e` the burst of writes slowed the NIC on the
  Gigabyte boards.** SMMU0 translates all four CX-7 functions and the root
  NVMe (and the Ceph USB disk) through one command queue
  (`docs/CX7_DMA_INVESTIGATION_2026-10.md` section 2.3); 257 NVMe writes
  within half a second queue map, unmap and invalidation commands ahead of
  the NIC's translations. One large write at the checkpoint did not stall a
  wave, a burst of many did. The snapshot store cannot be turned off to test
  this directly (run KD2).

## Files

- `stall_map-KD{1,3,4,5}.txt`: the late ranks wave by wave, with each node's
  vendor.
- `late_rank-KD{1,3,4,5}.txt`: per wave, the latest rank, the rank that
  waited least, and the wave's service time.
- `telemetry-KD{1,3,4,5}.txt`: GPU throttle counters, clocks, temperatures,
  memory, load and ping per node over each run's long prompt.
- `parts-KD{1,3,4,5}.txt`: the late rank's step parts against its peers', and
  each rank's lateness in the wave its node wrote a recurrent state.
- `host_jobs.txt`: each periodic host job's start times against its node's
  lateness, all four runs.
- `wave_clock-KD4-KD5.txt`: the wall-clock window of every wave with a late
  rank in KD4 and KD5.
- `snapshot-spark4-KD{4,5}.txt`, `snapshot_writes-spark4-KD{4,5}.txt`:
  spark4's snapshot store lines and its file writes in 10-second bins.
- `stall_map.py`, `late_rank.py`, `telemetry_window.py`, `wave_clock.py`,
  `state_align.py`, `event_align.py`, `late_parts.py`: the analyses (run on
  the hub).
