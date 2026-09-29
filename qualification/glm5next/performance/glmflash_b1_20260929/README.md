# GLM-5.3 Flash B1, lane glmflash-b1 (2026-09-29)

Single GB10 (sparkf), `glm5_next_batch_roofline --context 1024 --iterations 100 --graph 1 --route-readback 0`, TP16 rank shapes.
Every number comes from a perf window (`tools/perf_window.py glmflash-b1`). Variants are interleaved inside each round (`scripts/ab2.sh`), 4 rounds, median of per-run medians.
sparkf also ran its production GLM rank (idle between requests) and a second residentd during the windows (`ab/*gpu_before.txt`).

## Weight loads into L2 during the collective wait (committed design)

The bench has no collectives. `--round-spin-us N --round-wait 2` builds each of the 91 round points like a hardware-wait round:
1. two one-thread kernels, standing in for the request and publish kernels;
2. a wait-value graph node;
3. a host thread that releases the wait N us after the first kernel ran.

`--l2-prefetch 1` enables the change. `SparkGlm5NextL2PrefetchAfterRound` hangs a load kernel (up to 12 MiB, 48x256 `ld.global.cg`) off the node before the wait-value node and joins it into the next node.

`ab/ab_final_graph_node.txt`. In that run the host-released mode was still numbered `--round-wait 3`; the committed bench calls it `2`.

| B | wait/round | off ms | on ms | change |
|---|---:|---:|---:|---:|
| 1 | none | 12.808 | 12.965 | no load kernels placed (no wait node); noise |
| 1 | 40 us | 16.904 | 14.992 | -1.91 (-11.3%) |
| 1 | 50 us | 17.843 | 15.383 | -2.46 (-13.8%) |
| 8 | 40 us | 35.645 | 34.919 | -0.73 |
| 8 | 50 us | 36.631 | 35.211 | -1.42 |

`ROOFLINE-HASH` is identical for every variant at B1 and B8.

Roofline for the single-GPU bench step at B1 (2.56 GB/step, 243 GB/s peak, ceiling 95 tok/s):
- memory: 62% off and 70% on with 40 us waits; 81% with no waits;
- compute: about 1-2%;
- transport: not measured. The wait is emulated and moves no wire bytes.

## What did not work (kept for the record)
- `ab/ab_pf1_bulk_hint.txt`: `cp.async.bulk.prefetch.L2` hints. `ab/l2probe_sparkf.txt` shows that after a hint and a 60 us idle wait, an 8 MiB read still takes 27.5-30.6 us against 35.6 cold. After real loads the same read takes 8.2 us (1.0 TB/s from L2).
- `ab/ab_pf2_touch_before_round.txt`: the load kernel on a side stream forked before a spin-kernel round. The 48-CTA load kernel started first, and the round's kernel only started as its CTAs retired.
- `ab/ab_pf3_touch_after_round.txt`: the same side-stream fork with spin-kernel rounds looked good (B1 16.849 -> 14.758 ms), because a resident spin kernel is not delayed by the load kernel.
- Side stream forked before the round, with the load launched after it in API order: `ab/ab_side_stream_host_wait.txt`, B1 16.794 -> 15.248 ms.
  - An nsys trace with two pre-wait kernels showed the second one starting a median of 23.6 us late (p90 52 us).
  - On the fleet that delay would hit this rank's publish kernel, and every peer would wait for it.
  - The committed design forks after the last pre-wait kernel instead.

## Per-kernel baseline at main 590d3de (nsys, fastest of 30 replays)
- Kernel busy time is 12.18 ms; the span is 12.34 ms.
- BF16 skinny GEMVs run at 190-240 GB/s on an idle GB10.
- Remaining single-GPU gaps above the byte floor (10.5 ms):

| Kernel | Time | Note |
|---|---:|---|
| HC site | 1.25 ms | 0.58 ms of bytes |
| top-k warp | 0.26 ms | |
| RMSNorm | 0.25 ms | 113 launches |
| delta rule | 0.24 ms | |
| split latent attention | 0.30 ms | |
