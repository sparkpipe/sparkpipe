# GLM-5.3 Flash B1, lane glmflash-b1 (2026-09-29)

Single GB10 (sparkf), `glm5_next_batch_roofline --context 1024 --iterations 100 --graph 1 --route-readback 0`, TP16 rank shapes.
Every number comes from a perf window (`tools/perf_window.py glmflash-b1`). Variants are interleaved inside each round (`scripts/ab2.sh`), 4 rounds, median of per-run medians.
sparkf also ran its production GLM rank (idle between requests) and a second residentd during the windows (`ab/ab_pf3_gpu_before.txt`).

## Weight prefetch into L2 during the collective wait

The bench has no collectives. `--round-spin-us N` puts a one-CTA kernel that spins N microseconds at each of the 91 all-reduce points (embedding, after attention, after the MLP), where production waits for its peers. `--l2-prefetch 1` adds the change: before the round the main stream records an event, and after the round a side stream loads the weights the next kernels read (up to 12 MiB) into L2, then the main stream waits for it.

`ab/ab_pf3_touch_after_round.txt` (the committed code):

| B | spin us | off ms | on ms | change |
|---|---:|---:|---:|---:|
| 1 | 0 | 13.224 | 14.470 | +1.25 |
| 1 | 40 | 16.849 | 14.758 | -2.09 (-12.4%) |
| 1 | 50 | 17.846 | 15.159 | -2.69 (-15.1%) |
| 8 | 40 | 36.188 | 36.095 | -0.09 |
| 8 | 50 | 37.522 | 35.689 | -1.83 |

`ROOFLINE-HASH` is identical for every variant at B1 and B8: the change only moves reads, the arithmetic is unchanged.
Without waits the side kernel costs time (row 1), so the module enables it only when `tp_degree > 1`. Production B1 rounds average about 51 us (4.7 ms over 92 rounds).

Roofline, single-GPU bench step B1 (2.56 GB/step, 243 GB/s peak, ceiling 95 tok/s): off spin 40 memory 63% | on spin 40 memory 71% | no spin, no prefetch memory 80%. Compute about 1-2%. Transport is emulated by the spin (no wire bytes), so no transport percentage.

What did not work, kept for the record:
- `ab/ab_pf1_bulk_hint.txt`: `cp.async.bulk.prefetch.L2` hints on the main stream. B1 spin 40: 16.466 -> 16.848 ms. `ab/l2probe_sparkf.txt` shows why: after a hint and a 60 us idle wait, an 8 MiB read still takes 27.5-30.6 us against 35.6 cold, while data brought in by real loads reads in 8.2 us (1.0 TB/s from L2). The hints are mostly dropped.
- `ab/ab_pf2_touch_before_round.txt`: the load kernel forked before the round. The graph started the 48-CTA load kernel first and the spin kernel only started as its CTAs retired, so nothing overlapped (B1 spin 40: 16.439 -> 16.806). Enqueueing the load after the round (fork event before the round, launch after) lets the round start first; nsys then shows the round and the load running together.

## Per-kernel baseline at main 590d3de (nsys, fastest of 30 replays)
Kernel busy 12.18 ms, span 12.34 ms. BF16 skinny GEMVs run at 190-240 GB/s on an idle GB10 (same as kernels-w5). The remaining single-GPU gaps above the byte floor (10.5 ms) are HC site 1.25 ms (0.58 ms of bytes), top-k warp 0.26 ms, RMSNorm 0.25 ms (113 launches), delta rule 0.24 ms, split latent attention 0.30 ms.
