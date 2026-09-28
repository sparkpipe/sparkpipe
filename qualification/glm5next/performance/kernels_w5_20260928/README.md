# GLM-5.3 Flash B1 kernels, lane kernels-w5 (2026-09-28)

Single GB10 (sparke), `glm5_next_batch_roofline --context 1024 --iterations 100 --graph 1 --route-readback 0`, TP16 rank shapes, no collectives.
Every number comes from a perf window (`tools/perf_window.py kernels-w5`). Variants are interleaved inside each round.

## Result (ab/ab_prhead.txt, 7 rounds, median of per-run medians)

| rows | base 2ccae97 ms | PR head 90c1357 ms | change |
|---|---:|---:|---:|
| 1 | 15.216 (min 15.008) | 12.934 (min 12.809) | -2.28 ms, -15.0% |
| 8 | 32.029 (min 31.297) | 31.887 (min 31.203) | -0.14 ms |

ab/ab_final2.txt (9 rounds, same code but the older shuffle route scan): B1 15.442 -> 13.025, B8 32.162 -> 31.921.
Some rounds are 2-10 ms slower for both variants. A second residentd (ling lane, rank 14, `sp-ling-rd10`) ran on sparke during those windows (ab/ab_prhead_gpu_apps.txt).

End to end the step is bit-identical: `ROOFLINE-HASH` (FNV of the hidden streams after the run, plus the first token) is equal for base and every variant at rows 1 and 8, with and without graphs.

## Per commit (ab/ab_commits.txt, 5 rounds, B1 median ms)

| step | B1 ms |
|---|---:|
| base | 15.442 |
| + head staging, delta columns, route scan (hdr) | 14.210 |
| + attention 16-position groups (c5) | 13.670 |
| + RMSNorm register staging (c6) | 13.444 |
| + HC prefetch/dependent launch, conv fusion, rows-tile prefetch (final) | 12.93-13.03 (ab_final2, ab_prhead) |

Head alone (ab/ab_head.txt): 15.339 -> 14.425.
The `hdra`, `pdl` and `rms` columns in ab/ab_hc.txt are the `hdr` binary (the bench did not depend on inference/kernels; fixed by the Makefile commit), so ignore them.
Dropped after measurement: programmatic dependent launch plus L2 prefetch for every skinny GEMV (c7 vs c6: +0.04 ms, ab_commits; c11 vs c10: no change, ab_c12) and 8 delta-rule columns instead of 16 (c12 vs c11: no change).

## Multi-projection skinny launch (ab/ab_multi.txt, 7 rounds, branch perf/glm-kw5-multi)
KDA qkv_beta and decay_gate_down in one launch: B1 13.017 -> 12.922 ms (stdev 0.064 / 0.040), B8 32.067 -> 31.958. Base in that window 15.842 / 32.250.

## Bandwidth and per kernel
- stream_read_probe_sparke.txt: streaming read peaks at 243 GB/s (4-64 MB per launch), 237 GB/s at 1 GB; copy 222 GB/s read+write; an empty graph node costs 0.41 us; a 256 KB read costs 2.0 us, 1 MB 5.1 us.
- kernel_table_b1.txt: base vs final per kernel class (nsys graph-node trace, median replay). The BF16 skinny GEMVs were already at 190-240 GB/s on an idle GPU. The large losses were the B1 head (74 GB/s), the split latent attention (14 GB/s), the delta rule, route build and RMSNorm.
- The bytes floor of this step is 2.56 GB / 243 GB/s = 10.5 ms. The final step is 12.9 ms.

## Tests (tests/)
All new tests pass on sparke. They are bitwise against the previous kernel or an exact integer reference. `test_skinny_gemv` (line 249 every run, line 211 intermittently: grouped NPG 1 vs 4 differ) and `test_glm5_next_index_cp` line 107 fail on base 2ccae97 too.
