# GLM 5.3 Flash TP16 B1 roofline

The target is 80% of the memory roofline for one decode token per rank.

## Bytes per rank per token

Every rank streams its share of the weights once per token. The shapes come
from `spark_glm5_next_model.h` and the stage-pack sharding rules.

| Component | Per layer (MB) | Layers | Total (MB) |
| --- | ---: | ---: | ---: |
| KDA attention, BF16 | 19.17 | 34 | 652 |
| DSA/MLA attention and indexer, BF16 | 43.25 | 11 | 476 |
| HC mixing weights, f32, two sites | 3.15 | 45 | 142 |
| Router, shared expert and 8 routed FP8 experts | 18.49 | 42 | 776 |
| Dense MLP, BF16 | 18.87 | 3 | 57 |
| LM head shard | 79.3 | 1 | 79 |
| **Total** | | | **2181** |

At 273 GB/s this is 7.99 ms per token (125 tok/s). The 80% target is
10.0 ms per token (100 tok/s), and it includes collectives. 661 MB (30%) of
the per-rank bytes are replicated on all sixteen ranks: Q_A, KV_A, INDEX_Q,
INDEX_K, INDEX_COMPRESS_GATE, INDEX_HEAD, ROUTER, KDA_DECAY_GATE_DOWN and HC_FN.

## Where the time went

The CUPTI trace in [TP16_HARDWARE_PROFILE_20260922.md](TP16_HARDWARE_PROFILE_20260922.md)
records 54.6 ms of compute kernels per token. The largest items were:

- 14.0 ms of FP8 expert GEMM. 545 MB ran at 39 GB/s.
- 12.8 ms of BF16 GEMM. 1415 MB ran at 111 GB/s.
- 12.1 ms of HC mix. The three-block fix later cut this to about 4.4 ms.
- 4.3 ms of HC Sinkhorn.

Mesh kernels took 1.4 ms. Peer wait added 27 ms per token.

Three structural causes explain these numbers.

1. **Tensor-core GEMM at one row.** At one row the GEMM tiles N by 128 and does
   not split K. A projection therefore occupies `ceil(N/128)` SMs, each running
   a two-stage pipeline:

   | Projection | SMs used (of 48) |
   | --- | ---: |
   | KDA decay-gate-down (256 rows) | 2 |
   | Router | 3 |
   | Index K | 1 |
   | Routed W1 | 16 |

2. **One CTA or one thread per row.** HC Sinkhorn runs on a single thread and
   keeps its 4×4 matrix in local memory, because `hc` is a runtime value. HC
   pre-reduce and HC post each run as one CTA per row.
3. **Replicated weights.** They add 30% to every rank's bytes (see above).

## Changes in this branch

| Change | Kernels replaced | Expected per-token effect |
| --- | --- | --- |
| `inference/kernels/skinny.cuh` handles rows ≤ 4: one warp (or a 16- or 8-lane group) per output neuron, eight 16-byte weight loads in flight per lane, every SM busy. It serves all BF16 projections (shared `LaunchBf16Linear` for glm5_next, glm52, ling and laguna), the router, and the routed FP8 W1/W2 through the route map. | tensor-core GEMM for decode rows | BF16 12.8 → about 6 ms; FP8 14 → about 2.5 ms |
| `Glm5NextHcSiteKernel` fuses HC mix, Sinkhorn and pre-reduce into one eight-CTA cluster launch. K is split across the cluster, reduced in fixed order through distributed shared memory, and Sinkhorn runs in registers. | 3 launches per site, 90 sites | HC 8.7 → about 1 ms |
| HC post spreads each row over `HIDDEN / 256` CTAs. | HC post | about 1 ms |

The expected compute total is about 30 ms. Combined with unchanged waits, this
predicts roughly 40–50 ms per token on a dedicated fleet (about 20–25 tok/s),
up from about 78 ms. This is a prediction for the fleet measurement to test,
not a measured result.

## Numerics

These kernels reassociate floating-point sums:

- The skinny GEMV accumulates in f32 across lanes. The tensor-core path
  accumulated per MMA fragment.
- The FP8 skinny path multiplies the f32 block scale after the dot product.
  The tensor-core path rounded weight × scale to BF16 before the MMA, so the
  skinny result is closer to the exact value.
- HC mixes sum eight cluster partials in fixed order.

Results are deterministic run to run but not bitwise equal to the previous
build. Sinkhorn, pre-reduce and HC post are bitwise unchanged for identical
inputs. The GPU tests check against f64 references.

## Next steps, ordered by expected gain

1. Get a per-kernel CUPTI table for this build to rank the remaining
   11–15 ms of small kernels:
   - RMSNorm at one CTA per row;
   - KDA split, conv, sigmoid and copy launches;
   - the delta-rule grid of 4 CTAs;
   - the indexer.
2. Measure one 16-rank 8 KB all-reduce in isolation. There are at least 90
   collectives per token, so their latency floor decides whether 10 ms is
   reachable at TP16 at all.
3. Fuse the per-layer elementwise chains into the adjacent GEMV prologues and
   epilogues: norm into the input projection, split and conv into the
   `qkv_beta` epilogue, gate into the output projection.
4. Consider sharding the replicated projections behind one extra all-gather
   each. This trades up to 661 MB per rank for more collectives, and only pays
   if the collective latency from step 2 is small.
