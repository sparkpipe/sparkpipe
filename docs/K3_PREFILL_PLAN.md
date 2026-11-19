# Kimi K3.1 Pro 1M-token prefill plan

The goal is a cold 1,048,576-token Kimi K3.1 Pro prompt in about 3 minutes
on the 16 Sparks. The pass engine (`k3_streamed_prefill.cuh`,
`streamed.cuh`, `inference/runner/pass_prefill.cuh`) is the fast path: each
node owns its rows and streams the other ranks' weights. Measured on
2026-11-19 it prefills K3.1 at 1M in 764.9 s. Applying the DeepSeek V4.1
Flash 16-rank ideas (#1931: rows owned, the publisher's MX arithmetic in
native kernels, lean kernel loops, both ports on the wire) to it models
1M at about 225-260 s.

## Where it stands (2026-11-19)

| Item | Value | Source |
| --- | ---: | --- |
| K3.1 1M (998,468 tokens), pass engine | 764.9 s: MLA 461.4, KDA 157.5, MoE 143.5 | run KQ3 |
| K3.1 256K, pass engine | 156.9 s, one pass | run KQ2 |
| K3.1 256K, 2,048-row TP16 waves | 592.1 s | run KP2 |
| K3 1M (990,881 tokens), pass engine, dense MLA | 971.6 s | [PERFORMANCE_STATUS.md](../PERFORMANCE_STATUS.md), 2026-10-31 |
| K3.1 1M with the levers of section 3 | about 225-260 s | model |

Build: release `a242ef38e` (K3.1 adapter sha16 `e430d1f72200b6f7`) with the
capture fix device library of #1945 (sha16 `8b69d85be11a1a29`), weightd
`ee01d0d5665e4464`, all 16 Sparks, mesh lane 2, production deactivated.
Pass engine settings: `node_rows` 32,768 (the engine took 31,232 a node for
this prompt, two passes), 3 dense buffers, 28 expert slots, the expert pool
capped at 8 GiB, partner packs, residentd MemoryMax 80G (the 40.5 GB weight
window and 30.7 GB engine count against the unit; at 16G it is OOM-killed).
The 1M prompt is a pass-key needle (41,600 filler units, key at the middle).
Receipts:
[`qualification/k31_pro/prefill-20261118/`](../qualification/k31_pro/prefill-20261118/README.md).

## 1. The publisher's K3.1 and our stage

The checkpoint (`moonshotai/Kimi-K3.1-Pro`, revision `d9e99b0792a5`) ships
`config.json` and `configuration_kimi_k3.py` but no modeling file; its README
names vLLM, SGLang and TokenSpeed as the engines. The publisher column comes
from the config, the README and vLLM's K3.1 DSA module (day-0 support).

| Item | Publisher | Our stage |
| --- | --- | --- |
| Indexed layers | all 24 MLA layers (`dsa_layers`), own top-k per layer (`index_topk_freq` 1) | the same |
| Index shape | 32 heads x 128, NoPE, key LayerNorm with bias in FP32 | the same (`K3_INDEX_*`, `K3LayerMlaIndex`) |
| Score | `128^-0.5 * 32^-0.5 * sum_h w_h relu(q_h . k)`; q from `wq_b` of the normed q latent, k and w from one BF16 GEMM of the normed hidden | the same formula (`inference/kernels/index_score.cuh`) |
| Index precision | FP8 keys, 132 bytes a token with a ue8m0 scale per 128-wide key; FP8 queries, their scale folded into the head weights; DeepGEMM scoring | BF16 keys (256 bytes a token) and queries, BF16 MMA, FP32 sums |
| Selection | top 2,048 causal positions; a row with fewer attends all of them | the same; ties go to the earlier position |
| Sparse attention | FlashMLA sparse over an FP8 latent cache (`fp8_ds_mla`) | BF16 latent cache (1,152 bytes a position) |
| Routed experts | MXFP4 weights, "MXFP8 activations" in quantization-aware training | MXFP4 weights, BF16 activations (`LmCellMmaKernel`) |
| Attention, shared experts, dense MLP, router, latent projections | BF16 | BF16 |
| KDA | 69 layers, 96 heads x 128, conv 4, full-rank gate, lower bound -5; vLLM runs the chunked kernel | per-token delta rule, FP32 state |

## 2. Measured on the pass engine

| Pass | Keys to | Time | MLA | KDA | MoE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1M, pass 1 of 2 | 499,712 | 324.0 s | 172.4 | 78.1 | 72.0 |
| 1M, pass 2 of 2 | 998,467 | 440.9 s | 289.0 | 79.4 | 71.5 |
| 256K, one pass | 262,142 | 156.9 s | 73.0 | 42.8 | 40.0 |

Weights never held a pass back: each 1M pass streamed them in about 128 s
(7-8 s of dense broadcasts, 120 s of expert steps, 20 s of that reading the
packs) while the streamer waited 195 s and 300 s for the GPU.

Where the 256K pass spends its GPU time (rank 0 CUPTI trace, seconds):
stream GEMMs 29.2, mesh collective kernels about 41 (most of them the MLA
exchanges: query and key gathers, 16 x 2,048 index candidates a row, the
selected indices of every row, the values), experts 23.1, sparse attend 15.5
(10.9 TFLOPS) plus 4.5 of value projection, index score 14.5 (29 TFLOPS,
BF16), index query projections 3.9 in 46,048 launches, index select 2.6, KDA
scan 7.4. The MLA layers attend head-parallel: every node scores and attends
all of the pass's rows for its 6 heads and its 1/16 of the keys, so the index
work and every exchange grow with the pass, and the index with the context.

## 3. Levers on the pass engine, ranked by minutes saved at 1M

| # | Lever | DSV4.1 idea | 1M now | After | Saves |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Indexed MLA with rows owned: each node scores its own rows against replicated index keys (0.27 GB a layer at 1M), picks the top 2,048 inside the scoring kernel, and attends its own rows with all 96 heads sharing each row's 2,048 keys; the candidate, selected-index and value exchanges go away. The index kernel at tensor-core speed, then FP8 keys and queries as the publisher's engine | rows owned with attention; replicated indexer, own-row scoring, radix top-k; fused attention; lean kernels | 461 s | 80-100 s | 6.0-6.4 min |
| 2 | KDA: one chunked tensor-core scan instead of three token-serial passes, projections at GEMM speed | lean kernels | 157.5 s | 55-60 s | 1.6 min |
| 3 | Experts as MXFP8 activations x MXFP4 weights on block-scaled MMA, the BF16 parts at GEMM speed | native MX multiplies | 143.5 s | 75-90 s | 0.9-1.1 min |
| 4 | The weight stream on both 100G ports: about 64 s a pass instead of 128 | both device paths | hidden today | hidden | needed for 1-3 |

Levers 1-3 bring the GPU time of a 1M prompt to about 210-250 s, about 105-125
s a pass, under today's 128 s weight stream: without lever 4 two passes
floor at about 256 s. With it, 1M lands at about 225-260 s. Bits: lever 1 is
exact through its BF16 steps (same scores, same selection, the attention sum
order aside); the FP8 index, the chunked KDA scan and the MX experts move
bits and each needs a measured quality comparison before it is on.

## 4. Serving through the pass engine

- The decode step after a pass aborts today: the pass lane caps the expert
  pool at 8 GiB to make room for its window and engine, and the runner
  refuses a decode step until every expert group is resident
  (`StageRunnerStep` in `inference/runner/stage_runner.cu`). So the pass
  lane prefills and a full-pool lane decodes. The hand-off copies the
  sequence's KV (latent and index keys, about 2.1 GB a rank at 1M) and its
  KDA state (about 27 MB a rank), under a second on the 100G fabric. The
  other way is one lane that frees its pass buffers after the prefill and
  warms the full pool again (about 90 GB a node from the page cache or NVMe).
- Where the pass engine wins: a pass streams about 1.3 TB of expert weights
  a node, 128 s today, whatever the prompt. Waves take about 128 s for 64K
  tokens, so the threshold is about 64K now and about 32K after lever 4. On
  2026-11-07 a 33K prompt took 282 s through 2,048-row passes against about
  66 s in waves.

## 5. The TP16 wave path, below the threshold

Measured on 2026-11-18 (runs KP1 traced, KP2 untraced): a 256K cold prompt
in 592.1 s; a calm 2,048-row wave takes 3.94 s + 1.36 ms per 1K of context,
and 49% of waves stall by +1.0 s waiting on peers (late rank not identified,
no weightd transport errors). In a calm wave at 226K rank 0 computes 2.3 s
and spends 1.8 s in collectives; experts run at 19.5 TFLOPS, the index score
at 29.7, the sparse attend at 13.3, the projections at 41, and AttnRes and
the other row-wise passes take 0.68 s on every rank. Its levers: the stalls,
row-wise work owned per rank, the MLA key gather (#1943), whole experts per
rank, and the 224-wide routed up-projection that takes the slow GEMM path
(61 ms a calm wave).

## 6. Order of work

1. Lever 1 on the pass engine: the exact BF16 index tile kernel and the fused
   top 2,048 first (bit-for-bit tests against `LmWeightedSparseScoreKernel`
   and the current selection), then rows-owned scoring and attention in
   `K3StreamedMlaNode`, measured on a 256K and a 1M prompt.
2. The serving hand-off (section 4), so prompts above the threshold take the
   pass engine.
3. Levers 2, 3 and 4, each measured, each change of bits with its
   comparison; the FP8 index after its comparison.
4. The TP16 stalls, for the prompts below the threshold.
