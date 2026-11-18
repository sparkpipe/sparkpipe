# Kimi K3.1 Pro 1M-token prefill plan

The goal is a cold 1,048,576-token Kimi K3.1 Pro prompt in about 3 minutes
on the 16 Sparks. On 2026-10-31 Kimi K3 took 16.2 minutes on the streamed
pass engine. K3.1 today runs long prompts as 2,048-row TP16 waves, which
extrapolates to about 40-44 minutes for 1M tokens. The DeepSeek V4.1 Flash
16-rank climb (2.3K to 27.4K tok/s at 131K, #1931) is the playbook: rows
owned per rank, the publisher's MX arithmetic in native kernels, lean kernel
loops and overlapped wire. Applied to K3.1, the cost model below puts 1M at
about 3.5-4.1 minutes.

## Where it stands (2026-11-18)

| Item | Value | Source |
| --- | ---: | --- |
| K3 1M (990,881 tokens), pass engine, dense MLA | 971.6 s: MLA attention 694, KDA 137, MLP and experts 139 | [PERFORMANCE_STATUS.md](../PERFORMANCE_STATUS.md), 2026-10-31 |
| K3.1 256K cold prefill, production TP16 path | 592.1 s, 442.7 tok/s | run KP2 |
| K3.1 2,048-row wave without a stall | 3.94 s + 1.36 ms per 1K of context | KP2, 65 waves |
| K3.1 waves with a stall | 49%, +1.03 s each | KP2 |
| K3.1 1M on the production path | 39.6 min, 43.8 min with this run's stalls | extrapolated, section 3 |
| K3.1 1M on the pass engine with today's kernels | about 11.4 min | modeled, section 4 |
| K3.1 1M with the levers of section 5 | about 3.5-4.1 min | modeled |

Build: release `a242ef38e` (K3.1 adapter sha16 `e430d1f72200b6f7`) with the
capture fix device library of #1945 (sha16 `8b69d85be11a1a29`), weightd
`ee01d0d5665e4464`, all 16 Sparks, mesh lane 2, 2,048-row waves, the release
API on the hub, production deactivated. Receipts:
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
| KDA | 69 layers, 96 heads x 128, conv 4, full-rank gate, lower bound -5; vLLM runs the chunked kernel | per-token delta rule, FP32 state (`LmDeltaRuleColumnKernel`) |

Our index and latent cache are more precise than the publisher's engine;
our expert activations and per-token KDA differ from its arithmetic in the
other direction. None of these is a semantic difference.

## 2. Measured split on the production path

| Run | Rank 0 | Prompt (corpus offset) | Time | tok/s |
| --- | --- | --- | ---: | ---: |
| KP1 | CUPTI trace, complete (2.81M kernel records) | 4,096 warm-up, then 262,144 cold (0) | 656.8 s | 399 |
| KP2 | no tracer | 4,096 warm-up, then 262,144 cold (170,000) | 592.1 s | 442.7 |

Wave time by context, KP2 (a stall is a wave of 4.5 s or more):

| Context | Waves | Mean ms | Calm waves | Calm mean ms |
| --- | ---: | ---: | ---: | ---: |
| 0-32K | 16 | 4,567 | 8 | 3,945 |
| 32-64K | 16 | 4,826 | 6 | 4,099 |
| 64-128K | 32 | 4,468 | 20 | 4,027 |
| 128-192K | 32 | 4,833 | 12 | 4,209 |
| 192-256K | 31 | 4,467 | 19 | 4,229 |

Rank 0 inside a calm wave, KP1 (ms a wave; stages run on two streams and
overlap, so they sum to more than the compute union):

| Stage | 13.7K | 138K | 226K | At 226K |
| --- | ---: | ---: | ---: | --- |
| Wave | 3,951 | 4,184 | 4,141 | |
| GPU compute, union | 2,064 | 2,217 | 2,297 | |
| Mesh collectives, transfers and waits | 1,833 | 1,911 | 1,786 | |
| Routed experts | 644 | 637 | 639 | 19.5 TFLOPS |
| Dense projections | 375 | 396 | 384 | stream GEMM 41 TFLOPS; the 224-wide routed up-projection slice takes the slow `LmGemmKernel` (61 ms) |
| AttnRes score and mix | 301 | 302 | 302 | on every rank for all rows |
| Index score, select, merge | 29 | 162 | 246 | score 29.7 TFLOPS BF16, select 43 ps a score |
| Sparse MLA attend, merge, value | 183 | 186 | 193 | attend 13.3 TFLOPS |
| KDA, of it the scan | 191, 164 | 191, 164 | 191, 164 | token-serial scan, 1.16 µs a token step |
| MoE routing and finalize | 180 | 180 | 180 | |
| Other row-wise passes | 195 | 195 | 196 | |

Stalled waves (73 of 127 in KP1) spend 1.34 s more in the mesh arrival kernel:
rank 0 waits for its peers. The slow-path GEMM (+236 ms) and the sparse
attend (+123 ms) slow down beside the spinning arrival kernel. No weightd
transport error was logged; the late rank or link is not identified yet.

## 3. Why TP16 waves cannot reach minutes at 1M

- Even a calm wave spends 1.8 s of its 4.1 s in collectives. TP16 moves the
  activations: the MLA scatter alone is about 290 KB a row a layer (#1943),
  14.3 GB a wave a rank; with the per-layer all-reduces and the MoE exchange
  about 26 GB a wave, 12.6 MB a token (derived).
- Splitting a 2,048-row wave 16 ways makes small GEMMs: each expert sees about
  37 rows a wave on a rank (19.5 TFLOPS, its weight reads at half the memory
  bandwidth), the projections run at 41 TFLOPS.
- Row-wise work runs on all 16 ranks for every row: AttnRes, routing and the
  other row-wise passes take 0.68 s of each wave.
- The indexer adds 1.36 ms a wave per 1K of context: a wave at 1M carries 1.4 s
  more than one at 0.

The 512 waves of a 1M prompt sum to 39.6 minutes calm and 43.8 minutes at
KP2's stall rate.

## 4. Cost model for 1M on 16 nodes

The pass engine (`k3_streamed_prefill.cuh`, `streamed.cuh`) gives each node
its own rows (two passes of 31K rows a node for K3) and streams the other
ranks' weights to it, so linear layers, routing and row-wise work run once a
row at full GEMM size. Work a token, from the K3.1 config:

| Part | GFLOP a token |
| --- | ---: |
| MLA projections, index projections included (24 layers) | 11.5 |
| KDA projections (69 layers) | 61.2 |
| MoE BF16 parts: router, latent down and up, shared experts (92 layers) | 34.9 |
| Routed experts, MXFP4 (92 layers) | 97.2 |
| Dense layer 0 | 1.5 |
| Sparse MLA, 96 heads x 2,048 keys x (576 + 512) x 2 (24 layers) | 10.3 |
| Index, `24 x 8,192 x p` at position p | 103 at the 1M mean |

Per node at 1M (65,536 rows a node), with today's kernel rates and with the
rates of the levers:

| Stage | Work a node | Today | Today s | Lever rate | Lever s |
| --- | --- | --- | ---: | --- | ---: |
| Index scoring | 6.76e15 FLOP | 29.7 TFLOPS (KP1) | 228 | 120-150 TFLOPS FP8, fused | 45-56 |
| Index select, merge, exchange | 8.3e11 FP32 scores, 16 x 2,048 candidates a row, 390 GB of exchange | 43 ps a score (KP1) | 75 | fused, rows owned | 5 |
| Sparse MLA attention | 6.7e14 FLOP | attend 13.3 TFLOPS (KP1) | 70 | 45 TFLOPS, rows owned | 15 |
| MLA projections | 7.5e14 FLOP | 41 TFLOPS (KP1) | 18 | 80-90 TFLOPS | 8-9 |
| KDA | 4.0e15 FLOP, three token-serial scans | 137 s at 990,881 tokens (K3) | 145 | 80-90 TFLOPS, one chunked scan | 55-60 |
| MoE | 6.4e15 MXFP4 + 2.3e15 BF16 FLOP | 139 s at 990,881 tokens (K3) | 147 | 110-150 MX, 80-90 BF16 | 73-92 |
| Row-wise passes | rows owned | inside KDA and MoE | | fused | 10 |
| Total | | | about 683 (11.4 min) | | about 211-247 (3.5-4.1 min) |

The wire is hidden: a pass fetches 1.46 TB of weights a node (dense 109 GB and
experts 1.45 TB, 15/16 of them remote), about 63 s at two ports' 23 GB/s, plus
1.2 GB of latent keys and 0.27 GB of index keys a layer at 1M. Two passes
take 126 s of wire under 211-247 s of compute. Reaching 3 minutes takes the
last 30-60 s from kernel efficiency above the lever rates.

## 5. Levers ranked by minutes saved at 1M

| # | Lever | From DSV4.1 | Saves | Bits | Evidence |
| --- | --- | --- | ---: | --- | --- |
| 1 | K3.1 long prompts on the pass engine: rows owned a node, weights streamed under compute | rows owned per rank, overlap | 28-32 min | same per-row arithmetic; the indexed pass path has never run at scale, so it needs a needle and a paired comparison | K3 971.6 s; sections 3 and 4 |
| 2 | Index scoring at tensor-core speed: a BF16 tile kernel that stages each block's keys once and keeps the MMA k-order and head sum, then FP8 keys and queries as the publisher's engine | lean kernel loops; the publisher's arithmetic | 2.2 + 0.7 min | BF16 step exact; FP8 step moves bits and needs the quality comparison | score kernel 29.7 TFLOPS |
| 3 | KDA: one chunked tensor-core scan instead of three token-serial passes; projections at GEMM speed | lean kernel loops | 1.4 min | the chunked form moves bits (it is the publisher's engine's form) | K3 KDA 137 s; 1.16 µs a token step |
| 4 | Top 2,048 fused into the scoring, rows owned against replicated index keys: no FP32 score rows, no candidate exchange, no selected-index all-gather | replicated indexer, own-row scoring, radix top-k | 1.2 min | exact: same scores, same tie rule | select 43 ps a score; 390 GB a node |
| 5 | Experts as MXFP8 activations x MXFP4 weights on block-scaled MMA | native MX multiplies | 0.9-1.2 min | the publisher's QAT arithmetic; bits move, needs the quality comparison | K3 MLP and experts 139 s; our MXFP8 MMA 100-121 TFLOPS |
| 6 | Sparse MLA rows owned: a row's 96 heads share its 2,048 gathered latent keys | rows owned with attention, fused attention | 0.9 min | exact up to the attention sum order | attend 13.3 TFLOPS |
| 7 | Projections at 80-90 TFLOPS, AttnRes fused, passes overlapped | lean kernel loops | 0.3 min | exact | |

Levers 2-7 take the pass engine from about 11.4 to 3.5-4.1 minutes. With the
BF16 index tile kernel and no FP8 index the total is about 4.4-4.8 minutes.

The production TP16 path keeps serving shorter prompts; its own levers, by
share of a wave: stalls (49% of waves, +1.0 s), row-wise work owned per rank
(0.68 s a wave), the MLA key gather (#1943, +12-13% at 8-32K), whole experts
per rank (experts at 19.5 TFLOPS), and the 224-wide routed up-projection on
the slow GEMM (61 ms calm, 300 ms in a stalled wave).

## 6. Order of work

1. A K3.1 pass-engine lane: K3.1 partner packs on every node, a
   `streamed_prefill` record, and runs at 128K, 512K and 1M with a needle and a
   paired comparison against the TP16 path. Its CUPTI split replaces the
   modeled 11.4 minutes.
2. The BF16 index tile kernel with a bit-for-bit test against
   `LmWeightedSparseScoreKernel`, then the fused top 2,048 with a selection
   test (same positions, same ties), then rows-owned scoring on the pass engine.
3. The FP8 index: per-layer selection overlap and a teacher-forced comparison
   against BF16 on the long-context corpus, shown to the user before it
   becomes the default.
4. The chunked KDA scan, the MX experts (the DSV4.1 block-scaled kernels) and
   rows-owned sparse attention, each with its comparison.
5. The TP16 stalls: trace two to four ranks at once to find the late rank.
