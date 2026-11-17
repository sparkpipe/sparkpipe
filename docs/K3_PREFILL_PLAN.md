# Kimi K3.1 Pro 1M-token prefill plan

The goal is a cold 1,048,576-token Kimi K3.1 Pro prompt in about 3 minutes
on the 16 Sparks. The pass engine (`k3_streamed_prefill.cuh`,
`streamed.cuh`, `inference/runner/pass_prefill.cuh`) is the fast path: each
node owns its rows and streams the other ranks' weights. Measured on
2026-11-19 it prefilled K3.1 at 1M in 764.9 s from the release, 473.4 s
with this branch's exact changes. The DeepSeek V4.1 Flash 16-rank ideas
(#1931: rows owned, the publisher's MX arithmetic in native kernels, lean
kernel loops, both ports on the wire) and the levers measured since model 1M
at about 380-400 s on 16 Sparks with the publisher's arithmetic.

## Where it stands (2026-11-19)

| Item | Value | Source |
| --- | ---: | --- |
| K3.1 1M (998,468 tokens), pass engine | 764.9 s: MLA 461.4, KDA 157.5, MoE 143.5 | run KQ3 |
| The same with the staged index kernel | 687.0 s: MLA 393.7, KDA 147.8, MoE 143.7; rank 0's per-layer hidden absmax identical to KQ3 | run KQ4 |
| Plus row-owned index and attention, 96-head attend | 595.3 s: MLA 303.3, KDA 146.1, MoE 144.1; absmax identical to KQ3 | run KQ6 |
| Plus the sampled top-2,048 select | 529.8 s: MLA 239.5, KDA 145.1, MoE 143.7; absmax identical to KQ3 | run KQ7 |
| Plus the tail chunks on the tile kernel, the causal-bound skip, the row-tile value projection | 473.4 s: MLA 184.9, KDA 142.8, MoE 143.9; absmax identical to KQ3 | run KQ10 (KQ11 475.3 s, identical) |
| The same with the FP8 index (`index_fp8` 1) | 438.9 s: MLA 148.0, KDA 145.2, MoE 143.8; absmax moves up to 1.9% in pass 1 and up to 15.1% in pass 2 | run KQ12 |
| K3.1 1M needle answered by the same lane (mode switch, section 4) | BF16: prefill 477.1-480.6 s, the key correct, 528.6-529.0 s end to end; FP8 index: 440.7-446.4 s, correct, 485.6-489.1 s | runs KS3, KS6; KS5, KS7 |
| K3.1 256K, pass engine | 156.9 s, one pass | run KQ2 |
| K3.1 256K, 2,048-row TP16 waves | 592.1 s | run KP2 |
| K3 1M (990,881 tokens), pass engine, dense MLA | 971.6 s | [PERFORMANCE_STATUS.md](../PERFORMANCE_STATUS.md), 2026-10-31 |
| K3.1 1M with levers 1-6 of section 3 | about 380-400 s | model |

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

Where rank 0's GPU time goes at 1M with the sampled select (run KQ8: rank 0
under CUPTI, 529.4 s, the same as KQ7 untraced). Every kernel of the pass runs
on one stream and the GPU is busy for 99.4% of the prefill, so there are no
idle gaps to close; kernel time is the lever.

| Work | Rank 0 GPU s |
| --- | ---: |
| Index score: BF16 tile kernel 118.0, plus the 32-row tail chunk of each owner's share on the one-row kernel 22.6 | 140.6 |
| Stream GEMMs (BF16 projections, shared experts, the dense MLP) | 114.8 |
| Routed experts (`LmCellMmaKernel`, MXFP4 weights x BF16) | 83.9 |
| Mesh collectives (arrival waits 34.9, transfers 26.7, the rest 2.9) | 64.5 |
| KDA scan (`LmDeltaRuleChunkKernel` 28.2) and its small kernels | 42.0 |
| Index select (sampled 19.4, merge 2.7) | 22.1 |
| Value projection (`LmPerHeadProjectSplitKernel`) | 18.6 |
| SiTU, MoE finalize, routing | 18.2 |
| Sparse attend and merge | 16.3 |
| AttnRes | 9.3 |
| Norms, adds, gates and the rest | about 15 |

The rows add to 544.5 s; dependent launches let neighbouring kernels overlap.

The FP8 index (run KQ12) takes the MLA layers from 185.3 to 148.0 s and the
1M prompt to 438.9 s. It is the publisher's engine's arithmetic but changes
our bits: rank 0's per-layer hidden absmax moves at most 1.9% in pass 1 and
up to 15.1% at the last layer in pass 2, where rows attend up to 1M keys.
The absmax is the largest value over 31,232 rows, so one sensitive row can
move it; whether the answers move needs decode after the pass (section 4),
so BF16 stays the default.

GB10's MMA peak, measured with `mma.sync` loops free of dependencies at
2.44 GHz with FP32 accumulation: 123.6 TFLOPS BF16, 247 TFLOPS FP8 (E4M3).
Against it the index score kernel ran 55 TFLOPS on chunks of mixed lengths
(69% on full rows; its epilogue, key staging and per-row query loads each
cost 12-17%, and blocks past the causal bound still did that work), the
stream GEMMs 72-80 TFLOPS at the pass's shapes (58-65%), and the FP8 index
116 TFLOPS.

The collectives: the first exchange of each KDA layer, the convolution tails
(18 KB a node), waits 62.7 ms on average, the drift between nodes from the MoE
before it, whose time depends on each node's routing; the 15 later ones in
the layer wait about 4 ms. The weight stream runs on the same port
(`rocep1s0f1`) and traffic class (106, PFC priority 3) as weightd's mesh
rounds, so a small round can queue behind bulk writes.

## 3. Levers on the pass engine, ranked by what they save at 1M

| # | Lever | Bits | Rank 0 now (KQ8) | Saves at 1M | Status |
| --- | --- | --- | ---: | ---: | --- |
| 1 | Index score: the 32-row tails on the tile kernel; rows and tiles past the causal bound write -inf without loading queries | kept | 140.6 s | 1 and 2 together: 56.4 s | done, run KQ10 |
| 2 | Value projection on the row-tile kernel, the same FMA order per output | kept | 18.6 s | (with 1) | done, run KQ10 |
| 3 | FP8 index keys and queries as the publisher's engine (`streamed_prefill.index_fp8`) | the publisher's | about 95 s after 1 | 36.4 s (KQ12) | an option until the needle and logits compare it |
| 4 | Stream GEMM toward 85% of the BF16 peak, the same k order | kept | 114.8 s | 25-30 s | next |
| 5 | Collectives: the weight stream on its own priority class or port; fewer rounds in the KDA shard loop | kept | 64.5 s | 20-40 s | the class or port is a switch QoS decision |
| 6 | Chunked KDA scan, as vLLM runs it | the publisher's engine | 28.2 s | about 20 s | |
| 6a | The select's read folded into scoring: a sampled threshold before the score kernel, which then keeps only the scores above it | kept | 22.1 s | about 15 s | |
| 6b | SiTU into the expert up-projection's epilogue and the MoE finalize into the down-projection, over about 9 GB of expert intermediates a layer | kept, if the finalize keeps its sum order | 17.0 s | about 12 s | |
| 7 | MXFP8 activations for the routed experts (the README's QAT; the checkpoint's serving config leaves activations unquantized) | lower than the publisher's serving config | 83.9 s | 30-40 s | your call after a comparison |
| 8 | Both 100G ports for the weight stream: about 64 s a pass instead of 128 | kept | hidden | needed once a pass computes in under 128 s | |

The exact index steps so far (runs KQ4, KQ6, KQ7). The index kernel,
measured on a Spark with `tests/test_index_score_tiles_cuda.py` (bit for bit
against its one-tile form): the one-tile form scored a 128-row chunk over 1M
keys at 17.6 TFLOPS because each 8-row block read every key itself; the tile
kernel stages each key tile once in shared memory for 64 rows and runs at
54.9 TFLOPS (51.3 at 65,536 keys, 53.6 at 262,144) and 68 TFLOPS with lever
1. Giving each warp four key tiles at once kept the bits but ran 0.63-1.16x
and was dropped.

The local top 2,048 read each row's FP32 scores four or five times (radix
passes, then an emission scan): 13.06 ms for 128 rows over 1M keys. The
sampled select (`inference/kernels/index_select.cuh`) reads a strided sample
of the row, then the row once, keeping every score at or above a sampled key
in shared memory with a fine histogram beside it; an exact radix select over
that list keeps the same candidates (ties to the earlier position). When the
list overflows, a second read keeps the histogram bucket that holds the
threshold; when that overflows or the sample was wrong, the row takes the
radix select. 128 rows: 2.87 ms at 1M keys (187 GB/s), 0.76 ms at 262,144,
0.07 ms at 32,768; `tests/test_index_select_sampled_cuda.py` checks every row
against the radix select, ties and a sharded rank included.

What the levers reach: 1-6 bring a 1M prompt to about 380-400 s and 7 to
about 350 s on 16 Sparks. The BF16 GEMMs alone need about 70 s at the MMA
peak, and two passes cannot go under about 256 s while the weight stream
takes 128 s a pass. Three to four minutes on one 16-Spark fleet would need
precision below the publisher's in the dense GEMMs; with the publisher's
arithmetic it is in reach on 32 Sparks, where a 1M prompt fits one pass.
The earlier model of 225-260 s assumed the KDA and expert GEMMs could drop
by more than the BF16 peak allows.

## 4. Serving through the pass engine

- On a pass lane the decode step after a pass aborts: the lane caps the expert
  pool at 8 GiB to make room for its window and engine, and the runner
  refuses a decode step until every expert group is resident
  (`StageRunnerStep` in `inference/runner/stage_runner.cu`). The KV the pass
  wrote (latent and index keys, about 2.1 GB a rank at 1M) sits in weightd's
  KV pool, and the KDA state (about 27 MB a rank) in the lane's state pool.
- The mode switch, one lane: a lane whose `streamed_prefill` names
  `expert_pool_cap_bytes` starts with the whole arena resident and changes
  its weights around each streamed prefill. Before the pass the runner drops
  its decode graphs and leases, detaches the pack, has weightd reclaim the
  cold arena (`RECLAIM_PACK`), attaches it again with the pass cap (an 8 GiB
  slot pool and a private spine) and re-points the embedding, head and model
  weights (`rebind`); the model then builds its pass engine for the prefill
  and frees it after. Back the same way: detach, reclaim, attach whole, warm
  every expert group (acquire and release, as `weightd_warm --shared`),
  re-point, and the next decode captures new graphs. The KV pool and the
  mesh lane have their own weightd connections and the KDA states live in
  the lane's state pool, so none of them moves; the runner keeps the first
  attach's mesh mapping, which the device collective registered once.
- Measured: the switch to the pass pool takes 3.7-4.3 s on rank 0 (unmap
  0.7, reclaim 1.0, attach with the private spine 2.1), the switch back
  about 20 s (attach 5, warm 15) and 27 s after a 1M prefill (attach 12).
  KS2 answered a 100,869-token needle in 142.9 s end to end. The 1M needle
  (998,469 tokens) was answered correctly in all four 1M runs: BF16 prefill
  477.1 and 480.6 s, 528.6 and 529.0 s end to end (KS3, KS6); FP8 index
  446.4 and 440.7 s, 489.1 and 485.6 s (KS5, KS7). The switch leaves the
  prefill's bits unchanged (rank 0's absmax identical to the pass-lane runs
  KQ11 and KQ12).
- The ranks must enter the pass together. A long prompt makes the KV pool
  grow just as the lane detaches, and weightd answers every client on one
  thread, so on some ranks the DETACH round trip waits behind the grow (0 to
  10 s in KS5, 40 s on one rank in KS4). In KS4 the other ranks entered the
  pass's first exchange and ran out the device collective's 30 s round
  budget, and the needle failed. After each switch the runner now gathers a
  switch token from every rank in one round with a 120 s budget
  (`SparkTpDeviceCollectiveSetOperationTimeout` for that round) and checks
  that every rank reached the same switch; every other round keeps the
  lane's budget. In KS6 the 1M switch took 3.7 to 10.8 s across ranks and
  the others waited up to 7.1 s at the rendezvous. Growing the KV pool after
  the reclaim, or a weightd that does not hold its other clients behind a
  grow, would take those seconds off a 1M request.
- A grow sent just before the switch was answered after the arena was
  detached, when memory looks plentiful: on 2026-11-21 (run LB1) a 458K
  prompt took the pool from 400 to 800 chunks (5.9 to 11.7 GB) that way, the
  whole arena no longer fit beside it after the pass (1.5 GB short), and the
  request failed. The runner now holds the pool's growth from just before the
  switch to the pass pool until the whole arena is back
  (`SparkStageRunnerConfiguration.kv_pool_hold`): a grow already sent is
  waited out, then pages past the pool's size park or evict as on a full
  node. The prompt's pages were claimed by its waves before the switch, so
  the pass needs none. In every run since, the pool kept its size across
  each switch (200, 400 and 687 chunks), and every pass that finished
  switched back.
- Releasing only the expert chunks of one whole-arena attach would keep every
  pointer, but weightd maps a whole arena as one allocation and CUDA frees
  no part of one, so that needs a daemon change and a weightd release; the
  switch above runs on the deployed weightd.
- With a second fleet the same split is prefill and decode on different
  nodes: the prefill fleet runs the pass lane and the decode fleet copies
  the sequence's KV and state over the fabric (about 34 GB at 1M, a few
  seconds), with no switch on either side.
- The hand-off is also what lets a needle be answered after a pass, the
  check the FP8 index and every other change of bits needs before it is on.
- The FP8 index against BF16 (KS6 against KS7, the same lane and adapter):
  the 1M needle is correct in both; the 1M prefill takes 35 s (7%) less, the
  400K 4% less and the 131K none. A 400K corpus answer keeps its first 36 of
  64 greedy tokens: over those the chosen token's logprob moves 0.026 on
  average and at most 0.097, and token 36, a near tie under BF16 (margin
  0.062), swaps. A 131K answer keeps 26 of 32 (logprob moves 0.076 on
  average, at most 0.30; a 0.055 near tie swaps). BF16 stays the default
  until you decide.
- The FP8 index on a long-prompt suite (2026-11-21, partial,
  [`qualification/k31_pro/prefill-20261118/`](../qualification/k31_pro/prefill-20261118/README.md#fp8-index-on-a-long-prompt-suite-2026-11-21-partial)):
  20 prompts of RULER's retrieval and tracking tasks (mk, mv, mk3, vt) at
  131K, 458K and 1M, 140 questions an arm. The runs stopped after five
  paired prompts (36 questions): BF16 36 right, FP8 28; only BF16 8, only
  FP8 0 (exact McNemar p 0.0078); FP8 minus BF16 -22.2 points, 95%
  bootstrap interval over prompts -66.7 to 0.0. All eight are one prompt
  (mv-131k-0), whose first token is a 0.030 near tie under BF16 that FP8
  swaps: FP8 writes out its search and stops at the 384-token cap, its 13
  quoted pairs all right but none numbered. The other four answers are
  identical token for token (logprob moves 0.0012 on average, at most 0.23).
  mk-458k-0 prefills in 182.9 s against 204.5 s. BF16 also answered two 1M
  prompts, all 16 questions right (520.5 and 543.6 s end to end, passes
  480.5 and 493.0 s); they have no FP8 pair yet. The remaining 27 prompts
  need about 2.5 hours of full-fleet load.
- Where the pass engine wins: a pass streams about 1.3 TB of expert weights
  a node, 128 s today, whatever the prompt. Waves take about 128 s for 64K
  tokens, so the threshold is about 64K now and about 32K after lever 4. On
  2026-11-07 a 33K prompt took 282 s through 2,048-row passes against about
  66 s in waves.
- Open: reuse across turns. The exact-reuse rule (draft #1972, TECHDEBT
  "Exact reuse") says a request's tokens must not depend on what the cache
  held, and streamed arithmetic is not wave arithmetic. Today the batch
  engine streams whatever follows the cached prefix
  (`SparkModelBatchDecideDefer`): the span starts where the cache ended,
  ends at the prompt's checkpoint rather than at the request's cuts, and
  splits into passes from its own start, and the blocks it writes are
  published for later requests. Under the rule a prompt that streams is
  computed whole every time instead: it resumes from nothing and nothing
  resumes from it, so on a lane that streams every follow-up turn of a long
  session pays the full pass again (about 480 s at 1M in BF16). Reuse needs
  the streamed span to end at the request's cuts and its passes to split at
  fixed positions, so that a resumed request computes what a cold one does.
  No lane in `deployment/lanes.json` streams today, so nothing breaks yet;
  this has to be settled before one does.

## 5. The TP16 wave path, below the threshold

Measured on 2026-11-18 (runs KP1 traced, KP2 untraced): a 256K cold prompt
in 592.1 s; a calm 2,048-row wave takes 3.94 s + 1.36 ms per 1K of context,
and 49% of waves stall by +1.0 s waiting on peers. The late ranks (runs
KD1-KD5,
[`qualification/k31_pro/tp16-stalls-20261119/`](../qualification/k31_pro/tp16-stalls-20261119/README.md)):
on release `a242ef38e` the stalls follow the 16,384-token KV checkpoint,
where the release writes the chain the checkpoint closes to each rank's
snapshot store in one burst (256 KV pages and the recurrent state, 257 files
within half a second), and the ranks that arrive late there are Gigabyte
boards; the Lenovo nodes are never late. No GPU throttling, host stall or
memory pressure goes with them. PR #1949 (merged) spreads those writes over
the waves and removes the checkpoint stalls on both releases: the time the
checkpoint waves spend above the median fell from 74.1 to 1.7 s on
`a242ef38e` and from 18.7 to 6.1 s on `c2b714343`. On `c2b714343` a second
source costs about 20 s a 256K prompt away from the checkpoints, with or
without #1949: the prompt took 558.0 s with #1949 and 565.6 s without it,
against 520.1 s on `a242ef38e` with it. spark0 and sparkf go late together
about every 33 s of wall clock, matching no periodic job in the nodes'
journals, and a calm wave costs the same on both releases. In a calm wave at 226K rank 0 computes 2.3 s
and spends 1.8 s in collectives; experts run at 19.5 TFLOPS, the index score
at 29.7, the sparse attend at 13.3, the projections at 41, and AttnRes and
the other row-wise passes take 0.68 s on every rank. Its levers: the
checkpoint stalls (#1949, measured),
row-wise work owned per rank, the MLA key gather (#1943), whole experts per
rank, and the 224-wide routed up-projection that takes the slow GEMM path
(61 ms a calm wave).

## 6. Order of work

1. Done: the exact index steps and the value projection (runs KQ4-KQ10,
   764.9 -> 473.4 s) and the FP8 index as an option.
2. Done: the serving hand-off (section 4), the 1M needle answered by the
   same lane in BF16 and with the FP8 index, and the comparison; the FP8
   index waits for your decision. The long-prompt suite comparison is
   partial (5 of 20 prompts paired) and resumes on your word.
3. Exact levers by size: the stream GEMM toward the BF16 peak, the
   collectives (with a decision on the weight stream's class or port), the
   select folded into the score kernel, the SiTU and MoE finalize passes
   folded into the expert kernels.
4. The chunked KDA scan with its comparison; MXFP8 expert activations only
   on your call after a comparison.
5. Both ports for the weight stream once a pass computes in under 128 s.
6. The TP16 stalls, for the prompts below the threshold. PR #1949 (merged)
   removes the checkpoint share. Next is the second source on `c2b714343`
   (about 20 s at 256K, a 33-second wall-clock beat on spark0 and sparkf):
   per-second NIC and GPU counters on the late nodes during a run.
