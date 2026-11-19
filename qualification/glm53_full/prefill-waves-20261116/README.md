# GLM 5.3 Full prefill on sixteen Sparks: where a step goes, and one wave a step, 2026-11-16

The measure is an imagined sixteen-node reference built from the four-Spark
reference's own per-node rates (multiplies about 80 TFLOPS a node, sparse
attention 22, index scoring 28, everything else 1.5 times the multiplies),
applied to GLM 5.3 Full's shapes at the publisher's BF16 and capped by the
network at 11.5 GB/s a node: about 4,700, 4,600 and 4,300 tok/s at 8,192,
32,768 and 131,072 tokens. Every figure below is a percentage of those.

## What ran

- A test lane on all sixteen Sparks, mesh lane 0, its own root
  (`glm53_full_ab` in `tools/stage_lane.py`: the production record's model
  with another root, so the production record's root was not written), with
  the production record's settings: 8 sequences, 1,048,576 positions a
  sequence (16,384 pages), 131,072 logical pages, a 12 GiB state budget,
  `MemoryMax` 28G. Nothing else ran on the nodes; production was already
  deactivated through the swap controller.
- Packs: the production BF16 TP16 packs (`glm53full.bf16.tp16`, rank 0 pack
  SHA-256 `bafb7f39cd9f22e1…`).
- Firmware built on the build host from main `dc1f2face` and from this
  branch; the API is the hub's x86 build of the same trees. Adapter SHA-256:
  main `2edf20e1393d94b6…`, this branch `a722f7b42e00de4b…`.
- Requests: one request at a time, one output token, speculation off, text
  from the corpus the four-Spark work used, re-tokenized for GLM
  (`long.glm.tokens.u32`, SHA-256 `c591d3659f5cae1d…`, 232,933 tokens), a
  new offset for every prompt so nothing is cached. A figure is the second
  request after a lane starts (the first pays the warm-up). The answer check
  is 16 greedy tokens with the five likeliest at each after a 40,000-token
  prompt (`wave_check.py`), hashed.

## Main today

| Run | 8,192 tokens | 32,768 tokens | 131,072 tokens | 40,000-token answer |
| --- | ---: | ---: | ---: | --- |
| main, four lane starts over two holds | 1,462, 1,393, 1,453, 1,449 tok/s | 1,254, 1,336, 1,290, 1,266 | 1,022, 985, 996, 916 | `7ffa139edd31bcf1` every time |
| Mean, % of the reference | 1,439, 30.6% | 1,287, 28.0% | 980, 22.8% | |

glm-5.3 serves on today's main with the production record's settings. The
release a242ef38e did not qualify it because the swap controller's entry
names the old lane; that is the release kit's matter (PR 1937), not the
engine's.

## Where a step goes

A CUPTI trace of rank 0 (`libspark_cupti_trace.so`) during the same
requests; the trace slows the lane by about a quarter (1,087 / 856 / 690
tok/s), so kernel times are the trace's and the whole step is given both
ways. The step is the fifth 4,096-row step of the 32,768-token prompt
(positions 16,384 to 20,479), which the stage runs as two lanes of 2,048
rows on two streams. The reference budget is the imagined reference's time
for the same 4,096 rows at this position: experts 45.3 GFLOP a token,
dense multiplies 28.7, shared expert 5.7, sparse attention 22.2 (2,048 keys
a query), index scoring 2.9, over sixteen nodes at the rates above, and
half of their sum for everything else.

| Stage | Ours, ms a step | Reference budget, ms | Ours as % of budget |
| --- | ---: | ---: | ---: |
| Routed experts (grouped up and down projections, both lanes) | 1,502 | 145 | 1,036% |
| Dense multiplies (q_a, kv_a, q_b, both kv_b sides, o_proj, index projections) | 187 | 92 | 203% |
| Shared expert | 85 | 18 | 472% |
| Router (every rank, every row) | 133 | in the dense budget | |
| Sparse attention and its key union | 416 | 259 | 161% |
| Index scoring, merge and top-k | 59 | 27 | 219% |
| Row-wise passes (MoE finalize 277, route build 112, router top-k 66, norms 24, rope, KV store, digest) | 534 | | |
| Collectives with nothing else on the GPU | 1,579 | | |
| Idle | 21 | | |
| Row-wise passes, exposed collectives and idle together | 2,134 | 270 | 790% |
| Whole step | 4,114 traced, 3,050 untraced | 811 | 507% traced, 376% untraced |

The kernel times overlap where the two lanes run at once (2,919 ms of
non-collective kernels in 2,514 ms of wall time); the collective kernels
were 2,548 ms in all, 2,032 of them waiting for peers. The untraced step is
the API's own step time for the same positions (`batch_step` in the API
log: 3,050 ms on the second main run; one step of the first run took
4,585 ms).

What the trace shows, per lane and layer (2,048 rows):

- The routed experts are three quarters of the step's multiply time. Each
  rank holds a sixteenth of every expert (an intermediate slice of 128), so
  each layer reads the rank's whole 1.21 GB expert slice once a lane, the up
  projection reads every route's 12 KB row twice (two 128-wide column
  tiles), and the down projection writes a full-width BF16 partial for every
  route (98 KB a row) that the MoE finalize reads back: 4.8 ms, 2.7 ms and
  1.1 ms a lane and layer, all at the memory's rate (about 250 GB/s).
- The two lanes take turns at the collectives (rounds alternate strictly so
  every rank issues them in the same order), and a lane waits about a third
  of its time for its turn; with one lane's experts at 8.6 ms, the other
  lane's rounds cannot all hide under them. 1.6 s of the step has only
  round kernels on the GPU.
- Each lane gathers the context's old keys from all sixteen ranks at every
  layer, so the step gathers them twice.
- Sparse attention runs about 15.6 TFLOPS (the reference rate is 22): four
  heads a rank use each gathered key for four heads only.
- Every rank computes the router, its top-k and the route table for every
  row, which the experts' slicing needs.

## The change: one wave a step, of up to 16,384 rows

The stage runs one wave a step: no lane split, one arena, and a step of up
to 16,384 rows (the lane record chooses). For that the serving limit is
16,384 rows and a KV lane transaction holds 256 pages; the gathered
attention's decode-split partials (unreachable for a gathered wave, 539 MB
at 4,096 rows) are gone; the index selection buffer keeps the top-k's own
64 rows of 1M positions and the scoring and merge run in as many rows as
fit at the wave's row length. The rank arena is 2.99 GB at 4,096 rows
(main: two arenas of 4.33 GB), 4.06 GB at 8,192 and 6.21 GB at 16,384.

Same lane settings, the second request after each start, the arms in the
order main, this branch at 4,096, 16,384 and 8,192 rows, this branch at
4,096, main (the first hold had run the lanes' removal alone twice:
1,386 and 1,384 / 1,370 and 1,352 / 1,211 tok/s):

| Build, wave rows | 8,192 tokens | 32,768 tokens | 131,072 tokens | % of the reference |
| --- | ---: | ---: | ---: | --- |
| main, 4,096 (mean of four) | 1,439 tok/s | 1,287 | 980 | 30.6 / 28.0 / 22.8 |
| this branch, 4,096 | 1,450, 1,443 | 1,426, 1,419 | 1,254, 1,249 | 30.8 / 30.9 / 29.1 |
| this branch, 8,192 | 1,613 | 1,606 | 1,444 | 34.3 / 34.9 / 33.6 |
| this branch, 16,384 | 1,449 | 1,608 | 1,424 | 30.8 / 35.0 / 33.1 |

Every run answered the 40,000-token prompt with `7ffa139edd31bcf1`, main's
bytes. Against main this branch is +1% / +11% / +28% at the production
record's 4,096 rows, +12% / +25% / +47% at 8,192 and +1% / +25% / +45% at
16,384. The 8,192-row figure is one run.

- A 16,384-row lane started first after another workstream's lanes was
  still warming (1,127 / 1,380 / 1,231 tok/s, first request 23 s); it is
  left out of the table. An 8,192-token prompt is one step either way at
  8,192 and 16,384 rows; its figures differ by 11% on one run each.
- At 16,384 rows the KV snapshot saver defers saves (`kv_page_cache.c:2811`
  BUSY, recoverable) a few times a prompt; main never did. Free memory after
  start: 4.8 to 6.7 GiB a node at 16,384 rows, 5.9 to 9.2 at main.
- Not measured: a decode stream waits behind the wave in front of it, so
  larger waves trade stall length for prefill rate (on four Sparks: 4
  stalls up to 6.3 s at 16,384 rows against 16 up to 2.5 s at 4,096). The
  production record keeps 4,096 rows until that is measured and chosen.

## A reference-style layout for GLM on sixteen nodes

The layout the four-Spark work arrived at, carried to GLM 5.3 Full's shapes
and the publisher's BF16 arithmetic. FP8 codes on the wire and FP8 or FP4
multiplies do not carry over: the publisher's GLM code computes these
activations in BF16. Each step names its expected gain from the stage table
above (a 4,096-row step at 32,768 tokens, about 3.0 s untraced against the
reference's 0.81 s) and whether it keeps the bits.

1. **One wave a step, of up to 16,384 rows** (this change, measured above).
   The experts' 1.21 GB slice and the old keys are read once a wave instead
   of once a 2,048-row lane, and the lane arena's 4.3 GB pays for the larger
   wave. Bits kept.
2. **Whole experts per rank.** Rank r holds experts 16r to 16r+15 of every
   layer, whole (a pack plan like the four-Spark `whole` plan; each rank
   re-reads its 90 GB). Every rank already holds every normed row after the
   gather, so a rank runs only its own experts' routes and the MLP's
   reduce-scatter carries per-rank sums, the same bytes as today's slice
   partials. It removes the full-width BF16 partial each route writes today
   (98 KB a row and layer) and the finalize that reads it back, and makes
   each expert's down projection 2,048 deep instead of 128: experts and
   finalize from about 1.1 s to about 0.3 s a 4,096-row step at 16,384-row
   waves, 25 to 35%. The per-rank sums are not today's per-slice sums, so
   the bits move: score both builds with `tools/ab_api_score.py`. Decode
   then waits for the rank with the most of a step's experts, as on four
   Sparks (whole experts there: +4% prefill, -5 to -9% decode), so it is a
   pack option, not a replacement.
3. **Collectives under the multiplies.** With one wave a step every round
   is on the critical path: about 1.4 s of round kernels a 4,096-row step.
   Cut o_proj and the shared and expert down projections into column slices
   and reduce-scatter each slice on a second collective stream while the
   next is multiplied, in a fixed order on every rank (the four-Spark
   reference hides about 110 ms of a 1,816 ms step this way). Half of the
   round time hidden is 15 to 20%. Bits kept: the same sums in the same
   order.
4. **Row-owned router and indexer projections.** The rank that owns a row
   for the norm also computes its router logits, top-8, index query and
   index head weights, and the routes travel with the gathered normed row
   (64 bytes). Today every rank does this for every row: router 133 ms,
   top-k 66 ms, most of the 20 ms of index projections. About 6%. Bits kept.
5. **The spine in four groups of four.** Attention, the shared expert and
   the three dense layers tensor-parallel within a group of four, the rows
   split between the groups, experts whole across all sixteen (step 2).
   A rank's spine collectives then move a fifth of today's bytes (four
   ranks, a quarter of the rows), and each gathered key serves 16 heads
   instead of 4, which is what the attention kernel needs to reach the
   reference's 22 TFLOPS (it runs 15.6 today). The spine at four ranks is
   9.5 GB a rank instead of 2.4: steps 1 and 2 free the 4.3 GB lane arena
   and the 0.5 GB of split partials, and the KV and index caches of a
   sequence then live with its group. Exposed collectives from about 1.0 s
   to about 0.3 s and attention from 416 to about 300 ms a step: 30% or
   more. The largest step: a pack layout, group collectives and KV
   placement.
6. **Kernels.** The up projection's tile covering both of a slice's halves
   (each route's row read once instead of twice), a parallel route build
   (112 ms in one block today), a vectorized finalize: about 5% together.
   Bits kept.

Steps 1 to 6 together would bring a 4,096-row step to about 1.4 s, about
2,900 tok/s at 32,768 tokens or 63% of the reference; the rest is the
experts' multiply rate (the reference's 80 TFLOPS against our BF16 cell
kernel's) and the attention kernel.
