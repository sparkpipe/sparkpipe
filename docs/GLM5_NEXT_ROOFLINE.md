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

## Measured after PR #1205 (fleet, 2026-09-25)

| Metric | Before | After |
| --- | ---: | ---: |
| Warm decode tok/s | 4.4-6.5 | 7.3-9.1 |
| Graph replay p50 | 90-99 ms | 45.8 ms |
| Compute kernels per token | 54.6 ms | 21.6 ms |
| BF16 skinny GEMV per token | 12.8 ms | 6.2 ms (234 GB/s) |
| Routed FP8 experts per token | 14.0 ms | 3.2 ms (W1 1.8, W2 1.4) |
| HC site per token | ~9.5 ms | 2.7 ms |
| Latent attention per token | - | 2.5 ms (11 calls) |
| Router top-k per token | - | 1.05 ms (42 calls) |

An isolated 16-rank 8 KiB all-reduce through the shared weightd mesh costs
p50 174-319 µs (bimodal by rank) and p99 about 775 µs. That is roughly
27 ms per token over about 90 collectives, so collective latency, not
compute, is now the largest term.

## Collective relay (weightd)

A hardware-wait round crosses weightd's CPU relay twice: the local sweep
posts the RDMA writes, and the remote sweep completes the waiter. Before
this branch every doorbell-thread iteration touched all 512 doorbell cells
with a full barrier each and all 512 wait cells twice. It also printed one
`WD-SEEN` line to stderr per round while holding the wire lock. Only one
doorbell cell and one wait cell per configured band can carry valid work:
the one at the lane's local rank. The sweep now polls only those cells and
runs the full 512-cell scan once per millisecond, which keeps the error
reporting for invalid cells. The doorbell thread is pinned to the
highest-capacity CPUs.

Other threads now get priority on the wire lock over the spinning doorbell
thread. A lane that stays active without any doorbell, wait or send for
20 ms is polled every 200 µs instead of spun on. A client that disconnects
while active now quarantines only its own lane. A lane acquire with a
topology clears the quarantine after `SparkWeightdMeshLaneConfigure` proves
the lane is quiescent.

## Measured after PR #1208 (fleet, 2026-09-25)

| Metric | Value |
| --- | ---: |
| Warm decode tok/s | 27.6-29.8 |
| Token interval / chain p50 | 28-30 ms / 28.8 ms |
| Graph replay p50 / p90 | 25.5 / 44.5 ms |
| Compute kernels per token | 17.2 ms, 1950 launches |
| 8 KiB 16-rank all-reduce p50 / p90 / p99 / max | 167 µs / 472 µs / 3 ms / 12 ms |

While the all-reduce ladder ran, weightd spent 91% of its CPU in
`SparkWeightdMeshPoll`. Of that, 54.5% was in `open()` of peer records and
36.9% in `ibv_query_qp`, which is a firmware command. `TryWire` ran both
while holding the wire lock. Since waiters now have priority, the doorbell
thread waits out every such call, and that explains the millisecond tail.
Iteration 3 moves the peer-record reads and QP queries outside the lock. It
also turns CQ-error repair into a flag that the main thread services at most
every 10 ms. `WD-MESH-STATS` reports try-wire calls, record failures,
rewires, not-ready transitions and repairs every 10 s.

## Batching

Before iteration 3, only single-row waves used the captured graph. A batch
of two or more decode rows fell back to the eager per-layer path. Its
collectives also took the 8-phase binomial tree, because
`logical_sequence_count > 1`, and each phase is one relay round trip. Batched
decode was therefore slower per step than B1 by far more than the extra rows
justify.

Iteration 3 makes three changes:

- It captures one graph per row count (1-8) per slot.
- A hardware-wait collective whose payload fits one mesh slot (256 KiB,
  i.e. up to 32 rows of 4096 BF16 or 8 rows of 16384) runs as a single
  direct round.
- The skinny GEMV covers up to 8 dense rows and 8 routed tokens.

Weights are read once per step for all rows, so the expected step time grows
slowly with B. Routed experts grow with the number of distinct experts
selected.

## Pair links

Each Spark has a second port (nominally 200 Gb/s, about 100 Gb/s useful)
cabled directly to `rank XOR 1`. A hierarchical all-reduce runs in three
steps:

1. Pair sum over the direct link.
2. An 8-way exchange of half vectors: even ranks own half 0, odd ranks own
   half 1.
3. Return of the other half over the direct link.

This cuts switched bytes per rank from 15 x payload to 3.5 x payload. It
also needs three relay-mediated rounds instead of one.

| Payload | Direct all-to-all wire time | Hierarchical wire time |
| --- | ---: | ---: |
| B rows of 8 KiB | about 9.6 µs x B | about 2.9 µs x B |
| 256-row prefill chunk (2 MiB) | about 2.5 ms | about 0.6 ms |

With a per-round cost of 50-170 µs, the two extra rounds only pay above
roughly 25-30 rows. It is therefore the right algorithm for prefill and
large batches. It is the wrong one for B1-B8 decode, where one direct round
is latency bound.

Splitting a batch into row groups, so that one group's all-reduce overlaps
another group's compute, costs an extra read of the layer weights per group.
Decode is weight-bandwidth bound, so that trade does not pay at small B. The
overlap that does pay at B1 is using the collective wait to stream the next
projection's weights into L2. `test-skinny-gemv` now prints the device L2
size to size that experiment.

## Multi-row prefill and the poisoned sequence slot (#1210)

The B8 profile rejected prefill waves with
`submission_rejected status=1 kind=1 rows=64 lanes=1`. Two defects combined.

1. **The first multi-row wave failed in the collective.** The module sets
   `logical_sequence_count` to the number of sequences in the wave. A prefill
   wave of one sequence has 8 or 64 rows but a logical count of 1. The
   collective treated a logical count of 1 as a payload that fits one mesh
   slot. `RoundAdmit` returned `CAPACITY_EXCEEDED` for anything larger: the HC
   collective at 8 rows is 8 × 16384 × 2 bytes = 256 KiB, one trailer over the
   slot. Single-row prefill always fit, which is why B1 worked.
2. **The failure poisoned the resident sequence slot.** residentd bound the
   slot to the request's identity on every completion, failed or not. The
   batch engine sends a RELEASE only for a request whose resident state it
   has seen succeed, so after a failed first wave nothing unbound the slot.
   Every later submission on that slot failed `ValidatePersistentSlot` with
   `INVALID_ARGUMENT`. The host pipeline test reproduces the exact log line
   without the fix.

The fixes:

- In hardware-wait mode every collective now runs as direct all-to-all rounds.
  A payload larger than one slot is split into slot-sized chunks, one relay
  round each. The binomial tree is no longer used in hardware mode, and spin
  mode is unchanged.
- The slot gains a 64-byte trailer margin, so eight 16384-wide BF16 rows fit
  one slot exactly.

  | Collective | Before | After |
  | --- | --- | --- |
  | HC collective at B8 decode | 3 tree chunks × 8 phases = 24 relay rounds | 1 round |
  | 64-row prefill attention collective | failed | 2 rounds |
  | 64-row prefill HC collective | failed | 8 rounds |

- residentd no longer creates a binding from a failed completion. It
  invalidates the continuation lease on failure.
- The batch engine sends a RELEASE after an admitted prefill fails, because
  some ranks may have bound state. A rejected RELEASE ends the request instead
  of queueing another RELEASE.

## Execution order across ranks (4-stream deadlock)

Tensor-parallel ranks exchange collectives by mesh sequence number, not by
submission. Every rank must therefore execute submissions in the same order.

Each residentd runs submissions from a committed FIFO. Before this fix the
FIFO order depended on timing:

- A decision commit entered the FIFO when its message arrived.
- A continuation entered only once its local prepare succeeded. An adapter
  `BUSY` or an exhausted progress budget could defer that prepare to a later
  pass.
- The route scan starts at a rank-local position.

With one submission in flight the order cannot diverge. With several, one
rank could run a prefill while its peers ran a decode continuation. The ranks
then published different payloads under the same mesh sequence. The
collectives waited for tags that never came, until the 35 s completion drain
failed. The engine then went terminal and restarted, which is the connection
refused the API saw afterwards.

residentd now keeps the committed FIFO sorted by `submission_id`. It hands a
route to the adapter only when that route is the FIFO head and no active
route with a smaller `submission_id` is still before its commit (reserved,
resolving, preparing a continuation, or not yet queued). Every rank thus
executes in the client's submission order. The pipeline test defers one
stage's continuation prepare while a later prefill commits; without the fix
that stage runs the prefill first.

## Decode waves (8 streams at 1.5x one stream)

After the ordering fix, 8 streams completed but reached only 30.4 tok/s in
aggregate, and every graph replay was `rows=1`.

The batch engine dispatched each READY_DECODE request as soon as it was
ready, as long as a submission slot was free (`max_inflight_submission_count`
is 4). On a PARALLEL_FANOUT deployment the ranks execute one chain at a
time. So after the prefills, each stream became its own one-row decode chain
and the chains ran one after another. They never merged, because a request
that becomes ready always found a free slot.

The engine now caps decode submissions in flight at the pipeline depth:

| Deployment | Depth |
| --- | --- |
| Fanout TP | 1 |
| Hybrid TP+PP | `stage_count / parallel_group_size` |
| PP | `stage_count` |

A request that becomes ready while a decode wave is running waits for that
wave and joins the next one. Prefill and release submissions are not capped.

With 8 streams, one step then carries 8 rows. The weights are streamed once
for all 8 tokens instead of once per token.

## Large batches: what is and is not sharded

Per rank, with TP16, the caches are split as follows (`layer.cuh`,
`spark_glm5_next_model.h`):

| State | Layout per rank | Bytes |
| --- | --- | ---: |
| KDA recurrent state + conv windows (34 layers) | head-sharded, 4 of 64 heads | about 8.5 MiB per sequence |
| DSA latent KV (11 layers) | replicated, every token on every rank | 1024 B per token per layer |
| DSA indexer keys (11 layers) | replicated, every token on every rank | 514 B per token per layer |

Only the KDA state is 1/16 per rank. The DSA path stores the full latent
(`LmKvStoreKernel`) and the full indexer key on every rank. The attention
heads are split (`attn_heads = 64 / tp`), but MLA shares one latent across
all heads, so every rank also reads the same KV rows. Replicated DSA state
costs 16.5 KiB per token per rank:

| Batch × context | DSA state per rank |
| --- | ---: |
| 64 × 4K | 4.2 GiB |
| 64 × 32K | 33 GiB |
| 256 × 32K | 132 GiB (does not fit) |

It also costs bandwidth. The indexer scores every past token on every rank,
which reads 5.65 KB per token per step. At 32K context that is 185 MB per
row per step: 43 ms per step at B64, more than the weights.

## Batch roofline

Per step, each rank reads the following:

- dense weights: 1.65 GB, independent of B;
- one 1.573 MB slice for each distinct routed expert per routed layer (42
  layers). With uniform routing, B rows touch
  `288 × (1 − (1 − 8/288)^B)` experts: 58 at B8, 240 at B64, 280 at B128,
  288 at B256;
- DSA KV of `min(context, 2048)` tokens per row;
- indexer keys of `context` tokens per row.

At 1K context and 273 GB/s this gives these bounds for one model instance
across all 16 ranks:

| B | Bytes per step | Memory-bound step | Aggregate tok/s | Per stream |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5.6 GB | 21 ms | 390 | 49 |
| 64 | 18.6 GB | 68 ms | 940 | 15 |
| 128 | 22.3 GB | 82 ms | 1560 | 12 |
| 256 | 25.1 GB | 92 ms | 2780 | 11 |

About 3000 tok/s is an aggregate over up to 256 concurrent streams, and it
holds only if the following all hold:

1. collectives are overlapped with compute, or much cheaper than today;
2. contexts are short, or the DSA state is sharded (phase 3);
3. one execution workspace, not one per slot and per driver.

Each stream still sees about 11 tok/s.

The collectives at B256 are 92 per step, each carrying 2 MiB:

| Collective | Per collective | Per step |
| --- | ---: | ---: |
| Direct all-to-all | 15 × payload, 2.8 ms | 260 ms |
| Reduce-scatter + all-gather | 1.875 × payload in 2 rounds, 0.54 ms | 50 ms |

### Measuring it: `bench-glm5-next-batch`

`tools/glm5_next_batch_roofline.cu` links the module's real CUDA kernels
into a single-rank harness. Setup:

- It builds TP16 per-rank tensors with the exact stagepack shapes: FP8
  experts, BF16 dense weights, norms set to 1.0.
- By default it keeps 2 copies of each layer kind. Layer weights repeat
  every 2 layers, but one layer's bytes far exceed the 24 MB L2, so no
  step is served from cache.
- Caches are sized for the largest batch and context requested.

For each batch it then does the following:

1. Runs the full 45-layer decode step: attention, router, grouped experts
   and head. It runs no collectives.
2. Times the steps with CUDA events.
3. Reports the distinct experts that routing selected.

```sh
make bench-glm5-next-batch ROOFLINE_ARGS="--batches 1,8,32,64,128,256 --context 1024"
make bench-glm5-next-batch ROOFLINE_ARGS="--batches 64,256 --context 4096"
```

Output lines:

- `ROOFLINE`: `step_ms`, `compute_tok_s`, `distinct_experts` against
  `uniform_expect`, `step_gb`, `achieved_gbps`, `memory_bound_ms`, and
  `hidden_finite`.
- `ROOFLINE-FLEET`: adds the modelled collective cost (`--round-us`,
  `--nic-gbps`) for direct, RS/AG and RS/AG overlapped with compute.

`achieved_gbps` well below 273 at large B means the grouped expert kernels,
not the memory, are the limit.

### Execution workspace per driver

The module allocated a full device workspace per pipeline slot: hidden,
attention, MLP and head buffers sized for `execution_row_capacity`. Only one
chain runs at a time (`tp_chain_active`), and the chain lock is released
only after the completion has copied its outputs. Slots 1..n now alias slot
0's device workspace and keep their own host staging, events, graphs and MTP
buffers. With 4 slots this saves three workspaces per rank.

### Phases

1. **Measure the batch roofline, and use one execution workspace per
   driver.** Done.
2. **Reduce-scatter + all-gather** for BF16 sums of 12 rows or more.
   Done; see the next section. It needs the coordinated weightd roll.
3. **Context-parallel DSA.** Each rank keeps the KV and indexer keys of
   1/16 of the tokens. The work splits as follows:
   - scoring is local;
   - the global top-2048 is found with two small histogram all-reduces
     (the threshold bucket, then the exact threshold);
   - attention over local selected tokens produces per-head partial
     outputs with their log-sum-exp;
   - the head owners merge the partials.

   This cuts DSA memory and indexer bandwidth by 16×. It adds an
   all-gather of the latent query (64 KiB per row) and a partial exchange
   of the same size per DSA layer.
4. **Remove the host from the collective path.** This means two things:
   - GPU-initiated RDMA;
   - two micro-batches in flight, so that one computes while the other
     communicates.

   A node-level KV pool and scheduler shares cache capacity between
   drivers instead of preallocating per driver.

### Reduce-scatter + all-gather over slice routes

Doorbell word 3 used to be a peer mask. It is now a route with named
bitfields:

| Field | Bits |
| --- | ---: |
| `peer_mask` | 32 |
| `slice_bytes` | 24 |
| `mode` (FULL, SCATTER or GATHER) | 2 |
| `reserved` | 6 |

The modes post as follows:

- **FULL** posts the whole payload to every peer. It is the old behaviour,
  and a bare mask decodes to FULL.
- **SCATTER** posts to each peer `p` only bytes
  `[p * slice, (p + 1) * slice)`.
- **GATHER** posts the local rank's slice to every peer.

In all three modes the slot offsets and the 8-byte tail tag are the same
as before, so the wait protocol does not change.

For one chunk of a BF16 sum:

1. The kernel publishes the chunk with SCATTER.
2. Each rank reduces its own slice across the 16 slots and writes the
   result to the output.
3. Each rank publishes that slice with GATHER into the other ring slot.
4. After the second wait, the kernel copies every peer's slice into the
   output.

Each chunk takes two rounds and puts 1.875× the payload on the wire
instead of 15×. The sum order over peers is the same as in the direct
path, so results are bitwise identical.

The launcher picks RS/AG only if all of the following hold:

- the operation is a BF16 sum;
- the degree is at least 4;
- the payload is at least 49152 elements (12 rows of 4096);
- the local weightd advertises `SLICE_ROUTES` in the lane's wait entries.

The host phase accounting uses the same predicate.

**Compatibility.** The mesh record magic is bumped to `MESH0005`, so a
fleet with mixed weightd versions refuses to wire (`WD-MESH-ABI-MISMATCH`)
instead of mixing the two protocols. That makes this a full weightd +
weightd_warm roll. Drivers built before this change still work against the
new weightd, because they only write FULL routes. The kernel marker is now
`SPARK-TP-MESH-KERNELS-V11-SLICE-ROUTES`.

## Harness results and the per-expert skinny kernel (#1215 follow-up)

These are the harness numbers measured on one Spark with real kernels
(#1215):

| B | Step | Achieved | Memory bound |
| ---: | ---: | ---: | ---: |
| 1 | 20.0 ms | 128 GB/s | 9.4 ms |
| 8 | 39.3 ms | 151 GB/s | 21.8 ms |
| 32 | 203 ms | 61 GB/s | 45.6 ms |
| 256 | 582 ms | 40 GB/s | 86 ms |

Bandwidth efficiency falls by 4× between B8 and B256.

Up to 8 tokens (64 token-expert pairs), the routed experts run on
`LmSkinnyExperts`. That kernel handles each (token, expert) pair
independently, streaming the pair's expert weights once per pair. Above 64
pairs, the experts fell back to the tensor-core grouped GEMM. That GEMM is
tiled for dense row blocks, but at B32 an expert has about 1.5 rows, and at
B256 about 7.

`LmSkinnyGroupedExperts` replaces that fallback up to an average of 16 rows
per expert (576 tokens). How it works:

- One task per (expert, output neuron). It reads the expert's rows from the
  route build (`group_row_offset`, `route_source_token`) and processes them
  in blocks of 4 or 8.
- Each weight row is fetched from DRAM once per expert, not once per pair.
  Later row blocks re-read it from L2.
- Within a warp every task belongs to the same expert, so shuffle
  reductions stay warp-uniform.
- The per-lane accumulation order and the reduction are the same as in
  `LmSkinnyExperts`, so a row's result is bitwise equal to the per-pair
  kernel's result.
- It also covers prefill chunks, up to 576 tokens.

Above 16 rows per expert, the tensor-core GEMM is still used.

`test_skinny_gemv` has two parts:

- **Correctness.** It checks the new kernel against an f64 reference for 9,
  32 and 96 tokens, with one expert forced to 20 or more rows, using real
  `LmRouteBuild` offsets.
- **Timing.** It prints `TIMING grouped_experts` lines comparing the
  kernel with the tensor-core grouped GEMM at 9, 32, 64 and 256 tokens.

The harness also prints `ROOFLINE-PHASES`, a per-step breakdown into:

- KDA attention;
- DSA attention;
- attention post;
- route;
- experts;
- MLP post;
- begin + head.

The next kernel target can therefore be read off directly.

## Decode graphs above 8 rows

Graphs were captured for 1 to 8 rows only. A B16 wave therefore ran the
eager path, which issues about 1950 launches per step. Graph capture now
covers up to 64 rows.

A capture that fails at some row count used to disable graphs for the whole
slot and then terminate the engine. Now it marks only that row count
(`graph_failed_rows`). That row count then runs eager, which is logged once
as `GRAPH-CAPTURE-FAILED rows=N; this row count runs eager from now on`, and
every other row count keeps its graph.

## B16 and larger

The module's sequence cap is the compile-time batch bucket
(`SPARK_BATCH_BUCKET`). The Makefile builds every bucket from 1 to 1024
(`make -C modules/glm5_next_resident_decode_stage variants` /
`publish_variants`), and each is published under its own module ID:
`spark.glm5_next.resident_decode_stage.bf16.expert_fp8.h4096.l45.kda34.e288.k8.b<N>.v2`.

The `active=8` clamp came from deploying the b8 variant. A B16 profile needs
the b16 (or b32) variant; there is no code change.

## Wave width in the concurrency sweep

In the 8-stream sweep, every stream prefills 176 tokens through 8-row
chunks: about 22 chunks per stream, 176 prefill submissions in total. They
alternate with decode waves, which only take requests whose prefill has
finished. With 32 output tokens per stream, early streams finish decoding
before late streams finish prefill. So waves stay at 1 or 2 rows, and the
sweep time is mostly prefill.

To measure decode waves, run many more output tokens per stream than prompt
chunks. To make prefill itself faster, raise the prefill chunk size (see
below).

The prefill chunk is `execution_row_capacity` in the adapter configuration,
together with the engine's `max_prefill_rows_per_submission`. It is
independent of the sequence bucket: the module caps it at 65536, and the
mesh at `SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS` (128). With 64- or 128-row
chunks:

- a 176-token prompt takes 2–3 prefill submissions instead of 22;
- the routed experts use the per-expert kernel;
- the dense layers use the tensor-core GEMM.

The dense layers at more than 8 rows produce slightly different numerics
from the 1-row skinny path, so prefill logits are not bitwise equal to B1.

## Phase 3A: indexer context parallelism (exact)

Above 2048 tokens, every DSA layer used to score all of a row's context
pools on every rank:

- one pool is 4 tokens, with a 257-wide packed key per token;
- that is 5.65 KB per token per row per step across the 11 layers;
- at 32K context this is 185 MB per row per step, 43 ms at B64.

With `"dsa_index_context_parallel": 1` in the glm adapter configuration, the
work is split as follows:

- **Ownership.** A pool belongs to rank `page % tp` (pages of 64 tokens, 16
  pools each; `spark_glm5_next_index_cp.h`). Each rank scores only its own
  pools, which is 1/16 of the indexer reads.
- **Gather.** The local score rows are all-gathered on the main collective:
  op 0, `active_sequence_count = ceil(rows * local_stride / 2048)`. A permute
  kernel then puts them back in global pool order.
- **Selection.** The existing top-k, gather and expand kernels run unchanged
  on the full score array, on every rank.
- **Exactness.** Each pool is scored by exactly one rank, with the same code
  as the replicated path. The selection is therefore bitwise equal to
  replicated mode.
- **Cost.** One extra collective per DSA layer, and only when the wave's
  context exceeds 2048. It carries about `context` bytes per row (4 bytes
  per pool, received from 15 peers), rounded up to 8 KiB per rank.

The chain gets one new stage (`GATHER_INDEX`, appended to the enum so the
numeric stage values in existing logs do not change), in both the eager
chain and graph recording. The attention launch is split into
`AttentionScore` (HC site, norms, q_a, index projections and store, local
pool scores) and `AttentionSelect` (permute, top-k, q_b onward).

Limits:

- It requires `tp_degree >= 2`.
- The local scores must fit the gather at `execution_row_capacity`. At TP16
  that holds up to 128K `max_sequence_positions`. Beyond that, module init
  refuses with a message.
- Storage is unchanged: every rank still stores every index key and the
  full latent KV. Only the indexer reads are split.

Tests:

- `tests/test_glm5_next_index_cp_math.c` (host): ownership covers every pool
  exactly once for degrees 1–16; gather sizing.
- `tests/host_cuda/glm_index_kv_host.cu` (host, runs the real kernels via
  the host CUDA shim): context-parallel pool scores, gathered and permuted,
  are `memcmp`-equal to the replicated scores for degrees 2–16.
- `make test-glm5-next-index-cp` (GPU): the same comparison on the device,
  at 2049 and at 9001/4100/20000 tokens, for degrees 2, 3 and 16.

The harness takes `--index-cp 16` and adds the index read to `step_gb`.
It simulates the gather with device copies, so its timing covers the
compute side only.

## Phase 3B: sharding the latent KV itself (design, not implemented)

The latent KV (1024 B per token per DSA layer) is still replicated. Storing
1/16 per rank would mean computing attention where the KV lives:

1. Each rank computes partial softmax states for all 64 heads over its own
   selected positions.
2. The partials `(m, l, o[512])` move to the head owners.
3. The head owners merge them (`LmLatentAttentionDecodeSplitCombineKernel`
   already does exactly this, with partitions = ranks).

The problem is the traffic, per row per DSA layer:

| Transfer | Direction | Size |
| --- | --- | ---: |
| All-gather of the latent query | into each rank | about 64 KB |
| All-to-all of the partials | out of each rank | about 62 KB |

A 256 KiB mesh slot carries only 16 KiB per peer per SCATTER round, so at
B64 the all-to-all alone needs about 16 rounds per layer. That is about
18 ms per step at 100 µs per round, which is worse than the memory it
saves.

Two ways around this:

- **A 2D split.** For example 4 head groups × 4 context shards:
  - KV is stored at 1/4 per rank;
  - traffic is about 24 KB per row per DSA layer, exchanged within a group
    of 4.
- **Phase 4 first.** GPU-initiated RDMA with per-peer buffers sized to the
  exchange.

Until then, capacity for long contexts comes from the existing KV arena. It
evicts cold pages to the backing store (`resident_block_capacity <
logical_block_count`).

## Next steps, ordered by expected gain

1. Remeasure the ladder tail with the lock-free wiring scan, and measure
   batched decode at 1, 2, 4 and 8 concurrent streams.
2. Remove the CPU relay from the critical path: GPU-initiated RDMA
   (IBGDA-style), with the GPU writing work requests and ringing the NIC
   doorbell. A 16-rank 8 KiB all-reduce should then cost tens of
   microseconds.
3. Pair-link hierarchical all-reduce for prefill and B >= 16.
4. Fuse RMSNorm into the consuming GEMV and batch GEMVs that share an input,
   to cut the 1950 launches per token.
