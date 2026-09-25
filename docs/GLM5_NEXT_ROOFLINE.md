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
