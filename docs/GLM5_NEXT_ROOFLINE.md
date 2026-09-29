# GLM 5.3 Flash TP16 B1 roofline

The target is 80% of the memory roofline for one decode token per rank.
Output quality and acceptance gates are in
[GLM_PERFORMANCE_GATES.md](GLM_PERFORMANCE_GATES.md). Expert residency is in
[GLM_LAZY_DRIVER_INTEGRATION.md](GLM_LAZY_DRIVER_INTEGRATION.md).

## Where it stands (2026-09-28)

| Item | Value | Source |
| --- | ---: | --- |
| B1 decode, TP16 fleet, no speculation | 36 tok/s | lead-dev fleet measurement, 2026-09-28 |
| Time per token inside an 8-step graph chain | 23-24.5 ms | same measurement |
| 16-rank 8 KiB all-reduce p50 | 167 µs | "Measured after PR #1208" below |
| Memory floor | 7.99 ms (125 tok/s) | "Bytes per rank per token" below |
| 80% target | 10.0 ms (100 tok/s) | same |
| Best public 4-Spark GLM B1 result, no speculation | 23.2 tok/s | lead dev, 2026-09-28 |
| TP16 goal, 3-3.5x that result | 70-81 tok/s, 12.3-14.4 ms | arithmetic |
| COMPSEC-17, thinking off | 14/17 | [GLM_PERFORMANCE_GATES.md](GLM_PERFORMANCE_GATES.md) |

The measured engines were built from `dd3526b`, the head of PR #1243. The
x86 API on the rtx5090 was built from the same commit. The COMPSEC receipt
(`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md`)
lists the binary hashes. Main has moved past that build (`git log dd3526b..origin/main`
includes the i21-i33 driver fixes, #1255 and #1259), and nothing after
`dd3526b` has a fleet measurement yet. Quote a build identity with every
number.

The 36 tok/s was measured with this serving environment, which the
fleet-agent drop-in `20-serving.conf` sets on every Spark:

| Setting | Effect |
| --- | --- |
| `G5_GRAPH_PATH=1` | The agent passes it as `SPARK_GLM5_NEXT_GRAPH_PATH`, which is mandatory. The module fails with `INVALID_ARGUMENT` unless it is `0` or `1`, so deleting the drop-in stops the engines. Use `G5_GRAPH_PATH=0` for eager runs. |
| `G5_PIN_EXPERTS=1` | Passed as `SPARK_GLM5_NEXT_PIN_EXPERTS`. Pins all 12096 routed experts per rank. Graphs and linear chains run only with it. |
| `SPARK_TP_WAIT_MODE=hardware` | The collectives use hardware waits and chunked direct rounds. Unset or `spin` makes the GPU poll peer tails itself. |
| `G5_API_DISABLED=1` | Inert since #1261: the agent never starts an API. The API is `g53-api` on the rtx5090, port 8433. |
| `G5_WARMUP=0` | The agent's warmup hook does not run. |

Full pinning contradicts bounded residency (I29). See
[GLM_LAZY_DRIVER_INTEGRATION.md](GLM_LAZY_DRIVER_INTEGRATION.md) for the cost
and the redesign.

Collective latency is the dominant remaining cost (lead dev, 2026-09-28). A
chain step at 23-24.5 ms is 13-14.5 ms over the 10.0 ms target (arithmetic).
About 91 collective rounds per token (see "Linear eager chains") at the
167 µs p50 come to about 15.2 ms (arithmetic, assuming no overlap).

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

The CUPTI trace in [TP16_HARDWARE_PROFILE_20260922.md](archive/TP16_HARDWARE_PROFILE_20260922.md)
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

The HC site cluster synchronizes once. Each CTA arrives on the cluster
barrier at kernel start and waits for it after its K-slice dot, so every CTA
has started before any remote write. Each CTA then pushes its 25 partials
(24 mixes and the sum of squares) into every other CTA's shared memory. After
one `cluster.sync()` every CTA sums the eight partials in rank order and
computes the gates itself. The arithmetic and summation order match the
earlier kernel, which gathered the partials into CTA 0 and broadcast `pre`
with two more cluster syncs. The result is bitwise equal
(`test_glm5_next_hc_mix`, `ROOFLINE-HASH`). Only CTA 0 writes `mixes`, `pre`
and `post` to global memory.
Bench, sparkf, 5 interleaved rounds: B1 12.912 -> 12.831 ms, B8 31.569 -> 31.531 ms
(`qualification/glm5next/performance/glmflash_b1_20260929/ab/ab_hc1_one_sync.txt`).

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

## Relay timing per hop and per peer

The device counters split a token's collective time into source wait, peer
wait, copy and combine. At TP16, B1 the peer wait dominates at about 27 ms
per token. It has three possible causes:

- a peer that publishes late (a straggler);
- the relay hops on either node;
- the NIC path.

To tell them apart, weightd timestamps every round in its own monotonic
clock and prints a window every 10 s next to `WD-MESH-STATS`. No cross-node
clock sync is needed.

```
WD-MESH-TIMING posts=N post_us=p50/p99 ship_us=p50/p99 credits=N credit_us=p50/p99 gates=N gate_us=p50/p99 gate_ms=T self=N starts=N start_us=p50/p99 start_ms=T start_self=N worst_us=W worst_tag=E:R worst_closer=C peers=R:p50/p99/last/start_last/excess_us,...
```

| Field | Measures |
| --- | --- |
| `post_us` | From weightd first seeing the local GPU's doorbell to the RDMA writes being posted: the send-side relay reaction. |
| `ship_us` | From the post to the last write completion. That is the NIC round trip, and it also frees the source slot. |
| `credit_us` | From weightd seeing a hardware-wait source-credit gate (may the GPU reuse its slot?) to releasing it. A p50 near 1 means the credit was usually already there, so the handshake itself was the cost. |
| `gate_us`, `gate_ms` | From weightd seeing a hardware-wait peer gate to releasing it; `gate_ms` is the window's total, so `gate_ms / waves` is the wave time this rank's GPU spent waiting for peers. |
| `self` | Peer gates whose tails had all arrived before weightd first saw the gate. This rank reached its own gate last, so it closed the gate itself. |
| `starts`, `start_us`, `start_ms`, `start_self` | The same for the first round of each chain on a band (sequence 1 of the chain's epoch). On the hc band that round is the wave's embedding reduce, so it absorbs any skew in when ranks start the wave. |
| `worst_us`, `worst_tag`, `worst_closer` | The window's longest gate: its wait, its epoch and round, and the rank that closed it (the local rank for a self-closed gate). The epoch matches `worst_epochs` in `G5N-WAVE-TIMING`. |
| `peers` | For each physical sender: the arrival lag of its tail after this rank's own publish (p50 and p99), how many gates and chain-start gates its tail closed, and its excess: for each gate it closed, the time between the previous tail and its own, summed in µs. The excess is the wait that sender alone cost. weightd only sees a tail when it polls an open gate, so a lag is never shorter than the gap between this rank's publish and weightd first seeing its gate, and tails that land in the same poll count toward the higher band rank and add no excess. |

Values are upper bounds of power-of-two microsecond buckets. Peer lags need
hardware waits (`SPARK_TP_WAIT_MODE=hardware`); in spin mode the GPU polls
the tails itself and only `post_us` and `ship_us` are reported.

`python3 tools/mesh_timing_report.py 0=rank0.log 1=rank1.log ...` combines all
ranks' lines. It prints each rank's hop medians and mean gate time per
window, a receiver × sender matrix of median lag, each rank's share of all
gates and of chain-start gates closed, the excess wait charged to each rank,
and the ten worst gates. A rank closes a gate either as the last sender on
another rank or through its own `self` count, so the shares add up to 100%.

How to read it:

| Pattern | Conclusion | Next step |
| --- | --- | --- |
| One or two ranks close most gates, and their column shows much larger lag in every receiver's row | Stragglers | Look at those nodes: CPU placement of the doorbell thread, clocks and thermals, the NIC link and switch port |
| A rank closes many gates but its excess is small | It is last by microseconds; the skew costs nothing | Nothing on that rank |
| Most of `gate_ms` is `start_ms`, and one rank's excess is mostly start gates | That rank starts waves late | `G5N-WAVE-TIMING` on that rank: `wait_us`, `key_us`, `setup_us` |
| Lags similar for every sender, `post_us` tens of µs | The send-side relay is slow | Kernel-posted work requests |
| Lags similar for every sender, `ship_us` well above the wire time of one round's bytes at about 100 Gb/s useful (120 KiB takes about 10 µs) | The NIC or switch path | Pair links, fewer bytes per round |
| Lags similar and small, but the device peer wait is still large | The receive-side handoff costs the time (gate release, then GPU front end) | GPU waits directly on the peer tails |

`SPARK_WEIGHTD_MESH_DOORBELL_CPU=<cpu>` pins weightd's doorbell thread to
that one CPU instead of spreading it over all highest-capacity CPUs, so a
node's residentd threads can be kept off it (for example with `taskset`).
weightd refuses to start on a value that is not a CPU of the node.

### Measured after PR #1224 (fleet, 2026-09-26)

16 ranks, 8-stream load, hardware waits:

| Hop | p50 |
| --- | ---: |
| Send-side relay (doorbell to post) | 1 µs |
| NIC round trip (ship) | 128-256 µs |
| Credit gate | 1 µs |
| Peer gate release | 128-256 µs, p99 up to 32 ms |
| Arrival lag, every peer pair | 64-128 µs |

No rank's column stood out in the lag matrix, so there is no straggler node
or bad port at the median. The shares of gates closed were skewed: rank 1
closed 22.9%, rank 15 14%, ranks 12-14 7-9%, every other rank 2-5%. About
92 rounds at 128 µs is 12 ms of each 72 ms wave. Where the rest of the
35 ms outside compute goes is what the gate classes above and the wave
timeline below measure.

## Wave timeline per rank

Each rank's glm5_next module prints one line every 10 s from its completion
worker:

```
G5N-WAVE-TIMING rank=R waves=N rows=N steps=N prefill=N graph=N eager=N linear=N graph_path=P retries=N busy=C/S/L/A/O captures=N capture_ms=T idle_us=p50/p99 wait_us=p50/p99 key_us=p50/p99 setup_us=p50/p99 run_us=p50/p99 post_us=p50/p99 idle_ms=T wait_ms=T key_ms=T setup_ms=T run_ms=T post_ms=T graph_run_ms=T eager_run_ms=T linear_run_ms=T linear_walk_ms=T decode_wait_ms=T source_wait_ms=T peer_wait_ms=T copy_ms=T combine_ms=T worst_ms=W worst_request=Q worst_epochs=M/H worst_us=idle/wait/key/setup/run/post
```

The six intervals follow one frame, each starting where the previous one
ends:

| Interval | From | To |
| --- | --- | --- |
| `idle` | the previous frame's completion handed to residentd | the module first seeing this frame |
| `wait` | first sight | the chain claim. The `BUSY` retries happen here, while another chain holds the rank. |
| `key` | the chain claim | both bands' chain keys set: the activity handshakes and, on ranks other than 0, the wait for rank 0's epoch broadcast |
| `setup` | keys set | the first launch: the graph launch on the graph path, the first kernel on the eager path. Graph captures fall here. |
| `run` | the first launch | the host finishing the chain; for a linear chain, the completion worker finding the stream drained |
| `post` | the chain finished | the completion handed to residentd: host callback, completion worker, end of the collective chain |

`run` is the wave's GPU time as the host sees it, over every step of a
resident decode chain. The graph path waits for the whole graph on the
residentd thread before it finishes the chain, and the chain state machine
synchronizes the stream at every collective round, so on that path `run` also
holds the host work between rounds. A linear chain enqueues the whole wave and
returns, so its `run` ends when the completion worker finds the stream
drained, one host callback after the GPU finished.

The other fields:

| Field | Meaning |
| --- | --- |
| `steps` | decode steps the frames ran: one per frame, K for a resident decode chain of K steps |
| `prefill` | frames that were prefill chunks; the rest are decode waves |
| `graph`, `eager` | frames run as one CUDA graph, and the rest |
| `linear` | eager frames enqueued as one linear chain; the other eager frames ran the chain state machine |
| `graph_path` | 0: off by configuration; 1: on; 2: degraded, requested but turned off by a `GRAPH-FAILED`, so the rank runs eager until restart |
| `busy` | `BUSY` retries by what blocked the claim: another chain holds the rank, the stream still had work, the frame's pipeline slot was taken, one of its sequence lanes was taken, or something else (a retained lease, a cache-frame claim) |
| `captures`, `capture_ms` | graph captures and their time, inside `setup` |
| `graph_run_ms`, `eager_run_ms` | `run` split by path |
| `linear_run_ms` | the part of `eager_run_ms` that linear chains ran |
| `linear_walk_ms` | the part of `linear_run_ms` the host spent enqueueing the linear steps |
| `decode_wait_ms` | `wait` of decode waves only: the time they queue behind other chains, such as prefill chunks |

`wait` overlaps the `run` and `post` of the chain the frame queued behind, so
when frames queue, the intervals of all frames add up to more than the wall
time. The `_ms` fields are window totals; divide by `waves` for per-frame
means. The collective fields are the device-measured sums from
`COLLECTIVE-GPU-TIME`; `run` minus them is compute plus any host gaps inside
the run. `worst_*` names the window's slowest frame (first sight to
completion) by request id, which every rank shares, and by the main and hc
chain epochs, which match `worst_tag` in `WD-MESH-TIMING`.

`python3 tools/wave_timeline_report.py 0=residentd0.log ...` prints each
rank's per-frame budget with the busy reasons and path split, and the slowest
frames across ranks. Its `eager_run` column counts only the chain state
machine, `linear_run` the linear chains; lines printed before linear chains
existed count every non-graph frame as eager. `steps/decode` is the mean
number of steps per decode wave, 1.0 on lines printed before resident decode
chains. `linear_walk` is the host's share of `linear_run`; the difference is
the GPU still running after the host finished enqueueing.

### Chain and replay lines

Three more lines time single chains and waits. The completion worker prints
one `CHAIN-TIME` line per finished chain:

```
CHAIN-TIME slot=S path=P steps=N status=C total_ms=T walk_ms=T collective_host_submit_ms=T collective_host_submissions=N stage_ms=T/T/T/T/T/T/T/T
```

| Field | Meaning |
| --- | --- |
| `path` | how the chain ran: `graph`, `linear`, or `eager` for the chain state machine |
| `steps` | decode steps the frame ran; 1 unless it was a resident decode chain |
| `total_ms` | the chain from its start to the completion worker, over all of its steps |
| `walk_ms` | the part of `total_ms` the host spent enqueueing linear steps. A small `walk_ms` means the chain waited for the GPU and its peers. A full CUDA launch queue also blocks the walk, so a `walk_ms` close to `total_ms` shows a launch bound only if the GPU finished soon after the walk did. |
| `collective_host_submit_ms`, `collective_host_submissions` | host work submitting collective rounds, capture included. This is not the allreduce time of a graph replay; a linear chain checks its rounds once at the end, so its figure is launch work only. |
| `stage_ms` | host time the chain state machine spent in each of its eight stages (`chain_stage_ns`), reset after every line |

The graph path prints `GRAPH-REPLAY-TIME slot=S wall_ns=N stream_status=C`
after it waits for a replay. `wall_ns` includes compute and collective waits;
`stream_status` is the CUDA query result, and the sticky collective-error
check follows separately. `tp_device_collective.c` prints
`COLLECTIVE-WAIT-END rank=R slot_idx=I elapsed_since_previous_wait_end_us=N gpu_timestamp_ns=N`
from its arrival ring; the elapsed figure includes the computation between
the two waits. These interpretations come from the PR #1077 write-up
([archive/PR1077_SERVING_RELIABILITY.md](archive/PR1077_SERVING_RELIABILITY.md)).

### KDA state restore and capture

A prefix hit restores the sequence's KDA recurrent state and convolution
windows from the state store, and every publishing frame captures them.
The completion worker prints one line per 10 s window in which either
happened:

```
G5N-KDA-TIMING rank=R restores=N restore_bytes=B restore_us=T captures=N capture_bytes=B capture_us=T
```

`restore_us` spans the store read and the host-to-device copy of each
restore. `capture_us` spans the device-to-host copy and the store write of
each capture. Both are window totals on the host clock, and captures from
zero-row publish frames are included. Compare `restore_us` per restore with
the `key` and `setup` intervals of `G5N-WAVE-TIMING` to see what a hit adds
to its first frame.

The API prints one `engine_measurements` JSON line whenever a request ends
or a wave is rejected. It carries the engine's cumulative first-token count,
queue, prefill and TTFT totals, prefix hits and misses, stale-prefix
recomputes and rejected waves by status. Each `request_measurements` line
carries `first_dispatch_ns` and `stale_prefix_recomputes`, so a request's
queue time is `first_dispatch_ns - accepted_ns` and its prefill time runs
from `first_dispatch_ns` to its first token's timestamp.

### Iteration 15's `pre` held the GPU time

Iteration 15 printed `pre` (first sight to "the last launch") and `gpu` (the
last launch to the host callback). Both stamps were wrong. The launch stamp
was taken in `SparkGlm5NextFinishChain`, which the graph path reaches only
after `SparkGlm5NextGraphStep` has waited for the whole graph, and the eager
path only after the last round's stream synchronization. So `pre` held the
wave's GPU time and `gpu` only the callback latency. The 67 ms of `pre` per
wave measured after PR #1226 is wave execution plus queueing, not host
polling.

The retries were real, but they came from residentd, not the module. While
the head of the committed FIFO was `BUSY`, a later committed route in a
lower route slot got a no-op submit that counted as an adapter operation and
wrote the wake pipe. `poll` then never slept, and residentd retried the head
back to back: 18,000 retries per second against the 1,000 per second that
the 1 ms poll timeout allows. On adapters without asynchronous completion
the same no-op used up the one-operation budget of the pass, so the head
could not be submitted at all while that route sat in a lower slot.
`SparkModelResidentdSubmitAdapter` now reports whether it called the
adapter, and a route that is not the head is skipped without an operation
or a wake.

## Linear eager chains

After iteration 16 the fleet ran 94.1% of chains as graph replays, at 45.2 ms
each, and 5.9% eager, at 153.7 ms each. 54.9 ms of every eager chain was the
host blocked in collective rounds (`collective_host_submit_ms` in
`CHAIN-TIME`): the chain state machine read the round-control block back and
synchronized the stream after each of about 91 rounds, and every routed layer
added a routing readback, a weightd lease and a stream synchronization on
release. Eager chains are the prefill chunks and the decode waves the graph
gate turns away. Decode waves queued behind them for about 15 ms per wave.

When the stage's routed experts are all pinned, which the graph path already
requires, an eager chain now runs as a linear chain
(`SparkGlm5NextLinearChain`). The host enqueues the same launches the decode
graph records, from the same walk (`SparkGlm5NextWalkChain`, which
`SparkGlm5NextGraphRecord` also captures), in one pass:

- every collective is a stream-ordered hardware-wait round with no completion
  callback, no readback and no stream synchronization;
- routed layers address the pinned arena, like the graph, instead of taking a
  lease per layer. Both give the kernels the same arena base and the same
  per-layer offsets, so the weights read are the same;
- the host then hands the frame to the completion worker like any other
  chain, and residentd is free while the GPU runs it.

The rounds are checked once per chain. Every hardware-wait round adds one to
the device's `rounds_done` word unless it failed, and the first failure sets
the sticky `error_word`, after which later rounds skip their work. The
collective counts the rounds it deferred. After the completion worker has
drained the stream, `SparkTpDeviceCollectiveVerifyDeferred` reads each band's
round-control block once. A set `error_word` or a short `rounds_done` prints
`MESH-DEFERRED-ROUNDS-FAILED`; the module then broadcasts a cancel on both
bands and fails the frame, which rolls back its cache lanes like any other
failed frame.

A chain runs the state machine instead when experts are not pinned, with MTP,
speculative verify or the T1 trace, with `SPARK_GLM5_NEXT_GRAPH_RECORD_OPS`
set, or when either collective lacks hardware-wait rounds
(`SparkGlm5NextLinearEligible`).

To check a deployment: `linear` in `G5N-WAVE-TIMING` should cover the frames
that `eager` used to, `CHAIN-TIME` lines carry `path=linear` with
`collective_host_submit_ms` reduced to launch time, and `decode_wait_ms` per
decode wave falls by what the eager chains' host work cost. Greedy tokens
must match the state machine's. With `SPARK_GLM5_NEXT_GRAPH_PATH=0`,
`SPARK_GLM5_NEXT_PIN_EXPERTS=1` runs every chain linear and
`SPARK_GLM5_NEXT_PIN_EXPERTS=0` runs every chain through the state machine,
so the same prompts under both settings compare the two. What stays: a decode
wave that arrives behind a prefill chunk still waits for the chunk's GPU
time. Capturing prefill graphs, or ordering decode ahead of prefill, removes
that.

## Resident decode chains

After iteration 17, each of 8 streams got a token about every 62 ms, while a
graph chain ran for 46.3 ms. Most of the difference is work outside the chain
run that every frame pays: residentd and engine turns, the chain keys and
epoch, the host callbacks and the publish frames. The rest is queueing behind
prefill chunks. A decode frame now runs up to 8 decode steps
(`tokens_per_sequence`), so the per-frame work is paid once per chain.

How a chain runs:

- The adapter advertises `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN`,
  and the engine asks for K steps: the smallest of 8,
  `max_output_token_count` divided by the lane count, the budget left, the
  context left, and the positions left in each lane's 64-token block.
- `SparkGlm5NextValidateChain` accepts K > 1 only on a decode frame whose
  stage owns both the embedding and the head (every TP16 rank does). The frame
  must also have no state capture and no cache prefix or publish lane, and
  every row's K positions must fit in one KV page and under
  `max_sequence_positions`.
- Each step takes whichever path its `BEGIN` picks (graph replay, linear
  chain or state machine), at that step's positions. `BuildWave` recomputes
  the context bound every step, so a replay that would outgrow its captured
  bound recaptures, as it would between frames.
- At the end of a step, `SparkGlm5NextFinishChain` starts the next one.
  `SparkGlm5NextSettleStep` releases the step's expert lease, drains the
  stream (bounded at 35 s) and verifies both bands' deferred rounds. Then
  `SparkGlm5NextFeedStep` stores each row's token in the chain buffer,
  advances the row's position, and makes the token the row's next input.
- After the last step, `SparkGlm5NextGatherSteps` lays the tokens out
  lane-major (row × K + step). The frame completes with `tokens_per_sequence`
  = K, keeps K − 1 extra cache tokens per lane (as MTP's accepted drafts do),
  and advances each lane's next position by K − 1.
- The adapter sizes the output buffer and bounds the completion burst per
  frame with `SparkGlm5NextServingBurstLimit`: K for a chain, MTP depth + 1
  for a lone MTP sequence, 1 otherwise. It copies each lane's K tokens from
  that lane's row.
- Sampling noise is keyed by seed, position and token, so a sampled step
  draws what a single-step frame at that position would.

Cache publication: the engine used to queue a zero-row publish frame after
every chain. It now queues one only when the chain ended at a block boundary
or finished the request (`SparkModelBatchChainCheckpoint`), so a chain inside
a block costs one frame, not two. A chain cut short by EOS still publishes
nothing, because the resident state has run past the emitted tokens.

What chains leave in place, each tracked in [TECHDEBT](../TECHDEBT.md):

- the graph path waits for every replay on residentd's thread;
- a linear chain waits for each step's tokens before it enqueues the next
  step;
- chain frames run no MTP draft;
- the block cap shortens chains as batches grow. With random lane offsets
  the mean chain is 8 steps at 1 lane, 7.2 at 2, 6.5 at 4 and 5.4 at 8.

A chain also holds the rank for all K steps, so a prefill chunk that arrives
mid-chain waits for the whole chain.

To check a deployment: `steps/decode` in `tools/wave_timeline_report.py` is
the mean chain length of decode waves, and `CHAIN-TIME` lines carry
`steps=K`. At one stream, greedy tokens must match the single-step build's.

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
  round each. The binomial tree is no longer used in hardware mode. In spin
  mode, a single-sequence wave larger than one slot still took the host round
  and failed with `CAPACITY_EXCEEDED`. #1255 (`cf64e90`) sends such waves
  through chunked device rounds (`SparkTpDeviceCollectiveHostRound`). The
  deployed `dd3526b` engines predate #1255, so the 2026-09-28 API ran with
  `SPARK_MODEL_API_MAX_PREFILL_ROWS=8` (COMPSEC receipt).
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
| KDA recurrent state + conv windows (34 layers) | head-sharded, 4 of 64 heads | 8.9 MiB per sequence |
| DSA latent KV (11 layers) | replicated, every token on every rank | 1024 B per token per layer |
| DSA indexer keys (11 layers) | replicated, every token on every rank | 514 B per token per layer |

The KDA figure is arithmetic from `spark_glm5_next_model.h`. Per layer, the
state is 64 heads × 128 × 128 × 4 B = 4 MiB, and the Q, K and V convolution
windows are 3 × 64 heads × 128 × 4 history entries × 2 B = 192 KiB. The
module allocates both per rank divided by the TP degree
(`kda_state_layer_stride_bytes`, `kda_window_layer_stride_bytes`). Per
sequence, over 34 layers:

| Topology | State | Windows | Total |
| --- | ---: | ---: | ---: |
| TP1 | 136 MiB | 6.375 MiB | 142.375 MiB |
| TP4 | 34 MiB | 1.59375 MiB | 35.59375 MiB |
| TP16 | 8.5 MiB | 0.3984375 MiB | 8.8984375 MiB |

This is also what a KDA checkpoint must carry per sequence. The
`check_rank_state` case in `tests/test_glm5_next_stage_context.py` runs the
real allocator at TP1, TP4 and TP16 and checks the strides, pool sizes and
checkpoint page bytes.

Only the KDA state is 1/16 per rank. The DSA path stores the full latent
(`LmKvStoreKernel`) and the full indexer key on every rank. This violates
the owner's 1/16 requirement; the sharded kernels are in Phase 3B below,
and the serving path does not use them yet. The attention
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
- `ROOFLINE-GRAPH` (with `--graph 1`): the step is captured once as a CUDA
  graph and replayed `--iterations` times, as the production graph path
  runs it; prints the node count and the median, minimum, maximum and mean
  replay time. `--route-readback 0` builds the step the way resident chains
  do, without the host copy of the expert group offsets.

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

## One answer on every path (#1230)

In #1230's check, three cold runs of one prompt agreed with each other, three
runs that hit the prefix cache agreed with each other but not with the cold
runs, and three cold runs after another restart disagreed with each other.
Head argmax already breaks ties by the lowest token id, and collectives add in
rank order, so the flips come from logits that differ in their last bits
between paths. Which path a wave takes depends on the session:

- the first waves after a restart run eager until the experts are warm;
- a rank that loses its graph path (`graph_path=2`) runs eager until restart;
- a request runs on execution slot `request_id % 4`, whose graphs were
  captured at whatever contexts that slot saw first.

Three choices depended on the path instead of the context:

- **Split-KV.** A graph was captured with the context bound `context + 256`,
  and attention split its KV walk once that bound reached
  `decode_split_context_threshold` (64 on the fleet). An eager wave at
  context 20 did not split; a graph replay of the same wave did.
- **DSA selection.** A graph captured at context 1,900 had bound 2,156. It ran
  DSA selection, which starts above 2,048, for every context it later
  replayed, where an eager wave attends to every position.
- **The top-k.** Selection kept whole 8-bit histogram buckets and filled its
  512 slots in atomic order, so which pools it kept, and their order, changed
  from run to run.

Two bugs changed which positions were attended above 2,048 tokens:

- Attention read the selected positions with a row stride of 2,048, while the
  expansion wrote 2,051 per row. The tail was never attended, and row `r`
  read its list `3r` entries early, so later rows attended some of the
  previous row's selections and missed some of their own.
- A pool only partly inside a row's context got a score, so it could displace
  a complete pool. The expansion then emitted its positions, which the tail
  emits too, so they were attended twice. This happened on every graph replay
  (whose pool count comes from the bound) and for every shorter row in a
  multi-row wave.

What changed:

- **Graphs by regime.** Each slot keeps one graph per row count per regime:
  unsplit (context below the split threshold), split (up to 2,048) and
  selected (above 2,048). A bound stays inside its regime: `threshold - 1`
  for unsplit, 2,048 for split, `context + 256` for selected, capped at
  `max_sequence_positions`. Eager and graph waves decide split-KV with the
  same function, `LmLatentAttentionContextSplits`. The capture log reads
  `GRAPH-CAPTURE-OK rows=N regime=R bound=B`, with R 0, 1 or 2.
- **Exact top-k.** `LmTopkExactKernel` (`topk_exact.cuh`) selects the exact
  top 512 pools, breaks ties by the lowest pool index and writes them in index
  order. It replaces the histogram and gather kernels in glm5_next and in
  GLM 5.2.
- **Complete pools.** A pool scores only when all four of its tokens are
  inside the row's context. The expansion emits only complete pools, then the
  tail (context mod 4 tokens) once.
- **Stride.** Attention and the stage sideband read 2,051 positions per row.

The tail and stride fixes change output above 2,048 tokens. At or below 2,048,
output changes only where a graph replay used to split or select where an
eager wave did not.

Tests:

- `tests/test_topk_exact_host.py` runs the top-k on 64 host threads against a
  CPU reference: ties at the threshold, `-inf` padding, fewer candidates than
  k, and 8,192 pools. Four runs of each case must be identical.
- `tests/test_glm5_next_graph_regime.py` replays every context up to 4K and
  32K, and 400 random request histories, for eight split thresholds. Every
  replayed graph's bound covers the context and makes the same split and
  selection choices as eager. The previous policy fails it.
- `tests/test_glm5_next_index_kv.py` runs pool scores, top-k and expansion on
  host. The selection is the same at the eager pool count and at the graph
  bound, and a short row in a long wave attends each of its positions once.
  The previous pool scores and expansion each fail it.

Still open (TECHDEBT, Dynamic batching): a row's attention partitions depend
on the other rows in its wave, and no GPU test yet compares a graph replay's
logits with an eager wave's.

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
- **Selection.** The top-k and expand kernels run on the full score array,
  on every rank.
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

## Phase 3B: sharding the latent KV itself (kernels implemented, not serving)

Owner requirement (2026-09-28): at TP16 each node holds 1/16 of the KV
cache. Replicating the latent KV or the indexer keys violates it. The
shared kernels that store and attend over 1/16 per rank are done and
tested (#1334). The transport is a draft that does not yet pass its probe
(#1335). glm5_next and k3 do not use either yet, so production still
replicates.

### Layout

| Item | Rule |
| --- | --- |
| Owner of position `p` | `(p / grain) % degree` (`include/sparkpipe/spark_kv_shard.h`) |
| Latent KV grain | 1: each 64-token page puts 4 tokens on each of 16 ranks |
| Indexer key grain | 4: one DSA pool per rank per page, so pool scoring stays local |
| Pages | same page table, page ids and page count on every rank; each rank's page is `page_bytes / degree` |
| Refused | a degree that does not divide `page_slots / grain`, degree > 16, a rank outside the degree (`SparkKvShardValid`) |

Keeping the page identity global keeps prefix sharing, JIT-KV snapshot and
restore, and eviction page-granular and identical on all ranks. Each rank
saves and restores its own slice of every page.

### Attention

1. Each rank stores only the positions it owns (`LmKvShardStoreKernel`).
   KV_A is replicated, so every rank already has every row's latent.
2. The latent queries are all-gathered: 4 heads × 1 KiB per rank per row.
3. Each rank computes `(m, l, o[512])` for all 64 heads over its own keys
   (`LmLatentShardPartialKernel`). A row's keys come from the row alone:
   dense up to 2,048 keys, otherwise its selected list compacted in list
   order. The partitions are the ranks. No bit depends on wave composition.
4. The partials move to the head owners, 4 heads × 2,056 B per row per
   peer.
5. The head owner merges them in rank order (`LmLatentShardMergeKernel`,
   the same math as `LmLatentAttentionDecodeSplitCombineKernel`).

`LmKvShardReplicaView` runs step 3 over the replicated cache with the
ownership filter. It is the test oracle and is not a runtime path.

### Exchange per DSA layer

Numbers come from `SparkKvShardExchangePlanBuild` (4 heads per rank, fp32
partials, slot payload 262,192 B):

| B | Query out per rank | Query rounds | Partials per peer | SCATTER rounds | Wire out per rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 KiB | 1 | 8.0 KiB | 1 | 120 KiB |
| 8 | 32 KiB | 1 | 64 KiB | 5 | 964 KiB |
| 16 | 64 KiB | 1 | 128 KiB | 9 | 1.9 MiB |
| 64 | 256 KiB | 1 | 514 KiB | 33 | 7.5 MiB |
| 128 | 512 KiB | 2 | 1 MiB | 65 | 15 MiB |
| 256 | 1 MiB | 4 | 2 MiB | 129 | 30 MiB |
| 512 | 2 MiB | 8 | 4 MiB | 257 | 60 MiB |

The wire bytes are affordable at every batch size: 7.5 MiB at 25 GB/s is
about 0.3 ms per layer at B64. The 256 KiB mesh slots are the limit,
because a SCATTER round carries 16 KiB per peer. The chosen collective
shape is:

- **B ≤ 8:** SCATTER all-to-all over the existing slots (#1335), 1-5
  rounds per layer. Above 2,048 tokens of context, the query rides in the
  index-CP all-gather that already runs, so it costs no round of its own.
- **B ≥ 16:** per-peer exchange buffers sized to the exchange
  (GPU-initiated RDMA, Phase 4). Until those exist, the sharded path is
  still exact, but it pays the round counts above.

### B1 budget

B1 may cost at most 3% more per token: 0.75 ms on the 25.2 ms engine B1.

The estimate:

- **Extra rounds.** 2 per DSA layer (1 above 2,048 tokens) × 11 layers ×
  about 40 µs (the per-round floor in "Per-token budget" below). That is
  +0.9 ms, or +0.45 ms above 2,048 tokens.
- **Attention compute.** It drops from 72-83 µs to 42-45 µs per layer:
  -0.35 ms.
- **Net.** About +0.1 to +0.55 ms, inside the budget. This is an estimate
  until the fleet run below.

### Measured (sparkf GB10, `tests/cuda/kv_shard_cuda.cu` under `perf_window.py`, shared GPU)

Times are per rank per DSA layer.

Two runs on the shared GPU, first under heavier load, second at review:

| Case | KV per rank | Sharded partial + merge | Main's replicated heads kernel |
| --- | ---: | ---: | ---: |
| B1, 1k | 64 KiB of 1 MiB | 44.5 / 30.9 µs | 82.6 / 51.2 µs |
| B1, 8k (selected list) | 0.5 of 8 MiB | 41.5 / 41.2 µs | 71.9 / 59.4 µs |
| B8, 1k | 0.5 of 8 MiB | 211 / 78.6 µs | 198 / 137.8 µs |
| B8, 8k | 4 of 64 MiB | 185 / 113.5 µs | 190 / 159.3 µs |
| B64, 1k | 4 of 64 MiB | 664 / 563 µs | 525 / 419 µs |
| B64, 8k | 32 of 512 MiB | 964 / 851 µs | 909 / 780 µs |

At B1 both kernels are latency-bound: the sharded B1 8k partial reads about
128 KiB per rank per layer, about 1% of GB10 memory bandwidth, and does
about 1.5% of its fp32 compute.

Correctness in the same runs:

- Every case matches the replicated-storage oracle bit for bit, at grain
  1 (latent KV) and grain 4 (indexer-key ownership).
- Every owned slot holds the replicated bytes (CPU-shim test).
- Rows 0, B/2 and B-1 decoded alone have the same bits as inside the
  batch, so the row law holds for the store, the partials and the merge.
- `compute-sanitizer --tool memcheck --padding 256` reports no access
  outside the 1/degree pools.
- The all-to-all and all-gather receive layouts merge to the same bits.
- Against an f64 reference the worst difference is 1.2-2.4e-4 on the
  device and up to 4.8e-4 on the CPU shim. Both tests require less than
  2e-3. At the earlier bound of 1e-2, a merge that skipped the max rescale
  or a partial that dropped one key per rank still passed some cases.
- Against main's replicated kernel the difference is at most 4.9e-4.

That last difference is a reassociation, so COMPSEC-17 and MTP parity
must be requalified once the path serves. The same tests at rope 64 (the
k3 shape) and at degrees 4 and 8 pass too. The CPU-shim variant is
`tests/test_kv_shard_host.py`.

### Not done yet

- **Transport.** #1335 is a draft: the single-GPU mesh probe times out a
  peer gate at degree 2. Main's own probe also fails on sparkf today, at
  its first degree-16 case.
- **glm5_next wiring.**
  - Per-rank pools of `page_bytes / tp` for the latent and index caches
    and for the KV arena block.
  - `SparkGlm5NextPageCopy` per rank.
  - The sharded store.
  - Query gather, partial exchange and merge as chain stages in the
    eager, linear and graph paths.
  - The indexer on grain 4.
  - Module init refuses TP16 without sharding.
- **k3 wiring.** The same pieces at rope 64 (`K3_MLA_*`), on k3's own
  collective.
- **Fleet check.** B1 and B64 tok/s and KV bytes per rank against
  production, in an assigned weightd lane.
- **Large batches.** Without Phase 4 buffers, B64 pays 33 SCATTER rounds
  per DSA layer: about 11 × 33 × 40 µs = 14.5 ms on a 68 ms step (+21%).
  That cost needs its own budget before B ≥ 16 serves sharded.
- **Indexer keys.** No sharded store or scoring kernel exists for them
  yet. Only the grain-4 ownership rule is tested.
- **Prefix reuse and JIT-KV restore.** Untested until the per-rank page
  copy exists.

## Head and DSA attention at large batches

At B256 after the per-expert kernel, the harness showed `begin_head` at
106 ms and `attention_dsa` at 99 ms. Both kernels were written for one
row.

**Head.** `LmHeadCandidateKernel` runs one block per (row, 1024-token
tile). Each thread walks a whole 8 KB weight row with 2-byte loads, and
every row re-reads the full 79 MB vocabulary slice. The multi-row path now
uses `LmHeadCandidateRowsKernel`, which works on 16 rows × 128 tokens per
block:

- the weights come in as coalesced 64 × 64 tiles through shared memory
  (padded to stride 65, so the per-token reads are conflict-free);
- the weights are read once per 16 rows instead of once per row;
- each thread accumulates one token over `k` in the same order as the
  per-row kernel, so every token's score is bitwise equal;
- ties go to the lowest token id, both within a tile and in
  `LmHeadCommitKernel`.

`SPARK_GLM5_NEXT_HEAD_TILE` is now 128 (was 1024) and is shared by the
module allocation, the validation tools and the kernels. Single-row decode
uses the certified FP8 head and is unchanged. The single-row
`Glm5NextHead` (MTP) now has 76 candidate tiles instead of 10. Its scores
are identical, and the token can differ only on an exact score tie.

**Latent attention.** `LmLatentAttentionDecodeKernel` runs one block per
(row, head) and does a block-wide reduction for every attended position.
Each rank's 4 heads therefore read the same 1 KB latent row 4 times, with
a barrier per position. In decode waves (every row is a different
sequence, and there are at least 2 rows), `LmLatentAttentionHeadsKernel`
replaces it:

- one block per row (and per partition, if the split applies);
- 8 warps walk the positions in a strided pattern, and each warp loads a
  latent row once for all 4 heads;
- the dot products reduce with warp shuffles and there are no block
  barriers in the loop;
- the online-softmax states of the 8 warps are merged in a fixed order at
  the end;
- split partitions use the existing `(m, l, o)` layout and combine kernel.

The fp32 summation order differs from the per-head kernel, so the result
is not bitwise equal to it. Batched decode was already not bitwise equal
to B1, because the old split partition count depends on `rows × heads`.
Prefill waves and single-row decode keep the old kernel.

**Tests:**

- `tests/host_cuda/glm_rows_kernels_host.cu` (host, also covers the per-head projection rows kernel bitwise for 1–40 rows; run by
  `tests/test_glm5_next_rows_kernels_host.py`). It runs both kernels with
  real thread cooperation through `tests/host_cuda/lm_host_threads.cuh`,
  which uses one pthread per CUDA thread, barriers for `__syncthreads`, and
  a per-warp exchange for `__shfl_xor_sync`. The checks:
  - the head rows kernel is bitwise equal to the per-row kernel for 2–40
    rows, including an exact tie;
  - the attention output matches an f64 reference, split and unsplit, for
    1, 2 and 4 heads, with selected positions and `0xffffffff` holes.
- `make test-glm5-next-rows-kernels` (GPU): the same checks at full
  hidden size and vocabulary slice, plus `TIMING` lines for B8 and B256.

## DSA per-head projections (after #1218)

After #1218, the B256 harness showed `begin_head` at 8.2 ms but
`attention_dsa` still at 52.9 ms. Reading the KV once costs about 12 ms at
context 1024. Most of the gap is in two per-head projections that run
before and after the attention core:

- **Query absorb:** 256 → 512 per head, through `kv_b_key_transposed`.
- **Value up:** 512 → 256 per head, through `kv_b_value`.

Both ran on `LmPerHeadProjectKernel`, which has the same problem the head
had: one block per (row, head), each thread walking a whole weight row with
2-byte loads, and every row re-reading the head's 256 KB weight. At B256
that is 2 × 256 × 4 × 256 KB × 11 layers, about 5.8 GB per step of
uncoalesced reads.

`LmPerHeadProjectRowsKernel` computes one 64-output tile per block, for one
head and 16 rows (4 rows when the wave has at most 4). It shares its tile
loop with the head rows kernel (`inference/kernels/rows_tile.cuh`):

- coalesced 64 × 64 weight tiles through shared memory;
- sequential `k` accumulation, so its output is **bitwise equal** to the
  per-row kernel.

Because the output is bitwise equal, it is used at every row count,
including B1.

The all-heads attention walk now issues two positions' latent loads per
warp before it computes either of them. That doubles the memory-level
parallelism at large B, where each SM holds only one 8-warp block
(212 registers). Each warp still processes its positions in the same
order, so the result is unchanged.

## Grouped experts at large batches (after #1219)

At B256 the routed experts take about 190 ms of the 302 ms step. Their
weights stream in about 66 ms: 288 experts × 1.57 MB × 42 layers is
19 GB at 273 GB/s. The per-expert grouped kernel spent the difference on
three things:

- **Activation re-reads.** One lane group computed one output neuron. For
  every neuron, it re-read all of the expert's activation rows from L1/L2.
  At 8 rows that is 16 times more activation traffic than weight traffic.
  For W2 (K = 128) it also meant 1.2 million tiny tasks per call.
- **FP8 conversions.** The `cvt.e4m3x2` inline assembly was marked
  `volatile`, so the compiler could not reuse a converted weight chunk
  across the rows of a pass. Every row converted the same 16 weights again.
- **Scale index math.** Each 16-element chunk looked up its FP8 scale
  through the generic tensor path, which costs two runtime integer
  divisions.

The fixes:

- `LmSkinnyGroupedKernel` now gives each lane group four neurons
  (`LM_SKINNY_GROUPED_NEURONS`). One staged activation chunk serves all
  four neurons, and a W2 call has a quarter as many tasks.
- The skinny FP8 chunk converts through `LmE4m3PairToFloat2Pure`, whose
  assembly is not `volatile`, so the compiler can reuse a converted chunk
  across rows. Every other conversion in `dtype.cuh` stays `volatile` (see
  the measurement below for why).
- Skinny kernels compute the scale index directly: row stride times neuron,
  plus the chunk offset divided by the compile-time scale group. Launch
  validation requires the scale layout this assumes (one row per scale
  row, `Format::kScaleGroup` columns) and otherwise falls back to the
  tensor-core path.

Each neuron keeps its lane assignment, its chunk order and its reduction
tree. The grouped kernel is therefore bitwise equal to the one-neuron
version and to the per-pair kernel used up to 8 tokens.

**Measuring it.**

- The harness prints `ROOFLINE-EXPERTS` per batch. It splits the experts
  phase into `up_ms` (W1 and SwiGLU), `down_ms` (W2) and `combine_ms`
  (finalize, shared expert, add), and gives `up_gbps` and `down_gbps`
  against the distinct experts actually selected.
- `test-skinny-gemv` times the grouped kernel with four neurons, two
  neurons and one neuron against the tensor-core GEMM, for both shapes, at
  9, 32, 64 and 256 tokens.

**Tests.**

- `tests/host_cuda/skinny_grouped_host.cu` runs on the threaded host shim.
  It checks that the four-neuron kernel is bitwise equal to the one-neuron
  kernel and to the per-pair kernel, at 2, 5 and 20 tokens, for both the
  W1 and W2 shapes.
- The GPU test adds the same bitwise check for FP8 at 9, 32 and 96 tokens.

**Measured (first revision, PR #1221).** The grouped kernel was bitwise
equal everywhere and ran at 260 GB/s standalone, 3.67 times the
tensor-core path. B256 experts fell from 190 to 134 ms (up 71.8, down
38.7, combine 22.9 ms). But the B256 step rose from 302 to 344 ms: KDA
attention took 46 ms more, DSA 33 ms and the router 21 ms, reproducibly,
while B1 and B8 did not move.

The first revision had dropped `volatile` from every conversion in
`dtype.cuh`, not only the skinny one. The tensor-core GEMM calls the same
helpers (`LmFp8::Fragment` for FP8 weights, and in every instance, BF16
included, the UE4M3 and UE8M0 decodes inside the generic scale loader), and
its register allocation changed with them:
the 64-row-tile FP8 instance went from 117 to 60 registers and the BF16
instance, which the router uses at large batches, from 87 to 45. These
GEMMs serve the KDA, DSA and router projections from 32 rows up, which
matches the regression exactly. The skinny kernels serve B1 and B8, which
is why those did not move.

The revision keeps `volatile` everywhere except the skinny chunk. Compared
with main, every kernel outside the skinny family now compiles to
identical instructions (only relocated addresses differ). The grouped
kernels, the multi-row per-pair kernels and the W1 single-row kernel
compile exactly as in the measured revision. The only skinny kernels that
differ from it are the single-row ones with two or more neurons per lane
group; for GLM that is the W2 kernel at B1 and B8 (124 registers instead of
128, same occupancy).

**Checking which kernels a change touches.** Host tests cannot see register
allocation, so a shared-header change can move every kernel that includes
it. `make kernel-codegen-diff` compiles the harness (which instantiates
every GLM kernel) to a cubin and compares it with a base cubin kernel by
kernel. It normalizes relocated addresses, prints register, stack and local
memory use for each kernel that changed, and fails when a kernel outside
`ALLOW` changed (with no `ALLOW`, any change fails). It also fails when
either cubin yields no kernels, as happens when `nvdisasm` is not installed,
so a broken toolchain cannot pass as identical:

```sh
git checkout main && make build/glm5_next_batch_roofline.cubin && cp build/glm5_next_batch_roofline.cubin /tmp/base.cubin
git checkout <branch> && make kernel-codegen-diff BASE_CUBIN=/tmp/base.cubin ALLOW=LmSkinny
```

On the first revision it reports the nine `LmGemmKernel` instances as
unexpected. On the second it reports only skinny kernels.

## Per-token budget of the released TP16 build (09fdad6, 2026-09-28)

Source: the residentd logs of all 16 production ranks, aligned by chain
epoch with `tools/tp_chain_budget.py`, and the single-GPU bench in CUDA
graph mode (`bench-glm5-next-batch ROOFLINE_ARGS="--graph 1"`). Receipts:
`qualification/glm5next/performance/glm_perf_20260928_09fdad6/`. The
per-rank table comes from `tools/tp_chain_rank_compute.py <log dir> <lo ms>
<hi ms>` and the per-replay kernel gaps from
`tools/nsys_graph_replay_gaps.py <nsys cuda_gpu_trace csv>`.

B1 graph chains (472 chains that all 16 ranks ran, rank-mean replay wall
below 32 ms per step), per decode step:

| Part | ms | Note |
| --- | ---: | --- |
| GPU compute in the replay | 19.2-19.8 | replay wall minus device peer wait, copy and combine; spread across ranks 0.6 ms |
| Collective latency floor | 3.7 | peer wait of the last-arriving rank: about 40 us for each of 92 rounds |
| Rank skew | 1.0 | mean peer wait minus the floor |
| Copy and combine | 0.85 | |
| Host between replays | 1.4 | chain wall minus replay walls, per step |
| Total | about 26-27 | engine B1 is 25.2 ms per token |

Skew is not the main loss at B1: kernel time is. The same step on an idle
GB10 in graph mode (sparke, context 1024) takes 15.4-15.7 ms, and nsys
shows under 1 ms of launch gaps in the 1,462-node graph. The production
replay spends about 4 ms more on compute than the bench; the bench has no
collective kernels and uses `cudaMalloc` weights instead of the leased
expert pool, so that difference needs a profile of a real rank.

The last-arriving rank at B1 is rank 1 in 223 of 472 chains, then ranks 15,
8 and 12. Those nodes ran other lanes' work at the time: a three-process
CPU reference decoder on spark1, qwen27b on spark8, gemma4 on sparka to
sparkd and dev tests on sparkf. The same bench binary took 17.8-85 ms per
step on sparkf (median 23.9) against 15.3-18.0 ms on sparke. GPU or memory
bandwidth work on any production node slows every token.

At larger waves (replay wall 32-50 ms per step) the floor is 8.5 ms and skew
5 ms per step, so skew matters more there.

**Route readback removed from resident chains.** Every routed layer queued
a device-to-host copy of the expert group offsets and an event record, which
only the lazy expert path reads. Linear chains and graph captures require
every expert to be leased and never read them, but the copy sat between the
router and the expert GEMV in each of 42 layers. The walk now calls
`SparkGlm5NextLaunchCudaLayerMlpRouteResident`, which launches the same
kernels without the copy. The graph drops from 1,504 to 1,462 nodes. Six
alternating rounds of 100 replays on sparke, medians of the round medians:
B1 15.75 to 15.39 ms (-2.3%), B8 32.17 to 31.87 ms (-0.9%).

Kernel time at B1 on the idle bench (nsys node trace; shares of GPU busy
time): BF16 skinny GEMV 36%, FP8 expert GEMV 13%, HC site 10%, head 5%,
latent attention 5%, RMSNorm 3%. At B8 the FP8 expert GEMV is 55%.

## Next steps

Collective latency sets most of the B1 gap (see "Where it stands"). The
09fdad6 per-token budget above attributes most of the B1 step to kernel time
instead; both measurements stand until a profile of a real rank settles it.

1. Measure main on the fleet. Deploy a merged-main build, record its
   identity, and rerun B1, the 8-stream load and COMPSEC-17. Later changes
   should be measured against current code, not against `dd3526b`.
2. Remove the CPU relay from the critical path with GPU-initiated RDMA
   (IBGDA-style): the GPU writes the work requests and rings the NIC doorbell.
   The expectation, not yet measured, is a 16-rank 8 KiB all-reduce in tens
   of microseconds.
3. Take the host out of decode chains. Feed each step's tokens on the device,
   and stop waiting for every graph replay on residentd's thread (TECHDEBT;
   see "Resident decode chains").
4. Fuse RMSNorm into the consuming GEMV and batch GEMVs that share an input,
   to cut the launches per token (1950 at the #1208 measurement).
5. Restore bounded expert residency without losing graph speed
   ([GLM_LAZY_DRIVER_INTEGRATION.md](GLM_LAZY_DRIVER_INTEGRATION.md)).
6. Choose the topology by measurement. Qualify TP16, TP4xPP4 and PP16 each on
   its own (I38). Measure aggregate tok/s and per-stream latency from B1 up to
   the largest batch that fits, and serve with whichever measures best. The
   pair-link hierarchical all-reduce for prefill and B >= 16 belongs to this
   step.
