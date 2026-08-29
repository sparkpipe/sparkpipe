# Paired Dual-Link All-Reduce

This document specifies SparkPipe's unified collective service for TP4, TP8,
and TP16 on the sixteen-Spark combined fabric.

## Topology

Every rank has two mandatory rails:

```text
switch_rail: CRS804 all-to-all fabric
direct_rail: rank XOR 1 pairwise link
```

The direct pairs are `0<->1`, `2<->3`, `4<->5`, `6<->7`, `8<->9`,
`A<->B`, `C<->D`, and `E<->F`. A TP communicator contains complete direct
pairs. A pair never crosses PP stages or independent TP communicators.

Routes are pinned. A direct edge cannot escape through the switch, and a
switched edge cannot migrate to the direct interface. Missing either route
fails communicator construction.

## Algorithm selection

The immutable hardware profile maps this key to one plan:

```text
{communicator size, payload bytes, datatype, concurrency profile}
```

Selection is an O(1) lookup in the token path. TP8 and TP16 have independent
crossover tables. Batch labels are not selector inputs; effective payload
bytes include every active or speculative verification row.

### Recursive doubling

Recursive doubling handles small, latency-dominated payloads. Each step
exchanges and reduces the complete payload:

```text
XOR 1: direct rail
XOR 2: switched rail
XOR 4: switched rail for TP8 and TP16
XOR 8: switched rail for TP16
```

TP8 uses three dependent steps and TP16 uses four. Later steps consume the
result of earlier reductions, so one collective does not claim false overlap
between dependency-ordered rails.

### Recursive halving/doubling

The medium-payload plan performs a topology-aware reduce-scatter followed by
the reverse all-gather. For TP16, the reduce-scatter exchanges `S/2` over XOR
1 direct, then `S/4`, `S/8`, and `S/16` over switched XOR 2, 4, and 8. The
all-gather reverses those steps.

This plan transfers fewer bytes than recursive doubling and has fewer dependent
round trips than a ring. TP8 uses six dependency steps and TP16 uses eight.

### Counter-rotating split rings

The large-payload plan constructs one alternating physical cycle. The TP16
forward cycle is:

```text
0 --direct--> 1 --switch--> 2 --direct--> 3 --switch--> ...
E --direct--> F --switch--> 0
```

The reverse ring uses every edge in the opposite direction. Aligned,
alternating chunks are assigned to disjoint stripes:

```text
even chunks -> forward ring
odd chunks  -> reverse ring
```

Each stripe independently performs ring reduce-scatter and all-gather. Every
payload byte belongs to exactly one stripe.

At steady state every rank performs:

```text
direct TX + direct RX + switch TX + switch RX
```

The forward and reverse rings use independent queues, buffers, credits, and
progress state. One direction cannot wait behind completion of the other.

## Traffic balance

For `N` ranks and an `S`-byte tensor, a conventional one-port ring sends and
receives `2 * (N - 1) / N * S` through the switched port.

Each split ring carries `S/2`, so each physical port direction carries:

```text
(N - 1) / N * S
```

| Communicator | Switch TX | Switch RX | Direct TX | Direct RX |
| --- | ---: | ---: | ---: | ---: |
| TP16 | `0.9375S` | `0.9375S` | `0.9375S` | `0.9375S` |
| TP8 | `0.875S` | `0.875S` | `0.875S` | `0.875S` |

The switch remains half of the active communication fabric while carrying half
the bytes of a conventional one-port ring. The ideal gain is 2x. Full duplex
enables the two stripes; it does not create an additional 2x multiplier.

## Host-RDMA execution

Communicator initialization allocates fixed page-locked mapped host arenas,
registers them with the owning NIC, exposes their device mappings, creates
queues, and pre-posts a bounded receive window. The steady-state path:

1. receives NIC data into a registered slot;
2. publishes ordered readiness to the GPU;
3. reduces without an intervening CPU payload copy;
4. transmits an eligible earlier chunk; and
5. recycles a slot only after network and GPU ownership are released.

There is no per-layer connection, registration, allocation, model reload, or
communicator construction. There is no per-chunk CPU payload copy, device-wide
synchronization, or CPU-dispatched phase transition.

The large-payload pipeline overlaps receive `k`, reduce `k-1`, transmit `k-2`,
and preparation of a later registered slot. Chunk bytes and in-flight depth are
profiled together for exact model payloads.

## Correctness invariants

- Every byte belongs to exactly one stripe and reaches every rank exactly once.
- No all-gather exposes a partition before its reduction is complete.
- Slot reuse waits for send, receive, and GPU reduction ownership release.
- Message identity includes communicator, model step, layer, sequence, phase,
  stripe, and chunk.
- Duplicate or stale completion cannot advance a newer collective.
- Every rank selects the same algorithm and chunk geometry.
- Padding cannot affect model-visible output.
- Floating-point reassociation is compared within the model's accepted
  tolerance; integer and exactly representable fixtures remain exact.

## Instrumentation

Every size bucket and algorithm records invocation count, payload bytes, p50
and p99 latency, buffer and credit wait, GPU reduction time, per-rail TX/RX
bytes and throughput, in-flight depth, progress CPU time, and eligible-work
idle gaps.

Hardware interface counters prove physical routing. Application counters alone
cannot close the route or byte-balance gate. Timeout diagnostics identify the
rank, rail, phase, stripe, and chunk that stopped progressing.

Measurements and crossover values live only in
[`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md). Unfinished
implementation and qualification work lives only in
[`../TECHDEBT.md`](../TECHDEBT.md).
