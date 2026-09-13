# Topology switching: TP16 <-> PP16 in ~20 s, KV warm across the switch

The ring serves the same model under two deployment strategies, TP16 and
PP16, with necessarily different per-rank weight files. Switching must take
about twenty seconds with requests in flight, and switching BACK must resume
sequences from the NVMe KV tier wherever the 1 TB has not evicted them. This
document is the budget and the protocol; the machine is
`scheduler/topology_switch.c` behind `include/sparkpipe/spark_topology_switch.h`,
verified host-side by `tests/test_topology_switch.c` (gate: `topology
switch` in `tools/gates.sh`).

**Every time below is an estimate unless a line says measured.** The two
load-bearing constants — internal NVMe bandwidth and the swap fixed cost —
are unmeasured on this ring; both are configuration fields
(`SparkTopologySwitchConfiguration`) so the estimate and the machine read
the same numbers, and `SparkTopologySwitchEstimateBudget` recomputes the
whole table when hardware reports better ones.

## Why a switch does not invalidate KV

The weight side of a switch is total: TP16 and PP16 shard the same weights
differently, so a rank's resident bytes under one recipe are useless under
the other. The KV side is the opposite. The attention cache is MLA latent
KV, and the latent vector for a (layer, token) is defined by the model's
math, not by the topology: the latent projections are head-agnostic and
replicate (`include/sparkpipe/spark_tp_shard.h:23-26`), which is what makes
the latent cache byte-identical on every rank under TP. Under PP a stage
holds the same latent vectors for its own layers. A switch therefore
re-shards KV residency; it never changes KV content. The entire tier key
scheme exists to keep that true:

```text
tier key = SparkTopologySwitchKvKey(namespace, content_hash)

content_hash   chained token-run hash (cache/cache.h): commits to the
               token prefix and nothing else - no layer, rank, or degree,
               so strategy-neutral by construction.
namespace      hash of model identity + NEUTRAL cache geometry (layer
               count, slot bytes, block tokens, precision). The strategy
               is deliberately NOT hashed. Two recipes of one model share
               one namespace: a TP->PP switch finds the old recipe's
               records under the same names it published them. A different
               model, or a re-quantized one, gets a different namespace
               and correctly misses.
```

One record per block holds every layer's latent for the block's tokens,
layer-major. Under TP a resume reads the whole record — every rank needs
every layer. Under PP a stage reads its layers' slice, a contiguous range
at `record_base + layer_lo * slot_bytes * block_tokens`: re-sharding on
read is an offset computation, not a re-layout, so one neutral copy on the
tier serves both strategies and no per-strategy duplicate copies are
needed. (Duplicates would double the tier's write traffic to buy nothing
the slice does not provide.)

## The protocol

One rank's view; all ranks run it in lockstep and the coordinator's only
role is to say BEGIN. Admissions close synchronously inside
`SparkTopologySwitchBegin` — admitting into a recipe that is being unloaded
is how a sequence ends up bound to freed weights.

```text
QUIESCE     stop admitting; in-flight sequences run to their next token
            boundary. Bounded by one decode step: a step is atomic at the
            token boundary by design, so "let what is in flight land" can
            never wait longer. A sequence that finishes here is dropped
            from the checkpoint set - checkpointing finished work is the
            purest waste there is.
CHECKPOINT  per live sequence: pin every KV block on the tier (Publish is
            the tier's only eviction path, and the clock skips pinned
            blocks, cache/nvme_tier.c:333-336), then write one manifest
            record (position, block key set, recipe). Blocks already
            written back during serving are re-found, not re-written; the
            manifest is the one new write, and it is the crash-recovery
            record a restarted rank rebuilds from.
SWAP        the swap device unloads the old recipe's modules and streams
            the target recipe's packs from the rank's OWN NVMe - pre-staged
            there while the old recipe was serving. This phase is the
            whole budget.
RESUME      per sequence, total classification: every block still on the
            tier -> WARM, blocks planned into the tier's lookahead and the
            sequence re-admitted without prefill; any block absent ->
            RECOMPUTE, prefill from the prompt (the ordinary demand path
            still serves whatever suffix survived - the skip-prefill win
            is all-or-nothing because decode at position N needs the whole
            prefix). Pins drop as classification completes.
```

There is no CANCEL. Mid-swap the residency is split between recipes, the
one state worse than either alone; the protocol is crash-only, and the
manifests on the tier are what a restarted rank rebuilds from. Phases retry
on BUSY (a full tier finding no unpinned victim, a swap stream not yet
drained) instead of failing: a switch is an operator action, and the right
response to BUSY is to finish the current decode step and try again.

## The 20 s, in parts

From `SparkTopologySwitchEstimateBudget`; the serial total excludes resume,
which overlaps the first decode steps through the tier's lookahead.

```text
phase          cost                                   basis
quiesce        <= 1 decode step:
               B8 chat   ~0.1 s                        ESTIMATED: K3 B8 step
               B1024     ~0.9 s                        bytes 310/2,559 GB at
                                                        2,840 GB/s (80% roofline,
                                                        PERF_ROADMAP_2026-08-01.md
                                                        :300-306)
checkpoint     1,024 seqs x ~4 KB manifests            ESTIMATED at 2 GB/s write
               = ~4 MB -> ~2 ms
swap fixed     unload + allocator/arena reset + bind   ESTIMATED 0.5 s,
               = swap_fixed_microseconds               UNMEASURED placeholder
swap stream    ~100 GB/node at 5-7 GB/s internal NVMe  ESTIMATED: 14-20 s
               = weight_pack_bytes / read bandwidth    (techdebt.md model-swap
                                                        guarantee; bandwidth
                                                        PENDING per roadmap
                                                        :496-498)
--------------- -------------------------------------- -------------------
serial total   ~15-21 s, dominated by the stream
resume warm    <= 32 GB/node working set at 5-7 GB/s   ESTIMATED 4.6-6.4 s,
               = OVERLAPPED, not added to the total    overlapped via lookahead
                                                        (NVME_KV_SIZING.md:222-233)
```

The stream is ~100 GB/node either direction: the model is ~1.6 TB over 16
ranks, and TP and PP partition it differently but not unevenly. The
bandwidth floor the 20 s target implies: with 0.9 s quiesce (B1024 worst
case) and 0.5 s fixed, the stream has 18.6 s, so the drive must sustain
**>= 5.4 GB/s per node**. The assumed 5-7 GB/s internal range fits with
little headroom at its low end — which is why the load must be NVMe->GPU,
not network->GPU: the wire is not the constraint (dual 100 Gbps outruns the
drive), but a network fetch re-reads the packs from some OTHER node's drive
at the same 5-7 GB/s while that node's drive is also serving its own swap,
doubling the demand on the binding resource and adding a cross-node failure
domain to an operator action. Pre-staging makes each rank's swap
single-node and independent.

Pre-staging cost: two recipes x ~100 GB = ~200 GB of the internal drive
held for the alternate recipe while serving. Against the 1 TB that leaves
~800 GB for the KV tier and manifests — the layout NVME_KV_SIZING.md
already argues for (packs and the swap warm path on the internal drive; KV
may move to a larger external drive, since no 128K chat working set needs
internal speeds, ibid.:3-6).

What the budget does NOT contain, and why:

- **KV reload in the serial path.** Warm resume is planned into the tier's
  lookahead at classification time, so the first decode steps fetch ahead
  and the reload is traffic behind serving, not wall time before it. A
  cohort whose working set exceeds resident memory pays 4.6-6.4 s of
  overlapped fetch, invisible against the 20 s.
- **Recompute of cold sequences.** A sequence whose blocks were evicted
  from the 1 TB (or never written back) goes through prefill after the
  switch. At 128K of context that is the one cost the budget cannot
  absorb, and the protocol's answer is that it is bounded by the tier, not
  by the switch: the paused working set is <= 32 GB/node against 1 TB,
  ~30x headroom, and checkpoint pins make eviction between checkpoint and
  resume impossible (the clock steps over pinned records). Eviction can
  only happen BEFORE the switch is requested, under ordinary serving churn.
- **Manifest I/O.** Kilobytes per sequence; microseconds.

## Failure and retry semantics, briefly

Every phase's work is bounded bookkeeping plus non-blocking vtable calls;
`Advance` carries a per-phase cursor and `last_error` and retries the
failed unit next call. Checkpoint pins before it publishes (a manifest's
own slot acquisition could otherwise evict an unpinned block it names), and
`pins_done` makes a retried publish exact — a double pin leaks one count
that resume's single unpin never returns. Resume classifies against
`OffsetOf` and drops pins either way.

## Integration points (the hardware binding, at bring-up)

The machine is host-side schedule arithmetic over two vtables, the same
shape as the tier it sits on:

- `SparkTopologySwitchSwapDevice` — the rank's weight loader:
  `begin_swap` (non-blocking; DMA from the pre-staged packs into staging,
  copy up) and `poll_swap`. This is the piece that binds to
  `node/backend.c`'s module unload/load path.
- `SparkTopologySwitchWriteBlock` — the manifest PUT at the offset
  `SparkNvmeTierPublish` returns (the tier names records; the caller owns
  the write, `cache/nvme_tier.c` Publish contract).
- Scheduler lifecycle hooks: admission consults
  `SparkTopologySwitchAdmissionsOpen`; admit/complete mirror into
  `TrackSequence`/`SequenceComplete`; `SetSequenceKv` is kept current as
  blocks publish; the decode loop calls `SequenceAtBoundary` at each token
  boundary and `Advance` once per step, next to `SparkNvmeTierPump`.
- The coordinator broadcasts BEGIN with the target recipe descriptor; ranks
  need no further coordination because the protocol is deterministic given
  the same begin step.

## Verification

`tests/test_topology_switch.c` drives the machine with a mock swap device
(programmed swap latency in polls), a mock manifest write path, and the
real tier over a mock drive: quiesce holds until the last boundary and
completes the switch-bar-swap in one advance; checkpoint pins survive a
full tier's worth of eviction churn; warm sequences resume via the
lookahead (planned fetches, not demand stalls); absent and pre-switch-evicted
blocks classify RECOMPUTE; the budget arithmetic is asserted in numbers.
Gate: `topology switch` in `tools/gates.sh`; binary in `TEST_NAMES`.

Needs hardware: the swap fixed cost (0.5 s placeholder), the sustained
internal NVMe read rate under concurrent tier traffic (the >= 5.4 GB/s
floor), and the end-to-end 20 s claim itself.
