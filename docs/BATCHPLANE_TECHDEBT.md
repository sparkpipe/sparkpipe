# Batch-plane performance notes

Audit findings on the expert-queue batch plane host components. Fixed items
are recorded for provenance; open items are ordered by measured cost.

## Fixed

- **JIT KV pool eviction was O(fragments) per stage-in.** A linear scan over
  every fragment selected the eviction victim on each miss. Measured 281 us
  per require at 262144 fragments, scaling linearly with pool size; at 131072
  DRAM fragments and roughly 18000 stage-ins per longmem run this was seconds
  of pure host scan per rank, larger than the compute it scheduled. Replaced
  with a max-heap over DRAM-resident fragments keyed on next_need_ns, with the
  fragment's heap position tracked so an ETA change re-sifts in place. Now 0.13
  us per require, flat in pool size, a 2160x reduction. Tie-break on equal need
  evicts the higher fragment id deterministically for the ring SHA gates.

- **JIT KV pool transfer array hard-failed under prefetch bursts.** The pending
  transfer store was a plain array compacted only on Tick, and each stage-in
  queues up to two transfers (evict plus in), so 2048 un-ticked requires
  exhausted the 4096 slots and returned CAPACITY_EXCEEDED. A prefetch burst
  from a wave boundary hit this directly. Replaced with a ring buffer whose
  overflow completes the oldest in-flight transfer inline rather than refusing,
  so requires never hard-fail; overflow_drain_count records how often it
  happens so the transfer ceiling can be tuned from real traces.

## Open, ordered by measured cost

- **ExpertQueue NextFiring rescans all layer x expert slots from origin.**
  Every call walks up to layer_count x expert_count slots to find the next
  fireable one, so draining N firings is O(N x slots). Measured 2.2 us per
  fire, roughly 0.1 s per rank per longmem run. Modest next to the pool fix and
  a larger change (an intrusive ready-list of slots at or past threshold,
  pushed on enqueue crossing and on deadline, popped on fire), so deferred.

- **Serving-adapter per-dispatch H2D staging.** The decode serving adapter
  issues per-step cudaMemcpyAsync host-to-device for tokens and routing tables,
  a discrete-GPU pattern that on GB10 unified memory is a DRAM-to-DRAM copy plus
  launch latency. Bandwidth-negligible at KB-MB payloads, tens of us against
  16 ms stages, but convertible to mapped-host writes if the verify gate lands
  near 25 ms and needs shaving. Requires ring measurement to justify.

- **Workspace allocator residency unverified.** Whether the central workspace
  allocator uses mapped or managed allocations versus plain cudaMalloc lives in
  runtime host code outside the decode modules and could not be confirmed from
  the host sandbox. On GB10 both resolve to the same DRAM so steady-state
  bandwidth is identical; the only difference is a transient copy on load paths.
  One grep on the ring-side runtime settles it; fold into the packet-timing
  session by reporting per-stage copy counts and bytes.
