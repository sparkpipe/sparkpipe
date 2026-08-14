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

## Second audit pass

### Fixed

- **ExpertQueue init wrote a 1M-entry free list then zeroed a 24MB struct.**
  The eager free-list walk plus a full-struct memset cost 6309 us per init.
  The free list is now lazy: a high-water counter hands out unused rows in
  O(1) and only recycled rows go on the free list, so no init walk is needed;
  and the memset now covers only the header and slot array up to offsetof(rows)
  since the row pool is fully written by the allocator before any read. Init
  dropped to 1.8 us, a 3500x reduction. Row allocation and release are now
  single helpers used by both enqueue and firing rather than open-coded in
  three places.

### Cleared (checked, not a defect)

- **Overflow-drain versus DRAM capacity invariant.** Suspected that completing
  a stage-in inline during a transfer-ring overflow could push resident count
  past capacity between the pre-check and the heap insert. Stress with 7904
  forced drains at tight DRAM holds dram_resident + staging_in at exactly the
  cap with no breach: overflow drains complete stage-out and stage-in transfers
  in the FIFO order they were queued, so the accounting stays paired. Sound.

- **Firing emit cap over MAX_FIRING_ROWS.** A slot deeper than the emit array
  caps the firing at the array bound and leaves the remainder queued with a
  correctly advanced oldest arrival; memory-safe, no truncation of live rows.
  Locked with a test.

### Open, lower cost

- **JitKvPool init memset is 513 us.** The fragment array must be zeroed for
  the FREE-state sentinel, so unlike the queue it cannot skip the row region.
  Could zero only fragments[0..capacity) when capacity is below the max, but
  that trades a clear invariant for a rare-path saving and is deferred.

- **Config validation preambles are structurally similar across the three
  init functions** but validate different field sets, so they are left
  explicit rather than folded into a macro that would hide the field checks.

## Third pass: unconditional optimization

### Added — content-addressed KV dedup

The batch plane's dominant bandwidth term is attention, not the expert sweep
(measured 82 vs 48 GB/s per rank at the four-thousand-sequence point), and the
attention traffic is dominated by re-reading latent KV. Most levers there are
workload-conditional: FP8 latent is a quality tradeoff, and lane selection
co-scheduling depends on the unmeasured natural overlap of the eight lanes'
DSA selections. Content-addressed dedup is the one that is strictly never worse
in any workload: byte-identical prefix fragments shared across sequences (system
prompt, tool schemas, few-shot examples, shared longmem memory) resolve through
a content hash to a single physical fragment, refcounted, freed only at zero
references. With no sharing the physical assignment is identical to per-sequence
storage, a 1.00x floor with zero regression; at a realistic twenty-four-fragment
shared prefix across five hundred sequences it frees 5.3 GB per rank, which
converts directly into more DRAM-resident sequences and less NVMe paging, and
the shared physical fragment is read once for attention across every sequence
that points at it. Open-addressed table, linear probing, backward-shift delete
with cluster reinsert, no allocation. Tested for sharing, refcount lifecycle,
and probe-cluster integrity after a mid-cluster free.

### Footgun recorded — batch-plane structs must never be stack-allocated

The dedup struct is about 10 MB, the expert queue 24 MB, the JIT pool 9 MB.
A stack local of any of them overflows immediately; a test that did so segfaulted
and was caught by AddressSanitizer. There is no compile-time guard. These are
singletons by design and belong in static or caller-provided storage; anything
introducing a stack instance will crash at entry. Worth a static-assert on a
stack-hostile size if a guard mechanism is added.

### Measured — dedup is a capacity win, not a throughput win at the current point

Wiring dedup into the integrated sim's admission and measuring corrected the
earlier framing. At five hundred to four thousand sequences with a 2048-token
selection the working set fits DRAM, so the fifteen-second wave transit already
hides all NVMe latency and removing paging traffic leaves committed throughput
flat at 16601. Dedup's effect there is not tok/s but headroom: with a
twenty-four-fragment shared prefix the resident need at four thousand sequences
drops from 128000 fragments to under the 65556-fragment pool, so a population
that otherwise pages stays fully resident, and stage-ins fall from 48112 to
27081 to 12 as the shared prefix deepens. The sequence ceiling before the paging
wall roughly doubles, and NVMe pressure drops about threefold. This converts to
throughput only past the pool ceiling, above roughly seven thousand
select-2048 sequences or at longer contexts, which is the stack-more-runs
regime. The claim is therefore: dedup raises the resident sequence ceiling and
cuts NVMe pressure unconditionally, and turns into throughput exactly when the
plane becomes paging-bound. The integrated sim caps at four thousand sequences
and models NVMe as latency-hidden rather than a clock throttle, so the
throughput crossover itself is shown by the closed-form model, not the sim; that
sim limitation is the next thing to lift if the crossover regime needs direct
simulation.

## Fourth pass

### Fixed — sequence table admit was O(n) and completed slots leaked

Admit linear-scanned for a free slot on every call, so filling the 16384-entry
table was O(n squared): 83.6 ms to admit a full batch, and 10.3 us per admit
under steady realtime churn. Worse, Complete marked a sequence COMPLETE but
never returned its slot, so under realtime churn the table filled permanently
and admit would eventually fail forever. Both fixed with the same lazy free-list
the expert queue uses: a high-water counter hands out never-used slots in O(1),
Complete pushes the slot onto the free list, and Admit pops it. Admit dropped to
0.004 us and churn to 0.009 us, roughly 1300x and 1100x. A churn test locks the
recycling and the leak fix by admitting and completing sixty-four times on a
capacity-four table and asserting the freed index is reused.

### Cleared — dedup table is correct at the capacity boundary

Verified the open-addressed table refuses cleanly at 100 percent load
(CAPACITY_EXCEEDED at the exact capacity, no infinite probe) and still resolves
existing entries when full. Not a defect. But linear probing degrades past a 0.7
load factor, and the table has no resize, so a max_probe_length counter is now
exported and the header states the sizing contract: provision at least twice the
expected distinct-fragment count. The caller watches max_probe_length to catch a
too-small table before latency suffers.

### Watch — free-list pattern now in two components

The lazy free-list with high-water counter appears in the expert queue and the
sequence table. Two is a pattern, not yet a violation; the intrusive link field
differs between them (list_next in a row versus free_next in a sequence), so a
shared helper would need a generic intrusive-list contract. Extract if a third
component needs it.

## Correctness audit

### Fixed — overflow drain committed time travel

The transfer-ring overflow path completed the oldest in-flight transfer inline
regardless of its completion time, so a fragment could be marked DRAM-resident
while its transfer's done time was still in the future. Proven with a probe: at
a frozen clock a burst filled the ring and the drained fragment reported
resident at time zero with its DMA due 73.7 microseconds later. In the
production integration, where Tick is driven by real DMA completion events,
this records a fragment resident before its data has arrived and the GPU reads
garbage KV. Fixed: transfer space is reserved for the whole evict-plus-in pair
up front, the reservation drains only transfers whose done time has passed, and
when the oldest transfer is still in flight the require refuses with
CAPACITY_EXCEEDED, because a saturated NVMe has no other honest answer. The
burst stress test is restaged to both behaviors: frozen-time bursts must refuse
and must show nothing resident, advancing-time bursts drain opportunistically
and always succeed. The prior never-hard-fail behavior was the bug wearing a
feature costume.

### Fixed — require during eviction desynchronized the state machine

Found by a randomized property test, not by any hand-written case: requiring a
fragment in STAGING_OUT treated it as an ordinary miss and re-staged it in,
overwriting the state under a live outbound transfer; the outbound's completion
then clobbered the inbound state and the resident, staging-in, and staging-out
counters drifted from the true fragment states. The require path now refuses a
mid-eviction fragment; its outbound completes on a Tick, it lands in NVMe, and
the retry stages it in normally. Deterministic and race-free.

### Fixed — stale sequence handle acted on a recycled slot's next occupant

Proven with a probe: after a sequence completed and its slot was recycled, a
holdover index silently paused the slot's new occupant with a success status,
leaving an unrelated sequence stuck awaiting a tool it never requested. Handles
are now generation-tagged, fourteen index bits and eighteen generation bits,
the generation incremented when the slot frees, and every transition resolves
the handle and refuses a stale generation with NOT_FOUND. ABI version bumped.

### Added — randomized property tests in the suite

Fifty thousand randomized require, tick, and time-advance operations assert on
every step that the eviction heap root equals a brute-force scan over all DRAM
fragments including the tie-break, and periodically that the resident,
staging-in, and staging-out counters equal the true per-fragment state counts
and that resident plus staging-in never exceeds the DRAM capacity. A three
hundred thousand operation dedup run against a reference map verified physical
id assignment, sharing, refcounts, frees, and capacity refusal exactly; the
pool property test is the one that found the eviction race and lives in the
permanent suite.

### Verified correct under review

Heap remove's swap-with-last followed by sift-down then sift-up is correct: the
element rising into the vacated position during sift-down is a former child of
the removed element, so it is bounded by the grandparent and the trailing
sift-up terminates on its first comparison. The decrease-key sift-down-only
argument holds for its single caller. The dedup backward-shift delete preserves
probe clusters, exercised by both the hand-written mid-cluster free and the
property run. Partial-failure semantics of a batched require are idempotent on
retry since a staging-in fragment counts as a hit.
