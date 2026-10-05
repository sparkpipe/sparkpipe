# Cross-engine shared prefix pages

Two engines on one node that serve the same model with the same KV layout
read one published prefix from one set of device pages. weightd owns the
pages; each engine's KV binding maps them next to its private pages and
references them from its page tables like any other page.

## Scope

- **Who shares.** Bindings share when their layout digest matches: the
  same model, codec, geometry and context shard. Bindings that carry
  recurrent lane state (GLM Flash, K3) do not share: their prefix state
  lives in a per-engine state store that another engine cannot read. Such
  a binding logs `kv shared prefix off: recurrent state` and keeps every
  prefix private.
- **Opting in.** The node operator gives weightd
  `--kv-shared-window-bytes N` (or `SPARK_WEIGHTD_KV_SHARED_WINDOW_BYTES`).
  The bytes are carved from the KV reserve, and each layout key gets a
  window of that size.
  - Without the flag, every binding logs
    `kv shared prefix off: weightd has no shared window` and keeps its
    prefixes private.
  - With the flag, every binding of the layout maps the window. A failure
    to map it fails the binding loudly; nothing falls back to private
    pages quietly.

## Layout

Each region keeps its private pages `[0, P)` and gains the shared window
`[P, P + S)` right after them.

- **Page-major regions:** the window is one span.
- **Layer-major regions:** every layer's page array grows by `S` pages, so
  the window is one span per layer, and the layer stride becomes
  `(P + S) * layer_page_bytes`.

Drivers read the stride from the binding, so no kernel changes.

- **Alignment.** CUDA VMM maps whole chunks of at least the allocation
  granularity `G`. Every window span and every private piece must
  therefore start and end on `G`, so `P` and `S` are rounded to multiples
  of `A`, the least common multiple over regions of
  `G / gcd(G, page bytes of the region)`.
- **Private chunk size.** Private chunks tile each private piece exactly,
  so the private chunk size is the largest `G * 2^j` that divides every
  piece and keeps the pool within weightd's chunk limit. `P` is rounded
  down further when needed, and the binding logs the pages it gave up.
- **Mapping.** The binding reserves one address range for the whole
  layout. It maps its private pool chunks at the private offsets and the
  shared pool chunks at the window offsets.

## weightd

- **Attaching.** `KV_SHARED_ATTACH` (IPC ABI 12) attaches a connection to
  the shared pool for a key that names the layout digest. The request
  carries the page bytes and the layout's alignment `A`.
  - weightd sizes the window: `S` is the largest multiple of `A` that fits
    the configured bytes, capped at 65536 pages.
  - The first attach creates the pool: `S` pages' worth of chunks of size
    `G`, plus a metadata memfd that holds the shared index.
  - Later attaches must name the same page bytes and layout. weightd
    answers `SCHEMA_ERROR` otherwise.
- **Holders.** Each attached connection gets a holder index (0–62).
- **Detach and crash.** When a connection closes, weightd clears that
  holder's bits in every slot, and frees every slot that holder left
  `WRITING`. A crashed engine therefore never pins window pages.
- **Lifetime.** The pool is freed when its last holder detaches.
- **Caps.** At most 63 holders per pool: bit 63 of the mask is the
  reclaim lock.
- **Export.** Chunks are exported with `KV_POOL_EXPORT` like a private
  pool. Shared pools never resize.

## Shared index

The index is an array of `S` slots in the memfd, one per window page
(`include/sparkpipe/spark_kv_shared_index.h`, header-only so weightd and the
binding share it).

- **Slot fields:**
  - state: `FREE`, `WRITING` or `READY`;
  - the writer's holder index;
  - a 64-bit holder mask;
  - a generation;
  - a last-use stamp;
  - the chain identity and token count of the prefix ending on this page;
  - the parent slot and its generation.
- **Reserve.** A writer takes a `FREE` slot by compare-and-swap to
  `WRITING`. Failing that, it reclaims the least recently used `READY`
  slot nobody holds: it first swaps the holder mask from zero to the
  reclaim lock (bit 63), then swaps the state to `WRITING`.
- **Acquire.** An acquirer sets its holder bit. It backs off if the old
  mask carried the reclaim lock, or if the slot is no longer `READY` with
  the generation it read.
  - A held slot can never be locked, so its state and contents stay put
    until the last holder releases it.
  - A four-thread stress test caught an earlier design that swapped the
    state first: it briefly hid held slots from readers.
- **Publish.** It stores the identity, bumps the generation and releases
  `READY`. It runs only after the wave that wrote the page completed.

## Page cache

- **Allocation.** When the window has a free or reclaimable slot, a new
  mutable page comes from the window: an arena block flagged `SHARED`
  whose physical page is `P + slot`, where `P` is the binding's full
  private page count. That count stays fixed while the pool grows and
  shrinks the usable private pages.
  - The arena never parks or counts shared blocks.
  - The kernels write the page in place.
  - A partial block that is never published returns its slot as `FREE`
    when the sequence releases it.
- **Publication.** The entry keeps the shared block, and the slot becomes
  `READY` under the entry's identity, with the parent's slot. A chain is
  shared only from its root: a page whose parent is private stays private.
- **Lookup.** A lane whose prefix misses locally looks the identity up in
  the index, acquires the chain from the root, and imports each page as a
  local entry on the shared block. If any link changed, it stops at the
  last valid page and recomputes the rest.
- **Eviction.** Evicting a shared entry drops the holder bit; it never
  parks the page. Snapshot saves read shared pages like private ones.

## Engine

Each lane's engine predicts prefix hits from its own prefix index. The
engines of one model on the API host also read each other's index files:

- `model_api --peer-runtime-root PATH`, repeatable up to 8, names a peer
  lane's runtime root.
- Every two seconds the engine imports the committed records of any peer
  index file whose inode, size or modification time changed. Records for
  another model are refused by the index's model digest.

A record whose pages are no longer in the window answers `NOT_FOUND`
from every rank. The engine then recomputes that prefix and tombstones
it, the same path a missing snapshot takes.

## Evidence

- Host tests:
  - the index protocol, with four concurrent holders under
    ThreadSanitizer;
  - the arena's shared blocks;
  - the page cache's publication and import across two caches;
  - weightd's shared pools;
  - two bindings in one process: the second serves the first one's
    published prefix from the same window slot;
  - the engine's peer index import.
- The CUDA stub copies on map rather than aliasing, so the host tests
  check page tables and holders, not bytes.
- The fleet proof is still open: two GLM Full lanes on one node, where the
  second lane's prompt hits the first lane's prefix with no extra device
  pages and identical tokens.
