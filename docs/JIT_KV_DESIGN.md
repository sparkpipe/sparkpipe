# JIT-KV paged cache — design and implementation contract (2026-08-24)

Goal: B1024 lanes x 256k context with a 2.5TB backing store. The 2.5TB
is the LRU horizon (38M tokens = 14% of full B1024x256k for this model,
64 KiB/token); VRAM (34 GB pool = 524k tokens) holds the ACTIVE set.
The system parks and restores, never thrashes.

## The core insight that shapes everything

Attention decode READS a lane's whole context KV every token. A lane
whose KV is not in VRAM cannot decode - paging per-token is a 20-40x
cliff. Therefore: paging is for IDENTITY, not for active compute.

- ACTIVE lanes: full context resident in VRAM (the existing pool).
- PARKED lanes: KV saved to the backing store; zero VRAM footprint.
- SHARED PREFIXES: content-digested blocks deduplicated across lanes
  and requests - in VRAM while referenced, spillable to backing.

Parking/restoring generalizes the prefix publish/borrow machinery that
already works (bit-identical, merged): publish = save blocks + GDN
snapshot under an identity; borrow = re-attach + restore GDN. The pager
is that loop with a disk tier and a scheduler policy.

## Layering (matches the repo's ownership rules)

                     adapter (policy + accounting)        <- THIS design's core
                        park/restore ops, backpressure, slot maps
                     module (data movement)               <- thin ABI addition
                        save-block / restore-block / snapshot ops
                     backing store (runtime/spark_kv_backing)  <- LANDED
                        slot file, pread/pwrite, async worker

## Module ABI addition (frame-context ops, versioned flags)

New frame-context flags (same pattern as GDN_PREFIX_SNAPSHOT_OUT):
  KV_BLOCKS_SAVE_OUT   0x00001000u   (park: the frame's lane blocks ->
                                       host buffers, then the adapter's
                                       worker writes them to backing)
  KV_BLOCKS_RESTORE_IN 0x00002000u   (restore: backing -> host -> VRAM
                                       blocks, before the layer loop)

View struct (module-firmware header, mirrors SparkQwen38_27bGdnSnapshotView):
  typedef struct SparkQwen38_27bKvBlocksView {
      uint32_t abi_version, descriptor_bytes;
      uint32_t lane_index;
      uint32_t block_count;        /* blocks being parked/restored */
      uint32_t first_block;        /* ordinal within the lane table */
      void    *host_staging;       /* block_count * KV_BLOCK_BYTES, pinned */
  } SparkQwen38_27bKvBlocksView;

Data path per block (4 MiB): cudaMemcpyAsync D2D lane-block ->
staging is NOT needed for whole blocks: copy device block -> host
staging (one 4 MiB D2H), the worker pwrites it. Restore reverses.
One stream-ordered op per block; a park of N blocks = N x 4 MiB
D2H + pwrite, overlapped by the worker thread.

## Adapter: the pager (state per lane and per backing slot)

New adapter state:
  lane_parked[MAX_LANES]      : {identity[32], first_slot, slot_count,
                                token_count, gdn_snapshot_host}
  backing_slots               : free list over the backing file
  park_epoch[MAX_LANES]       : LRU for victim selection

PARK (scheduler-initiated, on CoverLane allocation failure):
  1. victim = LRU lane with no in-flight submission (never the caller)
  2. module frame with KV_BLOCKS_SAVE_OUT for the victim's blocks +
     GDN_PREFIX_SNAPSHOT_OUT (the existing op) -> host
  3. worker: pwrite blocks to fresh backing slots; record the map
  4. release the victim's VRAM blocks (refs: shared-prefix blocks stay
     pinned by their entries; only private blocks migrate)
  5. lane state -> parked

RESTORE (a submission arrives for a parked lane):
  1. worker: pread the slots -> staging
  2. module frame with KV_BLOCKS_RESTORE_IN + GDN restore before the
     lane's first compute frame (the prefix-borrow path, generalized)
  3. free the backing slots (or keep as the lane's spill copy while
     active - keep: re-park is then free until blocks mutate)

BACKPRESSURE (the anti-thrash rule):
  Offered load is bounded by (VRAM blocks + a park budget). When both
  are exhausted: requests QUEUE (admission control), they do not evict
  an active lane younger than the requester's priority. Priority from
  the API's deadline field (the README contract). The cliff never
  fires: a restored lane is fully resident before its first token.

## GDN state at 256k

Per-lane recurrent state is O(MB) - parked/restored with the lane in
the same op (host buffer alongside the block staging). Prefix-borrow
already proves the restore path is bit-exact.

## File format (runtime/spark_kv_backing.c - implemented now)

  header (4 KiB): magic "SPKVBS01", slot_bytes, slot_count, free hint
  slots: fixed stride, 4 MiB each (one KV block), positioned by index
  Concurrent-safe: single-writer worker; pwrite/pread at slot offsets.

## Performance model (why this is efficient)

- Park cost: N x (4 MiB D2H + pwrite). At NVMe ~10 GB/s and PCIe
  ~26 GB/s, a 256k-context lane (4096 blocks = 16 GiB) parks in ~1.6-2s
  of background I/O, overlapped with other lanes' decode.
- Restore cost: same magnitude - a B1024/256k fleet re-arming a lane
  pays seconds per lane, NOT per token. Round trip amortizes over the
  lane's next active period.
- Steady-state B1024: the fleet rotates active/parked sets; VRAM stays
  ~fully occupied by active lanes = the measured 24.5+ tok/s class
  decode shape; parked lanes cost zero VRAM.

## Failure modes and rules

- Backing full: park fails -> backpressure tightens (queue), never
  thrash; logged loudly ("backing horizon reached").
- Crash: backing slots are advisory state; on daemon restart the file
  is re-initialized (prefix digests re-validate on borrow; a stale
  slot read fails the digest check and falls back to cold recompute -
  the existing safe degradation).
- Bit-exactness: restore == never-parked (the prefix-borrow proof
  already covers the GDN half; the block half is a byte copy).

## Build order (this document is the contract)

1. LANDED: runtime/spark_kv_backing.c + header + unit test (slot file,
   alloc/free, round-trip integrity).
2. Module: the two frame-context ops + view struct (the D2H/H2D paths;
   ~100 lines, mirrors GdnPrefixTransfer).
3. Adapter: park/restore policy + worker thread + the generalized
   identity map (extends the prefix store; the borrow path becomes
   restore-from-backing when the entry has spilled).
4. Scheduler: backpressure in the engine's admission (cache-demand
   path already threads kv_physical_page_capacity - extend it with
   the park budget).
5. Fleet: B1024 limit raises (deployment JSON, module build params,
   position caps) with the pool sized by residency.
6. Tests: park/restore bit-exactness at B1/B16; backpressure under
   offered load 2x VRAM; crash-restart staleness fallback; the
   canonical O512 stream unchanged when the pager is idle.
