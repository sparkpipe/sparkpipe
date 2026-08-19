# GLM52 page_table <-> page-cache data-flow (corrected, verified against kv_cache.c)

For the record. Corrects the earlier "memcpy resident_slot_logical_block_indices ->
page_table" claim, which was the WRONG direction. Verified against the arena.

## Verified resident-slot semantics (cache/kv_cache.c)

- The kernel's device page table maps (sequence, page) -> RESIDENT SLOT:
  LmKvSlotRequired does pool + page_table[seq*stride+page] * kPageBytes + slot *
  kSlotBytes (inference/kernels/kv.cuh:285-325), and the arena places a block at
  key_device_base + key_block_stride_bytes * RESIDENT_SLOT
  (SparkKvCacheArenaAssignResidentSlot, kv_cache.c:998-1027). So the kernel's
  physical_page IS the resident slot.
- resident_slot_logical_block_indices[resident_slot] = logical_block is the
  REVERSE map (slot -> block), set by AssignResidentSlot (:1018-1019). It is NOT
  the kernel's page table source.
- A block becomes resident via SparkKvCacheArenaAcquireBlock (:738) then
  SparkKvCacheArenaMarkBlockResident (:1299), which assigns the slot + key
  address. AssignResidentSlot scans the free slot list in identity order, so
  marking blocks 0..N-1 in order yields slot i = block i (identity).

## Correct data-flow (the unit that replaces the raw identity table)

1. PinLane step (MISSING today): after SparkKvPageCachePrepareLane resolves the
   lane's logical pages, acquire + mark each one resident (mirror
   SparkDsv4PagedCachePinLane, dsv4_paged_cache.c:309-334):
   for each logical_page: SparkKvCacheArenaAcquireBlock then
   SparkKvCacheArenaMarkBlockResident. This is what populates the arena's
   resident slots + key addresses and lets eviction/prefetch move blocks.
2. Sync helper SparkGlm52KvSyncPageTable(state): rebuild the device page_table as
   (sequence, page) -> resident_slot. Per sequence: SparkKvPageCacheBuildLaneTable
   -> logical_page_indices[]; for each logical page, resident_slot =
   arena.blocks[logical_page].resident_slot_index; cudaMemcpy the row to
   page_table[seq*stride + page]. Call after init, after each predicate tail, and
   after CompleteLane. In the all-resident identity state this rebuilds the exact
   identity table today's kernel expects, so it is byte-safe.
3. Unmap on eviction/prefetch: when the arena evicts a block or the page store
   prefetches one, the sync writes LM_KV_PAGE_UNMAPPED for that (sequence, page),
   and LmKvSlotRequired already traps on unmapped (kv.cuh:300-311).

## Why this lands the raw-init deletion cleanly

Once PinLane + Sync are in, the identity builder (SparkGlm52BuildPageTable) and the
raw device page-table allocation are subsumed: the page table is a device cache of
the page cache's residency, and the kv_cache device pool is the arena's
key_device_base (already true). Net-negative: delete the identity builder + raw
page-table allocation (~30 lines), add PinLane + Sync (~35 lines).

## This turn's increment

Only the publish/rollback surfacing landed (SparkKvPageCacheCompleteLane /
RollbackLaneTransaction status now flows into async->completion.status instead of
(void)). PinLane + Sync + eviction/prefetch remain the next unit — they are only
correct once the arena's residency is actually driven, and touching the device
page table before then risks silent KV corruption that compilation will not catch.
