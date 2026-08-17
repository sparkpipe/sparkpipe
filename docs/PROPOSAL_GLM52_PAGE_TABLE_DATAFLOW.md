# GLM52 page_table <-> page-cache data-flow (statement, no code yet)

For the record: the next unit after completion-half + backing-path is the
page_table <-> page-cache data-flow, which unblocks the raw-init deletion.

## Today (post completion-half)

- The kernel reads a DEVICE identity page table
  (`SparkGlm52BuildPageTable`, `spark_glm52_resident_decode_stage_module.c`):
  `page_table[sequence * pages_per_sequence + page] = page` (identity), so
  `LmKvSlotRequired` (`inference/kernels/kv.cuh:250-326`) resolves
  `pool + physical_page * kPageBytes + slot * kSlotBytes` where physical_page
  is always the identity block index.
- The common page cache maintains the SAME mapping HOST-side in its four
  caller-owned tables (`entry_indices_by_logical_page`,
  `resident_slot_logical_block_indices`, `entries`, `sequences`), updated by
  PrepareLane / BeginLaneTransaction / CompleteLane / RollbackLaneTransaction /
  ReleaseLane. Today both are identity, so they agree by construction; the device
  page table is a redundant copy.

## What the data-flow unit replaces

1. **Delete `SparkGlm52BuildPageTable`** (the identity builder) and the module's
   raw `page_table` allocation. The device page table becomes a cache of the
   page cache's `resident_slot_logical_block_indices` array
   (resident-slot -> logical block), copied device-side.
2. **Add one sync helper** `SparkGlm52KvSyncPageTable(state)`: memcpy
   `state->kv_resident_slot_logical_block_indices` (host) -> the device
   `page_table`, called after every predicate tail (prepare/commit/abort) and
   after CompleteLane, so the kernel's `page_table[seq * stride + page]`
   resolves the LOGICAL block the page cache actually assigned.
3. **Unmap on eviction/prefetch**: when the arena evicts a block or the page
   store prefetches one, the sync writes `LM_KV_PAGE_UNMAPPED` into the device
   slot, and `LmKvSlotRequired` already traps on an unmapped page
   (`kv.cuh:300-311`) — the scheduler's admission guarantee (only resident
   blocks are admitted) is what prevents that trap in the steady state.

## Why this lands the raw-init deletion cleanly

With the device page table sourced from the page cache (not an identity builder),
the raw `kv_cache` layer-major pool + identity page table are fully subsumed by
the arena's block-major `key_device_base` + the page cache's block mapping. The
only remaining raw-init code is the `kv_cache` device allocation itself, which
becomes the arena's `key_device_base` (already true). Net-negative: delete the
identity builder + the redundant device page-table allocation (~30 lines),
replaced by the sync helper (~12 lines).

Blocked only on the eviction/prefetch enablement decision (whether GLM52 adopts
the page store's Writeback/Prefetch path or stays all-resident); the sync helper
is needed in both cases and can land first.
