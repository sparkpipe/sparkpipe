# JIT-KV vertical slice — report (2026-08-29)

Worktree /tmp/lane-jikvslice, branch `lane/jikv-slice` (off main @ 9eb58ef).
Scope: the first end-to-end proof of the JIT-KV contract — module
save/restore ops + the LRU pager + admission backpressure, running the REAL
arena (cache/kv_cache.c) and the REAL nvme tier (cache/nvme_tier.c, B3
digests) end to end in a host test, TERM-only (no CUDA, no reservations, no
fleet, no model tokens in shared code). The five mission proofs are green,
10/10 repeated runs. NO merge by this lane — integration requested.

## What landed

1. `include/sparkpipe/spark_kv_pager.h` + `cache/kv_pager.c` — the pager
   (the design doc's adapter layer): park/restore/admission policy joining
   the resident arena to the backing tier.
2. `SparkKvCacheArenaMarkParkedBlockResident` (cache/kv_cache.c +
   header decl) — the restore-half primitive. MarkBlockResident refuses
   BACKING_VALID blocks BY DESIGN (a plain mark would hand out a resident
   block nobody re-filled); the pager path had no way back onto the device.
   This is that counterpart: make-room via the normal LRU eviction, assign
   the slot, keep BACKING_VALID, keep DIRTY clear (re-park then costs
   nothing). Refuses never-parked (blank) blocks — those go through
   MarkBlockResident.
3. `tests/test_jit_kv_slice.c` — the cuda-stub host proof (Makefile rule +
   TEST_NAMES entry).

## The module seam (design doc step 2, model-neutral form)

The pager does not know CUDA. It calls two function pointers with a
`SparkKvPagerBlockView` — the model-neutral shape of the design's
KvBlocksView (lane_index, first_block, block_count, device plane addresses,
host_staging): `module_save` (device planes -> staging; the
KV_BLOCKS_SAVE_OUT 0x1000 seam) and `module_restore` (staging -> device
planes; KV_BLOCKS_RESTORE_IN 0x2000). The host test implements them as TERM
copies; the family wiring (dsv4 first, per JIT_KV_RESPONSE W1) implements
them with the frame ops unchanged. The backing write leg is likewise a
callback (pwrite in production): the tier stays read-vtable and hands a
reserved device_offset.

## The pager contract

- PARK = eviction. The pager installs itself as the arena's evict_function:
  stage the victim's planes, SHA-256 them, fold the digest into the tier's
  64-bit bucket key (B3 discipline: digest = identity, key = bucket; a
  collision cannot alias because the tier decides on digest), ReserveWrite,
  backing-write only when the payload is NEW (content-addressed dedup for
  free — two blocks with identical bytes park once), CommitWrite. IO-class
  failures abort the reservation and propagate so the arena's B1 degrade
  owns them (drop + recompute, never a wedge); BUSY propagates as
  backpressure; a digest collision stays loud.
- RESTORE = digest-verified page-in. Demand read via
  SparkNvmeTierRequestDemand/Pump (the tier verifies the landing; the pager
  re-verifies the buffer it copies from), then
  MarkParkedBlockResident (whose make-room may page the LRU victim OUT —
  the loop closes), then the module restore op. MISS answers NOT_FOUND
  (recompute path). A module-restore failure rolls the block straight back
  out (MarkBlockNonResident) — never resident-and-trusted.
- The deadlock the slice found and designed out: the tier protects a record
  whose staging is DEMAND-held from its own clock. A restore that held the
  tier's staging across make-room would let the make-room write-backs
  BUSY forever on the very record being read (reproduced, then fixed). The
  pager lands the verified bytes into its OWN staging plane (double buffer:
  save scratch / restore landing) and Consume-releases the tier buffer
  BEFORE MarkParkedBlockResident. Documented at the code and in
  KVCACHE_SUBSYSTEM_BOUNDARY.md 1.5a.
- ADMISSION (C1) = queue, never wedge. `SparkKvPagerAdmit` recomputes the
  arena's exact overflow arithmetic (resident + reserved + unassigned vs
  capacity), names the deficit, and checks it against (a) the exact
  parkable pool — the arena victim selector's own exclusions, so pinned /
  protected residents are structurally never victims — and (b) the park
  budget (tier records in use). Both fit -> ReserveUnassignedResidentBlocks
  (which trims to fit = does the parking); else the offer is QUEUED with
  the reason counted (admission_queued_device / admission_queued_backing).
  QUEUED holds no reservation; commit/release wrappers move the unassigned
  ownership per the arena's contract. Reserve's BUSY/CAPACITY also map to
  queued-with-reason; every other status stays loud.

## The five proofs (tests/test_jit_kv_slice.c, all green)

1. FILL BEYOND BUDGET: 16 logical blocks, resident capacity 4 (device
   budget 16 KiB), tier horizon 12 records. Lane A fills 4; lane B's
   demand 4 admits BY PARKING (deficit 4).
2. LRU PAGING OUT with digest-verified write-back: touching block 1 after
   the fill makes the victim order {0,2,3,1} — recency, not fill order
   (bounded page-out history in the pager statistics). Every eviction
   committed a tier record under its payload digest (publishes == 4,
   digest_mismatches == 0, backing writes == 4); parked blocks are
   backing-valid + non-resident and lane A still owns them.
3. REWIND: RestoreBlock(1) issues a real demand read (demand_loads >= 1),
   lands digest-verified (digest_verifications >= 1, mismatches 0), and the
   device planes are BIT-EXACT vs the true payload. Making room paged the
   LRU resident out (page-out history's newest entry). A re-dirtied rewind
   re-parks as a DEDUP: already_present, no backing write, tier record
   reused; an UNTOUCHED restored block re-parks silently (the arena skips
   the write-back for clean+backed blocks — the design's "re-park is free").
4. BACKPRESSURE: pinned (active) residents make parkable 0 -> the offer
   QUEUES with admission_queued_device; ten repeated offers queue healthy,
   zero page-outs, nothing evicted, nothing leaked. A full park horizon
   QUEUES with admission_queued_backing and stays queued. A tier-boundary
   BUSY (both tier records pinned) surfaces as a clean BUSY from the
   rewind — retryable, write_back_degraded_block_count == 0 (BUSY is
   backpressure, not a drop); after one unpin the SAME restore succeeds
   bit-exact. When the load drops (a lane completes, budget frees), the
   next offer ADMITS with zero parkings — the queue releases, never poisons.
5. THE BUDGET AT EVERY INSTANT: SparkKvPagerAssertDeviceBudget (resident +
   reserved + unassigned <= capacity; resident bytes <= configured device
   budget; tier slots <= slot count) runs after EVERY state change in every
   scenario, plus the arena's own residency gate. Init fences: a device
   budget above SPARK_KV_PAGER_DEVICE_LAW_BYTES (the 110 GiB device law) is
   refused; so is a budget smaller than the arena's whole resident
   capacity, undersized staging, and a second pager stealing an arena whose
   eviction is already owned.

## Gates, ratchets, artifacts

- `make offline-gates` (build-all, run-tests, package-manifest): PASS on
  this host (clang arm64, no nvcc; the CUDA artifacts SKIP per the gate
  contract). test_jit_kv_slice is registered in TEST_NAMES so the slice
  proof runs in every future offline pass. Honesty note: the FIRST full
  run failed exactly one gate — test_dry_law caught a model token in a
  pager-header comment ("(dsv4 first)"); reworded, and the second full run
  is green end to end. That is the gate doing its job on this lane's own
  code. The two failures the jit-safety lane saw on older main
  (test_dsv4_driver_source_contracts -lcuda, memory-contract re-pins) do
  NOT reproduce on this base.
- Size ratchet: 222168 -> 223008 EXACT (+840), in-commit justification:
  the pager (header + impl, 725 lines), the arena restore primitive (+13),
  its declaration (+13), the Makefile rule (+5). The test is excluded by
  construction. Complexity gate held unchanged (production max 75, mean
  7.85 — no new function above the file norm). Memory-contract gate green
  with no new parked entries.
- SHA256SUMS + PACKAGE_MANIFEST.json regenerated with the tools;
  `verify_package_manifest`: "package manifest, payload, and checksums
  match" (same repair the jit-safety lane made; no manual edits).

## Sequencing (docs/JIT_KV_RESPONSE.md)

This slice is the host half of W1: the pager + its arena primitive + the
tier are now one proven loop (admit -> park -> rewind -> restore ->
backpressure), family-neutral. What remains for W1 proper is the family
wiring — the dsv4 module implementing the save/restore seam with the
0x1000/0x2000 frame ops and the adapter calling SparkKvPagerAdmit in its
JIT-KV predicate — plus C2 (dispatch gates on restore-complete via the
tier's lookahead) and C4 (async park worker; the pager's save path is
already off the decode-critical-path-shaped: one seam call + one tier
reserve + one backing write per block, no arena re-entry). B2's orientation
contract is untouched (no glm5_next file was edited).

## Integration request

Please review and merge `lane/jikv-slice` (pushed to origin). Never merged
main here; branch is 9eb58ef + one commit.
