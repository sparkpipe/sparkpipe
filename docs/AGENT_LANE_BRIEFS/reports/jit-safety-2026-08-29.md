# JIT-KV safety lane — report (2026-08-29)

Worktree /tmp/lane-jitsafety, branch `lane/jit-safety` (off main @
04a419a, commit d00a92b). Scope: the four named bugs from kimi's
analysis, now B1-B4 of docs/JIT_KV_RESPONSE.md — the disqualifying-if-
unfixed list that gates the whole JIT-KV build and is the NEXT audit's
checklist. All four fixed, each with a reproducing test and a behavior
receipt. Both ratchets (size, CCN mean) moved with in-commit
justifications. No PR merged by this lane. PUSH/PR NOTE: the push to
origin failed — the stored GitHub token is rejected ("Invalid username
or token"; gh auth status: token invalid), so the branch is local to
this worktree; the coordinator should push lane/jit-safety and open the
PR when credentials are restored (never merge from this lane).

Gates on this host (clang arm64, no nvcc): all 38 C test binaries PASS;
every python gate PASS except two failures that are PRE-EXISTING on
pristine main and reproduce identically there (verified byte-for-byte
against the main checkout):

- tests/test_dsv4_driver_source_contracts.py — "missing CUDA Driver API
  validator link: -lcuda": the DRY wave 1 merge (61d6edc) moved the
  `-lcuda` link into the shared driver
  (modules/spark_resident_decode_stage_cuda_validation_common.sh:182)
  but the source contract at tests/test_dsv4_driver_source_contracts.py:436
  still pins it in the per-family script. Ownership: the DRY/dsv4 lane.
- tests/test_memory_contracts.py — parked-ratchet re-pin complaints on
  spark_qwen38_27b_resident_decode_stage_module.c, identical output on
  main. Ownership: whichever lane last shrank those call sites.

The ratchets moved with in-commit justifications (below). This lane also
regenerated SHA256SUMS and PACKAGE_MANIFEST.json via their tools, which
repairs the stale-manifest drift main was carrying (main:
verify_package_manifest FAIL(55); lane: `package manifest, payload, and
checksums match`). The CUDA compile gate needs nvcc and is a
target-hardware item, unchanged by this lane. The pre-existing main
`-Werror` const-discard in ring/transport/tp_device_collective.c (noted
by lane-hygiene) was not touched and does not affect these files.

---

## B1 — WRITE-BACK WEDGE (cache/kv_cache.c SparkKvCacheArenaEvictResidentBlock)

### Root cause

The eviction path called `evict_function` and, on any non-OK status,
returned it with the block still RESIDENT + DIRTY. The page store maps
ENOSPC/EFBIG (pwrite fails) to SPARK_STATUS_IO_ERROR and "no free
backing slot" to SPARK_STATUS_CAPACITY_EXCEEDED. Under a full disk the
write-back can therefore never succeed, so the dirty victim can never
stop being a victim, so no resident slot is ever freed: admission
returns IO_ERROR forever — a permanent serving wedge triggered by an
ordinary full NVMe tier.

### Fix (degrade, per the JIT-KV contract)

On `SPARK_STATUS_IO_ERROR` or `SPARK_STATUS_CAPACITY_EXCEEDED` the
eviction DEGRADES: the block is dropped — DIRTY and BACKING_VALID both
cleared — the resident slot is released, and eviction completes.
Clearing BACKING_VALID is the load-bearing half: restore
(SparkKvPageStorePrefetch) gates on it and answers NOT_FOUND, so no
caller is ever handed stale or partial bytes; the sequence recomputes
the block on demand. Degradations are counted in the arena
(`write_back_degraded_block_count`, uint32_t, saturating; placed in the
former `reserved1` padding so the struct size and ABI layout are
unchanged). BUSY keeps its existing backpressure semantics (async
write-back in flight); every other status (INTERNAL_ERROR,
INVALID_ARGUMENT, ...) still propagates loud — a program error must
never silently drop data.

### Reproducing tests (tests/test_kv_cache.c)

1. `SparkTestKvEvictionIoErrorDegradesInsteadOfWedging` — injected
   fault: every write-back returns IO_ERROR. Pre-fix this test hangs in
   the wedge (admission returns IO_ERROR forever); post-fix admissions
   keep succeeding, both dirty blocks are degraded, the degraded block
   shows !RESIDENT/!DIRTY/!BACKING_VALID, is re-admittable as a blank
   (recompute-on-demand), and once the fault clears the same block
   gets a real write-back again (degradation is per-attempt, not a
   poisoned state).
2. `SparkTestKvEvictionInternalErrorStaysLoud` — INTERNAL_ERROR
   propagates, the victim stays resident+dirty, degraded count stays 0.
3. `SparkTestKvPageStoreFullDiskDegradesAndServingContinues` — the
   full/errored-backing repro through the REAL page store: RLIMIT_FSIZE
   caps the backing file at one page (SIGXFSZ ignored), so the first
   write-back lands and every later one fails EFBIG → IO_ERROR on the
   worker thread — the exact shape of ENOSPC. Receipts: admissions
   keep succeeding; `write_back_degraded_block_count == 2`;
   `store.write_count` stays 1 and `backing_page_count` stays 1 (the
   failed reservation leaves no backing-state debris — the page store's
   RecordJob releases it); `SparkKvPageStorePrefetch` on the degraded
   block returns NOT_FOUND (never stale bytes); and the healthy block
   parked before the fault still restores byte-exact afterwards.
   25/25 repeated runs green (the BUSY-poll retry loops are
   race-tolerant by construction).

Receipt: serving continues post-degradation, no stale restore, no
debris, and the wedge is gone. verified in the three tests above.

## B2 — GLM5_NEXT ARENA GEOMETRY (modules/glm5_next_resident_decode_stage)

### Root cause (the contradiction)

glm5_next is hybrid: 11 DSA layers carry the compressed KV_A latent;
34 KDA layers carry no per-token KV (fp32 state + conv windows live in
their own stage pools). The device pool is allocated from the per-stage
DSA ordinals: `main_total = kv_layer_stride_bytes * state->kv_layer_count`
with `kv_layer_stride_bytes = page_count * 64 * KV_SLOT_BYTES`
(= 65,536 per layer). But the tier machinery was configured with the
STAGE's whole weight-layer count:

- `block_bytes = 64 * state->layer_count * KV_A_DIM * 2` (45 layers in
  the STAGE_COUNT=1 build) — 2,949,120 vs the real 720,896: 4.09x;
- `arena_configuration.layer_count = state->layer_count` — so the
  arena's derived block stride (`block_token_count * layer_count *
  kv_head_count * head_dim * bytes_per_scalar`) is the same 4.09x
  number.

The arena addresses resident slots as `key_device_base + slot *
key_block_stride_bytes` and the page store copies `page_bytes` per
block: restore writes and eviction reads run past the end of
`state->kv_cache` — the OOB DMA, the moment lanes wire to the page
directory. (Adjacent history confirmed the class: layer.cuh's
"first bring-up sized it x DSA_COUNT and the pool came out 65 GB" —
the kda-lane's double-multiplied stride, 64.96G → 5.9G.)

### Fix + fence

- `SparkGlm5NextKvInitialize` sizes `block_bytes` and
  `arena_configuration.layer_count` from `state->kv_layer_count` (this
  stage's DSA ordinals; also refuses `kv_layer_count == 0`).
- Fail-loud fence at init: after SparkKvBackendInitialize, the arena's
  `key_block_stride_bytes` must equal `block_bytes`, and the identities
  `block_bytes == kv_layer_stride_bytes / page_count * kv_layer_count`
  and `page_count * block_bytes == kv_layer_stride_bytes *
  kv_layer_count` must hold exactly — any future drift between the
  arena's address space and the allocated pool returns
  SPARK_STATUS_INTERNAL_ERROR at module init instead of corrupting
  silently.
- Compile-time assert: `KV_SLOT_BYTES == KV_ARENA_KV_HEAD_COUNT *
  KV_ARENA_HEAD_DIM * KV_BYTES_PER_SCALAR` (slot geometry and arena
  geometry describe the same per-DSA-layer page bytes).
- Layout-orientation contract documented at the wiring point: the
  device pool is LAYER-MAJOR (Glm5NextKv per-layer bases at
  `kv_cache + layer * kv_layer_stride_bytes`) while the arena names
  whole blocks contiguously — the page-directory wiring (W1) must
  translate between the two, and `evict_function` stays deliberately
  UNWIRED in this table until it does. The fence makes any silent
  sizing drift impossible; the comment makes the orientation trap
  impossible to miss.

Host verification: tests/test_glm5_next_geometry.py gains a B2 section
— the slot/arena byte identity (720,896 == 64*1024*11, and NOT the
45-layer 2,949,120) computed from the header macros against the
authoritative contract, plus source-contract checks pinning the fixed
lines (arena layer_count from kv_layer_count; no `state->layer_count`
in the arena config; the fence; the compile-time assert). The module
translation unit additionally passes `cc -fsyntax-only -Wall -Wextra
-Werror` with the cuda stub headers (no nvcc on this host — the full
compile is a target-hardware item).

## B3 — TIER CHECKSUMS (cache/nvme_tier.c, ABI 2 → 3)

### Root cause

The tier keyed slots on the bare 64-bit `content_hash` ("the same
chained hash cache/cache.h uses") and never looked at bytes: two
tenants whose blocks collide on 64 bits share one record silently —
the second writer's ReserveWrite was told `already_present`, the first
reader got whichever bytes were on the drive. No integrity anywhere
across the tier boundary.

### Fix — per-slot SHA-256 digests, verified on restore
(the prefix-cache content-digest design shape)

- Every `NvmeTierSlot` stores the SHA-256 of the payload it stands
  for. The 64-bit hash remains the bucket key; the digest is the
  identity. `SPARK_NVME_TIER_DIGEST_BYTES = SPARK_SHA256_DIGEST_BYTES`
  (shared implementation, src/spark_sha256.c).
- Writers present the digest: `SparkNvmeTierReserveWrite(hash, digest,
  out)` requires it (NULL/zero refused), records it at reserve,
  re-verifies at CommitWrite, and answers SPARK_STATUS_HASH_MISMATCH
  when an existing hash is presented under a different digest — a
  collision fails loud at the write boundary, never aliases.
- Readers present the digest where bytes move: RequestDemand (READY
  hands back a staging pointer) and Consume require it and answer
  HASH_MISMATCH on contradiction. Key-only bookkeeping (OffsetOf, Pin,
  PlanLookahead, WillBeResidentBy) may pass NULL — these classify,
  never deliver bytes — but a PRESENTED digest that contradicts the
  record still fails loud everywhere.
- Restore is verified: every landing in Pump is hashed
  (SparkSha256 over the staging buffer) and compared to the record's
  digest BEFORE the buffer becomes READY or a demand pointer. Mismatch
  → `digest_mismatches++`, the record is QUARANTINED (dropped from the
  index, slot recycled — so a corrupt on-drive record is never re-read
  forever; the next demand is an honest MISS → recompute) and Pump
  returns HASH_MISMATCH — loud, never wrong-KV.
- Statistics: `digest_verifications`, `digest_mismatches` appended.
- Consumers updated: scheduler/topology_switch.c (manifest writes
  present the real digest of the serialised manifest; Pin/OffsetOf/
  planning are key-only with the rationale documented) and
  tests/test_topology_switch.c. Makefile links spark_sha256.c into
  both test binaries.

### New tests (tests/test_nvme_tier.c, mock drive now a real byte image)

- corrupted-bytes test: flip one drive byte after commit → Pump answers
  HASH_MISMATCH, mismatch counted, the record quarantined (next demand
  MISS → recompute), slot recyclable by an honest write-back.
- collision test: same 64-bit hash, two payloads → ReserveWrite,
  RequestDemand, Consume and Pin under tenant B's digest all answer
  HASH_MISMATCH; tenant A unaffected; mismatches counted.
- hard-refusal test: all-zero/NULL digest on the decode path is
  INVALID_ARGUMENT, not a miss.
- every pre-existing tier test now runs digest-verified end-to-end
  (publish fills the drive image + presents the digest; every read
  presents it), with `digest_verifications >= landings &&
  digest_mismatches == 0` receipts added to the lookahead section.
  Full tier suite + topology switch suite green.

## B4 — BACKING-STORE HYGIENE (runtime/spark_kv_backing.c)

### Root cause

`open(path, O_RDWR|O_CREAT, 0644)` at caller-chosen paths (the test's
`/tmp/spark_kv_backing_test.bin` being the pattern): tenant KV at rest
world-readable at a predictable name.

### Fix

- Creation mode 0600, plus `fchmod(fd, 0600)` on EVERY open — files
  created by earlier builds at 0644 are migrated on first open
  (fchmod on the descriptor, no chmod-path race).
- `O_NOFOLLOW` (a symlink planted at the path is refused, not
  followed) and `O_CLOEXEC`.
- Namespaced paths: new `SparkKvBackingResolvePath(root, deployment,
  tenant, model)` composes `<root>/<deployment>/<tenant>/<model>.slots`
  with per-component hygiene ([A-Za-z0-9_.-], no leading dot, no
  separators — traversal is unrepresentable), and
  `SparkKvBackingCreateNamespaces` creates the directory chain at
  0700, tightening pre-existing loose directories to 0700 (the
  directory analogue of the file-permission migration). Documented in
  spark_kv_backing.h as the deployment contract.

### Tests (tools/spark_kv_backing_test.c)

Fresh file 0600; a 0644 file migrates to 0600 on open; a symlink at
the slot path is refused; namespaced layout composes exactly; four
traversal/separator/empty-id rejections; namespaces created 0700 and a
deliberately loosened 0755 namespace is tightened back. ALL PASS.

Note for the W1 wiring: glm5_next's page-store fallback directory
(`/tmp/sparkpipe_glm5_next_kv_<revision>`, mkdir 0700, page store
opens O_TMPFILE 0600) is already hygiene-clean; the backing-store
contract above is the one the lane wiring must use for the slot file.

---

## Ratchets (both moved with in-commit justification)

- tests/test_code_size.py: 214793 → 215374 exact (+581): the four
  safety fixes are the growth; ledger entry names each fix and why it
  is the cheapest safe design. `the authored codebase did not grow`
  relative to the new ceiling.
- tests/test_complexity_ceiling.py: production MEAN 7.81 → 7.84 (max
  unchanged at 75). Ledger entry names the functions that grew
  (SparkKvCacheArenaEvictResidentBlock, SparkNvmeTierReserveWrite/
  Pump, SparkKvBackingResolvePath, glm5_next KV init) and why each
  added branch is irreducible (degrade-vs-loud, digest verification,
  traversal rejection, one identity guard); every added branch returns
  a status, none nests. `the complexity ceiling held`.
- tools/generate_sha256sums.py + tools/generate_package_manifest.py
  rerun: `package manifest, payload, and checksums match` (repairs the
  stale manifest main carried; no manual edits to either artifact).

## Receipt summary (kimi's re-audit checklist)

| Bug | Fix | Repro/verify |
| --- | --- | --- |
| B1 write-back wedge | degrade on IO/CAPACITY, BACKING_VALID cleared, counter added | 3 tests incl. RLIMIT_FSIZE full-disk repro; 25/25 runs |
| B2 glm5_next arena OOB | machinery sized by DSA ordinals + init fence + compile-time assert + orientation contract | geometry test B2 section PASS; module -fsyntax-only -Werror clean |
| B3 tier checksums | per-slot SHA-256, verified on landing, collision = HASH_MISMATCH, quarantine | tier suite + topology suite green incl. 3 new B3 sections |
| B4 backing hygiene | 0600 + fchmod migration + O_NOFOLLOW + namespaced 0700 paths | backing test ALL PASS incl. migration/symlink/traversal |

Sequencing note (docs/JIT_KV_RESPONSE.md): B1-B4 were the gate for
everything else. B1's degrade path and B3's landing verification are
the exact mechanisms C1/C2 build on; W1's page-directory wiring owns
B2's orientation contract (comment marks the wiring point; the fence
holds the sizes fixed until then).
