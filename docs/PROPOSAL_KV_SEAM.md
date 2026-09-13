# Proposal: the shared KV seam (items 3.1 / 3.2 / 3.7 / 3.8 / 3.9)

KV-cache subsystem agent deliverable. This proposes the model-neutral KV seam
that the four model adopters (DSv4, GLM 5.2, Qwen 3.8, K3) consume, the joint
JIT-KV contract with the SCHEDULER agent, and the per-model migration sequence
with the host tests that pin today's behavior. PROPOSAL only: no shared or model
files are edited. Every claim is grep/read-verified; cited as file:line.

Companion to docs/KVCACHE_SUBSYSTEM_BOUNDARY.md (landed at eb70196). The two
reconstructions confirmed there are used verbatim: "N sparks store 1/N" = PP
slicing + TP head-sharding (docs/QWEN38_MAX_KV.md:122-125), and
"hash-consistent prefix sharding" = chained content hash (cache/cache.h:166-180)
+ fingerprinted rank keys (spark_stage_kv_client.h:40-42).

---

## 0. What "token-free" means here

The DRY gate is tests/test_dry_law.py. It scans the COMMON roots
(include/sparkpipe, cache, runtime, scheduler, node, ring, serving, api, text,
src, deployment, inference/stage, inference/kernels - tests/test_dry_law.py:13-27)
and fails any file whose PATH or CONTENT matches the model-token regex
(glm/glm52, kimi, k3, qwen, dsv4/deepseek, mimo25 - tests/test_dry_law.py:28-32).

Consequence for the seam:

- The seam STRUCTS and FUNCTION SIGNATURES live in the shared headers
  (include/sparkpipe/spark_kv_*.h, spark_nvme_tier.h, spark_prefix_cache.h) and
  must be token-free. They already are: every field is a generic name
  (head_dim, layer_count, bytes_per_scalar, model_id string).
- The per-model VALUES live in model-family headers (model-families/<model>/),
  which the DRY law deliberately exempts ("A model token belongs only in its
  model family, model module, model tool, or model test" -
  tests/test_dry_law.py:3-6).
- The pattern already exists and is the template: spark_k3_kv_geometry.h fills
  SparkKvCacheCapacityRequest, and tests/test_k3_kv_cache.c proves the seam is
  crossed with zero model symbols in the common machinery
  (tests/test_k3_kv_cache.c:1-7).

So the seam is NOT a new invented ABI. It is the EXISTING token-free interfaces,
nominated as the one seam, plus ONE consolidated config table each model fills
(section 1.2), plus the scheduler contract (section 2). That is the whole
proposal.

---

## 1. The model-neutral KV seam API

A model adopter consumes three layers, all token-free, all already in the tree:

    1. the ARENA           spark_kv_cache.h        blocks, residency, prefetch plan
    2. the PAGE DIRECTORY  spark_kv_page_cache.h   sequence<->page bindings, sharing
    3. the PAGE STORE      spark_kv_page_store.h   host backing + copy primitive
       (+ the NVMe TIER    spark_nvme_tier.h       JIT tier-3 lookahead)

The PREFIX CACHE (spark_prefix_cache.h) is consumed by the RUNTIME/SCHEDULER,
not by the model driver (section 2), so it is not part of the model-facing seam.

### 1.1 Exact signatures a model consumes

PAGE DIRECTORY (include/sparkpipe/spark_kv_page_cache.h:103-148). The model
driver passes the neutral SparkModelDriverCacheLane and receives logical page
indices; it never names its own byte layout here.

    SparkStatus SparkKvPageCacheInitialize(
        SparkKvPageCache *cache,
        const SparkKvPageCacheConfiguration *configuration);

    SparkStatus SparkKvPageCachePrepareLane(
        SparkKvPageCache *cache,
        const SparkModelDriverCacheLane *lane,
        uint32_t *logical_page_indices,
        uint32_t logical_page_capacity,
        uint32_t *logical_page_count_out);

    SparkStatus SparkKvPageCacheResolveLanePages(
        const SparkKvPageCache *cache,
        const SparkModelDriverCacheLane *lane,
        uint32_t *logical_page_indices,
        uint32_t logical_page_capacity,
        uint32_t *logical_page_count_out);

    SparkStatus SparkKvPageCacheBeginLaneTransaction(
        SparkKvPageCache *cache,
        const SparkModelDriverCacheLane *lane,
        uint32_t *mutable_logical_page_index_out,
        uint32_t *mutation_flags_out);

    SparkStatus SparkKvPageCacheRollbackLaneTransaction(
        SparkKvPageCache *cache,
        const SparkModelDriverCacheLane *lane,
        uint32_t mutation_flags);

    SparkStatus SparkKvPageCacheCompleteLane(
        SparkKvPageCache *cache,
        const SparkModelDriverCacheLane *lane);

    SparkStatus SparkKvPageCacheReleaseLane(
        SparkKvPageCache *cache,
        uint32_t resident_sequence_slot,
        uint64_t sequence_id);

    SparkStatus SparkKvPageCacheBuildLaneTable(
        SparkKvPageCache *cache,
        uint32_t resident_sequence_slot,
        uint64_t sequence_id,
        uint32_t *logical_page_indices,
        uint32_t logical_page_capacity,
        uint32_t *logical_page_count_out);

(Also SparkKvPageCacheGetLaneMutablePageDemand :118-121 and the non-transactional
SparkKvPageCacheBeginLane :122-125; a migration can start on the non-transactional
pair and adopt the transaction pair once prefill is enabled.)

PAGE STORE (include/sparkpipe/spark_kv_page_store.h:38-121). The model supplies
exactly one thing: the copy primitive that moves ITS physical page between device
and host. Everything else is common.

    typedef SparkStatus (*SparkKvPageStoreCopyFunction)(
        void *context,
        uint32_t direction,          /* SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST
                                        or COPY_HOST_TO_DEVICE, :18-19 */
        uintptr_t device_address,
        void *host_address,
        uint64_t bytes);

    SparkStatus SparkKvPageStoreInitialize(
        SparkKvPageStore *store,
        const SparkKvPageStoreConfiguration *configuration);

    SparkStatus SparkKvPageStoreBuildPath(
        char *path,
        uint32_t path_capacity,
        const char *backing_directory,
        const char *model_id,
        const char *model_revision,
        const char *node_id,
        uint32_t stage_index);

    SparkStatus SparkKvPageStoreWriteback(
        void *context,
        uint32_t logical_page_index,
        uint32_t physical_page_index,
        uint64_t generation,
        uintptr_t key_device_address,
        uint64_t key_bytes,
        uintptr_t value_device_address,
        uint64_t value_bytes);

    SparkStatus SparkKvPageStorePrefetch(
        SparkKvPageStore *store,
        SparkKvCacheArena *arena,
        uint32_t logical_page_index);

    SparkStatus SparkKvPageStoreProgress(
        SparkKvPageStore *store,
        SparkKvCacheArena *arena,
        uint32_t maximum_job_count);

    SparkStatus SparkKvPageStoreInvalidate(
        SparkKvPageStore *store,
        uint32_t logical_page_index,
        uint64_t generation);

The page-store header states the ownership contract outright: "The cache decides
what to evict and when to fetch it; a model driver supplies only the copy
primitive" (spark_kv_page_store.h:31-37).

NVMe TIER (include/sparkpipe/spark_nvme_tier.h:261-364). A model never calls the
decode-path RequestDemand directly; the SCHEDULER owns lookahead planning
(section 2). The tier is consumed by the common paged-cache path and by the
topology switch, not by model firmware.

    SparkStatus SparkNvmeTierPlanLookahead(
        SparkNvmeTier *tier,
        const SparkNvmeTierNeed *needs,
        uint32_t need_count,
        uint32_t step_now,
        SparkNvmeTierPlanReport *report_out);         /* :317-322 */

    SparkStatus SparkNvmeTierWillBeResidentBy(
        const SparkNvmeTier *tier,
        const uint64_t *content_hashes,
        uint32_t hash_count,
        uint32_t step_now,
        uint32_t step_deadline,
        SparkNvmeTierResidencyAssessment *assessment_out);  /* :354-360 */

    SparkStatus SparkNvmeTierReserveWrite / CommitWrite / AbortWrite /
                 OffsetOf / RequestDemand / Pump / Consume / Pin
                                             /* :277-351 */

ARENA (include/sparkpipe/spark_kv_cache.h:440-589) is the block/residency
substrate the page directory and page store build on; a model touches it only to
resolve a physical page view after the page directory hands back a logical page:

    SparkStatus SparkKvCacheArenaResolveBlock(
        const SparkKvCacheArena *arena,
        uint32_t logical_block_index,
        SparkKvCacheBlockView *block_view);          /* :565-568 */

    SparkStatus SparkKvCacheEstimateCapacity(
        const SparkKvCacheCapacityRequest *request,
        SparkKvCacheCapacityEstimate *estimate);     /* :440-442 */

    SparkStatus SparkKvCacheCalculateJitStageBudget(
        const SparkKvJitStageBudgetRequest *request,
        SparkKvJitStageBudget *budget);              /* :446-447 */

The neutral lane/identity types the above consume are already token-free:

    typedef struct SparkModelDriverCacheIdentity { uint8_t sha256[32]; }
                                            /* spark_model_driver.h:84-87 */

    typedef struct SparkModelDriverCacheLane {
        uint64_t sequence_id, sequence_position, request_generation,
                 step_generation;
        uint32_t resident_sequence_slot, context_token_count, prefix_token_count,
                 publish_token_count, flags, reserved;
        SparkModelDriverCacheIdentity prefix_identity, publish_identity;
    }                                       /* spark_model_driver.h:94-108 */

### 1.2 The per-model config table (one token-free struct, four model fills)

Proposal: nominate ONE consolidated, token-free config struct as the fill point,
so each model ships a single "here is my KV" table and the common core consumes
it uniformly. It is a thin wrapper over the three config structs that already
exist; no field moves, nothing is re-invented.

Proposed token-free struct (new, lives in include/sparkpipe/spark_kv_model_table.h;
name every field generically so test_dry_law passes):

    typedef struct SparkKvModelTable {
        /* geometry -> sizing */
        SparkKvCacheCapacityRequest   capacity_request;   /* spark_kv_cache.h:53-72 */
        /* arena block geometry + resident-slot geometry */
        SparkKvCacheConfiguration     arena_configuration;/* spark_kv_cache.h:350-369 */
        /* host backing + the one model-specific function */
        SparkKvPageStoreConfiguration page_store_config;  /* spark_kv_page_store.h:45-61 */
        SparkKvPageStoreCopyFunction  copy_function;
        void                         *copy_context;
        /* page directory capacities */
        uint32_t sequence_capacity;
        uint32_t entry_capacity;
        uint32_t hash_bucket_count;
        /* identity + backing */
        const char *model_id;
        const char *model_revision;
        const char *cache_layout_fingerprint;  /* text form for the store key */
    } SparkKvModelTable;

The common core gains one entry point (token-free):

    SparkStatus SparkKvModelTableValidate(const SparkKvModelTable *table);
    SparkStatus SparkKvBackendInitialize(
        const SparkKvModelTable *table,
        SparkKvCacheArena *arena,
        SparkKvPageCache *page_cache,
        SparkKvPageStore *page_store);

Each model fills SparkKvModelTable in ITS OWN header (model-families/<model>/),
where naming the model is legal. The per-model fill table is:

    model     layout                      block tok  layers      head_dim   bytes/sc  copy primitive
    ----      ------                      --------   ------      --------   --------  --------------
    dsv4      per-layer pool layout       128        43(+MTP)   layer-kind 2 (bf16)  SparkDsv4PageCopy
             (sliding/CSA/HCA)
    k3        COMPRESSED_KEY_VALUE        64         24 MLA     576 (512+64) 2       SparkK3PageCopy
             (MLA latents)                           +69 KDA slab
    glm52     COMPRESSED_KEY_VALUE_FP8    64/128     78         576 (512+64) 2/1(fp8) SparkGlm52PageCopy
             (latent 512 + rope 64)
    qwen38    FULL_KEY_VALUE              64         23 attn    2048        2       SparkQwen38PageCopy
             (K+V head-major, 4 KV heads)            +69 GDN

The exact constants to pin (all already in-tree, all model-side):

- dsv4: block = SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS 128
  (spark_dsv4_resident_decode_stage_firmware.h:17); codec =
  SPARK_DSV4_MODEL_KV_CACHE_CODEC = BF16 (spark_dsv4_model.h:68); per-layer
  sliding/compressed/indexer/compressor offsets (spark_dsv4_pool_layout.h:16-40).
- k3: layout = SPARK_K3_KV_LAYOUT = COMPRESSED_KEY_VALUE; latent 512, rope 64,
  bytes_per_scalar 2, 24 MLA + 69 KDA (spark_k3_kv_geometry.h:15-25); KDA slab
  sized by kernel config (spark_k3_kv_geometry.h:33-47).
- glm52: latent 512, rope 64 -> KV_A 576 (spark_glm52_model.h:11-12,88-90);
  FP8 scale block 128 (spark_glm52_model.h:21); heads 64, qk_nope 192, value
  head 256 (spark_glm52_model.h:10,14-15); KV_POOL_TOKENS 4194304 (:8).
- qwen38: 64 attn heads / 4 KV heads / head dim 256 (spark_qwen38_model.h:11-13);
  23 full-attention layers + 69 GDN (:27-28); token record = 2 x KV_DIM = 2048
  elements = 4 KiB/token/layer, head-major K then V (spark_qwen38_model.h:53-54;
  docs/QWEN38_MAX_KV.md:60-64).

Why a single table and not four ad-hoc fills: today only k3 ships a fill helper
(SparkK3KvFillCapacityRequest, spark_k3_kv_geometry.h:49-62); dsv4 inlines the
config in its paged-cache init (spark_dsv4_paged_cache.c:102-138), and
glm52/qwen38 ship nothing. One table makes the four adopters symmetric and makes
the "same tables" requirement in the coordinator's note mechanically true.

### 1.3 Expressing DRIVER_OWNS_KV with the same tables

The coordinator note requires: the seam must let glm52/qwen38/k3 (all currently
DRIVER_OWNS_KV, none JIT-KV - verified in KVCACHE_SUBSYSTEM_BOUNDARY.md 1.10)
express their path with the SAME tables as DSv4's JIT-KV path.

The two paths differ in exactly ONE bit of ownership, and both read the same
SparkKvModelTable:

- JIT-KV (DSv4): the RUNTIME owns the arena + page directory + prefix cache; the
  adapter declares cache_block_token_count != 0 (the JIT_KV gate,
  runtime/model_serving_adapter.c:97-100). The model fills the table's geometry
  and copy primitive and maps pages (spark_model_serving_adapter.h:146-149:
  "a JIT-KV driver only maps each page into its model-specific byte layout").
- DRIVER_OWNS_KV (glm52/qwen38/k3 today): the MODEL owns its KV arena; the
  runtime still owns admission and page budgets. The model fills the SAME table
  for geometry + byte layout + copy primitive, and the runtime reads the same
  fields for admission (page budgets come from SparkModelServingRuntimeLimits,
  not from the model).

The single structural requirement is: SparkKvModelTable MUST NOT be coupled to
the JIT_KV capability bit. It carries geometry and mechanics; the capability
bit only selects WHO instantiates the arena (runtime vs driver). That is the
whole DRIVER_OWNS_KV vs JIT-KV delta, and it is expressible by a table, not a
code fork.

---

## 2. JIT-KV joint contract with the SCHEDULER agent (INTERFACE)

This section is written as an INTERFACE: it states the split and the call
boundary, and it flags the three decisions to confirm with the scheduler agent.
It does NOT decide them (the scheduler agent is drawing its admission core in
parallel; this doc is the offer of the KV side of the boundary).

### 2.1 The scheduler owns: admission, page budgets, lookahead, prefix reuse

These are already scheduler-side today; the proposal is to keep them there and
make the boundary explicit.

1. Page budgets. The scheduler/runtime owns kv_logical_page_capacity and
   kv_physical_page_capacity (SparkModelServingRuntimeLimits,
   spark_model_serving_adapter.h:146-152). They flow: deployment spec ->
   model_resident_deployment.c:218-221 -> model_resident_client.c:942-945 ->
   model_batch_engine.c:1020-1021 -> model firmware via
   driver_compiler.c:588-589 (host_services.kv_logical/kv_physical_page_capacity).
   The KV core READS these, never sets them.

2. Admission arithmetic. The scheduler owns these pure functions (all in
   runtime/model_batch_engine.c, all take page budgets as INPUT, none touch KV
   bytes):
   - SparkModelBatchSchedulerRequestFitsPageCapacity(block_token_count,
     physical_page_capacity, prompt_token_count, output_token_budget) :205-221;
   - SparkModelBatchSchedulerCacheDemandFits(physical_page_capacity,
     used_page_count, additional_page_count) :193-203;
   - SparkModelBatchSchedulerPlanCacheBoundLaneCount(maximum_lane_count,
     physical_page_capacity, inflight_page_count) :180-191;
   - SparkModelBatchSchedulerPlanGroupSize :160-178, PlanMixedLaneCount :223-242,
     ChooseWorkKind :244-259.

3. Prefix reuse. The scheduler allocates and initializes the prefix cache from
   the adapter's cache_block_token_count (runtime/model_batch_engine.c:930-972),
   then probes/reserves/commits it (spark_prefix_cache.h:205-295). The KV core
   provides the hash index; the scheduler decides which prefix to reuse and when
   to publish (model_batch_engine.c:699-711, 1697-1707).

4. NVMe lookahead. The scheduler (topology switch) turns "what runs next" into
   SparkNvmeTierNeed entries and calls SparkNvmeTierPlanLookahead
   (scheduler/topology_switch.c:502-523). The tier performs the schedule
   arithmetic; the scheduler owns the schedule.

5. Prefetch resolution. The runtime drives adapter prefetch during distributed
   PREPARE and resolves it exactly once (COMMIT/ABORT) -
   SparkModelServingAdapterResolvePrefetch (runtime/model_serving_adapter.c:442-467,
   gated on JIT_KV+PREFETCH at :455-459). The adapter's prefetch/resolve_prefetch
   function pointers are the model-facing half (spark_model_serving_adapter.h:309-321).

### 2.2 The KV core owns: mechanics, never admission

For every scheduler decision above, the KV core exposes the MECHANISM, returns a
STATE, and makes no admission call:

    scheduler decision                 KV-core mechanism (query, no side-effect on admission)
    -------------------                ------------------------------------------------
    "does this request fit pages?"     SparkKvCacheArenaUnassignedResidentBlockCount
                                       + SparkKvCacheBlocksAvailable (spark_kv_cache.h:459-469;
                                       cache/cache.h:182-198 for the sizing formula)
    "is this prefix resident?"         SparkPrefixCacheProbeReusablePrefixResidency
                                       (spark_prefix_cache.h:238-244)
    "will these blocks be upstairs?"   SparkNvmeTierWillBeResidentBy
                                       (spark_nvme_tier.h:354-360)
    "what should I fetch next?"        SparkKvCacheArenaBuildPrefetchPlan /
                                       BuildPrefetchPlanFromSourceBlocks
                                       (spark_kv_cache.h:512-535)
    "evict to fit"                     SparkKvCacheArenaTrimResidentBlocks /
                                       EvictResidentBlocksToLimit (spark_kv_cache.h:551-563),
                                       SparkPrefixCacheTrimResidentBlocksByReuseScore
                                       (spark_prefix_cache.h:257-262)

The invariant that must hold through both agents' work: THE SCHEDULER NEVER
READS A MODEL'S BYTE LAYOUT, AND THE KV CORE NEVER CHOOSES ADMISSION. This is
already written into the architecture: "The orchestrator does not understand
... KV layout ... or JIT-KV policy. Those details belong inside model firmware"
(SPEC.md section 6), and "a JIT-KV driver only maps each page into its
model-specific byte layout" (spark_model_serving_adapter.h:146-149).

### 2.3 The three points to agree with the scheduler agent (not decided here)

- (A) Whether SparkKvModelTable (section 1.2) is consumed by the scheduler or
  only by the runtime resident/driver loader. Offer: the scheduler reads only the
  page budgets + cache_block_token_count; the table is consumed by the
  resident's backend initializer (SparkKvBackendInitialize), keeping the
  scheduler token-free.
- (B) Whether admission should call SparkNvmeTierWillBeResidentBy directly
  (confidence = ALL/PARTIAL/NONE, spark_nvme_tier.h:180-198) or go through the
  batch engine's pure fit functions only. Offer: keep WillBeResidentBy as a
  scheduler-side admission query against the tier, since the tier is already
  consumed by the topology switch (scheduler/topology_switch.c:523).
- (C) Who owns "evict to make room" under pressure: the scheduler names the
  target resident count and the protected set; the KV core performs the eviction
  (SparkKvCacheArenaTrimResidentBlocks / EvictResidentBlocksToLimit take
  protected_logical_block_indices from the CALLER - spark_kv_cache.h:551-563).
  Offer: scheduler supplies the protected set (the lookahead reservation), core
  picks victims by reuse score. This is the existing signature; confirm it stays.

---

## 3. Migration sequence per model (3.3-3.6) + pinning host tests

Order matters: finish the seam (section 1) and confirm the scheduler contract
(section 2) before any model moves, so the four adopters land against one
table, not four.

### 3.3 DSv4 (reconcile its two cache systems) - DSV4 model agent owns

Target: exactly one DSv4 page layout on top of the common arena. Today DSv4 has
both the model-specific spark_dsv4_cache_plan.c/.h + spark_dsv4_cache_arena.c/.h
(sliding/compressed/indexer/compressor arenas, spark_dsv4_cache_plan.h:52-86) and
the common-core consumer spark_dsv4_paged_cache.c/.h (wraps SparkKvCacheArena +
SparkKvPageCache, spark_dsv4_paged_cache.h:36-56). Decide which is the offline
sizing calculator (cache plan) and which is the serving page-layout mapping
(paged cache), then have the paged cache fill SparkKvModelTable like every other
model.

Pinned by: tests/test_dsv4_cache_plan.c (arena placement validation,
SparkTestValidateArenaPlacement), tests/test_dsv4_paged_cache.c (identity + lane),
tests/test_dsv4_serving_adapter.c / test_dsv4_tp16_serving_adapter.c /
test_dsv4_tp4_pp4_serving_adapter.c (the JIT_KV capability flag is asserted at
tests/test_dsv4_tp16_serving_adapter.c:56 and test_dsv4_tp4_pp4_serving_adapter.c:129).

### 3.4 GLM 5.2 (JIT-KV onto the common core) - GLM52 model agent owns

Target: replace the historical model-local JIT-KV backend (docs/archive/
GLM52_B1024_JIT_KV_INTEGRATION.md) with the common paged cache + NVMe tier.
GLM52's adapter is DRIVER_OWNS_KV only today (spark_glm52_serving_adapter.c:124).
Steps: (1) add a glm52 fill of SparkKvModelTable (latent 512 + rope 64 -> KV_A
576, FP8 scale block 128 - spark_glm52_model.h:11-12,21,88-90); (2) add the
glm52 copy primitive; (3) flip the adapter to JIT_KV + PREFETCH and set
cache_block_token_count, which the runtime gates (runtime/model_serving_adapter.c:97-100).

Pinned by: tests/test_kv_cache.c (the arena/page-store/prefix-cache mechanics
that become glm52's substrate), tests/test_nvme_tier.c (the JIT tier), and the
NVMe sizing cross-check tests/test_nvme_kv_estimate.py. No GLM52-specific KV host
test exists yet because the JIT-KV backend was archive-only; the new one should
mirror test_k3_kv_cache.c ("the seam, crossed").

### 3.5 Qwen 3.8 (+ qwen38_27b reference) - Qwen38 model agent owns

Target: adopt SparkKvPageStore + SparkKvStore, and finish TP 1/N head-sharding.
Today qwen38 links stage_kv_client.c + kv_store.c but the adapter is
DRIVER_OWNS_KV and the module ignores the page budgets (docs/QWEN38_MAX_KV.md:29-35).
Steps: (1) fill SparkKvModelTable for the 23 full-attention layers (2 x 4 heads x
256 x bf16 = 2048 elements/token, spark_qwen38_model.h:53-54); (2) wire the
store client (qwen38_27b's SparkQwen38_27bModuleOpenKvTier is the complete reference,
docs/QWEN38_MAX_KV.md:42-48); (3) activate head-parallel 1/tp_degree KV sharding
(currently refused at tp_degree > 1, docs/QWEN38_MAX_KV.md:127-130).

Pinned by: tests/test_kv_store.c (store contracts + lookahead),
tests/test_kv_mooncake.cpp (provider PUT/GET roundtrip), and the qwen38_27b reference
tests/test_qwen38_27b_serving_adapter.c. The qwen38 work-control tests landed in the
audit (QWEN38_MAX_KV.md:108-110, test_qwen38_work_control).

### 3.6 K3 (wire the capacity path; geometry header is already correct) - K3 model agent owns

Target: make SparkKvCacheEstimateCapacity the single sizing authority and retire
the stub serving adapter. K3's geometry header already fills
SparkKvCacheCapacityRequest (spark_k3_kv_geometry.h:49-62) and its pool sizing
consumes it (modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_pool_sizing.h),
but the serving adapter is a stub (capability_flags = 0, kv_cache_codec = 0,
cache_block_token_count = 0 - spark_k3_serving_adapter.c:451,461,478). K3 is the
cleanest adoption: fill SparkKvModelTable from the geometry header, keep the KDA
slab in the slot pool (not the token arena - spark_k3_kv_geometry.h:7-13).

Pinned by: tests/test_k3_kv_cache.c (the reference "seam crossed" test),
tests/test_k3_kv_geometry.py (serving-vs-kernel cross-check), tests/test_k3_pool_sizing.c,
tests/test_k3_serving_adapter_smoke.c.

### Host-test catalog (the pinning set for the whole migration)

    test                                     pins                                              line ref
    ----                                     ----                                              --------
    tests/test_kv_cache.c                    arena eviction/backpressure, prefetch plans,      :93-969
                                             page-store dirty-write/restore + direct-IO,
                                             prefix-cache reuse/dedup/transactional
    tests/test_nvme_tier.c                   budget, JIT lookahead, churn eviction,            :1-8
                                             demand-vs-prefetch priority (mock device)
    tests/test_kv_store.c                    store ABI contracts + pressure-limited lookahead  :66-88
    tests/test_kv_mooncake.cpp               provider PUT/GET block roundtrip                  :29
    tests/test_k3_kv_cache.c                 "the seam, crossed" (k3 on common machinery)      :1-7
    tests/test_k3_kv_geometry.py             serving geometry == kernel config                 :30-54
    tests/test_k3_pool_sizing.c              k3 pool sizing against geometry
    tests/test_cache.c                       LmCache sharing/eviction/CoW/JIT reservation      :1-5
    tests/test_dsv4_cache_plan.c             dsv4 cache-plan arena placement
    tests/test_dsv4_paged_cache.c            dsv4 paged-cache identity + lane
    tests/test_model_serving_adapter.c       JIT_KV/PREFETCH/DRIVER_OWNS_KV capability gates   :231-306
    tests/test_dry_law.py                    token-free gate for every new shared file          :35-55

The migration acceptance rule: every model lands with (a) its own fill of
SparkKvModelTable, (b) a test modeled on test_k3_kv_cache.c proving zero model
symbols leaked into the common machinery, and (c) test_dry_law.py still green.

---

## What I need from the scheduler agent and the coordinator

- Scheduler agent: confirm or counter the three offers in section 2.3 (table
  consumption point, admission query against WillBeResidentBy, protected-set
  ownership). This doc is the KV side; the split is not final until the scheduler
  agent agrees.
- Coordinator: approve SparkKvModelTable as the single token-free fill point
  (section 1.2) so the four model agents have one target; land the common
  SparkKvBackendInitialize helper once the scheduler contract is agreed.

---

## Coordinator note (2026-08-17)

SparkKvModelTable as the single token-free fill point is APPROVED in
principle (it matches the repo's table-driven pattern); it becomes
landable once the scheduler agent confirms or counters the three 2.3
offers. The 2.3 A/B/C sign-off is routed to the scheduler agent.

---

## Scheduler sign-off (2026-08-17, recorded by coordinator)

- (A) CONFIRMED: scheduler consumes only page budgets +
  cache_block_token_count; SparkKvModelTable lives in
  SparkKvBackendInitialize in the resident/driver loader.
- (B) COUNTERED (narrow): SparkNvmeTierWillBeResidentBy stays
  scheduler-side as a LOOKAHEAD/PLANNING query (topology-switch resume +
  tier prefetch), NOT an admission query. Admission remains on the batch
  engine's pure fit functions; ALL/PARTIAL/NONE is prefetch-timing
  confidence, never a capacity gate.
- (C) CONFIRMED: scheduler supplies target resident count + protected set;
  KV core picks victims by reuse score with
  protected_logical_block_indices from the caller.

SparkKvModelTable is now FULLY APPROVED as the single fill point.
