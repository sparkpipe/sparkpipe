# KV-Cache Subsystem Boundary

Subsystem agent deliverable. This document inventories every KV mechanism on the
unified branch, states the boundary contract between common machinery, per-model
page layouts, and scheduler/eviction policy, and lays out a consolidation plan
with owners. Every claim is grep/read-verified against the tree at
/Users/mac/dsh.sparkpipe/.agents/kv-cache and cited as file:line.

Scope note: this is a PROPOSAL document. No shared or model files are edited
here (charter lane: SUBSYSTEM agents review all of it, propose the shared
structure, and land through the coordinator).

---

## 1. Inventory of every KV mechanism on unified

There are TWO generations of KV machinery in the tree, plus a pluggable
distributed backing tier, plus per-model geometry headers. They are not all
wired to each other yet; section 1 records what exists, section 2 records who
owns what.

### 1.1 cache/cache.h - the inline LmCache reference arena (legacy spec)

cache/cache.h (723 lines) is a self-contained single-header implementation, not
a .c file. It is the canonical written-down spec of the cache's core ideas:

- "One arena, content-addressed sharing, JIT admission" - cache/cache.h:3.
- Two independent block states, REFERENCED vs RESIDENT - cache/cache.h:49-56.
- Chained content hash: a block matches only when the ENTIRE prefix before it
  matches (LmCacheHashBlock, cache/cache.h:166-180).
- LRU weighted by reuse count (LmCacheScore, cache/cache.h:238-251).
- JIT admission reserve (LmCacheReserve, cache/cache.h:431-449).
- Residency separate from pool eviction (cache/cache.h:476-505).
- Prefetch lanes + plan (cache/cache.h:603-688).

Who consumes it: only tests/test_cache.c includes the struct definitions
(tests/test_cache.c:6). But two production headers still anchor on its chained
content hash: include/sparkpipe/spark_nvme_tier.h:8,83 ("the same chained hash
cache/cache.h uses") and include/sparkpipe/spark_topology_switch.h:22 ("the
chained token-run hash from cache/cache.h"). So it is NOT dead - it is the
reference for the content-hash identity, but its allocator/residency logic is
duplicated by the real implementation below.

### 1.2 cache/kv_cache.c + include/sparkpipe/spark_kv_cache.h - the model-neutral arena (the real core)

The paged-cache arena: uniform opaque blocks, allocation/recycling, residency
slots, and prefetch planning. It is model-neutral; geometry arrives as
parameters.

- ABI v6 - spark_kv_cache.h:12.
- Block flags ALLOCATED/RESIDENT/RESIDENCY_RESERVED/DIRTY/BACKING_VALID -
  spark_kv_cache.h:24-28.
- SPARK_KV_CACHE_MAX_BLOCK_TOKENS 256 - spark_kv_cache.h:33.
- Layout vocabulary (the per-model precision/format axis): FULL_KEY_VALUE,
  COMPRESSED_KEY_VALUE, COMPRESSED_KEY_VALUE_FP8_E4M3, FULL_KEY_VALUE_FP8_E4M3 -
  spark_kv_cache.h:37-40.
- SparkKvCacheCapacityRequest (geometry in) and SparkKvCacheCapacityEstimate
  (sizing out) - spark_kv_cache.h:53-90.
- SparkKvCacheCalculateJitStageBudget (geometry->arena/backing budget) -
  spark_kv_cache.h:446-447, impl cache/kv_cache.c:154.
- SparkKvCacheEstimateCapacity - spark_kv_cache.h:440-442, impl
  cache/kv_cache.c:365.
- Arena block/recycle/retain/resident/pin/prefetch-plan API -
  spark_kv_cache.h:471-565.

Consumers: spark_kv_page_store.h, spark_prefix_cache.h, spark_kv_page_cache.h,
modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.h,
model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h, and tests.

### 1.3 cache/kv_page_cache.c + spark_kv_page_cache.h - model-neutral logical page directory

The header states the boundary contract in its own words:

> "Model-neutral logical page directory. It owns sequence-to-page bindings,
> immutable prefix sharing, and logical-page lifetime. Model drivers choose the
> arena block geometry and translate resident page slots into their own
> physical KV payload layout." - spark_kv_page_cache.h:23-28.

- Identity hash = FNV over sha256 identity + token_count -
  cache/kv_page_cache.c:7-24.
- Entry / Sequence structs - spark_kv_page_cache.h:30-54.
- Lane prepare/resolve/begin/complete/release + mutation transaction -
  spark_kv_page_cache.h:106-148.

Consumers: spark_dsv4_paged_cache.h, tests/test_kv_cache.c.

### 1.4 cache/kv_page_store.c + spark_kv_page_store.h - device-neutral page backing (host tier)

> "Device-neutral backing for opaque fixed-size KV pages. The cache decides what
> to evict and when to fetch it; a model driver supplies only the copy primitive
> needed to move its physical page between device and host memory." -
> spark_kv_page_store.h:31-37.

- Copy function contract - spark_kv_page_store.h:38-43.
- ANONYMOUS (volatile scratch) vs DIRECT_IO flags - spark_kv_page_store.h:21-27.
- Per-stage backing path keyed by model_id/model_revision/node_id/stage_index -
  spark_kv_page_store.h:92-99.
- Worker threads, backing pages, write-back/prefetch/progress -
  cache/kv_page_store.c:23-57.

Consumers: spark_kv_page_cache.h,
modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c.

### 1.5 cache/nvme_tier.c + spark_nvme_tier.h - Tier 3 JIT NVMe tier

> "Tier 3 of the KV cache: the NVMe tier, and it is a JIT tier." -
> spark_nvme_tier.h:3. Tier 1 = device memory, tier 2 = host memory, tier 3 =
> NVMe (spark_nvme_tier.h:5-8).

- JIT, not a cache of last resort: reads the admission schedule, issues reads
  early enough that transfer time hides inside compute - spark_nvme_tier.h:14-25.
- Clock (second-chance) eviction, generation-bumped, never a write-back on the
  decode path - cache/nvme_tier.c:5-27.
- Default budget 1 TB (spark_nvme_tier.h:50); bandwidth/step-time defaults
  spark_nvme_tier.h:61-69.
- Async block-device vtable (submit/poll/cancel) - spark_nvme_tier.h:109-140.
- Demand states (READY/IN_FLIGHT/STARTED/MISS) - spark_nvme_tier.h:146-153.
- Admission interface SparkNvmeTierWillBeResidentBy - spark_nvme_tier.h:354-360.

Consumers: spark_topology_switch.h, tests/test_nvme_tier.c. No model module
consumes it yet directly.

### 1.6 cache/prefix_cache.c + spark_prefix_cache.h - content-addressed prefix cache

The prompt-prefix reuse index, owned by the runtime batch engine (see 2.3).

- ABI v7 - spark_prefix_cache.h:12.
- Chained entry hash = parent_hash + block_hash + content_hash -
  spark_prefix_cache.h:59-65, SparkPrefixCacheHashBlock :193-196.
- Probe / Reserve / Commit / Cancel / CommitPrompt - spark_prefix_cache.h:205-295.
- Lookahead protection (SparkPrefixCacheProtectPromptLookahead) and reuse-scored
  resident eviction (SparkPrefixCacheTrimResidentBlocksByReuseScore) -
  spark_prefix_cache.h:249-262.

Consumers: runtime/model_batch_engine.c, tests/test_kv_cache.c.

### 1.7 cache/store/kv_store.c + spark_kv_store.h - pluggable distributed KV store interface

The family backing tier's provider ABI (the Mooncake slot-in).

- SparkKvStoreGetInterface symbol - spark_kv_store.h:12-13.
- Required caps: BATCH_GET | BATCH_PUT | PERSISTENT_SERVICE | PROVIDER_BUFFERS -
  spark_kv_store.h:28-36.
- Configuration carries model_fingerprint and cache_layout_fingerprint plus
  rank/layer slice - spark_kv_store.h:41-57.
- Interface (initialize/destroy/submit/poll/allocate/release) -
  spark_kv_store.h:107-119.
- Pressure-limited lookahead selector - spark_kv_store.h:130-139.

### 1.8 cache/store/stage_kv_client.c + spark_stage_kv_client.h - stage-module KV client

One instance per stage; loads the provider .so, builds fingerprinted keys,
submits/polls batches.

- Key binds model_fingerprint, cache_layout_fingerprint, rank_index, sequence_id,
  logical_block - spark_stage_kv_client.h:40-42; format
  SparkStageKvClientFormatKey :41.
- "The residency DECISION - which packets prefetch when - belongs to the runtime
  work-control layer ... this client is the provider plumbing that layer drives"
  - spark_stage_kv_client.h:24-28.
- Enablement is EXPLICIT ("none" or provider .so), no inferred default -
  spark_stage_kv_client.h:20-22.

### 1.9 modules/kv_mooncake/ - the Mooncake provider

spark_kv_mooncake.cpp (437 lines) implements SparkKvStoreInterface over
mooncake::DummyClient:

- Uses mooncake::DummyClient (setup_dummy) - spark_kv_mooncake.cpp:229-237.
- PUT pins a SINGLE replica on the local host (replica_num = 1u, preferred local
  hostname) - spark_kv_mooncake.cpp:141-147.
- Retries only NO_AVAILABLE_HANDLE, bounded at 1200 x 10 ms -
  spark_kv_mooncake.cpp:151-182.
- Exports SparkKvStoreGetInterface - spark_kv_mooncake.cpp:434-437.

modules/kv_mooncake/Makefile builds build/libkv_mooncake.so and fails closed
unless MOONCAKE_ROOT and the real Mooncake headers/libs are present
(Makefile:12,18-25). The deployment story is in docs/archive/mooncake_kv.md:
the real mooncake_client survives SparkPipe resident/module releases, so a
restart does not discard Mooncake state (mooncake_kv.md:14-15); keys include
fingerprints so a model/layout change cannot consume stale KV
(mooncake_kv.md:17-18).

### 1.10 JIT-KV flags

- Adapter capability: SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV
  UINT32_C(0x00000100) - spark_model_serving_adapter.h:62.
- Driver program flag: SPARK_MODEL_DRIVER_PROGRAM_FLAG_JIT_KV_CACHE 0x00000010 -
  spark_model_driver.h:23.
- Companion flags: SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV 0x80
  (spark_model_serving_adapter.h:60-61), SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE
  0x8 (spark_model_driver.h:22).
- The model-description JSON string "jit_kv_cache" maps to the program flag in
  runtime/pack/model_description.c:149-151.

Cross-field invariants enforced by the runtime (this is the real gate):

- cache_block_token_count != 0  iff  JIT_KV capability -
  runtime/model_serving_adapter.c:97-98.
- cache_block_token_count != 0  implies  PREFETCH | DRIVER_OWNS_KV -
  runtime/model_serving_adapter.c:99-100.
- For JIT_KV, page budgets must satisfy physical >= active_sequences,
  logical >= resident_sequences, physical <= logical; for non-JIT_KV both must
  be 0 - runtime/model_serving_adapter.c:163-176.
- resolve_prefetch is required iff JIT_KV + PREFETCH -
  runtime/model_serving_adapter.c:455-459.

Only one adapter declares JIT_KV today: DSv4
(modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:282,
with cache_block_token_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS
at :305-306). glm52, qwen38, qwen36 declare DRIVER_OWNS_KV only, and k3's
serving adapter is a stub (capability_flags = 0u, kv_cache_codec = 0u,
cache_block_token_count = 0u - spark_k3_serving_adapter.c:451,461,478).

### 1.11 Per-model KV geometry

- DSv4 - the only JIT-KV model. Block = 128 tokens
  (spark_dsv4_resident_decode_stage_firmware.h:17). Its paged cache wraps the
  COMMON arena + page cache + a dsv4-specific pool layout
  (spark_dsv4_paged_cache.h:36-56). DSv4 ALSO carries its own model-specific
  cache plan/arena (see 1.12): sliding/compressed/heavily-compressed attention
  classes with per-layer sliding/compressed/indexer/compressor offsets
  (spark_dsv4_cache_plan.h:24-30,52-86).
- K3 - spark_k3_kv_geometry.h is the cleanest example of the boundary: "K3's
  cache geometry for the common kv machinery, in the machinery's own request
  vocabulary. Two caches, two allocators: the 24 MLA layers page compressed
  latents through the token arena; the 69 KDA layers keep fixed-size recurrent
  state in a slot pool" (spark_k3_kv_geometry.h:7-13). It fills
  SparkKvCacheCapacityRequest (spark_k3_kv_geometry.h:49-62) and is
  cross-checked against inference/llms/kimi_k3/config.h by
  tests/test_k3_kv_geometry.py (:11-13). 1M context (spark_k3_model.h:32). The
  serving adapter is a stub (1.10).
- GLM 5.2 - 1M context (spark_glm52_model.h:7), KV_POOL_TOKENS 4194304 (:8), DSA
  selected 2048 tokens (:22), FP8 scale block 128 (:21). Adapter is
  DRIVER_OWNS_KV only (spark_glm52_serving_adapter.c:124). Its JIT-KV work is
  historical: docs/archive/GLM52_B1024_JIT_KV_INTEGRATION.md and
  docs/archive/GLM52_JIT_KV_BACKEND_AND_BLACKWELL_QKVO_20260702.md, and the
  measured status recorded JIT KV prefetch as NOT_WORKING with zero counters
  (docs/archive/GLM52_MEASURED_STATUS.md:75).
- Qwen 3.8 - docs/QWEN38_MAX_KV.md is the audit. Geometry: 64 attn heads / 4 KV
  heads / head dim 256 (spark_qwen38_model.h:11-13), 69 GDN + 23 full-attention
  layers (:27-28), 262144 native ctx (:19). Adapter is DRIVER_OWNS_KV +
  HIDDEN_TRANSPORT, no JIT_KV (spark_qwen38_serving_adapter.c:204). The audit's
  verdicts (JIT PARTIAL, sharded PARTIAL, exactness PARTIAL, dedicated ring NO)
  are at QWEN38_MAX_KV.md:9-14; the TP 4x-replication finding at :50-58; the
  head-parallel 1/tp_degree fix at :122-125.
- MiMo 2.5 - full + SWA hybrid, 4 full KV heads / 8 SWA KV heads, head dim 192
  (q) / 128 (v) (spark_mimo25_model.h:61-72), 1M positions (:57). Not a support
  target (ARCHITECTURE.md:211). No serving adapter in-tree.
- Qwen 3.6 - the complete reference for the stage KV store path
  (SparkQwen36ModuleOpenKvTier), per QWEN38_MAX_KV.md:42-48.

### 1.12 The storage contract (2.5 TB hot KV per spark; N sparks store 1/N)

The authoritative storage hierarchy is ARCHITECTURE.md:151-175:

- Every Spark: 4 TB internal NVMe + at least 4 TB external NVMe
  (ARCHITECTURE.md:153-154).
- Internal partition - 2.5 TB hot KV cache and resumable request state
  (:160), 1.0 TB rank-local shards for the active model working set (:161),
  0.5 TB OS/runtime/scratch (:162).
- External - at least 1 TB direct rank-local model access + remaining striped
  RAID-like model-data pool across the fleet (:166-169).

"N sparks store 1/N" is the deployment consequence of sharding, stated in code
and audit rather than as a single sentence: KV is sliced by PP stage (each stage
covers only its slice's layers) and by TP rank, so each spark holds 1/N of the
fleet's hot KV. Evidence:

- PP: correct today - "each stage's pool covers only its slice's attention
  layers" (QWEN38_MAX_KV.md:51-53).
- TP: replicated today (4x waste), fixed by head-parallel sharding so "each rank
  stores exactly 1/tp_degree of the KV heads" (QWEN38_MAX_KV.md:122-125).
- Store keys carry rank_index, "so a store ring can shard by rank"
  (QWEN38_MAX_KV.md:12).
- Per-rank deployment numbers: kv_logical_page_capacity 1048576,
  kv_physical_page_capacity 16384, and 4 TiB backing per host in
  examples/deployments/qwen38_fp8_tp4_pp4_host_rdma.spec.json:21-22,38-39 and
  examples/deployments/dsv4_pro_tp4_pp4_host_rdma.spec.json:21-22,57. (The
  13-node Flash spec uses a smaller 256 GiB per-rank backing:
  examples/deployments/dsv4_flash_pp13_host_rdma.json:21-22,32-33.)
- NVMe sizing treats the internal drive as the KV tier candidate, external as 2x
  slower (docs/archive/NVME_KV_SIZING.md:1-14), with per-model KV bytes/token
  (NVME_KV_SIZING.md:42-50).

---

## 2. The boundary contract

### 2.1 Common KV machinery vs per-model page layouts vs scheduler/eviction policy

Three-way split, all three already stated in the headers:

A. Common machinery (cache/ + the shared headers) owns blocks, residency,
paging, and JIT prefetch mechanics, never a model's byte layout:

- Arena: allocate/recycle/retain/resident/pin/prefetch-plan -
  spark_kv_cache.h:471-565.
- Logical page directory: "owns sequence-to-page bindings, immutable prefix
  sharing, and logical-page lifetime" - spark_kv_page_cache.h:23-28.
- Page store: "The cache decides what to evict and when to fetch it" -
  spark_kv_page_store.h:31-37.
- NVMe tier: JIT lookahead + clock eviction - spark_nvme_tier.h:3-27.

B. Per-model page layout is owned by the model driver, expressed as geometry
parameters + a physical byte mapping + a copy primitive:

- "Model drivers choose the arena block geometry and translate resident page
  slots into their own physical KV payload layout" - spark_kv_page_cache.h:26-27.
- "a model driver supplies only the copy primitive" - spark_kv_page_store.h:33-35.
- Geometry in the machinery's own vocabulary: SparkKvCacheCapacityRequest filled
  by spark_k3_kv_geometry.h:49-62; DSv4's physical pool layout
  (spark_dsv4_pool_layout.h:31-39).

C. Scheduler/eviction-policy ownership is explicitly split between the
runtime/scheduler and the model firmware:

- Page budgets are scheduler/runtime-owned: "The runtime and scheduler own these
  limits; a JIT-KV driver only maps each page into its model-specific byte
  layout" - spark_model_serving_adapter.h:146-149.
- The prefix cache (content-addressed reuse) is owned by the runtime batch
  engine, which allocates and initializes it from the adapter's
  cache_block_token_count - runtime/model_batch_engine.c:930-972 (block_token_count
  wired at :957).
- The authoritative statement of the seam: "The orchestrator does not understand
  attention, MoE, MTP, KV layout, quantization, CUDA graph topology, expert
  placement, or JIT-KV policy. Those details belong inside model firmware." -
  SPEC.md (section 6). Forcing KV internals into the orchestrator is listed as
  forbidden production behavior (SPEC.md:251).
- A model package binds "one exact adapter, driver, stage pack, weight format,
  KV contract, topology, and hardware profile" (ARCHITECTURE.md:180); model
  drivers own "weight layout, stage-local KV, recurrent state"
  (ARCHITECTURE.md:191-194).

### 2.2 The hash-consistent prefix sharding invariant

Note: the literal phrase "hash-consistent" does not appear in the tree (verified
by grep). The invariant the coordinator is pointing at is assembled from three
independently-present facts:

1. Content hash is chained, not per-block. A block matches only when the entire
   preceding prefix matches, so a prefix resolves to the SAME logical block on
   every consumer regardless of shard placement - cache/cache.h:166-180 and
   spark_prefix_cache.h:193-196.
2. Store keys bind identity + shard: model_fingerprint, cache_layout_fingerprint,
   rank_index, sequence_id, logical_block (spark_stage_kv_client.h:40-42), so a
   given (model, layout, rank) prefix deterministically lands on one rank's
   store - mooncake_kv.md:17-18, QWEN38_MAX_KV.md:12.
3. Stale KV can never be consumed: fingerprints in the key mean a model or
   layout change cannot read old KV - spark_stage_kv_client.h:25-28,
   mooncake_kv.md:17-18.

Net: the prefix's identity is a content hash; the shard is (model fingerprint x
layout fingerprint x rank); so a prefix is hash-addressed and rank-consistent,
which is what makes resumable KV safe across restarts and model swaps.

### 2.3 Coexistence / eviction behavior (resumable KV across model swaps)

- Pool eviction and residency eviction are different decisions and must not be
  conflated: a referenced block may be non-resident and a resident block may be
  unreferenced - cache/cache.h:49-56,476-505; separate LRU-weighted score for
  each (cache/cache.h:238-251, resident victim :488-505).
- Model activation changes an execution plan and residency assignment, not the
  public endpoint; "Eviction preserves resumable KV and model artifacts in the
  storage hierarchy" - ARCHITECTURE.md:145-149.
- The backing survives the resident process: the Mooncake client outlives
  SparkPipe resident/module releases, so "a SparkPipe restart does not discard
  Mooncake state" - mooncake_kv.md:14-15.
- Fingerprinted keys make coexistence safe: a swapped-in model cannot consume
  the previous model's KV - spark_stage_kv_client.h:25-28, mooncake_kv.md:17-18.
- NVMe eviction is a generation bump, never a write-back, because the bytes are
  re-fetchable/recomputable - spark_nvme_tier.h:21-25, cache/nvme_tier.c:20-27.

---

## 3. Consolidation plan (with owners)

The DRY law: shared code never names a model (tests/test_dry_law.py), model
facts live in tables/headers, generated files must match their generator. The
existing coordinator-verified plan is docs/DRY_CONSOLIDATION_PLAN.md. The KV
items below are additive to it.

### 3.1 One paged-cache core consumed by all models - KV-cache subsystem agent proposes; coordinator lands (shared)

The core already exists: spark_kv_cache + spark_kv_page_cache +
spark_kv_page_store + spark_prefix_cache + spark_nvme_tier. DSv4 is its only
production consumer (spark_dsv4_paged_cache.h:36-56). The work is adoption, not
invention:

- Publish the core's contract (this document's section 2) as the canonical
  boundary, so each model agent knows exactly the geometry->layout seam.
- The common core must expose SparkKvCacheEstimateCapacity /
  SparkKvCacheCalculateJitStageBudget as the single sizing path
  (spark_kv_cache.h:440-447). Today SparkKvCacheEstimateCapacity is defined but
  has no non-test caller (grep: only cache/kv_cache.c:365 and tests), so k3's
  geometry header is ahead of its consumer.

### 3.2 Retire or re-anchor cache/cache.h (LmCache) - KV-cache subsystem agent proposes; coordinator lands (shared)

cache/cache.h duplicates the arena/residency/prefetch logic now implemented in
spark_kv_cache.h. It is consumed for struct defs only by tests/test_cache.c:6,
but its chained content hash is still the anchor for spark_nvme_tier.h:83 and
spark_topology_switch.h:22. Options: (a) demote it to a spec-only header (no
mutable state) and re-anchor the two hash consumers on SparkPrefixCacheHashBlock;
or (b) keep it as the executable reference and make spark_kv_cache call into it.
(a) is preferred; it removes one representation of ownership (the
two-representation bug class cache/cache.h:13-17 warns about).

### 3.3 Reconcile DSv4's two cache systems - DSV4 model agent owns; KV-cache agent proposes the target seam

DSv4 has both the model-specific spark_dsv4_cache_plan.c/.h +
spark_dsv4_cache_arena.c/.h (sliding/compressed/indexer/compressor arenas,
spark_dsv4_cache_plan.h:52-86) and the common-core consumer
spark_dsv4_paged_cache.c/.h. Decide which is the page-layout mapping (the
common-core view is the JIT-KV one) and which is the offline sizing calculator,
then fold them so there is exactly one DSv4 page layout on top of the common
arena. KV-cache agent owns nothing under model-families/dsv4/ or
modules/dsv4_resident_decode_stage/; this is a review/propose lane.

### 3.4 Bring GLM 5.2 JIT-KV onto the common core - GLM52 model agent owns; KV-cache agent proposes

GLM52's JIT-KV is historical/archive (GLM52_B1024_JIT_KV_INTEGRATION.md) and its
adapter is DRIVER_OWNS_KV only (spark_glm52_serving_adapter.c:124). Proposal:
define GLM52's arena geometry (latent dim 512, rope 64, FP8 scale block 128 -
spark_glm52_model.h:11-12,21) as a SparkKvCacheCapacityRequest the way k3 does,
and adopt the common paged cache + NVMe tier instead of a model-local backend.

### 3.5 Qwen 3.8 (and the Qwen 3.6 reference) onto the common tier - Qwen38 model agent owns; KV-cache agent proposes

QWEN38_MAX_KV.md already maps the gap: the module links stage_kv_client.c +
kv_store.c but the adapter is DRIVER_OWNS_KV only and the module ignores the
page budgets (QWEN38_MAX_KV.md:29-35). Proposal: adopt the common
SparkKvPageStore + SparkKvStore path (qwen36's SparkQwen36ModuleOpenKvTier is
the reference, QWEN38_MAX_KV.md:42-48) and complete the TP 1/N head-sharding
(currently refused at tp_degree > 1, QWEN38_MAX_KV.md:127-130).

### 3.6 K3 geometry is already the boundary pattern - K3 model agent maintains the header; KV-cache agent proposes the capacity path

spark_k3_kv_geometry.h:49-62 fills SparkKvCacheCapacityRequest; the k3
pool-sizing header consumes it
(modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_pool_sizing.h). The
k3 serving adapter is a stub (spark_k3_serving_adapter.c:451,461,478). Proposal:
wire SparkKvCacheEstimateCapacity as the one sizing authority so the geometry
header and the stub adapter agree.

### 3.7 Prefix-cache protection policy - runtime/scheduler owns the consumer; KV-cache agent proposes the policy contract

The prefix cache is allocated/initialized by the runtime batch engine
(model_batch_engine.c:930-972), which is the correct owner of the residency
DECISION (spark_stage_kv_client.h:24-28). Proposal: the lookahead-protection and
reuse-scored-eviction knobs (spark_prefix_cache.h:249-262) become the single
protection policy, replacing any model-local lookahead logic.

### 3.8 NVMe tier stays common - KV-cache subsystem owns; no model edits

spark_nvme_tier is already model-free and device-behind-a-vtable
(spark_nvme_tier.h:109-140). Keep it; the only open item is wiring it into the
common paged-cache path (today no model module consumes it) and re-anchoring its
content-hash comment after 3.2.

### 3.9 Mooncake provider - KV-cache subsystem owns the interface; coordinator owns the provider build

spark_kv_store.h (interface) and spark_stage_kv_client.h (client) are shared
plumbing the subsystem owns. modules/kv_mooncake is the provider module. The
current provider is DummyClient-based and pins a single local replica
(spark_kv_mooncake.cpp:141-147), which matches the "each spark stores 1/N"
placement only after the rank sharding in 3.5 lands; flag this as the
correctness dependency for multi-spark coexistence.

### What I need from the coordinator

- Confirmation that "hash-consistent prefix sharding" (2.2) matches the intended
  invariant, since the literal phrase is absent from the tree and I reconstructed
  it from content-hash chaining + fingerprinted rank keys.
- Owner sign-off for the model-adoption items (3.3-3.6), which are outside the
  KV-cache subsystem's edit lane.

---

## Coordinator confirmation (2026-08-17)

The two flagged reconstructions are CONFIRMED against the tree:

1. "N sparks store 1/N of the KV" = PP slicing plus TP head-sharding. The
   head-parallel kernels landed in the Qwen38 wave make each rank store
   exactly 1/tp_degree of the KV heads (docs/QWEN38_MAX_KV.md:122-125);
   the pre-head-parallel replication note (same file :50-58) is the old
   state. The invariant is per-shape, not a global constant.
2. "Hash-consistent prefix sharding" is descriptive, not a literal string.
   It names two real mechanisms: the CHAINED content hash - a block matches
   only when the entire prefix before it matches (cache/cache.h:166-180) -
   and the fingerprinted rank keys (model + cache-layout fingerprints and
   rank index in SparkStageKvClientFormatKey, include/sparkpipe/
   spark_stage_kv_client.h:40-42).

Items 3.3-3.6 (DSV4, GLM52, Qwen38, K3 KV adoption) are SIGNED OFF to the
model agents; the kv-cache agent owns the shared seam proposals.
