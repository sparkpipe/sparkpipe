# TOP-10 KV-cache (top-speed assessment + ranked improvements)

KV-cache subsystem agent. Metric: maximize Solutions / (code size x 2)
(METRIC.md). DRY wins first (deleting a line is a solution at zero cost); then
performance levels 1 accurate-slow < 2 80% SOTA < 3 90% SOTA < 4 SOTA < 5
exceeds. Each item below names which solution it buys and an estimated
code-size delta. All facts cited file:line against origin/unified @ afb43a8.

## Assessment: the KV stack today

- The stack is COMPLETE but 1/4 ADOPTED. The common core (arena spark_kv_cache.h,
  page-dir spark_kv_page_cache.h, page-store spark_kv_page_store.h, NVMe tier
  spark_nvme_tier.h, prefix cache spark_prefix_cache.h) is ~11.4k lines and
  token-free, but ONLY DSv4 declares JIT_KV
  (modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:282,305).
  glm52/qwen38/qwen38_27b are DRIVER_OWNS_KV (spark_glm52_serving_adapter.c:124,
  spark_qwen38_serving_adapter.c:204); k3's adapter is a stub
  (spark_k3_serving_adapter.c:451,461,478).
- A SECOND GENERATION of the same logic still exists: cache/cache.h (723 lines,
  inline LmCache arena/residency/prefetch) duplicates spark_kv_cache.h. It is
  consumed for structs only by tests/test_cache.c:6, but its chained content hash
  is still the anchor for spark_nvme_tier.h:83 and spark_topology_switch.h:22.
- Two APIs are built but unconsumed: SparkKvCacheEstimateCapacity has no
  non-test caller (defined cache/kv_cache.c:365), and the NVMe tier has no model
  consumer (include/sparkpipe/spark_nvme_tier.h is included only by
  cache/nvme_tier.c and spark_topology_switch.h).
- The seam is now FULLY SIGNED: SparkKvModelTable approved as the single
  token-free fill point, scheduler confirmed A (table lives in resident) and C
  (scheduler supplies protected set), and narrowed B (WillBeResidentBy is
  lookahead-only, NOT an admission gate) - docs/PROPOSAL_KV_SEAM.md (scheduler
  sign-off note).
- The per-model gap is concentrated in Qwen 3.8: prefill refused, so the paged
  cache cannot take prompt chunks; TP replicates all 4 KV heads (4x waste) -
  docs/QWEN38_MAX_KV.md:11,50-58. Head-parallel kernels already landed but are
  refused at tp_degree > 1 pending the residual all-reduce (:122-130).
- Mooncake provider is a DummyClient pinning ONE local replica
  (modules/kv_mooncake/spark_kv_mooncake.cpp:141-147), so the 1/N rank-sharded
  coexistence contract is not yet exercised at fleet scale.

## TOP-10 improvements (ranked)

1. **Demote cache/cache.h to a spec, re-anchor the hash.** DRY (biggest
   deletion, do first). Strip the ~480 lines of executable LmCache
   arena/residency/prefetch that duplicate spark_kv_cache.h; keep the prose +
   the chained-hash definition as the reference; re-point the two hash consumers
   (spark_nvme_tier.h:83, spark_topology_switch.h:22) at
   SparkPrefixCacheHashBlock (spark_prefix_cache.h:193-196); port the unique
   copy-on-write/JIT-reservation cases from tests/test_cache.c (257L) into
   tests/test_kv_cache.c. Delta ~ -600. Owner: KV-cache agent (propose) +
   coordinator (land). First step: re-anchor the two hash consumers, run
   test_nvme_tier.c + test_dry_law.py.

2. **Land SparkKvModelTable + SparkKvBackendInitialize (already approved).** DRY
   + the enabler for 3-7 below. One token-free table replaces DSv4's inline
   config (spark_dsv4_paged_cache.c:102-138) and k3's partial fill
   (spark_k3_kv_geometry.h:49-62) with a single fill point + validator. Delta
   ~ +140 new / -150 removed (net ~0), but it collapses 4 ad-hoc fills to 1 and
   unblocks the whole adoption path. Owner: KV-cache agent + coordinator. First
   step: write include/sparkpipe/spark_kv_model_table.h (token-free) + the
   validator + a fill for DSv4 to prove byte-identical init.

3. **Wire SparkKvCacheEstimateCapacity as the single sizing authority.** DRY.
   It is defined (cache/kv_cache.c:365) with no non-test caller; k3 pool sizing
   computes the same numbers separately. Make every model size its arena through
   it. Delta ~ -100 (retire per-model sizing math). Owner: KV-cache agent +
   model agents. First step: switch modules/k3_resident_decode_stage's pool
   sizing to call it; pin with tests/test_k3_kv_cache.c.

4. **Reconcile DSv4's two cache systems.** DRY. DSv4 carries both an offline
   cache plan (spark_dsv4_cache_plan.h:52-86) and the serving paged cache
   (spark_dsv4_paged_cache.h:36-56); the per-layer 4096-aligned offset walking
   is computed twice. Share one per-layer layout walker; make the paged cache
   fill SparkKvModelTable like every other model. Delta ~ -300. Owner: DSV4
   model agent. First step: decide plan = offline sizing, paged = serving
   mapping, then de-duplicate the offset walker (pin: tests/test_dsv4_cache_plan.c
   + tests/test_dsv4_paged_cache.c).

5. **Qwen 3.8: activate TP 1/N head-sharding.** LEVEL 1->2 (capacity; the 1/N
   contract). Head-parallel kernels are landed but gated at tp_degree > 1
   pending the residual all-reduce (docs/QWEN38_MAX_KV.md:122-130). Finishing
   it removes the 4x TP replication (:50-58). Delta ~ +300. Owner: Qwen38 model
   agent. First step: wire SparkTpDeviceCollective residual + head-sliced
   q/k/v/o projections, then lift the refuse gate.

6. **Qwen 3.8: prefill + paged-cache adoption.** LEVEL 1->2/3. Prefill is
   refused today, so the cache fills only one decode token per step
   (docs/QWEN38_MAX_KV.md:11,29-35). Accept prompt chunks and page through the
   common store. Delta ~ +400. Owner: Qwen38 model agent. First step: module
   reads host_services page capacities (already delivered via
   runtime/pack/driver_compiler.c:588-589) and fills SparkKvModelTable.

7. **GLM 5.2: JIT-KV onto the common core + NVMe tier.** LEVEL 2->3. The
   archive JIT-KV backend is model-local and was measured NOT_WORKING with zero
   counters (docs/archive/GLM52_MEASURED_STATUS.md:75); the adapter is
   DRIVER_OWNS_KV only (spark_glm52_serving_adapter.c:124). Adopt the common
   paged cache + tier (GLM52's DSA cap holds B64 at 1M -
   docs/archive/NVME_KV_SIZING.md:143,154-155). Delta ~ +400. Owner: GLM52 model
   agent. First step: add a glm52 SparkKvModelTable fill (latent 576, FP8 scale
   128 - spark_glm52_model.h:11-12,21,88-90).

8. **Wire the NVMe tier into the common paged-cache path.** LEVEL 3->4 enabler
   (1M-context). The tier is built and host-tested (tests/test_nvme_tier.c) but
   no model consumes it. Connect SparkNvmeTierPlanLookahead to the arena
   prefetch plan inside SparkKvBackendInitialize so DSv4/GLM52 get JIT tier-3
   reads. Delta ~ +200. Owner: KV-cache agent + scheduler agent. First step:
   emit SparkNvmeTierNeed from the arena's non-resident blocks in the backend
   initializer (respect the signed B: lookahead only, never admission).

9. **K3: wire the capacity path, retire the stub adapter.** LEVEL 1->2. The
   geometry header already fills the capacity request (spark_k3_kv_geometry.h:49-62)
   and the seam-crossing test passes (tests/test_k3_kv_cache.c:1-7), but the
   serving adapter is a stub (spark_k3_serving_adapter.c:451,461,478). Fill
   SparkKvModelTable from the geometry header and give the adapter real
   capability_flags. Delta ~ +200. Owner: K3 model agent. First step: fill the
   table from spark_k3_kv_geometry.h; pin with tests/test_k3_kv_geometry.py.

10. **Mooncake provider: real rank-sharded backend (drop DummyClient
    single-replica).** Scale/correctness for the 1/N contract. The provider pins
    one local replica (spark_kv_mooncake.cpp:141-147), which is correct only
    AFTER TP sharding lands; the keys already carry rank_index
    (spark_stage_kv_client.h:40-42). Swap the DummyClient path for the real
    transfer-engine backend keyed by rank. Delta ~ +150. Owner: KV-cache agent
    (interface) + coordinator (provider build). First step: shard PUT by
    rank_index-derived segment; qualify restart-recovery per
    docs/archive/mooncake_kv.md:127-143.

## Net

Items 1-4 are DRY/structural (net ~ -900 lines, zero risk to correctness, do
them first and they pay for the rest). Items 5-10 are level buys totalling
~ +1,650 lines that move three models up the ladder (Qwen 1->2/3, GLM52 2->3,
K3 1->2) and make the already-built NVMe tier and 1/N coexistence real. The
sequence is: 2 first (it unblocks 5-9), then 1+3+4, then 5-10.
