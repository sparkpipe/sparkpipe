# DSV4 Pro KV cache audit: JIT, sharding, footprint, large capacity

Code-level audit of the dsv4 resident decode stage's KV path (paged cache,
pool layout, cache arena, admission, page store). All statements cite the
verified source sites.

## 1. JIT - yes, fully JIT

- The adapter declares SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV
  (spark_dsv4_serving_adapter.c descriptor), and the runtime enforces the
  JIT invariants in SparkModelServingAdapterValidateRuntimeLimits:
  kv_physical_page_capacity >= max_active_sequence_count,
  kv_logical_page_capacity >= resident_sequence_capacity,
  physical <= logical. The deployed config (16384 physical, 1048576
  logical) satisfies them.
- Pages are granted per submission at admission time
  (SparkDsv4PreparedCacheAdmission records, one per pipeline slot;
  SparkDsv4ModuleEvaluateAdmission + cache_admission_logical_pages), not
  pre-allocated for the full context: a lane only maps pages for the
  tokens it actually holds (window + compressed history + index-selected
  top-k), one block (128 tokens) per page.
- The page store (SparkKvPageStore, flags ANONYMOUS | DIRECT_IO) is the
  backing tier: evicted physical pages write back to the backing directory
  (deployed: /home/{host}/kvcache/dsv4_pro/tp4pp4.bf16, maximum 4 TiB per
  host) and are demand-fetched on later access - the "JIT tier" per
  include/sparkpipe/spark_nvme_tier.h. The cache arena itself is the
  device-resident physical pool: resident_state_bytes =
  physical_page_capacity * page_stride_bytes (module line 1571).

## 2. Sharding

- PP: each stage caches ONLY its own slice's layers. The pool layout
  (SparkDsv4PagedPoolBuildLayout) walks first_layer_index..+layer_count;
  per-layer attention/compressor/index offsets are stage-local. VERIFIED.
- TP: the KV computation is TP-distributed - WKV (and the compressor /
  indexer projections) are row-sharded by the sharder
  (tools/dsv4_pro_tp16_stagepack.py row_indices: 512 -> 128 rows per rank
  at TP4), while each rank holds 32 of the 128 query heads
  (SparkDsv4ModuleTpQueryHeads). The attention output is all-reduced
  across the TP group per layer (SparkDsv4ModuleReduceHidden), which
  carries the cross-rank attention combination. Each rank keeps its own
  cache (per-rank device arenas).
- CAVEAT (one detail not fully pinned by code reading): whether each
  rank's cache entries store the full 512-dim KV or only its 128-dim
  WKV band at a column offset inside the 512-wide entry stride. The pool
  layout prices entries at ATTN_HEAD_DIMENSION (512) BF16 per token, the
  kv slot buffer is allocated rows x 512, and the pair kernel writes
  second->rows (128) contiguous elements. The single-spark validations
  ran TP=1 (no TP sharding exercised); the first TP4 GPU run should
  confirm the band layout against the attention kernel's read pattern.

## 3. Footprint - only what attention needs

Per block (128 tokens) per layer per lane (SparkDsv4PagedPoolAppendLayer):

| Kind | Attention entries | Compressor state | Index (CSA only) |
| --- | --- | --- | --- |
| CSA | 128 window + 32 compressed, 512 dims BF16 = 163,840 B | 2 x 8,192 f32 = 65,536 B | 32 x 128 BF16 + 2 x 2,048 f32 = 24,576 B |
| HCA | 128 window + 1 compressed = 132,096 B | 2 x 65,536 f32 = 524,288 B | - |
| SWA/DSpark | 128 x 512 BF16 = 131,072 B | - | - |

The full uncompressed KV would be 128 x 512 x 2 = 131,072 B per block plus
the entire context; the cache instead holds the 128-token sliding window
plus COMPRESSED history (1 entry per 128 tokens on HCA, 1 per 4 on CSA)
plus the index-selected top-k (1024) - nothing else. Per-lane totals at
max_sequence_positions = 4096: CSA layer (128 + 1024) x 512 x 2 ~ 1.18 MB,
HCA layer (128 + 32) x 512 x 2 ~ 164 KB - versus 4.2 MB/layer for the full
4096-token KV. VERIFIED: the cache stores compressed + window only.

## 4. Large capacity on a dedicated spark ring - yes, with knobs

The model's native limit is SPARK_DSV4_MODEL_MAX_POSITIONS = 1,048,576
tokens; the deployment currently caps max_sequence_positions at 4096 (the
stage JSON) as a first-light choice, not a code limit. The knobs:

| Knob | Where | Deployed | Large-capacity direction |
| --- | --- | --- | --- |
| max_sequence_positions | stage JSON (per host config) | 4096 | up to 1,048,576 |
| kv_logical_page_capacity | resident config runtime_limits | 1048576 | >= ceil(max_seq/128) per lane x lanes |
| kv_physical_page_capacity | resident config runtime_limits | 16384 | hot-set sized: physical pages x page_stride <= unified memory budget |
| kv_backing_maximum_bytes | resident config | 4 TiB | history sized: lanes x per-lane compressed history |

Per-lane compressed history at 1M tokens: CSA layer (128 + 262,144) x
512 x 2 ~ 268.6 MB, HCA layer (128 + 8,192) x 512 x 2 ~ 8.5 MB; the
device pool holds only the hot physical pages, the page store demand-
fetches the rest from the backing tier, so a dedicated ring with large
disks scales the aggregate history without growing the device pool.

Sizing notes:
- page_stride_bytes (per 128-token page) for a 15-layer CSA stage ~ 4 MB,
  stage 0 (2 HCA + 14 CSA) ~ 5 MB (matches the observed page_kib ~1784-2180
  on the 4-layer validation slices).
- Physical pool = physical_pages x page_stride: 16384 x ~4.3 MB ~ 70 GB
  device on top of the ~57 GB weights - check the GB10 unified-memory
  budget before raising physical pages; raise backing bytes instead for
  history.
- The head-certified B1 path's shadow buffers are vocab-screening state,
  unrelated to the KV cache.
