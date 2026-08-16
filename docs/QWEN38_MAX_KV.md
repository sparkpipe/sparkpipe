# Qwen 3.8 Max KV cache audit - JIT, sharding, exactness, dedicated ring

Audit of the qwen38 KV path against four questions. Verdicts first, then the
evidence, then the capacity math and the kernel inventory. Fixes land in this
session; this doc records what was found before and after.

## Verdicts (before this session's fixes)

| Question | Verdict | Detail |
|---|---|---|
| Fully supports JIT | PARTIAL | The serving adapter grows lane blocks on demand from a host freelist (JIT within the resident pool) and releases them at position zero. The runtime already delivers logical/physical page capacities + a 4 TB-per-host NVMe backing contract through SparkFirmwareModuleHostServices, but the module IGNORES all of it: no KV tier, no store client, no paging. Prefill is refused (SPARK_STATUS_UNSUPPORTED), so the cache can only fill one decode token per step. |
| Sharded | PARTIAL | PP: yes - each stage's pool covers only its slice's attention layers (dense attn_ordinal_by_layer). TP: NO - attention is fully replicated, so every rank stores all 4 KV heads: 4x the needed capacity across the TP4 group. Store keys carry rank_index, so a store ring can shard by rank once the client is wired. |
| Not using more data than needed | PARTIAL | Token record is exactly K+V head-major: 2 x 4 heads x 256 x bf16 = 2048 elements = 4 KiB per token per layer, no padding, no Q-head waste, only attention layers stored (23 of 92). The 4x TP replication is the more-data-than-needed violation; per-rank record should be 512 elements = 1 KiB. |
| Large capacity on a dedicated spark ring | NO (module) | The deployment spec already declares kv_logical_page_capacity 1048576, kv_physical_page_capacity 16384, and a 4 TiB NVMe backing dataset per host; the KV store service + mooncake provider + service_address client path exist in-tree (qwen36 uses them). The qwen38 module links stage_kv_client.c + kv_store.c but never calls them. |

## Evidence

### JIT within the resident pool (adapter, works today)
spark_qwen38_serving_adapter.c: SparkQwen38ServingCoverLane grows a lane's
block list on demand (required = ceil(end_position / 64)), pulling physical
indices from free_blocks; a position-zero frame releases the lane's blocks
first; exhaustion returns SPARK_STATUS_CAPACITY_EXCEEDED and the whole
submission is dropped atomically so no partial allocation lingers. The block
table (logical lane -> physical pool index, lane_stride 4096 at full context)
is uploaded per submission and the decode kernel validates
required_block_count <= available_block_count before reading, NaN-poisoning
the head output on any violation.

### What the runtime already provides (ignored by the module)
SparkFirmwareModuleHostServices carries kv_logical_page_capacity,
kv_physical_page_capacity, kv_backing_directory, kv_backing_maximum_bytes.
The qwen38 deployment spec (examples/deployments/qwen38_fp8_tp4_pp4_host_rdma.spec.json)
declares logical 1048576 pages, physical 16384 pages, backing dataset
qwen38/tp4_pp4.bf16 with 4398046511104 bytes (4 TiB) per host. dsv4 and
glm52 modules consume exactly these fields; qwen38 reads none of them.

### The store path that exists in-tree
cache/store/stage_kv_client.c opens a pluggable KV store provider
(SparkKvStoreLoadInterfaceFromSharedObject), submits keyed GET/PUT batches
(key = model_fingerprint, cache_layout_fingerprint, rank_index, sequence_id,
logical_block), and polls completions. modules/kv_mooncake is the network
provider (mooncake transfer engine behind the interface). qwen36's module
(SparkQwen36ModuleOpenKvTier) is the complete reference: env
SPARK_QWEN36_STAGE_KV_STORE / _SERVICE / _SOCKET / _POOL_BYTES / _WORKERS,
geometry fingerprints, and model-families/qwen36/src/spark_qwen36_work_control.c
builds the restore/evict batches with the pressure-limited lookahead
selector. qwen38 links the same client but calls nothing, and its Makefile
declares the KV_* variables without exporting them.

### Sharding today
- PP: cache_layer_stride = 64 tokens x 2048 elements, cache_block_stride =
  cache_layer_stride x cache_layer_count where cache_layer_count is the
  SLICE's attention layers (5 or 6), not 23. Correct.
- TP: the attention linears are full-width (query 32768 fused rows, key/value
  1024 rows, output 8192 rows), SparkQwen38AttnPrepareKernel launches over
  all 64 query heads and writes all 4 KV heads on every rank, and the decode
  kernel reads kv_head = head / 16 from the local pool. Every rank of a TP4
  group holds the identical 4-head cache: 4x replication.

### Exactness
Token record = SPARK_QWEN38_MODEL_ATTN_CACHE_TOKEN_ELEMENTS = 2 x
ATTN_KV_DIMENSION = 2048 bf16 (K then V, head-major, post-RoPE). The pool is
cache_block_stride x kv_block_count elements zeroed at allocate. No waste
per record. The waste is the TP replication factor only.

## Capacity math (per rank, one lane)

| Scope | Elements/token | Bytes/token | Full context (262144) |
|---|---|---|---|
| One attention layer, replicated (today) | 2048 | 4096 | 1.0 GiB |
| One attention layer, TP4 head-sharded (fix) | 512 | 1024 | 256 MiB |
| Stage slice, 6 attn layers, replicated | 12288 | 24576 | 6.0 GiB |
| Stage slice, 6 attn layers, sharded | 3072 | 6144 | 1.5 GiB |

Per block (64 tokens, 6-layer stage): replicated 1.5 MiB, sharded 384 KiB.
Physical page capacity 16384 blocks on the deployed spec: sharded that is
6 GiB of resident KV per rank; the 4 TiB NVMe backing holds the rest of the
logical 1048576-page space (dedicated-ring grade capacity with the store
service at the other end of service_address).

GDN recurrent state is the larger resident cost per lane (128 value heads x
128x128 fp32 state x 69 GDN layers ~ 552 MiB per lane at full width) and is
the reason the work-control port keeps the qwen36 gdn record page: state
pages out with the KV blocks.

## Kernel inventory

Present and verified:
- SparkQwen38AttnPrepareKernel: per-(row,head) Q/K RMSNorm + partial RoPE,
  K/V write into the paged block at slot_mapping (group-leader blocks write).
- SparkQwen38AttnDecodeKernel: flash-style paged GQA decode, eight warps
  stripe the context, per-warp online softmax, fused sigmoid gate, one merge.
  Guards: context 0, block-count overrun, lane_stride violation -> NaN output.

Missing (this session's kernel work):
- Head-parallel variants of both kernels (local 16 Q heads + the rank's one
  KV head; per-rank cache strides /tp_degree) - the TP sharding fix.
- Prefill attention: chunked causal GQA over [base, base+token_count) that
  writes K/V blocks and produces per-position head outputs - the module
  refuses prefill today, so the paged cache cannot take prompt chunks.
- The residual combine for head-parallel attention (o_proj row-parallel ->
  8192-wide partial delta -> one all-reduce per attention layer), via
  SparkTpDeviceCollectiveSubmitBf16 following the dsv4 module pattern.

## Fix plan (this session)

Landed:
1. spark_qwen38_work_control ported (model-families/qwen38) with the qwen36
   building-block tests (plan math, lane-atomic restore batches, tier
   roundtrip through the mooncake provider) - test_qwen38_work_control PASS.
2. Module KV tier: host_services logical/physical page capacities recorded,
   SPARK_QWEN38_STAGE_KV_STORE / _SERVICE / _SOCKET / _POOL_BYTES / _WORKERS
   read (provider optional, default none = byte-identical today), geometry +
   layout fingerprints, client open (kv_tier_open log), the resident pool
   clamped to the declared physical window when the tier is active, and the
   decode path pages: per-lane required-block residency, window-slot
   assignment, round-robin eviction with dirty write-back, module-owned
   rewritten device block table + row slot mappings uploaded per frame, and
   dirty marking after the K/V write. Verified: tier opens against a real
   provider .so (mooncake dummy), and a frame without the adapter's decode
   batch view fails closed rather than paging blindly.
3. Head-parallel attention kernels landed: AttnPrepare/AttnDecode take
   (tp_degree, tp_rank), compute the rank's local query heads + local KV
   head slice, and use per-rank cache token strides - so each rank stores
   exactly 1/tp_degree of the KV heads (4x capacity win at TP4).
   SPARK_QWEN38_STAGE_TP_DEGREE unset keeps the replicated layout
   byte-identical (smoke-verified). tp_degree > 1 is REFUSED at initialize
   (fail closed) until the residual all-reduce (SparkTpDeviceCollective)
   and the head-sliced projections land; running head-parallel without the
   combine would silently emit rank-partial hiddens.

Still open (next sessions):
- Wire SparkTpDeviceCollective (hidden_transport backend, recursive
  doubling / split ring per the deployment config) into the module and
  slice the attention linear views (query/key/value row slices, o_proj
  row-parallel) - the one-step activation of TP-sharded KV.
- Prefill attention kernels (chunked causal GQA + paged K/V writes) and
  module prefill acceptance, so the paged cache takes prompt chunks.
- GDN state paging through the tier (the work-control gdn record slot is
  reserved; the placeholder record is dormant until the lane GDN-state
  record size is fixed).
