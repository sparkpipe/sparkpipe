/* General prefix-cache core for resident decode stages.
 *
 * Model-independent half of prefix caching: block-table-backed KV
 * allocation with content-addressed sharing, longest-common-prefix
 * matching at admit, private continuation on divergence, LRU eviction
 * over unreferenced blocks, and one-time geometry validation. The core
 * owns indices and token ids only; it never touches KV bytes, so every
 * resident stage links it unchanged (see the DRIVER CONTRACT below for
 * the model-specific half each stage implements).
 *
 * Divergence policy: PRIVATE CONTINUATION, chosen over copy-on-write.
 * Published blocks are immutable; a sequence always appends into its own
 * freshly allocated open block, so divergence needs no copy hook in any
 * driver (the core cannot issue device copies anyway - KV row layout is
 * driver-owned), shared state stays race-free at any batch width, and
 * the only cost is recompute of fewer than block_token_count tail
 * tokens on a mid-block admit.
 *
 * Threading contract: all calls run on the owning module's single host
 * thread, like every other per-module adapter structure.
 */
#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PREFIX_CACHE_CORE_ABI_VERSION 1u
#define SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreConfiguration))
#define SPARK_PREFIX_CACHE_CORE_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCore))
#define SPARK_PREFIX_CACHE_CORE_STATS_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreStats))
#define SPARK_PREFIX_CACHE_CORE_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreBlockTableView))

#define SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS 256u
#define SPARK_PREFIX_CACHE_CORE_NO_BLOCK 0xffffffffu

#define SPARK_PREFIX_CACHE_CORE_BLOCK_FREE 0u
#define SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE 1u
#define SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED 2u

typedef struct SparkPrefixCacheCoreConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	/* Driver KV geometry: tokens per block and KV bytes per block across
	 * every layer the stage owns. The core uses the token count for
	 * addressing and the stride for reuse accounting only. */
	uint32_t block_token_count;
	uint32_t reserved0;
	uint64_t block_stride_bytes;
	/* Pool and population ceilings. */
	uint32_t block_count;
	uint32_t max_sequence_count;
	uint32_t sequence_block_capacity;
	/* Content-index buckets; power of two. */
	uint32_t hash_bucket_count;
	uint32_t reserved1;
} SparkPrefixCacheCoreConfiguration;

typedef struct SparkPrefixCacheCoreStats
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t live_sequence_count;
	uint32_t used_block_count;
	uint32_t published_block_count;
	uint32_t free_block_count;
	uint64_t admit_count;
	uint64_t matched_block_count;
	uint64_t missed_block_count;
	uint64_t appended_token_count;
	uint64_t published_block_count_total;
	uint64_t evicted_block_count;
	uint64_t capacity_stall_count;
	uint64_t bytes_reused;
	uint64_t ticks;
} SparkPrefixCacheCoreStats;

/*
 * Per-step block table handed to a stage's attention kernels: lane-major
 * physical block indices plus per-lane counts, mirrored host and device.
 * Block i of a lane covers positions [i*block_token_count,
 * (i+1)*block_token_count). Every block except each lane's LAST is
 * published and therefore immutable and possibly shared; writes go only
 * to the lane's last block, at offset position % block_token_count.
 */
typedef struct SparkPrefixCacheCoreBlockTableView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t lane_count;
	uint32_t lane_stride;
	const uint32_t *device_physical_block_indices;
	const uint32_t *device_lane_block_counts;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_block_counts;
} SparkPrefixCacheCoreBlockTableView;

typedef struct SparkPrefixCacheCoreBlock
{
	uint32_t state;
	uint32_t reference_count;
	uint32_t token_count;
	uint32_t hash_next;
	uint32_t free_next;
	uint32_t lru_prev;
	uint32_t lru_next;
	uint64_t chain_hash;
	uint64_t last_used_tick;
	uint32_t *token_ids;
} SparkPrefixCacheCoreBlock;

typedef struct SparkPrefixCacheCoreSequence
{
	uint64_t sequence_id;
	uint32_t used;
	uint32_t block_count;
	uint64_t running_chain_hash;
	uint32_t *blocks;
} SparkPrefixCacheCoreSequence;

typedef struct SparkPrefixCacheCore
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t reserved0;
	uint64_t block_stride_bytes;
	uint32_t block_count;
	uint32_t max_sequence_count;
	uint32_t sequence_block_capacity;
	uint32_t hash_bucket_mask;
	SparkPrefixCacheCoreBlock *blocks;
	uint32_t *block_tokens;
	SparkPrefixCacheCoreSequence *sequences;
	uint32_t *sequence_blocks;
	uint32_t *hash_buckets;
	uint32_t free_block_head;
	uint32_t live_sequence_count;
	uint32_t lru_head;
	uint32_t lru_tail;
	uint64_t tick;
	uint64_t admit_count;
	uint64_t matched_block_count;
	uint64_t missed_block_count;
	uint64_t appended_token_count;
	uint64_t published_block_total;
	uint64_t evicted_block_count;
	uint64_t capacity_stall_count;
} SparkPrefixCacheCore;

/*
 * DRIVER CONTRACT - the model-specific half each resident stage implements:
 *
 * 1. GEOMETRY: fill configuration.block_token_count and
 *    block_stride_bytes from the stage's KV row layout, and size
 *    block_count to the physical pool the stage allocates as
 *    base + physical_index * block_stride_bytes. The core never reads
 *    or writes those bytes.
 * 2. KERNELS: after each rebuild of the lane set, upload the host
 *    mirrors with SparkStageModuleBlockTableUpload and read the table
 *    through SparkPrefixCacheCoreBlockTableView in every
 *    prefill/decode/verify walk, honoring the immutability rule
 *    documented on the view.
 * 3. TOKENS: call SparkPrefixCacheCoreAdmitSequence with the full
 *    prompt id array at submission, then
 *    SparkPrefixCacheCoreAppendTokens with every sampled id. The core
 *    keeps the host token mirror this needs.
 */
SparkStatus SparkPrefixCacheCoreValidateGeometry(
    const SparkPrefixCacheCoreConfiguration *configuration);

SparkStatus SparkPrefixCacheCoreInitialize(
    SparkPrefixCacheCore *core,
    const SparkPrefixCacheCoreConfiguration *configuration);

void SparkPrefixCacheCoreDestroy(SparkPrefixCacheCore *core);

/*
 * Admit a submission: match token_ids against published chains, attach
 * every matched block (sharing them read-only), append the unmatched
 * tail, and report matched_token_count_out - a multiple of
 * block_token_count - so the stage prefills from that position with
 * zero recompute of the shared prefix.
 */
SparkStatus SparkPrefixCacheCoreAdmitSequence(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out);

/* Append generated (or deferred prompt) ids to the sequence's open
 * block, publishing each block as it fills. */
SparkStatus SparkPrefixCacheCoreAppendTokens(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count);

/*
 * Grow a sequence with freshly allocated, still-empty PRIVATE tail blocks
 * until it holds at least min_block_count blocks, WITHOUT recording any
 * token ids. This is how a driver pre-reserves decode/speculation KV
 * coverage whose tokens do not exist yet (draft/verify rows are only
 * known after earlier frames run): the reserved blocks receive their real
 * ids through later AppendTokens calls, and a reserved block that never
 * fills is simply released with its sequence - it never publishes and
 * never matches. Fails with CAPACITY_EXCEEDED (after self-trim) if the
 * pool cannot supply the missing blocks; a failed call may leave a
 * shorter partial reservation, which the caller releases as usual.
 */
SparkStatus SparkPrefixCacheCoreReservePrivateTail(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t min_block_count);

/*
 * Copy the sequence's physical block list - every block the sequence
 * currently holds, including reserved-but-still-empty tails - into
 * block_indices_out. This is the authoritative table row a driver
 * uploads: coverage may exceed the token mirror, so a driver must build
 * rows from this list rather than from BuildBlockTable when reserved
 * blocks exist.
 */
SparkStatus SparkPrefixCacheCoreCopyBlockList(
    const SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t *block_indices_out,
    uint32_t block_index_capacity,
    uint32_t *block_count_out);

/* Fill block_indices_out with the physical blocks covering positions
 * [0, token_count) of the sequence. */
SparkStatus SparkPrefixCacheCoreBuildBlockTable(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *block_indices_out,
    uint32_t block_index_capacity,
    uint32_t *block_count_out);

uint32_t SparkPrefixCacheCoreSequenceTokenCount(
    const SparkPrefixCacheCore *core,
    uint64_t sequence_id);

/*
 * Finish a sequence: drop its open partial block, release its published
 * blocks into the evictable cache. The id may be reused by a later
 * AdmitSequence.
 */
SparkStatus SparkPrefixCacheCoreReleaseSequence(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id);

/* Evict least-recently-used published blocks with zero references
 * until free_block_count reaches free_target or nothing is evictable. */
SparkStatus SparkPrefixCacheCoreTrim(
    SparkPrefixCacheCore *core,
    uint32_t free_target,
    uint32_t *evicted_block_count_out);

void SparkPrefixCacheCoreQueryStats(
    const SparkPrefixCacheCore *core,
    SparkPrefixCacheCoreStats *stats);

#ifdef __cplusplus
}
#endif
