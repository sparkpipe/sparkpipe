#ifndef SPARKPIPE_SPARK_GLM52_PREFIX_CACHE_H
#define SPARKPIPE_SPARK_GLM52_PREFIX_CACHE_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PREFIX_CACHE_ABI_VERSION 5u
#define SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCacheConfiguration))
#define SPARK_GLM52_PREFIX_CACHE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCache))
#define SPARK_GLM52_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCacheEntry))
#define SPARK_GLM52_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCacheSequenceBinding))
#define SPARK_GLM52_PREFIX_CACHE_LOOKUP_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCacheLookup))
#define SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCacheReservation))
#define SPARK_GLM52_PREFIX_CACHE_PROMPT_HASH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCachePromptHash))
#define SPARK_GLM52_PREFIX_CACHE_HASH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PrefixCachePromptHash))

#define SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID 0x00000001u
#define SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_REUSABLE 0x00000002u
#define SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING 0x00000004u
#define SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY 0x00000008u

#define SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID 0x00000001u
#define SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_PENDING 0x00000002u

#define SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH 1469598103934665603ull
#define SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK 0xffffffffu
#define SPARK_GLM52_PREFIX_CACHE_NO_ENTRY 0xffffffffu

#ifndef SPARK_GLM52_PREFIX_CACHE_MAX_BLOCK_TOKENS
#define SPARK_GLM52_PREFIX_CACHE_MAX_BLOCK_TOKENS 256u
#endif

typedef struct SparkGlm52PrefixCacheEntry
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t token_count;
    uint32_t first_token_index;
    uint32_t physical_block_index;
    uint32_t reference_count;
    uint32_t reserved;
    uint32_t lookahead_priority;
    uint32_t lookahead_request_count;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t last_used_tick;
    uint64_t reservation_epoch;
    uint64_t committed_epoch;
    uint64_t lookahead_protection_epoch;
} SparkGlm52PrefixCacheEntry;

typedef struct SparkGlm52PrefixCacheSequenceBinding
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t entry_index;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t physical_block_index;
    uint32_t reserved;
    uint64_t sequence_id;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t acquire_epoch;
} SparkGlm52PrefixCacheSequenceBinding;

typedef struct SparkGlm52PrefixCachePromptHash
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t token_count;
    uint32_t hashed_token_count;
    uint32_t block_count;
    uint32_t last_block_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t parent_hash;
    uint64_t prompt_hash;
} SparkGlm52PrefixCachePromptHash;

typedef struct SparkGlm52PrefixCacheConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t entry_count;
    uint32_t physical_block_count;
    uint32_t sequence_binding_count;
    SparkGlm52PrefixCacheEntry *entries;
    SparkGlm52PrefixCacheSequenceBinding *sequence_bindings;
    SparkGlm52KvCacheArena *kv_cache_arena;
} SparkGlm52PrefixCacheConfiguration;

typedef struct SparkGlm52PrefixCacheLookup
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t requested_token_count;
    uint32_t matched_token_count;
    uint32_t matched_block_count;
    uint32_t next_token_index;
    uint32_t physical_block_index;
    uint32_t reserved;
    uint64_t sequence_id;
    uint64_t last_block_hash;
} SparkGlm52PrefixCacheLookup;

typedef struct SparkGlm52PrefixCacheReservation
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t requested_token_count;
    uint32_t reserved_token_count;
    uint32_t reusable_token_count;
    uint32_t physical_block_count;
    uint32_t cached_physical_block_count;
    uint32_t pending_physical_block_count;
    uint32_t physical_block_capacity;
    uint32_t reserved0;
    uint64_t sequence_id;
    uint64_t reservation_epoch;
    uint64_t last_block_hash;
    uint32_t *physical_block_indices;
} SparkGlm52PrefixCacheReservation;

typedef struct SparkGlm52PrefixCache
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t entry_count;
    uint32_t physical_block_count;
    uint32_t sequence_binding_count;
    SparkGlm52PrefixCacheEntry *entries;
    SparkGlm52PrefixCacheSequenceBinding *sequence_bindings;
    SparkGlm52KvCacheArena *kv_cache_arena;
    uint64_t tick;
    uint64_t operation_epoch;
    uint64_t lookup_count;
    uint64_t hit_count;
    uint64_t miss_count;
    uint64_t inserted_block_count;
    uint64_t evicted_block_count;
    uint64_t acquired_block_count;
    uint64_t released_block_count;
    uint64_t reserved_block_count;
    uint64_t committed_reserved_block_count;
    uint64_t cancelled_reserved_block_count;
    uint64_t live_only_block_count;
    uint64_t lookahead_protection_epoch;
    uint64_t lookahead_protected_block_count;
    uint64_t lookahead_protected_eviction_skip_count;
    uint64_t reuse_scored_resident_eviction_count;
    uint64_t reuse_scored_lookahead_eviction_count;
    uint64_t reuse_scored_untracked_eviction_count;
    uint64_t reuse_scored_capacity_stall_count;
} SparkGlm52PrefixCache;

SparkStatus SparkGlm52PrefixCacheInitialize(
    SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheConfiguration *configuration);

uint64_t SparkGlm52PrefixCacheHashBlock(
    const uint32_t *token_ids,
    uint32_t token_count,
    uint64_t parent_hash);

SparkStatus SparkGlm52PrefixCacheHashPromptTokens(
    uint32_t block_token_count,
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCachePromptHash *prompt_hash);

SparkStatus SparkGlm52PrefixCacheProbePrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup);

SparkStatus SparkGlm52PrefixCacheLookupPrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup);


SparkStatus SparkGlm52PrefixCacheProbePhysicalBlockTable(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *physical_block_count_out);

SparkStatus SparkGlm52PrefixCacheProbeReusablePrefixPrefetchSources(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *source_block_count_out);

SparkStatus SparkGlm52PrefixCacheProbeReusablePrefixResidency(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out);

SparkStatus SparkGlm52PrefixCacheResetLookaheadProtection(
    SparkGlm52PrefixCache *cache);

SparkStatus SparkGlm52PrefixCacheProtectPromptLookahead(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t demand_weight,
    uint32_t *protected_token_count_out,
    uint32_t *protected_block_count_out);

SparkStatus SparkGlm52PrefixCacheTrimResidentBlocksByReuseScore(
    SparkGlm52PrefixCache *cache,
    uint32_t max_resident_block_count,
    const uint32_t *hard_protected_physical_block_indices,
    uint32_t hard_protected_physical_block_count,
    uint32_t *evicted_block_count_out);



SparkStatus SparkGlm52PrefixCacheReservePrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheReservation *reservation);

SparkStatus SparkGlm52PrefixCacheCommitReservation(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch);

SparkStatus SparkGlm52PrefixCacheCancelReservation(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch);

SparkStatus SparkGlm52PrefixCacheCommitPrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup);

SparkStatus SparkGlm52PrefixCacheBuildPhysicalBlockTable(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out);

SparkStatus SparkGlm52PrefixCacheBuildSequencePrefetchSources(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *source_block_count_out);

SparkStatus SparkGlm52PrefixCacheProbeSequenceResidency(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *physical_block_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out);

SparkStatus SparkGlm52PrefixCacheBindCommittedPrefixFromSequence(
    SparkGlm52PrefixCache *cache,
    uint64_t source_sequence_id,
    uint64_t target_sequence_id,
    uint32_t token_count);

SparkStatus SparkGlm52PrefixCacheReleaseSequence(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id);

SparkStatus SparkGlm52PrefixCacheReset(
    SparkGlm52PrefixCache *cache);

#ifdef __cplusplus
}
#endif

#endif
