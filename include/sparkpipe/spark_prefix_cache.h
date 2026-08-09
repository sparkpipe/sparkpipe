#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PREFIX_CACHE_ABI_VERSION 7u
#define SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCacheConfiguration))
#define SPARK_PREFIX_CACHE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCache))
#define SPARK_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCacheEntry))
#define SPARK_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCacheSequenceBinding))
#define SPARK_PREFIX_CACHE_LOOKUP_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCacheLookup))
#define SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCacheReservation))
#define SPARK_PREFIX_CACHE_PROMPT_HASH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCachePromptHash))
#define SPARK_PREFIX_CACHE_HASH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPrefixCachePromptHash))

#define SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID 0x00000001u
#define SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE 0x00000002u
#define SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING 0x00000004u
#define SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY 0x00000008u

#define SPARK_PREFIX_CACHE_BINDING_FLAG_VALID 0x00000001u
#define SPARK_PREFIX_CACHE_BINDING_FLAG_PENDING 0x00000002u

#define SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH 1469598103934665603ull
#define SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK 0xffffffffu
#define SPARK_PREFIX_CACHE_NO_ENTRY 0xffffffffu

#define SPARK_PREFIX_CACHE_MAX_BLOCK_TOKENS 256u

typedef struct SparkPrefixCacheEntry
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t token_count;
    uint32_t first_token_index;
    uint32_t logical_block_index;
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
    uint32_t hash_next;
    uint32_t reserved1;
} SparkPrefixCacheEntry;

typedef struct SparkPrefixCacheSequenceBinding
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t entry_index;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t logical_block_index;
    uint32_t reserved;
    uint64_t sequence_id;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t acquire_epoch;
    uint32_t lookup_hash_next;
    uint32_t sequence_hash_next;
} SparkPrefixCacheSequenceBinding;

typedef struct SparkPrefixCachePromptHash
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
} SparkPrefixCachePromptHash;

typedef struct SparkPrefixCacheConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t entry_count;
    uint32_t logical_block_count;
    uint32_t sequence_binding_count;
    SparkPrefixCacheEntry *entries;
    SparkPrefixCacheSequenceBinding *sequence_bindings;
    SparkKvCacheArena *kv_cache_arena;
    uint32_t entry_hash_bucket_count;
    uint32_t binding_hash_bucket_count;
    uint32_t *entry_hash_bucket_heads;
    uint32_t *binding_lookup_hash_bucket_heads;
    uint32_t *binding_sequence_hash_bucket_heads;
} SparkPrefixCacheConfiguration;

typedef struct SparkPrefixCacheLookup
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t requested_token_count;
    uint32_t matched_token_count;
    uint32_t matched_block_count;
    uint32_t next_token_index;
    uint32_t logical_block_index;
    uint32_t reserved;
    uint64_t sequence_id;
    uint64_t last_block_hash;
} SparkPrefixCacheLookup;

typedef struct SparkPrefixCacheReservation
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t requested_token_count;
    uint32_t reserved_token_count;
    uint32_t reusable_token_count;
    uint32_t logical_block_count;
    uint32_t cached_logical_block_count;
    uint32_t pending_logical_block_count;
    uint32_t logical_block_capacity;
    uint32_t reserved0;
    uint64_t sequence_id;
    uint64_t reservation_epoch;
    uint64_t last_block_hash;
    uint32_t *logical_block_indices;
} SparkPrefixCacheReservation;

typedef struct SparkPrefixCache
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t entry_count;
    uint32_t logical_block_count;
    uint32_t sequence_binding_count;
    SparkPrefixCacheEntry *entries;
    SparkPrefixCacheSequenceBinding *sequence_bindings;
    SparkKvCacheArena *kv_cache_arena;
    uint32_t entry_hash_bucket_count;
    uint32_t binding_hash_bucket_count;
    uint32_t *entry_hash_bucket_heads;
    uint32_t *binding_lookup_hash_bucket_heads;
    uint32_t *binding_sequence_hash_bucket_heads;
    uint32_t free_entry_head;
    uint32_t free_binding_head;
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
} SparkPrefixCache;

SparkStatus SparkPrefixCacheInitialize(
    SparkPrefixCache *cache,
    const SparkPrefixCacheConfiguration *configuration);

uint64_t SparkPrefixCacheHashBlock(
    const uint32_t *token_ids,
    uint32_t token_count,
    uint64_t parent_hash);

SparkStatus SparkPrefixCacheHashPromptTokens(
    uint32_t block_token_count,
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCachePromptHash *prompt_hash);

SparkStatus SparkPrefixCacheProbePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup);

SparkStatus SparkPrefixCacheLookupPrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup);


SparkStatus SparkPrefixCacheProbeLogicalBlockTable(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *logical_block_indices,
    uint32_t logical_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *logical_block_count_out);

SparkStatus SparkPrefixCacheProbeReusablePrefixPrefetchSources(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *source_block_count_out);

SparkStatus SparkPrefixCacheProbeReusablePrefixResidency(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out);

SparkStatus SparkPrefixCacheResetLookaheadProtection(
    SparkPrefixCache *cache);

SparkStatus SparkPrefixCacheProtectPromptLookahead(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t demand_weight,
    uint32_t *protected_token_count_out,
    uint32_t *protected_block_count_out);

SparkStatus SparkPrefixCacheTrimResidentBlocksByReuseScore(
    SparkPrefixCache *cache,
    uint32_t max_resident_block_count,
    const uint32_t *hard_protected_logical_block_indices,
    uint32_t hard_protected_logical_block_count,
    uint32_t *evicted_block_count_out);



SparkStatus SparkPrefixCacheReservePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation);

SparkStatus SparkPrefixCacheReserveSequencePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation);

SparkStatus SparkPrefixCacheCommitReservation(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch);

SparkStatus SparkPrefixCacheCancelReservation(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch);

SparkStatus SparkPrefixCacheCommitPrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup);

/*
 * Ensure that a live sequence owns enough KV blocks to write token_count
 * positions.  Blocks added by this function are live-only: they are never
 * offered as content-addressed prompt-cache hits.  Repeating the call with
 * the same or a smaller token_count is idempotent.
 */
SparkStatus SparkPrefixCacheEnsureSequenceTokenCapacity(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count);

SparkStatus SparkPrefixCacheBuildLogicalBlockTable(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *logical_block_indices,
    uint32_t logical_block_capacity,
    uint32_t *logical_block_count_out);

SparkStatus SparkPrefixCacheBuildSequencePrefetchSources(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *source_block_count_out);

SparkStatus SparkPrefixCacheProbeSequenceResidency(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *logical_block_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out);

SparkStatus SparkPrefixCacheBindCommittedPrefixFromSequence(
    SparkPrefixCache *cache,
    uint64_t source_sequence_id,
    uint64_t target_sequence_id,
    uint32_t token_count);

SparkStatus SparkPrefixCacheReleaseSequence(
    SparkPrefixCache *cache,
    uint64_t sequence_id);

SparkStatus SparkPrefixCacheReset(
    SparkPrefixCache *cache);

#ifdef __cplusplus
}
#endif
