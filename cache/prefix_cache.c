#include "sparkpipe/spark_prefix_cache.h"
#include "sparkpipe/spark_sha256.h"

#include <string.h>

static uint32_t SparkPrefixCacheMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}


static uint32_t SparkPrefixCacheMaximumReusableTokenCount(
    const SparkPrefixCache *cache,
    uint32_t token_count)
{
    if (cache == 0 || token_count <= 1u)
    {
        return 0u;
    }
    return SparkRoundDownToMultipleU32(
        token_count - 1u,
        cache->block_token_count);
}

static uint32_t SparkPrefixCacheFullBlockTokenCount(
    const SparkPrefixCache *cache,
    uint32_t token_count)
{
    if (cache == 0)
    {
        return 0u;
    }
    return SparkRoundDownToMultipleU32(
        token_count,
        cache->block_token_count);
}

static uint64_t SparkPrefixCacheMixU64(
    uint64_t hash_value,
    uint64_t value)
{
    hash_value ^= value;
    hash_value *= 1099511628211ull;
    hash_value ^= hash_value >> 32u;
    return hash_value;
}

static uint32_t SparkPrefixCacheBucketIndex(
    uint64_t hash_value,
    uint32_t bucket_count)
{
    hash_value ^= hash_value >> 33u;
    hash_value *= UINT64_C(0xff51afd7ed558ccd);
    hash_value ^= hash_value >> 33u;
    return (uint32_t)(hash_value % bucket_count);
}

static uint32_t SparkPrefixCacheEntryBucket(
    const SparkPrefixCache *cache,
    uint64_t parent_hash,
    uint64_t block_hash,
    uint64_t content_hash,
    uint32_t first_token_index,
    uint32_t token_count)
{
    uint64_t hash_value;

    hash_value = SparkPrefixCacheMixU64(parent_hash, block_hash);
    hash_value = SparkPrefixCacheMixU64(hash_value, content_hash);
    hash_value = SparkPrefixCacheMixU64(
        hash_value,
        ((uint64_t)first_token_index << 32u) | token_count);
    return SparkPrefixCacheBucketIndex(
        hash_value,
        cache->entry_hash_bucket_count);
}

static uint32_t SparkPrefixCacheBindingLookupBucket(
    const SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t first_token_index)
{
    return SparkPrefixCacheBucketIndex(
        SparkPrefixCacheMixU64(sequence_id, first_token_index),
        cache->binding_hash_bucket_count);
}

static uint32_t SparkPrefixCacheBindingSequenceBucket(
    const SparkPrefixCache *cache,
    uint64_t sequence_id)
{
    return SparkPrefixCacheBucketIndex(
        sequence_id,
        cache->binding_hash_bucket_count);
}

static uint64_t SparkPrefixCacheHashBlockContent(
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint64_t hash_value;
    uint32_t token_index;

    hash_value = 7809847782465536322ull;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        hash_value = SparkPrefixCacheMixU64(
            hash_value,
            ((uint64_t)token_ids[token_index] << 1u) ^
                (uint64_t)(token_index + 0x9e3779b9u));
    }
    return SparkPrefixCacheMixU64(hash_value, token_count);
}

uint64_t SparkPrefixCacheHashBlock(
    const uint32_t *token_ids,
    uint32_t token_count,
    uint64_t parent_hash)
{
    uint64_t hash_value;
    uint32_t token_index;

    hash_value = parent_hash ^ 1099511628211ull;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        hash_value = SparkPrefixCacheMixU64(
            hash_value,
            (uint64_t)token_ids[token_index] +
                ((uint64_t)token_index << 32u));
    }
    return SparkPrefixCacheMixU64(hash_value, token_count);
}

SparkStatus SparkPrefixCacheHashPromptTokens(
    uint32_t block_token_count,
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCachePromptHash *prompt_hash)
{
    uint32_t token_offset;
    uint32_t block_count;
    uint64_t hash_value;

    if (prompt_hash == 0 ||
        block_token_count == 0u ||
        block_token_count > SPARK_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        (token_count != 0u && token_ids == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(prompt_hash, 0, sizeof(*prompt_hash));
    prompt_hash->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    prompt_hash->descriptor_bytes =
        SPARK_PREFIX_CACHE_PROMPT_HASH_DESCRIPTOR_BYTES;
    prompt_hash->token_count = token_count;
    prompt_hash->parent_hash = parent_hash;

    hash_value = parent_hash;
    token_offset = 0u;
    block_count = 0u;
    while (token_offset < token_count)
    {
        uint32_t current_block_token_count;

        current_block_token_count = SparkPrefixCacheMinimumU32(
            block_token_count,
            token_count - token_offset);
        hash_value = SparkPrefixCacheHashBlock(
            &token_ids[token_offset],
            current_block_token_count,
            hash_value);
        token_offset += current_block_token_count;
        block_count += 1u;
    }

    prompt_hash->hashed_token_count = token_count;
    prompt_hash->block_count = block_count;
    prompt_hash->last_block_token_count = token_count == 0u
        ? 0u
        : ((token_count - 1u) % block_token_count) + 1u;
    prompt_hash->prompt_hash = hash_value;
    return SPARK_STATUS_OK;
}

static void SparkPrefixCacheInitializeEntry(
    SparkPrefixCacheEntry *entry)
{
    memset(entry, 0, sizeof(*entry));
    entry->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    entry->descriptor_bytes = SPARK_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES;
    entry->logical_block_index = SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK;
    entry->hash_next = SPARK_PREFIX_CACHE_NO_ENTRY;
}

static void SparkPrefixCacheInitializeBinding(
    SparkPrefixCacheSequenceBinding *binding)
{
    memset(binding, 0, sizeof(*binding));
    binding->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    binding->descriptor_bytes = SPARK_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES;
    binding->entry_index = SPARK_PREFIX_CACHE_NO_ENTRY;
    binding->logical_block_index = SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK;
    binding->lookup_hash_next = SPARK_PREFIX_CACHE_NO_ENTRY;
    binding->sequence_hash_next = SPARK_PREFIX_CACHE_NO_ENTRY;
}

static SparkStatus SparkPrefixCacheValidate(
    const SparkPrefixCache *cache)
{
    if (cache == 0 ||
        cache->abi_version != SPARK_PREFIX_CACHE_ABI_VERSION ||
        cache->descriptor_bytes != SPARK_PREFIX_CACHE_DESCRIPTOR_BYTES ||
        cache->block_token_count == 0u ||
        cache->block_token_count > SPARK_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        cache->entry_count == 0u ||
        cache->logical_block_count == 0u ||
        cache->sequence_binding_count == 0u ||
        cache->entries == 0 ||
        cache->sequence_bindings == 0 ||
        ((cache->entry_hash_bucket_count != 0u) !=
            (cache->entry_hash_bucket_heads != 0)) ||
        ((cache->binding_hash_bucket_count != 0u) !=
            (cache->binding_lookup_hash_bucket_heads != 0)) ||
        ((cache->binding_hash_bucket_count != 0u) !=
            (cache->binding_sequence_hash_bucket_heads != 0)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkPrefixCacheEntryIsReusable(
    const SparkPrefixCache *cache,
    const SparkPrefixCacheEntry *entry)
{
    return cache != 0 && entry != 0 &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) != 0u &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) == 0u &&
        entry->token_count == cache->block_token_count;
}

static uint32_t SparkPrefixCacheEntryIndex(
    const SparkPrefixCache *cache,
    const SparkPrefixCacheEntry *entry)
{
    if (cache == 0 || entry == 0 || entry < cache->entries ||
        entry >= cache->entries + cache->entry_count)
    {
        return SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    return (uint32_t)(entry - cache->entries);
}


static void SparkPrefixCacheDigestBlock(
    const uint32_t *token_ids,
    uint32_t token_count,
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
    SparkSha256Context context;
    SparkSha256Initialize(&context);
    SparkSha256Update(&context, token_ids,
        (size_t)token_count * sizeof(uint32_t));
    SparkSha256Finalize(&context, digest);
}

static SparkPrefixCacheEntry *SparkPrefixCacheFindEntry(
    SparkPrefixCache *cache,
    uint64_t parent_hash,
    uint64_t block_hash,
    uint64_t content_hash,
    const uint8_t content_digest[SPARK_SHA256_DIGEST_BYTES],
    uint32_t first_token_index,
    uint32_t token_count,
    uint32_t reusable_only)
{
    uint32_t entry_index,visited;

    if (cache->entry_hash_bucket_count != 0u)
    {
        entry_index = cache->entry_hash_bucket_heads[
            SparkPrefixCacheEntryBucket(
                cache,
                parent_hash,
                block_hash,
                content_hash,
                first_token_index,
                token_count)];
        for (visited = 0u;
             entry_index != SPARK_PREFIX_CACHE_NO_ENTRY &&
                visited < cache->entry_count;
             ++visited)
        {
            SparkPrefixCacheEntry *entry;

            if (entry_index >= cache->entry_count)
            {
                return 0;
            }
            entry = &cache->entries[entry_index];
            if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
                entry->parent_hash == parent_hash &&
                entry->block_hash == block_hash &&
                entry->content_hash == content_hash &&
                entry->first_token_index == first_token_index &&
                entry->token_count == token_count &&
                ((entry->flags &
                    SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u ||
                 memcmp(entry->content_digest, content_digest,
                    SPARK_SHA256_DIGEST_BYTES) == 0) &&
                (reusable_only == 0u ||
                 SparkPrefixCacheEntryIsReusable(cache, entry)))
            {
                return entry;
            }
            entry_index = entry->hash_next;
        }
        return 0;
    }

    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            entry->parent_hash == parent_hash &&
            entry->block_hash == block_hash &&
            entry->content_hash == content_hash &&
            entry->first_token_index == first_token_index &&
            entry->token_count == token_count &&
            ((entry->flags &
                SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u ||
             memcmp(entry->content_digest, content_digest,
                SPARK_SHA256_DIGEST_BYTES) == 0) &&
            (reusable_only == 0u ||
             SparkPrefixCacheEntryIsReusable(cache, entry)))
        {
            return entry;
        }
    }
    return 0;
}

static SparkPrefixCacheSequenceBinding *SparkPrefixCacheFindBindingAtTokenOffset(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t first_token_index)
{
    uint32_t binding_index,visited;

    if (cache->binding_hash_bucket_count != 0u)
    {
        binding_index = cache->binding_lookup_hash_bucket_heads[
            SparkPrefixCacheBindingLookupBucket(
                cache,
                sequence_id,
                first_token_index)];
        for (visited = 0u;
             binding_index != SPARK_PREFIX_CACHE_NO_ENTRY &&
                visited < cache->sequence_binding_count;
             ++visited)
        {
            SparkPrefixCacheSequenceBinding *binding;

            if (binding_index >= cache->sequence_binding_count)
            {
                return 0;
            }
            binding = &cache->sequence_bindings[binding_index];
            if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
                binding->sequence_id == sequence_id &&
                binding->first_token_index == first_token_index)
            {
                return binding;
            }
            binding_index = binding->lookup_hash_next;
        }
        return 0;
    }

    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->first_token_index == first_token_index)
        {
            return binding;
        }
    }
    return 0;
}

static void SparkPrefixCacheLinkEntry(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entry)
{
    uint32_t bucket,entry_index;

    if (cache->entry_hash_bucket_count == 0u)
    {
        return;
    }
    entry_index = SparkPrefixCacheEntryIndex(cache, entry);
    bucket = SparkPrefixCacheEntryBucket(
        cache,
        entry->parent_hash,
        entry->block_hash,
        entry->content_hash,
        entry->first_token_index,
        entry->token_count);
    entry->hash_next = cache->entry_hash_bucket_heads[bucket];
    cache->entry_hash_bucket_heads[bucket] = entry_index;
}

static SparkStatus SparkPrefixCacheUnlinkEntry(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entry)
{
    uint32_t bucket,current,entry_index,previous,visited;

    if (cache->entry_hash_bucket_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    entry_index = SparkPrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    bucket = SparkPrefixCacheEntryBucket(
        cache,
        entry->parent_hash,
        entry->block_hash,
        entry->content_hash,
        entry->first_token_index,
        entry->token_count);
    previous = SPARK_PREFIX_CACHE_NO_ENTRY;
    current = cache->entry_hash_bucket_heads[bucket];
    for (visited = 0u;
         current != SPARK_PREFIX_CACHE_NO_ENTRY && visited < cache->entry_count;
         ++visited)
    {
        if (current >= cache->entry_count)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        if (current == entry_index)
        {
            if (previous == SPARK_PREFIX_CACHE_NO_ENTRY)
            {
                cache->entry_hash_bucket_heads[bucket] = entry->hash_next;
            }
            else
            {
                cache->entries[previous].hash_next = entry->hash_next;
            }
            entry->hash_next = SPARK_PREFIX_CACHE_NO_ENTRY;
            return SPARK_STATUS_OK;
        }
        previous = current;
        current = cache->entries[current].hash_next;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static void SparkPrefixCacheLinkBinding(
    SparkPrefixCache *cache,
    SparkPrefixCacheSequenceBinding *binding)
{
    uint32_t binding_index,bucket;

    if (cache->binding_hash_bucket_count == 0u)
    {
        return;
    }
    binding_index = (uint32_t)(binding - cache->sequence_bindings);
    bucket = SparkPrefixCacheBindingLookupBucket(
        cache,
        binding->sequence_id,
        binding->first_token_index);
    binding->lookup_hash_next =
        cache->binding_lookup_hash_bucket_heads[bucket];
    cache->binding_lookup_hash_bucket_heads[bucket] = binding_index;
    bucket = SparkPrefixCacheBindingSequenceBucket(
        cache,
        binding->sequence_id);
    binding->sequence_hash_next =
        cache->binding_sequence_hash_bucket_heads[bucket];
    cache->binding_sequence_hash_bucket_heads[bucket] = binding_index;
}

static SparkStatus SparkPrefixCacheUnlinkBindingChain(
    SparkPrefixCacheSequenceBinding *bindings,
    uint32_t binding_count,
    uint32_t *head,
    uint32_t binding_index,
    uint32_t use_sequence_chain)
{
    uint32_t current,next,previous,visited;

    previous = SPARK_PREFIX_CACHE_NO_ENTRY;
    current = *head;
    for (visited = 0u;
         current != SPARK_PREFIX_CACHE_NO_ENTRY && visited < binding_count;
         ++visited)
    {
        if (current >= binding_count)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        next = use_sequence_chain != 0u
            ? bindings[current].sequence_hash_next
            : bindings[current].lookup_hash_next;
        if (current == binding_index)
        {
            if (previous == SPARK_PREFIX_CACHE_NO_ENTRY)
            {
                *head = next;
            }
            else if (use_sequence_chain != 0u)
            {
                bindings[previous].sequence_hash_next = next;
            }
            else
            {
                bindings[previous].lookup_hash_next = next;
            }
            return SPARK_STATUS_OK;
        }
        previous = current;
        current = next;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkPrefixCacheUnlinkBinding(
    SparkPrefixCache *cache,
    SparkPrefixCacheSequenceBinding *binding)
{
    uint32_t binding_index,bucket;
    SparkStatus status;

    if (cache->binding_hash_bucket_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    binding_index = (uint32_t)(binding - cache->sequence_bindings);
    if (binding_index >= cache->sequence_binding_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    bucket = SparkPrefixCacheBindingLookupBucket(
        cache,
        binding->sequence_id,
        binding->first_token_index);
    status = SparkPrefixCacheUnlinkBindingChain(
        cache->sequence_bindings,
        cache->sequence_binding_count,
        &cache->binding_lookup_hash_bucket_heads[bucket],
        binding_index,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    bucket = SparkPrefixCacheBindingSequenceBucket(
        cache,
        binding->sequence_id);
    return SparkPrefixCacheUnlinkBindingChain(
        cache->sequence_bindings,
        cache->sequence_binding_count,
        &cache->binding_sequence_hash_bucket_heads[bucket],
        binding_index,
        1u);
}

static SparkPrefixCacheSequenceBinding *SparkPrefixCacheFindFreeBinding(
    SparkPrefixCache *cache)
{
    if (cache == 0 ||
        cache->free_binding_head == SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        return 0;
    }
    if (cache->free_binding_head >= cache->sequence_binding_count ||
        (cache->sequence_bindings[cache->free_binding_head].flags &
            SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u)
    {
        return 0;
    }
    return &cache->sequence_bindings[cache->free_binding_head];
}

static SparkStatus SparkPrefixCacheAcquireLogicalBlock(
    SparkPrefixCache *cache,
    uint32_t preferred_logical_block_index,
    uint32_t *logical_block_index_out)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaAcquireBlock(
            cache->kv_cache_arena,
            logical_block_index_out);
    }
    if (preferred_logical_block_index >= cache->logical_block_count)
    {
        *logical_block_index_out = SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *logical_block_index_out = preferred_logical_block_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheRecycleLogicalBlock(
    SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaRecycleBlock(
            cache->kv_cache_arena,
            logical_block_index);
    }
    return logical_block_index < cache->logical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefixCacheRetainLogicalBlock(
    SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaRetainBlock(
            cache->kv_cache_arena,
            logical_block_index);
    }
    return logical_block_index < cache->logical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefixCacheReleaseLogicalBlockReference(
    SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaReleaseBlockReference(
            cache->kv_cache_arena,
            logical_block_index);
    }
    return logical_block_index < cache->logical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefixCacheMarkLogicalBlockResident(
    SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaMarkBlockResident(
            cache->kv_cache_arena,
            logical_block_index);
    }
    return logical_block_index < cache->logical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefixCacheFillPrefetchSourceBlock(
    SparkPrefixCache *cache,
    const SparkPrefixCacheEntry *entry,
    SparkKvCachePrefetchSourceBlock *source_block)
{
    if (cache == 0 || entry == 0 || source_block == 0 ||
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
        entry->logical_block_index == SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK ||
        entry->logical_block_index >= cache->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(source_block, 0, sizeof(*source_block));
    source_block->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    source_block->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block->logical_block_index = entry->logical_block_index;
    source_block->token_capacity = cache->block_token_count;
    source_block->first_token_index = entry->first_token_index;
    source_block->token_count = entry->token_count;
    source_block->flags = SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS;
    source_block->parent_hash = entry->parent_hash;
    source_block->block_hash = entry->block_hash;
    source_block->content_hash = entry->content_hash;
    if (cache->kv_cache_arena != 0 &&
        entry->logical_block_index < cache->kv_cache_arena->logical_block_count)
    {
        source_block->generation =
            cache->kv_cache_arena->blocks[entry->logical_block_index].generation;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheFreeLogicalBlock(
    SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkKvCacheArenaFreeBlock(
            cache->kv_cache_arena,
            logical_block_index);
    }
    return logical_block_index < cache->logical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkPrefixCacheEntryHasCurrentLookaheadProtection(
    const SparkPrefixCache *cache,
    const SparkPrefixCacheEntry *entry)
{
    return cache != 0 &&
        entry != 0 &&
        cache->lookahead_protection_epoch != 0u &&
        entry->lookahead_protection_epoch == cache->lookahead_protection_epoch;
}

static uint32_t SparkPrefixCacheEntryIsEvictable(
    const SparkPrefixCacheEntry *entry)
{
    return entry != 0 &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) == 0u &&
        entry->reference_count == 0u;
}

static uint32_t SparkPrefixCacheUnprotectedVictimIsBetter(
    const SparkPrefixCacheEntry *candidate,
    const SparkPrefixCacheEntry *current)
{
    if (current == 0)
    {
        return 1u;
    }
    return candidate->last_used_tick < current->last_used_tick;
}

static uint32_t SparkPrefixCacheProtectedVictimIsBetter(
    const SparkPrefixCacheEntry *candidate,
    const SparkPrefixCacheEntry *current)
{
    if (current == 0)
    {
        return 1u;
    }
    if (candidate->lookahead_priority != current->lookahead_priority)
    {
        return candidate->lookahead_priority < current->lookahead_priority;
    }
    return candidate->last_used_tick < current->last_used_tick;
}

static SparkPrefixCacheEntry *SparkPrefixCacheSelectVictim(
    SparkPrefixCache *cache)
{
    SparkPrefixCacheEntry *unprotected_victim;
    SparkPrefixCacheEntry *protected_victim;
    uint32_t entry_index;

    if (cache->free_entry_head != SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        if (cache->free_entry_head >= cache->entry_count ||
            (cache->entries[cache->free_entry_head].flags &
                SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u)
        {
            return 0;
        }
        return &cache->entries[cache->free_entry_head];
    }

    unprotected_victim = 0;
    protected_victim = 0;
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u)
        {
            return entry;
        }
        if (!SparkPrefixCacheEntryIsEvictable(entry))
        {
            continue;
        }
        if (SparkPrefixCacheEntryHasCurrentLookaheadProtection(
                cache,
                entry))
        {
            if (SparkPrefixCacheProtectedVictimIsBetter(
                    entry,
                    protected_victim))
            {
                protected_victim = entry;
            }
            continue;
        }
        if (SparkPrefixCacheUnprotectedVictimIsBetter(
                entry,
                unprotected_victim))
        {
            unprotected_victim = entry;
        }
    }
    if (unprotected_victim != 0)
    {
        return unprotected_victim;
    }
    if (protected_victim != 0)
    {
        cache->lookahead_protected_eviction_skip_count += 1u;
    }
    return protected_victim;
}

static SparkStatus SparkPrefixCacheInvalidateEntry(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entry)
{
    uint32_t entry_index;
    SparkStatus status;

    if (entry == 0 ||
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (entry->reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    entry_index = SparkPrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (entry->logical_block_index != SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK)
    {
        status = SparkPrefixCacheFreeLogicalBlock(
            cache,
            entry->logical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkPrefixCacheUnlinkEntry(cache, entry);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkPrefixCacheInitializeEntry(entry);
    entry->reserved = cache->free_entry_head;
    cache->free_entry_head = entry_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheInstallEntry(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entry,
    uint32_t entry_flags,
    uint64_t parent_hash,
    uint64_t block_hash,
    uint64_t content_hash,
    const uint8_t content_digest[SPARK_SHA256_DIGEST_BYTES],
    uint32_t first_token_index,
    uint32_t token_count,
    uint64_t operation_epoch)
{
    uint32_t entry_index;
    uint32_t next_free_entry_index;
    uint32_t logical_block_index;
    uint32_t entry_was_valid;
    SparkStatus status;

    entry_index = SparkPrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    entry_was_valid =
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u;
    next_free_entry_index = SPARK_PREFIX_CACHE_NO_ENTRY;
    if (entry_was_valid != 0u)
    {
        if (entry->reference_count != 0u)
        {
            return SPARK_STATUS_BUSY;
        }
        logical_block_index = entry->logical_block_index;
        status = SparkPrefixCacheRecycleLogicalBlock(
            cache,
            logical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkPrefixCacheUnlinkEntry(cache, entry);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        cache->evicted_block_count += 1u;
    }
    else
    {
        if (cache->free_entry_head != entry_index)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        next_free_entry_index = entry->reserved;
        status = SparkPrefixCacheAcquireLogicalBlock(
            cache,
            entry_index,
            &logical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        cache->free_entry_head = next_free_entry_index;
    }

    entry->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    entry->descriptor_bytes = SPARK_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES;
    entry->flags = SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID | entry_flags;
    entry->token_count = token_count;
    entry->first_token_index = first_token_index;
    entry->logical_block_index = logical_block_index;
    entry->reference_count = 0u;
    entry->reserved = 0u;
    entry->parent_hash = parent_hash;
    entry->block_hash = block_hash;
    entry->content_hash = content_hash;
    memcpy(entry->content_digest, content_digest,
        SPARK_SHA256_DIGEST_BYTES);
    entry->reservation_epoch = operation_epoch;
    entry->committed_epoch = 0u;
    entry->lookahead_priority = 0u;
    entry->lookahead_request_count = 0u;
    entry->lookahead_protection_epoch = 0u;
    cache->tick += 1u;
    entry->last_used_tick = cache->tick;
    cache->inserted_block_count += 1u;
    if ((entry_flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u)
    {
        cache->live_only_block_count += 1u;
    }
    SparkPrefixCacheLinkEntry(cache, entry);
    return SPARK_STATUS_OK;
}

static void SparkPrefixCacheInitializeLookup(
    SparkPrefixCacheLookup *lookup,
    uint64_t sequence_id,
    uint32_t requested_token_count)
{
    memset(lookup, 0, sizeof(*lookup));
    lookup->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    lookup->descriptor_bytes = SPARK_PREFIX_CACHE_LOOKUP_DESCRIPTOR_BYTES;
    lookup->requested_token_count = requested_token_count;
    lookup->logical_block_index = SPARK_PREFIX_CACHE_NO_LOGICAL_BLOCK;
    lookup->sequence_id = sequence_id;
    lookup->last_block_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
}

static void SparkPrefixCachePrepareReservationOutput(
    SparkPrefixCacheReservation *reservation,
    uint64_t sequence_id,
    uint32_t requested_token_count,
    uint64_t operation_epoch)
{
    uint32_t *logical_block_indices;
    uint32_t logical_block_capacity;

    logical_block_indices = reservation->logical_block_indices;
    logical_block_capacity = reservation->logical_block_capacity;
    memset(reservation, 0, sizeof(*reservation));
    reservation->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation->descriptor_bytes = SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation->requested_token_count = requested_token_count;
    reservation->sequence_id = sequence_id;
    reservation->reservation_epoch = operation_epoch;
    reservation->last_block_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    reservation->logical_block_indices = logical_block_indices;
    reservation->logical_block_capacity = logical_block_capacity;
}

static SparkStatus SparkPrefixCacheReleaseBinding(
    SparkPrefixCache *cache,
    SparkPrefixCacheSequenceBinding *binding)
{
    uint32_t binding_index;
    SparkStatus status;

    if (binding == 0 ||
        (binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (binding->entry_index >= cache->entry_count ||
        cache->entries[binding->entry_index].reference_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    binding_index = (uint32_t)(binding - cache->sequence_bindings);
    if (binding_index >= cache->sequence_binding_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkPrefixCacheReleaseLogicalBlockReference(
        cache,
        binding->logical_block_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPrefixCacheUnlinkBinding(cache, binding);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    cache->entries[binding->entry_index].reference_count -= 1u;
    cache->released_block_count += 1u;
    SparkPrefixCacheInitializeBinding(binding);
    binding->reserved = cache->free_binding_head;
    cache->free_binding_head = binding_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheAcquireEntryForSequence(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    SparkPrefixCacheEntry *entry,
    uint64_t operation_epoch,
    uint32_t binding_is_pending)
{
    SparkPrefixCacheSequenceBinding *binding;
    uint32_t next_free_binding_index;
    uint32_t entry_index;
    SparkStatus status;

    entry_index = SparkPrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    binding = SparkPrefixCacheFindBindingAtTokenOffset(
        cache,
        sequence_id,
        entry->first_token_index);
    if (binding != 0)
    {
        if (binding->entry_index != entry_index)
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        if (binding_is_pending != 0u)
        {
            binding->flags |= SPARK_PREFIX_CACHE_BINDING_FLAG_PENDING;
        }
        return SPARK_STATUS_OK;
    }
    binding = SparkPrefixCacheFindFreeBinding(cache);
    if (binding == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkPrefixCacheRetainLogicalBlock(
        cache,
        entry->logical_block_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    next_free_binding_index = binding->reserved;
    if (cache->free_binding_head !=
        (uint32_t)(binding - cache->sequence_bindings))
    {
        (void)SparkPrefixCacheReleaseLogicalBlockReference(
            cache,
            entry->logical_block_index);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cache->free_binding_head = next_free_binding_index;

    binding->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    binding->descriptor_bytes = SPARK_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES;
    binding->flags = SPARK_PREFIX_CACHE_BINDING_FLAG_VALID;
    if (binding_is_pending != 0u)
    {
        binding->flags |= SPARK_PREFIX_CACHE_BINDING_FLAG_PENDING;
    }
    binding->entry_index = entry_index;
    binding->first_token_index = entry->first_token_index;
    binding->token_count = entry->token_count;
    binding->logical_block_index = entry->logical_block_index;
    binding->sequence_id = sequence_id;
    binding->parent_hash = entry->parent_hash;
    binding->block_hash = entry->block_hash;
    binding->acquire_epoch = operation_epoch;
    SparkPrefixCacheLinkBinding(cache, binding);
    entry->reference_count += 1u;
    cache->acquired_block_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheRollbackEpoch(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint64_t operation_epoch)
{
    uint32_t binding_index,entry_index,next_binding_index,visited;
    SparkStatus status;

    if (cache->binding_hash_bucket_count != 0u)
    {
        binding_index = cache->binding_sequence_hash_bucket_heads[
            SparkPrefixCacheBindingSequenceBucket(cache, sequence_id)];
        for (visited = 0u;
             binding_index != SPARK_PREFIX_CACHE_NO_ENTRY &&
                visited < cache->sequence_binding_count;
             ++visited)
        {
            SparkPrefixCacheEntry *entry;
            SparkPrefixCacheSequenceBinding *binding;

            if (binding_index >= cache->sequence_binding_count)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            binding = &cache->sequence_bindings[binding_index];
            next_binding_index = binding->sequence_hash_next;
            if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
                binding->sequence_id == sequence_id &&
                binding->acquire_epoch == operation_epoch)
            {
                entry_index = binding->entry_index;
                status = SparkPrefixCacheReleaseBinding(cache, binding);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
                entry = &cache->entries[entry_index];
                if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
                    (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u &&
                    entry->reservation_epoch == operation_epoch &&
                    entry->reference_count == 0u)
                {
                    status = SparkPrefixCacheInvalidateEntry(cache, entry);
                    if (status != SPARK_STATUS_OK)
                    {
                        return status;
                    }
                    cache->cancelled_reserved_block_count += 1u;
                }
            }
            binding_index = next_binding_index;
        }
        return SPARK_STATUS_OK;
    }

    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->acquire_epoch == operation_epoch)
        {
            status = SparkPrefixCacheReleaseBinding(cache, binding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u &&
            entry->reservation_epoch == operation_epoch &&
            entry->reference_count == 0u)
        {
            status = SparkPrefixCacheInvalidateEntry(cache, entry);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            cache->cancelled_reserved_block_count += 1u;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheInitialize(
    SparkPrefixCache *cache,
    const SparkPrefixCacheConfiguration *configuration)
{
    uint32_t binding_index,bucket_index,entry_index;

    if (cache == 0 || configuration == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->abi_version != SPARK_PREFIX_CACHE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->block_token_count == 0u ||
        configuration->block_token_count > SPARK_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        configuration->entry_count == 0u ||
        configuration->logical_block_count == 0u ||
        configuration->sequence_binding_count == 0u ||
        configuration->entries == 0 ||
        configuration->sequence_bindings == 0 ||
        ((configuration->entry_hash_bucket_count != 0u) !=
            (configuration->entry_hash_bucket_heads != 0)) ||
        ((configuration->binding_hash_bucket_count != 0u) !=
            (configuration->binding_lookup_hash_bucket_heads != 0)) ||
        ((configuration->binding_hash_bucket_count != 0u) !=
            (configuration->binding_sequence_hash_bucket_heads != 0)) ||
        (configuration->kv_cache_arena == 0 &&
            configuration->logical_block_count < configuration->entry_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->kv_cache_arena != 0 &&
        (configuration->kv_cache_arena->logical_block_count <
            configuration->logical_block_count ||
         configuration->kv_cache_arena->block_token_count !=
            configuration->block_token_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(cache, 0, sizeof(*cache));
    cache->abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    cache->descriptor_bytes = SPARK_PREFIX_CACHE_DESCRIPTOR_BYTES;
    cache->block_token_count = configuration->block_token_count;
    cache->entry_count = configuration->entry_count;
    cache->logical_block_count = configuration->logical_block_count;
    cache->sequence_binding_count = configuration->sequence_binding_count;
    cache->entries = configuration->entries;
    cache->sequence_bindings = configuration->sequence_bindings;
    cache->kv_cache_arena = configuration->kv_cache_arena;
    cache->entry_hash_bucket_count =
        configuration->entry_hash_bucket_count;
    cache->binding_hash_bucket_count =
        configuration->binding_hash_bucket_count;
    cache->entry_hash_bucket_heads =
        configuration->entry_hash_bucket_heads;
    cache->binding_lookup_hash_bucket_heads =
        configuration->binding_lookup_hash_bucket_heads;
    cache->binding_sequence_hash_bucket_heads =
        configuration->binding_sequence_hash_bucket_heads;
    cache->free_entry_head = 0u;
    cache->free_binding_head = 0u;
    for (bucket_index = 0u;
         bucket_index < cache->entry_hash_bucket_count;
         ++bucket_index)
    {
        cache->entry_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (bucket_index = 0u;
         bucket_index < cache->binding_hash_bucket_count;
         ++bucket_index)
    {
        cache->binding_lookup_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
        cache->binding_sequence_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheInitializeEntry(&cache->entries[entry_index]);
        cache->entries[entry_index].reserved =
            entry_index + 1u < cache->entry_count
                ? entry_index + 1u
                : SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheInitializeBinding(
            &cache->sequence_bindings[binding_index]);
        cache->sequence_bindings[binding_index].reserved =
            binding_index + 1u < cache->sequence_binding_count
                ? binding_index + 1u
                : SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheProbePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup)
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint32_t reusable_token_count;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || lookup == 0 || token_count == 0u || sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkPrefixCacheInitializeLookup(lookup, sequence_id, token_count);
    cache->lookup_count += 1u;
    reusable_token_count = SparkPrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    for (token_offset = 0u;
         token_offset < reusable_token_count;
         token_offset += cache->block_token_count)
    {
        SparkPrefixCacheEntry *entry;

        block_hash = SparkPrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkPrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        entry = 0;
        {
            SparkPrefixCacheSequenceBinding *own_binding;

            own_binding = SparkPrefixCacheFindBindingAtTokenOffset(
                cache,
                sequence_id,
                token_offset);
            if (own_binding != 0 &&
                own_binding->entry_index < cache->entry_count)
            {
                SparkPrefixCacheEntry *own_entry;

                own_entry = &cache->entries[own_binding->entry_index];
                if ((own_entry->flags &
                        SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
                    own_entry->first_token_index == token_offset &&
                    own_entry->token_count == cache->block_token_count &&
                    own_entry->parent_hash == parent_hash &&
                    own_entry->block_hash == block_hash &&
                    own_entry->content_hash == content_hash)
                {
                    entry = own_entry;
                }
            }
        }
        if (entry == 0)
        {
            { uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
            SparkPrefixCacheDigestBlock(&token_ids[token_offset],
                cache->block_token_count, digest);
            entry = SparkPrefixCacheFindEntry(
                cache,
                parent_hash,
                block_hash,
                content_hash,
                digest,
                token_offset,
                cache->block_token_count,
                1u); }
        }
        if (entry == 0)
        {
            break;
        }
        cache->tick += 1u;
        entry->last_used_tick = cache->tick;
        parent_hash = block_hash;
        lookup->matched_token_count += cache->block_token_count;
        lookup->matched_block_count += 1u;
        lookup->logical_block_index = entry->logical_block_index;
        lookup->last_block_hash = block_hash;
    }
    lookup->next_token_index = lookup->matched_token_count;
    if (lookup->matched_token_count != 0u)
    {
        cache->hit_count += 1u;
    }
    else
    {
        cache->miss_count += 1u;
    }
    return SPARK_STATUS_OK;
}


typedef struct SparkPrefixCacheWalk
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint32_t token_offset;
    uint32_t reusable_token_count;
    uint32_t matched_block_count;
} SparkPrefixCacheWalk;

static SparkPrefixCacheEntry *SparkPrefixCacheWalkNext(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheWalk *walk)
{
    SparkPrefixCacheEntry *entry;
    uint64_t content_hash;

    if (walk->matched_block_count == 0u && walk->token_offset == 0u)
    {
        walk->parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
        walk->reusable_token_count =
            SparkPrefixCacheMaximumReusableTokenCount(cache, token_count);
    }
    else
    {
        cache->tick += 1u;
        walk->parent_hash = walk->block_hash;
        walk->token_offset += cache->block_token_count;
    }
    if (walk->token_offset >= walk->reusable_token_count)
    {
        return 0;
    }
    walk->block_hash = SparkPrefixCacheHashBlock(
        &token_ids[walk->token_offset],
        cache->block_token_count,
        walk->parent_hash);
    content_hash = SparkPrefixCacheHashBlockContent(
        &token_ids[walk->token_offset],
        cache->block_token_count);
    { uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    SparkPrefixCacheDigestBlock(&token_ids[walk->token_offset],
        cache->block_token_count, digest);
    entry = SparkPrefixCacheFindEntry(
        cache,
        walk->parent_hash,
        walk->block_hash,
        content_hash,
        digest,
        walk->token_offset,
        cache->block_token_count,
        1u); }
    if (entry == 0)
    {
        return 0;
    }
    entry->last_used_tick = cache->tick + 1u;
    walk->matched_block_count += 1u;
    return entry;
}

SparkStatus SparkPrefixCacheProbeLogicalBlockTable(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *logical_block_indices,
    uint32_t logical_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *logical_block_count_out)
{
    uint32_t logical_block_count;
    SparkPrefixCacheWalk walk;
    SparkPrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u ||
        logical_block_indices == 0 || matched_token_count_out == 0 ||
        logical_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *matched_token_count_out = 0u;
    *logical_block_count_out = 0u;
    logical_block_count = 0u;
    memset(&walk, 0, sizeof(walk));
    while ((entry = SparkPrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (logical_block_count >= logical_block_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        logical_block_indices[logical_block_count] =
            entry->logical_block_index;
        logical_block_count += 1u;
    }

    *matched_token_count_out = logical_block_count * cache->block_token_count;
    *logical_block_count_out = logical_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheProbeReusablePrefixPrefetchSources(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *source_block_count_out)
{
    uint32_t source_block_count;
    SparkPrefixCacheWalk walk;
    SparkPrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u ||
        source_blocks == 0 || matched_token_count_out == 0 ||
        source_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *matched_token_count_out = 0u;
    *source_block_count_out = 0u;
    source_block_count = 0u;
    memset(&walk, 0, sizeof(walk));
    while ((entry = SparkPrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (source_block_count >= source_block_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkPrefixCacheFillPrefetchSourceBlock(
            cache,
            entry,
            &source_blocks[source_block_count]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        source_block_count += 1u;
    }

    *matched_token_count_out = source_block_count * cache->block_token_count;
    *source_block_count_out = source_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheProbeReusablePrefixResidency(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out)
{
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkPrefixCacheWalk walk;
    SparkPrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u ||
        matched_token_count_out == 0 || resident_block_count_out == 0 ||
        nonresident_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *matched_token_count_out = 0u;
    *resident_block_count_out = 0u;
    *nonresident_block_count_out = 0u;
    resident_block_count = 0u;
    nonresident_block_count = 0u;
    memset(&walk, 0, sizeof(walk));
    while ((entry = SparkPrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (cache->kv_cache_arena != 0 &&
            entry->logical_block_index <
                cache->kv_cache_arena->logical_block_count &&
            (cache->kv_cache_arena->blocks[entry->logical_block_index].flags &
                SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
        {
            nonresident_block_count += 1u;
        }
        else
        {
            resident_block_count += 1u;
        }
    }

    *matched_token_count_out =
        (resident_block_count + nonresident_block_count) *
        cache->block_token_count;
    *resident_block_count_out = resident_block_count;
    *nonresident_block_count_out = nonresident_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheResetLookaheadProtection(
    SparkPrefixCache *cache)
{
    uint32_t entry_index;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cache->lookahead_protection_epoch += 1u;
    if (cache->lookahead_protection_epoch == 0u)
    {
        cache->lookahead_protection_epoch = 1u;
        for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
        {
            cache->entries[entry_index].lookahead_protection_epoch = 0u;
            cache->entries[entry_index].lookahead_priority = 0u;
            cache->entries[entry_index].lookahead_request_count = 0u;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheProtectPromptLookahead(
    SparkPrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t demand_weight,
    uint32_t *protected_token_count_out,
    uint32_t *protected_block_count_out)
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint32_t reusable_token_count;
    uint32_t protected_block_count;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u ||
        protected_token_count_out == 0 || protected_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (cache->lookahead_protection_epoch == 0u)
    {
        status = SparkPrefixCacheResetLookaheadProtection(cache);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    reusable_token_count = SparkPrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    protected_block_count = 0u;
    for (token_offset = 0u;
         token_offset < reusable_token_count;
         token_offset += cache->block_token_count)
    {
        SparkPrefixCacheEntry *entry;

        block_hash = SparkPrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkPrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        { uint8_t digest_c[SPARK_SHA256_DIGEST_BYTES];
        SparkPrefixCacheDigestBlock(&token_ids[token_offset],
            cache->block_token_count, digest_c);
        entry = SparkPrefixCacheFindEntry(
            cache,
            parent_hash,
            block_hash,
            content_hash,
            digest_c,
            token_offset,
            cache->block_token_count,
            1u); }
        if (entry == 0)
        {
            break;
        }
        if (entry->lookahead_protection_epoch !=
            cache->lookahead_protection_epoch)
        {
            entry->lookahead_protection_epoch =
                cache->lookahead_protection_epoch;
            entry->lookahead_priority = demand_weight;
            entry->lookahead_request_count = 1u;
            cache->lookahead_protected_block_count += 1u;
        }
        else
        {
            if (demand_weight > entry->lookahead_priority)
            {
                entry->lookahead_priority = demand_weight;
            }
            if (entry->lookahead_request_count != UINT32_MAX)
            {
                entry->lookahead_request_count += 1u;
            }
        }
        protected_block_count += 1u;
        parent_hash = block_hash;
    }

    *protected_block_count_out = protected_block_count;
    *protected_token_count_out = protected_block_count * cache->block_token_count;
    return SPARK_STATUS_OK;
}


#define SPARK_PREFIX_CACHE_REUSE_SCORE_LOOKAHEAD_BASE 1000000000000000ull
#define SPARK_PREFIX_CACHE_REUSE_SCORE_PRIORITY_WEIGHT 1000000000ull
#define SPARK_PREFIX_CACHE_REUSE_SCORE_REQUEST_WEIGHT 10000000ull
#define SPARK_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT 1000000000000ull
#define SPARK_PREFIX_CACHE_REUSE_SCORE_TOKEN_DEPTH_WEIGHT 1024ull

static const SparkPrefixCacheEntry *SparkPrefixCacheFindResidentEntryForLogicalBlock(
    const SparkPrefixCache *cache,
    uint32_t logical_block_index)
{
    const SparkPrefixCacheEntry *best_entry;
    uint32_t entry_index;

    best_entry = 0;
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        const SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
            entry->logical_block_index != logical_block_index)
        {
            continue;
        }
        if (best_entry == 0 ||
            SparkPrefixCacheProtectedVictimIsBetter(
                best_entry,
                entry))
        {
            best_entry = entry;
        }
    }
    return best_entry;
}

static uint64_t SparkPrefixCacheClampU64Product(
    uint64_t left,
    uint64_t right)
{
    if (left != 0u && right > UINT64_MAX / left)
    {
        return UINT64_MAX;
    }
    return left * right;
}

static uint64_t SparkPrefixCacheAddScoreU64(
    uint64_t score,
    uint64_t addend)
{
    if (UINT64_MAX - score < addend)
    {
        return UINT64_MAX;
    }
    return score + addend;
}

static uint64_t SparkPrefixCacheComputeResidentBlockKeepScore(
    const SparkPrefixCache *cache,
    const SparkKvCacheBlock *block,
    const SparkPrefixCacheEntry *entry)
{
    uint64_t score;
    uint32_t prefix_token_depth;

    score = block->last_used_epoch;
    score = SparkPrefixCacheAddScoreU64(
        score,
        SparkPrefixCacheClampU64Product(
            block->reference_count,
            SPARK_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT));
    if (entry == 0)
    {
        return score;
    }

    score = SparkPrefixCacheAddScoreU64(score, entry->last_used_tick);
    score = SparkPrefixCacheAddScoreU64(
        score,
        SparkPrefixCacheClampU64Product(
            entry->reference_count,
            SPARK_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT));

    prefix_token_depth = entry->first_token_index + entry->token_count;
    score = SparkPrefixCacheAddScoreU64(
        score,
        SparkPrefixCacheClampU64Product(
            prefix_token_depth,
            SPARK_PREFIX_CACHE_REUSE_SCORE_TOKEN_DEPTH_WEIGHT));

    if (SparkPrefixCacheEntryHasCurrentLookaheadProtection(cache, entry))
    {
        score = SparkPrefixCacheAddScoreU64(
            score,
            SPARK_PREFIX_CACHE_REUSE_SCORE_LOOKAHEAD_BASE);
        score = SparkPrefixCacheAddScoreU64(
            score,
            SparkPrefixCacheClampU64Product(
                entry->lookahead_priority,
                SPARK_PREFIX_CACHE_REUSE_SCORE_PRIORITY_WEIGHT));
        score = SparkPrefixCacheAddScoreU64(
            score,
            SparkPrefixCacheClampU64Product(
                entry->lookahead_request_count,
                SPARK_PREFIX_CACHE_REUSE_SCORE_REQUEST_WEIGHT));
    }
    return score;
}

typedef struct SparkPrefixCacheResidentEvictionCandidate
{
    uint32_t logical_block_index;
    uint32_t has_prefix_entry;
    uint32_t has_lookahead_protection;
    uint64_t keep_score;
    uint64_t last_used_epoch;
} SparkPrefixCacheResidentEvictionCandidate;

static void SparkPrefixCacheInitializeResidentEvictionCandidate(
    SparkPrefixCacheResidentEvictionCandidate *candidate)
{
    memset(candidate, 0, sizeof(*candidate));
    candidate->logical_block_index = SPARK_KV_CACHE_NO_BLOCK;
    candidate->keep_score = UINT64_MAX;
    candidate->last_used_epoch = UINT64_MAX;
}

static uint32_t SparkPrefixCacheResidentEvictionCandidateIsBetter(
    const SparkPrefixCacheResidentEvictionCandidate *candidate,
    const SparkPrefixCacheResidentEvictionCandidate *current)
{
    if (current->logical_block_index == SPARK_KV_CACHE_NO_BLOCK)
    {
        return 1u;
    }
    if (candidate->keep_score != current->keep_score)
    {
        return candidate->keep_score < current->keep_score;
    }
    if (candidate->has_prefix_entry != current->has_prefix_entry)
    {
        return candidate->has_prefix_entry < current->has_prefix_entry;
    }
    if (candidate->last_used_epoch != current->last_used_epoch)
    {
        return candidate->last_used_epoch < current->last_used_epoch;
    }
    return candidate->logical_block_index < current->logical_block_index;
}

static SparkStatus SparkPrefixCacheSelectResidentReuseScoreVictim(
    SparkPrefixCache *cache,
    const uint32_t *hard_protected_logical_block_indices,
    uint32_t hard_protected_logical_block_count,
    SparkPrefixCacheResidentEvictionCandidate *victim_out)
{
    uint32_t logical_block_index;

    SparkPrefixCacheInitializeResidentEvictionCandidate(victim_out);
    for (logical_block_index = 0u;
         logical_block_index < cache->kv_cache_arena->logical_block_count;
         ++logical_block_index)
    {
        SparkPrefixCacheResidentEvictionCandidate candidate;
        const SparkPrefixCacheEntry *entry;
        const SparkKvCacheBlock *block;

        block = &cache->kv_cache_arena->blocks[logical_block_index];
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            SparkKvProtectedBlockListContainsBlock(
                hard_protected_logical_block_indices,
                hard_protected_logical_block_count,
                logical_block_index))
        {
            continue;
        }

        entry = SparkPrefixCacheFindResidentEntryForLogicalBlock(
            cache,
            logical_block_index);
        candidate.logical_block_index = logical_block_index;
        candidate.has_prefix_entry = entry != 0;
        candidate.has_lookahead_protection =
            SparkPrefixCacheEntryHasCurrentLookaheadProtection(
                cache,
                entry);
        candidate.keep_score = SparkPrefixCacheComputeResidentBlockKeepScore(
            cache,
            block,
            entry);
        candidate.last_used_epoch = block->last_used_epoch;
        if (SparkPrefixCacheResidentEvictionCandidateIsBetter(
                &candidate,
                victim_out))
        {
            *victim_out = candidate;
        }
    }
    return victim_out->logical_block_index == SPARK_KV_CACHE_NO_BLOCK
        ? SPARK_STATUS_NOT_FOUND
        : SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheTrimResidentBlocksByReuseScore(
    SparkPrefixCache *cache,
    uint32_t max_resident_block_count,
    const uint32_t *hard_protected_logical_block_indices,
    uint32_t hard_protected_logical_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (cache->kv_cache_arena == 0 ||
        (hard_protected_logical_block_count != 0u &&
         hard_protected_logical_block_indices == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (max_resident_block_count > cache->kv_cache_arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (cache->kv_cache_arena->resident_block_count > max_resident_block_count)
    {
        SparkPrefixCacheResidentEvictionCandidate victim;

        status = SparkPrefixCacheSelectResidentReuseScoreVictim(
            cache,
            hard_protected_logical_block_indices,
            hard_protected_logical_block_count,
            &victim);
        if (status != SPARK_STATUS_OK)
        {
            cache->reuse_scored_capacity_stall_count += 1u;
            if (cache->kv_cache_arena != 0)
            {
                cache->kv_cache_arena->resident_capacity_stall_count += 1u;
            }
            if (evicted_block_count_out != 0)
            {
                *evicted_block_count_out = evicted_block_count;
            }
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }

        status = SparkKvCacheArenaMarkBlockNonResident(
            cache->kv_cache_arena,
            victim.logical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            if (evicted_block_count_out != 0)
            {
                *evicted_block_count_out = evicted_block_count;
            }
            return status;
        }
        cache->kv_cache_arena->resident_evicted_block_count += 1u;
        cache->reuse_scored_resident_eviction_count += 1u;
        if (victim.has_lookahead_protection != 0u)
        {
            cache->reuse_scored_lookahead_eviction_count += 1u;
        }
        if (victim.has_prefix_entry == 0u)
        {
            cache->reuse_scored_untracked_eviction_count += 1u;
        }
        evicted_block_count += 1u;
    }

    if (evicted_block_count_out != 0)
    {
        *evicted_block_count_out = evicted_block_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheLookupPrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup)
{
    SparkPrefixCacheLookup probe;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t operation_epoch;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkPrefixCacheProbePrompt(
        cache,
        sequence_id,
        token_ids,
        token_count,
        &probe);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;
    parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    for (token_offset = 0u;
         token_offset < probe.matched_token_count;
         token_offset += cache->block_token_count)
    {
        SparkPrefixCacheEntry *entry;

        block_hash = SparkPrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkPrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        { uint8_t digest_c[SPARK_SHA256_DIGEST_BYTES];
        SparkPrefixCacheDigestBlock(&token_ids[token_offset],
            cache->block_token_count, digest_c);
        entry = SparkPrefixCacheFindEntry(
            cache,
            parent_hash,
            block_hash,
            content_hash,
            digest_c,
            token_offset,
            cache->block_token_count,
            1u); }
        if (entry == 0)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        status = SparkPrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            0u);
        if (status != SPARK_STATUS_OK)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        parent_hash = block_hash;
    }
    if (lookup != 0)
    {
        *lookup = probe;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefixCacheReservePromptInternal(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation,
    uint32_t allow_cross_sequence_reuse)
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t operation_epoch;
    uint32_t block_count;
    uint32_t block_index;
    uint32_t token_offset;
    uint32_t reusable_token_count;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u || sequence_id == 0u ||
        reservation == 0 ||
        reservation->abi_version != SPARK_PREFIX_CACHE_ABI_VERSION ||
        reservation->descriptor_bytes !=
            SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkCeilDivU32(
        token_count,
        cache->block_token_count);
    if (reservation->logical_block_indices != 0 &&
        reservation->logical_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;
    SparkPrefixCachePrepareReservationOutput(
        reservation,
        sequence_id,
        token_count,
        operation_epoch);
    reusable_token_count = SparkPrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    token_offset = 0u;

    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkPrefixCacheSequenceBinding *existing_binding;
        SparkPrefixCacheEntry *entry;
        uint32_t block_token_count;
        uint32_t is_full_block;
        uint32_t entry_flags;
        uint32_t binding_is_pending;

        block_token_count = SparkPrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - token_offset);
        is_full_block = block_token_count == cache->block_token_count;
        block_hash = SparkPrefixCacheHashBlock(
            &token_ids[token_offset],
            block_token_count,
            parent_hash);
        content_hash = SparkPrefixCacheHashBlockContent(
            &token_ids[token_offset],
            block_token_count);
        existing_binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        entry = 0;
        if (existing_binding != 0 && existing_binding->entry_index < cache->entry_count)
        {
            entry = &cache->entries[existing_binding->entry_index];
        }
        if (entry == 0 && allow_cross_sequence_reuse != 0u &&
            is_full_block != 0u && token_offset < reusable_token_count)
        {
            { uint8_t digest_d[SPARK_SHA256_DIGEST_BYTES];
            SparkPrefixCacheDigestBlock(&token_ids[token_offset],
                block_token_count, digest_d);
            entry = SparkPrefixCacheFindEntry(
                cache,
                parent_hash,
                block_hash,
                content_hash,
                digest_d,
                token_offset,
                block_token_count,
                1u); }
        }
        if (entry == 0)
        {
            entry = SparkPrefixCacheSelectVictim(cache);
            if (entry == 0)
            {
                SparkPrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            entry_flags = SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING;
            if (is_full_block == 0u || allow_cross_sequence_reuse == 0u)
            {
                entry_flags |= SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
            }
            { uint8_t install_digest[SPARK_SHA256_DIGEST_BYTES];
            SparkPrefixCacheDigestBlock(&token_ids[token_offset],
                block_token_count, install_digest);
            status = SparkPrefixCacheInstallEntry(
                cache,
                entry,
                entry_flags,
                parent_hash,
                block_hash,
                content_hash,
                install_digest,
                token_offset,
                block_token_count,
                operation_epoch); }
            if (status != SPARK_STATUS_OK)
            {
                SparkPrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return status;
            }
            reservation->pending_logical_block_count += 1u;
            cache->reserved_block_count += 1u;
        }
        else if (SparkPrefixCacheEntryIsReusable(cache, entry))
        {
            reservation->cached_logical_block_count += 1u;
            reservation->reusable_token_count += block_token_count;
        }
        else if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u)
        {
            reservation->pending_logical_block_count += 1u;
        }

        binding_is_pending =
            (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u;
        status = SparkPrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            binding_is_pending);
        if (status != SPARK_STATUS_OK)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        if (reservation->logical_block_indices != 0)
        {
            reservation->logical_block_indices[block_index] =
                entry->logical_block_index;
        }
        cache->tick += 1u;
        entry->last_used_tick = cache->tick;
        parent_hash = block_hash;
        token_offset += block_token_count;
    }

    reservation->reserved_token_count = token_count;
    reservation->logical_block_count = block_count;
    reservation->last_block_hash = parent_hash;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheReservePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation)
{
    return SparkPrefixCacheReservePromptInternal(
        cache,sequence_id,token_ids,token_count,reservation,1u);
}

SparkStatus SparkPrefixCacheReserveSequencePrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation)
{
    return SparkPrefixCacheReservePromptInternal(
        cache,sequence_id,token_ids,token_count,reservation,0u);
}

static SparkStatus SparkPrefixCacheCommitEntry(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entry,
    uint64_t reservation_epoch)
{
    SparkStatus status;

    if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) == 0u ||
        entry->reservation_epoch != reservation_epoch)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkPrefixCacheMarkLogicalBlockResident(
        cache,
        entry->logical_block_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    entry->flags &= ~SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING;
    if (entry->token_count == cache->block_token_count &&
        (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) == 0u)
    {
        entry->flags |= SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE;
    }
    else
    {
        entry->flags |= SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
    }
    entry->committed_epoch = reservation_epoch;
    cache->committed_reserved_block_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCommitReservation(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch)
{
    uint32_t binding_index,entry_index,visited;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || reservation_epoch == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (cache->binding_hash_bucket_count != 0u)
    {
        binding_index = cache->binding_sequence_hash_bucket_heads[
            SparkPrefixCacheBindingSequenceBucket(cache, sequence_id)];
        for (visited = 0u;
             binding_index != SPARK_PREFIX_CACHE_NO_ENTRY &&
                visited < cache->sequence_binding_count;
             ++visited)
        {
            SparkPrefixCacheSequenceBinding *binding;

            if (binding_index >= cache->sequence_binding_count)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            binding = &cache->sequence_bindings[binding_index];
            if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
                binding->sequence_id == sequence_id &&
                binding->acquire_epoch == reservation_epoch)
            {
                if (binding->entry_index >= cache->entry_count)
                {
                    return SPARK_STATUS_INTERNAL_ERROR;
                }
                status = SparkPrefixCacheCommitEntry(
                    cache,
                    &cache->entries[binding->entry_index],
                    reservation_epoch);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
                binding->flags &= ~SPARK_PREFIX_CACHE_BINDING_FLAG_PENDING;
            }
            binding_index = binding->sequence_hash_next;
        }
        return SPARK_STATUS_OK;
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        status = SparkPrefixCacheCommitEntry(
            cache,
            entry,
            reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->acquire_epoch == reservation_epoch)
        {
            binding->flags &= ~SPARK_PREFIX_CACHE_BINDING_FLAG_PENDING;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCancelReservation(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch)
{
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || reservation_epoch == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkPrefixCacheRollbackEpoch(
        cache,
        sequence_id,
        reservation_epoch);
}

SparkStatus SparkPrefixCacheCommitPrompt(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkPrefixCacheLookup *lookup)
{
    SparkPrefixCacheReservation reservation;
    uint32_t committed_token_count;
    SparkStatus status;

    if (lookup != 0)
    {
        SparkPrefixCacheInitializeLookup(lookup, sequence_id, token_count);
    }
    memset(&reservation, 0, sizeof(reservation));
    reservation.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes = SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    status = SparkPrefixCacheReservePrompt(
        cache,
        sequence_id,
        token_ids,
        token_count,
        &reservation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPrefixCacheCommitReservation(
        cache,
        sequence_id,
        reservation.reservation_epoch);
    if (status != SPARK_STATUS_OK)
    {
        SparkPrefixCacheCancelReservation(
            cache,
            sequence_id,
            reservation.reservation_epoch);
        return status;
    }
    if (lookup != 0)
    {
        committed_token_count = SparkPrefixCacheFullBlockTokenCount(
            cache,
            token_count);
        lookup->matched_token_count = committed_token_count;
        lookup->matched_block_count = committed_token_count / cache->block_token_count;
        lookup->next_token_index = committed_token_count;
        lookup->last_block_hash = reservation.last_block_hash;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheEnsureSequenceTokenCapacity(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count)
{
    SparkPrefixCacheSequenceBinding *short_binding;
    SparkPrefixCacheEntry *short_entry;
    uint64_t operation_epoch;
    uint64_t parent_hash;
    uint32_t block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkCeilDivU32(
        token_count,
        cache->block_token_count);
    short_binding = 0;
    short_entry = 0;
    parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;

    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkPrefixCacheSequenceBinding *binding;
        SparkPrefixCacheEntry *entry;
        uint64_t block_hash;
        uint64_t content_hash;
        uint32_t first_token_index;
        uint32_t required_block_token_count;

        first_token_index = block_index * cache->block_token_count;
        required_block_token_count = SparkPrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - first_token_index);
        binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            first_token_index);
        if (binding != 0)
        {
            if (binding->entry_index >= cache->entry_count ||
                binding->logical_block_index >= cache->logical_block_count ||
                binding->token_count == 0u ||
                binding->token_count > cache->block_token_count)
            {
                SparkPrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            entry = &cache->entries[binding->entry_index];
            if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
                (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u ||
                entry->first_token_index != first_token_index ||
                entry->logical_block_index != binding->logical_block_index ||
                entry->token_count != binding->token_count)
            {
                SparkPrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_BUSY;
            }
            if (binding->token_count < required_block_token_count)
            {
                if ((entry->flags &
                        SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) == 0u ||
                    entry->reference_count != 1u || short_binding != 0)
                {
                    SparkPrefixCacheRollbackEpoch(
                        cache,
                        sequence_id,
                        operation_epoch);
                    return SPARK_STATUS_MODULE_NOT_VALIDATED;
                }
                short_binding = binding;
                short_entry = entry;
            }
            parent_hash = binding->block_hash;
            continue;
        }

        entry = SparkPrefixCacheSelectVictim(cache);
        if (entry == 0)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        block_hash = SparkPrefixCacheMixU64(
            SparkPrefixCacheMixU64(parent_hash, sequence_id),
            first_token_index);
        content_hash = SparkPrefixCacheMixU64(
            block_hash,
            operation_epoch);
        { uint8_t live_digest[SPARK_SHA256_DIGEST_BYTES];
        memset(live_digest, 0, sizeof(live_digest));
        status = SparkPrefixCacheInstallEntry(
            cache,
            entry,
            SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING |
                SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY,
            parent_hash,
            block_hash,
            content_hash,
            live_digest,
            first_token_index,
            cache->block_token_count,
            operation_epoch); }
        if (status != SPARK_STATUS_OK)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        cache->reserved_block_count += 1u;
        status = SparkPrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        parent_hash = block_hash;
    }

    status = SparkPrefixCacheCommitReservation(
        cache,
        sequence_id,
        operation_epoch);
    if (status != SPARK_STATUS_OK)
    {
        SparkPrefixCacheRollbackEpoch(
            cache,
            sequence_id,
            operation_epoch);
        return status;
    }
    if (short_binding != 0)
    {
        short_binding->token_count = cache->block_token_count;
        short_entry->token_count = cache->block_token_count;
        short_entry->flags &= ~SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE;
        short_entry->flags |= SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheBuildLogicalBlockTable(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *logical_block_indices,
    uint32_t logical_block_capacity,
    uint32_t *logical_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        logical_block_indices == 0 || logical_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_count = SparkCeilDivU32(
        token_count,
        cache->block_token_count);
    if (logical_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkPrefixCacheSequenceBinding *binding;
        uint32_t required_block_token_count;
        uint32_t token_offset;

        token_offset = block_index * cache->block_token_count;
        required_block_token_count = SparkPrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - token_offset);

        binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0 || binding->token_count < required_block_token_count)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        logical_block_indices[block_index] = binding->logical_block_index;
    }
    *logical_block_count_out = block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheBuildSequencePrefetchSources(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *source_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        source_blocks == 0 || source_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_count = SparkCeilDivU32(
        token_count,
        cache->block_token_count);
    if (source_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    token_offset = 0u;
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkPrefixCacheSequenceBinding *binding;
        SparkPrefixCacheEntry *entry;

        binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0 || binding->entry_index >= cache->entry_count)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        entry = &cache->entries[binding->entry_index];
        status = SparkPrefixCacheFillPrefetchSourceBlock(
            cache,
            entry,
            &source_blocks[block_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        token_offset += binding->token_count;
    }
    *source_block_count_out = block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheProbeSequenceResidency(
    SparkPrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *logical_block_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    uint32_t token_offset;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        logical_block_count_out == 0 || resident_block_count_out == 0 ||
        nonresident_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkCeilDivU32(
        token_count,
        cache->block_token_count);
    token_offset = 0u;
    resident_block_count = 0u;
    nonresident_block_count = 0u;
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkPrefixCacheSequenceBinding *binding;

        binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (cache->kv_cache_arena != 0 &&
            binding->logical_block_index < cache->kv_cache_arena->logical_block_count &&
            (cache->kv_cache_arena->blocks[binding->logical_block_index].flags &
                SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
        {
            nonresident_block_count += 1u;
        }
        else
        {
            resident_block_count += 1u;
        }
        token_offset += binding->token_count;
    }

    *logical_block_count_out = block_count;
    *resident_block_count_out = resident_block_count;
    *nonresident_block_count_out = nonresident_block_count;
    return SPARK_STATUS_OK;
}


SparkStatus SparkPrefixCacheBindCommittedPrefixFromSequence(
    SparkPrefixCache *cache,
    uint64_t source_sequence_id,
    uint64_t target_sequence_id,
    uint32_t token_count)
{
    uint64_t operation_epoch;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (source_sequence_id == 0u || target_sequence_id == 0u ||
        source_sequence_id == target_sequence_id || token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;
    token_offset = 0u;
    while (token_offset < token_count)
    {
        SparkPrefixCacheSequenceBinding *source_binding;
        SparkPrefixCacheSequenceBinding *target_binding;
        SparkPrefixCacheEntry *entry;

        source_binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            source_sequence_id,
            token_offset);
        if (source_binding == 0 ||
            source_binding->entry_index >= cache->entry_count ||
            source_binding->token_count == 0u ||
            token_offset + source_binding->token_count > token_count)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_NOT_FOUND;
        }
        entry = &cache->entries[source_binding->entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
            (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_BUSY;
        }
        target_binding = SparkPrefixCacheFindBindingAtTokenOffset(
            cache,
            target_sequence_id,
            token_offset);
        if (target_binding != 0 && target_binding->entry_index !=
            source_binding->entry_index)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_DUPLICATE;
        }
        status = SparkPrefixCacheAcquireEntryForSequence(
            cache,
            target_sequence_id,
            entry,
            operation_epoch,
            0u);
        if (status != SPARK_STATUS_OK)
        {
            SparkPrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return status;
        }
        token_offset += source_binding->token_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheReleaseSequence(
    SparkPrefixCache *cache,
    uint64_t sequence_id)
{
    uint32_t binding_index,entry_index,next_binding_index,visited;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (cache->binding_hash_bucket_count != 0u)
    {
        binding_index = cache->binding_sequence_hash_bucket_heads[
            SparkPrefixCacheBindingSequenceBucket(cache, sequence_id)];
        for (visited = 0u;
             binding_index != SPARK_PREFIX_CACHE_NO_ENTRY &&
                visited < cache->sequence_binding_count;
             ++visited)
        {
            SparkPrefixCacheEntry *entry;
            SparkPrefixCacheSequenceBinding *binding;

            if (binding_index >= cache->sequence_binding_count)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            binding = &cache->sequence_bindings[binding_index];
            next_binding_index = binding->sequence_hash_next;
            if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
                binding->sequence_id == sequence_id)
            {
                entry_index = binding->entry_index;
                status = SparkPrefixCacheReleaseBinding(cache, binding);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
                entry = &cache->entries[entry_index];
                if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
                    (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) == 0u &&
                    entry->reference_count == 0u)
                {
                    status = SparkPrefixCacheInvalidateEntry(cache, entry);
                    if (status != SPARK_STATUS_OK)
                    {
                        return status;
                    }
                }
            }
            binding_index = next_binding_index;
        }
        return SPARK_STATUS_OK;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id)
        {
            status = SparkPrefixCacheReleaseBinding(cache, binding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            (entry->flags & SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) == 0u &&
            entry->reference_count == 0u)
        {
            status = SparkPrefixCacheInvalidateEntry(cache, entry);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheReset(
    SparkPrefixCache *cache)
{
    uint32_t binding_index,bucket_index,entry_index;
    SparkStatus status;

    status = SparkPrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkPrefixCacheInitializeEntry(&cache->entries[entry_index]);
        cache->entries[entry_index].reserved =
            entry_index + 1u < cache->entry_count
                ? entry_index + 1u
                : SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkPrefixCacheInitializeBinding(
            &cache->sequence_bindings[binding_index]);
        cache->sequence_bindings[binding_index].reserved =
            binding_index + 1u < cache->sequence_binding_count
                ? binding_index + 1u
                : SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (bucket_index = 0u;
         bucket_index < cache->entry_hash_bucket_count;
         ++bucket_index)
    {
        cache->entry_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    for (bucket_index = 0u;
         bucket_index < cache->binding_hash_bucket_count;
         ++bucket_index)
    {
        cache->binding_lookup_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
        cache->binding_sequence_hash_bucket_heads[bucket_index] =
            SPARK_PREFIX_CACHE_NO_ENTRY;
    }
    if (cache->kv_cache_arena != 0)
    {
        status = SparkKvCacheArenaReset(cache->kv_cache_arena);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    cache->tick = 0u;
    cache->free_entry_head = 0u;
    cache->free_binding_head = 0u;
    cache->operation_epoch = 0u;
    cache->lookup_count = 0u;
    cache->hit_count = 0u;
    cache->miss_count = 0u;
    cache->inserted_block_count = 0u;
    cache->evicted_block_count = 0u;
    cache->acquired_block_count = 0u;
    cache->released_block_count = 0u;
    cache->reserved_block_count = 0u;
    cache->committed_reserved_block_count = 0u;
    cache->cancelled_reserved_block_count = 0u;
    cache->live_only_block_count = 0u;
    cache->lookahead_protection_epoch = 0u;
    cache->lookahead_protected_block_count = 0u;
    cache->lookahead_protected_eviction_skip_count = 0u;
    return SPARK_STATUS_OK;
}
