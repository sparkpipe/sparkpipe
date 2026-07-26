#include "sparkpipe/spark_glm52_prefix_cache.h"

#include <string.h>

static uint32_t SparkGlm52PrefixCacheMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52PrefixCacheRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return 0u;
    }
    return value - (value % multiple);
}

static uint32_t SparkGlm52PrefixCacheCeilDivideU32(
    uint32_t numerator,
    uint32_t denominator)
{
    if (denominator == 0u)
    {
        return 0u;
    }
    return (numerator + denominator - 1u) / denominator;
}

static uint32_t SparkGlm52PrefixCacheMaximumReusableTokenCount(
    const SparkGlm52PrefixCache *cache,
    uint32_t token_count)
{
    if (cache == 0 || token_count <= 1u)
    {
        return 0u;
    }
    return SparkGlm52PrefixCacheRoundDownToMultiple(
        token_count - 1u,
        cache->block_token_count);
}

static uint32_t SparkGlm52PrefixCacheFullBlockTokenCount(
    const SparkGlm52PrefixCache *cache,
    uint32_t token_count)
{
    if (cache == 0)
    {
        return 0u;
    }
    return SparkGlm52PrefixCacheRoundDownToMultiple(
        token_count,
        cache->block_token_count);
}

static uint64_t SparkGlm52PrefixCacheMixU64(
    uint64_t hash_value,
    uint64_t value)
{
    hash_value ^= value;
    hash_value *= 1099511628211ull;
    hash_value ^= hash_value >> 32u;
    return hash_value;
}

static uint64_t SparkGlm52PrefixCacheHashBlockContent(
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint64_t hash_value;
    uint32_t token_index;

    hash_value = 7809847782465536322ull;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        hash_value = SparkGlm52PrefixCacheMixU64(
            hash_value,
            ((uint64_t)token_ids[token_index] << 1u) ^
                (uint64_t)(token_index + 0x9e3779b9u));
    }
    return SparkGlm52PrefixCacheMixU64(hash_value, token_count);
}

uint64_t SparkGlm52PrefixCacheHashBlock(
    const uint32_t *token_ids,
    uint32_t token_count,
    uint64_t parent_hash)
{
    uint64_t hash_value;
    uint32_t token_index;

    hash_value = parent_hash ^ 1099511628211ull;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        hash_value = SparkGlm52PrefixCacheMixU64(
            hash_value,
            (uint64_t)token_ids[token_index] +
                ((uint64_t)token_index << 32u));
    }
    return SparkGlm52PrefixCacheMixU64(hash_value, token_count);
}

SparkStatus SparkGlm52PrefixCacheHashPromptTokens(
    uint32_t block_token_count,
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCachePromptHash *prompt_hash)
{
    uint32_t token_offset;
    uint32_t block_count;
    uint64_t hash_value;

    if (prompt_hash == 0 ||
        block_token_count == 0u ||
        block_token_count > SPARK_GLM52_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        (token_count != 0u && token_ids == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(prompt_hash, 0, sizeof(*prompt_hash));
    prompt_hash->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prompt_hash->descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_PROMPT_HASH_DESCRIPTOR_BYTES;
    prompt_hash->token_count = token_count;
    prompt_hash->parent_hash = parent_hash;

    hash_value = parent_hash;
    token_offset = 0u;
    block_count = 0u;
    while (token_offset < token_count)
    {
        uint32_t current_block_token_count;

        current_block_token_count = SparkGlm52PrefixCacheMinimumU32(
            block_token_count,
            token_count - token_offset);
        hash_value = SparkGlm52PrefixCacheHashBlock(
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

static void SparkGlm52PrefixCacheInitializeEntry(
    SparkGlm52PrefixCacheEntry *entry)
{
    memset(entry, 0, sizeof(*entry));
    entry->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    entry->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES;
    entry->physical_block_index = SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK;
}

static void SparkGlm52PrefixCacheInitializeBinding(
    SparkGlm52PrefixCacheSequenceBinding *binding)
{
    memset(binding, 0, sizeof(*binding));
    binding->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    binding->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES;
    binding->entry_index = SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    binding->physical_block_index = SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK;
}

static SparkStatus SparkGlm52PrefixCacheValidate(
    const SparkGlm52PrefixCache *cache)
{
    if (cache == 0 ||
        cache->abi_version != SPARK_GLM52_PREFIX_CACHE_ABI_VERSION ||
        cache->descriptor_bytes != SPARK_GLM52_PREFIX_CACHE_DESCRIPTOR_BYTES ||
        cache->block_token_count == 0u ||
        cache->block_token_count > SPARK_GLM52_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        cache->entry_count == 0u ||
        cache->physical_block_count == 0u ||
        cache->sequence_binding_count == 0u ||
        cache->entries == 0 ||
        cache->sequence_bindings == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52PrefixCacheEntryIsReusable(
    const SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheEntry *entry)
{
    return cache != 0 && entry != 0 &&
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) != 0u &&
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) == 0u &&
        entry->token_count == cache->block_token_count;
}

static uint32_t SparkGlm52PrefixCacheEntryIndex(
    const SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheEntry *entry)
{
    if (cache == 0 || entry == 0 || entry < cache->entries ||
        entry >= cache->entries + cache->entry_count)
    {
        return SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    }
    return (uint32_t)(entry - cache->entries);
}

static SparkGlm52PrefixCacheEntry *SparkGlm52PrefixCacheFindEntry(
    SparkGlm52PrefixCache *cache,
    uint64_t parent_hash,
    uint64_t block_hash,
    uint64_t content_hash,
    uint32_t first_token_index,
    uint32_t token_count,
    uint32_t reusable_only)
{
    uint32_t entry_index;

    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            entry->parent_hash == parent_hash &&
            entry->block_hash == block_hash &&
            entry->content_hash == content_hash &&
            entry->first_token_index == first_token_index &&
            entry->token_count == token_count &&
            (reusable_only == 0u ||
             SparkGlm52PrefixCacheEntryIsReusable(cache, entry)))
        {
            return entry;
        }
    }
    return 0;
}

static SparkGlm52PrefixCacheSequenceBinding *SparkGlm52PrefixCacheFindBinding(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t entry_index)
{
    uint32_t binding_index;

    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->entry_index == entry_index)
        {
            return binding;
        }
    }
    return 0;
}

static SparkGlm52PrefixCacheSequenceBinding *SparkGlm52PrefixCacheFindBindingAtTokenOffset(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t first_token_index)
{
    uint32_t binding_index;

    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->first_token_index == first_token_index)
        {
            return binding;
        }
    }
    return 0;
}

static SparkGlm52PrefixCacheSequenceBinding *SparkGlm52PrefixCacheFindFreeBinding(
    SparkGlm52PrefixCache *cache)
{
    if (cache == 0 ||
        cache->free_binding_head == SPARK_GLM52_PREFIX_CACHE_NO_ENTRY)
    {
        return 0;
    }
    if (cache->free_binding_head >= cache->sequence_binding_count ||
        (cache->sequence_bindings[cache->free_binding_head].flags &
            SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u)
    {
        return 0;
    }
    return &cache->sequence_bindings[cache->free_binding_head];
}

static uint32_t SparkGlm52PrefixCachePhysicalBlockIsUsed(
    const SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    uint32_t entry_index;

    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        const SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            entry->physical_block_index == physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static SparkStatus SparkGlm52PrefixCacheAcquirePhysicalBlock(
    SparkGlm52PrefixCache *cache,
    uint32_t *physical_block_index_out)
{
    uint32_t physical_block_index;

    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaAcquireBlock(
            cache->kv_cache_arena,
            physical_block_index_out);
    }
    for (physical_block_index = 0u;
         physical_block_index < cache->physical_block_count;
         ++physical_block_index)
    {
        if (!SparkGlm52PrefixCachePhysicalBlockIsUsed(
                cache,
                physical_block_index))
        {
            *physical_block_index_out = physical_block_index;
            return SPARK_STATUS_OK;
        }
    }
    *physical_block_index_out = SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK;
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static SparkStatus SparkGlm52PrefixCacheRecyclePhysicalBlock(
    SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaRecycleBlock(
            cache->kv_cache_arena,
            physical_block_index);
    }
    return physical_block_index < cache->physical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52PrefixCacheRetainPhysicalBlock(
    SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaRetainBlock(
            cache->kv_cache_arena,
            physical_block_index);
    }
    return physical_block_index < cache->physical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52PrefixCacheReleasePhysicalBlockReference(
    SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaReleaseBlockReference(
            cache->kv_cache_arena,
            physical_block_index);
    }
    return physical_block_index < cache->physical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52PrefixCacheMarkPhysicalBlockResident(
    SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaMarkBlockResident(
            cache->kv_cache_arena,
            physical_block_index);
    }
    return physical_block_index < cache->physical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52PrefixCacheFillPrefetchSourceBlock(
    SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheEntry *entry,
    SparkGlm52KvCachePrefetchSourceBlock *source_block)
{
    if (cache == 0 || entry == 0 || source_block == 0 ||
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
        entry->physical_block_index == SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK ||
        entry->physical_block_index >= cache->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(source_block, 0, sizeof(*source_block));
    source_block->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    source_block->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block->physical_block_index = entry->physical_block_index;
    source_block->token_capacity = cache->block_token_count;
    source_block->first_token_index = entry->first_token_index;
    source_block->token_count = entry->token_count;
    source_block->flags = SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS;
    source_block->parent_hash = entry->parent_hash;
    source_block->block_hash = entry->block_hash;
    source_block->content_hash = entry->content_hash;
    if (cache->kv_cache_arena != 0 &&
        entry->physical_block_index < cache->kv_cache_arena->physical_block_count)
    {
        source_block->generation =
            cache->kv_cache_arena->blocks[entry->physical_block_index].generation;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PrefixCacheFreePhysicalBlock(
    SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    if (cache->kv_cache_arena != 0)
    {
        return SparkGlm52KvCacheArenaFreeBlock(
            cache->kv_cache_arena,
            physical_block_index);
    }
    return physical_block_index < cache->physical_block_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkGlm52PrefixCacheEntryHasCurrentLookaheadProtection(
    const SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheEntry *entry)
{
    return cache != 0 &&
        entry != 0 &&
        cache->lookahead_protection_epoch != 0u &&
        entry->lookahead_protection_epoch == cache->lookahead_protection_epoch;
}

static uint32_t SparkGlm52PrefixCacheEntryIsEvictable(
    const SparkGlm52PrefixCacheEntry *entry)
{
    return entry != 0 &&
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) == 0u &&
        entry->reference_count == 0u;
}

static uint32_t SparkGlm52PrefixCacheUnprotectedVictimIsBetter(
    const SparkGlm52PrefixCacheEntry *candidate,
    const SparkGlm52PrefixCacheEntry *current)
{
    if (current == 0)
    {
        return 1u;
    }
    return candidate->last_used_tick < current->last_used_tick;
}

static uint32_t SparkGlm52PrefixCacheProtectedVictimIsBetter(
    const SparkGlm52PrefixCacheEntry *candidate,
    const SparkGlm52PrefixCacheEntry *current)
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

static SparkGlm52PrefixCacheEntry *SparkGlm52PrefixCacheSelectVictim(
    SparkGlm52PrefixCache *cache)
{
    SparkGlm52PrefixCacheEntry *unprotected_victim;
    SparkGlm52PrefixCacheEntry *protected_victim;
    uint32_t entry_index;

    if (cache->free_entry_head != SPARK_GLM52_PREFIX_CACHE_NO_ENTRY)
    {
        if (cache->free_entry_head >= cache->entry_count ||
            (cache->entries[cache->free_entry_head].flags &
                SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u)
        {
            return 0;
        }
        return &cache->entries[cache->free_entry_head];
    }

    unprotected_victim = 0;
    protected_victim = 0;
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u)
        {
            return entry;
        }
        if (!SparkGlm52PrefixCacheEntryIsEvictable(entry))
        {
            continue;
        }
        if (SparkGlm52PrefixCacheEntryHasCurrentLookaheadProtection(
                cache,
                entry))
        {
            if (SparkGlm52PrefixCacheProtectedVictimIsBetter(
                    entry,
                    protected_victim))
            {
                protected_victim = entry;
            }
            continue;
        }
        if (SparkGlm52PrefixCacheUnprotectedVictimIsBetter(
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

static SparkStatus SparkGlm52PrefixCacheInvalidateEntry(
    SparkGlm52PrefixCache *cache,
    SparkGlm52PrefixCacheEntry *entry)
{
    uint32_t entry_index;
    SparkStatus status;

    if (entry == 0 ||
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (entry->reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    entry_index = SparkGlm52PrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_GLM52_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (entry->physical_block_index != SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK)
    {
        status = SparkGlm52PrefixCacheFreePhysicalBlock(
            cache,
            entry->physical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    SparkGlm52PrefixCacheInitializeEntry(entry);
    entry->reserved = cache->free_entry_head;
    cache->free_entry_head = entry_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PrefixCacheInstallEntry(
    SparkGlm52PrefixCache *cache,
    SparkGlm52PrefixCacheEntry *entry,
    uint32_t entry_flags,
    uint64_t parent_hash,
    uint64_t block_hash,
    uint64_t content_hash,
    uint32_t first_token_index,
    uint32_t token_count,
    uint64_t operation_epoch)
{
    uint32_t entry_index;
    uint32_t next_free_entry_index;
    uint32_t physical_block_index;
    uint32_t entry_was_valid;
    SparkStatus status;

    entry_index = SparkGlm52PrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_GLM52_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    entry_was_valid =
        (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u;
    next_free_entry_index = SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    if (entry_was_valid != 0u)
    {
        if (entry->reference_count != 0u)
        {
            return SPARK_STATUS_BUSY;
        }
        physical_block_index = entry->physical_block_index;
        status = SparkGlm52PrefixCacheRecyclePhysicalBlock(
            cache,
            physical_block_index);
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
        status = SparkGlm52PrefixCacheAcquirePhysicalBlock(
            cache,
            &physical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        cache->free_entry_head = next_free_entry_index;
    }

    entry->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    entry->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_ENTRY_DESCRIPTOR_BYTES;
    entry->flags = SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID | entry_flags;
    entry->token_count = token_count;
    entry->first_token_index = first_token_index;
    entry->physical_block_index = physical_block_index;
    entry->reference_count = 0u;
    entry->reserved = 0u;
    entry->parent_hash = parent_hash;
    entry->block_hash = block_hash;
    entry->content_hash = content_hash;
    entry->reservation_epoch = operation_epoch;
    entry->committed_epoch = 0u;
    entry->lookahead_priority = 0u;
    entry->lookahead_request_count = 0u;
    entry->lookahead_protection_epoch = 0u;
    cache->tick += 1u;
    entry->last_used_tick = cache->tick;
    cache->inserted_block_count += 1u;
    if ((entry_flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u)
    {
        cache->live_only_block_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52PrefixCacheInitializeLookup(
    SparkGlm52PrefixCacheLookup *lookup,
    uint64_t sequence_id,
    uint32_t requested_token_count)
{
    memset(lookup, 0, sizeof(*lookup));
    lookup->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    lookup->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_LOOKUP_DESCRIPTOR_BYTES;
    lookup->requested_token_count = requested_token_count;
    lookup->physical_block_index = SPARK_GLM52_PREFIX_CACHE_NO_PHYSICAL_BLOCK;
    lookup->sequence_id = sequence_id;
    lookup->last_block_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
}

static void SparkGlm52PrefixCachePrepareReservationOutput(
    SparkGlm52PrefixCacheReservation *reservation,
    uint64_t sequence_id,
    uint32_t requested_token_count,
    uint64_t operation_epoch)
{
    uint32_t *physical_block_indices;
    uint32_t physical_block_capacity;

    physical_block_indices = reservation->physical_block_indices;
    physical_block_capacity = reservation->physical_block_capacity;
    memset(reservation, 0, sizeof(*reservation));
    reservation->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    reservation->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation->requested_token_count = requested_token_count;
    reservation->sequence_id = sequence_id;
    reservation->reservation_epoch = operation_epoch;
    reservation->last_block_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    reservation->physical_block_indices = physical_block_indices;
    reservation->physical_block_capacity = physical_block_capacity;
}

static SparkStatus SparkGlm52PrefixCacheReleaseBinding(
    SparkGlm52PrefixCache *cache,
    SparkGlm52PrefixCacheSequenceBinding *binding)
{
    uint32_t binding_index;
    SparkStatus status;

    if (binding == 0 ||
        (binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) == 0u)
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
    status = SparkGlm52PrefixCacheReleasePhysicalBlockReference(
        cache,
        binding->physical_block_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    cache->entries[binding->entry_index].reference_count -= 1u;
    cache->released_block_count += 1u;
    SparkGlm52PrefixCacheInitializeBinding(binding);
    binding->reserved = cache->free_binding_head;
    cache->free_binding_head = binding_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PrefixCacheAcquireEntryForSequence(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    SparkGlm52PrefixCacheEntry *entry,
    uint64_t operation_epoch,
    uint32_t binding_is_pending)
{
    SparkGlm52PrefixCacheSequenceBinding *binding;
    uint32_t next_free_binding_index;
    uint32_t entry_index;
    SparkStatus status;

    entry_index = SparkGlm52PrefixCacheEntryIndex(cache, entry);
    if (entry_index == SPARK_GLM52_PREFIX_CACHE_NO_ENTRY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    binding = SparkGlm52PrefixCacheFindBinding(
        cache,
        sequence_id,
        entry_index);
    if (binding != 0)
    {
        if (binding_is_pending != 0u)
        {
            binding->flags |= SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_PENDING;
        }
        return SPARK_STATUS_OK;
    }
    binding = SparkGlm52PrefixCacheFindFreeBinding(cache);
    if (binding == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkGlm52PrefixCacheRetainPhysicalBlock(
        cache,
        entry->physical_block_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    next_free_binding_index = binding->reserved;
    if (cache->free_binding_head !=
        (uint32_t)(binding - cache->sequence_bindings))
    {
        (void)SparkGlm52PrefixCacheReleasePhysicalBlockReference(
            cache,
            entry->physical_block_index);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cache->free_binding_head = next_free_binding_index;

    binding->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    binding->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_BINDING_DESCRIPTOR_BYTES;
    binding->flags = SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID;
    if (binding_is_pending != 0u)
    {
        binding->flags |= SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_PENDING;
    }
    binding->entry_index = entry_index;
    binding->first_token_index = entry->first_token_index;
    binding->token_count = entry->token_count;
    binding->physical_block_index = entry->physical_block_index;
    binding->sequence_id = sequence_id;
    binding->parent_hash = entry->parent_hash;
    binding->block_hash = entry->block_hash;
    binding->acquire_epoch = operation_epoch;
    entry->reference_count += 1u;
    cache->acquired_block_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PrefixCacheRollbackEpoch(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint64_t operation_epoch)
{
    uint32_t binding_index;
    uint32_t entry_index;
    SparkStatus status;

    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->acquire_epoch == operation_epoch)
        {
            status = SparkGlm52PrefixCacheReleaseBinding(cache, binding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u &&
            entry->reservation_epoch == operation_epoch &&
            entry->reference_count == 0u)
        {
            status = SparkGlm52PrefixCacheInvalidateEntry(cache, entry);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            cache->cancelled_reserved_block_count += 1u;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheInitialize(
    SparkGlm52PrefixCache *cache,
    const SparkGlm52PrefixCacheConfiguration *configuration)
{
    uint32_t entry_index;
    uint32_t binding_index;

    if (cache == 0 || configuration == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->abi_version != SPARK_GLM52_PREFIX_CACHE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->block_token_count == 0u ||
        configuration->block_token_count > SPARK_GLM52_PREFIX_CACHE_MAX_BLOCK_TOKENS ||
        configuration->entry_count == 0u ||
        configuration->physical_block_count == 0u ||
        configuration->sequence_binding_count == 0u ||
        configuration->entries == 0 ||
        configuration->sequence_bindings == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->kv_cache_arena != 0 &&
        (configuration->kv_cache_arena->physical_block_count <
            configuration->physical_block_count ||
         configuration->kv_cache_arena->block_token_count !=
            configuration->block_token_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(cache, 0, sizeof(*cache));
    cache->abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    cache->descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_DESCRIPTOR_BYTES;
    cache->block_token_count = configuration->block_token_count;
    cache->entry_count = configuration->entry_count;
    cache->physical_block_count = configuration->physical_block_count;
    cache->sequence_binding_count = configuration->sequence_binding_count;
    cache->entries = configuration->entries;
    cache->sequence_bindings = configuration->sequence_bindings;
    cache->kv_cache_arena = configuration->kv_cache_arena;
    cache->free_entry_head = 0u;
    cache->free_binding_head = 0u;
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheInitializeEntry(&cache->entries[entry_index]);
        cache->entries[entry_index].reserved =
            entry_index + 1u < cache->entry_count
                ? entry_index + 1u
                : SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheInitializeBinding(
            &cache->sequence_bindings[binding_index]);
        cache->sequence_bindings[binding_index].reserved =
            binding_index + 1u < cache->sequence_binding_count
                ? binding_index + 1u
                : SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheProbePrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup)
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint32_t reusable_token_count;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || lookup == 0 || token_count == 0u || sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkGlm52PrefixCacheInitializeLookup(lookup, sequence_id, token_count);
    cache->lookup_count += 1u;
    reusable_token_count = SparkGlm52PrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    for (token_offset = 0u;
         token_offset < reusable_token_count;
         token_offset += cache->block_token_count)
    {
        SparkGlm52PrefixCacheEntry *entry;

        block_hash = SparkGlm52PrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkGlm52PrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        entry = 0;
        {
            SparkGlm52PrefixCacheSequenceBinding *own_binding;

            own_binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
                cache,
                sequence_id,
                token_offset);
            if (own_binding != 0 &&
                own_binding->entry_index < cache->entry_count)
            {
                SparkGlm52PrefixCacheEntry *own_entry;

                own_entry = &cache->entries[own_binding->entry_index];
                if ((own_entry->flags &
                        SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
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
            entry = SparkGlm52PrefixCacheFindEntry(
                cache,
                parent_hash,
                block_hash,
                content_hash,
                token_offset,
                cache->block_token_count,
                1u);
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
        lookup->physical_block_index = entry->physical_block_index;
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


typedef struct SparkGlm52PrefixCacheWalk
{
    uint64_t parent_hash;
    uint64_t block_hash;
    uint32_t token_offset;
    uint32_t reusable_token_count;
    uint32_t matched_block_count;
} SparkGlm52PrefixCacheWalk;

// Step the cached prefix chain one block at a time, returning the matched entry
// or null at the end of the chain, and touching recency as it goes. The three
// probes below differ only in what they do per block, so the chain hashing and
// lookup live here once rather than in each of them.
static SparkGlm52PrefixCacheEntry *SparkGlm52PrefixCacheWalkNext(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheWalk *walk)
{
    SparkGlm52PrefixCacheEntry *entry;
    uint64_t content_hash;

    if (walk->matched_block_count == 0u && walk->token_offset == 0u)
    {
        walk->parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
        walk->reusable_token_count =
            SparkGlm52PrefixCacheMaximumReusableTokenCount(cache, token_count);
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
    walk->block_hash = SparkGlm52PrefixCacheHashBlock(
        &token_ids[walk->token_offset],
        cache->block_token_count,
        walk->parent_hash);
    content_hash = SparkGlm52PrefixCacheHashBlockContent(
        &token_ids[walk->token_offset],
        cache->block_token_count);
    entry = SparkGlm52PrefixCacheFindEntry(
        cache,
        walk->parent_hash,
        walk->block_hash,
        content_hash,
        walk->token_offset,
        cache->block_token_count,
        1u);
    if (entry == 0)
    {
        return 0;
    }
    entry->last_used_tick = cache->tick + 1u;
    walk->matched_block_count += 1u;
    return entry;
}

SparkStatus SparkGlm52PrefixCacheProbePhysicalBlockTable(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *physical_block_count_out)
{
    uint32_t physical_block_count;
    SparkGlm52PrefixCacheWalk walk;
    SparkGlm52PrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u ||
        physical_block_indices == 0 || matched_token_count_out == 0 ||
        physical_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *matched_token_count_out = 0u;
    *physical_block_count_out = 0u;
    physical_block_count = 0u;
    memset(&walk, 0, sizeof(walk));
    while ((entry = SparkGlm52PrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (physical_block_count >= physical_block_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        physical_block_indices[physical_block_count] =
            entry->physical_block_index;
        physical_block_count += 1u;
    }

    *matched_token_count_out = physical_block_count * cache->block_token_count;
    *physical_block_count_out = physical_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheProbeReusablePrefixPrefetchSources(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *matched_token_count_out,
    uint32_t *source_block_count_out)
{
    uint32_t source_block_count;
    SparkGlm52PrefixCacheWalk walk;
    SparkGlm52PrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
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
    while ((entry = SparkGlm52PrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (source_block_count >= source_block_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkGlm52PrefixCacheFillPrefetchSourceBlock(
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

SparkStatus SparkGlm52PrefixCacheProbeReusablePrefixResidency(
    SparkGlm52PrefixCache *cache,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out)
{
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkGlm52PrefixCacheWalk walk;
    SparkGlm52PrefixCacheEntry *entry;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
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
    while ((entry = SparkGlm52PrefixCacheWalkNext(
        cache, token_ids, token_count, &walk)) != 0)
    {
        if (cache->kv_cache_arena != 0 &&
            entry->physical_block_index <
                cache->kv_cache_arena->physical_block_count &&
            (cache->kv_cache_arena->blocks[entry->physical_block_index].flags &
                SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
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

SparkStatus SparkGlm52PrefixCacheResetLookaheadProtection(
    SparkGlm52PrefixCache *cache)
{
    uint32_t entry_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
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

SparkStatus SparkGlm52PrefixCacheProtectPromptLookahead(
    SparkGlm52PrefixCache *cache,
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

    status = SparkGlm52PrefixCacheValidate(cache);
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
        status = SparkGlm52PrefixCacheResetLookaheadProtection(cache);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    reusable_token_count = SparkGlm52PrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    protected_block_count = 0u;
    for (token_offset = 0u;
         token_offset < reusable_token_count;
         token_offset += cache->block_token_count)
    {
        SparkGlm52PrefixCacheEntry *entry;

        block_hash = SparkGlm52PrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkGlm52PrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        entry = SparkGlm52PrefixCacheFindEntry(
            cache,
            parent_hash,
            block_hash,
            content_hash,
            token_offset,
            cache->block_token_count,
            1u);
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


#define SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_LOOKAHEAD_BASE 1000000000000000ull
#define SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_PRIORITY_WEIGHT 1000000000ull
#define SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_REQUEST_WEIGHT 10000000ull
#define SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT 1000000000000ull
#define SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_TOKEN_DEPTH_WEIGHT 1024ull

static uint32_t SparkGlm52PrefixCacheHardProtectedBlockListContainsBlock(
    const uint32_t *hard_protected_physical_block_indices,
    uint32_t hard_protected_physical_block_count,
    uint32_t physical_block_index)
{
    uint32_t protected_block_index;

    if (hard_protected_physical_block_count != 0u &&
        hard_protected_physical_block_indices == 0)
    {
        return 0u;
    }
    for (protected_block_index = 0u;
         protected_block_index < hard_protected_physical_block_count;
         ++protected_block_index)
    {
        if (hard_protected_physical_block_indices[protected_block_index] ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static const SparkGlm52PrefixCacheEntry *SparkGlm52PrefixCacheFindResidentEntryForPhysicalBlock(
    const SparkGlm52PrefixCache *cache,
    uint32_t physical_block_index)
{
    const SparkGlm52PrefixCacheEntry *best_entry;
    uint32_t entry_index;

    best_entry = 0;
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        const SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
            entry->physical_block_index != physical_block_index)
        {
            continue;
        }
        if (best_entry == 0 ||
            SparkGlm52PrefixCacheProtectedVictimIsBetter(
                best_entry,
                entry))
        {
            best_entry = entry;
        }
    }
    return best_entry;
}

static uint64_t SparkGlm52PrefixCacheClampU64Product(
    uint64_t left,
    uint64_t right)
{
    if (left != 0u && right > UINT64_MAX / left)
    {
        return UINT64_MAX;
    }
    return left * right;
}

static uint64_t SparkGlm52PrefixCacheAddScoreU64(
    uint64_t score,
    uint64_t addend)
{
    if (UINT64_MAX - score < addend)
    {
        return UINT64_MAX;
    }
    return score + addend;
}

static uint64_t SparkGlm52PrefixCacheComputeResidentBlockKeepScore(
    const SparkGlm52PrefixCache *cache,
    const SparkGlm52KvCacheBlock *block,
    const SparkGlm52PrefixCacheEntry *entry)
{
    uint64_t score;
    uint32_t prefix_token_depth;

    score = block->last_used_epoch;
    score = SparkGlm52PrefixCacheAddScoreU64(
        score,
        SparkGlm52PrefixCacheClampU64Product(
            block->reference_count,
            SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT));
    if (entry == 0)
    {
        return score;
    }

    score = SparkGlm52PrefixCacheAddScoreU64(score, entry->last_used_tick);
    score = SparkGlm52PrefixCacheAddScoreU64(
        score,
        SparkGlm52PrefixCacheClampU64Product(
            entry->reference_count,
            SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_REFERENCE_WEIGHT));

    prefix_token_depth = entry->first_token_index + entry->token_count;
    score = SparkGlm52PrefixCacheAddScoreU64(
        score,
        SparkGlm52PrefixCacheClampU64Product(
            prefix_token_depth,
            SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_TOKEN_DEPTH_WEIGHT));

    if (SparkGlm52PrefixCacheEntryHasCurrentLookaheadProtection(cache, entry))
    {
        score = SparkGlm52PrefixCacheAddScoreU64(
            score,
            SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_LOOKAHEAD_BASE);
        score = SparkGlm52PrefixCacheAddScoreU64(
            score,
            SparkGlm52PrefixCacheClampU64Product(
                entry->lookahead_priority,
                SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_PRIORITY_WEIGHT));
        score = SparkGlm52PrefixCacheAddScoreU64(
            score,
            SparkGlm52PrefixCacheClampU64Product(
                entry->lookahead_request_count,
                SPARK_GLM52_PREFIX_CACHE_REUSE_SCORE_REQUEST_WEIGHT));
    }
    return score;
}

typedef struct SparkGlm52PrefixCacheResidentEvictionCandidate
{
    uint32_t physical_block_index;
    uint32_t has_prefix_entry;
    uint32_t has_lookahead_protection;
    uint64_t keep_score;
    uint64_t last_used_epoch;
} SparkGlm52PrefixCacheResidentEvictionCandidate;

static void SparkGlm52PrefixCacheInitializeResidentEvictionCandidate(
    SparkGlm52PrefixCacheResidentEvictionCandidate *candidate)
{
    memset(candidate, 0, sizeof(*candidate));
    candidate->physical_block_index = SPARK_GLM52_KV_CACHE_NO_BLOCK;
    candidate->keep_score = UINT64_MAX;
    candidate->last_used_epoch = UINT64_MAX;
}

static uint32_t SparkGlm52PrefixCacheResidentEvictionCandidateIsBetter(
    const SparkGlm52PrefixCacheResidentEvictionCandidate *candidate,
    const SparkGlm52PrefixCacheResidentEvictionCandidate *current)
{
    if (current->physical_block_index == SPARK_GLM52_KV_CACHE_NO_BLOCK)
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
    return candidate->physical_block_index < current->physical_block_index;
}

static SparkStatus SparkGlm52PrefixCacheSelectResidentReuseScoreVictim(
    SparkGlm52PrefixCache *cache,
    const uint32_t *hard_protected_physical_block_indices,
    uint32_t hard_protected_physical_block_count,
    SparkGlm52PrefixCacheResidentEvictionCandidate *victim_out)
{
    uint32_t physical_block_index;

    SparkGlm52PrefixCacheInitializeResidentEvictionCandidate(victim_out);
    for (physical_block_index = 0u;
         physical_block_index < cache->kv_cache_arena->physical_block_count;
         ++physical_block_index)
    {
        SparkGlm52PrefixCacheResidentEvictionCandidate candidate;
        const SparkGlm52PrefixCacheEntry *entry;
        const SparkGlm52KvCacheBlock *block;

        block = &cache->kv_cache_arena->blocks[physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            SparkGlm52PrefixCacheHardProtectedBlockListContainsBlock(
                hard_protected_physical_block_indices,
                hard_protected_physical_block_count,
                physical_block_index))
        {
            continue;
        }

        entry = SparkGlm52PrefixCacheFindResidentEntryForPhysicalBlock(
            cache,
            physical_block_index);
        candidate.physical_block_index = physical_block_index;
        candidate.has_prefix_entry = entry != 0;
        candidate.has_lookahead_protection =
            SparkGlm52PrefixCacheEntryHasCurrentLookaheadProtection(
                cache,
                entry);
        candidate.keep_score = SparkGlm52PrefixCacheComputeResidentBlockKeepScore(
            cache,
            block,
            entry);
        candidate.last_used_epoch = block->last_used_epoch;
        if (SparkGlm52PrefixCacheResidentEvictionCandidateIsBetter(
                &candidate,
                victim_out))
        {
            *victim_out = candidate;
        }
    }
    return victim_out->physical_block_index == SPARK_GLM52_KV_CACHE_NO_BLOCK
        ? SPARK_STATUS_NOT_FOUND
        : SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheTrimResidentBlocksByReuseScore(
    SparkGlm52PrefixCache *cache,
    uint32_t max_resident_block_count,
    const uint32_t *hard_protected_physical_block_indices,
    uint32_t hard_protected_physical_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (cache->kv_cache_arena == 0 ||
        (hard_protected_physical_block_count != 0u &&
         hard_protected_physical_block_indices == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (max_resident_block_count > cache->kv_cache_arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (cache->kv_cache_arena->resident_block_count > max_resident_block_count)
    {
        SparkGlm52PrefixCacheResidentEvictionCandidate victim;

        status = SparkGlm52PrefixCacheSelectResidentReuseScoreVictim(
            cache,
            hard_protected_physical_block_indices,
            hard_protected_physical_block_count,
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

        status = SparkGlm52KvCacheArenaMarkBlockNonResident(
            cache->kv_cache_arena,
            victim.physical_block_index);
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

SparkStatus SparkGlm52PrefixCacheLookupPrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup)
{
    SparkGlm52PrefixCacheLookup probe;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t operation_epoch;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkGlm52PrefixCacheProbePrompt(
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
    parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    for (token_offset = 0u;
         token_offset < probe.matched_token_count;
         token_offset += cache->block_token_count)
    {
        SparkGlm52PrefixCacheEntry *entry;

        block_hash = SparkGlm52PrefixCacheHashBlock(
            &token_ids[token_offset],
            cache->block_token_count,
            parent_hash);
        content_hash = SparkGlm52PrefixCacheHashBlockContent(
            &token_ids[token_offset],
            cache->block_token_count);
        entry = SparkGlm52PrefixCacheFindEntry(
            cache,
            parent_hash,
            block_hash,
            content_hash,
            token_offset,
            cache->block_token_count,
            1u);
        if (entry == 0)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        status = SparkGlm52PrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            0u);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
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

static SparkStatus SparkGlm52PrefixCacheReservePromptInternal(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheReservation *reservation,
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

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (token_ids == 0 || token_count == 0u || sequence_id == 0u ||
        reservation == 0 ||
        reservation->abi_version != SPARK_GLM52_PREFIX_CACHE_ABI_VERSION ||
        reservation->descriptor_bytes !=
            SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkGlm52PrefixCacheCeilDivideU32(
        token_count,
        cache->block_token_count);
    if (reservation->physical_block_indices != 0 &&
        reservation->physical_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;
    SparkGlm52PrefixCachePrepareReservationOutput(
        reservation,
        sequence_id,
        token_count,
        operation_epoch);
    reusable_token_count = SparkGlm52PrefixCacheMaximumReusableTokenCount(
        cache,
        token_count);
    parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    token_offset = 0u;

    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *existing_binding;
        SparkGlm52PrefixCacheEntry *entry;
        uint32_t block_token_count;
        uint32_t is_full_block;
        uint32_t entry_flags;
        uint32_t binding_is_pending;

        block_token_count = SparkGlm52PrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - token_offset);
        is_full_block = block_token_count == cache->block_token_count;
        block_hash = SparkGlm52PrefixCacheHashBlock(
            &token_ids[token_offset],
            block_token_count,
            parent_hash);
        content_hash = SparkGlm52PrefixCacheHashBlockContent(
            &token_ids[token_offset],
            block_token_count);
        existing_binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
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
            entry = SparkGlm52PrefixCacheFindEntry(
                cache,
                parent_hash,
                block_hash,
                content_hash,
                token_offset,
                block_token_count,
                1u);
        }
        if (entry == 0)
        {
            entry = SparkGlm52PrefixCacheSelectVictim(cache);
            if (entry == 0)
            {
                SparkGlm52PrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            entry_flags = SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING;
            if (is_full_block == 0u || allow_cross_sequence_reuse == 0u)
            {
                entry_flags |= SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
            }
            status = SparkGlm52PrefixCacheInstallEntry(
                cache,
                entry,
                entry_flags,
                parent_hash,
                block_hash,
                content_hash,
                token_offset,
                block_token_count,
                operation_epoch);
            if (status != SPARK_STATUS_OK)
            {
                SparkGlm52PrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return status;
            }
            reservation->pending_physical_block_count += 1u;
            cache->reserved_block_count += 1u;
        }
        else if (SparkGlm52PrefixCacheEntryIsReusable(cache, entry))
        {
            reservation->cached_physical_block_count += 1u;
            reservation->reusable_token_count += block_token_count;
        }
        else if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u)
        {
            reservation->pending_physical_block_count += 1u;
        }

        binding_is_pending =
            (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u;
        status = SparkGlm52PrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            binding_is_pending);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        if (reservation->physical_block_indices != 0)
        {
            reservation->physical_block_indices[block_index] =
                entry->physical_block_index;
        }
        cache->tick += 1u;
        entry->last_used_tick = cache->tick;
        parent_hash = block_hash;
        token_offset += block_token_count;
    }

    reservation->reserved_token_count = token_count;
    reservation->physical_block_count = block_count;
    reservation->last_block_hash = parent_hash;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheReservePrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheReservation *reservation)
{
    return SparkGlm52PrefixCacheReservePromptInternal(
        cache,sequence_id,token_ids,token_count,reservation,1u);
}

SparkStatus SparkGlm52PrefixCacheReserveSequencePrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheReservation *reservation)
{
    return SparkGlm52PrefixCacheReservePromptInternal(
        cache,sequence_id,token_ids,token_count,reservation,0u);
}

SparkStatus SparkGlm52PrefixCacheCommitReservation(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch)
{
    uint32_t entry_index;
    uint32_t binding_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || reservation_epoch == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u &&
            entry->reservation_epoch == reservation_epoch)
        {
            status = SparkGlm52PrefixCacheMarkPhysicalBlockResident(
                cache,
                entry->physical_block_index);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            entry->flags &= ~SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING;
            if (entry->token_count == cache->block_token_count &&
                (entry->flags &
                    SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) == 0u)
            {
                entry->flags |= SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_REUSABLE;
            }
            else
            {
                entry->flags |= SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
            }
            entry->committed_epoch = reservation_epoch;
            cache->committed_reserved_block_count += 1u;
        }
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id &&
            binding->acquire_epoch == reservation_epoch)
        {
            binding->flags &= ~SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_PENDING;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheCancelReservation(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch)
{
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || reservation_epoch == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52PrefixCacheRollbackEpoch(
        cache,
        sequence_id,
        reservation_epoch);
}

SparkStatus SparkGlm52PrefixCacheCommitPrompt(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    SparkGlm52PrefixCacheLookup *lookup)
{
    SparkGlm52PrefixCacheReservation reservation;
    uint32_t committed_token_count;
    SparkStatus status;

    if (lookup != 0)
    {
        SparkGlm52PrefixCacheInitializeLookup(lookup, sequence_id, token_count);
    }
    memset(&reservation, 0, sizeof(reservation));
    reservation.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    status = SparkGlm52PrefixCacheReservePrompt(
        cache,
        sequence_id,
        token_ids,
        token_count,
        &reservation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52PrefixCacheCommitReservation(
        cache,
        sequence_id,
        reservation.reservation_epoch);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52PrefixCacheCancelReservation(
            cache,
            sequence_id,
            reservation.reservation_epoch);
        return status;
    }
    if (lookup != 0)
    {
        committed_token_count = SparkGlm52PrefixCacheFullBlockTokenCount(
            cache,
            token_count);
        lookup->matched_token_count = committed_token_count;
        lookup->matched_block_count = committed_token_count / cache->block_token_count;
        lookup->next_token_index = committed_token_count;
        lookup->last_block_hash = reservation.last_block_hash;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheEnsureSequenceTokenCapacity(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count)
{
    SparkGlm52PrefixCacheSequenceBinding *short_binding;
    SparkGlm52PrefixCacheEntry *short_entry;
    uint64_t operation_epoch;
    uint64_t parent_hash;
    uint32_t block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkGlm52PrefixCacheCeilDivideU32(
        token_count,
        cache->block_token_count);
    short_binding = 0;
    short_entry = 0;
    parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    cache->operation_epoch += 1u;
    operation_epoch = cache->operation_epoch;

    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;
        SparkGlm52PrefixCacheEntry *entry;
        uint64_t block_hash;
        uint64_t content_hash;
        uint32_t first_token_index;
        uint32_t required_block_token_count;

        first_token_index = block_index * cache->block_token_count;
        required_block_token_count = SparkGlm52PrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - first_token_index);
        binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            first_token_index);
        if (binding != 0)
        {
            if (binding->entry_index >= cache->entry_count ||
                binding->physical_block_index >= cache->physical_block_count ||
                binding->token_count == 0u ||
                binding->token_count > cache->block_token_count)
            {
                SparkGlm52PrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            entry = &cache->entries[binding->entry_index];
            if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
                (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u ||
                entry->first_token_index != first_token_index ||
                entry->physical_block_index != binding->physical_block_index ||
                entry->token_count != binding->token_count)
            {
                SparkGlm52PrefixCacheRollbackEpoch(
                    cache,
                    sequence_id,
                    operation_epoch);
                return SPARK_STATUS_BUSY;
            }
            if (binding->token_count < required_block_token_count)
            {
                /*
                 * Only the final, private, live-only prompt block may grow in
                 * place.  Growing a shared partial block would corrupt a fork
                 * and requires an explicit KV copy-on-write backend.
                 */
                if ((entry->flags &
                        SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) == 0u ||
                    entry->reference_count != 1u || short_binding != 0)
                {
                    SparkGlm52PrefixCacheRollbackEpoch(
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

        entry = SparkGlm52PrefixCacheSelectVictim(cache);
        if (entry == 0)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        block_hash = SparkGlm52PrefixCacheMixU64(
            SparkGlm52PrefixCacheMixU64(parent_hash, sequence_id),
            first_token_index);
        content_hash = SparkGlm52PrefixCacheMixU64(
            block_hash,
            operation_epoch);
        status = SparkGlm52PrefixCacheInstallEntry(
            cache,
            entry,
            SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING |
                SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY,
            parent_hash,
            block_hash,
            content_hash,
            first_token_index,
            cache->block_token_count,
            operation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        cache->reserved_block_count += 1u;
        status = SparkGlm52PrefixCacheAcquireEntryForSequence(
            cache,
            sequence_id,
            entry,
            operation_epoch,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                sequence_id,
                operation_epoch);
            return status;
        }
        parent_hash = block_hash;
    }

    status = SparkGlm52PrefixCacheCommitReservation(
        cache,
        sequence_id,
        operation_epoch);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52PrefixCacheRollbackEpoch(
            cache,
            sequence_id,
            operation_epoch);
        return status;
    }
    if (short_binding != 0)
    {
        short_binding->token_count = cache->block_token_count;
        short_entry->token_count = cache->block_token_count;
        short_entry->flags &= ~SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_REUSABLE;
        short_entry->flags |= SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheBuildPhysicalBlockTable(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        physical_block_indices == 0 || physical_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_count = SparkGlm52PrefixCacheCeilDivideU32(
        token_count,
        cache->block_token_count);
    if (physical_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;
        uint32_t required_block_token_count;
        uint32_t token_offset;

        token_offset = block_index * cache->block_token_count;
        required_block_token_count = SparkGlm52PrefixCacheMinimumU32(
            cache->block_token_count,
            token_count - token_offset);

        binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0 || binding->token_count < required_block_token_count)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        physical_block_indices[block_index] = binding->physical_block_index;
    }
    *physical_block_count_out = block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheBuildSequencePrefetchSources(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_capacity,
    uint32_t *source_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        source_blocks == 0 || source_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_count = SparkGlm52PrefixCacheCeilDivideU32(
        token_count,
        cache->block_token_count);
    if (source_block_capacity < block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    token_offset = 0u;
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;
        SparkGlm52PrefixCacheEntry *entry;

        binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0 || binding->entry_index >= cache->entry_count)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        entry = &cache->entries[binding->entry_index];
        status = SparkGlm52PrefixCacheFillPrefetchSourceBlock(
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

SparkStatus SparkGlm52PrefixCacheProbeSequenceResidency(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *physical_block_count_out,
    uint32_t *resident_block_count_out,
    uint32_t *nonresident_block_count_out)
{
    uint32_t block_count;
    uint32_t block_index;
    uint32_t token_offset;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u || token_count == 0u ||
        physical_block_count_out == 0 || resident_block_count_out == 0 ||
        nonresident_block_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_count = SparkGlm52PrefixCacheCeilDivideU32(
        token_count,
        cache->block_token_count);
    token_offset = 0u;
    resident_block_count = 0u;
    nonresident_block_count = 0u;
    for (block_index = 0u; block_index < block_count; ++block_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            sequence_id,
            token_offset);
        if (binding == 0)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (cache->kv_cache_arena != 0 &&
            binding->physical_block_index < cache->kv_cache_arena->physical_block_count &&
            (cache->kv_cache_arena->blocks[binding->physical_block_index].flags &
                SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
        {
            nonresident_block_count += 1u;
        }
        else
        {
            resident_block_count += 1u;
        }
        token_offset += binding->token_count;
    }

    *physical_block_count_out = block_count;
    *resident_block_count_out = resident_block_count;
    *nonresident_block_count_out = nonresident_block_count;
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52PrefixCacheBindCommittedPrefixFromSequence(
    SparkGlm52PrefixCache *cache,
    uint64_t source_sequence_id,
    uint64_t target_sequence_id,
    uint32_t token_count)
{
    uint64_t operation_epoch;
    uint32_t token_offset;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
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
        SparkGlm52PrefixCacheSequenceBinding *source_binding;
        SparkGlm52PrefixCacheSequenceBinding *target_binding;
        SparkGlm52PrefixCacheEntry *entry;

        source_binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            source_sequence_id,
            token_offset);
        if (source_binding == 0 ||
            source_binding->entry_index >= cache->entry_count ||
            source_binding->token_count == 0u ||
            token_offset + source_binding->token_count > token_count)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_NOT_FOUND;
        }
        entry = &cache->entries[source_binding->entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) == 0u ||
            (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_PENDING) != 0u)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_BUSY;
        }
        target_binding = SparkGlm52PrefixCacheFindBindingAtTokenOffset(
            cache,
            target_sequence_id,
            token_offset);
        if (target_binding != 0 && target_binding->entry_index !=
            source_binding->entry_index)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return SPARK_STATUS_DUPLICATE;
        }
        status = SparkGlm52PrefixCacheAcquireEntryForSequence(
            cache,
            target_sequence_id,
            entry,
            operation_epoch,
            0u);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52PrefixCacheRollbackEpoch(
                cache,
                target_sequence_id,
                operation_epoch);
            return status;
        }
        token_offset += source_binding->token_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheReleaseSequence(
    SparkGlm52PrefixCache *cache,
    uint64_t sequence_id)
{
    uint32_t binding_index;
    uint32_t entry_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheSequenceBinding *binding;

        binding = &cache->sequence_bindings[binding_index];
        if ((binding->flags & SPARK_GLM52_PREFIX_CACHE_BINDING_FLAG_VALID) != 0u &&
            binding->sequence_id == sequence_id)
        {
            status = SparkGlm52PrefixCacheReleaseBinding(cache, binding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheEntry *entry;

        entry = &cache->entries[entry_index];
        if ((entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_VALID) != 0u &&
            (entry->flags & SPARK_GLM52_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) == 0u &&
            entry->reference_count == 0u)
        {
            status = SparkGlm52PrefixCacheInvalidateEntry(cache, entry);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PrefixCacheReset(
    SparkGlm52PrefixCache *cache)
{
    uint32_t entry_index;
    uint32_t binding_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheValidate(cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (entry_index = 0u; entry_index < cache->entry_count; ++entry_index)
    {
        SparkGlm52PrefixCacheInitializeEntry(&cache->entries[entry_index]);
        cache->entries[entry_index].reserved =
            entry_index + 1u < cache->entry_count
                ? entry_index + 1u
                : SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    }
    for (binding_index = 0u;
         binding_index < cache->sequence_binding_count;
         ++binding_index)
    {
        SparkGlm52PrefixCacheInitializeBinding(
            &cache->sequence_bindings[binding_index]);
        cache->sequence_bindings[binding_index].reserved =
            binding_index + 1u < cache->sequence_binding_count
                ? binding_index + 1u
                : SPARK_GLM52_PREFIX_CACHE_NO_ENTRY;
    }
    if (cache->kv_cache_arena != 0)
    {
        status = SparkGlm52KvCacheArenaReset(cache->kv_cache_arena);
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
