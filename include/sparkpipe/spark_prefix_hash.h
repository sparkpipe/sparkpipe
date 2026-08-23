#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared prefix-hash primitives.
 *
 * There are exactly TWO prefix chain-hash constructions in the portable
 * KV/prefix homes, exposed here as named variants because they are NOT the
 * same algorithm and their outputs are NOT interchangeable:
 *
 *  - SparkPrefixHashPositionTagged: parent^prime seed, a per-token position
 *    term (token + index<<32), final length mix. This is the LOGICAL
 *    block-chain identity of cache/prefix_cache.c (SparkPrefixCacheHashBlock).
 *    It PERSISTS: nvme-tier tier records carry content_hash from this family
 *    (include/sparkpipe/spark_nvme_tier.h), so changing its output is a
 *    stored-identity migration across NVMe-tier data, not a refactor.
 *
 *  - SparkPrefixHashPlainChained: FNV offset basis, byte-fold of the parent
 *    hash then the raw token bytes, NO position term. This is the PHYSICAL
 *    block identity of runtime/prefix_cache.c (SparkPrefixCacheCore); its
 *    hashes are process-lifetime only (in-memory pool).
 *
 * The same input prefix hashes DIFFERENTLY under each variant by design.
 * Both bodies are byte-for-byte output-preserving ports of the local helpers
 * they replaced; do not merge them into one construction without pricing the
 * storage migration first (see the codesize memo
 * section 2). SparkPrefixHashMixU64 is the shared FNV mix step both cache-side
 * constructions (block chain and persisted content hash) are built on; it is
 * not itself a complete prefix identity.
 */

#define SPARK_PREFIX_HASH_FNV_PRIME UINT64_C(1099511628211)
#define SPARK_PREFIX_HASH_PLAIN_OFFSET_BASIS UINT64_C(14695981039346656037)

static inline uint64_t SparkPrefixHashMixU64(
    uint64_t hash_value,
    uint64_t value)
{
    hash_value ^= value;
    hash_value *= SPARK_PREFIX_HASH_FNV_PRIME;
    hash_value ^= hash_value >> 32u;
    return hash_value;
}

/* Variant 1: position-tagged chained block hash (cache-side, persisted). */
static inline uint64_t SparkPrefixHashPositionTagged(
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint64_t hash_value;
    uint32_t token_index;

    hash_value = parent_hash ^ SPARK_PREFIX_HASH_FNV_PRIME;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        hash_value = SparkPrefixHashMixU64(
            hash_value,
            (uint64_t)token_ids[token_index] +
                ((uint64_t)token_index << 32u));
    }
    return SparkPrefixHashMixU64(hash_value, token_count);
}

/* Variant 2: plain chained byte-fold (runtime-core side, process-lifetime). */
static inline uint64_t SparkPrefixHashPlainChained(
    uint64_t parent_hash,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint64_t hash_value;
    uint32_t token_index;
    uint32_t byte;

    hash_value = SPARK_PREFIX_HASH_PLAIN_OFFSET_BASIS;
    for (byte = 0u; byte < 8u; byte++)
    {
        hash_value ^= (parent_hash >> (uint64_t)(byte * 8u)) & 0xffull;
        hash_value *= SPARK_PREFIX_HASH_FNV_PRIME;
    }
    for (token_index = 0u; token_index < token_count; token_index++)
    {
        for (byte = 0u; byte < 4u; byte++)
        {
            hash_value ^=
                ((uint64_t)token_ids[token_index] >> (byte * 8u)) & 0xffull;
            hash_value *= SPARK_PREFIX_HASH_FNV_PRIME;
        }
    }
    return hash_value;
}

#ifdef __cplusplus
}
#endif
