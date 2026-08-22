#include "prefix_cache.h"

#include <stdlib.h>
#include <string.h>

#define CORE_HASH_OFFSET 14695981039346656037ull
#define CORE_HASH_PRIME 1099511628211ull

static uint64_t CoreChainHash(uint64_t parent_hash, const uint32_t *token_ids, uint32_t token_count)
{
	uint64_t hash;
	uint32_t index;
	uint32_t byte;
	hash = CORE_HASH_OFFSET;
	for (byte = 0u; byte < 8u; byte++)
	{
		hash ^= (parent_hash >> (uint64_t)(byte * 8u)) & 0xffull;
		hash *= CORE_HASH_PRIME;
	}
	for (index = 0u; index < token_count; index++)
	{
		for (byte = 0u; byte < 4u; byte++)
		{
			hash ^= ((uint64_t)token_ids[index] >> (byte * 8u)) & 0xffull;
			hash *= CORE_HASH_PRIME;
		}
	}
	return hash;
}

SparkStatus SparkPrefixCacheCoreValidateGeometry(
	const SparkPrefixCacheCoreConfiguration *configuration)
{
	if (configuration == 0 ||
	    configuration->abi_version != SPARK_PREFIX_CACHE_CORE_ABI_VERSION ||
	    configuration->descriptor_bytes !=
	        SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if (configuration->block_token_count == 0u ||
	    configuration->block_token_count > SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if (configuration->block_stride_bytes == 0u ||
	    configuration->block_count == 0u ||
	    configuration->max_sequence_count == 0u ||
	    configuration->sequence_block_capacity == 0u)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if (configuration->hash_bucket_count == 0u ||
	    (configuration->hash_bucket_count &
	     (configuration->hash_bucket_count - 1u)) != 0u)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
/* Geometry invariants end here: hot paths trust them. */
	return SPARK_STATUS_OK;
}

static void CoreLruUnlink(SparkPrefixCacheCore *core, uint32_t block_index)
{
	SparkPrefixCacheCoreBlock *block;
	block = &core->blocks[block_index];
	if (block->lru_prev != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		core->blocks[block->lru_prev].lru_next = block->lru_next;
	}
	else
	{
		core->lru_head = block->lru_next;
	}
	if (block->lru_next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		core->blocks[block->lru_next].lru_prev = block->lru_prev;
	}
	else
	{
		core->lru_tail = block->lru_prev;
	}
	block->lru_prev = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	block->lru_next = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
}

static void CoreLruPushMru(SparkPrefixCacheCore *core, uint32_t block_index)
{
	SparkPrefixCacheCoreBlock *block;
	block = &core->blocks[block_index];
	block->lru_prev = core->lru_tail;
	block->lru_next = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	if (core->lru_tail != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		core->blocks[core->lru_tail].lru_next = block_index;
	}
	else
	{
		core->lru_head = block_index;
	}
	core->lru_tail = block_index;
}

static void CoreLruTouch(SparkPrefixCacheCore *core, uint32_t block_index)
{
	core->tick++;
	core->blocks[block_index].last_used_tick = core->tick;
	CoreLruUnlink(core, block_index);
	CoreLruPushMru(core, block_index);
}

static void CoreIndexInsert(SparkPrefixCacheCore *core, uint32_t block_index)
{
	SparkPrefixCacheCoreBlock *block;
	uint32_t bucket;
	block = &core->blocks[block_index];
	bucket = (uint32_t)(block->chain_hash & (uint64_t)core->hash_bucket_mask);
	block->hash_next = core->hash_buckets[bucket];
	core->hash_buckets[bucket] = block_index;
}

static void CoreIndexRemove(SparkPrefixCacheCore *core, uint32_t block_index)
{
	SparkPrefixCacheCoreBlock *block;
	uint32_t bucket;
	uint32_t previous;
	uint32_t scan;
	block = &core->blocks[block_index];
	bucket = (uint32_t)(block->chain_hash & (uint64_t)core->hash_bucket_mask);
	previous = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	scan = core->hash_buckets[bucket];
	while (scan != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		if (scan == block_index)
		{
			if (previous == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
			{
				core->hash_buckets[bucket] = block->hash_next;
			}
			else
			{
				core->blocks[previous].hash_next = block->hash_next;
			}
			block->hash_next = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
			return;
		}
		previous = scan;
		scan = core->blocks[scan].hash_next;
	}
}

static uint32_t CoreIndexFind(
	const SparkPrefixCacheCore *core,
	uint64_t chain_hash,
	const uint32_t *token_ids)
{
	SparkPrefixCacheCoreBlock *block;
	uint32_t scan;
	/* Exact verification: the host token mirror rules out collisions. */
	scan = core->hash_buckets[(uint32_t)(chain_hash & (uint64_t)core->hash_bucket_mask)];
	while (scan != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		block = &core->blocks[scan];
		if (block->state == SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED &&
		    block->chain_hash == chain_hash &&
		    memcmp(block->token_ids, token_ids,
		           core->block_token_count * sizeof(uint32_t)) == 0)
		{
			return scan;
		}
		scan = block->hash_next;
	}
	return SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
}

/* Single return path to the physical pool: every free site keeps the
 * same state/linkage invariants by construction. */
static void CoreFreeBlock(SparkPrefixCacheCore *core, uint32_t block_index)
{
	SparkPrefixCacheCoreBlock *block;
	block = &core->blocks[block_index];
	block->state = SPARK_PREFIX_CACHE_CORE_BLOCK_FREE;
	block->free_next = core->free_block_head;
	core->free_block_head = block_index;
}

static uint32_t CoreAllocateOpenBlock(SparkPrefixCacheCore *core)
{
	SparkPrefixCacheCoreBlock *block;
	uint32_t block_index;
	uint32_t evicted;
	if (core->free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		(void)SparkPrefixCacheCoreTrim(core, 1u, &evicted);
	}
	if (core->free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		return SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	}
	block_index = core->free_block_head;
	block = &core->blocks[block_index];
	core->free_block_head = block->free_next;
	block->state = SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE;
	block->reference_count = 1u;
	block->token_count = 0u;
	block->chain_hash = 0u;
	return block_index;
}

static SparkStatus CoreFindSequence(
	const SparkPrefixCacheCore *core,
	uint64_t sequence_id,
	uint32_t *index_out)
{
	uint32_t index;
	for (index = 0u; index < core->max_sequence_count; index++)
	{
		if (core->sequences[index].used != 0u &&
		    core->sequences[index].sequence_id == sequence_id)
		{
			*index_out = index;
			return SPARK_STATUS_OK;
		}
	}
	return SPARK_STATUS_NOT_FOUND;
}

static uint32_t CoreCommittedTokens(const SparkPrefixCacheCore *core, const SparkPrefixCacheCoreSequence *sequence)
{
	SparkPrefixCacheCoreBlock *last;
	uint32_t last_block;
	if (sequence->block_count == 0u)
	{
		return 0u;
	}
	/* Reserved-but-empty private tails (ReservePrivateTail) hold no
	 * tokens: skip them so the committed count is the exact token
	 * mirror, never an overcount a caller could read blocks through. */
	last_block = sequence->block_count - 1u;
	while (last_block != 0u)
	{
		last = &core->blocks[sequence->blocks[last_block]];
		if (last->state != SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE ||
		    last->token_count != 0u)
		{
			break;
		}
		last_block--;
	}
	last = &core->blocks[sequence->blocks[last_block]];
	if (last->state == SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE)
	{
		return last_block * core->block_token_count + last->token_count;
	}
	return sequence->block_count * core->block_token_count;
}

SparkStatus SparkPrefixCacheCoreInitialize(
	SparkPrefixCacheCore *core,
	const SparkPrefixCacheCoreConfiguration *configuration)
{
	uint32_t block_index;
	uint32_t sequence_index;
	uint32_t bucket;
	uint64_t token_slot_count;
	SparkStatus status;
	if (core == 0 || configuration == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	/* Geometry invariants are checked once, here - never per call. */
	status = SparkPrefixCacheCoreValidateGeometry(configuration);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	memset(core, 0, sizeof(*core));
	core->abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	core->descriptor_bytes = SPARK_PREFIX_CACHE_CORE_DESCRIPTOR_BYTES;
	core->block_token_count = configuration->block_token_count;
	core->block_stride_bytes = configuration->block_stride_bytes;
	core->block_count = configuration->block_count;
	core->max_sequence_count = configuration->max_sequence_count;
	core->sequence_block_capacity = configuration->sequence_block_capacity;
	core->hash_bucket_mask = configuration->hash_bucket_count - 1u;
	core->free_block_head = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	core->lru_head = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	core->lru_tail = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	token_slot_count = (uint64_t)configuration->block_count *
	                   (uint64_t)configuration->block_token_count;
	core->blocks = (SparkPrefixCacheCoreBlock *)calloc(
	    configuration->block_count, sizeof(core->blocks[0]));
	core->block_tokens = (uint32_t *)calloc(
	    (size_t)token_slot_count, sizeof(core->block_tokens[0]));
	core->sequences = (SparkPrefixCacheCoreSequence *)calloc(
	    configuration->max_sequence_count, sizeof(core->sequences[0]));
	core->sequence_blocks = (uint32_t *)calloc(
	    (size_t)configuration->max_sequence_count *
	        configuration->sequence_block_capacity,
	    sizeof(core->sequence_blocks[0]));
	core->hash_buckets = (uint32_t *)calloc(
	    configuration->hash_bucket_count, sizeof(core->hash_buckets[0]));
	if (core->blocks == 0 || core->block_tokens == 0 ||
	    core->sequences == 0 || core->sequence_blocks == 0 ||
	    core->hash_buckets == 0)
	{
		SparkPrefixCacheCoreDestroy(core);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	for (block_index = 0u; block_index < core->block_count; block_index++)
	{
		core->blocks[block_index].state = SPARK_PREFIX_CACHE_CORE_BLOCK_FREE;
		core->blocks[block_index].reference_count = 0u;
		core->blocks[block_index].token_count = 0u;
		core->blocks[block_index].hash_next = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
		core->blocks[block_index].free_next = core->free_block_head;
		core->blocks[block_index].lru_prev = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
		core->blocks[block_index].lru_next = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
		core->blocks[block_index].chain_hash = 0u;
		core->blocks[block_index].last_used_tick = 0u;
		core->blocks[block_index].token_ids =
		    core->block_tokens +
		    (uint64_t)block_index * core->block_token_count;
		core->free_block_head = block_index;
	}
	for (sequence_index = 0u; sequence_index < core->max_sequence_count;
	     sequence_index++)
	{
		core->sequences[sequence_index].blocks =
		    core->sequence_blocks +
		    (uint64_t)sequence_index * core->sequence_block_capacity;
	}
	for (bucket = 0u; bucket <= core->hash_bucket_mask; bucket++)
	{
		core->hash_buckets[bucket] = SPARK_PREFIX_CACHE_CORE_NO_BLOCK;
	}
	return SPARK_STATUS_OK;
}

void SparkPrefixCacheCoreDestroy(SparkPrefixCacheCore *core)
{
	if (core == 0)
	{
		return;
	}
	free(core->blocks);
	free(core->block_tokens);
	free(core->sequences);
	free(core->sequence_blocks);
	free(core->hash_buckets);
	memset(core, 0, sizeof(*core));
}

SparkStatus SparkPrefixCacheCoreAdmitSequence(
	SparkPrefixCacheCore *core,
	uint64_t sequence_id,
	const uint32_t *token_ids,
	uint32_t token_count,
	uint32_t *matched_token_count_out)
{
	SparkPrefixCacheCoreSequence *sequence;
	uint32_t sequence_index;
	uint32_t full_block_count;
	uint32_t required_block_count;
	uint32_t matched_block_count;
	uint32_t matched_tokens;
	uint64_t chain_hash;
	uint32_t found;
	uint64_t next_chain_hash;
	SparkStatus status;
	if (core == 0 || token_ids == 0 || token_count == 0u ||
	    matched_token_count_out == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	/* A live id must be released before re-admission: a second slot
	 * with the same id would leak every reference the first holds. */
	if (CoreFindSequence(core, sequence_id, &sequence_index) ==
	    SPARK_STATUS_OK)
	{
		return SPARK_STATUS_DUPLICATE;
	}
	required_block_count =
	    (token_count + core->block_token_count - 1u) / core->block_token_count;
	if (required_block_count > core->sequence_block_capacity)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	sequence = 0;
	for (sequence_index = 0u; sequence_index < core->max_sequence_count;
	     sequence_index++)
	{
		if (core->sequences[sequence_index].used == 0u)
		{
			sequence = &core->sequences[sequence_index];
			break;
		}
	}
	if (sequence == 0)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	full_block_count = token_count / core->block_token_count;
	chain_hash = 0u;
	matched_block_count = 0u;
	while (matched_block_count < full_block_count)
	{
		next_chain_hash = CoreChainHash(
		    chain_hash,
		    token_ids + (uint64_t)matched_block_count * core->block_token_count,
		    core->block_token_count);
		found = CoreIndexFind(
		    core,
		    next_chain_hash,
		    token_ids + (uint64_t)matched_block_count * core->block_token_count);
		if (found == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
		{
			break;
		}
		/* Shared read-only attachment: every matcher holds a reference. */
		core->blocks[found].reference_count++;
		CoreLruTouch(core, found);
		sequence->blocks[matched_block_count] = found;
		chain_hash = next_chain_hash;
		matched_block_count++;
	}
	core->admit_count++;
	core->matched_block_count += matched_block_count;
	core->missed_block_count += full_block_count - matched_block_count;
	sequence->used = 1u;
	sequence->sequence_id = sequence_id;
	sequence->block_count = matched_block_count;
	sequence->running_chain_hash = chain_hash;
	core->live_sequence_count++;
	matched_tokens = matched_block_count * core->block_token_count;
	*matched_token_count_out = matched_tokens;
	if (matched_tokens < token_count)
	{
		status = SparkPrefixCacheCoreAppendTokens(
		    core,
		    sequence_id,
		    token_ids + matched_tokens,
		    token_count - matched_tokens);
		if (status != SPARK_STATUS_OK)
		{
			(void)SparkPrefixCacheCoreReleaseSequence(core, sequence_id);
			return status;
		}
	}
	return SPARK_STATUS_OK;
}

/* The open private slot every append must fill IN ORDER: the leftmost
 * private block with room. Reserved-but-empty tails (ReservePrivateTail)
 * sit after published blocks, so "last block" would fill them out of
 * order and break the correspondence between the token mirror and the
 * block ordinals a driver's table maps positions through. */
static SparkPrefixCacheCoreBlock *CoreOpenBlock(
    SparkPrefixCacheCore *core,
    const SparkPrefixCacheCoreSequence *sequence)
{
	SparkPrefixCacheCoreBlock *candidate;
	uint32_t index;
	for (index = 0u; index < sequence->block_count; index++)
	{
		candidate = &core->blocks[sequence->blocks[index]];
		if (candidate->state == SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE &&
		    candidate->token_count < core->block_token_count)
		{
			return candidate;
		}
	}
	return 0;
}

SparkStatus SparkPrefixCacheCoreAppendTokens(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count)
{
	SparkPrefixCacheCoreSequence *sequence;
	SparkPrefixCacheCoreBlock *block;
	uint32_t sequence_index;
	uint32_t block_index;
	uint32_t token_position;
	uint32_t committed_tokens;
	SparkStatus status;
	if (core == 0 || token_ids == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	status = CoreFindSequence(core, sequence_id, &sequence_index);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	sequence = &core->sequences[sequence_index];
	committed_tokens = CoreCommittedTokens(core, sequence);
	if ((uint64_t)committed_tokens + token_count >
	    (uint64_t)core->sequence_block_capacity * core->block_token_count)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	block = 0;
	for (token_position = 0u; token_position < token_count; token_position++)
	{
		if (block == 0)
		{
			block = CoreOpenBlock(core, sequence);
			if (block == 0)
			{
				/* Private continuation: the open block is always freshly
				 * allocated and never shared until it fills. */
				block_index = CoreAllocateOpenBlock(core);
				if (block_index == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
				{
					return SPARK_STATUS_CAPACITY_EXCEEDED;
				}
				sequence->blocks[sequence->block_count] = block_index;
				sequence->block_count++;
				block = &core->blocks[block_index];
			}
		}
		block->token_ids[block->token_count] = token_ids[token_position];
		block->token_count++;
		core->appended_token_count++;
		if (block->token_count == core->block_token_count)
		{
			/* Full: publish immutable and shareable. */
			/* Plain insert, NOT CoreLruTouch: the open block was
			 * never linked, and Touch's unlink would treat its
			 * NO_BLOCK lru_prev as list-head and wipe the chain. */
			block_index = sequence->blocks[sequence->block_count - 1u];
			block->chain_hash = CoreChainHash(
			    sequence->running_chain_hash,
			    block->token_ids,
			    core->block_token_count);
			block->state = SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED;
			CoreIndexInsert(core, block_index);
			CoreLruPushMru(core, block_index);
			core->tick++;
			block->last_used_tick = core->tick;
			core->published_block_total++;
			sequence->running_chain_hash = block->chain_hash;
			block = 0;
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCoreBuildBlockTable(
	SparkPrefixCacheCore *core,
	uint64_t sequence_id,
	uint32_t token_count,
	uint32_t *block_indices_out,
	uint32_t block_index_capacity,
	uint32_t *block_count_out)
{
	SparkPrefixCacheCoreSequence *sequence;
	uint32_t sequence_index;
	uint32_t required_block_count;
	uint32_t committed_tokens;
	uint32_t index;
	SparkStatus status;
	if (core == 0 || block_count_out == 0 ||
	    (token_count != 0u && block_indices_out == 0))
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	status = CoreFindSequence(core, sequence_id, &sequence_index);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	sequence = &core->sequences[sequence_index];
	committed_tokens = CoreCommittedTokens(core, sequence);
	if (token_count > committed_tokens)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	required_block_count =
	    (token_count + core->block_token_count - 1u) / core->block_token_count;
	if (required_block_count > block_index_capacity)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	for (index = 0u; index < required_block_count; index++)
	{
		block_indices_out[index] = sequence->blocks[index];
	}
	*block_count_out = required_block_count;
	return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCoreReservePrivateTail(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t min_block_count)
{
	SparkPrefixCacheCoreSequence *sequence;
	uint32_t sequence_index;
	uint32_t block_index;
	SparkStatus status;
	if (core == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	status = CoreFindSequence(core, sequence_id, &sequence_index);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	if (min_block_count > core->sequence_block_capacity)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	sequence = &core->sequences[sequence_index];
	while (sequence->block_count < min_block_count)
	{
		block_index = CoreAllocateOpenBlock(core);
		if (block_index == SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
		{
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		}
		sequence->blocks[sequence->block_count] = block_index;
		sequence->block_count++;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCoreCopyBlockList(
    const SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t *block_indices_out,
    uint32_t block_index_capacity,
    uint32_t *block_count_out)
{
	SparkPrefixCacheCoreSequence *sequence;
	uint32_t sequence_index;
	uint32_t index;
	SparkStatus status;
	if (core == 0 || block_count_out == 0 ||
	    block_indices_out == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	status = CoreFindSequence(core, sequence_id, &sequence_index);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	sequence = &core->sequences[sequence_index];
	if (sequence->block_count > block_index_capacity)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	for (index = 0u; index < sequence->block_count; index++)
	{
		block_indices_out[index] = sequence->blocks[index];
	}
	*block_count_out = sequence->block_count;
	return SPARK_STATUS_OK;
}

uint32_t SparkPrefixCacheCoreSequenceTokenCount(
	const SparkPrefixCacheCore *core,
	uint64_t sequence_id)
{
	uint32_t sequence_index;
	if (core == 0 || CoreFindSequence(core, sequence_id, &sequence_index) !=
	                  SPARK_STATUS_OK)
	{
		return 0u;
	}
	return CoreCommittedTokens(core, &core->sequences[sequence_index]);
}

SparkStatus SparkPrefixCacheCoreReleaseSequence(
	SparkPrefixCacheCore *core,
	uint64_t sequence_id)
{
	SparkPrefixCacheCoreSequence *sequence;
	SparkPrefixCacheCoreBlock *block;
	uint32_t sequence_index;
	uint32_t ordinal;
	uint32_t block_index;
	SparkStatus status;
	if (core == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	status = CoreFindSequence(core, sequence_id, &sequence_index);
	if (status != SPARK_STATUS_OK)
	{
		return status;
	}
	sequence = &core->sequences[sequence_index];
	for (ordinal = 0u; ordinal < sequence->block_count; ordinal++)
	{
		block_index = sequence->blocks[ordinal];
		block = &core->blocks[block_index];
		block->reference_count--;
		if (block->reference_count == 0u)
		{
			if (block->state == SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE)
			{
				/* The open partial block is never cached. */
				CoreFreeBlock(core, block_index);
			}
			/* Published blocks stay cached and LRU-evictable. */
		}
	}
	sequence->used = 0u;
	sequence->block_count = 0u;
	sequence->running_chain_hash = 0u;
	core->live_sequence_count--;
	return SPARK_STATUS_OK;
}

SparkStatus SparkPrefixCacheCoreTrim(
	SparkPrefixCacheCore *core,
	uint32_t free_target,
	uint32_t *evicted_block_count_out)
{
	SparkPrefixCacheCoreBlock *block;
	uint32_t scan;
	uint32_t previous;
	uint32_t freed;
	if (core == 0 || evicted_block_count_out == 0)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	freed = 0u;
	scan = core->lru_tail;
	while (scan != SPARK_PREFIX_CACHE_CORE_NO_BLOCK && freed < free_target)
	{
		previous = core->blocks[scan].lru_prev;
		block = &core->blocks[scan];
		if (block->state == SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED &&
		    block->reference_count == 0u)
		{
			CoreLruUnlink(core, scan);
			CoreIndexRemove(core, scan);
			CoreFreeBlock(core, scan);
			core->evicted_block_count++;
			freed++;
		}
		scan = previous;
	}
	if (freed < free_target)
	{
		core->capacity_stall_count++;
	}
	*evicted_block_count_out = freed;
	return SPARK_STATUS_OK;
}

void SparkPrefixCacheCoreQueryStats(
	const SparkPrefixCacheCore *core,
	SparkPrefixCacheCoreStats *stats)
{
	uint32_t block_index;
	if (core == 0 || stats == 0)
	{
		return;
	}
	memset(stats, 0, sizeof(*stats));
	stats->abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	stats->descriptor_bytes = SPARK_PREFIX_CACHE_CORE_STATS_DESCRIPTOR_BYTES;
	stats->live_sequence_count = core->live_sequence_count;
	stats->admit_count = core->admit_count;
	stats->matched_block_count = core->matched_block_count;
	stats->missed_block_count = core->missed_block_count;
	stats->appended_token_count = core->appended_token_count;
	stats->published_block_count_total = core->published_block_total;
	stats->evicted_block_count = core->evicted_block_count;
	stats->capacity_stall_count = core->capacity_stall_count;
	stats->bytes_reused = core->matched_block_count * core->block_stride_bytes;
	stats->ticks = core->tick;
	for (block_index = 0u; block_index < core->block_count; block_index++)
	{
		if (core->blocks[block_index].state ==
		    SPARK_PREFIX_CACHE_CORE_BLOCK_FREE)
		{
			stats->free_block_count++;
		}
		else
		{
			stats->used_block_count++;
		}
		if (core->blocks[block_index].state ==
		    SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED)
		{
			stats->published_block_count++;
		}
	}
}
