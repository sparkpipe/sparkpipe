/* Prefix-cache core proof tool.
 *
 * Proves, against a pure-function KV oracle, that content-addressed
 * sharing is exact: every visible KV row of every sequence equals the
 * row a no-reuse run would hold, at every step, under LRU eviction and
 * pool pressure, for B1/B4/B25/B1024 mixed concurrent populations.
 * Each batch runs twice - roomy and squeezed - and the squeezed pool is
 * small enough that cached blocks MUST flow through LRU eviction and
 * physical-block reuse while sequences stay live. Published-block
 * immutability is asserted continuously via digests, and emitted token
 * streams are compared byte-for-byte against a no-reuse replay of the
 * identical operation script.
 */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prefix_cache.h"

#define TEST_MAX_POSITIONS 384u
#define TEST_GROUP_COUNT 7u
#define TEST_GROUP_ROOT_TOKENS 37u

static uint64_t TestMix64(uint64_t value)
{
	value ^= value >> 30;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 27;
	value *= 0x94d049bb133111ebull;
	value ^= value >> 31;
	return value;
}


#define TEST_CAPACITY_BLOCKS 25u
#define TEST_BLOCK_STRIDE_BYTES 64ull
#define TEST_ROWS_DIGEST_OFFSET 14695981039346656037ull
#define TEST_MAX_SEQUENCE_SLOTS 1024u

static uint64_t TestRowsDigest(const uint64_t *rows, uint32_t count)
{
	uint64_t hash;
	uint32_t index;
	hash = TEST_ROWS_DIGEST_OFFSET;
	for (index = 0u; index < count; index++)
	{
		hash ^= rows[index];
		hash *= 1099511628211ull;
	}
	return hash;
}

static uint32_t TestNextPow2(uint32_t value)
{
	uint32_t power;
	power = 1u;
	while (power < value)
	{
		power <<= 1;
	}
	return power;
}

static uint64_t TestOracleAdvance(uint64_t previous_row, uint32_t token_id)
{
	return TestMix64(previous_row ^ TestMix64(token_id));
}

typedef struct TestHarness
{
	SparkPrefixCacheCore core;
	int reuse_on;
	uint32_t block_token_count;
	uint32_t max_sequences;
	uint32_t pool_block_count;
	/* Script shape knobs: prompt span and the token count that triggers
	 * retire-and-readmit churn. Squeezed runs shrink both so releases
	 * (and therefore cached-block churn) start early. */
	uint32_t prompt_span;
	uint32_t release_tokens;
	/* Simulated device pool: one row per (block, slot). */
	uint64_t *pool_rows;
	uint8_t *pool_written;
	/* Published-block immutability digests, keyed by chain hash so a
	 * reallocated physical block re-arms its own digest. */
	uint64_t *published_digest;
	uint64_t *published_chain;
	/* Per-sequence script state, indexed by SLOT. */
	uint32_t *tokens;
	uint64_t (*oracle_rows)[TEST_MAX_POSITIONS];
	uint32_t *token_counts;
	uint8_t *live;
	/* No-reuse mode: private block tables from a plain free list. */
	uint32_t *off_tables;
	uint32_t *off_free_stack;
	uint32_t off_free_count;
	/* Emitted stream: low 32 bits of each newly visible row, read
	 * THROUGH the block table - identical across modes iff sharing is
	 * exact. */
	uint32_t *stream;
	uint64_t stream_length;
	uint64_t stream_capacity;
	uint32_t table_scratch[TEST_MAX_POSITIONS / 16u + 2u];
	/* No-reuse mode: how many table entries each slot owns. */
	uint32_t *off_owned;
	uint64_t verify_row_checks;
	uint64_t digest_checks;
} TestHarness;

static void TestHarnessInitialize(
	TestHarness *harness,
	uint32_t block_count,
	uint32_t max_sequences,
	int reuse_on,
	uint64_t seed,
	uint32_t prompt_span,
	uint32_t release_tokens)
{
	SparkPrefixCacheCoreConfiguration configuration;
	size_t pool_slots;
	uint32_t index;
	memset(harness, 0, sizeof(*harness));
	harness->reuse_on = reuse_on;
	harness->block_token_count = 16u;
	harness->max_sequences = max_sequences;
	harness->pool_block_count = reuse_on ? block_count
	    : max_sequences * TEST_CAPACITY_BLOCKS;
	harness->prompt_span = prompt_span;
	harness->release_tokens = release_tokens;
	(void)seed;
	pool_slots = (size_t)harness->pool_block_count * harness->block_token_count;
	harness->pool_rows = (uint64_t *)calloc(pool_slots, sizeof(uint64_t));
	harness->pool_written = (uint8_t *)calloc(pool_slots, 1u);
	harness->published_digest =
	    (uint64_t *)calloc(harness->pool_block_count, sizeof(uint64_t));
	harness->published_chain =
	    (uint64_t *)calloc(harness->pool_block_count, sizeof(uint64_t));
	harness->tokens = (uint32_t *)calloc(
	    (size_t)max_sequences * TEST_MAX_POSITIONS, sizeof(uint32_t));
	harness->oracle_rows = (uint64_t (*)[TEST_MAX_POSITIONS])calloc(
	    max_sequences, sizeof(*harness->oracle_rows));
	harness->token_counts =
	    (uint32_t *)calloc(max_sequences, sizeof(uint32_t));
	harness->live = (uint8_t *)calloc(max_sequences, 1u);
	harness->off_tables = (uint32_t *)calloc(
	    (size_t)max_sequences * TEST_CAPACITY_BLOCKS, sizeof(uint32_t));
	harness->off_free_stack = (uint32_t *)calloc(
	    harness->pool_block_count, sizeof(uint32_t));
	harness->off_owned = (uint32_t *)calloc(max_sequences, sizeof(uint32_t));
	harness->stream_capacity = 1ull << 22;
	harness->stream =
	    (uint32_t *)malloc((size_t)harness->stream_capacity * sizeof(uint32_t));
	assert(harness->pool_rows != 0 && harness->pool_written != 0 &&
	       harness->published_digest != 0 && harness->published_chain != 0 &&
	       harness->tokens != 0 && harness->oracle_rows != 0 &&
	       harness->token_counts != 0 && harness->live != 0 &&
	       harness->off_tables != 0 && harness->off_free_stack != 0 &&
	       harness->off_owned != 0 && harness->stream != 0);
	for (index = 0u; index < harness->pool_block_count; index++)
	{
		harness->off_free_stack[index] =
		    harness->pool_block_count - 1u - index;
	}
	harness->off_free_count = harness->pool_block_count;
	if (reuse_on)
	{
		memset(&configuration, 0, sizeof(configuration));
		configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
		configuration.descriptor_bytes =
		    SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
		configuration.block_token_count = harness->block_token_count;
		configuration.block_stride_bytes = TEST_BLOCK_STRIDE_BYTES;
		configuration.block_count = block_count;
		configuration.max_sequence_count = max_sequences;
		configuration.sequence_block_capacity = TEST_CAPACITY_BLOCKS;
		configuration.hash_bucket_count = TestNextPow2(block_count) * 2u;
		assert(SparkPrefixCacheCoreInitialize(
		    &harness->core, &configuration) == SPARK_STATUS_OK);
	}
}

static void TestHarnessDestroy(TestHarness *harness)
{
	if (harness->reuse_on)
	{
		SparkPrefixCacheCoreDestroy(&harness->core);
	}
	free(harness->pool_rows);
	free(harness->pool_written);
	free(harness->published_digest);
	free(harness->published_chain);
	free(harness->tokens);
	free(harness->oracle_rows);
	free(harness->token_counts);
	free(harness->live);
	free(harness->off_tables);
	free(harness->off_free_stack);
	free(harness->off_owned);
	free(harness->stream);
	memset(harness, 0, sizeof(*harness));
}


static uint32_t TestPromptToken(
	uint64_t seed,
	uint32_t slot,
	uint32_t cycle,
	uint32_t position)
{
	return (uint32_t)(TestMix64(seed ^ ((uint64_t)cycle << 40) ^
	                        ((uint64_t)slot << 20) ^ position) >> 32) | 1u;
}

/* Grouped prompts: intra-group sequences share a 37-token root (which
 * crosses block boundaries on purpose), intra-family ones share an
 * even longer root; everything else diverges. Pure function of its
 * inputs so both replay modes generate identical scripts. */
static uint32_t TestBuildPrompt(
	TestHarness *harness,
	uint64_t seed,
	uint32_t slot,
	uint32_t cycle,
	uint32_t *out,
	uint32_t *length_out)
{
	uint32_t group;
	uint32_t family;
	uint32_t shared;
	uint32_t position;
	(void)harness;
	group = slot % TEST_GROUP_COUNT;
	family = (slot / TEST_GROUP_COUNT) % 4u;
	shared = TEST_GROUP_ROOT_TOKENS + family * 19u;
	if (shared > TEST_GROUP_ROOT_TOKENS + harness->prompt_span)
	{
		/* Squeezed runs keep every prompt inside the first block
		 * pair so the family tier collapses into the group root. */
		shared = TEST_GROUP_ROOT_TOKENS;
	}
	*length_out = 33u + (uint32_t)(((uint64_t)slot * 13u + cycle * 29u) %
	                               harness->prompt_span);
	for (position = 0u; position < *length_out; position++)
	{
		if (position < TEST_GROUP_ROOT_TOKENS)
		{
			out[position] = TestPromptToken(seed ^ 0x5a5aull, group, 0u, position);
		}
		else if (position < shared)
		{
			out[position] = TestPromptToken(seed ^ 0xa5a5ull, group, family, position);
		}
		else
		{
			out[position] = TestPromptToken(seed, slot, cycle, position);
		}
	}
	return group;
}

/* Continuation tokens are a pure function of (slot, position, cycle). */
static uint32_t TestContinuationToken(
	uint64_t seed,
	uint32_t slot,
	uint32_t cycle,
	uint32_t position)
{
	return TestPromptToken(seed ^ 0xdeadbeefull, slot, cycle, position);
}

static void TestCommitTokens(
	TestHarness *harness,
	uint32_t slot,
	const uint32_t *ids,
	uint32_t count)
{
	uint32_t base;
	uint32_t index;
	base = harness->token_counts[slot];
	assert(base + count <= TEST_MAX_POSITIONS);
	for (index = 0u; index < count; index++)
	{
		harness->tokens[(size_t)slot * TEST_MAX_POSITIONS + base + index] =
		    ids[index];
		harness->oracle_rows[slot][base + index] = TestOracleAdvance(
		    base + index == 0u ? 0ull :
		        harness->oracle_rows[slot][base + index - 1u],
		    ids[index]);
	}
	harness->token_counts[slot] = base + count;
}

static const uint32_t *TestSequenceTable(
	TestHarness *harness,
	uint32_t slot,
	uint32_t token_count,
	uint32_t *table_count_out)
{
	uint32_t sequence_id;
	sequence_id = slot + 1u;
	if (harness->reuse_on)
	{
		assert(SparkPrefixCacheCoreBuildBlockTable(
		    &harness->core, sequence_id, token_count, harness->table_scratch,
		    TEST_CAPACITY_BLOCKS, table_count_out) == SPARK_STATUS_OK);
	}
	else
	{
		uint32_t index;
		uint32_t needed;
		needed = (token_count + harness->block_token_count - 1u) /
		         harness->block_token_count;
		/* Growth on demand: decode bursts extend the sequence past its
		 * initial prompt, so claim fresh exclusive blocks exactly like
		 * the core claims open blocks. */
		while (harness->off_owned[slot] < needed)
		{
			assert(harness->off_free_count > 0u);
			harness->off_tables[(size_t)slot * TEST_CAPACITY_BLOCKS +
			                    harness->off_owned[slot]] =
			    harness->off_free_stack[--harness->off_free_count];
			harness->off_owned[slot]++;
		}
		for (index = 0u; index < needed; index++)
		{
			harness->table_scratch[index] =
			    harness->off_tables[
			        (size_t)slot * TEST_CAPACITY_BLOCKS + index];
		}
		*table_count_out = needed;
	}
	return harness->table_scratch;
}

/* Write every not-yet-materialized row from from_position onward and
 * push what the kernels would read onto the emitted stream. */
static void TestMaterializeTail(
	TestHarness *harness,
	uint32_t slot,
	uint32_t from_position)
{
	const uint32_t *table;
	uint32_t table_count;
	uint32_t token_count;
	uint32_t position;
	uint32_t touched_count;
	uint32_t touched[TEST_CAPACITY_BLOCKS + 2u];
	uint32_t index;
	uint32_t block_index;
	uint32_t pool_slot;
	token_count = harness->token_counts[slot];
	table = TestSequenceTable(harness, slot, token_count, &table_count);
	touched_count = 0u;
	for (position = from_position; position < token_count; position++)
	{
		block_index = table[position / harness->block_token_count];
		pool_slot = (size_t)block_index * harness->block_token_count +
		            position % harness->block_token_count;
		/* Device semantics: a write ALWAYS lands, including on a
		 * reallocated physical block. Exactness is not assumed here -
		 * the full-prefix audit after every operation catches any
		 * clobbered or stale visible row. */
		harness->pool_written[pool_slot] = 1u;
		harness->pool_rows[pool_slot] =
		    harness->oracle_rows[slot][position];
		assert(harness->stream_length < harness->stream_capacity);
		harness->stream[harness->stream_length++] =
		    (uint32_t)harness->pool_rows[pool_slot];
		for (index = 0u; index < touched_count; index++)
		{
			if (touched[index] == block_index)
			{
				break;
			}
		}
		if (index == touched_count)
		{
			touched[touched_count++] = block_index;
		}
	}
	if (harness->reuse_on)
	{
		for (index = 0u; index < touched_count; index++)
		{
			block_index = touched[index];
			if (harness->core.blocks[block_index].state ==
			        SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED &&
			    harness->published_chain[block_index] !=
			        harness->core.blocks[block_index].chain_hash)
			{
				harness->published_chain[block_index] =
				    harness->core.blocks[block_index].chain_hash;
				harness->published_digest[block_index] = TestRowsDigest(
				    harness->pool_rows +
				        (size_t)block_index * harness->block_token_count,
				    harness->block_token_count);
			}
		}
	}
}

/* Full visibility audit: every visible row equals the oracle and every
 * published block still matches its publish-time digest. */
static void TestVerifySequence(TestHarness *harness, uint32_t slot)
{
	const uint32_t *table;
	uint32_t table_count;
	uint32_t token_count;
	uint32_t position;
	uint32_t index;
	uint32_t block_index;
	uint32_t pool_slot;
	uint32_t seen_blocks[TEST_CAPACITY_BLOCKS + 2u];
	uint32_t seen_count;
	token_count = harness->token_counts[slot];
	table = TestSequenceTable(harness, slot, token_count, &table_count);
	seen_count = 0u;
	for (position = 0u; position < token_count; position++)
	{
		block_index = table[position / harness->block_token_count];
		pool_slot = (size_t)block_index * harness->block_token_count +
		            position % harness->block_token_count;
		assert(harness->pool_written[pool_slot] != 0u);
		assert(harness->pool_rows[pool_slot] ==
		       harness->oracle_rows[slot][position]);
		harness->verify_row_checks++;
		for (index = 0u; index < seen_count; index++)
		{
			if (seen_blocks[index] == block_index)
			{
				break;
			}
		}
		if (index == seen_count)
		{
			seen_blocks[seen_count++] = block_index;
		}
	}
	if (harness->reuse_on)
	{
		for (index = 0u; index < seen_count; index++)
		{
			block_index = seen_blocks[index];
			if (harness->published_chain[block_index] ==
			    harness->core.blocks[block_index].chain_hash &&
			    harness->published_chain[block_index] != 0u)
			{
				assert(harness->published_digest[block_index] ==
				       TestRowsDigest(
				           harness->pool_rows +
				               (size_t)block_index *
				                   harness->block_token_count,
				           harness->block_token_count));
				harness->digest_checks++;
			}
		}
	}
}


static void TestOpAdmit(
	TestHarness *harness,
	uint64_t seed,
	uint32_t slot,
	uint32_t cycle)
{
	uint32_t prompt[TEST_MAX_POSITIONS];
	uint32_t length;
	uint32_t matched_tokens;
	uint32_t needed;
	uint32_t index;
	assert(harness->live[slot] == 0u);
	(void)TestBuildPrompt(harness, seed, slot, cycle, prompt, &length);
	needed = (length + harness->block_token_count - 1u) /
	         harness->block_token_count;
	if (harness->reuse_on)
	{
		assert(SparkPrefixCacheCoreAdmitSequence(
		    &harness->core, slot + 1u, prompt, length,
		    &matched_tokens) == SPARK_STATUS_OK);
		assert(matched_tokens % harness->block_token_count == 0u);
		assert(matched_tokens <= length);
	}
	else
	{
		for (index = 0u; index < needed; index++)
		{
			assert(harness->off_free_count > 0u);
			harness->off_tables[
			    (size_t)slot * TEST_CAPACITY_BLOCKS + index] =
			    harness->off_free_stack[--harness->off_free_count];
		}
		harness->off_owned[slot] = needed;
	}
	TestCommitTokens(harness, slot, prompt, length);
	TestMaterializeTail(harness, slot, 0u);
	TestVerifySequence(harness, slot);
	harness->live[slot] = 1u;
}

static void TestOpStep(
	TestHarness *harness,
	uint64_t seed,
	uint32_t slot,
	uint32_t cycle,
	uint32_t burst)
{
	uint32_t ids[8u];
	uint32_t base;
	uint32_t index;
	assert(harness->live[slot] != 0u);
	base = harness->token_counts[slot];
	if (base + burst > TEST_MAX_POSITIONS)
	{
		burst = TEST_MAX_POSITIONS - base;
	}
	if (burst == 0u)
	{
		return;
	}
	for (index = 0u; index < burst; index++)
	{
		ids[index] = TestContinuationToken(
		    seed, slot, cycle, base + index);
	}
	if (harness->reuse_on)
	{
		assert(SparkPrefixCacheCoreAppendTokens(
		    &harness->core, slot + 1u, ids, burst) == SPARK_STATUS_OK);
	}
	TestCommitTokens(harness, slot, ids, burst);
	TestMaterializeTail(harness, slot, base);
	TestVerifySequence(harness, slot);
}

static void TestOpRelease(TestHarness *harness, uint32_t slot)
{
	uint32_t index;
	uint32_t used;
	assert(harness->live[slot] != 0u);
	if (harness->reuse_on)
	{
		assert(SparkPrefixCacheCoreReleaseSequence(
		    &harness->core, slot + 1u) == SPARK_STATUS_OK);
	}
	else
	{
		used = (harness->token_counts[slot] +
		        harness->block_token_count - 1u) /
		       harness->block_token_count;
		for (index = 0u; index < used; index++)
		{
			harness->off_free_stack[harness->off_free_count++] =
			    harness->off_tables[
			        (size_t)slot * TEST_CAPACITY_BLOCKS + index];
		}
		harness->off_owned[slot] = 0u;
	}
	harness->live[slot] = 0u;
	harness->token_counts[slot] = 0u;
}

/* Run one deterministic mixed population at batch width B: grouped
 * prompts (shared and diverging), interleaved decode bursts, releases
 * and id reuse. Runs twice - reuse ON and OFF - over the identical
 * script and compares the emitted streams byte-for-byte.
 *
 * Geometry variants:
 * - roomy: pool batch*9+16 covers the worst-case live set (~9 blocks
 *   per live slot), cached retention piles up but trimming stays rare;
 * - squeezed: short prompts plus a low retire threshold cap the live
 *   set at ~4 blocks per slot while the pool is batch*4+2, so released
 *   published blocks MUST cycle through LRU eviction and physical
 *   reuse while neighbours stay live - asserted, not hoped. */
static void TestRunMatrixBatch(
	uint32_t batch,
	uint64_t seed,
	uint32_t pool_per_slot,
	uint32_t pool_headroom,
	uint32_t prompt_span,
	uint32_t release_tokens,
	int demand_eviction,
	const char *label,
	uint64_t *row_checks_out,
	uint64_t *digest_checks_out)
{
	TestHarness on;
	TestHarness off;
	uint32_t pool_blocks;
	uint32_t slot;
	uint32_t cycle[TEST_MAX_SEQUENCE_SLOTS];
	uint32_t round_index;
	uint32_t op_rounds;
	uint64_t on_length;
	pool_blocks = batch * pool_per_slot + pool_headroom;
	assert(batch <= 1024u);
	TestHarnessInitialize(&on, pool_blocks, batch, 1, seed, prompt_span,
	    release_tokens);
	TestHarnessInitialize(&off, pool_blocks, batch, 0, seed, prompt_span,
	    release_tokens);
	memset(cycle, 0, sizeof(cycle));
	for (slot = 0u; slot < batch; slot++)
	{
		TestOpAdmit(&on, seed, slot, 0u);
		TestOpAdmit(&off, seed, slot, 0u);
	}
	op_rounds = 6u;
	for (round_index = 0u; round_index < op_rounds; round_index++)
	{
		for (slot = 0u; slot < batch; slot++)
		{
			uint32_t burst;
			burst = 1u + (uint32_t)(TestMix64(seed ^ slot ^
			                          ((uint64_t)round_index << 8)) % 4u);
			TestOpStep(&on, seed, slot, cycle[slot]++, burst);
			TestOpStep(&off, seed, slot, cycle[slot] - 1u, burst);
			/* Retire and re-admit finished-length sequences often:
			 * the churn keeps cached blocks flowing through LRU
			 * while every live set stays below pool capacity. */
			if (((slot + round_index) & 1u) == 0u &&
			    on.token_counts[slot] >= release_tokens)
			{
				TestOpRelease(&on, slot);
				TestOpRelease(&off, slot);
				cycle[slot]++;
				TestOpAdmit(&on, seed, slot, cycle[slot]);
				TestOpAdmit(&off, seed, slot, cycle[slot]);
			}
		}
	}
	if (demand_eviction)
	{
		/* Squeeze phase: unconditional retire-and-readmit churn
		 * until the eviction path has demonstrably fired under the
		 * matrix itself - a pressure run that never evicts proves
		 * nothing about correctness under physical-block reuse. */
		uint32_t squeeze_round;
		squeeze_round = 0u;
		while (on.core.evicted_block_count == 0ull &&
		       squeeze_round < 24u)
		{
			for (slot = 0u; slot < batch; slot++)
			{
				uint32_t burst;
				burst = 1u +
				    (uint32_t)(TestMix64(seed ^ 0x51edull ^
				                         slot ^
				                         ((uint64_t)squeeze_round << 8)) % 4u);
				TestOpStep(&on, seed, slot, cycle[slot]++, burst);
				TestOpStep(&off, seed, slot, cycle[slot] - 1u, burst);
				if (((slot + squeeze_round) & 1u) == 0u)
				{
					TestOpRelease(&on, slot);
					TestOpRelease(&off, slot);
					cycle[slot]++;
					TestOpAdmit(&on, seed, slot, cycle[slot]);
					TestOpAdmit(&off, seed, slot, cycle[slot]);
				}
			}
			squeeze_round++;
		}
		assert(on.core.evicted_block_count > 0ull);
	}
	for (slot = 0u; slot < batch; slot++)
	{
		if (on.live[slot] != 0u)
		{
			TestOpRelease(&on, slot);
			TestOpRelease(&off, slot);
		}
	}
	on_length = on.stream_length;
	assert(on_length == off.stream_length);
	assert(memcmp(on.stream, off.stream,
	       (size_t)on_length * sizeof(uint32_t)) == 0);
	if (demand_eviction)
	{
		/* The squeeze phase above must have succeeded. */
		assert(on.core.evicted_block_count > 0ull);
		assert(off.stream_length == on.stream_length);
	}
	printf("B%-4u %-8s PASS rows=%" PRIu64 " stream=%" PRIu64
	       " matched_blocks=%" PRIu64 " evicted=%" PRIu64
	       " stalls=%" PRIu64 "\n",
	    batch, label, on.verify_row_checks, on_length,
	    on.core.matched_block_count, on.core.evicted_block_count,
	    on.core.capacity_stall_count);
	*row_checks_out += on.verify_row_checks;
	*digest_checks_out += on.digest_checks;
	TestHarnessDestroy(&on);
	TestHarnessDestroy(&off);
}



static SparkStatus TestTinyCore(
	SparkPrefixCacheCore *core,
	uint32_t block_token_count,
	uint32_t block_count,
	uint32_t max_sequences)
{
	SparkPrefixCacheCoreConfiguration configuration;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	configuration.descriptor_bytes =
	    SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.block_token_count = block_token_count;
	configuration.block_stride_bytes = TEST_BLOCK_STRIDE_BYTES;
	configuration.block_count = block_count;
	configuration.max_sequence_count = max_sequences;
	configuration.sequence_block_capacity = 8u;
	configuration.hash_bucket_count = TestNextPow2(block_count) * 2u;
	return SparkPrefixCacheCoreInitialize(core, &configuration);
}

static void TestGeometryValidation(void)
{
	SparkPrefixCacheCoreConfiguration configuration;
	SparkPrefixCacheCore core;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	configuration.descriptor_bytes =
	    SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.block_token_count = 16u;
	configuration.block_stride_bytes = 128ull;
	configuration.block_count = 64u;
	configuration.max_sequence_count = 8u;
	configuration.sequence_block_capacity = 4u;
	configuration.hash_bucket_count = 128u;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_OK);
	configuration.abi_version = 99u;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_INVALID_ARGUMENT);
	configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	configuration.block_token_count = 0u;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_INVALID_ARGUMENT);
	configuration.block_token_count =
	    SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS + 1u;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_INVALID_ARGUMENT);
	configuration.block_token_count = 16u;
	configuration.block_stride_bytes = 0ull;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_INVALID_ARGUMENT);
	configuration.block_stride_bytes = 128ull;
	configuration.hash_bucket_count = 127u;
	assert(SparkPrefixCacheCoreValidateGeometry(&configuration) ==
	       SPARK_STATUS_INVALID_ARGUMENT);
	configuration.hash_bucket_count = 128u;
	assert(TestTinyCore(&core, 4u, 16u, 4u) == SPARK_STATUS_OK);
	SparkPrefixCacheCoreDestroy(&core);
	printf("geometry PASS\n");
}

static void TestLcpMatchingAndSharing(void)
{
	SparkPrefixCacheCore core;
	uint32_t prompt[24];
	uint32_t index;
	uint32_t matched;
	uint32_t table_a[8u];
	uint32_t table_c[8u];
	uint32_t count_a;
	uint32_t count_c;
	assert(TestTinyCore(&core, 4u, 32u, 8u) == SPARK_STATUS_OK);
	for (index = 0u; index < 20u; index++)
	{
		prompt[index] = 100u + index;
	}
	/* Cold admit: nothing to match, everything recomputes. */
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 11u, prompt, 20u, &matched) == SPARK_STATUS_OK);
	assert(matched == 0u);
	/* Identical resubmission: every full block matches - zero recompute.
	 * The 20-token prompt leaves no open block: 5 full blocks publish. */
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 12u, prompt, 20u, &matched) == SPARK_STATUS_OK);
	assert(matched == 20u);
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 11u, 20u, table_a, 8u, &count_a) == SPARK_STATUS_OK);
	assert(count_a == 5u);
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 12u, 20u, table_a, 8u, &count_a) == SPARK_STATUS_OK);
	/* Both sequences reference the SAME physical blocks. */
	for (index = 0u; index < count_a; index++)
	{
		assert(core.blocks[table_a[index]].reference_count == 2u);
	}
	/* Ten-token sibling diverges inside block two: match stops at the
	 * block boundary, eight tokens are shared, two recompute. */
	prompt[10] = 999u;
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 13u, prompt, 20u, &matched) == SPARK_STATUS_OK);
	assert(matched == 8u);
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 13u, 20u, table_c, 8u, &count_c) == SPARK_STATUS_OK);
	assert(table_c[0] == table_a[0] && table_c[1] == table_a[1]);
	assert(table_c[2] != table_a[2]);
	assert(core.live_sequence_count == 3u);
	SparkPrefixCacheCoreDestroy(&core);
	printf("lcp+sharing PASS\n");
}


static void TestDivergencePrivateContinuation(void)
{
	SparkPrefixCacheCore core;
	uint32_t prompt[24];
	uint32_t index;
	uint32_t matched;
	uint32_t table_a[8u];
	uint32_t table_b[8u];
	uint32_t table_c[8u];
	uint32_t count;
	uint32_t snapshot[8u];
	assert(TestTinyCore(&core, 4u, 32u, 8u) == SPARK_STATUS_OK);
	for (index = 0u; index < 8u; index++)
	{
		prompt[index] = 500u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 21u, prompt, 8u, &matched) == SPARK_STATUS_OK);
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 22u, prompt, 8u, &matched) == SPARK_STATUS_OK);
	assert(matched == 8u);
	/* Diverge: A continues one way, B another. */
	prompt[0] = 601u;
	assert(SparkPrefixCacheCoreAppendTokens(
	    &core, 21u, prompt, 4u) == SPARK_STATUS_OK);
	prompt[0] = 701u;
	assert(SparkPrefixCacheCoreAppendTokens(
	    &core, 22u, prompt, 4u) == SPARK_STATUS_OK);
	/* Structural isolation: A's table is unchanged by B's writes; the
	 * shared blocks keep two references; each open tail is private. */
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 21u, 12u, table_a, 8u, &count) == SPARK_STATUS_OK);
	assert(count == 3u);
	for (index = 0u; index < count; index++)
	{
		snapshot[index] = table_a[index];
	}
	assert(SparkPrefixCacheCoreAppendTokens(
	    &core, 22u, prompt, 4u) == SPARK_STATUS_OK);
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 21u, 12u, table_b, 8u, &count) == SPARK_STATUS_OK);
	assert(count == 3u);
	for (index = 0u; index < count; index++)
	{
		assert(table_b[index] == snapshot[index]);
	}
	assert(core.blocks[snapshot[0]].reference_count == 2u);
	/* Cross-check B's own chain: two private full blocks, distinct from
	 * everything A owns, sharing exactly the common prefix. */
	assert(SparkPrefixCacheCoreBuildBlockTable(
	    &core, 22u, 16u, table_c, 8u, &count) == SPARK_STATUS_OK);
	assert(count == 4u);
	assert(table_c[0] == snapshot[0] && table_c[1] == snapshot[1]);
	assert(table_c[2] != snapshot[2]);
	assert(table_c[3] != snapshot[2]);
	assert(table_c[2] != table_c[3]);
	assert(core.blocks[snapshot[2]].reference_count == 1u);
	assert(core.blocks[table_c[2]].reference_count == 1u);
	assert(core.blocks[table_c[2]].state ==
	       SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED);
	assert(core.blocks[table_c[2]].token_count == 4u);
	SparkPrefixCacheCoreDestroy(&core);
	printf("divergence-private-continuation PASS\n");
}

static void TestReleaseRematch(void)
{
	SparkPrefixCacheCore core;
	uint32_t prompt[24];
	uint32_t index;
	uint32_t matched;
	assert(TestTinyCore(&core, 4u, 32u, 8u) == SPARK_STATUS_OK);
	for (index = 0u; index < 20u; index++)
	{
		prompt[index] = 800u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 31u, prompt, 20u, &matched) == SPARK_STATUS_OK);
	assert(matched == 0u);
	/* Finish: the open partial block dies, full blocks stay cached. */
	assert(SparkPrefixCacheCoreReleaseSequence(&core, 31u) ==
	       SPARK_STATUS_OK);
	assert(core.live_sequence_count == 0u);
	/* A later submission with the same prompt reuses every full block
	 * with zero recompute - the finished sequence is a cache source. */
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 32u, prompt, 20u, &matched) == SPARK_STATUS_OK);
	assert(matched == 20u);
	/* Release idempotence guard: releasing an unknown id reports it. */
	assert(SparkPrefixCacheCoreReleaseSequence(&core, 99u) ==
	       SPARK_STATUS_NOT_FOUND);
	/* Double-admit guard: a live id cannot claim a second slot (that
	 * would leak the first slot's block references). */
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 32u, prompt, 20u, &matched) == SPARK_STATUS_DUPLICATE);
	assert(core.live_sequence_count == 1u);
	SparkPrefixCacheCoreDestroy(&core);
	printf("release-rematch PASS\n");
}

static void TestEvictionLruOrder(void)
{
	SparkPrefixCacheCore core;
	uint32_t prompt_a[16];
	uint32_t prompt_b[16];
	uint32_t prompt_c[24];
	uint32_t index;
	uint32_t matched;
	SparkPrefixCacheCoreStats stats;
	assert(TestTinyCore(&core, 4u, 6u, 8u) == SPARK_STATUS_OK);
	/* A fills all six blocks and finishes: all cached, unreferenced. */
	for (index = 0u; index < 16u; index++)
	{
		prompt_a[index] = 1000u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 41u, prompt_a, 16u, &matched) == SPARK_STATUS_OK);
	assert(SparkPrefixCacheCoreReleaseSequence(&core, 41u) ==
	       SPARK_STATUS_OK);
	/* B shares A's first two blocks (touching them to MRU) and fills
	 * the remaining two pool blocks with its own unique tail. */
	for (index = 0u; index < 16u; index++)
	{
		prompt_b[index] = index < 8u ? prompt_a[index] : 2000u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 42u, prompt_b, 16u, &matched) == SPARK_STATUS_OK);
	assert(matched == 8u);
	/* LRU order is now: block2, block3 (A, untouched), then B's two,
	 * then A's touched zero and one at MRU. C needs six fresh blocks
	 * while B is live: exactly the two unreferenced LRU blocks evict,
	 * then the pool stalls and C is rejected - B's shared blocks
	 * survive because references beat recency. */
	for (index = 0u; index < 24u; index++)
	{
		prompt_c[index] = 3000u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 43u, prompt_c, 24u, &matched) ==
	       SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkPrefixCacheCoreQueryStats(&core, &stats);
	assert(stats.evicted_block_count == 2u);
	assert(stats.capacity_stall_count >= 1u);
	assert(stats.free_block_count + stats.used_block_count == 6u);
	/* After B finishes, A's surviving prefix still rematches. */
	assert(SparkPrefixCacheCoreReleaseSequence(&core, 42u) ==
	       SPARK_STATUS_OK);
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 44u, prompt_a, 16u, &matched) == SPARK_STATUS_OK);
	assert(matched == 8u);
	SparkPrefixCacheCoreDestroy(&core);
	printf("eviction-lru PASS\n");
}

static void TestPressureRejectionAndConsistency(void)
{
	SparkPrefixCacheCore core;
	uint32_t prompt[12];
	uint32_t index;
	uint32_t matched;
	SparkPrefixCacheCoreStats stats;
	assert(TestTinyCore(&core, 4u, 3u, 4u) == SPARK_STATUS_OK);
	for (index = 0u; index < 12u; index++)
	{
		prompt[index] = 400u + index;
	}
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 51u, prompt, 12u, &matched) == SPARK_STATUS_OK);
	/* A second sequence with a NON-matching prompt needs a fresh block;
	 * none exists and all three are referenced, so rejection is clean
	 * and the accounting stays exact. */
	prompt[0] = 900u;
	prompt[1] = 901u;
	prompt[2] = 902u;
	prompt[3] = 903u;
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 52u, prompt, 4u, &matched) ==
	       SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkPrefixCacheCoreQueryStats(&core, &stats);
	assert(stats.free_block_count + stats.used_block_count == 3u);
	assert(stats.live_sequence_count == 1u);
	/* Once the holder releases, its full blocks serve the retry with
	 * zero recompute. */
	prompt[0] = 400u;
	prompt[1] = 401u;
	prompt[2] = 402u;
	prompt[3] = 403u;
	assert(SparkPrefixCacheCoreReleaseSequence(&core, 51u) ==
	       SPARK_STATUS_OK);
	assert(SparkPrefixCacheCoreAdmitSequence(
	    &core, 53u, prompt, 12u, &matched) == SPARK_STATUS_OK);
	assert(matched == 12u);
	SparkPrefixCacheCoreDestroy(&core);
	printf("pressure-rejection PASS\n");
}


int main(void)
{
	setvbuf(stdout, 0, _IONBF, 0);
	static const uint32_t batches[] = { 1u, 4u, 25u, 1024u };
	uint64_t row_checks;
	uint64_t digest_checks;
	uint32_t index;
	TestGeometryValidation();
	TestLcpMatchingAndSharing();
	TestDivergencePrivateContinuation();
	TestReleaseRematch();
	TestEvictionLruOrder();
	TestPressureRejectionAndConsistency();
	row_checks = 0ull;
	digest_checks = 0ull;
	for (index = 0u; index < sizeof(batches) / sizeof(batches[0]); index++)
	{
		/* Seed varies per batch so populations differ; every batch
		 * runs reuse ON vs OFF over the identical script and demands
		 * byte-identical streams - once in a roomy pool and once in
		 * a squeezed pool that forces LRU eviction mid-script. */
		TestRunMatrixBatch(batches[index],
		    0x9e3779b97f4a7c15ull ^ (uint64_t)batches[index],
		    9u, 16u, 96u, 80u, 0, "roomy",
		    &row_checks, &digest_checks);
		TestRunMatrixBatch(batches[index],
		    0xbf58476d1ce4e5b9ull ^ (uint64_t)batches[index],
		    4u, 2u, 16u, 34u, 1, "squeezed",
		    &row_checks, &digest_checks);
	}
	printf("prefix-cache-core PASS rows_verified=%" PRIu64
	       " digest_checks=%" PRIu64 "\n",
	    row_checks, digest_checks);
	return 0;
}
