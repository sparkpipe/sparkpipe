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
#include <time.h>

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

/* Monotonic wall clock for the MEASURED walk-cost case (PERF below). */
static uint64_t TestNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
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
	/* ---- MEASURED walk-cost instrumentation (PERF case only) ----
	 * model_skip switches the simulated device from "write always"
	 * (the correctness cells' pessimistic semantics) to "compute a row
	 * only when its physical block cannot already hold the final
	 * value": a block whose generation is unchanged since the row was
	 * written is read, not recomputed. Generations bump exactly at
	 * fresh physical claims - the write region of every core call -
	 * so a shared/matched published prefix is provably resident and
	 * costs zero recompute. */
	int model_skip;
	uint64_t recomputed_rows;
	uint64_t blocks_borrowed;
	uint64_t committed_tokens;
	uint64_t cover_ns;
	uint64_t verify_ns;
	uint64_t *block_gen;
	uint64_t *pool_gen;
	/* Prompt script source; defaults to TestBuildPrompt, overridden by
	 * the PERF workload's serving-shaped shared-prefix builder. */
	uint32_t (*prompt_builder)(struct TestHarness *, uint64_t, uint32_t,
	    uint32_t, uint32_t *, uint32_t *);
} TestHarness;

/* Grouped prompt builder; forward-declared because the harness default
 * wiring in TestHarnessInitialize points at it. */
static uint32_t TestBuildPrompt(
    TestHarness *harness,
    uint64_t seed,
    uint32_t slot,
    uint32_t cycle,
    uint32_t *out,
    uint32_t *length_out);

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
	harness->block_gen =
	    (uint64_t *)calloc(harness->pool_block_count, sizeof(uint64_t));
	harness->pool_gen = (uint64_t *)calloc(pool_slots, sizeof(uint64_t));
	harness->stream_capacity = 1ull << 22;
	harness->stream =
	    (uint32_t *)malloc((size_t)harness->stream_capacity * sizeof(uint32_t));
	assert(harness->pool_rows != 0 && harness->pool_written != 0 &&
	       harness->published_digest != 0 && harness->published_chain != 0 &&
	       harness->tokens != 0 && harness->oracle_rows != 0 &&
	       harness->token_counts != 0 && harness->live != 0 &&
	       harness->off_tables != 0 && harness->off_free_stack != 0 &&
	       harness->off_owned != 0 && harness->stream != 0 &&
	       harness->block_gen != 0 && harness->pool_gen != 0);
	harness->prompt_builder = TestBuildPrompt;
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
	free(harness->block_gen);
	free(harness->pool_gen);
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
	harness->committed_tokens += count;
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
			uint32_t claimed;
			assert(harness->off_free_count > 0u);
			claimed = harness->off_free_stack[--harness->off_free_count];
			harness->off_tables[(size_t)slot * TEST_CAPACITY_BLOCKS +
			                    harness->off_owned[slot]] = claimed;
			harness->off_owned[slot]++;
			if (harness->model_skip != 0)
			{
				/* Fresh physical claim: stale bytes must never
				 * survive, so the generation bumps and the
				 * claim counts as a borrow. */
				harness->block_gen[claimed]++;
				harness->blocks_borrowed++;
			}
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
	uint64_t timer_start;
	timer_start = TestNowNs();
	token_count = harness->token_counts[slot];
	table = TestSequenceTable(harness, slot, token_count, &table_count);
	touched_count = 0u;
	for (position = from_position; position < token_count; position++)
	{
		block_index = table[position / harness->block_token_count];
		pool_slot = (size_t)block_index * harness->block_token_count +
		            position % harness->block_token_count;
		if (harness->model_skip != 0 &&
		    harness->pool_written[pool_slot] != 0u &&
		    harness->pool_gen[pool_slot] == harness->block_gen[block_index])
		{
			/* Model-level reuse: the physical block generation is
			 * unchanged since this row was written, so the block
			 * still holds the final value - the device READS it
			 * (shared published prefix or untouched tail), it does
			 * not recompute it. */
		}
		else
		{
			/* Device semantics: a write ALWAYS lands, including on
			 * a reallocated physical block (its generation bumped
			 * at claim time, so stale bytes can never masquerade
			 * as resident). Exactness is not assumed here - the
			 * full-prefix audit after every operation catches any
			 * clobbered or stale visible row. */
			harness->pool_written[pool_slot] = 1u;
			harness->pool_gen[pool_slot] =
			    harness->block_gen[block_index];
			harness->pool_rows[pool_slot] =
			    harness->oracle_rows[slot][position];
			harness->recomputed_rows++;
		}
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
	harness->cover_ns += TestNowNs() - timer_start;
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
	uint64_t timer_start;
	timer_start = TestNowNs();
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
	harness->verify_ns += TestNowNs() - timer_start;
}


/* MEASURED-case helper: after a core call whose WRITE REGION starts at
 * ordinal `first_fresh_ordinal` (admit: matched_tokens/block_tokens;
 * append: the block count before the burst), bump generations and count
 * borrows for every physical block the call claimed fresh. Blocks below
 * that ordinal are matched/shared attaches - their bytes are provably
 * resident (a published block is immutable and only leaves the index
 * when evicted, at which point it cannot match anymore), so they cost
 * zero recompute and no borrow. */
static void TestNoteWriteRegion(
    TestHarness *harness,
    uint32_t slot,
    uint32_t token_count,
    uint32_t first_fresh_ordinal)
{
	uint32_t table[TEST_CAPACITY_BLOCKS];
	uint32_t table_count;
	uint32_t ordinal;
	assert(SparkPrefixCacheCoreBuildBlockTable(
	           &harness->core, slot + 1u, token_count, table,
	           TEST_CAPACITY_BLOCKS, &table_count) == SPARK_STATUS_OK);
	for (ordinal = first_fresh_ordinal; ordinal < table_count; ordinal++)
	{
		harness->block_gen[table[ordinal]]++;
		harness->blocks_borrowed++;
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
	harness->prompt_builder(harness, seed, slot, cycle, prompt, &length);
	needed = (length + harness->block_token_count - 1u) /
	         harness->block_token_count;
	if (harness->reuse_on)
	{
		assert(SparkPrefixCacheCoreAdmitSequence(
		    &harness->core, slot + 1u, prompt, length,
		    &matched_tokens) == SPARK_STATUS_OK);
		assert(matched_tokens % harness->block_token_count == 0u);
		assert(matched_tokens <= length);
		if (harness->model_skip != 0)
		{
			TestNoteWriteRegion(
			    harness, slot, length,
			    matched_tokens / harness->block_token_count);
		}
	}
	else
	{
		for (index = 0u; index < needed; index++)
		{
			uint32_t claimed;
			assert(harness->off_free_count > 0u);
			claimed = harness->off_free_stack[--harness->off_free_count];
			harness->off_tables[
			    (size_t)slot * TEST_CAPACITY_BLOCKS + index] =
			    claimed;
			if (harness->model_skip != 0)
			{
				harness->block_gen[claimed]++;
				harness->blocks_borrowed++;
			}
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
		if (harness->model_skip != 0)
		{
			TestNoteWriteRegion(
			    harness, slot, base + burst,
			    (base + harness->block_token_count - 1u) /
			        harness->block_token_count);
		}
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

/* ---- Decode-with-speculation conservation population (hard law) ----
 *
 * Permanent gate case per INTEGRATION.md "Completeness matrix" (audit
 * F3): multi-round decode via AppendTokens/BuildBlockTable crossing at
 * least two block boundaries per sequence, under mid-block admits and
 * divergences, with FREE-LIST CONSERVATION asserted after every single
 * core call: {true free list} U {blocks held by live sequences} U
 * {cached published blocks with zero references} PARTITIONS the physical
 * pool (no block in two sets, no block in none - the F1 leak class is a
 * block in none), and the stats-reported free count equals the true
 * free-list walk (the F2 class reports outstanding scratch as free).
 * Runs twice inside this one case: a roomy pool where nothing is ever
 * released, so the mandated two-set equation {free list} U {live-held}
 * == pool holds exactly and zero eviction is possible; and a squeezed
 * pool where retire-and-readmit churn forces LRU eviction mid-script -
 * evicted > 0 is asserted, not hoped. Ports 2-4: copy this case verbatim
 * and swap only the geometry numbers.
 */

#define TEST_SPEC_BLOCK_TOKENS 8u
#define TEST_SPEC_SLOT_COUNT 3u
#define TEST_SPEC_ROUND_COUNT 12u
#define TEST_SPEC_LENGTH_CAP 72u
#define TEST_SPEC_TABLE_CAPACITY 16u
#define TEST_SPEC_MAX_POOL 256u

static void TestSpecAssertConservation(
	SparkPrefixCacheCore *core,
	uint8_t *classes,
	uint8_t *holds,
	int strict_zero_cache)
{
	uint32_t index;
	uint32_t ordinal;
	uint32_t next;
	uint32_t free_walked;
	uint32_t live_held;
	uint32_t cached_count;
	SparkPrefixCacheCoreStats stats;
	memset(classes, 0, core->block_count);
	memset(holds, 0, core->block_count);
	free_walked = 0u;
	next = core->free_block_head;
	while (next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		assert(next < core->block_count);
		/* A repeat visit means a double-return or a cycle. */
		assert(classes[next] == 0u);
		classes[next] = 1u;
		free_walked++;
		next = core->blocks[next].free_next;
	}
	live_held = 0u;
	for (index = 0u; index < core->max_sequence_count; index++)
	{
		if (core->sequences[index].used == 0u)
		{
			continue;
		}
		for (ordinal = 0u; ordinal < core->sequences[index].block_count;
		     ordinal++)
		{
			next = core->sequences[index].blocks[ordinal];
			assert(next < core->block_count);
			/* A live-held block must never ALSO sit on the free
			 * list - that is accounting corruption. Sharing by
			 * several live sequences IS legal: counted once here,
			 * cross-checked against reference_count below. */
			assert(classes[next] != 1u);
			if (classes[next] == 0u)
			{
				classes[next] = 2u;
				live_held++;
			}
			holds[next]++;
		}
	}
	cached_count = 0u;
	for (index = 0u; index < core->block_count; index++)
	{
		if (classes[index] == 2u)
		{
			/* Every reference is owned by exactly one holding
			 * sequence slot: no leaked or phantom references. */
			assert(holds[index] ==
			       core->blocks[index].reference_count);
			continue;
		}
		if (classes[index] != 0u)
		{
			continue;
		}
		/* The only legal way to be reachable from neither the free
		 * list nor a live sequence is the finished-sequence cache:
		 * published, zero references, LRU-evictable. Anything else
		 * here is an orphaned block (the F1 class). */
		assert(core->blocks[index].state ==
		       SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED);
		assert(core->blocks[index].reference_count == 0u);
		classes[index] = 3u;
		cached_count++;
	}
	SparkPrefixCacheCoreQueryStats(core, &stats);
	/* The reported counter must match the true walk - never count
	 * outstanding scratch as free (the F2 class). */
	assert(stats.free_block_count == free_walked);
	if (strict_zero_cache)
	{
		assert(cached_count == 0u);
	}
	assert(free_walked + live_held + cached_count == core->block_count);
}

static void TestSpecRunPopulation(
	uint32_t pool_blocks,
	int churn,
	const char *label)
{
	SparkPrefixCacheCore core;
	SparkPrefixCacheCoreConfiguration configuration;
	SparkPrefixCacheCoreStats stats;
	uint8_t classes[TEST_SPEC_MAX_POOL];
	uint8_t holds[TEST_SPEC_MAX_POOL];
	uint32_t prompt[TEST_SPEC_LENGTH_CAP];
	uint32_t canon[TEST_SPEC_SLOT_COUNT][TEST_SPEC_LENGTH_CAP];
	uint32_t canon_length[TEST_SPEC_SLOT_COUNT];
	uint32_t draft[8u];
	uint32_t table[TEST_SPEC_TABLE_CAPACITY];
	uint32_t table_count;
	uint8_t live[TEST_SPEC_SLOT_COUNT];
	uint32_t round;
	uint32_t slot;
	uint32_t victim;
	uint32_t position;
	uint32_t draft_count;
	uint32_t take_length;
	uint32_t matched;
	uint32_t trim_evicted;
	uint32_t peak_blocks;
	uint64_t seed = 0x2545f4914f6cdd1dull;
	assert(pool_blocks <= TEST_SPEC_MAX_POOL);
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	configuration.descriptor_bytes =
	    SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.block_token_count = TEST_SPEC_BLOCK_TOKENS;
	configuration.block_stride_bytes = TEST_BLOCK_STRIDE_BYTES;
	configuration.block_count = pool_blocks;
	configuration.max_sequence_count = TEST_SPEC_SLOT_COUNT;
	configuration.sequence_block_capacity = TEST_SPEC_TABLE_CAPACITY;
	configuration.hash_bucket_count = TestNextPow2(pool_blocks) * 2u;
	assert(SparkPrefixCacheCoreInitialize(&core, &configuration) ==
	       SPARK_STATUS_OK);
	memset(canon_length, 0, sizeof(canon_length));
	memset(live, 0, sizeof(live));
	peak_blocks = 0u;
	for (round = 0u; round < TEST_SPEC_ROUND_COUNT; round++)
	{
		/* Slot 0 admits cold at round 0; slot 1 shares its first
		 * eleven tokens then DIVERGES INSIDE BLOCK ONE (mid-block
		 * admit: exactly one full block matches); slot 2 joins at
		 * round 3 sharing fifteen tokens of slot 0's prompt - again
		 * matched_tokens stops at the last FULL block. */
		if (round == 0u || round == 3u)
		{
			slot = round == 0u ? 0u : 2u;
			for (position = 0u; position < 20u; position++)
			{
				if (position < 15u)
				{
					prompt[position] = TestContinuationToken(
					    seed, 0u, 0u, position);
				}
				else
				{
					prompt[position] = TestContinuationToken(
					    seed ^ 0x77ull, slot, 0u, position);
				}
			}
			memcpy(canon[slot], prompt, 20u * sizeof(uint32_t));
			assert(SparkPrefixCacheCoreAdmitSequence(
			    &core, slot + 1u, prompt, 20u, &matched) ==
			       SPARK_STATUS_OK);
			if (slot == 2u && !churn)
			{
				assert(matched == 8u);
			}
			assert(matched % TEST_SPEC_BLOCK_TOKENS == 0u);
			assert(matched <= 20u);
			canon_length[slot] = 20u;
			live[slot] = 1u;
			TestSpecAssertConservation(&core, classes, holds, !churn);
		}
		if (round == 0u)
		{
			/* Slot 1: same first block as slot 0, divergent tail
			 * from token 11 on - private continuation. */
			for (position = 0u; position < 20u; position++)
			{
				prompt[position] = position < 11u ?
				    canon[0][position] :
				    TestContinuationToken(seed ^ 0xeeull, 1u,
				        0u, position);
			}
			memcpy(canon[1], prompt, 20u * sizeof(uint32_t));
			assert(SparkPrefixCacheCoreAdmitSequence(
			    &core, 2u, prompt, 20u, &matched) ==
			       SPARK_STATUS_OK);
			if (!churn)
			{
				assert(matched == 8u);
			}
			canon_length[1] = 20u;
			live[1] = 1u;
			TestSpecAssertConservation(&core, classes, holds, !churn);
		}
		/* Speculative decode rounds: each step appends a multi-token
		 * DRAFT burst (2..5) per live sequence - the accepted-token
		 * shape of speculation - so open blocks fill and publish
		 * across block boundaries every two-ish rounds. */
		for (slot = 0u; slot < TEST_SPEC_SLOT_COUNT; slot++)
		{
			if (live[slot] == 0u)
			{
				continue;
			}
			draft_count = 2u + ((round + slot) % 4u);
			if (canon_length[slot] + draft_count >
			    TEST_SPEC_LENGTH_CAP)
			{
				draft_count =
				    TEST_SPEC_LENGTH_CAP - canon_length[slot];
			}
			if (draft_count != 0u)
			{
				for (position = 0u; position < draft_count;
				     position++)
				{
					draft[position] = TestContinuationToken(
					    seed ^ 0xa11ceull, slot, 0u,
					    canon_length[slot] + position);
				}
				assert(SparkPrefixCacheCoreAppendTokens(
				    &core, slot + 1u, draft, draft_count) ==
				       SPARK_STATUS_OK);
				memcpy(canon[slot] + canon_length[slot], draft,
				    draft_count * sizeof(uint32_t));
				canon_length[slot] += draft_count;
				TestSpecAssertConservation(
				    &core, classes, holds, !churn);
			}
			assert(SparkPrefixCacheCoreSequenceTokenCount(
			           &core, slot + 1u) == canon_length[slot]);
			assert(SparkPrefixCacheCoreBuildBlockTable(
			    &core, slot + 1u, canon_length[slot], table,
			    TEST_SPEC_TABLE_CAPACITY, &table_count) ==
			       SPARK_STATUS_OK);
			assert(table_count ==
			       (canon_length[slot] + TEST_SPEC_BLOCK_TOKENS -
			           1u) /
			           TEST_SPEC_BLOCK_TOKENS);
			if (table_count > peak_blocks)
			{
				peak_blocks = table_count;
			}
		}
		/* Squeezed variant only: retire one sequence mid-population
		 * and re-admit it from a TRUNCATED canonical prefix (21
		 * tokens - not a block multiple - so the readmit is itself
		 * a mid-block admit against whatever cache survived). This
		 * churn is what pushes finished blocks through LRU eviction
		 * while neighbours stay live. A real driver trims for
		 * headroom; so does this script. */
		if (churn && round >= 2u && round % 3u == 2u)
		{
			victim = round % TEST_SPEC_SLOT_COUNT;
			if (live[victim] != 0u)
			{
				assert(SparkPrefixCacheCoreReleaseSequence(
				    &core, victim + 1u) == SPARK_STATUS_OK);
				live[victim] = 0u;
				TestSpecAssertConservation(&core, classes, holds, 0);
				take_length = canon_length[victim] > 21u ?
				    21u : canon_length[victim];
				assert(SparkPrefixCacheCoreAdmitSequence(
				    &core, victim + 1u, canon[victim],
				    take_length, &matched) == SPARK_STATUS_OK);
				assert(matched % TEST_SPEC_BLOCK_TOKENS == 0u);
				assert(matched <= take_length);
				canon_length[victim] = take_length;
				live[victim] = 1u;
				assert(SparkPrefixCacheCoreTrim(
				    &core, 2u, &trim_evicted) ==
				       SPARK_STATUS_OK);
				TestSpecAssertConservation(
				    &core, classes, holds, 0);
			}
		}
	}
	/* Mandate: at least two block-boundary crossings were exercised -
	 * a sequence must have grown past three blocks. */
	assert(peak_blocks >= 3u);
	SparkPrefixCacheCoreQueryStats(&core, &stats);
	if (!churn)
	{
		/* Roomy pool, no releases: strict equation, zero eviction. */
		assert(core.evicted_block_count == 0ull);
		for (slot = 0u; slot < TEST_SPEC_SLOT_COUNT; slot++)
		{
			assert(live[slot] != 0u);
		}
	}
	else
	{
		/* Pressure was real: eviction fired inside this case. */
		assert(core.evicted_block_count > 0ull);
	}
	printf("spec-conservation %-8s PASS pool=%u peak_blocks=%u "
	       "evicted=%" PRIu64 "\n",
	    label, pool_blocks, peak_blocks, core.evicted_block_count);
	SparkPrefixCacheCoreDestroy(&core);
}

static void TestDecodeWithSpeculationConservation(void)
{
	/* Roomy: nothing released, so {free list} U {live-held} == pool
	 * must hold EXACTLY every step. Squeezed: release/readmit churn
	 * under a pool that cannot hold the accumulated cache, so LRU
	 * eviction fires mid-population and the three-set partition is
	 * what conserves. Same case, both variants, as the law requires. */
	TestSpecRunPopulation(64u, 0, "roomy");
	TestSpecRunPopulation(30u, 1, "squeezed");
	printf("decode-with-speculation-conservation PASS\n");
}


/* ---- MEASURED prefix-reuse walk-cost case (queue task: first measured
 * performance evidence for this port) ----
 *
 * Serving-shaped shared-prefix workload: every prompt carries a 64-token
 * (4-block) root shared across the whole batch, prompt groups share one
 * extra family block, and every submission carries a unique divergent
 * tail; retire-and-readmit churn resubmits against the resident cache so
 * every admit after a root's first cold miss matches block-granular
 * prefixes and then DIVERGES - exactly the traffic prefix caching exists
 * for. The identical script runs reuse ON and OFF; the emitted streams
 * must stay byte-identical.
 *
 * Walk-cost model (host simulation - device kernel time is NOT claimed):
 * the simulated device computes a KV row only when its physical block
 * cannot already hold the final value. Generations bump exactly at fresh
 * physical claims (the write region of every core call), so matched/
 * shared published prefixes are provably resident and cost zero
 * recompute. Counters are exact, not sampled:
 *   recomputed rows ON  = total tokens - matched tokens   (asserted)
 *   recomputed rows OFF = total tokens
 * Wall clock covers the Cover (row materialization) and verify walks.
 * Timing is min-of-reps; treat it as order-of-magnitude evidence, the
 * recompute-savings counters as the hard numbers. */

#define PERF_ROUNDS 240u
#define PERF_REPS 3u
#define PERF_RELEASE_AT 140u

typedef struct TestPerfModeResult
{
	uint64_t recomputed_rows;
	uint64_t borrowed_blocks;
	uint64_t committed_tokens;
	uint64_t matched_tokens;
	uint64_t appended_tokens;
	uint64_t evicted_blocks;
	uint64_t best_walk_ns;
}
TestPerfModeResult;

static uint32_t TestPerfPromptToken(uint64_t salt, uint32_t a, uint32_t b)
{
	return (uint32_t)(TestMix64(salt ^ ((uint64_t)a << 24) ^
	                    (uint64_t)b) >> 32) | 1u;
}

/* Pure function of (seed, slot, cycle): ON and OFF generate identical
 * scripts. Roots are shared batch-wide, family tiers per group, tails
 * unique per (slot, cycle) - shared AND diverging prefixes every round. */
static uint32_t TestPerfBuildPrompt(
    TestHarness *harness,
    uint64_t seed,
    uint32_t slot,
    uint32_t cycle,
    uint32_t *out,
    uint32_t *length_out)
{
	uint32_t group;
	uint32_t shared;
	uint32_t position;
	(void)harness;
	group = slot % 5u;
	shared = 64u + (slot % 3u) * 16u;
	*length_out = 112u + ((slot * 13u + cycle * 29u) % 48u);
	for (position = 0u; position < *length_out; position++)
	{
		if (position < 64u)
		{
			out[position] =
			    TestPerfPromptToken(seed ^ 0x51ed2705ull, group, position);
		}
		else if (position < shared)
		{
			out[position] =
			    TestPerfPromptToken(seed ^ 0x2545f491ull, group, position);
		}
		else
		{
			out[position] = TestPerfPromptToken(seed,
			    slot ^ ((uint32_t)cycle << 8), position);
		}
	}
	return group;
}

static void TestPerfRunPopulation(
    TestHarness *on,
    TestHarness *off,
    uint64_t seed,
    uint32_t batch,
    uint32_t rounds)
{
	uint32_t cycle[TEST_MAX_SEQUENCE_SLOTS];
	uint32_t slot;
	uint32_t round_index;
	memset(cycle, 0, sizeof(cycle));
	for (slot = 0u; slot < batch; slot++)
	{
		TestOpAdmit(on, seed, slot, 0u);
		TestOpAdmit(off, seed, slot, 0u);
	}
	for (round_index = 0u; round_index < rounds; round_index++)
	{
		for (slot = 0u; slot < batch; slot++)
		{
			uint32_t burst;
			burst = 1u +
			    (uint32_t)(TestMix64(seed ^ slot ^
			                     ((uint64_t)round_index << 8)) % 4u);
			TestOpStep(on, seed, slot, cycle[slot]++, burst);
			TestOpStep(off, seed, slot, cycle[slot] - 1u, burst);
			/* Retire-and-readmit churn: finished-length sequences
			 * resubmit against the resident cache, matching their
			 * roots and diverging in the tail. */
			if (((slot + round_index) & 1u) == 0u &&
			    on->token_counts[slot] >= PERF_RELEASE_AT)
			{
				TestOpRelease(on, slot);
				TestOpRelease(off, slot);
				cycle[slot]++;
				TestOpAdmit(on, seed, slot, cycle[slot]);
				TestOpAdmit(off, seed, slot, cycle[slot]);
			}
		}
	}
	for (slot = 0u; slot < batch; slot++)
	{
		if (on->live[slot] != 0u)
		{
			TestOpRelease(on, slot);
			TestOpRelease(off, slot);
		}
	}
}

static void TestPerfCell(uint32_t batch, uint64_t seed)
{
	TestHarness on;
	TestHarness off;
	SparkPrefixCacheCoreStats stats;
	TestPerfModeResult on_result;
	TestPerfModeResult off_result;
	uint64_t pool_blocks;
	uint64_t walk_ns;
	uint32_t rep;
	memset(&on_result, 0, sizeof(on_result));
	memset(&off_result, 0, sizeof(off_result));
	/* Cache-resident regime: the pool holds the WHOLE run's distinct
	 * content (~230 blocks/slot measured headroom), so no cached prefix
	 * is ever evicted mid-run - the measurement isolates reuse value
	 * from LRU-churn noise (pressure behavior is the correctness
	 * cells' job). Asserted below via evicted == 0. */
	pool_blocks = batch * 384u + 16u;
	for (rep = 0u; rep < PERF_REPS; rep++)
	{
		uint64_t matched;
		TestHarnessInitialize(&on, pool_blocks, batch, 1, seed, 96u,
		    PERF_RELEASE_AT);
		TestHarnessInitialize(&off, pool_blocks, batch, 0, seed, 96u,
		    PERF_RELEASE_AT);
		on.model_skip = 1;
		off.model_skip = 1;
		on.prompt_builder = TestPerfBuildPrompt;
		off.prompt_builder = TestPerfBuildPrompt;
		TestPerfRunPopulation(&on, &off, seed, batch, PERF_ROUNDS);
		/* Correctness anchor: byte-identical visible streams. */
		assert(on.stream_length == off.stream_length);
		assert(memcmp(on.stream, off.stream,
		       (size_t)on.stream_length * sizeof(uint32_t)) == 0);
		/* Roomy-pool sanity: serving never refused an admit and the
		 * cache stayed resident (no eviction mid-measurement). */
		SparkPrefixCacheCoreQueryStats(&on.core, &stats);
		assert(stats.capacity_stall_count == 0ull);
		assert(stats.evicted_block_count == 0ull);
		/* Determinism: every rep executes the identical script. */
		if (rep != 0u)
		{
			assert(on_result.recomputed_rows == on.recomputed_rows);
			assert(off_result.recomputed_rows == off.recomputed_rows);
			assert(on_result.borrowed_blocks == on.blocks_borrowed);
			assert(off_result.borrowed_blocks == off.blocks_borrowed);
		}
		matched = stats.matched_block_count * on.block_token_count;
		assert(matched % on.block_token_count == 0ull);
		/* Accounting law: recompute avoided EXACTLY equals matched
		 * (block-granular) tokens - no hand-waving savings. */
		assert(off.recomputed_rows - on.recomputed_rows == matched);
		/* The core appended exactly the non-matched tokens. */
		assert(stats.appended_token_count ==
		       off.committed_tokens - matched);
		assert(on.recomputed_rows < off.recomputed_rows);
		on_result.recomputed_rows = on.recomputed_rows;
		off_result.recomputed_rows = off.recomputed_rows;
		on_result.borrowed_blocks = on.blocks_borrowed;
		off_result.borrowed_blocks = off.blocks_borrowed;
		on_result.matched_tokens = matched;
		on_result.appended_tokens = stats.appended_token_count;
		on_result.evicted_blocks = stats.evicted_block_count;
		walk_ns = on.cover_ns + on.verify_ns;
		if (rep == 0u || walk_ns < on_result.best_walk_ns)
		{
			on_result.best_walk_ns = walk_ns;
		}
		walk_ns = off.cover_ns + off.verify_ns;
		if (rep == 0u || walk_ns < off_result.best_walk_ns)
		{
			off_result.best_walk_ns = walk_ns;
		}
		TestHarnessDestroy(&on);
		TestHarnessDestroy(&off);
	}
	printf("perf b=%u mode=reuse-on  recomputed_rows=%" PRIu64
	       " borrowed_blocks=%" PRIu64 " matched_tokens=%" PRIu64
	       " evicted=%" PRIu64 " walk_us=%" PRIu64 "\n",
	    batch, on_result.recomputed_rows, on_result.borrowed_blocks,
	    on_result.matched_tokens, on_result.evicted_blocks,
	    on_result.best_walk_ns / 1000ull);
	printf("perf b=%u mode=reuse-off recomputed_rows=%" PRIu64
	       " borrowed_blocks=%" PRIu64 " matched_tokens=0 evicted=0"
	       " walk_us=%" PRIu64 "\n",
	    batch, off_result.recomputed_rows, off_result.borrowed_blocks,
	    off_result.best_walk_ns / 1000ull);
	{
		double savings_pct;
		double speedup;
		savings_pct = 100.0 *
		    (double)(off_result.recomputed_rows -
		             on_result.recomputed_rows) /
		    (double)off_result.recomputed_rows;
		speedup = (double)off_result.best_walk_ns /
		          (double)on_result.best_walk_ns;
		printf("PERF_RESULT b=%u recompute_off=%" PRIu64
		       " recompute_on=%" PRIu64 " tokens_avoided=%" PRIu64
		       " savings_pct=%.1f wall_off_us=%" PRIu64
		       " wall_on_us=%" PRIu64 " speedup_x=%.2f\n",
		    batch, off_result.recomputed_rows,
		    on_result.recomputed_rows,
		    off_result.recomputed_rows - on_result.recomputed_rows,
		    savings_pct, off_result.best_walk_ns / 1000ull,
		    on_result.best_walk_ns / 1000ull, speedup);
	}
}

static void TestPerfReuseSavings(void)
{
	printf("perf_reuse_savings: MEASURED host-side walk simulation "
	       "(shared-prefix workload with divergences; recompute counters "
	       "exact, wall clock min-of-%u reps)\n", PERF_REPS);
	TestPerfCell(1u, 0x9e3779b97f4a7c15ull ^ 1ull);
	TestPerfCell(4u, 0x9e3779b97f4a7c15ull ^ 4ull);
	TestPerfCell(25u, 0x9e3779b97f4a7c15ull ^ 25ull);
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
	TestDecodeWithSpeculationConservation();
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
	/* MEASURED prefix-reuse walk-cost evidence (B1/B4/B25, ON vs OFF). */
	TestPerfReuseSavings();
	printf("prefix-cache-core PASS rows_verified=%" PRIu64
	       " digest_checks=%" PRIu64 "\n",
	    row_checks, digest_checks);
	return 0;
}
