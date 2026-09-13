#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

/* Compiled in for the conservation cases: the gate drives the GLM-5.2
 * paged-KV port directly, at the adapter's per-round call pattern, and
 * asserts free-list conservation after EVERY step (the qwen36/qwen38
 * audit lessons 7+8). This is the unit half of the per-driver
 * prefix-cache gate family; the serving-boundary half lands with the
 * adapter lifecycle wiring (admit-on-submit / cover-per-round), exactly
 * as the qwen ports grew theirs. */
#include "spark_glm52_paged_kv.h"

/* Driver-true KV geometry: the compressed-latent block stride the core
 * prices reuse with comes from THIS model's cache headers - 64-token
 * blocks over (latent + rope) BF16 scalars across all 78 layers. */
#include "sparkpipe/spark_glm52_kv_geometry.h"

#define G1_BLOCK_TOKENS SPARK_GLM52_KV_BLOCK_TOKEN_COUNT
#define G1_BLOCKS_PER_LANE 32u
#define G1_LANES 4u
#define G1_POOL 96u
#define G1_ROUNDS 140u
#define G1_DRAFT 8u
#define G1_PROMPT_TOKENS (G1_BLOCK_TOKENS * 3u)

static uint32_t g1_table[G1_LANES * G1_BLOCKS_PER_LANE];
static uint32_t g1_counts[G1_LANES];

typedef struct GateFixture
{
	uint32_t *table;
	uint32_t *counts;
	uint32_t lane_count;
	uint32_t blocks_per_lane;
	uint32_t pool;
}
GateFixture;

/* The {free list} U {lane rows} U {live sequences} U {core LRU} == pool
 * partition walk, verbatim in shape from the qwen38 unit gate.
 * Structural corruption asserts; semantic violations count. */
static uint32_t GatePartitionAudit(const SparkGlm52PagedKv *cache,
	const GateFixture *fixture)
{
	uint8_t *in_free,*in_use;
	uint32_t i,ordinal,next,violations = 0u;
	in_free = (uint8_t *)calloc(fixture->pool,1u);
	in_use = (uint8_t *)calloc(fixture->pool,1u);
	assert(in_free != 0 && in_use != 0);
	next = cache->core.free_block_head;
	while ( next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
	{
		assert(next < fixture->pool);
		assert(in_free[next] == 0u); /* repeat visit == cycle */
		in_free[next] = 1u;
		next = cache->core.blocks[next].free_next;
	}
	/* Detached published blocks sit on the LRU (Trim recovers them). */
	next = cache->core.lru_head;
	while ( next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
	{
		assert(next < fixture->pool);
		in_use[next] = 1u;
		next = cache->core.blocks[next].lru_next;
	}
	for (i=0u; i<fixture->lane_count; i++)
		for (ordinal=0u; ordinal<fixture->blocks_per_lane; ordinal++)
		{
			uint32_t block =
				fixture->table[i * fixture->blocks_per_lane + ordinal];
			if ( ordinal >= fixture->counts[i] )
			{
				assert(block == SPARK_GLM52_PAGED_KV_NO_BLOCK);
				continue;
			}
			assert(block != SPARK_GLM52_PAGED_KV_NO_BLOCK &&
				block < fixture->pool);
			in_use[block] = 1u;
		}
	for (i=0u; i<cache->core.max_sequence_count; i++)
	{
		const SparkPrefixCacheCoreSequence *sequence =
			&cache->core.sequences[i];
		if ( sequence->used == 0u )
			continue;
		for (ordinal=0u; ordinal<sequence->block_count; ordinal++)
		{
			uint32_t block = cache->core.sequence_blocks[
				i * cache->core.sequence_block_capacity + ordinal];
			assert(block < fixture->pool);
			in_use[block] = 1u;
		}
	}
	for (i=0u; i<fixture->pool; i++)
	{
		if ( in_free[i] != 0u && in_use[i] != 0u )
			violations++; /* free AND in use */
		else if ( in_free[i] == 0u && in_use[i] == 0u )
			violations++; /* orphan: neither free nor reachable */
	}
	free(in_free);
	free(in_use);
	return(violations);
}

static void GateConservationCheck(const SparkGlm52PagedKv *cache,
	const GateFixture *fixture,const char *where,uint32_t step)
{
	uint32_t violations = GatePartitionAudit(cache,fixture);
	if ( violations != 0u )
		fprintf(stderr,"conservation %s step=%u violations=%u\n",
			where,step,violations);
	assert(violations == 0u);
}

static void GateFixtureInit(GateFixture *fixture,uint32_t *table,
	uint32_t *counts,uint32_t lane_count,uint32_t blocks_per_lane,
	uint32_t pool)
{
	fixture->table = table;
	fixture->counts = counts;
	fixture->lane_count = lane_count;
	fixture->blocks_per_lane = blocks_per_lane;
	fixture->pool = pool;
	memset(table,0xFF,(size_t)lane_count * blocks_per_lane *
		sizeof(table[0]));
	memset(counts,0,(size_t)lane_count * sizeof(counts[0]));
}

static void GateConfiguration(SparkGlm52PagedKvConfiguration *configuration,
	uint32_t lane_count,uint32_t blocks_per_lane,uint32_t pool,
	uint32_t checkpoint_slots)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->block_token_count = G1_BLOCK_TOKENS;
	configuration->lane_count = lane_count;
	configuration->blocks_per_lane = blocks_per_lane;
	configuration->physical_page_capacity = pool;
	configuration->logical_page_capacity = pool;
	configuration->checkpoint_slot_count = checkpoint_slots;
	configuration->block_stride_bytes = (uint64_t)G1_BLOCK_TOKENS *
		(SPARK_GLM52_MODEL_KV_A_DIMENSION +
			SPARK_GLM52_MODEL_ROPE_DIMENSION) * 2u *
		SPARK_GLM52_MODEL_LAYER_COUNT;
}

static void GateFill(uint32_t *tokens,uint32_t base,uint32_t count)
{
	uint32_t i;
	for (i=0u; i<count; i++)
		tokens[i] = base + i * 7u + 1u;
}

/* Mirror the adapter's post-frame checkpoint binding: offer+commit every
 * publish boundary up to end_position so later admits can clamp to a
 * WITNESSED depth. Without this step the recurrence law correctly
 * refuses all reuse (no donor state anywhere). */
static void GateBindWitnesses(SparkGlm52PagedKv *cache,uint32_t lane,
	uint64_t end_position)
{
	uint64_t boundary;
	uint32_t slot;
	if ( cache->reuse_enabled == 0u )
		return;
	for (boundary = G1_BLOCK_TOKENS; boundary <= end_position;
		boundary += G1_BLOCK_TOKENS)
		if ( SparkGlm52PagedKvCheckpointOffer(cache,lane,boundary,&slot)
			!= 0u )
			SparkGlm52PagedKvCheckpointCommit(cache,lane,slot,
				boundary);
}

/* CASE B: witnessed sharing, divergence, and mid-block admits. Lane 0
 * walks the root and binds witnesses; an identical prompt adopts the
 * donor's published physical blocks VERBATIM; a diverging prompt shares
 * exactly the longest common COMPLETE block run; a prompt diverging
 * inside the first block shares nothing. */
static void GateCaseSharing(void)
{
	SparkGlm52PagedKv cache;
	SparkGlm52PagedKvConfiguration configuration;
	SparkGlm52PagedKvMatch match;
	uint32_t root[3u * G1_BLOCK_TOKENS];
	uint32_t divergent[3u * G1_BLOCK_TOKENS];
	uint32_t inner[2u * G1_BLOCK_TOKENS];
	uint32_t sampled;
	uint32_t i;
	GateFixture fixture;

	GateFixtureInit(&fixture,g1_table,g1_counts,G1_LANES,
		G1_BLOCKS_PER_LANE,G1_POOL);
	GateConfiguration(&configuration,G1_LANES,G1_BLOCKS_PER_LANE,
		G1_POOL,4u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_OK);

	GateFill(root,1000u,G1_PROMPT_TOKENS);
	memcpy(divergent,root,sizeof(uint32_t) * 2u * G1_BLOCK_TOKENS);
	for (i = 2u * G1_BLOCK_TOKENS; i < G1_PROMPT_TOKENS; i++)
		divergent[i] = 9000u + i;
	memcpy(inner,root,sizeof(uint32_t) * (G1_BLOCK_TOKENS / 2u));
	for (i = G1_BLOCK_TOKENS / 2u; i < 2u * G1_BLOCK_TOKENS; i++)
		inner[i] = 8000u + i;

	/* Lane 0: cold walk of the root, then bind every boundary. */
	assert(SparkGlm52PagedKvAdmit(&cache,0u,root,G1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 0u && match.checkpoint_slot ==
		SPARK_GLM52_PAGED_KV_NO_SLOT);
	assert(SparkGlm52PagedKvCommittedTokens(&cache,0u) ==
		G1_PROMPT_TOKENS);
	GateConservationCheck(&cache,&fixture,"B/donor-admit",0u);
	GateBindWitnesses(&cache,0u,G1_PROMPT_TOKENS);

	/* The donor decodes one token past the root (block 4 opens). */
	sampled = 424242u;
	assert(SparkGlm52PagedKvCover(&cache,0u,G1_PROMPT_TOKENS + 1u,
		&sampled,1u) == SPARK_STATUS_OK);
	GateConservationCheck(&cache,&fixture,"B/donor-decode",1u);

	/* Lane 1: the identical prompt adopts the donor blocks verbatim -
	 * all three boundaries are witnessed, so the full run resumes. */
	assert(SparkGlm52PagedKvAdmit(&cache,1u,root,G1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 3u);
	assert(match.checkpoint_slot != SPARK_GLM52_PAGED_KV_NO_SLOT);
	for (i=0u; i<3u; i++)
		assert(g1_table[1u * G1_BLOCKS_PER_LANE + i] ==
			g1_table[0u * G1_BLOCKS_PER_LANE + i]);
	assert(SparkGlm52PagedKvCommittedTokens(&cache,1u) ==
		G1_PROMPT_TOKENS);
	GateConservationCheck(&cache,&fixture,"B/identical",2u);

	/* Lane 2: divergence inside block 3 shares exactly the two complete
	 * common blocks and takes a private continuation past them. */
	assert(SparkGlm52PagedKvAdmit(&cache,2u,divergent,G1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 2u);
	for (i=0u; i<2u; i++)
		assert(g1_table[2u * G1_BLOCKS_PER_LANE + i] ==
			g1_table[0u * G1_BLOCKS_PER_LANE + i]);
	assert(g1_table[2u * G1_BLOCKS_PER_LANE + 2u] !=
		g1_table[0u * G1_BLOCKS_PER_LANE + 2u]);
	assert(SparkGlm52PagedKvCommittedTokens(&cache,2u) ==
		G1_PROMPT_TOKENS);
	GateConservationCheck(&cache,&fixture,"B/divergent",3u);

	/* Lane 3: divergence inside the FIRST block publishes nothing
	 * shared - block-granular by construction, match stays empty. */
	assert(SparkGlm52PagedKvAdmit(&cache,3u,inner,
		2u * G1_BLOCK_TOKENS,&match) == SPARK_STATUS_OK);
	assert(match.block_count == 0u);
	GateConservationCheck(&cache,&fixture,"B/inner",4u);

	/* The core saw every admit; reuse is doing the accounting. */
	assert(cache.core.admit_count >= 4u);

	/* Teardown conserves: everything returns to the free list. */
	/* Teardown conserves: released sequences leave their published
	 * blocks cached for later matches (the reuse point); one Trim
	 * recovers every detached block and returns the WHOLE pool. */
	for (i=0u; i<G1_LANES; i++)
		SparkGlm52PagedKvLaneReset(&cache,i);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,G1_POOL,
			&evicted) == SPARK_STATUS_OK);
		assert(evicted > 0u); /* sharing kept blocks hot */
	}
	assert(SparkGlm52PagedKvFreeBlocks(&cache) == G1_POOL);
	GateConservationCheck(&cache,&fixture,"B/teardown",5u);
	SparkGlm52PagedKvDestroy(&cache);
	printf("glm52 prefix-cache CASE B sharing PASS\n");
}

/* CASE C: >G1_ROUNDS simulated B1 speculative rounds - Cover(end =
 * pos + 1) with one accepted token plus the speculative extension
 * Cover(end = pos + DRAFT + 1) with NO canonical tokens while scratch
 * is outstanding - crossing several block boundaries, conservation
 * after EVERY call, reuse ON and OFF, mid-stream lane reset with
 * outstanding scratch, conserving teardown. */
static void GateRunRounds(SparkGlm52PagedKv *cache,GateFixture *fixture,
	uint32_t lane,const char *where)
{
	uint32_t round,pos,tok;
	pos = G1_BLOCK_TOKENS;
	for (round=0u; round<G1_ROUNDS; round++)
	{
		tok = 500000u + round * 13u + lane * 7u;
		assert(SparkGlm52PagedKvCover(cache,lane,(uint64_t)pos + 1u,
			&tok,1u) == SPARK_STATUS_OK);
		GateConservationCheck(cache,fixture,where,round * 2u);
		assert(SparkGlm52PagedKvCover(cache,lane,
			(uint64_t)pos + 1u + G1_DRAFT + 1u,0,0) ==
			SPARK_STATUS_OK);
		GateConservationCheck(cache,fixture,where,round * 2u + 1u);
		pos += 1u;
		/* Reset with outstanding scratch before the row nears its
		 * width ceiling, then re-admit a fresh single-block root -
		 * the churn path the F1 audit class lived in. */
		if ( (pos + G1_DRAFT + 1u) / G1_BLOCK_TOKENS >=
			fixture->blocks_per_lane - 2u )
		{
			SparkGlm52PagedKvLaneReset(cache,lane);
			GateConservationCheck(cache,fixture,where,1000000u +
				round);
			tok = 700000u + round;
			assert(SparkGlm52PagedKvAdmit(cache,lane,&tok,1u,0) ==
				SPARK_STATUS_OK);
			GateConservationCheck(cache,fixture,where,2000000u +
				round);
			pos = G1_BLOCK_TOKENS;
			/* Re-admit bound the sequence at ONE token; cover back
			 * to a full block so pos arithmetic keeps holding. */
			{
				uint32_t filler;
				for (filler=pos; filler<G1_BLOCK_TOKENS; filler++)
				{
					uint32_t id = tok + filler;
					assert(SparkGlm52PagedKvCover(cache,lane,
						(uint64_t)filler + 1u,&id,1u) ==
						SPARK_STATUS_OK);
				}
			}
		}
	}
}

static void GateCaseSpeculative(void)
{
	SparkGlm52PagedKv cache;
	SparkGlm52PagedKvConfiguration configuration;
	GateFixture fixture;
	uint32_t root,i;

	/* Reuse ON. */
	GateFixtureInit(&fixture,g1_table,g1_counts,2u,
		G1_BLOCKS_PER_LANE,48u);
	GateConfiguration(&configuration,2u,G1_BLOCKS_PER_LANE,48u,2u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_OK);
	root = 31337u;
	assert(SparkGlm52PagedKvAdmit(&cache,0u,&root,1u,0) ==
		SPARK_STATUS_OK);
	assert(SparkGlm52PagedKvAdmit(&cache,1u,&root,1u,0) ==
		SPARK_STATUS_OK);
	GateRunRounds(&cache,&fixture,0u,"C/on-lane0");
	GateRunRounds(&cache,&fixture,1u,"C/on-lane1");
	for (i=0u; i<2u; i++)
		SparkGlm52PagedKvLaneReset(&cache,i);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,48u,&evicted) ==
			SPARK_STATUS_OK);
	}
	assert(SparkGlm52PagedKvFreeBlocks(&cache) == 48u);
	SparkGlm52PagedKvDestroy(&cache);
	printf("glm52 prefix-cache CASE C speculative ON PASS\n");

	/* Reuse OFF: same shape, pure scratch, still conserved. */
	GateFixtureInit(&fixture,g1_table,g1_counts,2u,
		G1_BLOCKS_PER_LANE,48u);
	GateConfiguration(&configuration,2u,G1_BLOCKS_PER_LANE,48u,0u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_OK);
	assert(SparkGlm52PagedKvAdmit(&cache,0u,&root,1u,0) ==
		SPARK_STATUS_OK);
	assert(SparkGlm52PagedKvAdmit(&cache,1u,&root,1u,0) ==
		SPARK_STATUS_OK);
	GateRunRounds(&cache,&fixture,0u,"C/off-lane0");
	GateRunRounds(&cache,&fixture,1u,"C/off-lane1");
	for (i=0u; i<2u; i++)
		SparkGlm52PagedKvLaneReset(&cache,i);
	assert(SparkGlm52PagedKvFreeBlocks(&cache) == 48u);
	SparkGlm52PagedKvDestroy(&cache);
	printf("glm52 prefix-cache CASE C speculative OFF PASS\n");
}

/* CASE D: squeezed pool under multi-root churn - every admission
 * completes or refuses CLEANLY up front, evictions happen, and the
 * partition survives every step including the failed ones. */
static void GateCaseSqueeze(void)
{
	SparkGlm52PagedKv cache;
	SparkGlm52PagedKvConfiguration configuration;
	SparkGlm52PagedKvMatch match;
	SparkPrefixCacheCoreStats stats;
	GateFixture fixture;
	uint32_t tokens[2u * G1_BLOCK_TOKENS];
	uint32_t root,step,status;

	GateFixtureInit(&fixture,g1_table,g1_counts,3u,8u,12u);
	GateConfiguration(&configuration,3u,8u,12u,2u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_OK);
	for (root=0u; root<6u; root++)
	{
		GateFill(tokens,root * 100000u + 31u,2u * G1_BLOCK_TOKENS);
		status = SparkGlm52PagedKvAdmit(&cache,root % 3u,tokens,
			2u * G1_BLOCK_TOKENS,&match);
		assert(status == SPARK_STATUS_OK ||
			status == SPARK_STATUS_CAPACITY_EXCEEDED);
		GateConservationCheck(&cache,&fixture,"D/admit",root * 4u);
		if ( status == SPARK_STATUS_OK )
		{
			GateBindWitnesses(&cache,root % 3u,
				2u * G1_BLOCK_TOKENS);
			GateConservationCheck(&cache,&fixture,"D/witness",
				root * 4u + 1u);
			step = root;
			assert(SparkGlm52PagedKvCover(&cache,root % 3u,
				2u * G1_BLOCK_TOKENS + 1u,&step,1u) ==
				SPARK_STATUS_OK ||
				SparkGlm52PagedKvFreeBlocks(&cache) == 0u);
			GateConservationCheck(&cache,&fixture,"D/decode",
				root * 4u + 2u);
		}
	}
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats);
	assert(stats.evicted_block_count > 0u);
	for (step=0u; step<3u; step++)
		SparkGlm52PagedKvLaneReset(&cache,step);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,12u,&evicted) ==
			SPARK_STATUS_OK);
	}
	assert(SparkGlm52PagedKvFreeBlocks(&cache) == 12u);
	GateConservationCheck(&cache,&fixture,"D/teardown",9999u);
	SparkGlm52PagedKvDestroy(&cache);
	printf("glm52 prefix-cache CASE D squeeze PASS\n");
}

int main(void)
{
	SparkGlm52PagedKv cache;
	SparkGlm52PagedKvConfiguration configuration;

	/* CASE A1: the engine refuses a lane-table row wider than THIS
	 * port's geometry-callback ceiling (the width is a per-stage fact).
	 * Validated before any state is built, so no rows are needed. */
	GateConfiguration(&configuration,1u,
		SPARK_GLM52_PAGED_KV_MAX_BLOCKS_PER_LANE + 1u,4u,2u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_INVALID_ARGUMENT);

	/* CASE A2: zero capacities refuse too. */
	GateConfiguration(&configuration,1u,G1_BLOCKS_PER_LANE,0u,2u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_INVALID_ARGUMENT);

	/* CASE A3: the driver-true geometry accepts and the port's private
	 * sequence-id range landed in the engine (base + lane). */
	GateConfiguration(&configuration,G1_LANES,G1_BLOCKS_PER_LANE,
		G1_POOL,4u);
	assert(SparkGlm52PagedKvInitialize(&cache,&configuration,g1_table,
		g1_counts) == SPARK_STATUS_OK);
	assert(cache.core.block_stride_bytes == (uint64_t)G1_BLOCK_TOKENS *
		(SPARK_GLM52_MODEL_KV_A_DIMENSION +
			SPARK_GLM52_MODEL_ROPE_DIMENSION) * 2u *
		SPARK_GLM52_MODEL_LAYER_COUNT);
	assert(cache.geometry->sequence_id_base ==
		SPARK_GLM52_PAGED_KV_SEQUENCE_BASE);
	SparkGlm52PagedKvDestroy(&cache);
	printf("glm52 prefix-cache CASE A geometry PASS\n");

	GateCaseSharing();
	GateCaseSpeculative();
	GateCaseSqueeze();
	printf("glm52 prefix-cache gate PASS\n");
	return(0);
}
