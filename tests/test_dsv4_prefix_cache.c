#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

/* Compiled in for the conservation cases: the gate drives the DSV4
 * paged-KV port directly, at the adapter's per-round call pattern, and
 * asserts free-list conservation after EVERY step (the qwen36/qwen38
 * audit lessons 7+8). This is the unit half of the per-driver
 * prefix-cache gate family; the serving-boundary half lands with the
 * adapter lifecycle wiring, exactly as the qwen ports grew theirs.
 *
 * Relationship to SparkDsv4PagedCache (R2-C3): that substrate keeps its
 * lazy page-RESIDENCY state machine over SparkKvPageCache; this gate
 * proves the REUSE half this port binds - content-addressed publishing,
 * LCP matching, witnessed sharing, LRU eviction - over the stage's real
 * pool geometry. */
#include "spark_dsv4_paged_kv.h"

/* Driver-true pool geometry: the page stride and 128-token blocks come
 * from the stage's own pool-layout builder, never a synthetic constant. */
#include "spark_dsv4_pool_layout.h"

#define V1_BLOCK_TOKENS SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS
#define V1_BLOCKS_PER_LANE 16u
#define V1_LANES 4u
#define V1_POOL 96u
#define V1_ROUNDS 140u
#define V1_DRAFT 8u
#define V1_PROMPT_TOKENS (V1_BLOCK_TOKENS * 3u)

static uint32_t v1_table[V1_LANES * V1_BLOCKS_PER_LANE];
static uint32_t v1_counts[V1_LANES];
/* Filled from SparkDsv4PagedPoolBuildLayout in main. */
static uint64_t v1_page_stride_bytes;

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
static uint32_t GatePartitionAudit(const SparkDsv4PagedKv *cache,
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
				assert(block == SPARK_DSV4_PAGED_KV_NO_BLOCK);
				continue;
			}
			assert(block != SPARK_DSV4_PAGED_KV_NO_BLOCK &&
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

static void GateConservationCheck(const SparkDsv4PagedKv *cache,
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

static void GateConfiguration(SparkDsv4PagedKvConfiguration *configuration,
	uint32_t lane_count,uint32_t blocks_per_lane,uint32_t pool,
	uint32_t checkpoint_slots)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->block_token_count = V1_BLOCK_TOKENS;
	configuration->lane_count = lane_count;
	configuration->blocks_per_lane = blocks_per_lane;
	configuration->physical_page_capacity = pool;
	configuration->logical_page_capacity = pool;
	configuration->checkpoint_slot_count = checkpoint_slots;
	configuration->block_stride_bytes = v1_page_stride_bytes;
}

static void GateFill(uint32_t *tokens,uint32_t base,uint32_t count)
{
	uint32_t i;
	for (i=0u; i<count; i++)
		tokens[i] = base + i * 7u + 1u;
}

/* Mirror the adapter's post-frame checkpoint binding. */
static void GateBindWitnesses(SparkDsv4PagedKv *cache,uint32_t lane,
	uint64_t end_position)
{
	uint64_t boundary;
	uint32_t slot;
	if ( cache->reuse_enabled == 0u )
		return;
	for (boundary = V1_BLOCK_TOKENS; boundary <= end_position;
		boundary += V1_BLOCK_TOKENS)
		if ( SparkDsv4PagedKvCheckpointOffer(cache,lane,boundary,&slot)
			!= 0u )
			SparkDsv4PagedKvCheckpointCommit(cache,lane,slot,boundary);
}

/* CASE B: witnessed sharing, divergence, and mid-block admits. */
static void GateCaseSharing(void)
{
	SparkDsv4PagedKv cache;
	SparkDsv4PagedKvConfiguration configuration;
	SparkDsv4PagedKvMatch match;
	uint32_t root[3u * V1_BLOCK_TOKENS];
	uint32_t divergent[3u * V1_BLOCK_TOKENS];
	uint32_t inner[2u * V1_BLOCK_TOKENS];
	uint32_t sampled;
	uint32_t i;
	GateFixture fixture;

	GateFixtureInit(&fixture,v1_table,v1_counts,V1_LANES,
		V1_BLOCKS_PER_LANE,V1_POOL);
	GateConfiguration(&configuration,V1_LANES,V1_BLOCKS_PER_LANE,
		V1_POOL,4u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_OK);

	GateFill(root,1000u,V1_PROMPT_TOKENS);
	memcpy(divergent,root,sizeof(uint32_t) * 2u * V1_BLOCK_TOKENS);
	for (i = 2u * V1_BLOCK_TOKENS; i < V1_PROMPT_TOKENS; i++)
		divergent[i] = 9000u + i;
	memcpy(inner,root,sizeof(uint32_t) * (V1_BLOCK_TOKENS / 2u));
	for (i = V1_BLOCK_TOKENS / 2u; i < 2u * V1_BLOCK_TOKENS; i++)
		inner[i] = 8000u + i;

	assert(SparkDsv4PagedKvAdmit(&cache,0u,root,V1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 0u && match.checkpoint_slot ==
		SPARK_DSV4_PAGED_KV_NO_SLOT);
	assert(SparkDsv4PagedKvCommittedTokens(&cache,0u) ==
		V1_PROMPT_TOKENS);
	GateConservationCheck(&cache,&fixture,"B/donor-admit",0u);
	GateBindWitnesses(&cache,0u,V1_PROMPT_TOKENS);

	sampled = 424242u;
	assert(SparkDsv4PagedKvCover(&cache,0u,V1_PROMPT_TOKENS + 1u,
		&sampled,1u) == SPARK_STATUS_OK);
	GateConservationCheck(&cache,&fixture,"B/donor-decode",1u);

	assert(SparkDsv4PagedKvAdmit(&cache,1u,root,V1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 3u);
	assert(match.checkpoint_slot != SPARK_DSV4_PAGED_KV_NO_SLOT);
	for (i=0u; i<3u; i++)
		assert(v1_table[1u * V1_BLOCKS_PER_LANE + i] ==
			v1_table[0u * V1_BLOCKS_PER_LANE + i]);
	assert(SparkDsv4PagedKvCommittedTokens(&cache,1u) ==
		V1_PROMPT_TOKENS);
	GateConservationCheck(&cache,&fixture,"B/identical",2u);

	assert(SparkDsv4PagedKvAdmit(&cache,2u,divergent,V1_PROMPT_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 2u);
	for (i=0u; i<2u; i++)
		assert(v1_table[2u * V1_BLOCKS_PER_LANE + i] ==
			v1_table[0u * V1_BLOCKS_PER_LANE + i]);
	assert(v1_table[2u * V1_BLOCKS_PER_LANE + 2u] !=
		v1_table[0u * V1_BLOCKS_PER_LANE + 2u]);
	GateConservationCheck(&cache,&fixture,"B/divergent",3u);

	assert(SparkDsv4PagedKvAdmit(&cache,3u,inner,2u * V1_BLOCK_TOKENS,
		&match) == SPARK_STATUS_OK);
	assert(match.block_count == 0u);
	GateConservationCheck(&cache,&fixture,"B/inner",4u);

	assert(cache.core.admit_count >= 4u);

	/* Teardown conserves: released sequences leave their published
	 * blocks cached for later matches (the reuse point); one Trim
	 * recovers every detached block and returns the WHOLE pool. */
	for (i=0u; i<V1_LANES; i++)
		SparkDsv4PagedKvLaneReset(&cache,i);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,V1_POOL,
			&evicted) == SPARK_STATUS_OK);
		assert(evicted > 0u); /* sharing kept blocks hot */
	}
	assert(SparkDsv4PagedKvFreeBlocks(&cache) == V1_POOL);
	GateConservationCheck(&cache,&fixture,"B/teardown",5u);
	SparkDsv4PagedKvDestroy(&cache);
	printf("dsv4 prefix-cache CASE B sharing PASS\n");
}

/* CASE C: speculative rounds, reuse ON and OFF. */
static void GateRunRounds(SparkDsv4PagedKv *cache,GateFixture *fixture,
	uint32_t lane,const char *where)
{
	uint32_t round,pos,tok;
	pos = V1_BLOCK_TOKENS;
	for (round=0u; round<V1_ROUNDS; round++)
	{
		tok = 500000u + round * 13u + lane * 7u;
		assert(SparkDsv4PagedKvCover(cache,lane,(uint64_t)pos + 1u,
			&tok,1u) == SPARK_STATUS_OK);
		GateConservationCheck(cache,fixture,where,round * 2u);
		assert(SparkDsv4PagedKvCover(cache,lane,
			(uint64_t)pos + 1u + V1_DRAFT + 1u,0,0) ==
			SPARK_STATUS_OK);
		GateConservationCheck(cache,fixture,where,round * 2u + 1u);
		pos += 1u;
		if ( (pos + V1_DRAFT + 1u) / V1_BLOCK_TOKENS >=
			fixture->blocks_per_lane - 2u )
		{
			SparkDsv4PagedKvLaneReset(cache,lane);
			GateConservationCheck(cache,fixture,where,1000000u +
				round);
			tok = 700000u + round;
			assert(SparkDsv4PagedKvAdmit(cache,lane,&tok,1u,0) ==
				SPARK_STATUS_OK);
			GateConservationCheck(cache,fixture,where,2000000u +
				round);
			pos = V1_BLOCK_TOKENS;
			{
				uint32_t filler;
				for (filler=pos; filler<V1_BLOCK_TOKENS; filler++)
				{
					uint32_t id = tok + filler;
					assert(SparkDsv4PagedKvCover(cache,lane,
						(uint64_t)filler + 1u,&id,1u) ==
						SPARK_STATUS_OK);
				}
			}
		}
	}
}

static void GateCaseSpeculative(void)
{
	SparkDsv4PagedKv cache;
	SparkDsv4PagedKvConfiguration configuration;
	GateFixture fixture;
	uint32_t root,i;

	GateFixtureInit(&fixture,v1_table,v1_counts,2u,
		V1_BLOCKS_PER_LANE,48u);
	GateConfiguration(&configuration,2u,V1_BLOCKS_PER_LANE,48u,2u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_OK);
	root = 31337u;
	assert(SparkDsv4PagedKvAdmit(&cache,0u,&root,1u,0) ==
		SPARK_STATUS_OK);
	assert(SparkDsv4PagedKvAdmit(&cache,1u,&root,1u,0) ==
		SPARK_STATUS_OK);
	GateRunRounds(&cache,&fixture,0u,"C/on-lane0");
	GateRunRounds(&cache,&fixture,1u,"C/on-lane1");
	for (i=0u; i<2u; i++)
		SparkDsv4PagedKvLaneReset(&cache,i);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,48u,&evicted) ==
			SPARK_STATUS_OK);
	}
	assert(SparkDsv4PagedKvFreeBlocks(&cache) == 48u);
	SparkDsv4PagedKvDestroy(&cache);
	printf("dsv4 prefix-cache CASE C speculative ON PASS\n");

	GateFixtureInit(&fixture,v1_table,v1_counts,2u,
		V1_BLOCKS_PER_LANE,48u);
	GateConfiguration(&configuration,2u,V1_BLOCKS_PER_LANE,48u,0u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_OK);
	assert(SparkDsv4PagedKvAdmit(&cache,0u,&root,1u,0) ==
		SPARK_STATUS_OK);
	assert(SparkDsv4PagedKvAdmit(&cache,1u,&root,1u,0) ==
		SPARK_STATUS_OK);
	GateRunRounds(&cache,&fixture,0u,"C/off-lane0");
	GateRunRounds(&cache,&fixture,1u,"C/off-lane1");
	for (i=0u; i<2u; i++)
		SparkDsv4PagedKvLaneReset(&cache,i);
	assert(SparkDsv4PagedKvFreeBlocks(&cache) == 48u);
	SparkDsv4PagedKvDestroy(&cache);
	printf("dsv4 prefix-cache CASE C speculative OFF PASS\n");
}

/* CASE D: squeezed pool under multi-root churn. */
static void GateCaseSqueeze(void)
{
	SparkDsv4PagedKv cache;
	SparkDsv4PagedKvConfiguration configuration;
	SparkDsv4PagedKvMatch match;
	SparkPrefixCacheCoreStats stats;
	GateFixture fixture;
	uint32_t tokens[2u * V1_BLOCK_TOKENS];
	uint32_t root,step,status;

	GateFixtureInit(&fixture,v1_table,v1_counts,3u,8u,12u);
	GateConfiguration(&configuration,3u,8u,12u,2u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_OK);
	for (root=0u; root<6u; root++)
	{
		GateFill(tokens,root * 100000u + 31u,2u * V1_BLOCK_TOKENS);
		status = SparkDsv4PagedKvAdmit(&cache,root % 3u,tokens,
			2u * V1_BLOCK_TOKENS,&match);
		assert(status == SPARK_STATUS_OK ||
			status == SPARK_STATUS_CAPACITY_EXCEEDED);
		GateConservationCheck(&cache,&fixture,"D/admit",root * 4u);
		if ( status == SPARK_STATUS_OK )
		{
			GateBindWitnesses(&cache,root % 3u,
				2u * V1_BLOCK_TOKENS);
			GateConservationCheck(&cache,&fixture,"D/witness",
				root * 4u + 1u);
			step = root;
			assert(SparkDsv4PagedKvCover(&cache,root % 3u,
				2u * V1_BLOCK_TOKENS + 1u,&step,1u) ==
				SPARK_STATUS_OK ||
				SparkDsv4PagedKvFreeBlocks(&cache) == 0u);
			GateConservationCheck(&cache,&fixture,"D/decode",
				root * 4u + 2u);
		}
	}
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats);
	assert(stats.evicted_block_count > 0u);
	for (step=0u; step<3u; step++)
		SparkDsv4PagedKvLaneReset(&cache,step);
	{
		uint32_t evicted;
		assert(SparkPrefixCacheCoreTrim(&cache.core,12u,&evicted) ==
			SPARK_STATUS_OK);
	}
	assert(SparkDsv4PagedKvFreeBlocks(&cache) == 12u);
	GateConservationCheck(&cache,&fixture,"D/teardown",9999u);
	SparkDsv4PagedKvDestroy(&cache);
	printf("dsv4 prefix-cache CASE D squeeze PASS\n");
}

int main(void)
{
	SparkDsv4PagedKv cache;
	SparkDsv4PagedKvConfiguration configuration;
	SparkDsv4PagedPoolLayout layout;

	/* The stage's own builder prices the pool: 128-token pages whose
	 * stride covers every layer region the Flash stage owns. */
	assert(SparkDsv4PagedPoolBuildLayout(0u,
		SPARK_DSV4_MODEL_LAYER_COUNT,&layout) == 0);
	assert(layout.block_token_count == V1_BLOCK_TOKENS);
	v1_page_stride_bytes = layout.page_stride_bytes;
	assert(v1_page_stride_bytes > 0u);

	GateConfiguration(&configuration,1u,
		SPARK_DSV4_PAGED_KV_MAX_BLOCKS_PER_LANE + 1u,4u,2u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_INVALID_ARGUMENT);

	GateConfiguration(&configuration,1u,V1_BLOCKS_PER_LANE,0u,2u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_INVALID_ARGUMENT);

	GateConfiguration(&configuration,V1_LANES,V1_BLOCKS_PER_LANE,
		V1_POOL,4u);
	assert(SparkDsv4PagedKvInitialize(&cache,&configuration,v1_table,
		v1_counts) == SPARK_STATUS_OK);
	assert(cache.core.block_stride_bytes == v1_page_stride_bytes);
	assert(cache.geometry->sequence_id_base ==
		SPARK_DSV4_PAGED_KV_SEQUENCE_BASE);
	SparkDsv4PagedKvDestroy(&cache);
	printf("dsv4 prefix-cache CASE A geometry PASS\n");

	GateCaseSharing();
	GateCaseSpeculative();
	GateCaseSqueeze();
	printf("dsv4 prefix-cache gate PASS\n");
	return(0);
}
