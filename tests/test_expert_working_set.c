#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_expert_working_set.h"
#include "sparkpipe/spark_step_verdict.h"

#define CHECK(condition) do { if ( !(condition) ) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#condition); exit(1); } } while (0)

#define LAYERS 5u
#define EXPERTS 40u
#define STRIDE 64u
#define RING 8u

typedef struct Pager
{
	uint8_t held[LAYERS * STRIDE];
	uint32_t calls;
	uint32_t fail_next;
	const SparkExpertWorkingSet *set;
} Pager;

static SparkStatus PagerAcquire(void *context,const uint32_t *keys,uint32_t count)
{
	Pager *pager = (Pager *)context;
	uint32_t index;
	pager->calls++;
	if ( pager->fail_next != 0u )
	{
		pager->fail_next = 0u;
		return(SPARK_STATUS_BUSY);
	}
	for (index=0u; index<count; index++)
	{
		CHECK(keys[index] < LAYERS * STRIDE);
		CHECK(index == 0u || keys[index - 1u] < keys[index]);
		CHECK(pager->held[keys[index]] == 0u);
		CHECK(SparkExpertWorkingSetCovered(pager->set,keys[index] / STRIDE,keys[index] % STRIDE) == 0u);
		pager->held[keys[index]] = 1u;
	}
	return(SPARK_STATUS_OK);
}

static void CheckCoverHeld(const SparkExpertWorkingSet *set,const Pager *pager)
{
	uint32_t layer,expert,count = 0u;
	for (layer=0u; layer<LAYERS; layer++)
		for (expert=0u; expert<EXPERTS; expert++)
		{
			CHECK(SparkExpertWorkingSetCovered(set,layer,expert) == pager->held[layer * STRIDE + expert]);
			count += pager->held[layer * STRIDE + expert];
		}
	CHECK(count == set->key_count);
}

static void Ring(uint32_t *miss,const uint32_t *entries,uint32_t recorded)
{
	uint32_t index;
	memset(miss,0,(SPARK_STEP_MISS_ENTRIES + RING) * sizeof(uint32_t));
	miss[SPARK_STEP_MISS_FLAG] = recorded != 0u ? 1u : 0u;
	miss[SPARK_STEP_MISS_COUNT] = recorded;
	for (index=0u; index<recorded && index<RING; index++)
		miss[SPARK_STEP_MISS_ENTRIES + index] = entries[index];
}

static void TestCreateAndAdd(void)
{
	SparkExpertWorkingSet set;
	Pager pager;
	uint32_t keys[4] = {1u * STRIDE + 33u,0u * STRIDE + 2u,1u * STRIDE + 33u,4u * STRIDE + 39u},missing = 0u,bad = 2u * STRIDE + EXPERTS;
	memset(&pager,0,sizeof(pager));
	pager.set = &set;
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,EXPERTS,8u,PagerAcquire,&pager) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,STRIDE,8u,0,&pager) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,STRIDE,LAYERS * EXPERTS + 1u,PagerAcquire,&pager) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,STRIDE,5u,PagerAcquire,&pager) == SPARK_STATUS_OK);
	CHECK(set.stride == 2u && SparkExpertWorkingSetCoverBytes(&set) == LAYERS * 2u * sizeof(uint32_t));
	CHECK(SparkExpertWorkingSetAdd(&set,&bad,1u) == SPARK_STATUS_INVALID_ARGUMENT && pager.calls == 0u);
	CHECK(SparkExpertWorkingSetAdd(&set,keys,4u) == SPARK_STATUS_OK);
	CHECK(set.key_count == 3u && set.generation == 1u && pager.calls == 1u);
	CHECK(set.cover[1u * 2u + 1u] == (UINT32_C(1) << 1u));
	CheckCoverHeld(&set,&pager);
	CHECK(SparkExpertWorkingSetAdd(&set,keys,4u) == SPARK_STATUS_OK && pager.calls == 1u && set.generation == 1u);
	pager.fail_next = 1u;
	keys[0] = 2u * STRIDE + 5u;
	CHECK(SparkExpertWorkingSetAdd(&set,keys,1u) == SPARK_STATUS_BUSY && set.grow_denied == 1u);
	CheckCoverHeld(&set,&pager);
	keys[1] = 2u * STRIDE + 6u;
	keys[2] = 2u * STRIDE + 7u;
	CHECK(SparkExpertWorkingSetAdd(&set,keys,3u) == SPARK_STATUS_CAPACITY_EXCEEDED && set.grow_denied == 2u && set.key_count == 3u);
	CheckCoverHeld(&set,&pager);
	CHECK(SparkExpertWorkingSetCheckAnchors(&set,0u,LAYERS,&missing) == SPARK_STATUS_UNSUPPORTED && missing == 2u);
	CHECK(SparkExpertWorkingSetCheckAnchors(&set,0u,2u,&missing) == SPARK_STATUS_OK && missing == UINT32_MAX);
	CHECK(SparkExpertWorkingSetCheckAnchors(&set,4u,2u,&missing) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkExpertWorkingSetDestroy(&set);
	printf("ok add is atomic: invalid, denied and over-cap growth leave the cover unchanged; bits are set only after acquire\n");
}

static void TestHarvest(void)
{
	SparkExpertWorkingSet set;
	SparkExpertMissHarvest harvest;
	Pager pager;
	uint32_t miss[SPARK_STEP_MISS_ENTRIES + RING],keys[RING];
	uint32_t in_order[5] = {1u * STRIDE + 9u,1u * STRIDE + 3u,1u * STRIDE + 9u,3u * STRIDE + 1u,3u * STRIDE + EXPERTS};
	uint32_t overflow_complete[10] = {2u * STRIDE + 1u,2u * STRIDE + 2u,4u * STRIDE + 3u,4u * STRIDE + 4u,4u * STRIDE + 5u,4u * STRIDE + 6u,4u * STRIDE + 7u,4u * STRIDE + 8u,4u * STRIDE + 9u,4u * STRIDE + 10u};
	uint32_t overflow_partial[10] = {2u * STRIDE + 1u,2u * STRIDE + 2u,2u * STRIDE + 3u,2u * STRIDE + 4u,2u * STRIDE + 5u,2u * STRIDE + 6u,2u * STRIDE + 7u,2u * STRIDE + 8u,2u * STRIDE + 9u,2u * STRIDE + 10u};
	uint32_t garbage_first[2] = {1u * STRIDE + EXPERTS,1u * STRIDE + 2u};
	uint32_t bad_layer[1] = {LAYERS * STRIDE};
	memset(&pager,0,sizeof(pager));
	pager.set = &set;
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,STRIDE,LAYERS * EXPERTS,PagerAcquire,&pager) == SPARK_STATUS_OK);
	Ring(miss,in_order,0u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_OK && harvest.key_count == 0u && harvest.first_layer == UINT32_MAX);
	Ring(miss,in_order,5u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_OK);
	CHECK(harvest.first_layer == 1u && harvest.key_count == 2u && keys[0] == 1u * STRIDE + 3u && keys[1] == 1u * STRIDE + 9u);
	Ring(miss,overflow_complete,10u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_OK);
	CHECK(harvest.first_layer == 2u && harvest.key_count == 2u && harvest.recorded == 10u && harvest.entries == RING && set.harvest_overflow == 1u);
	Ring(miss,overflow_partial,10u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_CAPACITY_EXCEEDED && set.harvest_overflow == 2u);
	Ring(miss,garbage_first,2u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_VALIDATION_FAILED);
	Ring(miss,bad_layer,1u);
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_VALIDATION_FAILED);
	Ring(miss,in_order,5u);
	miss[SPARK_STEP_MISS_FLAG] = 0u;
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_VALIDATION_FAILED);
	Ring(miss,in_order,0u);
	miss[SPARK_STEP_MISS_FLAG] = 1u;
	CHECK(SparkExpertWorkingSetHarvest(&set,miss,RING,keys,RING,&harvest) == SPARK_STATUS_VALIDATION_FAILED);
	SparkExpertWorkingSetDestroy(&set);
	printf("ok harvest acts on the first miss layer only, accepts overflow past it, refuses an incomplete first layer and corrupt rings\n");
}

static void TestRandomGrowth(void)
{
	SparkExpertWorkingSet set;
	Pager pager;
	uint32_t round,keys[16],count,index;
	uint64_t seed = UINT64_C(0x1234567);
	memset(&pager,0,sizeof(pager));
	pager.set = &set;
	CHECK(SparkExpertWorkingSetCreate(&set,LAYERS,EXPERTS,STRIDE,120u,PagerAcquire,&pager) == SPARK_STATUS_OK);
	for (round=0u; round<10000u; round++)
	{
		SparkStatus status;
		uint32_t before = set.key_count;
		seed = seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
		count = (uint32_t)(seed >> 60u) + 1u;
		for (index=0u; index<count; index++)
		{
			seed = seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
			keys[index] = (uint32_t)((seed >> 33u) % LAYERS) * STRIDE + (uint32_t)((seed >> 17u) % EXPERTS);
		}
		pager.fail_next = (round % 97u) == 0u ? 1u : 0u;
		status = SparkExpertWorkingSetAdd(&set,keys,count);
		CHECK(status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY || status == SPARK_STATUS_CAPACITY_EXCEEDED);
		CHECK(status == SPARK_STATUS_OK ? set.key_count >= before : set.key_count == before);
		CHECK(set.key_count <= set.cap_keys);
		pager.fail_next = 0u;
		CheckCoverHeld(&set,&pager);
	}
	CHECK(set.key_count > 100u && set.grow_denied != 0u);
	SparkExpertWorkingSetDestroy(&set);
	printf("ok 10000 random growth rounds keep cover == held and never exceed the cap\n");
}

static void TestReplayPolicy(void)
{
	SparkStepReplay replay;
	memset(&replay,0,sizeof(replay));
	replay.limit = 2u;
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_ROLLBACK_LOCAL) == SPARK_STEP_ACTION_REPLAY);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_ROLLBACK_REMOTE) == SPARK_STEP_ACTION_REPLAY);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_ROLLBACK_LOCAL) == SPARK_STEP_ACTION_EXHAUSTED);
	CHECK(replay.attempts == 0u && replay.exhausted == 1u && replay.local == 2u && replay.remote == 1u && replay.replays == 2u);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_ROLLBACK_REMOTE) == SPARK_STEP_ACTION_REPLAY);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_COMMIT) == SPARK_STEP_ACTION_COMMIT && replay.attempts == 0u && replay.commits == 1u);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_MIXED) == SPARK_STEP_ACTION_FAIL);
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_POISON_LOST) == SPARK_STEP_ACTION_FAIL);
	replay.limit = 0u;
	CHECK(SparkStepReplayNext(&replay,SPARK_STEP_VERDICT_ROLLBACK_LOCAL) == SPARK_STEP_ACTION_EXHAUSTED);
	printf("ok replay policy: bounded replays, then exhausted; mixed and lost poison fail\n");
}

int main(void)
{
	TestCreateAndAdd();
	TestHarvest();
	TestRandomGrowth();
	TestReplayPolicy();
	printf("PASS expert working set\n");
	return(0);
}
