#include "sparkpipe/spark_speculation_frame.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void TestBuckets(void)
{
	assert(SparkSpeculationFrameClass(3u,0u) == 0u && SparkSpeculationFrameClass(4u,0u) == 1u && SparkSpeculationFrameClass(7u,0u) == 1u);
	assert(SparkSpeculationFrameClass(8u,0u) == 2u && SparkSpeculationFrameClass(31u,0u) == 3u);
	assert(SparkSpeculationFrameClass(63u,0u) == 4u && SparkSpeculationFrameClass(64u,0u) == 5u);
	assert(SparkSpeculationFrameClass(64u,1u) == 2u * SPARK_SPECULATION_FRAME_MATCH_CLASSES - 1u && SparkSpeculationFrameClass(3u,1u) == SPARK_SPECULATION_FRAME_MATCH_CLASSES);
	assert(SparkSpeculationFrameModelClass(0u) == 2u * SPARK_SPECULATION_FRAME_MATCH_CLASSES && SparkSpeculationFrameModelClass(1u) == SPARK_SPECULATION_FRAME_CLASSES - 1u);
	assert(SparkSpeculationFrameClassIsModel(SparkSpeculationFrameModelClass(0u)) != 0u && SparkSpeculationFrameClassIsModel(SparkSpeculationFrameModelClass(1u)) != 0u);
	assert(SparkSpeculationFrameClassIsModel(SparkSpeculationFrameClass(64u,1u)) == 0u && SparkSpeculationFrameClassIsModel(SPARK_SPECULATION_FRAME_CLASSES) == 0u);
	assert(SparkSpeculationFrameDepthBucket(1u) == 0u && SparkSpeculationFrameDepthBucket(2u) == 1u);
	assert(SparkSpeculationFrameDepthBucket(3u) == 2u && SparkSpeculationFrameDepthBucket(4u) == 2u && SparkSpeculationFrameDepthBucket(5u) == 3u);
	assert(SparkSpeculationFrameDepthBucket(128u) == 7u && SparkSpeculationFrameDepthBucket(129u) == 8u && SparkSpeculationFrameDepthBucket(255u) == 8u);
	assert(SparkSpeculationFrameRowBucket(1u) == 0u && SparkSpeculationFrameRowBucket(8u) == 7u && SparkSpeculationFrameRowBucket(9u) == 8u && SparkSpeculationFrameRowBucket(16u) == 15u);
	assert(SparkSpeculationFrameRowBucket(17u) == 16u && SparkSpeculationFrameRowBucket(32u) == 16u && SparkSpeculationFrameRowBucket(256u) == 19u && SparkSpeculationFrameRowBucket(1024u) == 21u);
	assert(SparkSpeculationFrameRowBucketLimit(7u) == 8u && SparkSpeculationFrameRowBucketLimit(16u) == 32u && SparkSpeculationFrameRowBucketLimit(19u) == 256u && SparkSpeculationFrameRowBucketLimit(21u) == 1024u);
	assert(SparkSpeculationFrameContextBucket(1u) == 0u && SparkSpeculationFrameContextBucket(2048u) == 0u && SparkSpeculationFrameContextBucket(4096u) == 1u);
	assert(SparkSpeculationFrameContextBucket(1u << 30) == SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS - 1u);
}

static void TestHazard(void)
{
	SparkSpeculationHazard hazard;
	float expected = 1.0f,survival = 1.0f;
	uint32_t index;
	memset(&hazard,0,sizeof(hazard));
	for (index=1u; index<=8u; index++)
	{
		survival *= 1.0f - SPARK_SPECULATION_FRAME_HAZARD_PRIOR;
		expected += survival;
	}
	assert(fabsf(SparkSpeculationHazardAt(&hazard,0,0u) - SPARK_SPECULATION_FRAME_HAZARD_PRIOR) < 1e-6f);
	assert(fabsf(SparkSpeculationFrameExpected(&hazard,0,8u) - expected) < 1e-4f);
	assert(SparkSpeculationFrameExpected(&hazard,0,0u) == 1.0f);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&hazard,255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	assert(hazard.frames == 20u && hazard.events[8] == 0.0f && hazard.risk[8] > 100.0f);
	assert(SparkSpeculationHazardAt(&hazard,0,8u) < 0.001f);
	assert(SparkSpeculationFrameExpected(&hazard,0,255u) > 200.0f);
	memset(&hazard,0,sizeof(hazard));
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&hazard,8u,1u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	assert(hazard.events[SparkSpeculationFrameDepthBucket(2u)] > 10.0f && hazard.risk[SparkSpeculationFrameDepthBucket(3u)] == 0.0f);
	assert(SparkSpeculationHazardAt(&hazard,0,1u) > 0.9f);
	assert(SparkSpeculationFrameExpected(&hazard,0,8u) < 2.2f);
}

static void TestRequestBlend(void)
{
	SparkSpeculationHazard engine,request;
	uint32_t index;
	memset(&engine,0,sizeof(engine));
	memset(&request,0,sizeof(request));
	for (index=0u; index<50u; index++)
		SparkSpeculationHazardObserve(&engine,64u,64u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	for (index=0u; index<30u; index++)
		SparkSpeculationHazardObserve(&request,8u,0u,SPARK_SPECULATION_FRAME_REQUEST_DECAY);
	assert(SparkSpeculationFrameExpected(&engine,&request,8u) < SparkSpeculationFrameExpected(&engine,0,8u));
	assert(SparkSpeculationFrameExpected(&engine,&request,8u) < 3.5f && SparkSpeculationFrameExpected(&engine,0,8u) > 8.5f);
}

static void TestCost(void)
{
	SparkSpeculationFrameModel model;
	float mid,far;
	memset(&model,0,sizeof(model));
	assert(SparkSpeculationFrameCost(&model,8u,2048u) == 0.0f);
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 35000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,2048u,57000000ull);
	assert(fabsf(SparkSpeculationFrameTokenCost(&model,2048u) - 35e6f) < 1.0f && fabsf(SparkSpeculationFrameSingleCost(&model,2048u) - 57e6f) < 1.0f);
	assert(fabsf(SparkSpeculationFrameTokenCost(&model,200000u) - 35e6f) < 1.0f);
	assert(fabsf(SparkSpeculationFrameCost(&model,9u,2048u) - 57e6f * 2.0f) < 1.0f);
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,67000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,234000000ull);
	mid = SparkSpeculationFrameCost(&model,16u,2048u);
	far = SparkSpeculationFrameCost(&model,64u,2048u);
	assert(mid > 67e6f && mid < 234e6f);
	assert(fabsf(mid - (67e6f + (234e6f - 67e6f) * 14.0f / 31.0f)) < 2.0f);
	assert(far > 234e6f);
	assert(SparkSpeculationFrameCost(&model,16u,40000u) == mid);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,300000000ull);
	assert(SparkSpeculationFrameCost(&model,33u,2048u) > 234e6f && SparkSpeculationFrameCost(&model,33u,2048u) < 300e6f);
}

static void TestChoose(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice,again;
	SparkSpeculationHazard copy;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	request.feasible = 255u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 0u);
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 35000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,2048u,57000000ull);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 1u && choice.probe == 1u);
	request.feasible = 5u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 1u && choice.probe == 1u);
	request.feasible = 255u;
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,67000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,234000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,308000000ull);
	SparkSpeculationFrameObserveFrame(&model,256u,2048u,800000000ull);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts != 0u && choice.drafts <= 15u && choice.probe == 0u && choice.frame_ns < 234000000.0f);
	request.cost_cap_ns = 130000000.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts != 0u && choice.frame_ns <= 130000000.0f);
	request.cost_cap_ns = 60000000.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 0u);
	request.cost_cap_ns = 0.0f;
	SparkSpeculationFrameObserveFrame(&model,2u,16384u,36000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,16384u,46000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,16384u,54000000ull);
	SparkSpeculationFrameObserveFrame(&model,256u,16384u,68000000ull);
	request.context = 16384u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 255u && choice.probe == 0u && choice.expected_tokens > 200.0f);
	request.feasible = 100u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 95u);
	request.context = 2048u;
	request.feasible = 255u;
	request.match_length = 3u;
	for (index=0u; index<40u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(3u,0u)],8u,0u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 0u);
	memset(&copy,0,sizeof(copy));
	for (index=0u; index<6u; index++)
		SparkSpeculationHazardObserve(&copy,64u,64u,SPARK_SPECULATION_FRAME_REQUEST_DECAY);
	SparkSpeculationFrameChoose(&model,&copy,&request,&again);
	assert(again.drafts >= choice.drafts);
	request.match_length = 64u;
	request.sampled = 1u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.probe == 1u);
}

static void TestProbe(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice;
	SparkSpeculationHazard fresh,other;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	memset(&fresh,0,sizeof(fresh));
	memset(&other,0,sizeof(other));
	request.feasible = 255u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 35000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,2048u,57000000ull);
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,36000000ull);
	SparkSpeculationFrameObserveFrame(&model,9u,2048u,36500000ull);
	SparkSpeculationFrameObserveFrame(&model,17u,2048u,40000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,52000000ull);
	SparkSpeculationFrameObserveFrame(&model,256u,2048u,68000000ull);
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 8u && choice.probe == 1u && choice.frame_ns <= SPARK_SPECULATION_FRAME_MARGIN * 35e6f);
	request.feasible = 5u;
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 5u && choice.probe == 1u);
	request.feasible = 255u;
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 255u && choice.probe == 0u);
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 8u && choice.probe == 1u);
	SparkSpeculationHazardObserve(&fresh,8u,0u,SPARK_SPECULATION_FRAME_REQUEST_DECAY);
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 8u && choice.probe == 1u);
	SparkSpeculationHazardObserve(&fresh,8u,3u,SPARK_SPECULATION_FRAME_REQUEST_DECAY);
	request.proven_drafts = 3u;
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 11u && choice.drafts <= SPARK_SPECULATION_FRAME_GROWTH * 3u && choice.probe == 0u);
	request.match_length = 32u;
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(32u,0u)],255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,&other,&request,&choice);
	assert(choice.drafts != 0u && choice.drafts <= SPARK_SPECULATION_FRAME_GROWTH * 3u && choice.probe == 0u);
	request.match_length = 64u;
	request.proven_drafts = 16u;
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 63u && choice.drafts <= SPARK_SPECULATION_FRAME_GROWTH * 16u && choice.probe == 0u);
	request.proven_drafts = 255u;
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 255u && choice.probe == 0u);
	memset(&fresh,0,sizeof(fresh));
	request.proven_drafts = 0u;
	request.match_length = 3u;
	for (index=0u; index<40u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(3u,0u)],8u,0u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,&fresh,&request,&choice);
	assert(choice.drafts == 0u);
	printf("probe frames: a class the engine has not seen, or a request with no accepted draft in a class the engine would speculate on, gets a probe of the drafts whose frame costs at most the margin over a plain step; a rejected probe keeps the cap, the first accepted draft lifts it, after which a request's frames in every class grow at most to %u times the deepest draft it has had accepted in any class, and a class the engine knows fails never probes\n",SPARK_SPECULATION_FRAME_GROWTH);
}

static void TestBatch(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice alone,shared,crowded;
	uint32_t index;
	memset(&model,0,sizeof(model));
	assert(SparkSpeculationFrameBatchStepCost(&model,2u,2048u) == 0.0f && SparkSpeculationFrameBatchCost(&model,2u,8u,2048u) == 0.0f);
	SparkSpeculationFrameObserveBatchStep(&model,1u,8u,2048u,8u * 35000000ull);
	SparkSpeculationFrameObserveBatchStep(&model,17u,8u,2048u,8u * 35000000ull);
	assert(SparkSpeculationFrameBatchStepCost(&model,2u,2048u) == 0.0f);
	SparkSpeculationFrameObserveBatchStep(&model,4u,8u,2048u,8u * 50000000ull);
	assert(fabsf(SparkSpeculationFrameBatchStepCost(&model,4u,2048u) - 50e6f) < 256.0f && fabsf(SparkSpeculationFrameBatchStepCost(&model,4u,600000u) - 50e6f) < 256.0f);
	assert(fabsf(SparkSpeculationFrameBatchCost(&model,4u,12u,2048u) - 50e6f * 2.0f) < 256.0f && SparkSpeculationFrameBatchCost(&model,4u,3u,2048u) == 0.0f);
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 35000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,2048u,57000000ull);
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,67000000ull);
	SparkSpeculationFrameObserveFrame(&model,4u,2048u,40000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,234000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,308000000ull);
	assert(fabsf(SparkSpeculationFrameBatchSpread(&model,4u,2048u) - 10e6f) < 256.0f && SparkSpeculationFrameBatchSpread(&model,1u,2048u) == 0.0f);
	assert(fabsf(SparkSpeculationFrameBatchCost(&model,4u,33u,2048u) - SparkSpeculationFrameCost(&model,33u,2048u) - 10e6f) < 256.0f && SparkSpeculationFrameBatchCost(&model,1u,33u,2048u) == SparkSpeculationFrameCost(&model,33u,2048u));
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	memset(&request,0,sizeof(request));
	request.feasible = 64u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameChoose(&model,0,&request,&alone);
	request.lanes = 1u;
	SparkSpeculationFrameChoose(&model,0,&request,&shared);
	assert(alone.drafts != 0u && shared.drafts == alone.drafts && shared.frame_ns == alone.frame_ns);
	request.lanes = 4u;
	request.base_rows = 3u;
	request.base_tokens = 3.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&shared);
	assert(shared.drafts != 0u && shared.drafts + 3u + 1u <= 64u + 4u && fabsf(shared.frame_ns - SparkSpeculationFrameBatchCost(&model,4u,shared.drafts + 4u,2048u)) < 256.0f);
	request.base_rows = 40u;
	request.base_tokens = 3.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&crowded);
	assert(crowded.drafts == 0u || (3.0f + crowded.expected_tokens) / crowded.frame_ns > 4.0f / SparkSpeculationFrameBatchCost(&model,4u,41u,2048u));
	for (index=0u; index<40u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(3u,0u)],8u,0u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	request.match_length = 3u;
	request.base_rows = 3u;
	SparkSpeculationFrameChoose(&model,0,&request,&crowded);
	assert(crowded.drafts == 0u);
	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveBatchStep(&model,2u,8u,2048u,8u * 40000000ull);
	request.match_length = 64u;
	request.lanes = 2u;
	request.base_rows = 1u;
	request.base_tokens = 1.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&shared);
	assert(shared.drafts == 1u && shared.probe == 1u && shared.frame_ns > 40e6f);
}

static void TestMinimum(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice plain,floored,capped,edge;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	request.feasible = 255u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 62000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,2048u,71000000ull);
	SparkSpeculationFrameObserveFrame(&model,4u,2048u,116000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,260000000ull);
	SparkSpeculationFrameObserveFrame(&model,65u,2048u,345000000ull);
	SparkSpeculationFrameObserveFrame(&model,129u,2048u,520000000ull);
	SparkSpeculationFrameObserveFrame(&model,256u,2048u,780000000ull);
	for (index=0u; index<40u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],64u,32u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,0,&request,&plain);
	assert(plain.drafts != 0u && plain.drafts <= 32u);
	request.minimum = 64u;
	SparkSpeculationFrameChoose(&model,0,&request,&floored);
	assert(floored.drafts >= 64u && floored.expected_tokens >= plain.expected_tokens);
	request.cost_cap_ns = 300000000.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&capped);
	assert(capped.drafts != 0u && capped.drafts < 64u && capped.frame_ns <= 300000000.0f);
	request.cost_cap_ns = 0.0f;
	request.feasible = 40u;
	request.minimum = 40u;
	SparkSpeculationFrameChoose(&model,0,&request,&edge);
	assert(edge.drafts == 40u);
}

static void TestStale(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice;
	const uint32_t model_class = SparkSpeculationFrameClass(64u,0u);
	float cost;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	request.feasible = 3u;
	request.context = 256u;
	request.match_length = 64u;
	SparkSpeculationFrameObserveChain(&model,8u,256u,8u * 62000000ull);
	SparkSpeculationFrameObserveChain(&model,1u,256u,64000000ull);
	for (index=0u; index<40u; index++)
		SparkSpeculationHazardObserve(&model.classes[model_class],3u,3u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameObserveFrame(&model,2u,256u,80000000ull);
	SparkSpeculationFrameObserveFrame(&model,3u,256u,104000000ull);
	SparkSpeculationFrameObserveFrame(&model,4u,256u,144000000ull);
	for (index=0u; index<SPARK_SPECULATION_FRAME_STALE_FRAMES; index++)
		SparkSpeculationFrameObserveFrame(&model,3u,256u,94000000ull);
	assert(fabsf(SparkSpeculationFrameCost(&model,4u,256u) - 144e6f) < 1.0f);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 2u);
	SparkSpeculationFrameObserveFrame(&model,3u,256u,94000000ull);
	cost = SparkSpeculationFrameCost(&model,4u,256u);
	assert(cost > SparkSpeculationFrameCost(&model,3u,256u) && fabsf(cost - SparkSpeculationFrameCost(&model,3u,256u) * 12.0f / 11.0f) < 2.0f);
	assert(SparkSpeculationFrameCost(&model,3u,256u) < 95e6f && SparkSpeculationFrameCost(&model,3u,2048u) == SparkSpeculationFrameCost(&model,3u,256u));
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 3u);
	SparkSpeculationFrameObserveFrame(&model,4u,256u,100000000ull);
	assert(fabsf(SparkSpeculationFrameCost(&model,4u,256u) - 100e6f) < 1.0f);
	SparkSpeculationFrameObserveFrame(&model,4u,256u,300000000ull);
	assert(fabsf(SparkSpeculationFrameCost(&model,4u,256u) - 125e6f) < 1.0f);
	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveFrame(&model,4u,256u,90000000ull);
	for (index=0u; index<=SPARK_SPECULATION_FRAME_STALE_FRAMES; index++)
		SparkSpeculationFrameObserveFrame(&model,3u,256u,94000000ull);
	assert(fabsf(SparkSpeculationFrameCost(&model,4u,256u) - 90e6f) < 1.0f);
	SparkSpeculationFrameObserveFrame(&model,4u,256u,100000000ull);
	assert(fabsf(SparkSpeculationFrameCost(&model,4u,256u) - 100e6f) < 1.0f);
}

static void TestReach(void)
{
	assert(fabsf(SparkSpeculationFrameReach(1.0f,0.5f,0.0f,0.0f,0.3f) - 0.5f) < 1e-6f);
	assert(fabsf(SparkSpeculationFrameReach(1.0f,0.5f,6.0f,3.0f,0.3f) - 0.5f * 6.3f / 3.3f) < 1e-6f);
	assert(fabsf(SparkSpeculationFrameReach(1.0f,0.5f,1.0f,3.0f,0.3f) - 0.5f * 1.3f / 3.3f) < 1e-6f);
	assert(SparkSpeculationFrameReach(1.0f,0.9f,9.0f,3.0f,0.3f) == 1.0f);
	assert(SparkSpeculationFrameReach(0.4f,0.5f,6.0f,3.0f,0.3f) == 0.4f);
	assert(SparkSpeculationFrameReach(0.0f,0.5f,6.0f,3.0f,0.3f) == 0.0f);
}

static void TestColdHigherClassInheritance(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice inherited,small_feasible;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	SparkSpeculationFrameObserveChain(&model,1u,512u,200000000ull);
	SparkSpeculationFrameObserveFrame(&model,2u,512u,210000000ull);
	SparkSpeculationFrameObserveFrame(&model,8u,512u,230000000ull);
	SparkSpeculationFrameObserveFrame(&model,16u,512u,250000000ull);
	for (index=0u; index<6u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(6u,0u)],12u,12u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	request.feasible = 16u;
	request.context = 512u;
	request.match_length = 32u;
	SparkSpeculationFrameChoose(&model,0,&request,&inherited);
	assert(inherited.drafts >= 8u && inherited.probe == 0u);
	request.feasible = 6u;
	SparkSpeculationFrameChoose(&model,0,&request,&small_feasible);
	assert(small_feasible.drafts == 1u && small_feasible.probe == 1u);
}

static void TestBatchFrameCalibration(void)
{
	SparkSpeculationFrameModel model;
	float b1_cost,b8_mid,b8_high,after_up,after_down;
	uint32_t index;
	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 23500000ull);
	SparkSpeculationFrameObserveFrame(&model,16u,2048u,31000000ull);
	SparkSpeculationFrameObserveBatchStep(&model,8u,8u,2048u,8u * 50000000ull);
	assert(SparkSpeculationFrameBatchObserved(&model,8u,2048u) == 0u);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,72000000ull);
	assert(SparkSpeculationFrameBatchObserved(&model,8u,2048u) == 1u);
	assert(SparkSpeculationFrameBatchObserved(&model,4u,2048u) == 0u);
	b1_cost = SparkSpeculationFrameCost(&model,16u,2048u);
	assert(fabsf(b1_cost - 31e6f) < 2.0f);
	b8_mid = SparkSpeculationFrameBatchCost(&model,8u,16u,2048u);
	b8_high = SparkSpeculationFrameBatchCost(&model,8u,32u,2048u);
	assert(fabsf(b8_mid - 72e6f) < 2.0f);
	assert(b8_high > 85e6f && b8_high < 120e6f);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,92000000ull);
	after_up = SparkSpeculationFrameBatchCost(&model,8u,16u,2048u);
	assert(fabsf(after_up - 82e6f) < 2.0f);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,70000000ull);
	after_down = SparkSpeculationFrameBatchCost(&model,8u,16u,2048u);
	assert(fabsf(after_down - 79e6f) < 2.0f);
	SparkSpeculationFrameObserveBatchFrame(&model,6u,12u,2048u,45000000ull);
	assert(SparkSpeculationFrameBatchObserved(&model,6u,2048u) == 1u);
	assert(SparkSpeculationFrameBatchObserved(&model,8u,2048u) == 0u);
	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 5000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,5000000ull);
	assert(fabsf(SparkSpeculationFrameBatchCost(&model,8u,32u,2048u) - 5e6f) < 2.0f);
	for (index=0u; index<=SPARK_SPECULATION_FRAME_STALE_FRAMES; index++)
		SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,6000000ull);
	assert(SparkSpeculationFrameBatchCost(&model,8u,16u,2048u) > 5900000.0f);
}

static void TestColdClassInheritanceCap(void)
{
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice;
	uint32_t index;
	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 25000000ull);
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,26000000ull);
	SparkSpeculationFrameObserveFrame(&model,17u,2048u,30000000ull);
	SparkSpeculationFrameObserveFrame(&model,65u,2048u,40000000ull);
	for (index=0u; index<10u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(8u,0u)],12u,12u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	assert(SparkSpeculationFrameClassHazard(&model,SparkSpeculationFrameClass(64u,0u),1u) == &model.classes[SparkSpeculationFrameClass(8u,0u)]);
	assert(SparkSpeculationFrameClassHazard(&model,SparkSpeculationFrameClass(64u,0u),0u) == &model.classes[SparkSpeculationFrameClass(64u,0u)]);
	request.feasible = 64u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.probe == 0u && choice.drafts >= 8u && choice.drafts <= SPARK_SPECULATION_FRAME_PROBE_DRAFTS);
	request.lanes = 8u;
	request.base_rows = 7u;
	request.base_tokens = 7.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.probe == 1u && choice.drafts <= SPARK_SPECULATION_FRAME_PROBE_DRAFTS);
}

static void SimulateB1LegacyRate(const SparkSpeculationFrameModel *model, const SparkSpeculationHazard *hazard, uint32_t context,
	uint32_t max_drafts, uint32_t *drafts_out, float *v_out)
{
	const float base = SparkSpeculationFrameTokenCost(model,context);
	float best_rate = 0.0f;
	uint32_t drafts,best = 0u;
	for (drafts=1u; drafts<=max_drafts; drafts++)
	{
		const float cost = SparkSpeculationFrameCost(model,drafts + 1u,context);
		const float expected = SparkSpeculationFrameExpected(hazard,0,drafts);
		const float rate = cost > 0.0f ? expected / cost : 0.0f;
		if ( rate * base >= SPARK_SPECULATION_FRAME_MARGIN && rate > best_rate )
		{
			best_rate = rate;
			best = drafts;
		}
	}
	*drafts_out = best;
	*v_out = best != 0u ? SparkSpeculationFrameCost(model,best + 1u,context) / base : 1.0f;
}

static void SimulateB8Policy(const SparkSpeculationFrameModel *model, uint32_t lanes, uint32_t context,
	const uint32_t *buckets, const float *expected, uint32_t count, float wait_ns, uint32_t use_v_score,
	uint32_t *rows_out, float *v_out, float *alpha_out)
{
	const float step = SparkSpeculationFrameBatchStepCost(model,lanes,context);
	float best_obj = 0.0f;
	uint32_t index,best_i = count;
	for (index=0u; index<count; index++)
	{
		const float cost = SparkSpeculationFrameBatchCost(model,lanes,buckets[index],context) + wait_ns;
		const float rate = cost > 0.0f ? expected[index] / cost : 0.0f;
		const float obj = use_v_score != 0u ? SparkSpeculationFrameScore(rate,cost,step,lanes) : rate;
		if ( rate * step >= SPARK_SPECULATION_FRAME_MARGIN * (float)lanes && obj > best_obj )
		{
			best_obj = obj;
			best_i = index;
		}
	}
	if ( best_i >= count )
	{
		*rows_out = 0u;
		*v_out = 1.0f;
		*alpha_out = 1.0f;
		return;
	}
	*rows_out = buckets[best_i];
	*v_out = (SparkSpeculationFrameBatchCost(model,lanes,buckets[best_i],context) + wait_ns) / step;
	*alpha_out = expected[best_i] / (float)lanes;
}

static void TestPlatformVerifyEfficiency(void)
{
	static const uint32_t b8_rows[5] = { 16u, 24u, 32u, 48u, 64u };
	static const float b8_edit_exp[5] = { 16.0f, 24.0f, 32.0f, 48.0f, 64.0f };
	static const float b8_neural_exp[5] = { 13.6f, 20.0f, 24.8f, 30.0f, 32.0f };
	static const float b8_weak_exp[5] = { 11.2f, 13.0f, 14.0f, 14.5f, 15.0f };
	SparkSpeculationFrameModel model;
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice;
	uint32_t index,old_d,old_r,new_r;
	float old_v1,new_v1,old_v8,new_v8,old_a8,new_a8,alpha_mean;

	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 25000000ull);
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,25500000ull);
	SparkSpeculationFrameObserveFrame(&model,16u,2048u,31000000ull);
	SparkSpeculationFrameObserveFrame(&model,32u,2048u,39000000ull);
	SparkSpeculationFrameObserveFrame(&model,48u,2048u,55000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,72000000ull);
	SparkSpeculationFrameObserveBatchStep(&model,8u,8u,2048u,8u * 27000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,28350000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,32u,2048u,32265000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,64u,2048u,52650000ull);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],64u,64u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,0u,&old_r,&old_v8,&old_a8);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,1u,&new_r,&new_v8,&new_a8);
	assert(old_r == 64u && new_r == 32u && new_v8 < old_v8 && new_v8 <= 1.25f && new_a8 >= 3.9f);
	printf("p2-sim 1x-dense B8: rows %u->%u V_b8 %.3f->%.3f alpha_b8 %.2f->%.2f\n",old_r,new_r,old_v8,new_v8,old_a8,new_a8);

	memset(&model,0,sizeof(model));
	memset(&request,0,sizeof(request));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 23500000ull);
	SparkSpeculationFrameObserveFrame(&model,6u,2048u,42700000ull);
	SparkSpeculationFrameObserveFrame(&model,9u,2048u,55000000ull);
	SparkSpeculationFrameObserveFrame(&model,12u,2048u,58400000ull);
	SparkSpeculationFrameObserveFrame(&model,21u,2048u,82400000ull);
	SparkSpeculationFrameObserveFrame(&model,40u,2048u,122000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,175000000ull);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],64u,64u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SimulateB1LegacyRate(&model,&model.classes[SparkSpeculationFrameClass(64u,0u)],2048u,63u,&old_d,&old_v1);
	request.feasible = 63u;
	request.context = 2048u;
	request.match_length = 64u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	new_v1 = choice.frame_ns / 23500000.0f;
	alpha_mean = (2.47f + 3.97f + choice.expected_tokens) / 3.0f;
	assert(old_d >= 39u && choice.drafts == 11u && new_v1 < 2.55f && new_v1 * 2.0f < old_v1 && alpha_mean >= 4.0f);
	SparkSpeculationFrameObserveBatchStep(&model,8u,8u,2048u,8u * 50000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,72000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,24u,2048u,95000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,32u,2048u,115000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,48u,2048u,155000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,64u,2048u,195000000ull);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,0u,&old_r,&old_v8,&old_a8);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,1u,&new_r,&new_v8,&new_a8);
	assert(old_r == 64u && new_r == 16u && new_v8 < 1.50f && new_v8 * 2.0f < old_v8);
	printf("p2-sim 4x-tp4 B1: drafts %u->%u V_b1 %.3f->%.3f alpha_mean=%.2f | B8: rows %u->%u V_b8 %.3f->%.3f\n",
		old_d,choice.drafts,old_v1,new_v1,alpha_mean,old_r,new_r,old_v8,new_v8);

	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveChain(&model,8u,2048u,8u * 40000000ull);
	SparkSpeculationFrameObserveFrame(&model,6u,2048u,60000000ull);
	SparkSpeculationFrameObserveFrame(&model,8u,2048u,72000000ull);
	SparkSpeculationFrameObserveFrame(&model,12u,2048u,96000000ull);
	SparkSpeculationFrameObserveFrame(&model,16u,2048u,124000000ull);
	SparkSpeculationFrameObserveFrame(&model,24u,2048u,168000000ull);
	SparkSpeculationFrameObserveFrame(&model,48u,2048u,288000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,384000000ull);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],64u,64u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SimulateB1LegacyRate(&model,&model.classes[SparkSpeculationFrameClass(64u,0u)],2048u,63u,&old_d,&old_v1);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	new_v1 = choice.frame_ns / 40000000.0f;
	alpha_mean = (1.0f + 1.0f + choice.expected_tokens) / 3.0f;
	assert(choice.drafts == 11u && new_v1 <= 2.45f && new_v1 * 2.5f < old_v1 && alpha_mean >= 4.0f);
	SparkSpeculationFrameObserveBatchStep(&model,8u,8u,2048u,8u * 80000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,140000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,24u,2048u,192000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,32u,2048u,240000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,48u,2048u,340000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,64u,2048u,440000000ull);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,0u,&old_r,&old_v8,&old_a8);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_edit_exp,5u,0.0f,1u,&new_r,&new_v8,&new_a8);
	assert(old_r == 64u && new_r == 16u && new_v8 <= 1.80f && new_v8 * 3.0f < old_v8);
	printf("p2-sim 16x-tp16 B1-lookup: drafts %u->%u V_b1 %.3f->%.3f alpha_mean=%.2f | B8-edit: rows %u->%u V_b8 %.3f->%.3f\n",
		old_d,choice.drafts,old_v1,new_v1,alpha_mean,old_r,new_r,old_v8,new_v8);

	memset(&model,0,sizeof(model));
	SparkSpeculationFrameObserveBatchStep(&model,8u,8u,2048u,8u * 80000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,16u,2048u,140000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,32u,2048u,184000000ull);
	SparkSpeculationFrameObserveBatchFrame(&model,8u,64u,2048u,340000000ull);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_neural_exp,5u,16000000.0f,1u,&new_r,&new_v8,&new_a8);
	assert(new_r == 24u && new_v8 <= 2.35f && new_a8 >= 2.4f);
	printf("p2-sim 16x-tp16 B8-neural: rows=%u V_b8=%.3f alpha_b8=%.2f (unprofitable rounds declined)\n",new_r,new_v8,new_a8);
	SimulateB8Policy(&model,8u,2048u,b8_rows,b8_weak_exp,5u,16000000.0f,1u,&new_r,&new_v8,&new_a8);
	assert(new_r == 0u);
}

int main(void)
{
	TestBuckets();
	TestHazard();
	TestRequestBlend();
	TestCost();
	TestMinimum();
	TestChoose();
	TestProbe();
	TestBatch();
	TestStale();
	TestReach();
	TestColdHigherClassInheritance();
	TestBatchFrameCalibration();
	TestColdClassInheritanceCap();
	TestPlatformVerifyEfficiency();
	printf("speculation frame: hazards per match class, interpolated frame costs and the chain price decide every frame; a frame shared by several lanes prices drafts against the rows and tokens already in it, with the extra cost of separate sequences learned from batch steps; a cost not refreshed for %u frames yields to its fresh neighbours and is replaced when measured again: ok\n",SPARK_SPECULATION_FRAME_STALE_FRAMES);
	return(0);
}
