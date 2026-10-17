#include "sparkpipe/spark_speculation_frame.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void TestBuckets(void)
{
	assert(SparkSpeculationFrameClass(3u,0u) == 0u && SparkSpeculationFrameClass(7u,0u) == 0u);
	assert(SparkSpeculationFrameClass(8u,0u) == 1u && SparkSpeculationFrameClass(31u,0u) == 2u);
	assert(SparkSpeculationFrameClass(63u,0u) == 3u && SparkSpeculationFrameClass(64u,0u) == 4u);
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
	assert(choice.drafts == SPARK_SPECULATION_FRAME_PROBE_DRAFTS && choice.probe == 1u);
	request.feasible = 5u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 5u && choice.probe == 1u);
	request.feasible = 255u;
	SparkSpeculationFrameObserveFrame(&model,2u,2048u,67000000ull);
	SparkSpeculationFrameObserveFrame(&model,33u,2048u,234000000ull);
	SparkSpeculationFrameObserveFrame(&model,64u,2048u,308000000ull);
	SparkSpeculationFrameObserveFrame(&model,256u,2048u,800000000ull);
	for (index=0u; index<20u; index++)
		SparkSpeculationHazardObserve(&model.classes[SparkSpeculationFrameClass(64u,0u)],255u,255u,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 255u && choice.probe == 0u && choice.expected_tokens > 200.0f);
	request.feasible = 100u;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 100u);
	request.feasible = 255u;
	request.cost_cap_ns = 130000000.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts != 0u && choice.frame_ns <= 130000000.0f);
	request.cost_cap_ns = 60000000.0f;
	SparkSpeculationFrameChoose(&model,0,&request,&choice);
	assert(choice.drafts == 0u);
	request.cost_cap_ns = 0.0f;
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

int main(void)
{
	TestBuckets();
	TestHazard();
	TestRequestBlend();
	TestCost();
	TestMinimum();
	TestChoose();
	printf("speculation frame: hazards per match class, interpolated frame costs and the chain price decide every frame: ok\n");
	return(0);
}
