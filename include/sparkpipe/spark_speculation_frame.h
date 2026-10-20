#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_FRAME_MATCH_CLASSES 5u
#define SPARK_SPECULATION_FRAME_MODEL_CLASSES 2u
#define SPARK_SPECULATION_FRAME_CLASSES (2u * SPARK_SPECULATION_FRAME_MATCH_CLASSES + SPARK_SPECULATION_FRAME_MODEL_CLASSES)
#define SPARK_SPECULATION_FRAME_DEPTH_BUCKETS 9u
#define SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS 16u
#define SPARK_SPECULATION_FRAME_ROW_BUCKETS 22u
#define SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS 9u
#define SPARK_SPECULATION_FRAME_MAX_DRAFTS 255u
#define SPARK_SPECULATION_FRAME_PROBE_DRAFTS 16u
#define SPARK_SPECULATION_FRAME_HAZARD_PRIOR 0.2f
#define SPARK_SPECULATION_FRAME_REQUEST_WEIGHT 8.0f
#define SPARK_SPECULATION_FRAME_ENGINE_DECAY 0.995f
#define SPARK_SPECULATION_FRAME_REQUEST_DECAY 0.98f
#define SPARK_SPECULATION_FRAME_COST_WEIGHT 0.125f
#define SPARK_SPECULATION_FRAME_MARGIN 1.05f

typedef struct SparkSpeculationHazard
{
	float risk[SPARK_SPECULATION_FRAME_DEPTH_BUCKETS];
	float events[SPARK_SPECULATION_FRAME_DEPTH_BUCKETS];
	uint32_t frames;
} SparkSpeculationHazard;

typedef struct SparkSpeculationCost
{
	float ns;
	float rows;
	float samples;
} SparkSpeculationCost;

typedef struct SparkSpeculationFrameModel
{
	SparkSpeculationHazard classes[SPARK_SPECULATION_FRAME_CLASSES];
	SparkSpeculationCost frame[SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS][SPARK_SPECULATION_FRAME_ROW_BUCKETS];
	SparkSpeculationCost token[SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS];
	SparkSpeculationCost single[SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS];
} SparkSpeculationFrameModel;

typedef struct SparkSpeculationFrameRequest
{
	uint32_t feasible;
	uint32_t minimum;
	uint32_t context;
	uint32_t match_length;
	uint32_t sampled;
	float cost_cap_ns;
} SparkSpeculationFrameRequest;

typedef struct SparkSpeculationFrameChoice
{
	uint32_t drafts;
	uint32_t probe;
	float expected_tokens;
	float frame_ns;
	float chain_token_ns;
} SparkSpeculationFrameChoice;

uint32_t SparkSpeculationFrameClass(uint32_t match_length, uint32_t sampled);
uint32_t SparkSpeculationFrameModelClass(uint32_t sampled);
uint32_t SparkSpeculationFrameClassIsModel(uint32_t frame_class);
uint32_t SparkSpeculationFrameDepthBucket(uint32_t draft_index);
uint32_t SparkSpeculationFrameRowBucket(uint32_t rows);
uint32_t SparkSpeculationFrameRowBucketLimit(uint32_t bucket);
uint32_t SparkSpeculationFrameContextBucket(uint32_t context);
void SparkSpeculationHazardObserve(SparkSpeculationHazard *hazard, uint32_t drafts, uint32_t accepted, float decay);
float SparkSpeculationHazardAt(const SparkSpeculationHazard *engine, const SparkSpeculationHazard *request, uint32_t bucket);
float SparkSpeculationFrameExpected(const SparkSpeculationHazard *engine, const SparkSpeculationHazard *request, uint32_t drafts);
void SparkSpeculationFrameObserveFrame(SparkSpeculationFrameModel *model, uint32_t rows, uint32_t context, uint64_t ns);
void SparkSpeculationFrameObserveChain(SparkSpeculationFrameModel *model, uint32_t steps, uint32_t context, uint64_t ns);
float SparkSpeculationFrameCost(const SparkSpeculationFrameModel *model, uint32_t rows, uint32_t context);
float SparkSpeculationFrameTokenCost(const SparkSpeculationFrameModel *model, uint32_t context);
float SparkSpeculationFrameSingleCost(const SparkSpeculationFrameModel *model, uint32_t context);
void SparkSpeculationFrameChoose(const SparkSpeculationFrameModel *model, const SparkSpeculationHazard *request_hazard,
	const SparkSpeculationFrameRequest *request, SparkSpeculationFrameChoice *choice);

#ifdef __cplusplus
}
#endif
