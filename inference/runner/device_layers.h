#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_LAYERS_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_LAYERS_H

#include <stdint.h>

#include "inference/runner/device_stage_model.h"

#define SPARK_DEVICE_GATED_DELTA_STATE_POOL 0u
#define SPARK_DEVICE_GATED_DELTA_WINDOW_POOL 1u
#define SPARK_DEVICE_GATED_DELTA_POOLS 2u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkDeviceGatedDeltaShape
{
	uint32_t key_heads;
	uint32_t value_heads;
	uint32_t key_dimension;
	uint32_t value_dimension;
	uint32_t conv_taps;
	float epsilon;
} SparkDeviceGatedDeltaShape;

typedef struct SparkDeviceGatedDeltaWeights
{
	SparkTensor mix;
	SparkTensor gates;
	SparkTensor conv;
	SparkTensor log_scale;
	SparkTensor bias;
	SparkTensor norm;
	SparkTensor out;
} SparkDeviceGatedDeltaWeights;

typedef struct SparkDeviceGatedDeltaScratch
{
	SparkTensor mix;
	SparkTensor gates;
	SparkTensor delta;
	SparkTensor core;
	SparkTensor gated;
	SparkTensor keep;
} SparkDeviceGatedDeltaScratch;

typedef struct SparkDeviceAttentionShape
{
	uint32_t heads;
	uint32_t kv_heads;
	uint32_t head_dimension;
	uint32_t rope_dimension;
	double rope_theta;
	float epsilon;
	float scale;
	uint32_t output_gate;
	uint32_t key_is_value;
	uint32_t value_norm;
	uint32_t window;
	uint32_t rope_pairs;
	uint32_t rope_rounded_terms;
} SparkDeviceAttentionShape;

typedef struct SparkDeviceAttentionRope
{
	SparkHeadRopeParams query;
	SparkHeadRopeParams key;
	SparkHeadRopeParams value;
} SparkDeviceAttentionRope;

typedef struct SparkDeviceAttentionWeights
{
	SparkTensor qkv;
	SparkTensor out;
	SparkTensor q_norm;
	SparkTensor k_norm;
} SparkDeviceAttentionWeights;

typedef struct SparkDeviceAttentionScratch
{
	SparkTensor qkv;
	SparkTensor query;
	SparkTensor key;
	SparkTensor value;
	SparkTensor records;
	SparkTensor gated;
} SparkDeviceAttentionScratch;

typedef struct SparkDeviceFfnWeights
{
	SparkTensor gate_up;
	SparkTensor down;
} SparkDeviceFfnWeights;

typedef struct SparkDeviceExpertsShape
{
	uint32_t experts;
	uint32_t top_k;
	uint32_t intermediate;
	uint32_t activation;
	float epsilon;
} SparkDeviceExpertsShape;

typedef struct SparkDeviceExpertsWeights
{
	SparkTensor router;
	SparkTensor router_scale;
	SparkTensor expert_scale;
	SparkTensor input_norm;
	SparkTensor gate;
	SparkTensor up;
	SparkTensor down;
} SparkDeviceExpertsWeights;

typedef struct SparkDeviceExpertsScratch
{
	SparkTensor routed;
	SparkTensor logits;
	SparkTensor experts;
	SparkTensor weights;
	SparkTensor normed;
	SparkTensor gate_up;
	SparkTensor act;
	SparkTensor out;
} SparkDeviceExpertsScratch;

typedef struct SparkDeviceFfnScratch
{
	SparkTensor gate_up;
	SparkTensor act;
} SparkDeviceFfnScratch;

uint32_t SparkDeviceGatedDeltaQk(const SparkDeviceGatedDeltaShape *shape);
uint32_t SparkDeviceGatedDeltaValues(const SparkDeviceGatedDeltaShape *shape);
uint32_t SparkDeviceGatedDeltaChannels(const SparkDeviceGatedDeltaShape *shape);
uint32_t SparkDeviceGatedDeltaMix(const SparkDeviceGatedDeltaShape *shape);
uint32_t SparkDeviceGatedDeltaGates(const SparkDeviceGatedDeltaShape *shape);
uint64_t SparkDeviceGatedDeltaStateBytes(const SparkDeviceGatedDeltaShape *shape);
uint64_t SparkDeviceGatedDeltaWindowBytes(const SparkDeviceGatedDeltaShape *shape);
uint64_t SparkDeviceGatedDeltaKeepBytes(const SparkDeviceGatedDeltaShape *shape, uint32_t rows, uint32_t layers);
SparkStatus SparkDeviceGatedDeltaBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceGatedDeltaShape *shape, SparkDeviceGatedDeltaWeights *bound);
SparkStatus SparkDeviceGatedDelta(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceGatedDeltaShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceGatedDeltaWeights *weights, uint32_t recurrent_layer,
	const SparkDeviceGatedDeltaScratch *scratch, SparkTensor normed, SparkTensor output);
SparkStatus SparkDeviceGatedDeltaKeep(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceGatedDeltaShape *shape,
	const SparkDeviceKeepTensors *keep, const SparkDeviceGatedDeltaWeights *weights, uint32_t recurrent_layer, const SparkDeviceGatedDeltaScratch *scratch);

uint32_t SparkDeviceAttentionQueryWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionKvWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionFusedWidth(const SparkDeviceAttentionShape *shape);
void SparkDeviceAttentionRopePrepare(const SparkDeviceAttentionShape *shape, SparkDeviceAttentionRope *rope);
SparkStatus SparkDeviceAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceAttentionShape *shape, SparkDeviceAttentionWeights *bound);
SparkStatus SparkDeviceAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceAttentionShape *shape,
	const SparkDeviceAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceAttentionScratch *scratch, SparkTensor normed, SparkTensor output);

SparkStatus SparkDeviceFfnBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, uint32_t ffn,
	SparkDeviceFfnWeights *bound);
SparkStatus SparkDeviceGatedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, uint32_t activation, const SparkDeviceStepTensors *tensors,
	const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output);

uint64_t SparkDeviceExpertsScratchBytes(const SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape, uint32_t part);
SparkStatus SparkDeviceExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound);
SparkStatus SparkDeviceExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, const SparkDeviceStepTensors *tensors,
	uint32_t layer, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor output);
SparkStatus SparkDeviceBranchNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors, SparkTensor first,
	const SparkTensor *second, const SparkTensor *first_weight, const SparkTensor *second_weight, SparkTensor output_weight, const SparkTensor *scalar);
SparkStatus SparkDeviceResidualNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors,
	const SparkTensor *addend, SparkTensor weight, SparkTensor normed);
SparkStatus SparkDeviceLinear(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, SparkTensor w, SparkTensor y,
	uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension);

#ifdef __cplusplus
}
#endif

#endif
