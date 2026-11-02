#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_LAYERS_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_LAYERS_H

#include <stdint.h>

#include "inference/runner/device_stage_model.h"
#include "sparkpipe/spark_weight_codec.h"

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
	uint32_t sigmoid_gate;
} SparkDeviceGatedDeltaShape;

typedef struct SparkDeviceGatedDeltaWeights
{
	SparkDeviceWeight mix;
	SparkDeviceWeight gates;
	SparkTensor conv;
	SparkTensor log_scale;
	SparkTensor bias;
	SparkTensor norm;
	SparkDeviceWeight out;
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

typedef struct SparkDeviceChannelDeltaShape
{
	uint32_t heads;
	uint32_t dimension;
	uint32_t conv_taps;
	uint32_t low_rank;
	float epsilon;
	float l2_epsilon;
	float minimum_log_decay;
	uint32_t beta_mode;
	uint32_t norm_f32;
	uint32_t norm_round;
	uint32_t conv_f32;
	uint32_t gate_full_rank;
	float prenorm_epsilon;
} SparkDeviceChannelDeltaShape;

typedef struct SparkDeviceChannelDeltaWeights
{
	SparkDeviceWeight qkv;
	SparkDeviceWeight beta;
	SparkDeviceWeight decay_down;
	SparkDeviceWeight decay;
	SparkDeviceWeight gate_down;
	SparkDeviceWeight gate;
	SparkTensor conv;
	SparkTensor log_scale;
	SparkTensor bias;
	SparkTensor norm;
	SparkDeviceWeight out;
} SparkDeviceChannelDeltaWeights;

typedef struct SparkDeviceChannelDeltaScratch
{
	SparkTensor qkv;
	SparkTensor beta;
	SparkTensor low;
	SparkTensor decay;
	SparkTensor gate;
	SparkTensor delta;
	SparkTensor core;
	SparkTensor gated;
} SparkDeviceChannelDeltaScratch;

#define SPARK_LATENT_GATE_HEAD 1u
#define SPARK_LATENT_GATE_ELEMENT 2u

typedef struct SparkDeviceLatentAttentionShape
{
	uint32_t heads;
	uint32_t nope;
	uint32_t rope;
	uint32_t latent;
	uint32_t value_dimension;
	double rope_theta;
	float epsilon;
	float scale;
	uint32_t head_gate;
	uint32_t query_lora;
	uint32_t norm_bf16;
	uint32_t sinks;
	const char *tensor_prefix;
	uint32_t query_concat;
	uint32_t query_gate;
	float rope_factor;
	float rope_beta_fast;
	float rope_beta_slow;
	uint32_t rope_original;
	uint32_t rope_unrotated;
	uint32_t gate_unrounded;
	uint32_t rope_half;
	uint32_t window;
	uint32_t differential;
} SparkDeviceLatentAttentionShape;

typedef struct SparkDeviceLatentAttentionRope
{
	SparkHeadRopeParams query;
	SparkHeadRopeParams key;
} SparkDeviceLatentAttentionRope;

typedef struct SparkDeviceLatentAttentionWeights
{
	SparkDeviceWeight query_down;
	SparkTensor query_norm;
	SparkTensor sinks;
	SparkDeviceWeight query;
	SparkDeviceWeight kv;
	SparkTensor kv_norm;
	SparkTensor key_up;
	SparkTensor value_up;
	SparkDeviceWeight gate;
	SparkDeviceWeight out;
	SparkDeviceWeight lambda;
} SparkDeviceLatentAttentionWeights;

typedef struct SparkDeviceLatentAttentionScratch
{
	SparkTensor query_lora;
	SparkTensor query;
	SparkTensor kv;
	SparkTensor gate;
	SparkTensor packed;
	SparkTensor records;
	SparkTensor latent;
	SparkTensor value;
	SparkTensor lambda;
	SparkTensor differential;
} SparkDeviceLatentAttentionScratch;

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
	uint32_t gate_projection;
	uint32_t key_is_value;
	uint32_t value_norm;
	uint32_t qk_weightless;
	uint32_t qk_plain;
	float query_multiply;
	uint32_t value_dimension;
	float value_scale;
	uint32_t sinks;
	uint32_t window;
	uint32_t rope_pairs;
	uint32_t rope_rounded_terms;
	float rope_scale;
	float yarn_factor;
	float yarn_original;
	float yarn_beta_fast;
	float yarn_beta_slow;
	uint32_t head_gate;
	const char *tensor_prefix;
} SparkDeviceAttentionShape;

typedef struct SparkDeviceAttentionRope
{
	SparkHeadRopeParams query;
	SparkHeadRopeParams key;
	SparkHeadRopeParams value;
} SparkDeviceAttentionRope;

typedef struct SparkDeviceAttentionWeights
{
	SparkDeviceWeight qkv;
	SparkDeviceWeight gate;
	SparkDeviceWeight out;
	SparkTensor q_norm;
	SparkTensor k_norm;
	SparkTensor sinks;
} SparkDeviceAttentionWeights;

typedef struct SparkDeviceAttentionScratch
{
	SparkTensor qkv;
	SparkTensor query;
	SparkTensor key;
	SparkTensor value;
	SparkTensor records;
	SparkTensor gate;
	SparkTensor gated;
	SparkTensor qkv_local;
	SparkTensor qkv_gathered;
} SparkDeviceAttentionScratch;

typedef struct SparkDeviceFfnWeights
{
	SparkDeviceWeight gate_up;
	SparkDeviceWeight down;
} SparkDeviceFfnWeights;

typedef struct SparkDeviceExpertsShape
{
	uint32_t experts;
	uint32_t top_k;
	uint32_t intermediate;
	uint32_t activation;
	float epsilon;
	float routed_scale;
	uint32_t round_logits;
	uint32_t groups;
	uint32_t topk_groups;
	float limit;
	const char *router;
	uint32_t tensor_parallel;
} SparkDeviceExpertsShape;

typedef struct SparkDeviceExpertsWeights
{
	SparkTensor router;
	SparkTensor router_scale;
	SparkTensor expert_scale;
	SparkTensor bias;
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
	SparkTensor sorted;
} SparkDeviceExpertsScratch;

typedef struct SparkDeviceLowRankGateWeights
{
	SparkDeviceWeight down;
	SparkDeviceWeight up;
} SparkDeviceLowRankGateWeights;

typedef struct SparkDeviceLowRankGateScratch
{
	SparkTensor bottleneck;
	SparkTensor gate;
} SparkDeviceLowRankGateScratch;

typedef struct SparkDeviceExpertMatrix
{
	uint32_t codec;
	uint32_t scale_group;
	SparkTensor values;
	SparkTensor scale;
	SparkTensor global;
} SparkDeviceExpertMatrix;

typedef struct SparkDeviceRoutedFfnShape
{
	uint32_t experts;
	uint32_t top_k;
	uint32_t intermediate;
	uint32_t activation;
	uint32_t weighted;
	float limit;
	float up_limit;
	uint32_t width;
} SparkDeviceRoutedFfnShape;

typedef struct SparkDeviceRoutedPoly
{
	SparkTensor params;
	float clamp;
	float bias_clamp;
	float scale;
	float epsilon;
} SparkDeviceRoutedPoly;

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

uint32_t SparkDeviceChannelDeltaWidth(const SparkDeviceChannelDeltaShape *shape);
uint64_t SparkDeviceChannelDeltaStateBytes(const SparkDeviceChannelDeltaShape *shape);
uint64_t SparkDeviceChannelDeltaWindowBytes(const SparkDeviceChannelDeltaShape *shape);
SparkStatus SparkDeviceChannelDeltaBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceChannelDeltaShape *shape, SparkDeviceChannelDeltaWeights *bound);
SparkStatus SparkDeviceChannelDelta(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceChannelDeltaShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceChannelDeltaWeights *weights, uint32_t recurrent_layer,
	const SparkDeviceChannelDeltaScratch *scratch, SparkTensor normed, SparkTensor output);

uint32_t SparkDeviceLatentAttentionSlot(const SparkDeviceLatentAttentionShape *shape);
void SparkDeviceLatentAttentionRopePrepare(const SparkDeviceLatentAttentionShape *shape, uint32_t local_heads, SparkDeviceLatentAttentionRope *rope);
SparkStatus SparkDeviceLatentAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceLatentAttentionShape *shape, SparkDeviceLatentAttentionWeights *bound);
SparkStatus SparkDeviceLatentAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceLatentAttentionShape *shape,
	const SparkDeviceLatentAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceLatentAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceLatentAttentionScratch *scratch, SparkTensor normed, SparkTensor output);

uint32_t SparkDeviceAttentionQueryWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionKvWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionFusedWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionValueWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionOutputWidth(const SparkDeviceAttentionShape *shape);
uint32_t SparkDeviceAttentionHeadGateRows(const SparkDeviceAttentionShape *shape, uint32_t degree);
void SparkDeviceAttentionRopePrepare(const SparkDeviceAttentionShape *shape, SparkDeviceAttentionRope *rope);
SparkStatus SparkDeviceAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceAttentionShape *shape, SparkDeviceAttentionWeights *bound);
SparkStatus SparkDeviceAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceAttentionShape *shape,
	const SparkDeviceAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceAttentionScratch *scratch, SparkTensor normed, SparkTensor output);

SparkStatus SparkDeviceFfnBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, uint32_t ffn,
	SparkDeviceFfnWeights *bound);
SparkStatus SparkDeviceFfnBindNamed(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *gate_up,
	const char *down, uint32_t ffn, SparkDeviceFfnWeights *bound);
SparkStatus SparkDeviceGatedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, uint32_t activation, float limit, const SparkDeviceStepTensors *tensors,
	const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output);
SparkStatus SparkDeviceGatedFfnLimits(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, uint32_t activation, float limit, float up_limit,
	const SparkDeviceStepTensors *tensors, const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output);

uint64_t SparkDeviceExpertsScratchBytes(const SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape, uint32_t part);
SparkStatus SparkDeviceExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound);
SparkStatus SparkDeviceExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, const SparkDeviceStepTensors *tensors,
	uint32_t layer, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor output);
SparkStatus SparkDeviceRoutedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceRoutedFfnShape *shape, uint32_t rows,
	const SparkDeviceExpertMatrix *gate, const SparkDeviceExpertMatrix *up, const SparkDeviceExpertMatrix *down, const SparkDeviceExpertsScratch *scratch,
	SparkTensor input);
SparkStatus SparkDeviceRoutedFfnPoly(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceRoutedFfnShape *shape, uint32_t rows,
	const SparkDeviceExpertMatrix *gate, const SparkDeviceExpertMatrix *up, const SparkDeviceExpertMatrix *down, const SparkDeviceRoutedPoly *poly,
	const SparkDeviceExpertsScratch *scratch, SparkTensor input);
SparkStatus SparkDeviceRoutedFfnQdq(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceRoutedFfnShape *shape, uint32_t rows,
	const SparkDeviceExpertMatrix *gate, const SparkDeviceExpertMatrix *up, const SparkDeviceExpertMatrix *down, const SparkDeviceExpertsScratch *scratch,
	SparkTensor input);
SparkStatus SparkDeviceSigmoidExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound);
SparkStatus SparkDeviceSigmoidExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor normed,
	SparkTensor shared, SparkTensor output);
SparkStatus SparkDeviceBranchNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors, SparkTensor first,
	const SparkTensor *second, const SparkTensor *first_weight, const SparkTensor *second_weight, SparkTensor output_weight, const SparkTensor *scalar);
SparkStatus SparkDeviceResidualNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors,
	const SparkTensor *addend, SparkTensor weight, SparkTensor normed);
SparkStatus SparkDeviceLinear(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, SparkTensor w, SparkTensor y,
	uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension);
SparkStatus SparkDeviceLowRankGateBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *prefix,
	uint32_t rank, SparkDeviceLowRankGateWeights *bound);
SparkStatus SparkDeviceLowRankGate(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t rows, uint32_t rank, const SparkDeviceLowRankGateWeights *weights,
	const SparkDeviceLowRankGateScratch *scratch, SparkTensor values);
SparkStatus SparkDeviceLinearWeight(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, const SparkDeviceWeight *w,
	SparkTensor y, uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension);

#ifdef __cplusplus
}
#endif

#endif
