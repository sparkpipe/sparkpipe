#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen38_max_model.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"

/*
 * Qwen 3.8 Max resident decode stage, hardware validation (sm_121a).
 *
 * Ported from the qwen38_27b harness (the family's proven template) at the
 * Max geometry, plus the two checks this module's production path adds: the
 * MXFP4-E2M1 routed-expert kernels against an exact dequant oracle, and the
 * rank-local TP4 kernel geometry at rank 0 (whole-head GDN shard + the
 * 128-expert MXFP4 shard), the collective-free half of the TP story - the
 * cross-rank half gates at the fleet window.
 *
 * The MODULE tier loads a stage pack through the module's own
 * Initialize/Execute (the pack path comes from SPARK_QWEN38_MAX_STAGE_*
 * exactly as production), drives decode frames, and checks determinism
 * across a fresh instance. The module is decode-only (prefill frames fail
 * closed by design; the chunk kernels are validated at the KERNEL tier
 * against the recurrence oracle instead) and Admit/Snapshot are fail-closed
 * stubs, which this harness asserts rather than assumes.
 *
 * Every comparison prints its numbers; the thresholds are the guard, the
 * numbers are the evidence.
 */

#define SPARK_QWEN38_MAX_VALIDATION_ROWS 4u
#define SPARK_QWEN38_MAX_VALIDATION_ATTN_TOKENS 5u
#define SPARK_QWEN38_MAX_VALIDATION_CHUNK_TOKENS 64u
/* Five rows: an admitted B1 decode shape (1/5/7/8/16/32/64/1024). */
#define SPARK_QWEN38_MAX_VALIDATION_MOE_ROWS 5u
#ifndef SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES 8u
#endif
#define SPARK_QWEN38_MAX_VALIDATION_KV_LANES SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES

#define SPARK_QWEN38_MAX_VAL_DK SPARK_QWEN38_MAX_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN38_MAX_VAL_DV SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN38_MAX_VAL_HEADS SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT
#define SPARK_QWEN38_MAX_VAL_CONV SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS
#define SPARK_QWEN38_MAX_VAL_GVA (SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN38_MAX_VAL_ATTN_GROUP (SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT)

extern "C" cudaError_t SparkQwen38MaxConfigureCudaKernels(void);
extern "C" cudaError_t SparkQwen38MaxLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38MaxGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38MaxGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen38MaxLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38MaxGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen38MaxLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38MaxGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen38MaxLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38MaxGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen38MaxLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38MaxAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkQwen38MaxLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38MaxKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkQwen38MaxLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen38MaxGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen38MaxLaunchGateScores(cudaStream_t stream, const SparkQwen38MaxLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern "C" cudaError_t SparkQwen38MaxLaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern "C" cudaError_t SparkQwen38MaxLaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t expert_mxfp4, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern "C" cudaError_t SparkQwen38MaxLaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen38MaxLinearView *w1, const SparkQwen38MaxLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkQwen38MaxLaunchExpertDown(cudaStream_t stream, const SparkQwen38MaxLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkQwen38MaxLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern "C" SparkStatus SparkQwen38MaxResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
extern "C" SparkStatus SparkQwen38MaxResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
extern "C" SparkStatus SparkQwen38MaxResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
extern "C" SparkStatus SparkQwen38MaxResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
extern "C" void SparkQwen38MaxResidentDecodeStageDestroy(void *module_state);

static uint32_t SparkQwen38MaxValRandomState;

static uint32_t SparkQwen38MaxValNext(void)
{
	uint32_t state = SparkQwen38MaxValRandomState;
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	SparkQwen38MaxValRandomState = state;
	return(state);
}

static float SparkQwen38MaxValUniform(float scale)
{
	return(((float)(SparkQwen38MaxValNext() & 0xffffu) / 65535.0f - 0.5f) * 2.0f * scale);
}

static uint16_t SparkQwen38MaxValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	/* Round to nearest even, matching __float2bfloat16: the 27b template's
	 * truncating converter is fine for inputs both sides read back, but the
	 * MoE oracle quantizes OUTPUTS with it, where truncation vs rounding is
	 * a 2x-ulp systematic bias. */
	bits += 0x7fffu + ((bits >> 16) & 1u);
	return((uint16_t)(bits >> 16));
}

static float SparkQwen38MaxValFromBf16(uint16_t value)
{
	float out = 0.0f;
	uint32_t bits = ((uint32_t)value) << 16;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static void SparkQwen38MaxValFillBf16(uint16_t *packed, float *exact, uint64_t count, float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		float value = SparkQwen38MaxValUniform(scale);
		if ( exact != 0 )
			exact[index] = value;
		packed[index] = SparkQwen38MaxValBf16(value);
	}
}

static int SparkQwen38MaxValFail(const char *check, const char *detail)
{
	printf("qwen38_max_validation FAIL check=%s detail=%s\n",check,detail);
	return(1);
}

static int SparkQwen38MaxValCuda(cudaError_t error, const char *check)
{
	if ( error != cudaSuccess )
	{
		printf("qwen38_max_validation FAIL check=%s cuda=%s\n",check,cudaGetErrorString(error));
		return(1);
	}
	return(0);
}

typedef struct SparkQwen38MaxValMetrics
{
	double relative_l2;
	double cosine;
	double max_absolute;
} SparkQwen38MaxValMetrics;

static void SparkQwen38MaxValMeasure(SparkQwen38MaxValMetrics *metrics, const float *actual, const float *reference, uint64_t count)
{
	double norm = 0.0,difference = 0.0,max_absolute = 0.0;
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		double delta = (double)actual[index] - (double)reference[index];
		norm += (double)reference[index] * (double)reference[index];
		difference += delta * delta;
		if ( delta < 0.0 )
			delta = -delta;
		if ( delta > max_absolute )
			max_absolute = delta;
	}
	metrics->relative_l2 = norm > 0.0 ? sqrt(difference / norm) : sqrt(difference);
	metrics->cosine = 1.0;
	metrics->max_absolute = max_absolute;
	if ( norm > 0.0 && difference > 0.0 )
	{
		double dot = 0.0,actual_norm = 0.0;
		for (index = 0u; index < count; index++)
		{
			dot += (double)actual[index] * (double)reference[index];
			actual_norm += (double)actual[index] * (double)actual[index];
		}
		metrics->cosine = dot / sqrt(actual_norm * norm);
	}
}

static int SparkQwen38MaxValReport(const char *check, const SparkQwen38MaxValMetrics *metrics, double max_relative_l2, double minimum_cosine)
{
	printf("qwen38_max_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)0,metrics->relative_l2,metrics->cosine,metrics->max_absolute);
	if ( !(metrics->relative_l2 <= max_relative_l2) || !(metrics->cosine >= minimum_cosine) )
	{
		printf("qwen38_max_validation FAIL check=%s threshold relative_l2<=%.3g cosine>=%.9g\n",check,max_relative_l2,minimum_cosine);
		return(1);
	}
	return(0);
}

/* -- fp32 CPU oracles (the 27b reference.c formulas at Max geometry) ------- */

static float SparkQwen38MaxValSilu(float value)
{
	return(value / (1.0f + expf(-value)));
}

static void SparkQwen38MaxValL2Norm(const float *input, float *output, uint32_t dimension)
{
	uint32_t element;
	float total = 0.0f;
	for (element = 0u; element < dimension; element++)
		total += input[element] * input[element];
	total = 1.0f / sqrtf(total + 1e-6f);
	for (element = 0u; element < dimension; element++)
		output[element] = input[element] * total;
}

static void SparkQwen38MaxValGdnRecurrence(const float *q, const float *k, const float *v, const float *g, const float *beta, float *state, float *output, uint32_t tokens)
{
	float qn[SPARK_QWEN38_MAX_VAL_DK],kn[SPARK_QWEN38_MAX_VAL_DK],delta[SPARK_QWEN38_MAX_VAL_DV];
	float scale = 1.0f / sqrtf((float)SPARK_QWEN38_MAX_VAL_DK),decay,kv_mem;
	uint32_t token,row,column;
	for (token = 0u; token < tokens; token++)
	{
		SparkQwen38MaxValL2Norm(q + ((uint64_t)token * SPARK_QWEN38_MAX_VAL_DK),qn,SPARK_QWEN38_MAX_VAL_DK);
		SparkQwen38MaxValL2Norm(k + ((uint64_t)token * SPARK_QWEN38_MAX_VAL_DK),kn,SPARK_QWEN38_MAX_VAL_DK);
		for (row = 0u; row < SPARK_QWEN38_MAX_VAL_DK; row++)
			qn[row] *= scale;
		decay = expf(g[token]);
		for (row = 0u; row < SPARK_QWEN38_MAX_VAL_DK; row++)
			for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
				state[(row * SPARK_QWEN38_MAX_VAL_DV) + column] *= decay;
		for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
		{
			kv_mem = 0.0f;
			for (row = 0u; row < SPARK_QWEN38_MAX_VAL_DK; row++)
				kv_mem += state[(row * SPARK_QWEN38_MAX_VAL_DV) + column] * kn[row];
			delta[column] = (v[((uint64_t)token * SPARK_QWEN38_MAX_VAL_DV) + column] - kv_mem) * beta[token];
		}
		for (row = 0u; row < SPARK_QWEN38_MAX_VAL_DK; row++)
			for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
				state[(row * SPARK_QWEN38_MAX_VAL_DV) + column] += kn[row] * delta[column];
		for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
		{
			kv_mem = 0.0f;
			for (row = 0u; row < SPARK_QWEN38_MAX_VAL_DK; row++)
				kv_mem += state[(row * SPARK_QWEN38_MAX_VAL_DV) + column] * qn[row];
			output[((uint64_t)token * SPARK_QWEN38_MAX_VAL_DV) + column] = kv_mem;
		}
	}
}

static void SparkQwen38MaxValRope(float *vector, uint32_t rope_dim, uint32_t position, float theta)
{
	uint32_t pair,half = rope_dim / 2u;
	float frequency,angle,cosine,sine,low,high;
	for (pair = 0u; pair < half; pair++)
	{
		frequency = powf(theta,-((float)(2u * pair) / (float)rope_dim));
		angle = (float)position * frequency;
		cosine = cosf(angle);
		sine = sinf(angle);
		low = vector[pair];
		high = vector[pair + half];
		vector[pair] = (low * cosine) - (high * sine);
		vector[pair + half] = (high * cosine) + (low * sine);
	}
}

static void SparkQwen38MaxValRmsNorm(const float *input, const float *weight, float *output, uint32_t dimension, float epsilon)
{
	uint32_t element;
	float variance = 0.0f,inverse;
	for (element = 0u; element < dimension; element++)
		variance += input[element] * input[element];
	inverse = 1.0f / sqrtf((variance / (float)dimension) + epsilon);
	for (element = 0u; element < dimension; element++)
		output[element] = input[element] * inverse * weight[element];
}

/* One kv head's group of query heads against its cache, fused [query|gate]
 * input, sigmoid gate on the output - the pinned modeling_qwen3_5 form. */
static void SparkQwen38MaxValAttention(const float *q_fused, const float *k_cache, const float *v_cache, const float *q_norm_weight, float *output, uint32_t group, uint32_t tokens, float epsilon)
{
	float qh[SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION],scores[SPARK_QWEN38_MAX_VALIDATION_ATTN_TOKENS],probability;
	float scale = 1.0f / sqrtf((float)SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION),maximum,total;
	uint32_t head,element,token;
	for (head = 0u; head < group; head++)
	{
		const float *fused = q_fused + ((uint64_t)head * 2u * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
		SparkQwen38MaxValRmsNorm(fused,q_norm_weight,qh,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,epsilon);
		SparkQwen38MaxValRope(qh,SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION,tokens - 1u,SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_THETA);
		maximum = -3.0e38f;
		for (token = 0u; token < tokens; token++)
		{
			probability = 0.0f;
			for (element = 0u; element < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION; element++)
				probability += qh[element] * k_cache[((uint64_t)token * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) + element];
			scores[token] = probability * scale;
			if ( scores[token] > maximum )
				maximum = scores[token];
		}
		total = 0.0f;
		for (token = 0u; token < tokens; token++)
		{
			scores[token] = expf(scores[token] - maximum);
			total += scores[token];
		}
		for (element = 0u; element < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION; element++)
		{
			probability = 0.0f;
			for (token = 0u; token < tokens; token++)
				probability += (scores[token] / total) * v_cache[((uint64_t)token * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) + element];
			output[((uint64_t)head * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) + element] =
				probability * (1.0f / (1.0f + expf(-fused[SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION + element])));
		}
	}
}

/* MXFP4-E2M1 dequant, bit-identical to SparkLmDecodeE2m1 x SparkLmDecodeE8m0:
 * magnitude table indexed by the low three nibble bits, sign in bit 3, one
 * scale byte per 32 elements at scale[row * (columns/32) + k/32]. */
static float SparkQwen38MaxValDecodeE2m1(uint32_t nibble)
{
	static const float magnitude[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float value = magnitude[nibble & 7u];
	return((nibble & 8u) != 0u ? -value : value);
}

static float SparkQwen38MaxValDecodeE8m0(uint32_t byte_value)
{
	return(byte_value == 255u ? 0.0f : exp2f((float)(int32_t)byte_value - 127.0f));
}

static float SparkQwen38MaxValDequantMxfp4(const uint8_t *payload, const uint8_t *scales, uint64_t row, uint32_t columns, uint32_t column)
{
	uint8_t pair = payload[(row * (uint64_t)(columns / 2u)) + (column >> 1u)];
	uint32_t nibble = (column & 1u) != 0u ? (pair >> 4u) : (pair & 0x0fu);
	return(SparkQwen38MaxValDecodeE2m1(nibble) * SparkQwen38MaxValDecodeE8m0(scales[(row * (uint64_t)(columns / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE)) + (column / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE)]));
}

/* E4M3 grid round-to-nearest-even (cvt.rn.satfinite semantics: values past
 * the finite max clamp to 448, subnormals step at 2^-9). */
static float SparkQwen38MaxValE4m3Quantize(float value)
{
	float sign = value < 0.0f ? -1.0f : 1.0f,magnitude = fabsf(value),grid;
	if ( magnitude < 0.015625f )
		grid = nearbyintf(magnitude / 0.001953125f) * 0.001953125f;
	else
	{
		int exponent = (int)floorf(log2f(magnitude));
		float unit,steps;
		if ( exponent > 8 )
			return(448.0f * sign);
		unit = exp2f((float)(exponent - 3));
		steps = nearbyintf(magnitude / unit);
		if ( steps >= 16.0f )
		{
			exponent += 1;
			unit = exp2f((float)(exponent - 3));
			steps = 8.0f;
		}
		grid = steps * unit;
		if ( grid > 448.0f )
			grid = 448.0f;
	}
	return(sign * grid);
}

/* The B1 kernels' activation codec: one UE8M0 scale per 128-element block
 * (scale = 2^ceil(log2(amax/448))), each element quantized to E4M3 on that
 * scale and decoded back to float. Replicated exactly so the oracle and the
 * kernel consume identical activations. */
static void SparkQwen38MaxValQdq128(float *row, uint32_t width)
{
	uint32_t base;
	for (base = 0u; base < width; base += 128u)
	{
		float amax = 0.0f,scale;
		uint32_t element;
		uint32_t span = width - base < 128u ? width - base : 128u;
		for (element = 0u; element < span; element++)
			if ( fabsf(row[base + element]) > amax )
				amax = fabsf(row[base + element]);
		scale = exp2f(ceilf(log2f(fmaxf(amax,1.0e-4f) / 448.0f)));
		for (element = 0u; element < span; element++)
			row[base + element] = SparkQwen38MaxValE4m3Quantize(row[base + element] / scale) * scale;
	}
}

static float SparkQwen38MaxValBf16Round(float value)
{
	return(SparkQwen38MaxValFromBf16(SparkQwen38MaxValBf16(value)));
}

/* -- shared device fixture ------------------------------------------------- */

typedef struct SparkQwen38MaxValDevice
{
	SparkQwen38MaxGdnLayerWeights gdn_weights;
	SparkQwen38MaxAttnLayerWeights attn_weights;
	SparkQwen38MaxGdnStatePool pool;
	uint16_t *conv_weight;      /* CONV x 4 */
	float *a_log;               /* HEADS */
	float *dt_bias;             /* HEADS */
	uint16_t *gdn_norm_weight;  /* DV */
	uint16_t *q_norm_weight;    /* ATTN_HEAD_DIM */
	uint16_t *k_norm_weight;
	float *state;               /* 2 lanes x HEADS x DK x DV */
	uint16_t *conv_tail;        /* 2 lanes x CONV x 3 */
	uint32_t *cold;             /* 2 */
	uint32_t *lane_indices;     /* 2 */
	uint16_t *qkv;              /* tokens x CONV */
	uint16_t *conv_out;         /* tokens x CONV */
	uint16_t *core_out;         /* tokens x GDN_VALUE_DIM */
	uint16_t *z_bf16;           /* tokens x GDN_VALUE_DIM */
	uint16_t *gated_out;        /* tokens x GDN_VALUE_DIM */
	uint16_t *ba_bf16;          /* tokens x HEADS x 2 (decay|beta halves) */
	float *log_decay;           /* tokens x HEADS */
	float *beta;                /* tokens x HEADS */
	float *chunk_qn;
	float *chunk_kn;
	float *chunk_cum_g;
	float *chunk_decay;
	float *chunk_attn;
	float *chunk_w;
	float *chunk_kg;
} SparkQwen38MaxValDevice;

static int SparkQwen38MaxValDeviceSetup(SparkQwen38MaxValDevice *device)
{
	uint64_t state_elements = 2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV;
	uint64_t vector_floats = (uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN38_MAX_VAL_DK;
	uint64_t matrix_floats = (uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS;
	uint32_t tokens = SPARK_QWEN38_MAX_VALIDATION_CHUNK_TOKENS;
	cudaError_t error;
	memset(device,0,sizeof(*device));
	error = cudaMalloc((void **)&device->conv_weight,(uint64_t)SPARK_QWEN38_MAX_VAL_CONV * SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->a_log,SPARK_QWEN38_MAX_VAL_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->dt_bias,SPARK_QWEN38_MAX_VAL_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gdn_norm_weight,SPARK_QWEN38_MAX_VAL_DV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->q_norm_weight,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->k_norm_weight,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->state,state_elements * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_tail,2ull * SPARK_QWEN38_MAX_VAL_CONV * (SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL - 1u) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->cold,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->lane_indices,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->qkv,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_out,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->core_out,(uint64_t)tokens * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->z_bf16,(uint64_t)tokens * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gated_out,(uint64_t)tokens * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->ba_bf16,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_HEADS * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->log_decay,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->beta,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_qn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_cum_g,(uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_decay,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_attn,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_w,(uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN38_MAX_VAL_DV * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kg,vector_floats * sizeof(float));
	if (error != cudaSuccess)
		return(SparkQwen38MaxValCuda(error,"device_alloc"));
	/* One GDN layer per lane, two lanes: the pool strides the launchers
	 * read, mirroring the module's AllocatePools at layer_count 1, tp 1. */
	device->pool.abi_version = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	device->pool.lane_capacity = 2u;
	device->pool.gdn_layer_count = 1u;
	device->pool.state_f32 = device->state;
	device->pool.state_layer_stride_elements = (uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV;
	device->pool.state_lane_stride_elements = device->pool.state_layer_stride_elements;
	device->pool.conv_tail_bf16 = device->conv_tail;
	device->pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_QWEN38_MAX_VAL_CONV * (SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL - 1u);
	device->pool.conv_tail_lane_stride_elements = device->pool.conv_tail_layer_stride_elements;
	device->pool.state_cold_by_row = device->cold;
	device->gdn_weights.conv_weight_bf16 = device->conv_weight;
	device->gdn_weights.a_log_f32 = device->a_log;
	device->gdn_weights.dt_bias_f32 = device->dt_bias;
	device->gdn_weights.gdn_norm_weight_bf16 = device->gdn_norm_weight;
	device->attn_weights.query_norm_weight_bf16 = device->q_norm_weight;
	device->attn_weights.key_norm_weight_bf16 = device->k_norm_weight;
	return(0);
}

/* -- kernel tier ----------------------------------------------------------- */

static int SparkQwen38MaxValCheckDecayBeta(SparkQwen38MaxValDevice *device)
{
	uint16_t host_ba[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS * 2u];
	float host_a[SPARK_QWEN38_MAX_VAL_HEADS],host_bias[SPARK_QWEN38_MAX_VAL_HEADS];
	float host_decay[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS];
	float host_beta[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS];
	float ref_decay[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS];
	float ref_beta[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS];
	SparkQwen38MaxValMetrics metrics;
	uint32_t row,head;
	uint64_t index;
	float shifted;
	cudaError_t error;
	SparkQwen38MaxValRandomState = 11u;
	SparkQwen38MaxValFillBf16(host_ba,0,SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS * 2u,2.0f);
	for (head = 0u; head < SPARK_QWEN38_MAX_VAL_HEADS; head++)
	{
		host_a[head] = SparkQwen38MaxValUniform(1.0f);
		host_bias[head] = SparkQwen38MaxValUniform(1.0f);
	}
	error = cudaMemcpy(device->ba_bf16,host_ba,sizeof(host_ba),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->a_log,host_a,sizeof(host_a),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->dt_bias,host_bias,sizeof(host_bias),cudaMemcpyHostToDevice);
	/* decay_pre and beta_pre are separate rows x HEADS buffers; ba_bf16
	 * holds them back to back. */
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchDecayBeta(cudaStreamPerThread,device->ba_bf16,device->ba_bf16 + SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS,&device->gdn_weights,device->log_decay,device->beta,SPARK_QWEN38_MAX_VALIDATION_ROWS,1u);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(host_decay,device->log_decay,sizeof(host_decay),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(host_beta,device->beta,sizeof(host_beta),cudaMemcpyDeviceToHost);
	if (SparkQwen38MaxValCuda(error,"decay_beta") != 0)
		return(1);
	for (row = 0u; row < SPARK_QWEN38_MAX_VALIDATION_ROWS; row++)
		for (head = 0u; head < SPARK_QWEN38_MAX_VAL_HEADS; head++)
		{
			index = ((uint64_t)row * SPARK_QWEN38_MAX_VAL_HEADS) + head;
			shifted = SparkQwen38MaxValFromBf16(host_ba[index]) + host_bias[head];
			ref_decay[index] = -expf(host_a[head]) * (shifted > 20.0f ? shifted : logf(1.0f + expf(shifted)));
			ref_beta[index] = 1.0f / (1.0f + expf(-SparkQwen38MaxValFromBf16(host_ba[SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS + index])));
		}
	SparkQwen38MaxValMeasure(&metrics,host_decay,ref_decay,SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS);
	if (SparkQwen38MaxValReport("decay_gate",&metrics,1e-5,0.99999999) != 0)
		return(1);
	SparkQwen38MaxValMeasure(&metrics,host_beta,ref_beta,SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_VAL_HEADS);
	return(SparkQwen38MaxValReport("write_gate",&metrics,1e-5,0.99999999));
}

/* One GDN recurrence step per row: row 0 cold, row 1 warm with a random
 * resident state, compared against the per-head oracle recurrence. The GVA
 * ratio is 8:1 at this geometry (key head = value head / 8). */
static int SparkQwen38MaxValCheckGdnStep(SparkQwen38MaxValDevice *device)
{
	uint64_t state_elements = (uint64_t)SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV;
	uint16_t *host_conv = (uint16_t *)calloc(2ull * SPARK_QWEN38_MAX_VAL_CONV,sizeof(uint16_t));
	float *exact = (float *)calloc(2ull * SPARK_QWEN38_MAX_VAL_CONV,sizeof(float));
	float *state_host = (float *)calloc(2ull * state_elements,sizeof(float));
	float *state_reference = (float *)calloc(2ull * state_elements,sizeof(float));
	float *oracle_out = (float *)calloc(2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	float *actual = (float *)calloc(2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	uint16_t *core_packed = (uint16_t *)calloc(2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV,sizeof(uint16_t));
	uint32_t lanes[2] = {0u,1u};
	uint32_t cold[2] = {1u,0u};
	SparkQwen38MaxValMetrics metrics;
	uint32_t row,head;
	cudaError_t error;
	if (host_conv == 0 || exact == 0 || state_host == 0 || state_reference == 0 || oracle_out == 0 || actual == 0 || core_packed == 0)
		return(SparkQwen38MaxValFail("gdn_step","host_alloc"));
	SparkQwen38MaxValRandomState = 37u;
	SparkQwen38MaxValFillBf16(host_conv,exact,2ull * SPARK_QWEN38_MAX_VAL_CONV,1.0f);
	/* The kernel reads bf16-rounded projections; the oracle must too, or
	 * the comparison measures bf16 storage noise, not the recurrence. */
	{
		uint64_t index;
		for (index = 0u; index < 2ull * SPARK_QWEN38_MAX_VAL_CONV; index++)
			exact[index] = SparkQwen38MaxValFromBf16(host_conv[index]);
	}
	{
		uint64_t index;
		for (index = 0u; index < state_elements; index++)
			state_host[state_elements + index] = SparkQwen38MaxValUniform(0.25f); /* lane 1 warm */
		memcpy(state_reference,state_host,2ull * state_elements * sizeof(float));
	}
	{
		float host_log_decay[2 * SPARK_QWEN38_MAX_VAL_HEADS],host_beta[2 * SPARK_QWEN38_MAX_VAL_HEADS];
		uint32_t head_index;
		for (head_index = 0u; head_index < 2u * SPARK_QWEN38_MAX_VAL_HEADS; head_index++)
		{
			host_log_decay[head_index] = SparkQwen38MaxValUniform(0.5f) - 0.5f;
			host_beta[head_index] = 0.25f + fabsf(SparkQwen38MaxValUniform(0.5f));
		}
		error = cudaMemcpy(device->state,state_host,2ull * state_elements * sizeof(float),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->qkv,host_conv,2ull * SPARK_QWEN38_MAX_VAL_CONV * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,host_log_decay,sizeof(host_log_decay),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->beta,host_beta,sizeof(host_beta),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
		if (error == cudaSuccess)
			error = SparkQwen38MaxLaunchGdnStep(cudaStreamPerThread,device->qkv,device->log_decay,device->beta,&device->pool,device->core_out,device->lane_indices,2u,0u,1u);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		if (error == cudaSuccess) error = cudaMemcpy(core_packed,device->core_out,2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (error == cudaSuccess) error = cudaMemcpy(state_host,device->state,2ull * state_elements * sizeof(float),cudaMemcpyDeviceToHost);
		if (SparkQwen38MaxValCuda(error,"gdn_step") != 0)
		{
			free(host_conv); free(exact); free(state_host); free(state_reference);
			free(oracle_out); free(actual); free(core_packed);
			return(1);
		}
		for (row = 0u; row < 2u; row++)
			for (head = 0u; head < SPARK_QWEN38_MAX_VAL_HEADS; head++)
			{
				uint32_t key_head = head / SPARK_QWEN38_MAX_VAL_GVA;
				const float *q = exact + ((uint64_t)row * SPARK_QWEN38_MAX_VAL_CONV) + ((uint64_t)key_head * SPARK_QWEN38_MAX_VAL_DK);
				const float *k = q + SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION;
				const float *v = exact + ((uint64_t)row * SPARK_QWEN38_MAX_VAL_CONV) + (2ull * SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV);
				float g = cold[row] != 0u ? -30.0f : host_log_decay[(row * SPARK_QWEN38_MAX_VAL_HEADS) + head];
				float beta = host_beta[(row * SPARK_QWEN38_MAX_VAL_HEADS) + head];
				SparkQwen38MaxValGdnRecurrence(q,k,v,&g,&beta,
					state_reference + ((uint64_t)row * state_elements) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV),
					oracle_out + ((uint64_t)row * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV),1u);
			}
		{
			uint64_t index;
			for (index = 0u; index < 2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV; index++)
				actual[index] = SparkQwen38MaxValFromBf16(core_packed[index]);
		}
	}
	SparkQwen38MaxValMeasure(&metrics,actual,oracle_out,2ull * SPARK_QWEN38_MAX_VAL_HEADS * SPARK_QWEN38_MAX_VAL_DV);
	if (SparkQwen38MaxValReport("gdn_step_output",&metrics,5e-3,0.99999) != 0)
	{
		free(host_conv); free(exact); free(state_host); free(state_reference);
		free(oracle_out); free(actual); free(core_packed);
		return(1);
	}
	SparkQwen38MaxValMeasure(&metrics,state_host,state_reference,2ull * state_elements);
	free(host_conv); free(exact); free(state_host); free(state_reference);
	free(oracle_out); free(actual); free(core_packed);
	return(SparkQwen38MaxValReport("gdn_step_state",&metrics,1e-3,0.999999));
}

/* The rank-local TP4 GDN step: rank 0 owns value heads [0,32) with key
 * heads [0,4), conv channels q512|k512|v4096. Same oracle over the rank's
 * head slice - the whole-head cut must be invisible to the math. */
static int SparkQwen38MaxValCheckGdnStepTp4(void)
{
	const uint32_t tp = 4u,local_heads = SPARK_QWEN38_MAX_VAL_HEADS / tp;
	const uint32_t local_qk = (SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT / tp) * SPARK_QWEN38_MAX_VAL_DK;
	const uint32_t local_conv = (2u * local_qk) + (local_heads * SPARK_QWEN38_MAX_VAL_DV);
	uint64_t state_elements = (uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV;
	SparkQwen38MaxValDevice device;
	uint16_t *host_conv = (uint16_t *)calloc(local_conv,sizeof(uint16_t));
	float *exact = (float *)calloc(local_conv,sizeof(float));
	float *state_host = (float *)calloc(state_elements,sizeof(float));
	float *oracle_out = (float *)calloc((uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	float *actual = (float *)calloc((uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	uint16_t *core_packed = (uint16_t *)calloc((uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(uint16_t));
	uint32_t lanes[1] = {0u};
	uint32_t cold[1] = {0u};
	float host_log_decay[SPARK_QWEN38_MAX_VAL_HEADS / 4u],host_beta[SPARK_QWEN38_MAX_VAL_HEADS / 4u];
	SparkQwen38MaxValMetrics metrics;
	uint32_t head;
	cudaError_t error;
	if (host_conv == 0 || exact == 0 || state_host == 0 || oracle_out == 0 || actual == 0 || core_packed == 0)
		return(SparkQwen38MaxValFail("gdn_step_tp4","host_alloc"));
	if (SparkQwen38MaxValDeviceSetup(&device) != 0)
		return(1);
	/* Rebind the pool and weight pointers to rank-local widths. */
	device.pool.state_layer_stride_elements = state_elements;
	device.pool.state_lane_stride_elements = state_elements;
	device.pool.conv_tail_layer_stride_elements = (uint64_t)local_conv * (SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL - 1u);
	device.pool.conv_tail_lane_stride_elements = device.pool.conv_tail_layer_stride_elements;
	SparkQwen38MaxValRandomState = 73u;
	SparkQwen38MaxValFillBf16(host_conv,exact,local_conv,1.0f);
	{
		uint32_t index;
		for (index = 0u; index < local_conv; index++)
			exact[index] = SparkQwen38MaxValFromBf16(host_conv[index]);
	}
	{
		uint64_t index;
		for (index = 0u; index < state_elements; index++)
			state_host[index] = SparkQwen38MaxValUniform(0.25f);
	}
	for (head = 0u; head < local_heads; head++)
	{
		host_log_decay[head] = SparkQwen38MaxValUniform(0.5f) - 0.5f;
		host_beta[head] = 0.25f + fabsf(SparkQwen38MaxValUniform(0.5f));
	}
	error = cudaMemcpy(device.state,state_host,state_elements * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device.qkv,host_conv,local_conv * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device.log_decay,host_log_decay,sizeof(host_log_decay),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device.beta,host_beta,sizeof(host_beta),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device.cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device.lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchGdnStep(cudaStreamPerThread,device.qkv,device.log_decay,device.beta,&device.pool,device.core_out,device.lane_indices,1u,0u,tp);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(core_packed,device.core_out,(uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SparkQwen38MaxValCuda(error,"gdn_step_tp4") != 0)
	{
		cudaFree(device.state); free(host_conv); free(exact); free(state_host); free(oracle_out); free(actual); free(core_packed);
		return(1);
	}
	for (head = 0u; head < local_heads; head++)
	{
		uint32_t key_head = head / SPARK_QWEN38_MAX_VAL_GVA;
		const float *q = exact + ((uint64_t)key_head * SPARK_QWEN38_MAX_VAL_DK);
		const float *k = q + local_qk;
		const float *v = exact + (2ull * local_qk) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV);
		SparkQwen38MaxValGdnRecurrence(q,k,v,&host_log_decay[head],&host_beta[head],
			state_host + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV),
			oracle_out + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV),1u);
	}
	{
		uint64_t index;
		for (index = 0u; index < (uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV; index++)
			actual[index] = SparkQwen38MaxValFromBf16(core_packed[index]);
	}
	SparkQwen38MaxValMeasure(&metrics,actual,oracle_out,(uint64_t)local_heads * SPARK_QWEN38_MAX_VAL_DV);
	{
		/* Free the fixture's allocations: the setup helper has no teardown,
		 * so release the big ones explicitly to keep the tier's footprint
		 * bounded for the checks that follow. */
		void *marks[] = {device.conv_weight,device.a_log,device.dt_bias,device.gdn_norm_weight,device.q_norm_weight,device.k_norm_weight,device.state,device.conv_tail,device.cold,device.lane_indices,device.qkv,device.conv_out,device.core_out,device.z_bf16,device.gated_out,device.ba_bf16,device.log_decay,device.beta,device.chunk_qn,device.chunk_kn,device.chunk_cum_g,device.chunk_decay,device.chunk_attn,device.chunk_w,device.chunk_kg};
		uint32_t mark;
		for (mark = 0u; mark < sizeof(marks) / sizeof(marks[0]); mark++)
			if ( marks[mark] != 0 )
				cudaFree(marks[mark]);
	}
	free(host_conv); free(exact); free(state_host); free(oracle_out); free(actual); free(core_packed);
	return(SparkQwen38MaxValReport("gdn_step_tp4",&metrics,5e-3,0.99999));
}

static int SparkQwen38MaxValCheckGatedNorm(SparkQwen38MaxValDevice *device)
{
	uint64_t elements = (uint64_t)SPARK_QWEN38_MAX_VALIDATION_ROWS * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION;
	uint16_t *packed = (uint16_t *)calloc(elements * 2u,sizeof(uint16_t));
	float *exact = (float *)calloc(elements * 2u,sizeof(float));
	float *expected = (float *)calloc(elements,sizeof(float));
	float *actual = (float *)calloc(elements,sizeof(float));
	uint16_t *norm_packed = (uint16_t *)calloc(SPARK_QWEN38_MAX_VAL_DV,sizeof(uint16_t));
	float *norm_exact = (float *)calloc(SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	SparkQwen38MaxValMetrics metrics;
	uint32_t row,head,column;
	cudaError_t error;
	if (packed == 0 || exact == 0 || expected == 0 || actual == 0 || norm_packed == 0 || norm_exact == 0)
		return(SparkQwen38MaxValFail("gated_norm","host_alloc"));
	SparkQwen38MaxValRandomState = 51u;
	SparkQwen38MaxValFillBf16(packed,exact,elements,1.0f);              /* core */
	SparkQwen38MaxValFillBf16(packed + elements,exact + elements,elements,1.0f); /* z */
	SparkQwen38MaxValFillBf16(norm_packed,norm_exact,SPARK_QWEN38_MAX_VAL_DV,0.5f);
	/* Round-trip the oracle inputs through bf16: the kernel reads the
	 * packed values, and the comparison must isolate the kernel's math. */
	{
		uint64_t index;
		for (index = 0u; index < elements * 2u; index++)
			exact[index] = SparkQwen38MaxValFromBf16(packed[index]);
	}
	{
		uint32_t index;
		for (index = 0u; index < SPARK_QWEN38_MAX_VAL_DV; index++)
			norm_exact[index] = SparkQwen38MaxValFromBf16(norm_packed[index]);
	}
	error = cudaMemcpy(device->core_out,packed,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->z_bf16,packed + elements,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->gdn_norm_weight,norm_packed,SPARK_QWEN38_MAX_VAL_DV * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchGatedNorm(cudaStreamPerThread,device->core_out,device->z_bf16,&device->gdn_weights,device->gated_out,SPARK_QWEN38_MAX_VALIDATION_ROWS,SPARK_QWEN38_MAX_MODEL_RMS_NORM_EPSILON,1u);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(packed,device->gated_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SparkQwen38MaxValCuda(error,"gated_norm") != 0)
	{
		free(packed); free(exact); free(expected); free(actual); free(norm_packed); free(norm_exact);
		return(1);
	}
	for (row = 0u; row < SPARK_QWEN38_MAX_VALIDATION_ROWS; row++)
		for (head = 0u; head < SPARK_QWEN38_MAX_VAL_HEADS; head++)
		{
			float variance = 0.0f,inverse;
			for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
			{
				uint64_t index = ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV) + column;
				variance += exact[index] * exact[index];
			}
			inverse = 1.0f / sqrtf((variance / (float)SPARK_QWEN38_MAX_VAL_DV) + SPARK_QWEN38_MAX_MODEL_RMS_NORM_EPSILON);
			for (column = 0u; column < SPARK_QWEN38_MAX_VAL_DV; column++)
			{
				uint64_t index = ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV) + column;
				expected[index] = exact[index] * inverse * norm_exact[column] * SparkQwen38MaxValSilu(exact[elements + index]);
				actual[index] = SparkQwen38MaxValFromBf16(packed[index]);
			}
		}
	SparkQwen38MaxValMeasure(&metrics,actual,expected,elements);
	free(packed); free(exact); free(expected); free(actual); free(norm_packed); free(norm_exact);
	/* The gated norm's output is bf16 (like the GDN step's), so five nines is
	 * the family's established bound for this check - qwen38_27b and
	 * qwen4_flash both measure cosine ~0.9999986 here on passing runs. */
	return(SparkQwen38MaxValReport("gated_norm",&metrics,2e-3,0.99999));
}

/* Attention prepare + decode against the fused-gate GQA oracle: one row,
 * all 64 heads over one kv head group each, 5-token context from the
 * module's own cache write. */
static int SparkQwen38MaxValCheckAttention(SparkQwen38MaxValDevice *device)
{
	const uint32_t tokens = SPARK_QWEN38_MAX_VALIDATION_ATTN_TOKENS;
	const uint32_t kv_heads = SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT;
	const uint64_t token_elements = 2ull * kv_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t cache_elements = (uint64_t)tokens * token_elements;
	uint16_t *q_packed = (uint16_t *)calloc(2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,sizeof(uint16_t));
	float *q_exact = (float *)calloc(2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,sizeof(float));
	uint16_t *k_packed = (uint16_t *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION,sizeof(uint16_t));
	float *k_exact = (float *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION,sizeof(float));
	uint16_t *v_packed = (uint16_t *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION,sizeof(uint16_t));
	uint16_t *qn_packed = (uint16_t *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(uint16_t));
	float *qn_exact = (float *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
	uint16_t *kn_packed = (uint16_t *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(uint16_t));
	float *kn_exact = (float *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
	uint16_t *cache = (uint16_t *)calloc(cache_elements,sizeof(uint16_t));
	uint16_t *head_out = (uint16_t *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,sizeof(uint16_t));
	float *expected = (float *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,sizeof(float));
	float *actual = (float *)calloc(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,sizeof(float));
	uint32_t slot_mapping[1];
	uint64_t positions[1];
	uint32_t context_lengths[1];
	uint32_t lane_indices[1];
	uint32_t zero_lane = 0u;
	SparkQwen38MaxValMetrics metrics;
	cudaError_t error;
	if (q_packed == 0 || q_exact == 0 || k_packed == 0 || k_exact == 0 || v_packed == 0 || qn_packed == 0 || qn_exact == 0 || kn_packed == 0 || kn_exact == 0 || cache == 0 || head_out == 0 || expected == 0 || actual == 0)
		return(SparkQwen38MaxValFail("attn_decode","host_alloc"));
	(void)k_exact;
	SparkQwen38MaxValRandomState = 67u;
	SparkQwen38MaxValFillBf16(q_packed,q_exact,2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION,1.0f);
	SparkQwen38MaxValFillBf16(k_packed,0,SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION,1.0f);
	SparkQwen38MaxValFillBf16(v_packed,0,SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION,1.0f);
	SparkQwen38MaxValFillBf16(qn_packed,qn_exact,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,0.5f);
	SparkQwen38MaxValFillBf16(kn_packed,kn_exact,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,0.5f);
	{
		uint64_t index;
		for (index = 0u; index < 2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION; index++)
			q_exact[index] = SparkQwen38MaxValFromBf16(q_packed[index]);
		for (index = 0u; index < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION; index++)
		{
			qn_exact[index] = SparkQwen38MaxValFromBf16(qn_packed[index]);
			kn_exact[index] = SparkQwen38MaxValFromBf16(kn_packed[index]);
		}
	}
	error = cudaMemcpy(device->qkv,q_packed,2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->core_out,k_packed,SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->gated_out,v_packed,SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->q_norm_weight,qn_packed,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->k_norm_weight,kn_packed,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	slot_mapping[0] = tokens - 1u;
	positions[0] = tokens - 1u;
	context_lengths[0] = tokens;
	lane_indices[0] = 0u;
	(void)lane_indices;
	/* Stage the kv cache through five consecutive single-row prepares at
	 * positions 0..4, exactly how the decode path fills it. */
	void *kv_cache = 0;
	error = cudaMalloc(&kv_cache,cache_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMemset(kv_cache,0,cache_elements * sizeof(uint16_t));
	{
		uint32_t token;
		for (token = 0u; token < tokens && error == cudaSuccess; token++)
		{
			uint32_t slots[1];
			uint64_t pos[1];
			slots[0] = token;
			pos[0] = token;
			/* Prepare normalizes and rope's the query IN PLACE, so each
			 * staging iteration must start from the pristine projection -
			 * only the final pass's query survives to the decode. */
			error = cudaMemcpy(device->qkv,q_packed,2ull * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
			if (error == cudaSuccess)
				error = SparkQwen38MaxLaunchAttnPrepare(cudaStreamPerThread,device->qkv,device->core_out,device->gated_out,&device->attn_weights,kv_cache,slots,pos,1u,0u,token_elements,cache_elements,SPARK_QWEN38_MAX_MODEL_RMS_NORM_EPSILON,1u,0u);
		}
	}
	if (error == cudaSuccess)
	{
		SparkQwen38MaxKvBlockTableView table;
		uint32_t blocks[1] = {0u};
		uint32_t counts[1] = {1u};
		uint32_t *device_blocks,*device_counts;
		error = cudaMalloc((void **)&device_blocks,sizeof(blocks));
		if (error == cudaSuccess) error = cudaMemcpy(device_blocks,blocks,sizeof(blocks),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMalloc((void **)&device_counts,sizeof(counts));
		if (error == cudaSuccess) error = cudaMemcpy(device_counts,counts,sizeof(counts),cudaMemcpyHostToDevice);
		memset(&table,0,sizeof(table));
		table.abi_version = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
		table.descriptor_bytes = sizeof(table);
		table.block_token_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		table.lane_count = 1u;
		table.lane_stride = 1u;
		table.lane_capacity = 1u;
		table.physical_block_indices = device_blocks;
		table.lane_physical_block_counts = device_counts;
		table.host_physical_block_indices = blocks;
		table.host_lane_physical_block_counts = counts;
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,&zero_lane,sizeof(zero_lane),cudaMemcpyHostToDevice);
		if (error == cudaSuccess)
			error = SparkQwen38MaxLaunchAttnDecode(cudaStreamPerThread,device->qkv,kv_cache,&table,device->lane_indices,context_lengths,device->conv_out,1u,0u,token_elements,cache_elements,1u,0u);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		if (error == cudaSuccess) error = cudaMemcpy(head_out,device->conv_out,SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (error == cudaSuccess) error = cudaMemcpy(cache,kv_cache,cache_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (error == cudaSuccess)
		{
			cudaFree(device_blocks);
			cudaFree(device_counts);
		}
	}
	if (SparkQwen38MaxValCuda(error,"attn_decode") != 0)
	{
		if (kv_cache != 0) cudaFree(kv_cache);
		free(q_packed); free(q_exact); free(k_packed); free(k_exact); free(v_packed); free(qn_packed); free(qn_exact); free(kn_packed); free(kn_exact); free(cache); free(head_out); free(expected); free(actual);
		return(1);
	}
	/* Oracle: per kv head, its group of query heads attends over the staged
	 * cache rows; the fused gate is the sigmoid of the query half. */
	{
		uint32_t kv_head;
		for (kv_head = 0u; kv_head < kv_heads; kv_head++)
		{
			float *k_rows = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
			float *v_rows = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
			uint32_t token,element;
			for (token = 0u; token < tokens; token++)
				for (element = 0u; element < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION; element++)
				{
					uint64_t base = ((uint64_t)token * token_elements) + ((uint64_t)kv_head * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) + element;
					k_rows[(uint64_t)token * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION + element] = SparkQwen38MaxValFromBf16(cache[base]);
					v_rows[(uint64_t)token * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION + element] = SparkQwen38MaxValFromBf16(cache[base + ((uint64_t)kv_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION)]);
				}
			SparkQwen38MaxValAttention(
				q_exact + ((uint64_t)kv_head * SPARK_QWEN38_MAX_VAL_ATTN_GROUP * 2u * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION),
				k_rows,v_rows,qn_exact,
				expected + ((uint64_t)kv_head * SPARK_QWEN38_MAX_VAL_ATTN_GROUP * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION),
				SPARK_QWEN38_MAX_VAL_ATTN_GROUP,tokens,SPARK_QWEN38_MAX_MODEL_RMS_NORM_EPSILON);
			free(k_rows); free(v_rows);
		}
		{
			uint64_t index;
			for (index = 0u; index < SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION; index++)
				actual[index] = SparkQwen38MaxValFromBf16(head_out[index]);
		}
	}
	SparkQwen38MaxValMeasure(&metrics,actual,expected,SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION);
	if (kv_cache != 0) cudaFree(kv_cache);
	free(q_packed); free(q_exact); free(k_packed); free(k_exact); free(v_packed); free(qn_packed); free(qn_exact); free(kn_packed); free(kn_exact); free(cache); free(head_out); free(expected); free(actual);
	return(SparkQwen38MaxValReport("attn_decode",&metrics,5e-3,0.99999));
}

/* The chunked prefill core vs the recurrence oracle over 64 tokens, one
 * lane, warm start: the module's prefill formulation, proven at the kernel
 * tier (the module refuses prefill frames by design). */
static int SparkQwen38MaxValCheckGdnChunk(SparkQwen38MaxValDevice *device)
{
	const uint32_t tokens = SPARK_QWEN38_MAX_VALIDATION_CHUNK_TOKENS;
	const uint32_t heads = SPARK_QWEN38_MAX_VAL_HEADS;
	uint64_t state_elements = (uint64_t)heads * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV;
	uint16_t *host_conv = (uint16_t *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV,sizeof(uint16_t));
	float *exact = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV,sizeof(float));
	float *state_host = (float *)calloc(state_elements,sizeof(float));
	float *state_reference = (float *)calloc(state_elements,sizeof(float));
	float *oracle_out = (float *)calloc((uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	float *actual = (float *)calloc((uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
	uint16_t *core_packed = (uint16_t *)calloc((uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(uint16_t));
	float *host_log_decay = (float *)calloc((uint64_t)tokens * heads,sizeof(float));
	float *host_beta = (float *)calloc((uint64_t)tokens * heads,sizeof(float));
	uint32_t cold[2] = {0u,0u};
	SparkQwen38MaxValMetrics metrics;
	uint32_t head;
	cudaError_t error;
	if (host_conv == 0 || exact == 0 || state_host == 0 || state_reference == 0 || oracle_out == 0 || actual == 0 || core_packed == 0 || host_log_decay == 0 || host_beta == 0)
		return(SparkQwen38MaxValFail("gdn_chunk","host_alloc"));
	SparkQwen38MaxValRandomState = 91u;
	SparkQwen38MaxValFillBf16(host_conv,exact,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV,1.0f);
	{
		uint64_t index;
		for (index = 0u; index < (uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV; index++)
			exact[index] = SparkQwen38MaxValFromBf16(host_conv[index]);
	}
	{
		uint64_t index;
		for (index = 0u; index < state_elements; index++)
			state_host[index] = SparkQwen38MaxValUniform(0.25f);
		memcpy(state_reference,state_host,state_elements * sizeof(float));
		for (index = 0u; index < (uint64_t)tokens * heads; index++)
		{
			host_log_decay[index] = SparkQwen38MaxValUniform(0.5f) - 0.75f;
			host_beta[index] = 0.25f + fabsf(SparkQwen38MaxValUniform(0.5f));
		}
	}
	error = cudaMemcpy(device->state,state_host,state_elements * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->conv_out,host_conv,(uint64_t)tokens * SPARK_QWEN38_MAX_VAL_CONV * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,host_log_decay,(uint64_t)tokens * heads * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->beta,host_beta,(uint64_t)tokens * heads * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchGdnChunk(cudaStreamPerThread,device->conv_out,device->log_decay,device->beta,device->chunk_qn,device->chunk_kn,device->chunk_cum_g,device->chunk_decay,device->chunk_attn,device->chunk_w,device->chunk_kg,&device->pool,device->core_out,0u,tokens,0u,1u);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(core_packed,device->core_out,(uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(state_host,device->state,state_elements * sizeof(float),cudaMemcpyDeviceToHost);
	if (SparkQwen38MaxValCuda(error,"gdn_chunk") != 0)
	{
		free(host_conv); free(exact); free(state_host); free(state_reference); free(oracle_out); free(actual); free(core_packed); free(host_log_decay); free(host_beta);
		return(1);
	}
	for (head = 0u; head < heads; head++)
	{
		uint32_t key_head = head / SPARK_QWEN38_MAX_VAL_GVA;
		uint32_t token;
		/* The oracle consumes per-token q/k/v slices for this head. */
		float *q = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_VAL_DK,sizeof(float));
		float *k = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_VAL_DK,sizeof(float));
		float *v = (float *)calloc((uint64_t)tokens * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
		float *g = (float *)calloc(tokens,sizeof(float));
		float *beta = (float *)calloc(tokens,sizeof(float));
		for (token = 0u; token < tokens; token++)
		{
			uint64_t row = (uint64_t)token * SPARK_QWEN38_MAX_VAL_CONV;
			uint32_t element;
			for (element = 0u; element < SPARK_QWEN38_MAX_VAL_DK; element++)
			{
				q[(uint64_t)token * SPARK_QWEN38_MAX_VAL_DK + element] = exact[row + ((uint64_t)key_head * SPARK_QWEN38_MAX_VAL_DK) + element];
				k[(uint64_t)token * SPARK_QWEN38_MAX_VAL_DK + element] = exact[row + SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION + ((uint64_t)key_head * SPARK_QWEN38_MAX_VAL_DK) + element];
			}
			for (element = 0u; element < SPARK_QWEN38_MAX_VAL_DV; element++)
				v[(uint64_t)token * SPARK_QWEN38_MAX_VAL_DV + element] = exact[row + (2ull * SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DV) + element];
			g[token] = host_log_decay[(uint64_t)token * heads + head];
			beta[token] = host_beta[(uint64_t)token * heads + head];
		}
		SparkQwen38MaxValGdnRecurrence(q,k,v,g,beta,
			state_reference + ((uint64_t)head * SPARK_QWEN38_MAX_VAL_DK * SPARK_QWEN38_MAX_VAL_DV),
			oracle_out + ((uint64_t)head * tokens * SPARK_QWEN38_MAX_VAL_DV),tokens);
		free(q); free(k); free(v); free(g); free(beta);
	}
	{
		uint64_t index;
		for (index = 0u; index < (uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV; index++)
			actual[index] = SparkQwen38MaxValFromBf16(core_packed[index]);
	}
	/* The chunk output layout is [token][head][dv] on device but the oracle
	 * walked [head][token][dv]; repack the oracle for the comparison. */
	{
		float *repacked = (float *)calloc((uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV,sizeof(float));
		uint32_t token;
		for (token = 0u; token < tokens; token++)
			for (head = 0u; head < heads; head++)
				memcpy(repacked + ((uint64_t)token * heads + head) * SPARK_QWEN38_MAX_VAL_DV,
					oracle_out + ((uint64_t)head * tokens + token) * SPARK_QWEN38_MAX_VAL_DV,
					SPARK_QWEN38_MAX_VAL_DV * sizeof(float));
		SparkQwen38MaxValMeasure(&metrics,actual,repacked,(uint64_t)tokens * heads * SPARK_QWEN38_MAX_VAL_DV);
		free(repacked);
	}
	if (SparkQwen38MaxValReport("gdn_chunk_output",&metrics,5e-3,0.99999) != 0)
	{
		free(host_conv); free(exact); free(state_host); free(state_reference); free(oracle_out); free(actual); free(core_packed); free(host_log_decay); free(host_beta);
		return(1);
	}
	SparkQwen38MaxValMeasure(&metrics,state_host,state_reference,state_elements);
	free(host_conv); free(exact); free(state_host); free(state_reference); free(oracle_out); free(actual); free(core_packed); free(host_log_decay); free(host_beta);
	return(SparkQwen38MaxValReport("gdn_chunk_state",&metrics,1e-3,0.999999));
}

/* -- module tier ----------------------------------------------------------- */

typedef struct SparkQwen38MaxValCapture
{
	uint16_t hidden[SPARK_QWEN38_MAX_VALIDATION_KV_LANES * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION];
	uint32_t sends;
} SparkQwen38MaxValCapture;

static SparkStatus SparkQwen38MaxValCaptureSend(SparkHiddenTransportSession *session, const SparkHiddenTransportPacket *packet)
{
	SparkQwen38MaxValCapture *capture = (SparkQwen38MaxValCapture *)session;
	uint64_t bytes = (uint64_t)packet->active_sequence_count * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION * 2u;
	if (packet->hidden_bf16 == 0 || packet->active_sequence_count > SPARK_QWEN38_MAX_VALIDATION_KV_LANES ||
		packet->hidden_dimension != SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION)
		return(SPARK_STATUS_VALIDATION_FAILED);
	if (cudaMemcpy(capture->hidden,packet->hidden_bf16,bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
		return(SPARK_STATUS_IO_ERROR);
	capture->sends++;
	return(SPARK_STATUS_OK);
}

typedef struct SparkQwen38MaxValModule
{
	void *state;
	uint32_t token_ids[SPARK_QWEN38_MAX_VALIDATION_KV_LANES];
	uint32_t output_token_ids[SPARK_QWEN38_MAX_VALIDATION_KV_LANES];
	uint32_t head_stage;
	uint32_t host_blocks[SPARK_QWEN38_MAX_VALIDATION_KV_LANES];
	uint32_t host_counts[SPARK_QWEN38_MAX_VALIDATION_KV_LANES];
	uint32_t *device_blocks;
	uint32_t *device_counts;
	SparkQwen38MaxKvBlockTableView table;
	SparkQwen38MaxResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkQwen38MaxValCapture capture;
} SparkQwen38MaxValModule;

static int SparkQwen38MaxValModuleInitialize(SparkQwen38MaxValModule *module)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkStatus status;
	const char *stage_count_text;
	uint32_t lane;
	cudaError_t error;
	memset(module,0,sizeof(*module));
	/* head_stage: the module owns the final head iff it is the whole-stack
	 * last stage (STAGE_COUNT == 1). TP_DEGREE 1 is the whole-stack tier
	 * here; a tp>1 module tier needs the collective (fleet window). */
	stage_count_text = getenv("SPARK_QWEN38_MAX_STAGE_COUNT");
	module->head_stage = stage_count_text != 0 && strcmp(stage_count_text,"1") == 0 ? 1u : 0u;
	for (lane = 0u; lane < SPARK_QWEN38_MAX_VALIDATION_KV_LANES; lane++)
	{
		module->host_blocks[lane] = lane;
		module->host_counts[lane] = 1u;
	}
	error = cudaMalloc((void **)&module->device_blocks,sizeof(module->host_blocks));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_blocks,module->host_blocks,sizeof(module->host_blocks),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMalloc((void **)&module->device_counts,sizeof(module->host_counts));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_counts,module->host_counts,sizeof(module->host_counts),cudaMemcpyHostToDevice);
	if (SparkQwen38MaxValCuda(error,"module_table_alloc") != 0)
		return(1);
	module->table.abi_version = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	module->table.descriptor_bytes = sizeof(module->table);
	module->table.block_token_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	module->table.lane_count = SPARK_QWEN38_MAX_VALIDATION_KV_LANES;
	module->table.lane_stride = 1u;
	module->table.lane_capacity = SPARK_QWEN38_MAX_VALIDATION_KV_LANES;
	module->table.physical_block_indices = module->device_blocks;
	module->table.lane_physical_block_counts = module->device_counts;
	module->table.host_physical_block_indices = module->host_blocks;
	module->table.host_lane_physical_block_counts = module->host_counts;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "Qwen/Qwen3.8-2.4T-A95B";
	configuration.model_revision = "validation";
	configuration.stage_name = "qwen38_max_resident_decode_stage";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "qwen38_max_resident_decode_stage";
	configuration.configuration_json = "{}";
	configuration.configuration_json_bytes = 2u;
	memset(&host_services,0,sizeof(host_services));
	host_services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services.descriptor_bytes = sizeof(host_services);
	host_services.node_id = "spark-qwen38_max-validator";
	host_services.node_target = "cuda.sm121.qwen38.resident_decode_stage.fp8";
	host_services.execution_stream = (void *)cudaStreamPerThread;
	status = SparkQwen38MaxResidentDecodeStageInitialize(&configuration,&host_services,&module->state);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"qwen38_max_validation failure=module_initialize status=%d\n",(int)status);
		return(1);
	}
	return(0);
}

static int SparkQwen38MaxValModuleExecute(SparkQwen38MaxValModule *module, uint32_t rows, uint32_t position)
{
	SparkStatus status;
	memset(&module->context,0,sizeof(module->context));
	memset(&module->frame,0,sizeof(module->frame));
	module->context.abi_version = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	module->context.descriptor_bytes = sizeof(module->context);
	module->context.flags =
		SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
		SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
		(module->head_stage == 0u
			? SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT
			: 0u);
	module->context.kv_block_table = &module->table;
	module->context.hidden_output_transport_session = module->head_stage != 0u ? 0 : (SparkHiddenTransportSession *)&module->capture;
	module->context.hidden_output_send_function = module->head_stage != 0u ? 0 : SparkQwen38MaxValCaptureSend;
	module->frame.program_id = 1u;
	module->frame.tokens_per_sequence = 1u;
	module->frame.request_id = 1u;
	module->frame.sequence_id = 1u;
	module->frame.sequence_position = position;
	module->frame.active_slot_count = rows;
	module->frame.new_token_count = rows;
	module->frame.execution_stream = (void *)cudaStreamPerThread;
	module->frame.buffers = module->buffers;
	module->frame.buffer_count = module->head_stage != 0u ? 2u : 1u;
	module->frame.user_context = &module->context;
	memset(module->buffers,0,sizeof(module->buffers));
	module->buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	module->buffers[0].address = module->token_ids;
	module->buffers[0].bytes = rows * sizeof(uint32_t);
	if ( module->head_stage != 0u )
	{
		module->buffers[1].slot = 1u;
		module->buffers[1].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		module->buffers[1].address = module->output_token_ids;
		module->buffers[1].bytes = sizeof(module->output_token_ids);
	}
	status = SparkQwen38MaxResidentDecodeStageExecute(module->state,&module->frame);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"qwen38_max_validation failure=module_execute rows=%u status=%d\n",rows,(int)status);
		return(1);
	}
	return(0);
}

static int SparkQwen38MaxValCheckFinite(const char *check, const uint16_t *hidden, uint64_t rows)
{
	uint64_t index,count = rows * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
	for (index = 0u; index < count; index++)
		if (isfinite(SparkQwen38MaxValFromBf16(hidden[index])) == 0)
			return(SparkQwen38MaxValFail(check,"nonfinite"));
	return(0);
}

static int SparkQwen38MaxValCheckModule(void)
{
	SparkQwen38MaxValModule module;
	uint16_t decode_hidden[SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION];
	uint16_t rerun_hidden[SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION];
	uint32_t decode_token = 0u,rerun_token = 0u,row;
	SparkModelDriverAdmissionRequest admission;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	if (SparkQwen38MaxValModuleInitialize(&module) != 0)
		return(1);
	/* Admission and snapshot are fail-closed stubs on this module by
	 * design; assert they refuse cleanly rather than pretend to accept. */
	memset(&admission,0,sizeof(admission));
	admission.descriptor_bytes = sizeof(admission);
	admission.program_id = 1u;
	admission.frame_flags = 0u;
	admission.new_token_count = 1u;
	admission.active_slot_count = 1u;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkQwen38MaxResidentDecodeStageAdmit(module.state,&admission,&decision);
	if (status != SPARK_STATUS_UNSUPPORTED)
		return(SparkQwen38MaxValFail("module_admit","expected_fail_closed"));
	memset(&snapshot,0,sizeof(snapshot));
	SparkModelDriverInitializeRuntimeSnapshot(&snapshot,1u);
	status = SparkQwen38MaxResidentDecodeStageSnapshot(module.state,1u,&snapshot);
	if (status != SPARK_STATUS_UNSUPPORTED)
		return(SparkQwen38MaxValFail("module_snapshot","expected_fail_closed"));
	printf("qwen38_max_validation check=module_admit_snapshot admit=fail_closed snapshot=fail_closed\n");
	for (row = 0u; row < SPARK_QWEN38_MAX_VALIDATION_KV_LANES; row++)
		module.token_ids[row] = 1000u + (row * 371u) % 200000u;
	if (SparkQwen38MaxValModuleExecute(&module,SPARK_QWEN38_MAX_VALIDATION_KV_LANES,0u) != 0)
		return(1);
	if (module.capture.sends != (module.head_stage != 0u ? 0u : 1u))
		return(SparkQwen38MaxValFail("module_decode",module.head_stage != 0u ? "unexpected_hidden_send" : "no_hidden_send"));
	if ( module.head_stage == 0u )
	{
		if (SparkQwen38MaxValCheckFinite("module_decode",module.capture.hidden,SPARK_QWEN38_MAX_VALIDATION_KV_LANES) != 0)
			return(1);
		memcpy(decode_hidden,module.capture.hidden,sizeof(decode_hidden));
	}
	else
	{
		decode_token = module.output_token_ids[0];
		if ( decode_token >= SPARK_QWEN38_MAX_MODEL_OUTPUT_VOCAB_COUNT )
			return(SparkQwen38MaxValFail("module_decode","out_of_vocab"));
		printf("qwen38_max_validation check=module_decode token=%u in_vocab=1\n",decode_token);
	}
	/* Determinism: a fresh instance must reproduce the first lane's hidden
	 * (or token) bit for bit. */
	SparkQwen38MaxResidentDecodeStageDestroy(module.state);
	cudaFree(module.device_blocks);
	cudaFree(module.device_counts);
	{
		SparkQwen38MaxValModule rerun;
		if (SparkQwen38MaxValModuleInitialize(&rerun) != 0)
			return(1);
		for (row = 0u; row < SPARK_QWEN38_MAX_VALIDATION_KV_LANES; row++)
			rerun.token_ids[row] = 1000u + (row * 371u) % 200000u;
		if (SparkQwen38MaxValModuleExecute(&rerun,SPARK_QWEN38_MAX_VALIDATION_KV_LANES,0u) != 0)
			return(1);
		if ( rerun.head_stage == 0u )
		{
			memcpy(rerun_hidden,rerun.capture.hidden,sizeof(rerun_hidden));
			if (memcmp(decode_hidden,rerun_hidden,sizeof(decode_hidden)) != 0)
				return(SparkQwen38MaxValFail("module_determinism","fresh_instance_mismatch"));
		}
		else
		{
			rerun_token = rerun.output_token_ids[0];
			if (rerun_token != decode_token)
				return(SparkQwen38MaxValFail("module_determinism","fresh_instance_token_mismatch"));
		}
		SparkQwen38MaxResidentDecodeStageDestroy(rerun.state);
		cudaFree(rerun.device_blocks);
		cudaFree(rerun.device_counts);
		printf("qwen38_max_validation check=module_determinism bit_exact=1\n");
	}
	return(0);
}

/* The routed-expert MXFP4 path against an exact dequant oracle: route four
 * rows through ten experts each on the rank-0 shard of a tp4 view (128 of
 * 512 experts), run the B1 W13-activation and W2 kernels plus the pair
 * reduce, and compare the mixture with a per-expert fp32 CPU evaluation of
 * the SAME quantized bytes. This is the production codec's kernel gate (it
 * replaces the dropped FP8-vs-MXFP4 parity experiment: there is no FP8 twin
 * of the AMD weights, and the oracle is the stronger check anyway). */
static int SparkQwen38MaxValCheckMoeMxfp4(SparkQwen38MaxValDevice *device)
{
	const uint32_t tp = 4u,experts_per_rank = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT / tp;
	const uint32_t rows = SPARK_QWEN38_MAX_VALIDATION_MOE_ROWS;
	const uint32_t topk = SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN;
	const uint32_t hidden = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
	const uint32_t intermediate = SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	uint64_t w13_rows = (uint64_t)experts_per_rank * intermediate;
	uint64_t w2_rows = (uint64_t)experts_per_rank * hidden;
	uint64_t w13_payload_bytes = w13_rows * hidden / 2u;
	uint64_t w13_scale_bytes = w13_rows * hidden / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE;
	uint64_t w2_payload_bytes = w2_rows * intermediate / 2u;
	uint64_t w2_scale_bytes = w2_rows * intermediate / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE;
	uint16_t *host_input = (uint16_t *)calloc((uint64_t)rows * hidden,sizeof(uint16_t));
	float *input_exact = (float *)calloc((uint64_t)rows * hidden,sizeof(float));
	uint8_t *w1_payload = (uint8_t *)calloc(w13_payload_bytes,1);
	uint8_t *w1_scales = (uint8_t *)calloc(w13_scale_bytes,1);
	uint8_t *w3_payload = (uint8_t *)calloc(w13_payload_bytes,1);
	uint8_t *w3_scales = (uint8_t *)calloc(w13_scale_bytes,1);
	uint8_t *w2_payload = (uint8_t *)calloc(w2_payload_bytes,1);
	uint8_t *w2_scales = (uint8_t *)calloc(w2_scale_bytes,1);
	uint32_t *route_expert = (uint32_t *)calloc((uint64_t)rows * topk,sizeof(uint32_t));
	float *route_weights = (float *)calloc((uint64_t)rows * topk,sizeof(float));
	float *expected = (float *)calloc((uint64_t)rows * hidden,sizeof(float));
	float *actual = (float *)calloc((uint64_t)rows * hidden,sizeof(float));
	uint16_t *host_delta = (uint16_t *)calloc((uint64_t)rows * hidden,sizeof(uint16_t));
	SparkQwen38MaxLinearView w1_view,w3_view,w2_view;
	uint32_t *indices_u32 = 0,*inverse_u32 = 0,*grouped_rows_u32 = 0,*group_offset_u32 = 0,*prefix_w1_u32 = 0,*prefix_w2_u32 = 0;
	float *weights_f32 = 0;
	void *input_bf16 = 0,*activated_bf16 = 0,*slot_out_bf16 = 0,*delta_bf16 = 0;
	uint8_t *w1_payload_d = 0,*w1_scales_d = 0,*w3_payload_d = 0,*w3_scales_d = 0,*w2_payload_d = 0,*w2_scales_d = 0;
	uint64_t allocation_bytes = ((uint64_t)rows * topk + SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT + 2u) * 4u;
	uint32_t multiprocessor_count = 1u;
	SparkQwen38MaxValMetrics metrics;
	uint32_t row,slot;
	cudaError_t error = cudaSuccess;
	(void)device;
	if (host_input == 0 || input_exact == 0 || w1_payload == 0 || w1_scales == 0 || w3_payload == 0 || w3_scales == 0 || w2_payload == 0 || w2_scales == 0 || route_expert == 0 || route_weights == 0 || expected == 0 || actual == 0 || host_delta == 0)
		return(SparkQwen38MaxValFail("moe_mxfp4","host_alloc"));
	SparkQwen38MaxValRandomState = 131u;
	SparkQwen38MaxValFillBf16(host_input,input_exact,(uint64_t)rows * hidden,1.0f);
	{
		uint64_t index;
		for (index = 0u; index < (uint64_t)rows * hidden; index++)
			input_exact[index] = SparkQwen38MaxValFromBf16(host_input[index]);
	}
	/* Deterministic random-looking bytes: a 64 KiB xorshift block tiled over
	 * the payloads; scale codes pinned near unity. The oracle dequantizes
	 * the exact same bytes, so any bytes are valid weights. */
	{
		static uint8_t block[65536];
		uint32_t state = 0x1234567u,index;
		uint64_t offset;
		for (index = 0u; index < sizeof(block); index++)
		{
			state ^= state << 13;
			state ^= state >> 17;
			state ^= state << 5;
			block[index] = (uint8_t)(state >> 24);
		}
		/* Even payload bytes become nibble pairs with both halves populated;
		 * keep every byte nonzero so the weights exercise the full grid. */
		for (index = 0u; index < 65536u; index += 2u)
			block[index] = (uint8_t)(120u + (block[index] & 7u));
		{
			static uint8_t rotated[65536],shifted[65536];
			memcpy(rotated,block + 32768u,32768u);
			memcpy(rotated + 32768u,block,32768u);
			memcpy(shifted,block + 17u,65519u);
			memcpy(shifted + 65519u,block,17u);
			for (offset = 0u; offset < w13_payload_bytes; offset += sizeof(block))
			{
				uint64_t step = w13_payload_bytes - offset < sizeof(block) ? w13_payload_bytes - offset : sizeof(block);
				memcpy(w1_payload + offset,block,(size_t)step);
				memcpy(w3_payload + offset,rotated,(size_t)step);
			}
			for (offset = 0u; offset < w2_payload_bytes; offset += sizeof(block))
			{
				uint64_t step = w2_payload_bytes - offset < sizeof(block) ? w2_payload_bytes - offset : sizeof(block);
				memcpy(w2_payload + offset,shifted,(size_t)step);
			}
		}
		memset(w1_scales,124u,(size_t)w13_scale_bytes);
		memset(w3_scales,123u,(size_t)w13_scale_bytes);
		memset(w2_scales,125u,(size_t)w2_scale_bytes);
	}
	/* Route every row through ten DISTINCT rank-0 experts (the shard owns
	 * experts [0,128); global ids below stay inside it). */
	for (row = 0u; row < rows; row++)
		for (slot = 0u; slot < topk; slot++)
		{
			route_expert[(uint64_t)row * topk + slot] = ((row * 7u + slot * 13u) % (experts_per_rank / 2u)) + ((slot & 1u) * (experts_per_rank / 2u));
			route_weights[(uint64_t)row * topk + slot] = 0.05f + 0.09f * (float)slot;
		}
	error = cudaMalloc(&input_bf16,(uint64_t)rows * hidden * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc(&activated_bf16,(uint64_t)rows * topk * intermediate * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc(&slot_out_bf16,(uint64_t)rows * topk * hidden * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc(&delta_bf16,(uint64_t)rows * hidden * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc(&indices_u32,(uint64_t)rows * topk * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&inverse_u32,allocation_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&grouped_rows_u32,allocation_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&group_offset_u32,(SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&prefix_w1_u32,(SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&prefix_w2_u32,(SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&weights_f32,(uint64_t)rows * topk * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc(&w1_payload_d,w13_payload_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&w1_scales_d,w13_scale_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&w3_payload_d,w13_payload_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&w3_scales_d,w13_scale_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&w2_payload_d,w2_payload_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&w2_scales_d,w2_scale_bytes);
	if (error == cudaSuccess) error = cudaMemcpy(input_bf16,host_input,(uint64_t)rows * hidden * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(indices_u32,route_expert,(uint64_t)rows * topk * sizeof(uint32_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(weights_f32,route_weights,(uint64_t)rows * topk * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w1_payload_d,w1_payload,w13_payload_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w1_scales_d,w1_scales,w13_scale_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w3_payload_d,w3_payload,w13_payload_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w3_scales_d,w3_scales,w13_scale_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w2_payload_d,w2_payload,w2_payload_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(w2_scales_d,w2_scales,w2_scale_bytes,cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
	{
		int32_t sm_count = 0;
		if (cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0) == cudaSuccess && sm_count > 0)
			multiprocessor_count = (uint32_t)sm_count;
	}
	memset(&w1_view,0,sizeof(w1_view));
	memset(&w3_view,0,sizeof(w3_view));
	memset(&w2_view,0,sizeof(w2_view));
	w1_view.abi_version = w3_view.abi_version = w2_view.abi_version = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	w1_view.weight_format = w3_view.weight_format = w2_view.weight_format = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1;
	w1_view.input_dimension = w3_view.input_dimension = hidden;
	w1_view.output_dimension = w3_view.output_dimension = (uint32_t)w13_rows;
	w2_view.input_dimension = intermediate;
	w2_view.output_dimension = (uint32_t)w2_rows;
	w1_view.weight_payload = w1_payload_d;
	w1_view.weight_scale_e8m0 = w1_scales_d;
	w3_view.weight_payload = w3_payload_d;
	w3_view.weight_scale_e8m0 = w3_scales_d;
	w2_view.weight_payload = w2_payload_d;
	w2_view.weight_scale_e8m0 = w2_scales_d;
	w1_view.weight_payload_bytes = w13_payload_bytes;
	w3_view.weight_payload_bytes = w13_payload_bytes;
	w2_view.weight_payload_bytes = w2_payload_bytes;
	w1_view.weight_scale_bytes = w13_scale_bytes;
	w3_view.weight_scale_bytes = w13_scale_bytes;
	w2_view.weight_scale_bytes = w2_scale_bytes;
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchMoeRoute(cudaStreamPerThread,indices_u32,rows,intermediate,1u,group_offset_u32,inverse_u32,grouped_rows_u32,prefix_w1_u32,prefix_w2_u32);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchFusedExpertW13Act(cudaStreamPerThread,&w1_view,&w3_view,input_bf16,grouped_rows_u32,group_offset_u32,prefix_w1_u32,activated_bf16,rows,intermediate,SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT,multiprocessor_count,tp,0u);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchExpertDown(cudaStreamPerThread,&w2_view,activated_bf16,group_offset_u32,prefix_w2_u32,slot_out_bf16,rows,intermediate,hidden,multiprocessor_count,tp,0u);
	if (error == cudaSuccess)
		error = SparkQwen38MaxLaunchMoePairReduceOverwrite(cudaStreamPerThread,slot_out_bf16,inverse_u32,weights_f32,delta_bf16,rows,hidden);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(host_delta,delta_bf16,(uint64_t)rows * hidden * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SparkQwen38MaxValCuda(error,"moe_mxfp4") != 0)
	{
		free(host_input); free(input_exact); free(w1_payload); free(w1_scales); free(w3_payload); free(w3_scales); free(w2_payload); free(w2_scales); free(route_expert); free(route_weights); free(expected); free(actual); free(host_delta);
		return(1);
	}
	/* Oracle: the B1 pipeline exactly as the kernels execute it - the
	 * activation row is FP8-E4M3/UE8M0 quantize-dequantized per 128-block
	 * before W13 and again before W2, the gate/up dots are bf16-rounded
	 * then limit-clamped, and every inter-stage store is bf16. */
	for (row = 0u; row < rows; row++)
	{
		float *x = (float *)calloc(hidden,sizeof(float));
		float *y = expected + ((uint64_t)row * hidden);
		uint32_t element;
		for (element = 0u; element < hidden; element++)
		{
			x[element] = input_exact[(uint64_t)row * hidden + element];
			y[element] = 0.0f;
		}
		SparkQwen38MaxValQdq128(x,hidden);
		for (slot = 0u; slot < topk; slot++)
		{
			uint32_t expert = route_expert[(uint64_t)row * topk + slot];
			float weight = route_weights[(uint64_t)row * topk + slot];
			float *activated = (float *)calloc(intermediate,sizeof(float));
			float *down = (float *)calloc(hidden,sizeof(float));
			uint32_t neuron;
			for (neuron = 0u; neuron < intermediate; neuron++)
			{
				uint64_t w13_row = (uint64_t)expert * intermediate + neuron;
				float gate = 0.0f,up = 0.0f;
				uint32_t column;
				for (column = 0u; column < hidden; column++)
				{
					/* fmaf: the kernel's dot rounds each product into the
					 * accumulation once, matching cvt/mma semantics. */
					gate = fmaf(SparkQwen38MaxValDequantMxfp4(w1_payload,w1_scales,w13_row,hidden,column),x[column],gate);
					up = fmaf(SparkQwen38MaxValDequantMxfp4(w3_payload,w3_scales,w13_row,hidden,column),x[column],up);
				}
				gate = SparkQwen38MaxValBf16Round(gate);
				up = SparkQwen38MaxValBf16Round(up);
				gate = gate > SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT ? SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT : gate;
				up = up > SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT ? SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT : (up < -SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT ? -SPARK_QWEN38_MAX_MODEL_SWIGLU_LIMIT : up);
				activated[neuron] = SparkQwen38MaxValFromBf16(SparkQwen38MaxValBf16(SparkQwen38MaxValSilu(gate) * up));
			}
			SparkQwen38MaxValQdq128(activated,intermediate);
			for (element = 0u; element < hidden; element++)
			{
				uint64_t w2_row = (uint64_t)expert * hidden + element;
				float sum = 0.0f;
				uint32_t neuron;
				for (neuron = 0u; neuron < intermediate; neuron++)
					sum = fmaf(SparkQwen38MaxValDequantMxfp4(w2_payload,w2_scales,w2_row,intermediate,neuron),activated[neuron],sum);
				down[element] = SparkQwen38MaxValFromBf16(SparkQwen38MaxValBf16(sum));
			}
			for (element = 0u; element < hidden; element++)
				y[element] += weight * down[element];
			free(activated);
			free(down);
		}
		free(x);
	}
	{
		uint64_t index;
		for (index = 0u; index < (uint64_t)rows * hidden; index++)
			actual[index] = SparkQwen38MaxValFromBf16(host_delta[index]);
	}
	SparkQwen38MaxValMeasure(&metrics,actual,expected,(uint64_t)rows * hidden);
	free(host_input); free(input_exact); free(w1_payload); free(w1_scales); free(w3_payload); free(w3_scales); free(w2_payload); free(w2_scales); free(route_expert); free(route_weights); free(expected); free(actual); free(host_delta);
	if (indices_u32 != 0) cudaFree(indices_u32);
	if (inverse_u32 != 0) cudaFree(inverse_u32);
	if (grouped_rows_u32 != 0) cudaFree(grouped_rows_u32);
	if (group_offset_u32 != 0) cudaFree(group_offset_u32);
	if (prefix_w1_u32 != 0) cudaFree(prefix_w1_u32);
	if (prefix_w2_u32 != 0) cudaFree(prefix_w2_u32);
	if (weights_f32 != 0) cudaFree(weights_f32);
	if (input_bf16 != 0) cudaFree(input_bf16);
	if (activated_bf16 != 0) cudaFree(activated_bf16);
	if (slot_out_bf16 != 0) cudaFree(slot_out_bf16);
	if (delta_bf16 != 0) cudaFree(delta_bf16);
	if (w1_payload_d != 0) cudaFree(w1_payload_d);
	if (w1_scales_d != 0) cudaFree(w1_scales_d);
	if (w3_payload_d != 0) cudaFree(w3_payload_d);
	if (w3_scales_d != 0) cudaFree(w3_scales_d);
	if (w2_payload_d != 0) cudaFree(w2_payload_d);
	if (w2_scales_d != 0) cudaFree(w2_scales_d);
	return(SparkQwen38MaxValReport("moe_mxfp4",&metrics,2e-2,0.999));
}

int main(int argc, char **argv)
{
	SparkQwen38MaxValDevice device;
	int result = 0;
	if (argc != 2 || strlen(argv[1]) != 64u)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if (SparkQwen38MaxValCuda(SparkQwen38MaxConfigureCudaKernels(),"configure") != 0)
		return(1);
	if (SparkQwen38MaxValDeviceSetup(&device) != 0)
		return(1);
	if (result == 0) result = SparkQwen38MaxValCheckDecayBeta(&device);
	if (result == 0) result = SparkQwen38MaxValCheckGdnStep(&device);
	if (result == 0) result = SparkQwen38MaxValCheckGatedNorm(&device);
	if (result == 0) result = SparkQwen38MaxValCheckAttention(&device);
	if (result == 0) result = SparkQwen38MaxValCheckGdnChunk(&device);
	if (result == 0) result = SparkQwen38MaxValCheckMoeMxfp4(&device);
	if (result == 0) result = SparkQwen38MaxValCheckGdnStepTp4();
	if (result == 0) result = SparkQwen38MaxValCheckModule();
	if (result == 0)
		printf("qwen38_max_validation PASS\n");
	return(result);
}
