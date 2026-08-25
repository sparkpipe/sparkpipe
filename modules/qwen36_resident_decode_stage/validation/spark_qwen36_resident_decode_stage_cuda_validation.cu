#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"

/*
 * Qwen 3.6 resident decode stage, hardware validation (sm_121a).
 *
 * Two tiers, one binary. The KERNEL tier runs the module's CUDA kernels on
 * synthetic inputs and compares against fp32 CPU oracles - the same
 * formulas validation/spark_qwen36_reference.c pins against modeling_qwen3_5,
 * restated here so this translation unit stays self-contained under nvcc.
 * The MODULE tier loads a stage pack through the module's own
 * Initialize/Execute (the pack path comes from SPARK_QWEN36_STAGE_* exactly
 * as production), drives a prefill-then-decode flow on two lanes with a
 * capture transport standing in for the ring, and checks determinism across
 * a fresh instance plus decode-vs-prefill cross-path agreement. Synthetic
 * packs exercise geometry; the real stage-0 pack runs the same checks on
 * the model's own weights.
 *
 * Every comparison prints its numbers; the thresholds are the guard, the
 * numbers are the evidence.
 */

#define SPARK_QWEN36_VALIDATION_ROWS 4u
#define SPARK_QWEN36_VALIDATION_PREFILL_TOKENS 8u
#define SPARK_QWEN36_VALIDATION_CHUNK_TOKENS 128u
#define SPARK_QWEN36_VALIDATION_ATTN_TOKENS 5u
/* KV table lanes must match the module's max_active_sequence_count
 * (the module validates lane_count == its own max). Use the firmware
 * header's build-time value, not a hardcoded 8 — this was the B16
 * validation failure: the module at B16 expected lane_count=16 but
 * the validator always supplied 8. */
#ifndef SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES 8u
#endif
#define SPARK_QWEN36_VALIDATION_KV_LANES SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES

#define SPARK_QWEN36_DK SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN36_DV SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN36_HEADS SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT
#define SPARK_QWEN36_CONV SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS

extern "C" cudaError_t SparkQwen36ConfigureCudaKernels(void);
extern "C" cudaError_t SparkQwen36LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern "C" cudaError_t SparkQwen36LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen36GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern "C" cudaError_t SparkQwen36LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern "C" cudaError_t SparkQwen36LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen36GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern "C" cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern "C" cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern "C" cudaError_t SparkQwen36LaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);

static uint32_t SparkQwen36ValRandomState;

static uint32_t SparkQwen36ValNext(void)
{
	SparkQwen36ValRandomState = SparkQwen36ValRandomState * 1664525u + 1013904223u;
	return(SparkQwen36ValRandomState >> 8u);
}

static float SparkQwen36ValUniform(float scale)
{
	return(((float)(int32_t)(SparkQwen36ValNext() & 0xffffu) - 32768.0f) * scale / 32768.0f);
}

static uint16_t SparkQwen36ValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits += 0x8000u; /* round to nearest even on the truncate */
	return((uint16_t)(bits >> 16u));
}

static float SparkQwen36ValFromBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float converted;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}

/* Fill a host buffer with random values, returned both as the bf16 they
 * become on device and as the exact upcast the oracle must see. */
static void SparkQwen36ValFillBf16(uint16_t *packed, float *exact, uint64_t count, float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SparkQwen36ValBf16(SparkQwen36ValUniform(scale));
		if (exact != 0)
			exact[index] = SparkQwen36ValFromBf16(packed[index]);
	}
}

static int SparkQwen36ValFail(const char *check, const char *detail)
{
	fprintf(stderr,"qwen36_validation failure=%s detail=%s\n",check,detail);
	return(1);
}

static int SparkQwen36ValCuda(cudaError_t error, const char *check)
{
	if (error == cudaSuccess)
		return(0);
	fprintf(stderr,"qwen36_validation failure=%s cuda=%s\n",check,cudaGetErrorString(error));
	return(1);
}

typedef struct SparkQwen36ValMetrics
{
	double difference_l2;
	double reference_l2;
	double actual_l2;
	double dot;
	double maximum_absolute;
	uint64_t count;
} SparkQwen36ValMetrics;

static void SparkQwen36ValMeasure(SparkQwen36ValMetrics *metrics, const float *actual, const float *reference, uint64_t count)
{
	uint64_t index;
	double difference;
	memset(metrics,0,sizeof(*metrics));
	metrics->count = count;
	for (index = 0u; index < count; index++)
	{
		difference = (double)actual[index] - (double)reference[index];
		metrics->difference_l2 += difference * difference;
		metrics->reference_l2 += (double)reference[index] * (double)reference[index];
		metrics->actual_l2 += (double)actual[index] * (double)actual[index];
		metrics->dot += (double)actual[index] * (double)reference[index];
		if (fabs((double)actual[index] - (double)reference[index]) > metrics->maximum_absolute)
			metrics->maximum_absolute = fabs((double)actual[index] - (double)reference[index]);
	}
}

static int SparkQwen36ValReport(const char *check, const SparkQwen36ValMetrics *metrics, double max_relative_l2, double minimum_cosine)
{
	double relative_l2 = metrics->reference_l2 > 0.0
		? sqrt(metrics->difference_l2 / metrics->reference_l2) : INFINITY;
	double cosine = metrics->actual_l2 > 0.0 && metrics->reference_l2 > 0.0
		? metrics->dot / sqrt(metrics->actual_l2 * metrics->reference_l2) : 0.0;
	printf("qwen36_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)metrics->count,relative_l2,cosine,metrics->maximum_absolute);
	if (isfinite(relative_l2) == 0 || relative_l2 > max_relative_l2)
		return(SparkQwen36ValFail(check,"relative_l2"));
	if (cosine < minimum_cosine)
		return(SparkQwen36ValFail(check,"cosine"));
	return(0);
}

/* -- fp32 CPU oracles: the reference.c formulas, restated ---------------- */

static float SparkQwen36ValSilu(float value)
{
	return(value / (1.0f + expf(-value)));
}

static void SparkQwen36ValL2Norm(const float *input, float *output, uint32_t dimension)
{
	uint32_t element;
	float total = 0.0f;
	for (element = 0u; element < dimension; element++)
		total += input[element] * input[element];
	total = 1.0f / sqrtf(total + 1e-6f);
	for (element = 0u; element < dimension; element++)
		output[element] = input[element] * total;
}

static void SparkQwen36ValGdnRecurrence(const float *q, const float *k, const float *v, const float *g, const float *beta, float *state, float *output, uint32_t tokens)
{
	float qn[SPARK_QWEN36_DK],kn[SPARK_QWEN36_DK],delta[SPARK_QWEN36_DV];
	float scale = 1.0f / sqrtf((float)SPARK_QWEN36_DK),decay,kv_mem;
	uint32_t token,row,column;
	for (token = 0u; token < tokens; token++)
	{
		SparkQwen36ValL2Norm(q + ((uint64_t)token * SPARK_QWEN36_DK),qn,SPARK_QWEN36_DK);
		SparkQwen36ValL2Norm(k + ((uint64_t)token * SPARK_QWEN36_DK),kn,SPARK_QWEN36_DK);
		for (row = 0u; row < SPARK_QWEN36_DK; row++)
			qn[row] *= scale;
		decay = expf(g[token]);
		for (row = 0u; row < SPARK_QWEN36_DK; row++)
			for (column = 0u; column < SPARK_QWEN36_DV; column++)
				state[(row * SPARK_QWEN36_DV) + column] *= decay;
		for (column = 0u; column < SPARK_QWEN36_DV; column++)
		{
			kv_mem = 0.0f;
			for (row = 0u; row < SPARK_QWEN36_DK; row++)
				kv_mem += state[(row * SPARK_QWEN36_DV) + column] * kn[row];
			delta[column] = (v[((uint64_t)token * SPARK_QWEN36_DV) + column] - kv_mem) * beta[token];
		}
		for (row = 0u; row < SPARK_QWEN36_DK; row++)
			for (column = 0u; column < SPARK_QWEN36_DV; column++)
				state[(row * SPARK_QWEN36_DV) + column] += kn[row] * delta[column];
		for (column = 0u; column < SPARK_QWEN36_DV; column++)
		{
			kv_mem = 0.0f;
			for (row = 0u; row < SPARK_QWEN36_DK; row++)
				kv_mem += state[(row * SPARK_QWEN36_DV) + column] * qn[row];
			output[((uint64_t)token * SPARK_QWEN36_DV) + column] = kv_mem;
		}
	}
}

static void SparkQwen36ValRope(float *vector, uint32_t rope_dim, uint32_t position, float theta)
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

static void SparkQwen36ValRmsNorm(const float *input, const float *weight, float *output, uint32_t dimension, float epsilon)
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
 * input, sigmoid on the output - reference.c's SparkQwen36RefAttention. */
static void SparkQwen36ValAttention(const float *q_fused, const float *k_cache, const float *v_cache, const float *q_norm_weight, float *output, uint32_t group, uint32_t tokens, float epsilon)
{
	float qh[SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION],scores[SPARK_QWEN36_VALIDATION_ATTN_TOKENS],probability;
	float scale = 1.0f / sqrtf((float)SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION),maximum,total;
	uint32_t head,element,token;
	for (head = 0u; head < group; head++)
	{
		const float *fused = q_fused + ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
		SparkQwen36ValRmsNorm(fused,q_norm_weight,qh,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,epsilon);
		SparkQwen36ValRope(qh,SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION,tokens - 1u,SPARK_QWEN36_MODEL_ATTN_ROPE_THETA);
		maximum = -3.0e38f;
		for (token = 0u; token < tokens; token++)
		{
			probability = 0.0f;
			for (element = 0u; element < SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION; element++)
				probability += qh[element] * k_cache[((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + element];
			scores[token] = probability * scale;
			if (scores[token] > maximum)
				maximum = scores[token];
		}
		total = 0.0f;
		for (token = 0u; token < tokens; token++)
		{
			scores[token] = expf(scores[token] - maximum);
			total += scores[token];
		}
		for (element = 0u; element < SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION; element++)
		{
			probability = 0.0f;
			for (token = 0u; token < tokens; token++)
				probability += (scores[token] / total) * v_cache[((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + element];
			output[((uint64_t)head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + element] =
				probability * (1.0f / (1.0f + expf(-fused[SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION + element])));
		}
	}
}

/* -- shared device fixture ----------------------------------------------- */

typedef struct SparkQwen36ValDevice
{
	SparkQwen36GdnLayerWeights gdn_weights;
	SparkQwen36AttnLayerWeights attn_weights;
	SparkQwen36GdnStatePool pool;
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
} SparkQwen36ValDevice;

static int SparkQwen36ValDeviceSetup(SparkQwen36ValDevice *device)
{
	uint64_t state_elements = 2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DK * SPARK_QWEN36_DV;
	uint64_t vector_floats = (uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN36_DK;
	uint64_t matrix_floats = (uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS;
	uint32_t tokens = SPARK_QWEN36_VALIDATION_CHUNK_TOKENS;
	cudaError_t error;
	memset(device,0,sizeof(*device));
	error = cudaMalloc((void **)&device->conv_weight,(uint64_t)SPARK_QWEN36_CONV * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->a_log,SPARK_QWEN36_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->dt_bias,SPARK_QWEN36_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gdn_norm_weight,SPARK_QWEN36_DV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->q_norm_weight,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->k_norm_weight,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->state,state_elements * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_tail,2ull * SPARK_QWEN36_CONV * (SPARK_QWEN36_MODEL_GDN_CONV_KERNEL - 1u) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->cold,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->lane_indices,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->qkv,(uint64_t)tokens * SPARK_QWEN36_CONV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_out,(uint64_t)tokens * SPARK_QWEN36_CONV * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->core_out,(uint64_t)tokens * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->z_bf16,(uint64_t)tokens * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gated_out,(uint64_t)tokens * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->ba_bf16,(uint64_t)tokens * SPARK_QWEN36_HEADS * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->log_decay,(uint64_t)tokens * SPARK_QWEN36_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->beta,(uint64_t)tokens * SPARK_QWEN36_HEADS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_qn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_cum_g,(uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_decay,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_attn,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_w,(uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS * SPARK_QWEN36_DV * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kg,vector_floats * sizeof(float));
	if (error != cudaSuccess)
		return(SparkQwen36ValCuda(error,"device_alloc"));
	/* One GDN layer per lane, two lanes: the pool strides the launchers
	 * read, mirroring the module's AllocatePools at layer_count 1. */
	device->pool.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	device->pool.lane_capacity = 2u;
	device->pool.gdn_layer_count = 1u;
	device->pool.state_f32 = device->state;
	device->pool.state_layer_stride_elements = (uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_DK * SPARK_QWEN36_DV;
	device->pool.state_lane_stride_elements = device->pool.state_layer_stride_elements;
	device->pool.conv_tail_bf16 = device->conv_tail;
	device->pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_QWEN36_CONV * (SPARK_QWEN36_MODEL_GDN_CONV_KERNEL - 1u);
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

/* -- kernel tier ---------------------------------------------------------- */

static int SparkQwen36ValCheckDecayBeta(SparkQwen36ValDevice *device)
{
	uint16_t host_ba[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS * 2u];
	float host_a[SPARK_QWEN36_HEADS],host_bias[SPARK_QWEN36_HEADS];
	float host_decay[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS];
	float host_beta[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS];
	float ref_decay[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS];
	float ref_beta[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS];
	SparkQwen36ValMetrics metrics;
	uint32_t row,head;
	uint64_t index;
	float shifted;
	cudaError_t error;
	SparkQwen36ValRandomState = 11u;
	SparkQwen36ValFillBf16(host_ba,0,SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS * 2u,2.0f);
	for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
	{
		host_a[head] = SparkQwen36ValUniform(1.0f);
		host_bias[head] = SparkQwen36ValUniform(1.0f);
	}
	error = cudaMemcpy(device->ba_bf16,host_ba,sizeof(host_ba),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->a_log,host_a,sizeof(host_a),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->dt_bias,host_bias,sizeof(host_bias),cudaMemcpyHostToDevice);
	/* decay_pre and beta_pre are separate rows x HEADS buffers; ba_bf16
	 * holds them back to back. */
	if (error == cudaSuccess)
		error = SparkQwen36LaunchDecayBeta(cudaStreamPerThread,device->ba_bf16,device->ba_bf16 + SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS,&device->gdn_weights,device->log_decay,device->beta,SPARK_QWEN36_VALIDATION_ROWS);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(host_decay,device->log_decay,sizeof(host_decay),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(host_beta,device->beta,sizeof(host_beta),cudaMemcpyDeviceToHost);
	if (SparkQwen36ValCuda(error,"decay_beta") != 0)
		return(1);
	for (row = 0u; row < SPARK_QWEN36_VALIDATION_ROWS; row++)
		for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
		{
			index = ((uint64_t)row * SPARK_QWEN36_HEADS) + head;
			shifted = SparkQwen36ValFromBf16(host_ba[index]) + host_bias[head];
			ref_decay[index] = -expf(host_a[head]) * (shifted > 20.0f ? shifted : logf(1.0f + expf(shifted)));
			ref_beta[index] = 1.0f / (1.0f + expf(-SparkQwen36ValFromBf16(host_ba[SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS + index])));
		}
	SparkQwen36ValMeasure(&metrics,host_decay,ref_decay,SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS);
	if (SparkQwen36ValReport("decay_gate",&metrics,1e-5,0.99999999) != 0)
		return(1);
	SparkQwen36ValMeasure(&metrics,host_beta,ref_beta,SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_HEADS);
	return(SparkQwen36ValReport("write_gate",&metrics,1e-5,0.99999999));
}

/* One decode conv step per row over two sequential tokens, checking the tail
 * carry against the single-channel oracle run on every channel. */
static int SparkQwen36ValCheckConv(SparkQwen36ValDevice *device)
{
	uint16_t *host_qkv,*host_out;
	float *exact,*expected,*actual,*tails;
	uint32_t token,row;
	uint32_t cold[2] = {1u,1u};
	uint32_t lanes[2] = {0u,1u};
	/* Two sequential decode tokens for each of the two lanes: the launches
	 * and the oracle both walk this [token][lane][channel] frame. */
	const uint64_t frame_elements = (uint64_t)SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_CONV;
	uint64_t channel;
	cudaError_t error;
	SparkQwen36ValMetrics metrics;
	host_qkv = (uint16_t *)calloc(frame_elements,sizeof(uint16_t));
	host_out = (uint16_t *)calloc(frame_elements,sizeof(uint16_t));
	exact = (float *)calloc(frame_elements,sizeof(float));
	expected = (float *)calloc(frame_elements,sizeof(float));
	actual = (float *)calloc(frame_elements,sizeof(float));
	tails = (float *)calloc(frame_elements * 3u,sizeof(float));
	if (host_qkv == 0 || host_out == 0 || exact == 0 || expected == 0 || actual == 0 || tails == 0)
		return(SparkQwen36ValFail("conv_update","host_alloc"));
	SparkQwen36ValRandomState = 23u;
	SparkQwen36ValFillBf16(host_qkv,exact,frame_elements,1.0f);
	{
		uint16_t *weight_packed = (uint16_t *)calloc((uint64_t)SPARK_QWEN36_CONV * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL,sizeof(uint16_t));
		float *weight_exact = (float *)calloc((uint64_t)SPARK_QWEN36_CONV * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL,sizeof(float));
		if (weight_packed == 0 || weight_exact == 0)
			return(SparkQwen36ValFail("conv_update","weight_alloc"));
		SparkQwen36ValFillBf16(weight_packed,weight_exact,(uint64_t)SPARK_QWEN36_CONV * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL,0.5f);
		error = cudaMemcpy(device->conv_weight,weight_packed,(uint64_t)SPARK_QWEN36_CONV * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->qkv,host_qkv,frame_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
		for (token = 0u; token < 2u && error == cudaSuccess; token++)
		{
			if (token != 0u)
			{
				cold[0] = 0u; cold[1] = 0u;
				error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
			}
			/* One token per launch, one row per lane: the launcher consumes
			 * row r's qkv row, so slide the window by handing it row r. */
			if (error == cudaSuccess)
				error = SparkQwen36LaunchConvUpdate(cudaStreamPerThread,device->qkv + (uint64_t)token * 2u * SPARK_QWEN36_CONV,&device->gdn_weights,device->conv_out + (uint64_t)token * 2u * SPARK_QWEN36_CONV,&device->pool,device->lane_indices,2u,0u);
			if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		}
		if (error == cudaSuccess)
			error = cudaMemcpy(host_out,device->conv_out,frame_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (SparkQwen36ValCuda(error,"conv_update") != 0)
			return(1);
		for (row = 0u; row < 2u; row++)
			for (channel = 0u; channel < SPARK_QWEN36_CONV; channel++)
			{
				/* The oracle conv, one channel, fed the two tokens in order
				 * with the tail carried between them. */
				float input[2],output[2],tail[3];
				uint64_t base = ((uint64_t)row * SPARK_QWEN36_CONV) + channel;
				input[0] = exact[base];
				input[1] = exact[2ull * SPARK_QWEN36_CONV + base];
				tail[0] = tail[1] = tail[2] = 0.0f;
				{
					uint32_t t,token_index;
					for (token_index = 0u; token_index < 2u; token_index++)
					{
						float window[4],accumulator = 0.0f;
						window[0] = token_index >= 3u ? input[token_index - 3u] : tail[token_index];
						window[1] = token_index >= 2u ? input[token_index - 2u] : tail[token_index + 1u];
						window[2] = token_index >= 1u ? input[token_index - 1u] : tail[token_index + 2u];
						window[3] = input[token_index];
						for (t = 0u; t < 4u; t++)
							accumulator += window[t] * weight_exact[(channel * 4u) + t];
						output[token_index] = SparkQwen36ValSilu(accumulator);
					}
					tail[2] = input[1];
					tail[1] = input[0];
					tail[0] = 0.0f;
					(void)tail;
				}
				expected[(uint64_t)0u * 2u * SPARK_QWEN36_CONV + base] = output[0];
				expected[(uint64_t)1u * 2u * SPARK_QWEN36_CONV + base] = output[1];
			}
		for (token = 0u; token < 2u; token++)
			for (uint64_t i = 0u; i < 2ull * SPARK_QWEN36_CONV; i++)
				actual[(uint64_t)token * 2ull * SPARK_QWEN36_CONV + i] =
					SparkQwen36ValFromBf16(host_out[(uint64_t)token * 2ull * SPARK_QWEN36_CONV + i]);
		free(weight_packed);
		free(weight_exact);
	}
	SparkQwen36ValMeasure(&metrics,actual,expected,frame_elements);
	free(host_qkv); free(host_out); free(exact); free(expected); free(actual); free(tails);
	return(SparkQwen36ValReport("conv_update",&metrics,5e-3,0.99999));
}

/* One GDN recurrence step per row: row 0 cold, row 1 warm with a random
 * resident state, compared against the per-head oracle recurrence. */
static int SparkQwen36ValCheckGdnStep(SparkQwen36ValDevice *device)
{
	uint64_t state_elements = (uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_DK * SPARK_QWEN36_DV;
	uint16_t *host_conv = (uint16_t *)calloc(2ull * SPARK_QWEN36_CONV,sizeof(uint16_t));
	float *exact = (float *)calloc(2ull * SPARK_QWEN36_CONV,sizeof(float));
	float *state_host = (float *)calloc(2ull * state_elements,sizeof(float));
	float *state_reference = (float *)calloc(2ull * state_elements,sizeof(float));
	float *oracle_out = (float *)calloc(2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV,sizeof(float));
	float *actual = (float *)calloc(2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV,sizeof(float));
	uint16_t *core_packed = (uint16_t *)calloc(2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV,sizeof(uint16_t));
	uint32_t lanes[2] = {0u,1u};
	uint32_t cold[2] = {1u,0u};
	SparkQwen36ValMetrics metrics;
	uint32_t row,head;
	cudaError_t error;
	if (host_conv == 0 || exact == 0 || state_host == 0 || state_reference == 0 || oracle_out == 0 || actual == 0 || core_packed == 0)
		return(SparkQwen36ValFail("gdn_step","host_alloc"));
	SparkQwen36ValRandomState = 37u;
	SparkQwen36ValFillBf16(host_conv,exact,2ull * SPARK_QWEN36_CONV,1.0f);
	{
		uint64_t index;
		for (index = 0u; index < state_elements; index++)
			state_host[state_elements + index] = SparkQwen36ValUniform(0.25f); /* lane 1 warm */
		memcpy(state_reference,state_host,2ull * state_elements * sizeof(float));
	}
	{
		float host_log_decay[2 * SPARK_QWEN36_HEADS],host_beta[2 * SPARK_QWEN36_HEADS];
		uint32_t head_index;
		for (head_index = 0u; head_index < 2u * SPARK_QWEN36_HEADS; head_index++)
		{
			host_log_decay[head_index] = SparkQwen36ValUniform(0.5f) - 0.5f;
			host_beta[head_index] = 0.25f + fabsf(SparkQwen36ValUniform(0.5f));
		}
		error = cudaMemcpy(device->state,state_host,2ull * state_elements * sizeof(float),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->qkv,host_conv,2ull * SPARK_QWEN36_CONV * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,host_log_decay,sizeof(host_log_decay),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->beta,host_beta,sizeof(host_beta),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
		if (error == cudaSuccess)
			error = SparkQwen36LaunchGdnStep(cudaStreamPerThread,device->qkv,device->log_decay,device->beta,&device->pool,device->core_out,device->lane_indices,2u,0u);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		if (error == cudaSuccess) error = cudaMemcpy(core_packed,device->core_out,2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (error == cudaSuccess) error = cudaMemcpy(state_host,device->state,2ull * state_elements * sizeof(float),cudaMemcpyDeviceToHost);
		if (SparkQwen36ValCuda(error,"gdn_step") != 0)
			return(1);
		/* Oracle: per (row, value head), key head h/3, one token. */
		for (row = 0u; row < 2u; row++)
			for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
			{
				uint32_t key_head = head / 3u;
				const float *q = exact + ((uint64_t)row * SPARK_QWEN36_CONV) + ((uint64_t)key_head * SPARK_QWEN36_DK);
				const float *k = q + SPARK_QWEN36_MODEL_GDN_QK_DIMENSION;
				const float *v = exact + ((uint64_t)row * SPARK_QWEN36_CONV) + (2ull * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_DV);
				float g = cold[row] != 0u ? -30.0f : host_log_decay[(row * SPARK_QWEN36_HEADS) + head];
				float beta = host_beta[(row * SPARK_QWEN36_HEADS) + head];
				SparkQwen36ValGdnRecurrence(q,k,v,&g,&beta,
					state_reference + ((uint64_t)row * state_elements) + ((uint64_t)head * SPARK_QWEN36_DK * SPARK_QWEN36_DV),
					oracle_out + ((uint64_t)row * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV) + ((uint64_t)head * SPARK_QWEN36_DV),1u);
			}
	}
	{
		uint64_t index;
		for (index = 0u; index < 2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV; index++)
			actual[index] = SparkQwen36ValFromBf16(core_packed[index]);
	}
	SparkQwen36ValMeasure(&metrics,actual,oracle_out,2ull * SPARK_QWEN36_HEADS * SPARK_QWEN36_DV);
	if (SparkQwen36ValReport("gdn_step_output",&metrics,5e-3,0.99999) != 0)
		return(1);
	SparkQwen36ValMeasure(&metrics,state_host,state_reference,2ull * state_elements);
	free(host_conv); free(exact); free(state_host); free(state_reference);
	free(oracle_out); free(actual); free(core_packed);
	return(SparkQwen36ValReport("gdn_step_state",&metrics,1e-3,0.999999));
}

static int SparkQwen36ValCheckGatedNorm(SparkQwen36ValDevice *device)
{
	uint64_t elements = (uint64_t)SPARK_QWEN36_VALIDATION_ROWS * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION;
	uint16_t *packed = (uint16_t *)calloc(elements * sizeof(uint16_t),sizeof(uint16_t));
	float *exact = (float *)calloc(elements * sizeof(uint16_t),sizeof(float));
	float *expected = (float *)calloc(elements,sizeof(float));
	float *actual = (float *)calloc(elements,sizeof(float));
	uint16_t *norm_packed = (uint16_t *)calloc(SPARK_QWEN36_DV,sizeof(uint16_t));
	float *norm_exact = (float *)calloc(SPARK_QWEN36_DV,sizeof(float));
	SparkQwen36ValMetrics metrics;
	uint32_t row,head,element;
	cudaError_t error;
	if (packed == 0 || exact == 0 || expected == 0 || actual == 0 || norm_packed == 0 || norm_exact == 0)
		return(SparkQwen36ValFail("gated_norm","host_alloc"));
	SparkQwen36ValRandomState = 51u;
	SparkQwen36ValFillBf16(packed,exact,elements,1.0f);              /* core */
	SparkQwen36ValFillBf16(packed + elements,exact + elements,elements,1.0f); /* z */
	SparkQwen36ValFillBf16(norm_packed,norm_exact,SPARK_QWEN36_DV,0.5f);
	error = cudaMemcpy(device->core_out,packed,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->z_bf16,packed + elements,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->gdn_norm_weight,norm_packed,SPARK_QWEN36_DV * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SparkQwen36LaunchGatedNorm(cudaStreamPerThread,device->core_out,device->z_bf16,&device->gdn_weights,device->gated_out,SPARK_QWEN36_VALIDATION_ROWS,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(packed,device->gated_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SparkQwen36ValCuda(error,"gated_norm") != 0)
		return(1);
	for (row = 0u; row < SPARK_QWEN36_VALIDATION_ROWS; row++)
		for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
		{
			uint64_t base = ((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_DV);
			float variance = 0.0f,inverse;
			for (element = 0u; element < SPARK_QWEN36_DV; element++)
				variance += exact[base + element] * exact[base + element];
			inverse = 1.0f / sqrtf((variance / (float)SPARK_QWEN36_DV) + SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
			for (element = 0u; element < SPARK_QWEN36_DV; element++)
				expected[base + element] = exact[base + element] * inverse * norm_exact[element] *
					SparkQwen36ValSilu(exact[elements + base + element]);
		}
	for (row = 0u; row < elements; row++)
		actual[row] = SparkQwen36ValFromBf16(packed[row]);
	SparkQwen36ValMeasure(&metrics,actual,expected,elements);
	free(packed); free(exact); free(expected); free(actual); free(norm_packed); free(norm_exact);
	return(SparkQwen36ValReport("gated_norm",&metrics,5e-3,0.99999));
}

/* Five tokens through AttnPrepare into a one-block paged cache, then one
 * decode of the newest position, against the per-kv-head oracle. */
static int SparkQwen36ValCheckAttention(SparkQwen36ValDevice *device)
{
	const uint32_t tokens = SPARK_QWEN36_VALIDATION_ATTN_TOKENS;
	const uint64_t fused_elements = (uint64_t)tokens * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION;
	const uint64_t kv_elements = (uint64_t)tokens * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION;
	const uint64_t cache_elements = (uint64_t)SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS;
	uint16_t *q_packed = (uint16_t *)calloc(fused_elements,sizeof(uint16_t));
	uint16_t *k_packed = (uint16_t *)calloc(kv_elements,sizeof(uint16_t));
	uint16_t *v_packed = (uint16_t *)calloc(kv_elements,sizeof(uint16_t));
	float *q_exact = (float *)calloc(fused_elements,sizeof(float));
	float *k_exact = (float *)calloc(kv_elements,sizeof(float));
	float *v_exact = (float *)calloc(kv_elements,sizeof(float));
	float *expected = (float *)calloc(SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION,sizeof(float));
	float *actual = (float *)calloc(SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION,sizeof(float));
	uint16_t *out_packed = (uint16_t *)calloc(SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION,sizeof(uint16_t));
	uint16_t *norm_packed = (uint16_t *)calloc(2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,sizeof(uint16_t));
	float *q_norm_exact = (float *)calloc(SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
	float *k_norm_exact = (float *)calloc(SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
	uint32_t slot_mapping[SPARK_QWEN36_VALIDATION_ATTN_TOKENS];
	uint64_t positions[SPARK_QWEN36_VALIDATION_ATTN_TOKENS];
	uint32_t block_indices[1] = {0u};
	uint32_t block_counts[1] = {1u};
	uint32_t lane_zero[1] = {0u};
	uint32_t context[1] = {SPARK_QWEN36_VALIDATION_ATTN_TOKENS};
	uint32_t token;
	cudaError_t error;
	SparkQwen36ValMetrics metrics;
	SparkQwen36KvBlockTableView table;
	void *kv_cache = 0;
	uint16_t *device_q = 0,*device_k = 0,*device_v = 0,*device_out = 0;
	uint32_t *device_slots = 0,*device_blocks = 0,*device_counts = 0,*device_lane = 0,*device_context = 0;
	uint64_t *device_positions = 0;
	if (q_packed == 0 || k_packed == 0 || v_packed == 0 || q_exact == 0 || k_exact == 0 ||
		v_exact == 0 || expected == 0 || actual == 0 || out_packed == 0 || norm_packed == 0 ||
		q_norm_exact == 0 || k_norm_exact == 0)
		return(SparkQwen36ValFail("attn_decode","host_alloc"));
	SparkQwen36ValRandomState = 67u;
	SparkQwen36ValFillBf16(q_packed,q_exact,fused_elements,0.5f);
	SparkQwen36ValFillBf16(k_packed,k_exact,kv_elements,0.5f);
	SparkQwen36ValFillBf16(v_packed,v_exact,kv_elements,0.5f);
	SparkQwen36ValFillBf16(norm_packed,q_norm_exact,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,0.5f);
	SparkQwen36ValFillBf16(norm_packed + SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,k_norm_exact,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,0.5f);
	for (token = 0u; token < tokens; token++)
	{
		slot_mapping[token] = token;       /* block 0, slots 0..tokens-1 */
		positions[token] = token;
	}
	error = cudaMalloc(&kv_cache,cache_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMemset(kv_cache,0,cache_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_q,fused_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_k,kv_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_v,kv_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_out,SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_slots,sizeof(slot_mapping));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_positions,sizeof(positions));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_blocks,sizeof(block_indices));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_counts,sizeof(block_counts));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_lane,sizeof(lane_zero));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_context,sizeof(context));
	if (error == cudaSuccess) error = cudaMemcpy(device_q,q_packed,fused_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_k,k_packed,kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_v,v_packed,kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_slots,slot_mapping,sizeof(slot_mapping),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_positions,positions,sizeof(positions),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_blocks,block_indices,sizeof(block_indices),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_counts,block_counts,sizeof(block_counts),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_lane,lane_zero,sizeof(lane_zero),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_context,context,sizeof(context),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->q_norm_weight,norm_packed,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->k_norm_weight,norm_packed + SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	/* The cache layout: one layer, one block. layer stride == block stride
	 * at cache_layer_count 1, mirroring AllocatePools. */
	if (error == cudaSuccess)
		error = SparkQwen36LaunchAttnPrepare(cudaStreamPerThread,device_q,device_k,device_v,&device->attn_weights,kv_cache,device_slots,device_positions,tokens,0u,cache_elements,cache_elements,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	table.descriptor_bytes = sizeof(table);
	table.block_token_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	table.lane_count = 1u;
	table.lane_stride = 1u;
	table.lane_capacity = 1u;
	table.physical_block_indices = device_blocks;
	table.lane_physical_block_counts = device_counts;
	table.host_physical_block_indices = block_indices;
	table.host_lane_physical_block_counts = block_counts;
	if (error == cudaSuccess)
		error = SparkQwen36LaunchAttnDecode(cudaStreamPerThread,device_q + ((uint64_t)(tokens - 1u) * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION),kv_cache,&table,device_lane,device_context,device_out,1u,0u,cache_elements,cache_elements);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(out_packed,device_out,SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SparkQwen36ValCuda(error,"attn_decode") != 0)
		return(1);
	/* Oracle: normalize + rope the cached K at its position, then one
	 * RefAttention call per kv head over the newest fused query row. */
	{
		float *k_cache = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
		float *v_cache = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,sizeof(float));
		uint32_t kv_head,element;
		if (k_cache == 0 || v_cache == 0)
			return(SparkQwen36ValFail("attn_decode","oracle_alloc"));
		for (kv_head = 0u; kv_head < SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT; kv_head++)
		{
			for (token = 0u; token < tokens; token++)
			{
				const float *k_row = k_exact + (((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION));
				float *cache_row = k_cache + ((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
				SparkQwen36ValRmsNorm(k_row,k_norm_exact,cache_row,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
				SparkQwen36ValRope(cache_row,SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION,token,SPARK_QWEN36_MODEL_ATTN_ROPE_THETA);
				for (element = 0u; element < SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION; element++)
					v_cache[((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + element] =
						v_exact[((uint64_t)token * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + element];
			}
			SparkQwen36ValAttention(
				q_exact + (((uint64_t)(tokens - 1u) * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION) + ((uint64_t)kv_head * 6u * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION)),
				k_cache,v_cache,q_norm_exact,
				expected + ((uint64_t)kv_head * 6u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION),
				6u,tokens,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
		}
		free(k_cache);
		free(v_cache);
	}
	for (token = 0u; token < SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION; token++)
		actual[token] = SparkQwen36ValFromBf16(out_packed[token]);
	SparkQwen36ValMeasure(&metrics,actual,expected,SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION);
	cudaFree(kv_cache); cudaFree(device_q); cudaFree(device_k); cudaFree(device_v);
	cudaFree(device_out); cudaFree(device_slots); cudaFree(device_positions);
	cudaFree(device_blocks); cudaFree(device_counts); cudaFree(device_lane); cudaFree(device_context);
	free(q_packed); free(k_packed); free(v_packed); free(q_exact); free(k_exact);
	free(v_exact); free(expected); free(actual); free(out_packed); free(norm_packed);
	free(q_norm_exact); free(k_norm_exact);
	return(SparkQwen36ValReport("attn_decode",&metrics,5e-3,0.99999));
}

/* 128 tokens (two chunks) through the chunk kernels on one lane, state and
 * per-token outputs against the recurrence oracle on the same inputs. */
static int SparkQwen36ValCheckGdnChunk(SparkQwen36ValDevice *device)
{
	const uint32_t tokens = SPARK_QWEN36_VALIDATION_CHUNK_TOKENS;
	const uint64_t conv_elements = (uint64_t)tokens * SPARK_QWEN36_CONV;
	const uint64_t out_elements = (uint64_t)tokens * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION;
	const uint64_t state_elements = (uint64_t)SPARK_QWEN36_HEADS * SPARK_QWEN36_DK * SPARK_QWEN36_DV;
	uint16_t *conv_packed = (uint16_t *)calloc(conv_elements,sizeof(uint16_t));
	float *exact = (float *)calloc(conv_elements,sizeof(float));
	uint16_t *out_packed = (uint16_t *)calloc(out_elements,sizeof(uint16_t));
	float *actual = (float *)calloc(out_elements,sizeof(float));
	float *expected = (float *)calloc(out_elements,sizeof(float));
	float *state_device = (float *)calloc(state_elements,sizeof(float));
	float *state_oracle = (float *)calloc(state_elements,sizeof(float));
	float *log_decay = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_HEADS,sizeof(float));
	float *beta = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_HEADS,sizeof(float));
	uint32_t head,token;
	uint64_t index;
	cudaError_t error;
	SparkQwen36ValMetrics metrics;
	if (conv_packed == 0 || exact == 0 || out_packed == 0 || actual == 0 || expected == 0 ||
		state_device == 0 || state_oracle == 0 || log_decay == 0 || beta == 0)
		return(SparkQwen36ValFail("gdn_chunk","host_alloc"));
	SparkQwen36ValRandomState = 83u;
	SparkQwen36ValFillBf16(conv_packed,exact,conv_elements,1.0f);
	for (index = 0u; index < (uint64_t)tokens * SPARK_QWEN36_HEADS; index++)
	{
		log_decay[index] = SparkQwen36ValUniform(0.25f) - 0.25f;
		beta[index] = 0.25f + fabsf(SparkQwen36ValUniform(0.5f));
	}
	error = cudaMemset(device->state,0,state_elements * sizeof(float));
	if (error == cudaSuccess) error = cudaMemcpy(device->qkv,conv_packed,conv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,log_decay,(uint64_t)tokens * SPARK_QWEN36_HEADS * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->beta,beta,(uint64_t)tokens * SPARK_QWEN36_HEADS * sizeof(float),cudaMemcpyHostToDevice);
	/* The launcher walks ONE chunk per call (token_count <= GDN_CHUNK_TOKENS);
	 * the module loops it over the frame, so do the same here. */
	for (uint32_t base = 0u; base < tokens && error == cudaSuccess; base += SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS)
		error = SparkQwen36LaunchGdnChunk(cudaStreamPerThread,
			device->qkv + ((uint64_t)base * SPARK_QWEN36_CONV),
			device->log_decay + ((uint64_t)base * SPARK_QWEN36_HEADS),
			device->beta + ((uint64_t)base * SPARK_QWEN36_HEADS),
			device->chunk_qn,device->chunk_kn,device->chunk_cum_g,device->chunk_decay,
			device->chunk_attn,device->chunk_w,device->chunk_kg,
			&device->pool,device->core_out + ((uint64_t)base * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION),
			0u,SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS,0u);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(out_packed,device->core_out,out_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(state_device,device->state,state_elements * sizeof(float),cudaMemcpyDeviceToHost);
	if (SparkQwen36ValCuda(error,"gdn_chunk") != 0)
		return(1);
	for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
	{
		uint32_t key_head = head / 3u;
		float *q_head = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_DK,sizeof(float));
		float *k_head = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_DK,sizeof(float));
		float *v_head = (float *)calloc((uint64_t)tokens * SPARK_QWEN36_DV,sizeof(float));
		float *g_head = (float *)calloc(tokens,sizeof(float));
		float *b_head = (float *)calloc(tokens,sizeof(float));
		if (q_head == 0 || k_head == 0 || v_head == 0 || g_head == 0 || b_head == 0)
			return(SparkQwen36ValFail("gdn_chunk","oracle_alloc"));
		for (token = 0u; token < tokens; token++)
		{
			uint64_t base = (uint64_t)token * SPARK_QWEN36_CONV;
			memcpy(q_head + ((uint64_t)token * SPARK_QWEN36_DK),exact + base + ((uint64_t)key_head * SPARK_QWEN36_DK),SPARK_QWEN36_DK * sizeof(float));
			memcpy(k_head + ((uint64_t)token * SPARK_QWEN36_DK),exact + base + SPARK_QWEN36_MODEL_GDN_QK_DIMENSION + ((uint64_t)key_head * SPARK_QWEN36_DK),SPARK_QWEN36_DK * sizeof(float));
			memcpy(v_head + ((uint64_t)token * SPARK_QWEN36_DV),exact + base + (2ull * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_DV),SPARK_QWEN36_DV * sizeof(float));
			g_head[token] = log_decay[((uint64_t)token * SPARK_QWEN36_HEADS) + head];
			b_head[token] = beta[((uint64_t)token * SPARK_QWEN36_HEADS) + head];
		}
		SparkQwen36ValGdnRecurrence(q_head,k_head,v_head,g_head,b_head,
			state_oracle + ((uint64_t)head * SPARK_QWEN36_DK * SPARK_QWEN36_DV),
			expected + ((uint64_t)head * tokens * SPARK_QWEN36_DV),tokens);
		free(q_head); free(k_head); free(v_head); free(g_head); free(b_head);
	}
	/* The recurrence oracle writes each head's outputs into its own
	 * tokens x dv slab; the kernel's core_out is [token][head][dv] -
	 * re-map before comparing. */
	{
		float *expected_ordered = (float *)calloc(out_elements,sizeof(float));
		if (expected_ordered == 0)
			return(SparkQwen36ValFail("gdn_chunk","order_alloc"));
		for (token = 0u; token < tokens; token++)
			for (head = 0u; head < SPARK_QWEN36_HEADS; head++)
				memcpy(expected_ordered + ((uint64_t)token * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_DV),
					expected + ((uint64_t)head * tokens * SPARK_QWEN36_DV) + ((uint64_t)token * SPARK_QWEN36_DV),
					SPARK_QWEN36_DV * sizeof(float));
		for (index = 0u; index < out_elements; index++)
			actual[index] = SparkQwen36ValFromBf16(out_packed[index]);
		SparkQwen36ValMeasure(&metrics,actual,expected_ordered,out_elements);
		free(expected_ordered);
	}
	if (SparkQwen36ValReport("gdn_chunk_output",&metrics,2e-2,0.999) != 0)
		return(1);
	SparkQwen36ValMeasure(&metrics,state_device,state_oracle,state_elements);
	free(conv_packed); free(exact); free(out_packed); free(actual); free(expected);
	free(state_device); free(state_oracle); free(log_decay); free(beta);
	return(SparkQwen36ValReport("gdn_chunk_state",&metrics,2e-2,0.999));
}

/* -- module tier ---------------------------------------------------------- */

typedef struct SparkQwen36ValCapture
{
	/* A prefill frame sends token_count rows, so the capture must hold the
	 * larger of the prefill width and the decode row count. */
	uint16_t hidden[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
	uint32_t sends;
} SparkQwen36ValCapture;

static SparkStatus SparkQwen36ValCaptureSend(SparkHiddenTransportSession *session, const SparkHiddenTransportPacket *packet)
{
	SparkQwen36ValCapture *capture = (SparkQwen36ValCapture *)session;
	uint64_t bytes = (uint64_t)packet->active_sequence_count * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * 2u;
	if (packet->hidden_bf16 == 0 || packet->active_sequence_count > SPARK_QWEN36_VALIDATION_PREFILL_TOKENS ||
		packet->hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION)
		return(SPARK_STATUS_VALIDATION_FAILED);
	if (cudaMemcpy(capture->hidden,packet->hidden_bf16,bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
		return(SPARK_STATUS_IO_ERROR);
	capture->sends++;
	return(SPARK_STATUS_OK);
}

typedef struct SparkQwen36ValModule
{
	void *state;
	/* Prefill frames carry PREFILL_TOKENS ids; decode uses only ROWS lanes. */
	uint32_t token_ids[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS];
	uint32_t output_token_ids[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS];
	uint32_t head_stage;
	uint32_t lanes[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS];
	uint64_t positions[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS];
	uint64_t sequence_ids[SPARK_QWEN36_VALIDATION_PREFILL_TOKENS];
	uint32_t host_blocks[SPARK_QWEN36_VALIDATION_KV_LANES];
	uint32_t host_counts[SPARK_QWEN36_VALIDATION_KV_LANES];
	uint32_t *device_blocks;
	uint32_t *device_counts;
	SparkQwen36KvBlockTableView table;
	SparkQwen36DecodeBatchView decode_batch;
	SparkQwen36PrefillFrameView prefill_view;
	SparkQwen36ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkQwen36ValCapture capture;
} SparkQwen36ValModule;

static int SparkQwen36ValModuleInitialize(SparkQwen36ValModule *module)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkStatus status;
	const char *stage_count_text;
	uint32_t lane;
	cudaError_t error;
	memset(module,0,sizeof(*module));
	/* head_stage: the stage owns the final head iff it is the whole-stack
	 * last stage (STAGE_COUNT == 1), independent of TP_DEGREE (TP1
	 * full-width is whole-stack and owns the head). */
	stage_count_text = getenv("SPARK_QWEN36_STAGE_COUNT");
	module->head_stage = stage_count_text != 0 && strcmp(stage_count_text,"1") == 0 ? 1u : 0u;
	for (lane = 0u; lane < SPARK_QWEN36_VALIDATION_KV_LANES; lane++)
	{
		module->host_blocks[lane] = lane;
		module->host_counts[lane] = 1u;
	}
	error = cudaMalloc((void **)&module->device_blocks,sizeof(module->host_blocks));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_blocks,module->host_blocks,sizeof(module->host_blocks),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMalloc((void **)&module->device_counts,sizeof(module->host_counts));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_counts,module->host_counts,sizeof(module->host_counts),cudaMemcpyHostToDevice);
	if (SparkQwen36ValCuda(error,"module_table_alloc") != 0)
		return(1);
	module->table.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	module->table.descriptor_bytes = sizeof(module->table);
	module->table.block_token_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	module->table.lane_count = SPARK_QWEN36_VALIDATION_KV_LANES;
	module->table.lane_stride = 1u;
	module->table.lane_capacity = SPARK_QWEN36_VALIDATION_KV_LANES;
	module->table.physical_block_indices = module->device_blocks;
	module->table.lane_physical_block_counts = module->device_counts;
	module->table.host_physical_block_indices = module->host_blocks;
	module->table.host_lane_physical_block_counts = module->host_counts;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "Qwen/Qwen3.6-27B";
	configuration.model_revision = "validation";
	configuration.stage_name = "qwen36_resident_decode_stage";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "qwen36_resident_decode_stage";
	configuration.configuration_json = "{}";
	configuration.configuration_json_bytes = 2u;
	memset(&host_services,0,sizeof(host_services));
	host_services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services.descriptor_bytes = sizeof(host_services);
	host_services.node_id = "spark-qwen36-validator";
	host_services.node_target = "cuda.sm121.qwen36.resident_decode_stage.bf16";
	host_services.execution_stream = (void *)cudaStreamPerThread;
	status = SparkQwen36ResidentDecodeStageInitialize(&configuration,&host_services,&module->state);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"qwen36_validation failure=module_initialize status=%d\n",(int)status);
		return(1);
	}
	return(0);
}

/* flags select decode vs prefill; rows and the lane/position/sequence arrays
 * are already staged on the module struct. */
static int SparkQwen36ValModuleExecute(SparkQwen36ValModule *module, uint32_t prefill, uint32_t rows, uint32_t lane, uint32_t draft_count, const SparkQwen36MtpDraftView *draft_view)
{
	SparkStatus status;
	memset(&module->context,0,sizeof(module->context));
	memset(&module->frame,0,sizeof(module->frame));
	module->context.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	module->context.descriptor_bytes = sizeof(module->context);
	module->context.flags =
		SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
		(module->head_stage == 0u
			? SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT
			: 0u) |
		(prefill != 0u
			? SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW
			: SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) |
		(draft_count != 0u ? SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER : 0u);
	module->context.mtp_draft = draft_view;
	module->context.kv_block_table = &module->table;
	if (prefill != 0u)
	{
		module->prefill_view.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		module->prefill_view.descriptor_bytes = sizeof(module->prefill_view);
		module->prefill_view.lane_index = lane;
		module->prefill_view.token_count = rows;
		module->prefill_view.base_position = module->positions[0];
		module->prefill_view.sequence_id = module->sequence_ids[0];
		module->context.prefill_frame = &module->prefill_view;
	}
	else
	{
		module->decode_batch.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		module->decode_batch.descriptor_bytes = sizeof(module->decode_batch);
		module->decode_batch.row_count = rows;
		module->decode_batch.row_lane_indices = module->lanes;
		module->decode_batch.row_positions = module->positions;
		module->decode_batch.row_sequence_ids = module->sequence_ids;
		module->context.decode_batch = &module->decode_batch;
	}
	module->context.hidden_output_transport_session = module->head_stage != 0u ? 0 : (SparkHiddenTransportSession *)&module->capture;
	module->context.hidden_output_send_function = module->head_stage != 0u ? 0 : SparkQwen36ValCaptureSend;
	module->frame.program_id = 1u;
	module->frame.tokens_per_sequence = 1u;
	module->frame.request_id = 1u;
	module->frame.sequence_id = module->sequence_ids[0];
	module->frame.sequence_position = module->positions[0];
	module->frame.active_slot_count = prefill != 0u ? 1u : rows;
	module->frame.new_token_count = rows;
	module->frame.flags = prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
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
	status = SparkQwen36ResidentDecodeStageExecute(module->state,&module->frame);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"qwen36_validation failure=module_execute prefill=%u rows=%u status=%d\n",prefill,rows,(int)status);
		return(1);
	}
	return(0);
}

static int SparkQwen36ValCheckFinite(const char *check, const uint16_t *hidden, uint64_t rows)
{
	uint64_t index,count = rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	for (index = 0u; index < count; index++)
		if (isfinite(SparkQwen36ValFromBf16(hidden[index])) == 0)
			return(SparkQwen36ValFail(check,"nonfinite"));
	return(0);
}

/* MTP draft chain qualification (A1 step 2): continue lane 0 with a decode
 * at position 9 and draft the next two tokens via MTP_DRAFT_AFTER, then check
 * the draft ids are in-vocab. There is no CPU-oracle MTP reference, so
 * in-vocab is the feasible qualification here; determinism is covered by the
 * module_determinism check above (bit-exact fresh-instance re-execution). */
static int SparkQwen36ValCheckMtpDraft(SparkQwen36ValModule *module)
{
	SparkQwen36MtpDraftView draft_view;
	uint32_t draft;
	module->token_ids[0] = 4242u;
	module->lanes[0] = 0u;
	module->positions[0] = SPARK_QWEN36_VALIDATION_PREFILL_TOKENS + 1u;
	module->sequence_ids[0] = 1u;
	memset(&draft_view,0,sizeof(draft_view));
	draft_view.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION;
	draft_view.descriptor_bytes = sizeof(draft_view);
	draft_view.lane_index = 0u;
	draft_view.draft_token_count = 2u;
	draft_view.base_position = (uint64_t)SPARK_QWEN36_VALIDATION_PREFILL_TOKENS + 2u;
	draft_view.sequence_id = 1u;
	draft_view.row_token_ids = module->token_ids;
	if (SparkQwen36ValModuleExecute(module,0u,1u,0u,2u,&draft_view) != 0)
		return(1);
	for (draft = 0u; draft < 2u; draft++)
	{
		if (module->output_token_ids[1u + draft] >= SPARK_QWEN36_MODEL_VOCAB_COUNT)
			return(SparkQwen36ValFail("module_mtp_draft","out_of_vocab"));
	}
	printf("qwen36_validation check=module_mtp_draft in_vocab=1 drafts=[%u,%u]\n",module->output_token_ids[1],module->output_token_ids[2]);
	return(0);
}

/* The module flow: prefill 8 tokens on lane 0, decode position 8 on lane 0;
 * prefill the same 8 on lane 1 plus a 1-token warm prefill at position 8.
 * The decode path (recurrent step) and the warm-prefill path (chunk walk)
 * must land on the same hidden. Returns bit-exactness separately. */
static int SparkQwen36ValCheckModule(void)
{
	SparkQwen36ValModule module;
	uint16_t decode_hidden[SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
	uint16_t prefill_hidden[SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
	uint16_t rerun_hidden[SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
	uint32_t decode_token = 0u,prefill_token = 0u,rerun_token = 0u;
	uint32_t index;
	SparkQwen36ValMetrics metrics;
	SparkModelDriverAdmissionRequest admission;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	if (SparkQwen36ValModuleInitialize(&module) != 0)
		return(1);
	/* Admission and snapshot smoke: a decode admit must accept, and the
	 * snapshot must succeed. Run before the frames so pipeline slots are free. */
	memset(&admission,0,sizeof(admission));
	admission.descriptor_bytes = sizeof(admission);
	admission.program_id = 1u;
	admission.frame_flags = 0u;
	admission.new_token_count = 1u;
	admission.active_slot_count = 1u;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkQwen36ResidentDecodeStageAdmit(module.state,&admission,&decision);
	if (status != SPARK_STATUS_OK || decision.accepted == 0u)
		return(SparkQwen36ValFail("module_admit","rejected"));
	memset(&snapshot,0,sizeof(snapshot));
	SparkModelDriverInitializeRuntimeSnapshot(&snapshot,1u);
	status = SparkQwen36ResidentDecodeStageSnapshot(module.state,1u,&snapshot);
	if (status != SPARK_STATUS_OK)
		return(SparkQwen36ValFail("module_snapshot","status"));
	printf("qwen36_validation check=module_admit_snapshot admit=ok snapshot=ok\n");
	for (index = 0u; index < SPARK_QWEN36_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 1u;
	if (SparkQwen36ValModuleExecute(&module,1u,SPARK_QWEN36_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
		return(1);
	if (module.capture.sends != (module.head_stage != 0u ? 0u : 1u))
		return(SparkQwen36ValFail("module_prefill",module.head_stage != 0u ? "unexpected_hidden_send" : "no_hidden_send"));
	if (module.head_stage == 0u && SparkQwen36ValCheckFinite("module_prefill",module.capture.hidden,SPARK_QWEN36_VALIDATION_PREFILL_TOKENS) != 0)
		return(1);
	/* Decode position 8 on lane 0, token chosen arbitrarily but fixed. */
	module.token_ids[0] = 4242u;
	module.lanes[0] = 0u;
	module.positions[0] = SPARK_QWEN36_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 1u;
	if (SparkQwen36ValModuleExecute(&module,0u,1u,0u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(decode_hidden,module.capture.hidden,sizeof(decode_hidden));
		if (SparkQwen36ValCheckFinite("module_decode",decode_hidden,1u) != 0)
			return(1);
	}
	else
		decode_token = module.output_token_ids[0];
	/* Lane 1: same 8 tokens as a fresh sequence, then a warm 1-token
	 * prefill at position 8 - the chunk-walk twin of lane 0's decode. */
	for (index = 0u; index < SPARK_QWEN36_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 2u;
	if (SparkQwen36ValModuleExecute(&module,1u,SPARK_QWEN36_VALIDATION_PREFILL_TOKENS,1u,0u,0) != 0)
		return(1);
	module.token_ids[0] = 4242u;
	module.positions[0] = SPARK_QWEN36_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 2u;
	if (SparkQwen36ValModuleExecute(&module,1u,1u,1u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(prefill_hidden,module.capture.hidden,sizeof(prefill_hidden));
		{
			float actual[SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
			float reference[SPARK_QWEN36_MODEL_HIDDEN_DIMENSION];
			for (index = 0u; index < SPARK_QWEN36_MODEL_HIDDEN_DIMENSION; index++)
			{
				actual[index] = SparkQwen36ValFromBf16(decode_hidden[index]);
				reference[index] = SparkQwen36ValFromBf16(prefill_hidden[index]);
			}
			SparkQwen36ValMeasure(&metrics,actual,reference,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION);
			if (SparkQwen36ValReport("module_decode_vs_prefill",&metrics,5e-2,0.999) != 0)
				return(1);
		}
	}
	else
	{
		prefill_token = module.output_token_ids[0];
		printf("qwen36_validation check=module_decode_vs_prefill decode_token=%u prefill_token=%u bit_exact=%d\n",decode_token,prefill_token,decode_token == prefill_token ? 1 : 0);
		if (decode_token != prefill_token)
			return(SparkQwen36ValFail("module_decode_vs_prefill","token_mismatch"));
	}
	/* Determinism: a fresh instance must reproduce lane 0's decode hidden
	 * bit for bit. */
	{
		SparkQwen36ValModule rerun;
		if (SparkQwen36ValModuleInitialize(&rerun) != 0)
			return(1);
		for (index = 0u; index < SPARK_QWEN36_VALIDATION_PREFILL_TOKENS; index++)
			rerun.token_ids[index] = 1000u + (index * 37u) % 200000u;
		rerun.positions[0] = 0u;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen36ValModuleExecute(&rerun,1u,SPARK_QWEN36_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
			return(1);
		rerun.token_ids[0] = 4242u;
		rerun.lanes[0] = 0u;
		rerun.positions[0] = SPARK_QWEN36_VALIDATION_PREFILL_TOKENS;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen36ValModuleExecute(&rerun,0u,1u,0u,0u,0) != 0)
			return(1);
		if (module.head_stage == 0u)
		{
			memcpy(rerun_hidden,rerun.capture.hidden,sizeof(rerun_hidden));
			if (memcmp(decode_hidden,rerun_hidden,sizeof(decode_hidden)) != 0)
				return(SparkQwen36ValFail("module_determinism","fresh_instance_mismatch"));
		}
		else
		{
			rerun_token = rerun.output_token_ids[0];
			if (rerun_token != decode_token)
				return(SparkQwen36ValFail("module_determinism","fresh_instance_token_mismatch"));
		}
		SparkQwen36ResidentDecodeStageDestroy(rerun.state);
		cudaFree(rerun.device_blocks);
		cudaFree(rerun.device_counts);
		printf("qwen36_validation check=module_determinism bit_exact=1\n");
	}
	if (module.head_stage != 0u && SparkQwen36ValCheckMtpDraft(&module) != 0)
		return(1);
	SparkQwen36ResidentDecodeStageDestroy(module.state);
	cudaFree(module.device_blocks);
	cudaFree(module.device_counts);
	return(0);
}

int main(int argc, char **argv)
{
	SparkQwen36ValDevice device;
	int result = 0;
	if (argc != 2 || strlen(argv[1]) != 64u)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if (SparkQwen36ValCuda(SparkQwen36ConfigureCudaKernels(),"configure") != 0)
		return(1);
	if (SparkQwen36ValDeviceSetup(&device) != 0)
		return(1);
	if (result == 0) result = SparkQwen36ValCheckDecayBeta(&device);
	if (result == 0) result = SparkQwen36ValCheckConv(&device);
	if (result == 0) result = SparkQwen36ValCheckGdnStep(&device);
	if (result == 0) result = SparkQwen36ValCheckGatedNorm(&device);
	if (result == 0) result = SparkQwen36ValCheckAttention(&device);
	if (result == 0) result = SparkQwen36ValCheckGdnChunk(&device);
	if (result == 0) result = SparkQwen36ValCheckModule();
	if (result == 0)
		printf("qwen36_validation PASS\n");
	return(result);
}
