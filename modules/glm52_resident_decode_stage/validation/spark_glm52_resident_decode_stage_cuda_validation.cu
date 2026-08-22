#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "spark_glm52_resident_decode_stage_internal.h"

/*
 * GLM 5.2 resident decode stage, hardware validation v1 (sm_121a).
 *
 * A minimal retained-receipt numerical gate: ONE dense layer's forward -
 * the attention chunk plus the dense MLP chunk, driven through the module's
 * own exported launchers (SparkGlm52LaunchCudaWaveBegin / LayerAttention /
 * LayerMlp) exactly as SparkGlm52TpChainAdvance drives them - compared
 * against an fp32 CPU oracle that restates the shared-kernel formulas here,
 * self-contained under nvcc and sharing no code with what it validates.
 * The indexer is legitimately absent: the fixture's context fits
 * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT, the case the production
 * prologue skips the indexer for. The routed-expert tier and the DSA sparse
 * tier are later validators' work; this one exists so require_gpu_validator
 * stops pointing at nothing and validate/publish/publish_variants carry a
 * numerical receipt again.
 *
 * Every comparison prints its numbers; the thresholds are the guard, the
 * numbers are the evidence.
 */

#define SPARK_GLM52_VALIDATION_TOKENS 4u
#define SPARK_GLM52_VALIDATION_LANES 1u
#define SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE 2u
#define SPARK_GLM52_VALIDATION_EMBED_ROWS 8u

#define SPARK_GLM52_VHIDDEN SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_VQUERY_A SPARK_GLM52_MODEL_QUERY_A_DIMENSION
#define SPARK_GLM52_VQK_NOPE SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION
#define SPARK_GLM52_VROPE SPARK_GLM52_MODEL_ROPE_DIMENSION
#define SPARK_GLM52_VLATENT SPARK_GLM52_MODEL_LATENT_DIMENSION
#define SPARK_GLM52_VHEADS SPARK_GLM52_MODEL_HEAD_COUNT
#define SPARK_GLM52_VVALUE_DIM SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION
#define SPARK_GLM52_VQ_ROWS (SPARK_GLM52_VHEADS * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE))
#define SPARK_GLM52_VATTN_COLS (SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM)
#define SPARK_GLM52_VDENSE_INTER SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_VGATE_UP_ROWS (2u * SPARK_GLM52_VDENSE_INTER)
/* Glm52Kv geometry (source/cuda/config.h + layer.cuh): block-major pool. */
#define SPARK_GLM52_VKV_SLOT_ELEMENTS (SPARK_GLM52_VLATENT + SPARK_GLM52_VROPE)
#define SPARK_GLM52_VKV_SLOT_BYTES (SPARK_GLM52_VKV_SLOT_ELEMENTS * 2u)
#define SPARK_GLM52_VKV_PAGE_SLOTS 64u
#define SPARK_GLM52_VKV_LAYER_BYTES (SPARK_GLM52_VKV_SLOT_BYTES * SPARK_GLM52_VKV_PAGE_SLOTS)
#define SPARK_GLM52_VKV_PAGE_BYTES (SPARK_GLM52_VKV_LAYER_BYTES * SPARK_GLM52_MODEL_LAYER_COUNT)
#define SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS 6u

extern "C" int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count);

/* -- fixtures -------------------------------------------------------------- */

static uint32_t SparkGlm52ValRandomState;

static uint32_t SparkGlm52ValNext(void)
{
	SparkGlm52ValRandomState = SparkGlm52ValRandomState * 1664525u + 1013904223u;
	return(SparkGlm52ValRandomState >> 8u);
}

/* Round to nearest even: the LmFloatToBf16 contract (dtype.cuh). */
static uint16_t SparkGlm52ValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return((uint16_t)(bits >> 16u));
}

static float SparkGlm52ValFromBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float converted;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}

static void SparkGlm52ValFill(uint16_t *packed,float *exact,uint64_t count,float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SparkGlm52ValBf16((((float)(int32_t)(SparkGlm52ValNext() & 0xffffu) - 32768.0f) * scale) / 32768.0f);
		exact[index] = SparkGlm52ValFromBf16(packed[index]);
	}
}

static void SparkGlm52ValFillNorm(uint16_t *packed,float *exact,uint64_t count)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SparkGlm52ValBf16(1.0f + (((float)(int32_t)(SparkGlm52ValNext() & 0xffu) - 128.0f) * 0.001f));
		exact[index] = SparkGlm52ValFromBf16(packed[index]);
	}
}

static int SparkGlm52ValFail(const char *check,const char *detail)
{
	fprintf(stderr,"glm52_validation failure=%s detail=%s\n",check,detail);
	return(1);
}

typedef struct SparkGlm52ValMetrics
{
	double difference_l2;
	double reference_l2;
	double actual_l2;
	double dot;
	double maximum_absolute;
	uint64_t count;
} SparkGlm52ValMetrics;

static void SparkGlm52ValMeasure(SparkGlm52ValMetrics *metrics,const float *actual,const float *reference,uint64_t count)
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
		if ( fabs(difference) > metrics->maximum_absolute )
			metrics->maximum_absolute = fabs(difference);
	}
}

static int SparkGlm52ValReport(const char *check,const SparkGlm52ValMetrics *metrics,double max_relative_l2,double minimum_cosine)
{
	double relative_l2 = metrics->reference_l2 > 0.0
		? sqrt(metrics->difference_l2 / metrics->reference_l2) : INFINITY;
	double cosine = metrics->actual_l2 > 0.0 && metrics->reference_l2 > 0.0
		? metrics->dot / sqrt(metrics->actual_l2 * metrics->reference_l2) : 0.0;
	printf("glm52_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)metrics->count,relative_l2,cosine,metrics->maximum_absolute);
	if ( isfinite(relative_l2) == 0 || relative_l2 > max_relative_l2 )
		return(SparkGlm52ValFail(check,"relative_l2"));
	if ( cosine < minimum_cosine )
		return(SparkGlm52ValFail(check,"cosine"));
	return(0);
}

/* -- device fixture --------------------------------------------------------- */

typedef struct SparkGlm52ValMatrix
{
	uint16_t *device;
	float *host;            /* exact fp32 upcast, rows x columns row-major */
	uint32_t rows;
	uint32_t columns;
} SparkGlm52ValMatrix;

typedef struct SparkGlm52ValFixture
{
	SparkGlm52LayerWeights weights;
	SparkGlm52ExecutionSlot slot;
	SparkGlm52CudaWave wave;
	cudaStream_t stream;
	uint32_t multiprocessors;
	SparkGlm52ValMatrix embedding;
	SparkGlm52ValMatrix attn_norm,q_a,q_a_norm,q_b,kv_a,kv_a_norm;
	SparkGlm52ValMatrix kv_b_key,kv_b_value,attn_output;
	SparkGlm52ValMatrix post_attn_norm,dense_gate_up,dense_down;
	uint16_t *hidden,*residual,*normed,*q_compressed,*q_bf16,*query_rope;
	uint16_t *query_latent,*kv_slot,*attention_latent,*attention_value,*attention_out;
	uint16_t *gate_up,*intermediate;
	uint16_t *scratch_small;
	float *float_scratch;
	uint32_t *token_ids,*resident_slots,*positions,*context_lengths;
	uint32_t *dense_row_offset,*dense_tile_prefix;
	uint32_t *page_table,*route_scratch;
	uint64_t *head_maxloc;
	uint8_t *kv_cache;
	uint32_t *kv_access_error;
	uint32_t host_page_table[SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE];
	uint32_t host_index_ordinals[1];
	uint32_t host_zero;
	uint32_t host_position;
	uint32_t host_token;
} SparkGlm52ValFixture;

static int SparkGlm52ValAllocMatrix(SparkGlm52ValMatrix *matrix,uint32_t rows,uint32_t columns,int mode,float scale)
{
	uint16_t *packed;
	uint64_t count = (uint64_t)rows * columns;
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->host = (float *)malloc(count * sizeof(float));
	packed = (uint16_t *)malloc(count * sizeof(uint16_t));
	if ( matrix->host == 0 || packed == 0 )
		return(SparkGlm52ValFail("fixture","host_alloc"));
	if ( cudaMalloc((void **)&matrix->device,count * sizeof(uint16_t)) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","device_alloc"));
	SparkGlm52ValRandomState += 101u;
	if ( mode == 1 )
		SparkGlm52ValFillNorm(packed,matrix->host,count);
	else
		SparkGlm52ValFill(packed,matrix->host,count,scale);
	if ( cudaMemcpy(matrix->device,packed,count * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","weight_upload"));
	free(packed);
	return(0);
}

static void SparkGlm52ValFreeMatrix(SparkGlm52ValMatrix *matrix)
{
	free(matrix->host);
	cudaFree(matrix->device);
	memset(matrix,0,sizeof(*matrix));
}

static void *SparkGlm52ValAllocZeroed(uint64_t bytes)
{
	void *pointer;
	if ( cudaMalloc(&pointer,bytes != 0u ? bytes : 16u) != cudaSuccess )
		return(0);
	if ( cudaMemset(pointer,0,bytes != 0u ? bytes : 16u) != cudaSuccess )
	{
		cudaFree(pointer);
		return(0);
	}
	return(pointer);
}

static int SparkGlm52ValFixtureSetup(SparkGlm52ValFixture *fixture)
{
	uint64_t pool_bytes;
	uint32_t lane,pages;
	memset(fixture,0,sizeof(*fixture));
	fixture->host_index_ordinals[0] = UINT32_MAX;
	fixture->host_zero = 0u;
	fixture->host_token = 0u;
	fixture->host_position = 0u;
	if ( cudaStreamCreate(&fixture->stream) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","stream_create"));
	if ( SparkGlm52ConfigureCudaModule(&fixture->multiprocessors) != 0 )
		return(SparkGlm52ValFail("configure","not_sm121"));
	/* Dense layer 0 synthetic weights. The scales keep every intermediate
	 * in an honest numeric range through six stacked projections. */
	if ( SparkGlm52ValAllocMatrix(&fixture->embedding,SPARK_GLM52_VALIDATION_EMBED_ROWS,SPARK_GLM52_VHIDDEN,0,0.05f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->attn_norm,1u,SPARK_GLM52_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_a,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_a_norm,1u,SPARK_GLM52_VQUERY_A,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_b,SPARK_GLM52_VQ_ROWS,SPARK_GLM52_VQUERY_A,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_a,SPARK_GLM52_VKV_SLOT_ELEMENTS,SPARK_GLM52_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_a_norm,1u,SPARK_GLM52_VLATENT,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_b_key,SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT,SPARK_GLM52_VQK_NOPE,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_b_value,SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM,SPARK_GLM52_VLATENT,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->attn_output,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VATTN_COLS,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->post_attn_norm,1u,SPARK_GLM52_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->dense_gate_up,SPARK_GLM52_VGATE_UP_ROWS,SPARK_GLM52_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->dense_down,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDENSE_INTER,0,0.01f) != 0 )
		return(1);
	fixture->hidden = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->residual = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->normed = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->q_compressed = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VQUERY_A * 8u * sizeof(uint16_t));
	fixture->q_bf16 = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VQ_ROWS * 8u * sizeof(uint16_t));
	fixture->query_rope = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VROPE * 8u * sizeof(uint16_t));
	fixture->query_latent = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t));
	fixture->kv_slot = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VKV_SLOT_ELEMENTS * 8u * sizeof(uint16_t));
	fixture->attention_latent = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t));
	fixture->attention_value = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM * 8u * sizeof(uint16_t));
	fixture->attention_out = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->gate_up = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * 8u * sizeof(uint16_t));
	fixture->intermediate = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * 8u * sizeof(uint16_t));
	fixture->scratch_small = (uint16_t *)SparkGlm52ValAllocZeroed(8192u * sizeof(uint16_t));
	fixture->float_scratch = (float *)SparkGlm52ValAllocZeroed(8192u * sizeof(float));
	fixture->token_ids = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->resident_slots = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->positions = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->context_lengths = (uint32_t *)SparkGlm52ValAllocZeroed(SPARK_GLM52_VALIDATION_LANES * sizeof(uint32_t));
	fixture->dense_row_offset = (uint32_t *)SparkGlm52ValAllocZeroed(4u * sizeof(uint32_t));
	fixture->dense_tile_prefix = (uint32_t *)SparkGlm52ValAllocZeroed(4u * sizeof(uint32_t));
	fixture->page_table = (uint32_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE * sizeof(uint32_t));
	fixture->route_scratch = (uint32_t *)SparkGlm52ValAllocZeroed(4096u * sizeof(uint32_t));
	fixture->head_maxloc = (uint64_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint64_t));
	pool_bytes = (uint64_t)SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE * SPARK_GLM52_VKV_PAGE_BYTES;
	fixture->kv_cache = (uint8_t *)SparkGlm52ValAllocZeroed(pool_bytes);
	fixture->kv_access_error = (uint32_t *)SparkGlm52ValAllocZeroed(SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS * sizeof(uint32_t));
	if ( fixture->hidden == 0 || fixture->residual == 0 || fixture->normed == 0 ||
		fixture->q_compressed == 0 || fixture->q_bf16 == 0 || fixture->query_rope == 0 ||
		fixture->query_latent == 0 || fixture->kv_slot == 0 || fixture->attention_latent == 0 ||
		fixture->attention_value == 0 || fixture->attention_out == 0 || fixture->gate_up == 0 ||
		fixture->intermediate == 0 || fixture->scratch_small == 0 || fixture->float_scratch == 0 ||
		fixture->token_ids == 0 || fixture->resident_slots == 0 || fixture->positions == 0 ||
		fixture->context_lengths == 0 || fixture->dense_row_offset == 0 ||
		fixture->dense_tile_prefix == 0 || fixture->page_table == 0 || fixture->route_scratch == 0 ||
		fixture->head_maxloc == 0 || fixture->kv_access_error == 0 || fixture->kv_cache == 0 )
		return(SparkGlm52ValFail("fixture","device_alloc"));
	for (lane = 0u; lane < SPARK_GLM52_VALIDATION_LANES; lane++)
		for (pages = 0u; pages < SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE; pages++)
			fixture->host_page_table[lane * SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE + pages] =
				lane * SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE + pages;
	if ( cudaMemcpy(fixture->page_table,fixture->host_page_table,sizeof(fixture->host_page_table),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","page_table_upload"));
	return(0);
}

static void SparkGlm52ValFixtureDestroy(SparkGlm52ValFixture *fixture)
{
	SparkGlm52ValFreeMatrix(&fixture->embedding);
	SparkGlm52ValFreeMatrix(&fixture->attn_norm);
	SparkGlm52ValFreeMatrix(&fixture->q_a);
	SparkGlm52ValFreeMatrix(&fixture->q_a_norm);
	SparkGlm52ValFreeMatrix(&fixture->q_b);
	SparkGlm52ValFreeMatrix(&fixture->kv_a);
	SparkGlm52ValFreeMatrix(&fixture->kv_a_norm);
	SparkGlm52ValFreeMatrix(&fixture->kv_b_key);
	SparkGlm52ValFreeMatrix(&fixture->kv_b_value);
	SparkGlm52ValFreeMatrix(&fixture->attn_output);
	SparkGlm52ValFreeMatrix(&fixture->post_attn_norm);
	SparkGlm52ValFreeMatrix(&fixture->dense_gate_up);
	SparkGlm52ValFreeMatrix(&fixture->dense_down);
	cudaFree(fixture->hidden); cudaFree(fixture->residual); cudaFree(fixture->normed);
	cudaFree(fixture->q_compressed); cudaFree(fixture->q_bf16); cudaFree(fixture->query_rope);
	cudaFree(fixture->query_latent); cudaFree(fixture->kv_slot); cudaFree(fixture->attention_latent);
	cudaFree(fixture->attention_value); cudaFree(fixture->attention_out); cudaFree(fixture->gate_up);
	cudaFree(fixture->intermediate); cudaFree(fixture->scratch_small); cudaFree(fixture->float_scratch);
	cudaFree(fixture->token_ids); cudaFree(fixture->resident_slots); cudaFree(fixture->positions);
	cudaFree(fixture->context_lengths); cudaFree(fixture->dense_row_offset); cudaFree(fixture->dense_tile_prefix);
	cudaFree(fixture->page_table); cudaFree(fixture->route_scratch); cudaFree(fixture->head_maxloc);
	cudaFree(fixture->kv_access_error); cudaFree(fixture->kv_cache);
	if ( fixture->stream != 0 )
		cudaStreamDestroy(fixture->stream);
}

/* Bind the fixture into the production wave exactly as BindLayer does for a
 * whole-stack TP1 rank's local layer 0. */
static void SparkGlm52ValBuildWave(SparkGlm52ValFixture *fixture,uint32_t token,uint32_t position)
{
	SparkGlm52CudaWave *wave = &fixture->wave;
	SparkGlm52ExecutionSlot *slot = &fixture->slot;
	fixture->host_token = token;
	fixture->host_position = position;
	memset(slot,0,sizeof(*slot));
	slot->stream = fixture->stream;
	slot->hidden_bf16 = fixture->hidden;
	slot->residual_bf16 = fixture->residual;
	slot->normed_bf16 = fixture->normed;
	slot->q_compressed_bf16 = fixture->q_compressed;
	slot->q_bf16 = fixture->q_bf16;
	slot->query_latent_bf16 = fixture->query_latent;
	slot->query_rope_bf16 = fixture->query_rope;
	slot->index_query_bf16 = fixture->scratch_small;
	slot->index_key_bf16 = fixture->scratch_small;
	slot->index_head_weight_bf16 = fixture->scratch_small;
	slot->kv_slot_bf16 = fixture->kv_slot;
	slot->attention_latent_bf16 = fixture->attention_latent;
	slot->attention_value_bf16 = fixture->attention_value;
	slot->attention_out_bf16 = fixture->attention_out;
	slot->gate_up_bf16 = fixture->gate_up;
	slot->intermediate_bf16 = fixture->intermediate;
	slot->expert_out_bf16 = fixture->scratch_small;
	slot->shared_out_bf16 = fixture->scratch_small;
	slot->router_logits_f32 = fixture->float_scratch;
	slot->selection_scores_f32 = fixture->float_scratch;
	slot->selected_positions = fixture->route_scratch;
	slot->route_expert = fixture->route_scratch;
	slot->route_weight = fixture->float_scratch;
	slot->route_source_token = fixture->route_scratch;
	slot->route_packed_row = fixture->route_scratch;
	slot->group_row_offset = fixture->route_scratch;
	slot->group_tile_prefix_w1 = fixture->route_scratch;
	slot->group_tile_prefix_w2 = fixture->route_scratch;
	slot->head_candidate_score = fixture->float_scratch;
	slot->head_candidate_token = fixture->route_scratch;
	slot->output_token = fixture->route_scratch;
	slot->output_score = fixture->float_scratch;
	slot->head_maxloc_u64 = fixture->head_maxloc;
	slot->dense_row_offset = fixture->dense_row_offset;
	slot->dense_tile_prefix = fixture->dense_tile_prefix;
	slot->context_lengths = fixture->context_lengths;
	slot->token_ids = fixture->token_ids;
	slot->resident_slots = fixture->resident_slots;
	slot->positions = fixture->positions;
	slot->kv_access_error = fixture->kv_access_error;
	fixture->weights.attn_norm_bf16 = fixture->attn_norm.device;
	fixture->weights.q_a_bf16 = fixture->q_a.device;
	fixture->weights.q_a_norm_bf16 = fixture->q_a_norm.device;
	fixture->weights.q_b_bf16 = fixture->q_b.device;
	fixture->weights.kv_a_bf16 = fixture->kv_a.device;
	fixture->weights.kv_a_norm_bf16 = fixture->kv_a_norm.device;
	fixture->weights.kv_b_key_transposed_bf16 = fixture->kv_b_key.device;
	fixture->weights.kv_b_value_bf16 = fixture->kv_b_value.device;
	fixture->weights.index_q_bf16 = fixture->scratch_small;
	fixture->weights.index_k_bf16 = fixture->scratch_small;
	fixture->weights.index_head_bf16 = fixture->scratch_small;
	fixture->weights.index_norm_weight_bf16 = fixture->scratch_small;
	fixture->weights.index_norm_bias_bf16 = fixture->scratch_small;
	fixture->weights.dense_gate_up_bf16 = fixture->dense_gate_up.device;
	fixture->weights.dense_down_bf16 = fixture->dense_down.device;
	fixture->weights.router_bf16 = fixture->scratch_small;
	fixture->weights.router_correction_f32 = fixture->float_scratch;
	fixture->weights.expert_up_gate_payload = fixture->scratch_small;
	fixture->weights.expert_down_payload = fixture->scratch_small;
	fixture->weights.shared_gate_up_bf16 = fixture->scratch_small;
	fixture->weights.shared_down_bf16 = fixture->scratch_small;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = 0u;
	wave->first_layer_index = 0u;
	wave->layer_count = 1u;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = 1u;
	wave->maximum_context = position + 1u;
	wave->resident_sequence_capacity = SPARK_GLM52_VALIDATION_LANES;
	wave->max_sequence_positions = SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
	wave->pages_per_sequence = SPARK_GLM52_VALIDATION_PAGES_PER_SEQUENCE;
	wave->owns_embedding = 1u;
	wave->owns_final_head = 0u;
	wave->sideband_input = 0u;
	wave->sideband_output = 0u;
	wave->boundary_row_offset = 0u;
	wave->sideband_row_offset = 0u;
	wave->host_token_ids = &fixture->host_token;
	wave->host_resident_slots = &fixture->host_zero;
	wave->host_positions = &fixture->host_position;
	wave->embedding_bf16 = fixture->embedding.device;
	wave->layers = &fixture->weights;
	wave->slot = slot;
	wave->kv_cache = fixture->kv_cache;
	wave->kv_layer_stride_bytes = SPARK_GLM52_VKV_LAYER_BYTES;
	wave->index_cache = fixture->kv_cache;
	wave->index_layer_stride_bytes = 0u;
	wave->index_ordinal_by_local_layer = fixture->host_index_ordinals;
	wave->page_table = fixture->page_table;
	wave->multiprocessor_count = fixture->multiprocessors;
}

static int SparkGlm52ValRunWalk(SparkGlm52ValFixture *fixture)
{
	static const uint32_t tokens[SPARK_GLM52_VALIDATION_TOKENS] = {1u,3u,2u,6u};
	uint32_t step;
	int32_t status;
	for (step = 0u; step < SPARK_GLM52_VALIDATION_TOKENS; step++)
	{
		SparkGlm52ValBuildWave(fixture,tokens[step],step);
		status = SparkGlm52LaunchCudaWaveBegin(&fixture->wave);
		if ( status != 0 )
			return(SparkGlm52ValFail("wave_begin","status"));
		if ( SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 )
			return(SparkGlm52ValFail("layer_attention","status"));
		if ( SparkGlm52LaunchCudaLayerMlp(&fixture->wave,0u) != 0 )
			return(SparkGlm52ValFail("layer_mlp","status"));
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			return(SparkGlm52ValFail("walk","sync"));
	}
	return(0);
}

/* -- fp32 CPU oracle -------------------------------------------------------- */

typedef struct SparkGlm52ValOracle
{
	const SparkGlm52ValFixture *fixture;
	float hidden[SPARK_GLM52_VHIDDEN];
	float residual[SPARK_GLM52_VHIDDEN];
	float normed[SPARK_GLM52_VHIDDEN];
	float q_compressed[SPARK_GLM52_VQUERY_A];
	float q_row[SPARK_GLM52_VQ_ROWS];
	float q_rope[SPARK_GLM52_VHEADS * SPARK_GLM52_VROPE];
	float query_latent[SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT];
	float kv_slot[SPARK_GLM52_VKV_SLOT_ELEMENTS];
	float attention_latent[SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT];
	float attention_value[SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM];
	float attention_out[SPARK_GLM52_VHIDDEN];
	float gate_up[SPARK_GLM52_VGATE_UP_ROWS];
	float intermediate[SPARK_GLM52_VDENSE_INTER];
	/* cached slots, host mirror: [position][element] */
	float cache[SPARK_GLM52_VALIDATION_TOKENS][SPARK_GLM52_VKV_SLOT_ELEMENTS];
} SparkGlm52ValOracle;

/* C[r][o] = bf16(sum_k A[r][k] * W[o][k]); A/W already exact fp32 upcasts. */
static void SparkGlm52ValGemmRow(const float *activation,const float *weights,float *output,uint32_t input_dimension,uint32_t output_dimension)
{
	uint32_t column,index;
	float total;
	for (column = 0u; column < output_dimension; column++)
	{
		total = 0.0f;
		for (index = 0u; index < input_dimension; index++)
			total += activation[index] * weights[((uint64_t)column * input_dimension) + index];
		output[column] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(total));
	}
}

static void SparkGlm52ValRmsNorm(const float *input,const float *weight,float *output,uint32_t dimension,float epsilon)
{
	float total = 0.0f,scale;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		total += input[index] * input[index];
	scale = 1.0f / sqrtf((total / (float)dimension) + epsilon);
	for (index = 0u; index < dimension; index++)
		output[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(input[index] * scale * weight[index]));
}

static void SparkGlm52ValRotatePair(float *low,float *high,float angle)
{
	float cosine = cosf(angle),sine = sinf(angle);
	float a = *low,b = *high;
	*low = (a * cosine) - (b * sine);
	*high = (a * sine) + (b * cosine);
}

/* One token through the layer, mirroring the chunk chain op for op and
 * bf16 materialization boundary for boundary. */
static int SparkGlm52ValOracleToken(SparkGlm52ValOracle *oracle,const SparkGlm52ValFixture *fixture,uint32_t token,uint32_t position)
{
	const SparkGlm52ValFixture *fx = fixture;
	float sum,value,pair_low,pair_high,angle,score;
	float scores[SPARK_GLM52_VALIDATION_TOKENS];
	float maximum,running_sum;
	uint32_t index,head,element,step,context;
	const float *slot;
	/* WaveBegin: embedding row into hidden, zero residual. */
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		oracle->hidden[index] = fx->embedding.host[((uint64_t)token * SPARK_GLM52_VHIDDEN) + index];
		oracle->residual[index] = 0.0f;
	}
	/* Attention chunk. attn-norm: residual add fused, residual stored pre-scale. */
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		sum = oracle->hidden[index] + oracle->residual[index];
		oracle->residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		oracle->hidden[index] = sum;
	}
	SparkGlm52ValRmsNorm(oracle->hidden,fx->attn_norm.host,oracle->normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValGemmRow(oracle->normed,fx->q_a.host,oracle->q_compressed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VQUERY_A);
	SparkGlm52ValRmsNorm(oracle->q_compressed,fx->q_a_norm.host,oracle->q_compressed,SPARK_GLM52_VQUERY_A,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValGemmRow(oracle->q_compressed,fx->q_b.host,oracle->q_row,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VQ_ROWS);
	SparkGlm52ValGemmRow(oracle->normed,fx->kv_a.host,oracle->kv_slot,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VKV_SLOT_ELEMENTS);
	/* kv_a-norm covers only the latent half of the 576-wide row. */
	SparkGlm52ValRmsNorm(oracle->kv_slot,fx->kv_a_norm.host,oracle->kv_slot,SPARK_GLM52_VLATENT,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	/* Query rope: extract [192..256) of every packed head, rotate, store. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
		{
			pair_low = oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index)];
			pair_high = oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index + 1u)];
			angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
			SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
			oracle->q_rope[(head * SPARK_GLM52_VROPE) + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
			oracle->q_rope[(head * SPARK_GLM52_VROPE) + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
		}
	/* Key-side absorption: per-head W[512][192] over the RAW nope slice. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
		{
			sum = 0.0f;
			for (index = 0u; index < SPARK_GLM52_VQK_NOPE; index++)
				sum += oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + index] *
					fx->kv_b_key.host[((uint64_t)head * SPARK_GLM52_VLATENT + element) * SPARK_GLM52_VQK_NOPE + index];
			oracle->query_latent[(head * SPARK_GLM52_VLATENT) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		}
	/* Cache store: rope the latent row's tail IN PLACE, then copy. */
	for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
	{
		pair_low = oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index];
		pair_high = oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u];
		angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
		SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
		oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
		oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
	}
	memcpy(oracle->cache[position],oracle->kv_slot,sizeof(oracle->cache[position]));
	/* Latent decode attention: dense positions 0..position, causal skip none. */
	context = position + 1u;
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
	{
		maximum = -3.0e38f;
		running_sum = 0.0f;
		for (step = 0u; step < context; step++)
		{
			slot = oracle->cache[step];
			score = 0.0f;
			for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
				score += oracle->query_latent[(head * SPARK_GLM52_VLATENT) + element] * slot[element];
			for (element = 0u; element < SPARK_GLM52_VROPE; element++)
				score += oracle->q_rope[(head * SPARK_GLM52_VROPE) + element] * slot[SPARK_GLM52_VLATENT + element];
			scores[step] = score * SPARK_GLM52_MODEL_QK_SCALE;
			if ( scores[step] > maximum )
				maximum = scores[step];
		}
		for (step = 0u; step < context; step++)
			running_sum += expf(scores[step] - maximum);
		if ( running_sum < 1.0e-20f )
			running_sum = 1.0e-20f;
		for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
		{
			value = 0.0f;
			for (step = 0u; step < context; step++)
				value += expf(scores[step] - maximum) * oracle->cache[step][element];
			oracle->attention_latent[(head * SPARK_GLM52_VLATENT) + element] =
				SparkGlm52ValFromBf16(SparkGlm52ValBf16(value / running_sum));
		}
	}
	/* Value up-projection: per-head W[256][512]. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (element = 0u; element < SPARK_GLM52_VVALUE_DIM; element++)
		{
			sum = 0.0f;
			for (index = 0u; index < SPARK_GLM52_VLATENT; index++)
				sum += oracle->attention_latent[(head * SPARK_GLM52_VLATENT) + index] *
					fx->kv_b_value.host[((uint64_t)head * SPARK_GLM52_VVALUE_DIM + element) * SPARK_GLM52_VLATENT + index];
			oracle->attention_value[(head * SPARK_GLM52_VVALUE_DIM) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		}
	SparkGlm52ValGemmRow(oracle->attention_value,fx->attn_output.host,oracle->attention_out,SPARK_GLM52_VATTN_COLS,SPARK_GLM52_VHIDDEN);
	/* MLP chunk: mlp-norm, fused [up|gate] stack, silu-mul, down. */
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		sum = oracle->attention_out[index] + oracle->residual[index];
		oracle->residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		oracle->hidden[index] = sum;
	}
	SparkGlm52ValRmsNorm(oracle->hidden,fx->post_attn_norm.host,oracle->normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValGemmRow(oracle->normed,fx->dense_gate_up.host,oracle->gate_up,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VGATE_UP_ROWS);
	/* gate_first=false: up is the FIRST half, gate the SECOND. */
	for (index = 0u; index < SPARK_GLM52_VDENSE_INTER; index++)
	{
		float gate = oracle->gate_up[SPARK_GLM52_VDENSE_INTER + index];
		float up = oracle->gate_up[index];
		oracle->intermediate[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16((gate / (1.0f + expf(-gate))) * up));
	}
	SparkGlm52ValGemmRow(oracle->intermediate,fx->dense_down.host,oracle->hidden,SPARK_GLM52_VDENSE_INTER,SPARK_GLM52_VHIDDEN);
	return(0);
}

int main(int argc,char **argv)
{
	SparkGlm52ValFixture fixture;
	SparkGlm52ValOracle oracle;
	SparkGlm52ValMetrics metrics;
	uint16_t *device_hidden,*device_residual;
	float run_hidden[SPARK_GLM52_VHIDDEN];
	float run_residual[SPARK_GLM52_VHIDDEN];
	float actual[SPARK_GLM52_VHIDDEN];
	float reference[SPARK_GLM52_VHIDDEN];
	uint32_t index,error_words[SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS];
	int result;
	if ( argc != 2 || strlen(argv[1]) != 64u )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	memset(&fixture,0,sizeof(fixture));
	if ( SparkGlm52ValFixtureSetup(&fixture) != 0 )
		return(1);
	device_hidden = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	device_residual = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	if ( device_hidden == 0 || device_residual == 0 )
		return(SparkGlm52ValFail("compare","host_alloc"));
	/* Walk one: the measured run. */
	if ( SparkGlm52ValRunWalk(&fixture) != 0 )
		return(1);
	if ( cudaMemcpy(error_words,fixture.kv_access_error,sizeof(error_words),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SparkGlm52ValFail("kv_access","error_readback"));
	for (index = 0u; index < SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS; index++)
		if ( error_words[index] != 0u )
			return(SparkGlm52ValFail("kv_access","error_word_set"));
	if ( cudaMemcpy(device_hidden,fixture.hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(device_residual,fixture.residual,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SparkGlm52ValFail("compare","hidden_readback"));
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		run_hidden[index] = SparkGlm52ValFromBf16(device_hidden[index]);
		run_residual[index] = SparkGlm52ValFromBf16(device_residual[index]);
	}
	/* The oracle walks the same four tokens on the host. */
	memset(&oracle,0,sizeof(oracle));
	{
		static const uint32_t tokens[SPARK_GLM52_VALIDATION_TOKENS] = {1u,3u,2u,6u};
		uint32_t step;
		for (step = 0u; step < SPARK_GLM52_VALIDATION_TOKENS; step++)
			SparkGlm52ValOracleToken(&oracle,&fixture,tokens[step],step);
	}
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		actual[index] = run_hidden[index];
		reference[index] = oracle.hidden[index];
	}
	SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
	result = SparkGlm52ValReport("layer_forward_hidden",&metrics,2e-2,0.999);
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		actual[index] = run_residual[index];
		reference[index] = oracle.residual[index];
	}
	SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
	if ( result == 0 )
		result = SparkGlm52ValReport("layer_forward_residual",&metrics,2e-2,0.999);
	/* Determinism: the identical walk on the identical fixture must
	 * reproduce the committed streams bit for bit - the property every
	 * replay/restore argument downstream borrows from the chain. */
	if ( result == 0 )
	{
		int mismatch = 0;
		if ( SparkGlm52ValRunWalk(&fixture) != 0 )
			return(1);
		if ( cudaMemcpy(device_hidden,fixture.hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(device_residual,fixture.residual,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
			return(SparkGlm52ValFail("determinism","rerun_readback"));
		for (index = 0u; index < SPARK_GLM52_VHIDDEN && mismatch == 0; index++)
		{
			if ( device_hidden[index] != SparkGlm52ValBf16(run_hidden[index]) )
				mismatch = 1;
			if ( device_residual[index] != SparkGlm52ValBf16(run_residual[index]) )
				mismatch = 1;
		}
		printf("glm52_validation check=determinism elements=%u bit_exact=%d\n",
			(unsigned)(2u * SPARK_GLM52_VHIDDEN),mismatch == 0 ? 1 : 0);
		if ( mismatch != 0 )
			result = SparkGlm52ValFail("determinism","repeat_walk_mismatch");
	}
	free(device_hidden);
	free(device_residual);
	SparkGlm52ValFixtureDestroy(&fixture);
	if ( result == 0 )
		printf("glm52_validation PASS\n");
	return(result);
}
