#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"
#include "spark_glm5_next_tap_ring.h"

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the compiled archive's expert codec"
#endif

#define SPARK_GLM5_NEXT_TAP_CHECK_EXPERT_CODEC_ID 5u
#if GLM5_NEXT_EXPERT_WEIGHT_CODEC != SPARK_GLM5_NEXT_TAP_CHECK_EXPERT_CODEC_ID
#error "the tap ring harness synthesizes fp8 expert payloads; build the archive with EXPERT_CODEC=fp8"
#endif

#define SPARK_GLM5_NEXT_TAP_CHECK_LAYERS 43u
#define SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS 33u
#define SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS 10u
#define SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS 6u
#define SPARK_GLM5_NEXT_TAP_CHECK_DECODE_STEPS 12u
#define SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS \
	(SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS + SPARK_GLM5_NEXT_TAP_CHECK_DECODE_STEPS)
#define SPARK_GLM5_NEXT_TAP_CHECK_PAGES 2u
#define SPARK_GLM5_NEXT_TAP_CHECK_PAGE_TOKENS 64u
#define SPARK_GLM5_NEXT_TAP_CHECK_MAX_POSITIONS \
	(SPARK_GLM5_NEXT_TAP_CHECK_PAGES * SPARK_GLM5_NEXT_TAP_CHECK_PAGE_TOKENS)
#define SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY 8u
#define SPARK_GLM5_NEXT_TAP_CHECK_WINDOW 8u
#define SPARK_GLM5_NEXT_TAP_CHECK_LANES 2u
#define SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILE 1024u
#define SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILES \
	((SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT + SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILE - 1u) / \
	 SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILE)
#define SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS (1u << 22)
#define SPARK_GLM5_NEXT_TAP_CHECK_KV_ACCESS_WORDS 6u

#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ 0.02f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_ATTN_OUT 0.01f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_DENSE 0.01f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_KDA_OUT 0.005f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_CONV 0.05f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_ROUTER 0.002f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_SHARED 0.005f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_EMBEDDING 0.02f
#define SPARK_GLM5_NEXT_TAP_CHECK_SCALE_HEAD 0.01f
#define SPARK_GLM5_NEXT_TAP_CHECK_EXPERT_SCALE 6.103515625e-05f
#define SPARK_GLM5_NEXT_TAP_CHECK_KDA_DECAY_BIAS -0.25f
#define SPARK_GLM5_NEXT_TAP_CHECK_KDA_HEAD_LOG_SCALE 0.1f
#define SPARK_GLM5_NEXT_TAP_CHECK_ROUTER_CORRECTION 4.0f

#define SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_SLOT_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION * \
	 SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL * \
	 SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES)
#define SPARK_GLM5_NEXT_TAP_CHECK_STATE_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS * \
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER)
#define SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS * 3u * \
	 SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_SLOT_BYTES)
#define SPARK_GLM5_NEXT_TAP_CHECK_KV_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_PAGES * SPARK_GLM5_NEXT_TAP_CHECK_PAGE_TOKENS * \
	 SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES)
#define SPARK_GLM5_NEXT_TAP_CHECK_INDEX_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_PAGES * SPARK_GLM5_NEXT_TAP_CHECK_PAGE_TOKENS * \
	 SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u)
#define SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS * sizeof(uint16_t))

static_assert(SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS <= SPARK_GLM5_NEXT_TAP_CHECK_MAX_POSITIONS,
	"tap check walk must fit the page table");
static_assert(SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS > 2u * SPARK_GLM5_NEXT_TAP_CHECK_WINDOW,
	"tap check must wrap the ring window at least twice");
static_assert(SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS <= SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY,
	"prefill wave must fit the scratch row capacity");
static_assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT == 5u,
	"the drafter contract carries five tap layers");

typedef struct SparkGlm5NextTapCheckWeights
{
	uint16_t *norm_hidden_bf16;
	uint16_t *norm_query_a_bf16;
	uint16_t *norm_kv_a_bf16;
	uint16_t *norm_index_bf16;
	uint16_t *kda_qkv_beta_bf16;
	uint16_t *kda_decay_gate_down_bf16;
	uint16_t *kda_decay_up_bf16;
	uint16_t *kda_gate_up_bf16;
	uint16_t *kda_q_conv_bf16;
	uint16_t *kda_k_conv_bf16;
	uint16_t *kda_v_conv_bf16;
	uint16_t *kda_out_bf16;
	float *kda_out_norm_f32;
	float *kda_decay_bias_f32;
	float *kda_head_log_scale_f32;
	uint16_t *q_a_bf16;
	uint16_t *q_b_bf16;
	uint16_t *kv_a_bf16;
	uint16_t *kv_b_key_bf16;
	uint16_t *kv_b_value_bf16;
	uint16_t *attn_output_bf16;
	uint16_t *index_q_bf16;
	uint16_t *index_k_bf16;
	uint16_t *index_head_bf16;
	uint16_t *index_compress_gate_bf16;
	float *index_compress_ape_f32;
	uint16_t *dense_gate_up_bf16;
	uint16_t *dense_down_bf16;
	uint16_t *router_bf16;
	float *router_correction_f32;
	uint16_t *shared_gate_up_bf16;
	uint16_t *shared_down_bf16;
	uint8_t *expert_up_gate_payload;
	uint8_t *expert_up_gate_scale;
	uint8_t *expert_down_payload;
	uint8_t *expert_down_scale;
	float *hc_attn_fn_f32;
	float *hc_attn_base_f32;
	float *hc_attn_scale_f32;
	float *hc_ffn_fn_f32;
	float *hc_ffn_base_f32;
	float *hc_ffn_scale_f32;
	uint16_t *embedding_bf16;
	uint16_t *lm_head_bf16;
} SparkGlm5NextTapCheckWeights;

typedef struct SparkGlm5NextTapCheckFixture
{
	cudaStream_t stream;
	uint32_t multiprocessor_count;
	uint64_t random_state;
	SparkGlm5NextTapCheckWeights weights;
	SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_TAP_CHECK_LAYERS];
	SparkGlm5NextExecutionSlot slot;
	SparkGlm5NextCudaWave wave;
	SparkGlm5NextTapRing *ring;
	void *begin_event;
	void *op_event;
	void *done_event;
	uint32_t capture_pending;
	double capture_mean_ms;
	double capture_d2h_ms;
	double capture_max_ms;
	uint32_t capture_count;
	uint8_t *kda_state_pools;
	uint8_t *kda_window_pools;
	uint8_t *kv_cache;
	uint8_t *index_cache;
	uint32_t *page_table;
	uint32_t *kda_state_index_device;
	uint32_t index_ordinal_by_local_layer[SPARK_GLM5_NEXT_TAP_CHECK_LAYERS];
	uint32_t kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_TAP_CHECK_LAYERS];
	uint16_t *reference_taps;
	uint32_t host_token_ids[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY];
	uint32_t host_positions[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY];
	uint32_t host_resident_slots[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY];
	uint32_t host_run_begin[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY + 1u];
	uint32_t host_run_state_index[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY];
	uint32_t host_output[SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY];
} SparkGlm5NextTapCheckFixture;

static const uint32_t SparkGlm5NextTapCheckLayers[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT] =
	SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYERS;

static int SparkGlm5NextTapCheckFail(const char *check,const char *detail)
{
	printf("FAIL %s: %s\n",check,detail);
	return(1);
}

static int SparkGlm5NextTapCheckCuda(cudaError_t error,const char *check,const char *detail)
{
	if ( error == cudaSuccess )
		return(0);
	printf("FAIL %s: %s cuda=%s\n",check,detail,cudaGetErrorString(error));
	return(1);
}

static uint32_t SparkGlm5NextTapCheckNext(uint64_t *state)
{
	uint64_t value;
	value = *state;
	value ^= value << 13u;
	value ^= value >> 7u;
	value ^= value << 17u;
	*state = value;
	return((uint32_t)(value >> 32u));
}

static uint16_t SparkGlm5NextTapCheckBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)(bits >> 16u));
}

static uint16_t SparkGlm5NextTapCheckRoundBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static void *SparkGlm5NextTapCheckAlloc(uint64_t bytes,const char *name)
{
	void *pointer;
	pointer = 0;
	if ( bytes == 0u || cudaMalloc(&pointer,(size_t)bytes) != cudaSuccess )
	{
		printf("FAIL alloc: %s (%llu bytes)\n",name,(unsigned long long)bytes);
		return(0);
	}
	return(pointer);
}

static int SparkGlm5NextTapCheckUpload(
	const void *device,
	const void *host,
	uint64_t bytes,
	const char *name)
{
	return(SparkGlm5NextTapCheckCuda(cudaMemcpy((void *)device,host,(size_t)bytes,cudaMemcpyHostToDevice),"upload",name));
}

static int SparkGlm5NextTapCheckFillBf16(
	SparkGlm5NextTapCheckFixture *fixture,
	uint16_t **out,
	uint64_t elements,
	float scale,
	uint32_t constant_one,
	const char *name)
{
	uint16_t *chunk;
	uint16_t *device;
	uint64_t offset;
	device = (uint16_t *)SparkGlm5NextTapCheckAlloc(elements * sizeof(uint16_t),name);
	if ( device == 0 )
		return(1);
	chunk = (uint16_t *)malloc((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS * sizeof(uint16_t));
	if ( chunk == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host",name));
	for ( offset = 0u; offset < elements; )
	{
		uint64_t count = elements - offset;
		uint64_t index;
		if ( count > SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index < count; index++ )
		{
			float value;
			if ( constant_one != 0u )
				value = 1.0f;
			else
				value = (((float)(int32_t)(SparkGlm5NextTapCheckNext(&fixture->random_state) & 0xffffu) - 32768.0f) / 32768.0f) * scale;
			chunk[index] = SparkGlm5NextTapCheckBf16(value);
		}
		if ( SparkGlm5NextTapCheckUpload(device + offset,chunk,count * sizeof(uint16_t),name) != 0 )
		{
			free(chunk);
			return(1);
		}
		offset += count;
	}
	free(chunk);
	*out = device;
	return(0);
}

static int SparkGlm5NextTapCheckFillF32(
	float **out,
	uint64_t count,
	float value,
	const char *name)
{
	float *host;
	float *device;
	uint64_t index;
	device = (float *)SparkGlm5NextTapCheckAlloc(count * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((size_t)(count * sizeof(float)));
	if ( host == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host",name));
	for ( index = 0u; index < count; index++ )
		host[index] = value;
	if ( SparkGlm5NextTapCheckUpload(device,host,count * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextTapCheckFillF32Grid(
	float **out,
	uint64_t count,
	uint32_t modulus,
	float scale,
	const char *name)
{
	float *host;
	float *device;
	uint64_t index;
	device = (float *)SparkGlm5NextTapCheckAlloc(count * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((size_t)(count * sizeof(float)));
	if ( host == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host",name));
	for ( index = 0u; index < count; index++ )
		host[index] = ((float)(index % modulus) - (float)(modulus / 2u)) * scale;
	if ( SparkGlm5NextTapCheckUpload(device,host,count * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextTapCheckFillExperts(
	SparkGlm5NextTapCheckFixture *fixture,
	uint8_t **payload_out,
	uint8_t **scale_out,
	uint32_t rows,
	uint32_t columns,
	const char *name)
{
	uint64_t payload_bytes;
	uint64_t scale_bytes;
	uint64_t offset;
	uint8_t *payload;
	uint8_t *scale_device;
	uint8_t *chunk;
	payload_bytes = SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_FP8_E4M3,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * rows,columns);
	scale_bytes = SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_FP8_E4M3,
		SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,rows,columns);
	if ( payload_bytes == 0u || scale_bytes == 0u || (scale_bytes % sizeof(float)) != 0u )
		return(SparkGlm5NextTapCheckFail("expert_shape",name));
	payload = (uint8_t *)SparkGlm5NextTapCheckAlloc(payload_bytes,name);
	scale_device = (uint8_t *)SparkGlm5NextTapCheckAlloc(scale_bytes,name);
	if ( payload == 0 || scale_device == 0 )
		return(1);
	chunk = (uint8_t *)malloc((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS);
	if ( chunk == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host",name));
	for ( offset = 0u; offset < payload_bytes; )
	{
		uint64_t count = payload_bytes - offset;
		uint64_t index;
		if ( count > SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index < count; index++ )
		{
			uint8_t raw = (uint8_t)SparkGlm5NextTapCheckNext(&fixture->random_state);
			if ( (raw & 0x7fu) == 0x7fu )
				raw &= 0x7eu;
			chunk[index] = raw;
		}
		if ( SparkGlm5NextTapCheckUpload(payload + offset,chunk,count,name) != 0 )
		{
			free(chunk);
			return(1);
		}
		offset += count;
	}
	for ( offset = 0u; offset < scale_bytes; )
	{
		uint64_t count = scale_bytes - offset;
		uint64_t index;
		if ( count > SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_TAP_CHECK_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index + sizeof(float) <= count; index += sizeof(float) )
		{
			float value = SPARK_GLM5_NEXT_TAP_CHECK_EXPERT_SCALE;
			memcpy(chunk + index,&value,sizeof(value));
		}
		if ( SparkGlm5NextTapCheckUpload(scale_device + offset,chunk,count,name) != 0 )
		{
			free(chunk);
			return(1);
		}
		offset += count;
	}
	free(chunk);
	*payload_out = payload;
	*scale_out = scale_device;
	return(0);
}

static int SparkGlm5NextTapCheckFillRouterCorrection(float **out,const char *name)
{
	float *host;
	float *device;
	uint32_t index;
	device = (float *)SparkGlm5NextTapCheckAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float));
	if ( host == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host",name));
	for ( index = 0u; index < SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT; index++ )
		host[index] = index < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K ?
			SPARK_GLM5_NEXT_TAP_CHECK_ROUTER_CORRECTION : -SPARK_GLM5_NEXT_TAP_CHECK_ROUTER_CORRECTION;
	if ( SparkGlm5NextTapCheckUpload(device,host,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextTapCheckBuildWeights(SparkGlm5NextTapCheckFixture *fixture)
{
	SparkGlm5NextTapCheckWeights *weights;
	uint64_t hidden;
	uint32_t kda_dim;
	weights = &fixture->weights;
	hidden = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	kda_dim = SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION;
	if ( SparkGlm5NextTapCheckFillBf16(fixture,&weights->norm_hidden_bf16,hidden,0.0f,1u,"norm_hidden") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->norm_query_a_bf16,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,0.0f,1u,"norm_query_a") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->norm_kv_a_bf16,SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION,0.0f,1u,"norm_kv_a") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->norm_index_bf16,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION,0.0f,1u,"norm_index") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_qkv_beta_bf16,(3u * kda_dim + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT) * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_DENSE,0u,"kda_qkv_beta") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_decay_gate_down_bf16,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_DENSE,0u,"kda_decay_gate_down") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_decay_up_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"kda_decay_up") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_gate_up_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"kda_gate_up") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_q_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_CONV,0u,"kda_q_conv") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_k_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_CONV,0u,"kda_k_conv") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_v_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_CONV,0u,"kda_v_conv") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kda_out_bf16,hidden * kda_dim,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_KDA_OUT,0u,"kda_out") != 0 ||
		SparkGlm5NextTapCheckFillF32(&weights->kda_out_norm_f32,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION,1.0f,"kda_out_norm") != 0 ||
		SparkGlm5NextTapCheckFillF32(&weights->kda_decay_bias_f32,kda_dim,SPARK_GLM5_NEXT_TAP_CHECK_KDA_DECAY_BIAS,"kda_decay_bias") != 0 ||
		SparkGlm5NextTapCheckFillF32(&weights->kda_head_log_scale_f32,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,SPARK_GLM5_NEXT_TAP_CHECK_KDA_HEAD_LOG_SCALE,"kda_head_log_scale") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->q_a_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"q_a") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->q_b_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"q_b") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kv_a_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"kv_a") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kv_b_key_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"kv_b_key") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->kv_b_value_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"kv_b_value") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->attn_output_bf16,hidden * SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_ATTN_OUT,0u,"attn_output") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->index_q_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"index_q") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->index_k_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"index_k") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->index_head_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_ATTN_OUT,0u,"index_head") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->index_compress_gate_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_PROJ,0u,"index_compress_gate") != 0 ||
		SparkGlm5NextTapCheckFillF32Grid(&weights->index_compress_ape_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION,5u,0.25f,"index_compress_ape") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->dense_gate_up_bf16,2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_DENSE,0u,"dense_gate_up") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->dense_down_bf16,hidden * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_DENSE,0u,"dense_down") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->router_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_ROUTER,0u,"router") != 0 ||
		SparkGlm5NextTapCheckFillRouterCorrection(&weights->router_correction_f32,"router_correction") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->shared_gate_up_bf16,2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_SHARED,0u,"shared_gate_up") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->shared_down_bf16,hidden * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_SHARED,0u,"shared_down") != 0 ||
		SparkGlm5NextTapCheckFillExperts(fixture,&weights->expert_up_gate_payload,&weights->expert_up_gate_scale,2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,"expert_up_gate") != 0 ||
		SparkGlm5NextTapCheckFillExperts(fixture,&weights->expert_down_payload,&weights->expert_down_scale,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,"expert_down") != 0 ||
		SparkGlm5NextTapCheckFillF32Grid(&weights->hc_attn_fn_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden,7u,0.01f,"hc_attn_fn") != 0 ||
		SparkGlm5NextTapCheckFillF32Grid(&weights->hc_attn_base_f32,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,5u,0.25f,"hc_attn_base") != 0 ||
		SparkGlm5NextTapCheckFillF32(&weights->hc_attn_scale_f32,SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT,0.5f,"hc_attn_scale") != 0 ||
		SparkGlm5NextTapCheckFillF32Grid(&weights->hc_ffn_fn_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden,9u,0.01f,"hc_ffn_fn") != 0 ||
		SparkGlm5NextTapCheckFillF32Grid(&weights->hc_ffn_base_f32,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,3u,0.25f,"hc_ffn_base") != 0 ||
		SparkGlm5NextTapCheckFillF32(&weights->hc_ffn_scale_f32,SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT,0.5f,"hc_ffn_scale") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->embedding_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_EMBEDDING,0u,"embedding") != 0 ||
		SparkGlm5NextTapCheckFillBf16(fixture,&weights->lm_head_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT * hidden,SPARK_GLM5_NEXT_TAP_CHECK_SCALE_HEAD,0u,"lm_head") != 0 )
		return(1);
	return(0);
}

#define SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(field,rows,columns) \
	slot->field = (uint16_t *)SparkGlm5NextTapCheckAlloc( \
		(uint64_t)(rows) * (columns) * sizeof(uint16_t),#field); \
	if ( slot->field == 0 ) \
		return(1);
#define SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(field,rows,columns) \
	slot->field = (float *)SparkGlm5NextTapCheckAlloc( \
		(uint64_t)(rows) * (columns) * sizeof(float),#field); \
	if ( slot->field == 0 ) \
		return(1);
#define SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(field,count) \
	slot->field = (uint32_t *)SparkGlm5NextTapCheckAlloc( \
		(uint64_t)(count) * sizeof(uint32_t),#field); \
	if ( slot->field == 0 ) \
		return(1);

static int SparkGlm5NextTapCheckBuildScratch(SparkGlm5NextTapCheckFixture *fixture)
{
	SparkGlm5NextExecutionSlot *slot;
	uint64_t rows;
	uint64_t hidden;
	uint64_t kda_dim;
	slot = &fixture->slot;
	rows = SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY;
	hidden = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	kda_dim = SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION;
	memset(slot,0,sizeof(*slot));
	slot->stream = fixture->stream;
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(hidden_bf16,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(residual_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(normed_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(tap_stage_bf16,rows,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(q_compressed_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(q_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(query_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION)
	slot->query_rope_bf16 = 0;
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(index_query_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(index_key_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(index_gate_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(index_packed_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(selected_pools,rows * SPARK_GLM5_NEXT_MODEL_INDEX_POOL_SELECT_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(index_head_weight_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kv_slot_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(attention_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(attention_value_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(attention_out_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(gate_up_bf16,rows,2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(intermediate_bf16,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(expert_out_bf16,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(shared_out_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(fused_qkvb_bf16,rows,3u * kda_dim + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(fused_decay_gate_bf16,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_decay_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_gate_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_beta_logit,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_gate_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_decay_logit_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(kda_output_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(kda_retention,rows,kda_dim)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(kda_write_gate,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(hc_mixes_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(hc_pre_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(hc_post_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(hc_comb_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(hc_collapsed_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(hc_snapshot_bf16,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_BF16(hc_mean_bf16,rows,hidden)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(router_logits_f32,rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(selection_scores_f32,rows,SPARK_GLM5_NEXT_TAP_CHECK_MAX_POSITIONS / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(selected_positions,rows * SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(route_expert,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(route_weight,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(route_source_token,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(route_packed_row,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(group_row_offset,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(group_tile_prefix_w1,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(group_tile_prefix_w2,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(token_ids,rows)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(resident_slots,rows)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(positions,rows)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(context_lengths,1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(dense_row_offset,4u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(dense_tile_prefix,4u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(run_begin,rows + 1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(run_state_index,rows)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(output_token,rows)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(output_score,rows,1u)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_F32(head_candidate_score,rows,SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILES)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(head_candidate_token,rows * SPARK_GLM5_NEXT_TAP_CHECK_HEAD_TILES)
	SPARK_GLM5_NEXT_TAP_CHECK_SCRATCH_U32(kv_access_error,SPARK_GLM5_NEXT_TAP_CHECK_KV_ACCESS_WORDS)
	slot->head_maxloc_u64 = (uint64_t *)SparkGlm5NextTapCheckAlloc(rows * sizeof(uint64_t),"head_maxloc_u64");
	if ( slot->head_maxloc_u64 == 0 )
		return(1);
	slot->attention_split_partials_f32 = (float *)SparkGlm5NextTapCheckAlloc(
		SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT),
		"attention_split_partials_f32");
	if ( slot->attention_split_partials_f32 == 0 )
		return(1);
	return(0);
}

static void SparkGlm5NextTapCheckBindLayers(SparkGlm5NextTapCheckFixture *fixture)
{
	SparkGlm5NextTapCheckWeights *weights;
	uint32_t local,kda,dsa;
	weights = &fixture->weights;
	kda = 0u;
	dsa = 0u;
	for ( local = 0u; local < SPARK_GLM5_NEXT_TAP_CHECK_LAYERS; local++ )
	{
		SparkGlm5NextLayerWeights *layer;
		layer = &fixture->layers[local];
		memset(layer,0,sizeof(*layer));
		layer->attn_norm_bf16 = weights->norm_hidden_bf16;
		layer->post_attn_norm_bf16 = weights->norm_hidden_bf16;
		layer->hc_attn_fn_f32 = weights->hc_attn_fn_f32;
		layer->hc_attn_base_f32 = weights->hc_attn_base_f32;
		layer->hc_attn_scale_f32 = weights->hc_attn_scale_f32;
		layer->hc_ffn_fn_f32 = weights->hc_ffn_fn_f32;
		layer->hc_ffn_base_f32 = weights->hc_ffn_base_f32;
		layer->hc_ffn_scale_f32 = weights->hc_ffn_scale_f32;
		layer->dense_gate_up_bf16 = weights->dense_gate_up_bf16;
		layer->dense_down_bf16 = weights->dense_down_bf16;
		layer->router_bf16 = weights->router_bf16;
		layer->router_correction_f32 = weights->router_correction_f32;
		layer->expert_up_gate_payload = weights->expert_up_gate_payload;
		layer->expert_up_gate_scale = weights->expert_up_gate_scale;
		layer->expert_down_payload = weights->expert_down_payload;
		layer->expert_down_scale = weights->expert_down_scale;
		layer->shared_gate_up_bf16 = weights->shared_gate_up_bf16;
		layer->shared_down_bf16 = weights->shared_down_bf16;
		if ( SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(local) != 0u )
		{
			layer->kda_qkv_beta_bf16 = weights->kda_qkv_beta_bf16;
			layer->kda_decay_gate_down_bf16 = weights->kda_decay_gate_down_bf16;
			layer->kda_decay_up_bf16 = weights->kda_decay_up_bf16;
			layer->kda_gate_up_bf16 = weights->kda_gate_up_bf16;
			layer->kda_q_conv_bf16 = weights->kda_q_conv_bf16;
			layer->kda_k_conv_bf16 = weights->kda_k_conv_bf16;
			layer->kda_v_conv_bf16 = weights->kda_v_conv_bf16;
			layer->kda_decay_bias_f32 = weights->kda_decay_bias_f32;
			layer->kda_head_log_scale_f32 = weights->kda_head_log_scale_f32;
			layer->kda_out_norm_bf16 = weights->kda_out_norm_f32;
			layer->kda_out_bf16 = weights->kda_out_bf16;
			fixture->kda_ordinal_by_local_layer[local] = kda++;
			fixture->index_ordinal_by_local_layer[local] = UINT32_MAX;
		}
		else
		{
			layer->q_a_bf16 = weights->q_a_bf16;
			layer->q_a_norm_bf16 = weights->norm_query_a_bf16;
			layer->q_b_bf16 = weights->q_b_bf16;
			layer->kv_a_bf16 = weights->kv_a_bf16;
			layer->kv_a_norm_bf16 = weights->norm_kv_a_bf16;
			layer->kv_b_key_transposed_bf16 = weights->kv_b_key_bf16;
			layer->kv_b_value_bf16 = weights->kv_b_value_bf16;
			layer->attn_output_bf16 = weights->attn_output_bf16;
			layer->index_q_bf16 = weights->index_q_bf16;
			layer->index_k_bf16 = weights->index_k_bf16;
			layer->index_head_bf16 = weights->index_head_bf16;
			layer->index_norm_weight_bf16 = weights->norm_index_bf16;
			layer->index_norm_bias_bf16 = weights->norm_index_bf16;
			layer->index_compress_ape_f32 = weights->index_compress_ape_f32;
			layer->index_compress_gate_bf16 = weights->index_compress_gate_bf16;
			fixture->kda_ordinal_by_local_layer[local] = UINT32_MAX;
			fixture->index_ordinal_by_local_layer[local] = dsa++;
		}
	}
	if ( kda != SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS || dsa != SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS )
	{
		printf("FAIL bind: ordinal counts kda %u dsa %u\n",kda,dsa);
		exit(1);
	}
}

static void SparkGlm5NextTapCheckBuildWave(
	SparkGlm5NextTapCheckFixture *fixture,
	const uint32_t *tokens,
	uint32_t first_position,
	uint32_t row_count)
{
	SparkGlm5NextCudaWave *wave;
	uint32_t row;
	wave = &fixture->wave;
	for ( row = 0u; row < row_count; row++ )
	{
		fixture->host_token_ids[row] = tokens[row];
		fixture->host_positions[row] = first_position + row;
		fixture->host_resident_slots[row] = 0u;
	}
	fixture->host_run_begin[0] = 0u;
	fixture->host_run_begin[1] = row_count;
	fixture->host_run_state_index[0] = 0u;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = 0u;
	wave->first_layer_index = 0u;
	wave->layer_count = SPARK_GLM5_NEXT_TAP_CHECK_LAYERS;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = row_count;
	wave->maximum_context = first_position + row_count;
	wave->resident_sequence_capacity = 1u;
	wave->max_sequence_positions = SPARK_GLM5_NEXT_TAP_CHECK_MAX_POSITIONS;
	wave->execution_row_capacity = SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY;
	wave->pages_per_sequence = SPARK_GLM5_NEXT_TAP_CHECK_PAGES;
	wave->owns_embedding = 1u;
	wave->owns_final_head = 1u;
	wave->host_token_ids = fixture->host_token_ids;
	wave->host_resident_slots = fixture->host_resident_slots;
	wave->host_positions = fixture->host_positions;
	wave->embedding_bf16 = fixture->weights.embedding_bf16;
	wave->final_norm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->lm_head_bf16 = fixture->weights.lm_head_bf16;
	wave->layers = fixture->layers;
	wave->slot = &fixture->slot;
	wave->kv_cache = fixture->kv_cache;
	wave->kv_layer_stride_bytes = SPARK_GLM5_NEXT_TAP_CHECK_KV_BYTES;
	wave->index_cache = fixture->index_cache;
	wave->index_layer_stride_bytes = SPARK_GLM5_NEXT_TAP_CHECK_INDEX_BYTES;
	wave->index_ordinal_by_local_layer = fixture->index_ordinal_by_local_layer;
	wave->kda_ordinal_by_local_layer = fixture->kda_ordinal_by_local_layer;
	wave->kda_state_pools = fixture->kda_state_pools;
	wave->kda_state_layer_stride_bytes = SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER;
	wave->kda_q_window_pool = fixture->kda_window_pools;
	wave->kda_k_window_pool = fixture->kda_window_pools +
		SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_SLOT_BYTES;
	wave->kda_v_window_pool = fixture->kda_window_pools +
		2u * SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_SLOT_BYTES;
	wave->kda_window_layer_stride_bytes = SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_SLOT_BYTES;
	wave->kda_state_index = fixture->kda_state_index_device;
	wave->kda_layer_count = SPARK_GLM5_NEXT_TAP_CHECK_KDA_ORDINALS;
	wave->run_count = 1u;
	wave->sequence_row_begin = fixture->slot.run_begin;
	wave->run_state_index = fixture->slot.run_state_index;
	wave->host_sequence_row_begin = fixture->host_run_begin;
	wave->host_run_state_index = fixture->host_run_state_index;
	wave->commit = 1u;
	wave->page_table = fixture->page_table;
	wave->multiprocessor_count = fixture->multiprocessor_count;
	wave->decode_split_context_threshold = 0u;
	wave->attention_split_partials_f32 = fixture->slot.attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		SPARK_GLM5_NEXT_TAP_CHECK_ROW_CAPACITY,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT);
}

static int SparkGlm5NextTapCheckLaunchFail(const char *label,const char *stage,int32_t status)
{
	printf("FAIL %s: %s status %d cuda %s\n",label,stage,(int)status,
		cudaGetErrorString(cudaGetLastError()));
	return(1);
}

static int SparkGlm5NextTapCheckLaunchFailLayer(const char *label,const char *stage,uint32_t local,int32_t status)
{
	printf("FAIL %s: %s local %u status %d cuda %s\n",label,stage,local,(int)status,
		cudaGetErrorString(cudaGetLastError()));
	return(1);
}

static int SparkGlm5NextTapCheckResetKvAccess(SparkGlm5NextTapCheckFixture *fixture,const char *label)
{
	return(SparkGlm5NextTapCheckCuda(cudaMemsetAsync(fixture->slot.kv_access_error,0,
		SPARK_GLM5_NEXT_TAP_CHECK_KV_ACCESS_WORDS * sizeof(uint32_t),fixture->stream),label,"kv_access_reset"));
}

static int SparkGlm5NextTapCheckCheckKvAccess(SparkGlm5NextTapCheckFixture *fixture,const char *label)
{
	uint32_t kv_access[SPARK_GLM5_NEXT_TAP_CHECK_KV_ACCESS_WORDS];
	if ( SparkGlm5NextTapCheckCuda(cudaMemcpy(kv_access,fixture->slot.kv_access_error,
		sizeof(kv_access),cudaMemcpyDeviceToHost),label,"kv_access_readback") != 0 )
		return(1);
	if ( kv_access[0] != 0u )
	{
		printf("FAIL %s: kv_access_error code %u kind %u row %u seq %u pos %u page %u\n",
			label,kv_access[0],kv_access[1],kv_access[2],kv_access[3],kv_access[4],kv_access[5]);
		return(1);
	}
	return(0);
}

static uint32_t SparkGlm5NextTapCheckTapIndex(uint32_t local_layer)
{
	uint32_t tap_index;
	for ( tap_index = 0u; tap_index < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT; tap_index++ )
		if ( SparkGlm5NextTapCheckLayers[tap_index] == local_layer )
			return(tap_index);
	return(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT);
}

static uint16_t *SparkGlm5NextTapCheckReference(
	SparkGlm5NextTapCheckFixture *fixture,
	uint32_t position,
	uint32_t tap_index)
{
	return(fixture->reference_taps +
		((uint64_t)position * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT + tap_index) *
		SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS);
}

static int SparkGlm5NextTapCheckCapture(SparkGlm5NextTapCheckFixture *fixture,uint32_t tap_index)
{
	cudaError_t error;
	SparkStatus status;
	uint32_t row;
	float mean_ms,d2h_ms;
	if ( fixture->capture_pending != 0u &&
		SparkGlm5NextTapCheckCuda(cudaStreamWaitEvent(fixture->stream,(cudaEvent_t)fixture->done_event,0u),"capture","stage_reuse") != 0 )
		return(1);
	error = cudaEventRecord((cudaEvent_t)fixture->begin_event,fixture->stream);
	if ( error == cudaSuccess )
		error = SparkGlm5NextLaunchHcMeanRows(fixture->stream,fixture->slot.hidden_bf16,fixture->slot.tap_stage_bf16,fixture->wave.row_count);
	if ( error == cudaSuccess )
		error = cudaEventRecord((cudaEvent_t)fixture->op_event,fixture->stream);
	if ( SparkGlm5NextTapCheckCuda(error,"capture","mean") != 0 )
		return(1);
	status = SparkGlm5NextTapRingEnqueueCapture(fixture->ring,fixture->stream,fixture->op_event,fixture->done_event,
		fixture->slot.tap_stage_bf16,fixture->wave.row_count,fixture->host_resident_slots,fixture->host_positions,tap_index);
	if ( status != SPARK_STATUS_OK )
	{
		printf("FAIL capture: enqueue tap %u status %d\n",tap_index,(int)status);
		return(1);
	}
	for ( row = 0u; row < fixture->wave.row_count; row++ )
	{
		error = cudaMemcpyAsync(
			SparkGlm5NextTapCheckReference(fixture,fixture->host_positions[row],tap_index),
			fixture->slot.tap_stage_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,
			SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES,cudaMemcpyDeviceToHost,fixture->stream);
		if ( SparkGlm5NextTapCheckCuda(error,"capture","reference") != 0 )
			return(1);
	}
	fixture->capture_pending = 1u;
	if ( SparkGlm5NextTapCheckCuda(cudaEventSynchronize((cudaEvent_t)fixture->done_event),"capture","done") != 0 )
		return(1);
	mean_ms = 0.0f;
	d2h_ms = 0.0f;
	if ( SparkGlm5NextTapCheckCuda(cudaEventElapsedTime(&mean_ms,(cudaEvent_t)fixture->begin_event,(cudaEvent_t)fixture->op_event),"capture","mean_elapsed") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaEventElapsedTime(&d2h_ms,(cudaEvent_t)fixture->op_event,(cudaEvent_t)fixture->done_event),"capture","d2h_elapsed") != 0 )
		return(1);
	fixture->capture_mean_ms += mean_ms;
	fixture->capture_d2h_ms += d2h_ms;
	if ( (double)mean_ms + d2h_ms > fixture->capture_max_ms )
		fixture->capture_max_ms = (double)mean_ms + d2h_ms;
	fixture->capture_count++;
	return(0);
}

static int SparkGlm5NextTapCheckRunWave(SparkGlm5NextTapCheckFixture *fixture,const char *label)
{
	uint32_t local;
	int32_t status;
	if ( SparkGlm5NextTapCheckResetKvAccess(fixture,label) != 0 )
		return(1);
	status = SparkGlm5NextLaunchCudaWaveBegin(&fixture->wave);
	if ( status != 0 )
		return(SparkGlm5NextTapCheckLaunchFail(label,"begin",status));
	for ( local = 0u; local < SPARK_GLM5_NEXT_TAP_CHECK_LAYERS; local++ )
	{
		status = SparkGlm5NextLaunchCudaLayerAttention(&fixture->wave,local);
		if ( status != 0 )
			return(SparkGlm5NextTapCheckLaunchFailLayer(label,"attention",local,status));
		status = SparkGlm5NextLaunchCudaLayerAttentionPost(&fixture->wave,local);
		if ( status != 0 )
			return(SparkGlm5NextTapCheckLaunchFailLayer(label,"attention_post",local,status));
		status = SparkGlm5NextLaunchCudaLayerMlp(&fixture->wave,local);
		if ( status != 0 )
			return(SparkGlm5NextTapCheckLaunchFailLayer(label,"mlp",local,status));
		status = SparkGlm5NextLaunchCudaLayerMlpPost(&fixture->wave,local);
		if ( status != 0 )
			return(SparkGlm5NextTapCheckLaunchFailLayer(label,"mlp_post",local,status));
		if ( SparkGlm5NextTapCheckTapIndex(local) != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT &&
			SparkGlm5NextTapCheckCapture(fixture,SparkGlm5NextTapCheckTapIndex(local)) != 0 )
			return(1);
	}
	status = SparkGlm5NextLaunchCudaWaveHead(&fixture->wave);
	if ( status != 0 )
		return(SparkGlm5NextTapCheckLaunchFail(label,"head",status));
	if ( SparkGlm5NextTapCheckCuda(SparkGlm5NextLaunchHeadMaxlocUnpack(fixture->stream,
		fixture->slot.head_maxloc_u64,fixture->slot.output_token,fixture->wave.row_count),label,"maxloc_unpack") != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckCuda(cudaMemcpyAsync(fixture->host_output,fixture->slot.output_token,
		(uint64_t)fixture->wave.row_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,fixture->stream),label,"output_readback") != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckCuda(cudaStreamSynchronize(fixture->stream),label,"sync") != 0 )
		return(1);
	return(SparkGlm5NextTapCheckCheckKvAccess(fixture,label));
}

static const uint32_t SPARK_GLM5_NEXT_TAP_CHECK_PROMPT[SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS] =
	{ 11u, 902u, 47u, 1888u, 5u, 203u };

static int SparkGlm5NextTapCheckFixtureBuild(SparkGlm5NextTapCheckFixture *fixture)
{
	uint32_t host_page_table[SPARK_GLM5_NEXT_TAP_CHECK_PAGES];
	uint32_t zero;
	uint32_t index;
	cudaError_t error;
	SparkStatus status;
	memset(fixture,0,sizeof(*fixture));
	fixture->random_state = 0x5eed1234u;
	if ( SparkGlm5NextConfigureCudaModule(&fixture->multiprocessor_count) != 0 )
	{
		SparkGlm5NextTapCheckFail("configure","sm_121 target required");
		return(1);
	}
	if ( SparkGlm5NextTapCheckCuda(cudaStreamCreate(&fixture->stream),"fixture","stream") != 0 )
		return(1);
	error = cudaEventCreateWithFlags((cudaEvent_t *)&fixture->begin_event,0u);
	if ( error == cudaSuccess )
		error = cudaEventCreateWithFlags((cudaEvent_t *)&fixture->op_event,0u);
	if ( error == cudaSuccess )
		error = cudaEventCreateWithFlags((cudaEvent_t *)&fixture->done_event,0u);
	if ( SparkGlm5NextTapCheckCuda(error,"fixture","events") != 0 )
		return(1);
	status = SparkGlm5NextTapRingCreate(SPARK_GLM5_NEXT_TAP_CHECK_LANES,SPARK_GLM5_NEXT_TAP_CHECK_WINDOW,
		SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,
		&fixture->ring);
	if ( status != SPARK_STATUS_OK )
	{
		printf("FAIL fixture: tap ring create status %d\n",(int)status);
		return(1);
	}
	if ( SparkGlm5NextTapCheckBuildWeights(fixture) != 0 ||
		SparkGlm5NextTapCheckBuildScratch(fixture) != 0 )
		return(1);
	fixture->kda_state_pools = (uint8_t *)SparkGlm5NextTapCheckAlloc(
		SPARK_GLM5_NEXT_TAP_CHECK_STATE_BYTES,"kda_state_pools");
	fixture->kda_window_pools = (uint8_t *)SparkGlm5NextTapCheckAlloc(
		SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_BYTES,"kda_window_pools");
	fixture->kv_cache = (uint8_t *)SparkGlm5NextTapCheckAlloc(
		(uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_KV_BYTES,"kv_cache");
	fixture->index_cache = (uint8_t *)SparkGlm5NextTapCheckAlloc(
		(uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_INDEX_BYTES,"index_cache");
	fixture->page_table = (uint32_t *)SparkGlm5NextTapCheckAlloc(
		SPARK_GLM5_NEXT_TAP_CHECK_PAGES * sizeof(uint32_t),"page_table");
	fixture->kda_state_index_device = (uint32_t *)SparkGlm5NextTapCheckAlloc(
		sizeof(uint32_t),"kda_state_index");
	if ( fixture->kda_state_pools == 0 || fixture->kda_window_pools == 0 || fixture->kv_cache == 0 ||
		fixture->index_cache == 0 || fixture->page_table == 0 || fixture->kda_state_index_device == 0 )
		return(1);
	for ( index = 0u; index < SPARK_GLM5_NEXT_TAP_CHECK_PAGES; index++ )
		host_page_table[index] = index;
	zero = 0u;
	if ( SparkGlm5NextTapCheckUpload(fixture->page_table,host_page_table,sizeof(host_page_table),"page_table") != 0 ||
		SparkGlm5NextTapCheckUpload(fixture->kda_state_index_device,&zero,sizeof(zero),"kda_state_index") != 0 )
		return(1);
	fixture->slot.kda_state_index = fixture->kda_state_index_device;
	fixture->reference_taps = (uint16_t *)malloc(
		(uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT *
		SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES);
	if ( fixture->reference_taps == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host","reference_taps"));
	if ( SparkGlm5NextTapCheckCuda(cudaMemset(fixture->kda_state_pools,0,(size_t)SPARK_GLM5_NEXT_TAP_CHECK_STATE_BYTES),"fixture","kda_state_clear") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaMemset(fixture->kda_window_pools,0,(size_t)SPARK_GLM5_NEXT_TAP_CHECK_WINDOW_BYTES),"fixture","kda_window_clear") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaMemset(fixture->kv_cache,0,(size_t)((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_KV_BYTES)),"fixture","kv_clear") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaMemset(fixture->index_cache,0,(size_t)((uint64_t)SPARK_GLM5_NEXT_TAP_CHECK_DSA_ORDINALS * SPARK_GLM5_NEXT_TAP_CHECK_INDEX_BYTES)),"fixture","index_clear") != 0 )
		return(1);
	SparkGlm5NextTapCheckBindLayers(fixture);
	return(0);
}

static int SparkGlm5NextTapCheckRun(SparkGlm5NextTapCheckFixture *fixture)
{
	uint32_t current;
	uint32_t step;
	SparkGlm5NextTapCheckBuildWave(fixture,SPARK_GLM5_NEXT_TAP_CHECK_PROMPT,
		0u,SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS);
	if ( SparkGlm5NextTapCheckRunWave(fixture,"prefill") != 0 )
		return(1);
	current = fixture->host_output[SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS - 1u];
	for ( step = 0u; step < SPARK_GLM5_NEXT_TAP_CHECK_DECODE_STEPS; step++ )
	{
		SparkGlm5NextTapCheckBuildWave(fixture,&current,
			SPARK_GLM5_NEXT_TAP_CHECK_PREFILL_ROWS + step,1u);
		if ( SparkGlm5NextTapCheckRunWave(fixture,"decode") != 0 )
			return(1);
		current = fixture->host_output[0];
	}
	printf("run: %u positions captured over %u waves (%u captures)\n",
		SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS,SPARK_GLM5_NEXT_TAP_CHECK_DECODE_STEPS + 1u,fixture->capture_count);
	return(0);
}

static int SparkGlm5NextTapCheckExpectNotFound(
	SparkGlm5NextTapCheckFixture *fixture,
	uint32_t lane,
	uint32_t position,
	const char *label)
{
	const uint16_t *row;
	SparkStatus status;
	row = 0;
	status = SparkGlm5NextTapRingRead(fixture->ring,lane,position,&row);
	if ( status != SPARK_STATUS_NOT_FOUND )
	{
		printf("FAIL %s: read lane %u position %u expected NOT_FOUND got status %d\n",
			label,lane,position,(int)status);
		return(1);
	}
	return(0);
}

static int SparkGlm5NextTapCheckVerifyWindow(SparkGlm5NextTapCheckFixture *fixture)
{
	const uint16_t *row;
	SparkStatus status;
	uint32_t position,tap;
	if ( SparkGlm5NextTapCheckCuda(cudaDeviceSynchronize(),"verify","drain") != 0 )
		return(1);
	if ( SparkGlm5NextTapRingCommitAnchor(fixture->ring,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS) != SPARK_STATUS_OK )
		return(SparkGlm5NextTapCheckFail("verify","anchor_commit"));
	for ( position = 0u; position < SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS; position++ )
	{
		if ( position + SPARK_GLM5_NEXT_TAP_CHECK_WINDOW < SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS )
		{
			if ( SparkGlm5NextTapCheckExpectNotFound(fixture,0u,position,"verify_stale") != 0 )
				return(1);
			continue;
		}
		status = SparkGlm5NextTapRingRead(fixture->ring,0u,position,&row);
		if ( status != SPARK_STATUS_OK )
		{
			printf("FAIL verify: read position %u status %d\n",position,(int)status);
			return(1);
		}
		for ( tap = 0u; tap < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT; tap++ )
			if ( memcmp(row + (uint64_t)tap * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,
				SparkGlm5NextTapCheckReference(fixture,position,tap),SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES) != 0 )
			{
				printf("FAIL verify: ring position %u tap %u diverges from the wave residual\n",position,tap);
				return(1);
			}
	}
	if ( SparkGlm5NextTapCheckExpectNotFound(fixture,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS,"verify_anchor") != 0 ||
		SparkGlm5NextTapCheckExpectNotFound(fixture,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS + 82u,"verify_uncaptured") != 0 ||
		SparkGlm5NextTapCheckExpectNotFound(fixture,1u,12u,"verify_lane_silent") != 0 )
		return(1);
	if ( SparkGlm5NextTapRingCommitAnchor(fixture->ring,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS - 1u) != SPARK_STATUS_OK )
		return(SparkGlm5NextTapCheckFail("verify","anchor_rollback"));
	if ( SparkGlm5NextTapCheckExpectNotFound(fixture,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS - 1u,"verify_rejected_row") != 0 )
		return(1);
	if ( SparkGlm5NextTapRingCommitAnchor(fixture->ring,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS) != SPARK_STATUS_OK )
		return(SparkGlm5NextTapCheckFail("verify","anchor_restore"));
	printf("verify: window [%u,%u) byte-exact vs wave residual; stale, uncommitted and uncaptured reads rejected\n",
		SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS - SPARK_GLM5_NEXT_TAP_CHECK_WINDOW,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS);
	return(0);
}

static int SparkGlm5NextTapCheckVerifyResidual(SparkGlm5NextTapCheckFixture *fixture)
{
	const uint16_t *row;
	uint16_t *host_streams;
	uint16_t *host_mean;
	uint32_t element,stream;
	SparkStatus status;
	host_streams = (uint16_t *)malloc((uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT *
		SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
	host_mean = (uint16_t *)malloc(SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES);
	if ( host_streams == 0 || host_mean == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host","residual_check"));
	if ( SparkGlm5NextTapCheckCuda(cudaMemcpy(host_streams,fixture->slot.hidden_bf16,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
		cudaMemcpyDeviceToHost),"residual","readback") != 0 )
		return(1);
	for ( element = 0u; element < SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION; element++ )
	{
		float value = 0.0f;
		for ( stream = 0u; stream < SPARK_GLM5_NEXT_MODEL_HC_MULT; stream++ )
		{
			uint32_t raw = (uint32_t)host_streams[(uint64_t)stream * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION + element] << 16u;
			float element_value;
			memcpy(&element_value,&raw,sizeof(element_value));
			value += element_value;
		}
		value /= (float)SPARK_GLM5_NEXT_MODEL_HC_MULT;
		host_mean[element] = SparkGlm5NextTapCheckRoundBf16(value);
	}
	status = SparkGlm5NextTapRingRead(fixture->ring,0u,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS - 1u,&row);
	if ( status != SPARK_STATUS_OK )
		return(SparkGlm5NextTapCheckFail("residual","ring_read"));
	if ( memcmp(host_mean,SparkGlm5NextTapCheckReference(fixture,SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS - 1u,
		SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT - 1u),SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES) != 0 ||
		memcmp(host_mean,row + (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT - 1u) *
			SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES) != 0 )
	{
		printf("FAIL residual: tap is not the byte-exact HC mean of the post-MlpPost hidden at the last position\n");
		return(1);
	}
	free(host_streams);
	free(host_mean);
	printf("verify: ring tap at the last position is byte-exact the HC mean of the post-MlpPost residual\n");
	return(0);
}

static int SparkGlm5NextTapCheckVerifyLanes(SparkGlm5NextTapCheckFixture *fixture)
{
	const uint16_t *row;
	uint16_t *host_pattern;
	uint32_t lanes[2];
	uint32_t positions[2];
	uint32_t row_index;
	uint64_t element;
	cudaError_t error;
	SparkStatus status;
	host_pattern = (uint16_t *)malloc(2u * SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES);
	if ( host_pattern == 0 )
		return(SparkGlm5NextTapCheckFail("alloc_host","lane_pattern"));
	for ( element = 0u; element < 2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS; element++ )
		host_pattern[element] = SparkGlm5NextTapCheckBf16(
			(((float)(int32_t)(SparkGlm5NextTapCheckNext(&fixture->random_state) & 0xffffu) - 32768.0f) / 32768.0f) * 0.5f);
	if ( fixture->capture_pending != 0u &&
		SparkGlm5NextTapCheckCuda(cudaStreamWaitEvent(fixture->stream,(cudaEvent_t)fixture->done_event,0u),"lanes","stage_reuse") != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckCuda(cudaMemcpyAsync(fixture->slot.tap_stage_bf16,host_pattern,
		2u * SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES,cudaMemcpyHostToDevice,fixture->stream),"lanes","upload") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaStreamSynchronize(fixture->stream),"lanes","upload_sync") != 0 )
		return(1);
	lanes[0] = 1u;
	lanes[1] = 0u;
	positions[0] = 25u;
	positions[1] = 26u;
	error = cudaEventRecord((cudaEvent_t)fixture->op_event,fixture->stream);
	if ( SparkGlm5NextTapCheckCuda(error,"lanes","op_event") != 0 )
		return(1);
	status = SparkGlm5NextTapRingEnqueueCapture(fixture->ring,fixture->stream,fixture->op_event,fixture->done_event,
		fixture->slot.tap_stage_bf16,2u,lanes,positions,3u);
	if ( status != SPARK_STATUS_OK )
	{
		printf("FAIL lanes: enqueue status %d\n",(int)status);
		return(1);
	}
	if ( SparkGlm5NextTapCheckCuda(cudaEventSynchronize((cudaEvent_t)fixture->done_event),"lanes","done") != 0 ||
		SparkGlm5NextTapCheckCuda(cudaDeviceSynchronize(),"lanes","drain") != 0 )
		return(1);
	if ( SparkGlm5NextTapRingCommitAnchor(fixture->ring,1u,26u) != SPARK_STATUS_OK ||
		SparkGlm5NextTapRingCommitAnchor(fixture->ring,0u,27u) != SPARK_STATUS_OK )
		return(SparkGlm5NextTapCheckFail("lanes","anchor_commit"));
	for ( row_index = 0u; row_index < 2u; row_index++ )
	{
		status = SparkGlm5NextTapRingRead(fixture->ring,lanes[row_index],positions[row_index],&row);
		if ( status != SPARK_STATUS_OK )
		{
			printf("FAIL lanes: read lane %u position %u status %d\n",lanes[row_index],positions[row_index],(int)status);
			return(1);
		}
		if ( memcmp(row + 3u * (uint64_t)SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,
			host_pattern + (uint64_t)row_index * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_WIDTH_ELEMENTS,
			SPARK_GLM5_NEXT_TAP_CHECK_TAP_BYTES) != 0 )
		{
			printf("FAIL lanes: lane %u position %u tap 3 payload mismatch\n",lanes[row_index],positions[row_index]);
			return(1);
		}
	}
	if ( SparkGlm5NextTapCheckExpectNotFound(fixture,1u,17u,"lanes_uncaptured") != 0 )
		return(1);
	free(host_pattern);
	printf("verify: per-lane ring addressing byte-exact for lanes {1,0} at positions {25,26}\n");
	return(0);
}

int main(int argc,char **argv)
{
	static SparkGlm5NextTapCheckFixture fixture;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("glm5_next tap ring: configuration %s\n",argv[1]);
	if ( SparkGlm5NextTapCheckFixtureBuild(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckRun(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckVerifyWindow(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckVerifyResidual(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextTapCheckVerifyLanes(&fixture) != 0 )
		return(1);
	printf("capture cost: %u captures, mean kernel avg %.4f ms, ring D2H avg %.4f ms, capture total max %.4f ms\n",
		fixture.capture_count,
		fixture.capture_count != 0u ? fixture.capture_mean_ms / fixture.capture_count : 0.0,
		fixture.capture_count != 0u ? fixture.capture_d2h_ms / fixture.capture_count : 0.0,
		fixture.capture_max_ms);
	printf("PASS glm5_next tap ring: %u positions x %u taps byte-exact; wrap, anchor, uncaptured and lane gates verified\n",
		SPARK_GLM5_NEXT_TAP_CHECK_POSITIONS,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_TAP_LAYER_COUNT);
	SparkGlm5NextTapRingDestroy(fixture.ring);
	return(0);
}
