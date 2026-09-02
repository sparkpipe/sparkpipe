#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the compiled archive's expert codec"
#endif

#define SPARK_GLM5_NEXT_MTP_PARITY_EXPERT_CODEC_ID 5u
#if GLM5_NEXT_EXPERT_WEIGHT_CODEC != SPARK_GLM5_NEXT_MTP_PARITY_EXPERT_CODEC_ID
#error "the MTP parity harness synthesizes fp8 expert payloads; build the archive with EXPERT_CODEC=fp8"
#endif

#define SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS 6u
#define SPARK_GLM5_NEXT_MTP_PARITY_DECODE_TOKENS 80u
#define SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS \
	(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u)
#define SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS \
	(SPARK_GLM5_NEXT_MTP_PARITY_DECODE_TOKENS + SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS)
#define SPARK_GLM5_NEXT_MTP_PARITY_LAYERS 4u
#define SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS 3u
#define SPARK_GLM5_NEXT_MTP_PARITY_PAGES 2u
#define SPARK_GLM5_NEXT_MTP_PARITY_PAGE_TOKENS 64u
#define SPARK_GLM5_NEXT_MTP_PARITY_MAX_POSITIONS \
	(SPARK_GLM5_NEXT_MTP_PARITY_PAGES * SPARK_GLM5_NEXT_MTP_PARITY_PAGE_TOKENS)
#define SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY 8u
#define SPARK_GLM5_NEXT_MTP_PARITY_SNAPSHOT_COUNT \
	(SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS)
#define SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILE 1024u
#define SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILES \
	((SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT + SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILE - 1u) / \
	 SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILE)
#define SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS (1u << 22)
#define SPARK_GLM5_NEXT_MTP_PARITY_KV_ACCESS_WORDS 6u

#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ 0.02f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_ATTN_OUT 0.01f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_DENSE 0.01f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_KDA_OUT 0.005f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_CONV 0.05f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_ROUTER 0.002f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_SHARED 0.005f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_EMBEDDING 0.02f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_HEAD 0.01f
#define SPARK_GLM5_NEXT_MTP_PARITY_SCALE_EH_PROJ 0.01f
#define SPARK_GLM5_NEXT_MTP_PARITY_EXPERT_SCALE 6.103515625e-05f
#define SPARK_GLM5_NEXT_MTP_PARITY_KDA_DECAY_BIAS -0.25f
#define SPARK_GLM5_NEXT_MTP_PARITY_KDA_HEAD_LOG_SCALE 0.1f
#define SPARK_GLM5_NEXT_MTP_PARITY_ROUTER_CORRECTION 4.0f

#define SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_SLOT_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION * \
	 SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL * \
	 SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES)
#define SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * \
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER)
#define SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * 3u * \
	 SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_SLOT_BYTES)
#define SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_PAGES * SPARK_GLM5_NEXT_MTP_PARITY_PAGE_TOKENS * \
	 SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES)
#define SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES \
	((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_PAGES * SPARK_GLM5_NEXT_MTP_PARITY_PAGE_TOKENS * \
	 SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u)
#define SPARK_GLM5_NEXT_MTP_PARITY_SNAPSHOT_BLOCK_BYTES \
	(SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES + SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES + \
	 SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES + SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES)

static_assert(
	SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS + \
		SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS <= SPARK_GLM5_NEXT_MTP_PARITY_MAX_POSITIONS,
	"parity walk must fit the page table");
static_assert(SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS <= SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY,
	"verify wave must fit the scratch row capacity");
static_assert(SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS <= SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY,
	"prefill wave must fit the scratch row capacity");
static_assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH == 2u,
	"the parity harness drives exactly the depth-2 chain");

typedef enum SparkGlm5NextMtpParityMode
{
	SPARK_GLM5_NEXT_MTP_PARITY_MODE_ORGANIC = 0,
	SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_REJECT,
	SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_MID,
	SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_ACCEPT,
	SPARK_GLM5_NEXT_MTP_PARITY_MODE_COUNT
} SparkGlm5NextMtpParityMode;

typedef struct SparkGlm5NextMtpParityWeights
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
	uint16_t *mtp_eh_proj_bf16;
} SparkGlm5NextMtpParityWeights;

typedef struct SparkGlm5NextMtpParityFixture
{
	cudaStream_t stream;
	uint32_t multiprocessor_count;
	uint64_t random_state;
	SparkGlm5NextMtpParityWeights weights;
	SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_MTP_PARITY_LAYERS];
	SparkGlm5NextLayerWeights mtp_layer;
	SparkGlm5NextExecutionSlot slot;
	SparkGlm5NextCudaWave wave;
	SparkGlm5NextCudaWave draft_wave;
	uint64_t kda_replay_layer_bytes;
	uint8_t *kda_state_pools;
	uint8_t *kda_window_pools;
	uint8_t *kv_cache;
	uint8_t *index_cache;
	uint32_t *page_table;
	uint32_t *kda_state_index_device;
	uint32_t index_ordinal_by_local_layer[SPARK_GLM5_NEXT_MTP_PARITY_LAYERS];
	uint32_t kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_MTP_PARITY_LAYERS];
	uint16_t *draft_hidden_bf16;
	uint32_t host_token_ids[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY];
	uint32_t host_positions[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY];
	uint32_t host_resident_slots[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY];
	uint32_t host_run_begin[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY + 1u];
	uint32_t host_run_state_index[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY];
	uint32_t host_output[SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY];
	uint32_t reference_tokens[SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS];
	uint8_t *snapshots;
} SparkGlm5NextMtpParityFixture;

static int SparkGlm5NextMtpParityFail(const char *check,const char *detail)
{
	printf("FAIL %s: %s\n",check,detail);
	return(1);
}

static int SparkGlm5NextMtpParityCuda(cudaError_t error,const char *check,const char *detail)
{
	if ( error == cudaSuccess )
		return(0);
	printf("FAIL %s: %s cuda=%s\n",check,detail,cudaGetErrorString(error));
	return(1);
}

static uint32_t SparkGlm5NextMtpParityNext(uint64_t *state)
{
	uint64_t value;
	value = *state;
	value ^= value << 13u;
	value ^= value >> 7u;
	value ^= value << 17u;
	*state = value;
	return((uint32_t)(value >> 32u));
}

static uint16_t SparkGlm5NextMtpParityBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)(bits >> 16u));
}

static void *SparkGlm5NextMtpParityAlloc(uint64_t bytes,const char *name)
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

static int SparkGlm5NextMtpParityUpload(
	const void *device,
	const void *host,
	uint64_t bytes,
	const char *name)
{
	return(SparkGlm5NextMtpParityCuda(cudaMemcpy((void *)device,host,(size_t)bytes,cudaMemcpyHostToDevice),"upload",name));
}

static int SparkGlm5NextMtpParityFillBf16(
	SparkGlm5NextMtpParityFixture *fixture,
	uint16_t **out,
	uint64_t elements,
	float scale,
	uint32_t constant_one,
	const char *name)
{
	uint16_t *chunk;
	uint16_t *device;
	uint64_t offset;
	device = (uint16_t *)SparkGlm5NextMtpParityAlloc(elements * sizeof(uint16_t),name);
	if ( device == 0 )
		return(1);
	chunk = (uint16_t *)malloc((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS * sizeof(uint16_t));
	if ( chunk == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host",name));
	for ( offset = 0u; offset < elements; )
	{
		uint64_t count = elements - offset;
		uint64_t index;
		if ( count > SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index < count; index++ )
		{
			float value;
			if ( constant_one != 0u )
				value = 1.0f;
			else
				value = (((float)(int32_t)(SparkGlm5NextMtpParityNext(&fixture->random_state) & 0xffffu) - 32768.0f) / 32768.0f) * scale;
			chunk[index] = SparkGlm5NextMtpParityBf16(value);
		}
		if ( SparkGlm5NextMtpParityUpload(device + offset,chunk,count * sizeof(uint16_t),name) != 0 )
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

static int SparkGlm5NextMtpParityFillF32(
	float **out,
	uint64_t count,
	float value,
	const char *name)
{
	float *host;
	float *device;
	uint64_t index;
	device = (float *)SparkGlm5NextMtpParityAlloc(count * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((size_t)(count * sizeof(float)));
	if ( host == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host",name));
	for ( index = 0u; index < count; index++ )
		host[index] = value;
	if ( SparkGlm5NextMtpParityUpload(device,host,count * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextMtpParityFillF32Grid(
	float **out,
	uint64_t count,
	uint32_t modulus,
	float scale,
	const char *name)
{
	float *host;
	float *device;
	uint64_t index;
	device = (float *)SparkGlm5NextMtpParityAlloc(count * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((size_t)(count * sizeof(float)));
	if ( host == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host",name));
	for ( index = 0u; index < count; index++ )
		host[index] = ((float)(index % modulus) - (float)(modulus / 2u)) * scale;
	if ( SparkGlm5NextMtpParityUpload(device,host,count * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextMtpParityFillExperts(
	SparkGlm5NextMtpParityFixture *fixture,
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
		return(SparkGlm5NextMtpParityFail("expert_shape",name));
	payload = (uint8_t *)SparkGlm5NextMtpParityAlloc(payload_bytes,name);
	scale_device = (uint8_t *)SparkGlm5NextMtpParityAlloc(scale_bytes,name);
	if ( payload == 0 || scale_device == 0 )
		return(1);
	chunk = (uint8_t *)malloc((uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS);
	if ( chunk == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host",name));
	for ( offset = 0u; offset < payload_bytes; )
	{
		uint64_t count = payload_bytes - offset;
		uint64_t index;
		if ( count > SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index < count; index++ )
		{
			uint8_t raw = (uint8_t)SparkGlm5NextMtpParityNext(&fixture->random_state);
			if ( (raw & 0x7fu) == 0x7fu )
				raw &= 0x7eu;
			chunk[index] = raw;
		}
		if ( SparkGlm5NextMtpParityUpload(payload + offset,chunk,count,name) != 0 )
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
		if ( count > SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS )
			count = SPARK_GLM5_NEXT_MTP_PARITY_FILL_CHUNK_ELEMENTS;
		for ( index = 0u; index + sizeof(float) <= count; index += sizeof(float) )
		{
			float value = SPARK_GLM5_NEXT_MTP_PARITY_EXPERT_SCALE;
			memcpy(chunk + index,&value,sizeof(value));
		}
		if ( SparkGlm5NextMtpParityUpload(scale_device + offset,chunk,count,name) != 0 )
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

static int SparkGlm5NextMtpParityFillRouterCorrection(float **out,const char *name)
{
	float *host;
	float *device;
	uint32_t index;
	device = (float *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float),name);
	if ( device == 0 )
		return(1);
	host = (float *)malloc((uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float));
	if ( host == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host",name));
	for ( index = 0u; index < SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT; index++ )
		host[index] = index < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K ?
			SPARK_GLM5_NEXT_MTP_PARITY_ROUTER_CORRECTION : -SPARK_GLM5_NEXT_MTP_PARITY_ROUTER_CORRECTION;
	if ( SparkGlm5NextMtpParityUpload(device,host,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * sizeof(float),name) != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	*out = device;
	return(0);
}

static int SparkGlm5NextMtpParityBuildWeights(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkGlm5NextMtpParityWeights *weights;
	uint64_t hidden;
	uint32_t kda_dim;
	weights = &fixture->weights;
	hidden = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	kda_dim = SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION;
	if ( SparkGlm5NextMtpParityFillBf16(fixture,&weights->norm_hidden_bf16,hidden,0.0f,1u,"norm_hidden") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->norm_query_a_bf16,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,0.0f,1u,"norm_query_a") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->norm_kv_a_bf16,SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION,0.0f,1u,"norm_kv_a") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->norm_index_bf16,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION,0.0f,1u,"norm_index") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_qkv_beta_bf16,(3u * kda_dim + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT) * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_DENSE,0u,"kda_qkv_beta") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_decay_gate_down_bf16,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_DENSE,0u,"kda_decay_gate_down") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_decay_up_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"kda_decay_up") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_gate_up_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"kda_gate_up") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_q_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_CONV,0u,"kda_q_conv") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_k_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_CONV,0u,"kda_k_conv") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_v_conv_bf16,(uint64_t)kda_dim * SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_CONV,0u,"kda_v_conv") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kda_out_bf16,hidden * kda_dim,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_KDA_OUT,0u,"kda_out") != 0 ||
		SparkGlm5NextMtpParityFillF32(&weights->kda_out_norm_f32,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION,1.0f,"kda_out_norm") != 0 ||
		SparkGlm5NextMtpParityFillF32(&weights->kda_decay_bias_f32,kda_dim,SPARK_GLM5_NEXT_MTP_PARITY_KDA_DECAY_BIAS,"kda_decay_bias") != 0 ||
		SparkGlm5NextMtpParityFillF32(&weights->kda_head_log_scale_f32,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,SPARK_GLM5_NEXT_MTP_PARITY_KDA_HEAD_LOG_SCALE,"kda_head_log_scale") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->q_a_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"q_a") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->q_b_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"q_b") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kv_a_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"kv_a") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kv_b_key_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"kv_b_key") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->kv_b_value_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"kv_b_value") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->attn_output_bf16,hidden * SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_ATTN_OUT,0u,"attn_output") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->index_q_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION * SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"index_q") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->index_k_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"index_k") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->index_head_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_ATTN_OUT,0u,"index_head") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->index_compress_gate_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_PROJ,0u,"index_compress_gate") != 0 ||
		SparkGlm5NextMtpParityFillF32Grid(&weights->index_compress_ape_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION,5u,0.25f,"index_compress_ape") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->dense_gate_up_bf16,2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_DENSE,0u,"dense_gate_up") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->dense_down_bf16,hidden * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_DENSE,0u,"dense_down") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->router_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_ROUTER,0u,"router") != 0 ||
		SparkGlm5NextMtpParityFillRouterCorrection(&weights->router_correction_f32,"router_correction") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->shared_gate_up_bf16,2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_SHARED,0u,"shared_gate_up") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->shared_down_bf16,hidden * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_SHARED,0u,"shared_down") != 0 ||
		SparkGlm5NextMtpParityFillExperts(fixture,&weights->expert_up_gate_payload,&weights->expert_up_gate_scale,2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,"expert_up_gate") != 0 ||
		SparkGlm5NextMtpParityFillExperts(fixture,&weights->expert_down_payload,&weights->expert_down_scale,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,"expert_down") != 0 ||
		SparkGlm5NextMtpParityFillF32Grid(&weights->hc_attn_fn_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden,7u,0.01f,"hc_attn_fn") != 0 ||
		SparkGlm5NextMtpParityFillF32Grid(&weights->hc_attn_base_f32,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,5u,0.25f,"hc_attn_base") != 0 ||
		SparkGlm5NextMtpParityFillF32(&weights->hc_attn_scale_f32,SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT,0.5f,"hc_attn_scale") != 0 ||
		SparkGlm5NextMtpParityFillF32Grid(&weights->hc_ffn_fn_f32,(uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden,9u,0.01f,"hc_ffn_fn") != 0 ||
		SparkGlm5NextMtpParityFillF32Grid(&weights->hc_ffn_base_f32,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,3u,0.25f,"hc_ffn_base") != 0 ||
		SparkGlm5NextMtpParityFillF32(&weights->hc_ffn_scale_f32,SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT,0.5f,"hc_ffn_scale") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->embedding_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_EMBEDDING,0u,"embedding") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->lm_head_bf16,(uint64_t)SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_HEAD,0u,"lm_head") != 0 ||
		SparkGlm5NextMtpParityFillBf16(fixture,&weights->mtp_eh_proj_bf16,hidden * 2u * hidden,SPARK_GLM5_NEXT_MTP_PARITY_SCALE_EH_PROJ,0u,"mtp_eh_proj") != 0 )
		return(1);
	return(0);
}

#define SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(field,rows,columns) \
	slot->field = (uint16_t *)SparkGlm5NextMtpParityAlloc( \
		(uint64_t)(rows) * (columns) * sizeof(uint16_t),#field); \
	if ( slot->field == 0 ) \
		return(1);
#define SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(field,rows,columns) \
	slot->field = (float *)SparkGlm5NextMtpParityAlloc( \
		(uint64_t)(rows) * (columns) * sizeof(float),#field); \
	if ( slot->field == 0 ) \
		return(1);
#define SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(field,count) \
	slot->field = (uint32_t *)SparkGlm5NextMtpParityAlloc( \
		(uint64_t)(count) * sizeof(uint32_t),#field); \
	if ( slot->field == 0 ) \
		return(1);

static int SparkGlm5NextMtpParityBuildScratch(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkGlm5NextExecutionSlot *slot;
	uint64_t rows;
	uint64_t hidden;
	uint64_t kda_dim;
	slot = &fixture->slot;
	rows = SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY;
	hidden = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	kda_dim = SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION;
	memset(slot,0,sizeof(*slot));
	slot->stream = fixture->stream;
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(hidden_bf16,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(residual_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(normed_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(q_compressed_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(q_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(query_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION)
	slot->query_rope_bf16 = 0;
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(index_query_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(index_key_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(index_gate_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(index_packed_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(selected_pools,rows * SPARK_GLM5_NEXT_MODEL_INDEX_POOL_SELECT_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(index_head_weight_bf16,rows,SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kv_slot_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(attention_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(attention_value_bf16,rows,SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(attention_out_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(gate_up_bf16,rows,2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(intermediate_bf16,rows,SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(expert_out_bf16,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(shared_out_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(fused_qkvb_bf16,rows,3u * kda_dim + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(fused_decay_gate_bf16,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_decay_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_gate_latent_bf16,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_beta_logit,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_gate_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_decay_logit_bf16,rows,kda_dim)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(kda_output_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(kda_retention,rows,kda_dim)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(kda_write_gate,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(hc_mixes_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(hc_pre_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(hc_post_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(hc_comb_f32,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(hc_collapsed_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(hc_snapshot_bf16,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(hc_mean_bf16,rows,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(router_logits_f32,rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(selection_scores_f32,rows,SPARK_GLM5_NEXT_MTP_PARITY_MAX_POSITIONS / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(selected_positions,rows * SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(route_expert,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(route_weight,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(route_source_token,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(route_packed_row,rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(group_row_offset,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(group_tile_prefix_w1,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(group_tile_prefix_w2,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(token_ids,rows)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(resident_slots,rows)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(positions,rows)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(context_lengths,1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(dense_row_offset,4u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(dense_tile_prefix,4u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(run_begin,rows + 1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(run_state_index,rows)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(output_token,rows)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(output_score,rows,1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_F32(head_candidate_score,rows,SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILES)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(head_candidate_token,rows * SPARK_GLM5_NEXT_MTP_PARITY_HEAD_TILES)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(kv_access_error,SPARK_GLM5_NEXT_MTP_PARITY_KV_ACCESS_WORDS)
	slot->head_maxloc_u64 = (uint64_t *)SparkGlm5NextMtpParityAlloc(rows * sizeof(uint64_t),"head_maxloc_u64");
	if ( slot->head_maxloc_u64 == 0 )
		return(1);
	slot->attention_split_partials_f32 = (float *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT),
		"attention_split_partials_f32");
	if ( slot->attention_split_partials_f32 == 0 )
		return(1);
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(mtp_hidden_bf16,1u,hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_BF16(mtp_concat_bf16,1u,2u * hidden)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(mtp_positions,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(mtp_context,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(mtp_page_table,1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(mtp_sequence,1u)
	SPARK_GLM5_NEXT_MTP_PARITY_SCRATCH_U32(mtp_committed,1u)
	slot->mtp_kv_pool = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES,"mtp_kv_pool");
	slot->mtp_index_pool = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u *
			SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT,"mtp_index_pool");
	slot->mtp_replay_steps = SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS *
			SPARK_GLM5_NEXT_MTP_REPLAY_STEP_BYTES,"mtp_replay_steps");
	slot->mtp_conv_scratch = (uint16_t *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS * kda_dim * sizeof(uint16_t),"mtp_conv_scratch");
	slot->kda_replay_pool = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * fixture->kda_replay_layer_bytes,"kda_replay_pool");
	if ( slot->mtp_kv_pool == 0 || slot->mtp_index_pool == 0 || slot->mtp_replay_steps == 0 ||
		slot->mtp_conv_scratch == 0 || slot->kda_replay_pool == 0 )
		return(1);
	return(0);
}

static void SparkGlm5NextMtpParityBindLayers(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkGlm5NextMtpParityWeights *weights;
	uint32_t local;
	weights = &fixture->weights;
	for ( local = 0u; local < SPARK_GLM5_NEXT_MTP_PARITY_LAYERS; local++ )
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
	}
	for ( local = 0u; local < SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS; local++ )
	{
		SparkGlm5NextLayerWeights *layer;
		layer = &fixture->layers[local];
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
		layer->dense_gate_up_bf16 = weights->dense_gate_up_bf16;
		layer->dense_down_bf16 = weights->dense_down_bf16;
		fixture->kda_ordinal_by_local_layer[local] = local;
		fixture->index_ordinal_by_local_layer[local] = UINT32_MAX;
	}
	{
		SparkGlm5NextLayerWeights *layer;
		layer = &fixture->layers[SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS];
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
		layer->router_bf16 = weights->router_bf16;
		layer->router_correction_f32 = weights->router_correction_f32;
		layer->expert_up_gate_payload = weights->expert_up_gate_payload;
		layer->expert_up_gate_scale = weights->expert_up_gate_scale;
		layer->expert_down_payload = weights->expert_down_payload;
		layer->expert_down_scale = weights->expert_down_scale;
		layer->shared_gate_up_bf16 = weights->shared_gate_up_bf16;
		layer->shared_down_bf16 = weights->shared_down_bf16;
		fixture->kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS] = UINT32_MAX;
		fixture->index_ordinal_by_local_layer[SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS] = 0u;
	}
	fixture->mtp_layer = fixture->layers[SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS];
}

static void SparkGlm5NextMtpParityBuildWave(
	SparkGlm5NextMtpParityFixture *fixture,
	const uint32_t *tokens,
	uint32_t first_position,
	uint32_t row_count,
	uint32_t commit,
	uint32_t mtp_verify)
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
	wave->layer_count = SPARK_GLM5_NEXT_MTP_PARITY_LAYERS;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = row_count;
	wave->maximum_context = first_position + row_count;
	wave->resident_sequence_capacity = 1u;
	wave->max_sequence_positions = SPARK_GLM5_NEXT_MTP_PARITY_MAX_POSITIONS;
	wave->execution_row_capacity = SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY;
	wave->pages_per_sequence = SPARK_GLM5_NEXT_MTP_PARITY_PAGES;
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
	wave->kv_layer_stride_bytes = SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES;
	wave->index_cache = fixture->index_cache;
	wave->index_layer_stride_bytes = SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES;
	wave->index_ordinal_by_local_layer = fixture->index_ordinal_by_local_layer;
	wave->kda_ordinal_by_local_layer = fixture->kda_ordinal_by_local_layer;
	wave->kda_state_pools = fixture->kda_state_pools;
	wave->kda_state_layer_stride_bytes = SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER;
	wave->kda_q_window_pool = fixture->kda_window_pools;
	wave->kda_k_window_pool = fixture->kda_window_pools +
		SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_SLOT_BYTES;
	wave->kda_v_window_pool = fixture->kda_window_pools +
		2u * SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS * SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_SLOT_BYTES;
	wave->kda_window_layer_stride_bytes = SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_SLOT_BYTES;
	wave->kda_state_index = fixture->kda_state_index_device;
	wave->kda_layer_count = SPARK_GLM5_NEXT_MTP_PARITY_KDA_ORDINALS;
	wave->run_count = 1u;
	wave->sequence_row_begin = fixture->slot.run_begin;
	wave->run_state_index = fixture->slot.run_state_index;
	wave->host_sequence_row_begin = fixture->host_run_begin;
	wave->host_run_state_index = fixture->host_run_state_index;
	wave->commit = commit;
	wave->mtp_verify = mtp_verify;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->mtp_layer_weights = &fixture->mtp_layer;
	wave->mtp_eh_proj_bf16 = fixture->weights.mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->mtp_hnorm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->mtp_shared_norm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->kda_replay_layer_bytes = fixture->kda_replay_layer_bytes;
	wave->page_table = fixture->page_table;
	wave->multiprocessor_count = fixture->multiprocessor_count;
	wave->decode_split_context_threshold = 0u;
	wave->attention_split_partials_f32 = fixture->slot.attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		SPARK_GLM5_NEXT_MTP_PARITY_ROW_CAPACITY,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT);
}

static void SparkGlm5NextMtpParityBuildDraftWave(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkGlm5NextCudaWave *wave;
	wave = &fixture->draft_wave;
	memset(wave,0,sizeof(*wave));
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->owns_final_head = 1u;
	wave->multiprocessor_count = fixture->multiprocessor_count;
	wave->embedding_bf16 = fixture->weights.embedding_bf16;
	wave->lm_head_bf16 = fixture->weights.lm_head_bf16;
	wave->mtp_layer_weights = &fixture->mtp_layer;
	wave->mtp_eh_proj_bf16 = fixture->weights.mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->mtp_hnorm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->mtp_shared_norm_bf16 = fixture->weights.norm_hidden_bf16;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->slot = &fixture->slot;
}

static int SparkGlm5NextMtpParityRunWave(SparkGlm5NextMtpParityFixture *fixture,const char *label)
{
	uint32_t local;
	uint32_t kv_access[SPARK_GLM5_NEXT_MTP_PARITY_KV_ACCESS_WORDS];
	if ( SparkGlm5NextMtpParityCuda(cudaMemsetAsync(fixture->slot.kv_access_error,0,
		SPARK_GLM5_NEXT_MTP_PARITY_KV_ACCESS_WORDS * sizeof(uint32_t),fixture->stream),label,"kv_access_reset") != 0 )
		return(1);
	if ( SparkGlm5NextLaunchCudaWaveBegin(&fixture->wave) != 0 )
		return(SparkGlm5NextMtpParityFail(label,"begin"));
	for ( local = 0u; local < SPARK_GLM5_NEXT_MTP_PARITY_LAYERS; local++ )
	{
		if ( SparkGlm5NextLaunchCudaLayerAttention(&fixture->wave,local) != 0 )
			return(SparkGlm5NextMtpParityFail(label,"attention"));
		if ( SparkGlm5NextLaunchCudaLayerAttentionPost(&fixture->wave,local) != 0 )
			return(SparkGlm5NextMtpParityFail(label,"attention_post"));
		if ( SparkGlm5NextLaunchCudaLayerMlp(&fixture->wave,local) != 0 )
			return(SparkGlm5NextMtpParityFail(label,"mlp"));
		if ( SparkGlm5NextLaunchCudaLayerMlpPost(&fixture->wave,local) != 0 )
			return(SparkGlm5NextMtpParityFail(label,"mlp_post"));
	}
	if ( SparkGlm5NextLaunchCudaWaveHead(&fixture->wave) != 0 )
		return(SparkGlm5NextMtpParityFail(label,"head"));
	if ( SparkGlm5NextMtpParityCuda(SparkGlm5NextLaunchHeadMaxlocUnpack(fixture->stream,
		fixture->slot.head_maxloc_u64,fixture->slot.output_token,fixture->wave.row_count),label,"maxloc_unpack") != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityCuda(cudaMemcpyAsync(fixture->host_output,fixture->slot.output_token,
		(uint64_t)fixture->wave.row_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,fixture->stream),label,"output_readback") != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityCuda(cudaStreamSynchronize(fixture->stream),label,"sync") != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityCuda(cudaMemcpy(kv_access,fixture->slot.kv_access_error,
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

static int SparkGlm5NextMtpParityStashHidden(SparkGlm5NextMtpParityFixture *fixture,uint32_t row)
{
	return(SparkGlm5NextMtpParityCuda(cudaMemcpyAsync(fixture->draft_hidden_bf16,
		fixture->slot.hc_mean_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
		cudaMemcpyDeviceToDevice,fixture->stream),"stash","hidden"));
}

static int SparkGlm5NextMtpParityResetPools(SparkGlm5NextMtpParityFixture *fixture)
{
	if ( SparkGlm5NextMtpParityCuda(cudaMemset(fixture->kda_state_pools,0,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES),"reset","kda_state") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemset(fixture->kda_window_pools,0,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES),"reset","kda_windows") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemset(fixture->kv_cache,0,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES),"reset","kv_cache") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemset(fixture->index_cache,0,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES),"reset","index_cache") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaStreamSynchronize(fixture->stream),"reset","sync") != 0 )
		return(1);
	return(0);
}

static uint8_t *SparkGlm5NextMtpParitySnapshotBlock(
	SparkGlm5NextMtpParityFixture *fixture,
	uint32_t snapshot_index)
{
	return(fixture->snapshots +
		(uint64_t)snapshot_index * SPARK_GLM5_NEXT_MTP_PARITY_SNAPSHOT_BLOCK_BYTES);
}

static int SparkGlm5NextMtpParitySnapshot(SparkGlm5NextMtpParityFixture *fixture,uint32_t snapshot_index)
{
	uint8_t *block;
	block = SparkGlm5NextMtpParitySnapshotBlock(fixture,snapshot_index);
	if ( SparkGlm5NextMtpParityCuda(cudaMemcpy(block,fixture->kda_state_pools,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES,cudaMemcpyDeviceToHost),"snapshot","kda_state") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemcpy(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES,fixture->kda_window_pools,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES,cudaMemcpyDeviceToHost),"snapshot","kda_windows") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemcpy(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES +
			SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES,fixture->kv_cache,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES,cudaMemcpyDeviceToHost),"snapshot","kv_cache") != 0 ||
		SparkGlm5NextMtpParityCuda(cudaMemcpy(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES +
			SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES + SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES,fixture->index_cache,
		(size_t)SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES,cudaMemcpyDeviceToHost),"snapshot","index_cache") != 0 )
		return(1);
	return(0);
}

static int SparkGlm5NextMtpParityCompareRegion(
	const uint8_t *expected,
	const void *device,
	uint64_t bytes,
	uint8_t *scratch,
	const char *label,
	const char *region,
	uint32_t snapshot_index)
{
	uint64_t offset;
	if ( SparkGlm5NextMtpParityCuda(cudaMemcpy(scratch,device,(size_t)bytes,cudaMemcpyDeviceToHost),label,region) != 0 )
		return(1);
	for ( offset = 0u; offset < bytes; offset++ )
		if ( scratch[offset] != expected[offset] )
		{
			printf("FAIL %s: %s diverges at snapshot %u byte %llu: spec %02x baseline %02x\n",
				label,region,snapshot_index,(unsigned long long)offset,
				(unsigned)scratch[offset],(unsigned)expected[offset]);
			return(1);
		}
	return(0);
}

static int SparkGlm5NextMtpParityCompareState(
	SparkGlm5NextMtpParityFixture *fixture,
	uint32_t snapshot_index,
	uint8_t *scratch,
	const char *label)
{
	const uint8_t *block;
	uint64_t kv_bytes;
	uint64_t index_bytes;
	block = SparkGlm5NextMtpParitySnapshotBlock(fixture,snapshot_index);
	if ( SparkGlm5NextMtpParityCompareRegion(block,fixture->kda_state_pools,
		SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES,scratch,label,"kda_state",snapshot_index) != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityCompareRegion(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES,
		fixture->kda_window_pools,SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES,scratch,label,"kda_windows",snapshot_index) != 0 )
		return(1);
	kv_bytes = (uint64_t)snapshot_index * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	if ( kv_bytes > SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES )
		kv_bytes = SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES;
	if ( SparkGlm5NextMtpParityCompareRegion(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES +
		SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES,fixture->kv_cache,kv_bytes,scratch,label,"kv_cache",snapshot_index) != 0 )
		return(1);
	index_bytes = (uint64_t)snapshot_index * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	if ( index_bytes > SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES )
		index_bytes = SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES;
	if ( SparkGlm5NextMtpParityCompareRegion(block + SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES +
		SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES + SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES,
		fixture->index_cache,index_bytes,scratch,label,"index_cache",snapshot_index) != 0 )
		return(1);
	return(0);
}

static const uint32_t SPARK_GLM5_NEXT_MTP_PARITY_PROMPT[SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS] =
	{ 11u, 902u, 47u, 1888u, 5u, 203u };

static int SparkGlm5NextMtpParityFixtureBuild(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkGlm5NextKdaReplayLayout replay_layout;
	uint32_t host_page_table[SPARK_GLM5_NEXT_MTP_PARITY_PAGES];
	uint32_t mtp_meta[2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
	uint32_t zero;
	uint32_t index;
	memset(fixture,0,sizeof(*fixture));
	fixture->random_state = 0x5eed1234u;
	if ( SparkGlm5NextMtpParityCuda(cudaStreamCreate(&fixture->stream),"fixture","stream") != 0 )
		return(1);
	replay_layout = SparkGlm5NextKdaReplayLayoutFor(
		SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS);
	fixture->kda_replay_layer_bytes = replay_layout.layer_bytes;
	if ( SparkGlm5NextMtpParityBuildWeights(fixture) != 0 ||
		SparkGlm5NextMtpParityBuildScratch(fixture) != 0 )
		return(1);
	fixture->kda_state_pools = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES,"kda_state_pools");
	fixture->kda_window_pools = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_MTP_PARITY_WINDOW_BYTES,"kda_window_pools");
	fixture->kv_cache = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_MTP_PARITY_KV_BYTES,"kv_cache");
	fixture->index_cache = (uint8_t *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_MTP_PARITY_INDEX_BYTES,"index_cache");
	fixture->page_table = (uint32_t *)SparkGlm5NextMtpParityAlloc(
		SPARK_GLM5_NEXT_MTP_PARITY_PAGES * sizeof(uint32_t),"page_table");
	fixture->kda_state_index_device = (uint32_t *)SparkGlm5NextMtpParityAlloc(
		sizeof(uint32_t),"kda_state_index");
	fixture->draft_hidden_bf16 = (uint16_t *)SparkGlm5NextMtpParityAlloc(
		(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),"draft_hidden");
	if ( fixture->kda_state_pools == 0 || fixture->kda_window_pools == 0 || fixture->kv_cache == 0 ||
		fixture->index_cache == 0 || fixture->page_table == 0 || fixture->kda_state_index_device == 0 ||
		fixture->draft_hidden_bf16 == 0 )
		return(1);
	for ( index = 0u; index < SPARK_GLM5_NEXT_MTP_PARITY_PAGES; index++ )
		host_page_table[index] = index;
	zero = 0u;
	for ( index = 0u; index < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; index++ )
	{
		mtp_meta[index] = index;
		mtp_meta[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + index] = index + 1u;
	}
	if ( SparkGlm5NextMtpParityUpload(fixture->page_table,host_page_table,sizeof(host_page_table),"page_table") != 0 ||
		SparkGlm5NextMtpParityUpload(fixture->kda_state_index_device,&zero,sizeof(zero),"kda_state_index") != 0 ||
		SparkGlm5NextMtpParityUpload(fixture->slot.mtp_positions,mtp_meta,
			SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),"mtp_positions") != 0 ||
		SparkGlm5NextMtpParityUpload(fixture->slot.mtp_context,
			mtp_meta + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
			SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),"mtp_context") != 0 ||
		SparkGlm5NextMtpParityUpload(fixture->slot.mtp_page_table,&zero,sizeof(zero),"mtp_page_table") != 0 ||
		SparkGlm5NextMtpParityUpload(fixture->slot.mtp_sequence,&zero,sizeof(zero),"mtp_sequence") != 0 )
		return(1);
	fixture->slot.kda_state_index = fixture->kda_state_index_device;
	fixture->snapshots = (uint8_t *)malloc(
		(uint64_t)SPARK_GLM5_NEXT_MTP_PARITY_SNAPSHOT_COUNT * SPARK_GLM5_NEXT_MTP_PARITY_SNAPSHOT_BLOCK_BYTES);
	if ( fixture->snapshots == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host","snapshots"));
	SparkGlm5NextMtpParityBindLayers(fixture);
	SparkGlm5NextMtpParityBuildDraftWave(fixture);
	return(0);
}

static int SparkGlm5NextMtpParityRunBaseline(SparkGlm5NextMtpParityFixture *fixture)
{
	uint32_t step;
	if ( SparkGlm5NextMtpParityResetPools(fixture) != 0 )
		return(1);
	SparkGlm5NextMtpParityBuildWave(fixture,SPARK_GLM5_NEXT_MTP_PARITY_PROMPT,
		0u,SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS,1u,0u);
	if ( SparkGlm5NextMtpParityRunWave(fixture,"baseline_prefill") != 0 )
		return(1);
	fixture->reference_tokens[0] = fixture->host_output[SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS - 1u];
	if ( SparkGlm5NextMtpParitySnapshot(fixture,SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS) != 0 )
		return(1);
	for ( step = 0u; step + 1u < SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS; step++ )
	{
		SparkGlm5NextMtpParityBuildWave(fixture,&fixture->reference_tokens[step],
			SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + step,1u,1u,0u);
		if ( SparkGlm5NextMtpParityRunWave(fixture,"baseline_decode") != 0 )
			return(1);
		fixture->reference_tokens[step + 1u] = fixture->host_output[0];
		if ( SparkGlm5NextMtpParitySnapshot(fixture,SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + step + 1u) != 0 )
			return(1);
	}
	printf("baseline: %u reference tokens recorded\n",SPARK_GLM5_NEXT_MTP_PARITY_REF_TOKENS);
	return(0);
}

static const char *SparkGlm5NextMtpParityModeName(uint32_t mode)
{
	switch ( mode )
	{
	case SPARK_GLM5_NEXT_MTP_PARITY_MODE_ORGANIC: return("organic");
	case SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_REJECT: return("force_reject");
	case SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_MID: return("force_mid");
	case SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_ACCEPT: return("force_accept");
	default: return("unknown");
	}
}

static int SparkGlm5NextMtpParityRunSpeculative(SparkGlm5NextMtpParityFixture *fixture)
{
	SparkSpeculationPolicyVerifyResult result;
	uint8_t *scratch;
	uint32_t counts[SPARK_GLM5_NEXT_MTP_PARITY_MODE_COUNT];
	uint32_t draft_tokens[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
	uint32_t verify_tokens[SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS];
	uint32_t emitted;
	uint32_t current;
	uint32_t step;
	uint32_t mode;
	scratch = (uint8_t *)malloc(SPARK_GLM5_NEXT_MTP_PARITY_STATE_BYTES);
	if ( scratch == 0 )
		return(SparkGlm5NextMtpParityFail("alloc_host","compare_scratch"));
	for ( mode = 0u; mode < SPARK_GLM5_NEXT_MTP_PARITY_MODE_COUNT; mode++ )
		counts[mode] = 0u;
	if ( SparkGlm5NextMtpParityResetPools(fixture) != 0 )
		return(1);
	SparkGlm5NextMtpParityBuildWave(fixture,SPARK_GLM5_NEXT_MTP_PARITY_PROMPT,
		0u,SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS,1u,0u);
	if ( SparkGlm5NextMtpParityRunWave(fixture,"spec_prefill") != 0 )
		return(1);
	if ( fixture->host_output[SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS - 1u] != fixture->reference_tokens[0] )
	{
		printf("FAIL spec_prefill: prefill token %u != baseline %u\n",
			fixture->host_output[SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS - 1u],fixture->reference_tokens[0]);
		return(1);
	}
	if ( SparkGlm5NextMtpParityStashHidden(fixture,SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS - 1u) != 0 )
		return(1);
	emitted = 1u;
	current = fixture->reference_tokens[0];
	step = 0u;
	while ( emitted < SPARK_GLM5_NEXT_MTP_PARITY_DECODE_TOKENS )
	{
		SparkStatus status;
		uint32_t expected;
		uint32_t committed;
		uint32_t index;
		mode = step % SPARK_GLM5_NEXT_MTP_PARITY_MODE_COUNT;
		if ( SparkGlm5NextLaunchCudaMtpDraft(&fixture->draft_wave,0,
			fixture->draft_hidden_bf16,current,draft_tokens) != 0 )
			return(SparkGlm5NextMtpParityFail("spec_draft","launch"));
		counts[mode]++;
		expected = UINT32_MAX;
		if ( mode == SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_REJECT )
		{
			draft_tokens[0] = (fixture->reference_tokens[emitted] + 1u) % SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
			expected = 0u;
		}
		else if ( mode == SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_MID )
		{
			draft_tokens[0] = fixture->reference_tokens[emitted];
			draft_tokens[1] = (fixture->reference_tokens[emitted + 1u] + 1u) % SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
			expected = 1u;
		}
		else if ( mode == SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_ACCEPT )
		{
			draft_tokens[0] = fixture->reference_tokens[emitted];
			draft_tokens[1] = fixture->reference_tokens[emitted + 1u];
			expected = 2u;
		}
		verify_tokens[0] = current;
		verify_tokens[1] = draft_tokens[0];
		verify_tokens[2] = draft_tokens[1];
		SparkGlm5NextMtpParityBuildWave(fixture,verify_tokens,
			SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + emitted - 1u,
			SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS,0u,1u);
		if ( SparkGlm5NextMtpParityRunWave(fixture,"spec_verify") != 0 )
			return(1);
		status = SparkSpeculationPolicyResolveVerifierTokens(
			draft_tokens,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
			fixture->host_output,SPARK_GLM5_NEXT_MTP_PARITY_VERIFY_ROWS,
			SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,&result);
		if ( status != SPARK_STATUS_OK )
			return(SparkGlm5NextMtpParityFail("spec_resolve","status"));
		if ( result.committed_token_count != result.accepted_draft_token_count + 1u )
			return(SparkGlm5NextMtpParityFail("spec_resolve","committed != accepted + 1"));
		if ( expected != UINT32_MAX && result.accepted_draft_token_count != expected )
		{
			printf("FAIL spec_verify: %s step %u expected %u accepted, got %u (draft %u %u verifier %u %u %u)\n",
				SparkGlm5NextMtpParityModeName(mode),step,expected,result.accepted_draft_token_count,
				draft_tokens[0],draft_tokens[1],
				fixture->host_output[0],fixture->host_output[1],fixture->host_output[2]);
			return(1);
		}
		committed = result.committed_token_count;
		for ( index = 0u; index < committed; index++ )
			if ( fixture->host_output[index] != fixture->reference_tokens[emitted + index] )
			{
				printf("FAIL spec_verify: token mismatch at emitted position %u: spec %u baseline %u (step %u mode %s)\n",
					emitted + index,fixture->host_output[index],fixture->reference_tokens[emitted + index],
					step,SparkGlm5NextMtpParityModeName(mode));
				return(1);
			}
		if ( SparkGlm5NextMtpParityStashHidden(fixture,committed - 1u) != 0 )
			return(1);
		if ( SparkGlm5NextLaunchCudaMtpCommit(&fixture->wave,committed) != 0 )
			return(SparkGlm5NextMtpParityFail("spec_commit","fold"));
		if ( SparkGlm5NextMtpParityCuda(cudaStreamSynchronize(fixture->stream),"spec_commit","sync") != 0 )
			return(1);
		if ( SparkGlm5NextMtpParityCompareState(fixture,
			SPARK_GLM5_NEXT_MTP_PARITY_PREFILL_ROWS + emitted + committed - 1u,scratch,"spec_state") != 0 )
			return(1);
		current = fixture->host_output[committed - 1u];
		emitted += committed;
		step++;
	}
	free(scratch);
	for ( mode = 0u; mode < SPARK_GLM5_NEXT_MTP_PARITY_MODE_COUNT; mode++ )
		if ( counts[mode] == 0u )
		{
			printf("FAIL spec_verify: mode %s never executed\n",SparkGlm5NextMtpParityModeName(mode));
			return(1);
		}
	printf("spec: %u tokens emitted in %u steps (organic %u reject %u mid %u accept %u)\n",
		emitted,step,counts[SPARK_GLM5_NEXT_MTP_PARITY_MODE_ORGANIC],
		counts[SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_REJECT],
		counts[SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_MID],
		counts[SPARK_GLM5_NEXT_MTP_PARITY_MODE_FORCE_ACCEPT]);
	return(0);
}

int main(int argc,char **argv)
{
	static SparkGlm5NextMtpParityFixture fixture;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("glm5_next MTP parity: configuration %s\n",argv[1]);
	if ( SparkGlm5NextConfigureCudaModule(&fixture.multiprocessor_count) != 0 )
		return(SparkGlm5NextMtpParityFail("configure","sm_121 target required"));
	if ( SparkGlm5NextMtpParityFixtureBuild(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityRunBaseline(&fixture) != 0 )
		return(1);
	if ( SparkGlm5NextMtpParityRunSpeculative(&fixture) != 0 )
		return(1);
	printf("PASS glm5_next MTP parity: %u tokens, spec == baseline byte-exact (tokens, KDA state, conv windows, KV, index)\n",
		SPARK_GLM5_NEXT_MTP_PARITY_DECODE_TOKENS);
	return(0);
}
