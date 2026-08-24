/* Frozen reference revision, glm52 stage-pack contract (cleanup round
 * 2026-08-23): the last blessed state before the standalone
 * spark_glm52_stagepack_format.h was folded into the family internal
 * header. Verbatim concatenation of the then-current internal header and
 * the complete former format header, under the single include name the
 * parity driver now uses; "runtime/" resolves to the frozen mechanics
 * beside this file, and <cuda_runtime.h> resolves through the same
 * tests/cuda_stub include path every module gate uses. Any drift in
 * accepted/received/rejected streams trips tests/test_stagepack_parity.py. */
#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

typedef struct SparkGlm52LayerWeights
{
	const void *attn_norm_bf16;
	const void *q_a_bf16;
	const void *q_a_norm_bf16;
	const void *q_b_bf16;
	const void *kv_a_bf16;
	const void *kv_a_norm_bf16;
	const void *kv_b_key_transposed_bf16;
	const void *kv_b_value_bf16;
	const void *attn_output_bf16;
	const void *post_attn_norm_bf16;
	const void *index_q_bf16;
	const void *index_k_bf16;
	const void *index_head_bf16;
	const void *index_norm_weight_bf16;
	const void *index_norm_bias_bf16;
	const void *dense_gate_up_bf16;
	const void *dense_down_bf16;
	const void *router_bf16;
	const float *router_correction_f32;
	const void *expert_up_gate_payload;
	const void *expert_up_gate_scale;
	const void *expert_down_payload;
	const void *expert_down_scale;
	const void *shared_gate_up_bf16;
	const void *shared_down_bf16;
} SparkGlm52LayerWeights;

typedef struct SparkGlm52ExecutionSlot
{
	void *stream;
	void *host_staging;
	uint32_t *host_token_ids;
	uint32_t *host_resident_slots;
	uint32_t *host_positions;
	uint32_t *host_output_token_ids;
	uint32_t *host_kv_access_error;
	uint32_t *token_ids;
	uint32_t *resident_slots;
	uint32_t *positions;
	uint32_t *context_lengths;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *q_compressed_bf16;
	uint16_t *q_bf16;
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *index_query_bf16;
	uint16_t *index_key_bf16;
	uint16_t *index_head_weight_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_value_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits_f32;
	float *selection_scores_f32;
	uint32_t *selected_positions;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
	uint64_t *head_maxloc_u64;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	void *kv_access_error;
	/* DFlash2 drafter tap staging: one arena row id per execution row,
	 * uploaded at wave build and consumed by SparkGlm52LaunchDsparkTapStore
	 * at each aux capture layer. */
	uint32_t *dspark_tap_row_indices;
} SparkGlm52ExecutionSlot;

typedef struct SparkGlm52CudaWave
{
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t row_count;
	uint32_t maximum_context;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t pages_per_sequence;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t sideband_input;
	uint32_t sideband_output;
	uint64_t boundary_row_offset;
	uint64_t sideband_row_offset;
	const uint32_t *host_token_ids;
	const uint32_t *host_resident_slots;
	const uint32_t *host_positions;
	const void *hidden_input_bf16;
	void *hidden_output_bf16;
	const void *sideband_input_u32;
	void *sideband_output_u32;
	uint32_t *host_output_token_ids;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	const SparkGlm52LayerWeights *layers;
	SparkGlm52ExecutionSlot *slot;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	const uint32_t *index_ordinal_by_local_layer;
	const uint32_t *page_table;
	uint32_t multiprocessor_count;
} SparkGlm52CudaWave;

#ifdef __cplusplus
extern "C" {
#endif

int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave);
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave);
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer);
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer);
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave);
cudaError_t SparkGlm52LaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset);
cudaError_t SparkGlm52LaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count);
/* Shared stage-module accumulate pair (runtime/stage_module_kernels.cuh);
 * the private glm52 bodies were deleted by the naming/audit round. */
#include "runtime/stage_module_kernels.h"
/* DFlash2 drafter support: copy one aux-capture layer's hidden rows into the
 * drafter's device tap arena. tap_row_indices holds one arena row index per
 * wave row; the layer's vector lands at arena_base +
 * tap_row*row_stride_elements + tap_index*hidden_dimension. */
cudaError_t SparkGlm52LaunchDsparkTapStore(cudaStream_t stream,const void *hidden_bf16,const uint32_t *tap_row_indices,uint32_t tap_index,uint32_t row_count,uint32_t hidden_dimension,uint16_t *arena_base,uint64_t arena_row_stride_elements);
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count);

#ifdef __cplusplus
}
#endif


#include <stdint.h>
#include <string.h>

#include "runtime/spark_stagepack_reader.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_GLM52_STAGEPACK_MAGIC UINT32_C(0x32534c47)
#define SPARK_GLM52_STAGEPACK_FORMAT_VERSION 3u
#define SPARK_GLM52_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES 256u
#define SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES 65u
#define SPARK_GLM52_STAGEPACK_SHA256_BYTES 32u
#define SPARK_GLM52_STAGEPACK_FLAG_MTP UINT32_C(0x00000001)
#define SPARK_GLM52_STAGEPACK_KNOWN_FLAGS SPARK_GLM52_STAGEPACK_FLAG_MTP

typedef enum SparkGlm52StagePackPayloadType
{
	SPARK_GLM52_STAGEPACK_PAYLOAD_BF16 = SPARK_STAGE_PACK_PAYLOAD_BF16,
	SPARK_GLM52_STAGEPACK_PAYLOAD_F32 = SPARK_STAGE_PACK_PAYLOAD_F32,
	SPARK_GLM52_STAGEPACK_PAYLOAD_U32 = SPARK_STAGE_PACK_PAYLOAD_U32,
	SPARK_GLM52_STAGEPACK_PAYLOAD_PACKED_WEIGHT = SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT
} SparkGlm52StagePackPayloadType;

typedef enum SparkGlm52StagePackTensorKind
{
	SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM = 3,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_A = 4,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM = 5,
	SPARK_GLM52_STAGEPACK_TENSOR_Q_B = 6,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_A = 7,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM = 8,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED = 9,
	SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE = 10,
	SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT = 11,
	SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM = 12,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q = 13,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K = 14,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD = 15,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT = 16,
	SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS = 17,
	SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP = 18,
	SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN = 19,
	SPARK_GLM52_STAGEPACK_TENSOR_ROUTER = 20,
	SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION = 21,
	SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE = 22,
	SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN = 23,
	SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP = 24,
	SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN = 25,
	SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT = 26
} SparkGlm52StagePackTensorKind;

typedef struct SparkGlm52StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t codec_abi_version;
	uint32_t flags;
	uint32_t tensor_count;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t total_layer_count;
	uint32_t hidden_dimension;
	uint32_t vocab_count;
	uint32_t routed_expert_count;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t kv_cache_codec;
	uint32_t reserved0;
	uint32_t reserved1;
	uint64_t directory_offset;
	uint64_t file_bytes;
	char model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES];
	uint8_t contract_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint8_t source_config_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint8_t pack_recipe_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
} SparkGlm52StagePackHeader;

typedef struct SparkGlm52StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t payload_type;
	uint32_t weight_codec;
	uint32_t scale_encoding;
	uint32_t group_count;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkGlm52StagePackEntry;

/* The normalized reader shape, under this family's historical name. */
typedef SparkStagePackShape SparkGlm52StagePackTensorShape;

#define SPARK_GLM52_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkGlm52StagePackHeader))
#define SPARK_GLM52_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkGlm52StagePackEntry))

static inline uint32_t SparkGlm52StagePackLayerIsDense(uint32_t layer_index)
{
	return(layer_index < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackLayerHasFullIndexer(uint32_t layer_index)
{
	return(SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(layer_index));
}

static inline uint32_t SparkGlm52StagePackKindIsGlobal(uint32_t tensor_kind)
{
	return(tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsIndexer(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q && tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsDense(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP || tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN ? 1u : 0u);
}

static inline uint32_t SparkGlm52StagePackKindIsRouted(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_ROUTER && tensor_kind <= SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN ? 1u : 0u);
}

/* TP sharding policy: which kinds are row- or column-sharded across ranks.
 * Must match tools/glm52_resident_stagepack.py exactly; the mechanics live
 * in the shared reader (SparkStagePackApplyShard). */
static inline uint32_t SparkGlm52StagePackTpShardPolicy(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD:
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B:
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE:
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP:
		return(SPARK_STAGE_PACK_SHARD_ROWS);
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT:
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN:
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN:
		return(SPARK_STAGE_PACK_SHARD_COLUMNS);
	default:
		return(SPARK_STAGE_PACK_SHARD_NONE);
	}
}

/* Pack header TP identity: reserved0 = tp_degree, reserved1 = tp_rank. */
static inline uint32_t SparkGlm52StagePackHeaderTpDegree(const SparkGlm52StagePackHeader *header)
{
	return(header != 0 ? header->reserved0 : 0u);
}

static inline uint32_t SparkGlm52StagePackHeaderTpRank(const SparkGlm52StagePackHeader *header)
{
	return(header != 0 ? header->reserved1 : 0u);
}

static inline int32_t SparkGlm52StagePackExpectedShape(uint32_t tensor_kind,uint32_t layer_index,uint32_t expert_codec,uint32_t tp_degree,SparkGlm52StagePackTensorShape *shape)
{
	uint32_t global;
	if ( shape == 0 || tensor_kind >= SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT || tp_degree == 0u )
		return(-1);
	memset(shape,0,sizeof(*shape));
	global = SparkGlm52StagePackKindIsGlobal(tensor_kind);
	if ( (global != 0u && layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER) || (global == 0u && layer_index >= SPARK_GLM52_MODEL_LAYER_COUNT) )
		return(-2);
	if ( SparkGlm52StagePackKindIsIndexer(tensor_kind) != 0u && SparkGlm52StagePackLayerHasFullIndexer(layer_index) == 0u )
		return(-3);
	if ( SparkGlm52StagePackKindIsDense(tensor_kind) != SparkGlm52StagePackLayerIsDense(layer_index) && (SparkGlm52StagePackKindIsDense(tensor_kind) != 0u || SparkGlm52StagePackKindIsRouted(tensor_kind) != 0u) )
		return(-4);
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM:
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_QUERY_A_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_QK_HEAD_DIMENSION,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_LATENT_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: SparkStagePackShapeBf16(shape,SPARK_GLM52_MODEL_HEAD_COUNT,SPARK_GLM52_MODEL_LATENT_DIMENSION,SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: SparkStagePackShapeBf16(shape,SPARK_GLM52_MODEL_HEAD_COUNT,SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION,SPARK_GLM52_MODEL_LATENT_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION,SPARK_GLM52_MODEL_QUERY_A_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT:
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: SparkStagePackShapeBf16(shape,1u,1u,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: SparkStagePackShapeBf16(shape,1u,2u * SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: SparkStagePackShapeWords(shape,SPARK_GLM52_STAGEPACK_PAYLOAD_F32,1u,1u,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE:
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN:
		if ( expert_codec < SPARK_WEIGHT_CODEC_INT6 || expert_codec > SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
			return(-5);
		shape->payload_type = SPARK_GLM52_STAGEPACK_PAYLOAD_PACKED_WEIGHT;
		shape->weight_codec = expert_codec;
		shape->scale_encoding = SparkWeightCodecScaleEncoding(expert_codec);
		shape->group_count = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
		shape->rows = tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE ? 2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION : SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
		shape->columns = tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE ? SPARK_GLM52_MODEL_HIDDEN_DIMENSION : SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
		break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: SparkStagePackShapeBf16(shape,1u,2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION); break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: SparkStagePackShapeBf16(shape,1u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION); break;
	default: return(-6);
	}
	if ( SparkStagePackApplyShard(shape,SparkGlm52StagePackTpShardPolicy(tensor_kind),tp_degree) != 0 )
		return(-7);
	return(0);
}

