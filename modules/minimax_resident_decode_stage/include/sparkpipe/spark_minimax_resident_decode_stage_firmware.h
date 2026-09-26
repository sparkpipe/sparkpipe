#ifndef SPARKPIPE_SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "llm_defines.h"
#include "sparkpipe/spark_minimax_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_stagepack_format.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION SPARK_LLM_DECODE_BATCH_VIEW_ABI_VERSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION SPARK_LLM_PREFILL_FRAME_VIEW_ABI_VERSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION SPARK_LLM_KV_BLOCK_TABLE_ABI_VERSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION SPARK_LLM_LINEAR_VIEW_ABI_VERSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT SPARK_LLM_MAX_PIPELINE_SLOT_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS SPARK_LLM_KV_BLOCK_TOKENS
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID SPARK_LLM_INVALID_TOKEN_ID
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_NO_BLOCK SPARK_LLM_NO_BLOCK

#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_MINIMAX_TEXT_LAYER_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_COUNT SPARK_MINIMAX_TEXT_ATTENTION_HEAD_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT SPARK_MINIMAX_TEXT_KV_HEAD_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION SPARK_MINIMAX_TEXT_HEAD_DIMENSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_QUERY_DIMENSION SPARK_MINIMAX_TEXT_QUERY_DIMENSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_DIMENSION SPARK_MINIMAX_TEXT_KV_DIMENSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FFN_INTERMEDIATE_DIMENSION SPARK_MINIMAX_TEXT_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT SPARK_MINIMAX_TEXT_VOCAB_COUNT
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAXIMUM_CONTEXT_TOKENS SPARK_MINIMAX_TEXT_MAXIMUM_CONTEXT_TOKENS
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON SPARK_MINIMAX_TEXT_RMS_NORM_EPSILON
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_END_OF_TEXT_TOKEN_ID SPARK_MINIMAX_TEXT_END_OF_TEXT_TOKEN_ID
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES SPARK_MINIMAX_TEXT_HIDDEN_BF16_BYTES
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_BF16_ELEMENT_BYTES SPARK_MINIMAX_TEXT_BF16_ELEMENT_BYTES

#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 SPARK_STAGEPACK_FORMAT_WEIGHT_BF16

/* nvcc compiles this header inside the C++17 .cu translation unit; the
 * shared shim convention is spark_driver_defines.h SPARK_LLM_STATIC_ASSERT
 * (self-contained here because this header does not chain it). */
#if defined(__cplusplus)
#define SPARK_MINIMAX_STATIC_ASSERT(condition,message) static_assert(condition,message)
#else
#define SPARK_MINIMAX_STATIC_ASSERT(condition,message) _Static_assert(condition,message)
#endif

SPARK_MINIMAX_STATIC_ASSERT(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == 0u,"minimax packs carry bf16 weight code zero");
SPARK_MINIMAX_STATIC_ASSERT(SPARK_LLM_TIED_WORD_EMBEDDINGS == 0u,"minimax serves the untied language modeling head");

typedef struct SparkMinimaxLinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	uint32_t row_base;
	const void *weight_payload_bf16;
	uint64_t weight_payload_bytes;
} SparkMinimaxLinearView;

typedef struct SparkMinimaxAttentionLayerWeights
{
	SparkMinimaxLinearView query;
	SparkMinimaxLinearView key;
	SparkMinimaxLinearView value;
	SparkMinimaxLinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkMinimaxAttentionLayerWeights;

typedef struct SparkMinimaxMlpLayerWeights
{
	SparkMinimaxLinearView gate;
	SparkMinimaxLinearView up;
	SparkMinimaxLinearView down;
} SparkMinimaxMlpLayerWeights;

#define SPARK_ABI_TYPE(name) SparkMinimax##name
#include "sparkpipe/family/abi/spark_abi_kv_block_table_view.h"

typedef struct SparkMinimaxPipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *slot_mapping;
	const uint32_t *context_lengths;
	const uint64_t *row_positions;
	void *hidden_bf16;
	void *normalized_bf16;
	void *query_bf16;
	void *key_bf16;
	void *value_bf16;
	void *query_roped_bf16;
	void *attended_bf16;
	void *attn_output_bf16;
	void *mlp_gate_bf16;
	void *mlp_up_bf16;
	void *mlp_down_bf16;
	void *argmax_reduce_u64;
	float *rope_angle_f32;
} SparkMinimaxPipelineSlot;

typedef struct SparkMinimaxResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t kv_cache_block_count;
	uint32_t enable_cuda_graph_replay;
	float rms_norm_epsilon;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t local_query_head_count;
	uint32_t local_kv_head_count;
	uint32_t local_kv_dimension;
	uint32_t local_ffn_dimension;
	uint32_t local_vocab_rows;
	uint32_t local_vocab_base;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	const SparkMinimaxAttentionLayerWeights *attention_weights_by_layer;
	const SparkMinimaxMlpLayerWeights *mlp_weights_by_layer;
	void *kv_cache_bf16;
	uint64_t kv_cache_layer_stride;
	uint64_t kv_cache_block_stride;
	const SparkMinimaxPipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkMinimaxResidentDecodeStageNodeContext;

#include "sparkpipe/family/abi/spark_abi_decode_batch_view.h"

typedef struct SparkMinimaxPrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
	const uint32_t *row_token_ids;
} SparkMinimaxPrefillFrameView;

#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u

#include "sparkpipe/family/abi/spark_abi_frame_context.h"
#undef SPARK_ABI_TYPE

SparkStatus SparkMinimaxResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration,const SparkFirmwareModuleHostServices *host_services,void **module_state);
SparkStatus SparkMinimaxResidentDecodeStageExecute(void *module_state,SparkModelDriverFrame *frame);
SparkStatus SparkMinimaxResidentDecodeStageAdmit(void *module_state,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkMinimaxResidentDecodeStageSnapshot(void *module_state,uint32_t program_id,SparkModelDriverRuntimeSnapshot *snapshot);
void SparkMinimaxResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
