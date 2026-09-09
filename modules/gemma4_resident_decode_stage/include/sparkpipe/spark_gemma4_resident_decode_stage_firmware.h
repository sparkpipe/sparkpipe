#ifndef SPARKPIPE_SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_gemma4_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 32u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 512u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 7u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED 8u

typedef struct SparkGemma4LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkGemma4LinearView;

typedef struct SparkGemma4SlidingLayerWeights
{
	SparkGemma4LinearView query;
	SparkGemma4LinearView kv_fused;
	SparkGemma4LinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkGemma4SlidingLayerWeights;

typedef struct SparkGemma4FullLayerWeights
{
	SparkGemma4LinearView query;
	SparkGemma4LinearView key;
	SparkGemma4LinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkGemma4FullLayerWeights;

typedef struct SparkGemma4DenseMlpWeights
{
	SparkGemma4LinearView gate_up;
	SparkGemma4LinearView down;
} SparkGemma4DenseMlpWeights;

#if SPARK_GEMMA4_MODEL_MOE_BLOCK
typedef struct SparkGemma4MoeLayerWeights
{
	SparkGemma4LinearView router_proj;
	const float *per_expert_scale_f32;
	SparkGemma4LinearView experts_gate_up;
	SparkGemma4LinearView experts_down;
} SparkGemma4MoeLayerWeights;
#endif

typedef struct SparkGemma4KvBlockTableView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t lane_count;
	uint32_t lane_stride;
	uint32_t lane_capacity;
	const uint32_t *physical_block_indices;
	const uint32_t *lane_physical_block_counts;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_physical_block_counts;
} SparkGemma4KvBlockTableView;

typedef struct SparkGemma4PipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *slot_mapping;
	const uint32_t *context_lengths;
	void *hidden_input_bf16;
	void *hidden_bf16;
	void *residual_bf16;
	void *normalized_bf16;
	void *sliding_query_bf16;
	void *sliding_kv_bf16;
	void *full_query_bf16;
	void *full_key_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *mlp_gate_up_bf16;
	void *mlp_down_bf16;
	void *branch_bf16;
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	uint32_t *moe_inverse_u32;
	uint32_t *moe_grouped_rows_u32;
	uint32_t *moe_tile_prefix_w1_u32;
	uint32_t *moe_tile_prefix_w2_u32;
#endif
	void *argmax_score_f32;
	void *argmax_token_ids;
} SparkGemma4PipelineSlot;

typedef struct SparkGemma4ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t max_active_sequence_count;
	uint32_t max_prefill_tokens;
	uint32_t pipeline_slot_count;
	uint32_t kv_cache_block_count;
	uint32_t enable_cuda_graph_replay;
	float rms_norm_epsilon;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *full_rope_table_f32;
	const void *layer_input_norms_by_layer_bf16[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_post_attention_norms_by_layer_bf16[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_pre_feedforward_norms_by_layer_bf16[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_post_feedforward_norms_by_layer_bf16[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const SparkGemma4SlidingLayerWeights *sliding_weights_by_layer;
	const SparkGemma4FullLayerWeights *full_weights_by_layer;
	const SparkGemma4DenseMlpWeights *mlp_weights_by_layer;
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	const SparkGemma4MoeLayerWeights *moe_weights_by_layer;
#endif
	void *sliding_kv_cache_bf16;
	void *full_kv_cache_bf16;
	uint64_t sliding_kv_layer_stride;
	uint64_t full_kv_layer_stride;
	const SparkGemma4PipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkGemma4ResidentDecodeStageNodeContext;

typedef struct SparkGemma4DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
	const uint32_t *row_token_ids;
	const uint32_t *context_lengths;
} SparkGemma4DecodeBatchView;

typedef struct SparkGemma4PrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
} SparkGemma4PrefillFrameView;

#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u

typedef SparkStatus (*SparkGemma4HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkGemma4HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SparkGemma4ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkGemma4KvBlockTableView *kv_block_table;
	const SparkGemma4DecodeBatchView *decode_batch;
	const SparkGemma4PrefillFrameView *prefill_frame;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkGemma4HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkGemma4HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkGemma4ResidentDecodeStageFrameContext;

SparkStatus SparkGemma4ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkGemma4ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkGemma4ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkGemma4ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkGemma4ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
