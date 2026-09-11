#ifndef SPARKPIPE_SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_muse_glimmer_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION 1u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u

#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 32u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 512u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u

typedef struct SparkMuseGlimmerLinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkMuseGlimmerLinearView;

typedef struct SparkMuseGlimmerLayerWeights
{
	SparkMuseGlimmerLinearView qgkv;
	SparkMuseGlimmerLinearView output;
	SparkMuseGlimmerLinearView gate_up;
	SparkMuseGlimmerLinearView down;
	const void *input_norm_weight_bf16;
	const void *post_attention_norm_weight_bf16;
	const void *pre_feedforward_norm_weight_bf16;
	const void *post_feedforward_norm_weight_bf16;
} SparkMuseGlimmerLayerWeights;

typedef struct SparkMuseGlimmerKvBlockTableView
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
} SparkMuseGlimmerKvBlockTableView;

typedef struct SparkMuseGlimmerPipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *row_positions_u32;
	const uint32_t *context_lengths;
	const uint32_t *window_positions;
	void *hidden_input_bf16;
	void *hidden_bf16;
	void *residual_bf16;
	void *normalized_bf16;
	void *fused_qgkv_bf16;
	void *query_gate_bf16;
	void *query_bf16;
	void *attn_gate_bf16;
	void *key_bf16;
	void *value_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *gate_up_bf16;
	void *intermediate_bf16;
	void *delta_bf16;
	void *argmax_score_f32;
	void *argmax_token_ids;
} SparkMuseGlimmerPipelineSlot;

typedef struct SparkMuseGlimmerResidentDecodeStageNodeContext
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
	const void *lm_head_weight_bf16;
	const SparkMuseGlimmerLayerWeights *weights_by_layer;
	void *kv_cache_bf16;
	const SparkMuseGlimmerPipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkMuseGlimmerResidentDecodeStageNodeContext;

typedef struct SparkMuseGlimmerDecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkMuseGlimmerDecodeBatchView;

typedef struct SparkMuseGlimmerPrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
} SparkMuseGlimmerPrefillFrameView;

#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u
#define SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY 0x00000040u

typedef SparkStatus (*SparkMuseGlimmerHiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkMuseGlimmerHiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SparkMuseGlimmerResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkMuseGlimmerKvBlockTableView *kv_block_table;
	const SparkMuseGlimmerDecodeBatchView *decode_batch;
	const SparkMuseGlimmerPrefillFrameView *prefill_frame;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkMuseGlimmerHiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkMuseGlimmerHiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkMuseGlimmerResidentDecodeStageFrameContext;

SparkStatus SparkMuseGlimmerResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkMuseGlimmerResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkMuseGlimmerResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkMuseGlimmerResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkMuseGlimmerResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
