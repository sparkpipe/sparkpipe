#ifndef SPARKPIPE_SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_qwen4_flash_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS 8u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS 8u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u

#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 32u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP 4096u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 512u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 3u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 4u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 6u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 7u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED 8u

typedef struct SparkQwen4FlashLinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkQwen4FlashLinearView;

typedef struct SparkQwen4FlashGdnLayerWeights
{
	SparkQwen4FlashLinearView qkv;
	SparkQwen4FlashLinearView gate;
	SparkQwen4FlashLinearView beta;
	SparkQwen4FlashLinearView decay;
	SparkQwen4FlashLinearView output;
	const void *conv_weight_bf16;
	const float *a_log_f32;
	const float *dt_bias_f32;
	const void *gdn_norm_weight_bf16;
} SparkQwen4FlashGdnLayerWeights;

typedef struct SparkQwen4FlashAttnLayerWeights
{
	SparkQwen4FlashLinearView query;
	SparkQwen4FlashLinearView key;
	SparkQwen4FlashLinearView value;
	SparkQwen4FlashLinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkQwen4FlashAttnLayerWeights;

typedef struct SparkQwen4FlashMoeWeights
{
	SparkQwen4FlashLinearView gate;
	SparkQwen4FlashLinearView experts_w1;
	SparkQwen4FlashLinearView experts_w3;
	SparkQwen4FlashLinearView experts_w2;
	SparkQwen4FlashLinearView shared_gate;
	SparkQwen4FlashLinearView shared_up;
	SparkQwen4FlashLinearView shared_down;
	const void *shared_gate_weight_bf16;
} SparkQwen4FlashMoeWeights;

typedef struct SparkQwen4FlashHcWeights
{
	const void *hc_norm_weight_bf16;
	SparkQwen4FlashLinearView mix_down;
	SparkQwen4FlashLinearView mix_up;
	SparkQwen4FlashLinearView block_inject;
} SparkQwen4FlashHcWeights;

typedef struct SparkQwen4FlashHcMixer
{
	const void *hc_norm_weight_bf16;
	SparkQwen4FlashLinearView mix_down;
	SparkQwen4FlashLinearView mix_up;
} SparkQwen4FlashHcMixer;

typedef struct SparkQwen4FlashIndexerWeights
{
	SparkQwen4FlashLinearView index_qk;
	const void *q_norm_weight_bf16;
	const void *k_norm_weight_bf16;
} SparkQwen4FlashIndexerWeights;

typedef struct SparkQwen4FlashPleWeights
{
	SparkQwen4FlashLinearView key_proj;
	SparkQwen4FlashLinearView value_proj;
	const void *norm_key_weight_bf16;
	const void *norm_query_weight_bf16;
	const void *norm_conv_weight_bf16;
	const void *conv_weight_bf16;
	const int64_t *layer_multipliers;
	const int64_t *head_vocab_sizes;
	const int64_t *head_offsets;
	const void *ngram_embedding_bf16;
} SparkQwen4FlashPleWeights;

typedef struct SparkQwen4FlashMtpWeights
{
	SparkQwen4FlashLinearView fc;
	const void *embed_norm_weight_bf16;
	const void *hidden_norm_weight_bf16;
	SparkQwen4FlashHcMixer readout_mixer;
	SparkQwen4FlashHcWeights attention_hc;
	SparkQwen4FlashHcWeights mlp_hc;
	SparkQwen4FlashIndexerWeights indexer;
	SparkQwen4FlashAttnLayerWeights attention;
	SparkQwen4FlashMoeWeights moe;
} SparkQwen4FlashMtpWeights;

typedef struct SparkQwen4FlashGdnStatePool
{
	uint32_t abi_version;
	uint32_t lane_capacity;
	uint32_t gdn_layer_count;
	uint32_t reserved0;
	float *state_f32;
	uint64_t state_lane_stride_elements;
	uint64_t state_layer_stride_elements;
	void *conv_tail_bf16;
	uint64_t conv_tail_lane_stride_elements;
	uint64_t conv_tail_layer_stride_elements;
	uint32_t *state_cold_by_row;
} SparkQwen4FlashGdnStatePool;

typedef struct SparkQwen4FlashKvBlockTableView
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
} SparkQwen4FlashKvBlockTableView;

typedef struct SparkQwen4FlashPipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *slot_mapping;
	const uint32_t *context_lengths;
	void *hidden_input_bf16;
	void *hidden_bf16;
	void *normalized_bf16;
	void *attn_query_bf16;
	void *attn_key_bf16;
	void *attn_value_bf16;
	void *attn_gate_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *gdn_conv_workspace_bf16;
	void *gdn_query_bf16;
	void *gdn_key_bf16;
	void *gdn_value_bf16;
	void *gdn_gate_bf16;
	void *gdn_ba_bf16;
	void *gdn_log_decay_f32;
	void *gdn_beta_f32;
	void *gdn_core_output_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	void *moe_indices_u32;
	float *moe_weights_f32;
	uint32_t *moe_inverse_u32;
	uint32_t *moe_grouped_rows_u32;
	uint32_t *moe_tile_prefix_w1_u32;
	uint32_t *moe_tile_prefix_w2_u32;
	void *argmax_score_f32;
	void *argmax_token_ids;
	float *chunk_qn_f32;
	float *chunk_kn_f32;
	float *chunk_cum_g_f32;
	float *chunk_decay_f32;
	float *chunk_attn_f32;
	float *chunk_w_f32;
	float *chunk_kg_f32;
	uint32_t *mtp_draft_token_ids;
} SparkQwen4FlashPipelineSlot;

typedef struct SparkQwen4FlashResidentDecodeStageNodeContext
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
	const void *attention_norm_weights_by_layer_bf16[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_weights_by_layer_bf16[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const SparkQwen4FlashGdnLayerWeights *gdn_weights_by_layer;
	const SparkQwen4FlashAttnLayerWeights *attn_weights_by_layer;
	const SparkQwen4FlashMoeWeights *moe_weights_by_layer;
	SparkQwen4FlashGdnStatePool gdn_state_pool;
	void *kv_cache_bf16;
	const SparkQwen4FlashPipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkQwen4FlashResidentDecodeStageNodeContext;

typedef struct SparkQwen4FlashDecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkQwen4FlashDecodeBatchView;

typedef struct SparkQwen4FlashPrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
} SparkQwen4FlashPrefillFrameView;

typedef struct SparkQwen4FlashMtpDraftView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t draft_token_count;
	uint64_t base_position;
	uint64_t sequence_id;
	const uint32_t *row_token_ids;
} SparkQwen4FlashMtpDraftView;

typedef struct SparkQwen4FlashGdnSnapshotView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t snapshot_index;
	uint32_t reserved0;
} SparkQwen4FlashGdnSnapshotView;

#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER 0x00000020u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY 0x00000040u
#define SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST 0x00000080u

typedef SparkStatus (*SparkQwen4FlashHiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkQwen4FlashHiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SparkQwen4FlashResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkQwen4FlashKvBlockTableView *kv_block_table;
	const SparkQwen4FlashDecodeBatchView *decode_batch;
	const SparkQwen4FlashPrefillFrameView *prefill_frame;
	const SparkQwen4FlashMtpDraftView *mtp_draft;
	const SparkQwen4FlashGdnSnapshotView *gdn_snapshot;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkQwen4FlashHiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkQwen4FlashHiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkQwen4FlashResidentDecodeStageFrameContext;

SparkStatus SparkQwen4FlashResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkQwen4FlashResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkQwen4FlashResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkQwen4FlashResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkQwen4FlashResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
