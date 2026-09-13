#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmGdnStageLinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} LmGdnStageLinearView;

cudaError_t LmGdnStageConfigureKernels(void);
cudaError_t LmGdnStageLaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
cudaError_t LmGdnStageLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
cudaError_t LmGdnStageLaunchLinear(cudaStream_t stream, const LmGdnStageLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
cudaError_t LmGdnStageLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride);
cudaError_t LmGdnStageLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const float *a_log_f32, const float *dt_bias_f32, float *log_decay_f32, float *beta_f32, uint32_t row_count);
cudaError_t LmGdnStageLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *state_f32, void *core_out_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride);
cudaError_t LmGdnStageLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const void *norm_weight_bf16, void *output_bf16, uint32_t row_count, float epsilon);
cudaError_t LmGdnStageLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const void *query_norm_weight_bf16, const void *key_norm_weight_bf16, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank);
cudaError_t LmGdnStageLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const uint32_t *block_indices, const uint32_t *block_counts, uint32_t lane_stride, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank);
cudaError_t LmGdnStageLaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride);
cudaError_t LmGdnStageLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, float *state_f32, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride);
cudaError_t LmGdnStageLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
cudaError_t LmGdnStageLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
cudaError_t LmGdnStageLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
cudaError_t LmGdnStageLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
cudaError_t LmGdnStageLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
cudaError_t LmGdnStageLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
cudaError_t LmGdnStageLaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
cudaError_t LmGdnStageLaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
cudaError_t LmGdnStageLaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension);
cudaError_t LmGdnStageLaunchGateScores(cudaStream_t stream, const LmGdnStageLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
cudaError_t LmGdnStageLaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
cudaError_t LmGdnStageLaunchFusedExpertW13Act(cudaStream_t stream, const LmGdnStageLinearView *w1, const LmGdnStageLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count);
cudaError_t LmGdnStageLaunchExpertDown(cudaStream_t stream, const LmGdnStageLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count);
cudaError_t LmGdnStageLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension);
cudaError_t LmGdnStageLaunchGroupedExpertLinear(cudaStream_t stream, const LmGdnStageLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank);
cudaError_t LmGdnStageLaunchGroupedExpertTileLinear(cudaStream_t stream, const LmGdnStageLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, void *output_bf16, uint32_t source_row_count, uint32_t tp_degree, uint32_t tp_rank);

#ifdef __cplusplus
}
#endif
