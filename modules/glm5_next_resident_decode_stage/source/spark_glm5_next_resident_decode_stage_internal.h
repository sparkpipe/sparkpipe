#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_status.h"

typedef struct SparkGlm5NextLayerWeights
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
	/* KDA (pack-V2 fusions; conv/bias/log-scale arrive rank-sliced). */
	const void *kda_qkv_beta_bf16;
	const void *kda_decay_gate_down_bf16;
	const void *kda_decay_up_bf16;
	const void *kda_gate_up_bf16;
	const void *kda_q_conv_bf16;
	const void *kda_k_conv_bf16;
	const void *kda_v_conv_bf16;
	const float *kda_decay_bias_f32;
	const float *kda_head_log_scale_f32;
	const void *kda_out_norm_bf16;
	const void *kda_out_bf16;
	/* Hyper-connections (fn/base/scale stored F32). */
	const void *hc_attn_fn_f32;
	const void *hc_attn_base_f32;
	const void *hc_attn_scale_f32;
	const void *hc_ffn_fn_f32;
	const void *hc_ffn_base_f32;
	const void *hc_ffn_scale_f32;
	/* Indexer kpool compressor. */
	const void *index_compress_ape_f32;
	const void *index_compress_gate_bf16;
} SparkGlm5NextLayerWeights;

typedef struct SparkGlm5NextExecutionSlot
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
	/* Multi-row runs (chunked prefill): consecutive wave rows of one
	 * resident slot form a single sequential run through the KDA
	 * recurrence. Run structure is staged per wave like the row arrays;
	 * run_count+1 begin entries, run_count slot-keyed state indices. */
	uint32_t *host_run_begin;
	uint32_t *host_run_state_index;
	uint32_t *run_begin;
	uint32_t *run_state_index;
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *q_compressed_bf16;
	uint16_t *q_bf16;
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *index_query_bf16;
	uint16_t *index_key_bf16;
	uint16_t *index_gate_bf16;
	uint16_t *index_packed_bf16;
	uint32_t *selected_pools;
	uint16_t *index_head_weight_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_value_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	/* KDA per-layer state and scratch. */
	uint8_t *kda_state_pool;
	const uint32_t *kda_state_index;
	uint16_t *kda_qkv_window_pool;
	uint16_t *fused_qkvb_bf16;
	uint16_t *fused_decay_gate_bf16;
	uint16_t *kda_decay_latent_bf16;
	uint16_t *kda_gate_latent_bf16;
	uint16_t *kda_beta_logit;
	uint16_t *kda_gate_bf16;
	uint16_t *kda_decay_logit_bf16;
	uint16_t *kda_output_bf16;
	float *kda_retention;
	float *kda_write_gate;
	/* Hyper-connection scratch. */
	float *hc_mixes_f32;
	float *hc_pre_f32;
	float *hc_post_f32;
	float *hc_comb_f32;
	uint16_t *hc_collapsed_bf16;
	uint16_t *hc_snapshot_bf16;
	uint16_t *hc_mean_bf16;
	float *router_logits_f32;
	float *selection_scores_f32;
	float *attention_split_partials_f32;
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
	/* R1 screened-head scratch (per head-owning rank shard size). */
	void *head_certified_scratch;
	uint32_t *head_certified_candidates;
	uint32_t *head_screened_count;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	void *kv_access_error;
} SparkGlm5NextExecutionSlot;

typedef struct SparkGlm5NextCudaWave
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
	/* R1 certified-FP8 shadow of the lm_head shard (built on-device at
	 * load on head-owning ranks; 0 elsewhere and before the build). */
	const uint8_t *head_certified_fp8_payload;
	const float *head_certified_fp8_scale_f32;
	const float *head_certified_fp8_norm_f32;
	const SparkGlm5NextLayerWeights *layers;
	SparkGlm5NextExecutionSlot *slot;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	const uint32_t *index_ordinal_by_local_layer;
	/* KDA per-layer pools: fp32 state + the three bf16 conv windows,
	 * indexed by KDA ordinal (only 34 layers carry them). */
	uint8_t *kda_state_pools;
	uint64_t kda_state_layer_stride_bytes;
	uint8_t *kda_q_window_pool;
	uint8_t *kda_k_window_pool;
	uint8_t *kda_v_window_pool;
	uint64_t kda_window_layer_stride_bytes;
	const uint32_t *kda_ordinal_by_local_layer;
	const uint32_t *kda_state_index;
	/* Run structure for this wave (device arrays, staged by the module):
	 * sequence_row_begin[run_count+1] row ranges, run_state_index[run_count]
	 * resident-slot state keys. Decode = run per row; a same-slot chunk is
	 * one multi-row run through the recurrence kernels. */
	uint32_t run_count;
	const uint32_t *sequence_row_begin;
	const uint32_t *run_state_index;
	const uint32_t *host_sequence_row_begin;
	const uint32_t *host_run_state_index;
	/* Row capacity of the slot buffers (execution, not sequences): a
	 * multi-row wave may carry up to this many rows. */
	uint32_t execution_row_capacity;
	uint32_t kda_layer_count;
	const uint32_t *page_table;
	uint32_t multiprocessor_count;
	/* R3 flash-decode: 0 keeps the single-pass decode attention byte-for-
	 * byte; above the threshold the position range splits across CTAs and a
	 * combine pass merges the per-partition softmax states. */
	uint32_t decode_split_context_threshold;
	float *attention_split_partials_f32;
	uint64_t attention_split_partial_blocks;
} SparkGlm5NextCudaWave;

#ifdef __cplusplus
extern "C" {
#endif

int32_t SparkGlm5NextLaunchCudaWave(const SparkGlm5NextCudaWave *wave);
int32_t SparkGlm5NextLaunchCudaWaveBegin(const SparkGlm5NextCudaWave *wave);
int32_t SparkGlm5NextLaunchCudaLayerAttention(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);
int32_t SparkGlm5NextLaunchCudaLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);
/* HC placement on the REDUCED sublayer output (chain reduce-completion stages). */
int32_t SparkGlm5NextLaunchCudaLayerAttentionPost(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);
int32_t SparkGlm5NextLaunchCudaLayerMlpPost(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);
int32_t SparkGlm5NextLaunchCudaWaveHead(const SparkGlm5NextCudaWave *wave);
cudaError_t SparkGlm5NextLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset);
cudaError_t SparkGlm5NextLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count);
cudaError_t SparkGlm5NextLaunchHeadCertifiedQuantize(cudaStream_t stream,const void *head_bf16,uint8_t *certified_payload,float *certified_scale_f32,float *certified_norm_f32,uint32_t vocabulary,uint32_t hidden_dimension);
cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width);
cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);
int32_t SparkGlm5NextConfigureCudaModule(uint32_t *multiprocessor_count);

#ifdef __cplusplus
}
#endif
