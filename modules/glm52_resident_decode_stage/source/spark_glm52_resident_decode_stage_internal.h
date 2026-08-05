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
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	void *kv_access_error;
} SparkGlm52ExecutionSlot;

typedef struct SparkGlm52CudaWave
{
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
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

int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave);
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count);
