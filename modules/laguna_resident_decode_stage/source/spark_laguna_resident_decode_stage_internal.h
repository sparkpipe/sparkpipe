#pragma once

#include <stdint.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_status.h"

typedef struct SparkLagunaLayerWeights
{
	const void *attn_norm_bf16;
	const void *fused_qkv_bf16;
	const void *q_norm_bf16;
	const void *k_norm_bf16;
	const void *attn_output_bf16;
	const void *attn_gate_bf16;
	const void *post_attn_norm_bf16;
	const void *dense_gate_up_bf16;
	const void *dense_down_bf16;
	const void *router_bf16;
	const float *router_correction_f32;
	const void *expert_gate_up_payload;
	const void *expert_gate_up_scale;
	const void *expert_down_payload;
	const void *expert_down_scale;
	uint64_t expert_gate_up_payload_offset;
	uint64_t expert_gate_up_scale_offset;
	uint64_t expert_down_payload_offset;
	uint64_t expert_down_scale_offset;
	const void *shared_gate_up_bf16;
	const void *shared_down_bf16;
} SparkLagunaLayerWeights;

typedef struct SparkLagunaExecutionSlot
{
	void *stream;
	void *route_ready_event;
	uint32_t route_recorded;
	void *host_staging;
	uint32_t *host_token_ids;
	uint32_t *host_resident_slots;
	uint32_t *host_positions;
	uint32_t *host_output_token_ids;
	uint32_t *host_kv_access_error;
	uint32_t *host_group_row_offset;
	uint32_t *token_ids;
	uint32_t *resident_slots;
	uint32_t *positions;
	uint32_t *context_lengths;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *qkv_bf16;
	uint16_t *q_bf16;
	uint16_t *k_bf16;
	uint16_t *v_bf16;
	uint16_t *attention_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_bf16;
	uint32_t *window_positions;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits_f32;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
	uint64_t *head_maxloc_u64;
	void *head_certified_scratch;
	uint32_t *head_certified_candidates;
	uint32_t *head_screened_count;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	void *kv_access_error;
} SparkLagunaExecutionSlot;

typedef struct SparkLagunaCudaWave
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
	uint64_t boundary_row_offset;
	const uint32_t *host_token_ids;
	const uint32_t *host_resident_slots;
	const uint32_t *host_positions;
	const void *hidden_input_bf16;
	void *hidden_output_bf16;
	uint32_t *host_output_token_ids;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	const uint8_t *head_certified_fp8_payload;
	const float *head_certified_fp8_scale_f32;
	const float *head_certified_fp8_norm_f32;
	const float *yarn_inv_freq;
	const SparkLagunaLayerWeights *layers;
	uint32_t lazy_experts;
	uint32_t expert_lease_local_layer;
	const uint8_t *expert_lease_base;
	SparkLagunaExecutionSlot *slot;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint32_t commit;
	uint32_t execution_row_capacity;
	const uint32_t *page_table;
	uint32_t multiprocessor_count;
	uint32_t decode_split_context_threshold;
} SparkLagunaCudaWave;

#ifdef __cplusplus
extern "C" {
#endif

int32_t SparkLagunaLaunchCudaWave(const SparkLagunaCudaWave *wave);
int32_t SparkLagunaLaunchCudaWaveBegin(const SparkLagunaCudaWave *wave);
SparkStatus SparkLagunaLaunchOpWait(cudaStream_t stream,void *flag_device,uint64_t wait_value);
int32_t SparkLagunaLaunchCudaLayerAttention(const SparkLagunaCudaWave *wave,uint32_t local_layer);
int32_t SparkLagunaLaunchCudaLayerMlp(const SparkLagunaCudaWave *wave,uint32_t local_layer);
int32_t SparkLagunaLaunchCudaLayerMlpRoute(const SparkLagunaCudaWave *wave,uint32_t local_layer);
cudaError_t SparkLagunaPollCudaLayerMlpRoute(const SparkLagunaCudaWave *wave);
int32_t SparkLagunaLaunchCudaLayerMlpExperts(const SparkLagunaCudaWave *wave,uint32_t local_layer);
int32_t SparkLagunaLaunchCudaLayerAttentionPost(const SparkLagunaCudaWave *wave,uint32_t local_layer);
int32_t SparkLagunaLaunchCudaLayerMlpPost(const SparkLagunaCudaWave *wave,uint32_t local_layer);
int32_t SparkLagunaLaunchCudaWaveHead(const SparkLagunaCudaWave *wave);
cudaError_t SparkLagunaLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset);
cudaError_t SparkLagunaLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count);
cudaError_t SparkLagunaLaunchHeadCertifiedQuantize(cudaStream_t stream,const void *head_bf16,uint8_t *certified_payload,float *certified_scale_f32,float *certified_norm_f32,uint32_t vocabulary,uint32_t hidden_dimension);
cudaError_t SparkLagunaLaunchDirectSum(cudaStream_t stream,void *destination,const void *const *rank_devices,uint32_t local_rank,uint32_t rows,uint32_t width);
cudaError_t SparkLagunaLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width);
cudaError_t SparkLagunaLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);
int32_t SparkLagunaConfigureCudaModule(uint32_t *multiprocessor_count);

#ifdef __cplusplus
}
#endif
