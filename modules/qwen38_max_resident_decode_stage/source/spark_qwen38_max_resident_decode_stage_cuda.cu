#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/route.cuh"
#include "runtime/launch.h"
#include "common/common_gdn_stage_kernels.h"

#include "common/common_gdn_stage_kernels.cu"

static LmGdnStageLinearView SparkQwen38MaxLinearToCommon(const SparkQwen38MaxLinearView *view)
{
	LmGdnStageLinearView common;
	common.abi_version = view->abi_version;
	common.weight_format = view->weight_format;
	common.input_dimension = view->input_dimension;
	common.output_dimension = view->output_dimension;
	common.weight_payload = view->weight_payload;
	common.weight_scale_e8m0 = view->weight_scale_e8m0;
	common.weight_payload_bytes = view->weight_payload_bytes;
	common.weight_scale_bytes = view->weight_scale_bytes;
	return(common);
}

extern "C" cudaError_t SparkQwen38MaxConfigureCudaKernels(void)
{
	return(LmGdnStageConfigureKernels());
}

extern "C" cudaError_t SparkQwen38MaxLaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	return(LmGdnStageLaunchFusedResidualRmsNorm(stream,hidden_bf16,delta_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon));
}

extern "C" cudaError_t SparkQwen38MaxLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	return(LmGdnStageLaunchRmsNorm(stream,input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon));
}

extern "C" cudaError_t SparkQwen38MaxLaunchLinear(cudaStream_t stream, const SparkQwen38MaxLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	LmGdnStageLinearView common = SparkQwen38MaxLinearToCommon(view);
	return(LmGdnStageLaunchLinear(stream,&common,input_bf16,output_bf16,row_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38MaxGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38MaxGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	return(LmGdnStageLaunchConvUpdate(stream,qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38MaxGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	return(LmGdnStageLaunchDecayBeta(stream,decay_pre_bf16,beta_pre_bf16,weights->a_log_f32,weights->dt_bias_f32,log_decay_f32,beta_f32,row_count,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38MaxGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	return(LmGdnStageLaunchGdnStep(stream,conv_out_bf16,log_decay_f32,beta_f32,pool->state_f32,core_out_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->state_lane_stride_elements,pool->state_layer_stride_elements,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38MaxGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon)
{
	return(LmGdnStageLaunchGatedNorm(stream,core_bf16,z_bf16,weights->gdn_norm_weight_bf16,output_bf16,row_count,epsilon,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38MaxAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank)
{
	return(LmGdnStageLaunchAttnPrepare(stream,q_fused_bf16,k_bf16,v_bf16,weights->query_norm_weight_bf16,weights->key_norm_weight_bf16,kv_cache_bf16,slot_mapping,row_positions,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,epsilon,tp_degree,tp_rank));
}

extern "C" cudaError_t SparkQwen38MaxLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38MaxKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank)
{
	return(LmGdnStageLaunchAttnDecode(stream,q_fused_bf16,kv_cache_bf16,table->physical_block_indices,table->lane_physical_block_counts,table->lane_stride,row_lane_indices,context_lengths,head_out_bf16,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,tp_degree,tp_rank));
}

extern "C" cudaError_t SparkQwen38MaxLaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38MaxGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38MaxGdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
	return(LmGdnStageLaunchChunkConv(stream,qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,lane_index,token_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen38MaxGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
	return(LmGdnStageLaunchGdnChunk(stream,conv_out_bf16,log_decay_f32,beta_f32,workspace_qn,workspace_kn,workspace_cum_g,workspace_decay,workspace_attn,workspace_w,workspace_kg,pool->state_f32,core_out_bf16,lane_index,token_count,gdn_layer_ordinal,pool->state_lane_stride_elements,pool->state_layer_stride_elements,1u));
}

extern "C" cudaError_t SparkQwen38MaxLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	return(LmGdnStageLaunchEmbeddingGather(stream,token_ids,embedding_bf16,hidden_bf16,row_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	return(LmGdnStageLaunchTpCombineAdd(stream,destination_bf16,source_bf16,row_count,width));
}

extern "C" cudaError_t SparkQwen38MaxLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	return(LmGdnStageLaunchResidualAdd(stream,hidden_bf16,delta_bf16,row_count,dimension));
}

extern "C" cudaError_t SparkQwen38MaxLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(LmGdnStageLaunchHeadShadowQuantize(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SparkQwen38MaxLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	return(LmGdnStageLaunchHeadScreenedArgmax(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,candidate_ids,candidate_counts,output_token_ids,row_count,candidate_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	return(LmGdnStageLaunchHeadArgmax(stream,hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,candidate_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGateScores(cudaStream_t stream, const SparkQwen38MaxLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	LmGdnStageLinearView common = SparkQwen38MaxLinearToCommon(gate);
	return(LmGdnStageLaunchGateScores(stream,&common,input_bf16,scores_f32,row_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	return(LmGdnStageLaunchGateSelect(stream,scores_f32,bias_f32,row_count,expert_count,topk,route_scale,indices_u32,weights_f32));
}

extern "C" cudaError_t SparkQwen38MaxLaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	return(LmGdnStageLaunchSwiGlu(stream,gate_bf16,up_bf16,row_count,dimension));
}

extern "C" cudaError_t SparkQwen38MaxLaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension)
{
	return(LmGdnStageLaunchSharedGate(stream,accum_bf16,gate_weight_bf16,gate_input_bf16,row_count,dimension));
}

extern "C" cudaError_t SparkQwen38MaxLaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2)
{
	return(LmGdnStageLaunchMoeRoute(stream,route_expert,rows,expert_width,group_row_offset,route_packed_row,route_source_token,group_tile_prefix_w1,group_tile_prefix_w2));
}

extern "C" cudaError_t SparkQwen38MaxLaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen38MaxLinearView *w1, const SparkQwen38MaxLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count)
{
	LmGdnStageLinearView common_w1 = SparkQwen38MaxLinearToCommon(w1);
	LmGdnStageLinearView common_w3 = SparkQwen38MaxLinearToCommon(w3);
	return(LmGdnStageLaunchFusedExpertW13Act(stream,&common_w1,&common_w3,input_bf16,route_source_token,group_row_offset,group_tile_prefix,activated_bf16,rows,expert_width,limit,multiprocessor_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchExpertDown(cudaStream_t stream, const SparkQwen38MaxLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count)
{
	LmGdnStageLinearView common_stacked = SparkQwen38MaxLinearToCommon(stacked);
	return(LmGdnStageLaunchExpertDown(stream,&common_stacked,input_bf16,group_row_offset,group_tile_prefix,output_bf16,rows,expert_width,hidden_dimension,multiprocessor_count));
}

extern "C" cudaError_t SparkQwen38MaxLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(LmGdnStageLaunchMoePairReduceOverwrite(stream,slot_out_bf16,inverse_map,pair_weights_f32,output_bf16,row_count,hidden_dimension));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGroupedExpertLinear(cudaStream_t stream, const SparkQwen38MaxLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank)
{
	LmGdnStageLinearView common = SparkQwen38MaxLinearToCommon(view);
	return(LmGdnStageLaunchGroupedExpertLinear(stream,&common,input_bf16,source_row_map,group_row_offset,group_tile_prefix,output_bf16,source_row_count,multiprocessor_count,tp_degree,tp_rank));
}

extern "C" cudaError_t SparkQwen38MaxLaunchGroupedExpertTileLinear(cudaStream_t stream, const SparkQwen38MaxLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, void *output_bf16, uint32_t source_row_count, uint32_t tp_degree, uint32_t tp_rank)
{
	LmGdnStageLinearView common = SparkQwen38MaxLinearToCommon(view);
	return(LmGdnStageLaunchGroupedExpertTileLinear(stream,&common,input_bf16,source_row_map,group_row_offset,output_bf16,source_row_count,tp_degree,tp_rank));
}
