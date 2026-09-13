#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "spark_qwen36_dspark_internal.h"

 



















#ifdef __cplusplus
extern "C" {
#endif

extern uint64_t SparkQwen36DsparkHeadTopKChunkKeyCount(uint32_t row_count, uint32_t top_k);
extern cudaError_t SparkQwen36LaunchDsparkHeadTopK(cudaStream_t stream, const SparkQwen36LinearView *head, const void *hidden_bf16, uint64_t *chunk_keys, uint32_t *top_candidate_ids, float *top_scores_f32, void *top_scores_bf16, uint32_t row_count, uint32_t candidate_offset, uint32_t top_k);
extern cudaError_t SparkQwen36LaunchDsparkSelector(cudaStream_t stream, const void *hidden_bf16, const void *hidden_projection_bf16, const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids, const uint32_t *anchor_token_ids, const float *unary_f32, void *context_gate_bf16, float *edges_f32, uint32_t *draft_token_ids, uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank, uint32_t hidden_dimension);

#ifdef __cplusplus
}
#endif

 



typedef struct SparkQwen36DsparkSelectorWorkspace
{
	uint64_t *chunk_keys;        
	uint32_t *candidate_ids;     
	float *candidate_scores;     
	void *context_gate_bf16;     
	float *edges_f32;            
	uint32_t *draft_token_ids;   
} SparkQwen36DsparkSelectorWorkspace;

static inline uint64_t SparkQwen36DsparkSelectorWorkspaceBytes(uint32_t slot_count, uint32_t top_k, uint32_t rank)
{
	return((SparkQwen36DsparkHeadTopKChunkKeyCount(slot_count,top_k) * sizeof(uint64_t)) +
		((uint64_t)slot_count * top_k * sizeof(uint32_t)) +
		((uint64_t)slot_count * top_k * sizeof(float)) +
		((uint64_t)slot_count * rank * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES) +
		((uint64_t)slot_count * top_k * top_k * sizeof(float)) +
		((uint64_t)slot_count * sizeof(uint32_t)));
}

 





static inline cudaError_t SparkQwen36DsparkSelectorEmit(
	cudaStream_t stream,
	const SparkQwen36LinearView *head,
	const void *mask_hidden_bf16,
	const void *hidden_projection_bf16,
	const void *predecessor_bf16,
	const void *successor_bf16,
	const uint32_t *anchor_token_id_device,
	const SparkQwen36DsparkSelectorWorkspace *workspace,
	uint32_t slot_count,
	uint32_t top_k,
	uint32_t rank,
	uint32_t hidden_dimension,
	uint32_t *host_draft_token_ids)
{
	cudaError_t error;
	if ( head == 0 || mask_hidden_bf16 == 0 || hidden_projection_bf16 == 0 || predecessor_bf16 == 0 ||
	     successor_bf16 == 0 || anchor_token_id_device == 0 || workspace == 0 || host_draft_token_ids == 0 )
		return(cudaErrorInvalidValue);
	if ( slot_count == 0u || top_k == 0u || rank == 0u || hidden_dimension == 0u )
		return(cudaErrorInvalidValue);
	 

	error = SparkQwen36LaunchDsparkHeadTopK(stream,head,mask_hidden_bf16,workspace->chunk_keys,
		workspace->candidate_ids,workspace->candidate_scores,0,slot_count,0u,top_k);
	 


	if ( error == cudaSuccess )
		error = SparkQwen36LaunchDsparkSelector(stream,mask_hidden_bf16,hidden_projection_bf16,
			predecessor_bf16,successor_bf16,workspace->candidate_ids,anchor_token_id_device,
			workspace->candidate_scores,workspace->context_gate_bf16,workspace->edges_f32,
			workspace->draft_token_ids,0,1u,slot_count,top_k,rank,hidden_dimension);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(host_draft_token_ids,workspace->draft_token_ids,
			(size_t)slot_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}
