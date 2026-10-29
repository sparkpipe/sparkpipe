#pragma once

#include "inference/kernels/tp_reduce.cuh"
#include "inference/llms/kimi_k3/slice.cuh"

static int32_t K3StreamedAccumulate(float *accumulator, const uint16_t *value, uint64_t elements, uint32_t first, cudaStream_t stream)
{
	if ( elements == 0u )
		return(LM_LAUNCH_OK);
	LmTpAccumulateBf16Kernel<<<(uint32_t)((elements + 255u) / 256u), 256u, 0, stream>>>(accumulator,value,elements,first);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t K3StreamedRound(uint16_t *destination, const float *accumulator, uint64_t elements, cudaStream_t stream)
{
	if ( elements == 0u || elements > UINT32_MAX )
		return(elements == 0u ? LM_LAUNCH_OK : LM_LAUNCH_ERR_SHAPE);
	LmTpF32ToBf16Kernel<<<(uint32_t)((elements + 255u) / 256u), 256u, 0, stream>>>(destination,accumulator,(uint32_t)elements);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format>
static int32_t K3StreamedMoe(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, float *latent_accumulator,
	float *hidden_accumulator, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	const uint64_t latent_elements = (uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN, hidden_elements = (uint64_t)rows * K3_HIDDEN;
	uint32_t shard;
	int32_t status;
	if ( shards == 0 || shard_count == 0u || slice * shard_count != K3_ROUTED_EXPERT_HIDDEN || !K3_EXPERT_CELLS(b) || b->tp_sharded == 0u ||
		latent_accumulator == 0 || hidden_accumulator == 0 || rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = LmStreamGemmDense<LmBf16Format>(b->router_weight,LmScaleTensorNone(),b->normed_bf16,(uint16_t *)0,b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,
		0u,0u,multiprocessors,stream);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
		status = LmStreamGemmDense<LmBf16Format>(shards[shard].routed_down_weight,LmScaleTensorNone(),b->normed_bf16,b->latent_full_bf16,(float *)0,rows,K3_HIDDEN,
			slice,K3_ROUTED_EXPERT_HIDDEN,shard * slice,multiprocessors,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMoeSelect(b,rows,packed_rows,stream);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		status = K3LayerMoeCells<Format>(b,rows,packed_rows,multiprocessors,stream,1u);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(latent_accumulator,b->latent_bf16,latent_elements,shard == 0u ? 1u : 0u,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3LayerSharedUp<Format>(b,rows,b->shared_mid_bf16,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3LayerSharedDown<Format>(b,rows,b->shared_mid_bf16,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,b->shared_out_bf16,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	if ( status == LM_LAUNCH_OK )
		status = K3StreamedRound(b->latent_full_bf16,latent_accumulator,latent_elements,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
		b->latent_full_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_full_bf16,K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		LM_LAUNCH((K3LatentSliceKernel<K3_LAYER_THREADS>), dim3((slice + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->latent_full_bf16,b->latent_bf16,slice,shard * slice,K3_ROUTED_EXPERT_HIDDEN);
		status = K3LayerMoeOutput<Format>(b,rows,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,b->hidden_bf16,hidden_elements,0u,stream);
	}
	if ( status == LM_LAUNCH_OK )
		status = K3StreamedRound(b->hidden_bf16,hidden_accumulator,hidden_elements,stream);
	if ( status == LM_LAUNCH_OK )
		K3PartialAdd(b,b->hidden_bf16,rows,stream);
	return(status);
}

struct K3StreamedKdaState
{
	uint8_t *state_pool;
	uint16_t *q_window;
	uint16_t *k_window;
	uint16_t *v_window;
};

template<class Format>
static int32_t K3StreamedKda(K3LayerBuffers *b, const K3LayerWeights *shards, const K3StreamedKdaState *states, uint32_t shard_count, uint16_t *partial,
	float *hidden_accumulator, uint32_t boundary, uint32_t rows, uint32_t sequences, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN;
	uint32_t shard;
	int32_t status = LM_LAUNCH_OK;
	if ( shards == 0 || states == 0 || shard_count == 0u || partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u ||
		K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS) * shard_count != K3_KDA_HEADS )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->kda_state_pool = states[shard].state_pool;
		b->kda_q_window = states[shard].q_window;
		b->kda_k_window = states[shard].k_window;
		b->kda_v_window = states[shard].v_window;
		status = K3LayerKdaBody<Format>(b,rows,sequences,1u,partial,(uint16_t *)0,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,partial,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	if ( status == LM_LAUNCH_OK )
		status = K3StreamedRound(partial,hidden_accumulator,hidden_elements,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( boundary != 0u )
		K3PartialSet(b,partial,rows,stream);
	else
		K3PartialAdd(b,partial,rows,stream);
	return(LM_LAUNCH_OK);
}
