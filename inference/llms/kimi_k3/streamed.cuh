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

template<class Format>
static int32_t K3StreamedDenseMlp(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, uint16_t *partial, float *hidden_accumulator,
	uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN;
	uint32_t shard;
	int32_t status = LM_LAUNCH_OK;
	if ( shards == 0 || shard_count == 0u || partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		status = K3LayerDenseMlpBody<Format>(b,rows,partial,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,partial,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	if ( status == LM_LAUNCH_OK )
		status = K3StreamedRound(partial,hidden_accumulator,hidden_elements,stream);
	if ( status == LM_LAUNCH_OK )
		K3PartialAdd(b,partial,rows,stream);
	return(status);
}

template<class Format>
static int32_t K3StreamedSlice(K3LayerBuffers *b, const K3LayerWeights *weights, const K3StreamedKdaState *states, uint32_t shard_count,
	uint32_t first_layer, uint32_t layer_count, uint16_t *partial, float *latent_accumulator, float *hidden_accumulator,
	uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset, layer, boundary, recurrent = 0u;
	int32_t status = LM_LAUNCH_OK;
	for (offset = 0u; offset < layer_count && status == LM_LAUNCH_OK; offset++)
	{
		const K3LayerWeights *shards = weights + (uint64_t)offset * shard_count;
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS || K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT )
			return(LM_LAUNCH_ERR_SHAPE);
		K3BindLayer(&shards[0],b);
		boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
		if ( layer > 0u )
			K3AttnRes(b,b->attnres_attn_weight,((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( boundary != 0u )
		{
			if ( layer == 0u )
				K3PartialSet(b,b->hidden_bf16,rows,stream);
			K3BankStore(b,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
		}
		status = K3StreamedKda<Format>(b,shards,states + (uint64_t)recurrent * shard_count,shard_count,partial,hidden_accumulator,boundary,rows,1u,
			multiprocessors,stream);
		recurrent++;
		if ( status != LM_LAUNCH_OK )
			break;
		K3BindLayer(&shards[0],b);
		K3AttnRes(b,b->attnres_mlp_weight,(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		status = layer < K3_FIRST_ROUTED_LAYER
			? K3StreamedDenseMlp<Format>(b,shards,shard_count,partial,hidden_accumulator,rows,multiprocessors,stream)
			: K3StreamedMoe<Format>(b,shards,shard_count,latent_accumulator,hidden_accumulator,rows,packed_rows,multiprocessors,stream);
	}
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( first_layer + layer_count == K3_LAYERS )
	{
		K3AttnRes(b,b->attnres_out_weight,((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(LM_LAUNCH_OK);
	}
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,b->hidden_bf16,rows,K3_HIDDEN);
	return(LM_LAUNCH_OK);
}

template<uint32_t HEADS, class Geometry>
struct K3StreamedHeadKv
{
	static constexpr uint32_t kSlotBytes = HEADS * (K3_MHA_QK + K3_V_HEAD_DIM) * 2u;
	static constexpr uint32_t kPageSlots = Geometry::kPageSlots;
	static constexpr uint32_t kPageBytes = kSlotBytes * Geometry::kPageSlots;
	static constexpr bool kGrows = Geometry::kGrows;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return Geometry::PageOf(position); }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return Geometry::SlotInPage(position); }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return Geometry::PagesForTokens(tokens); }
};

struct K3StreamedMlaScratch
{
	uint8_t *latent_keys;
	uint8_t *expanded;
	float *partials;
	uint32_t key_capacity;
};

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedMla(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, const K3StreamedMlaScratch *scratch,
	uint16_t *partial, float *hidden_accumulator, uint32_t boundary, uint32_t rows, uint32_t key_count, uint32_t multiprocessors, cudaStream_t stream)
{
	typedef K3StreamedHeadKv<HEADS,Geometry> Expanded;
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN, key_pitch = (uint64_t)HEADS * (K3_MHA_QK + K3_V_HEAD_DIM);
	const uint64_t stride = (uint64_t)rows * LmGqaShardRecordFloats(HEADS,K3_V_HEAD_DIM,1u);
	LmKvShardGatherView view;
	uint32_t shard;
	int32_t status;
	if ( shards == 0 || scratch == 0 || shard_count * HEADS != K3_MLA_HEADS || partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u ||
		b->kv_shard.degree != 1u || key_count == 0u || key_count > scratch->key_capacity || b->shard_gather_list == 0 || b->shard_gather_offset == 0 ||
		b->shard_gather_context == 0 || shards[0].mla_kv_b_key_full_weight == 0 || shards[0].mla_kv_b_value_full_weight == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	status = K3LayerMlaDown<Format>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmKvShardStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->cache_shard,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	if ( LmKvShardGatherPackLaunch<Geometry,K3_LAYER_THREADS>(b->cache_shard,b->shard_gather_list,b->shard_gather_offset,b->shard_gather_context,1u,
			key_count,scratch->latent_keys,stream) != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	b->shard_mha = 1u;
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		const uint64_t key_rows = (uint64_t)shard * HEADS * K3_QK_NOPE_DIM, value_rows = (uint64_t)shard * HEADS * K3_V_HEAD_DIM;
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_b_weight,0,b->query_bf16,rows,K3_Q_LORA_RANK,HEADS * K3_MHA_QK,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		if ( LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_QK_NOPE_DIM>(scratch->latent_keys,key_count,
				(const uint16_t *)shards[0].mla_kv_b_key_full_weight + key_rows * K3_KV_LORA_RANK,HEADS,scratch->expanded,key_pitch,K3_MHA_QK,stream) != cudaSuccess ||
			LmLatentRopeBroadcastLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM>(scratch->latent_keys,key_count,HEADS,scratch->expanded,key_pitch,K3_MHA_QK,K3_QK_NOPE_DIM,stream) != cudaSuccess ||
			LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>(scratch->latent_keys,key_count,
				(const uint16_t *)shards[0].mla_kv_b_value_full_weight + value_rows * K3_KV_LORA_RANK,HEADS,scratch->expanded + (uint64_t)HEADS * K3_MHA_QK * 2u,
				key_pitch,K3_V_HEAD_DIM,stream) != cudaSuccess )
			status = LM_LAUNCH_ERR_LAUNCH;
		else if ( LmKvShardGatherViewInitialize<Expanded>(&view,b->cache_shard.pages,b->kv_shard,scratch->expanded,b->shard_gather_offset,b->shard_gather_context) != 0 )
			status = LM_LAUNCH_ERR_SHAPE;
		else if ( LmGqaShardPartialLaunch<Expanded,LmKvShardGatherView,HEADS,K3_MHA_QK,K3_V_HEAD_DIM>(view,b->query_bf16,HEADS,b->sequence_of_row,
				b->context_length,b->positions,K3_MLA_QK_SCALE,scratch->partials,stride,rows,stream) != cudaSuccess ||
			LmGqaShardMergeLaunch<K3_V_HEAD_DIM>(scratch->partials,stride,1u,HEADS,0u,b->value_bf16,rows,stream) != cudaSuccess )
			status = LM_LAUNCH_ERR_LAUNCH;
		if ( status == LM_LAUNCH_OK )
			status = K3LayerMlaOutput<Format>(b,rows,(uint16_t *)0,0u,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,b->attention_out_bf16,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	b->shard_mha = 0u;
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
