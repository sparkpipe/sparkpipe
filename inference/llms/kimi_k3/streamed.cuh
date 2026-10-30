#pragma once

#include "sparkpipe/spark_tp_device_collective.h"
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

struct K3StreamedJoin
{
	uint16_t *weight;
	uint16_t *activation;
	uint64_t weight_capacity;
	uint64_t activation_capacity;
};

static int32_t K3StreamedJoinColumns(uint16_t *destination, uint32_t destination_columns, uint32_t first_column, const void *part, uint32_t part_columns,
	uint32_t rows, cudaStream_t stream)
{
	return(cudaMemcpy2DAsync(destination + first_column,(size_t)destination_columns * sizeof(uint16_t),part,(size_t)part_columns * sizeof(uint16_t),
		(size_t)part_columns * sizeof(uint16_t),rows,cudaMemcpyDeviceToDevice,stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format>
static int32_t K3StreamedMoeBegin(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, const K3StreamedJoin *join, float *latent_accumulator,
	uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN), shared = K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE);
	uint32_t shard;
	int32_t status;
	if ( shards == 0 || shard_count == 0u || slice * shard_count != K3_ROUTED_EXPERT_HIDDEN || !K3_EXPERT_CELLS(b) || b->tp_sharded == 0u ||
		latent_accumulator == 0 || rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u || join == 0 || join->weight == 0 || join->activation == 0 ||
		join->weight_capacity < (uint64_t)K3_HIDDEN * 2u * shard_count * shared || join->weight_capacity < (uint64_t)K3_HIDDEN * K3_ROUTED_EXPERT_HIDDEN ||
		join->activation_capacity < (uint64_t)rows * shard_count * shared )
		return(LM_LAUNCH_ERR_SHAPE);
	for (shard = 0u; shard < shard_count; shard++)
		if ( shards[shard].routed_down_weight == 0 || shards[shard].shared_w1_weight == 0 || shards[shard].shared_w2_weight == 0 || shards[shard].shared_w1_scale != 0 )
			return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = LmStreamGemmDense<LmBf16Format>(b->router_weight,LmScaleTensorNone(),b->normed_bf16,(uint16_t *)0,b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,
		0u,0u,multiprocessors,stream);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
		status = K3StreamedJoinColumns(join->weight + (uint64_t)shard * slice * K3_HIDDEN,K3_HIDDEN,0u,shards[shard].routed_down_weight,K3_HIDDEN,slice,stream);
	if ( status == LM_LAUNCH_OK )
		status = LmStreamGemmDense<LmBf16Format>(join->weight,LmScaleTensorNone(),b->normed_bf16,b->latent_full_bf16,(float *)0,rows,K3_HIDDEN,
			K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,0u,multiprocessors,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMoeSelect(b,rows,packed_rows,stream);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
		status = K3StreamedJoinColumns(join->weight + (uint64_t)shard * 2u * shared * K3_HIDDEN,K3_HIDDEN,0u,shards[shard].shared_w1_weight,K3_HIDDEN,2u * shared,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3Project<LmBf16Format>(b,b->normed_bf16,join->weight,0,b->gate_up_bf16,rows,K3_HIDDEN,2u * shard_count * shared,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows * shard_count, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,join->activation,shared,(uint64_t)shared,K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
		status = K3StreamedJoinColumns(join->weight,shard_count * shared,shard * shared,shards[shard].shared_w2_weight,shared,K3_HIDDEN,stream);
	return(status);
}

template<class Format>
static int32_t K3StreamedMoeShard(K3LayerBuffers *b, const K3LayerWeights *weights, uint32_t shard, uint32_t member, uint32_t members, uint32_t shard_count,
	const K3StreamedJoin *join, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t channels;
	int32_t status;
	if ( weights == 0 || shard >= shard_count || member >= members || join == 0 || weights->expert_w1_weight == 0 || weights->expert_w2_weight == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(weights,b);
	b->tp_rank = shard;
	channels = b->expert_w1_output / 2u;
	if ( b->tp_sharded == 0u || b->expert_interleave == 0u || b->expert_w2_input != channels )
		return(LM_LAUNCH_ERR_SHAPE);
	status = LmCellMmaExperts(b->expert_w1_weight,b->latent_full_bf16,b->gate_up_bf16,b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,0u,
		K3_ROUTED_EXPERT_HIDDEN,b->expert_w1_output,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16 + (uint64_t)member * channels,channels,(uint64_t)members * channels,K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	return(LM_LAUNCH_OK);
}

static int32_t K3StreamedMoeDown(K3LayerBuffers *b, const void *const *down_weights, uint32_t members, uint32_t first, float *latent_accumulator, uint32_t rows,
	uint32_t packed_rows, cudaStream_t stream)
{
	int32_t status;
	if ( down_weights == 0 || members == 0u || members > LM_CELL_MMA_MAX_SEGMENTS || latent_accumulator == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	status = LmCellMmaExpertsSegmented(down_weights,members,b->intermediate_bf16,b->gate_up_bf16,b->group_row_offset,0,K3_EXPERTS,packed_rows,1u,
		members * b->expert_w2_input,K3_ROUTED_EXPERT_HIDDEN,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((K3_ROUTED_EXPERT_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,LmLaunchGridRows(rows)), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16,rows,K3_TOP_K,K3_ROUTED_EXPERT_HIDDEN);
	return(K3StreamedAccumulate(latent_accumulator,b->latent_bf16,(uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN,first,stream));
}

template<class Format>
static int32_t K3StreamedMoeEnd(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, const K3StreamedJoin *join, float *latent_accumulator,
	uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN), shared = K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE);
	uint32_t shard;
	int32_t status;
	K3BindLayer(&shards[0],b);
	status = K3Project<LmBf16Format>(b,join->activation,join->weight,0,b->shared_out_bf16,rows,shard_count * shared,K3_HIDDEN,multiprocessors,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3StreamedRound(b->latent_full_bf16,latent_accumulator,(uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
		b->latent_full_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_full_bf16,K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
		status = K3StreamedJoinColumns(join->weight,K3_ROUTED_EXPERT_HIDDEN,shard * slice,shards[shard].routed_up_weight,slice,K3_HIDDEN,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3Project<LmBf16Format>(b,b->latent_full_bf16,join->weight,0,b->hidden_bf16,rows,K3_ROUTED_EXPERT_HIDDEN,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,LmLaunchGridRows(rows)),K3_LAYER_THREADS,0,stream,
		b->hidden_bf16,b->shared_out_bf16,b->hidden_bf16,rows,K3_HIDDEN);
	K3PartialAdd(b,b->hidden_bf16,rows,stream);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3StreamedMoe(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, uint32_t members, const K3StreamedJoin *join,
	float *latent_accumulator, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const void *down_weights[LM_CELL_MMA_MAX_SEGMENTS];
	uint32_t shard;
	int32_t status;
	if ( members == 0u || members > LM_CELL_MMA_MAX_SEGMENTS || shard_count % members != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	status = K3StreamedMoeBegin<Format>(b,shards,shard_count,join,latent_accumulator,rows,packed_rows,multiprocessors,stream);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		down_weights[shard % members] = shards[shard].expert_w2_weight;
		status = K3StreamedMoeShard<Format>(b,&shards[shard],shard,shard % members,members,shard_count,join,rows,packed_rows,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK && shard % members == members - 1u )
			status = K3StreamedMoeDown(b,down_weights,members,shard + 1u == members ? 1u : 0u,latent_accumulator,rows,packed_rows,stream);
	}
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3StreamedMoeEnd<Format>(b,shards,shard_count,join,latent_accumulator,rows,multiprocessors,stream));
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

__global__ static void K3StreamedStateIdentityKernel(float *state, uint64_t matrices)
{
	const uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	const uint64_t within = index % ((uint64_t)K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM);
	if ( index < matrices * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM )
		state[index] = within / K3_KDA_VALUE_DIM == within % K3_KDA_VALUE_DIM ? 1.0f : 0.0f;
}

__global__ static void K3StreamedStateComposeKernel(const float *transform, const float *incoming, const float *start, float *outgoing)
{
	const uint64_t head = blockIdx.x, row = blockIdx.y, column = threadIdx.x;
	const float *m = transform + head * K3_KDA_KEY_DIM * K3_KDA_KEY_DIM + row * K3_KDA_KEY_DIM;
	const float *s = incoming + head * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM + column;
	float total = start != 0 ? start[(head * K3_KDA_KEY_DIM + row) * K3_KDA_VALUE_DIM + column] : 0.0f;
	for (uint32_t inner = 0u; inner < K3_KDA_KEY_DIM; inner++)
		total = __fmaf_rn(m[inner],s[(uint64_t)inner * K3_KDA_VALUE_DIM],total);
	outgoing[(head * K3_KDA_KEY_DIM + row) * K3_KDA_VALUE_DIM + column] = total;
}

__global__ static void K3StreamedConvSeedKernel(uint16_t *window, const uint16_t *input, const uint32_t *chunk_row_begin, uint32_t channels, uint32_t chunks)
{
	const uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	const uint32_t chunk = 1u + (uint32_t)(index / channels), channel = (uint32_t)(index % channels);
	if ( chunk >= chunks )
		return;
	for (uint32_t tap = 0u; tap < K3_KDA_CONV_KERNEL; tap++)
		window[((uint64_t)chunk * channels + channel) * K3_KDA_CONV_KERNEL + tap] =
			input[((uint64_t)chunk_row_begin[chunk] - K3_KDA_CONV_KERNEL + tap) * channels + channel];
}

typedef void (*K3StreamedCollective)(void *context, void *stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *send, void *receive);

__global__ static void K3StreamedConvTailKernel(uint16_t *window, const uint16_t *input, uint32_t rows, uint32_t channels)
{
	const uint32_t channel = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( channel >= channels )
		return;
	for (uint32_t tap = 0u; tap < K3_KDA_CONV_KERNEL; tap++)
		window[(uint64_t)channel * K3_KDA_CONV_KERNEL + tap] = input[((uint64_t)rows - K3_KDA_CONV_KERNEL + tap) * channels + channel];
}

struct K3StreamedKdaExchange
{
	uint32_t node;
	uint32_t nodes;
	uint32_t chunks;
	const uint32_t *chunk_row_begin;
	const uint32_t *chunk_slots;
	uint16_t *tail_send;
	uint16_t *tails;
	float *summary_send;
	float *summaries;
	float *ends;
	float *transforms;
	float *incoming;
	float *running;
	uint16_t *q_windows;
	uint16_t *k_windows;
	uint16_t *v_windows;
	const uint16_t *zeros;
	uint16_t *scratch;
	K3StreamedCollective collective;
	void *context;
};

static int32_t K3StreamedFold(const float *transforms, const float *starts, float *running, float *destination, uint32_t heads, uint32_t chunks,
	uint64_t slot_floats, cudaStream_t stream)
{
	uint32_t chunk, current = 0u;
	if ( cudaMemcpyAsync(running,starts != 0 ? starts : transforms,(size_t)slot_floats * sizeof(float),cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	for (chunk = 1u; chunk < chunks; chunk++)
	{
		K3StreamedStateComposeKernel<<<dim3(heads,K3_KDA_KEY_DIM),K3_KDA_VALUE_DIM,0,stream>>>(transforms + (uint64_t)chunk * slot_floats,running + current * slot_floats,
			starts != 0 ? starts + (uint64_t)chunk * slot_floats : (const float *)0,running + (current ^ 1u) * slot_floats);
		current ^= 1u;
	}
	if ( cudaMemcpyAsync(destination,running + current * slot_floats,(size_t)slot_floats * sizeof(float),cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3StreamedKdaNode(K3LayerBuffers *b, const K3LayerWeights *shards, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaExchange *x, uint16_t *partial, const K3StreamedJoin *join, uint32_t boundary, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint32_t channels[3] = {heads * K3_KDA_KEY_DIM,heads * K3_KDA_KEY_DIM,heads * K3_KDA_VALUE_DIM};
	const uint64_t window_offset[3] = {0u,(uint64_t)channels[0] * K3_KDA_CONV_KERNEL,(uint64_t)(channels[0] + channels[1]) * K3_KDA_CONV_KERNEL};
	const uint64_t tail_elements = (uint64_t)(channels[0] + channels[1] + channels[2]) * K3_KDA_CONV_KERNEL;
	const uint64_t slot_floats = (uint64_t)heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM, slot_bytes = slot_floats * sizeof(float);
	const uint32_t *row_begin = b->sequence_row_begin, *state_index = b->kda_state_index;
	uint32_t shard;
	int32_t status = LM_LAUNCH_OK;
	if ( shards == 0 || states == 0 || x == 0 || x->collective == 0 || x->nodes == 0u || x->node >= x->nodes || x->chunks == 0u ||
		rows < (uint64_t)x->chunks * K3_KDA_CONV_KERNEL || partial == 0 || join == 0 || join->weight == 0 || join->activation == 0 || b->tp_sharded == 0u || heads * shard_count != K3_KDA_HEADS ||
		K3_KDA_KEY_DIM != K3_KDA_VALUE_DIM || x->chunk_row_begin == 0 || x->chunk_slots == 0 || x->tail_send == 0 || x->tails == 0 || x->summary_send == 0 || x->summaries == 0 || x->ends == 0 ||
		x->transforms == 0 || x->incoming == 0 || x->running == 0 || x->zeros == 0 || x->scratch == 0 || x->q_windows == 0 || x->k_windows == 0 || x->v_windows == 0 ||
		join->weight_capacity < (uint64_t)K3_HIDDEN * shard_count * channels[2] || join->activation_capacity < (uint64_t)rows * shard_count * channels[2] )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		uint16_t *const windows[3] = {x->q_windows,x->k_windows,x->v_windows};
		uint16_t *const carried[3] = {states[shard].q_window,states[shard].k_window,states[shard].v_window};
		const uint16_t *const last_tail = x->tails + (uint64_t)(x->nodes - 1u) * tail_elements;
		uint32_t which, node, chunk, current = 0u;
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->sequence_row_begin = x->chunk_row_begin;
		b->kda_state_index = x->chunk_slots;
		status = K3LayerKdaProject<Format>(b,rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		{
			const uint16_t *const inputs[3] = {b->query_bf16,b->key_bf16,b->value_bf16};
			for (which = 0u; which < 3u; which++)
				K3StreamedConvTailKernel<<<(channels[which] + 255u) / 256u,256u,0,stream>>>(x->tail_send + window_offset[which],inputs[which],rows,channels[which]);
			x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,1u,(uint32_t)tail_elements,x->tail_send,x->tails);
			for (which = 0u; which < 3u; which++)
			{
				const uint64_t window_elements = (uint64_t)channels[which] * K3_KDA_CONV_KERNEL;
				const uint16_t *seed = x->node == 0u ? carried[which] : x->tails + (uint64_t)(x->node - 1u) * tail_elements + window_offset[which];
				if ( cudaMemcpyAsync(windows[which],seed,(size_t)window_elements * sizeof(uint16_t),cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
					return(LM_LAUNCH_ERR_LAUNCH);
				if ( x->chunks > 1u )
					K3StreamedConvSeedKernel<<<(uint32_t)(((uint64_t)(x->chunks - 1u) * channels[which] + 255u) / 256u),256u,0,stream>>>(
						windows[which],inputs[which],x->chunk_row_begin,channels[which],x->chunks);
			}
		}
		b->kda_q_window = windows[0];
		b->kda_k_window = windows[1];
		b->kda_v_window = windows[2];
		status = K3LayerKdaPrepare<Format>(b,rows,x->chunks,1u,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK || cudaMemsetAsync(x->ends,0,(size_t)(slot_bytes * x->chunks),stream) != cudaSuccess )
			return(status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_LAUNCH);
		b->kda_state_pool = (uint8_t *)x->ends;
		status = K3LayerKdaRecur(b,b->value_bf16,x->scratch,x->chunks,1u,stream);
		K3StreamedStateIdentityKernel<<<(uint32_t)((slot_floats * x->chunks + 255u) / 256u),256u,0,stream>>>(x->transforms,(uint64_t)heads * x->chunks);
		b->kda_state_pool = (uint8_t *)x->transforms;
		if ( status == LM_LAUNCH_OK )
			status = K3LayerKdaRecur(b,x->zeros,x->scratch,x->chunks,1u,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedFold(x->transforms,x->ends,x->running,x->summary_send,heads,x->chunks,slot_floats,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedFold(x->transforms,(const float *)0,x->running,x->summary_send + slot_floats,heads,x->chunks,slot_floats,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,1u,(uint32_t)slot_bytes,x->summary_send,x->summaries);
		if ( cudaMemcpyAsync(x->running,states[shard].state_pool,(size_t)slot_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		for (node = 0u; node < x->nodes; node++)
		{
			const float *summary = x->summaries + (uint64_t)node * 2u * slot_floats;
			if ( node == x->node && cudaMemcpyAsync(x->incoming,x->running + current * slot_floats,(size_t)slot_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
				return(LM_LAUNCH_ERR_LAUNCH);
			K3StreamedStateComposeKernel<<<dim3(heads,K3_KDA_KEY_DIM),K3_KDA_VALUE_DIM,0,stream>>>(summary + slot_floats,x->running + current * slot_floats,summary,
				x->running + (current ^ 1u) * slot_floats);
			current ^= 1u;
		}
		if ( cudaMemcpyAsync(states[shard].state_pool,x->running + current * slot_floats,(size_t)slot_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		for (which = 0u; which < 3u; which++)
			if ( cudaMemcpyAsync(carried[which],last_tail + window_offset[which],(size_t)channels[which] * K3_KDA_CONV_KERNEL * sizeof(uint16_t),
					cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
				return(LM_LAUNCH_ERR_LAUNCH);
		for (chunk = 1u; chunk < x->chunks; chunk++)
			K3StreamedStateComposeKernel<<<dim3(heads,K3_KDA_KEY_DIM),K3_KDA_VALUE_DIM,0,stream>>>(x->transforms + (uint64_t)(chunk - 1u) * slot_floats,
				x->incoming + (uint64_t)(chunk - 1u) * slot_floats,x->ends + (uint64_t)(chunk - 1u) * slot_floats,x->incoming + (uint64_t)chunk * slot_floats);
		b->kda_state_pool = (uint8_t *)x->incoming;
		status = K3LayerKdaRecur(b,b->value_bf16,b->attention_out_bf16,x->chunks,0u,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		status = K3LayerKdaGate(b,rows,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedJoinColumns(join->activation,shard_count * channels[2],shard * channels[2],b->attention_out_bf16,channels[2],rows,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedJoinColumns(join->weight,shard_count * channels[2],shard * channels[2],b->kda_out_weight,channels[2],K3_HIDDEN,stream);
	}
	b->sequence_row_begin = row_begin;
	b->kda_state_index = state_index;
	if ( status == LM_LAUNCH_OK )
		status = K3Project<LmBf16Format>(b,join->activation,join->weight,0,partial,rows,shard_count * channels[2],K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( boundary != 0u )
		K3PartialSet(b,partial,rows,stream);
	else
		K3PartialAdd(b,partial,rows,stream);
	return(LM_LAUNCH_OK);
}

struct K3StreamedMlaExchange
{
	uint32_t node;
	uint32_t nodes;
	uint32_t pass_rows;
	uint32_t first_key;
	uint32_t key_capacity;
	uint16_t *q_latents;
	uint8_t *latent_keys;
	uint16_t *query;
	uint8_t *expanded;
	float *partials;
	uint16_t *merged;
	uint16_t *value_send;
	uint16_t *values;
	const uint32_t *pass_positions;
	const uint32_t *pass_sequence_of_row;
	const uint32_t *pass_row_offset;
	uint32_t *pass_tile_prefix;
	const uint32_t *key_offset;
	const uint32_t *key_context;
	LmKvShardView store;
	const uint32_t *store_sequence_of_row;
	const uint32_t *older_sequence;
	const uint32_t *older_offset;
	const uint32_t *older_context;
	uint8_t *older_send;
	uint8_t *older_receive;
	uint64_t older_capacity;
	uint32_t expanded_keys;
	uint32_t key_block;
	float *running;
	K3StreamedCollective collective;
	void *context;
};

template<uint32_t SLOT_BYTES>
__global__ static void K3StreamedOlderKeysKernel(SparkKvShard shard, const uint8_t *gathered, uint32_t first_local, uint32_t rows_per_rank, uint32_t keys, uint8_t *latent_keys)
{
	static_assert(SLOT_BYTES % 16u == 0u, "a key moves as whole 16-byte words");
	for (uint64_t slot = blockIdx.x; slot < (uint64_t)shard.degree * rows_per_rank; slot += gridDim.x)
	{
		SparkKvShard owner = shard;
		uint32_t position;
		owner.rank = (uint32_t)(slot / rows_per_rank);
		position = SparkKvShardLocalPosition(owner,first_local + (uint32_t)(slot % rows_per_rank));
		if ( position >= keys )
			continue;
		for (uint32_t index = threadIdx.x; index < SLOT_BYTES / 16u; index += blockDim.x)
			((uint4 *)(latent_keys + (uint64_t)position * SLOT_BYTES))[index] = ((const uint4 *)(gathered + slot * SLOT_BYTES))[index];
	}
}

template<uint32_t VALUE_DIM>
__global__ static void K3StreamedFoldRecordsKernel(float *running, const float *block, uint32_t heads, uint32_t rows, uint32_t first)
{
	const uint32_t width = heads * VALUE_DIM, record = LmGqaShardRecordFloats(heads,VALUE_DIM,1u);
	float *r = running + (uint64_t)blockIdx.x * record;
	const float *b = block + (uint64_t)blockIdx.x * record;
	uint32_t index;
	if ( blockIdx.x >= rows )
		return;
	for (index = threadIdx.x; index < width; index += blockDim.x)
	{
		const uint32_t head = index / VALUE_DIM;
		const float old_max = first != 0u ? -INFINITY : r[width + 2u * head], new_max = b[width + 2u * head], top = fmaxf(old_max,new_max);
		const float keep = old_max == -INFINITY ? 0.0f : __expf(old_max - top), take = new_max == -INFINITY ? 0.0f : __expf(new_max - top);
		r[index] = (first != 0u ? 0.0f : r[index] * keep) + b[index] * take;
	}
	__syncthreads();
	for (index = threadIdx.x; index < heads; index += blockDim.x)
	{
		const float old_max = first != 0u ? -INFINITY : r[width + 2u * index], new_max = b[width + 2u * index], top = fmaxf(old_max,new_max);
		const float keep = old_max == -INFINITY ? 0.0f : __expf(old_max - top), take = new_max == -INFINITY ? 0.0f : __expf(new_max - top);
		r[width + 2u * index + 1u] = (first != 0u ? 0.0f : r[width + 2u * index + 1u] * keep) + b[width + 2u * index + 1u] * take;
		r[width + 2u * index] = top;
	}
}

template<class Geometry>
static int32_t K3StreamedMlaKeys(const K3StreamedMlaExchange *x, cudaStream_t stream)
{
	const uint32_t degree = x->store.shard.degree;
	uint32_t older_rows, chunk_rows, first_local;
	if ( degree == 0u )
		return(LM_LAUNCH_OK);
	if ( x->store_sequence_of_row == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmKvShardStoreKernel<Geometry,K3_LAYER_THREADS>), x->pass_rows, K3_LAYER_THREADS, 0, stream,
		x->store,(const uint16_t *)(x->latent_keys + (uint64_t)x->first_key * Geometry::kSlotBytes),x->store_sequence_of_row,x->pass_positions,x->pass_rows,
		Geometry::kSlotBytes / 2u);
	if ( x->first_key == 0u )
		return(LM_LAUNCH_OK);
	older_rows = SparkKvShardGatherKeys(x->store.shard,x->first_key);
	chunk_rows = (uint32_t)(x->older_capacity / ((uint64_t)degree * Geometry::kSlotBytes));
	chunk_rows = chunk_rows < older_rows ? chunk_rows : older_rows;
	if ( x->older_sequence == 0 || x->older_offset == 0 || x->older_context == 0 || x->older_send == 0 || x->older_receive == 0 || chunk_rows == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	for (first_local = 0u; first_local < older_rows; first_local += chunk_rows)
	{
		const uint32_t count = older_rows - first_local < chunk_rows ? older_rows - first_local : chunk_rows;
		if ( LmKvShardGatherPackRangeLaunch<Geometry,K3_LAYER_THREADS>(x->store,x->older_sequence,x->older_offset,x->older_context,1u,first_local,first_local + count,
				x->older_send - (uint64_t)first_local * Geometry::kSlotBytes,stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,count,Geometry::kSlotBytes / 2u,x->older_send,x->older_receive);
		LM_LAUNCH((K3StreamedOlderKeysKernel<Geometry::kSlotBytes>), LM_KV_SHARD_UNPACK_BLOCKS, K3_LAYER_THREADS, 0, stream,
			x->store.shard,x->older_receive,first_local,count,x->first_key,x->latent_keys);
	}
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedMlaNode(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, const K3StreamedMlaExchange *x,
	uint16_t *partial, const K3StreamedJoin *join, uint32_t boundary, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	typedef K3StreamedHeadKv<HEADS,Geometry> Expanded;
	const uint64_t key_pitch = (uint64_t)HEADS * (K3_MHA_QK + K3_V_HEAD_DIM);
	const uint64_t value_elements = (uint64_t)HEADS * K3_V_HEAD_DIM;
	uint16_t *const own_value = b->value_bf16;
	const uint32_t *const own_row_offset = b->dense_row_offset;
	uint32_t *const own_tile_prefix = b->dense_tile_prefix;
	LmKvShardGatherView view;
	uint64_t stride, key_count, key_first, block, blocks;
	uint32_t shard, groups;
	int32_t status;
	if ( shards == 0 || x == 0 || x->collective == 0 || x->nodes == 0u || x->node >= x->nodes || shard_count * HEADS != K3_MLA_HEADS || shard_count % x->nodes != 0u ||
		partial == 0 || join == 0 || join->weight == 0 || join->activation == 0 || join->weight_capacity < (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM ||
		join->activation_capacity < (uint64_t)rows * K3_MLA_HEADS * K3_V_HEAD_DIM || b->tp_sharded == 0u || b->kv_shard.degree != 1u || rows == 0u || (uint64_t)rows * x->nodes != x->pass_rows ||
		x->q_latents == 0 || x->latent_keys == 0 || x->query == 0 || x->expanded == 0 || x->partials == 0 || x->merged == 0 || x->value_send == 0 || x->values == 0 ||
		x->pass_positions == 0 || x->pass_sequence_of_row == 0 || x->pass_row_offset == 0 || x->pass_tile_prefix == 0 || x->key_offset == 0 || x->key_context == 0 ||
		shards[x->node].mla_kv_b_key_full_weight == 0 || shards[x->node].mla_kv_b_value_full_weight == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	groups = shard_count / x->nodes;
	key_count = (uint64_t)x->first_key + x->pass_rows;
	stride = (uint64_t)x->pass_rows * LmGqaShardRecordFloats(HEADS,K3_V_HEAD_DIM,1u);
	block = x->key_block != 0u && x->key_block < key_count ? x->key_block : key_count;
	blocks = (key_count + block - 1u) / block;
	if ( key_count > x->key_capacity || (x->expanded_keys != 0u && block > x->expanded_keys) || (blocks > 1u && x->running == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	status = K3LayerMlaDown<Format>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,rows,K3_Q_LORA_RANK,b->latent_bf16,x->q_latents);
	x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,rows,Geometry::kSlotBytes / 2u,b->kv_slot_bf16,
		x->latent_keys + (uint64_t)x->first_key * Geometry::kSlotBytes);
	status = K3StreamedMlaKeys<Geometry>(x,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	b->dense_row_offset = x->pass_row_offset;
	b->dense_tile_prefix = x->pass_tile_prefix;
	for (shard = x->node; shard < shard_count && status == LM_LAUNCH_OK; shard += x->nodes)
	{
		const uint64_t key_rows = (uint64_t)shard * HEADS * K3_QK_NOPE_DIM, value_rows = (uint64_t)shard * HEADS * K3_V_HEAD_DIM;
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		status = K3Project<LmBf16Format>(b,x->q_latents,b->mla_q_b_weight,0,x->query,x->pass_rows,K3_Q_LORA_RANK,HEADS * K3_MHA_QK,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		for (key_first = 0u; key_first < key_count && status == LM_LAUNCH_OK; key_first += block)
		{
			const uint32_t keys = (uint32_t)(key_count - key_first < block ? key_count - key_first : block);
			const uint8_t *latent = x->latent_keys + key_first * Geometry::kSlotBytes;
			if ( LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_QK_NOPE_DIM>(latent,keys,
					(const uint16_t *)shards[x->node].mla_kv_b_key_full_weight + key_rows * K3_KV_LORA_RANK,HEADS,x->expanded,key_pitch,K3_MHA_QK,stream) != cudaSuccess ||
				LmLatentRopeBroadcastLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM>(latent,keys,HEADS,x->expanded,key_pitch,K3_MHA_QK,K3_QK_NOPE_DIM,stream) != cudaSuccess ||
				LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>(latent,keys,
					(const uint16_t *)shards[x->node].mla_kv_b_value_full_weight + value_rows * K3_KV_LORA_RANK,HEADS,x->expanded + (uint64_t)HEADS * K3_MHA_QK * 2u,
					key_pitch,K3_V_HEAD_DIM,stream) != cudaSuccess )
				status = LM_LAUNCH_ERR_LAUNCH;
			else if ( LmKvShardGatherViewInitialize<Expanded>(&view,b->cache_shard.pages,b->kv_shard,x->expanded,x->key_offset,x->key_context) != 0 ||
				LmKvShardGatherViewWindow(&view,(uint32_t)key_first,(uint32_t)key_first + keys) != 0 )
				status = LM_LAUNCH_ERR_SHAPE;
			else if ( LmGqaShardPartialLaunch<Expanded,LmKvShardGatherView,HEADS,K3_MHA_QK,K3_V_HEAD_DIM>(view,x->query,HEADS,x->pass_sequence_of_row,
					x->key_context,x->pass_positions,K3_MLA_QK_SCALE,x->partials,stride,x->pass_rows,stream) != cudaSuccess )
				status = LM_LAUNCH_ERR_LAUNCH;
			else if ( blocks > 1u )
				LM_LAUNCH((K3StreamedFoldRecordsKernel<K3_V_HEAD_DIM>), x->pass_rows, 256u, 0, stream, x->running,x->partials,HEADS,x->pass_rows,key_first == 0u ? 1u : 0u);
		}
		if ( status != LM_LAUNCH_OK )
			break;
		if ( LmGqaShardMergeLaunch<K3_V_HEAD_DIM>(blocks > 1u ? x->running : x->partials,stride,1u,HEADS,0u,x->merged,x->pass_rows,stream) != cudaSuccess ||
			cudaMemcpy2DAsync(x->value_send + (uint64_t)(shard / x->nodes) * rows * value_elements,(size_t)groups * rows * value_elements * sizeof(uint16_t),x->merged,
				(size_t)rows * value_elements * sizeof(uint16_t),(size_t)rows * value_elements * sizeof(uint16_t),x->nodes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			status = LM_LAUNCH_ERR_LAUNCH;
	}
	b->dense_row_offset = own_row_offset;
	b->dense_tile_prefix = own_tile_prefix;
	if ( status != LM_LAUNCH_OK )
		return(status);
	x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL,groups * rows,(uint32_t)value_elements,x->value_send,x->values);
	b->shard_mha = 1u;
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->value_bf16 = x->values + ((uint64_t)(shard % x->nodes) * groups + shard / x->nodes) * rows * value_elements;
		status = K3LayerMlaGated<Format>(b,rows,0u,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedJoinColumns(join->activation,K3_MLA_HEADS * K3_V_HEAD_DIM,shard * (uint32_t)value_elements,b->value_bf16,(uint32_t)value_elements,rows,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedJoinColumns(join->weight,K3_MLA_HEADS * K3_V_HEAD_DIM,shard * (uint32_t)value_elements,b->mla_out_weight,(uint32_t)value_elements,K3_HIDDEN,stream);
	}
	b->shard_mha = 0u;
	b->value_bf16 = own_value;
	if ( status == LM_LAUNCH_OK )
		status = K3Project<LmBf16Format>(b,join->activation,join->weight,0,partial,rows,K3_MLA_HEADS * K3_V_HEAD_DIM,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( boundary != 0u )
		K3PartialSet(b,partial,rows,stream);
	else
		K3PartialAdd(b,partial,rows,stream);
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedLayerAttention(K3LayerBuffers *b, const K3LayerWeights *shards, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaExchange *kda, const K3StreamedMlaExchange *mla, const K3StreamedJoin *join, uint32_t layer, uint16_t *partial, float *hidden_accumulator,
	uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
	int32_t status;
	if ( layer >= K3_LAYERS || (K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT && mla == 0) || (K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT && states == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	if ( layer > 0u )
		K3AttnRes(b,b->attnres_attn_weight,((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
	if ( boundary != 0u )
	{
		if ( layer == 0u )
			K3PartialSet(b,b->hidden_bf16,rows,stream);
		K3BankStore(b,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
	}
	if ( K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT )
		status = K3StreamedMlaNode<Format,Geometry,HEADS>(b,shards,shard_count,mla,partial,join,boundary,rows,multiprocessors,stream);
	else if ( kda != 0 )
		status = K3StreamedKdaNode<Format>(b,shards,states,shard_count,kda,partial,join,boundary,rows,multiprocessors,stream);
	else
		status = K3StreamedKda<Format>(b,shards,states,shard_count,partial,hidden_accumulator,boundary,rows,1u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	K3BindLayer(&shards[0],b);
	K3AttnRes(b,b->attnres_mlp_weight,(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedSlice(K3LayerBuffers *b, const K3LayerWeights *weights, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaExchange *kda, const K3StreamedMlaExchange *mla, const K3StreamedJoin *join,
	uint32_t first_layer, uint32_t layer_count, uint16_t *partial, float *latent_accumulator, float *hidden_accumulator,
	uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset, layer, recurrent = 0u, attention = 0u;
	int32_t status = LM_LAUNCH_OK;
	for (offset = 0u; offset < layer_count && status == LM_LAUNCH_OK; offset++)
	{
		const K3LayerWeights *shards = weights + (uint64_t)offset * shard_count;
		const uint32_t kind = K3_LAYER_KIND(first_layer + offset);
		layer = first_layer + offset;
		status = K3StreamedLayerAttention<Format,Geometry,HEADS>(b,shards,kind == LM_LAYER_RECURRENT ? states + (uint64_t)recurrent * shard_count : 0,shard_count,kda,
			kind != LM_LAYER_RECURRENT && mla != 0 ? mla + attention : 0,join,layer,partial,hidden_accumulator,rows,multiprocessors,stream);
		if ( kind == LM_LAYER_RECURRENT )
			recurrent++;
		else
			attention++;
		if ( status != LM_LAUNCH_OK )
			break;
		status = layer < K3_FIRST_ROUTED_LAYER
			? K3StreamedDenseMlp<Format>(b,shards,shard_count,partial,hidden_accumulator,rows,multiprocessors,stream)
			: K3StreamedMoe<Format>(b,shards,shard_count,1u,join,latent_accumulator,rows,packed_rows,multiprocessors,stream);
	}
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( first_layer + layer_count == K3_LAYERS )
	{
		K3AttnRes(b,b->attnres_out_weight,((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(LM_LAUNCH_OK);
	}
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,LmLaunchGridRows(rows)),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,b->hidden_bf16,rows,K3_HIDDEN);
	return(LM_LAUNCH_OK);
}

struct K3StreamedVocab
{
	const uint16_t *slice[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	uint32_t slice_rows;
	uint32_t slices;
};

__global__ static void K3StreamedEmbedKernel(K3StreamedVocab vocab, const uint32_t *tokens, uint16_t *hidden, uint32_t rows)
{
	const uint32_t row = blockIdx.x, token = row < rows ? tokens[row] : 0u, slice = token / vocab.slice_rows;
	const uint4 *source = slice < vocab.slices ? (const uint4 *)(vocab.slice[slice] + (uint64_t)(token % vocab.slice_rows) * K3_HIDDEN) : 0;
	uint4 *target = (uint4 *)(hidden + (uint64_t)row * K3_HIDDEN), zero = {0u,0u,0u,0u};
	if ( row >= rows )
		return;
	for (uint32_t index = threadIdx.x; index < K3_HIDDEN * sizeof(uint16_t) / 16u; index += blockDim.x)
		target[index] = source != 0 ? source[index] : zero;
}

static int32_t K3StreamedEmbed(const K3StreamedVocab *vocab, const uint32_t *tokens, uint16_t *hidden, uint32_t rows, cudaStream_t stream)
{
	if ( vocab == 0 || vocab->slice_rows == 0u || vocab->slices == 0u || vocab->slices > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE || tokens == 0 || hidden == 0 || rows == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((K3StreamedEmbedKernel), rows, K3_LAYER_THREADS, 0, stream, *vocab,tokens,hidden,rows);
	return(LM_LAUNCH_OK);
}
