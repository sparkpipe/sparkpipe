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

struct K3StreamedKdaSegments
{
	uint32_t chunks;
	uint16_t *const *q_windows;
	uint16_t *const *k_windows;
	uint16_t *const *v_windows;
	const uint32_t *chunk_row_begin;
	const uint32_t *chunk_slots;
	uint8_t *const *ends;
	uint8_t *const *transforms;
	uint8_t *const *incoming;
	const uint16_t *zeros;
	uint16_t *base;
	uint16_t *correction;
};

template<class Format>
static int32_t K3StreamedKdaSegmented(K3LayerBuffers *b, const K3LayerWeights *shards, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaSegments *segments, uint16_t *partial, float *hidden_accumulator, uint32_t boundary, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN;
	const uint32_t heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint64_t slot_bytes = K3_KDA_RANK_STATE_SLOT_BYTES(heads,0u), matrices = (uint64_t)heads * segments->chunks;
	const uint32_t *row_begin = b->sequence_row_begin, *state_index = b->kda_state_index;
	uint32_t shard, chunk;
	int32_t status = LM_LAUNCH_OK;
	if ( shards == 0 || states == 0 || segments == 0 || segments->chunks < 2u || partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u ||
		heads * shard_count != K3_KDA_HEADS || segments->zeros == 0 || segments->base == 0 || segments->correction == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		float *ends = (float *)segments->ends[shard], *transforms = (float *)segments->transforms[shard], *incoming = (float *)segments->incoming[shard];
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->sequence_row_begin = row_begin;
		b->kda_state_index = state_index;
		b->kda_state_pool = states[shard].state_pool;
		b->kda_q_window = states[shard].q_window;
		b->kda_k_window = states[shard].k_window;
		b->kda_v_window = states[shard].v_window;
		status = K3LayerKdaProject<Format>(b,rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		{
			uint16_t *const windows[3] = {segments->q_windows[shard],segments->k_windows[shard],segments->v_windows[shard]};
			const uint16_t *const inputs[3] = {b->query_bf16,b->key_bf16,b->value_bf16};
			uint16_t *const sequence_windows[3] = {states[shard].q_window,states[shard].k_window,states[shard].v_window};
			const uint32_t channels[3] = {heads * K3_KDA_KEY_DIM,heads * K3_KDA_KEY_DIM,heads * K3_KDA_VALUE_DIM};
			for (uint32_t which = 0u; which < 3u; which++)
			{
				const uint64_t window_bytes = (uint64_t)channels[which] * K3_KDA_CONV_KERNEL * sizeof(uint16_t);
				if ( cudaMemcpyAsync(windows[which],sequence_windows[which],(size_t)window_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
					return(LM_LAUNCH_ERR_LAUNCH);
				K3StreamedConvSeedKernel<<<(uint32_t)(((uint64_t)(segments->chunks - 1u) * channels[which] + 255u) / 256u),256u,0,stream>>>(
					windows[which],inputs[which],segments->chunk_row_begin,channels[which],segments->chunks);
			}
			b->sequence_row_begin = segments->chunk_row_begin;
			b->kda_state_index = segments->chunk_slots;
			b->kda_q_window = windows[0];
			b->kda_k_window = windows[1];
			b->kda_v_window = windows[2];
			status = K3LayerKdaPrepare<Format>(b,rows,segments->chunks,1u,multiprocessors,stream);
			for (uint32_t which = 0u; which < 3u && status == LM_LAUNCH_OK; which++)
			{
				const uint64_t window_bytes = (uint64_t)channels[which] * K3_KDA_CONV_KERNEL * sizeof(uint16_t);
				if ( cudaMemcpyAsync(sequence_windows[which],windows[which] + (uint64_t)(segments->chunks - 1u) * window_bytes / sizeof(uint16_t),(size_t)window_bytes,
						cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
					return(LM_LAUNCH_ERR_LAUNCH);
			}
		}
		if ( status != LM_LAUNCH_OK )
			break;
		if ( cudaMemsetAsync(ends,0,(size_t)(slot_bytes * segments->chunks),stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		b->kda_state_pool = (uint8_t *)ends;
		status = K3LayerKdaRecur(b,b->value_bf16,segments->base,segments->chunks,1u,stream);
		K3StreamedStateIdentityKernel<<<(uint32_t)((matrices * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM + 255u) / 256u),256u,0,stream>>>(transforms,matrices);
		b->kda_state_pool = (uint8_t *)transforms;
		if ( status == LM_LAUNCH_OK )
			status = K3LayerKdaRecur(b,segments->zeros,segments->correction,segments->chunks,1u,stream);
		if ( status != LM_LAUNCH_OK || cudaMemcpyAsync(incoming,states[shard].state_pool,(size_t)slot_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			return(status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_LAUNCH);
		for (chunk = 1u; chunk < segments->chunks; chunk++)
			K3StreamedStateComposeKernel<<<dim3(heads,K3_KDA_KEY_DIM),K3_KDA_VALUE_DIM,0,stream>>>(
				transforms + (uint64_t)(chunk - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_KEY_DIM,
				incoming + (uint64_t)(chunk - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM,
				ends + (uint64_t)(chunk - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM,
				incoming + (uint64_t)chunk * heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM);
		K3StreamedStateComposeKernel<<<dim3(heads,K3_KDA_KEY_DIM),K3_KDA_VALUE_DIM,0,stream>>>(
			transforms + (uint64_t)(segments->chunks - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_KEY_DIM,
			incoming + (uint64_t)(segments->chunks - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM,
			ends + (uint64_t)(segments->chunks - 1u) * heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM,
			(float *)states[shard].state_pool);
		b->kda_state_pool = (uint8_t *)incoming;
		status = K3LayerKdaRecur(b,segments->zeros,segments->correction,segments->chunks,0u,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),dim3((heads * K3_KDA_VALUE_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),K3_LAYER_THREADS,0,stream,
			segments->base,segments->correction,b->attention_out_bf16,rows,heads * K3_KDA_VALUE_DIM);
		status = K3LayerKdaFinish<Format>(b,rows,partial,(uint16_t *)0,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,partial,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	b->sequence_row_begin = row_begin;
	b->kda_state_index = state_index;
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

#define K3_STREAMED_KDA_TAILS 1u
#define K3_STREAMED_KDA_SUMMARIES 2u

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
	uint16_t *tails;
	uint8_t *summaries;
	float *incoming;
	float *running;
	uint16_t *q_window;
	uint16_t *k_window;
	uint16_t *v_window;
	const uint32_t *row_begin;
	const uint32_t *slot;
	const uint16_t *zeros;
	uint16_t *base;
	uint16_t *correction;
	void (*exchange)(void *context, void *stream, uint32_t phase, uint32_t shard);
	void *context;
};

template<class Format>
static int32_t K3StreamedKdaNode(K3LayerBuffers *b, const K3LayerWeights *shards, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaExchange *x, uint16_t *partial, float *hidden_accumulator, uint32_t boundary, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN;
	const uint32_t heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint32_t channels[3] = {heads * K3_KDA_KEY_DIM,heads * K3_KDA_KEY_DIM,heads * K3_KDA_VALUE_DIM};
	const uint64_t window_offset[3] = {0u,(uint64_t)channels[0] * K3_KDA_CONV_KERNEL,(uint64_t)(channels[0] + channels[1]) * K3_KDA_CONV_KERNEL};
	const uint64_t tail_elements = (uint64_t)(channels[0] + channels[1] + channels[2]) * K3_KDA_CONV_KERNEL;
	const uint64_t slot_floats = (uint64_t)heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM, slot_bytes = slot_floats * sizeof(float);
	const uint32_t *row_begin = b->sequence_row_begin, *state_index = b->kda_state_index;
	uint32_t shard;
	int32_t status = LM_LAUNCH_OK;
	if ( shards == 0 || states == 0 || x == 0 || x->exchange == 0 || x->nodes == 0u || x->node >= x->nodes || rows < K3_KDA_CONV_KERNEL ||
		partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u || heads * shard_count != K3_KDA_HEADS || K3_KDA_KEY_DIM != K3_KDA_VALUE_DIM ||
		x->tails == 0 || x->summaries == 0 || x->incoming == 0 || x->running == 0 || x->zeros == 0 || x->base == 0 || x->correction == 0 ||
		x->row_begin == 0 || x->slot == 0 || x->q_window == 0 || x->k_window == 0 || x->v_window == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		uint16_t *const windows[3] = {x->q_window,x->k_window,x->v_window};
		uint16_t *const carried[3] = {states[shard].q_window,states[shard].k_window,states[shard].v_window};
		uint16_t *const own_tail = x->tails + ((uint64_t)x->node * shard_count + shard) * tail_elements;
		const uint16_t *const last_tail = x->tails + ((uint64_t)(x->nodes - 1u) * shard_count + shard) * tail_elements;
		float *const own = (float *)x->summaries + ((uint64_t)x->node * shard_count + shard) * 2u * slot_floats;
		uint32_t which, node, current = 0u;
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->sequence_row_begin = x->row_begin;
		b->kda_state_index = x->slot;
		status = K3LayerKdaProject<Format>(b,rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		{
			const uint16_t *const inputs[3] = {b->query_bf16,b->key_bf16,b->value_bf16};
			for (which = 0u; which < 3u; which++)
				K3StreamedConvTailKernel<<<(channels[which] + 255u) / 256u,256u,0,stream>>>(own_tail + window_offset[which],inputs[which],rows,channels[which]);
		}
		x->exchange(x->context,(void *)(uintptr_t)stream,K3_STREAMED_KDA_TAILS,shard);
		for (which = 0u; which < 3u; which++)
		{
			const uint16_t *seed = x->node == 0u ? carried[which] : x->tails + ((uint64_t)(x->node - 1u) * shard_count + shard) * tail_elements + window_offset[which];
			if ( cudaMemcpyAsync(windows[which],seed,(size_t)channels[which] * K3_KDA_CONV_KERNEL * sizeof(uint16_t),cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
				return(LM_LAUNCH_ERR_LAUNCH);
		}
		b->kda_q_window = windows[0];
		b->kda_k_window = windows[1];
		b->kda_v_window = windows[2];
		status = K3LayerKdaPrepare<Format>(b,rows,1u,1u,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK || cudaMemsetAsync(own,0,(size_t)slot_bytes,stream) != cudaSuccess )
			return(status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_LAUNCH);
		b->kda_state_pool = (uint8_t *)own;
		status = K3LayerKdaRecur(b,b->value_bf16,x->base,1u,1u,stream);
		K3StreamedStateIdentityKernel<<<(uint32_t)((slot_floats + 255u) / 256u),256u,0,stream>>>(own + slot_floats,heads);
		b->kda_state_pool = (uint8_t *)(own + slot_floats);
		if ( status == LM_LAUNCH_OK )
			status = K3LayerKdaRecur(b,x->zeros,x->correction,1u,1u,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		x->exchange(x->context,(void *)(uintptr_t)stream,K3_STREAMED_KDA_SUMMARIES,shard);
		if ( cudaMemcpyAsync(x->running,states[shard].state_pool,(size_t)slot_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		for (node = 0u; node < x->nodes; node++)
		{
			const float *summary = (const float *)x->summaries + ((uint64_t)node * shard_count + shard) * 2u * slot_floats;
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
		b->kda_state_pool = (uint8_t *)x->incoming;
		status = K3LayerKdaRecur(b,x->zeros,x->correction,1u,0u,stream);
		if ( status != LM_LAUNCH_OK )
			break;
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),dim3((channels[2] + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),K3_LAYER_THREADS,0,stream,
			x->base,x->correction,b->attention_out_bf16,rows,channels[2]);
		status = K3LayerKdaFinish<Format>(b,rows,partial,(uint16_t *)0,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,partial,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	b->sequence_row_begin = row_begin;
	b->kda_state_index = state_index;
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

#define K3_STREAMED_MLA_LATENTS 3u
#define K3_STREAMED_MLA_VALUES 4u

struct K3StreamedMlaExchange
{
	uint32_t node;
	uint32_t nodes;
	uint32_t pass_rows;
	uint32_t first_row;
	uint32_t first_key;
	uint32_t key_capacity;
	uint16_t *q_latents;
	uint8_t *latent_keys;
	uint16_t *query;
	uint8_t *expanded;
	float *partials;
	uint16_t *values;
	const uint32_t *pass_positions;
	const uint32_t *pass_sequence_of_row;
	const uint32_t *pass_row_offset;
	uint32_t *pass_tile_prefix;
	const uint32_t *key_offset;
	const uint32_t *key_context;
	void (*exchange)(void *context, void *stream, uint32_t phase, uint32_t shard);
	void *context;
};

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedMlaNode(K3LayerBuffers *b, const K3LayerWeights *shards, uint32_t shard_count, const K3StreamedMlaExchange *x,
	uint16_t *partial, float *hidden_accumulator, uint32_t boundary, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	typedef K3StreamedHeadKv<HEADS,Geometry> Expanded;
	const uint64_t hidden_elements = (uint64_t)rows * K3_HIDDEN, key_pitch = (uint64_t)HEADS * (K3_MHA_QK + K3_V_HEAD_DIM);
	const uint64_t value_elements = (uint64_t)HEADS * K3_V_HEAD_DIM;
	uint16_t *const own_value = b->value_bf16;
	const uint32_t *const own_row_offset = b->dense_row_offset;
	uint32_t *const own_tile_prefix = b->dense_tile_prefix;
	LmKvShardGatherView view;
	uint64_t stride, key_count;
	uint32_t shard;
	int32_t status;
	if ( shards == 0 || x == 0 || x->exchange == 0 || x->nodes == 0u || x->node >= x->nodes || shard_count * HEADS != K3_MLA_HEADS ||
		partial == 0 || hidden_accumulator == 0 || b->tp_sharded == 0u || b->kv_shard.degree != 1u || rows == 0u || x->first_row + rows > x->pass_rows ||
		x->q_latents == 0 || x->latent_keys == 0 || x->query == 0 || x->expanded == 0 || x->partials == 0 || x->values == 0 ||
		x->pass_positions == 0 || x->pass_sequence_of_row == 0 || x->pass_row_offset == 0 || x->pass_tile_prefix == 0 || x->key_offset == 0 || x->key_context == 0 ||
		shards[0].mla_kv_b_key_full_weight == 0 || shards[0].mla_kv_b_value_full_weight == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	key_count = (uint64_t)x->first_key + x->pass_rows;
	stride = (uint64_t)x->pass_rows * LmGqaShardRecordFloats(HEADS,K3_V_HEAD_DIM,1u);
	if ( key_count > x->key_capacity )
		return(LM_LAUNCH_ERR_SHAPE);
	K3BindLayer(&shards[0],b);
	status = K3LayerMlaDown<Format>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	if ( cudaMemcpyAsync(x->q_latents + (uint64_t)x->first_row * K3_Q_LORA_RANK,b->latent_bf16,(size_t)rows * K3_Q_LORA_RANK * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,stream) != cudaSuccess ||
		cudaMemcpyAsync(x->latent_keys + ((uint64_t)x->first_key + x->first_row) * Geometry::kSlotBytes,b->kv_slot_bf16,(size_t)rows * Geometry::kSlotBytes,
			cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	x->exchange(x->context,(void *)(uintptr_t)stream,K3_STREAMED_MLA_LATENTS,0u);
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
		if ( LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_QK_NOPE_DIM>(x->latent_keys,(uint32_t)key_count,
				(const uint16_t *)shards[0].mla_kv_b_key_full_weight + key_rows * K3_KV_LORA_RANK,HEADS,x->expanded,key_pitch,K3_MHA_QK,stream) != cudaSuccess ||
			LmLatentRopeBroadcastLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM>(x->latent_keys,(uint32_t)key_count,HEADS,x->expanded,key_pitch,K3_MHA_QK,K3_QK_NOPE_DIM,stream) != cudaSuccess ||
			LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>(x->latent_keys,(uint32_t)key_count,
				(const uint16_t *)shards[0].mla_kv_b_value_full_weight + value_rows * K3_KV_LORA_RANK,HEADS,x->expanded + (uint64_t)HEADS * K3_MHA_QK * 2u,
				key_pitch,K3_V_HEAD_DIM,stream) != cudaSuccess )
			status = LM_LAUNCH_ERR_LAUNCH;
		else if ( LmKvShardGatherViewInitialize<Expanded>(&view,b->cache_shard.pages,b->kv_shard,x->expanded,x->key_offset,x->key_context) != 0 )
			status = LM_LAUNCH_ERR_SHAPE;
		else if ( LmGqaShardPartialLaunch<Expanded,LmKvShardGatherView,HEADS,K3_MHA_QK,K3_V_HEAD_DIM>(view,x->query,HEADS,x->pass_sequence_of_row,
				x->key_context,x->pass_positions,K3_MLA_QK_SCALE,x->partials,stride,x->pass_rows,stream) != cudaSuccess ||
			LmGqaShardMergeLaunch<K3_V_HEAD_DIM>(x->partials,stride,1u,HEADS,0u,x->values + (uint64_t)shard * x->pass_rows * value_elements,x->pass_rows,stream) != cudaSuccess )
			status = LM_LAUNCH_ERR_LAUNCH;
	}
	b->dense_row_offset = own_row_offset;
	b->dense_tile_prefix = own_tile_prefix;
	if ( status != LM_LAUNCH_OK )
		return(status);
	x->exchange(x->context,(void *)(uintptr_t)stream,K3_STREAMED_MLA_VALUES,0u);
	b->shard_mha = 1u;
	for (shard = 0u; shard < shard_count && status == LM_LAUNCH_OK; shard++)
	{
		K3BindLayer(&shards[shard],b);
		b->tp_rank = shard;
		b->value_bf16 = x->values + ((uint64_t)shard * x->pass_rows + x->first_row) * value_elements;
		status = K3LayerMlaOutput<Format>(b,rows,(uint16_t *)0,0u,multiprocessors,stream);
		if ( status == LM_LAUNCH_OK )
			status = K3StreamedAccumulate(hidden_accumulator,b->attention_out_bf16,hidden_elements,shard == 0u ? 1u : 0u,stream);
	}
	b->shard_mha = 0u;
	b->value_bf16 = own_value;
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

template<class Format, class Geometry, uint32_t HEADS>
static int32_t K3StreamedSlice(K3LayerBuffers *b, const K3LayerWeights *weights, const K3StreamedKdaState *states, uint32_t shard_count,
	const K3StreamedKdaExchange *kda, const K3StreamedMlaExchange *mla,
	uint32_t first_layer, uint32_t layer_count, uint16_t *partial, float *latent_accumulator, float *hidden_accumulator,
	uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset, layer, boundary, recurrent = 0u, attention = 0u;
	int32_t status = LM_LAUNCH_OK;
	for (offset = 0u; offset < layer_count && status == LM_LAUNCH_OK; offset++)
	{
		const K3LayerWeights *shards = weights + (uint64_t)offset * shard_count;
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS || (K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT && mla == 0) )
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
		if ( K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT )
			status = K3StreamedMlaNode<Format,Geometry,HEADS>(b,shards,shard_count,mla + attention++,partial,hidden_accumulator,boundary,rows,multiprocessors,stream);
		else if ( kda != 0 )
			status = K3StreamedKdaNode<Format>(b,shards,states + (uint64_t)recurrent++ * shard_count,shard_count,kda,partial,hidden_accumulator,boundary,rows,
				multiprocessors,stream);
		else
			status = K3StreamedKda<Format>(b,shards,states + (uint64_t)recurrent++ * shard_count,shard_count,partial,hidden_accumulator,boundary,rows,1u,
				multiprocessors,stream);
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
