#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/ling_resident_decode_stage/source/cuda/unity.cu"
#include "spark_ling_resident_decode_stage_internal.h"

#define SPARK_LING_CUDA_THREADS 256u

__global__ static void SparkLingBoundaryLoadKernel(
	const uint16_t *boundary,
	uint16_t *hidden,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source,destination;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= LING_HIDDEN )
		return;
	source = (first_row + row) * (uint64_t)LING_HIDDEN;
	destination = row * (uint64_t)LING_HIDDEN;
	hidden[destination + element] = boundary[source + element];
}

__global__ static void SparkLingBoundaryStoreKernel(
	const uint16_t *hidden,
	const uint16_t *pending,
	uint16_t *boundary,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source,destination;
	float value;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= LING_HIDDEN )
		return;
	source = row * (uint64_t)LING_HIDDEN;
	destination = (first_row + row) * (uint64_t)LING_HIDDEN;
	value = LmBf16ToFloat(hidden[source + element]) +
		LmBf16ToFloat(pending[source + element]);
	boundary[destination + element] = LmFloatToBf16(value);
}

__global__ static void SparkLingEmbeddingKernel(
	const uint32_t *token_ids,
	const uint16_t *embedding,
	uint16_t *hidden,
	uint32_t row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t element,row,source,destination;
	uint32_t token,vocab_per_rank,rank_offset;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= LING_HIDDEN )
		return;
	vocab_per_rank = LING_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * LING_HIDDEN + element;
	destination = row * (uint64_t)LING_HIDDEN;
	hidden[destination + element] =
		(token >= rank_offset && token < rank_offset + vocab_per_rank)
			? embedding[source] : 0u;
}

__global__ static void SparkLingEmbeddingKernel(
	const uint32_t *token_ids,
	const uint16_t *embedding,
	uint16_t *streams,
	uint32_t row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t element,row,source,stream;
	uint32_t token,vocab_per_rank,rank_offset;
	uint16_t value;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= LING_HIDDEN )
		return;
	vocab_per_rank = LING_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * LING_HIDDEN + element;
	value = (token >= rank_offset && token < rank_offset + vocab_per_rank) ? embedding[source] : 0u;
	for ( stream = 0u; stream < LING_HC; ++stream )
		streams[((row * (uint64_t)LING_HC + stream) * LING_HIDDEN) + element] = value;
}

__global__ static void SparkLingWaveMetadataKernel(
	const uint32_t *resident_slots,
	const uint32_t *positions,
	uint32_t *context_lengths,
	uint32_t *dense_row_offset,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		context_lengths[resident_slots[row]] = positions[row] + 1u;
	if ( row == 0u )
	{
		dense_row_offset[0] = 0u;
		dense_row_offset[1] = row_count;
	}
}

static int32_t SparkLingCudaStatus(cudaError_t status)
{
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ uint32_t SparkLingOrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkLingHeadMaxlocPackKernel(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count,
	uint32_t rank_offset)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SparkLingOrderedHeadScore(scores[row]) << 32u) |
			(UINT32_MAX - (token_ids[row] + rank_offset));
}

static __global__ void SparkLingHeadMaxlocUnpackKernel(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

static __device__ __forceinline__ float2 SparkLingLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkLingStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

static __global__ void SparkLingAccumAddKernel(
	void *destination_bf16,
	const void *source_bf16,
	uint32_t row_count,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = SparkLingLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLingLoadBf16Pair(source_bf16,offset + element);
		SparkLingStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkLingAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkLingLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLingHeadMaxlocPackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(scores,token_ids,maxloc,row_count,rank_offset);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkLingLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLingHeadMaxlocUnpackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(maxloc,token_ids,row_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkLingLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkLingAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkLingLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLingAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

__global__ static void SparkLingKdaResetKernel(
	uint8_t *state_pools, uint64_t state_layer_stride, uint64_t state_slot_bytes,
	uint8_t *q_windows, uint8_t *k_windows, uint8_t *v_windows,
	uint64_t window_layer_stride, uint64_t qk_window_slot_bytes, uint64_t v_window_slot_bytes,
	const uint32_t *state_index, const uint32_t *positions, uint32_t layer_count, uint32_t rows)
{
	uint32_t layer = blockIdx.x, row = blockIdx.y, i;
	uint64_t slot;
	uint8_t *base;
	if ( row >= rows || layer >= layer_count || positions[row] != 0u )
		return;
	slot = state_index[row];
	base = state_pools + (uint64_t)layer * state_layer_stride + slot * state_slot_bytes;
	for ( i = threadIdx.x; i < state_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = q_windows + (uint64_t)layer * window_layer_stride + slot * qk_window_slot_bytes;
	for ( i = threadIdx.x; i < qk_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = k_windows + (uint64_t)layer * window_layer_stride + slot * qk_window_slot_bytes;
	for ( i = threadIdx.x; i < qk_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = v_windows + (uint64_t)layer * window_layer_stride + slot * v_window_slot_bytes;
	for ( i = threadIdx.x; i < v_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
}

static int32_t SparkLingStageWaveMetadata(const SparkLingCudaWave *wave)
{
	SparkLingExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->resident_slots,wave->host_resident_slots,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->positions,wave->host_positions,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && wave->owns_embedding != 0u )
		error = cudaMemcpyAsync(slot->token_ids,wave->host_token_ids,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkLingWaveMetadataKernel<<<(wave->row_count + SPARK_LING_CUDA_THREADS - 1u) / SPARK_LING_CUDA_THREADS,SPARK_LING_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error == cudaSuccess && wave->kda_state_pools != 0 &&
		wave->kda_layer_count != 0u )
	{
		SparkLingKdaResetKernel<<<dim3(wave->kda_layer_count,wave->row_count),SPARK_LING_CUDA_THREADS,0,stream>>>(
			wave->kda_state_pools,wave->kda_state_layer_stride_bytes,
			(uint64_t)SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER,
			wave->kda_q_window_pool,wave->kda_k_window_pool,wave->kda_v_window_pool,
			wave->kda_window_layer_stride_bytes,
			(uint64_t)(SPARK_LING_MODEL_KDA_HEAD_COUNT / wave->tp_degree) * LING_KDA_KEY_DIM * LING_KDA_CONV_KERNEL * 2u,
			(uint64_t)(SPARK_LING_MODEL_KDA_HEAD_COUNT / wave->tp_degree) * LING_KDA_VALUE_DIM * LING_KDA_CONV_KERNEL * 2u,
			wave->kda_state_index,slot->positions,wave->kda_layer_count,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLingCudaStatus(error));
}

static int32_t SparkLingStageWaveBoundary(const SparkLingCudaWave *wave)
{
	SparkLingExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_embedding != 0u )
	{
		SparkLingEmbeddingKernel<<<dim3((LING_HIDDEN + SPARK_LING_CUDA_THREADS - 1u) / SPARK_LING_CUDA_THREADS,wave->row_count),SPARK_LING_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkLingBoundaryLoadKernel<<<dim3((LING_HIDDEN + SPARK_LING_CUDA_THREADS - 1u) / SPARK_LING_CUDA_THREADS,wave->row_count),SPARK_LING_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLingCudaStatus(error));
}

static void SparkLingBuildKvView(
	LmKvView *view,
	uint8_t *pool,
	const SparkLingCudaWave *wave)
{
	view->pool = pool;
	view->page_table = wave->page_table;
	view->page_table_stride = wave->pages_per_sequence;
	view->sequence_count = wave->resident_sequence_capacity;
	view->pool_page_count = wave->resident_sequence_capacity * wave->pages_per_sequence;
	view->access_error = (LmKvAccessError *)wave->slot->kv_access_error;
}

static uint32_t SparkLingKvOrdinalOf(uint32_t layer)
{
	if ( layer >= SPARK_LING_MODEL_LAYER_COUNT ||
	     !SPARK_LING_MODEL_LAYER_IS_MLA(layer) )
		return(UINT32_MAX);
	return(layer / SPARK_LING_MODEL_ATTENTION_PERIOD);
}

static void SparkLingBindLayer(
	const SparkLingCudaWave *wave,
	uint32_t local_layer,
	LingLayerBuffers *buffers)
{
	const SparkLingLayerWeights *weight;
	SparkLingExecutionSlot *slot;
	uint32_t kda_ordinal;
	uint32_t layer;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	layer = wave->first_layer_index + local_layer;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->layer_index = layer;
	buffers->attn_heads = SPARK_LING_MODEL_MLA_HEAD_COUNT / wave->tp_degree;
	buffers->q_b_rows = buffers->attn_heads * (SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION);
	buffers->attn_output_columns = buffers->attn_heads * SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION;
	buffers->dense_gate_up_rows = 2u * SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->head_vocabulary = SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT / wave->tp_degree;
	buffers->kda_heads = SPARK_LING_MODEL_KDA_HEAD_COUNT / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->q_a_weight = weight->q_bf16;
	buffers->kv_a_weight = weight->kv_a_bf16;
	buffers->kv_a_norm_weight = weight->kv_a_norm_bf16;
	{
		uint64_t head_offset = (uint64_t)wave->tp_rank * buffers->attn_heads;
		uint64_t key_head_stride = (uint64_t)LING_LATENT * LING_QK_NOPE_DIM;
		uint64_t value_head_stride = (uint64_t)LING_VALUE_DIM * LING_LATENT;
		buffers->kv_b_key_transposed_weight = (const uint16_t *)weight->kv_b_key_transposed_bf16 + head_offset * key_head_stride;
		buffers->kv_b_value_weight = (const uint16_t *)weight->kv_b_value_bf16 + head_offset * value_head_stride;
	}
	buffers->attn_gate_weight = weight->attn_gate_bf16;
	buffers->qk_scale = SPARK_LING_MODEL_MLA_QK_SCALE;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_weight = weight->dense_gate_up_bf16;
	buffers->dense_up_weight = weight->dense_gate_up_bf16 == 0 ? 0 : (const uint16_t *)weight->dense_gate_up_bf16 + ((uint64_t)LING_DENSE_INTERMEDIATE * LING_HIDDEN);
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->dense_gate_up_fused = weight->dense_gate_up_bf16 != 0 ? 1u : 0u;
	buffers->expert_w1_weight = weight->expert_up_gate_payload;
	buffers->expert_w1_scale = weight->expert_up_gate_scale;
	buffers->expert_w2_weight = weight->expert_down_payload;
	buffers->expert_w2_scale = weight->expert_down_scale;
	buffers->shared_gate_up_weight = weight->shared_gate_up_bf16;
	buffers->shared_down_weight = weight->shared_down_bf16;
	buffers->kda_qkv_beta_weight = weight->kda_qkv_beta_bf16;
	buffers->kda_decay_weight = weight->kda_decay_proj_bf16;
	buffers->kda_gate_weight = weight->kda_gate_proj_bf16;
	buffers->kda_q_conv_weight = weight->kda_q_conv_bf16;
	buffers->kda_k_conv_weight = weight->kda_k_conv_bf16;
	buffers->kda_v_conv_weight = weight->kda_v_conv_bf16;
	buffers->kda_decay_bias = (const float *)weight->kda_decay_bias_f32;
	buffers->kda_head_log_scale = (const float *)weight->kda_head_log_scale_f32;
	buffers->kda_out_norm_weight = weight->kda_out_norm_bf16;
	buffers->kda_out_weight = weight->kda_out_bf16;
	buffers->hidden_bf16 = slot->hidden_bf16;
	buffers->residual_bf16 = slot->residual_bf16;
	buffers->normed_bf16 = slot->normed_bf16;
	buffers->q_bf16 = slot->q_bf16;
	buffers->residual_bf16 = local_layer == 0u ? 0 : slot->attention_out_bf16;
	buffers->query_latent_bf16 = slot->query_latent_bf16;
	buffers->query_rope_bf16 = slot->query_rope_bf16;
	buffers->attn_gate_bf16 = slot->attn_gate_bf16;
	buffers->kv_slot_bf16 = slot->kv_slot_bf16;
	buffers->attention_latent_bf16 = slot->attention_latent_bf16;
	buffers->attention_value_bf16 = slot->attention_value_bf16;
	buffers->attention_out_bf16 = slot->attention_out_bf16;
	buffers->gate_up_bf16 = slot->gate_up_bf16;
	buffers->intermediate_bf16 = slot->intermediate_bf16;
	buffers->expert_out_bf16 = slot->expert_out_bf16;
	buffers->shared_out_bf16 = slot->shared_out_bf16;
	buffers->router_logits = slot->router_logits_f32;
	buffers->attention_split_partials = wave->attention_split_partials_f32;
	buffers->attention_split_partial_blocks = wave->attention_split_partial_blocks;
	buffers->decode_split_context_threshold = wave->decode_split_context_threshold;
	buffers->route_expert = slot->route_expert;
	buffers->route_weight = slot->route_weight;
	buffers->route_source_token = slot->route_source_token;
	buffers->route_packed_row = slot->route_packed_row;
	buffers->head_candidate_score = slot->head_candidate_score;
	buffers->head_candidate_token = slot->head_candidate_token;
	buffers->output_token = slot->output_token;
	buffers->output_score = slot->output_score;
	buffers->group_row_offset = slot->group_row_offset;
	buffers->group_tile_prefix_w1 = slot->group_tile_prefix_w1;
	buffers->group_tile_prefix_w2 = slot->group_tile_prefix_w2;
	buffers->sequence_of_row = slot->resident_slots;
	buffers->context_length = slot->context_lengths;
	buffers->positions = slot->positions;
	buffers->row_positions = slot->positions;
	buffers->fused_qkvb_bf16 = slot->fused_qkvb_bf16;
	buffers->kda_beta_logit = slot->kda_beta_logit;
	buffers->kda_gate_bf16 = slot->kda_gate_bf16;
	buffers->kda_decay_logit_bf16 = slot->kda_decay_logit_bf16;
	buffers->kda_output_bf16 = slot->kda_output_bf16;
	buffers->kda_retention = slot->kda_retention;
	buffers->kda_write_gate = slot->kda_write_gate;
	buffers->kda_state_index = wave->kda_state_index;
	buffers->sequence_row_begin = 0;
	kda_ordinal = wave->kda_ordinal_by_local_layer[local_layer];
	if ( kda_ordinal != UINT32_MAX )
	{
		buffers->kda_state_pool = wave->kda_state_pools + (uint64_t)kda_ordinal * wave->kda_state_layer_stride_bytes;
		buffers->kda_state_slot_bytes = LING_KDA_STATE_BYTES_PER_LAYER;
		buffers->kda_q_window = (uint16_t *)(wave->kda_q_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_k_window = (uint16_t *)(wave->kda_k_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_v_window = (uint16_t *)(wave->kda_v_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
	}
	if ( wave->kv_ordinal_by_local_layer[local_layer] != UINT32_MAX )
		SparkLingBuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)wave->kv_ordinal_by_local_layer[local_layer] * wave->kv_layer_stride_bytes),wave);
}

static int32_t SparkLingValidateWaveShape(const SparkLingCudaWave *wave);

static int32_t SparkLingRunLayerAttention(const SparkLingCudaWave *wave,uint32_t local_layer)
{
	LingLayerBuffers buffers;
	uint32_t layer;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	stream = (cudaStream_t)wave->slot->stream;
	SparkLingBindLayer(wave,local_layer,&buffers);
	if ( SPARK_LING_MODEL_LAYER_IS_KDA(layer) )
	{
		return(LingLayerKda(&buffers,wave->row_count,wave->row_count,1u,wave->multiprocessor_count,stream));
	}
	status = LingLayerAttention(&buffers,wave->row_count,wave->maximum_context,layer,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LM_LAUNCH_OK);
}

static int32_t SparkLingRunLayerMlp(const SparkLingCudaWave *wave,uint32_t local_layer)
{
	LingLayerBuffers buffers;
	uint32_t layer,packed_rows;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * LING_TOP_K;
	stream = (cudaStream_t)wave->slot->stream;
	SparkLingBindLayer(wave,local_layer,&buffers);
	status = layer < LING_FIRST_ROUTED_LAYER ? LingLayerDenseMlp(&buffers,wave->row_count,wave->multiprocessor_count,stream) : LingLayerMoe<LING_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LM_LAUNCH_OK);
}

static int32_t SparkLingRunLayers(const SparkLingCudaWave *wave)
{
	uint32_t local;
	int32_t status;
	for (local=0u; local<wave->layer_count; local++)
	{
		status = SparkLingRunLayerAttention(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkLingRunLayerMlp(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

static int32_t SparkLingRunHead(const SparkLingCudaWave *wave)
{
	SparkLingExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	int32_t status;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_final_head != 0u )
	{
		LingLayerBuffers buffers;
		uint32_t rank_offset;
		SparkLingBindLayer(wave,wave->layer_count - 1u,&buffers);
		rank_offset = wave->tp_rank * buffers.head_vocabulary;
		status = LingHeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		error = SparkLingLaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkLingBoundaryStoreKernel<<<dim3((LING_HIDDEN + SPARK_LING_CUDA_THREADS - 1u) / SPARK_LING_CUDA_THREADS,wave->row_count),SPARK_LING_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,slot->attention_out_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLingCudaStatus(error));
}

static int32_t SparkLingValidateWaveShape(const SparkLingCudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || wave->row_count > wave->resident_sequence_capacity || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkLingLaunchCudaWaveBegin(const SparkLingCudaWave *wave)
{
	int32_t status;
	status = SparkLingValidateWaveShape(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLingStageWaveMetadata(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLingStageWaveBoundary(wave);
	return(status);
}

extern "C" int32_t SparkLingLaunchCudaLayerAttention(const SparkLingCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkLingValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLingRunLayerAttention(wave,local_layer));
}

extern "C" int32_t SparkLingLaunchCudaLayerMlp(const SparkLingCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkLingValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLingRunLayerMlp(wave,local_layer));
}

extern "C" int32_t SparkLingLaunchCudaWaveHead(const SparkLingCudaWave *wave)
{
	int32_t status;
	status = SparkLingValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SparkLingRunHead(wave));
}

extern "C" int32_t SparkLingLaunchCudaWave(const SparkLingCudaWave *wave)
{
	int32_t status;
	status = SparkLingLaunchCudaWaveBegin(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLingRunLayers(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLingRunHead(wave);
	return(status);
}

extern "C" int32_t SparkLingConfigureCudaModule(uint32_t *multiprocessor_count)
{
	cudaDeviceProp properties;
	int32_t device;
	cudaError_t error;
	if ( multiprocessor_count == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	error = cudaGetDevice(&device);
	if ( error == cudaSuccess )
		error = cudaGetDeviceProperties(&properties,device);
	if ( error != cudaSuccess || properties.major != 12 || properties.minor != 1 || properties.multiProcessorCount <= 0 )
		return(LM_LAUNCH_ERR_LAUNCH);
	*multiprocessor_count = (uint32_t)properties.multiProcessorCount;
	return(LM_LAUNCH_OK);
}
