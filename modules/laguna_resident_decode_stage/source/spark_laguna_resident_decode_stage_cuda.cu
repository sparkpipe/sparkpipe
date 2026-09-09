#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/laguna_resident_decode_stage/source/cuda/unity.cu"
#include "spark_laguna_resident_decode_stage_internal.h"
#include "inference/kernels/tp_reduce.cuh"
#include "sparkpipe/spark_tp_device_collective.h"

#define SPARK_LAGUNA_CUDA_THREADS 256u

__global__ static void SparkLagunaBoundaryLoadKernel(
	const uint16_t *boundary,
	uint16_t *streams,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source,width;
	width = ((uint64_t)LAGUNA_HC * LAGUNA_HIDDEN);
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= width )
		return;
	source = ((first_row + row) * width);
	streams[(row * width) + element] = boundary[source + element];
}

__global__ static void SparkLagunaBoundaryStoreKernel(
	const uint16_t *streams,
	uint16_t *boundary,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,destination,width;
	width = ((uint64_t)LAGUNA_HC * LAGUNA_HIDDEN);
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= width )
		return;
	destination = ((first_row + row) * width);
	boundary[destination + element] = streams[(row * width) + element];
}

__global__ static void SparkLagunaEmbeddingKernel(
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
	if ( row >= row_count || element >= LAGUNA_HIDDEN )
		return;
	vocab_per_rank = LAGUNA_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * LAGUNA_HIDDEN + element;
	value = (token >= rank_offset && token < rank_offset + vocab_per_rank) ? embedding[source] : 0u;
	for ( stream = 0u; stream < LAGUNA_HC; ++stream )
		streams[((row * (uint64_t)LAGUNA_HC + stream) * LAGUNA_HIDDEN) + element] = value;
}

__global__ static void SparkLagunaWaveMetadataKernel(
	const uint32_t *resident_slots,
	const uint32_t *positions,
	uint32_t *context_lengths,
	uint32_t *dense_row_offset,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
	{
		for ( row = 0u; row < row_count; ++row )
		{
			uint32_t slot = resident_slots[row];
			uint32_t len = positions[row] + 1u;
			if ( len > context_lengths[slot] )
				context_lengths[slot] = len;
		}
		dense_row_offset[0] = 0u;
		dense_row_offset[1] = row_count;
	}
}

static int32_t SparkLagunaCudaStatus(cudaError_t status)
{
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ uint32_t SparkLagunaOrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkLagunaHeadMaxlocPackKernel(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count,
	uint32_t rank_offset)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SparkLagunaOrderedHeadScore(scores[row]) << 32u) |
			(UINT32_MAX - (token_ids[row] + rank_offset));
}

static __global__ void SparkLagunaHeadMaxlocUnpackKernel(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

static __device__ __forceinline__ float2 SparkLagunaLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkLagunaStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

static __global__ void SparkLagunaAccumAddKernel(
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
		destination_pair = SparkLagunaLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLagunaLoadBf16Pair(source_bf16,offset + element);
		SparkLagunaStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkLagunaAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkLagunaLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLagunaHeadMaxlocPackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(scores,token_ids,maxloc,row_count,rank_offset);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkLagunaLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLagunaHeadMaxlocUnpackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(maxloc,token_ids,row_count);
	return(cudaPeekAtLastError());
}

static uint32_t SparkLagunaProbeReduction(cudaStream_t stream,const uint16_t *const *ranks,uint32_t local_rank,uint32_t rows,uint32_t width)
{
	static uint32_t count = 0u;
	uint32_t rank;
	char label[32];
	if ( local_rank != 0u || rows != 1u || width != LAGUNA_HIDDEN || count >= 2u || getenv("SPARK_LAGUNA_PROBE_VEC") == 0 || LagunaKdaProbeVecLayer(0) != 0 )
		return(0u);
	count++;
	for (rank=0u; rank<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT; rank++)
		if ( ranks[rank] != 0 )
		{
			snprintf(label,sizeof(label),"reduce_rank%u",rank);
			LagunaProbeVecU16(stream,ranks[rank],width,0u,count,label);
		}
	return(count);
}

extern "C" cudaError_t SparkLagunaLaunchDirectSum(cudaStream_t stream,void *destination,const void *const *rank_devices,uint32_t local_rank,uint32_t rows,uint32_t width)
{
	LmTpBf16Contributions<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT> inputs = {};
	uint32_t rank,pass;
	cudaError_t error;
	if ( destination == 0 || rank_devices == 0 || local_rank >= SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT || rows == 0u || width == 0u )
		return(cudaErrorInvalidValue);
	for (rank=0u; rank<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT; rank++)
		inputs.rank[rank] = (const uint16_t *)(rank == local_rank ? destination : rank_devices[rank]);
	pass = SparkLagunaProbeReduction(stream,inputs.rank,local_rank,rows,width);
	LmTpBf16SumKernel<<<rows,256u,0u,stream>>>((uint16_t *)destination,inputs,rows,width);
	error = cudaPeekAtLastError();
	if ( error == cudaSuccess && pass != 0u )
		LagunaProbeVecU16(stream,(const uint16_t *)destination,width,0u,pass,"reduce_result");
	return(error);
}

extern "C" cudaError_t SparkLagunaLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkLagunaAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkLagunaLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkLagunaAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

__global__ static void SparkLagunaKdaResetKernel(
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

static int32_t SparkLagunaStageWaveMetadata(const SparkLagunaCudaWave *wave)
{
	SparkLagunaExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->resident_slots,wave->host_resident_slots,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->positions,wave->host_positions,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && wave->run_count != 0u )
	{
		error = cudaMemcpyAsync(slot->run_begin,wave->host_sequence_row_begin,((uint64_t)wave->run_count + 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(slot->run_state_index,wave->host_run_state_index,(uint64_t)wave->run_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess && wave->owns_embedding != 0u )
		error = cudaMemcpyAsync(slot->token_ids,wave->host_token_ids,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkLagunaWaveMetadataKernel<<<(wave->row_count + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error == cudaSuccess && wave->kda_state_pools != 0 &&
		wave->kda_layer_count != 0u )
	{
		SparkLagunaKdaResetKernel<<<dim3(wave->kda_layer_count,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(
			wave->kda_state_pools,wave->kda_state_layer_stride_bytes,
			(uint64_t)SPARK_LAGUNA_MODEL_KDA_STATE_BYTES_PER_LAYER / wave->tp_degree,
			wave->kda_q_window_pool,wave->kda_k_window_pool,wave->kda_v_window_pool,
			wave->kda_window_layer_stride_bytes,
			(uint64_t)(SPARK_LAGUNA_MODEL_KDA_HEAD_COUNT / wave->tp_degree) * LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_CONV_KERNEL * 2u,
			(uint64_t)(SPARK_LAGUNA_MODEL_KDA_HEAD_COUNT / wave->tp_degree) * LAGUNA_KDA_VALUE_DIM * LAGUNA_KDA_CONV_KERNEL * 2u,
			slot->resident_slots,slot->positions,wave->kda_layer_count,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLagunaCudaStatus(error));
}

static int32_t SparkLagunaStageWaveBoundary(const SparkLagunaCudaWave *wave)
{
	SparkLagunaExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_embedding != 0u )
	{
		SparkLagunaEmbeddingKernel<<<dim3((LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkLagunaBoundaryLoadKernel<<<dim3((LAGUNA_HC * LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_input == 0u || wave->maximum_context <= LAGUNA_DSA_SELECTED )
		return(SparkLagunaCudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)LAGUNA_DSA_SELECTED;
	error = cudaMemcpyAsync(slot->selected_positions,(const uint32_t *)wave->sideband_input_u32 + sideband_offset,(uint64_t)wave->row_count * LAGUNA_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	return(SparkLagunaCudaStatus(error));
}

static void SparkLagunaBuildKvView(
	LmKvView *view,
	uint8_t *pool,
	const SparkLagunaCudaWave *wave)
{
	view->pool = pool;
	view->page_table = wave->page_table;
	view->page_table_stride = wave->pages_per_sequence;
	view->sequence_count = wave->resident_sequence_capacity;
	view->pool_page_count = wave->resident_sequence_capacity * wave->pages_per_sequence;
	view->access_error = (LmKvAccessError *)wave->slot->kv_access_error;
}

static uint32_t index_ordinal_of(const SparkLagunaCudaWave *wave,uint32_t local_layer,uint32_t layer)
{
	(void)wave;
	(void)local_layer;
	if ( layer < 3u || layer >= SPARK_LAGUNA_MODEL_LAYER_COUNT ||
	     SPARK_LAGUNA_MODEL_LAYER_IS_KDA(layer) )
		return(UINT32_MAX);
	return((layer - 3u) / SPARK_LAGUNA_MODEL_ATTENTION_PERIOD);
}

static void SparkLagunaBindLayer(
	const SparkLagunaCudaWave *wave,
	uint32_t local_layer,
	LagunaLayerBuffers *buffers)
{
	const SparkLagunaLayerWeights *weight;
	SparkLagunaExecutionSlot *slot;
	uint32_t index_ordinal;
	uint32_t kda_ordinal;
	uint32_t layer;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	layer = wave->first_layer_index + local_layer;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->layer_index = layer;
	buffers->attn_heads = SPARK_LAGUNA_MODEL_MLA_HEAD_COUNT / wave->tp_degree;
	buffers->q_b_rows = buffers->attn_heads * (SPARK_LAGUNA_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_LAGUNA_MODEL_MLA_QK_ROPE_HEAD_DIMENSION);
	buffers->attn_output_columns = buffers->attn_heads * SPARK_LAGUNA_MODEL_MLA_VALUE_HEAD_DIMENSION;
	buffers->dense_gate_up_rows = 2u * SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->head_vocabulary = SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT / wave->tp_degree;
	buffers->kda_heads = SPARK_LAGUNA_MODEL_KDA_HEAD_COUNT / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->q_a_weight = weight->q_a_bf16;
	buffers->q_a_norm_weight = weight->q_a_norm_bf16;
	buffers->q_b_weight = weight->q_b_bf16;
	buffers->kv_a_weight = weight->kv_a_bf16;
	buffers->kv_a_norm_weight = weight->kv_a_norm_bf16;
	{
		uint64_t head_offset = (uint64_t)wave->tp_rank * buffers->attn_heads;
		uint64_t key_head_stride = (uint64_t)LAGUNA_LATENT * LAGUNA_QK_NOPE_DIM;
		uint64_t value_head_stride = (uint64_t)LAGUNA_VALUE_DIM * LAGUNA_LATENT;
		buffers->kv_b_key_transposed_weight = (const uint16_t *)weight->kv_b_key_transposed_bf16 + head_offset * key_head_stride;
		buffers->kv_b_value_weight = (const uint16_t *)weight->kv_b_value_bf16 + head_offset * value_head_stride;
	}
	buffers->index_q_weight = weight->index_q_bf16;
	buffers->index_k_weight = weight->index_k_bf16;
	buffers->index_head_weight = weight->index_head_bf16;
	buffers->index_norm_weight = weight->index_norm_weight_bf16;
	buffers->index_norm_bias = weight->index_norm_bias_bf16;
	buffers->index_compress_ape = (const float *)weight->index_compress_ape_f32;
	buffers->index_compress_gate = weight->index_compress_gate_bf16;
	buffers->qk_scale = SPARK_LAGUNA_MODEL_MLA_QK_SCALE;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_weight = weight->dense_gate_up_bf16;
	buffers->dense_up_weight = weight->dense_gate_up_bf16 == 0 ? 0 : (const uint16_t *)weight->dense_gate_up_bf16 + ((uint64_t)LAGUNA_DENSE_INTERMEDIATE * LAGUNA_HIDDEN);
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->dense_gate_up_fused = weight->dense_gate_up_bf16 != 0 ? 1u : 0u;
	buffers->expert_w1_weight = weight->expert_up_gate_payload;
	buffers->expert_w1_scale = weight->expert_up_gate_scale;
	buffers->expert_w2_weight = weight->expert_down_payload;
	buffers->expert_w2_scale = weight->expert_down_scale;
	if ( wave->lazy_experts != 0u )
	{
		buffers->expert_w1_weight = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_up_gate_payload_offset;
		buffers->expert_w1_scale = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_up_gate_scale_offset;
		buffers->expert_w2_weight = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_down_payload_offset;
		buffers->expert_w2_scale = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_down_scale_offset;
	}
	buffers->shared_gate_up_weight = weight->shared_gate_up_bf16;
	buffers->shared_down_weight = weight->shared_down_bf16;
	buffers->kda_qkv_beta_weight = weight->kda_qkv_beta_bf16;
	buffers->kda_decay_gate_down_weight = weight->kda_decay_gate_down_bf16;
	buffers->kda_decay_up_weight = weight->kda_decay_up_bf16;
	buffers->kda_gate_up_weight = weight->kda_gate_up_bf16;
	buffers->kda_q_conv_weight = weight->kda_q_conv_bf16;
	buffers->kda_k_conv_weight = weight->kda_k_conv_bf16;
	buffers->kda_v_conv_weight = weight->kda_v_conv_bf16;
	buffers->kda_decay_bias = (const float *)weight->kda_decay_bias_f32;
	buffers->kda_head_log_scale = (const float *)weight->kda_head_log_scale_f32;
	buffers->kda_out_norm_weight = weight->kda_out_norm_bf16;
	buffers->kda_out_weight = weight->kda_out_bf16;
	buffers->hc_attn_fn = (const float *)weight->hc_attn_fn_f32;
	buffers->hc_attn_base = (const float *)weight->hc_attn_base_f32;
	buffers->hc_attn_scale = (const float *)weight->hc_attn_scale_f32;
	buffers->hc_ffn_fn = (const float *)weight->hc_ffn_fn_f32;
	buffers->hc_ffn_base = (const float *)weight->hc_ffn_base_f32;
	buffers->hc_ffn_scale = (const float *)weight->hc_ffn_scale_f32;
	buffers->hidden_bf16 = slot->hidden_bf16;
	buffers->residual_bf16 = slot->residual_bf16;
	buffers->normed_bf16 = slot->normed_bf16;
	buffers->q_compressed_bf16 = slot->q_compressed_bf16;
	buffers->q_bf16 = slot->q_bf16;
	buffers->query_latent_bf16 = slot->query_latent_bf16;
	buffers->query_rope_bf16 = slot->query_rope_bf16;
	buffers->index_query_bf16 = slot->index_query_bf16;
	buffers->index_key_bf16 = slot->index_key_bf16;
	buffers->index_gate_bf16 = slot->index_gate_bf16;
	buffers->index_packed_bf16 = slot->index_packed_bf16;
	buffers->selected_pools = slot->selected_pools;
	buffers->index_head_weight_bf16 = slot->index_head_weight_bf16;
	buffers->kv_slot_bf16 = slot->kv_slot_bf16;
	buffers->attention_latent_bf16 = slot->attention_latent_bf16;
	buffers->attention_value_bf16 = slot->attention_value_bf16;
	buffers->attention_out_bf16 = slot->attention_out_bf16;
	buffers->gate_up_bf16 = slot->gate_up_bf16;
	buffers->intermediate_bf16 = slot->intermediate_bf16;
	buffers->expert_out_bf16 = slot->expert_out_bf16;
	buffers->shared_out_bf16 = slot->shared_out_bf16;
	buffers->router_logits = slot->router_logits_f32;
	buffers->selection_scores = slot->selection_scores_f32;
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
	buffers->selected_positions = slot->selected_positions;
	buffers->selected_position_count = LAGUNA_DSA_SELECTED;
	buffers->fused_qkvb_bf16 = slot->fused_qkvb_bf16;
	buffers->fused_decay_gate_bf16 = slot->fused_decay_gate_bf16;
	buffers->kda_decay_latent_bf16 = slot->kda_decay_latent_bf16;
	buffers->kda_gate_latent_bf16 = slot->kda_gate_latent_bf16;
	buffers->kda_beta_logit = slot->kda_beta_logit;
	buffers->kda_gate_bf16 = slot->kda_gate_bf16;
	buffers->kda_decay_logit_bf16 = slot->kda_decay_logit_bf16;
	buffers->kda_output_bf16 = slot->kda_output_bf16;
	buffers->kda_retention = slot->kda_retention;
	buffers->kda_write_gate = slot->kda_write_gate;
	buffers->kda_state_index = wave->run_count != 0u ? wave->run_state_index : wave->kda_state_index;
	buffers->sequence_row_begin = wave->sequence_row_begin;
	kda_ordinal = wave->kda_ordinal_by_local_layer[local_layer];
	if ( kda_ordinal != UINT32_MAX )
	{
		buffers->kda_state_pool = wave->kda_state_pools + (uint64_t)kda_ordinal * wave->kda_state_layer_stride_bytes;
		buffers->kda_state_slot_bytes = LAGUNA_KDA_STATE_BYTES_PER_LAYER / wave->tp_degree;
		buffers->kda_q_window = (uint16_t *)(wave->kda_q_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_k_window = (uint16_t *)(wave->kda_k_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_v_window = (uint16_t *)(wave->kda_v_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		if ( wave->mtp_verify != 0u && slot->kda_replay_pool != 0 )
		{
			buffers->kda_replay_layer = slot->kda_replay_pool + (uint64_t)kda_ordinal * wave->kda_replay_layer_bytes;
			buffers->kda_replay_steps = wave->row_count;
		}
	}
	buffers->hc_mixes_f32 = slot->hc_mixes_f32;
	buffers->hc_pre_f32 = slot->hc_pre_f32;
	buffers->hc_post_f32 = slot->hc_post_f32;
	buffers->hc_comb_f32 = slot->hc_comb_f32;
	buffers->hc_collapsed_bf16 = slot->hc_collapsed_bf16;
	buffers->hc_snapshot_bf16 = slot->hc_snapshot_bf16;
	buffers->hc_mean_bf16 = slot->hc_mean_bf16;
	index_ordinal = wave->index_ordinal_by_local_layer[local_layer];
	if ( index_ordinal != UINT32_MAX )
		SparkLagunaBuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)index_ordinal * wave->kv_layer_stride_bytes),wave);
	{
		uint32_t dsa_ordinal = index_ordinal_of(wave,local_layer,layer);
		if ( dsa_ordinal != UINT32_MAX )
			SparkLagunaBuildKvView(&buffers->index_cache,wave->index_cache + ((uint64_t)dsa_ordinal * wave->index_layer_stride_bytes),wave);
	}
}

static int32_t SparkLagunaValidateWaveShape(const SparkLagunaCudaWave *wave);

static int32_t SparkLagunaRunLayerAttention(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	LagunaLayerBuffers buffers;
	uint32_t layer;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	stream = (cudaStream_t)wave->slot->stream;
	SparkLagunaBindLayer(wave,local_layer,&buffers);
	status = LagunaHcSite(&buffers,buffers.hc_attn_fn,buffers.hc_attn_base,buffers.hc_attn_scale,wave->row_count,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( wave->tp_rank == 0u && LagunaKdaProbeDeep(&buffers) )
	{
		LAGUNA_KDA_PROBE_RAW(stream,layer,"attnsite_collapsed",buffers.hc_collapsed_bf16);
		LAGUNA_KDA_PROBE_RAW(stream,layer,"attnsite_attn_norm_weight",buffers.attn_norm_weight);
	}
	if ( SPARK_LAGUNA_MODEL_LAYER_IS_KDA(layer) )
	{
		return(LagunaLayerKda(&buffers,wave->row_count,wave->run_count != 0u ? wave->run_count : wave->row_count,wave->commit,wave->multiprocessor_count,stream));
	}
	status = LagunaLayerAttention(&buffers,wave->row_count,wave->maximum_context,layer,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LM_LAUNCH_OK);
}

static int32_t SparkLagunaRunLayerMlpRoute(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	LagunaLayerBuffers buffers;
	uint32_t layer,packed_rows;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * LAGUNA_TOP_K;
	stream = (cudaStream_t)wave->slot->stream;
	SparkLagunaBindLayer(wave,local_layer,&buffers);
	if ( wave->tp_rank == 0u && LagunaKdaProbeDeep(&buffers) )
	{
		LAGUNA_KDA_PROBE_RAW(stream,layer,"mlpsite_collapsed",buffers.hc_collapsed_bf16);
		LAGUNA_KDA_PROBE_RAW(stream,layer,"mlpsite_mlp_norm_weight",buffers.mlp_norm_weight);
	}
	status = LagunaHcSite(&buffers,buffers.hc_ffn_fn,buffers.hc_ffn_base,buffers.hc_ffn_scale,wave->row_count,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = layer < LAGUNA_FIRST_ROUTED_LAYER ? LagunaLayerDenseMlp(&buffers,wave->row_count,wave->multiprocessor_count,stream) : LagunaLayerMoeRoute<LAGUNA_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LM_LAUNCH_OK);
}

static int32_t SparkLagunaRunLayerMlpExperts(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	LagunaLayerBuffers buffers;
	if ( (wave->first_layer_index + local_layer) < LAGUNA_FIRST_ROUTED_LAYER )
		return(LM_LAUNCH_OK);
	SparkLagunaBindLayer(wave,local_layer,&buffers);
	return(LagunaLayerMoeExperts<LAGUNA_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,wave->row_count * LAGUNA_TOP_K,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream));
}

static int32_t SparkLagunaRunLayerMlp(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	int32_t status = SparkLagunaRunLayerMlpRoute(wave,local_layer);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SparkLagunaRunLayerMlpExperts(wave,local_layer));
}


static int32_t SparkLagunaRunLayerHcPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	LagunaLayerBuffers buffers;
	int32_t status;
	cudaStream_t stream;
	if ( wave == 0 || wave->slot == 0 || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	stream = (cudaStream_t)wave->slot->stream;
	SparkLagunaBindLayer(wave,local_layer,&buffers);
	status = LagunaHcPost(&buffers,buffers.attention_out_bf16,wave->row_count,stream);
	return(status);
}

extern "C" int32_t SparkLagunaLaunchCudaLayerAttentionPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	if ( SparkLagunaValidateWaveShape(wave) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLagunaRunLayerHcPost(wave,local_layer));
}

extern "C" int32_t SparkLagunaLaunchCudaLayerMlpPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	if ( SparkLagunaValidateWaveShape(wave) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLagunaRunLayerHcPost(wave,local_layer));
}

static int32_t SparkLagunaRunLayers(const SparkLagunaCudaWave *wave)
{
	uint32_t local;
	int32_t status;
	for (local=0u; local<wave->layer_count; local++)
	{
		status = SparkLagunaRunLayerAttention(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkLagunaRunLayerMlp(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

static int32_t SparkLagunaRunHead(const SparkLagunaCudaWave *wave)
{
	SparkLagunaExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	int32_t status;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_final_head != 0u )
	{
		LagunaLayerBuffers buffers;
		uint32_t rank_offset;
		SparkLagunaBindLayer(wave,wave->layer_count - 1u,&buffers);
		LagunaHcHeadMeanKernel<<<wave->row_count,SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,slot->hc_mean_bf16,wave->row_count,LAGUNA_HC,LAGUNA_HIDDEN);
		error = cudaPeekAtLastError();
		if ( error != cudaSuccess )
			return(SparkLagunaCudaStatus(error));
		rank_offset = wave->tp_rank * buffers.head_vocabulary;
		if ( wave->row_count == 1u && wave->head_certified_fp8_payload != 0 )
			status = LagunaHeadCertifiedB1(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->head_certified_fp8_payload,wave->head_certified_fp8_scale_f32,wave->head_certified_fp8_norm_f32,slot->head_certified_scratch,slot->head_certified_candidates,slot->head_screened_count,0u,buffers.head_vocabulary,stream);
		else
			status = LagunaHeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		error = SparkLagunaLaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkLagunaBoundaryStoreKernel<<<dim3((LAGUNA_HC * LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_output == 0u )
		return(SparkLagunaCudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)LAGUNA_DSA_SELECTED;
	if ( wave->maximum_context > LAGUNA_DSA_SELECTED )
		error = cudaMemcpyAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,slot->selected_positions,(uint64_t)wave->row_count * LAGUNA_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	else
		error = cudaMemsetAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,0,(uint64_t)wave->row_count * LAGUNA_DSA_SELECTED * sizeof(uint32_t),stream);
	return(SparkLagunaCudaStatus(error));
}

static int32_t SparkLagunaValidateWaveShape(const SparkLagunaCudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || (wave->execution_row_capacity != 0u && wave->row_count > wave->execution_row_capacity) || (wave->execution_row_capacity == 0u && wave->row_count > wave->resident_sequence_capacity) || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkLagunaLaunchCudaWaveBegin(const SparkLagunaCudaWave *wave)
{
	int32_t status;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLagunaStageWaveMetadata(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLagunaStageWaveBoundary(wave);
	return(status);
}

__global__ void SparkLagunaOpWaitKernel(
	volatile unsigned long long *flag,
	unsigned long long value)
{
	while (*flag < value)
		__nanosleep(100u);
}

extern "C" SparkStatus SparkLagunaLaunchOpWait(
	cudaStream_t stream,
	void *flag_device,
	uint64_t wait_value)
{
	if ( stream == 0 || flag_device == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkLagunaOpWaitKernel<<<1,1,0,stream>>>(
		(volatile unsigned long long *)flag_device,
		(unsigned long long)wait_value);
	return cudaPeekAtLastError() == cudaSuccess ?
		SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR;
}

extern "C" int32_t SparkLagunaLaunchCudaLayerAttention(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLagunaRunLayerAttention(wave,local_layer));
}

extern "C" int32_t SparkLagunaLaunchCudaLayerMlp(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLagunaRunLayerMlp(wave,local_layer));
}

extern "C" int32_t SparkLagunaLaunchCudaLayerMlpRoute(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	uint32_t routed;
	cudaError_t error = cudaSuccess;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( wave->slot->route_ready_event == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	wave->slot->route_recorded = 0u;
	routed = (wave->first_layer_index + local_layer) >= LAGUNA_FIRST_ROUTED_LAYER;
	if ( routed != 0u && wave->slot->host_group_row_offset == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	status = SparkLagunaRunLayerMlpRoute(wave,local_layer);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( routed != 0u )
		error = cudaMemcpyAsync(wave->slot->host_group_row_offset,wave->slot->group_row_offset,(LAGUNA_EXPERTS + 1u) * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)wave->slot->stream);
	if ( error == cudaSuccess )
		error = cudaEventRecord((cudaEvent_t)wave->slot->route_ready_event,(cudaStream_t)wave->slot->stream);
	if ( error != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	wave->slot->route_recorded = 1u;
	return(LM_LAUNCH_OK);
}

extern "C" cudaError_t SparkLagunaPollCudaLayerMlpRoute(const SparkLagunaCudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->route_ready_event == 0 || wave->slot->route_recorded == 0u )
		return(cudaErrorInvalidValue);
	return(cudaEventQuery((cudaEvent_t)wave->slot->route_ready_event));
}

extern "C" int32_t SparkLagunaLaunchCudaLayerMlpExperts(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkLagunaRunLayerMlpExperts(wave,local_layer));
}

extern "C" int32_t SparkLagunaLaunchCudaWaveHead(const SparkLagunaCudaWave *wave)
{
	int32_t status;
	status = SparkLagunaValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SparkLagunaRunHead(wave));
}

extern "C" int32_t SparkLagunaLaunchCudaWave(const SparkLagunaCudaWave *wave)
{
	int32_t status;
	status = SparkLagunaLaunchCudaWaveBegin(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLagunaRunLayers(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkLagunaRunHead(wave);
	return(status);
}

extern "C" int32_t SparkLagunaConfigureCudaModule(uint32_t *multiprocessor_count)
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

static int32_t SparkLagunaMtpReduceRows(const SparkLagunaMtpDraftOps *ops,uint16_t *rows_bf16)
{
	if ( ops == 0 || ops->reduce_rows_bf16 == 0 )
		return(LM_LAUNCH_OK);
	return(ops->reduce_rows_bf16(ops->context,rows_bf16,1u,LAGUNA_HIDDEN) == SPARK_STATUS_OK ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t SparkLagunaMtpReduceHead(const SparkLagunaMtpDraftOps *ops,uint64_t *maxloc)
{
	if ( ops == 0 || ops->reduce_max_u64 == 0 )
		return(LM_LAUNCH_OK);
	return(ops->reduce_max_u64(ops->context,maxloc,1u) == SPARK_STATUS_OK ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t SparkLagunaBindMtpLayer(
	const SparkLagunaCudaWave *wave,
	uint32_t step,
	LagunaLayerBuffers *buffers)
{
	static const uint32_t mtp_no_ordinal[1] = { UINT32_MAX };
	SparkLagunaCudaWave draft;
	SparkLagunaExecutionSlot *slot;
	if ( wave == 0 || buffers == 0 || wave->slot == 0 || wave->mtp_layer_weights == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	slot = wave->slot;
	draft = *wave;
	draft.first_layer_index = LAGUNA_MTP_LAYER_INDEX;
	draft.layer_count = 1u;
	draft.layers = wave->mtp_layer_weights;
	draft.kda_ordinal_by_local_layer = mtp_no_ordinal;
	draft.index_ordinal_by_local_layer = mtp_no_ordinal;
	draft.kda_state_pools = 0;
	draft.kda_q_window_pool = 0;
	draft.kda_k_window_pool = 0;
	draft.kda_v_window_pool = 0;
	draft.kda_layer_count = 0u;
	draft.kv_cache = 0;
	draft.index_cache = 0;
	SparkLagunaBindLayer(&draft,0u,buffers);
	buffers->hc_collapsed_bf16 = slot->mtp_hidden_bf16;
	buffers->hc_mean_bf16 = slot->mtp_hidden_bf16;
	buffers->positions = slot->mtp_positions + step;
	buffers->row_positions = slot->mtp_positions + step;
	buffers->context_length = slot->mtp_context + step;
	buffers->sequence_of_row = slot->mtp_sequence;
	if ( slot->kv_access_error == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	buffers->cache.pool = (uint8_t *)slot->mtp_kv_pool;
	buffers->cache.page_table = slot->mtp_page_table;
	buffers->cache.page_table_stride = 1u;
	buffers->cache.sequence_count = 1u;
	buffers->cache.pool_page_count = 1u;
	buffers->cache.access_error = (LmKvAccessError *)slot->kv_access_error;
	buffers->index_cache = buffers->cache;
	buffers->index_cache.pool = (uint8_t *)slot->mtp_index_pool;
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkLagunaLaunchCudaMtpDraft(
	const SparkLagunaCudaWave *wave,
	const SparkLagunaMtpDraftOps *ops,
	uint16_t *committed_hidden_bf16,
	uint32_t first_token,
	uint32_t *host_draft_tokens)
{
	SparkLagunaExecutionSlot *slot;
	LagunaLayerBuffers buffers;
	const uint16_t *hidden_input;
	cudaStream_t stream;
	cudaError_t error;
	uint32_t step,token;
	uint32_t row_window[2];
	int32_t status;
	if ( wave == 0 || host_draft_tokens == 0 || committed_hidden_bf16 == 0 || wave->slot == 0 ||
		wave->mtp_layer_weights == 0 || wave->mtp_eh_proj_bf16 == 0 || wave->mtp_enorm_bf16 == 0 ||
		wave->mtp_hnorm_bf16 == 0 || wave->mtp_shared_norm_bf16 == 0 || wave->embedding_bf16 == 0 ||
		wave->lm_head_bf16 == 0 || wave->owns_final_head == 0u || wave->mtp_draft_depth == 0u ||
		wave->mtp_draft_depth > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH )
		return(LM_LAUNCH_ERR_SHAPE);
	slot = wave->slot;
	if ( slot->mtp_hidden_bf16 == 0 || slot->mtp_concat_bf16 == 0 || slot->mtp_kv_pool == 0 ||
		slot->mtp_index_pool == 0 || slot->mtp_page_table == 0 || slot->mtp_sequence == 0 ||
		slot->mtp_positions == 0 || slot->mtp_context == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	stream = (cudaStream_t)slot->stream;
	row_window[0] = 0u;
	row_window[1] = 1u;
	error = cudaMemcpyAsync(slot->dense_row_offset,row_window,sizeof(row_window),cudaMemcpyHostToDevice,stream);
	if ( error != cudaSuccess )
		return(SparkLagunaCudaStatus(error));
	token = first_token;
	for ( step = 0u; step < wave->mtp_draft_depth; ++step )
	{
		hidden_input = step == 0u ? (const uint16_t *)committed_hidden_bf16 : (const uint16_t *)slot->mtp_hidden_bf16;
		error = cudaMemcpyAsync(slot->token_ids,&token,sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
		{
			SparkLagunaEmbeddingKernel<<<dim3((LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,1u),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,1u,wave->tp_degree,wave->tp_rank);
			error = cudaPeekAtLastError();
		}
		if ( error != cudaSuccess )
			return(SparkLagunaCudaStatus(error));
		status = SparkLagunaMtpReduceRows(ops,slot->hidden_bf16);
		if ( status != LM_LAUNCH_OK )
			return(status);
		LM_LAUNCH(
			(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
			1u,LAGUNA_LAYER_THREADS,(LAGUNA_HIDDEN + 8u) * sizeof(float),stream,
			slot->hidden_bf16,0,(const uint16_t *)wave->mtp_enorm_bf16,0,slot->mtp_concat_bf16 + LAGUNA_HIDDEN,
			LAGUNA_HIDDEN,LAGUNA_HIDDEN,LAGUNA_RMS_EPSILON);
		LM_LAUNCH(
			(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
			1u,LAGUNA_LAYER_THREADS,(LAGUNA_HIDDEN + 8u) * sizeof(float),stream,
			hidden_input,0,(const uint16_t *)wave->mtp_hnorm_bf16,0,slot->mtp_concat_bf16,
			LAGUNA_HIDDEN,LAGUNA_HIDDEN,LAGUNA_RMS_EPSILON);
		status = LagunaLaunchBf16Linear(slot->mtp_concat_bf16,wave->mtp_eh_proj_bf16,slot->mtp_hidden_bf16,
			slot->dense_row_offset,slot->dense_tile_prefix,1u,2u * LAGUNA_HIDDEN,LAGUNA_HIDDEN,LAGUNA_HIDDEN,0u,wave->multiprocessor_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkLagunaBindMtpLayer(wave,step,&buffers);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = LagunaLayerAttention(&buffers,1u,step + 1u,LAGUNA_MTP_LAYER_INDEX,wave->multiprocessor_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkLagunaMtpReduceRows(ops,slot->attention_out_bf16);
		if ( status != LM_LAUNCH_OK )
			return(status);
		LM_LAUNCH(
			(LmAddRowsKernel<LAGUNA_LAYER_THREADS>),
			dim3((LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS,1u),LAGUNA_LAYER_THREADS,0,stream,
			slot->mtp_hidden_bf16,slot->attention_out_bf16,slot->mtp_hidden_bf16,1u,LAGUNA_HIDDEN);
		status = LagunaLayerMoe<LAGUNA_EXPERT_WEIGHT_CODEC>(&buffers,1u,LAGUNA_TOP_K,wave->multiprocessor_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkLagunaMtpReduceRows(ops,slot->attention_out_bf16);
		if ( status != LM_LAUNCH_OK )
			return(status);
		LM_LAUNCH(
			(LmAddRowsKernel<LAGUNA_LAYER_THREADS>),
			dim3((LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS,1u),LAGUNA_LAYER_THREADS,0,stream,
			slot->mtp_hidden_bf16,slot->attention_out_bf16,slot->mtp_hidden_bf16,1u,LAGUNA_HIDDEN);
		status = LagunaHead(&buffers,wave->mtp_shared_norm_bf16,wave->lm_head_bf16,0,buffers.head_vocabulary,1u,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		error = SparkLagunaLaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,1u,wave->tp_rank * buffers.head_vocabulary);
		if ( error == cudaSuccess && SparkLagunaMtpReduceHead(ops,slot->head_maxloc_u64) != LM_LAUNCH_OK )
			error = cudaErrorUnknown;
		if ( error == cudaSuccess )
			error = SparkLagunaLaunchHeadMaxlocUnpack(stream,slot->head_maxloc_u64,slot->output_token,1u);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(&host_draft_tokens[step],slot->output_token,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		if ( error != cudaSuccess )
			return(SparkLagunaCudaStatus(error));
		token = host_draft_tokens[step];
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkLagunaLaunchCudaMtpCommit(
	const SparkLagunaCudaWave *wave,
	uint32_t committed_steps)
{
	SparkLagunaExecutionSlot *slot;
	SparkLagunaKdaReplayLayout layout;
	LmReplayStep host_steps[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u];
	const SparkLagunaLayerWeights *weight;
	LmReplayStep *device_steps;
	cudaStream_t stream;
	cudaError_t error;
	uint8_t *record;
	uint32_t local,ordinal,rank_heads,rank_qk,rank_v,steps_capacity,step;
	static_assert(sizeof(LmReplayStep) == SPARK_LAGUNA_MTP_REPLAY_STEP_BYTES,"replay step record size changed; re-price the staging buffer");
	if ( wave == 0 || wave->slot == 0 || wave->layers == 0 || wave->mtp_verify == 0u ||
		committed_steps == 0u || committed_steps > wave->row_count ||
		wave->row_count > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u )
		return(LM_LAUNCH_ERR_SHAPE);
	slot = wave->slot;
	if ( slot->kda_replay_pool == 0 || slot->mtp_replay_steps == 0 || slot->mtp_committed == 0 ||
		slot->mtp_conv_scratch == 0 || slot->resident_slots == 0 || slot->run_begin == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	stream = (cudaStream_t)slot->stream;
	rank_heads = SPARK_LAGUNA_MODEL_KDA_HEAD_COUNT / wave->tp_degree;
	rank_qk = rank_heads * LAGUNA_KDA_KEY_DIM;
	rank_v = rank_heads * LAGUNA_KDA_VALUE_DIM;
	steps_capacity = wave->row_count;
	layout = SparkLagunaKdaReplayLayoutFor(rank_heads,steps_capacity);
	error = cudaMemcpyAsync(slot->mtp_committed,&committed_steps,sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	for ( local = 0u; error == cudaSuccess && local < wave->layer_count; ++local )
	{
		ordinal = wave->kda_ordinal_by_local_layer[local];
		if ( ordinal == UINT32_MAX )
			continue;
		weight = &wave->layers[local];
		record = slot->kda_replay_pool + (uint64_t)ordinal * wave->kda_replay_layer_bytes;
		for ( step = 0u; step < steps_capacity; ++step )
		{
			host_steps[step].key_bf16 = (const uint16_t *)(record + layout.key_offset) + (uint64_t)step * rank_qk;
			host_steps[step].value_bf16 = (const uint16_t *)(record + layout.value_offset) + (uint64_t)step * rank_v;
			host_steps[step].retention = (const float *)(record + layout.retention_offset) + (uint64_t)step * rank_qk;
			host_steps[step].write_gate = (const float *)(record + layout.write_gate_offset) + (uint64_t)step * rank_heads;
		}
		device_steps = (LmReplayStep *)slot->mtp_replay_steps + (uint64_t)ordinal * steps_capacity;
		error = cudaMemcpyAsync(device_steps,host_steps,(uint64_t)steps_capacity * sizeof(LmReplayStep),cudaMemcpyHostToDevice,stream);
		if ( error != cudaSuccess )
			break;
		LM_LAUNCH(
			(LmReplayFoldKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_KEY_DIM,LAGUNA_KDA_VALUE_DIM,float>),
			dim3(1u,rank_heads),LAGUNA_LAYER_THREADS,0,stream,
			wave->kda_state_pools + (uint64_t)ordinal * wave->kda_state_layer_stride_bytes,
			LAGUNA_KDA_STATE_BYTES_PER_LAYER / wave->tp_degree,
			slot->resident_slots,
			device_steps,
			slot->mtp_committed,
			rank_heads,1u,1u);
		LM_LAUNCH(
			(LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
			dim3(1u,(rank_qk + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),LAGUNA_LAYER_THREADS,0,stream,
			(uint16_t *)(wave->kda_q_window_pool + (uint64_t)ordinal * wave->kda_window_layer_stride_bytes),
			slot->resident_slots,slot->run_begin,slot->mtp_committed,
			(const uint16_t *)(record + layout.pre_q_offset),(const uint16_t *)weight->kda_q_conv_bf16,
			slot->mtp_conv_scratch,rank_qk,1u,1u);
		LM_LAUNCH(
			(LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
			dim3(1u,(rank_qk + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),LAGUNA_LAYER_THREADS,0,stream,
			(uint16_t *)(wave->kda_k_window_pool + (uint64_t)ordinal * wave->kda_window_layer_stride_bytes),
			slot->resident_slots,slot->run_begin,slot->mtp_committed,
			(const uint16_t *)(record + layout.pre_k_offset),(const uint16_t *)weight->kda_k_conv_bf16,
			slot->mtp_conv_scratch,rank_qk,1u,1u);
		LM_LAUNCH(
			(LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
			dim3(1u,(rank_v + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),LAGUNA_LAYER_THREADS,0,stream,
			(uint16_t *)(wave->kda_v_window_pool + (uint64_t)ordinal * wave->kda_window_layer_stride_bytes),
			slot->resident_slots,slot->run_begin,slot->mtp_committed,
			(const uint16_t *)(record + layout.pre_v_offset),(const uint16_t *)weight->kda_v_conv_bf16,
			slot->mtp_conv_scratch,rank_v,1u,1u);
		error = cudaPeekAtLastError();
	}
	return(SparkLagunaCudaStatus(error));
}
