#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/laguna_resident_decode_stage/source/cuda/unity.cu"
#include "spark_laguna_resident_decode_stage_internal.h"
#include "inference/kernels/tp_reduce.cuh"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"

#define SPARK_LAGUNA_CUDA_THREADS 256u

__global__ static void SparkLagunaBoundaryLoadKernel(
	const uint16_t *boundary,
	uint16_t *hidden,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source,width;
	width = (uint64_t)LAGUNA_HIDDEN;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= width )
		return;
	source = ((first_row + row) * width);
	hidden[(row * width) + element] = boundary[source + element];
}

__global__ static void SparkLagunaBoundaryStoreKernel(
	const uint16_t *hidden,
	uint16_t *boundary,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,destination,width;
	width = (uint64_t)LAGUNA_HIDDEN;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= width )
		return;
	destination = ((first_row + row) * width);
	boundary[destination + element] = hidden[(row * width) + element];
}

__global__ static void SparkLagunaEmbeddingKernel(
	const uint32_t *token_ids,
	const uint16_t *embedding,
	uint16_t *hidden,
	uint32_t row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t element,row,source;
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
	hidden[((uint64_t)row * LAGUNA_HIDDEN) + element] = value;
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

extern "C" cudaError_t SparkLagunaLaunchDirectSum(cudaStream_t stream,void *destination,const void *const *rank_devices,uint32_t local_rank,uint32_t rows,uint32_t width)
{
	LmTpBf16Contributions<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT> inputs = {};
	uint32_t rank;
	cudaError_t error;
	if ( destination == 0 || rank_devices == 0 || local_rank >= SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT || rows == 0u || width == 0u )
		return(cudaErrorInvalidValue);
	for (rank=0u; rank<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT; rank++)
		inputs.rank[rank] = (const uint16_t *)(rank == local_rank ? destination : rank_devices[rank]);
	LmTpBf16SumKernel<<<rows,256u,0u,stream>>>((uint16_t *)destination,inputs,rows,width);
	error = cudaPeekAtLastError();
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
	if ( error == cudaSuccess && wave->owns_embedding != 0u )
		error = cudaMemcpyAsync(slot->token_ids,wave->host_token_ids,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkLagunaWaveMetadataKernel<<<(wave->row_count + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLagunaCudaStatus(error));
}

static int32_t SparkLagunaStageWaveBoundary(const SparkLagunaCudaWave *wave)
{
	SparkLagunaExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	if ( wave->owns_embedding != 0u )
	{
		SparkLagunaEmbeddingKernel<<<dim3((LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkLagunaBoundaryLoadKernel<<<dim3((LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error == cudaSuccess )
		error = cudaMemsetAsync(slot->residual_bf16,0,(uint64_t)wave->row_count * LAGUNA_HIDDEN * sizeof(uint16_t),stream);
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

static void SparkLagunaBindLayer(
	const SparkLagunaCudaWave *wave,
	uint32_t local_layer,
	LagunaLayerBuffers *buffers)
{
	const SparkLagunaLayerWeights *weight;
	SparkLagunaExecutionSlot *slot;
	uint32_t layer;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	layer = wave->first_layer_index + local_layer;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->layer_index = layer;
	buffers->q_heads = LAGUNA_LAYER_HEADS(layer) / wave->tp_degree;
	buffers->qkv_rows = buffers->q_heads * LAGUNA_HEAD_DIM +
		2u * (SPARK_LAGUNA_MODEL_ATTENTION_KV_HEAD_COUNT / wave->tp_degree) * LAGUNA_HEAD_DIM;
	buffers->attn_output_columns = buffers->q_heads * LAGUNA_HEAD_DIM;
	buffers->gate_rows = buffers->q_heads;
	buffers->dense_gate_up_rows = 2u * SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->fused_qkv_weight = weight->fused_qkv_bf16;
	buffers->q_norm_weight = weight->q_norm_bf16;
	buffers->k_norm_weight = weight->k_norm_bf16;
	buffers->qk_scale = LAGUNA_ATTN_SCALE;
	buffers->yarn_inv_freq = wave->yarn_inv_freq;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->gate_weight = weight->attn_gate_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_up_weight = weight->dense_gate_up_bf16;
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->expert_w1_weight = weight->expert_gate_up_payload;
	buffers->expert_w1_scale = weight->expert_gate_up_scale;
	buffers->expert_w2_weight = weight->expert_down_payload;
	buffers->expert_w2_scale = weight->expert_down_scale;
	if ( wave->lazy_experts != 0u )
	{
		buffers->expert_w1_weight = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_gate_up_payload_offset;
		buffers->expert_w1_scale = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_gate_up_scale_offset;
		buffers->expert_w2_weight = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_down_payload_offset;
		buffers->expert_w2_scale = (wave->expert_lease_base == 0 || wave->expert_lease_local_layer != local_layer) ? 0 : wave->expert_lease_base + weight->expert_down_scale_offset;
	}
	buffers->shared_gate_up_weight = weight->shared_gate_up_bf16;
	buffers->shared_down_weight = weight->shared_down_bf16;
	buffers->hidden_bf16 = slot->hidden_bf16;
	buffers->residual_bf16 = slot->residual_bf16;
	buffers->normed_bf16 = slot->normed_bf16;
	buffers->qkv_bf16 = slot->qkv_bf16;
	buffers->q_bf16 = slot->q_bf16;
	buffers->k_bf16 = slot->k_bf16;
	buffers->v_bf16 = slot->v_bf16;
	buffers->attention_bf16 = slot->attention_bf16;
	buffers->attention_out_bf16 = slot->attention_out_bf16;
	buffers->gate_bf16 = slot->gate_bf16;
	buffers->window_positions = slot->window_positions;
	buffers->gate_up_bf16 = slot->gate_up_bf16;
	buffers->intermediate_bf16 = slot->intermediate_bf16;
	buffers->expert_out_bf16 = slot->expert_out_bf16;
	buffers->shared_out_bf16 = slot->shared_out_bf16;
	buffers->router_logits = slot->router_logits_f32;
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
	SparkLagunaBuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)local_layer * wave->kv_layer_stride_bytes),wave);
}

static int32_t SparkLagunaValidateWaveShape(const SparkLagunaCudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || (wave->execution_row_capacity != 0u && wave->row_count > wave->execution_row_capacity) || (wave->execution_row_capacity == 0u && wave->row_count > wave->resident_sequence_capacity) || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree != SPARK_LAGUNA_MODEL_TENSOR_PARALLEL_DEGREE )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

static int32_t SparkLagunaRunLayerAttention(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	LagunaLayerBuffers buffers;
	int32_t status;
	cudaStream_t stream;
	stream = (cudaStream_t)wave->slot->stream;
	SparkLagunaBindLayer(wave,local_layer,&buffers);
	status = LagunaLayerAttention(&buffers,wave->row_count,wave->maximum_context,wave->multiprocessor_count,stream);
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
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	if ( wave->owns_final_head != 0u )
	{
		LagunaLayerBuffers buffers;
		uint32_t rank_offset;
		SparkLagunaBindLayer(wave,wave->layer_count - 1u,&buffers);
		rank_offset = wave->tp_rank * (LAGUNA_VOCAB / wave->tp_degree);
		status = LagunaHeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		error = SparkLagunaLaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkLagunaBoundaryStoreKernel<<<dim3((LAGUNA_HIDDEN + SPARK_LAGUNA_CUDA_THREADS - 1u) / SPARK_LAGUNA_CUDA_THREADS,wave->row_count),SPARK_LAGUNA_CUDA_THREADS,0,stream>>>(slot->residual_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkLagunaCudaStatus(error));
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

extern "C" int32_t SparkLagunaLaunchCudaLayerAttentionPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkLagunaLaunchCudaLayerMlpPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
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

extern "C" SparkStatus SparkLagunaStageYarnTableUpload(float *device_inv_freq,void *stream)
{
	float host_inv_freq[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	if ( device_inv_freq == 0 || stream == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	LagunaBuildYarnInvFrequency(host_inv_freq,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION,
		SPARK_LAGUNA_MODEL_ROPE_FULL_THETA,
		SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW);
	if ( cudaMemcpy(device_inv_freq,host_inv_freq,sizeof(host_inv_freq),cudaMemcpyHostToDevice) != cudaSuccess )
		return SPARK_STATUS_DRIVER_LOAD_ERROR;
	return SPARK_STATUS_OK;
}
