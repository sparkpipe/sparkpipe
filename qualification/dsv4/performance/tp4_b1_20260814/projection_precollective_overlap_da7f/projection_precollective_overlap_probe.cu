#include <algorithm>
#include <atomic>
#include <chrono>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_lm_kernels.cuh"

#define POC_LAYER_COUNT SPARK_DSV4_MODEL_LAYER_COUNT
#define POC_CSA_LAYER_COUNT 21u
#define POC_QUERY_RANK SPARK_DSV4_MODEL_QUERY_LORA_RANK
#define POC_LOCAL_QUERY_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / 4u)
#define POC_INDEX_DIMENSION SPARK_DSV4_MODEL_INDEX_DIMENSION
#define POC_INDEX_WEIGHT_COUNT SPARK_DSV4_MODEL_INDEX_HEAD_COUNT
#define POC_TOPK_COLUMNS (SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + SPARK_DSV4_MODEL_INDEX_TOP_K)
#define POC_REPETITION_COUNT 31u
#define POC_WARMUP_COUNT 4u
#define POC_FLUSH_BYTES (256u * 1024u * 1024u)

#define POC_CUDA(call) do { cudaError_t poc_error = (call); if ( poc_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(poc_error)); return(-1); } } while (0)

typedef struct PocTiming
{
	float median_ms,p10_ms,p90_ms,mean_ms;
} PocTiming;

typedef struct PocOutputs
{
	uint16_t *query_rank_bf16;
	uint16_t *query_bf16;
	uint16_t *index_query_bf16;
	uint16_t *index_weights_bf16;
	float *index_weights_f32;
	int32_t *attention_indices;
	uint32_t *slot_counts;
	uint32_t *attention_slot_counts;
} PocOutputs;

typedef struct PocBuffers
{
	uint16_t *query_rank_input_bf16;
	uint16_t *query_norm_gain_bf16;
	uint16_t *normalized_bf16;
	uint8_t *query_weight_fp8;
	uint8_t *query_scale_e8m0;
	uint8_t *index_query_weight_fp8;
	uint8_t *index_query_scale_e8m0;
	uint16_t *index_weight_bf16;
	uint64_t *row_positions;
	uint8_t *flush;
	uint32_t *flush_sink;
	volatile uint32_t *collective_request_host;
	volatile uint32_t *collective_done_host;
	uint32_t *collective_request_device;
	uint32_t *collective_done_device;
	PocOutputs control;
	PocOutputs candidate;
} PocBuffers;

static cudaError_t PocStreamWriteValue32(cudaStream_t stream,uint32_t *address,
	uint32_t value)
{
	CUresult result = cuStreamWriteValue32((CUstream)stream,
		(CUdeviceptr)(uintptr_t)address,value,CU_STREAM_WRITE_VALUE_DEFAULT);
	return(result == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown);
}

static cudaError_t PocStreamWaitValue32(cudaStream_t stream,uint32_t *address,
	uint32_t value)
{
	CUresult result = cuStreamWaitValue32((CUstream)stream,
		(CUdeviceptr)(uintptr_t)address,value,CU_STREAM_WAIT_VALUE_GEQ);
	return(result == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown);
}

static __global__ void PocFillBf16Kernel(uint16_t *values,uint64_t count,
	uint32_t seed,float center,float scale)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t value;
	float converted;
	for (; index<count; index+=stride)
	{
		value = ((uint32_t)index * UINT32_C(747796405)) + seed;
		value = ((value >> ((value >> 28u) + 4u)) ^ value) *
			UINT32_C(277803737);
		value = (value >> 22u) ^ value;
		converted = center + ((float)((int32_t)(value & 255u) - 127) * scale);
		values[index] = __bfloat16_as_ushort(__float2bfloat16_rn(converted));
	}
}

static __global__ void PocFillFp8Kernel(uint8_t *values,uint64_t count,
	uint32_t seed)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t value;
	for (; index<count; index+=stride)
	{
		value = ((uint32_t)index * UINT32_C(747796405)) + seed;
		value ^= value >> 16u;
		values[index] = (uint8_t)(0x18u + (value & 0x0fu) |
			((value >> 7u) & 0x80u));
	}
}

static __global__ void PocFlushKernel(const uint4 *values,uint64_t count,
	uint32_t *sink)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t checksum = 0u;
	for (; index<count; index+=stride)
	{
		uint4 value = values[index];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	if ( checksum != 0u )
		atomicXor(sink,checksum);
}

static __device__ __forceinline__ uint32_t PocAttentionWindowSlot(
	uint64_t position,uint32_t column,uint32_t window_token_count)
{
	uint32_t first;
	if ( window_token_count == 0u || column >= window_token_count )
		return(UINT32_MAX);
	first = position + 1u < window_token_count ? 0u :
		(uint32_t)(((position % window_token_count) + 1u) % window_token_count);
	return((first + column) % window_token_count);
}

static __global__ void PocBuildAttentionIndicesKernel(
	const uint64_t *row_positions,int32_t *indices,uint32_t *slot_counts,
	uint32_t *attention_slot_counts,uint32_t column_count,
	uint32_t index_slot_capacity,uint32_t layer_kind)
{
	uint32_t attention_slots,column,compressed,valid_index_slots,window;
	uint64_t position = row_positions[0];
	window = position + 1u < SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS ?
		(uint32_t)(position + 1u) : SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
	compressed = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA ?
		(uint32_t)((position + 1u) / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO) : 0u;
	valid_index_slots = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ?
		(uint32_t)((position + 1u) / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO) : 0u;
	valid_index_slots = min(valid_index_slots,index_slot_capacity);
	attention_slots = window;
	if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		attention_slots += compressed;
	else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		attention_slots = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
			min(valid_index_slots,SPARK_DSV4_MODEL_INDEX_TOP_K);
	attention_slots = min(attention_slots,column_count);
	if ( threadIdx.x == 0u )
	{
		slot_counts[0] = valid_index_slots;
		attention_slot_counts[0] = attention_slots;
	}
	for (column=threadIdx.x; column<column_count; column+=blockDim.x)
	{
		if ( column < window )
			indices[column] = (int32_t)PocAttentionWindowSlot(position,column,
				SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
		else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA &&
			column < window + compressed )
			indices[column] = (int32_t)(SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
				column - window);
		else
			indices[column] = -1;
	}
}

static __global__ void PocWidenKernel(const uint16_t *input_bf16,
	float *output_f32,uint32_t width,float scale)
{
	uint32_t element;
	for (element=threadIdx.x; element<width; element+=blockDim.x)
		output_f32[element] = __bfloat162float(__ushort_as_bfloat16(
			input_bf16[element])) * scale;
}

static uint32_t PocCsaOrdinal(uint32_t layer)
{
	uint32_t ordinal = 0u,index;
	for (index=0u; index<layer; index++)
		ordinal += SparkDsv4ModelLayerKind(index) ==
			SPARK_DSV4_MODEL_LAYER_KIND_CSA ? 1u : 0u;
	return(ordinal);
}

static int32_t PocAllocateOutputs(PocOutputs *output)
{
	POC_CUDA(cudaMalloc(&output->query_rank_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&output->query_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_LOCAL_QUERY_DIMENSION * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&output->index_query_bf16,
		(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_DIMENSION * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&output->index_weights_bf16,
		(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&output->index_weights_f32,
		(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT * sizeof(float)));
	POC_CUDA(cudaMalloc(&output->attention_indices,
		(uint64_t)POC_LAYER_COUNT * POC_TOPK_COLUMNS * sizeof(int32_t)));
	POC_CUDA(cudaMalloc(&output->slot_counts,
		POC_LAYER_COUNT * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&output->attention_slot_counts,
		POC_LAYER_COUNT * sizeof(uint32_t)));
	return(0);
}

static int32_t PocAllocate(PocBuffers *buffers,cudaStream_t stream)
{
	uint64_t query_weight_count = (uint64_t)POC_LAYER_COUNT *
		POC_LOCAL_QUERY_DIMENSION * POC_QUERY_RANK;
	uint64_t query_scale_count = query_weight_count / 128u;
	uint64_t index_query_weight_count = (uint64_t)POC_CSA_LAYER_COUNT *
		POC_INDEX_DIMENSION * POC_QUERY_RANK;
	uint64_t index_query_scale_count = index_query_weight_count / 128u;
	uint64_t index_weight_count = (uint64_t)POC_CSA_LAYER_COUNT *
		POC_INDEX_WEIGHT_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	std::vector<uint64_t> positions(POC_LAYER_COUNT);
	uint32_t layer;
	memset(buffers,0,sizeof(*buffers));
	POC_CUDA(cudaMalloc(&buffers->query_rank_input_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&buffers->query_norm_gain_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&buffers->normalized_bf16,
		(uint64_t)POC_LAYER_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
		sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&buffers->query_weight_fp8,query_weight_count));
	POC_CUDA(cudaMalloc(&buffers->query_scale_e8m0,query_scale_count));
	POC_CUDA(cudaMalloc(&buffers->index_query_weight_fp8,
		index_query_weight_count));
	POC_CUDA(cudaMalloc(&buffers->index_query_scale_e8m0,
		index_query_scale_count));
	POC_CUDA(cudaMalloc(&buffers->index_weight_bf16,
		index_weight_count * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&buffers->row_positions,
		POC_LAYER_COUNT * sizeof(uint64_t)));
	POC_CUDA(cudaMalloc(&buffers->flush,POC_FLUSH_BYTES));
	POC_CUDA(cudaMalloc(&buffers->flush_sink,sizeof(uint32_t)));
	POC_CUDA(cudaHostAlloc((void **)&buffers->collective_request_host,
		2u * sizeof(uint32_t),cudaHostAllocMapped));
	buffers->collective_done_host = buffers->collective_request_host + 1u;
	POC_CUDA(cudaHostGetDevicePointer(&buffers->collective_request_device,
		(void *)buffers->collective_request_host,0u));
	POC_CUDA(cudaHostGetDevicePointer(&buffers->collective_done_device,
		(void *)buffers->collective_done_host,0u));
	if ( PocAllocateOutputs(&buffers->control) < 0 ||
		PocAllocateOutputs(&buffers->candidate) < 0 )
		return(-2);
	PocFillBf16Kernel<<<4096u,256u,0,stream>>>(
		buffers->query_rank_input_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK,UINT32_C(0x41f20cab),
		0.0f,1.0f / 4096.0f);
	PocFillBf16Kernel<<<4096u,256u,0,stream>>>(
		buffers->query_norm_gain_bf16,
		(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK,UINT32_C(0x7b2de519),
		1.0f,1.0f / 32768.0f);
	PocFillBf16Kernel<<<4096u,256u,0,stream>>>(buffers->normalized_bf16,
		(uint64_t)POC_LAYER_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
		UINT32_C(0x937d71ac),0.0f,1.0f / 4096.0f);
	PocFillFp8Kernel<<<4096u,256u,0,stream>>>(buffers->query_weight_fp8,
		query_weight_count,UINT32_C(0x28b7c113));
	PocFillFp8Kernel<<<4096u,256u,0,stream>>>(
		buffers->index_query_weight_fp8,index_query_weight_count,
		UINT32_C(0x981dcf22));
	POC_CUDA(cudaMemsetAsync(buffers->query_scale_e8m0,127,
		query_scale_count,stream));
	POC_CUDA(cudaMemsetAsync(buffers->index_query_scale_e8m0,127,
		index_query_scale_count,stream));
	PocFillBf16Kernel<<<4096u,256u,0,stream>>>(buffers->index_weight_bf16,
		index_weight_count,UINT32_C(0xc4a63f0d),0.0f,1.0f / 8192.0f);
	for (layer=0u; layer<POC_LAYER_COUNT; layer++)
		positions[layer] = 4095u + layer;
	POC_CUDA(cudaMemcpyAsync(buffers->row_positions,positions.data(),
		POC_LAYER_COUNT * sizeof(uint64_t),cudaMemcpyHostToDevice,stream));
	POC_CUDA(cudaMemsetAsync(buffers->flush,0xa5,POC_FLUSH_BYTES,stream));
	POC_CUDA(cudaMemsetAsync(buffers->flush_sink,0,sizeof(uint32_t),stream));
	POC_CUDA(cudaStreamSynchronize(stream));
	return(0);
}

static int32_t PocResetOutput(cudaStream_t stream,const PocOutputs *output)
{
	if ( cudaMemsetAsync(output->query_rank_bf16,0,
			(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK * sizeof(uint16_t),stream) !=
		cudaSuccess )
		return(-1);
	if ( cudaMemsetAsync(output->query_bf16,0,
			(uint64_t)POC_LAYER_COUNT * POC_LOCAL_QUERY_DIMENSION *
			sizeof(uint16_t),stream) != cudaSuccess )
		return(-2);
	if ( cudaMemsetAsync(output->index_query_bf16,0,
			(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_DIMENSION *
			sizeof(uint16_t),stream) != cudaSuccess )
		return(-3);
	if ( cudaMemsetAsync(output->index_weights_bf16,0,
			(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT *
			sizeof(uint16_t),stream) != cudaSuccess )
		return(-4);
	if ( cudaMemsetAsync(output->index_weights_f32,0,
			(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT *
			sizeof(float),stream) != cudaSuccess )
		return(-5);
	if ( cudaMemsetAsync(output->attention_indices,0,
			(uint64_t)POC_LAYER_COUNT * POC_TOPK_COLUMNS * sizeof(int32_t),stream) !=
		cudaSuccess )
		return(-6);
	if ( cudaMemsetAsync(output->slot_counts,0,
			POC_LAYER_COUNT * sizeof(uint32_t),stream) != cudaSuccess )
		return(-7);
	if ( cudaMemsetAsync(output->attention_slot_counts,0,
			POC_LAYER_COUNT * sizeof(uint32_t),stream) != cudaSuccess )
		return(-8);
	return(0);
}

static int32_t PocLaunchStageTopk(cudaStream_t stream,const PocBuffers *buffers,
	const PocOutputs *output,uint32_t layer)
{
	PocBuildAttentionIndicesKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
		buffers->row_positions + layer,
		output->attention_indices + (uint64_t)layer * POC_TOPK_COLUMNS,
		output->slot_counts + layer,output->attention_slot_counts + layer,
		POC_TOPK_COLUMNS,262144u,SparkDsv4ModelLayerKind(layer));
	return(cudaGetLastError() == cudaSuccess ? 0 : -1);
}

static int32_t PocLaunchIndexWeights(cudaStream_t stream,
	const PocBuffers *buffers,const PocOutputs *output,uint32_t layer)
{
	uint32_t ordinal = PocCsaOrdinal(layer);
	uint16_t *output_bf16 = output->index_weights_bf16 +
		(uint64_t)ordinal * POC_INDEX_WEIGHT_COUNT;
	float *output_f32 = output->index_weights_f32 +
		(uint64_t)ordinal * POC_INDEX_WEIGHT_COUNT;
	const uint16_t *weight = buffers->index_weight_bf16 +
		(uint64_t)ordinal * POC_INDEX_WEIGHT_COUNT *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	const uint16_t *input = buffers->normalized_bf16 +
		(uint64_t)layer * SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	float scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) /
		sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error = SparkLmHostLaunchSm121DecodeLinear<32u,
		SPARK_ACTIVATION_CODEC_NONE>(stream,SPARK_LM_WEIGHT_FORMAT_BF16,weight,0,
		input,output_bf16,1u,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
		POC_INDEX_WEIGHT_COUNT);
	if ( error == cudaSuccess )
		PocWidenKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(output_bf16,
			output_f32,POC_INDEX_WEIGHT_COUNT,scale);
	return(error == cudaSuccess && cudaGetLastError() == cudaSuccess ? 0 : -1);
}

static int32_t PocLaunchQueryRankPost(cudaStream_t stream,
	const PocBuffers *buffers,const PocOutputs *output,uint32_t layer)
{
	const uint16_t *input = buffers->query_rank_input_bf16 +
		(uint64_t)layer * POC_QUERY_RANK;
	const uint16_t *gain = buffers->query_norm_gain_bf16 +
		(uint64_t)layer * POC_QUERY_RANK;
	uint16_t *result = output->query_rank_bf16 +
		(uint64_t)layer * POC_QUERY_RANK;
	SparkLmRmsNormKernel<<<1u,SPARK_LM_CTA_THREADS,
		POC_QUERY_RANK * sizeof(float),stream>>>(input,gain,result,1u,
		POC_QUERY_RANK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	return(cudaGetLastError() == cudaSuccess ? 0 : -1);
}

static int32_t PocLaunchQueryProjection(cudaStream_t stream,
	const PocBuffers *buffers,const PocOutputs *output,uint32_t layer)
{
	uint64_t payload_stride = (uint64_t)POC_LOCAL_QUERY_DIMENSION *
		POC_QUERY_RANK;
	uint64_t scale_stride = payload_stride / 128u;
	return(SparkLmHostLaunchSm121DecodeLinear<128u,
		SPARK_ACTIVATION_CODEC_NONE>(stream,SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,
		buffers->query_weight_fp8 + (uint64_t)layer * payload_stride,
		buffers->query_scale_e8m0 + (uint64_t)layer * scale_stride,
		output->query_rank_bf16 + (uint64_t)layer * POC_QUERY_RANK,
		output->query_bf16 + (uint64_t)layer * POC_LOCAL_QUERY_DIMENSION,1u,
		POC_QUERY_RANK,POC_LOCAL_QUERY_DIMENSION) == cudaSuccess ? 0 : -1);
}

static int32_t PocLaunchIndexQuery(cudaStream_t stream,const PocBuffers *buffers,
	const PocOutputs *output,uint32_t layer)
{
	uint32_t ordinal = PocCsaOrdinal(layer);
	uint64_t payload_stride = (uint64_t)POC_INDEX_DIMENSION * POC_QUERY_RANK;
	uint64_t scale_stride = payload_stride / 128u;
	return(SparkLmHostLaunchSm121DecodeLinear<128u,
		SPARK_ACTIVATION_CODEC_NONE>(stream,SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,
		buffers->index_query_weight_fp8 + (uint64_t)ordinal * payload_stride,
		buffers->index_query_scale_e8m0 + (uint64_t)ordinal * scale_stride,
		output->query_rank_bf16 + (uint64_t)layer * POC_QUERY_RANK,
		output->index_query_bf16 + (uint64_t)ordinal * POC_INDEX_DIMENSION,1u,
		POC_QUERY_RANK,POC_INDEX_DIMENSION) == cudaSuccess ? 0 : -1);
}

static void PocProgressCollectives(volatile uint32_t *request,
	volatile uint32_t *done,uint64_t delay_ns,uint64_t *actual_total_ns)
{
	uint32_t layer;
	uint64_t total = 0u;
	for (layer=1u; layer<=POC_LAYER_COUNT; layer++)
	{
		while ( __atomic_load_n(request,__ATOMIC_ACQUIRE) < layer )
			std::this_thread::yield();
		auto start = std::chrono::steady_clock::now();
		uint64_t elapsed = 0u;
		while ( elapsed < delay_ns )
		{
			std::atomic_signal_fence(std::memory_order_seq_cst);
			elapsed = (uint64_t)std::chrono::duration_cast<
				std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
				start).count();
		}
		total += elapsed;
		__atomic_store_n(done,layer,__ATOMIC_RELEASE);
	}
	*actual_total_ns = total;
}

static int32_t PocRunStep(cudaStream_t primary,cudaStream_t prologue,
	cudaStream_t kv,cudaEvent_t start,cudaEvent_t stop,
	cudaEvent_t *projection_ready,cudaEvent_t *milestone,
	cudaEvent_t *prologue_done,cudaEvent_t *kv_done,const PocBuffers *buffers,
	uint32_t candidate,uint64_t delay_ns,float *elapsed_ms,
	uint64_t *actual_collective_ns)
{
	const PocOutputs *output = candidate != 0u ? &buffers->candidate :
		&buffers->control;
	uint32_t layer,kind;
	__atomic_store_n(buffers->collective_request_host,0u,__ATOMIC_RELEASE);
	__atomic_store_n(buffers->collective_done_host,0u,__ATOMIC_RELEASE);
	if ( PocResetOutput(primary,output) < 0 )
		return(-1);
	PocFlushKernel<<<4096u,256u,0u,primary>>>((const uint4 *)buffers->flush,
		POC_FLUSH_BYTES / sizeof(uint4),buffers->flush_sink);
	if ( cudaStreamSynchronize(primary) != cudaSuccess )
		return(-2);
	std::thread progress(PocProgressCollectives,
		buffers->collective_request_host,buffers->collective_done_host,delay_ns,
		actual_collective_ns);
	if ( cudaEventRecord(start,primary) != cudaSuccess )
		return(-3);
	for (layer=0u; layer<POC_LAYER_COUNT; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		if ( cudaEventRecord(projection_ready[layer],primary) != cudaSuccess )
			return(-4);
		if ( candidate != 0u )
		{
			if ( cudaStreamWaitEvent(prologue,projection_ready[layer],0u) !=
				cudaSuccess )
				return(-5);
			if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA &&
				PocLaunchIndexWeights(prologue,buffers,output,layer) < 0 )
				return(-6);
			if ( PocLaunchStageTopk(prologue,buffers,output,layer) < 0 )
				return(-7);
			if ( cudaEventRecord(prologue_done[layer],prologue) != cudaSuccess )
				return(-8);
		}
		if ( PocStreamWriteValue32(primary,buffers->collective_request_device,
				layer + 1u) != cudaSuccess ||
			PocStreamWaitValue32(primary,buffers->collective_done_device,
				layer + 1u) != cudaSuccess )
			return(-9);
		if ( candidate == 0u &&
			PocLaunchStageTopk(primary,buffers,output,layer) < 0 )
			return(-10);
		if ( PocLaunchQueryRankPost(primary,buffers,output,layer) < 0 ||
			cudaEventRecord(milestone[layer],primary) != cudaSuccess )
			return(-11);
		if ( cudaStreamWaitEvent(kv,milestone[layer],0u) != cudaSuccess )
			return(-12);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		{
			if ( PocLaunchIndexQuery(kv,buffers,output,layer) < 0 )
				return(-13);
			if ( candidate == 0u &&
				PocLaunchIndexWeights(kv,buffers,output,layer) < 0 )
				return(-14);
		}
		if ( cudaEventRecord(kv_done[layer],kv) != cudaSuccess )
			return(-15);
		if ( PocLaunchQueryProjection(primary,buffers,output,layer) < 0 ||
			cudaStreamWaitEvent(primary,kv_done[layer],0u) != cudaSuccess )
			return(-16);
		if ( candidate != 0u && cudaStreamWaitEvent(primary,
				prologue_done[layer],0u) != cudaSuccess )
			return(-17);
	}
	if ( cudaEventRecord(stop,primary) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess ||
		cudaEventElapsedTime(elapsed_ms,start,stop) != cudaSuccess )
		return(-18);
	progress.join();
	return(0);
}

static PocTiming PocSummarize(std::vector<float> values)
{
	PocTiming timing = {};
	uint32_t index;
	std::sort(values.begin(),values.end());
	for (index=0u; index<values.size(); index++)
		timing.mean_ms += values[index];
	timing.mean_ms /= (float)values.size();
	timing.median_ms = values[values.size() / 2u];
	timing.p10_ms = values[values.size() / 10u];
	timing.p90_ms = values[(values.size() * 9u) / 10u];
	return(timing);
}

static int32_t PocMeasure(cudaStream_t primary,cudaStream_t prologue,
	cudaStream_t kv,cudaEvent_t start,cudaEvent_t stop,
	cudaEvent_t *projection_ready,cudaEvent_t *milestone,
	cudaEvent_t *prologue_done,cudaEvent_t *kv_done,const PocBuffers *buffers,
	uint64_t delay_ns,PocTiming *control,PocTiming *candidate,
	uint64_t *mean_actual_collective_ns)
{
	std::vector<float> control_times,candidate_times;
	float elapsed;
	uint64_t actual,total_actual = 0u;
	uint32_t repetition,warmup,run_count = 0u;
	for (warmup=0u; warmup<POC_WARMUP_COUNT; warmup++)
	{
		if ( PocRunStep(primary,prologue,kv,start,stop,projection_ready,milestone,
			prologue_done,kv_done,buffers,warmup & 1u,delay_ns,&elapsed,&actual) < 0 )
			return(-1);
	}
	for (repetition=0u; repetition<POC_REPETITION_COUNT; repetition++)
	{
		uint32_t first = repetition & 1u;
		if ( PocRunStep(primary,prologue,kv,start,stop,projection_ready,milestone,
			prologue_done,kv_done,buffers,first,delay_ns,&elapsed,&actual) < 0 )
			return(-2);
		(first != 0u ? candidate_times : control_times).push_back(elapsed);
		total_actual += actual;
		run_count++;
		if ( PocRunStep(primary,prologue,kv,start,stop,projection_ready,milestone,
			prologue_done,kv_done,buffers,first ^ 1u,delay_ns,&elapsed,&actual) < 0 )
			return(-3);
		(first != 0u ? control_times : candidate_times).push_back(elapsed);
		total_actual += actual;
		run_count++;
	}
	*control = PocSummarize(control_times);
	*candidate = PocSummarize(candidate_times);
	*mean_actual_collective_ns = total_actual /
		((uint64_t)run_count * POC_LAYER_COUNT);
	return(0);
}

static int32_t PocCompareBuffer(const void *control,const void *candidate,
	uint64_t bytes,const char *name)
{
	std::vector<uint8_t> first(bytes),second(bytes);
	if ( cudaMemcpy(first.data(),control,bytes,cudaMemcpyDeviceToHost) !=
		cudaSuccess || cudaMemcpy(second.data(),candidate,bytes,
		cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-1);
	if ( memcmp(first.data(),second.data(),bytes) != 0 )
	{
		fprintf(stderr,"exactness failed buffer=%s bytes=%llu\n",name,
			(unsigned long long)bytes);
		return(-2);
	}
	return(0);
}

static int32_t PocCheckExact(const PocBuffers *buffers)
{
#define POC_COMPARE(field,bytes) do { if ( PocCompareBuffer(buffers->control.field,buffers->candidate.field,(bytes),#field) < 0 ) return(-1); } while (0)
	POC_COMPARE(query_rank_bf16,(uint64_t)POC_LAYER_COUNT * POC_QUERY_RANK * sizeof(uint16_t));
	POC_COMPARE(query_bf16,(uint64_t)POC_LAYER_COUNT * POC_LOCAL_QUERY_DIMENSION * sizeof(uint16_t));
	POC_COMPARE(index_query_bf16,(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_DIMENSION * sizeof(uint16_t));
	POC_COMPARE(index_weights_bf16,(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT * sizeof(uint16_t));
	POC_COMPARE(index_weights_f32,(uint64_t)POC_CSA_LAYER_COUNT * POC_INDEX_WEIGHT_COUNT * sizeof(float));
	POC_COMPARE(attention_indices,(uint64_t)POC_LAYER_COUNT * POC_TOPK_COLUMNS * sizeof(int32_t));
	POC_COMPARE(slot_counts,POC_LAYER_COUNT * sizeof(uint32_t));
	POC_COMPARE(attention_slot_counts,POC_LAYER_COUNT * sizeof(uint32_t));
#undef POC_COMPARE
	return(0);
}

int main(void)
{
	PocBuffers buffers;
	PocTiming control,candidate;
	cudaDeviceProp properties;
	cudaStream_t primary,prologue,kv;
	cudaEvent_t start,stop;
	cudaEvent_t projection_ready[POC_LAYER_COUNT],milestone[POC_LAYER_COUNT];
	cudaEvent_t prologue_done[POC_LAYER_COUNT],kv_done[POC_LAYER_COUNT];
	uint64_t delays[] = {44000u,60000u};
	uint64_t actual_delay;
	uint32_t layer,index;
	POC_CUDA(cudaGetDeviceProperties(&properties,0));
	POC_CUDA(cudaStreamCreateWithFlags(&primary,cudaStreamNonBlocking));
	POC_CUDA(cudaStreamCreateWithFlags(&prologue,cudaStreamNonBlocking));
	POC_CUDA(cudaStreamCreateWithFlags(&kv,cudaStreamNonBlocking));
	POC_CUDA(cudaEventCreate(&start));
	POC_CUDA(cudaEventCreate(&stop));
	for (layer=0u; layer<POC_LAYER_COUNT; layer++)
	{
		POC_CUDA(cudaEventCreateWithFlags(&projection_ready[layer],
			cudaEventDisableTiming));
		POC_CUDA(cudaEventCreateWithFlags(&milestone[layer],
			cudaEventDisableTiming));
		POC_CUDA(cudaEventCreateWithFlags(&prologue_done[layer],
			cudaEventDisableTiming));
		POC_CUDA(cudaEventCreateWithFlags(&kv_done[layer],
			cudaEventDisableTiming));
	}
	if ( PocAllocate(&buffers,primary) < 0 )
		return(2);
	printf("device=%s sm_count=%d layers=%u csa_layers=%u rows=1 tp_degree=4 collective_payload_bytes=%u query_consumer_payload_bytes_per_layer=%llu index_query_consumer_payload_bytes_per_csa_layer=%llu index_weight_payload_bytes_per_csa_layer=%llu repetitions=%u\n",
		properties.name,properties.multiProcessorCount,POC_LAYER_COUNT,
		POC_CSA_LAYER_COUNT,(uint32_t)(SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
			sizeof(uint16_t)),
		(unsigned long long)POC_LOCAL_QUERY_DIMENSION * POC_QUERY_RANK,
		(unsigned long long)POC_INDEX_DIMENSION * POC_QUERY_RANK,
		(unsigned long long)POC_INDEX_WEIGHT_COUNT *
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
		POC_REPETITION_COUNT);
	for (index=0u; index<sizeof(delays) / sizeof(delays[0]); index++)
	{
		if ( PocMeasure(primary,prologue,kv,start,stop,projection_ready,milestone,
				prologue_done,kv_done,&buffers,delays[index],&control,&candidate,
				&actual_delay) < 0 )
			return(3);
		if ( PocCheckExact(&buffers) < 0 )
			return(4);
		printf("collective_target_ns=%llu collective_actual_mean_ns=%llu exact_all_bf16_and_metadata=true control_median_ms=%.6f candidate_median_ms=%.6f gain_percent=%.6f control_p10_ms=%.6f control_p90_ms=%.6f candidate_p10_ms=%.6f candidate_p90_ms=%.6f control_mean_ms=%.6f candidate_mean_ms=%.6f\n",
			(unsigned long long)delays[index],(unsigned long long)actual_delay,
			control.median_ms,candidate.median_ms,
			100.0f * ((control.median_ms / candidate.median_ms) - 1.0f),
			control.p10_ms,control.p90_ms,candidate.p10_ms,candidate.p90_ms,
			control.mean_ms,candidate.mean_ms);
	}
	return(0);
}
