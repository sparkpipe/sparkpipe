#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"

#define PROBE_BOUNDARY_COUNT 86u
#define PROBE_REPETITION_COUNT 31u
#define PROBE_FLUSH_BYTES (256u * 1024u * 1024u)
#define PROBE_COLLECTIVE_NS 60000ull

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(probe_error)); return(-1); } } while (0)

typedef struct ProbeState
{
	void *streams_bf16,*initial_bf16,*residual_bf16,*reduced_bf16;
	float *partials_f32,*mixes_f32,*pre_f32,*post_f32,*comb_f32;
} ProbeState;

typedef struct ProbeShared
{
	float *fn_f32,*scale3_f32,*base_f32;
	uint8_t *flush;
	cudaStream_t stream;
	cudaEvent_t start,stop;
} ProbeShared;

static __global__ void ProbeFillBf16Kernel(void *destination,uint64_t count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		((__nv_bfloat16 *)destination)[index] = __float2bfloat16(
			(float)((int32_t)((index * 17u + 11u) % 97u) - 48) / 64.0f);
}

static __global__ void ProbeFillFloatKernel(float *destination,uint64_t count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		destination[index] = (float)((int32_t)((index * 29u + 7u) % 101u) - 50) / 256.0f;
}

static __global__ void ProbeDelayKernel(uint64_t nanoseconds)
{
	uint64_t begin,now;
	if ( blockIdx.x != 0u || threadIdx.x != 0u )
		return;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
	do
	{
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	} while ( now - begin < nanoseconds );
}

static __global__ void ProbeHcPreReduceResidualKernel(
	const void *streams_bf16,const float *pre_f32,void *reduced_bf16,
	void *residual_bf16,uint32_t row_count,uint32_t hc,uint32_t dimension,
	uint32_t tiles_per_row)
{
	uint32_t row = blockIdx.y,tile = blockIdx.x,element,stream;
	float value;
	if ( row >= row_count )
		return;
	for (element=tile*blockDim.x+threadIdx.x; element<dimension;
		element+=tiles_per_row*blockDim.x)
	{
		value = 0.0f;
		for (stream=0u; stream<hc; stream++)
		{
			uint64_t index = (((uint64_t)row * hc) + stream) * dimension + element;
			__nv_bfloat16 raw = ((const __nv_bfloat16 *)streams_bf16)[index];
			((__nv_bfloat16 *)residual_bf16)[index] = raw;
			value += pre_f32[((uint64_t)row * hc) + stream] * __bfloat162float(raw);
		}
		SparkLmFloatToBf16(reduced_bf16,((uint64_t)row * dimension) + element,value);
	}
}

static cudaError_t ProbeLaunchHcPreReduceResidual(cudaStream_t stream,
	const void *streams_bf16,const float *pre_f32,void *reduced_bf16,
	void *residual_bf16)
{
	dim3 grid(SPARK_DSV4_HC_MINIMUM_BLOCKS,1u);
	ProbeHcPreReduceResidualKernel<<<grid,SPARK_DSV4_HC_ELEMENT_TILE,0,stream>>>(
		streams_bf16,pre_f32,reduced_bf16,residual_bf16,1u,
		SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
		SPARK_DSV4_HC_MINIMUM_BLOCKS);
	return(cudaGetLastError());
}

static int32_t ProbeAllocateState(ProbeState *state)
{
	uint64_t stream_bytes = (uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t);
	uint64_t partial_bytes = (uint64_t)SPARK_DSV4_HC_SPLIT_K_COUNT * SPARK_DSV4_HC_SPLIT_K_PARTIALS * sizeof(float);
	memset(state,0,sizeof(*state));
	PROBE_CUDA(cudaMalloc(&state->streams_bf16,stream_bytes));
	PROBE_CUDA(cudaMalloc(&state->initial_bf16,stream_bytes));
	PROBE_CUDA(cudaMalloc(&state->residual_bf16,stream_bytes));
	PROBE_CUDA(cudaMalloc(&state->reduced_bf16,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc(&state->partials_f32,partial_bytes));
	PROBE_CUDA(cudaMalloc(&state->mixes_f32,SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&state->pre_f32,SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&state->post_f32,SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&state->comb_f32,SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float)));
	return(0);
}

static void ProbeFreeState(ProbeState *state)
{
	cudaFree(state->comb_f32);
	cudaFree(state->post_f32);
	cudaFree(state->pre_f32);
	cudaFree(state->mixes_f32);
	cudaFree(state->partials_f32);
	cudaFree(state->reduced_bf16);
	cudaFree(state->residual_bf16);
	cudaFree(state->initial_bf16);
	cudaFree(state->streams_bf16);
}

static int32_t ProbeInitialize(ProbeShared *shared,ProbeState *control,
	ProbeState *candidate)
{
	uint64_t stream_elements = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	uint64_t fn_elements = (uint64_t)PROBE_BOUNDARY_COUNT * SPARK_DSV4_MODEL_HC_MIX_ROWS * stream_elements;
	memset(shared,0,sizeof(*shared));
	if ( ProbeAllocateState(control) < 0 || ProbeAllocateState(candidate) < 0 )
		return(-1);
	PROBE_CUDA(cudaStreamCreateWithFlags(&shared->stream,cudaStreamNonBlocking));
	PROBE_CUDA(cudaEventCreate(&shared->start));
	PROBE_CUDA(cudaEventCreate(&shared->stop));
	PROBE_CUDA(cudaMalloc(&shared->fn_f32,fn_elements * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&shared->scale3_f32,3u * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&shared->base_f32,SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float)));
	PROBE_CUDA(cudaMalloc(&shared->flush,PROBE_FLUSH_BYTES));
	ProbeFillBf16Kernel<<<64u,256u,0,shared->stream>>>(control->initial_bf16,stream_elements);
	PROBE_CUDA(cudaMemcpyAsync(candidate->initial_bf16,control->initial_bf16,stream_elements*sizeof(uint16_t),cudaMemcpyDeviceToDevice,shared->stream));
	ProbeFillFloatKernel<<<(fn_elements+255u)/256u,256u,0,shared->stream>>>(shared->fn_f32,fn_elements);
	ProbeFillFloatKernel<<<1u,256u,0,shared->stream>>>(shared->scale3_f32,3u);
	ProbeFillFloatKernel<<<1u,256u,0,shared->stream>>>(shared->base_f32,SPARK_DSV4_MODEL_HC_MIX_ROWS);
	PROBE_CUDA(cudaStreamSynchronize(shared->stream));
	return(0);
}

static void ProbeDestroy(ProbeShared *shared,ProbeState *control,
	ProbeState *candidate)
{
	cudaFree(shared->flush);
	cudaFree(shared->base_f32);
	cudaFree(shared->scale3_f32);
	cudaFree(shared->fn_f32);
	cudaEventDestroy(shared->stop);
	cudaEventDestroy(shared->start);
	cudaStreamDestroy(shared->stream);
	ProbeFreeState(candidate);
	ProbeFreeState(control);
}

static cudaError_t ProbeRunBoundary(ProbeShared *shared,ProbeState *state,
	uint32_t boundary,uint32_t fused)
{
	const float *fn = shared->fn_f32 + (uint64_t)boundary * SPARK_DSV4_MODEL_HC_MIX_ROWS * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	cudaError_t error;
	if ( fused == 0u )
		error = cudaMemcpyAsync(state->residual_bf16,state->streams_bf16,
			(uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS*sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,shared->stream);
	else
		error = cudaSuccess;
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcMixSplitKSinkhorn(shared->stream,
			state->streams_bf16,fn,shared->scale3_f32,shared->base_f32,
			state->partials_f32,state->mixes_f32,1u,
			SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,
			SPARK_DSV4_MODEL_HC_MIX_ROWS,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON,SPARK_DSV4_MODEL_HC_EPSILON,
			state->pre_f32,state->post_f32,state->comb_f32);
	if ( error == cudaSuccess )
		error = fused != 0u ? ProbeLaunchHcPreReduceResidual(shared->stream,
			state->streams_bf16,state->pre_f32,state->reduced_bf16,
			state->residual_bf16) : SparkDsv4LaunchHcPreReduce(shared->stream,
			state->streams_bf16,state->pre_f32,state->reduced_bf16,1u,
			SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		ProbeDelayKernel<<<1u,1u,0,shared->stream>>>(PROBE_COLLECTIVE_NS);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPost(shared->stream,state->reduced_bf16,
			state->residual_bf16,state->post_f32,state->comb_f32,
			state->streams_bf16,1u,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(error == cudaSuccess ? cudaGetLastError() : error);
}

static int32_t ProbeTimedStep(ProbeShared *shared,ProbeState *state,
	uint32_t fused,float *milliseconds)
{
	uint32_t boundary;
	PROBE_CUDA(cudaMemcpyAsync(state->streams_bf16,state->initial_bf16,
		(uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS*sizeof(uint16_t),
		cudaMemcpyDeviceToDevice,shared->stream));
	PROBE_CUDA(cudaMemsetAsync(shared->flush,(int)fused,PROBE_FLUSH_BYTES,
		shared->stream));
	PROBE_CUDA(cudaStreamSynchronize(shared->stream));
	PROBE_CUDA(cudaEventRecord(shared->start,shared->stream));
	for (boundary=0u; boundary<PROBE_BOUNDARY_COUNT; boundary++)
		PROBE_CUDA(ProbeRunBoundary(shared,state,boundary,fused));
	PROBE_CUDA(cudaEventRecord(shared->stop,shared->stream));
	PROBE_CUDA(cudaEventSynchronize(shared->stop));
	PROBE_CUDA(cudaEventElapsedTime(milliseconds,shared->start,shared->stop));
	return(0);
}

static int32_t ProbeCompare(const ProbeState *control,
	const ProbeState *candidate)
{
	uint64_t stream_bytes = (uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t);
	uint64_t reduced_bytes = (uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	uint8_t *first = (uint8_t *)malloc((size_t)stream_bytes);
	uint8_t *second = (uint8_t *)malloc((size_t)stream_bytes);
	int32_t mismatch = 0;
	if ( first == 0 || second == 0 )
		mismatch = -1;
	if ( mismatch == 0 && (cudaMemcpy(first,control->streams_bf16,stream_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(second,candidate->streams_bf16,stream_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || memcmp(first,second,(size_t)stream_bytes) != 0) )
		mismatch = -2;
	if ( mismatch == 0 && (cudaMemcpy(first,control->residual_bf16,stream_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(second,candidate->residual_bf16,stream_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || memcmp(first,second,(size_t)stream_bytes) != 0) )
		mismatch = -3;
	if ( mismatch == 0 && (cudaMemcpy(first,control->reduced_bf16,reduced_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(second,candidate->reduced_bf16,reduced_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || memcmp(first,second,(size_t)reduced_bytes) != 0) )
		mismatch = -4;
	free(second);
	free(first);
	return(mismatch);
}

static int ProbeFloatCompare(const void *first,const void *second)
{
	float a = *(const float *)first,b = *(const float *)second;
	return(a < b ? -1 : a > b ? 1 : 0);
}

static float ProbeMedian(float *values)
{
	qsort(values,PROBE_REPETITION_COUNT,sizeof(values[0]),ProbeFloatCompare);
	return(values[PROBE_REPETITION_COUNT / 2u]);
}

int main(void)
{
	ProbeShared shared;
	ProbeState control,candidate;
	float control_ms[PROBE_REPETITION_COUNT],candidate_ms[PROBE_REPETITION_COUNT];
	float control_median,candidate_median;
	uint32_t repetition,warmup;
	if ( ProbeInitialize(&shared,&control,&candidate) < 0 )
		return(1);
	for (warmup=0u; warmup<4u; warmup++)
		if ( ProbeTimedStep(&shared,&control,0u,&control_ms[0]) < 0 || ProbeTimedStep(&shared,&candidate,1u,&candidate_ms[0]) < 0 )
			return(2);
	for (repetition=0u; repetition<PROBE_REPETITION_COUNT; repetition++)
	{
		if ( ((repetition & 1u) == 0u && (ProbeTimedStep(&shared,&control,0u,&control_ms[repetition]) < 0 || ProbeTimedStep(&shared,&candidate,1u,&candidate_ms[repetition]) < 0)) || ((repetition & 1u) != 0u && (ProbeTimedStep(&shared,&candidate,1u,&candidate_ms[repetition]) < 0 || ProbeTimedStep(&shared,&control,0u,&control_ms[repetition]) < 0)) )
			return(3);
		if ( ProbeCompare(&control,&candidate) != 0 )
			return(4);
		printf("pair=%u control_ms=%.6f candidate_ms=%.6f gain_percent=%.6f exact=1\n",repetition+1u,control_ms[repetition],candidate_ms[repetition],100.0f*(control_ms[repetition]-candidate_ms[repetition])/control_ms[repetition]);
	}
	control_median = ProbeMedian(control_ms);
	candidate_median = ProbeMedian(candidate_ms);
	printf("summary control_median_ms=%.6f candidate_median_ms=%.6f gain_percent=%.6f exact=1 boundaries=%u residual_bytes=%u collective_ns=%llu\n",control_median,candidate_median,100.0f*(control_median-candidate_median)/control_median,PROBE_BOUNDARY_COUNT,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS*2u,(unsigned long long)PROBE_COLLECTIVE_NS);
	ProbeDestroy(&shared,&control,&candidate);
	return(0);
}
