#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"

#define PROBE_BOUNDARY_COUNT 86u
#define PROBE_MAX_REPETITIONS 31u
#define PROBE_WARMUP_COUNT 4u
#define PROBE_FLUSH_BYTES (256u * 1024u * 1024u)
#define PROBE_BLOCK_THREADS SPARK_DSV4_HC_ELEMENT_TILE

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(probe_error)); return(-1); } } while (0)

typedef struct ProbeInputs
{
	void *streams_bf16;
	float *fn_f32;
	float *scale3_f32;
	float *base_f32;
	float *partials_f32;
} ProbeInputs;

typedef struct ProbeOutputs
{
	float *mixes_f32;
	float *pre_f32;
	float *post_f32;
	float *comb_f32;
	void *reduced_bf16;
	void *residual_bf16;
} ProbeOutputs;

typedef struct ProbeRuntime
{
	ProbeInputs inputs;
	ProbeOutputs control;
	ProbeOutputs candidate;
	uint8_t *flush;
	cudaStream_t stream;
	cudaEvent_t start;
	cudaEvent_t stop;
	uint32_t cooperative_capacity;
} ProbeRuntime;

static __global__ void ProbeFillBf16Kernel(void *destination,uint64_t count,
	uint32_t seed)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	for (; index<count; index+=stride)
	{
		int32_t raw = (int32_t)((index * 17u + (uint64_t)seed * 31u + 11u) % 127u) - 63;
		((__nv_bfloat16 *)destination)[index] = __float2bfloat16((float)raw / 64.0f);
	}
}

static __global__ void ProbeFillFnKernel(float *destination,uint64_t count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	for (; index<count; index+=stride)
	{
		int32_t raw = (int32_t)((index * 29u + (index >> 14u) * 43u + 7u) % 257u) - 128;
		destination[index] = (float)raw / 1024.0f;
	}
}

static __global__ void ProbeFillBoundaryWeightsKernel(float *scale3_f32,
	float *base_f32)
{
	uint32_t boundary = blockIdx.x * blockDim.x + threadIdx.x,index;
	if ( boundary >= PROBE_BOUNDARY_COUNT )
		return;
	for (index=0u; index<3u; index++)
		scale3_f32[(uint64_t)boundary * 3u + index] = 0.25f +
		(float)((boundary * 7u + index * 11u) % 37u) / 64.0f;
	for (index=0u; index<SPARK_DSV4_MODEL_HC_MIX_ROWS; index++)
		base_f32[(uint64_t)boundary * SPARK_DSV4_MODEL_HC_MIX_ROWS + index] =
		(float)((int32_t)((boundary * 13u + index * 19u) % 53u) - 26) / 128.0f;
}

static __global__ void ProbeHcFinalizePreReduceCooperativeKernel(
	const float *partials_f32,const float *scale3_f32,const float *base_f32,
	const void *streams_bf16,uint32_t iterations,float rms_epsilon,
	float hc_epsilon,float *mixes_f32,float *pre_f32,float *post_f32,
	float *comb_f32,void *reduced_bf16,void *residual_bf16)
{
	__shared__ float mixes[SPARK_DSV4_MODEL_HC_MIX_ROWS];
	__shared__ float comb[SPARK_DSV4_MODEL_HC_STREAM_COUNT *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	__shared__ float sums[SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	uint32_t lane = threadIdx.x,split,element,stream;
	uint64_t partial_base,index;
	float total = 0.0f,inverse,value;
	__nv_bfloat16 raw;
	if ( blockIdx.x == 0u )
	{
		if ( lane < SPARK_DSV4_HC_SPLIT_K_PARTIALS )
			for (split=0u; split<SPARK_DSV4_HC_SPLIT_K_COUNT; split++)
			{
				partial_base = (uint64_t)split * SPARK_DSV4_HC_SPLIT_K_PARTIALS;
				total += partials_f32[partial_base + lane];
			}
		if ( lane == 0u )
			sums[0] = rsqrtf(total /
				(float)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS + rms_epsilon);
		__syncthreads();
		if ( lane > 0u && lane < SPARK_DSV4_HC_SPLIT_K_PARTIALS )
		{
			inverse = sums[0];
			mixes[lane - 1u] = total * inverse;
			mixes_f32[lane - 1u] = total * inverse;
		}
		__syncthreads();
		SparkDsv4HcParallelSinkhorn(mixes,scale3_f32,base_f32,iterations,
			hc_epsilon,pre_f32,post_f32,comb_f32,0u,comb,sums);
	}
	grid.sync();
	for (element=blockIdx.x * blockDim.x + threadIdx.x;
		element<SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		element+=gridDim.x * blockDim.x)
	{
		value = 0.0f;
		for (stream=0u; stream<SPARK_DSV4_MODEL_HC_STREAM_COUNT; stream++)
		{
			index = (uint64_t)stream * SPARK_DSV4_MODEL_HIDDEN_DIMENSION + element;
			raw = ((const __nv_bfloat16 *)streams_bf16)[index];
			((__nv_bfloat16 *)residual_bf16)[index] = raw;
			value += pre_f32[stream] * __bfloat162float(raw);
		}
		SparkLmFloatToBf16(reduced_bf16,element,value);
	}
}

static int32_t ProbeAllocateOutputs(ProbeOutputs *outputs)
{
	uint64_t boundary_count = PROBE_BOUNDARY_COUNT;
	memset(outputs,0,sizeof(*outputs));
	PROBE_CUDA(cudaMalloc((void **)&outputs->mixes_f32,boundary_count *
		SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&outputs->pre_f32,boundary_count *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&outputs->post_f32,boundary_count *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&outputs->comb_f32,boundary_count *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT *
		sizeof(float)));
	PROBE_CUDA(cudaMalloc(&outputs->reduced_bf16,boundary_count *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc(&outputs->residual_bf16,boundary_count *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t)));
	return(0);
}

static void ProbeFreeOutputs(ProbeOutputs *outputs)
{
	cudaFree(outputs->residual_bf16);
	cudaFree(outputs->reduced_bf16);
	cudaFree(outputs->comb_f32);
	cudaFree(outputs->post_f32);
	cudaFree(outputs->pre_f32);
	cudaFree(outputs->mixes_f32);
	memset(outputs,0,sizeof(*outputs));
}

static int32_t ProbeAllocateInputs(ProbeInputs *inputs)
{
	uint64_t boundaries = PROBE_BOUNDARY_COUNT;
	uint64_t flat = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	memset(inputs,0,sizeof(*inputs));
	PROBE_CUDA(cudaMalloc(&inputs->streams_bf16,boundaries * flat *
		sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&inputs->fn_f32,boundaries *
		SPARK_DSV4_MODEL_HC_MIX_ROWS * flat * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&inputs->scale3_f32,boundaries * 3u *
		sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&inputs->base_f32,boundaries *
		SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&inputs->partials_f32,boundaries *
		SPARK_DSV4_HC_SPLIT_K_COUNT * SPARK_DSV4_HC_SPLIT_K_PARTIALS *
		sizeof(float)));
	return(0);
}

static void ProbeFreeInputs(ProbeInputs *inputs)
{
	cudaFree(inputs->partials_f32);
	cudaFree(inputs->base_f32);
	cudaFree(inputs->scale3_f32);
	cudaFree(inputs->fn_f32);
	cudaFree(inputs->streams_bf16);
	memset(inputs,0,sizeof(*inputs));
}

static cudaError_t ProbePreparePartials(ProbeRuntime *runtime)
{
	uint64_t flat = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	uint64_t partial_stride = (uint64_t)SPARK_DSV4_HC_SPLIT_K_COUNT *
		SPARK_DSV4_HC_SPLIT_K_PARTIALS;
	uint32_t boundary;
	for (boundary=0u; boundary<PROBE_BOUNDARY_COUNT; boundary++)
		SparkDsv4HcMixSplitKKernel<<<SPARK_DSV4_HC_SPLIT_K_COUNT,
			SPARK_LM_CTA_THREADS,0,runtime->stream>>>(
			(const uint16_t *)runtime->inputs.streams_bf16 +
				(uint64_t)boundary * flat,
			runtime->inputs.fn_f32 + (uint64_t)boundary *
				SPARK_DSV4_MODEL_HC_MIX_ROWS * flat,
			runtime->inputs.partials_f32 + (uint64_t)boundary * partial_stride,
			1u,(uint32_t)flat);
	return(cudaGetLastError());
}

static int32_t ProbeInitialize(ProbeRuntime *runtime)
{
	uint64_t stream_count = (uint64_t)PROBE_BOUNDARY_COUNT *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	uint64_t fn_count = stream_count * SPARK_DSV4_MODEL_HC_MIX_ROWS;
	int32_t blocks_per_sm,device,sm_count,cooperative;
	memset(runtime,0,sizeof(*runtime));
	if ( ProbeAllocateInputs(&runtime->inputs) < 0 ||
		ProbeAllocateOutputs(&runtime->control) < 0 ||
		ProbeAllocateOutputs(&runtime->candidate) < 0 )
		return(-1);
	PROBE_CUDA(cudaStreamCreateWithFlags(&runtime->stream,cudaStreamNonBlocking));
	PROBE_CUDA(cudaEventCreate(&runtime->start));
	PROBE_CUDA(cudaEventCreate(&runtime->stop));
	PROBE_CUDA(cudaMalloc((void **)&runtime->flush,PROBE_FLUSH_BYTES));
	ProbeFillBf16Kernel<<<1024u,256u,0,runtime->stream>>>(
		runtime->inputs.streams_bf16,stream_count,41u);
	ProbeFillFnKernel<<<65535u,256u,0,runtime->stream>>>(runtime->inputs.fn_f32,
		fn_count);
	ProbeFillBoundaryWeightsKernel<<<1u,128u,0,runtime->stream>>>(
		runtime->inputs.scale3_f32,runtime->inputs.base_f32);
	PROBE_CUDA(ProbePreparePartials(runtime));
	PROBE_CUDA(cudaStreamSynchronize(runtime->stream));
	PROBE_CUDA(cudaGetDevice(&device));
	PROBE_CUDA(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,
		device));
	if ( cooperative == 0 )
		return(-2);
	PROBE_CUDA(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,
		device));
	PROBE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
		ProbeHcFinalizePreReduceCooperativeKernel,PROBE_BLOCK_THREADS,0u));
	runtime->cooperative_capacity = (uint32_t)(blocks_per_sm * sm_count);
	return(0);
}

static void ProbeDestroy(ProbeRuntime *runtime)
{
	cudaFree(runtime->flush);
	cudaEventDestroy(runtime->stop);
	cudaEventDestroy(runtime->start);
	cudaStreamDestroy(runtime->stream);
	ProbeFreeOutputs(&runtime->candidate);
	ProbeFreeOutputs(&runtime->control);
	ProbeFreeInputs(&runtime->inputs);
}

static void ProbeBoundaryPointers(const ProbeRuntime *runtime,
	const ProbeOutputs *outputs,uint32_t boundary,const void **streams,
	const float **partials,const float **scale3,const float **base,float **mixes,
	float **pre,float **post,float **comb,void **reduced,void **residual)
{
	*streams = (const uint16_t *)runtime->inputs.streams_bf16 +
		(uint64_t)boundary * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	*partials = runtime->inputs.partials_f32 + (uint64_t)boundary *
		SPARK_DSV4_HC_SPLIT_K_COUNT * SPARK_DSV4_HC_SPLIT_K_PARTIALS;
	*scale3 = runtime->inputs.scale3_f32 + (uint64_t)boundary * 3u;
	*base = runtime->inputs.base_f32 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HC_MIX_ROWS;
	*mixes = outputs->mixes_f32 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HC_MIX_ROWS;
	*pre = outputs->pre_f32 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT;
	*post = outputs->post_f32 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT;
	*comb = outputs->comb_f32 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT;
	*reduced = (uint16_t *)outputs->reduced_bf16 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	*residual = (uint16_t *)outputs->residual_bf16 + (uint64_t)boundary *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
}

static cudaError_t ProbeLaunchControl(ProbeRuntime *runtime,
	ProbeOutputs *outputs,uint32_t boundary)
{
	const void *streams;
	const float *partials,*scale3,*base;
	float *mixes,*pre,*post,*comb;
	void *reduced,*residual;
	cudaError_t error;
	ProbeBoundaryPointers(runtime,outputs,boundary,&streams,&partials,&scale3,
		&base,&mixes,&pre,&post,&comb,&reduced,&residual);
	SparkDsv4HcMixSplitKFinalizeKernel<<<1u,64u,0,runtime->stream>>>(partials,
		scale3,base,1u,SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON,SPARK_DSV4_MODEL_HC_EPSILON,
		mixes,pre,post,comb);
	error = cudaGetLastError();
	if ( error != cudaSuccess )
		return(error);
	SparkDsv4HcPreReduceKernel<<<dim3(SPARK_DSV4_HC_MINIMUM_BLOCKS,1u),
		SPARK_DSV4_HC_ELEMENT_TILE,0,runtime->stream>>>(streams,pre,reduced,
		residual,1u,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_HC_MINIMUM_BLOCKS);
	return(cudaGetLastError());
}

static cudaError_t ProbeLaunchCandidate(ProbeRuntime *runtime,
	ProbeOutputs *outputs,uint32_t boundary,uint32_t grid_blocks)
{
	const void *streams;
	const float *partials,*scale3,*base;
	float *mixes,*pre,*post,*comb;
	void *reduced,*residual;
	uint32_t iterations = SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS;
	float rms_epsilon = SPARK_DSV4_MODEL_RMS_NORM_EPSILON;
	float hc_epsilon = SPARK_DSV4_MODEL_HC_EPSILON;
	void *arguments[13];
	ProbeBoundaryPointers(runtime,outputs,boundary,&streams,&partials,&scale3,
		&base,&mixes,&pre,&post,&comb,&reduced,&residual);
	arguments[0] = (void *)&partials;
	arguments[1] = (void *)&scale3;
	arguments[2] = (void *)&base;
	arguments[3] = (void *)&streams;
	arguments[4] = (void *)&iterations;
	arguments[5] = (void *)&rms_epsilon;
	arguments[6] = (void *)&hc_epsilon;
	arguments[7] = (void *)&mixes;
	arguments[8] = (void *)&pre;
	arguments[9] = (void *)&post;
	arguments[10] = (void *)&comb;
	arguments[11] = (void *)&reduced;
	arguments[12] = (void *)&residual;
	return(cudaLaunchCooperativeKernel(
		(const void *)ProbeHcFinalizePreReduceCooperativeKernel,
		dim3(grid_blocks),dim3(PROBE_BLOCK_THREADS),arguments,0u,runtime->stream));
}

static int32_t ProbeTimedStep(ProbeRuntime *runtime,ProbeOutputs *outputs,
	uint32_t candidate,uint32_t grid_blocks,uint32_t salt,float *milliseconds)
{
	uint32_t boundary;
	PROBE_CUDA(cudaMemsetAsync(runtime->flush,(int)salt,PROBE_FLUSH_BYTES,
		runtime->stream));
	PROBE_CUDA(cudaStreamSynchronize(runtime->stream));
	PROBE_CUDA(cudaEventRecord(runtime->start,runtime->stream));
	for (boundary=0u; boundary<PROBE_BOUNDARY_COUNT; boundary++)
		PROBE_CUDA(candidate != 0u ? ProbeLaunchCandidate(runtime,outputs,boundary,
			grid_blocks) : ProbeLaunchControl(runtime,outputs,boundary));
	PROBE_CUDA(cudaEventRecord(runtime->stop,runtime->stream));
	PROBE_CUDA(cudaEventSynchronize(runtime->stop));
	PROBE_CUDA(cudaEventElapsedTime(milliseconds,runtime->start,runtime->stop));
	return(0);
}

static int32_t ProbeCompareBuffer(const void *first,const void *second,
	uint64_t bytes,const char *label)
{
	uint8_t *a = (uint8_t *)malloc((size_t)bytes);
	uint8_t *b = (uint8_t *)malloc((size_t)bytes);
	int32_t result = 0;
	if ( a == 0 || b == 0 ) result = -1;
	if ( result == 0 && cudaMemcpy(a,first,bytes,cudaMemcpyDeviceToHost) !=
		cudaSuccess ) result = -2;
	if ( result == 0 && cudaMemcpy(b,second,bytes,cudaMemcpyDeviceToHost) !=
		cudaSuccess ) result = -3;
	if ( result == 0 && memcmp(a,b,(size_t)bytes) != 0 ) result = -4;
	if ( result != 0 ) fprintf(stderr,"exactness_failed buffer=%s code=%d\n",
		label,result);
	free(b);
	free(a);
	return(result);
}

static int32_t ProbeCompare(const ProbeRuntime *runtime)
{
	uint64_t boundaries = PROBE_BOUNDARY_COUNT;
	if ( ProbeCompareBuffer(runtime->control.mixes_f32,
		runtime->candidate.mixes_f32,boundaries *
		SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float),"mixes_f32") != 0 ) return(-1);
	if ( ProbeCompareBuffer(runtime->control.pre_f32,runtime->candidate.pre_f32,
		boundaries * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),
		"pre_f32") != 0 ) return(-2);
	if ( ProbeCompareBuffer(runtime->control.post_f32,runtime->candidate.post_f32,
		boundaries * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),
		"post_f32") != 0 ) return(-3);
	if ( ProbeCompareBuffer(runtime->control.comb_f32,runtime->candidate.comb_f32,
		boundaries * SPARK_DSV4_MODEL_HC_STREAM_COUNT *
		SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),"comb_f32") != 0 ) return(-4);
	if ( ProbeCompareBuffer(runtime->control.reduced_bf16,
		runtime->candidate.reduced_bf16,boundaries *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
		"reduced_bf16") != 0 ) return(-5);
	if ( ProbeCompareBuffer(runtime->control.residual_bf16,
		runtime->candidate.residual_bf16,boundaries *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t),
		"residual_bf16") != 0 ) return(-6);
	return(ProbeCompareBuffer(runtime->control.residual_bf16,
		runtime->inputs.streams_bf16,boundaries *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t),
		"raw_residual_input"));
}

static int ProbeFloatCompare(const void *first,const void *second)
{
	float a = *(const float *)first,b = *(const float *)second;
	return(a < b ? -1 : a > b ? 1 : 0);
}

static float ProbeMedian(float *values,uint32_t count)
{
	qsort(values,count,sizeof(values[0]),ProbeFloatCompare);
	return(values[count / 2u]);
}

int main(int argument_count,char **arguments)
{
	ProbeRuntime runtime;
	float control_ms[PROBE_MAX_REPETITIONS],candidate_ms[PROBE_MAX_REPETITIONS];
	float control_median,candidate_median,saving_ms;
	uint32_t grid_blocks,repetitions,repetition,warmup;
	if ( argument_count < 2 || argument_count > 3 )
	{
		fprintf(stderr,"usage: %s GRID_BLOCKS [REPETITIONS]\n",arguments[0]);
		return(2);
	}
	grid_blocks = (uint32_t)strtoul(arguments[1],0,10);
	repetitions = argument_count == 3 ? (uint32_t)strtoul(arguments[2],0,10) :
		PROBE_MAX_REPETITIONS;
	if ( grid_blocks == 0u || grid_blocks > SPARK_DSV4_HC_MINIMUM_BLOCKS ||
		repetitions == 0u || repetitions > PROBE_MAX_REPETITIONS )
		return(2);
	if ( ProbeInitialize(&runtime) < 0 )
		return(1);
	if ( grid_blocks > runtime.cooperative_capacity )
	{
		fprintf(stderr,"grid=%u cooperative_capacity=%u\n",grid_blocks,
			runtime.cooperative_capacity);
		return(2);
	}
	for (warmup=0u; warmup<PROBE_WARMUP_COUNT; warmup++)
		if ( ProbeTimedStep(&runtime,&runtime.control,0u,grid_blocks,warmup,
			&control_ms[0]) < 0 || ProbeTimedStep(&runtime,&runtime.candidate,1u,
			grid_blocks,warmup + 17u,&candidate_ms[0]) < 0 ) return(3);
	for (repetition=0u; repetition<repetitions; repetition++)
	{
		if ( (repetition & 1u) == 0u )
		{
			if ( ProbeTimedStep(&runtime,&runtime.control,0u,grid_blocks,
				repetition + 31u,&control_ms[repetition]) < 0 ||
				ProbeTimedStep(&runtime,&runtime.candidate,1u,grid_blocks,
				repetition + 73u,&candidate_ms[repetition]) < 0 ) return(4);
		}
		else if ( ProbeTimedStep(&runtime,&runtime.candidate,1u,grid_blocks,
			repetition + 73u,&candidate_ms[repetition]) < 0 ||
			ProbeTimedStep(&runtime,&runtime.control,0u,grid_blocks,
			repetition + 31u,&control_ms[repetition]) < 0 ) return(4);
		if ( ProbeCompare(&runtime) != 0 ) return(5);
		printf("pair=%u grid=%u control_ms=%.6f candidate_ms=%.6f gain_percent=%.6f exact=1\n",
			repetition + 1u,grid_blocks,control_ms[repetition],
			candidate_ms[repetition],100.0f *
			(control_ms[repetition] - candidate_ms[repetition]) /
			control_ms[repetition]);
	}
	control_median = ProbeMedian(control_ms,repetitions);
	candidate_median = ProbeMedian(candidate_ms,repetitions);
	saving_ms = control_median - candidate_median;
	printf("summary grid=%u cooperative_capacity=%u control_median_ms=%.6f candidate_median_ms=%.6f gain_percent=%.6f exact=1 boundaries=%u control_us_per_boundary=%.6f candidate_us_per_boundary=%.6f saving_us_per_boundary=%.6f model_step_saving_ms=%.6f\n",
		grid_blocks,runtime.cooperative_capacity,control_median,candidate_median,
		100.0f * saving_ms / control_median,PROBE_BOUNDARY_COUNT,
		control_median * 1000.0f / PROBE_BOUNDARY_COUNT,
		candidate_median * 1000.0f / PROBE_BOUNDARY_COUNT,
		saving_ms * 1000.0f / PROBE_BOUNDARY_COUNT,saving_ms);
	ProbeDestroy(&runtime);
	return(0);
}
