#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"

#define PROBE_CSA_LAYER_COUNT 21u
#define PROBE_REPETITION_COUNT 31u
#define PROBE_FLUSH_BYTES (128u * 1024u * 1024u)

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(probe_error)); return(-1); } } while (0)

typedef struct ProbeBuffers
{
	void *initial_bf16,*control_bf16,*candidate_bf16;
	float *freqs_f32;
	uint64_t *positions_u64;
	uint8_t *flush;
	cudaStream_t stream;
	cudaEvent_t start,stop;
} ProbeBuffers;

static __global__ void ProbeFillBf16Kernel(void *destination,uint64_t count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		((__nv_bfloat16 *)destination)[index] = __float2bfloat16(
			(float)((int32_t)((index * 43u + 19u) % 149u) - 74) / 61.0f);
}

static __global__ void ProbeFillFreqsKernel(float *freqs,uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		freqs[index] = 0.000017f * (float)(index + 1u);
}

static __global__ void ProbeIndexerPostKernel(
	void *data_bf16,const float *freqs_f32,const uint64_t *row_positions,
	uint32_t head_count,uint32_t head_dim,uint32_t rope_dim,
	uint32_t quant_block)
{
	extern __shared__ float hadamard_shared[];
	uint32_t head = blockIdx.x,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	void *head_data;
	if ( head >= head_count )
		return;
	SparkDsv4RopeRow(data_bf16,freqs_f32,row_positions,0u,head,head_count,
		head_dim,rope_dim,0u);
	__syncthreads();
	SparkDsv4HadamardRow(data_bf16,head,head_dim,hadamard_shared);
	__syncthreads();
	head_data = (uint8_t *)data_bf16 +
		(uint64_t)head * head_dim * sizeof(uint16_t);
	if ( warp * quant_block < head_dim )
		SparkDsv4QuantSimGroup(head_data,0u,warp,lane,head_dim,head_dim,
			quant_block,SPARK_DSV4_MODEL_FP4_MAX,1u);
}

static int32_t ProbeInitialize(ProbeBuffers *buffers)
{
	uint64_t layer_elements = SPARK_DSV4_MODEL_INDEX_DIMENSION;
	uint64_t elements = PROBE_CSA_LAYER_COUNT * layer_elements;
	uint32_t index;
	memset(buffers,0,sizeof(*buffers));
	PROBE_CUDA(cudaStreamCreateWithFlags(&buffers->stream,cudaStreamNonBlocking));
	PROBE_CUDA(cudaEventCreate(&buffers->start));
	PROBE_CUDA(cudaEventCreate(&buffers->stop));
	PROBE_CUDA(cudaMalloc(&buffers->initial_bf16,elements*sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc(&buffers->control_bf16,elements*sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc(&buffers->candidate_bf16,elements*sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc(&buffers->freqs_f32,
		(SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION/2u)*sizeof(float)));
	PROBE_CUDA(cudaMalloc(&buffers->positions_u64,
		PROBE_CSA_LAYER_COUNT*sizeof(uint64_t)));
	PROBE_CUDA(cudaMalloc(&buffers->flush,PROBE_FLUSH_BYTES));
	ProbeFillBf16Kernel<<<(elements+255u)/256u,256u,0,buffers->stream>>>(
		buffers->initial_bf16,elements);
	ProbeFillFreqsKernel<<<1u,256u,0,buffers->stream>>>(buffers->freqs_f32,
		SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION/2u);
	for (index=0u; index<PROBE_CSA_LAYER_COUNT; index++)
	{
		uint64_t position = 12288u + index * 83u;
		PROBE_CUDA(cudaMemcpyAsync(buffers->positions_u64+index,&position,
			sizeof(position),cudaMemcpyHostToDevice,buffers->stream));
	}
	PROBE_CUDA(cudaStreamSynchronize(buffers->stream));
	return(0);
}

static void ProbeDestroy(ProbeBuffers *buffers)
{
	cudaFree(buffers->flush);
	cudaFree(buffers->positions_u64);
	cudaFree(buffers->freqs_f32);
	cudaFree(buffers->candidate_bf16);
	cudaFree(buffers->control_bf16);
	cudaFree(buffers->initial_bf16);
	cudaEventDestroy(buffers->stop);
	cudaEventDestroy(buffers->start);
	cudaStreamDestroy(buffers->stream);
}

static int32_t ProbeTimed(ProbeBuffers *buffers,uint32_t fused,float *elapsed)
{
	uint64_t layer_elements = SPARK_DSV4_MODEL_INDEX_DIMENSION;
	uint64_t bytes = layer_elements * PROBE_CSA_LAYER_COUNT * sizeof(uint16_t);
	void *data = fused != 0u ? buffers->candidate_bf16 : buffers->control_bf16;
	uint32_t layer;
	PROBE_CUDA(cudaMemcpyAsync(data,buffers->initial_bf16,bytes,
		cudaMemcpyDeviceToDevice,buffers->stream));
	PROBE_CUDA(cudaMemsetAsync(buffers->flush,(int)fused,PROBE_FLUSH_BYTES,
		buffers->stream));
	PROBE_CUDA(cudaStreamSynchronize(buffers->stream));
	PROBE_CUDA(cudaEventRecord(buffers->start,buffers->stream));
	for (layer=0u; layer<PROBE_CSA_LAYER_COUNT; layer++)
	{
		void *layer_data = (uint8_t *)data +
			layer * layer_elements * sizeof(uint16_t);
		if ( fused != 0u )
			ProbeIndexerPostKernel<<<SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
				SPARK_LM_CTA_THREADS,
				SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION*sizeof(float),
				buffers->stream>>>(layer_data,buffers->freqs_f32,
				buffers->positions_u64+layer,
				SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
				SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
				SPARK_DSV4_MODEL_FP4_QUANT_BLOCK);
		else
		{
			PROBE_CUDA(SparkDsv4LaunchRope(buffers->stream,layer_data,
				buffers->freqs_f32,buffers->positions_u64+layer,1u,
				SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
				SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u));
			PROBE_CUDA(SparkDsv4LaunchHadamard(buffers->stream,layer_data,
				SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
				SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION));
			PROBE_CUDA(SparkDsv4LaunchQuantSim(buffers->stream,layer_data,1u,
				SPARK_DSV4_MODEL_INDEX_DIMENSION,
				SPARK_DSV4_MODEL_INDEX_DIMENSION,
				SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u));
		}
	}
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaEventRecord(buffers->stop,buffers->stream));
	PROBE_CUDA(cudaEventSynchronize(buffers->stop));
	PROBE_CUDA(cudaEventElapsedTime(elapsed,buffers->start,buffers->stop));
	return(0);
}

static int32_t ProbeExact(const ProbeBuffers *buffers)
{
	uint64_t bytes = (uint64_t)PROBE_CSA_LAYER_COUNT *
		SPARK_DSV4_MODEL_INDEX_DIMENSION * sizeof(uint16_t);
	uint8_t *control = (uint8_t *)malloc((size_t)bytes);
	uint8_t *candidate = (uint8_t *)malloc((size_t)bytes);
	int32_t status = 0;
	if ( control == 0 || candidate == 0 )
		status = -1;
	if ( status == 0 &&
		(cudaMemcpy(control,buffers->control_bf16,bytes,
		cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(candidate,buffers->candidate_bf16,bytes,
		cudaMemcpyDeviceToHost) != cudaSuccess ||
		memcmp(control,candidate,(size_t)bytes) != 0) )
		status = -2;
	free(candidate);
	free(control);
	return(status);
}

static int ProbeFloatCompare(const void *first,const void *second)
{
	float a = *(const float *)first,b = *(const float *)second;
	return(a < b ? -1 : a > b ? 1 : 0);
}

static float ProbeMedian(float *values)
{
	qsort(values,PROBE_REPETITION_COUNT,sizeof(values[0]),ProbeFloatCompare);
	return(values[PROBE_REPETITION_COUNT/2u]);
}

int main(void)
{
	ProbeBuffers buffers;
	float control_ms[PROBE_REPETITION_COUNT];
	float candidate_ms[PROBE_REPETITION_COUNT];
	float control_median,candidate_median;
	uint32_t repetition,warmup;
	if ( ProbeInitialize(&buffers) < 0 )
		return(1);
	for (warmup=0u; warmup<4u; warmup++)
		if ( ProbeTimed(&buffers,0u,&control_ms[0]) < 0 ||
			ProbeTimed(&buffers,1u,&candidate_ms[0]) < 0 )
			return(2);
	for (repetition=0u; repetition<PROBE_REPETITION_COUNT; repetition++)
	{
		if ( ((repetition&1u)==0u &&
			(ProbeTimed(&buffers,0u,&control_ms[repetition])<0 ||
			ProbeTimed(&buffers,1u,&candidate_ms[repetition])<0)) ||
			((repetition&1u)!=0u &&
			(ProbeTimed(&buffers,1u,&candidate_ms[repetition])<0 ||
			ProbeTimed(&buffers,0u,&control_ms[repetition])<0)) )
			return(3);
		if ( ProbeExact(&buffers) != 0 )
			return(4);
		printf("pair=%u control_ms=%.6f candidate_ms=%.6f "
			"gain_percent=%.6f exact=1\n",repetition+1u,
			control_ms[repetition],candidate_ms[repetition],
			100.0f*(control_ms[repetition]-candidate_ms[repetition]) /
			control_ms[repetition]);
	}
	control_median = ProbeMedian(control_ms);
	candidate_median = ProbeMedian(candidate_ms);
	printf("summary control_median_ms=%.6f candidate_median_ms=%.6f "
		"gain_percent=%.6f exact=1 csa_layers=%u index_heads=%u "
		"head_dim=%u\n",control_median,candidate_median,
		100.0f*(control_median-candidate_median)/control_median,
		PROBE_CSA_LAYER_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
		SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	ProbeDestroy(&buffers);
	return(0);
}
