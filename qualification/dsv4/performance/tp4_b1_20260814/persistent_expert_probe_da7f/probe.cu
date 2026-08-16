#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_lm_kernels.cuh"

#define POC_GROUP_COUNT 256u
#define POC_ACTIVE_GROUP_COUNT 6u
#define POC_HIDDEN_DIMENSION 4096u
#define POC_EXPERT_DIMENSION 512u
#define POC_W13_TILE_N 32u
#define POC_W2_TILE_N 128u
#define POC_CONTROL_W13_BLOCKS_PER_SM 2u
#define POC_CONTROL_W2_BLOCKS_PER_SM 4u
#define POC_REPETITION_COUNT 101u
#define POC_WARMUP_COUNT 8u
#define POC_FLUSH_BYTES (256u * 1024u * 1024u)

#define POC_CUDA(call) do { cudaError_t poc_error = (call); if ( poc_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(poc_error)); return(-1); } } while (0)

typedef struct PocDeviceBuffers
{
	uint8_t *w1_payload,*w1_scale,*w3_payload,*w3_scale;
	uint8_t *w2_payload,*w2_scale,*flush;
	__nv_bfloat16 *input,*control_up,*control_output;
	__nv_bfloat16 *candidate_up,*candidate_output;
	uint32_t *source_rows,*row_offsets,*w13_prefix,*w2_prefix,*flush_sink;
} PocDeviceBuffers;

typedef struct PocTiming
{
	float median_ms,p10_ms,p90_ms,mean_ms;
} PocTiming;

namespace cg = cooperative_groups;

static const uint32_t PocActiveGroups[POC_ACTIVE_GROUP_COUNT] = {
	7u,43u,91u,129u,186u,241u
};

static __global__ void PocFlushKernel(const uint4 *values,uint64_t count,
	uint32_t *sink)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t value = 0u;
	for (; index<count; index+=stride)
	{
		uint4 item = values[index];
		value ^= item.x ^ item.y ^ item.z ^ item.w;
	}
	if ( value != 0u )
		atomicXor(sink,value);
}

template<uint32_t W13_TILE_N,uint32_t W2_TILE_N>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void PocPersistentExpertChainKernel(
	const uint8_t *w1_payload,const uint8_t *w1_scale,
	const uint8_t *w3_payload,const uint8_t *w3_scale,
	const uint8_t *w2_payload,const uint8_t *w2_scale,
	const void *input_bf16,const uint32_t *source_rows,
	const uint32_t *row_offsets,const uint32_t *w13_prefix,
	const uint32_t *w2_prefix,void *up_bf16,void *output_bf16,
	uint32_t group_count)
{
	extern __shared__ float shared_input[];
	cg::grid_group grid = cg::this_grid();
	uint32_t task,group,row,neuron_base;
	uint32_t total_tasks = __ldg(w13_prefix + group_count);
	for (task=blockIdx.x; task<total_tasks; task+=gridDim.x)
	{
		SparkLmSm121B1ExpertTask<W13_TILE_N>(row_offsets,w13_prefix,
			group_count,POC_EXPERT_DIMENSION,task,&group,&row,&neuron_base);
		SparkLmSm121B1ExpertW13Task<W13_TILE_N>(w1_payload,w1_scale,
			w3_payload,w3_scale,input_bf16,source_rows,up_bf16,1u,group,row,
			neuron_base,POC_HIDDEN_DIMENSION,POC_EXPERT_DIMENSION,10.0f,
			shared_input);
	}
	grid.sync();
	total_tasks = __ldg(w2_prefix + group_count);
	for (task=blockIdx.x; task<total_tasks; task+=gridDim.x)
	{
		SparkLmSm121B1ExpertTask<W2_TILE_N>(row_offsets,w2_prefix,
			group_count,POC_HIDDEN_DIMENSION,task,&group,&row,&neuron_base);
		SparkLmSm121B1ExpertW2Task<W2_TILE_N>(w2_payload,w2_scale,up_bf16,
			output_bf16,group,row,neuron_base,POC_EXPERT_DIMENSION,
			POC_HIDDEN_DIMENSION,shared_input);
	}
}

static __global__ void PocFillInputKernel(__nv_bfloat16 *values,uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		values[index] = __float2bfloat16(((float)((index * 37u) % 251u) -
			125.0f) / 128.0f);
}

static __global__ void PocFillBytesKernel(uint8_t *values,uint64_t count,
	uint32_t seed,uint32_t scale)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t value;
	for (; index<count; index+=stride)
	{
		value = ((uint32_t)index * UINT32_C(747796405)) + seed;
		value = ((value >> ((value >> 28u) + 4u)) ^ value) *
			UINT32_C(277803737);
		value = (value >> 22u) ^ value;
		values[index] = scale != 0u ? (uint8_t)(124u + (value % 7u)) :
			(uint8_t)value;
	}
}

static int32_t PocSetRoutes(cudaStream_t stream,PocDeviceBuffers *buffers)
{
	std::vector<uint32_t> offsets(POC_GROUP_COUNT + 1u);
	std::vector<uint32_t> source_rows(POC_ACTIVE_GROUP_COUNT);
	std::vector<uint32_t> w13_prefix(POC_GROUP_COUNT + 1u);
	std::vector<uint32_t> w2_prefix(POC_GROUP_COUNT + 1u);
	uint32_t index,active,active_index = 0u;
	for (index=0u; index<=POC_GROUP_COUNT; index++)
	{
		active = active_index;
		offsets[index] = active;
		w13_prefix[index] = active *
			(POC_EXPERT_DIMENSION / POC_W13_TILE_N);
		w2_prefix[index] = active *
			(POC_HIDDEN_DIMENSION / POC_W2_TILE_N);
		if ( index < POC_ACTIVE_GROUP_COUNT )
			source_rows[index] = 0u;
		if ( (index < POC_GROUP_COUNT) &&
			(active_index < POC_ACTIVE_GROUP_COUNT) &&
			(index == PocActiveGroups[active_index]) )
			active_index++;
	}
	if ( cudaMemcpyAsync(buffers->source_rows,source_rows.data(),
			source_rows.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,
			stream) != cudaSuccess )
		return(-1);
	if ( cudaMemcpyAsync(buffers->row_offsets,offsets.data(),
			offsets.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) !=
		cudaSuccess )
		return(-2);
	if ( cudaMemcpyAsync(buffers->w13_prefix,w13_prefix.data(),
			w13_prefix.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) !=
		cudaSuccess )
		return(-3);
	if ( cudaMemcpyAsync(buffers->w2_prefix,w2_prefix.data(),
			w2_prefix.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) !=
		cudaSuccess )
		return(-4);
	return(0);
}

static int32_t PocAllocate(PocDeviceBuffers *buffers,cudaStream_t stream)
{
	uint64_t w13_payload_bytes = (uint64_t)POC_GROUP_COUNT *
		POC_EXPERT_DIMENSION * POC_HIDDEN_DIMENSION / 2u;
	uint64_t w13_scale_bytes = (uint64_t)POC_GROUP_COUNT *
		POC_EXPERT_DIMENSION * (POC_HIDDEN_DIMENSION / 32u);
	uint64_t w2_payload_bytes = (uint64_t)POC_GROUP_COUNT *
		POC_HIDDEN_DIMENSION * POC_EXPERT_DIMENSION / 2u;
	uint64_t w2_scale_bytes = (uint64_t)POC_GROUP_COUNT *
		POC_HIDDEN_DIMENSION * (POC_EXPERT_DIMENSION / 32u);
	uint64_t up_bytes = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_EXPERT_DIMENSION * sizeof(uint16_t);
	uint64_t output_bytes = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_HIDDEN_DIMENSION * sizeof(uint16_t);
	memset(buffers,0,sizeof(*buffers));
	POC_CUDA(cudaMalloc(&buffers->w1_payload,w13_payload_bytes));
	POC_CUDA(cudaMalloc(&buffers->w1_scale,w13_scale_bytes));
	POC_CUDA(cudaMalloc(&buffers->w3_payload,w13_payload_bytes));
	POC_CUDA(cudaMalloc(&buffers->w3_scale,w13_scale_bytes));
	POC_CUDA(cudaMalloc(&buffers->w2_payload,w2_payload_bytes));
	POC_CUDA(cudaMalloc(&buffers->w2_scale,w2_scale_bytes));
	POC_CUDA(cudaMalloc(&buffers->input,POC_HIDDEN_DIMENSION * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&buffers->control_up,up_bytes));
	POC_CUDA(cudaMalloc(&buffers->control_output,output_bytes));
	POC_CUDA(cudaMalloc(&buffers->candidate_up,up_bytes));
	POC_CUDA(cudaMalloc(&buffers->candidate_output,output_bytes));
	POC_CUDA(cudaMalloc(&buffers->source_rows,
		POC_ACTIVE_GROUP_COUNT * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->row_offsets,
		(POC_GROUP_COUNT + 1u) * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->w13_prefix,
		(POC_GROUP_COUNT + 1u) * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->w2_prefix,
		(POC_GROUP_COUNT + 1u) * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->flush,POC_FLUSH_BYTES));
	POC_CUDA(cudaMalloc(&buffers->flush_sink,sizeof(uint32_t)));
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w1_payload,
		w13_payload_bytes,UINT32_C(0x63b790e1),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w1_scale,
		w13_scale_bytes,UINT32_C(0x8f41c729),1u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w3_payload,
		w13_payload_bytes,UINT32_C(0xd2a5663b),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w3_scale,
		w13_scale_bytes,UINT32_C(0x17e3bd85),1u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w2_payload,
		w2_payload_bytes,UINT32_C(0x4ab981d7),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->w2_scale,
		w2_scale_bytes,UINT32_C(0xb6502fc1),1u);
	POC_CUDA(cudaMemsetAsync(buffers->flush,0xa5,POC_FLUSH_BYTES,stream));
	POC_CUDA(cudaMemsetAsync(buffers->flush_sink,0,sizeof(uint32_t),stream));
	PocFillInputKernel<<<(POC_HIDDEN_DIMENSION + 255u) / 256u,256u,0,
		stream>>>(buffers->input,POC_HIDDEN_DIMENSION);
	if ( PocSetRoutes(stream,buffers) < 0 )
		return(-2);
	POC_CUDA(cudaStreamSynchronize(stream));
	return(0);
}

static cudaError_t PocLaunchControl(cudaStream_t stream,
	const cudaDeviceProp *properties,const PocDeviceBuffers *buffers)
{
	SparkLmSm121B1ExpertW13Kernel<POC_W13_TILE_N>
		<<<properties->multiProcessorCount * POC_CONTROL_W13_BLOCKS_PER_SM,
		SPARK_LM_CTA_THREADS,POC_HIDDEN_DIMENSION * sizeof(float),stream>>>(
		buffers->w1_payload,buffers->w1_scale,buffers->w3_payload,
		buffers->w3_scale,buffers->input,buffers->source_rows,
		buffers->row_offsets,buffers->w13_prefix,buffers->control_up,1u,
		POC_GROUP_COUNT,POC_HIDDEN_DIMENSION,POC_EXPERT_DIMENSION,10.0f);
	if ( cudaGetLastError() != cudaSuccess )
		return(cudaGetLastError());
	SparkLmSm121B1ExpertW2Kernel<POC_W2_TILE_N>
		<<<properties->multiProcessorCount * POC_CONTROL_W2_BLOCKS_PER_SM,
		SPARK_LM_CTA_THREADS,POC_EXPERT_DIMENSION * sizeof(float),stream>>>(
		buffers->w2_payload,buffers->w2_scale,buffers->control_up,
		buffers->row_offsets,buffers->w2_prefix,buffers->control_output,
		POC_GROUP_COUNT,POC_EXPERT_DIMENSION,POC_HIDDEN_DIMENSION);
	return(cudaGetLastError());
}

static cudaError_t PocLaunchPersistent(cudaStream_t stream,
	const PocDeviceBuffers *buffers,uint32_t block_count)
{
	const uint8_t *w1_payload = buffers->w1_payload;
	const uint8_t *w1_scale = buffers->w1_scale;
	const uint8_t *w3_payload = buffers->w3_payload;
	const uint8_t *w3_scale = buffers->w3_scale;
	const uint8_t *w2_payload = buffers->w2_payload;
	const uint8_t *w2_scale = buffers->w2_scale;
	const void *input = buffers->input;
	const uint32_t *source_rows = buffers->source_rows;
	const uint32_t *row_offsets = buffers->row_offsets;
	const uint32_t *w13_prefix = buffers->w13_prefix;
	const uint32_t *w2_prefix = buffers->w2_prefix;
	void *up = buffers->candidate_up;
	void *output = buffers->candidate_output;
	uint32_t group_count = POC_GROUP_COUNT;
	void *arguments[] = {&w1_payload,&w1_scale,&w3_payload,&w3_scale,
		&w2_payload,&w2_scale,&input,&source_rows,&row_offsets,&w13_prefix,
		&w2_prefix,&up,&output,&group_count};
	return(cudaLaunchCooperativeKernel(
		(void *)PocPersistentExpertChainKernel<POC_W13_TILE_N,POC_W2_TILE_N>,
		dim3(block_count),dim3(SPARK_LM_CTA_THREADS),arguments,
		POC_HIDDEN_DIMENSION * sizeof(float),stream));
}

static void PocFlush(cudaStream_t stream,const PocDeviceBuffers *buffers)
{
	PocFlushKernel<<<4096u,256u,0,stream>>>((const uint4 *)buffers->flush,
		POC_FLUSH_BYTES / sizeof(uint4),buffers->flush_sink);
}

static int32_t PocCheckExact(cudaStream_t stream,const cudaDeviceProp *properties,
	const PocDeviceBuffers *buffers,uint32_t block_count,uint64_t *up_digest,
	uint64_t *output_digest)
{
	uint64_t up_elements = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_EXPERT_DIMENSION;
	uint64_t output_elements = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_HIDDEN_DIMENSION;
	std::vector<uint16_t> control_up(up_elements),candidate_up(up_elements);
	std::vector<uint16_t> control_output(output_elements);
	std::vector<uint16_t> candidate_output(output_elements);
	uint64_t index,digest;
	if ( PocLaunchControl(stream,properties,buffers) != cudaSuccess ||
		PocLaunchPersistent(stream,buffers,block_count) != cudaSuccess ||
		cudaStreamSynchronize(stream) != cudaSuccess )
		return(-1);
	if ( cudaMemcpy(control_up.data(),buffers->control_up,
			up_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(candidate_up.data(),buffers->candidate_up,
			up_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(control_output.data(),buffers->control_output,
			output_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(candidate_output.data(),buffers->candidate_output,
			output_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-2);
	if ( memcmp(control_up.data(),candidate_up.data(),
			up_elements * sizeof(uint16_t)) != 0 )
		return(-3);
	if ( memcmp(control_output.data(),candidate_output.data(),
			output_elements * sizeof(uint16_t)) != 0 )
		return(-4);
	digest = UINT64_C(1469598103934665603);
	for (index=0u; index<up_elements; index++)
		digest = (digest ^ control_up[index]) * UINT64_C(1099511628211);
	*up_digest = digest;
	digest = UINT64_C(1469598103934665603);
	for (index=0u; index<output_elements; index++)
		digest = (digest ^ control_output[index]) * UINT64_C(1099511628211);
	*output_digest = digest;
	return(0);
}

static int32_t PocTimeOne(cudaStream_t stream,cudaEvent_t start,
	cudaEvent_t stop,const cudaDeviceProp *properties,
	const PocDeviceBuffers *buffers,uint32_t candidate,uint32_t block_count,
	float *elapsed_ms)
{
	PocFlush(stream,buffers);
	if ( cudaEventRecord(start,stream) != cudaSuccess )
		return(-1);
	if ( (candidate == 0u ? PocLaunchControl(stream,properties,buffers) :
		PocLaunchPersistent(stream,buffers,block_count)) != cudaSuccess )
		return(-2);
	if ( cudaEventRecord(stop,stream) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess ||
		cudaEventElapsedTime(elapsed_ms,start,stop) != cudaSuccess )
		return(-3);
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

static int32_t PocMeasure(cudaStream_t stream,cudaEvent_t start,
	cudaEvent_t stop,const cudaDeviceProp *properties,
	const PocDeviceBuffers *buffers,uint32_t block_count,PocTiming *control,
	PocTiming *candidate)
{
	std::vector<float> control_times,candidate_times;
	float elapsed;
	uint32_t repetition,warmup;
	for (warmup=0u; warmup<POC_WARMUP_COUNT; warmup++)
	{
		if ( PocTimeOne(stream,start,stop,properties,buffers,warmup & 1u,
			block_count,&elapsed) < 0 )
			return(-1);
	}
	for (repetition=0u; repetition<POC_REPETITION_COUNT; repetition++)
	{
		if ( (repetition & 1u) == 0u )
		{
			if ( PocTimeOne(stream,start,stop,properties,buffers,0u,
					block_count,&elapsed) < 0 )
				return(-2);
			control_times.push_back(elapsed);
			if ( PocTimeOne(stream,start,stop,properties,buffers,1u,
					block_count,&elapsed) < 0 )
				return(-3);
			candidate_times.push_back(elapsed);
		}
		else
		{
			if ( PocTimeOne(stream,start,stop,properties,buffers,1u,
					block_count,&elapsed) < 0 )
				return(-4);
			candidate_times.push_back(elapsed);
			if ( PocTimeOne(stream,start,stop,properties,buffers,0u,
					block_count,&elapsed) < 0 )
				return(-5);
			control_times.push_back(elapsed);
		}
	}
	*control = PocSummarize(control_times);
	*candidate = PocSummarize(candidate_times);
	return(0);
}

int main(void)
{
	PocDeviceBuffers buffers;
	PocTiming control,candidate;
	cudaDeviceProp properties;
	cudaEvent_t start,stop;
	cudaStream_t stream;
	uint64_t up_digest,output_digest;
	uint32_t blocks_per_sm,block_count;
	int32_t maximum_blocks_per_sm;
	POC_CUDA(cudaGetDeviceProperties(&properties,0));
	POC_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	POC_CUDA(cudaEventCreate(&start));
	POC_CUDA(cudaEventCreate(&stop));
	if ( properties.cooperativeLaunch == 0 )
	{
		fprintf(stderr,"cooperative launch unsupported\n");
		return(2);
	}
	POC_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&maximum_blocks_per_sm,
		PocPersistentExpertChainKernel<POC_W13_TILE_N,POC_W2_TILE_N>,
		SPARK_LM_CTA_THREADS,POC_HIDDEN_DIMENSION * sizeof(float)));
	if ( PocAllocate(&buffers,stream) < 0 )
		return(3);
	printf("device=%s sm_count=%d cooperative=true max_resident_blocks_per_sm=%d repetitions=%u\n",
		properties.name,properties.multiProcessorCount,maximum_blocks_per_sm,
		POC_REPETITION_COUNT);
	for (blocks_per_sm=2u; blocks_per_sm<=4u; blocks_per_sm++)
	{
		if ( blocks_per_sm > (uint32_t)maximum_blocks_per_sm )
			continue;
		block_count = blocks_per_sm * properties.multiProcessorCount;
		if ( PocCheckExact(stream,&properties,&buffers,block_count,&up_digest,
				&output_digest) < 0 )
		{
			fprintf(stderr,"exactness failed blocks_per_sm=%u\n",blocks_per_sm);
			return(4);
		}
		if ( PocMeasure(stream,start,stop,&properties,&buffers,block_count,
				&control,&candidate) < 0 )
		{
			fprintf(stderr,"timing failed blocks_per_sm=%u\n",blocks_per_sm);
			return(5);
		}
		printf("blocks_per_sm=%u exact_up=true exact_output=true up_digest=%016llx output_digest=%016llx control_median_ms=%.6f candidate_median_ms=%.6f speedup=%.6f control_p10_ms=%.6f control_p90_ms=%.6f candidate_p10_ms=%.6f candidate_p90_ms=%.6f control_mean_ms=%.6f candidate_mean_ms=%.6f\n",
			blocks_per_sm,(unsigned long long)up_digest,
			(unsigned long long)output_digest,control.median_ms,
			candidate.median_ms,control.median_ms / candidate.median_ms,
			control.p10_ms,control.p90_ms,candidate.p10_ms,candidate.p90_ms,
			control.mean_ms,candidate.mean_ms);
	}
	return(0);
}
