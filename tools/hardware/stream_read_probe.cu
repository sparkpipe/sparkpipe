#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { fprintf(stderr,"PROBE-FAIL line=%d cuda=%s\n",__LINE__,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<uint32_t DEPTH>
__global__ __launch_bounds__(256) void ProbeReadKernel(const uint4 *data, uint64_t count, float *sink)
{
	uint64_t index, stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t u, acc = 0u;
	uint4 value[DEPTH];
	for ( index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride * DEPTH )
	{
		#pragma unroll
		for ( u = 0u; u < DEPTH; u++ )
			value[u] = index + u * stride < count ? __ldcs(data + index + u * stride) : make_uint4(0u,0u,0u,0u);
		#pragma unroll
		for ( u = 0u; u < DEPTH; u++ )
			acc ^= value[u].x ^ value[u].y ^ value[u].z ^ value[u].w;
	}
	if ( acc == 0x12345678u )
		sink[0] = (float)acc;
}

__global__ __launch_bounds__(256) void ProbeCopyKernel(const uint4 *source, uint4 *target, uint64_t count)
{
	uint64_t index, stride = (uint64_t)gridDim.x * blockDim.x;
	for ( index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride )
		__stcs(target + index,__ldcs(source + index));
}

__global__ void ProbeEmptyKernel(void)
{
}

typedef struct ProbeResult
{
	double median_us, min_us;
}
ProbeResult;

static ProbeResult ProbeGraph(cudaStream_t stream, uint32_t launches, uint32_t repeats, void (*record)(cudaStream_t, uint32_t, void *), void *context)
{
	cudaGraph_t graph;
	cudaGraphExec_t exec;
	cudaEvent_t start, stop;
	std::vector<double> samples;
	ProbeResult result;
	uint32_t index;
	float ms;
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	PROBE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
	for ( index = 0u; index < launches; index++ )
		record(stream,index,context);
	PROBE_CUDA(cudaStreamEndCapture(stream,&graph));
	PROBE_CUDA(cudaGraphInstantiate(&exec,graph,0));
	PROBE_CUDA(cudaGraphLaunch(exec,stream));
	PROBE_CUDA(cudaStreamSynchronize(stream));
	for ( index = 0u; index < repeats; index++ )
	{
		PROBE_CUDA(cudaEventRecord(start,stream));
		PROBE_CUDA(cudaGraphLaunch(exec,stream));
		PROBE_CUDA(cudaEventRecord(stop,stream));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&ms,start,stop));
		samples.push_back((double)ms * 1000.0 / launches);
	}
	std::sort(samples.begin(),samples.end());
	result.median_us = samples[samples.size() / 2u];
	result.min_us = samples[0];
	PROBE_CUDA(cudaGraphExecDestroy(exec));
	PROBE_CUDA(cudaGraphDestroy(graph));
	return(result);
}

typedef struct ProbeReadContext
{
	const uint8_t *base;
	uint64_t arena, bytes;
	float *sink;
	uint32_t blocks, depth;
}
ProbeReadContext;

static void ProbeRecordRead(cudaStream_t stream, uint32_t launch, void *opaque)
{
	ProbeReadContext *context = (ProbeReadContext *)opaque;
	uint64_t slots = context->arena / context->bytes, offset = (launch % slots) * context->bytes;
	const uint4 *data = (const uint4 *)(context->base + offset);
	if ( context->depth == 4u )
		ProbeReadKernel<4u><<<context->blocks,256,0,stream>>>(data,context->bytes / 16u,context->sink);
	else if ( context->depth == 8u )
		ProbeReadKernel<8u><<<context->blocks,256,0,stream>>>(data,context->bytes / 16u,context->sink);
	else
		ProbeReadKernel<1u><<<context->blocks,256,0,stream>>>(data,context->bytes / 16u,context->sink);
}

static void ProbeRecordCopy(cudaStream_t stream, uint32_t launch, void *opaque)
{
	ProbeReadContext *context = (ProbeReadContext *)opaque;
	uint64_t half = context->arena / 2u, slots = half / context->bytes, offset = (launch % slots) * context->bytes;
	ProbeCopyKernel<<<context->blocks,256,0,stream>>>((const uint4 *)(context->base + offset),(uint4 *)(context->base + half + offset),context->bytes / 16u);
}

static void ProbeRecordEmpty(cudaStream_t stream, uint32_t launch, void *opaque)
{
	(void)launch;
	(void)opaque;
	ProbeEmptyKernel<<<1,32,0,stream>>>();
}

int main(int argc, char **argv)
{
	static const uint64_t sizes[] = {262144ull,524288ull,1048576ull,2097152ull,4194304ull,8388608ull,16777216ull,33554432ull,67108864ull,268435456ull,1073741824ull};
	static const uint32_t depths[] = {1u,4u,8u};
	cudaDeviceProp properties;
	cudaStream_t stream;
	ProbeReadContext context;
	ProbeResult result, best;
	uint64_t arena = 4ull << 30u;
	uint32_t size_index, depth_index, blocks_per_sm, best_blocks, best_depth, launches;
	void *base;
	float *sink;
	(void)argc;
	(void)argv;
	PROBE_CUDA(cudaGetDeviceProperties(&properties,0));
	PROBE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	PROBE_CUDA(cudaMalloc(&base,arena));
	PROBE_CUDA(cudaMemset(base,1,arena));
	PROBE_CUDA(cudaMalloc((void **)&sink,sizeof(*sink)));
	printf("PROBE-DEVICE name=%s sm=%d.%d multiprocessors=%d l2_bytes=%d\n",properties.name,properties.major,properties.minor,properties.multiProcessorCount,properties.l2CacheSize);
	memset(&context,0,sizeof(context));
	context.base = (const uint8_t *)base;
	context.arena = arena;
	context.sink = sink;
	result = ProbeGraph(stream,2000u,9u,ProbeRecordEmpty,&context);
	printf("PROBE-EMPTY per_launch_us median=%.3f min=%.3f\n",result.median_us,result.min_us);
	for ( size_index = 0u; size_index < sizeof(sizes) / sizeof(sizes[0]); size_index++ )
	{
		best.median_us = 1e30;
		best.min_us = 1e30;
		best_blocks = best_depth = 0u;
		launches = sizes[size_index] >= 268435456ull ? 20u : 400u;
		context.bytes = sizes[size_index];
		for ( depth_index = 0u; depth_index < sizeof(depths) / sizeof(depths[0]); depth_index++ )
			for ( blocks_per_sm = 1u; blocks_per_sm <= 8u; blocks_per_sm *= 2u )
			{
				context.depth = depths[depth_index];
				context.blocks = blocks_per_sm * properties.multiProcessorCount;
				result = ProbeGraph(stream,launches,7u,ProbeRecordRead,&context);
				if ( result.median_us < best.median_us )
				{
					best = result;
					best_blocks = context.blocks;
					best_depth = context.depth;
				}
			}
		printf("PROBE-READ bytes=%llu best_us=%.2f min_us=%.2f gbps=%.1f blocks=%u depth=%u\n",(unsigned long long)sizes[size_index],best.median_us,best.min_us,(double)sizes[size_index] / (best.median_us * 1e3),best_blocks,best_depth);
	}
	for ( size_index = 6u; size_index < sizeof(sizes) / sizeof(sizes[0]); size_index++ )
	{
		best.median_us = 1e30;
		context.bytes = sizes[size_index];
		for ( blocks_per_sm = 1u; blocks_per_sm <= 16u; blocks_per_sm *= 2u )
		{
			context.blocks = blocks_per_sm * properties.multiProcessorCount;
			result = ProbeGraph(stream,sizes[size_index] >= 268435456ull ? 10u : 100u,7u,ProbeRecordCopy,&context);
			if ( result.median_us < best.median_us )
				best = result;
		}
		printf("PROBE-COPY bytes=%llu best_us=%.2f read_plus_write_gbps=%.1f\n",(unsigned long long)sizes[size_index],best.median_us,2.0 * (double)sizes[size_index] / (best.median_us * 1e3));
	}
	return(0);
}
