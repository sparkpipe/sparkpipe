#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_l2_prefetch_shape.h"

#define SPARK_L2_PREFETCH_RANGES 8u
#define SPARK_L2_PREFETCH_THREADS 256u
#define SPARK_L2_PREFETCH_SEARCH 16u

typedef struct SparkL2PrefetchPlan
{
	const uint8_t *base[SPARK_L2_PREFETCH_RANGES];
	uint32_t bytes[SPARK_L2_PREFETCH_RANGES];
	uint32_t count,total,cap,misaligned;
}
SparkL2PrefetchPlan;

static __device__ uint32_t spark_l2_prefetch_sink;

static __global__ void __launch_bounds__(SPARK_L2_PREFETCH_THREADS) SparkL2PrefetchKernel(const __grid_constant__ SparkL2PrefetchPlan plan)
{
	const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x,stride = gridDim.x * blockDim.x;
	uint32_t range,index,count,folded = 0u;
	uint4 value;
	for (range=0u; range<plan.count; range++)
	{
		count = plan.bytes[range] / 16u;
		for (index=first; index<count; index+=stride)
		{
			value = __ldcg((const uint4 *)plan.base[range] + index);
			folded ^= value.x ^ value.y ^ value.z ^ value.w;
		}
	}
	if ( folded == 0x9e3779b9u )
		spark_l2_prefetch_sink = folded;
}

static inline void SparkL2PrefetchPlanBegin(SparkL2PrefetchPlan *plan,const SparkL2PrefetchShape *shape)
{
	memset(plan,0,sizeof(*plan));
	plan->cap = shape->bytes;
}

static inline void SparkL2PrefetchAdd(SparkL2PrefetchPlan *plan,const void *base,uint64_t bytes)
{
	uint64_t room;
	if ( base != 0 && ((uintptr_t)base % 16u) != 0u )
		plan->misaligned = 1u;
	if ( base == 0 || plan->misaligned != 0u || plan->count == SPARK_L2_PREFETCH_RANGES || plan->total >= plan->cap )
		return;
	room = plan->cap - plan->total;
	bytes = (bytes < room ? bytes : room) & ~(uint64_t)15u;
	if ( bytes == 0u )
		return;
	plan->base[plan->count] = (const uint8_t *)base;
	plan->bytes[plan->count] = (uint32_t)bytes;
	plan->count++;
	plan->total += (uint32_t)bytes;
}

static inline cudaGraphNode_t SparkL2PrefetchForkNode(cudaGraphNode_t node)
{
	CUgraphNodeType type;
	CUgraphNode previous;
	size_t count;
	uint32_t step;
	for (step=0u; step<SPARK_L2_PREFETCH_SEARCH; step++)
	{
		count = 0u;
		if ( cuGraphNodeGetType((CUgraphNode)node,&type) != CUDA_SUCCESS || cuGraphNodeGetDependencies((CUgraphNode)node,0,0,&count) != CUDA_SUCCESS || count != 1u ||
		     cuGraphNodeGetDependencies((CUgraphNode)node,&previous,0,&count) != CUDA_SUCCESS )
			return(0);
		if ( type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP )
			return((cudaGraphNode_t)previous);
		node = (cudaGraphNode_t)previous;
	}
	return(0);
}

static inline cudaError_t SparkL2PrefetchPlaceAfterWait(cudaStream_t stream,const SparkL2PrefetchPlan *plan,const SparkL2PrefetchShape *shape,uint32_t *placed)
{
	cudaStreamCaptureStatus capture;
	cudaGraph_t graph;
	const cudaGraphNode_t *dependencies;
	cudaGraphNode_t fork,prefetch;
	cudaKernelNodeParams params;
	SparkL2PrefetchPlan copy;
	size_t count;
	void *arguments[1];
	if ( placed == 0 || plan == 0 || SparkL2PrefetchShapeValid(shape) == 0u || plan->misaligned != 0u )
		return(cudaErrorInvalidValue);
	*placed = 0u;
	if ( plan->count == 0u )
		return(cudaSuccess);
	if ( cudaStreamGetCaptureInfo(stream,&capture,0,&graph,&dependencies,0,&count) != cudaSuccess )
		return(cudaErrorUnknown);
	if ( capture != cudaStreamCaptureStatusActive || count != 1u )
		return(cudaSuccess);
	fork = SparkL2PrefetchForkNode(dependencies[0]);
	if ( fork == 0 )
		return(cudaSuccess);
	copy = *plan;
	memset(&params,0,sizeof(params));
	arguments[0] = &copy;
	params.func = (void *)SparkL2PrefetchKernel;
	params.gridDim = dim3(shape->blocks);
	params.blockDim = dim3(SPARK_L2_PREFETCH_THREADS);
	params.kernelParams = arguments;
	if ( cudaGraphAddKernelNode(&prefetch,graph,&fork,1u,&params) != cudaSuccess ||
	     cudaStreamUpdateCaptureDependencies(stream,&prefetch,0,1u,cudaStreamAddCaptureDependencies) != cudaSuccess )
		return(cudaErrorUnknown);
	*placed = 1u;
	return(cudaSuccess);
}
