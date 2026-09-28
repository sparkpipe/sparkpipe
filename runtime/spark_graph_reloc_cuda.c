#include "sparkpipe/spark_graph_reloc.h"
#include "sparkpipe/spark_error_site.h"
#include <stddef.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

_Static_assert(sizeof(void *) == 8u,"relocation words are 64-bit device addresses");

static SparkStatus SparkGraphRelocCudaRefuse(SparkGraphRelocFault *fault,SparkStatus status,uint32_t reason,uint32_t node,uint64_t word,uint32_t detail)
{
	if ( fault != 0 )
	{
		fault->reason = reason;
		fault->node = node;
		fault->byte_offset = 0u;
		fault->word = word;
		fault->detail = detail;
	}
	SPARK_FAIL(status);
}

uint32_t SparkGraphRelocCudaProbe(void *context,uint64_t word)
{
	struct cudaPointerAttributes attributes;
	(void)context;
	memset(&attributes,0,sizeof(attributes));
	if ( cudaPointerGetAttributes(&attributes,(const void *)(uintptr_t)word) != cudaSuccess )
	{
		(void)cudaGetLastError();
		return(0u);
	}
	return(attributes.type != cudaMemoryTypeUnregistered ? 1u : 0u);
}

static SparkStatus SparkGraphRelocCudaLayout(const void *symbol,SparkGraphRelocParam *params,uint32_t *count,uint32_t *bytes)
{
	cudaFunction_t function = 0;
	CUresult result = CUDA_SUCCESS;
	size_t offset = 0u,size = 0u;
	uint32_t index;
	*count = 0u;
	*bytes = 0u;
	if ( cudaGetFuncBySymbol(&function,symbol) != cudaSuccess || function == 0 )
	{
		(void)cudaGetLastError();
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	for (index=0u; index<SPARK_GRAPH_RELOC_KERNEL_PARAMS_MAX; index++)
	{
		result = cuFuncGetParamInfo((CUfunction)function,index,&offset,&size);
		if ( result != CUDA_SUCCESS )
			break;
		if ( offset > UINT32_MAX || size > UINT32_MAX - offset )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		params[index].offset = (uint32_t)offset;
		params[index].bytes = (uint32_t)size;
		if ( (uint32_t)(offset + size) > *bytes )
			*bytes = (uint32_t)(offset + size);
	}
	if ( result != CUDA_ERROR_INVALID_VALUE )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*count = index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocCudaKernel(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,cudaGraphNode_t handle,SparkGraphRelocFault *fault)
{
	SparkGraphRelocParam params[SPARK_GRAPH_RELOC_KERNEL_PARAMS_MAX];
	SparkGraphRelocNodeInput input;
	struct cudaKernelNodeParams kernel;
	uint32_t count = 0u,bytes = 0u,index;
	uint8_t *blob;
	memset(&kernel,0,sizeof(kernel));
	if ( cudaGraphKernelNodeGetParams(handle,&kernel) != cudaSuccess || kernel.extra != 0 || SparkGraphRelocCudaLayout(kernel.func,params,&count,&bytes) != SPARK_STATUS_OK || (count != 0u && kernel.kernelParams == 0) )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_NODE_QUERY,image->node_count,(uint64_t)(uintptr_t)kernel.func,cudaGraphNodeTypeKernel));
	blob = SparkGraphRelocImageTail(image,bytes);
	if ( blob == 0 )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,image->node_count,bytes,cudaGraphNodeTypeKernel));
	for (index=0u; index<count; index++)
		memcpy(blob + params[index].offset,kernel.kernelParams[index],params[index].bytes);
	memset(&input,0,sizeof(input));
	input.handle = handle;
	input.function = kernel.func;
	input.params = params;
	input.kind = SPARK_GRAPH_RELOC_NODE_KERNEL;
	input.param_count = count;
	input.blob_bytes = bytes;
	return(SparkGraphRelocImageCommit(image,registry,&input,fault));
}

static SparkStatus SparkGraphRelocCudaCommit(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,cudaGraphNode_t handle,uint32_t kind,const SparkGraphRelocParam *params,uint32_t param_count,uint32_t bytes,SparkGraphRelocFault *fault)
{
	SparkGraphRelocNodeInput input;
	memset(&input,0,sizeof(input));
	input.handle = handle;
	input.params = params;
	input.kind = kind;
	input.param_count = param_count;
	input.blob_bytes = bytes;
	return(SparkGraphRelocImageCommit(image,registry,&input,fault));
}

static cudaError_t SparkGraphRelocCudaRead(cudaGraphNode_t handle,uint32_t kind,uint8_t *blob)
{
	switch ( kind )
	{
	case SPARK_GRAPH_RELOC_NODE_COPY: return(cudaGraphMemcpyNodeGetParams(handle,(struct cudaMemcpy3DParms *)blob));
	case SPARK_GRAPH_RELOC_NODE_FILL: return(cudaGraphMemsetNodeGetParams(handle,(struct cudaMemsetParams *)blob));
	case SPARK_GRAPH_RELOC_NODE_HOST: return(cudaGraphHostNodeGetParams(handle,(struct cudaHostNodeParams *)blob));
	case SPARK_GRAPH_RELOC_NODE_EVENT_RECORD: return(cudaGraphEventRecordNodeGetEvent(handle,(cudaEvent_t *)blob));
	case SPARK_GRAPH_RELOC_NODE_EVENT_WAIT: return(cudaGraphEventWaitNodeGetEvent(handle,(cudaEvent_t *)blob));
	default: return(cudaSuccess);
	}
}

static const SparkGraphRelocParam SparkGraphRelocCudaCopyWords[2] =
{
	{(uint32_t)(offsetof(struct cudaMemcpy3DParms,srcPtr) + offsetof(struct cudaPitchedPtr,ptr)),8u},
	{(uint32_t)(offsetof(struct cudaMemcpy3DParms,dstPtr) + offsetof(struct cudaPitchedPtr,ptr)),8u}
};
static const SparkGraphRelocParam SparkGraphRelocCudaFillWords[1] = {{(uint32_t)offsetof(struct cudaMemsetParams,dst),8u}};
static const SparkGraphRelocParam SparkGraphRelocCudaHostWords[1] = {{(uint32_t)offsetof(struct cudaHostNodeParams,userData),8u}};

static SparkStatus SparkGraphRelocCudaFixed(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,cudaGraphNode_t handle,uint32_t kind,SparkGraphRelocFault *fault)
{
	static const uint32_t sizes[] = {0u,0u,sizeof(struct cudaMemcpy3DParms),sizeof(struct cudaMemsetParams),sizeof(struct cudaHostNodeParams),sizeof(cudaEvent_t),sizeof(cudaEvent_t)};
	static const SparkGraphRelocParam *const words[] = {0,0,SparkGraphRelocCudaCopyWords,SparkGraphRelocCudaFillWords,SparkGraphRelocCudaHostWords,0,0};
	static const uint32_t counts[] = {0u,0u,2u,1u,1u,0u,0u};
	uint8_t *blob = SparkGraphRelocImageTail(image,sizes[kind]);
	if ( blob == 0 )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,image->node_count,sizes[kind],kind));
	if ( SparkGraphRelocCudaRead(handle,kind,blob) != cudaSuccess )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_NODE_QUERY,image->node_count,0u,kind));
	if ( kind == SPARK_GRAPH_RELOC_NODE_COPY && (((struct cudaMemcpy3DParms *)blob)->srcArray != 0 || ((struct cudaMemcpy3DParms *)blob)->dstArray != 0) )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_UNSUPPORTED,SPARK_GRAPH_RELOC_REASON_NODE_KIND,image->node_count,0u,kind));
	return(SparkGraphRelocCudaCommit(image,registry,handle,kind,words[kind],counts[kind],sizes[kind],fault));
}

static SparkStatus SparkGraphRelocCudaNode(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,cudaGraphNode_t handle,SparkGraphRelocFault *fault)
{
	enum cudaGraphNodeType type;
	if ( cudaGraphNodeGetType(handle,&type) != cudaSuccess )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_NODE_QUERY,image->node_count,0u,0u));
	switch ( type )
	{
	case cudaGraphNodeTypeKernel: return(SparkGraphRelocCudaKernel(image,registry,handle,fault));
	case cudaGraphNodeTypeMemcpy: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_COPY,fault));
	case cudaGraphNodeTypeMemset: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_FILL,fault));
	case cudaGraphNodeTypeHost: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_HOST,fault));
	case cudaGraphNodeTypeEventRecord: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_EVENT_RECORD,fault));
	case cudaGraphNodeTypeWaitEvent: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_EVENT_WAIT,fault));
	case cudaGraphNodeTypeEmpty: return(SparkGraphRelocCudaFixed(image,registry,handle,SPARK_GRAPH_RELOC_NODE_EMPTY,fault));
	default: return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_UNSUPPORTED,SPARK_GRAPH_RELOC_REASON_NODE_KIND,image->node_count,0u,(uint32_t)type));
	}
}

SparkStatus SparkGraphRelocCaptureCuda(void *graph,const SparkGraphRelocRegistry *registry,uint32_t slot,SparkGraphRelocImage *image,SparkGraphRelocFault *fault)
{
	SparkStatus status = SPARK_STATUS_OK;
	size_t count = 0u;
	uint32_t index;
	if ( graph == 0 || registry == 0 || image == 0 || registry->sealed == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkGraphRelocImageReset(image,slot);
	if ( cudaGraphGetNodes((cudaGraph_t)graph,0,&count) != cudaSuccess )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_NODE_QUERY,SPARK_GRAPH_RELOC_NONE,0u,0u));
	if ( count > image->node_capacity )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,SPARK_GRAPH_RELOC_NONE,count,0u));
	if ( count != 0u && cudaGraphGetNodes((cudaGraph_t)graph,(cudaGraphNode_t *)image->handles,&count) != cudaSuccess )
		return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_NODE_QUERY,SPARK_GRAPH_RELOC_NONE,count,0u));
	for (index=0u; status==SPARK_STATUS_OK && index<(uint32_t)count; index++)
		status = SparkGraphRelocCudaNode(image,registry,(cudaGraphNode_t)image->handles[index],fault);
	return(status);
}

static SparkStatus SparkGraphRelocCudaSetKernel(cudaGraphExec_t exec,const SparkGraphRelocImage *image,uint32_t index,const uint8_t *bytes)
{
	void *values[SPARK_GRAPH_RELOC_KERNEL_PARAMS_MAX];
	const SparkGraphRelocNode *node = &image->nodes[index];
	struct cudaKernelNodeParams kernel;
	uint32_t param;
	memset(&kernel,0,sizeof(kernel));
	if ( node->param_count > SPARK_GRAPH_RELOC_KERNEL_PARAMS_MAX || cudaGraphKernelNodeGetParams((cudaGraphNode_t)image->handles[index],&kernel) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for (param=0u; param<node->param_count; param++)
		values[param] = (void *)(uintptr_t)(bytes + image->params[node->param_first + param].offset);
	kernel.kernelParams = values;
	kernel.extra = 0;
	if ( cudaGraphExecKernelNodeSetParams(exec,(cudaGraphNode_t)image->handles[index],&kernel) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocCudaSet(cudaGraphExec_t exec,const SparkGraphRelocImage *image,uint32_t index,const uint8_t *bytes)
{
	cudaGraphNode_t handle = (cudaGraphNode_t)image->handles[index];
	cudaError_t error = cudaErrorInvalidValue;
	switch ( image->nodes[index].kind )
	{
	case SPARK_GRAPH_RELOC_NODE_KERNEL: return(SparkGraphRelocCudaSetKernel(exec,image,index,bytes));
	case SPARK_GRAPH_RELOC_NODE_COPY: error = cudaGraphExecMemcpyNodeSetParams(exec,handle,(const struct cudaMemcpy3DParms *)bytes); break;
	case SPARK_GRAPH_RELOC_NODE_FILL: error = cudaGraphExecMemsetNodeSetParams(exec,handle,(const struct cudaMemsetParams *)bytes); break;
	case SPARK_GRAPH_RELOC_NODE_HOST: error = cudaGraphExecHostNodeSetParams(exec,handle,(const struct cudaHostNodeParams *)bytes); break;
	default: break;
	}
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGraphRelocApplyCuda(void *exec,const SparkGraphRelocImage *image,const uint8_t *blob,uint32_t force,uint32_t *applied,SparkGraphRelocFault *fault)
{
	const SparkGraphRelocNode *node;
	uint32_t index;
	if ( exec == 0 || image == 0 || blob == 0 || applied == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*applied = 0u;
	for (index=0u; index<image->node_count; index++)
	{
		node = &image->nodes[index];
		if ( node->site_count == 0u || (force == 0u && memcmp(blob + node->blob_offset,image->blob + node->blob_offset,node->blob_bytes) == 0) )
			continue;
		if ( SparkGraphRelocCudaSet((cudaGraphExec_t)exec,image,index,blob + node->blob_offset) != SPARK_STATUS_OK )
		{
			(void)cudaGetLastError();
			return(SparkGraphRelocCudaRefuse(fault,SPARK_STATUS_IO_ERROR,SPARK_GRAPH_RELOC_REASON_APPLY,index,0u,node->kind));
		}
		(*applied)++;
	}
	return(SPARK_STATUS_OK);
}
