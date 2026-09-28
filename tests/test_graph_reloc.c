#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "sparkpipe/spark_graph_reloc.h"

#define DEVICE_BASE UINT64_C(0x7f0000000000)
#define STAGING_SLOT0 UINT64_C(0x7f0000100000)
#define STAGING_SLOT1 UINT64_C(0x7f0000200000)
#define WEIGHT_BASE UINT64_C(0x7f0010000000)
#define SHADOW_DEVICE UINT64_C(0x7f0800000000)
#define ROGUE UINT64_C(0x7f0000300000)
#define PROBED UINT64_C(0x0000500000000000)
#define FAKE_NODES 8u

enum { ID_HIDDEN = 1u, ID_STAGING = 2u, ID_WEIGHT = 3u };

typedef struct FakeKernel
{
	uint32_t count;
	size_t offsets[4];
	size_t sizes[4];
} FakeKernel;

typedef struct FakeArgs
{
	uint64_t staging;
	uint32_t a;
	uint32_t b;
} FakeArgs;

typedef struct FakeNode
{
	enum cudaGraphNodeType type;
	void *values[4];
	uint64_t hidden;
	uint32_t rows;
	FakeArgs args;
	uint64_t seed;
	void **extra;
	struct cudaMemcpy3DParms copy;
	struct cudaMemsetParams fill;
	struct cudaHostNodeParams host;
	cudaEvent_t event;
	uint8_t applied[64];
	uint32_t applied_count;
} FakeNode;

typedef struct FakeGraph
{
	FakeNode nodes[FAKE_NODES];
	uint32_t count;
} FakeGraph;

static const FakeKernel KERNEL = {4u,{0u,8u,16u,32u},{8u,4u,16u,8u}};
static uint32_t HOST_COOKIE;
static FakeGraph GRAPHS[2];

static void FakeHost(void *user_data)
{
	(void)user_data;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph,cudaGraphNode_t *nodes,size_t *count)
{
	FakeGraph *fake = (FakeGraph *)graph;
	uint32_t index;
	if ( nodes == 0 )
	{
		*count = fake->count;
		return(cudaSuccess);
	}
	assert(*count == fake->count);
	for (index=0u; index<fake->count; index++)
		nodes[index] = &fake->nodes[index];
	return(cudaSuccess);
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,enum cudaGraphNodeType *type)
{
	*type = ((FakeNode *)node)->type;
	return(cudaSuccess);
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,struct cudaKernelNodeParams *params)
{
	FakeNode *fake = (FakeNode *)node;
	memset(params,0,sizeof(*params));
	fake->values[0] = &fake->hidden;
	fake->values[1] = &fake->rows;
	fake->values[2] = &fake->args;
	fake->values[3] = &fake->seed;
	params->func = (void *)&KERNEL;
	params->kernelParams = fake->values;
	params->extra = fake->extra;
	return(cudaSuccess);
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,struct cudaMemcpy3DParms *params)
{
	*params = ((FakeNode *)node)->copy;
	return(cudaSuccess);
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,struct cudaMemsetParams *params)
{
	*params = ((FakeNode *)node)->fill;
	return(cudaSuccess);
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node,struct cudaHostNodeParams *params)
{
	*params = ((FakeNode *)node)->host;
	return(cudaSuccess);
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node,cudaEvent_t *event)
{
	*event = ((FakeNode *)node)->event;
	return(cudaSuccess);
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node,cudaEvent_t *event)
{
	*event = ((FakeNode *)node)->event;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaKernelNodeParams *params)
{
	FakeNode *fake = (FakeNode *)node;
	uint32_t index;
	assert(exec != 0 && params->func == (void *)&KERNEL && params->extra == 0);
	for (index=0u; index<KERNEL.count; index++)
		memcpy(fake->applied + KERNEL.offsets[index],params->kernelParams[index],KERNEL.sizes[index]);
	fake->applied_count++;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaMemcpy3DParms *params)
{
	FakeNode *fake = (FakeNode *)node;
	assert(exec != 0);
	fake->copy = *params;
	fake->applied_count++;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaMemsetParams *params)
{
	FakeNode *fake = (FakeNode *)node;
	assert(exec != 0);
	fake->fill = *params;
	fake->applied_count++;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaHostNodeParams *params)
{
	FakeNode *fake = (FakeNode *)node;
	assert(exec != 0);
	fake->host = *params;
	fake->applied_count++;
	return(cudaSuccess);
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t *function,const void *symbol)
{
	*function = (cudaFunction_t)(uintptr_t)symbol;
	return(cudaSuccess);
}

CUresult cuFuncGetParamInfo(CUfunction function,size_t index,size_t *offset,size_t *size)
{
	const FakeKernel *kernel = (const FakeKernel *)(uintptr_t)function;
	if ( index >= kernel->count )
		return(CUDA_ERROR_INVALID_VALUE);
	*offset = kernel->offsets[index];
	*size = kernel->sizes[index];
	return(CUDA_SUCCESS);
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,const void *pointer)
{
	attributes->type = (uint64_t)(uintptr_t)pointer == PROBED ? cudaMemoryTypeDevice : cudaMemoryTypeUnregistered;
	return(cudaSuccess);
}

static void BuildGraph(FakeGraph *graph,uint64_t hidden,uint64_t staging,uint64_t seed,cudaEvent_t event)
{
	memset(graph,0,sizeof(*graph));
	graph->count = 5u;
	graph->nodes[0].type = cudaGraphNodeTypeKernel;
	graph->nodes[0].hidden = hidden + 64u;
	graph->nodes[0].rows = 8u;
	graph->nodes[0].args.staging = staging + 16u;
	graph->nodes[0].args.a = 3u;
	graph->nodes[0].args.b = 4u;
	graph->nodes[0].seed = seed;
	graph->nodes[1].type = cudaGraphNodeTypeMemcpy;
	graph->nodes[1].copy.srcPtr.ptr = (void *)(uintptr_t)(WEIGHT_BASE + 128u);
	graph->nodes[1].copy.dstPtr.ptr = (void *)(uintptr_t)staging;
	graph->nodes[1].copy.extent.width = 32u;
	graph->nodes[1].copy.kind = cudaMemcpyDeviceToHost;
	graph->nodes[2].type = cudaGraphNodeTypeMemset;
	graph->nodes[2].fill.dst = (void *)(uintptr_t)hidden;
	graph->nodes[2].fill.width = 64u;
	graph->nodes[2].fill.elementSize = 1u;
	graph->nodes[3].type = cudaGraphNodeTypeHost;
	graph->nodes[3].host.fn = FakeHost;
	graph->nodes[3].host.userData = &HOST_COOKIE;
	graph->nodes[4].type = cudaGraphNodeTypeEventRecord;
	graph->nodes[4].event = event;
}

static void Register(SparkGraphRelocRegistry *registry,uint64_t hidden)
{
	SparkGraphRelocRegistryReset(registry);
	assert(SparkGraphRelocRegister(registry,ID_WEIGHT,(void *)(uintptr_t)WEIGHT_BASE,UINT64_C(1) << 20,3u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(registry,ID_STAGING,(void *)(uintptr_t)STAGING_SLOT1,1024u,2u,1u) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(registry,ID_HIDDEN,(void *)(uintptr_t)hidden,4096u,1u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(registry,ID_STAGING,(void *)(uintptr_t)STAGING_SLOT0,1024u,2u,0u) == SPARK_STATUS_OK);
	assert(SparkGraphRelocSeal(registry,0) == SPARK_STATUS_OK);
}

static SparkGraphRelocWorkspace *Workspace(uint32_t nodes,uint64_t margin)
{
	SparkGraphRelocCapacity capacity = {4096u,16u,nodes,64u,64u};
	SparkGraphRelocWorkspace *workspace = 0;
	assert(SparkGraphRelocWorkspaceCreate(&capacity,margin,&workspace) == SPARK_STATUS_OK);
	return(workspace);
}

static void TestRegistry(void)
{
	SparkGraphRelocRegion regions[4];
	SparkGraphRelocKey keys[4];
	SparkGraphRelocRegistry registry;
	SparkGraphRelocFault fault;
	SparkGraphRelocMatch match;
	SparkGraphRelocRegistryInit(&registry,regions,keys,4u,UINT64_C(1) << 30);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)0x100u,64u,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)DEVICE_BASE,0u,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)DEVICE_BASE,256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,2u,(void *)(uintptr_t)(DEVICE_BASE + 255u),64u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocSeal(&registry,&fault) == SPARK_STATUS_VALIDATION_FAILED && fault.reason == SPARK_GRAPH_RELOC_REASON_REGION);
	SparkGraphRelocRegistryReset(&registry);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)DEVICE_BASE,256u,0u,0u) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)(DEVICE_BASE + 4096u),256u,0u,0u) == SPARK_STATUS_OK);
	assert(SparkGraphRelocSeal(&registry,&fault) == SPARK_STATUS_DUPLICATE && fault.reason == SPARK_GRAPH_RELOC_REASON_REGION);
	SparkGraphRelocRegistryReset(&registry);
	assert(SparkGraphRelocRegister(&registry,2u,(void *)(uintptr_t)(DEVICE_BASE + 256u),256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,1u,(void *)(uintptr_t)DEVICE_BASE,256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,3u,(void *)(uintptr_t)(DEVICE_BASE + 8192u),256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,4u,(void *)(uintptr_t)(DEVICE_BASE + 16384u),256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&registry,5u,(void *)(uintptr_t)(DEVICE_BASE + 32768u),256u,0u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkGraphRelocSeal(&registry,&fault) == SPARK_STATUS_OK);
	SparkGraphRelocClassify(&registry,DEVICE_BASE + 17u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_REGION && registry.regions[match.region].id == 1u && match.offset == 17u);
	SparkGraphRelocClassify(&registry,DEVICE_BASE + 8192u + 256u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_REGION && registry.regions[match.region].id == 3u && match.offset == 256u);
	SparkGraphRelocClassify(&registry,DEVICE_BASE + 256u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_AMBIGUOUS);
	SparkGraphRelocClassify(&registry,DEVICE_BASE + 12288u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_UNKNOWN);
	SparkGraphRelocClassify(&registry,0u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_CONSTANT);
	SparkGraphRelocClassify(&registry,PROBED,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_CONSTANT);
	registry.probe = SparkGraphRelocCudaProbe;
	SparkGraphRelocClassify(&registry,PROBED,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_UNKNOWN);
	SparkGraphRelocClassify(&registry,PROBED + 8u,&match);
	assert(match.verdict == SPARK_GRAPH_RELOC_WORD_CONSTANT);
	assert(SparkGraphRelocFind(&registry,4u,SPARK_GRAPH_RELOC_SLOT_SHARED) != SPARK_GRAPH_RELOC_NONE);
	assert(registry.regions[SparkGraphRelocFind(&registry,4u,SPARK_GRAPH_RELOC_SLOT_SHARED)].base == DEVICE_BASE + 16384u);
	assert(SparkGraphRelocFind(&registry,4u,0u) == SPARK_GRAPH_RELOC_NONE);
	assert(SparkGraphRelocFind(&registry,9u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_GRAPH_RELOC_NONE);
}

static void CheckSites(const SparkGraphRelocImage *image,uint32_t slot)
{
	static const uint32_t nodes[5] = {0u,0u,1u,1u,2u};
	static const uint32_t ids[5] = {ID_HIDDEN,ID_STAGING,ID_WEIGHT,ID_STAGING,ID_HIDDEN};
	static const uint64_t offsets[5] = {64u,16u,128u,0u,0u};
	uint32_t index;
	assert(image->node_count == 5u && image->kernel_count == 1u && image->site_count == 5u);
	for (index=0u; index<5u; index++)
	{
		assert(image->sites[index].node == nodes[index]);
		assert(image->sites[index].region_id == ids[index]);
		assert(image->sites[index].offset == offsets[index]);
		assert(image->sites[index].region_slot == (ids[index] == ID_STAGING ? slot : SPARK_GRAPH_RELOC_SLOT_SHARED));
	}
	assert(image->sites[0].byte_offset == 0u && image->sites[1].byte_offset == 16u);
	assert(image->nodes[3].site_count == 0u && image->nodes[4].site_count == 0u);
}

static void TestTwoCaptures(void)
{
	SparkGraphRelocWorkspace *workspace = Workspace(16u,UINT64_C(1) << 30);
	SparkGraphRelocImage *first = &workspace->images[0],*second = &workspace->images[1];
	SparkGraphRelocFault fault;
	uint32_t applied = 0u;
	FakeArgs args;
	Register(&workspace->registry,DEVICE_BASE);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,(cudaEvent_t)(uintptr_t)0x11u);
	BuildGraph(&GRAPHS[1],DEVICE_BASE,STAGING_SLOT1,7u,(cudaEvent_t)(uintptr_t)0x11u);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,first,&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&workspace->registry,1u,second,&fault) == SPARK_STATUS_OK);
	CheckSites(first,0u);
	CheckSites(second,1u);
	assert(SparkGraphRelocValidatePair(first,second,&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRebase(first,&workspace->registry,1u,workspace->patch_blob,first->blob_capacity,&fault) == SPARK_STATUS_OK);
	assert(memcmp(workspace->patch_blob,second->blob,(size_t)second->blob_bytes) == 0);
	assert(memcmp(workspace->patch_blob,first->blob,(size_t)first->blob_bytes) != 0);
	assert(SparkGraphRelocApplyCuda((void *)&GRAPHS[0],first,workspace->patch_blob,0u,&applied,&fault) == SPARK_STATUS_OK);
	assert(applied == 2u);
	assert(GRAPHS[0].nodes[0].applied_count == 1u && GRAPHS[0].nodes[1].applied_count == 1u && GRAPHS[0].nodes[2].applied_count == 0u);
	memcpy(&args,GRAPHS[0].nodes[0].applied + 16u,sizeof(args));
	assert(args.staging == STAGING_SLOT1 + 16u && args.a == 3u && args.b == 4u);
	assert((uint64_t)(uintptr_t)GRAPHS[0].nodes[1].copy.dstPtr.ptr == STAGING_SLOT1);
	assert((uint64_t)(uintptr_t)GRAPHS[0].nodes[1].copy.srcPtr.ptr == WEIGHT_BASE + 128u);
	assert(SparkGraphRelocApplyCuda((void *)&GRAPHS[0],first,workspace->patch_blob,1u,&applied,&fault) == SPARK_STATUS_OK);
	assert(applied == 3u && GRAPHS[0].nodes[2].applied_count == 1u && GRAPHS[0].nodes[3].applied_count == 0u);
	SparkGraphRelocWorkspaceDestroy(workspace);
}

static void TestPlantedScalars(void)
{
	SparkGraphRelocWorkspace *workspace = Workspace(16u,UINT64_C(1) << 30);
	SparkGraphRelocWorkspace *shadow = Workspace(16u,0u);
	SparkGraphRelocFault fault;
	Register(&workspace->registry,DEVICE_BASE);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,STAGING_SLOT0,0);
	BuildGraph(&GRAPHS[1],DEVICE_BASE,STAGING_SLOT1,STAGING_SLOT0,0);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&workspace->registry,1u,&workspace->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_SLOT && fault.node == 0u && fault.byte_offset == 32u && fault.word == STAGING_SLOT0);
	Register(&shadow->registry,DEVICE_BASE);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,DEVICE_BASE,0);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&shadow->registry,0u,&shadow->images[0],&fault) == SPARK_STATUS_OK);
	Register(&shadow->registry,SHADOW_DEVICE);
	BuildGraph(&GRAPHS[1],SHADOW_DEVICE,STAGING_SLOT0,DEVICE_BASE,0);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&shadow->registry,0u,&shadow->images[1],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocValidatePair(&shadow->images[0],&shadow->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_SHAPE && fault.node == 0u);
	Register(&workspace->registry,SHADOW_DEVICE);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&workspace->registry,0u,&workspace->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_UNKNOWN && fault.byte_offset == 32u && fault.word == DEVICE_BASE);
	SparkGraphRelocWorkspaceDestroy(shadow);
	SparkGraphRelocWorkspaceDestroy(workspace);
}

static void TestRefusals(void)
{
	SparkGraphRelocWorkspace *workspace = Workspace(16u,UINT64_C(1) << 30);
	SparkGraphRelocWorkspace *small = Workspace(2u,UINT64_C(1) << 30);
	void *extra[1] = {0};
	SparkGraphRelocFault fault;
	Register(&workspace->registry,DEVICE_BASE);
	Register(&small->registry,DEVICE_BASE);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,0);
	GRAPHS[0].nodes[0].args.staging = ROGUE;
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_UNKNOWN && fault.node == 0u && fault.byte_offset == 16u && fault.word == ROGUE);
	assert(strcmp(SparkGraphRelocReasonName(fault.reason),"GRAPH-RELOC-UNKNOWN") == 0);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,0);
	GRAPHS[0].nodes[2].type = cudaGraphNodeTypeMemAlloc;
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_UNSUPPORTED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_NODE_KIND && fault.node == 2u && fault.detail == cudaGraphNodeTypeMemAlloc);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,0);
	GRAPHS[0].nodes[0].extra = extra;
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_IO_ERROR);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_NODE_QUERY && fault.node == 0u);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,0);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&small->registry,0u,&small->images[0],&fault) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_CAPACITY);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,(cudaEvent_t)(uintptr_t)0x11u);
	BuildGraph(&GRAPHS[1],DEVICE_BASE,STAGING_SLOT1,7u,(cudaEvent_t)(uintptr_t)0x12u);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&workspace->registry,1u,&workspace->images[1],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocValidatePair(&workspace->images[0],&workspace->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_CONSTANT && fault.node == 4u && fault.word == 0x12u);
	GRAPHS[1].nodes[4].event = (cudaEvent_t)(uintptr_t)0x11u;
	GRAPHS[1].nodes[0].args.a = 5u;
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[1],&workspace->registry,1u,&workspace->images[1],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocValidatePair(&workspace->images[0],&workspace->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_CONSTANT && fault.node == 0u && fault.byte_offset == 24u);
	SparkGraphRelocWorkspaceDestroy(small);
	SparkGraphRelocWorkspaceDestroy(workspace);
}

static void TestRebaseTargets(void)
{
	SparkGraphRelocWorkspace *workspace = Workspace(16u,UINT64_C(1) << 30);
	SparkGraphRelocWorkspace *target = Workspace(16u,UINT64_C(1) << 30);
	SparkGraphRelocFault fault;
	Register(&workspace->registry,DEVICE_BASE);
	BuildGraph(&GRAPHS[0],DEVICE_BASE,STAGING_SLOT0,7u,0);
	assert(SparkGraphRelocCaptureCuda(&GRAPHS[0],&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRebase(&workspace->images[0],&workspace->registry,2u,workspace->patch_blob,workspace->images[0].blob_capacity,&fault) == SPARK_STATUS_NOT_FOUND);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_TARGET && fault.word == ID_STAGING);
	SparkGraphRelocRegistryReset(&target->registry);
	assert(SparkGraphRelocRegister(&target->registry,ID_WEIGHT,(void *)(uintptr_t)WEIGHT_BASE,64u,3u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&target->registry,ID_HIDDEN,(void *)(uintptr_t)SHADOW_DEVICE,4096u,1u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRegister(&target->registry,ID_STAGING,(void *)(uintptr_t)STAGING_SLOT0,1024u,2u,0u) == SPARK_STATUS_OK);
	assert(SparkGraphRelocSeal(&target->registry,&fault) == SPARK_STATUS_OK);
	assert(SparkGraphRelocRebase(&workspace->images[0],&target->registry,0u,workspace->patch_blob,workspace->images[0].blob_capacity,&fault) == SPARK_STATUS_NOT_FOUND);
	assert(fault.reason == SPARK_GRAPH_RELOC_REASON_TARGET && fault.node == 1u);
	assert(SparkGraphRelocRebase(&workspace->images[0],&target->registry,0u,workspace->patch_blob,8u,&fault) == SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkGraphRelocWorkspaceDestroy(target);
	SparkGraphRelocWorkspaceDestroy(workspace);
}

int main(void)
{
	SparkGraphRelocWorkspace *workspace = 0;
	SparkGraphRelocCapacity empty = {0u,0u,0u,0u,0u};
	assert(SparkGraphRelocWorkspaceCreate(&empty,0u,&workspace) == SPARK_STATUS_INVALID_ARGUMENT && workspace == 0);
	TestRegistry();
	TestTwoCaptures();
	TestPlantedScalars();
	TestRefusals();
	TestRebaseTargets();
	printf("PASS graph relocation registry, walker, two-capture validation, rebase and exec patch\n");
	return(0);
}
