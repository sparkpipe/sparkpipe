#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_graph_reloc.h"

#define ROWS 4u
#define WIDTH 256u
#define WEIGHT_COUNT (8u * WIDTH)
#define SLOTS 2u
#define REPEATS 200u
#define GUARD 512u

#define CHECK(condition) do { if ( !(condition) ) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#condition); exit(1); } } while (0)
#define CUDA_CHECK(call) do { cudaError_t check_error = (call); if ( check_error != cudaSuccess ) { fprintf(stderr,"FAIL %s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(check_error)); exit(1); } } while (0)

enum { ID_WEIGHT = 1u, ID_HIDDEN, ID_SCRATCH, ID_OUTPUT, ID_HOST, ID_ROGUE };

typedef struct ScratchView
{
	float *scratch;
	uint32_t rows;
	uint32_t width;
} ScratchView;

typedef struct Slot
{
	float *hidden;
	float *scratch;
	float *output;
	float *host;
} Slot;

static __global__ void ProjectKernel(const float *hidden,const float *weight,ScratchView view,uint32_t expert)
{
	uint32_t row = blockIdx.x,column = threadIdx.x,k;
	float sum = view.scratch[row * view.width + column];
	for (k=0u; k<8u; k++)
		sum += hidden[row * WIDTH + (column + k * 31u) % WIDTH] * weight[((expert + k) % 7u) * WIDTH + column];
	view.scratch[row * view.width + column] = sum;
}

static __global__ void FinishKernel(const float *scratch,float *output,float scale,const float *weight)
{
	uint32_t index = blockIdx.x * WIDTH + threadIdx.x;
	output[index] = tanhf(scratch[index] * scale) + weight[threadIdx.x];
}

static float Value(uint64_t seed)
{
	seed = (seed + UINT64_C(0x9e3779b97f4a7c15)) * UINT64_C(0xbf58476d1ce4e5b9);
	seed ^= seed >> 29u;
	return((float)((seed >> 40u) & 0xffffu) / 32768.0f - 1.0f);
}

static void Fill(float *device,uint32_t count,uint64_t seed)
{
	float *host = (float *)malloc(count * sizeof(float));
	uint32_t index;
	CHECK(host != 0);
	for (index=0u; index<count; index++)
		host[index] = Value(seed + index);
	CUDA_CHECK(cudaMemcpy(device,host,count * sizeof(float),cudaMemcpyHostToDevice));
	free(host);
}

static void SlotCreate(Slot *slot,uint32_t index)
{
	CUDA_CHECK(cudaMalloc((void **)&slot->hidden,ROWS * WIDTH * sizeof(float) + GUARD));
	CUDA_CHECK(cudaMalloc((void **)&slot->scratch,ROWS * WIDTH * sizeof(float) + GUARD));
	CUDA_CHECK(cudaMalloc((void **)&slot->output,ROWS * WIDTH * sizeof(float) + GUARD));
	CUDA_CHECK(cudaHostAlloc((void **)&slot->host,ROWS * WIDTH * sizeof(float) + GUARD,cudaHostAllocDefault));
	Fill(slot->hidden,ROWS * WIDTH,UINT64_C(1000) * (index + 1u));
}

static void Record(cudaStream_t stream,const Slot *slot,const float *weight,const float *rogue)
{
	ScratchView view;
	view.scratch = slot->scratch;
	view.rows = ROWS;
	view.width = WIDTH;
	CUDA_CHECK(cudaMemsetAsync(slot->scratch,0,ROWS * WIDTH * sizeof(float),stream));
	ProjectKernel<<<ROWS,WIDTH,0,stream>>>(slot->hidden,weight + WIDTH,view,3u);
	FinishKernel<<<ROWS,WIDTH,0,stream>>>(slot->scratch,slot->output,0.25f,rogue != 0 ? rogue : weight);
	CUDA_CHECK(cudaMemcpyAsync(slot->host,slot->output,ROWS * WIDTH * sizeof(float),cudaMemcpyDeviceToHost,stream));
	CUDA_CHECK(cudaGetLastError());
}

static cudaGraph_t Capture(cudaStream_t stream,const Slot *slot,const float *weight,const float *rogue)
{
	cudaGraph_t graph;
	CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
	Record(stream,slot,weight,rogue);
	CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
	return(graph);
}

static void Register(SparkGraphRelocRegistry *registry,const float *weight,const Slot *slots)
{
	uint32_t index;
	SparkGraphRelocFault fault;
	SparkGraphRelocRegistryReset(registry);
	CHECK(SparkGraphRelocRegister(registry,ID_WEIGHT,weight,WEIGHT_COUNT * sizeof(float),1u,SPARK_GRAPH_RELOC_SLOT_SHARED) == SPARK_STATUS_OK);
	for (index=0u; index<SLOTS; index++)
	{
		CHECK(SparkGraphRelocRegister(registry,ID_HIDDEN,slots[index].hidden,ROWS * WIDTH * sizeof(float),2u,index) == SPARK_STATUS_OK);
		CHECK(SparkGraphRelocRegister(registry,ID_SCRATCH,slots[index].scratch,ROWS * WIDTH * sizeof(float),2u,index) == SPARK_STATUS_OK);
		CHECK(SparkGraphRelocRegister(registry,ID_OUTPUT,slots[index].output,ROWS * WIDTH * sizeof(float),2u,index) == SPARK_STATUS_OK);
		CHECK(SparkGraphRelocRegister(registry,ID_HOST,slots[index].host,ROWS * WIDTH * sizeof(float),3u,index) == SPARK_STATUS_OK);
	}
	CHECK(SparkGraphRelocSeal(registry,&fault) == SPARK_STATUS_OK);
	registry->probe = SparkGraphRelocCudaProbe;
}

static void Run(cudaGraphExec_t exec,cudaStream_t stream,const Slot *slot,float *out)
{
	CUDA_CHECK(cudaMemsetAsync(slot->output,0xff,ROWS * WIDTH * sizeof(float),stream));
	memset(slot->host,0,ROWS * WIDTH * sizeof(float));
	CUDA_CHECK(cudaGraphLaunch(exec,stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	memcpy(out,slot->host,ROWS * WIDTH * sizeof(float));
}

static SparkStatus Walk(cudaGraph_t graph,const SparkGraphRelocRegistry *registry,uint32_t slot,SparkGraphRelocImage *image,SparkGraphRelocFault *fault)
{
	SparkStatus status;
	memset(fault,0,sizeof(*fault));
	status = SparkGraphRelocCaptureCuda(graph,registry,slot,image,fault);
	if ( status != SPARK_STATUS_OK )
		printf("walk slot=%u status=%d %s node=%u byte=%u word=0x%llx detail=%u\n",slot,(int)status,SparkGraphRelocReasonName(fault->reason),fault->node,fault->byte_offset,(unsigned long long)fault->word,fault->detail);
	return(status);
}

static double Now(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec + (double)now.tv_nsec * 1e-9);
}

int main(void)
{
	static float fresh[ROWS * WIDTH],patched[ROWS * WIDTH],moved[ROWS * WIDTH],own[ROWS * WIDTH];
	SparkGraphRelocCapacity capacity = {UINT64_C(1) << 16,16u,64u,256u,256u};
	SparkGraphRelocWorkspace *workspace = 0,*target = 0;
	SparkGraphRelocFault fault;
	Slot slots[SLOTS];
	float *weight,*weight_moved,*rogue;
	cudaStream_t stream;
	cudaGraph_t graph0,graph1,graph_moved,graph_rogue;
	cudaGraphExec_t exec0,exec1,exec_moved;
	uint32_t applied = 0u,index;
	double start,capture_us,patch_us;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	CUDA_CHECK(cudaMalloc((void **)&weight,WEIGHT_COUNT * sizeof(float) + GUARD));
	CUDA_CHECK(cudaMalloc((void **)&weight_moved,WEIGHT_COUNT * sizeof(float) + GUARD));
	CUDA_CHECK(cudaMalloc((void **)&rogue,WEIGHT_COUNT * sizeof(float) + GUARD));
	Fill(weight,WEIGHT_COUNT,UINT64_C(77));
	CUDA_CHECK(cudaMemcpy(weight_moved,weight,WEIGHT_COUNT * sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(rogue,weight,WEIGHT_COUNT * sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaDeviceSynchronize());
	for (index=0u; index<SLOTS; index++)
		SlotCreate(&slots[index],index);
	CHECK(SparkGraphRelocWorkspaceCreate(&capacity,UINT64_C(1) << 30,&workspace) == SPARK_STATUS_OK);
	CHECK(SparkGraphRelocWorkspaceCreate(&capacity,UINT64_C(1) << 30,&target) == SPARK_STATUS_OK);
	Register(&workspace->registry,weight,slots);
	graph0 = Capture(stream,&slots[0],weight,0);
	graph1 = Capture(stream,&slots[1],weight,0);
	CHECK(Walk(graph0,&workspace->registry,0u,&workspace->images[0],&fault) == SPARK_STATUS_OK);
	CHECK(Walk(graph1,&workspace->registry,1u,&workspace->images[1],&fault) == SPARK_STATUS_OK);
	CHECK(workspace->images[0].node_count == 4u && workspace->images[0].kernel_count == 2u);
	CHECK(workspace->images[0].site_count == workspace->images[1].site_count && workspace->images[0].site_count >= 8u);
	CHECK(SparkGraphRelocValidatePair(&workspace->images[0],&workspace->images[1],&fault) == SPARK_STATUS_OK);
	printf("ok two captures on slots 0 and 1 differ only at %u relocation sites (%llu words scanned)\n",workspace->images[0].site_count,(unsigned long long)workspace->images[0].word_count);
	CUDA_CHECK(cudaGraphInstantiate(&exec0,graph0,0));
	CUDA_CHECK(cudaGraphInstantiate(&exec1,graph1,0));
	Run(exec0,stream,&slots[0],own);
	Run(exec1,stream,&slots[1],fresh);
	CHECK(memcmp(own,fresh,sizeof(fresh)) != 0);
	CHECK(SparkGraphRelocRebase(&workspace->images[0],&workspace->registry,1u,workspace->patch_blob,workspace->images[0].blob_capacity,&fault) == SPARK_STATUS_OK);
	CHECK(memcmp(workspace->patch_blob,workspace->images[1].blob,(size_t)workspace->images[1].blob_bytes) == 0);
	CHECK(SparkGraphRelocApplyCuda(exec0,&workspace->images[0],workspace->patch_blob,0u,&applied,&fault) == SPARK_STATUS_OK);
	CHECK(applied == 4u);
	Run(exec0,stream,&slots[1],patched);
	CHECK(memcmp(patched,fresh,sizeof(fresh)) == 0);
	printf("ok slot-0 exec patched in place to slot 1 (%u nodes) gives output bit-identical to a fresh slot-1 capture\n",applied);
	SparkGraphRelocRegistryReset(&target->registry);
	Register(&target->registry,weight_moved,slots);
	CHECK(SparkGraphRelocRebase(&workspace->images[0],&target->registry,1u,target->patch_blob,workspace->images[0].blob_capacity,&fault) == SPARK_STATUS_OK);
	start = Now();
	for (index=0u; index<REPEATS; index++)
	{
		CHECK(SparkGraphRelocRebase(&workspace->images[0],&target->registry,1u,target->patch_blob,workspace->images[0].blob_capacity,&fault) == SPARK_STATUS_OK);
		CHECK(SparkGraphRelocApplyCuda(exec0,&workspace->images[0],target->patch_blob,1u,&applied,&fault) == SPARK_STATUS_OK);
	}
	patch_us = (Now() - start) * 1e6 / REPEATS;
	graph_moved = Capture(stream,&slots[1],weight_moved,0);
	CUDA_CHECK(cudaGraphInstantiate(&exec_moved,graph_moved,0));
	Run(exec_moved,stream,&slots[1],fresh);
	CUDA_CHECK(cudaMemsetAsync(weight,0xff,WEIGHT_COUNT * sizeof(float),stream));
	Run(exec0,stream,&slots[1],moved);
	CHECK(memcmp(moved,fresh,sizeof(fresh)) == 0);
	printf("ok shared weight base moved: patched exec reads only the new base (old base poisoned) and matches a fresh capture bitwise\n");
	start = Now();
	for (index=0u; index<REPEATS; index++)
	{
		cudaGraph_t graph = Capture(stream,&slots[1],weight_moved,0);
		cudaGraphExec_t exec;
		CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));
		CUDA_CHECK(cudaGraphExecDestroy(exec));
		CUDA_CHECK(cudaGraphDestroy(graph));
	}
	capture_us = (Now() - start) * 1e6 / REPEATS;
	graph_rogue = Capture(stream,&slots[0],weight,rogue);
	CHECK(Walk(graph_rogue,&workspace->registry,0u,&workspace->images[1],&fault) == SPARK_STATUS_VALIDATION_FAILED);
	CHECK(fault.reason == SPARK_GRAPH_RELOC_REASON_UNKNOWN && fault.word == (uint64_t)(uintptr_t)rogue);
	printf("ok a kernel argument pointing at unregistered device memory is refused: %s node=%u\n",SparkGraphRelocReasonName(fault.reason),fault.node);
	printf("TIMING 4-node graph: capture+instantiate %.1f us, rebase+patch %.1f us\n",capture_us,patch_us);
	CUDA_CHECK(cudaGraphExecDestroy(exec0));
	CUDA_CHECK(cudaGraphExecDestroy(exec1));
	CUDA_CHECK(cudaGraphExecDestroy(exec_moved));
	CUDA_CHECK(cudaGraphDestroy(graph0));
	CUDA_CHECK(cudaGraphDestroy(graph1));
	CUDA_CHECK(cudaGraphDestroy(graph_moved));
	CUDA_CHECK(cudaGraphDestroy(graph_rogue));
	SparkGraphRelocWorkspaceDestroy(target);
	SparkGraphRelocWorkspaceDestroy(workspace);
	printf("PASS relocatable graphs: patched-in-place graphs equal freshly captured ones on real CUDA graphs\n");
	return(0);
}
