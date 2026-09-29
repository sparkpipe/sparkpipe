#define main SparkGlm52ValidatorMain
#include "spark_glm52_resident_decode_stage_cuda_validation.cu"
#undef main

#include <algorithm>
#include <time.h>
#include <cuda.h>

#include "sparkpipe/spark_glm52_graph_regime.h"

#define BENCH_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define BENCH_SPLIT_THRESHOLD 64u
#define BENCH_POSITION 200u
#define BENCH_WARM 3u
#define BENCH_REPLAYS 25u
#define BENCH_PEERS 16u
#define BENCH_THREADS 256u

typedef struct BenchRig
{
	SparkGlm52ValFixture fixture;
	SparkGlm52LayerWeights layers[BENCH_LAYERS];
	uint32_t ordinals[BENCH_LAYERS];
	uint32_t *host_words;
	uint16_t *boundary;
	uint16_t *gather;
	uint16_t *peer_slots;
	uint16_t *combined;
	uint64_t *request_word;
	uint64_t *ready_word;
	uint32_t rounds_per_step;
	uint32_t emulate_rounds;
	uint32_t wait_ns;
	uint32_t l2_prefetch;
	uint32_t l2_placed;
	SparkL2PrefetchShape l2_shape;
	uint32_t tp_degree;
} BenchRig;

typedef struct BenchResult
{
	double launch_us;
	double gpu_ms;
	double wall_ms;
	size_t nodes;
	uint32_t rounds;
	uint32_t l2_placed;
} BenchResult;

__global__ static void BenchRequestKernel(uint64_t *request,uint64_t kind)
{
	*(volatile uint64_t *)request = kind;
}

__global__ static void BenchGuardKernel(const uint64_t *ready,uint64_t *request)
{
	if ( *(volatile const uint64_t *)ready != 1u )
		*(volatile uint64_t *)request = UINT64_MAX;
}

__global__ static void BenchPeerLatencyKernel(uint32_t wait_ns)
{
	uint64_t start,now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < wait_ns );
}

__global__ static void BenchPublishKernel(uint16_t *slot,const uint16_t *local,uint32_t count)
{
	uint32_t index;
	for (index=threadIdx.x; index<count; index+=blockDim.x)
		slot[index] = local[index];
}

__global__ static void BenchCombineKernel(const uint16_t *slots,uint16_t *output,uint32_t count,uint32_t peers)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t peer;
	float total = 0.0f;
	if ( index >= count )
		return;
	for (peer=0u; peer<peers; peer++)
		total += __uint_as_float((uint32_t)slots[(uint64_t)peer * count + index] << 16u);
	output[index] = (uint16_t)(__float_as_uint(total) >> 16u);
}

static cudaError_t BenchWaitNode(cudaStream_t stream,uint64_t *ready)
{
	CUstreamCaptureStatus capture;
	cuuint64_t id;
	CUgraph graph;
	const CUgraphNode *dependencies;
	size_t dependency_count;
	CUstreamBatchMemOpParams operation = {};
	CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
	CUgraphNode wait;
	if ( cuStreamGetCaptureInfo((CUstream)stream,&capture,&id,&graph,&dependencies,0,&dependency_count) != CUDA_SUCCESS )
		return(cudaErrorUnknown);
	if ( capture != CU_STREAM_CAPTURE_STATUS_ACTIVE )
		return(cuStreamWaitValue64((CUstream)stream,(CUdeviceptr)ready,1u,CU_STREAM_WAIT_VALUE_EQ) == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown);
	operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
	operation.waitValue.address = (CUdeviceptr)ready;
	operation.waitValue.value64 = 1u;
	operation.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
	if ( cuCtxGetCurrent(&params.ctx) != CUDA_SUCCESS )
		return(cudaErrorUnknown);
	params.count = 1u;
	params.paramArray = &operation;
	if ( cuGraphAddBatchMemOpNode(&wait,graph,dependencies,dependency_count,&params) != CUDA_SUCCESS )
		return(cudaErrorUnknown);
	if ( cuStreamUpdateCaptureDependencies((CUstream)stream,&wait,0,1u,CU_STREAM_SET_CAPTURE_DEPENDENCIES) != CUDA_SUCCESS )
		return(cudaErrorUnknown);
	return(cudaSuccess);
}

static int BenchRound(BenchRig *rig,const uint16_t *local,uint32_t layer,uint32_t site)
{
	cudaStream_t stream = rig->fixture.stream;
	uint32_t count = SPARK_GLM52_VHIDDEN;
	uint32_t placed = 0u;
	rig->rounds_per_step++;
	if ( rig->emulate_rounds == 0u )
		return(0);
	BenchRequestKernel<<<1,1,0,stream>>>(rig->request_word,1u);
	if ( BenchWaitNode(stream,rig->ready_word) != cudaSuccess )
		return(1);
	BenchGuardKernel<<<1,1,0,stream>>>(rig->ready_word,rig->request_word);
	BenchPublishKernel<<<1,BENCH_THREADS,0,stream>>>(rig->peer_slots,local,count);
	BenchRequestKernel<<<1,1,0,stream>>>(rig->request_word,2u);
	if ( BenchWaitNode(stream,rig->ready_word) != cudaSuccess )
		return(1);
	if ( rig->wait_ns != 0u )
		BenchPeerLatencyKernel<<<1,1,0,stream>>>(rig->wait_ns);
	BenchGuardKernel<<<1,1,0,stream>>>(rig->ready_word,rig->request_word);
	if ( rig->l2_prefetch != 0u && site != UINT32_MAX )
	{
		if ( SparkGlm52L2PrefetchAfterRound(&rig->fixture.wave,layer,site,&rig->l2_shape,&placed) != 0 )
			return(1);
		rig->l2_placed += placed;
	}
	BenchCombineKernel<<<(count + BENCH_THREADS - 1u) / BENCH_THREADS,BENCH_THREADS,0,stream>>>(rig->peer_slots,rig->combined,count,BENCH_PEERS);
	return(cudaPeekAtLastError() == cudaSuccess ? 0 : 1);
}

static void BenchBuild(BenchRig *rig,uint32_t split)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	SparkGlm52ValBuildWave(&rig->fixture,0u,1u,BENCH_POSITION);
	for (layer=0u; layer<BENCH_LAYERS; layer++)
	{
		rig->layers[layer] = rig->fixture.weights;
		rig->ordinals[layer] = 0u;
	}
	rig->host_words[0] = 1u;
	rig->host_words[1] = 0u;
	rig->host_words[2] = BENCH_POSITION;
	wave->layers = rig->layers;
	wave->layer_count = BENCH_LAYERS;
	wave->index_ordinal_by_local_layer = rig->ordinals;
	wave->host_token_ids = rig->host_words;
	wave->host_resident_slots = rig->host_words + 1;
	wave->host_positions = rig->host_words + 2;
	wave->tp_degree = rig->tp_degree;
	wave->tp_rank = 0u;
	wave->owns_final_head = 0u;
	wave->hidden_output_bf16 = rig->boundary;
	wave->boundary_row_offset = 0u;
	wave->decode_split_context_threshold = BENCH_SPLIT_THRESHOLD;
	wave->max_sequence_positions = SPARK_GLM52_VALIDATION_DSA_CONTEXT;
	wave->projection_split = split;
	rig->fixture.slot.projection_gather_bf16 = split != 0u ? rig->gather : 0;
}

static int BenchWalk(BenchRig *rig,uint32_t split)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	rig->rounds_per_step = 0u;
	rig->l2_placed = 0u;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 || BenchRound(rig,rig->fixture.hidden,0u,SPARK_GLM52_L2_SITE_BEGIN) != 0 )
		return(1);
	for (layer=0u; layer<BENCH_LAYERS; layer++)
	{
		if ( split != 0u )
		{
			if ( SparkGlm52LaunchCudaLayerAttentionProject(wave,layer) != 0 || BenchRound(rig,rig->gather,layer,SPARK_GLM52_L2_SITE_PROJECTION) != 0 ||
				SparkGlm52LaunchCudaLayerAttentionCore(wave,layer) != 0 )
				return(2);
		}
		else if ( SparkGlm52LaunchCudaLayerAttention(wave,layer) != 0 )
			return(3);
		if ( BenchRound(rig,rig->fixture.attention_out,layer,SPARK_GLM52_L2_SITE_ATTENTION) != 0 || SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 ||
			BenchRound(rig,rig->fixture.hidden,layer,SPARK_GLM52_L2_SITE_MLP) != 0 )
			return(4);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 || BenchRound(rig,rig->fixture.hidden,0u,UINT32_MAX) != 0 )
		return(5);
	return(0);
}

static double BenchNowUs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec * 1e6 + (double)now.tv_nsec / 1e3);
}

static double BenchMedian(double *values,uint32_t count)
{
	std::sort(values,values + count);
	return(values[count / 2u]);
}

static int BenchRun(BenchRig *rig,uint32_t split,uint32_t graph_mode,BenchResult *result)
{
	cudaStream_t stream = rig->fixture.stream;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	cudaEvent_t begin,end;
	double launch[BENCH_REPLAYS],gpu[BENCH_REPLAYS],wall[BENCH_REPLAYS];
	uint32_t replay,regime,bound;
	float elapsed;
	double t0,t1;
	memset(result,0,sizeof(*result));
	if ( SparkGlm52ValResetStreams(&rig->fixture) != 0 || cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess )
		return(10);
	BenchBuild(rig,split);
	regime = SparkGlm52GraphRegime(BENCH_POSITION + 1u,BENCH_SPLIT_THRESHOLD,rig->fixture.wave.max_sequence_positions,&bound);
	(void)regime;
	if ( graph_mode != 0u )
	{
		rig->fixture.wave.maximum_context = bound;
		if ( cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
			return(11);
		if ( BenchWalk(rig,split) != 0 )
		{
			(void)cudaStreamEndCapture(stream,&graph);
			return(12);
		}
		result->l2_placed = rig->l2_placed;
		if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess || graph == 0 )
			return(13);
		rig->fixture.wave.maximum_context = BENCH_POSITION + 1u;
		if ( cudaGraphGetNodes(graph,0,&result->nodes) != cudaSuccess )
			return(14);
		if ( cudaGraphInstantiate(&exec,graph,0) != cudaSuccess || cudaGraphUpload(exec,stream) != cudaSuccess )
			return(15);
		(void)cudaGraphDestroy(graph);
	}
	for (replay=0u; replay<BENCH_WARM + BENCH_REPLAYS; replay++)
	{
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			return(16);
		t0 = BenchNowUs();
		if ( cudaEventRecord(begin,stream) != cudaSuccess )
			return(17);
		if ( graph_mode != 0u )
		{
			if ( cudaGraphLaunch(exec,stream) != cudaSuccess )
				return(18);
		}
		else if ( BenchWalk(rig,split) != 0 )
			return(19);
		t1 = BenchNowUs();
		if ( cudaEventRecord(end,stream) != cudaSuccess || cudaEventSynchronize(end) != cudaSuccess || cudaEventElapsedTime(&elapsed,begin,end) != cudaSuccess )
			return(20);
		if ( replay >= BENCH_WARM )
		{
			launch[replay - BENCH_WARM] = t1 - t0;
			gpu[replay - BENCH_WARM] = (double)elapsed;
			wall[replay - BENCH_WARM] = (BenchNowUs() - t0) / 1000.0;
		}
	}
	if ( SparkGlm52ValCheckAccessError(&rig->fixture) != 0 )
		return(21);
	result->launch_us = BenchMedian(launch,BENCH_REPLAYS);
	result->gpu_ms = BenchMedian(gpu,BENCH_REPLAYS);
	result->wall_ms = BenchMedian(wall,BENCH_REPLAYS);
	result->rounds = rig->rounds_per_step;
	if ( exec != 0 )
		(void)cudaGraphExecDestroy(exec);
	(void)cudaEventDestroy(begin);
	(void)cudaEventDestroy(end);
	return(0);
}

int main(int argc,char **argv)
{
	static const char *const names[2] = {"replicated","split"};
	BenchRig *rig;
	BenchResult result;
	uint32_t split,rounds,graph_mode;
	int status,failures = 0;
	uint64_t one = 1u;
	rig = (BenchRig *)calloc(1u,sizeof(*rig));
	if ( rig == 0 || SparkGlm52ValFixtureSetup(&rig->fixture) != 0 )
		return(1);
	rig->tp_degree = 16u;
	rig->wait_ns = argc >= 5 ? (uint32_t)atoi(argv[4]) * 1000u : 0u;
	rig->l2_prefetch = argc >= 6 ? (uint32_t)atoi(argv[5]) : 0u;
	rig->l2_shape.bytes = argc >= 7 ? (uint32_t)atoi(argv[6]) : SPARK_L2_PREFETCH_BYTES_DEFAULT;
	rig->l2_shape.blocks = argc >= 8 ? (uint32_t)atoi(argv[7]) : SPARK_L2_PREFETCH_BLOCKS_DEFAULT;
	if ( SparkL2PrefetchShapeValid(&rig->l2_shape) == 0u )
	{
		fprintf(stderr,"glm52_step_bench: bytes must be a multiple of %u up to %u and blocks 1..%u\n",SPARK_L2_PREFETCH_BYTES_STEP,SPARK_L2_PREFETCH_BYTES_MAX,SPARK_L2_PREFETCH_BLOCKS_MAX);
		return(2);
	}
	rig->fixture.split_threshold = BENCH_SPLIT_THRESHOLD;
	rig->fixture.split_partials = (float *)SparkGlm52ValAllocZeroed(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(1u,SPARK_GLM52_MODEL_HEAD_COUNT));
	if ( rig->fixture.split_partials == 0 ||
		cudaHostAlloc((void **)&rig->host_words,4u * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess ||
		cudaMalloc((void **)&rig->boundary,2u * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->gather,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->peer_slots,(uint64_t)BENCH_PEERS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->combined,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->request_word,sizeof(uint64_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->ready_word,sizeof(uint64_t)) != cudaSuccess ||
		cudaMemset(rig->gather,0,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(rig->peer_slots,0,(uint64_t)BENCH_PEERS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemcpy(rig->ready_word,&one,sizeof(one),cudaMemcpyHostToDevice) != cudaSuccess )
		return(1);
	for (split=0u; split<2u; split++)
		for (rounds=0u; rounds<2u; rounds++)
			for (graph_mode=0u; graph_mode<2u; graph_mode++)
			{
				if ( argc >= 4 && (split != (uint32_t)atoi(argv[1]) || rounds != (uint32_t)atoi(argv[2]) || graph_mode != (uint32_t)atoi(argv[3])) )
					continue;
				rig->emulate_rounds = rounds;
				status = BenchRun(rig,split,graph_mode,&result);
				if ( status != 0 )
				{
					fprintf(stderr,"glm52_step_bench FAIL projections=%s rounds=%s mode=%s status=%d cuda=%s\n",names[split],rounds != 0u ? "emulated" : "none",graph_mode != 0u ? "graph" : "linear",status,cudaGetErrorString(cudaGetLastError()));
					failures++;
					continue;
				}
				printf("glm52_step_bench tp=%u layers=%u position=%u projections=%s rounds=%s collectives=%u peer_wait_us=%u l2_prefetch=%u l2_bytes=%u l2_blocks=%u l2_placed=%u mode=%s nodes=%zu launch_us=%.1f gpu_ms=%.3f wall_ms=%.3f\n",
					rig->tp_degree,BENCH_LAYERS,BENCH_POSITION,names[split],rounds != 0u ? "emulated" : "none",result.rounds,rig->wait_ns / 1000u,rig->l2_prefetch,rig->l2_shape.bytes,rig->l2_shape.blocks,result.l2_placed,graph_mode != 0u ? "graph" : "linear",result.nodes,result.launch_us,result.gpu_ms,result.wall_ms);
			}
	SparkGlm52ValFixtureDestroy(&rig->fixture);
	printf("glm52_step_bench %s\n",failures == 0 ? "PASS" : "FAIL");
	return(failures == 0 ? 0 : 1);
}
