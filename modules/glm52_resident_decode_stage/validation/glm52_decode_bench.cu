#define main SparkGlm52ValidatorMain
#include "spark_glm52_resident_decode_stage_cuda_validation.cu"
#undef main

#include <algorithm>
#include <time.h>
#include <cuda.h>

#include "sparkpipe/spark_glm52_graph_regime.h"
#include "sparkpipe/llm_defines.h"
#include "common/common_glm_cuda_tree/spark_glm_cuda_config.h"

#define DB_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define DB_MAX_ROWS SPARK_GLM52_VAL_MAX_ROWS
#define DB_SPLIT_THRESHOLD 64u
#define DB_WARM 3u
#define DB_REPLAYS 25u
#define DB_PEERS 16u
#define DB_THREADS 256u
#define DB_TP 16u
#define DB_EXPERTS 256u
#define DB_MAX_CONTEXT 2048u
#define DB_PAGES ((DB_MAX_CONTEXT + SPARK_GLM52_VKV_PAGE_SLOTS - 1u) / SPARK_GLM52_VKV_PAGE_SLOTS)
#define DB_PEAK_GBPS 243.0

typedef struct DbRig
{
	SparkGlm52ValFixture fixture;
	SparkGlm52LayerWeights layers[DB_LAYERS];
	uint32_t ordinals[DB_LAYERS];
	uint32_t host_tokens[DB_MAX_ROWS];
	uint32_t host_slots[DB_MAX_ROWS];
	uint32_t host_positions[DB_MAX_ROWS];
	uint32_t host_page_table[DB_MAX_ROWS * DB_PAGES];
	uint32_t *pinned_words;
	uint16_t *boundary,*gather,*peer_slots,*combined;
	uint64_t *request_word,*ready_word;
	uint8_t *wide_w1,*wide_w2;
	uint32_t rows,position,emulate_rounds,wide_experts,rounds_per_step;
	uint32_t distinct_experts[DB_LAYERS];
} DbRig;

__global__ static void DbRequestKernel(uint64_t *request,uint64_t kind)
{
	*(volatile uint64_t *)request = kind;
}

__global__ static void DbGuardKernel(const uint64_t *ready,uint64_t *request)
{
	if ( *(volatile const uint64_t *)ready != 1u )
		*(volatile uint64_t *)request = UINT64_MAX;
}

__global__ static void DbPublishKernel(uint16_t *slot,const uint16_t *local,uint32_t count)
{
	uint32_t index;
	for (index=blockIdx.x * blockDim.x + threadIdx.x; index<count; index+=blockDim.x * gridDim.x)
		slot[index] = local[index];
}

__global__ static void DbCombineKernel(const uint16_t *slots,uint16_t *output,uint32_t count,uint32_t peers)
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

__global__ static void DbFillFp8Kernel(uint8_t *payload,uint64_t bytes,uint32_t seed)
{
	uint64_t index;
	uint32_t hash;
	for (index=(uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index<bytes; index+=(uint64_t)blockDim.x * gridDim.x)
	{
		hash = (uint32_t)index * 2654435761u ^ seed;
		hash ^= hash >> 15u;
		hash *= 2246822519u;
		hash ^= hash >> 13u;
		payload[index] = (uint8_t)((hash & 0x80u) | (0x20u + (hash >> 8u) % 0x18u));
	}
}

static cudaError_t DbWaitNode(cudaStream_t stream,uint64_t *ready)
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

static int DbRound(DbRig *rig,const uint16_t *local)
{
	cudaStream_t stream = rig->fixture.stream;
	uint32_t count = rig->rows * SPARK_GLM52_VHIDDEN;
	rig->rounds_per_step++;
	if ( rig->emulate_rounds == 0u )
		return(0);
	DbRequestKernel<<<1,1,0,stream>>>(rig->request_word,1u);
	if ( DbWaitNode(stream,rig->ready_word) != cudaSuccess )
		return(1);
	DbGuardKernel<<<1,1,0,stream>>>(rig->ready_word,rig->request_word);
	DbPublishKernel<<<rig->rows,DB_THREADS,0,stream>>>(rig->peer_slots,local,count);
	DbRequestKernel<<<1,1,0,stream>>>(rig->request_word,2u);
	if ( DbWaitNode(stream,rig->ready_word) != cudaSuccess )
		return(1);
	DbGuardKernel<<<1,1,0,stream>>>(rig->ready_word,rig->request_word);
	DbCombineKernel<<<(count + DB_THREADS - 1u) / DB_THREADS,DB_THREADS,0,stream>>>(rig->peer_slots,rig->combined,count,DB_PEERS);
	return(cudaPeekAtLastError() == cudaSuccess ? 0 : 1);
}

static void *DbAlloc(uint64_t bytes)
{
	return(SparkGlm52ValAllocZeroed(bytes));
}

static int DbSetup(DbRig *rig)
{
	SparkGlm52ValFixture *fx = &rig->fixture;
	uint64_t w1_bytes = (uint64_t)DB_EXPERTS * (SPARK_GLM52_VW1_ROWS / DB_TP) * SPARK_GLM52_VEXPERT_COLUMNS;
	uint64_t w2_bytes = (uint64_t)DB_EXPERTS * SPARK_GLM52_VW2_ROWS * (SPARK_GLM52_VW2_COLUMNS / DB_TP);
	uint32_t lane,page;
	uint64_t one = 1u;
	if ( SparkGlm52ValFixtureSetup(fx) != 0 )
		return(1);
	fx->split_threshold = DB_SPLIT_THRESHOLD;
	fx->split_partials = (float *)DbAlloc(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(DB_MAX_ROWS,SPARK_GLM52_MODEL_HEAD_COUNT));
	fx->kv_cache = (uint8_t *)DbAlloc((uint64_t)DB_MAX_ROWS * DB_PAGES * SPARK_GLM52_VKV_PAGE_BYTES);
	fx->index_cache = (uint8_t *)DbAlloc((uint64_t)DB_MAX_ROWS * DB_PAGES * SPARK_GLM52_VINDEX_PAGE_BYTES);
	fx->page_table = (uint32_t *)DbAlloc(sizeof(rig->host_page_table));
	fx->context_lengths = (uint32_t *)DbAlloc(DB_MAX_ROWS * sizeof(uint32_t));
	fx->expert_out = (uint16_t *)DbAlloc((uint64_t)DB_MAX_ROWS * SPARK_GLM52_VTOP_K * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	fx->shared_out = (uint16_t *)DbAlloc((uint64_t)DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	fx->router_logits = (float *)DbAlloc((uint64_t)DB_MAX_ROWS * DB_EXPERTS * sizeof(float));
	fx->selection_scores = (float *)DbAlloc((uint64_t)DB_MAX_ROWS * SPARK_GLM52_VALIDATION_DSA_CONTEXT * sizeof(float));
	fx->selected_positions = (uint32_t *)DbAlloc((uint64_t)DB_MAX_ROWS * SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t));
	rig->wide_w1 = (uint8_t *)DbAlloc(w1_bytes);
	rig->wide_w2 = (uint8_t *)DbAlloc(w2_bytes);
	if ( fx->split_partials == 0 || fx->kv_cache == 0 || fx->index_cache == 0 || fx->page_table == 0 || fx->context_lengths == 0 ||
		fx->expert_out == 0 || fx->shared_out == 0 || fx->router_logits == 0 || fx->selection_scores == 0 || fx->selected_positions == 0 ||
		rig->wide_w1 == 0 || rig->wide_w2 == 0 ||
		cudaHostAlloc((void **)&rig->pinned_words,4u * DB_MAX_ROWS * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess ||
		cudaMalloc((void **)&rig->boundary,2u * DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->gather,DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->peer_slots,(uint64_t)DB_PEERS * DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->combined,DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->request_word,sizeof(uint64_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->ready_word,sizeof(uint64_t)) != cudaSuccess ||
		cudaMemset(rig->gather,0,DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(rig->peer_slots,0,(uint64_t)DB_PEERS * DB_MAX_ROWS * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemcpy(rig->ready_word,&one,sizeof(one),cudaMemcpyHostToDevice) != cudaSuccess )
		return(2);
	DbFillFp8Kernel<<<1024,256>>>(rig->wide_w1,w1_bytes,0x1234u);
	DbFillFp8Kernel<<<1024,256>>>(rig->wide_w2,w2_bytes,0x5678u);
	for (lane=0u; lane<DB_MAX_ROWS; lane++)
		for (page=0u; page<DB_PAGES; page++)
			rig->host_page_table[lane * DB_PAGES + page] = lane * DB_PAGES + page;
	if ( cudaMemcpy(fx->page_table,rig->host_page_table,sizeof(rig->host_page_table),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaDeviceSynchronize() != cudaSuccess )
		return(3);
	return(0);
}

static int DbReset(DbRig *rig)
{
	SparkGlm52ValFixture *fx = &rig->fixture;
	float bias[DB_EXPERTS];
	uint32_t expert;
	for (expert=0u; expert<DB_EXPERTS; expert++)
		bias[expert] = rig->wide_experts != 0u ? 0.0f : expert < SPARK_GLM52_VALIDATION_EXPERT_SLOTS ? 4.0f : -4.0f;
	if ( cudaMemcpy(fx->correction_bias,bias,sizeof(bias),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemset(fx->kv_cache,0,(uint64_t)DB_MAX_ROWS * DB_PAGES * SPARK_GLM52_VKV_PAGE_BYTES) != cudaSuccess ||
		cudaMemset(fx->hidden,0,(uint64_t)SPARK_GLM52_VHIDDEN * DB_MAX_ROWS * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fx->residual,0,(uint64_t)SPARK_GLM52_VHIDDEN * DB_MAX_ROWS * sizeof(uint16_t)) != cudaSuccess )
		return(1);
	return(0);
}

static void DbBuild(DbRig *rig)
{
	SparkGlm52ValFixture *fx = &rig->fixture;
	SparkGlm52CudaWave *wave = &fx->wave;
	uint32_t layer,row;
	SparkGlm52ValBuildWave(fx,0u,1u,rig->position);
	if ( rig->wide_experts != 0u )
	{
		fx->weights.expert_up_gate_payload = rig->wide_w1;
		fx->weights.expert_down_payload = rig->wide_w2;
	}
	for (layer=0u; layer<DB_LAYERS; layer++)
	{
		rig->layers[layer] = fx->weights;
		rig->ordinals[layer] = 0u;
	}
	for (row=0u; row<rig->rows; row++)
	{
		rig->pinned_words[row] = row % SPARK_GLM52_VALIDATION_EMBED_ROWS;
		rig->pinned_words[DB_MAX_ROWS + row] = row;
		rig->pinned_words[2u * DB_MAX_ROWS + row] = rig->position;
	}
	wave->layers = rig->layers;
	wave->layer_count = DB_LAYERS;
	wave->index_ordinal_by_local_layer = rig->ordinals;
	wave->host_token_ids = rig->pinned_words;
	wave->host_resident_slots = rig->pinned_words + DB_MAX_ROWS;
	wave->host_positions = rig->pinned_words + 2u * DB_MAX_ROWS;
	wave->row_count = rig->rows;
	wave->resident_sequence_capacity = DB_MAX_ROWS;
	wave->execution_row_capacity = DB_MAX_ROWS;
	wave->pages_per_sequence = DB_PAGES;
	wave->kv_cache = fx->kv_cache;
	wave->index_cache = fx->index_cache;
	wave->page_table = fx->page_table;
	wave->attention_split_partials_f32 = fx->split_partials;
	wave->attention_split_partial_blocks = SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(DB_MAX_ROWS,SPARK_GLM52_MODEL_HEAD_COUNT);
	wave->tp_degree = DB_TP;
	wave->tp_rank = 0u;
	wave->owns_final_head = 0u;
	wave->hidden_output_bf16 = rig->boundary;
	wave->boundary_row_offset = 0u;
	wave->decode_split_context_threshold = DB_SPLIT_THRESHOLD;
	wave->max_sequence_positions = DB_MAX_CONTEXT;
	wave->maximum_context = rig->position + 1u;
	wave->projection_split = 1u;
	fx->slot.projection_gather_bf16 = rig->gather;
}

static int DbWalk(DbRig *rig,uint32_t count_experts)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer,index,distinct;
	uint32_t routes[DB_MAX_ROWS * SPARK_GLM52_VTOP_K];
	uint8_t seen[DB_EXPERTS];
	rig->rounds_per_step = 0u;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 || DbRound(rig,rig->fixture.hidden) != 0 )
		return(1);
	for (layer=0u; layer<DB_LAYERS; layer++)
	{
		if ( SparkGlm52LaunchCudaLayerAttentionProject(wave,layer) != 0 || DbRound(rig,rig->gather) != 0 ||
			SparkGlm52LaunchCudaLayerAttentionCore(wave,layer) != 0 || DbRound(rig,rig->fixture.attention_out) != 0 ||
			SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 )
			return(2);
		if ( count_experts != 0u )
		{
			rig->distinct_experts[layer] = 0u;
			if ( layer >= SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
			{
				if ( cudaMemcpyAsync(routes,rig->fixture.route_expert,(uint64_t)rig->rows * SPARK_GLM52_VTOP_K * sizeof(uint32_t),cudaMemcpyDeviceToHost,rig->fixture.stream) != cudaSuccess ||
					cudaStreamSynchronize(rig->fixture.stream) != cudaSuccess )
					return(3);
				memset(seen,0,sizeof(seen));
				distinct = 0u;
				for (index=0u; index<rig->rows * SPARK_GLM52_VTOP_K; index++)
					if ( routes[index] < DB_EXPERTS && seen[routes[index]]++ == 0u )
						distinct++;
				rig->distinct_experts[layer] = distinct;
			}
		}
		if ( DbRound(rig,rig->fixture.hidden) != 0 )
			return(4);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 || DbRound(rig,rig->fixture.hidden) != 0 )
		return(5);
	return(0);
}

static double DbNowUs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec * 1e6 + (double)now.tv_nsec / 1e3);
}

static double DbMedian(double *values,uint32_t count)
{
	std::sort(values,values + count);
	return(values[count / 2u]);
}

static void DbSlice(uint32_t total,uint32_t *count)
{
	uint32_t granules = total / GLM_PROJECTION_GRANULE, base = granules / DB_TP, extra = granules % DB_TP;
	*count = GLM_PROJECTION_GRANULE * (base + (extra > 0u ? 1u : 0u));
}

static double DbIdealBytes(const DbRig *rig,double *expert_bytes,double *kv_bytes)
{
	const double heads = SPARK_GLM52_MODEL_HEAD_COUNT / DB_TP;
	const double hidden = SPARK_GLM52_VHIDDEN;
	uint32_t q_count,kv_count,layer;
	double attention,dense,routed,index_layer,expert,total = 0.0;
	DbSlice(SPARK_GLM52_VQUERY_A,&q_count);
	DbSlice(SPARK_GLM52_VKV_SLOT_ELEMENTS,&kv_count);
	attention = 2.0 * (hidden + (double)(q_count + kv_count) * hidden + SPARK_GLM52_VQUERY_A + SPARK_GLM52_VLATENT +
		heads * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE) * SPARK_GLM52_VQUERY_A +
		heads * SPARK_GLM52_VLATENT * SPARK_GLM52_VQK_NOPE + heads * SPARK_GLM52_VVALUE_DIM * SPARK_GLM52_VLATENT +
		hidden * heads * SPARK_GLM52_VVALUE_DIM + hidden);
	index_layer = 2.0 * (SPARK_GLM52_VDSA_DIM * hidden + 2.0 * SPARK_GLM52_VDSA_DIM);
	dense = 2.0 * (3.0 * hidden * (SPARK_GLM52_VDENSE_INTER / DB_TP));
	routed = 2.0 * (DB_EXPERTS * hidden + 3.0 * hidden * (SPARK_GLM52_VEXPERT_INTER / DB_TP)) + 4.0 * DB_EXPERTS;
	expert = 3.0 * hidden * (SPARK_GLM52_VEXPERT_INTER / DB_TP) + 4.0 * 3.0 * (SPARK_GLM52_VEXPERT_INTER / DB_TP);
	*expert_bytes = 0.0;
	*kv_bytes = (double)rig->rows * (rig->position + 1u) * SPARK_GLM52_VKV_SLOT_BYTES * DB_LAYERS;
	for (layer=0u; layer<DB_LAYERS; layer++)
	{
		total += attention + (GlmLayerHasFullIndexer(layer) != 0u ? index_layer : 0.0);
		if ( layer < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
			total += dense;
		else
		{
			total += routed;
			*expert_bytes += expert * rig->distinct_experts[layer];
		}
	}
	return(total + *expert_bytes + *kv_bytes);
}

static int DbRun(DbRig *rig,uint32_t graph_mode)
{
	cudaStream_t stream = rig->fixture.stream;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	cudaEvent_t begin,end;
	double gpu[DB_REPLAYS],launch[DB_REPLAYS];
	double t0,ideal,experts,kv,step_ms,mean_distinct = 0.0;
	uint32_t replay,layer,bound,routed_layers = 0u;
	size_t nodes = 0u;
	float elapsed;
	if ( DbReset(rig) != 0 || cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess )
		return(10);
	DbBuild(rig);
	(void)SparkGlm52GraphRegime(rig->position + 1u,DB_SPLIT_THRESHOLD,DB_MAX_CONTEXT,&bound);
	if ( graph_mode != 0u )
	{
		rig->fixture.wave.maximum_context = bound;
		if ( cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
			return(11);
		if ( DbWalk(rig,0u) != 0 )
		{
			(void)cudaStreamEndCapture(stream,&graph);
			return(12);
		}
		if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess || graph == 0 || cudaGraphGetNodes(graph,0,&nodes) != cudaSuccess ||
			cudaGraphInstantiate(&exec,graph,0) != cudaSuccess || cudaGraphUpload(exec,stream) != cudaSuccess )
			return(13);
		(void)cudaGraphDestroy(graph);
		rig->fixture.wave.maximum_context = rig->position + 1u;
	}
	for (replay=0u; replay<DB_WARM + DB_REPLAYS; replay++)
	{
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			return(14);
		t0 = DbNowUs();
		if ( cudaEventRecord(begin,stream) != cudaSuccess )
			return(15);
		if ( graph_mode != 0u ? cudaGraphLaunch(exec,stream) != cudaSuccess : DbWalk(rig,0u) != 0 )
			return(16);
		if ( replay >= DB_WARM )
			launch[replay - DB_WARM] = DbNowUs() - t0;
		if ( cudaEventRecord(end,stream) != cudaSuccess || cudaEventSynchronize(end) != cudaSuccess || cudaEventElapsedTime(&elapsed,begin,end) != cudaSuccess )
			return(17);
		if ( replay >= DB_WARM )
			gpu[replay - DB_WARM] = elapsed;
	}
	if ( exec != 0 )
		(void)cudaGraphExecDestroy(exec);
	if ( cudaStreamSynchronize(stream) != cudaSuccess || DbWalk(rig,1u) != 0 || cudaStreamSynchronize(stream) != cudaSuccess ||
		SparkGlm52ValCheckAccessError(&rig->fixture) != 0 )
		return(18);
	for (layer=SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER; layer<DB_LAYERS; layer++, routed_layers++)
		mean_distinct += rig->distinct_experts[layer];
	mean_distinct /= routed_layers;
	ideal = DbIdealBytes(rig,&experts,&kv);
	step_ms = DbMedian(gpu,DB_REPLAYS);
	printf("glm52_decode_bench tp=%u rows=%u position=%u experts=%s distinct_mean=%.1f rounds=%s collectives=%u mode=%s nodes=%zu launch_us=%.1f gpu_ms=%.3f gpu_min_ms=%.3f\n",
		DB_TP,rig->rows,rig->position,rig->wide_experts != 0u ? "wide256" : "fixed8",mean_distinct,rig->emulate_rounds != 0u ? "emulated" : "none",
		rig->rounds_per_step,graph_mode != 0u ? "graph" : "linear",nodes,DbMedian(launch,DB_REPLAYS),step_ms,gpu[0]);
	printf("glm52_decode_bench roofline @B=%u (single-GPU bench step, rank-local TP16, head excluded): memory %.1f%% (ideal %.3f GB = spine %.3f + experts %.3f + kv %.4f; floor %.2f ms at %.0f GB/s; ceiling %.1f tok/s vs measured %.1f tok/s)\n",
		rig->rows,100.0 * ideal / (step_ms * 1e-3) / (DB_PEAK_GBPS * 1e9),ideal / 1e9,(ideal - experts - kv) / 1e9,experts / 1e9,kv / 1e9,
		ideal / (DB_PEAK_GBPS * 1e9) * 1e3,DB_PEAK_GBPS,rig->rows * DB_PEAK_GBPS * 1e9 / ideal,rig->rows * 1e3 / step_ms);
	(void)cudaEventDestroy(begin);
	(void)cudaEventDestroy(end);
	return(0);
}

int main(int argc,char **argv)
{
	DbRig *rig;
	int status;
	cudaDeviceProp properties;
	if ( argc != 6 )
	{
		fprintf(stderr,"usage: glm52_decode_bench <rows 1..16> <position 1..2047> <rounds 0|1> <graph 0|1> <experts 0=fixed8|1=wide256>\n");
		return(2);
	}
	rig = (DbRig *)calloc(1u,sizeof(*rig));
	if ( rig == 0 )
		return(1);
	rig->rows = (uint32_t)atoi(argv[1]);
	rig->position = (uint32_t)atoi(argv[2]);
	rig->emulate_rounds = (uint32_t)atoi(argv[3]);
	rig->wide_experts = (uint32_t)atoi(argv[5]);
	if ( rig->rows == 0u || rig->rows > DB_MAX_ROWS || rig->position == 0u || rig->position >= DB_MAX_CONTEXT )
	{
		fprintf(stderr,"glm52_decode_bench: rows must be 1..%u and position 1..%u\n",DB_MAX_ROWS,DB_MAX_CONTEXT - 1u);
		return(2);
	}
	if ( cudaGetDeviceProperties(&properties,0) == cudaSuccess )
		printf("glm52_decode_bench device=%s sms=%d l2_bytes=%d\n",properties.name,properties.multiProcessorCount,properties.l2CacheSize);
	status = DbSetup(rig);
	if ( status == 0 )
		status = DbRun(rig,(uint32_t)atoi(argv[4]));
	if ( status != 0 )
		fprintf(stderr,"glm52_decode_bench FAIL status=%d cuda=%s\n",status,cudaGetErrorString(cudaGetLastError()));
	printf("glm52_decode_bench %s\n",status == 0 ? "PASS" : "FAIL");
	return(status == 0 ? 0 : 1);
}
