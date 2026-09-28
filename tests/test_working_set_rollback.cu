#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_step_verdict.h"
#include "sparkpipe/spark_expert_working_set.h"
#include "inference/kernels/expert_cover.cuh"
#include "inference/kernels/state_snapshot.cuh"

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_LAUNCH 1
#define SPARK_FAMILY(name) TestWs##name
#include "sparkpipe/family/glm/spark_glm_head_maxloc.cuh"

#define RANKS 16u
#define LAYERS 6u
#define EXPERTS 40u
#define TOPK 2u
#define HIDDEN 64u
#define VOCAB_SHARD 32u
#define VOCAB (RANKS * VOCAB_SHARD)
#define ROWS 2u
#define STATE_ROWS 4u
#define PACK_STRIDE 64u
#define RING_CAPACITY 64u
#define RING_WORDS (SPARK_STEP_MISS_ENTRIES + RING_CAPACITY)
#define STEPS 32u
#define COVER_STRIDE ((EXPERTS + 31u) / 32u)
#define LONE_RANK 5u

#define CHECK(condition) do { if ( !(condition) ) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#condition); exit(1); } } while (0)
#define CUDA_CHECK(call) do { cudaError_t check_error = (call); if ( check_error != cudaSuccess ) { fprintf(stderr,"FAIL %s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(check_error)); exit(1); } } while (0)

typedef struct Rank
{
	float *hidden;
	float *partial;
	float *state;
	float *decay;
	float *pool;
	float *head_scores;
	uint32_t *head_tokens;
	uint64_t *maxloc;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *cover;
	uint32_t *ring_host;
	uint32_t *ring_device;
	uint32_t *tokens_host;
	uint32_t *tokens_device;
	SparkStateSpan *spans;
	uint8_t *snapshot;
	float *master;
	SparkExpertWorkingSet set;
} Rank;

typedef struct World
{
	Rank ranks[RANKS];
	float **partials;
	float **hiddens;
	uint64_t **maxlocs;
	uint32_t *input_tokens;
	uint32_t *state_index;
	uint32_t partial;
	cudaGraphExec_t exec;
	size_t nodes;
} World;

static float *EMBED;
static float *ROUTER;

static uint64_t Mix(uint64_t value)
{
	value += UINT64_C(0x9e3779b97f4a7c15);
	value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
	value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
	return(value ^ (value >> 31u));
}

static float Uniform(uint64_t seed)
{
	return((float)((Mix(seed) >> 40u) & 0xffffffu) / 8388608.0f - 1.0f);
}

static __global__ void EmbedKernel(float *hidden,const float *embed,const uint32_t *tokens)
{
	uint32_t row = blockIdx.x,d = threadIdx.x;
	hidden[row * HIDDEN + d] = embed[(uint64_t)tokens[row] * HIDDEN + d];
}

static __global__ void RecurrentKernel(float *state,const float *decay,const float *hidden,float *partial,const uint32_t *state_index,uint32_t layer)
{
	uint32_t row = blockIdx.x,d = threadIdx.x;
	float *cell = state + ((uint64_t)layer * STATE_ROWS + state_index[row]) * HIDDEN + d;
	*cell = decay[layer * HIDDEN + d] * *cell + hidden[row * HIDDEN + d];
	partial[row * HIDDEN + d] = 0.0625f * tanhf(*cell);
}

static __global__ void AllReduceSumKernel(float *const *partials,float *const *hiddens)
{
	uint32_t index = blockIdx.x * HIDDEN + threadIdx.x,rank;
	float sum = 0.0f;
	for (rank=0u; rank<RANKS; rank++)
		sum += partials[rank][index];
	for (rank=0u; rank<RANKS; rank++)
		hiddens[rank][index] += sum;
}

static __global__ void RouterKernel(const float *hidden,const float *router,uint32_t layer,uint32_t *route_expert,float *route_weight)
{
	uint32_t row = threadIdx.x,expert,d,best[TOPK],k,j;
	float logit,top[TOPK],norm;
	if ( row >= ROWS )
		return;
	for (k=0u; k<TOPK; k++)
	{
		top[k] = -INFINITY;
		best[k] = 0u;
	}
	for (expert=0u; expert<EXPERTS; expert++)
	{
		logit = 0.0f;
		for (d=0u; d<HIDDEN; d++)
			logit += hidden[row * HIDDEN + d] * router[((uint64_t)layer * EXPERTS + expert) * HIDDEN + d];
		for (k=0u; k<TOPK; k++)
			if ( logit > top[k] )
			{
				for (j=TOPK - 1u; j>k; j--)
				{
					top[j] = top[j - 1u];
					best[j] = best[j - 1u];
				}
				top[k] = logit;
				best[k] = expert;
				break;
			}
	}
	norm = 0.0f;
	for (k=0u; k<TOPK; k++)
		norm += expf(top[k] - top[0]);
	for (k=0u; k<TOPK; k++)
	{
		route_expert[row * TOPK + k] = best[k];
		route_weight[row * TOPK + k] = expf(top[k] - top[0]) / norm;
	}
}

static __global__ void ExpertKernel(const float *hidden,const float *pool,const uint32_t *route_expert,const float *route_weight,uint32_t layer,float *partial)
{
	uint32_t row = blockIdx.x,d = threadIdx.x,k;
	float sum = 0.0f,x = tanhf(hidden[row * HIDDEN + d]);
	for (k=0u; k<TOPK; k++)
		sum += route_weight[row * TOPK + k] * pool[((uint64_t)layer * EXPERTS + route_expert[row * TOPK + k]) * HIDDEN + d] * x;
	partial[row * HIDDEN + d] = 0.125f * sum;
}

static __global__ void HeadKernel(const float *hidden,const float *embed,uint32_t rank,float *scores,uint32_t *tokens)
{
	uint32_t row = threadIdx.x,token,d,best = 0u;
	float score,top = -INFINITY;
	if ( row >= ROWS )
		return;
	for (token=0u; token<VOCAB_SHARD; token++)
	{
		score = 0.0f;
		for (d=0u; d<HIDDEN; d++)
			score += hidden[row * HIDDEN + d] * embed[((uint64_t)rank * VOCAB_SHARD + token) * HIDDEN + d];
		if ( score > top )
		{
			top = score;
			best = token;
		}
	}
	scores[row] = top;
	tokens[row] = best;
}

static __global__ void AllReduceMaxKernel(uint64_t *const *maxlocs)
{
	uint32_t row = threadIdx.x,rank;
	uint64_t value = 0u;
	if ( row >= ROWS )
		return;
	for (rank=0u; rank<RANKS; rank++)
		value = maxlocs[rank][row] > value ? maxlocs[rank][row] : value;
	for (rank=0u; rank<RANKS; rank++)
		maxlocs[rank][row] = value;
}

static SparkStatus AcquireExperts(void *context,const uint32_t *keys,uint32_t count)
{
	Rank *rank = (Rank *)context;
	uint32_t index,layer,expert;
	uint64_t offset;
	for (index=0u; index<count; index++)
	{
		layer = keys[index] / PACK_STRIDE;
		expert = keys[index] % PACK_STRIDE;
		offset = ((uint64_t)layer * EXPERTS + expert) * HIDDEN;
		if ( cudaMemcpy(rank->pool + offset,rank->master + offset,HIDDEN * sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void UploadCover(Rank *rank)
{
	CUDA_CHECK(cudaMemcpy(rank->cover,rank->set.cover,SparkExpertWorkingSetCoverBytes(&rank->set),cudaMemcpyHostToDevice));
}

static void RankCreate(Rank *rank,uint32_t index,uint32_t partial)
{
	float *decay = (float *)malloc(LAYERS * HIDDEN * sizeof(float));
	float *state = (float *)malloc((uint64_t)LAYERS * STATE_ROWS * HIDDEN * sizeof(float));
	float *nan_pool = (float *)malloc((uint64_t)LAYERS * EXPERTS * HIDDEN * sizeof(float));
	SparkStateSpan spans[LAYERS];
	uint64_t count = (uint64_t)LAYERS * EXPERTS * HIDDEN,item,snapshot_bytes;
	uint32_t layer;
	memset(rank,0,sizeof(*rank));
	CHECK(decay != 0 && state != 0 && nan_pool != 0);
	rank->master = (float *)malloc(count * sizeof(float));
	CHECK(rank->master != 0);
	for (item=0u; item<count; item++)
	{
		rank->master[item] = Uniform(UINT64_C(0x1000000) * (index + 1u) + item);
		nan_pool[item] = __builtin_nanf("");
	}
	for (item=0u; item<(uint64_t)LAYERS * HIDDEN; item++)
		decay[item] = 0.7f + 0.25f * Uniform(UINT64_C(0x5000000) * (index + 1u) + item);
	for (item=0u; item<(uint64_t)LAYERS * STATE_ROWS * HIDDEN; item++)
		state[item] = 0.5f * Uniform(UINT64_C(0x9000000) * (index + 1u) + item);
	CUDA_CHECK(cudaMalloc((void **)&rank->hidden,ROWS * HIDDEN * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->partial,ROWS * HIDDEN * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->state,(uint64_t)LAYERS * STATE_ROWS * HIDDEN * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->decay,LAYERS * HIDDEN * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->pool,count * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->head_scores,ROWS * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->head_tokens,ROWS * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&rank->maxloc,ROWS * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void **)&rank->route_expert,ROWS * TOPK * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&rank->route_weight,ROWS * TOPK * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&rank->tokens_device,ROWS * sizeof(uint32_t)));
	CUDA_CHECK(cudaHostAlloc((void **)&rank->tokens_host,ROWS * sizeof(uint32_t),cudaHostAllocDefault));
	CUDA_CHECK(cudaMemcpy(rank->decay,decay,LAYERS * HIDDEN * sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(rank->state,state,(uint64_t)LAYERS * STATE_ROWS * HIDDEN * sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(rank->pool,partial != 0u ? nan_pool : rank->master,count * sizeof(float),cudaMemcpyHostToDevice));
	if ( partial != 0u )
	{
		CUDA_CHECK(cudaMalloc((void **)&rank->cover,(uint64_t)LAYERS * COVER_STRIDE * sizeof(uint32_t)));
		CUDA_CHECK(cudaHostAlloc((void **)&rank->ring_host,RING_WORDS * sizeof(uint32_t),cudaHostAllocMapped));
		memset(rank->ring_host,0,RING_WORDS * sizeof(uint32_t));
		CUDA_CHECK(cudaHostGetDevicePointer((void **)&rank->ring_device,rank->ring_host,0));
		for (layer=0u; layer<LAYERS; layer++)
		{
			spans[layer].base = (uint8_t *)(rank->state + (uint64_t)layer * STATE_ROWS * HIDDEN);
			spans[layer].row_stride = HIDDEN * sizeof(float);
			spans[layer].row_bytes = HIDDEN * sizeof(float);
			spans[layer].state_rows = STATE_ROWS;
		}
		snapshot_bytes = SparkStateSpansLayout(spans,LAYERS,ROWS,0);
		CHECK(snapshot_bytes == (uint64_t)LAYERS * ROWS * HIDDEN * sizeof(float));
		CUDA_CHECK(cudaMalloc((void **)&rank->spans,sizeof(spans)));
		CUDA_CHECK(cudaMemcpy(rank->spans,spans,sizeof(spans),cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc((void **)&rank->snapshot,snapshot_bytes));
		CHECK(SparkExpertWorkingSetCreate(&rank->set,LAYERS,EXPERTS,PACK_STRIDE,LAYERS * EXPERTS,AcquireExperts,rank) == SPARK_STATUS_OK);
	}
	free(decay);
	free(state);
	free(nan_pool);
}

static void WorldSeed(World *world,uint32_t lone)
{
	uint32_t anchors[LAYERS],extra[LAYERS * EXPERTS],layer,rank,count = 0u,missing = 0u,index;
	for (layer=0u; layer<LAYERS; layer++)
	{
		anchors[layer] = layer * PACK_STRIDE + (layer * 7u + 3u) % EXPERTS;
		for (index=0u; index<(lone != 0u ? EXPERTS : 6u); index++)
			extra[count++] = layer * PACK_STRIDE + (lone != 0u ? index : (uint32_t)(Mix(layer * 97u + index) % EXPERTS));
	}
	for (rank=0u; rank<RANKS; rank++)
	{
		CHECK(SparkExpertWorkingSetCheckAnchors(&world->ranks[rank].set,0u,LAYERS,&missing) == SPARK_STATUS_UNSUPPORTED && missing == 0u);
		CHECK(SparkExpertWorkingSetAdd(&world->ranks[rank].set,anchors,LAYERS) == SPARK_STATUS_OK);
		if ( rank != LONE_RANK )
			CHECK(SparkExpertWorkingSetAdd(&world->ranks[rank].set,extra,count) == SPARK_STATUS_OK);
		CHECK(SparkExpertWorkingSetCheckAnchors(&world->ranks[rank].set,0u,LAYERS,&missing) == SPARK_STATUS_OK);
		UploadCover(&world->ranks[rank]);
	}
}

static void WorldLaunch(World *world,cudaStream_t stream)
{
	uint32_t layer,rank;
	Rank *item;
	for (rank=0u; rank<RANKS; rank++)
	{
		item = &world->ranks[rank];
		if ( world->partial != 0u )
		{
			CUDA_CHECK(cudaMemsetAsync(item->ring_device,0,RING_WORDS * sizeof(uint32_t),stream));
			CUDA_CHECK(LmStateSpansCopy(stream,item->spans,LAYERS,HIDDEN * sizeof(float) / SPARK_STATE_SPAN_ALIGN,item->snapshot,world->state_index,ROWS,0u));
		}
		EmbedKernel<<<ROWS,HIDDEN,0,stream>>>(item->hidden,EMBED,world->input_tokens);
	}
	for (layer=0u; layer<LAYERS; layer++)
	{
		for (rank=0u; rank<RANKS; rank++)
			RecurrentKernel<<<ROWS,HIDDEN,0,stream>>>(world->ranks[rank].state,world->ranks[rank].decay,world->ranks[rank].hidden,world->ranks[rank].partial,world->state_index,layer);
		AllReduceSumKernel<<<ROWS,HIDDEN,0,stream>>>(world->partials,world->hiddens);
		for (rank=0u; rank<RANKS; rank++)
		{
			item = &world->ranks[rank];
			RouterKernel<<<1,32,0,stream>>>(item->hidden,ROUTER,layer,item->route_expert,item->route_weight);
			if ( world->partial != 0u )
				LmExpertCoverKernel<<<1,32,0,stream>>>(item->route_expert,item->cover,COVER_STRIDE,EXPERTS,layer,PACK_STRIDE,RING_CAPACITY,ROWS * TOPK,0,item->ring_device);
			ExpertKernel<<<ROWS,HIDDEN,0,stream>>>(item->hidden,item->pool,item->route_expert,item->route_weight,layer,item->partial);
		}
		AllReduceSumKernel<<<ROWS,HIDDEN,0,stream>>>(world->partials,world->hiddens);
	}
	for (rank=0u; rank<RANKS; rank++)
	{
		item = &world->ranks[rank];
		HeadKernel<<<1,32,0,stream>>>(item->hidden,EMBED,rank,item->head_scores,item->head_tokens);
		TestWsHeadMaxlocPackKernel<<<1,32,0,stream>>>(item->head_scores,item->head_tokens,item->maxloc,ROWS,rank * VOCAB_SHARD);
		if ( world->partial != 0u )
			LmHeadMissPoisonKernel<<<1,32,0,stream>>>(item->ring_device,item->maxloc,ROWS);
	}
	AllReduceMaxKernel<<<1,32,0,stream>>>(world->maxlocs);
	for (rank=0u; rank<RANKS; rank++)
	{
		item = &world->ranks[rank];
		LmHeadMaxlocUnpackPoisonKernel<<<1,32,0,stream>>>(item->maxloc,item->tokens_device,ROWS);
		CUDA_CHECK(cudaMemcpyAsync(item->tokens_host,item->tokens_device,ROWS * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
	}
	CUDA_CHECK(cudaGetLastError());
}

static void WorldCreate(World *world,uint32_t partial,uint32_t lone,cudaStream_t stream)
{
	uint32_t rank,state_index[ROWS] = {3u,1u};
	float *partials[RANKS],*hiddens[RANKS];
	uint64_t *maxlocs[RANKS];
	cudaGraph_t graph;
	memset(world,0,sizeof(*world));
	world->partial = partial;
	for (rank=0u; rank<RANKS; rank++)
	{
		RankCreate(&world->ranks[rank],rank,partial);
		partials[rank] = world->ranks[rank].partial;
		hiddens[rank] = world->ranks[rank].hidden;
		maxlocs[rank] = world->ranks[rank].maxloc;
	}
	CUDA_CHECK(cudaMalloc((void **)&world->partials,sizeof(partials)));
	CUDA_CHECK(cudaMalloc((void **)&world->hiddens,sizeof(hiddens)));
	CUDA_CHECK(cudaMalloc((void **)&world->maxlocs,sizeof(maxlocs)));
	CUDA_CHECK(cudaMalloc((void **)&world->input_tokens,ROWS * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&world->state_index,ROWS * sizeof(uint32_t)));
	CUDA_CHECK(cudaMemcpy(world->partials,partials,sizeof(partials),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(world->hiddens,hiddens,sizeof(hiddens),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(world->maxlocs,maxlocs,sizeof(maxlocs),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(world->state_index,state_index,sizeof(state_index),cudaMemcpyHostToDevice));
	if ( partial != 0u )
		WorldSeed(world,lone);
	CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
	WorldLaunch(world,stream);
	CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
	CUDA_CHECK(cudaGraphGetNodes(graph,0,&world->nodes));
	CUDA_CHECK(cudaGraphInstantiate(&world->exec,graph,0));
	CUDA_CHECK(cudaGraphDestroy(graph));
}

typedef struct RunStats
{
	uint64_t graph_launches;
	uint64_t eager_steps;
	uint64_t rollback_steps;
	uint64_t local;
	uint64_t remote;
	uint64_t lone_misses;
	uint64_t grown;
	double seconds;
} RunStats;

static double Now(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec + (double)now.tv_nsec * 1e-9);
}

static void RecoverRank(World *world,Rank *rank,SparkStepVerdict verdict,cudaStream_t stream,RunStats *stats)
{
	uint32_t keys[RING_CAPACITY];
	SparkExpertMissHarvest harvest;
	uint64_t before = rank->set.grown;
	CUDA_CHECK(LmStateSpansCopy(stream,rank->spans,LAYERS,HIDDEN * sizeof(float) / SPARK_STATE_SPAN_ALIGN,rank->snapshot,world->state_index,ROWS,1u));
	CHECK(SparkExpertWorkingSetHarvest(&rank->set,rank->ring_host,RING_CAPACITY,keys,RING_CAPACITY,&harvest) == SPARK_STATUS_OK);
	CHECK((verdict == SPARK_STEP_VERDICT_ROLLBACK_LOCAL) == (harvest.key_count != 0u));
	if ( harvest.key_count != 0u )
	{
		CHECK(SparkExpertWorkingSetAdd(&rank->set,keys,harvest.key_count) == SPARK_STATUS_OK);
		CHECK(rank->set.grown > before);
		UploadCover(rank);
	}
	stats->grown += rank->set.grown - before;
}

static void EagerStep(World *world,cudaStream_t stream)
{
	uint32_t routes[ROWS * TOPK],keys[ROWS * TOPK],layer,rank,item;
	Rank *entry;
	for (rank=0u; rank<RANKS; rank++)
		EmbedKernel<<<ROWS,HIDDEN,0,stream>>>(world->ranks[rank].hidden,EMBED,world->input_tokens);
	for (layer=0u; layer<LAYERS; layer++)
	{
		for (rank=0u; rank<RANKS; rank++)
			RecurrentKernel<<<ROWS,HIDDEN,0,stream>>>(world->ranks[rank].state,world->ranks[rank].decay,world->ranks[rank].hidden,world->ranks[rank].partial,world->state_index,layer);
		AllReduceSumKernel<<<ROWS,HIDDEN,0,stream>>>(world->partials,world->hiddens);
		for (rank=0u; rank<RANKS; rank++)
		{
			entry = &world->ranks[rank];
			RouterKernel<<<1,32,0,stream>>>(entry->hidden,ROUTER,layer,entry->route_expert,entry->route_weight);
			CUDA_CHECK(cudaMemcpyAsync(routes,entry->route_expert,sizeof(routes),cudaMemcpyDeviceToHost,stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			for (item=0u; item<ROWS * TOPK; item++)
			{
				CHECK(routes[item] < EXPERTS);
				keys[item] = layer * PACK_STRIDE + routes[item];
			}
			CHECK(AcquireExperts(entry,keys,ROWS * TOPK) == SPARK_STATUS_OK);
			ExpertKernel<<<ROWS,HIDDEN,0,stream>>>(entry->hidden,entry->pool,entry->route_expert,entry->route_weight,layer,entry->partial);
		}
		AllReduceSumKernel<<<ROWS,HIDDEN,0,stream>>>(world->partials,world->hiddens);
	}
	for (rank=0u; rank<RANKS; rank++)
	{
		entry = &world->ranks[rank];
		HeadKernel<<<1,32,0,stream>>>(entry->hidden,EMBED,rank,entry->head_scores,entry->head_tokens);
		TestWsHeadMaxlocPackKernel<<<1,32,0,stream>>>(entry->head_scores,entry->head_tokens,entry->maxloc,ROWS,rank * VOCAB_SHARD);
	}
	AllReduceMaxKernel<<<1,32,0,stream>>>(world->maxlocs);
	for (rank=0u; rank<RANKS; rank++)
	{
		entry = &world->ranks[rank];
		LmHeadMaxlocUnpackPoisonKernel<<<1,32,0,stream>>>(entry->maxloc,entry->tokens_device,ROWS);
		CUDA_CHECK(cudaMemcpyAsync(entry->tokens_host,entry->tokens_device,ROWS * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void RunChain(World *world,const uint32_t *prompt,uint32_t *tokens,uint32_t limit,cudaStream_t stream,RunStats *stats)
{
	SparkStepReplay replay[RANKS];
	SparkStepVerdict verdict[RANKS];
	SparkStepAction action[RANKS];
	uint32_t input[ROWS],step,rank,row,locals;
	double start = Now();
	memset(replay,0,sizeof(replay));
	for (rank=0u; rank<RANKS; rank++)
		replay[rank].limit = limit;
	memcpy(input,prompt,sizeof(input));
	for (step=0u; step<STEPS; step++)
	{
		CUDA_CHECK(cudaMemcpyAsync(world->input_tokens,input,sizeof(input),cudaMemcpyHostToDevice,stream));
		for (;;)
		{
			CUDA_CHECK(cudaGraphLaunch(world->exec,stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			stats->graph_launches++;
			locals = 0u;
			for (rank=0u; rank<RANKS; rank++)
			{
				const volatile uint32_t *ring = world->ranks[rank].ring_host;
				verdict[rank] = SparkStepVerdictClassify(world->ranks[rank].tokens_host,ROWS,ring != 0 ? ring[SPARK_STEP_MISS_FLAG] : 0u);
				action[rank] = SparkStepReplayNext(&replay[rank],verdict[rank]);
				CHECK(action[rank] == action[0]);
				CHECK(action[rank] == SPARK_STEP_ACTION_COMMIT || action[rank] == SPARK_STEP_ACTION_REPLAY || action[rank] == SPARK_STEP_ACTION_EXHAUSTED);
				locals += verdict[rank] == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
				stats->local += verdict[rank] == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
				stats->remote += verdict[rank] == SPARK_STEP_VERDICT_ROLLBACK_REMOTE ? 1u : 0u;
			}
			if ( action[0] == SPARK_STEP_ACTION_COMMIT )
				break;
			CHECK(locals != 0u);
			stats->lone_misses += locals < RANKS && verdict[LONE_RANK] == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
			stats->rollback_steps++;
			for (rank=0u; rank<RANKS; rank++)
				RecoverRank(world,&world->ranks[rank],verdict[rank],stream,stats);
			if ( action[0] == SPARK_STEP_ACTION_EXHAUSTED )
			{
				for (rank=0u; rank<RANKS; rank++)
					CHECK(replay[rank].attempts == 0u && replay[rank].exhausted == replay[0].exhausted);
				EagerStep(world,stream);
				stats->eager_steps++;
				break;
			}
		}
		for (rank=0u; rank<RANKS; rank++)
			for (row=0u; row<ROWS; row++)
				CHECK(world->ranks[rank].tokens_host[row] == world->ranks[0].tokens_host[row] && world->ranks[rank].tokens_host[row] < VOCAB);
		for (row=0u; row<ROWS; row++)
		{
			tokens[step * ROWS + row] = world->ranks[0].tokens_host[row];
			input[row] = world->ranks[0].tokens_host[row];
		}
	}
	stats->seconds = Now() - start;
}

static void CompareState(const World *left,const World *right,uint32_t check_touched)
{
	uint64_t count = (uint64_t)LAYERS * STATE_ROWS * HIDDEN;
	float *a = (float *)malloc(count * sizeof(float)),*b = (float *)malloc(count * sizeof(float));
	uint32_t rank;
	CHECK(a != 0 && b != 0);
	for (rank=0u; rank<RANKS; rank++)
	{
		CUDA_CHECK(cudaMemcpy(a,left->ranks[rank].state,count * sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(b,right->ranks[rank].state,count * sizeof(float),cudaMemcpyDeviceToHost));
		CHECK((memcmp(a,b,count * sizeof(float)) == 0) == (check_touched == 0u));
	}
	free(a);
	free(b);
}

static void ResetState(World *world,const World *pristine)
{
	uint32_t rank;
	for (rank=0u; rank<RANKS; rank++)
		CUDA_CHECK(cudaMemcpy(world->ranks[rank].state,pristine->ranks[rank].state,(uint64_t)LAYERS * STATE_ROWS * HIDDEN * sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaDeviceSynchronize());
}

static void CheckUntouchedSlots(const World *world)
{
	float row[HIDDEN];
	uint32_t rank,layer,slot;
	for (rank=0u; rank<RANKS; rank++)
		for (layer=0u; layer<LAYERS; layer++)
			for (slot=0u; slot<STATE_ROWS; slot+=2u)
			{
				uint32_t d;
				CUDA_CHECK(cudaMemcpy(row,world->ranks[rank].state + ((uint64_t)layer * STATE_ROWS + slot) * HIDDEN,sizeof(row),cudaMemcpyDeviceToHost));
				for (d=0u; d<HIDDEN; d++)
					CHECK(row[d] == 0.5f * Uniform(UINT64_C(0x9000000) * (rank + 1u) + ((uint64_t)layer * STATE_ROWS + slot) * HIDDEN + d));
			}
}

int main(void)
{
	static uint32_t reference[STEPS * ROWS],repeat[STEPS * ROWS],cold[STEPS * ROWS],warm[STEPS * ROWS],single[STEPS * ROWS],eager[STEPS * ROWS];
	uint32_t prompt[ROWS] = {17u,301u};
	uint64_t index,embed_count = (uint64_t)VOCAB * HIDDEN,router_count = (uint64_t)LAYERS * EXPERTS * HIDDEN;
	float *host = (float *)malloc((embed_count > router_count ? embed_count : router_count) * sizeof(float));
	World *full = (World *)calloc(1u,sizeof(World)),*pristine = (World *)calloc(1u,sizeof(World)),*partial = (World *)calloc(1u,sizeof(World)),*lone = (World *)calloc(1u,sizeof(World)),*capped = (World *)calloc(1u,sizeof(World));
	RunStats full_stats,repeat_stats,cold_stats,warm_stats,lone_stats,capped_stats;
	cudaStream_t stream;
	uint32_t rank;
	CHECK(host != 0 && full != 0 && pristine != 0 && partial != 0 && lone != 0 && capped != 0);
	memset(&full_stats,0,sizeof(full_stats));
	memset(&repeat_stats,0,sizeof(repeat_stats));
	memset(&cold_stats,0,sizeof(cold_stats));
	memset(&warm_stats,0,sizeof(warm_stats));
	memset(&lone_stats,0,sizeof(lone_stats));
	memset(&capped_stats,0,sizeof(capped_stats));
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	for (index=0u; index<embed_count; index++)
		host[index] = Uniform(UINT64_C(0x20000000) + index);
	CUDA_CHECK(cudaMalloc((void **)&EMBED,embed_count * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(EMBED,host,embed_count * sizeof(float),cudaMemcpyHostToDevice));
	for (index=0u; index<router_count; index++)
		host[index] = 0.5f * Uniform(UINT64_C(0x30000000) + index);
	CUDA_CHECK(cudaMalloc((void **)&ROUTER,router_count * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(ROUTER,host,router_count * sizeof(float),cudaMemcpyHostToDevice));
	WorldCreate(full,0u,0u,stream);
	WorldCreate(pristine,0u,0u,stream);
	WorldCreate(partial,1u,0u,stream);
	WorldCreate(lone,1u,1u,stream);
	WorldCreate(capped,1u,0u,stream);
	CompareState(full,partial,0u);
	RunChain(full,prompt,reference,LAYERS + 1u,stream,&full_stats);
	CompareState(full,pristine,1u);
	CheckUntouchedSlots(full);
	ResetState(full,pristine);
	RunChain(full,prompt,repeat,LAYERS + 1u,stream,&repeat_stats);
	CHECK(memcmp(reference,repeat,sizeof(reference)) == 0);
	printf("ok all-pinned graph is deterministic: %u steps x %u rows, nodes=%zu\n",STEPS,ROWS,full->nodes);
	RunChain(partial,prompt,cold,LAYERS + 1u,stream,&cold_stats);
	CHECK(memcmp(reference,cold,sizeof(reference)) == 0);
	CompareState(full,partial,0u);
	CheckUntouchedSlots(partial);
	printf("ok cold working set: tokens and recurrent state bit-identical to all-pinned; launches=%llu rollbacks=%llu local=%llu remote=%llu lone-rank=%llu grown=%llu nodes=%zu\n",
		(unsigned long long)cold_stats.graph_launches,(unsigned long long)cold_stats.rollback_steps,(unsigned long long)cold_stats.local,(unsigned long long)cold_stats.remote,(unsigned long long)cold_stats.lone_misses,(unsigned long long)cold_stats.grown,partial->nodes);
	CHECK(cold_stats.rollback_steps != 0u && cold_stats.local != 0u && cold_stats.graph_launches == STEPS + cold_stats.rollback_steps);
	for (rank=0u; rank<RANKS; rank++)
		CHECK(partial->ranks[rank].set.key_count < LAYERS * EXPERTS);
	ResetState(partial,pristine);
	ResetState(full,pristine);
	RunChain(full,prompt,repeat,LAYERS + 1u,stream,&repeat_stats);
	RunChain(partial,prompt,warm,LAYERS + 1u,stream,&warm_stats);
	CHECK(memcmp(reference,warm,sizeof(reference)) == 0 && warm_stats.rollback_steps == 0u && warm_stats.graph_launches == STEPS);
	CompareState(full,partial,0u);
	printf("ok warm working set: zero rollbacks, bit-identical; keys held rank0=%u rank%u=%u of %u\n",partial->ranks[0].set.key_count,LONE_RANK,partial->ranks[LONE_RANK].set.key_count,LAYERS * EXPERTS);
	RunChain(lone,prompt,single,LAYERS + 1u,stream,&lone_stats);
	CHECK(memcmp(reference,single,sizeof(reference)) == 0);
	CompareState(full,lone,0u);
	CHECK(lone_stats.rollback_steps != 0u && lone_stats.local == lone_stats.rollback_steps && lone_stats.lone_misses == lone_stats.rollback_steps && lone_stats.remote == (RANKS - 1u) * lone_stats.rollback_steps);
	printf("ok single-rank miss: rank %u alone lacked experts; every rollback was local on rank %u and remote on the other %u ranks; rollbacks=%llu grown=%llu, bit-identical\n",LONE_RANK,LONE_RANK,RANKS - 1u,(unsigned long long)lone_stats.rollback_steps,(unsigned long long)lone_stats.grown);
	ResetState(full,pristine);
	RunChain(full,prompt,repeat,LAYERS + 1u,stream,&repeat_stats);
	RunChain(capped,prompt,eager,2u,stream,&capped_stats);
	CHECK(memcmp(reference,eager,sizeof(reference)) == 0);
	CompareState(full,capped,0u);
	CheckUntouchedSlots(capped);
	CHECK(capped_stats.eager_steps != 0u && capped_stats.graph_launches == STEPS - capped_stats.eager_steps + capped_stats.rollback_steps);
	printf("ok replay limit 2 on a cold working set: %llu steps exhausted their replays and ran eager; tokens and recurrent state bit-identical to all-pinned; launches=%llu rollbacks=%llu\n",
		(unsigned long long)capped_stats.eager_steps,(unsigned long long)capped_stats.graph_launches,(unsigned long long)capped_stats.rollback_steps);
	printf("TIMING toy chain %u steps: all-pinned %.3f ms/step, working-set warm %.3f ms/step (graph nodes %zu vs %zu)\n",STEPS,repeat_stats.seconds * 1e3 / STEPS,warm_stats.seconds * 1e3 / STEPS,full->nodes,partial->nodes);
	printf("PASS working-set graph: a missing expert is detected on every rank, rolled back and replayed with output bit-identical to the all-pinned graph\n");
	return(0);
}
