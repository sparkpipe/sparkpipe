#define main PocLinearPairSweepUnusedMain
#include "dsv4_linear_pair_sweep.cu"
#undef main

#include <cooperative_groups.h>

#define POC_GATE_LAYERS 43u
#define POC_GATE_EXPERTS 256u
#define POC_GATE_TOPK 6u
#define POC_GATE_HIDDEN 4096u
#define POC_GATE_EXPERT_WIDTH 512u
#define POC_GATE_HASH_LAYERS 3u
#define POC_GATE_VOCAB 129280u
#define POC_GATE_CANDIDATES 2u
#define POC_GATE_DISCOVERY_ROUNDS 9u
#define POC_GATE_VALIDATION_ROUNDS 15u
#define POC_GATE_EXACT_SEEDS 7u
#define POC_GATE_CHAIN_REPEATS 8u
#define POC_GATE_KIND_WEIGHT 16u
#define POC_GATE_KIND_BIAS 17u
#define POC_GATE_KIND_TID2EID 18u
#define POC_GATE_KIND_SHARED_W1 22u
#define POC_GATE_KIND_SHARED_W2 23u
#define POC_GATE_KIND_SHARED_W3 24u
#define POC_GATE_SWIGLU_LIMIT 10.0f
#define POC_GATE_ROUTE_SCALE 1.5f

typedef struct PocGateWeights
{
	uint8_t *gate_bf16,*shared_w1,*shared_w2,*shared_w3;
	uint8_t *shared_w1_scale,*shared_w2_scale,*shared_w3_scale;
	float *bias_f32;
	uint32_t *tid2eid_u32;
	uint64_t gate_stride,w1_stride,w2_stride,w3_stride;
	uint64_t w1_scale_stride,w2_scale_stride,w3_scale_stride;
	uint64_t bias_stride,tid_stride,source_digest;
	uint32_t gate_sources,bias_sources,tid_sources,shared_sources;
} PocGateWeights;

typedef struct PocGateBuffers
{
	__nv_bfloat16 *inputs,*shared_activated,*shared_output;
	float *scores,*route_weights;
	uint32_t *token_id,*route_experts,*group_offsets,*route_packed;
	uint32_t *route_source,*tile_prefix_up,*tile_prefix_down;
} PocGateBuffers;

typedef struct PocGateEvents
{
	cudaEvent_t kickoff,aux_done[POC_GATE_LAYERS];
	cudaEvent_t layer_done[POC_GATE_LAYERS],aux_final;
} PocGateEvents;

typedef struct PocGateState
{
	PocGateWeights weights;
	PocGateBuffers buffers;
	cudaStream_t primary,auxiliary;
	PocGateEvents events;
} PocGateState;

typedef struct PocGateGraph
{
	cudaGraph_t graph;
	cudaGraphExec_t executable;
} PocGateGraph;

typedef struct PocGateCandidateResult
{
	float discovery_control[POC_GATE_DISCOVERY_ROUNDS];
	float discovery_candidate[POC_GATE_DISCOVERY_ROUNDS];
	float validation_control[POC_GATE_VALIDATION_ROUNDS];
	float validation_candidate[POC_GATE_VALIDATION_ROUNDS];
	uint32_t validation_wins,bitwise_exact,accepted;
} PocGateCandidateResult;

typedef struct PocGateResults
{
	PocGateCandidateResult candidate[POC_GATE_CANDIDATES];
	uint64_t exact_digest;
} PocGateResults;

static const char *PocGateCandidateNames[POC_GATE_CANDIDATES] =
{
	"score_then_fused_select_route","cooperative_all_fused_gate_route"
};

static __global__ void PocGateScoresKernel(const void *weight_bf16,
	const void *input_bf16,float *scores_f32)
{
	extern __shared__ float gate_shared[];
	uint32_t warp_count = blockDim.x / SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t expert = blockIdx.x * warp_count + warp,element;
	float accumulator;
	for (element=threadIdx.x; element<POC_GATE_HIDDEN; element+=blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,element);
	__syncthreads();
	if ( expert >= POC_GATE_EXPERTS )
		return;
	accumulator = SparkLmDotRowBf16(gate_shared,weight_bf16,expert,
		POC_GATE_HIDDEN,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		scores_f32[expert] = sqrtf(SparkLmSoftplus(accumulator));
}

static __global__ void PocGateSelectKernel(const float *scores_f32,
	const float *bias_f32,const uint32_t *tid2eid_u32,
	const uint32_t *token_ids,uint32_t *indices_u32,float *weights_f32)
{
	__shared__ uint64_t ordered_keys[POC_GATE_EXPERTS];
	uint64_t selected_key;
	uint32_t expert,rank = threadIdx.x,selected_expert;
	float selected_score,selected_total;
	if ( tid2eid_u32 != 0 )
		selected_expert = rank < POC_GATE_TOPK ?
			tid2eid_u32[(uint64_t)token_ids[0] * POC_GATE_TOPK + rank] :
			UINT32_MAX;
	else
	{
		for (expert=threadIdx.x; expert<POC_GATE_EXPERTS;
			expert+=blockDim.x)
			ordered_keys[expert] = SparkLmOrderedTopKKey(scores_f32[expert] +
				(bias_f32 != 0 ? bias_f32[expert] : 0.0f),expert);
		__syncthreads();
		SparkLmBitonicSortKeysAscending<POC_GATE_EXPERTS>(ordered_keys);
		selected_key = rank < POC_GATE_TOPK ?
			ordered_keys[POC_GATE_EXPERTS - 1u - rank] : 0u;
		selected_expert = selected_key != 0u ?
			UINT32_MAX - (uint32_t)selected_key : UINT32_MAX;
	}
	selected_score = rank < POC_GATE_TOPK && selected_expert < POC_GATE_EXPERTS ?
		scores_f32[selected_expert] : 0.0f;
	if ( threadIdx.x < SPARK_LM_WARP_LANES )
	{
		selected_total = SparkLmWarpReduceSum(selected_score);
		selected_total = __shfl_sync(0xffffffffu,selected_total,0u);
		if ( rank < POC_GATE_TOPK )
		{
			indices_u32[rank] = selected_expert;
			weights_f32[rank] = selected_total > 0.0f ?
				selected_score / selected_total * POC_GATE_ROUTE_SCALE : 0.0f;
		}
	}
}

static __device__ void PocGateBuildRoute(const uint32_t *route_expert,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,uint32_t *count)
{
	uint32_t index,expert,packed;
	for (index=threadIdx.x; index<POC_GATE_EXPERTS; index+=blockDim.x)
		count[index] = 0u;
	__syncthreads();
	for (index=threadIdx.x; index<POC_GATE_TOPK; index+=blockDim.x)
		atomicAdd(&count[route_expert[index]],1u);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		uint32_t total = 0u,held,up = 0u,down = 0u,rows;
		for (index=0u; index<POC_GATE_EXPERTS; index++)
		{
			held = count[index];
			group_row_offset[index] = total;
			count[index] = total;
			total += held;
			rows = held != 0u ? 1u : 0u;
			tile_prefix_up[index] = up;
			tile_prefix_down[index] = down;
			up += rows * 16u;
			down += rows * 32u;
		}
		group_row_offset[POC_GATE_EXPERTS] = total;
		tile_prefix_up[POC_GATE_EXPERTS] = up;
		tile_prefix_down[POC_GATE_EXPERTS] = down;
	}
	__syncthreads();
	for (index=threadIdx.x; index<POC_GATE_TOPK; index+=blockDim.x)
	{
		expert = route_expert[index];
		packed = atomicAdd(&count[expert],1u);
		route_packed_row[index] = packed;
		route_source_token[packed] = 0u;
	}
}

static __global__ void PocGateRouteBuildKernel(const uint32_t *route_expert,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down)
{
	__shared__ uint32_t count[POC_GATE_EXPERTS];
	PocGateBuildRoute(route_expert,group_row_offset,route_packed_row,
		route_source_token,tile_prefix_up,tile_prefix_down,count);
}

static __device__ void PocGateSelectRouteBody(const float *scores_f32,
	const float *bias_f32,const uint32_t *tid2eid_u32,
	const uint32_t *token_ids,uint32_t *indices_u32,float *weights_f32,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,uint64_t *shared_words)
{
	__shared__ uint32_t selected[POC_GATE_TOPK];
	uint64_t selected_key;
	uint32_t expert,rank = threadIdx.x,selected_expert;
	float selected_score,selected_total;
	if ( tid2eid_u32 != 0 )
		selected_expert = rank < POC_GATE_TOPK ?
			tid2eid_u32[(uint64_t)token_ids[0] * POC_GATE_TOPK + rank] :
			UINT32_MAX;
	else
	{
		for (expert=threadIdx.x; expert<POC_GATE_EXPERTS;
			expert+=blockDim.x)
			shared_words[expert] = SparkLmOrderedTopKKey(scores_f32[expert] +
				(bias_f32 != 0 ? bias_f32[expert] : 0.0f),expert);
		__syncthreads();
		SparkLmBitonicSortKeysAscending<POC_GATE_EXPERTS>(shared_words);
		selected_key = rank < POC_GATE_TOPK ?
			shared_words[POC_GATE_EXPERTS - 1u - rank] : 0u;
		selected_expert = selected_key != 0u ?
			UINT32_MAX - (uint32_t)selected_key : UINT32_MAX;
	}
	selected_score = rank < POC_GATE_TOPK && selected_expert < POC_GATE_EXPERTS ?
		scores_f32[selected_expert] : 0.0f;
	if ( threadIdx.x < SPARK_LM_WARP_LANES )
	{
		selected_total = SparkLmWarpReduceSum(selected_score);
		selected_total = __shfl_sync(0xffffffffu,selected_total,0u);
		if ( rank < POC_GATE_TOPK )
		{
			indices_u32[rank] = selected_expert;
			selected[rank] = selected_expert;
			weights_f32[rank] = selected_total > 0.0f ?
				selected_score / selected_total * POC_GATE_ROUTE_SCALE : 0.0f;
		}
	}
	__syncthreads();
	PocGateBuildRoute(selected,group_row_offset,route_packed_row,
		route_source_token,tile_prefix_up,tile_prefix_down,
		(uint32_t *)shared_words);
}

static __global__ void PocGateSelectRouteKernel(const float *scores_f32,
	const float *bias_f32,const uint32_t *tid2eid_u32,
	const uint32_t *token_ids,uint32_t *indices_u32,float *weights_f32,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down)
{
	__shared__ uint64_t shared_words[POC_GATE_EXPERTS];
	PocGateSelectRouteBody(scores_f32,bias_f32,tid2eid_u32,token_ids,
		indices_u32,weights_f32,group_row_offset,route_packed_row,
		route_source_token,tile_prefix_up,tile_prefix_down,shared_words);
}

static __global__ void PocGateAllFusedKernel(const void *weight_bf16,
	const void *input_bf16,float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,
	uint32_t *indices_u32,float *weights_f32,uint32_t *group_row_offset,
	uint32_t *route_packed_row,uint32_t *route_source_token,
	uint32_t *tile_prefix_up,uint32_t *tile_prefix_down)
{
	extern __shared__ float gate_shared[];
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	uint32_t warp_count = blockDim.x / SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t expert = blockIdx.x * warp_count + warp,element;
	float accumulator;
	for (element=threadIdx.x; element<POC_GATE_HIDDEN; element+=blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,element);
	__syncthreads();
	if ( expert < POC_GATE_EXPERTS )
	{
		accumulator = SparkLmDotRowBf16(gate_shared,weight_bf16,expert,
			POC_GATE_HIDDEN,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			scores_f32[expert] = sqrtf(SparkLmSoftplus(accumulator));
	}
	grid.sync();
	if ( blockIdx.x != 0u )
		return;
	PocGateSelectRouteBody(scores_f32,bias_f32,tid2eid_u32,token_ids,
		indices_u32,weights_f32,group_row_offset,route_packed_row,
		route_source_token,tile_prefix_up,tile_prefix_down,
		(uint64_t *)gate_shared);
}

static const PocPackEntry *PocGateEntry(const std::vector<PocPackEntry> &entries,
	uint32_t layer,uint32_t kind)
{
	return(PocFindEntry(entries,layer,kind));
}

static int32_t PocGateCopyBytes(int32_t fd,uint64_t offset,uint64_t bytes,
	uint8_t *device,uint64_t *digest)
{
	std::vector<uint8_t> host(bytes);
	if ( PocReadExact(fd,host.data(),bytes,offset) < 0 )
		return(-1);
	*digest = PocFnvBytes(host.data(),bytes,*digest);
	return(cudaMemcpy(device,host.data(),bytes,cudaMemcpyHostToDevice) ==
		cudaSuccess ? 0 : -2);
}

static int32_t PocGateAllocateWeights(PocGateWeights *weights)
{
	weights->gate_stride = (uint64_t)POC_GATE_EXPERTS * POC_GATE_HIDDEN * 2u;
	weights->w1_stride = weights->w3_stride =
		(uint64_t)POC_GATE_EXPERT_WIDTH * POC_GATE_HIDDEN;
	weights->w2_stride = (uint64_t)POC_GATE_HIDDEN * POC_GATE_EXPERT_WIDTH;
	weights->w1_scale_stride = weights->w3_scale_stride =
		(uint64_t)POC_GATE_EXPERT_WIDTH * (POC_GATE_HIDDEN / 128u);
	weights->w2_scale_stride =
		(uint64_t)POC_GATE_HIDDEN * (POC_GATE_EXPERT_WIDTH / 128u);
	weights->bias_stride = POC_GATE_EXPERTS * sizeof(float);
	weights->tid_stride = (uint64_t)POC_GATE_VOCAB * POC_GATE_TOPK *
		sizeof(uint32_t);
	if ( cudaMalloc(&weights->gate_bf16,weights->gate_stride * POC_GATE_LAYERS) !=
		cudaSuccess || cudaMalloc(&weights->shared_w1,weights->w1_stride *
		POC_GATE_LAYERS) != cudaSuccess || cudaMalloc(&weights->shared_w2,
		weights->w2_stride * POC_GATE_LAYERS) != cudaSuccess || cudaMalloc(
		&weights->shared_w3,weights->w3_stride * POC_GATE_LAYERS) != cudaSuccess ||
		cudaMalloc(&weights->shared_w1_scale,weights->w1_scale_stride *
		POC_GATE_LAYERS) != cudaSuccess || cudaMalloc(&weights->shared_w2_scale,
		weights->w2_scale_stride * POC_GATE_LAYERS) != cudaSuccess || cudaMalloc(
		&weights->shared_w3_scale,weights->w3_scale_stride * POC_GATE_LAYERS) !=
		cudaSuccess || cudaMalloc(&weights->bias_f32,weights->bias_stride *
		POC_GATE_LAYERS) != cudaSuccess || cudaMalloc(&weights->tid2eid_u32,
		weights->tid_stride * POC_GATE_HASH_LAYERS) != cudaSuccess )
		return(-1);
	return(0);
}

static int32_t PocGateLoadOneLayer(int32_t fd,
	const std::vector<PocPackEntry> &entries,PocGateWeights *weights,
	uint32_t target,uint32_t source)
{
	const PocPackEntry *gate,*w1,*w2,*w3,*route;
	uint32_t hash = target < POC_GATE_HASH_LAYERS ? 1u : 0u;
	gate = PocGateEntry(entries,source,POC_GATE_KIND_WEIGHT);
	w1 = PocGateEntry(entries,source,POC_GATE_KIND_SHARED_W1);
	w2 = PocGateEntry(entries,source,POC_GATE_KIND_SHARED_W2);
	w3 = PocGateEntry(entries,source,POC_GATE_KIND_SHARED_W3);
	route = PocGateEntry(entries,source,hash != 0u ? POC_GATE_KIND_TID2EID :
		POC_GATE_KIND_BIAS);
	if ( gate == 0 || w1 == 0 || w2 == 0 || w3 == 0 || route == 0 )
		return(-1);
	if ( PocGateCopyBytes(fd,gate->payload_offset,weights->gate_stride,
		weights->gate_bf16 + (uint64_t)target * weights->gate_stride,
		&weights->source_digest) < 0 || PocGateCopyBytes(fd,w1->payload_offset,
		weights->w1_stride,weights->shared_w1 + (uint64_t)target *
		weights->w1_stride,&weights->source_digest) < 0 || PocGateCopyBytes(fd,
		w2->payload_offset,weights->w2_stride,weights->shared_w2 +
		(uint64_t)target * weights->w2_stride,&weights->source_digest) < 0 ||
		PocGateCopyBytes(fd,w3->payload_offset,weights->w3_stride,
		weights->shared_w3 + (uint64_t)target * weights->w3_stride,
		&weights->source_digest) < 0 )
		return(-2);
	if ( PocGateCopyBytes(fd,w1->scale_offset,weights->w1_scale_stride,
		weights->shared_w1_scale + (uint64_t)target * weights->w1_scale_stride,
		&weights->source_digest) < 0 || PocGateCopyBytes(fd,w2->scale_offset,
		weights->w2_scale_stride,weights->shared_w2_scale + (uint64_t)target *
		weights->w2_scale_stride,&weights->source_digest) < 0 ||
		PocGateCopyBytes(fd,w3->scale_offset,weights->w3_scale_stride,
		weights->shared_w3_scale + (uint64_t)target * weights->w3_scale_stride,
		&weights->source_digest) < 0 )
		return(-3);
	if ( hash != 0u )
		return(PocGateCopyBytes(fd,route->payload_offset,weights->tid_stride,
			(uint8_t *)weights->tid2eid_u32 + (uint64_t)target *
			weights->tid_stride,&weights->source_digest));
	return(PocGateCopyBytes(fd,route->payload_offset,weights->bias_stride,
		(uint8_t *)weights->bias_f32 + (uint64_t)target * weights->bias_stride,
		&weights->source_digest));
}

static int32_t PocGateLoadWeights(int32_t fd,
	const std::vector<PocPackEntry> &entries,const PocPackHeader *header,
	PocGateWeights *weights)
{
	uint32_t layer,source;
	memset(weights,0,sizeof(*weights));
	weights->source_digest = UINT64_C(1469598103934665603);
	if ( header->layer_count < 4u || PocGateAllocateWeights(weights) < 0 )
		return(-1);
	for (layer=0u; layer<POC_GATE_LAYERS; layer++)
	{
		source = layer < POC_GATE_HASH_LAYERS ? layer :
			POC_GATE_HASH_LAYERS + ((layer - POC_GATE_HASH_LAYERS) %
			(header->layer_count - POC_GATE_HASH_LAYERS));
		if ( PocGateLoadOneLayer(fd,entries,weights,layer,source) < 0 )
			return(-2);
	}
	weights->gate_sources = header->layer_count;
	weights->bias_sources = header->layer_count - POC_GATE_HASH_LAYERS;
	weights->tid_sources = POC_GATE_HASH_LAYERS;
	weights->shared_sources = header->layer_count;
	return(0);
}

static int32_t PocGateAllocateBuffers(PocGateBuffers *buffers)
{
	memset(buffers,0,sizeof(*buffers));
	if ( cudaMalloc(&buffers->inputs,(uint64_t)POC_GATE_LAYERS *
		POC_GATE_HIDDEN * sizeof(uint16_t)) != cudaSuccess || cudaMalloc(
		&buffers->shared_activated,POC_GATE_EXPERT_WIDTH * sizeof(uint16_t)) !=
		cudaSuccess || cudaMalloc(&buffers->shared_output,POC_GATE_HIDDEN *
		sizeof(uint16_t)) != cudaSuccess || cudaMalloc(&buffers->scores,
		POC_GATE_EXPERTS * sizeof(float)) != cudaSuccess || cudaMalloc(
		&buffers->route_weights,POC_GATE_TOPK * sizeof(float)) != cudaSuccess ||
		cudaMalloc(&buffers->token_id,sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc(&buffers->route_experts,POC_GATE_TOPK * sizeof(uint32_t)) !=
		cudaSuccess || cudaMalloc(&buffers->group_offsets,(POC_GATE_EXPERTS + 1u) *
		sizeof(uint32_t)) != cudaSuccess || cudaMalloc(&buffers->route_packed,
		POC_GATE_TOPK * sizeof(uint32_t)) != cudaSuccess || cudaMalloc(
		&buffers->route_source,POC_GATE_TOPK * sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc(&buffers->tile_prefix_up,(POC_GATE_EXPERTS + 1u) *
		sizeof(uint32_t)) != cudaSuccess || cudaMalloc(&buffers->tile_prefix_down,
		(POC_GATE_EXPERTS + 1u) * sizeof(uint32_t)) != cudaSuccess )
		return(-1);
	return(0);
}

static const uint8_t *PocGateLayerBytes(const uint8_t *base,uint64_t stride,
	uint32_t layer)
{
	return(base + (uint64_t)layer * stride);
}

static const float *PocGateBias(const PocGateState *state,uint32_t layer)
{
	return(layer < POC_GATE_HASH_LAYERS ? 0 :
		(const float *)PocGateLayerBytes((const uint8_t *)state->weights.bias_f32,
		state->weights.bias_stride,layer));
}

static const uint32_t *PocGateTid(const PocGateState *state,uint32_t layer)
{
	return(layer < POC_GATE_HASH_LAYERS ?
		(const uint32_t *)PocGateLayerBytes((const uint8_t *)
		state->weights.tid2eid_u32,state->weights.tid_stride,layer) : 0);
}

static cudaError_t PocGateLaunchShared(PocGateState *state,uint32_t layer,
	cudaStream_t stream)
{
	const void *input = state->buffers.inputs + (uint64_t)layer * POC_GATE_HIDDEN;
	cudaError_t error;
	error = SparkLmHostLaunchSm121FusedDenseW13<
		SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(stream,
		PocGateLayerBytes(state->weights.shared_w1,state->weights.w1_stride,layer),
		PocGateLayerBytes(state->weights.shared_w1_scale,
		state->weights.w1_scale_stride,layer),PocGateLayerBytes(
		state->weights.shared_w3,state->weights.w3_stride,layer),
		PocGateLayerBytes(state->weights.shared_w3_scale,
		state->weights.w3_scale_stride,layer),input,
		state->buffers.shared_activated,1u,POC_GATE_HIDDEN,
		POC_GATE_EXPERT_WIDTH,POC_GATE_SWIGLU_LIMIT);
	if ( error != cudaSuccess )
		return(error);
	return(SparkLmHostLaunchSm121DecodeLinear<128u,
		SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0>(stream,
		SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,PocGateLayerBytes(
		state->weights.shared_w2,state->weights.w2_stride,layer),
		PocGateLayerBytes(state->weights.shared_w2_scale,
		state->weights.w2_scale_stride,layer),state->buffers.shared_activated,
		state->buffers.shared_output,1u,POC_GATE_EXPERT_WIDTH,POC_GATE_HIDDEN));
}

static cudaError_t PocGateLaunchScores(PocGateState *state,uint32_t layer,
	cudaStream_t stream)
{
	PocGateScoresKernel<<<32u,256u,POC_GATE_HIDDEN * sizeof(float),stream>>>(
		PocGateLayerBytes(state->weights.gate_bf16,state->weights.gate_stride,layer),
		state->buffers.inputs + (uint64_t)layer * POC_GATE_HIDDEN,
		state->buffers.scores);
	return(cudaGetLastError());
}

static cudaError_t PocGateLaunchBaseline(PocGateState *state,uint32_t layer,
	cudaStream_t stream)
{
	cudaError_t error = PocGateLaunchScores(state,layer,stream);
	if ( error == cudaSuccess )
	{
		PocGateSelectKernel<<<1u,256u,0u,stream>>>(state->buffers.scores,
			PocGateBias(state,layer),PocGateTid(state,layer),
			state->buffers.token_id,state->buffers.route_experts,
			state->buffers.route_weights);
		error = cudaGetLastError();
	}
	if ( error == cudaSuccess )
	{
		PocGateRouteBuildKernel<<<1u,256u,0u,stream>>>(
			state->buffers.route_experts,state->buffers.group_offsets,
			state->buffers.route_packed,state->buffers.route_source,
			state->buffers.tile_prefix_up,state->buffers.tile_prefix_down);
		error = cudaGetLastError();
	}
	return(error);
}

static cudaError_t PocGateLaunchSelectRoute(PocGateState *state,uint32_t layer,
	cudaStream_t stream)
{
	cudaError_t error = PocGateLaunchScores(state,layer,stream);
	if ( error != cudaSuccess )
		return(error);
	PocGateSelectRouteKernel<<<1u,256u,0u,stream>>>(state->buffers.scores,
		PocGateBias(state,layer),PocGateTid(state,layer),state->buffers.token_id,
		state->buffers.route_experts,state->buffers.route_weights,
		state->buffers.group_offsets,state->buffers.route_packed,
		state->buffers.route_source,state->buffers.tile_prefix_up,
		state->buffers.tile_prefix_down);
	return(cudaGetLastError());
}

static cudaError_t PocGateLaunchAllFused(PocGateState *state,uint32_t layer,
	cudaStream_t stream)
{
	const void *weight = PocGateLayerBytes(state->weights.gate_bf16,
		state->weights.gate_stride,layer);
	const void *input = state->buffers.inputs + (uint64_t)layer * POC_GATE_HIDDEN;
	const float *bias = PocGateBias(state,layer);
	const uint32_t *tid = PocGateTid(state,layer);
	void *arguments[] = {&weight,&input,&state->buffers.scores,&bias,&tid,
		&state->buffers.token_id,&state->buffers.route_experts,
		&state->buffers.route_weights,&state->buffers.group_offsets,
		&state->buffers.route_packed,&state->buffers.route_source,
		&state->buffers.tile_prefix_up,&state->buffers.tile_prefix_down};
	return(cudaLaunchCooperativeKernel((void *)PocGateAllFusedKernel,
		dim3(32u),dim3(256u),arguments,POC_GATE_HIDDEN * sizeof(float),stream));
}

static cudaError_t PocGateLaunchCandidate(PocGateState *state,uint32_t layer,
	uint32_t candidate,cudaStream_t stream)
{
	return(candidate == 0u ? PocGateLaunchSelectRoute(state,layer,stream) :
		PocGateLaunchAllFused(state,layer,stream));
}

static int32_t PocGateCreateEvents(PocGateEvents *events)
{
	uint32_t layer;
	if ( cudaEventCreateWithFlags(&events->kickoff,cudaEventDisableTiming) !=
		cudaSuccess || cudaEventCreateWithFlags(&events->aux_final,
		cudaEventDisableTiming) != cudaSuccess )
		return(-1);
	for (layer=0u; layer<POC_GATE_LAYERS; layer++)
		if ( cudaEventCreateWithFlags(&events->aux_done[layer],
			cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(
			&events->layer_done[layer],cudaEventDisableTiming) != cudaSuccess )
			return(-2);
	return(0);
}

static int32_t PocGateCapture(PocGateState *state,int32_t candidate,
	PocGateGraph *result)
{
	uint32_t layer;
	cudaError_t error;
	if ( cudaStreamBeginCapture(state->primary,cudaStreamCaptureModeGlobal) !=
		cudaSuccess || cudaEventRecord(state->events.kickoff,state->primary) !=
		cudaSuccess || cudaStreamWaitEvent(state->auxiliary,
		state->events.kickoff,0u) != cudaSuccess )
		return(-1);
	for (layer=0u; layer<POC_GATE_LAYERS; layer++)
	{
		error = PocGateLaunchShared(state,layer,state->auxiliary);
		if ( error == cudaSuccess )
			error = candidate < 0 ? PocGateLaunchBaseline(state,layer,
				state->primary) : PocGateLaunchCandidate(state,layer,
				(uint32_t)candidate,state->primary);
		if ( error != cudaSuccess || cudaEventRecord(state->events.aux_done[layer],
			state->auxiliary) != cudaSuccess || cudaStreamWaitEvent(state->primary,
			state->events.aux_done[layer],0u) != cudaSuccess || cudaEventRecord(
			state->events.layer_done[layer],state->primary) != cudaSuccess ||
			cudaStreamWaitEvent(state->auxiliary,state->events.layer_done[layer],0u) !=
			cudaSuccess )
			return(-2);
	}
	if ( cudaEventRecord(state->events.aux_final,state->auxiliary) != cudaSuccess ||
		cudaStreamWaitEvent(state->primary,state->events.aux_final,0u) !=
		cudaSuccess || cudaStreamEndCapture(state->primary,&result->graph) !=
		cudaSuccess )
		return(-3);
	return(cudaGraphInstantiate(&result->executable,result->graph,0ull) ==
		cudaSuccess ? 0 : -4);
}

static void PocGateAppendBytes(std::vector<uint8_t> *output,const void *device,
	uint64_t bytes)
{
	uint64_t old = output->size();
	output->resize(old + bytes);
	cudaMemcpy(output->data() + old,device,bytes,cudaMemcpyDeviceToHost);
}

static int32_t PocGateCopyOutputs(PocGateState *state,
	std::vector<uint8_t> *output)
{
	PocGateBuffers *buffers = &state->buffers;
	output->clear();
	PocGateAppendBytes(output,buffers->scores,POC_GATE_EXPERTS * sizeof(float));
	PocGateAppendBytes(output,buffers->route_experts,
		POC_GATE_TOPK * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->route_weights,
		POC_GATE_TOPK * sizeof(float));
	PocGateAppendBytes(output,buffers->group_offsets,
		(POC_GATE_EXPERTS + 1u) * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->route_packed,
		POC_GATE_TOPK * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->route_source,
		POC_GATE_TOPK * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->tile_prefix_up,
		(POC_GATE_EXPERTS + 1u) * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->tile_prefix_down,
		(POC_GATE_EXPERTS + 1u) * sizeof(uint32_t));
	PocGateAppendBytes(output,buffers->shared_output,
		POC_GATE_HIDDEN * sizeof(uint16_t));
	return(cudaGetLastError() == cudaSuccess ? 0 : -1);
}

static int32_t PocGateRunLayer(PocGateState *state,uint32_t layer,
	int32_t candidate)
{
	cudaError_t error = PocGateLaunchShared(state,layer,state->auxiliary);
	if ( error == cudaSuccess )
		error = candidate < 0 ? PocGateLaunchBaseline(state,layer,state->primary) :
			PocGateLaunchCandidate(state,layer,(uint32_t)candidate,state->primary);
	return(error == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess ?
		0 : -1);
}

static int32_t PocGateCheckExact(PocGateState *state,PocGateResults *results)
{
	static const uint32_t token_ids[POC_GATE_EXACT_SEEDS] =
		{0u,1u,17u,4096u,65535u,100000u,129279u};
	std::vector<uint8_t> reference,candidate_output;
	uint32_t seed,layer,candidate;
	results->exact_digest = UINT64_C(1469598103934665603);
	for (seed=0u; seed<POC_GATE_EXACT_SEEDS; seed++)
	{
		PocFillInputs<<<((uint64_t)POC_GATE_LAYERS * POC_GATE_HIDDEN + 255u) /
			256u,256u,0u,state->primary>>>(state->buffers.inputs,POC_GATE_LAYERS,
			611u + seed * 977u);
		if ( cudaMemcpyAsync(state->buffers.token_id,&token_ids[seed],
			sizeof(uint32_t),cudaMemcpyHostToDevice,state->primary) != cudaSuccess ||
			cudaStreamSynchronize(state->primary) != cudaSuccess )
			return(-1);
		for (layer=0u; layer<POC_GATE_LAYERS; layer++)
		{
			if ( PocGateRunLayer(state,layer,-1) < 0 ||
				PocGateCopyOutputs(state,&reference) < 0 )
				return(-2);
			results->exact_digest = PocFnvBytes(reference.data(),reference.size(),
				results->exact_digest);
			for (candidate=0u; candidate<POC_GATE_CANDIDATES; candidate++)
			{
				if ( PocGateRunLayer(state,layer,(int32_t)candidate) < 0 ||
					PocGateCopyOutputs(state,&candidate_output) < 0 )
					return(-3);
				if ( reference.size() != candidate_output.size() || memcmp(
					reference.data(),candidate_output.data(),reference.size()) != 0 )
					return(-4 - (int32_t)candidate);
			}
		}
	}
	for (candidate=0u; candidate<POC_GATE_CANDIDATES; candidate++)
		results->candidate[candidate].bitwise_exact = 1u;
	return(0);
}

static int32_t PocGateTimeGraph(cudaGraphExec_t executable,cudaStream_t stream,
	cudaEvent_t start,cudaEvent_t stop,float *milliseconds)
{
	float elapsed;
	uint32_t repeat;
	if ( cudaEventRecord(start,stream) != cudaSuccess )
		return(-1);
	for (repeat=0u; repeat<POC_GATE_CHAIN_REPEATS; repeat++)
		if ( cudaGraphLaunch(executable,stream) != cudaSuccess )
			return(-2);
	if ( cudaEventRecord(stop,stream) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess || cudaEventElapsedTime(
		&elapsed,start,stop) != cudaSuccess )
		return(-3);
	*milliseconds = elapsed / (float)POC_GATE_CHAIN_REPEATS;
	return(0);
}

static int32_t PocGateMeasure(PocGateState *state,PocGateGraph *control,
	PocGateGraph *candidates,PocGateResults *results,cudaEvent_t start,
	cudaEvent_t stop)
{
	PocGateCandidateResult *result;
	uint32_t candidate,round,first,warmup;
	for (candidate=0u; candidate<POC_GATE_CANDIDATES; candidate++)
	{
		result = &results->candidate[candidate];
		for (warmup=0u; warmup<3u; warmup++)
			if ( cudaGraphLaunch(candidates[candidate].executable,state->primary) !=
				cudaSuccess )
				return(-1);
		if ( cudaStreamSynchronize(state->primary) != cudaSuccess )
			return(-2);
		for (round=0u; round<POC_GATE_DISCOVERY_ROUNDS; round++)
		{
			first = (round + candidate) & 1u;
			if ( PocGateTimeGraph(first == 0u ? control->executable :
				candidates[candidate].executable,state->primary,start,stop,
				first == 0u ? &result->discovery_control[round] :
				&result->discovery_candidate[round]) < 0 || PocGateTimeGraph(
				first == 0u ? candidates[candidate].executable : control->executable,
				state->primary,start,stop,first == 0u ?
				&result->discovery_candidate[round] :
				&result->discovery_control[round]) < 0 )
				return(-3);
		}
		for (round=0u; round<POC_GATE_VALIDATION_ROUNDS; round++)
		{
			first = round & 1u;
			if ( PocGateTimeGraph(first == 0u ? control->executable :
				candidates[candidate].executable,state->primary,start,stop,
				first == 0u ? &result->validation_control[round] :
				&result->validation_candidate[round]) < 0 || PocGateTimeGraph(
				first == 0u ? candidates[candidate].executable : control->executable,
				state->primary,start,stop,first == 0u ?
				&result->validation_candidate[round] :
				&result->validation_control[round]) < 0 )
				return(-4);
			if ( result->validation_candidate[round] <
				result->validation_control[round] )
				result->validation_wins++;
		}
		result->accepted = result->bitwise_exact != 0u && PocMedian(
			result->validation_candidate,POC_GATE_VALIDATION_ROUNDS) < PocMedian(
			result->validation_control,POC_GATE_VALIDATION_ROUNDS) &&
			result->validation_wins >= 11u ? 1u : 0u;
	}
	return(0);
}

static void PocGatePrintResult(FILE *file,const PocGateCandidateResult *result,
	uint32_t candidate,uint32_t last)
{
	float control = PocMedian(result->validation_control,
		POC_GATE_VALIDATION_ROUNDS);
	float measured = PocMedian(result->validation_candidate,
		POC_GATE_VALIDATION_ROUNDS);
	fprintf(file,"    {\"name\":\"%s\",\"bitwise_exact\":%s,",
		PocGateCandidateNames[candidate],result->bitwise_exact != 0u ?
		"true" : "false");
	fprintf(file,"\"discovery_control_round_ms\":");
	PocPrintArray(file,result->discovery_control,POC_GATE_DISCOVERY_ROUNDS);
	fprintf(file,",\"discovery_candidate_round_ms\":");
	PocPrintArray(file,result->discovery_candidate,POC_GATE_DISCOVERY_ROUNDS);
	fprintf(file,",\"validation_control_round_ms\":");
	PocPrintArray(file,result->validation_control,POC_GATE_VALIDATION_ROUNDS);
	fprintf(file,",\"validation_candidate_round_ms\":");
	PocPrintArray(file,result->validation_candidate,POC_GATE_VALIDATION_ROUNDS);
	fprintf(file,",\"control_median_ms\":%.9f,\"candidate_median_ms\":%.9f,",
		control,measured);
	fprintf(file,"\"gain_percent\":%.9f,\"candidate_wins\":%u,",
		100.0f * (control / measured - 1.0f),result->validation_wins);
	fprintf(file,"\"accepted\":%s}%s\n",result->accepted != 0u ? "true" :
		"false",last != 0u ? "" : ",");
}

static int32_t PocGatePrintReceipt(const char *path,const char *pack_path,
	const char *source_sha,const char *binary_sha,const PocPackHeader *header,
	const cudaDeviceProp *properties,const PocGateState *state,
	const PocGateResults *results)
{
	FILE *file = fopen(path,"w");
	uint32_t candidate;
	if ( file == 0 )
		return(-1);
	fprintf(file,"{\n  \"schema\":\"dsv4_b1_gate_route_fusion_fast_gate_v1\",\n");
	fprintf(file,"  \"source_commit\":\"%s\",\n",POC_SOURCE_COMMIT);
	fprintf(file,"  \"source_sha256\":\"%s\",\n",source_sha);
	fprintf(file,"  \"binary_sha256\":\"%s\",\n",binary_sha);
	fprintf(file,"  \"model_revision\":\"%s\",\n",POC_MODEL_REVISION);
	fprintf(file,"  \"device\":{\"host\":\"spark3\",\"name\":\"%s\",",
		properties->name);
	fprintf(file,"\"cc\":\"%d.%d\",\"sm_count\":%d},\n",properties->major,
		properties->minor,properties->multiProcessorCount);
	fprintf(file,"  \"pack\":{\"path\":\"%s\",\"file_bytes\":%llu,",
		pack_path,(unsigned long long)header->file_bytes);
	fprintf(file,"\"first_layer\":%u,\"layer_count\":%u,",
		header->first_layer_index,header->layer_count);
	fprintf(file,"\"payload_source\":\"actual_ga_tp4_pp4_stagepack_rows_cycled_to_43_layers\"},\n");
	fprintf(file,"  \"actual_sources\":{\"gate\":%u,\"bias\":%u,\"tid2eid\":%u,",
		state->weights.gate_sources,state->weights.bias_sources,
		state->weights.tid_sources);
	fprintf(file,"\"shared_expert\":%u,\"source_fnv64\":\"%016llx\"},\n",
		state->weights.shared_sources,
		(unsigned long long)state->weights.source_digest);
	fprintf(file,"  \"shape\":{\"layers\":43,\"rows\":1,\"hidden\":4096,");
	fprintf(file,"\"experts\":256,\"topk\":6,\"hash_layers\":3,");
	fprintf(file,"\"tp4_shared_expert_width\":512,\"route_tile_m\":16,");
	fprintf(file,"\"route_up_tiles_per_expert\":16,\"route_down_tiles_per_expert\":32},\n");
	fprintf(file,"  \"method\":{\"control\":");
	fprintf(file,"\"accepted_gate_scores_then_gate_select_then_route_build_on_primary\",");
	fprintf(file,"\"overlap\":");
	fprintf(file,"\"actual_fp8_shared_expert_w13_then_w2_on_auxiliary_with_per_layer_join\",");
	fprintf(file,"\"quality\":");
	fprintf(file,"\"unchanged_fp32_score_accumulation_sort_ties_route_order_and_bf16_spine\",");
	fprintf(file,"\"chain_repeats_per_sample\":%u,",POC_GATE_CHAIN_REPEATS);
	fprintf(file,"\"discovery_rounds\":%u,\"validation_rounds\":%u,",
		POC_GATE_DISCOVERY_ROUNDS,POC_GATE_VALIDATION_ROUNDS);
	fprintf(file,"\"exact_seeds\":%u,\"acceptance\":",POC_GATE_EXACT_SEEDS);
	fprintf(file,"\"positive_validation_median_and_at_least_11_of_15_paired_wins_no_minimum_gain\"},\n");
	fprintf(file,"  \"exact_digest_fnv64\":\"%016llx\",\n",
		(unsigned long long)results->exact_digest);
	fprintf(file,"  \"candidates\":[\n");
	for (candidate=0u; candidate<POC_GATE_CANDIDATES; candidate++)
		PocGatePrintResult(file,&results->candidate[candidate],candidate,
			candidate + 1u == POC_GATE_CANDIDATES);
	fprintf(file,"  ]\n}\n");
	fclose(file);
	return(0);
}

static int32_t PocGateInitialize(const char *pack_path,PocGateState *state,
	PocPackHeader *header,std::vector<PocPackEntry> *entries,int32_t *fd)
{
	memset(state,0,sizeof(*state));
	if ( PocLoadPack(pack_path,header,entries,fd) < 0 ||
		PocGateLoadWeights(*fd,*entries,header,&state->weights) < 0 ||
		PocGateAllocateBuffers(&state->buffers) < 0 || cudaStreamCreateWithFlags(
		&state->primary,cudaStreamNonBlocking) != cudaSuccess ||
		cudaStreamCreateWithFlags(&state->auxiliary,cudaStreamNonBlocking) !=
		cudaSuccess || PocGateCreateEvents(&state->events) < 0 )
		return(-1);
	return(0);
}

int main(int argc,char **argv)
{
	PocPackHeader header;
	std::vector<PocPackEntry> entries;
	PocGateState state;
	PocGateGraph control,candidates[POC_GATE_CANDIDATES];
	PocGateResults results;
	cudaDeviceProp properties;
	cudaEvent_t start,stop;
	int32_t fd,error;
	uint32_t candidate,token_id = 12345u;
	if ( argc != 5 )
		return(1);
	memset(&control,0,sizeof(control));
	memset(candidates,0,sizeof(candidates));
	memset(&results,0,sizeof(results));
	fprintf(stderr,"load=actual_gate_and_shared_expert_weights\n");
	if ( PocGateInitialize(argv[1],&state,&header,&entries,&fd) < 0 ||
		cudaGetDeviceProperties(&properties,0) != cudaSuccess ||
		cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) !=
		cudaSuccess )
		return(2);
	fprintf(stderr,"exact=all_43_layers_both_candidates\n");
	error = PocGateCheckExact(&state,&results);
	if ( error < 0 )
	{
		fprintf(stderr,"exact_failure=%d cuda=%s\n",error,
			cudaGetErrorString(cudaGetLastError()));
		return(3);
	}
	if ( cudaMemcpy(state.buffers.token_id,&token_id,sizeof(token_id),
		cudaMemcpyHostToDevice) != cudaSuccess )
		return(4);
	fprintf(stderr,"capture=production_overlap_graph\n");
	if ( PocGateCapture(&state,-1,&control) < 0 )
		return(5);
	for (candidate=0u; candidate<POC_GATE_CANDIDATES; candidate++)
	{
		fprintf(stderr,"capture=%s\n",PocGateCandidateNames[candidate]);
		if ( PocGateCapture(&state,(int32_t)candidate,&candidates[candidate]) < 0 )
			return(6);
	}
	fprintf(stderr,"timing=interleaved_full_43_layer_overlap_graphs\n");
	if ( PocGateMeasure(&state,&control,candidates,&results,start,stop) < 0 )
		return(7);
	if ( PocGatePrintReceipt(argv[2],argv[1],argv[3],argv[4],&header,
		&properties,&state,&results) < 0 )
		return(8);
	close(fd);
	return(0);
}
