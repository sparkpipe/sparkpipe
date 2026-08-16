#define main PocLinearPairSweepUnusedMain
#include "dsv4_linear_pair_sweep.cu"
#undef main

#define POC_BUNDLE_LAYER_COUNT 43u
#define POC_BUNDLE_POLICY_COUNT 9u
#define POC_BUNDLE_DISCOVERY_ROUNDS 9u
#define POC_BUNDLE_VALIDATION_ROUNDS 15u
#define POC_BUNDLE_EXACT_SEEDS 7u
#define POC_BUNDLE_CHAIN_REPEATS 8u
#define POC_BUNDLE_TILE_WARPS 16u

typedef struct PocBundleTile
{
	const void *payload;
	const uint8_t *scale;
	void *output_bf16;
	uint32_t neuron_base,neuron_count,weight_format,reserved0;
} PocBundleTile;

typedef struct PocBundleLayer
{
	uint32_t kind,fp8_launch,aux_launch,tile_offset,tile_count;
} PocBundleLayer;

typedef struct PocBundleTensor
{
	const void *payload;
	const uint8_t *scale;
	void *output_bf16;
	uint32_t neurons,weight_format;
} PocBundleTensor;

typedef struct PocBundleEvents
{
	cudaEvent_t kickoff,aux0_done[POC_BUNDLE_LAYER_COUNT];
	cudaEvent_t aux1_done[POC_BUNDLE_LAYER_COUNT];
	cudaEvent_t layer_done[POC_BUNDLE_LAYER_COUNT];
	cudaEvent_t aux0_final,aux1_final;
} PocBundleEvents;

typedef struct PocBundleGraph
{
	cudaGraph_t graph;
	cudaGraphExec_t executable;
} PocBundleGraph;

typedef struct PocBundleState
{
	PocFamily families[POC_FAMILY_COUNT];
	PocBundleLayer layers[POC_BUNDLE_LAYER_COUNT];
	std::vector<PocBundleTile> host_tiles;
	PocBundleTile *device_tiles;
	cudaStream_t primary,aux0,aux1;
	PocBundleEvents events;
} PocBundleState;

typedef struct PocBundleResults
{
	float discovery_control[POC_BUNDLE_POLICY_COUNT]
		[POC_BUNDLE_DISCOVERY_ROUNDS];
	float discovery_candidate[POC_BUNDLE_POLICY_COUNT]
		[POC_BUNDLE_DISCOVERY_ROUNDS];
	float validation_control[POC_BUNDLE_VALIDATION_ROUNDS];
	float validation_candidate[POC_BUNDLE_VALIDATION_ROUNDS];
	uint64_t exact_digest;
	uint32_t selected_policy,validation_wins,bitwise_exact,accepted;
} PocBundleResults;

static const uint32_t PocBundleGrids[POC_BUNDLE_POLICY_COUNT] =
{
	8u,12u,16u,20u,24u,32u,40u,48u,64u
};

static __global__ void PocProjectionBundleKernel(const PocBundleTile *tiles,
	uint32_t tile_count,const void *input_bf16)
{
	extern __shared__ float shared_input[];
	PocBundleTile tile;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t pair,pair_count = POC_HIDDEN >> 1u,tile_index,neuron;
	float accumulator;
	float2 input_pair;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		input_pair = SparkLmLoadBf16Pair(input_bf16,pair);
		shared_input[pair << 1u] = input_pair.x;
		shared_input[(pair << 1u) + 1u] = input_pair.y;
	}
	__syncthreads();
	for (tile_index=blockIdx.x; tile_index<tile_count;
		tile_index+=gridDim.x)
	{
		tile = tiles[tile_index];
		if ( warp >= tile.neuron_count )
			continue;
		neuron = tile.neuron_base + warp;
		if ( tile.weight_format == POC_FORMAT_FP8 )
			accumulator = SparkLmDotRowFp8<128u>(shared_input,tile.payload,
				tile.scale,neuron,POC_HIDDEN,lane);
		else
			accumulator = SparkLmDotRowBf16(shared_input,tile.payload,neuron,
				POC_HIDDEN,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(tile.output_bf16,neuron,accumulator);
	}
}

static const uint8_t *PocBundlePayload(const PocFamily *family,
	uint32_t launch,uint32_t second)
{
	return((second != 0u ? family->second_payload : family->first_payload) +
		(uint64_t)launch * (second != 0u ? family->second_payload_stride :
		family->first_payload_stride));
}

static const uint8_t *PocBundleScale(const PocFamily *family,uint32_t launch,
	uint32_t second)
{
	const uint8_t *scale = second != 0u ? family->second_scale :
		family->first_scale;
	uint64_t stride = second != 0u ? family->second_scale_stride :
		family->first_scale_stride;
	return(scale == 0 ? 0 : scale + (uint64_t)launch * stride);
}

static void PocBundleMakeTensor(const PocFamily *family,uint32_t launch,
	uint32_t second,PocBundleTensor *tensor)
{
	tensor->payload = PocBundlePayload(family,launch,second);
	tensor->scale = PocBundleScale(family,launch,second);
	tensor->output_bf16 = second != 0u ? family->second_output_bf16 :
		family->first_output_bf16;
	tensor->neurons = second != 0u ? family->second_output :
		family->first_output;
	tensor->weight_format = family->fp8 != 0u ? POC_FORMAT_FP8 :
		POC_FORMAT_BF16;
}

static void PocBundleAppendRoundRobin(std::vector<PocBundleTile> *tiles,
	const PocBundleTensor *tensors,uint32_t tensor_count)
{
	PocBundleTile tile;
	uint32_t round,tensor,base,added = 1u;
	for (round=0u; added != 0u; round++)
	{
		added = 0u;
		base = round * POC_BUNDLE_TILE_WARPS;
		for (tensor=0u; tensor<tensor_count; tensor++)
		{
			if ( base >= tensors[tensor].neurons )
				continue;
			memset(&tile,0,sizeof(tile));
			tile.payload = tensors[tensor].payload;
			tile.scale = tensors[tensor].scale;
			tile.output_bf16 = tensors[tensor].output_bf16;
			tile.neuron_base = base;
			tile.neuron_count = std::min(POC_BUNDLE_TILE_WARPS,
				tensors[tensor].neurons - base);
			tile.weight_format = tensors[tensor].weight_format;
			tiles->push_back(tile);
			added = 1u;
		}
	}
}

static void PocBundleBuildLayer(PocBundleState *state,uint32_t layer,
	uint32_t kind,uint32_t aux_launch)
{
	PocBundleTensor tensors[6];
	PocBundleLayer *entry = &state->layers[layer];
	uint32_t tensor_count = 0u;
	memset(entry,0,sizeof(*entry));
	entry->kind = kind;
	entry->fp8_launch = layer;
	entry->aux_launch = aux_launch;
	entry->tile_offset = (uint32_t)state->host_tiles.size();
	if ( kind == 0u )
		return;
	PocBundleMakeTensor(&state->families[0],layer,0u,&tensors[tensor_count++]);
	PocBundleMakeTensor(&state->families[0],layer,1u,&tensors[tensor_count++]);
	PocBundleMakeTensor(&state->families[kind],aux_launch,0u,
		&tensors[tensor_count++]);
	PocBundleMakeTensor(&state->families[kind],aux_launch,1u,
		&tensors[tensor_count++]);
	if ( kind == 1u )
	{
		PocBundleMakeTensor(&state->families[3],aux_launch,0u,
			&tensors[tensor_count++]);
		PocBundleMakeTensor(&state->families[3],aux_launch,1u,
			&tensors[tensor_count++]);
	}
	PocBundleAppendRoundRobin(&state->host_tiles,tensors,tensor_count);
	entry->tile_count = (uint32_t)state->host_tiles.size() - entry->tile_offset;
}

static int32_t PocBundleBuildLayers(PocBundleState *state)
{
	uint32_t layer,csa = 0u,hca = 0u,kind,aux;
	for (layer=0u; layer<POC_BUNDLE_LAYER_COUNT; layer++)
	{
		kind = layer < 2u ? 0u : ((layer & 1u) == 0u ? 1u : 2u);
		aux = kind == 1u ? csa++ : (kind == 2u ? hca++ : 0u);
		PocBundleBuildLayer(state,layer,kind,aux);
	}
	if ( csa != 21u || hca != 20u || state->host_tiles.size() != 2144u )
		return(-1);
	if ( cudaMalloc(&state->device_tiles,state->host_tiles.size() *
		sizeof(PocBundleTile)) != cudaSuccess )
		return(-2);
	return(cudaMemcpy(state->device_tiles,state->host_tiles.data(),
		state->host_tiles.size() * sizeof(PocBundleTile),
		cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -3);
}

static cudaError_t PocBundleLaunchFp8(PocBundleState *state,uint32_t launch,
	const void *input,cudaStream_t stream)
{
	PocFamily *family = &state->families[0];
	return(SparkLmHostLaunchSm121DecodeLinearPair<128u,
		SPARK_ACTIVATION_CODEC_NONE>(stream,PocBundlePayload(family,launch,0u),
		PocBundleScale(family,launch,0u),family->first_output,
		PocBundlePayload(family,launch,1u),PocBundleScale(family,launch,1u),
		family->second_output,input,family->first_output_bf16,
		family->second_output_bf16,1u,POC_HIDDEN));
}

static cudaError_t PocBundleLaunchBf16(PocBundleState *state,uint32_t family_id,
	uint32_t launch,const void *input,cudaStream_t stream)
{
	PocFamily *family = &state->families[family_id];
	return(SparkLmHostLaunchBf16LinearPair(stream,
		PocBundlePayload(family,launch,0u),PocBundlePayload(family,launch,1u),
		input,family->first_output_bf16,family->second_output_bf16,1u,
		POC_HIDDEN,family->first_output));
}

static cudaError_t PocBundleLaunchCandidate(PocBundleState *state,
	uint32_t layer,uint32_t policy,cudaStream_t stream)
{
	PocBundleLayer *entry = &state->layers[layer];
	const void *input = state->families[0].inputs +
		(uint64_t)layer * POC_HIDDEN;
	uint32_t blocks;
	if ( entry->kind == 0u )
		return(PocBundleLaunchFp8(state,layer,input,stream));
	blocks = std::min(PocBundleGrids[policy],entry->tile_count);
	PocProjectionBundleKernel<<<blocks,POC_BUNDLE_TILE_WARPS *
		SPARK_LM_WARP_LANES,POC_HIDDEN * sizeof(float),stream>>>(
		state->device_tiles + entry->tile_offset,entry->tile_count,input);
	return(cudaGetLastError());
}

static cudaError_t PocBundleLaunchProduction(PocBundleState *state,
	uint32_t layer)
{
	PocBundleLayer *entry = &state->layers[layer];
	const void *input = state->families[0].inputs +
		(uint64_t)layer * POC_HIDDEN;
	cudaError_t error;
	if ( entry->kind != 0u )
	{
		error = PocBundleLaunchBf16(state,entry->kind,entry->aux_launch,input,
			state->aux0);
		if ( error != cudaSuccess )
			return(error);
	}
	if ( entry->kind == 1u )
	{
		error = PocBundleLaunchBf16(state,3u,entry->aux_launch,input,state->aux1);
		if ( error != cudaSuccess )
			return(error);
	}
	return(PocBundleLaunchFp8(state,layer,input,state->primary));
}

static int32_t PocBundleCreateEvent(cudaEvent_t *event)
{
	return(cudaEventCreateWithFlags(event,cudaEventDisableTiming) == cudaSuccess ?
		0 : -1);
}

static int32_t PocBundleCreateEvents(PocBundleEvents *events)
{
	uint32_t layer;
	if ( PocBundleCreateEvent(&events->kickoff) < 0 ||
		PocBundleCreateEvent(&events->aux0_final) < 0 ||
		PocBundleCreateEvent(&events->aux1_final) < 0 )
		return(-1);
	for (layer=0u; layer<POC_BUNDLE_LAYER_COUNT; layer++)
		if ( PocBundleCreateEvent(&events->aux0_done[layer]) < 0 ||
			PocBundleCreateEvent(&events->aux1_done[layer]) < 0 ||
			PocBundleCreateEvent(&events->layer_done[layer]) < 0 )
			return(-2);
	return(0);
}

static int32_t PocBundleCaptureProduction(PocBundleState *state,
	PocBundleGraph *bundle_graph)
{
	PocBundleEvents *events = &state->events;
	cudaError_t error;
	uint32_t layer;
	if ( cudaStreamBeginCapture(state->primary,cudaStreamCaptureModeGlobal) !=
		cudaSuccess || cudaEventRecord(events->kickoff,state->primary) !=
		cudaSuccess || cudaStreamWaitEvent(state->aux0,events->kickoff,0u) !=
		cudaSuccess || cudaStreamWaitEvent(state->aux1,events->kickoff,0u) !=
		cudaSuccess )
		return(-1);
	for (layer=0u; layer<POC_BUNDLE_LAYER_COUNT; layer++)
	{
		error = PocBundleLaunchProduction(state,layer);
		if ( error != cudaSuccess ||
			cudaEventRecord(events->aux0_done[layer],state->aux0) != cudaSuccess ||
			cudaEventRecord(events->aux1_done[layer],state->aux1) != cudaSuccess ||
			cudaStreamWaitEvent(state->primary,events->aux0_done[layer],0u) !=
			cudaSuccess || cudaStreamWaitEvent(state->primary,
			events->aux1_done[layer],0u) != cudaSuccess ||
			cudaEventRecord(events->layer_done[layer],state->primary) !=
			cudaSuccess || cudaStreamWaitEvent(state->aux0,
			events->layer_done[layer],0u) != cudaSuccess ||
			cudaStreamWaitEvent(state->aux1,events->layer_done[layer],0u) !=
			cudaSuccess )
			return(-2);
	}
	if ( cudaEventRecord(events->aux0_final,state->aux0) != cudaSuccess ||
		cudaEventRecord(events->aux1_final,state->aux1) != cudaSuccess ||
		cudaStreamWaitEvent(state->primary,events->aux0_final,0u) != cudaSuccess ||
		cudaStreamWaitEvent(state->primary,events->aux1_final,0u) != cudaSuccess ||
		cudaStreamEndCapture(state->primary,&bundle_graph->graph) != cudaSuccess )
		return(-3);
	return(cudaGraphInstantiate(&bundle_graph->executable,bundle_graph->graph,
		0ull) == cudaSuccess ? 0 : -4);
}

static int32_t PocBundleCaptureCandidate(PocBundleState *state,uint32_t policy,
	PocBundleGraph *bundle_graph)
{
	uint32_t layer;
	if ( cudaStreamBeginCapture(state->primary,cudaStreamCaptureModeGlobal) !=
		cudaSuccess )
		return(-1);
	for (layer=0u; layer<POC_BUNDLE_LAYER_COUNT; layer++)
		if ( PocBundleLaunchCandidate(state,layer,policy,state->primary) !=
			cudaSuccess )
			return(-2);
	if ( cudaStreamEndCapture(state->primary,&bundle_graph->graph) != cudaSuccess )
		return(-3);
	return(cudaGraphInstantiate(&bundle_graph->executable,bundle_graph->graph,
		0ull) == cudaSuccess ? 0 : -4);
}

static int32_t PocBundleCopyOutput(const PocFamily *family,
	std::vector<uint16_t> *output,uint32_t *offset)
{
	uint32_t first = family->first_output,second = family->second_output;
	if ( cudaMemcpy(output->data() + *offset,family->first_output_bf16,
		first * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-1);
	*offset += first;
	if ( cudaMemcpy(output->data() + *offset,family->second_output_bf16,
		second * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-2);
	*offset += second;
	return(0);
}

static int32_t PocBundleCopyLayerOutputs(PocBundleState *state,uint32_t layer,
	std::vector<uint16_t> *output)
{
	PocBundleLayer *entry = &state->layers[layer];
	uint32_t count = 384u,offset = 0u;
	if ( entry->kind == 1u )
		count += 640u;
	else if ( entry->kind == 2u )
		count += 256u;
	output->resize(count);
	if ( PocBundleCopyOutput(&state->families[0],output,&offset) < 0 )
		return(-1);
	if ( entry->kind != 0u && PocBundleCopyOutput(
		&state->families[entry->kind],output,&offset) < 0 )
		return(-2);
	if ( entry->kind == 1u && PocBundleCopyOutput(&state->families[3],output,
		&offset) < 0 )
		return(-3);
	return(offset == count ? 0 : -4);
}

static int32_t PocBundleRunProductionLayer(PocBundleState *state,
	uint32_t layer)
{
	if ( PocBundleLaunchProduction(state,layer) != cudaSuccess ||
		cudaDeviceSynchronize() != cudaSuccess )
		return(-1);
	return(0);
}

static int32_t PocBundleCheckExact(PocBundleState *state,
	PocBundleResults *results)
{
	std::vector<uint16_t> reference,candidate;
	uint32_t seed,layer,policy;
	results->exact_digest = UINT64_C(1469598103934665603);
	for (seed=0u; seed<POC_BUNDLE_EXACT_SEEDS; seed++)
	{
		PocFillInputs<<<((uint64_t)POC_BUNDLE_LAYER_COUNT * POC_HIDDEN + 255u) /
			256u,256u,0u,state->primary>>>(state->families[0].inputs,
			POC_BUNDLE_LAYER_COUNT,401u + seed * 977u);
		if ( cudaStreamSynchronize(state->primary) != cudaSuccess )
			return(-1);
		for (layer=2u; layer<POC_BUNDLE_LAYER_COUNT; layer++)
		{
			if ( PocBundleRunProductionLayer(state,layer) < 0 ||
				PocBundleCopyLayerOutputs(state,layer,&reference) < 0 )
				return(-2);
			results->exact_digest = PocFnvBytes((const uint8_t *)reference.data(),
				reference.size() * sizeof(uint16_t),results->exact_digest);
			for (policy=0u; policy<POC_BUNDLE_POLICY_COUNT; policy++)
			{
				if ( PocBundleLaunchCandidate(state,layer,policy,state->primary) !=
					cudaSuccess || cudaStreamSynchronize(state->primary) != cudaSuccess ||
					PocBundleCopyLayerOutputs(state,layer,&candidate) < 0 )
					return(-3);
				if ( reference.size() != candidate.size() || memcmp(reference.data(),
					candidate.data(),reference.size() * sizeof(uint16_t)) != 0 )
					return(-4);
			}
		}
	}
	results->bitwise_exact = 1u;
	return(0);
}

static int32_t PocBundleTimeGraph(cudaGraphExec_t executable,
	cudaStream_t stream,cudaEvent_t start,cudaEvent_t stop,float *milliseconds)
{
	float elapsed;
	uint32_t repeat;
	if ( cudaEventRecord(start,stream) != cudaSuccess )
		return(-1);
	for (repeat=0u; repeat<POC_BUNDLE_CHAIN_REPEATS; repeat++)
		if ( cudaGraphLaunch(executable,stream) != cudaSuccess )
			return(-2);
	if ( cudaEventRecord(stop,stream) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess ||
		cudaEventElapsedTime(&elapsed,start,stop) != cudaSuccess )
		return(-3);
	*milliseconds = elapsed / (float)POC_BUNDLE_CHAIN_REPEATS;
	return(0);
}

static int32_t PocBundleDiscovery(PocBundleState *state,PocBundleGraph *control,
	PocBundleGraph *candidates,PocBundleResults *results,cudaEvent_t start,
	cudaEvent_t stop)
{
	float best = 1e30f,median;
	uint32_t round,position,policy,first,warmup;
	for (policy=0u; policy<POC_BUNDLE_POLICY_COUNT; policy++)
		for (warmup=0u; warmup<2u; warmup++)
			if ( cudaGraphLaunch(candidates[policy].executable,state->primary) !=
				cudaSuccess )
				return(-1);
	if ( cudaStreamSynchronize(state->primary) != cudaSuccess )
		return(-2);
	for (round=0u; round<POC_BUNDLE_DISCOVERY_ROUNDS; round++)
	{
		PocFillInputs<<<((uint64_t)POC_BUNDLE_LAYER_COUNT * POC_HIDDEN + 255u) /
			256u,256u,0u,state->primary>>>(state->families[0].inputs,
			POC_BUNDLE_LAYER_COUNT,5003u + round * 101u);
		for (position=0u; position<POC_BUNDLE_POLICY_COUNT; position++)
		{
			policy = (position * 5u + round * 2u) % POC_BUNDLE_POLICY_COUNT;
			first = (round + policy) & 1u;
			if ( PocBundleTimeGraph(first == 0u ? control->executable :
				candidates[policy].executable,state->primary,start,stop,
				first == 0u ? &results->discovery_control[policy][round] :
				&results->discovery_candidate[policy][round]) < 0 ||
				PocBundleTimeGraph(first == 0u ? candidates[policy].executable :
				control->executable,state->primary,start,stop,first == 0u ?
				&results->discovery_candidate[policy][round] :
				&results->discovery_control[policy][round]) < 0 )
				return(-3);
		}
	}
	for (policy=0u; policy<POC_BUNDLE_POLICY_COUNT; policy++)
	{
		median = PocMedian(results->discovery_candidate[policy],
			POC_BUNDLE_DISCOVERY_ROUNDS);
		if ( median < best )
		{
			best = median;
			results->selected_policy = policy;
		}
	}
	return(0);
}

static int32_t PocBundleValidation(PocBundleState *state,
	PocBundleGraph *control,PocBundleGraph *candidate,PocBundleResults *results,
	cudaEvent_t start,cudaEvent_t stop)
{
	uint32_t round,first;
	for (round=0u; round<POC_BUNDLE_VALIDATION_ROUNDS; round++)
	{
		PocFillInputs<<<((uint64_t)POC_BUNDLE_LAYER_COUNT * POC_HIDDEN + 255u) /
			256u,256u,0u,state->primary>>>(state->families[0].inputs,
			POC_BUNDLE_LAYER_COUNT,9001u + round * 313u);
		first = round & 1u;
		if ( PocBundleTimeGraph(first == 0u ? control->executable :
			candidate->executable,state->primary,start,stop,first == 0u ?
			&results->validation_control[round] :
			&results->validation_candidate[round]) < 0 ||
			PocBundleTimeGraph(first == 0u ? candidate->executable :
			control->executable,state->primary,start,stop,first == 0u ?
			&results->validation_candidate[round] :
			&results->validation_control[round]) < 0 )
			return(-1);
		if ( results->validation_candidate[round] <
			results->validation_control[round] )
			results->validation_wins++;
	}
	results->accepted = results->bitwise_exact != 0u &&
		PocMedian(results->validation_candidate,POC_BUNDLE_VALIDATION_ROUNDS) <
		PocMedian(results->validation_control,POC_BUNDLE_VALIDATION_ROUNDS) &&
		results->validation_wins >= 11u ? 1u : 0u;
	return(0);
}

static void PocBundlePrintDiscovery(FILE *file,const PocBundleResults *results)
{
	uint32_t policy;
	for (policy=0u; policy<POC_BUNDLE_POLICY_COUNT; policy++)
	{
		fprintf(file,"%s\"grid_%u\":{\"control_round_ms\":",
			policy == 0u ? "" : ",",PocBundleGrids[policy]);
		PocPrintArray(file,results->discovery_control[policy],
			POC_BUNDLE_DISCOVERY_ROUNDS);
		fprintf(file,",\"candidate_round_ms\":");
		PocPrintArray(file,results->discovery_candidate[policy],
			POC_BUNDLE_DISCOVERY_ROUNDS);
		fprintf(file,",\"control_median_ms\":%.9f,\"candidate_median_ms\":%.9f}",
			PocMedian(results->discovery_control[policy],
				POC_BUNDLE_DISCOVERY_ROUNDS),
			PocMedian(results->discovery_candidate[policy],
				POC_BUNDLE_DISCOVERY_ROUNDS));
	}
}

static int32_t PocBundlePrintReceipt(const char *path,const char *pack_path,
	const char *source_sha,const char *binary_sha,const PocPackHeader *header,
	const cudaDeviceProp *properties,const PocBundleState *state,
	const PocBundleResults *results)
{
	FILE *file = fopen(path,"w");
	float control,candidate;
	uint32_t family;
	if ( file == 0 )
		return(-1);
	control = PocMedian(results->validation_control,
		POC_BUNDLE_VALIDATION_ROUNDS);
	candidate = PocMedian(results->validation_candidate,
		POC_BUNDLE_VALIDATION_ROUNDS);
	fprintf(file,"{\n  \"schema\":\"dsv4_b1_projection_bundle_fast_gate_v1\",\n");
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
	fprintf(file,"\"first_layer\":%u,\"layer_count\":%u,\"tp_degree\":4,",
		header->first_layer_index,header->layer_count);
	fprintf(file,"\"tp_rank_slice\":%u,\"payload_source\":",POC_TP_RANK);
	fprintf(file,"\"actual_ga_stagepack_rows_cycled_to_true_43_layer_shape\"},\n");
	fprintf(file,"  \"source_families\":[");
	for (family=0u; family<POC_FAMILY_COUNT; family++)
		fprintf(file,"%s{\"name\":\"%s\",\"actual_pairs\":%u,",
			family == 0u ? "" : ",",state->families[family].name,
			state->families[family].source_count),
		fprintf(file,"\"source_fnv64\":\"%016llx\"}",
			(unsigned long long)state->families[family].source_digest);
	fprintf(file,"],\n  \"shape\":{\"layers\":43,\"swa\":2,\"csa\":21,");
	fprintf(file,"\"hca\":20,\"csa_outputs\":[384,512,128],");
	fprintf(file,"\"hca_outputs\":[384,256],\"input_dimension\":4096,");
	fprintf(file,"\"candidate_tiles\":%llu},\n",
		(unsigned long long)state->host_tiles.size());
	fprintf(file,"  \"method\":{\"control\":");
	fprintf(file,"\"accepted_three_stream_production_pair_kernels_with_per_layer_event_join_in_one_cuda_graph\"");
	fprintf(file,",\"candidate\":");
	fprintf(file,"\"one_16_warp_static_schedule_persistent_multi_tensor_kernel_per_non_swa_layer\"");
	fprintf(file,",\"arithmetic\":");
	fprintf(file,"\"unchanged_per_neuron_fp32_fma_and_warp_reduction_bf16_spine_no_quantization\"");
	fprintf(file,",\"working_set\":\"full_rank_local_model_projection_weights_exceed_l2\",");
	fprintf(file,"\"chain_repeats_per_sample\":%u,",POC_BUNDLE_CHAIN_REPEATS);
	fprintf(file,"\"discovery_rounds\":%u,\"validation_rounds\":%u,",
		POC_BUNDLE_DISCOVERY_ROUNDS,POC_BUNDLE_VALIDATION_ROUNDS);
	fprintf(file,"\"exact_seeds\":%u,\"acceptance\":",POC_BUNDLE_EXACT_SEEDS);
	fprintf(file,"\"positive_validation_median_and_at_least_11_of_15_paired_wins_no_minimum_gain\"},\n");
	fprintf(file,"  \"exact\":{\"bitwise\":%s,\"digest_fnv64\":\"%016llx\"},\n",
		results->bitwise_exact != 0u ? "true" : "false",
		(unsigned long long)results->exact_digest);
	fprintf(file,"  \"discovery\":{");
	PocBundlePrintDiscovery(file,results);
	fprintf(file,"},\n  \"selected\":{\"grid_ctas\":%u},\n",
		PocBundleGrids[results->selected_policy]);
	fprintf(file,"  \"validation\":{\"control_round_ms\":");
	PocPrintArray(file,results->validation_control,
		POC_BUNDLE_VALIDATION_ROUNDS);
	fprintf(file,",\"candidate_round_ms\":");
	PocPrintArray(file,results->validation_candidate,
		POC_BUNDLE_VALIDATION_ROUNDS);
	fprintf(file,",\"control_median_ms\":%.9f,\"candidate_median_ms\":%.9f,",
		control,candidate);
	fprintf(file,"\"gain_percent\":%.9f,\"candidate_wins\":%u,",
		100.0f * (control / candidate - 1.0f),results->validation_wins);
	fprintf(file,"\"round_count\":%u,\"accepted\":%s}\n}\n",
		POC_BUNDLE_VALIDATION_ROUNDS,results->accepted != 0u ? "true" : "false");
	fclose(file);
	return(0);
}

static int32_t PocBundleLoadState(const char *pack_path,PocBundleState *state,
	PocPackHeader *header,std::vector<PocPackEntry> *entries,int32_t *fd)
{
	uint32_t family;
	memset(state->layers,0,sizeof(state->layers));
	memset(&state->events,0,sizeof(state->events));
	state->device_tiles = 0;
	state->primary = 0;
	state->aux0 = 0;
	state->aux1 = 0;
	state->host_tiles.clear();
	PocInitializeFamilies(state->families);
	if ( PocLoadPack(pack_path,header,entries,fd) < 0 ||
		cudaStreamCreateWithFlags(&state->primary,cudaStreamNonBlocking) !=
		cudaSuccess || cudaStreamCreateWithFlags(&state->aux0,
		cudaStreamNonBlocking) != cudaSuccess || cudaStreamCreateWithFlags(
		&state->aux1,cudaStreamNonBlocking) != cudaSuccess )
		return(-1);
	for (family=0u; family<POC_FAMILY_COUNT; family++)
	{
		fprintf(stderr,"load=%s\n",state->families[family].name);
		if ( PocLoadFamily(*fd,*entries,&state->families[family]) < 0 )
			return(-2);
	}
	if ( PocBundleBuildLayers(state) < 0 || PocBundleCreateEvents(
		&state->events) < 0 )
		return(-3);
	return(0);
}

int main(int argc,char **argv)
{
	PocPackHeader header;
	std::vector<PocPackEntry> entries;
	PocBundleState state;
	PocBundleGraph control,candidates[POC_BUNDLE_POLICY_COUNT];
	PocBundleResults results;
	cudaDeviceProp properties;
	cudaEvent_t start,stop;
	int32_t fd,error;
	uint32_t policy;
	if ( argc != 5 )
		return(1);
	memset(&control,0,sizeof(control));
	memset(candidates,0,sizeof(candidates));
	memset(&results,0,sizeof(results));
	if ( PocBundleLoadState(argv[1],&state,&header,&entries,&fd) < 0 ||
		cudaGetDeviceProperties(&properties,0) != cudaSuccess ||
		cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) !=
		cudaSuccess )
		return(2);
	fprintf(stderr,"exact=all_non_swa_layers_all_policies\n");
	error = PocBundleCheckExact(&state,&results);
	if ( error < 0 )
	{
		fprintf(stderr,"exact_failure=%d cuda=%s\n",error,
			cudaGetErrorString(cudaGetLastError()));
		return(3);
	}
	fprintf(stderr,"capture=production\n");
	if ( PocBundleCaptureProduction(&state,&control) < 0 )
		return(4);
	for (policy=0u; policy<POC_BUNDLE_POLICY_COUNT; policy++)
	{
		fprintf(stderr,"capture=candidate_grid_%u\n",PocBundleGrids[policy]);
		if ( PocBundleCaptureCandidate(&state,policy,&candidates[policy]) < 0 )
			return(5);
	}
	fprintf(stderr,"discovery=full_43_layer_graph\n");
	if ( PocBundleDiscovery(&state,&control,candidates,&results,start,stop) < 0 )
		return(6);
	fprintf(stderr,"validation=grid_%u\n",
		PocBundleGrids[results.selected_policy]);
	if ( PocBundleValidation(&state,&control,
		&candidates[results.selected_policy],&results,start,stop) < 0 )
		return(7);
	if ( PocBundlePrintReceipt(argv[2],argv[1],argv[3],argv[4],&header,
		&properties,&state,&results) < 0 )
		return(8);
	close(fd);
	return(0);
}
