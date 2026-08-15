// K3 resident decode stage runner: embed -> slice -> head over the dispatch,
// with the TP4 all-reduce wiring (see the header for the ownership layout).
//
// The slice's layer_collective hook fires after every layer and all-reduces
// the input-sharded projection destinations the layer wrote WITHOUT the
// partial fold (tp_sharded): attention_out (kda_out / mla_out), hidden
// (routed_up / dense_down) and shared_out (shared_w2). The attention fold
// keeps the fused path's rule - SET at a block boundary (the slice already
// set the partial to the LOCAL attention output there, and the summed value
// replaces it), ADD otherwise; the MLP folds are ALWAYS adds, because the
// boundary restart already happened with the attention output and the MoE
// half accumulates on top of it. The embedding (stage 0) and the head
// candidates (last stage) use the same ring collective; the head exchange
// is a slot-encoded SUM - each rank fills only its slot, so the sum over
// the rank slots IS the all-gather of the four local argmaxes and the
// winner reduces locally.

#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"

typedef struct SparkK3RunnerState
{
	SparkK3ModuleState module;
	SparkK3Dispatch dispatch;
	SparkTpCollective collective;
	int collective_created;
	uint32_t rows;              /* the step in flight, for the hook */
	const uint16_t *embed_weight;
	const uint16_t *head_norm_weight;
	const uint16_t *head_weight;
	uint32_t vocab;
	uint32_t vocab_slice_rows;
	/* host staging for the BF16 collective: values + scratch */
	uint16_t *staging_values;
	uint16_t *staging_scratch;
	uint32_t staging_capacity;
	/* head candidate slot exchange: rows x (2 * tp_degree) floats */
	float *head_slots_host;
	float *head_slots_device;
	uint32_t head_slots_capacity;
	/* per-step device arrays the dispatch step consumes */
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *head_candidate_token;
	float *head_candidate_score;
	uint32_t head_tiles;
	uint32_t *output_token;
	float *output_score;
	uint32_t *output_token_host;
	float *output_score_host;
	cudaStream_t stream;
	uint32_t max_rows;
	uint32_t max_context;
	uint32_t multiprocessors;
} SparkK3RunnerState;

static uint32_t K3RunnerFirstLayer(uint32_t stage_index)
{
	static const uint32_t first[4] = { 0u, 24u, 47u, 70u };
	return(first[stage_index % 4u]);
}

static uint32_t K3RunnerLayerCount(uint32_t stage_index)
{
	static const uint32_t count[4] = { 24u, 23u, 23u, 23u };
	return(count[stage_index % 4u]);
}

// One BF16 all-reduce of a rows x K3_HIDDEN device tensor: sync the stream,
// stage to the host, all-reduce in place, upload. The sync-per-projection is
// the host-collective tier's known cost; the device-direct tier replaces it
// without changing this file's contract.
static SparkStatus K3RunnerReduceBf16(SparkK3RunnerState *state, cudaStream_t stream,
	const uint16_t *device_values, uint32_t rows)
{
	SparkStatus status;
	uint32_t elements = rows * K3_HIDDEN;
	if ( state->collective_created == 0 )
		return SPARK_STATUS_OK;
	cudaError_t error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	error = cudaMemcpy(state->staging_values, device_values,
		(uint64_t)elements * 2u, cudaMemcpyDeviceToHost);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = SparkTpCollectiveAllReduceSumBf16(&state->collective,
		state->staging_values, elements, state->staging_scratch);
	if ( status != SPARK_STATUS_OK )
		return status;
	error = cudaMemcpy((void *)device_values, state->staging_values,
		(uint64_t)elements * 2u, cudaMemcpyHostToDevice);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

// The slice's per-layer hook. Attention_out for every layer (kda_out at
// KDA layers, mla_out at MLA), hidden for the MoE routed_up (and layer 0's
// dense_down), shared_out for the MoE shared_w2.
static void K3RunnerLayerCollective(void *context, void *stream_void, uint32_t layer)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)context;
	K3LayerBuffers *b = state->dispatch.buffers;
	cudaStream_t stream = (cudaStream_t)stream_void;
	uint32_t rows = state->rows;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u;
	if ( K3RunnerReduceBf16(state, stream, b->attention_out_bf16, rows) != SPARK_STATUS_OK )
		return;
	if ( boundary != 0u )
		K3PartialSet(b, b->attention_out_bf16, rows, stream);
	else
		K3PartialAdd(b, b->attention_out_bf16, rows, stream);
	if ( K3RunnerReduceBf16(state, stream, b->hidden_bf16, rows) != SPARK_STATUS_OK )
		return;
	K3PartialAdd(b, b->hidden_bf16, rows, stream);
	if ( layer == 0u )
		return;
	if ( K3RunnerReduceBf16(state, stream, b->shared_out_bf16, rows) != SPARK_STATUS_OK )
		return;
	K3PartialAdd(b, b->shared_out_bf16, rows, stream);
}
SparkStatus SparkK3StageRunnerInitialize(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerConfiguration *configuration)
{
	SparkK3RunnerState *state;
	SparkStatus status;
	SparkK3PackEntry entry;
	uint32_t routes;
	if ( runner == 0 || configuration == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( configuration->abi_version != SPARK_K3_STAGE_RUNNER_ABI_VERSION ||
		configuration->stage_index >= 4u ||
		configuration->stage_count != 4u ||
		configuration->rank_pack_path == 0 ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_input_row_count == 0u ||
		configuration->kv_page_bytes == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(runner, 0, sizeof(*runner));
	state = new SparkK3RunnerState;
	memset(state, 0, sizeof(*state));
	runner->abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->descriptor_bytes = SPARK_K3_STAGE_RUNNER_BYTES;
	runner->flags = configuration->flags;
	runner->stage_index = configuration->stage_index;
	runner->stage_count = configuration->stage_count;
	runner->tp_degree = configuration->tp_degree;
	runner->tp_rank = configuration->tp_rank;
	runner->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
	runner->owns_final_head = configuration->stage_index + 1u == configuration->stage_count ? 1u : 0u;
	runner->private_state = state;
	state->stream = (cudaStream_t)configuration->execution_stream;
	state->max_rows = configuration->max_input_row_count;
	state->max_context = configuration->resident_sequence_capacity;
	state->multiprocessors = configuration->multiprocessors;
	runner->stats.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->stats.descriptor_bytes = (uint32_t)sizeof(SparkK3StageRunnerStats);
	/* Pack + bind + pools + device objects. */
	status = SparkK3ModuleInitialize(&state->module,
		configuration->rank_pack_path,
		K3RunnerFirstLayer(configuration->stage_index),
		K3RunnerLayerCount(configuration->stage_index));
	if ( status != SPARK_STATUS_OK )
		{ delete state; return status; }
	if ( SparkK3DispatchCreate(&state->dispatch,&state->module.sizing,
		configuration->max_active_sequence_count,
		configuration->max_input_row_count,
		configuration->kv_pages_per_sequence,
		configuration->kv_page_bytes, 0) != SPARK_K3_DISPATCH_OK )
		{ SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	if ( SparkK3DispatchRegisterPack(&state->module.pack) != SPARK_K3_DISPATCH_OK ||
		SparkK3DispatchBindWeights(&state->dispatch,&state->module.pack,
			state->module.bound,state->module.bound_count) != SPARK_K3_DISPATCH_OK )
		{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	/* The page tables start all-zero (every position maps to physical page
	 * 0); the serving tier owns the real mappings and rewrites them before
	 * publishing a step, but an uninitialised table would be a wild read. */
	cudaMemset(state->dispatch.page_table, 0,
		(uint64_t)state->module.sizing.mla_layer_count *
		configuration->kv_pages_per_sequence * 4u);
	state->vocab = state->module.pack.config.vocab;
	/* The rank pack's embed/lm_head are ALREADY the rank's slice: the
	 * sharder row-split them by the PACK's tp degree, so the rank's rows
	 * come from the tensor's own shape, never vocab / run_degree (which
	 * double-divides for a TP4 pack run single-rank, or vice versa). */
	if ( SparkK3PackLoadEntry(&state->module.pack,"model.embed_tokens.weight",&entry) == 0 &&
		entry.shape_count >= 1u )
		state->vocab_slice_rows = entry.shape[0];
	else
		state->vocab_slice_rows = state->vocab;
	/* The TP contract: sharded ranks defer the partial epilogues to the hook. */
	state->dispatch.buffers->tp_sharded = configuration->tp_degree > 1u ? 1u : 0u;
	if ( configuration->tp_degree > 1u )
	{
		if ( configuration->tp_collective == 0 )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INVALID_ARGUMENT; }
		state->dispatch.slice_state->layer_collective = K3RunnerLayerCollective;
		state->dispatch.slice_state->collective_context = state;
		status = SparkTpCollectiveCreate(configuration->tp_collective,&state->collective);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return status; }
		state->collective_created = 1;
	}
	/* Stage 0 and the head stage need the model-level tensors. */
	if ( runner->owns_embedding != 0u &&
		SparkK3PackLoadEntry(&state->module.pack,"model.embed_tokens.weight",&entry) == 0 )
		state->embed_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
	if ( runner->owns_final_head != 0u )
	{
		if ( SparkK3PackLoadEntry(&state->module.pack,"model.norm.weight",&entry) == 0 )
			state->head_norm_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
		if ( SparkK3PackLoadEntry(&state->module.pack,"lm_head.weight",&entry) == 0 )
			state->head_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
	}
	/* Host staging + head slots + the per-step device arrays. */
	state->staging_capacity = configuration->max_input_row_count * K3_HIDDEN;
	state->staging_values = new uint16_t[state->staging_capacity];
	state->staging_scratch = new uint16_t[state->staging_capacity];
	state->head_slots_capacity = configuration->max_input_row_count * 2u * configuration->tp_degree;
	state->head_slots_host = new float[state->head_slots_capacity];
	cudaMalloc(&state->head_slots_device,(uint64_t)state->head_slots_capacity * 4u);
	routes = configuration->max_input_row_count * K3_TOP_K;
	cudaMalloc(&state->route_expert,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_packed_row,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_source_token,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_weight,(uint64_t)routes * 4u);
	cudaMalloc(&state->group_row_offset,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w1,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w2,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->dense_row_offset, 8u);
	cudaMalloc(&state->dense_tile_prefix, 8u);
	state->head_tiles = (state->vocab_slice_rows + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	cudaMalloc(&state->head_candidate_token,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->head_candidate_score,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->output_token,
		(uint64_t)configuration->max_input_row_count * 4u);
	cudaMalloc(&state->output_score,
		(uint64_t)configuration->max_input_row_count * 4u);
	state->output_token_host = new uint32_t[configuration->max_input_row_count];
	state->output_score_host = new float[configuration->max_input_row_count];
	return SPARK_STATUS_OK;
}

// The head's cross-rank argmax: the slot-encoded sum (see the header).
static SparkStatus K3RunnerHeadExchange(SparkK3RunnerState *state,
	uint32_t rows, uint32_t tp_degree, uint32_t tp_rank,
	uint32_t *output_token_ids, float *output_scores)
{
	uint32_t slots = rows * 2u * tp_degree;
	SparkStatus status;
	uint32_t row, rank_slot, best, r;
	if ( tp_degree <= 1u )
		return SPARK_STATUS_OK;
	memset(state->head_slots_host, 0, (uint64_t)slots * 4u);
	for ( row = 0u; row < rows; ++row )
	{
		rank_slot = tp_rank * 2u;
		state->head_slots_host[row * 2u * tp_degree + rank_slot] = output_scores[row];
		state->head_slots_host[row * 2u * tp_degree + rank_slot + 1u] = (float)output_token_ids[row];
	}
	/* The collective exchanges HOST buffers (the wire is TCP); the slot
	 * vector lives in host memory end to end. */
	status = SparkTpCollectiveAllReduceSumF32(&state->collective,
		state->head_slots_host, slots, (float *)state->staging_values);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( row = 0u; row < rows; ++row )
	{
		best = 0u;
		for ( r = 1u; r < tp_degree; ++r )
		{
			float bs = state->head_slots_host[row * 2u * tp_degree + best * 2u];
			float rs = state->head_slots_host[row * 2u * tp_degree + r * 2u];
			if ( rs > bs )
				best = r;
		}
		output_token_ids[row] = (uint32_t)state->head_slots_host[row * 2u * tp_degree + best * 2u + 1u];
		output_scores[row] = state->head_slots_host[row * 2u * tp_degree + best * 2u];
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerSubmit(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerDispatch *dispatch)
{
	SparkK3RunnerState *state;
	SparkK3StepInput in;
	K3LayerBuffers *b;
	cudaStream_t stream;
	uint32_t rows, sequences, packed_rows;
	uint32_t dense_offsets[2];
	int32_t status;
	SparkStatus exchange_status;
	SparkModelDriverCompletion completion;
	uint32_t *host_tokens;
	float *host_scores;
	uint32_t i;
	if ( runner == 0 || dispatch == 0 || runner->private_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	rows = dispatch->row_count;
	if ( rows == 0u || rows > state->max_rows ||
		(runner->owns_embedding != 0u && dispatch->token_ids == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	stream = state->stream;
	state->rows = rows;
	b = state->dispatch.buffers;
	sequences = rows;
	packed_rows = rows * K3_TOP_K;
	memset(&in, 0, sizeof(in));
	/* Stage 0 embeds; the others consume the transported hidden stream. */
	if ( runner->owns_embedding != 0u )
	{
		status = K3Embedding(state->embed_weight, dispatch->token_ids,
			b->hidden_bf16, rows, runner->tp_rank * state->vocab_slice_rows,
			state->vocab_slice_rows, stream);
		if ( status != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( K3RunnerReduceBf16(state, stream, b->hidden_bf16, rows) != SPARK_STATUS_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
	}
	else
	{
		if ( dispatch->hidden_input_bf16 == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		cudaMemcpy(b->hidden_bf16, dispatch->hidden_input_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	}
	in.hidden_in = b->hidden_bf16;
	in.positions = dispatch->positions;
	in.context_length = dispatch->context_length;
	in.sequence_of_row = dispatch->sequence_of_row;
	in.sequence_row_begin = 0; /* pure decode: row i is sequence i */
	in.kda_state_index = dispatch->kda_state_index;
	in.route_expert = state->route_expert;
	in.route_packed_row = state->route_packed_row;
	in.route_source_token = state->route_source_token;
	in.route_weight = state->route_weight;
	in.group_row_offset = state->group_row_offset;
	in.group_tile_prefix_w1 = state->group_tile_prefix_w1;
	in.group_tile_prefix_w2 = state->group_tile_prefix_w2;
	in.dense_row_offset = state->dense_row_offset;
	in.dense_tile_prefix = state->dense_tile_prefix;
	in.head_candidate_score = state->head_candidate_score;
	in.head_candidate_token = state->head_candidate_token;
	in.output_token = state->output_token;
	in.output_score = state->output_score;
	dense_offsets[0] = 0u;
	dense_offsets[1] = rows;
	cudaMemcpy(state->dense_row_offset, dense_offsets, 8u, cudaMemcpyHostToDevice);
	status = SparkK3DispatchStep(&state->dispatch, &in, rows, sequences,
		1u, packed_rows, state->max_context, state->multiprocessors, stream);
	if ( status != SPARK_K3_DISPATCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	/* The head stage commits the tokens. */
	if ( runner->owns_final_head != 0u )
	{
		status = K3Head(b, state->head_norm_weight, state->head_weight, 0,
			state->vocab_slice_rows, rows, stream);
		if ( status != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		cudaStreamSynchronize(stream);
		cudaMemcpy(state->output_token_host, state->output_token,
			(uint64_t)rows * 4u, cudaMemcpyDeviceToHost);
		cudaMemcpy(state->output_score_host, state->output_score,
			(uint64_t)rows * 4u, cudaMemcpyDeviceToHost);
		exchange_status = K3RunnerHeadExchange(state, rows, runner->tp_degree,
			runner->tp_rank, state->output_token_host, state->output_score_host);
		if ( exchange_status != SPARK_STATUS_OK )
			return exchange_status;
		if ( dispatch->output_token_ids != 0 )
			cudaMemcpy(dispatch->output_token_ids, state->output_token_host,
				(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
		if ( dispatch->output_scores != 0 )
			cudaMemcpy(dispatch->output_scores, state->output_score_host,
				(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
	}
	else if ( dispatch->hidden_output_bf16 != 0 )
	{
		cudaMemcpy(dispatch->hidden_output_bf16, b->hidden_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	}
	runner->stats.submitted_count++;
	runner->stats.completed_count++;
	if ( dispatch->completion_function != 0 )
	{
		if ( runner->owns_final_head == 0u )
		{
			memset(state->output_token_host, 0, (uint64_t)rows * 4u);
		}
		memset(&completion, 0, sizeof(completion));
		completion.request_id = dispatch->request_id;
		completion.sequence_id = dispatch->sequence_id;
		completion.sequence_position = dispatch->sequence_position;
		completion.accepted_token_count = rows;
		completion.token_count = rows;
		completion.tokens_per_sequence = 1u;
		for ( i = 0u; i < rows && i < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY; ++i )
			completion.token_ids[i] = state->output_token_host[i];
		completion.status = SPARK_STATUS_OK;
		dispatch->completion_function(dispatch->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerGetStats(
	const SparkK3StageRunner *runner,
	SparkK3StageRunnerStats *stats_out)
{
	if ( runner == 0 || stats_out == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	*stats_out = runner->stats;
	return SPARK_STATUS_OK;
}

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner)
{
	SparkK3RunnerState *state;
	if ( runner == 0 || runner->private_state == 0 )
		return;
	state = (SparkK3RunnerState *)runner->private_state;
	if ( state->collective_created != 0 )
		SparkTpCollectiveDestroy(&state->collective);
	SparkK3DispatchDestroy(&state->dispatch);
	/* The dispatch registered the pack mmap for UVA weight access; a
	 * re-initialise remaps (often the same address) and registering an
	 * already-registered region fails, so the runner unregisters before the
	 * munmap. */
	if ( state->module.pack.mapping != 0 )
		cudaHostUnregister((void *)state->module.pack.mapping);
	SparkK3ModuleDestroy(&state->module);
	delete[] state->staging_values;
	delete[] state->staging_scratch;
	delete[] state->head_slots_host;
	delete[] state->output_token_host;
	delete[] state->output_score_host;
	cudaFree(state->head_slots_device);
	cudaFree(state->route_expert);
	cudaFree(state->route_packed_row);
	cudaFree(state->route_source_token);
	cudaFree(state->route_weight);
	cudaFree(state->group_row_offset);
	cudaFree(state->group_tile_prefix_w1);
	cudaFree(state->group_tile_prefix_w2);
	cudaFree(state->dense_row_offset);
	cudaFree(state->dense_tile_prefix);
	cudaFree(state->head_candidate_token);
	cudaFree(state->head_candidate_score);
	cudaFree(state->output_token);
	cudaFree(state->output_score);
	delete state;
	runner->private_state = 0;
}
