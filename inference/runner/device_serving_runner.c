#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_sampling.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_runner_model.h"

typedef struct DeviceServing
{
	SparkDeviceStageRunner *device;
	const SparkStageRunnerModelInterface *model;
	uint32_t max_rows;
	uint32_t head_rows;
	SparkDeviceBuffer scratch;
	uint32_t *lane_slots;
	uint32_t *row_of;
	uint32_t *tokens;
	uint32_t *positions;
	uint32_t *slots;
	uint32_t *heads;
	uint32_t *head_of;
	uint32_t *output_tokens;
	float *output_scores;
	SparkSampleRow *rules;
} DeviceServing;

static SparkStatus ServingRefuse(SparkStatus status, const char *what)
{
	fprintf(stderr,"device_serving_runner: %s\n",what);
	return(status);
}

static DeviceServing *ServingOf(const SparkStageRunner *runner)
{
	return(runner != 0 ? (DeviceServing *)runner->private_state : 0);
}

uint32_t SparkStageRunnerImplementationFlags(void)
{
	return(SPARK_STAGE_RUNNER_FLAG_OWN_PACK);
}

static SparkStatus ServingScratch(DeviceServing *serving)
{
	const uint64_t rows = serving->max_rows;
	uint32_t *base;
	SparkStatus status = SparkDeviceAllocate(&serving->scratch,SPARK_DEVICE_SPACE_SHARED_COHERENT,rows * 9u * sizeof(uint32_t),
		"device serving rows");
	if ( status != SPARK_STATUS_OK )
		return(status);
	base = (uint32_t *)serving->scratch.host;
	serving->lane_slots = base;
	serving->row_of = base + rows;
	serving->tokens = base + 2u * rows;
	serving->positions = base + 3u * rows;
	serving->slots = base + 4u * rows;
	serving->heads = base + 5u * rows;
	serving->output_tokens = base + 6u * rows;
	serving->output_scores = (float *)(base + 7u * rows);
	serving->head_of = base + 8u * rows;
	serving->rules = (SparkSampleRow *)calloc(serving->head_rows,sizeof(*serving->rules));
	return(serving->rules != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkStageRunnerInitialize(SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration,
	const SparkStageRunnerModelInterface *model)
{
	SparkDeviceStageRunnerConfig config;
	DeviceServing *serving;
	SparkStatus status;
	if ( runner == 0 || configuration == 0 || model == 0 || configuration->abi_version != SPARK_STAGE_RUNNER_ABI_VERSION ||
		configuration->rank_pack_path == 0 || configuration->max_input_row_count == 0u || configuration->resident_sequence_capacity == 0u ||
		configuration->kv_pages_per_sequence == 0u )
		return(ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a device stage serves one pack with rows, resident sequences and KV pages"));
	if ( configuration->tp_degree > 1u || configuration->stage_count > 1u )
		return(ServingRefuse(SPARK_STATUS_UNSUPPORTED,"a device stage serves the whole model on one device"));
	serving = (DeviceServing *)calloc(1u,sizeof(*serving));
	if ( serving == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&config,0,sizeof(config));
	config.pack_path = configuration->rank_pack_path;
	config.model = model;
	config.max_rows = configuration->max_input_row_count;
	config.sequences = configuration->resident_sequence_capacity;
	config.max_context = configuration->kv_pages_per_sequence * SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	config.head_rows = configuration->head_row_capacity != 0u ? configuration->head_row_capacity : configuration->max_input_row_count;
	config.state_budget_bytes = configuration->state_budget_bytes;
	config.linear_weight_codec = configuration->linear_weight_codec;
	config.output_vocabulary = configuration->output_vocabulary;
	config.external_kv = 1u;
	serving->model = model;
	serving->max_rows = config.max_rows;
	serving->head_rows = config.head_rows;
	status = SparkDeviceStageRunnerOpen(&serving->device,&config);
	if ( status == SPARK_STATUS_OK )
		status = ServingScratch(serving);
	if ( status != SPARK_STATUS_OK )
	{
		if ( serving->device != 0 )
			SparkDeviceStageRunnerClose(serving->device);
		free(serving);
		return(status);
	}
	memset(runner,0,sizeof(*runner));
	runner->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	runner->descriptor_bytes = SPARK_STAGE_RUNNER_BYTES;
	runner->flags = SparkStageRunnerImplementationFlags();
	runner->stage_count = 1u;
	runner->tp_degree = 1u;
	runner->owns_embedding = 1u;
	runner->owns_final_head = 1u;
	runner->private_state = serving;
	return(SPARK_STATUS_OK);
}

static SparkStatus ServingRead(SparkDeviceQueue *queue, uint32_t *host_view, const SparkDeviceBuffer *scratch, const void *address, uint64_t bytes)
{
	SparkDeviceRange source, destination;
	SparkStatus status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)address,bytes,&source);
	destination.buffer = scratch;
	destination.offset = (uint64_t)((uint8_t *)host_view - (uint8_t *)scratch->host);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceCopy(queue,destination,source,bytes);
	return(status == SPARK_STATUS_OK ? SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) : status);
}

static SparkStatus ServingWrite(SparkDeviceQueue *queue, void *address, const SparkDeviceBuffer *scratch, const void *host_view, uint64_t bytes)
{
	SparkDeviceRange source, destination;
	SparkStatus status;
	if ( address == 0 )
		return(SPARK_STATUS_OK);
	status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)address,bytes,&destination);
	source.buffer = scratch;
	source.offset = (uint64_t)((const uint8_t *)host_view - (const uint8_t *)scratch->host);
	return(status == SPARK_STATUS_OK ? SparkDeviceCopy(queue,destination,source,bytes) : status);
}

static SparkStatus ServingGroup(DeviceServing *serving, const SparkStageRunnerDispatch *dispatch, uint32_t prefill, uint32_t *head_count)
{
	const uint32_t rows = dispatch->row_count, lanes = dispatch->active_sequence_count;
	uint32_t lane, row, grouped = 0u;
	*head_count = 0u;
	for ( lane = 0u; lane < lanes; lane++ )
	{
		const uint32_t first = grouped;
		for ( row = 0u; row < rows; row++ )
			if ( dispatch->host_sequence_of_row[row] == serving->lane_slots[lane] )
			{
				serving->row_of[grouped] = row;
				serving->tokens[grouped] = dispatch->token_ids[row];
				serving->positions[grouped] = dispatch->host_positions[row];
				serving->slots[grouped] = serving->lane_slots[lane];
				if ( prefill == 0u )
					serving->heads[(*head_count)++] = grouped;
				grouped++;
			}
		if ( grouped == first )
			return(ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"an active sequence has no row in the dispatch"));
		if ( prefill != 0u )
			serving->heads[(*head_count)++] = grouped - 1u;
	}
	return(grouped == rows ? SPARK_STATUS_OK : ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a row belongs to no active sequence"));
}

static SparkStatus ServingRules(DeviceServing *serving, const SparkStageRunnerDispatch *dispatch, uint32_t head_count)
{
	uint32_t head, entry;
	for ( head = 0u; head < dispatch->row_count; head++ )
		serving->head_of[head] = UINT32_MAX;
	for ( head = 0u; head < head_count; head++ )
		serving->head_of[serving->row_of[serving->heads[head]]] = head;
	for ( entry = 0u; entry < dispatch->distribution_count; entry++ )
	{
		const SparkRowSampling *rule = &dispatch->distribution_rules[entry];
		const uint32_t row = dispatch->distribution_rows[entry];
		if ( row >= dispatch->row_count || serving->head_of[row] == UINT32_MAX )
			return(ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a sampled row is not a head row of the dispatch"));
		serving->rules[entry].seed = rule->seed;
		serving->rules[entry].inverse_temperature = rule->inverse_temperature;
		serving->rules[entry].nucleus = rule->nucleus;
		serving->rules[entry].top_k = rule->top_k;
		serving->rules[entry].logprobs = rule->logprobs;
		serving->rules[entry].head_row = serving->head_of[row];
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus ServingSample(DeviceServing *serving, const SparkStageRunnerDispatch *dispatch, uint32_t step, uint32_t *tokens)
{
	SparkDeviceStageRunnerSamples samples;
	uint32_t entry;
	SparkStatus status;
	for ( entry = 0u; entry < dispatch->distribution_count; entry++ )
		serving->rules[entry].position = dispatch->distribution_positions[entry] + step;
	status = SparkDeviceStageRunnerSample(serving->device,dispatch->distribution_count,serving->rules,&samples);
	for ( entry = 0u; status == SPARK_STATUS_OK && entry < dispatch->distribution_count; entry++ )
	{
		tokens[dispatch->distribution_rows[entry]] = samples.tokens[entry];
		if ( dispatch->distribution_logprobs != 0 && dispatch->distribution_rules[entry].logprobs != 0u )
			memcpy(&dispatch->distribution_logprobs[(uint64_t)entry * SPARK_SAMPLING_MAX_LOGPROBS],&samples.logprobs[(uint64_t)entry * SPARK_SAMPLE_MAX_LOGPROBS],
				(size_t)dispatch->distribution_rules[entry].logprobs * sizeof(SparkSamplingLogprob));
	}
	return(status);
}

static void ServingCompletion(const SparkStageRunnerDispatch *dispatch, const uint32_t *tokens, uint32_t rows)
{
	SparkModelDriverCompletion completion;
	uint32_t row;
	if ( dispatch->completion_function == 0 )
		return;
	memset(&completion,0,sizeof(completion));
	completion.request_id = dispatch->request_id;
	completion.sequence_id = dispatch->sequence_id;
	completion.sequence_position = dispatch->sequence_position;
	completion.accepted_token_count = rows;
	completion.token_count = rows;
	completion.tokens_per_sequence = 1u;
	for ( row = 0u; row < rows && row < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY; row++ )
		completion.token_ids[row] = tokens[row];
	completion.status = SPARK_STATUS_OK;
	dispatch->completion_function(dispatch->completion_context,&completion);
}

SparkStatus SparkStageRunnerSubmit(SparkStageRunner *runner, const SparkStageRunnerDispatch *dispatch)
{
	DeviceServing *serving = ServingOf(runner);
	SparkDeviceStageRunnerBatch batch;
	SparkDeviceStageRunnerResult result;
	SparkDeviceQueue *queue;
	uint32_t rows, steps, step, row, head, head_count = 0u, verify, prefill;
	SparkStatus status;
	if ( serving == 0 || dispatch == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows = dispatch->row_count;
	steps = dispatch->chain_steps > 1u ? dispatch->chain_steps : 1u;
	verify = (dispatch->flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY) != 0u ? 1u : 0u;
	prefill = (dispatch->flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( (dispatch->flags & ~SPARK_STAGE_RUNNER_DISPATCH_KNOWN_FLAGS) != 0u || (dispatch->flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER) != 0u ||
		dispatch->hidden_input_bf16 != 0 )
		return(ServingRefuse(SPARK_STATUS_UNSUPPORTED,"a device stage runs whole-model rows; deferred prefill and boundary input are not served"));
	if ( dispatch->distribution_count != 0u && (dispatch->distribution_rows == 0 || dispatch->distribution_positions == 0 ||
		dispatch->distribution_rules == 0 || dispatch->distribution_count > serving->head_rows) )
		return(ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"sampled rows name their rows, positions and rules, within the head capacity"));
	if ( rows == 0u || dispatch->active_sequence_count == 0u || dispatch->active_sequence_count > rows || (uint64_t)rows * steps > serving->max_rows ||
		dispatch->token_ids == 0 || dispatch->host_positions == 0 || dispatch->host_sequence_of_row == 0 || dispatch->recurrent_index == 0 ||
		(steps > 1u && (rows != dispatch->active_sequence_count || verify != 0u || prefill != 0u)) )
		return(ServingRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a dispatch names rows within capacity, its host row tables and its sequences; a chain decodes "
			"one row per sequence"));
	queue = SparkDeviceStageRunnerQueue(serving->device);
	status = ServingRead(queue,serving->lane_slots,&serving->scratch,dispatch->recurrent_index,(uint64_t)dispatch->active_sequence_count * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = ServingGroup(serving,dispatch,prefill,&head_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( head_count > serving->head_rows )
		return(ServingRefuse(SPARK_STATUS_CAPACITY_EXCEEDED,"the dispatch needs more head rows than the device stage holds"));
	status = ServingRules(serving,dispatch,head_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(serving->output_tokens,0,(size_t)rows * steps * sizeof(uint32_t));
	memset(serving->output_scores,0,(size_t)rows * steps * sizeof(float));
	memset(&batch,0,sizeof(batch));
	batch.rows = rows;
	batch.token_ids = serving->tokens;
	batch.positions = serving->positions;
	batch.sequence_of_row = serving->slots;
	batch.head_rows = head_count;
	batch.head_row_index = serving->heads;
	batch.verify = verify;
	for ( step = 0u; status == SPARK_STATUS_OK && step < steps; step++ )
	{
		uint32_t *tokens = serving->output_tokens + (uint64_t)step * rows;
		float *scores = serving->output_scores + (uint64_t)step * rows;
		const uint32_t *previous = step != 0u ? tokens - rows : tokens;
		for ( row = 0u; step != 0u && row < rows; row++ )
		{
			serving->tokens[row] = previous[serving->row_of[row]];
			serving->positions[row]++;
		}
		status = SparkDeviceStageRunnerStep(serving->device,&batch,&result);
		for ( head = 0u; status == SPARK_STATUS_OK && head < head_count; head++ )
		{
			tokens[serving->row_of[serving->heads[head]]] = result.argmax[head];
			scores[serving->row_of[serving->heads[head]]] = result.logprob[4u * head];
		}
		if ( status == SPARK_STATUS_OK && dispatch->distribution_count != 0u )
			status = ServingSample(serving,dispatch,step,tokens);
	}
	if ( status == SPARK_STATUS_OK )
		status = ServingWrite(queue,dispatch->output_token_ids,&serving->scratch,serving->output_tokens,(uint64_t)rows * steps * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = ServingWrite(queue,dispatch->output_scores,&serving->scratch,serving->output_scores,(uint64_t)rows * steps * sizeof(float));
	runner->stats.submitted_count++;
	if ( status != SPARK_STATUS_OK )
	{
		runner->stats.failed_count++;
		runner->stats.last_status = (uint32_t)status;
		return(status);
	}
	runner->stats.completed_count++;
	ServingCompletion(dispatch,serving->output_tokens,rows);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageRunnerResetSlots(SparkStageRunner *runner, const uint32_t *slots, uint32_t count)
{
	DeviceServing *serving = ServingOf(runner);
	SparkStatus status = serving != 0 && (count == 0u || slots != 0) ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
	uint32_t index;
	for ( index = 0u; status == SPARK_STATUS_OK && index < count; index++ )
		status = SparkDeviceStageRunnerResetSequence(serving->device,slots[index]);
	return(status);
}

SparkStatus SparkStageRunnerIdle(SparkStageRunner *runner)
{
	return(ServingOf(runner) != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkStageRunnerSessionReset(SparkStageRunner *runner)
{
	return(ServingOf(runner) != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkStageRunnerGetStats(const SparkStageRunner *runner, SparkStageRunnerStats *stats_out)
{
	if ( runner == 0 || stats_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*stats_out = runner->stats;
	return(SPARK_STATUS_OK);
}

void SparkStageRunnerDestroy(SparkStageRunner *runner)
{
	DeviceServing *serving = ServingOf(runner);
	if ( serving == 0 )
		return;
	if ( serving->scratch.handle != 0u )
		(void)SparkDeviceFree(&serving->scratch);
	free(serving->rules);
	if ( serving->device != 0 )
		SparkDeviceStageRunnerClose(serving->device);
	free(serving);
	runner->private_state = 0;
}

uint32_t SparkStageRunnerKvLayerCount(const SparkStageRunner *runner)
{
	const DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerGeometry(serving->device)->kv_layer_count : 0u);
}

SparkStatus SparkStageRunnerAttachKv(SparkStageRunner *runner, const SparkStageRunnerKv *kv)
{
	DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerAttachKv(serving->device,kv) : SPARK_STATUS_INVALID_ARGUMENT);
}

uint64_t SparkStageRunnerRecurrentBytes(const SparkStageRunner *runner)
{
	const DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerGeometry(serving->device)->recurrent_bytes : 0u);
}

uint32_t SparkStageRunnerVerifyCapable(const SparkStageRunner *runner)
{
	const DeviceServing *serving = ServingOf(runner);
	return(serving != 0 && (SparkDeviceStageRunnerGeometry(serving->device)->recurrent_bytes == 0u || serving->model->keep_rows != 0) ? 1u : 0u);
}

uint64_t SparkStageRunnerTakeCaptureNs(SparkStageRunner *runner)
{
	(void)runner;
	return(0u);
}

SparkStatus SparkStageRunnerKeepRows(SparkStageRunner *runner, const uint32_t *kept, uint32_t sequences)
{
	DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerKeepRows(serving->device,kept,sequences) : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkStageRunnerCheck(SparkStageRunner *runner)
{
	return(ServingOf(runner) != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkStageRunnerSetTaps(SparkStageRunner *runner, const SparkStageRunnerTapEntry *entries, uint32_t entry_count, uint32_t capture)
{
	(void)runner; (void)entries; (void)capture;
	return(entry_count == 0u ? SPARK_STATUS_OK : ServingRefuse(SPARK_STATUS_UNSUPPORTED,"a device stage has no hidden-state taps"));
}

SparkStatus SparkStageRunnerTapShape(const SparkStageRunner *runner, uint32_t *hidden, uint32_t *total_layers)
{
	(void)runner; (void)hidden; (void)total_layers;
	return(SPARK_STATUS_UNSUPPORTED);
}

const uint16_t *SparkStageRunnerTapRows(const SparkStageRunner *runner, uint32_t step, uint32_t rows, uint64_t *row_stride_elements)
{
	(void)runner; (void)step; (void)rows; (void)row_stride_elements;
	return(0);
}

uint32_t SparkStageRunnerTapFirstRow(const SparkStageRunner *runner)
{
	(void)runner;
	return(0u);
}

SparkStatus SparkStageRunnerRecurrentCopy(SparkStageRunner *runner, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, SparkDeviceQueue *queue)
{
	DeviceServing *serving = ServingOf(runner);
	if ( serving == 0 || (queue != 0 && queue != SparkDeviceStageRunnerQueue(serving->device)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkDeviceStageRunnerRecurrentCopy(serving->device,to_buffer,slot,(SparkDeviceAddress)(uintptr_t)buffer,bytes));
}

SparkStatus SparkStageRunnerPackIdentity(const SparkStageRunner *runner, uint8_t *digest, uint32_t digest_bytes)
{
	const DeviceServing *serving = ServingOf(runner);
	SparkDeviceStageRunnerInfo info;
	if ( serving == 0 || digest == 0 || digest_bytes < SPARK_SHA256_DIGEST_BYTES ||
		SparkDeviceStageRunnerInfoGet(serving->device,&info) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkSha256HexToDigest(info.pack_sha256,digest));
}

const void *SparkStageRunnerProbeBuffers(const SparkStageRunner *runner)
{
	(void)runner;
	return(0);
}

void *SparkStageRunnerModel(const SparkStageRunner *runner)
{
	const DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerModel(serving->device) : 0);
}

SparkDeviceQueue *SparkStageRunnerQueue(const SparkStageRunner *runner)
{
	const DeviceServing *serving = ServingOf(runner);
	return(serving != 0 ? SparkDeviceStageRunnerQueue(serving->device) : 0);
}
