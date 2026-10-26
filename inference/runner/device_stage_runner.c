#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "inference/runner/device_ops.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_sha256.h"

#define RUNNER_TABLE_ARRAYS 10u

typedef struct RunnerTables
{
	uint32_t *token_ids;
	uint32_t *positions;
	uint32_t *sequence_of_row;
	uint32_t *row_indices;
	uint32_t *row_begin;
	uint32_t *recurrent_index;
	uint32_t *context_length;
	uint32_t *head_rows;
	uint32_t *targets;
	uint32_t *kept;
	uint64_t offsets[RUNNER_TABLE_ARRAYS];
} RunnerTables;

struct SparkDeviceStageRunner
{
	SparkDeviceStageRunnerConfig config;
	SparkDeviceQueue *queue;
	SparkDeviceOps *ops;
	int pack_fd;
	SparkDeviceBuffer weights;
	void *model;
	SparkStageRunnerModelGeometry geometry;
	SparkStageRunnerServices services;
	SparkStageRunnerConfiguration configuration;
	SparkStageRunnerDeviceWeights device_weights;
	SparkDeviceBuffer tables;
	RunnerTables table;
	SparkDeviceBuffer hidden;
	SparkDeviceBuffer head_hidden;
	SparkDeviceBuffer head_normed;
	SparkDeviceBuffer logits;
	SparkDeviceBuffer logprob;
	SparkDeviceBuffer argmax;
	SparkDeviceBuffer kv_pool;
	SparkDeviceBuffer page_table;
	uint32_t pages_per_sequence;
	uint32_t *next_position;
	uint8_t *seen;
	SparkStageRunnerStep verify_step;
	uint32_t verify_pending;
	uint32_t device_open;
	SparkDeviceStageRunnerInfo info;
};

static SparkStatus RunnerFail(SparkStatus status, const char *what)
{
	fprintf(stderr,"device_stage_runner: %s (status %d; device: %s %s)\n",what,(int)status,SparkDeviceLastError()->operation,SparkDeviceLastError()->text);
	return(status);
}

static uint64_t RunnerNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static SparkStatus RunnerRound(void *context, void *device_queue, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	(void)context; (void)device_queue; (void)operation; (void)stage; (void)rows; (void)row_elements; (void)local; (void)full;
	fprintf(stderr,"device_stage_runner: a single-device stage has no collective round; the model asked for one\n");
	return(SPARK_STATUS_UNSUPPORTED);
}

static uint32_t RunnerPublished(void *context)
{
	(void)context;
	return(0u);
}

static void RunnerFailService(void *context, SparkStatus status)
{
	(void)context;
	fprintf(stderr,"device_stage_runner: the model reported failure %d\n",(int)status);
}

static SparkStatus RunnerExpertWeights(void *context, uint32_t layer, const uint32_t *group_row_offset_device, uint32_t routed_rows,
	const void **w1, const void **w2)
{
	(void)context; (void)layer; (void)group_row_offset_device; (void)routed_rows; (void)w1; (void)w2;
	fprintf(stderr,"device_stage_runner: routed expert leases need weightd, which the device runner does not have\n");
	return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus RunnerVerifyPack(SparkDeviceStageRunner *runner)
{
	char expected[128], hex[SPARK_SHA256_HEX_BYTES], path[4096];
	FILE *file;
	SparkStatus status;
	snprintf(path,sizeof(path),"%s.sha256",runner->config.pack_path);
	file = fopen(path,"r");
	if ( file == 0 || fscanf(file,"%64s",expected) != 1 )
	{
		if ( file != 0 )
			fclose(file);
		return(RunnerFail(SPARK_STATUS_NOT_FOUND,"the pack has no readable .sha256 record"));
	}
	fclose(file);
	status = SparkSha256File(runner->config.pack_path,hex);
	if ( status != SPARK_STATUS_OK )
		return(RunnerFail(status,"cannot hash the pack"));
	if ( strcmp(hex,expected) != 0 )
		return(RunnerFail(SPARK_STATUS_HASH_MISMATCH,"the pack bytes do not match its .sha256 record"));
	memcpy(runner->info.pack_sha256,hex,sizeof(runner->info.pack_sha256));
	return(SparkSha256HexToDigest(hex,runner->device_weights.pack_sha256));
}

static SparkStatus RunnerMapPack(SparkDeviceStageRunner *runner)
{
	struct stat stat_buffer;
	SparkStatus status;
	runner->pack_fd = open(runner->config.pack_path,O_RDONLY);
	if ( runner->pack_fd < 0 || fstat(runner->pack_fd,&stat_buffer) != 0 || stat_buffer.st_size <= 0 )
		return(RunnerFail(SPARK_STATUS_IO_ERROR,"cannot open the rank pack"));
	if ( runner->config.verify_pack != 0u )
	{
		status = RunnerVerifyPack(runner);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	status = SparkDeviceMapFile(&runner->weights,runner->pack_fd,0u,(uint64_t)stat_buffer.st_size,"rank pack");
	if ( status != SPARK_STATUS_OK )
		return(RunnerFail(status,"cannot map the rank pack into device memory"));
	runner->device_weights.pack_address = runner->weights.device;
	runner->device_weights.pack_bytes = runner->weights.bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus RunnerAllocate(SparkDeviceBuffer *buffer, SparkDeviceSpace space, uint64_t bytes, const char *owner, uint64_t *total)
{
	SparkStatus status = SparkDeviceAllocate(buffer,space,bytes,owner);
	if ( status != SPARK_STATUS_OK )
		return(RunnerFail(status,owner));
	*total += bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus RunnerTablesCreate(SparkDeviceStageRunner *runner, uint64_t *total)
{
	const uint64_t counts[RUNNER_TABLE_ARRAYS] = { runner->config.max_rows, runner->config.max_rows, runner->config.max_rows,
		runner->config.max_rows, (uint64_t)runner->config.sequences + 1u, runner->config.sequences, runner->config.sequences,
		runner->config.head_rows, runner->config.head_rows, runner->config.sequences };
	uint32_t **arrays[RUNNER_TABLE_ARRAYS] = { &runner->table.token_ids, &runner->table.positions, &runner->table.sequence_of_row,
		&runner->table.row_indices, &runner->table.row_begin, &runner->table.recurrent_index, &runner->table.context_length,
		&runner->table.head_rows, &runner->table.targets, &runner->table.kept };
	uint64_t cursor = 0u;
	uint32_t index;
	SparkStatus status;
	for ( index = 0u; index < RUNNER_TABLE_ARRAYS; index++ )
	{
		runner->table.offsets[index] = cursor;
		cursor += (counts[index] * 4u + 255u) / 256u * 256u;
	}
	status = RunnerAllocate(&runner->tables,SPARK_DEVICE_SPACE_SHARED_COHERENT,cursor,"runner row tables",total);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for ( index = 0u; index < RUNNER_TABLE_ARRAYS; index++ )
		*arrays[index] = (uint32_t *)((uint8_t *)runner->tables.host + runner->table.offsets[index]);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunnerKvCreate(SparkDeviceStageRunner *runner, uint64_t *total)
{
	const SparkStageRunnerModelGeometry *g = &runner->geometry;
	const uint32_t pool_pages = runner->pages_per_sequence * runner->config.sequences;
	SparkStageRunnerKv kv;
	uint32_t *table, index;
	SparkStatus status;
	if ( g->kv_layer_count == 0u || g->kv_layer_page_bytes == 0u || g->kv_second_layer_count != 0u )
		return(RunnerFail(SPARK_STATUS_UNSUPPORTED,"the device runner holds one paged KV pool; the model's cache layout needs another"));
	status = RunnerAllocate(&runner->kv_pool,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,(uint64_t)g->kv_layer_count * pool_pages * g->kv_layer_page_bytes,
		"runner KV pool",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->page_table,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)pool_pages * sizeof(uint32_t),"runner page table",total);
	if ( status != SPARK_STATUS_OK )
		return(status);
	table = (uint32_t *)runner->page_table.host;
	for ( index = 0u; index < pool_pages; index++ )
		table[index] = index;
	memset(&kv,0,sizeof(kv));
	kv.pool = (uint8_t *)(uintptr_t)runner->kv_pool.device;
	kv.layer_stride_bytes = (uint64_t)pool_pages * g->kv_layer_page_bytes;
	kv.layer_page_bytes = g->kv_layer_page_bytes;
	kv.layer_count = g->kv_layer_count;
	kv.page_table = (const uint32_t *)(uintptr_t)runner->page_table.device;
	kv.page_table_stride = runner->pages_per_sequence;
	kv.pool_page_count = pool_pages;
	kv.sequence_count = runner->config.sequences;
	kv.context_shard.degree = 1u;
	kv.context_shard.rank = 0u;
	kv.context_shard.grain = 1u;
	status = runner->config.model->attach_kv(runner->model,&kv);
	return(status == SPARK_STATUS_OK ? SPARK_STATUS_OK : RunnerFail(status,"the model refused the KV pool"));
}

static SparkStatus RunnerBuffers(SparkDeviceStageRunner *runner, uint64_t *total)
{
	const uint64_t hidden = runner->geometry.hidden, vocab = runner->geometry.vocab, head = runner->config.head_rows;
	SparkStatus status = RunnerTablesCreate(runner,total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->hidden,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)runner->config.max_rows * hidden * sizeof(uint16_t),"runner hidden rows",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->head_hidden,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,head * hidden * sizeof(uint16_t),"runner head rows",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->head_normed,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,head * hidden * sizeof(uint16_t),"runner head normed rows",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->logits,SPARK_DEVICE_SPACE_SHARED_COHERENT,head * vocab * sizeof(float),"runner logits",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->logprob,SPARK_DEVICE_SPACE_SHARED_COHERENT,head * 16u,"runner log-probabilities",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerAllocate(&runner->argmax,SPARK_DEVICE_SPACE_SHARED_COHERENT,head * sizeof(uint32_t),"runner argmax",total);
	if ( status == SPARK_STATUS_OK )
		status = RunnerKvCreate(runner,total);
	return(status);
}

static SparkStatus RunnerOpenModel(SparkDeviceStageRunner *runner)
{
	const SparkStageRunnerModelInterface *model = runner->config.model;
	SparkStageRunnerConfiguration *c = &runner->configuration;
	SparkStageRunnerModelOpen request;
	SparkStatus status;
	memset(c,0,sizeof(*c));
	c->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	c->descriptor_bytes = SPARK_STAGE_RUNNER_CONFIGURATION_BYTES;
	c->stage_index = 0u;
	c->stage_count = 1u;
	c->tp_degree = 1u;
	c->tp_rank = 0u;
	c->max_active_sequence_count = runner->config.sequences;
	c->resident_sequence_capacity = runner->config.sequences;
	c->max_input_row_count = runner->config.max_rows;
	c->kv_pages_per_sequence = runner->pages_per_sequence;
	c->state_budget_bytes = runner->config.state_budget_bytes;
	c->rank_pack_path = runner->config.pack_path;
	c->execution_stream = runner->queue;
	c->linear_weight_codec = runner->config.linear_weight_codec;
	c->head_row_capacity = runner->config.head_rows;
	runner->services.round = RunnerRound;
	runner->services.published = RunnerPublished;
	runner->services.fail = RunnerFailService;
	runner->services.expert_weights = RunnerExpertWeights;
	request.configuration = c;
	request.services = &runner->services;
	request.stream = runner->queue;
	status = model->open(&request,&runner->model,&runner->geometry);
	if ( status != SPARK_STATUS_OK )
		return(RunnerFail(status,"the model refused to open"));
	status = model->bind_device(runner->model,&runner->device_weights,&runner->geometry);
	return(status == SPARK_STATUS_OK ? SPARK_STATUS_OK : RunnerFail(status,"the model could not bind its tensors"));
}

SparkStatus SparkDeviceStageRunnerOpen(SparkDeviceStageRunner **out, const SparkDeviceStageRunnerConfig *config)
{
	SparkDeviceStageRunner *runner;
	const SparkStageRunnerModelInterface *model = config != 0 ? config->model : 0;
	uint64_t total = 0u;
	SparkStatus status;
	*out = 0;
	if ( config == 0 || model == 0 || model->abi_version != SPARK_STAGE_RUNNER_MODEL_ABI_VERSION || model->open == 0 || model->bind_device == 0 ||
		model->close == 0 || model->step == 0 || model->attach_kv == 0 || model->reset_slot == 0 || config->pack_path == 0 ||
		config->max_rows == 0u || config->sequences == 0u || config->max_context == 0u || config->head_rows == 0u ||
		config->head_rows > config->max_rows )
	{
		fprintf(stderr,"device_stage_runner: needs a model with open, bind_device, close, step, attach_kv and reset_slot, a pack, and nonzero rows, "
			"sequences, context and head rows\n");
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	runner = (SparkDeviceStageRunner *)calloc(1u,sizeof(*runner));
	if ( runner == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	runner->config = *config;
	runner->pack_fd = -1;
	runner->pages_per_sequence = (config->max_context + SPARK_DEVICE_RUNNER_PAGE_SLOTS - 1u) / SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	runner->next_position = (uint32_t *)calloc(config->sequences,sizeof(uint32_t));
	runner->seen = (uint8_t *)calloc(config->sequences,1u);
	status = runner->next_position != 0 && runner->seen != 0 ? SparkDeviceInitialize(0) : SPARK_STATUS_CAPACITY_EXCEEDED;
	runner->device_open = status == SPARK_STATUS_OK ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueCreate(&runner->queue,"device stage runner");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceOpsAcquire(&runner->ops);
	if ( status == SPARK_STATUS_OK )
		status = RunnerMapPack(runner);
	if ( status == SPARK_STATUS_OK )
		status = RunnerOpenModel(runner);
	if ( status == SPARK_STATUS_OK )
		status = RunnerBuffers(runner,&total);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDeviceStageRunnerClose(runner);
		return(status);
	}
	runner->info.hidden = runner->geometry.hidden;
	runner->info.vocab = runner->geometry.vocab;
	runner->info.layers = runner->geometry.total_layers;
	runner->info.pack_bytes = runner->weights.bytes;
	runner->info.streamed_bytes_per_token = runner->weights.bytes - runner->geometry.embed_bytes;
	runner->info.device_state_bytes = total;
	snprintf(runner->info.device,sizeof(runner->info.device),"%s",SparkDeviceGetProperties()->name);
	snprintf(runner->info.target,sizeof(runner->info.target),"%s",SparkDeviceGetProperties()->target);
	*out = runner;
	return(SPARK_STATUS_OK);
}

static SparkStatus RunnerValidate(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch, uint32_t *sequences)
{
	uint32_t row, count = 0u;
	if ( batch->rows == 0u || batch->rows > runner->config.max_rows || batch->head_rows == 0u || batch->head_rows > runner->config.head_rows ||
		batch->token_ids == 0 || batch->positions == 0 || batch->sequence_of_row == 0 || batch->head_row_index == 0 )
		return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"a step needs rows within the runner's capacity and head rows within its head capacity"));
	if ( batch->verify != 0u && runner->geometry.recurrent_bytes != 0u && runner->config.model->keep_rows == 0 )
		return(RunnerFail(SPARK_STATUS_UNSUPPORTED,"a verify step needs a model that keeps verified rows of its recurrent state"));
	memset(runner->seen,0,runner->config.sequences);
	for ( row = 0u; row < batch->rows; row++ )
	{
		const uint32_t slot = batch->sequence_of_row[row];
		const uint32_t first = row == 0u || batch->sequence_of_row[row - 1u] != slot;
		if ( slot >= runner->config.sequences || batch->token_ids[row] >= runner->geometry.vocab || batch->positions[row] >= runner->config.max_context )
			return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"a row names a sequence, token or position outside the runner"));
		if ( first != 0u )
		{
			if ( runner->seen[slot] != 0u )
				return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"each sequence's rows must be contiguous in a step"));
			runner->seen[slot] = 1u;
			count++;
		}
		if ( batch->positions[row] != (first != 0u ? runner->next_position[slot] : batch->positions[row - 1u] + 1u) )
			return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"rows must continue each sequence at its next position, in order"));
	}
	for ( row = 0u; row < batch->head_rows; row++ )
		if ( batch->head_row_index[row] >= batch->rows || (batch->targets != 0 && batch->targets[row] >= runner->geometry.vocab) )
			return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"a head row or target lies outside the step"));
	*sequences = count;
	return(SPARK_STATUS_OK);
}

static void RunnerWriteTables(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch, uint32_t sequences)
{
	RunnerTables *t = &runner->table;
	uint32_t row, lane = 0u;
	memcpy(t->token_ids,batch->token_ids,(size_t)batch->rows * sizeof(*t->token_ids));
	memcpy(t->positions,batch->positions,(size_t)batch->rows * sizeof(*t->positions));
	memcpy(t->sequence_of_row,batch->sequence_of_row,(size_t)batch->rows * sizeof(*t->sequence_of_row));
	for ( row = 0u; row < batch->rows; row++ )
	{
		t->row_indices[row] = row;
		if ( row == 0u || batch->sequence_of_row[row] != batch->sequence_of_row[row - 1u] )
		{
			t->row_begin[lane] = row;
			t->recurrent_index[lane] = batch->sequence_of_row[row];
			lane++;
		}
		t->context_length[batch->sequence_of_row[row]] = batch->positions[row] + 1u;
	}
	t->row_begin[sequences] = batch->rows;
	memcpy(t->head_rows,batch->head_row_index,(size_t)batch->head_rows * sizeof(*t->head_rows));
	for ( row = 0u; row < batch->head_rows; row++ )
		t->targets[row] = batch->targets != 0 ? batch->targets[row] : 0xffffffffu;
}

static const uint32_t *RunnerDeviceTable(const SparkDeviceStageRunner *runner, uint32_t index)
{
	return((const uint32_t *)(uintptr_t)(runner->tables.device + runner->table.offsets[index]));
}

static SparkTensor RunnerTable(const SparkDeviceStageRunner *runner, uint32_t index, uint64_t count)
{
	return(SparkTensorAt(&runner->tables,runner->table.offsets[index],count * 4u));
}

static SparkStatus RunnerHead(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch)
{
	const SparkStageRunnerModelGeometry *g = &runner->geometry;
	const uint32_t rows = batch->head_rows;
	SparkNormParams norm;
	SparkGemvParams head;
	SparkLogprobParams logprob;
	SparkStatus status;
	status = SparkOpsGather(runner->ops,runner->queue,SparkTensorAt(&runner->hidden,0u,(uint64_t)batch->rows * g->hidden * 2u),RunnerTable(runner,7u,rows),
		SparkTensorAt(&runner->head_hidden,0u,(uint64_t)rows * g->hidden * 2u),rows,g->hidden,batch->rows);
	memset(&norm,0,sizeof(norm));
	norm.rows = rows;
	norm.dimension = g->hidden;
	norm.input_stride = norm.output_stride = g->hidden;
	norm.epsilon = g->rms_epsilon;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormResidual(runner->ops,runner->queue,SparkTensorAt(&runner->head_hidden,0u,(uint64_t)rows * g->hidden * 2u),0,
			SparkTensorAt(&runner->weights,g->head_norm_offset,g->head_norm_bytes),SparkTensorAt(&runner->head_normed,0u,(uint64_t)rows * g->hidden * 2u),&norm);
	memset(&head,0,sizeof(head));
	head.rows = rows;
	head.input_dimension = g->hidden;
	head.output_dimension = g->vocab;
	head.input_stride = g->hidden;
	head.output_stride = g->vocab;
	head.output_f32 = 1u;
	head.row_exact = batch->verify;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinear(runner->ops,runner->queue,SparkTensorAt(&runner->head_normed,0u,(uint64_t)rows * g->hidden * 2u),
			SparkTensorAt(&runner->weights,g->head_offset,g->head_bytes),SparkTensorAt(&runner->logits,0u,(uint64_t)rows * g->vocab * 4u),&head);
	if ( status == SPARK_STATUS_OK && g->logit_softcap != 0.0f )
		status = SparkOpsSoftcap(runner->ops,runner->queue,SparkTensorAt(&runner->logits,0u,(uint64_t)rows * g->vocab * sizeof(float)),(uint64_t)rows * g->vocab,
			g->logit_softcap);
	memset(&logprob,0,sizeof(logprob));
	logprob.rows = rows;
	logprob.vocabulary = g->vocab;
	logprob.logit_stride = g->vocab;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLogprob(runner->ops,runner->queue,SparkTensorAt(&runner->logits,0u,(uint64_t)rows * g->vocab * 4u),RunnerTable(runner,8u,rows),
			SparkTensorAt(&runner->logprob,0u,(uint64_t)rows * 16u),SparkTensorAt(&runner->argmax,0u,(uint64_t)rows * 4u),&logprob);
	return(status);
}

SparkStatus SparkDeviceStageRunnerStep(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch, SparkDeviceStageRunnerResult *result)
{
	const SparkStageRunnerModelGeometry *g = &runner->geometry;
	SparkStageRunnerStep step;
	uint32_t sequences = 0u, row, error = 0u;
	const uint64_t start = RunnerNowNs();
	SparkStatus status = RunnerValidate(runner,batch,&sequences);
	if ( status != SPARK_STATUS_OK )
		return(status);
	RunnerWriteTables(runner,batch,sequences);
	status = SparkOpsGather(runner->ops,runner->queue,SparkTensorAt(&runner->weights,g->embed_offset,g->embed_bytes),RunnerTable(runner,0u,batch->rows),
		SparkTensorAt(&runner->hidden,0u,(uint64_t)batch->rows * g->hidden * 2u),batch->rows,g->hidden,g->embed_rows);
	memset(&step,0,sizeof(step));
	step.hidden_bf16 = (uint16_t *)(uintptr_t)runner->hidden.device;
	step.positions = RunnerDeviceTable(runner,1u);
	step.sequence_of_row = RunnerDeviceTable(runner,2u);
	step.sequence_row_indices = RunnerDeviceTable(runner,3u);
	step.sequence_row_begin = RunnerDeviceTable(runner,4u);
	step.recurrent_index = RunnerDeviceTable(runner,5u);
	step.context_length = RunnerDeviceTable(runner,6u);
	step.token_ids = RunnerDeviceTable(runner,0u);
	step.rows = batch->rows;
	step.sequences = sequences;
	step.commit = batch->verify != 0u ? 0u : 1u;
	step.host_positions = runner->table.positions;
	step.host_sequence_of_row = runner->table.sequence_of_row;
	step.route_override = batch->route_override;
	step.route_override_width = batch->route_override != 0 ? batch->route_override_width : 0u;
	if ( status == SPARK_STATUS_OK )
		status = runner->config.model->step(runner->model,&step,runner->queue);
	if ( status == SPARK_STATUS_OK )
		status = RunnerHead(runner,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(runner->queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status == SPARK_STATUS_OK && SparkDeviceOpsTakeError(runner->ops,&error) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"device_stage_runner: a kernel raised the device error word (row %u)\n",error - 1u);
		status = SPARK_STATUS_IO_ERROR;
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkDeviceQueueWait(runner->queue,SPARK_DEVICE_WAIT_FOREVER);
		return(RunnerFail(status,"the step failed"));
	}
	runner->verify_pending = batch->verify;
	runner->verify_step = step;
	for ( row = 0u; batch->verify == 0u && row < batch->rows; row++ )
		runner->next_position[batch->sequence_of_row[row]] = batch->positions[row] + 1u;
	result->argmax = (const uint32_t *)runner->argmax.host;
	result->logprob = (const float *)runner->logprob.host;
	result->logits = (const float *)runner->logits.host;
	result->logit_stride = g->vocab;
	result->step_ns = RunnerNowNs() - start;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageRunnerKeepRows(SparkDeviceStageRunner *runner, const uint32_t *kept, uint32_t sequences)
{
	const RunnerTables *t = &runner->table;
	uint32_t sequence, error = 0u;
	SparkStatus status;
	if ( runner->verify_pending == 0u || kept == 0 || sequences != runner->verify_step.sequences )
		return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"keep_rows follows a verify step and names a count for each of its sequences"));
	for ( sequence = 0u; sequence < sequences; sequence++ )
		if ( kept[sequence] > t->row_begin[sequence + 1u] - t->row_begin[sequence] )
			return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"a sequence keeps more rows than its verify step held"));
	if ( runner->config.model->keep_rows == 0 && runner->geometry.recurrent_bytes != 0u )
		return(RunnerFail(SPARK_STATUS_UNSUPPORTED,"the model keeps no verified rows"));
	runner->verify_pending = 0u;
	memcpy(t->kept,kept,(size_t)sequences * sizeof(*t->kept));
	if ( runner->config.model->keep_rows != 0 )
	{
		status = runner->config.model->keep_rows(runner->model,&runner->verify_step,RunnerDeviceTable(runner,9u),runner->queue);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceQueueWait(runner->queue,SPARK_DEVICE_WAIT_FOREVER);
		if ( status == SPARK_STATUS_OK && SparkDeviceOpsTakeError(runner->ops,&error) != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
		if ( status != SPARK_STATUS_OK )
			return(RunnerFail(status,"keep_rows failed"));
	}
	for ( sequence = 0u; sequence < sequences; sequence++ )
		runner->next_position[t->recurrent_index[sequence]] = t->positions[t->row_begin[sequence]] + kept[sequence];
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageRunnerResetSequence(SparkDeviceStageRunner *runner, uint32_t sequence)
{
	SparkStatus status;
	if ( sequence >= runner->config.sequences )
		return(RunnerFail(SPARK_STATUS_INVALID_ARGUMENT,"reset names a sequence outside the runner"));
	status = runner->config.model->reset_slot(runner->model,sequence,runner->queue);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(runner->queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status == SPARK_STATUS_OK )
		runner->next_position[sequence] = 0u;
	return(status == SPARK_STATUS_OK ? SPARK_STATUS_OK : RunnerFail(status,"the model could not reset the sequence"));
}

SparkStatus SparkDeviceStageRunnerInfoGet(const SparkDeviceStageRunner *runner, SparkDeviceStageRunnerInfo *info)
{
	*info = runner->info;
	return(SPARK_STATUS_OK);
}

void *SparkDeviceStageRunnerModel(const SparkDeviceStageRunner *runner)
{
	return(runner->model);
}

void SparkDeviceStageRunnerClose(SparkDeviceStageRunner *runner)
{
	SparkDeviceBuffer *buffers[] = { &runner->tables, &runner->hidden, &runner->head_hidden, &runner->head_normed, &runner->logits, &runner->logprob,
		&runner->argmax, &runner->kv_pool, &runner->page_table };
	uint32_t index;
	if ( runner == 0 )
		return;
	if ( runner->queue != 0 )
		(void)SparkDeviceQueueWait(runner->queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( runner->model != 0 )
		runner->config.model->close(runner->model);
	for ( index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		if ( buffers[index]->handle != 0u )
			(void)SparkDeviceFree(buffers[index]);
	if ( runner->weights.handle != 0u )
		(void)SparkDeviceFree(&runner->weights);
	if ( runner->pack_fd >= 0 )
		close(runner->pack_fd);
	if ( runner->ops != 0 )
		SparkDeviceOpsRelease(runner->ops);
	if ( runner->queue != 0 )
		(void)SparkDeviceQueueDestroy(runner->queue);
	if ( runner->device_open != 0u )
		(void)SparkDeviceShutdown();
	free(runner->next_position);
	free(runner->seen);
	free(runner);
}
