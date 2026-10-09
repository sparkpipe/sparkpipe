#include "inference/runner/stage_serving_adapter.c"
#include "modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c"
#include "tests/test_k3_runner_stub.h"
#include "tests/test_weightd_kv_server.h"

#include <assert.h>
#include <ftw.h>
#include <stdlib.h>
#include <unistd.h>

static char DIRECTORY[64];
static char SNAPSHOT_DIRECTORY[64];
static uint32_t COMPLETIONS;
static SparkModelServingCompletion LAST;
static uint64_t NEXT_ID = 1u;

typedef struct TestK3Work
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[4];
	uint32_t lane_of_row[8];
	uint64_t positions[8];
	uint64_t sequences[8];
	uint32_t tokens[8];
} TestK3Work;

static void Completion(void *context, const SparkModelServingCompletion *completion)
{
	(void)context;
	COMPLETIONS++;
	LAST = *completion;
}

static int RemoveEntry(const char *path, const struct stat *info, int flag, struct FTW *walk)
{
	(void)info;
	(void)flag;
	(void)walk;
	return(remove(path));
}

static SparkStatus BindWith(SparkStageServingState *state, uint32_t logical, uint32_t physical)
{
	SparkModelServingAdapterConfiguration configuration;
	memset(&configuration, 0, sizeof(configuration));
	configuration.runtime_limits.kv_logical_page_capacity = logical;
	configuration.runtime_limits.kv_physical_page_capacity = physical;
	configuration.kv_backing_directory = DIRECTORY;
	configuration.kv_backing_maximum_bytes = UINT64_C(1) << 26;
	configuration.kv_snapshot_directory = SNAPSHOT_DIRECTORY;
	configuration.kv_snapshot_maximum_bytes = UINT64_C(1) << 26;
	return(StageServingBindKv(state, &configuration));
}

static SparkStageServingState *Open(void)
{
	SparkStageServingState *state = (SparkStageServingState *)calloc(1u, sizeof(*state));
	SparkMemoryBuffer *host[6], *device[8];
	uint32_t index;
	assert(state != 0);
	state->model = &K3StageServing;
	host[0] = &state->positions_host; host[1] = &state->context_host; host[2] = &state->state_host;
	host[3] = &state->runs_host; host[4] = &state->seqslot_host; host[5] = &state->order_host;
	device[0] = &state->positions_device; device[1] = &state->context_device; device[2] = &state->state_device;
	device[3] = &state->runs_device; device[4] = &state->seqslot_device; device[5] = &state->order_device;
	device[6] = &state->output_tokens; device[7] = &state->output_scores;
	for ( index = 0u; index < 6u; index++ )
		assert(SparkMemoryBufferAllocate(host[index], SPARK_MEMORY_SPACE_HOST_COHERENT, 9u * sizeof(uint32_t)) == SPARK_STATUS_OK);
	for ( index = 0u; index < 8u; index++ )
		assert(SparkMemoryBufferAllocate(device[index], SPARK_MEMORY_SPACE_DEVICE_PRIVATE, 9u * sizeof(uint32_t)) == SPARK_STATUS_OK);
	state->max_rows = 8u;
	state->runner_config.max_input_row_count = 8u;
	state->runner_config.max_active_sequence_count = 4u;
	state->runner_config.resident_sequence_capacity = 4u;
	state->runner_config.kv_pages_per_sequence = 2u;
	state->runner_config.execution_stream = (void *)(uintptr_t)7u;
	state->completion_function = Completion;
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES);
	strcpy(DIRECTORY, "/tmp/sparkpipe-k3-kv-XXXXXX");
	assert(mkdtemp(DIRECTORY) != 0);
	strcpy(SNAPSHOT_DIRECTORY, "/tmp/sparkpipe-k3-snapshot-XXXXXX");
	assert(mkdtemp(SNAPSHOT_DIRECTORY) != 0);
	test_attach_calls = 0u;
	memset(test_recurrent_lanes, 0, sizeof(test_recurrent_lanes));
	COMPLETIONS = 0u;
	assert(BindWith(state, 16u, 8u) == SPARK_STATUS_OK);
	assert(test_attach_calls == 1u && test_attached_kv.pool == state->kv.region_base[0] && test_attached_kv.page_table == state->kv.page_table);
	assert(test_attached_kv.layer_count == test_kv_layers && test_attached_kv.pool_page_count == 8u && test_attached_kv.page_table_stride == state->kv.pages_per_sequence);
	assert(test_attached_kv.layer_stride_bytes == state->kv.region_layer_stride_bytes[0] && test_attached_kv.sequence_count == 4u);
	assert(test_attached_kv.layer_page_bytes == (uint64_t)SPARK_K3_KV_PAGE_SLOTS * SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_KV_BYTES_PER_SCALAR);
	assert(state->kv.recurrent.lane_bytes == TEST_K3_RECURRENT_BYTES && state->kv.pages_per_sequence == 2u);
	return(state);
}

static void Close(SparkStageServingState *state)
{
	StageServingDestroy(state);
	assert(rmdir(DIRECTORY) == 0);
	assert(nftw(SNAPSHOT_DIRECTORY, RemoveEntry, 16, FTW_DEPTH | FTW_PHYS) == 0);
}

static void WorkInit(TestK3Work *work, uint32_t kind)
{
	uint64_t id = NEXT_ID++;
	memset(work, 0, sizeof(*work));
	work->submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	work->submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	work->submission.work_kind = kind;
	work->submission.submission_id = work->submission.request_id = work->submission.transaction_id = id;
	work->submission.control_generation = 1u;
	work->submission.request_generation = 1u;
	work->submission.step_generation = id;
	work->submission.tokens_per_sequence = 1u;
	work->submission.lanes = work->lanes;
	work->submission.row_lane_indices = work->lane_of_row;
	work->submission.row_positions = work->positions;
	work->submission.row_sequence_ids = work->sequences;
	work->submission.token_ids = work->tokens;
}

static void WorkLane(TestK3Work *work, uint64_t sequence, uint32_t slot, uint64_t position, uint32_t context)
{
	SparkModelServingLane *lane = &work->lanes[work->submission.active_sequence_count];
	lane->request_id = work->submission.request_id;
	lane->request_generation = 1u;
	lane->step_generation = work->submission.step_generation;
	lane->sequence_id = sequence;
	lane->sequence_position = position;
	lane->resident_sequence_slot = slot;
	lane->context_token_count = context;
	work->submission.sequence_id = sequence;
	work->submission.sequence_position = position;
	work->submission.active_sequence_count++;
	work->submission.lane_count++;
}

static void WorkIdentity(SparkModelServingCacheIdentity *identity, uint8_t seed)
{
	uint32_t index;
	for ( index = 0u; index < sizeof(identity->sha256); index++ )
		identity->sha256[index] = (uint8_t)(seed + index);
}

static void WorkPublish(TestK3Work *work, uint32_t tokens, uint8_t seed)
{
	SparkModelServingLane *lane = &work->lanes[work->submission.active_sequence_count - 1u];
	lane->flags |= SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
	lane->cache_publish_token_count = tokens;
	WorkIdentity(&lane->cache_publish_identity, seed);
}

static void WorkPrefix(TestK3Work *work, uint32_t tokens, uint8_t seed)
{
	SparkModelServingLane *lane = &work->lanes[work->submission.active_sequence_count - 1u];
	lane->flags |= SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX;
	lane->cache_prefix_token_count = tokens;
	WorkIdentity(&lane->cache_prefix_identity, seed);
}

static void WorkRows(TestK3Work *work, uint32_t lane, uint64_t first, uint32_t count)
{
	uint32_t index, row;
	for ( index = 0u; index < count; index++ )
	{
		row = work->submission.row_count++;
		work->lane_of_row[row] = lane;
		work->positions[row] = first + index;
		work->sequences[row] = work->lanes[lane].sequence_id;
		work->tokens[row] = 100u + row;
	}
	work->submission.new_token_count = work->submission.row_count;
	work->submission.token_count = work->submission.row_count;
}

static SparkStatus Run(SparkStageServingState *state, TestK3Work *work)
{
	SparkStatus status = StageServingPrefetch(state, &work->submission, 1u);
	if ( status == SPARK_STATUS_OK )
		status = StageServingResolvePrefetch(state, &work->submission, SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT);
	if ( status == SPARK_STATUS_OK )
		status = StageServingSubmit(state, &work->submission);
	return(status);
}

static void Release(SparkStageServingState *state, uint64_t sequence, uint32_t slot)
{
	TestK3Work work;
	uint32_t completions = COMPLETIONS;
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_RELEASE);
	WorkLane(&work, sequence, slot, 0u, 0u);
	assert(StageServingSubmit(state, &work.submission) == SPARK_STATUS_OK && COMPLETIONS == completions + 1u);
}

static void TestK3PrefillDecodeRelease(void)
{
	SparkStageServingState *state = Open();
	SparkStageKvRecurrentCounters counters;
	TestK3Work work;
	uint32_t index, dispatches;
	for ( index = 0u; index < TEST_K3_RECURRENT_BYTES; index++ )
		test_recurrent_lanes[0][index] = (uint8_t)(0xc0u + index);
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 1u, 0u, 0u, 4u);
	WorkPublish(&work, 4u, 0x90u);
	WorkRows(&work, 0u, 0u, 4u);
	assert(Run(state, &work) == SPARK_STATUS_OK && test_dispatch_rows == 4u && COMPLETIONS == 1u && LAST.accepted_token_count == 4u);
	assert(atomic_load(&state->kv.lane_next_positions[0]) == 4u && SparkStageKvBindingResidentCount(&state->kv) == 1u);
	assert(state->kv.lanes[0].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && state->kv.page_cache.published_page_count == 1u);
	assert(atomic_load(&state->lane_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE);
	SparkStageKvBindingTakeRecurrentCounters(&state->kv, &counters);
	assert(counters.captures == 1u && counters.capture_bytes == TEST_K3_RECURRENT_BYTES && test_recurrent_to_buffer == 1u);
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_DECODE);
	WorkLane(&work, 1u, 0u, 4u, 5u);
	WorkRows(&work, 0u, 4u, 1u);
	assert(Run(state, &work) == SPARK_STATUS_OK && COMPLETIONS == 2u && atomic_load(&state->kv.lane_next_positions[0]) == 5u);
	dispatches = test_dispatch_calls;
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_DECODE);
	WorkLane(&work, 1u, 0u, 7u, 8u);
	WorkRows(&work, 0u, 7u, 1u);
	assert(Run(state, &work) != SPARK_STATUS_OK && test_dispatch_calls == dispatches && COMPLETIONS == 2u);
	assert(atomic_load(&state->kv.lane_next_positions[0]) == 5u && atomic_load(&state->lane_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE);
	test_reset_calls = 0u;
	Release(state, 1u, 0u);
	assert(test_reset_calls == 1u && test_reset_count == 1u && test_reset_slots[0] == 0u && SparkStageKvBindingResidentCount(&state->kv) == 0u);
	test_dispatch_status = SPARK_STATUS_IO_ERROR;
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 5u, 3u, 0u, 2u);
	WorkRows(&work, 0u, 0u, 2u);
	assert(Run(state, &work) == SPARK_STATUS_IO_ERROR && COMPLETIONS == 3u);
	test_dispatch_status = SPARK_STATUS_OK;
	assert(state->kv.lanes[3].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && atomic_load(&state->kv.lane_bound[3]) == 0u);
	assert(SparkStageKvBindingResidentCount(&state->kv) == 0u && atomic_load(&state->lane_states[3]) == SPARK_STAGE_MODULE_SLOT_FREE);
	Release(state, 5u, 3u);
	assert(state->kv.lanes[3].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && SparkStageKvBindingResidentCount(&state->kv) == 0u);
	Close(state);
	printf("K3 on the KV binding: prefill publishes and captures KDA state, decode continues, a position jump or a failed run binds nothing, release resets the slot and succeeds after a failed run: ok\n");
}

static void TestK3PrefixRestore(void)
{
	SparkStageServingState *state = Open();
	SparkStageKvRecurrentCounters counters;
	TestK3Work work;
	uint64_t published;
	uint32_t index;
	for ( index = 0u; index < TEST_K3_RECURRENT_BYTES; index++ )
		test_recurrent_lanes[0][index] = (uint8_t)(0x30u + index);
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 1u, 0u, 0u, 4u);
	WorkPublish(&work, 4u, 0x91u);
	WorkRows(&work, 0u, 0u, 4u);
	assert(Run(state, &work) == SPARK_STATUS_OK);
	Release(state, 1u, 0u);
	SparkStageKvBindingTakeRecurrentCounters(&state->kv, &counters);
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 2u, 1u, 4u, 5u);
	WorkPrefix(&work, 4u, 0x91u);
	WorkRows(&work, 0u, 4u, 1u);
	assert(StageServingPrefetch(state, &work.submission, 1u) == SPARK_STATUS_OK);
	assert((state->kv.lane_state_flags[1] & SPARK_STAGE_KV_LANE_STATE_RESTORE_READY) != 0u);
	assert(StageServingResolvePrefetch(state, &work.submission, SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
	memset(test_recurrent_lanes[1], 0, TEST_K3_RECURRENT_BYTES);
	assert(StageServingSubmit(state, &work.submission) == SPARK_STATUS_OK);
	assert(memcmp(test_recurrent_lanes[1], test_recurrent_lanes[0], TEST_K3_RECURRENT_BYTES) == 0 && test_recurrent_from_buffer >= 1u);
	assert(atomic_load(&state->kv.lane_next_positions[1]) == 5u);
	SparkStageKvBindingTakeRecurrentCounters(&state->kv, &counters);
	assert(counters.restores == 1u && counters.restore_bytes == TEST_K3_RECURRENT_BYTES);
	published = state->kv.page_cache.published_page_count;
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH);
	WorkLane(&work, 2u, 1u, 5u, 5u);
	WorkPublish(&work, 5u, 0x92u);
	work.submission.tokens_per_sequence = 0u;
	assert(Run(state, &work) == SPARK_STATUS_OK && LAST.accepted_token_count == 0u);
	assert(state->kv.page_cache.published_page_count > published);
	SparkStageKvBindingTakeRecurrentCounters(&state->kv, &counters);
	assert(counters.captures == 1u);
	Close(state);
	printf("K3 on the KV binding: a prefix hit on another slot restores the captured KDA state at claim, and a publish-only frame publishes and records: ok\n");
}

static void TestK3Reset(void)
{
	SparkStageServingState *state = Open();
	TestK3Work work;
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 1u, 2u, 0u, 3u);
	WorkRows(&work, 0u, 0u, 3u);
	assert(Run(state, &work) == SPARK_STATUS_OK && SparkStageKvBindingResidentCount(&state->kv) == 1u);
	test_reset_calls = 0u;
	assert(StageServingReset(state, 2u) == SPARK_STATUS_OK);
	assert(test_reset_calls == 1u && test_reset_count == 4u && SparkStageKvBindingResidentCount(&state->kv) == 0u);
	assert(state->kv.reset_generation == 2u && atomic_load(&state->reset_generation) == 2u);
	assert(StageServingReset(state, 2u) == SPARK_STATUS_VALIDATION_FAILED && test_reset_calls == 1u);
	WorkInit(&work, SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	WorkLane(&work, 3u, 2u, 0u, 2u);
	WorkRows(&work, 0u, 0u, 2u);
	assert(StageServingSubmit(state, &work.submission) == SPARK_STATUS_INVALID_ARGUMENT);
	work.submission.control_generation = 2u;
	assert(Run(state, &work) == SPARK_STATUS_OK && SparkStageKvBindingResidentCount(&state->kv) == 1u);
	Close(state);
	printf("K3 on the KV binding: reset clears every slot's KDA state and the binding, refuses a repeated generation, and stale submissions are refused: ok\n");
}

static void TestK3Refusals(void)
{
	SparkStageServingState *state = (SparkStageServingState *)calloc(1u, sizeof(*state));
	assert(state != 0);
	state->model = &K3StageServing;
	state->max_rows = 8u;
	state->runner_config.resident_sequence_capacity = 4u;
	state->runner_config.kv_pages_per_sequence = 2u;
	strcpy(DIRECTORY, "/tmp/sparkpipe-k3-kv-XXXXXX");
	assert(mkdtemp(DIRECTORY) != 0);
	strcpy(SNAPSHOT_DIRECTORY, "/tmp/sparkpipe-k3-snapshot-XXXXXX");
	assert(mkdtemp(SNAPSHOT_DIRECTORY) != 0);
	test_pack_status = SPARK_STATUS_INVALID_ARGUMENT;
	assert(BindWith(state, 16u, 8u) == SPARK_STATUS_INVALID_ARGUMENT && state->kv.mutex_initialized == 0u);
	test_pack_status = SPARK_STATUS_OK;
	assert(BindWith(state, 4u, 8u) != SPARK_STATUS_OK && state->kv.mutex_initialized == 0u);
	SparkStageKvBindingDestroy(&state->kv);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
	assert(rmdir(DIRECTORY) == 0);
	assert(nftw(SNAPSHOT_DIRECTORY, RemoveEntry, 16, FTW_DEPTH | FTW_PHYS) == 0);
	printf("K3 on the KV binding: no pack identity or too few logical pages refuse the binding: ok\n");
}

static void TestK3ContextShard(void)
{
	SparkStageServingState *state = (SparkStageServingState *)calloc(1u, sizeof(*state));
	const uint64_t page = (uint64_t)SPARK_K3_KV_PAGE_SLOTS * SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_KV_BYTES_PER_SCALAR;
	uint32_t rows;
	assert(state != 0);
	state->model = &K3StageServing;
	state->max_rows = 8u;
	state->runner_config.max_input_row_count = 8u;
	state->runner_config.max_active_sequence_count = 4u;
	state->runner_config.resident_sequence_capacity = 4u;
	state->runner_config.kv_pages_per_sequence = 2u;
	state->runner_config.tp_degree = 16u;
	state->runner_config.tp_rank = 5u;
	strcpy(DIRECTORY, "/tmp/sparkpipe-k3-kv-XXXXXX");
	assert(mkdtemp(DIRECTORY) != 0);
	strcpy(SNAPSHOT_DIRECTORY, "/tmp/sparkpipe-k3-snapshot-XXXXXX");
	assert(mkdtemp(SNAPSHOT_DIRECTORY) != 0);
	test_attach_calls = 0u;
	assert(BindWith(state, 16u, 8u) == SPARK_STATUS_OK && test_attach_calls == 1u);
	assert(state->kv.context_shard.degree == 16u && state->kv.context_shard.rank == 5u && state->kv.context_shard.grain == 1u);
	assert(test_attached_kv.context_shard.degree == 16u && test_attached_kv.context_shard.rank == 5u && test_attached_kv.context_shard.grain == 1u);
	assert(test_attached_kv.layer_page_bytes == page);
	assert(state->kv.region_packed_page_bytes[0] == (uint64_t)test_kv_layers * page / 16u);
	assert(test_attached_kv.layer_stride_bytes == state->kv.region_layer_stride_bytes[0] &&
		test_attached_kv.layer_stride_bytes == (uint64_t)test_attached_kv.pool_page_count * page / 16u);
	for ( rows = 1u; rows <= 512u; rows++ )
		assert(SparkK3KvShardFits(16u, rows) == 1u &&
			SparkK3KvShardQueryStride(rows, 16u) >= (uint64_t)rows * 6u * (SPARK_K3_MODEL_MLA_LATENT_DIMENSION + SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION) &&
			SparkK3KvShardPartialStride(rows, 16u) >= (uint64_t)rows * 6u * (SPARK_K3_MODEL_MLA_LATENT_DIMENSION + 2u));
	assert(SparkK3KvShardFits(1u, 1u) == 0u && SparkK3KvShardFits(32u, 1u) == 0u && SparkK3KvShardFits(5u, 1u) == 0u);
	for ( rows = 1u; rows <= 64u; rows++ )
	{
		uint32_t degree;
		assert(SparkK3KvShardSequenceCapacity(rows, 16u) == rows);
		for ( degree = 2u; degree <= 16u; degree *= 2u )
			assert(SparkK3KvShardFits(degree, rows) == 1u &&
				SparkK3KvShardSequenceCapacity(rows, degree) >= rows &&
				SparkK3KvShardSequenceCapacity(rows, degree) >= SparkK3KvShardQuerySequences(rows, degree) &&
				SparkK3KvShardSequenceCapacity(rows, degree) >= SparkK3KvShardPartialSequences(rows, degree) &&
				SparkK3KvShardQueryStride(rows, degree) >= (uint64_t)rows * (96u / degree) * (SPARK_K3_MODEL_MLA_LATENT_DIMENSION + SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION));
	}
	assert(SparkK3KvShardQuerySequences(64u, 4u) > 64u && SparkK3KvShardSequenceCapacity(64u, 4u) == SparkK3KvShardPartialSequences(64u, 4u));
	assert(SparkK3TpSequenceCapacity(64u, 16u) == 878u && SparkK3TpSequenceCapacity(1u, 16u) == 14u);
	for ( rows = 1u; rows <= 64u; rows++ )
		assert(SparkK3TpSequenceCapacity(rows, 4u) >= SparkK3KvShardSequenceCapacity(rows, 4u) &&
			(uint64_t)SparkK3TpSequenceCapacity(rows, 16u) * SPARK_K3_TP_ROW_ELEMENTS >= (uint64_t)rows * SPARK_K3_TP_GATE_UP_ROW_ELEMENTS &&
			(uint64_t)SparkK3TpSequences((uint64_t)rows * SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION) * SPARK_K3_TP_ROW_ELEMENTS >= (uint64_t)rows * SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION);
	SparkStageKvBindingDestroy(&state->kv);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
	assert(rmdir(DIRECTORY) == 0);
	assert(nftw(SNAPSHOT_DIRECTORY, RemoveEntry, 16, FTW_DEPTH | FTW_PHYS) == 0);
	printf("K3 on the KV binding: at TP16 rank 5 the binding splits context 5/16 and the runner gets 1/16 pages: ok\n");
}

int main(void)
{
	setvbuf(stdout, 0, _IONBF, 0);
	TestKvServerStart(UINT64_C(64) << 20);
	TestK3PrefillDecodeRelease();
	TestK3PrefixRestore();
	TestK3Reset();
	TestK3Refusals();
	TestK3ContextShard();
	TestKvServerFinish();
	printf("PASS k3 kv binding: K3 admits, claims, restores and finishes through the common KV binding with its KDA state as the recurrent record\n");
	return(0);
}
