#include "inference/runner/stage_serving_adapter.c"
#include "modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c"

#include "inference/llms/kimi_k3/generated_config.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#if SPARK_K3_SERVING_TOPOLOGY == 16
#define TEST_K3_DEPLOYMENT "modules/k3_resident_decode_stage/configs/model_resident_tp16.json"
#else
#define TEST_K3_DEPLOYMENT "modules/k3_resident_decode_stage/configs/model_resident.json"
#endif

#include "tests/test_k3_runner_stub.h"

#include <pthread.h>

static int32_t TestK3Check(int32_t condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static int32_t TestK3Descriptor(void)
{
	const SparkModelServingAdapterInterface *adapter = SparkModelServingAdapterGetInterface();
	const SparkModelServingAdapterDescriptor *descriptor = &K3ServingDescriptor;
	uint32_t index, stage_layers = 0u, tiled = 1u;
	int32_t failures = 0;
	failures += TestK3Check(adapter != 0 && adapter->descriptor == descriptor,
		"the adapter interface exports the K3 descriptor");
	failures += TestK3Check(SparkModelServingAdapterValidateInterface(adapter,
		0u) == SPARK_STATUS_OK &&
		(descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE) != 0u &&
		(descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH) != 0u,
		"the common validator accepts the K3 adapter: it restores cached prefixes and serves publish frames (I23)");
	failures += TestK3Check(descriptor->cache_block_token_count == SPARK_K3_KV_PAGE_SLOTS &&
		descriptor->cache_block_token_count == K3_KV_PAGE_SLOTS,
		"the cache block is one KV page of the kernel's slot count");
	failures += TestK3Check(strcmp(descriptor->driver_program_name, "k3") == 0,
		"the descriptor names the k3 driver program");
#if SPARK_K3_SERVING_TOPOLOGY == 16
	failures += TestK3Check(strcmp(descriptor->adapter_id, "k3-tp16") == 0 &&
		descriptor->parallel_group_size == 0u && descriptor->stage_count == 16u &&
		(descriptor->capability_flags &
			(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP |
			 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT)) == 0u &&
		(descriptor->capability_flags &
			SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) != 0u,
		"TP16: sixteen parallel ranks, no pipeline hand-off");
	for (index = 0u; index < SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT; index++)
		if ( (index < 16u && descriptor->stage_layer_counts[index] !=
				SPARK_K3_MODEL_LAYER_COUNT) ||
			(index < 12u && (descriptor->boundary_sideband_kinds[index] != 0u ||
				descriptor->boundary_sideband_bytes_per_sequence[index] != 0u)) )
			tiled = 0u;
	failures += TestK3Check(tiled != 0u &&
		descriptor->layer_count == SPARK_K3_MODEL_LAYER_COUNT,
		"TP16: every rank carries all 93 layers and no residual-bank sideband");
	return(failures);
#endif
	failures += TestK3Check(descriptor->parallel_group_size != 0u &&
		descriptor->stage_count == SPARK_K3_PP_STAGE_COUNT * descriptor->parallel_group_size,
		"the ranks are the PP stages times the TP group");
	for (index = 0u; index < descriptor->stage_count && descriptor->parallel_group_size != 0u; index++)
	{
		if ( descriptor->stage_layer_counts[index] !=
			SPARK_K3_PP_STAGE_LAYERS(index / descriptor->parallel_group_size) )
			tiled = 0u;
		if ( index % descriptor->parallel_group_size == 0u )
			stage_layers += descriptor->stage_layer_counts[index];
	}
	failures += TestK3Check(tiled != 0u && stage_layers == SPARK_K3_MODEL_LAYER_COUNT &&
		descriptor->layer_count == SPARK_K3_MODEL_LAYER_COUNT,
		"every rank carries its PP stage's layers and the stages tile the model");
	return(failures);
}

static int32_t TestK3Deployment(const char *path)
{
	SparkModelResidentDeployment deployment;
	SparkModelResidentDeployment zero_pages;
	SparkModelResidentDeployment short_pages;
	int32_t failures = 0;
	printf("deployment %s\n", path);
	memset(&deployment, 0, sizeof(deployment));
	if ( TestK3Check(SparkModelResidentDeploymentLoad(path, &deployment) == SPARK_STATUS_OK,
		"residentd's loader parses the deployment") != 0 )
		return(1);
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&deployment,
		&K3ServingDescriptor) == SPARK_STATUS_OK,
		"residentd accepts the K3 deployment for the prefix-reusing adapter");
	failures += TestK3Check(deployment.eos_token_count == 1u &&
		deployment.eos_token_ids[0] == K3_EOS_TOKEN,
		"the model EOS is the authoritative contract's end_of_text");
	zero_pages = deployment;
	zero_pages.runtime_limits.kv_logical_page_capacity = 0u;
	zero_pages.runtime_limits.kv_physical_page_capacity = 0u;
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&zero_pages,
		&K3ServingDescriptor) != SPARK_STATUS_OK,
		"zero KV page capacities are refused");
	short_pages = deployment;
	short_pages.runtime_limits.kv_physical_page_capacity =
		deployment.runtime_limits.max_active_sequence_count - 1u;
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&short_pages,
		&K3ServingDescriptor) != SPARK_STATUS_OK,
		"fewer physical pages than active sequences are refused");
	SparkModelResidentDeploymentDestroy(&deployment);
	return(failures);
}

static int32_t TestK3ReleaseValidation(void)
{
	SparkStageServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[3];
	uint32_t slots[4];
	int32_t failures = 0;
	memset(&state, 0, sizeof(state));
	state.runner_config.max_active_sequence_count = 4u;
	memset(&submission, 0, sizeof(submission));
	memset(lanes, 0, sizeof(lanes));
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission.lanes = lanes;
	submission.lane_count = 2u;
	submission.active_sequence_count = 2u;
	lanes[0].resident_sequence_slot = 3u;
	lanes[1].resident_sequence_slot = 1u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_OK &&
		StageServingReleaseSlots(&state, &submission, slots) == SPARK_STATUS_OK && slots[0] == 3u && slots[1] == 1u,
		"a RELEASE names each released slot once");
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_OK,
		"a publish-only frame is a rowless submission like a RELEASE");
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	lanes[1].resident_sequence_slot = 4u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a released slot outside the state pool is refused");
	lanes[1].resident_sequence_slot = 3u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a slot released twice in one RELEASE is refused");
	lanes[1].resident_sequence_slot = 1u;
	submission.lanes = 0;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a RELEASE naming sequences without lanes is refused");
	submission.lanes = lanes;
	submission.row_count = 1u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a RELEASE with rows is refused");
	submission.row_count = 0u;
	submission.control_generation = 4u;
	atomic_store(&state.reset_generation, 5u);
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a submission from before the last reset is refused");
	return(failures);
}

static int32_t TestK3SharedRows(void)
{
	static const uint64_t sequences[8] = { 7u, 7u, 7u, 9u, 9u, 9u, 11u, 11u };
	SparkStageServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[8];
	int32_t failures = 0;
	memset(&state, 0, sizeof(state));
	state.model = &K3StageServing;
	state.max_rows = 8u;
	state.max_shared_rows = 4u;
	state.runner_config.max_active_sequence_count = 8u;
	memset(&submission, 0, sizeof(submission));
	memset(lanes, 0, sizeof(lanes));
	submission.lanes = lanes;
	submission.lane_count = 2u;
	submission.row_sequence_ids = sequences;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.active_sequence_count = 2u;
	submission.row_count = 6u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_CAPACITY_EXCEEDED,
		"a prefill wave of two sequences above the shared rows is refused before any work");
	submission.row_count = 4u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_OK,
		"a prefill wave of two sequences within the shared rows is accepted");
	submission.lane_count = 1u;
	submission.active_sequence_count = 1u;
	submission.row_count = 8u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_OK,
		"a single-sequence prefill may use every row");
	submission.row_count = 9u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_CAPACITY_EXCEEDED,
		"no wave exceeds max_rows");
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.lane_count = 6u;
	submission.active_sequence_count = 6u;
	submission.row_count = 6u;
	failures += TestK3Check(StageServingValidateSubmission(&state, &submission) == SPARK_STATUS_CAPACITY_EXCEEDED,
		"a non-prefill wave above the shared rows is refused");
	return(failures);
}

static const uint32_t *TestK3Host(const SparkMemoryBuffer *buffer)
{
	return((const uint32_t *)buffer->pointer);
}

static int32_t TestK3Grouping(void)
{
	static const uint64_t positions[5] = { 0u, 1u, 2u, 0u, 1u };
	static const uint32_t lane_of_row[5] = { 0u, 0u, 0u, 1u, 1u };
	static const uint64_t sequences[5] = { 7u, 7u, 7u, 9u, 9u };
	SparkStageServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	SparkStatus status = SPARK_STATUS_OK;
	SparkMemoryBuffer *host[] = { &state.positions_host, &state.context_host,
		&state.state_host, &state.runs_host, &state.seqslot_host, &state.order_host, &state.target_host };
	uint32_t row_slots[8], index, active = 0u;
	uint64_t row_sequences[8], row_positions[8];
	int32_t failures = 0;
	memset(&state, 0, sizeof(state));
	state.max_rows = 8u;
	state.runner_config.max_active_sequence_count = 4u;
	for ( index = 0u; status == SPARK_STATUS_OK && index < sizeof(host) / sizeof(host[0]); index++ )
		status = SparkMemoryBufferAllocate(host[index], SPARK_MEMORY_SPACE_HOST_COHERENT, 9u * sizeof(uint32_t));
	if ( TestK3Check(status == SPARK_STATUS_OK, "the grouping buffers allocate") != 0 )
		return(1);
	memset(&submission, 0, sizeof(submission));
	memset(lanes, 0, sizeof(lanes));
	lanes[0].resident_sequence_slot = 2u;
	lanes[1].resident_sequence_slot = 0u;
	lanes[0].state_target_token_count = 16384u;
	lanes[1].state_target_token_count = 2u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.lanes = lanes;
	submission.lane_count = 2u;
	submission.active_sequence_count = 2u;
	submission.row_count = 5u;
	submission.row_positions = positions;
	submission.row_lane_indices = lane_of_row;
	submission.row_sequence_ids = sequences;
	status = StageServingGroupRows(&state, &submission, 5u, &active);
	failures += TestK3Check(status == SPARK_STATUS_OK && active == 2u,
		"a two-sequence prefill of five rows groups into two sequences");
	failures += TestK3Check(TestK3Host(&state.runs_host)[0] == 0u && TestK3Host(&state.runs_host)[1] == 3u &&
		TestK3Host(&state.runs_host)[2] == 5u,
		"the recurrent state walks each sequence's rows in order: row runs 0, 3, 5");
	failures += TestK3Check(TestK3Host(&state.seqslot_host)[0] == 2u && TestK3Host(&state.seqslot_host)[1] == 0u,
		"the recurrent state is indexed per sequence: slots 2 then 0");
	failures += TestK3Check(TestK3Host(&state.target_host)[0] == 16384u && TestK3Host(&state.target_host)[1] == 2u,
		"each sequence carries its lane's next kept-state position to the runner");
	failures += TestK3Check(TestK3Host(&state.state_host)[0] == 2u && TestK3Host(&state.state_host)[2] == 2u &&
		TestK3Host(&state.state_host)[3] == 0u && TestK3Host(&state.state_host)[4] == 0u,
		"attention KV stays indexed per row");
	failures += TestK3Check(TestK3Host(&state.context_host)[2] == 3u && TestK3Host(&state.context_host)[0] == 2u &&
		TestK3Host(&state.context_host)[1] == 0u && TestK3Host(&state.context_host)[3] == 0u,
		"attention reads each sequence's context by its slot: slot 2 holds 3 tokens, slot 0 holds 2");
	StageServingContinuityRows(&state, &submission, 5u, active, row_slots, row_sequences, row_positions);
	failures += TestK3Check(row_slots[0] == 2u && row_slots[1] == 0u && row_positions[0] == 0u && row_positions[1] == 0u &&
		row_sequences[0] == 7u && row_sequences[1] == 9u && row_slots[2] == 2u && row_positions[2] == 1u &&
		row_positions[3] == 2u && row_slots[4] == 0u && row_positions[4] == 1u,
		"the continuity rows lead with each sequence's first row, then each sequence's remaining rows in order");
	{
		static const uint64_t wave_positions[5] = { 0u, 0u, 1u, 1u, 2u };
		static const uint32_t wave_lane_of_row[5] = { 0u, 1u, 0u, 1u, 1u };
		static const uint64_t decode_positions[2] = { 11u, 4u };
		static const uint32_t decode_lanes[2] = { 1u, 0u };
		static const uint32_t one_lane[5] = { 0u, 0u, 0u, 0u, 0u };
		static const uint32_t past_lanes[5] = { 0u, 1u, 2u, 1u, 1u };
		submission.row_positions = wave_positions;
		submission.row_lane_indices = wave_lane_of_row;
		status = StageServingGroupRows(&state, &submission, 5u, &active);
		failures += TestK3Check(status == SPARK_STATUS_OK && active == 2u &&
			TestK3Host(&state.runs_host)[1] == 2u && TestK3Host(&state.runs_host)[2] == 5u &&
			TestK3Host(&state.order_host)[0] == 0u && TestK3Host(&state.order_host)[1] == 2u &&
			TestK3Host(&state.order_host)[2] == 1u && TestK3Host(&state.order_host)[3] == 3u && TestK3Host(&state.order_host)[4] == 4u,
			"a wave prefill (rows interleaved across sequences) walks each sequence's own rows in order");
		failures += TestK3Check(TestK3Host(&state.state_host)[0] == 2u && TestK3Host(&state.state_host)[1] == 0u &&
			TestK3Host(&state.state_host)[2] == 2u && TestK3Host(&state.context_host)[2] == 2u && TestK3Host(&state.context_host)[0] == 3u,
			"a wave prefill keeps each row's slot and each slot's context");
		submission.row_lane_indices = one_lane;
		failures += TestK3Check(StageServingGroupRows(&state, &submission, 5u, &active) == SPARK_STATUS_VALIDATION_FAILED,
			"a sequence with no rows is refused before admission");
		submission.row_lane_indices = past_lanes;
		failures += TestK3Check(StageServingGroupRows(&state, &submission, 5u, &active) == SPARK_STATUS_VALIDATION_FAILED,
			"a row naming a sequence past the submission's lanes is refused before admission");
		submission.row_lane_indices = wave_lane_of_row;
		lanes[0].resident_sequence_slot = 4u;
		failures += TestK3Check(StageServingGroupRows(&state, &submission, 5u, &active) == SPARK_STATUS_VALIDATION_FAILED,
			"a row whose slot is outside the sequence pool is refused before admission");
		lanes[0].resident_sequence_slot = 1u;
		lanes[1].resident_sequence_slot = 1u;
		failures += TestK3Check(StageServingGroupRows(&state, &submission, 5u, &active) == SPARK_STATUS_VALIDATION_FAILED,
			"two sequences on one slot are refused before admission");
		lanes[0].resident_sequence_slot = 2u;
		submission.row_count = 2u;
		submission.row_positions = decode_positions;
		submission.row_lane_indices = decode_lanes;
		status = StageServingGroupRows(&state, &submission, 2u, &active);
		failures += TestK3Check(status == SPARK_STATUS_OK && TestK3Host(&state.context_host)[1] == 12u &&
			TestK3Host(&state.context_host)[2] == 5u && TestK3Host(&state.context_host)[0] == 0u &&
			TestK3Host(&state.seqslot_host)[0] == 2u && TestK3Host(&state.seqslot_host)[1] == 1u &&
			TestK3Host(&state.order_host)[0] == 1u && TestK3Host(&state.order_host)[1] == 0u,
			"a decode whose rows are not in lane order gives each slot its own context and each lane its own row");
	}
	for ( index = 0u; index < sizeof(host) / sizeof(host[0]); index++ )
		SparkMemoryBufferFree(host[index]);
	return(failures);
}

static SparkStatus TestK3ParseTaps(const char *text, SparkStageServingState *state)
{
	SparkJsonDocument doc;
	SparkStatus status;
	memset(&doc, 0, sizeof(doc));
	memset(state, 0, sizeof(*state));
	state->model = &K3StageServing;
	state->tap_socket = -1;
	status = SparkJsonParseText(text, strlen(text), &doc);
	if ( status == SPARK_STATUS_OK )
		status = StageServingParseTaps(state, &doc, SparkJsonGetRootToken(&doc));
	SparkJsonDocumentDestroy(&doc);
	return(status);
}

static int32_t TestK3VerifyPublishRefused(void)
{
	static const uint64_t positions[2] = { 40u, 41u };
	static const uint32_t lane_of_row[2] = { 0u, 0u };
	static const uint64_t sequences[2] = { 7u, 7u };
	SparkStageServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	SparkStatus status = SPARK_STATUS_OK;
	SparkMemoryBuffer *host[] = { &state.positions_host, &state.context_host,
		&state.state_host, &state.runs_host, &state.seqslot_host, &state.order_host, &state.target_host };
	uint32_t index, untouched = 1u;
	int32_t failures = 0;
	memset(&state, 0, sizeof(state));
	state.model = &K3StageServing;
	state.max_rows = 8u;
	state.max_shared_rows = 8u;
	state.runner_config.max_active_sequence_count = 4u;
	for ( index = 0u; status == SPARK_STATUS_OK && index < sizeof(host) / sizeof(host[0]); index++ )
		status = SparkMemoryBufferAllocate(host[index], SPARK_MEMORY_SPACE_HOST_COHERENT, 9u * sizeof(uint32_t));
	if ( TestK3Check(status == SPARK_STATUS_OK, "the verify buffers allocate") != 0 )
		return(1);
	for ( index = 0u; index < sizeof(host) / sizeof(host[0]); index++ )
		memset((void *)TestK3Host(host[index]), 0xab, 9u * sizeof(uint32_t));
	memset(&submission, 0, sizeof(submission));
	memset(&lane, 0, sizeof(lane));
	lane.resident_sequence_slot = 1u;
	lane.sequence_position = 40u;
	lane.context_token_count = 42u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN | SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
	lane.cache_publish_token_count = 42u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.flags = SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY;
	submission.tokens_per_sequence = 2u;
	submission.lanes = &lane;
	submission.lane_count = 1u;
	submission.active_sequence_count = 1u;
	submission.row_count = 2u;
	submission.row_positions = positions;
	submission.row_lane_indices = lane_of_row;
	submission.row_sequence_ids = sequences;
	status = StageServingSubmit(&state, &submission);
	for ( index = 0u; index < sizeof(host) / sizeof(host[0]); index++ )
		for ( uint32_t word = 0u; word < 9u; word++ )
			untouched &= TestK3Host(host[index])[word] == 0xababababu ? 1u : 0u;
	failures += TestK3Check(status == SPARK_STATUS_INVALID_ARGUMENT && untouched != 0u,
		"a verify frame with a publishing lane is refused before any row reaches the runner, so no recurrent state is captured ahead of keep_rows");
	for ( index = 0u; index < sizeof(host) / sizeof(host[0]); index++ )
		SparkMemoryBufferFree(host[index]);
	return(failures);
}

static void TestK3ReleaseTaps(SparkStageServingState *state)
{
	StageServingTapClose(state);
	free(state->tap_buffer);
	free(state->tap_file);
	free(state->tap_stream);
}

static int TestK3Listen(uint32_t *port)
{
	struct sockaddr_in address;
	socklen_t bytes = (socklen_t)sizeof(address);
	int fd = socket(AF_INET, SOCK_STREAM, 0);
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if ( fd < 0 || bind(fd, (const struct sockaddr *)&address, bytes) != 0 || listen(fd, 1) != 0 ||
		getsockname(fd, (struct sockaddr *)&address, &bytes) != 0 )
	{
		if ( fd >= 0 )
			(void)close(fd);
		return(-1);
	}
	*port = ntohs(address.sin_port);
	return(fd);
}

typedef struct TestK3Reader
{
	int fd;
	uint8_t *data;
	size_t bytes;
	size_t got;
} TestK3Reader;

static void *TestK3Receive(void *argument)
{
	TestK3Reader *reader = (TestK3Reader *)argument;
	while ( reader->got < reader->bytes )
	{
		const ssize_t part = recv(reader->fd, reader->data + reader->got, reader->bytes - reader->got, 0);
		if ( part <= 0 )
			break;
		reader->got += (size_t)part;
	}
	return(0);
}

static int32_t TestK3TapStream(void)
{
	SparkStageServingState state;
	char endpoint[64];
	uint8_t *sent, *received;
	const size_t bytes = STAGE_SERVING_TAP_BUFFER_BYTES + 12345u;
	uint32_t port = 0u, index, failed = 0u;
	int listener, peer;
	TestK3Reader reader;
	pthread_t thread;
	int32_t failures = 0;
	failures += TestK3Check(TestK3ParseTaps("{\"entries\":\"o7,final\",\"rank\":15,\"stream\":\"127.0.0.1:9\"}", &state) == SPARK_STATUS_OK &&
		state.tap_stream != 0 && state.tap_file == 0 && state.tap_count == 2u && state.tap_rank == 15u, "a tap stream endpoint configures taps without a file");
	TestK3ReleaseTaps(&state);
	failures += TestK3Check(TestK3ParseTaps("{\"entries\":\"o7\",\"stream\":\"127.0.0.1:9\",\"file\":\"/tmp/t\",\"file_max_bytes\":10}", &state) ==
		SPARK_STATUS_SCHEMA_ERROR, "taps refuse a file and a stream together");
	TestK3ReleaseTaps(&state);
	failures += TestK3Check(TestK3ParseTaps("{\"entries\":\"o7\",\"stream\":\"127.0.0.1:9\",\"file_max_bytes\":10}", &state) == SPARK_STATUS_SCHEMA_ERROR,
		"a tap stream refuses a file byte limit");
	TestK3ReleaseTaps(&state);
	failures += TestK3Check(TestK3ParseTaps("{\"entries\":\"o7\"}", &state) == SPARK_STATUS_SCHEMA_ERROR, "taps need a file or a stream");
	TestK3ReleaseTaps(&state);
	failures += TestK3Check(TestK3ParseTaps("{\"entries\":\"o7\",\"file\":\"/tmp/t\"}", &state) == SPARK_STATUS_SCHEMA_ERROR,
		"a tap file still needs its byte limit");
	TestK3ReleaseTaps(&state);
	listener = TestK3Listen(&port);
	failures += TestK3Check(listener >= 0, "a loopback listener for the tap stream");
	if ( listener < 0 )
		return(failures);
	(void)snprintf(endpoint, sizeof(endpoint), "127.0.0.1:%u", port);
	memset(&state, 0, sizeof(state));
	state.model = &K3StageServing;
	state.tap_socket = StageServingTapConnect(endpoint, STAGE_SERVING_TAP_CONNECT_MS);
	state.tap_buffer = (uint8_t *)malloc(STAGE_SERVING_TAP_BUFFER_BYTES);
	peer = accept(listener, 0, 0);
	sent = (uint8_t *)malloc(bytes);
	received = (uint8_t *)malloc(bytes);
	for (index = 0u; sent != 0 && index < bytes; index++)
		sent[index] = (uint8_t)(index * 131u + 7u);
	failures += TestK3Check(state.tap_socket >= 0 && peer >= 0 && state.tap_buffer != 0 && sent != 0 && received != 0, "the tap rank connects to the stream endpoint");
	if ( state.tap_socket >= 0 && peer >= 0 && state.tap_buffer != 0 && sent != 0 && received != 0 )
	{
		reader.fd = peer;
		reader.data = received;
		reader.bytes = bytes;
		reader.got = 0u;
		failures += TestK3Check(pthread_create(&thread, 0, TestK3Receive, &reader) == 0, "a receiver thread for the tap stream");
		failures += TestK3Check(StageServingTapWrite(&state, sent, 100u) == 0 && state.tap_buffered == 100u, "small tap writes are buffered");
		failures += TestK3Check(StageServingTapWrite(&state, sent + 100u, bytes - 100u) == 0 && StageServingTapFlush(&state) == 0 &&
			state.tap_buffered == 0u, "a write larger than the buffer and a flush send everything");
		(void)pthread_join(thread, 0);
		failures += TestK3Check(reader.got == bytes && memcmp(sent, received, bytes) == 0, "the receiver gets every byte in order");
		(void)close(peer);
		peer = -1;
		for (index = 0u; index < 64u && failed == 0u; index++)
			if ( StageServingTapWrite(&state, sent, 65536u) != 0 || StageServingTapFlush(&state) != 0 )
				failed = 1u;
		failures += TestK3Check(failed != 0u, "a closed receiver fails the sends instead of killing the process");
	}
	if ( peer >= 0 )
		(void)close(peer);
	(void)close(listener);
	TestK3ReleaseTaps(&state);
	failures += TestK3Check(StageServingTapConnect(endpoint, STAGE_SERVING_TAP_RECONNECT_MS) < 0, "an endpoint with no listener is refused");
	failures += TestK3Check(StageServingTapConnect("127.0.0.1", 50) < 0 && StageServingTapConnect("host:12", 50) < 0 && StageServingTapConnect("127.0.0.1:0", 50) < 0,
		"malformed endpoints are refused");
	memset(&state, 0, sizeof(state));
	state.model = &K3StageServing;
	state.tap_socket = -1;
	state.tap_stream = strdup(endpoint);
	memset(&state.tap_hello, 0x5c, sizeof(state.tap_hello));
	failures += TestK3Check(StageServingTapStreamConnect(&state, STAGE_SERVING_TAP_RECONNECT_MS) == 0u && state.tap_stream_down != 0u && state.tap_socket < 0 &&
		state.tap_retry_ns > StageServingNowNs(), "an unreachable tap stream pauses the taps with a retry time instead of failing");
	listener = TestK3Listen(&port);
	if ( listener >= 0 )
	{
		uint8_t hello[sizeof(state.tap_hello)];
		free(state.tap_stream);
		(void)snprintf(endpoint, sizeof(endpoint), "127.0.0.1:%u", port);
		state.tap_stream = strdup(endpoint);
		failures += TestK3Check(StageServingTapStreamConnect(&state, STAGE_SERVING_TAP_RECONNECT_MS) != 0u && state.tap_stream_down == 0u && state.tap_socket >= 0,
			"the tap stream connects again once its endpoint listens");
		peer = accept(listener, 0, 0);
		reader.fd = peer;
		reader.data = hello;
		reader.bytes = sizeof(hello);
		reader.got = 0u;
		(void)TestK3Receive(&reader);
		failures += TestK3Check(peer >= 0 && reader.got == sizeof(hello) && memcmp(hello, &state.tap_hello, sizeof(hello)) == 0,
			"a reconnected stream starts with the stream hello");
		if ( peer >= 0 )
			(void)close(peer);
		(void)close(listener);
	}
	TestK3ReleaseTaps(&state);
	free(sent);
	free(received);
	return(failures);
}

int main(int argc, char **argv)
{
	int32_t failures = TestK3Descriptor();
	failures += TestK3ReleaseValidation();
	failures += TestK3SharedRows();
	failures += TestK3Grouping();
	failures += TestK3VerifyPublishRefused();
	failures += TestK3TapStream();
	int index;
	if ( argc == 1 )
		failures += TestK3Deployment(TEST_K3_DEPLOYMENT);
	for (index = 1; index < argc; index++)
		failures += TestK3Deployment(argv[index]);
	printf("test_k3_serving_adapter: %d failures\n", (int)failures);
	return(failures != 0 ? 1 : 0);
}
