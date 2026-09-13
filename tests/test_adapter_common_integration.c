/*
 * Integration gate for the shared serving-adapter skeleton
 * (runtime/adapter_common.[ch]) exercised the way its four consumers embed
 * it: one family-shaped pending-record layout per profile (dsv4-, glm52-,
 * qwen36- and qwen38-style tails behind a SparkAdapterPendingCore head),
 * each against its OWN parameterized fake model driver
 * (tests/fake_driver/adapter_fake_driver.c compiled per -DPROF_* profile),
 * loaded through the REAL SparkAdapterLoadDriver -> dlopen path.
 *
 * Pinned here, per family profile AND across all four:
 *   1. Configuration law   - shared validation accepts the family shape and
 *      rejects wrong program name / bad stage index / ABI byte drift;
 *   2. Prologue mirroring  - slot count, limits, routes land verbatim;
 *   3. Driver contract     - both capability schools (kind 0 runtime-limits,
 *      kind 1 exact-mask) load through one path; a drifted revision is
 *      refused TARGET_MISMATCH;
 *   4. Slot-table law      - stride-parameterized claim/available scans over
 *      family records with distinct tails; slots beyond the configured count
 *      never activate even when pre-marked; identity lands in the core;
 *   5. Submission open     - shared open gate accepts a wire-valid decode
 *      submission and refuses once quiesced;
 *   6. Completion routing  - match predicate agrees on all four identity
 *      fields; the orphan route fired through the DRIVER-wired callback
 *      lands in the shared counter;
 *   7. Wake trampoline     - forwards to the configuration's route;
 *   8. Residency law       - plain builder zeroes residency and stamps ABI;
 *      WithResidency copies the anchor; NULL keeps it zeroed;
 *   9. Snapshot merge      - driver counters + availability (zeroed while
 *      quiescing) + orphan-inflated rejections;
 *  10. Quiesce lifecycle   - BUSY on any claimed slot, BUSY on driver
 *      activity, OK when idle, latch permanent;
 *  11. Destroy guards      - teardown idempotent;
 *  12. Env parsing trio    - DefaultOn / ExactZero / Truthy semantics incl.
 *      the "01" divergence; loud-fail setenv; %u formatter.
 *  13. Admit-frame trio    - NULL guards refuse INVALID_ARGUMENT; accept
 *      without submit_on_apply leaves the driver's submitted_count
 *      untouched; accept + submit_on_apply fires the program submit exactly
 *      once; the from-frame request path (NULL submission) admits too;
 *  14. Direct orphan pin   - the PUBLIC SparkAdapterOrphanDriverCompletion
 *      bumps the shared counter and a NULL context is a safe no-op (the
 *      driver-wired route itself stays covered by section 6);
 *  15. Direct wake + counters - the PUBLIC SparkAdapterDispatchWake forwards
 *      once and tolerates NULL; AccumulateFrameCounters sums accepted/
 *      queue-delay/service across completions;
 *  16. Spin-lock law       - acquire pins the word to 1, release returns it
 *      to 0, and four hammering threads keep the protected counter exact.
 *
 * SparkAdapterLoadTpCollective is deliberately NOT duplicated here: the
 * golden-stanza gate tests/test_adapter_tp_collective.c IS its integration
 * coverage (one concern, one gate).
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_status.h"
#include "adapter_common.h"
#include "adapter_fake_driver.h"

static uint32_t g_checks;
static uint32_t g_failures;

#define CHECK(condition) 	do { g_checks++; if ( !(condition) ) { 		g_failures++; 		(void)fprintf(stderr,"FAIL %s:%d check #%d: %s\n", 			__FILE__,__LINE__,g_checks,#condition); } } while (0)

static const SparkAdapterPendingCore *TestCore(
	const void *pending_table,size_t record_bytes,uint32_t index)
{
	return((const SparkAdapterPendingCore *)
		((const char *)pending_table + index * record_bytes));
}

/* ---- completion/wake recorders ----------------------------------------- */

static uint32_t g_completion_count;
static uint64_t g_last_completed_request_id;
static uint32_t g_wake_count;

static void TestCompletionRoute(void *context,
	const SparkModelServingCompletion *completion)
{
	(void)context;
	g_completion_count++;
	g_last_completed_request_id =
		completion != 0 ? completion->request_id : 0u;
}

static void TestWakeRoute(void *context)
{
	(void)context;
	g_wake_count++;
}

/* ---- family-shaped embedders ------------------------------------------- */

typedef struct Dsv4StyleRecord
{
	SparkAdapterPendingCore core;
	uint64_t speculation_credit;
} Dsv4StyleRecord;

typedef struct Glm52StyleRecord
{
	SparkAdapterPendingCore core;
	uint32_t tap_mask;
	uint64_t anchor_generation;
} Glm52StyleRecord;

typedef struct Qwen36StyleRecord
{
	SparkAdapterPendingCore core;
	uint32_t emit_rows[4];
} Qwen36StyleRecord;

typedef struct Qwen38StyleRecord
{
	SparkAdapterPendingCore core;
	uint32_t cache_lane_count;
	uint64_t prefix_token_count;
} Qwen38StyleRecord;

typedef struct FamilyProfile
{
	const char *label;
	const char *model_id;
	const char *model_revision;
	const char *stage_name;
	const char *program_name;
	const char *driver_so;
	uint32_t check_kind;
	uint32_t pin_target;
	uint32_t inflight_slots;
	size_t record_bytes;
} FamilyProfile;

#define TAIL_MAGIC UINT64_C(0xA11CE00000000001)

static const FamilyProfile kProfiles[4] =
{
	{ "dsv4-style",   "integration.family.a", "rev-a", "stage.a", "prog.a",
	  "build/test_adapter_fake_a_driver.so",
	  1u, 1u, 4u, sizeof(Dsv4StyleRecord) },
	{ "glm52-style",  "integration.family.b", "rev-b", "stage.b", "prog.b",
	  "build/test_adapter_fake_b_driver.so",
	  0u, 1u, 8u, sizeof(Glm52StyleRecord) },
	{ "qwen36-style", "integration.family.c", "rev-c", "stage.c", "prog.c",
	  "build/test_adapter_fake_c_driver.so",
	  0u, 0u, 2u, sizeof(Qwen36StyleRecord) },
	{ "qwen38-style", "integration.family.d", "rev-d", "stage.d", "prog.d",
	  "build/test_adapter_fake_d_driver.so",
	  1u, 1u, 6u, sizeof(Qwen38StyleRecord) }
};

/* ---- builders ----------------------------------------------------------- */

static void TestBuildDescriptor(const FamilyProfile *profile,
	SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes =
		SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags =
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION;
	descriptor->stage_count = 1u;
	descriptor->layer_count = 1u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 6144u;
	descriptor->boundary_element_bytes = sizeof(uint16_t);
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 16u;
	descriptor->max_active_sequence_count = 16u;
	descriptor->max_input_row_count = 256u;
	descriptor->max_resident_sequence_count = 256u;
	descriptor->max_output_token_count = 16u;
	descriptor->adapter_id = "integration.adapter";
	descriptor->model_id = profile->model_id;
	descriptor->model_revision = profile->model_revision;
	descriptor->driver_program_name = profile->program_name;
	descriptor->artifact_sha256 =
		"0123456789abcdef0123456789abcdef"
		"0123456789abcdef0123456789abcdef";
	descriptor->stage_layer_counts[0] = 1u;
}

static void TestBuildLimits(const FamilyProfile *profile,
	SparkModelServingRuntimeLimits *limits)
{
	memset(limits,0,sizeof(*limits));
	limits->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits->descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits->max_inflight_submission_count = profile->inflight_slots;
	limits->max_active_sequence_count = profile->inflight_slots;
	limits->max_input_row_count = profile->inflight_slots * 4u;
	limits->resident_sequence_capacity = profile->inflight_slots * 4u;
}

static void TestBuildConfiguration(const FamilyProfile *profile,
	const SparkModelServingRuntimeLimits *limits,
	SparkModelServingAdapterConfiguration *configuration)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes =
		SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = 0u;
	configuration->stage_index = 0u;
	configuration->runtime_limits = *limits;
	configuration->runtime_root = "build/tmp";
	configuration->node_id = "node.integration";
	configuration->node_target = "cpu.test";
	configuration->adapter_configuration_path = "unused.json";
	configuration->driver_shared_object_path = profile->driver_so;
	configuration->driver_program_name = profile->program_name;
	configuration->execution_stream = (void *)1;
	configuration->completion_function = TestCompletionRoute;
	configuration->wake_function = TestWakeRoute;
}

static void TestBuildDecodeSubmission(uint64_t salt,
	SparkModelServingLane *lane,uint32_t token_ids[1],
	uint32_t row_lane_indices[1],uint64_t row_positions[1],
	uint64_t row_sequence_ids[1],SparkModelServingSubmission *submission)
{
	memset(lane,0,sizeof(*lane));
	lane->request_id = 1000ull + salt;
	lane->request_generation = 3ull + salt;
	lane->step_generation = 5ull + salt;
	lane->sequence_id = 7000ull + salt;
	lane->sequence_position = 41ull + salt;
	lane->resident_sequence_slot = 0u;
	lane->context_token_count = 42u;
	lane->input_token_id = 9u;
	lane->flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	token_ids[0] = 55u;
	row_lane_indices[0] = 0u;
	row_positions[0] = 41ull + salt;
	row_sequence_ids[0] = 7000ull + salt;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->request_id = 1000ull + salt;
	submission->sequence_id = 7000ull + salt;
	submission->sequence_position = 41ull + salt;
	submission->submission_id = 90000ull + salt;
	submission->control_generation = 1ull + salt;
	submission->transaction_id = 2ull + salt;
	submission->dispatch_generation = 3ull + salt;
	submission->request_generation = 3ull + salt;
	submission->step_generation = 5ull + salt;
	submission->active_sequence_count = 1u;
	submission->new_token_count = 1u;
	submission->lane_count = 1u;
	submission->row_count = 1u;
	submission->token_count = 1u;
	submission->tokens_per_sequence = 1u;
	submission->lanes = lane;
	submission->token_ids = token_ids;
	submission->row_lane_indices = row_lane_indices;
	submission->row_positions = row_positions;
	submission->row_sequence_ids = row_sequence_ids;
}

static void TestBuildDriverCompletion(
	const SparkAdapterPendingCore *core,uint32_t program_id,
	SparkModelDriverCompletion *completion)
{
	memset(completion,0,sizeof(*completion));
	completion->request_id = core->identity.request_id;
	completion->sequence_id = core->identity.sequence_id;
	completion->sequence_position = core->identity.sequence_position;
	completion->program_id = program_id;
	completion->accepted_token_count = 3u;
}

static void TestStampTailActive(void *pending_table,
	const FamilyProfile *profile,uint32_t index)
{
	if ( pending_table == kProfiles + 0 || profile == &kProfiles[0] )
	{
		Dsv4StyleRecord *records = pending_table;
		records[index].core.active = 1u;
		records[index].speculation_credit = TAIL_MAGIC;
	}
	else if ( profile == &kProfiles[1] )
	{
		Glm52StyleRecord *records = pending_table;
		records[index].core.active = 1u;
		records[index].tap_mask = 0x1Fu;
	}
	else if ( profile == &kProfiles[2] )
	{
		Qwen36StyleRecord *records = pending_table;
		records[index].core.active = 1u;
		records[index].emit_rows[0] = 17u;
	}
	else
	{
		Qwen38StyleRecord *records = pending_table;
		records[index].core.active = 1u;
		records[index].prefix_token_count = 256ull;
	}
}

static void TestDeactivate(void *pending_table,
	const FamilyProfile *profile,uint32_t index)
{
	((SparkAdapterPendingCore *)((char *)pending_table +
		index * profile->record_bytes))->active = 0u;
}

/* ---- per-family scenario ------------------------------------------------ */

static void TestRunFamily(const FamilyProfile *profile)
{
	char label[64];
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	SparkModelServingAdapterConfiguration configuration;
	SparkAdapterCommonState common;
	Dsv4StyleRecord recordsA[16];
	Glm52StyleRecord recordsB[16];
	Qwen36StyleRecord recordsC[16];
	Qwen38StyleRecord recordsD[16];
	void *pending_table;
	SparkAdapterDriverContract contract;
	SparkModelServingLane lane_storage[2];
	uint32_t token_storage[2];
	uint32_t lane_index_storage[2];
	uint64_t position_storage[2];
	uint64_t sequence_storage[2];
	SparkModelServingSubmission submission;
	SparkModelDriverCompletion driver_completion;
	SparkModelServingCompletion completion;
	SparkModelDriverResidencyToken residency_anchor;
	SparkFakeDriverInstance *instance;
	SparkStatus status;
	int32_t claimed;
	uint32_t index;

	snprintf(label,sizeof(label),"%s",profile->label);
	(void)label;
	memset(&common,0,sizeof(common));

	switch ( (int)(profile - kProfiles) )
	{
	case 0: pending_table = recordsA; break;
	case 1: pending_table = recordsB; break;
	case 2: pending_table = recordsC; break;
	default: pending_table = recordsD; break;
	}
	for (index=0u; index<16u; index++)
		memset((char *)pending_table + index * profile->record_bytes,
			0,profile->record_bytes);

	TestBuildDescriptor(profile,&descriptor);
	TestBuildLimits(profile,&limits);
	TestBuildConfiguration(profile,&limits,&configuration);

	/* 1. Configuration law. */
	status = SparkAdapterValidateConfiguration(&descriptor,&configuration,
		profile->program_name,1u);
	CHECK(status == SPARK_STATUS_OK);
	status = SparkAdapterValidateConfiguration(&descriptor,&configuration,
		"not.the.program",1u);
	CHECK(status == SPARK_STATUS_INVALID_ARGUMENT);
	configuration.stage_index = 5u;
	status = SparkAdapterValidateConfiguration(&descriptor,&configuration,
		profile->program_name,1u);
	CHECK(status == SPARK_STATUS_INVALID_ARGUMENT);
	configuration.stage_index = 0u;
	configuration.descriptor_bytes += 1u;
	status = SparkAdapterValidateConfiguration(&descriptor,&configuration,
		profile->program_name,1u);
	CHECK(status == SPARK_STATUS_ABI_MISMATCH);
	configuration.descriptor_bytes -= 1u;

	/* 2. Prologue mirroring. */
	status = SparkAdapterInitializePrologue(&common,&configuration);
	CHECK(status == SPARK_STATUS_OK);
	CHECK(common.pipeline_slot_count == profile->inflight_slots);
	CHECK(common.max_active_sequence_count == profile->inflight_slots);
	CHECK(common.sink.function == TestCompletionRoute);
	CHECK(common.wake_function == TestWakeRoute);
	common.pending = pending_table;
	common.pending_element_bytes = profile->record_bytes;

	/* 3. Driver contract: both schools, drifted identity refused. */
	contract.model_id = profile->model_id;
	contract.model_revision = "drifted-revision";
	contract.stage_name = profile->stage_name;
	contract.target = profile->pin_target ? "cpu.test" : 0;
	contract.description_sha256 =
		"0123456789abcdef0123456789abcdef"
		"0123456789abcdef0123456789abcdef";
	contract.required_program_flags =
		SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED;
	contract.check_kind = profile->check_kind;
	contract.node_context = 0;
	status = SparkAdapterLoadDriver(&common,&configuration,&contract);
	CHECK(status == SPARK_STATUS_TARGET_MISMATCH);
	contract.model_revision = profile->model_revision;
	status = SparkAdapterLoadDriver(&common,&configuration,&contract);
	CHECK(status == SPARK_STATUS_OK);
	CHECK(common.driver_instance != 0 && common.program != 0);
	instance = common.driver_instance;
	CHECK(strcmp(instance->request.node_id,"node.integration") == 0);

	/* 4. Slot-table law over the FAMILY record layout. */
	CHECK(SparkAdapterAvailableSubmissionCount(pending_table,
		profile->record_bytes,profile->inflight_slots) ==
			profile->inflight_slots);
	for (index=0u; index<profile->inflight_slots; index++)
	{
		claimed = SparkAdapterPendingClaim(pending_table,
			profile->record_bytes,profile->inflight_slots);
		CHECK(claimed == (int32_t)index);
		TestStampTailActive(pending_table,profile,(uint32_t)claimed);
	}
	claimed = SparkAdapterPendingClaim(pending_table,profile->record_bytes,
		profile->inflight_slots);
	CHECK(claimed == -1);
	/* Slots beyond the configured count never participate even pre-marked. */
	TestStampTailActive(pending_table,profile,profile->inflight_slots + 4u);
	CHECK(SparkAdapterAvailableSubmissionCount(pending_table,
		profile->record_bytes,profile->inflight_slots) == 0u);
	/* Release every slot except 0: the captured-active slot the later
	 * completion/quiesce legs reason about. */
	for (index=1u; index<profile->inflight_slots; index++)
		TestDeactivate(pending_table,profile,index);

	/* Capture identity into slot 0 through the shared helper; tails stay
	 * family-owned (the canary value survives). */
	TestBuildDecodeSubmission(7ull,&lane_storage[0],&token_storage[0],
		&lane_index_storage[0],&position_storage[0],&sequence_storage[0],
		&submission);
	SparkAdapterPendingCapture(pending_table,&submission);
	CHECK(TestCore(pending_table,profile->record_bytes,0)->
		identity.request_id == 1007ull);
	CHECK(TestCore(pending_table,profile->record_bytes,0)->work_kind ==
		SPARK_MODEL_SERVING_WORK_KIND_DECODE);
	if ( profile == &kProfiles[0] )
		CHECK(recordsA[0].speculation_credit == TAIL_MAGIC);

	/* Row/lane capture maps. */
	{
		uint32_t last_rows_by_lane[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
		memset(last_rows_by_lane,0xFF,sizeof(last_rows_by_lane));
		SparkAdapterCaptureLastRowByLane(&submission,last_rows_by_lane);
		CHECK(last_rows_by_lane[0] == 0u);
	}
	{
		uint32_t slots_per_lane[2];
		uint32_t slots_by_row[2];
		SparkModelServingLane two_lanes[2];
		uint32_t two_tokens[2];
		uint32_t two_lane_indices[2] = { 0u,1u };
		uint64_t two_positions[2] = { 48ull,49ull };
		uint64_t two_sequences[2];
		SparkModelServingSubmission two_row_submission;
		memset(slots_per_lane,0xEE,sizeof(slots_per_lane));
		SparkAdapterCaptureResidentSlotsPerLane(&submission,1u,slots_per_lane);
		CHECK(slots_per_lane[0] == 0u);
		two_sequences[0] = 7007ull;
		two_sequences[1] = 8007ull;
		two_lanes[0] = lane_storage[0];
		two_lanes[1] = lane_storage[0];
		two_lanes[1].sequence_id = 8007ull;
		two_lanes[1].request_id = 1008ull;
		two_lanes[1].sequence_position = 48ull;
		two_lanes[1].resident_sequence_slot = 1u;
		two_tokens[0] = 55u;
		two_tokens[1] = 56u;
		memcpy(&two_row_submission,&submission,sizeof(two_row_submission));
		two_row_submission.lane_count = 2u;
		two_row_submission.row_count = 2u;
		two_row_submission.token_count = 2u;
		two_row_submission.new_token_count = 2u;
		two_row_submission.lanes = two_lanes;
		two_row_submission.token_ids = two_tokens;
		two_row_submission.row_lane_indices = two_lane_indices;
		two_row_submission.row_positions = two_positions;
		two_row_submission.row_sequence_ids = two_sequences;
		memset(slots_per_lane,0xEE,sizeof(slots_per_lane));
		SparkAdapterCaptureResidentSlotsPerLane(&two_row_submission,2u,
			slots_per_lane);
		CHECK(slots_per_lane[0] == 0u && slots_per_lane[1] == 1u);
		memset(slots_by_row,0xEE,sizeof(slots_by_row));
		SparkAdapterCaptureResidentSlotsPerRow(&two_row_submission,slots_by_row);
		CHECK(slots_by_row[0] == 0u && slots_by_row[1] == 1u);
	}

	/* 5. Shared submission-open gate. */
	status = SparkAdapterValidateSubmissionOpen(&common,&descriptor,
		&submission);
	CHECK(status == SPARK_STATUS_OK);

	/* 6. Completion routing: exact match plus four single-field drifts. */
	TestBuildDriverCompletion(TestCore(pending_table,profile->record_bytes,0),
		common.program->program_id,&driver_completion);
	CHECK(SparkAdapterDriverCompletionMatches(&driver_completion,
		driver_completion.request_id,driver_completion.sequence_id,
		driver_completion.sequence_position,
		driver_completion.program_id) == 1u);
	CHECK(SparkAdapterDriverCompletionMatches(&driver_completion,
		driver_completion.request_id + 1u,driver_completion.sequence_id,
		driver_completion.sequence_position,
		driver_completion.program_id) == 0u);
	CHECK(SparkAdapterDriverCompletionMatches(&driver_completion,
		driver_completion.request_id,driver_completion.sequence_id + 1u,
		driver_completion.sequence_position,
		driver_completion.program_id) == 0u);
	CHECK(SparkAdapterDriverCompletionMatches(&driver_completion,
		driver_completion.request_id,driver_completion.sequence_id,
		driver_completion.sequence_position + 1u,
		driver_completion.program_id) == 0u);
	CHECK(SparkAdapterDriverCompletionMatches(&driver_completion,
		driver_completion.request_id,driver_completion.sequence_id,
		driver_completion.sequence_position,
		driver_completion.program_id + 1u) == 0u);
	/* Orphan route fired through the DRIVER-wired callback. */
	g_completion_count = 0u;
	instance->rejected_count = 2ull;
	driver_completion.request_id += 777ull; /* matches nothing reserved */
	instance->request.completion_function(instance->request.completion_context,
		&driver_completion);
	CHECK(common.orphan_completion_count == 1ull);
	CHECK(g_completion_count == 0u); /* orphans never reach the sink */

	/* 7. Wake trampoline forwards to the configuration route. */
	g_wake_count = 0u;
	instance->request.wake_function(instance->request.wake_context);
	CHECK(g_wake_count == 1u);

	/* 8. Residency law: zeroed by default, stamped ABI, copied anchor. */
	residency_anchor.word0 = UINT64_C(0x123456789abcdef0);
	residency_anchor.word1 = UINT64_C(0x0fedcba987654321);
	residency_anchor.generation = 9ull;
	residency_anchor.owner = 4ull;
	memset(&completion,0xAB,sizeof(completion));
	SparkAdapterBuildCompletionHeader(&completion,
		TestCore(pending_table,profile->record_bytes,0));
	CHECK(completion.abi_version ==
		SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION);
	CHECK(completion.request_id == 1007ull &&
		completion.sequence_id == 7007ull &&
		completion.sequence_position == 48ull);
	CHECK(completion.residency.word0 == 0ull &&
		completion.residency.owner == 0ull);
	memset(&completion,0,sizeof(completion));
	SparkAdapterBuildCompletionHeaderWithResidency(&completion,
		TestCore(pending_table,profile->record_bytes,0),&residency_anchor);
	CHECK(completion.residency.generation == 9ull &&
		completion.residency.word0 == residency_anchor.word0);
	memset(&completion,0,sizeof(completion));
	SparkAdapterBuildCompletionHeaderWithResidency(&completion,
		TestCore(pending_table,profile->record_bytes,0),0);
	CHECK(completion.residency.word0 == 0ull);
	CHECK(SparkAdapterClampAcceptedTokenCount(UINT64_MAX) == UINT32_MAX);
	CHECK(SparkAdapterClampAcceptedTokenCount(5ull) == 5u);

	/* 9. Snapshot merge pre-quiesce: availability counts free slots only. */
	instance->submitted_count = 10ull;
	instance->completed_count = 7ull;
	{
		SparkModelServingAdapterSnapshot snapshot;
		status = SparkAdapterSnapshot(&common,&snapshot);
		CHECK(status == SPARK_STATUS_OK);
		CHECK(snapshot.available_submission_count ==
			profile->inflight_slots - 1u); /* slot 0 captured-active */
		CHECK(snapshot.active_submission_count == 1u);
		CHECK(snapshot.submitted_count == 10ull &&
			snapshot.completed_count == 7ull);
		CHECK(snapshot.rejected_count == 3ull); /* 2 driver + 1 orphan */
	}

	/* 9b. Admit-frame trio through the REAL loaded fake driver. */
	{
		SparkModelDriverFrame frame;
		memset(&frame,0,sizeof(frame));
		CHECK(SparkAdapterAdmitFrame(0,common.driver_instance,
			common.program,&submission,&frame,0u) ==
			SPARK_STATUS_INVALID_ARGUMENT);
		CHECK(SparkAdapterAdmitFrame(&common.driver,common.driver_instance,
			0,&submission,&frame,0u) == SPARK_STATUS_INVALID_ARGUMENT);
		CHECK(SparkAdapterAdmitFrame(&common.driver,common.driver_instance,
			common.program,&submission,0,0u) ==
			SPARK_STATUS_INVALID_ARGUMENT);
		instance->submitted_count = 41ull;
		status = SparkAdapterAdmitFrame(&common.driver,
			common.driver_instance,common.program,&submission,&frame,0u);
		CHECK(status == SPARK_STATUS_OK);
		CHECK(instance->submitted_count == 41ull); /* no submit requested */
		CHECK((frame.flags &
			SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
		status = SparkAdapterAdmitFrame(&common.driver,
			common.driver_instance,common.program,&submission,&frame,1u);
		CHECK(status == SPARK_STATUS_OK);
		CHECK(instance->submitted_count == 42ull); /* submit_on_apply fired */
		status = SparkAdapterAdmitFrame(&common.driver,
			common.driver_instance,common.program,0,&frame,0u);
		CHECK(status == SPARK_STATUS_OK); /* from-frame request path */
	}

	/* 9c. Direct public-symbol pins: wake trampoline, counter sums, orphan. */
	g_wake_count = 0u;
	SparkAdapterDispatchWake(&common);
	CHECK(g_wake_count == 1u);
	SparkAdapterDispatchWake(0); /* NULL context is a safe no-op */
	driver_completion.accepted_token_count = 3ull;
	driver_completion.queue_delay_ns = 11ull;
	driver_completion.service_time_ns = 130ull;
	{
		uint64_t accepted_tokens = 0ull,delay_ns = 0ull,service_ns = 0ull;
		SparkAdapterAccumulateFrameCounters(&accepted_tokens,&delay_ns,
			&service_ns,&driver_completion);
		driver_completion.accepted_token_count = 4ull;
		driver_completion.queue_delay_ns = 22ull;
		driver_completion.service_time_ns = 260ull;
		SparkAdapterAccumulateFrameCounters(&accepted_tokens,&delay_ns,
			&service_ns,&driver_completion);
		CHECK(accepted_tokens == 7ull && delay_ns == 33ull &&
			service_ns == 390ull);
	}
	{
		const uint64_t orphans_before = common.orphan_completion_count;
		SparkModelDriverCompletion orphan_payload;
		memset(&orphan_payload,0,sizeof(orphan_payload));
		SparkAdapterOrphanDriverCompletion(&common,&orphan_payload);
		SparkAdapterOrphanDriverCompletion(0,0); /* NULL-safe no-op */
		CHECK(common.orphan_completion_count == orphans_before + 1ull);
	}

	/* 10. Quiesce lifecycle. */
	status = SparkAdapterQuiesce(&common,1000ull);
	CHECK(status == SPARK_STATUS_BUSY); /* slot 0 still claimed */
	TestDeactivate(pending_table,profile,0);
	instance->active_submission_count = 1u;
	status = SparkAdapterQuiesce(&common,1000ull);
	CHECK(status == SPARK_STATUS_BUSY); /* driver reports activity */
	instance->active_submission_count = 0u;
	status = SparkAdapterQuiesce(&common,1000ull);
	CHECK(status == SPARK_STATUS_OK);
	status = SparkAdapterQuiesce(&common,0ull);
	CHECK(status == SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkAdapterValidateSubmissionOpen(&common,&descriptor,
		&submission);
	CHECK(status == SPARK_STATUS_BUSY); /* latch permanent */

	/* 11. Destroy guard + idempotent teardown. */
	CHECK(SparkAdapterDestroyReady(&common) != 0u);
	SparkAdapterTeardownDriver(&common);
	/* The skeleton unloads the archive; the instance pointer itself is
	 * caller-owned (family destroy callbacks release their own state). */
	CHECK(common.driver.interface == 0 &&
		common.driver.dynamic_library == 0);
	SparkAdapterTeardownDriver(&common); /* second call is a safe no-op */
}

/* ---- env parsing trio + formatter --------------------------------------- */

static void TestEnvironmentLaws(void)
{
	char buffer[32];
	const char *on_name = "SPARK_TEST_INTEG_DEFAULT_ON";
	const char *exact_name = "SPARK_TEST_INTEG_EXACT_ZERO";
	const char *truthy_name = "SPARK_TEST_INTEG_TRUTHY";
	unsetenv(on_name);
	unsetenv(exact_name);
	unsetenv(truthy_name);
	CHECK(SparkAdapterEnvFlagDefaultOn(on_name) == 1u);
	setenv(on_name,"",1); CHECK(SparkAdapterEnvFlagDefaultOn(on_name) == 1u);
	setenv(on_name,"0",1); CHECK(SparkAdapterEnvFlagDefaultOn(on_name) == 0u);
	setenv(on_name,"00",1); CHECK(SparkAdapterEnvFlagDefaultOn(on_name) == 1u);
	setenv(on_name,"off",1); CHECK(SparkAdapterEnvFlagDefaultOn(on_name) == 1u);
	CHECK(SparkAdapterEnvFlagDefaultOffExactZero(exact_name) == 0u);
	setenv(exact_name,"",1);
	CHECK(SparkAdapterEnvFlagDefaultOffExactZero(exact_name) == 0u);
	setenv(exact_name,"0",1);
	CHECK(SparkAdapterEnvFlagDefaultOffExactZero(exact_name) == 0u);
	setenv(exact_name,"01",1);
	CHECK(SparkAdapterEnvFlagDefaultOffExactZero(exact_name) == 1u);
	CHECK(SparkAdapterEnvFlagDefaultOffTruthy(truthy_name) == 0u);
	setenv(truthy_name,"",1);
	CHECK(SparkAdapterEnvFlagDefaultOffTruthy(truthy_name) == 0u);
	setenv(truthy_name,"01",1);
	CHECK(SparkAdapterEnvFlagDefaultOffTruthy(truthy_name) == 0u);
	setenv(truthy_name,"1",1);
	CHECK(SparkAdapterEnvFlagDefaultOffTruthy(truthy_name) == 1u);
	unsetenv(on_name);
	unsetenv(exact_name);
	unsetenv(truthy_name);
	/* Loud-fail staging: NULL name/value refuse, valid pair round-trips. */
	CHECK(SparkAdapterSetEnvironmentText("SPARK_TEST_INTEG_STAGED","v") ==
		SPARK_STATUS_OK);
	CHECK(strcmp(SparkAdapterEnvText("SPARK_TEST_INTEG_STAGED"),"v") == 0);
	CHECK(SparkAdapterSetEnvironmentText(0,"v") == SPARK_STATUS_INTERNAL_ERROR);
	unsetenv("SPARK_TEST_INTEG_STAGED");
	CHECK(SparkAdapterEnvText("SPARK_TEST_INTEG_ABSENT") == 0);
	SparkAdapterFormatEnvironmentUnsigned(buffer,sizeof(buffer),0u);
	CHECK(strcmp(buffer,"0") == 0);
	SparkAdapterFormatEnvironmentUnsigned(buffer,sizeof(buffer),42u);
	CHECK(strcmp(buffer,"42") == 0);
	SparkAdapterFormatEnvironmentUnsigned(buffer,sizeof(buffer),UINT32_MAX);
	CHECK(strcmp(buffer,"4294967295") == 0);
}

/* ---- spin-lock law: held-value pinning + concurrent exactness ----------- */

enum { SPIN_THREADS = 4,SPIN_ITERS = 20000 };

static _Atomic uint32_t g_spin_lock;
static uint64_t g_spin_protected;

static void *SpinHammer(void *argument)
{
	uint32_t index;
	(void)argument;
	for (index = 0u; index < (uint32_t)SPIN_ITERS; index++)
	{
		SparkAdapterSpinLockAcquire(&g_spin_lock);
		g_spin_protected++;
		SparkAdapterSpinLockRelease(&g_spin_lock);
	}
	return(0);
}

static void TestSpinLockLaws(void)
{
	pthread_t threads[SPIN_THREADS];
	uint32_t index;
	atomic_init(&g_spin_lock,0u);
	g_spin_protected = 0ull;
	SparkAdapterSpinLockAcquire(&g_spin_lock);
	CHECK(g_spin_lock == 1u); /* held value pins to exactly one */
	SparkAdapterSpinLockRelease(&g_spin_lock);
	CHECK(g_spin_lock == 0u); /* release returns the word to zero */
	for (index = 0u; index < (uint32_t)SPIN_THREADS; index++)
		CHECK(pthread_create(&threads[index],0,SpinHammer,0) == 0);
	for (index = 0u; index < (uint32_t)SPIN_THREADS; index++)
		CHECK(pthread_join(threads[index],0) == 0);
	CHECK(g_spin_protected == (uint64_t)SPIN_THREADS * (uint64_t)SPIN_ITERS);
	CHECK(g_spin_lock == 0u); /* hammer leaves the word unlocked */
}

int main(void)
{
	uint32_t index;
	TestEnvironmentLaws();
	TestSpinLockLaws();
	for (index=0u; index<sizeof(kProfiles)/sizeof(kProfiles[0]); index++)
		TestRunFamily(&kProfiles[index]);
	printf("%s adapter-common integration: %u checks, %u failures\n",
		g_failures != 0u ? "FAIL" : "PASS",g_checks,g_failures);
	return(g_failures != 0u ? 1 : 0);
}
