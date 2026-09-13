// runtime/adapter_common.c - integration coverage for every shared
// serving-adapter helper, host-side, no GPU.
//
// Strategy mirrors how the helpers are consumed by the four family
// adapters: a fake SparkModelDriverInterface vtable stands in for the
// loaded driver so Quiesce/Snapshot/DestroyReady/TeardownDriver/AdmitFrame
// exercise their real control flow against scripted snapshot/admit
// callbacks. LoadDriver's dlopen happy path stays covered by the family
// dylib builds and fleet smokes; here it is gated through its error path.
// LoadTpCollective runs real stanzas through SparkJsonParseText.
//
// Helper inventory covered (30): PendingCapture, PendingClaim,
// AvailableSubmissionCount, CaptureLastRowByLane,
// CaptureResidentSlotsPerLane, CaptureResidentSlotsPerRow,
// OrphanDriverCompletion, DispatchWake, DriverCompletionMatches,
// AccumulateFrameCounters, BuildCompletionHeader,
// BuildCompletionHeaderWithResidency, ClampAcceptedTokenCount, EnvText,
// EnvFlagDefaultOn, EnvFlagDefaultOffExactZero, EnvFlagDefaultOffTruthy,
// SpinLockAcquire, SpinLockRelease, SetEnvironmentText,
// FormatEnvironmentUnsigned, ValidateConfiguration, InitializePrologue,
// LoadDriver, ValidateSubmissionOpen, Quiesce, Snapshot, DestroyReady,
// TeardownDriver, AdmitFrame, LoadTpCollective.

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/adapter_common.h"
#include "sparkpipe/spark_admission.h"

static int32_t failures = 0;

#define CHECK_STATUS(label, got, want) \
	do { expect((got) == (want), label); } while (0)

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if (!condition)
		++failures;
}

/* ---- fixtures ---------------------------------------------------------- */

static const uint32_t kTokenIds[4] = { 9u, 8u, 7u, 6u };
static const uint32_t kLaneOfRow[4] = { 0u, 1u, 0u, 1u };
static const uint64_t kPositionOfRow[4] = { 10u, 20u, 11u, 21u };
static const uint64_t kSequenceOfRow[4] = { 77u, 88u, 77u, 88u };
static const uint32_t kSlotOfLane[4] = { 3u, 1u, 0u, 2u };

static void attach_rows(SparkModelServingSubmission *s)
{
	static SparkModelServingLane lanes[4];
	uint32_t i;
	for (i = 0u; i < 4u; i++)
	{
		memset(&lanes[i], 0, sizeof(lanes[i]));
		lanes[i].resident_sequence_slot = kSlotOfLane[i];
		lanes[i].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	}
	/* Lane 0 is a fully valid LIVE lane under the runtime validator:
	 * identity present, context = last position + 1. */
	lanes[0].request_id = 2002u;
	lanes[0].request_generation = 43u;
	lanes[0].step_generation = 44u;
	lanes[0].sequence_id = 77u;
	lanes[0].sequence_position = 10u;
	lanes[0].context_token_count = 11u;
	s->lanes = lanes;
	s->token_ids = kTokenIds;
	s->row_lane_indices = kLaneOfRow;
	s->row_positions = kPositionOfRow;
	s->row_sequence_ids = kSequenceOfRow;
}

static SparkModelServingSubmission make_submission(void)
{
	SparkModelServingSubmission s;
	memset(&s, 0, sizeof(s));
	s.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	s.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	s.submission_id = 1001u;
	s.request_id = 2002u;
	s.sequence_id = 3003u;
	s.sequence_position = 7u;
	s.control_generation = 41u;
	s.transaction_id = 5005u;
	s.dispatch_generation = 42u;
	s.request_generation = 43u;
	s.step_generation = 44u;
	s.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	s.row_count = 1u;
	s.lane_count = 1u;
	s.active_sequence_count = 1u;
	s.token_count = 1u;
	s.new_token_count = 1u;
	s.tokens_per_sequence = 1u;
	attach_rows(&s);
	return s;
}

typedef struct FakeDriverState
{
	SparkLoadedModelDriver handle;
	void *instance;
	SparkModelDriverInterface iface;
	SparkModelDriverProgramDescriptor program;
	SparkModelDriverProgramProfile profile;
	int create_calls;
	int destroy_calls;
	int admit_calls;
	int snapshot_calls;
	uint32_t captured_program_id;
	int snapshot_mode; /* 0 idle-ok, 1 active-busy, 2 error */
	SparkStatus admit_status;
	void *admit_seen_instance;
} FakeDriverState;

static SparkStatus fake_create(const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	FakeDriverState *d = (FakeDriverState *)request->completion_context;
	d->create_calls++;
	*driver_instance = d;
	return SPARK_STATUS_OK;
}

static void fake_destroy(void *driver_instance)
{
	((FakeDriverState *)driver_instance)->destroy_calls++;
}

static SparkStatus fake_admit(void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	FakeDriverState *d = (FakeDriverState *)driver_instance;
	d->admit_calls++;
	d->captured_program_id = request->program_id;
	d->admit_seen_instance = driver_instance;
	if (d->admit_status != SPARK_STATUS_OK)
		return d->admit_status;
	memset(decision, 0, sizeof(*decision));
	decision->descriptor_bytes = (uint32_t)sizeof(*decision);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	return SPARK_STATUS_OK;
}

static SparkStatus fake_snapshot(void *driver_instance, uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	FakeDriverState *d = (FakeDriverState *)driver_instance;
	d->snapshot_calls++;
	d->captured_program_id = program_id;
	if (d->snapshot_mode == 2)
		return SPARK_STATUS_INTERNAL_ERROR;
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->available_dispatch_slot_count =
		d->snapshot_mode == 1 ? 0u : 16u;
	snapshot->active_submission_count =
		d->snapshot_mode == 1 ? 2u : 0u;
	snapshot->submitted_count = 100u;
	snapshot->completed_count = 98u;
	snapshot->rejected_count = 3u;
	snapshot->resident_sequence_count = 5u;
	snapshot->resident_token_count = 55u;
	snapshot->kv_token_capacity = 4096u;
	snapshot->device_memcpy_bytes_per_submit = 12u;
	snapshot->host_staging_bytes_per_submit = 34u;
	return SPARK_STATUS_OK;
}

static void fake_driver_init(FakeDriverState *d)
{
	memset(d, 0, sizeof(*d));
	SparkLoadedModelDriverReset(&d->handle);
	d->iface.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION;
	d->iface.interface_bytes = (uint32_t)sizeof(d->iface);
	d->iface.create = fake_create;
	d->iface.destroy = fake_destroy;
	d->iface.admit = fake_admit;
	d->iface.snapshot = fake_snapshot;
	d->handle.interface = &d->iface;
	d->instance = d;
	d->program.program_id = 7u;
	d->program.max_inflight = 16u;
	d->profile.descriptor_bytes = (uint32_t)sizeof(d->profile);
	d->profile.max_active_slots = 16u;
	d->profile.max_new_tokens = 16u;
	d->program.profile = &d->profile;
	d->admit_status = SPARK_STATUS_OK;
	d->snapshot_mode = 0;
}

static SparkAdapterCommonState make_common(FakeDriverState *d,
	SparkAdapterPendingCore *table, size_t stride, uint32_t slots)
{
	SparkAdapterCommonState common;
	memset(&common, 0, sizeof(common));
	common.driver = d->handle;
	common.driver_instance = d->instance;
	common.program = &d->program;
	common.pending = table;
	common.pending_element_bytes = stride;
	common.pipeline_slot_count = slots;
	return common;
}

static SparkModelServingAdapterDescriptor make_descriptor(void)
{
	SparkModelServingAdapterDescriptor d;
	memset(&d, 0, sizeof(d));
	d.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	d.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	d.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE;
	d.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	d.boundary_element_count = 7168u;
	d.boundary_element_bytes = (uint32_t)sizeof(uint16_t);
	d.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	d.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	d.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	d.adapter_id = "spark.ac.test";
	d.model_id = "ac/test";
	d.model_revision = "r1";
	d.driver_program_name = "p";
	d.artifact_sha256 =
		"1111111111111111111111111111111111111111111111111111111111111111";
	d.stage_layer_counts[0] = 4u;
	d.stage_layer_counts[1] = 4u;
	d.minimum_efficient_submission_row_count = 1u;
	d.stage_count = 2u;
	d.layer_count = 8u;
	d.max_inflight_submission_count = 4u;
	d.max_active_sequence_count = 2u;
	d.max_input_row_count = 2u;
	d.max_resident_sequence_count = 4u;
	d.max_output_token_count = 16u;
	return d;
}

static size_t offsetof_identity_tail(void);

/* ---- pending table ------------------------------------------------------ */

static void test_pending_table(void)
{
	SparkAdapterPendingCore table[4];
	SparkModelServingSubmission s = make_submission();
	SparkAdapterPendingCore *core;
	struct Padded { SparkAdapterPendingCore core; char tail[32]; };
	memset(table, 0, sizeof(table));

	expect(SparkAdapterPendingClaim(table, sizeof(table[0]), 0u) == -1,
		"claim zero-slot table busy");
	expect(SparkAdapterPendingClaim(table, sizeof(table[0]), 4u) == 0,
		"claim first free slot 0");
	core = &table[0];
	SparkAdapterPendingCapture(core, &s);
	core->active = 1u;
	expect(core->identity.request_id == 2002u &&
		core->identity.step_generation == 44u &&
		core->identity.control_generation == 41u &&
		core->row_count == 1u && core->lane_count == 1u &&
		core->active_sequence_count == 1u &&
		core->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE,
		"capture echoes shape + nine-field identity");
	expect(SparkAdapterPendingClaim(table, sizeof(table[0]), 4u) == 1,
		"claim skips active slot");
	table[1].active = 1u;
	table[2].active = 1u;
	table[3].active = 1u;
	expect(SparkAdapterPendingClaim(table, sizeof(table[0]), 4u) == -1,
		"claim full table -> -1");
	expect(SparkAdapterAvailableSubmissionCount(table,
		sizeof(table[0]), 4u) == 0u, "available full table 0");
	table[2].active = 0u;
	expect(SparkAdapterAvailableSubmissionCount(table,
		sizeof(table[0]), 4u) == 1u, "available counts inactive");
}

static void test_padded_stride(void)
{
	struct Padded { SparkAdapterPendingCore core; char tail[32]; };
	struct Padded wide[3];
	memset(wide, 0, sizeof(wide));
	wide[1].core.active = 1u;
	expect(SparkAdapterPendingClaim(wide, sizeof(wide[0]), 3u) == 0,
		"padded-stride claim finds slot 0");
	wide[0].core.active = 1u;
	expect(SparkAdapterPendingClaim(wide, sizeof(wide[0]), 3u) == 2,
		"padded-stride claim walks past wide record");
	expect(SparkAdapterAvailableSubmissionCount(wide,
		sizeof(wide[0]), 3u) == 1u, "padded-stride availability");
}

/* ---- row captures -------------------------------------------------------- */

static void test_row_captures(void)
{
	SparkModelServingSubmission s = make_submission();
	uint32_t last_row_by_lane[4] = { 99u, 99u, 99u, 99u };
	uint32_t per_lane[4];
	uint32_t per_row[4];
	uint32_t i;
	int ok;
	s.row_count = 4u;
	s.lane_count = 4u;
	s.active_sequence_count = 4u;
	attach_rows(&s);

	SparkAdapterCaptureLastRowByLane(&s, last_row_by_lane);
	/* Rows 0,2 belong to lane 0; rows 1,3 to lane 1: LAST wins. */
	expect(last_row_by_lane[0] == 2u && last_row_by_lane[1] == 3u &&
		last_row_by_lane[2] == 99u && last_row_by_lane[3] == 99u,
		"last-row-by-lane keeps final row per lane");

	SparkAdapterCaptureResidentSlotsPerLane(&s, 4u, per_lane);
	ok = 1;
	for (i = 0u; i < 4u; i++)
		if (per_lane[i] != kSlotOfLane[i])
			ok = 0;
	expect(ok, "slots-per-lane mirrors lanes[] order");

	SparkAdapterCaptureResidentSlotsPerRow(&s, per_row);
	ok = 1;
	for (i = 0u; i < 4u; i++)
		if (per_row[i] != kSlotOfLane[kLaneOfRow[i]])
			ok = 0;
	expect(ok, "slots-per-row maps row -> own-lane slot");
}

/* ---- completion routing --------------------------------------------------- */

static int wake_calls = 0;
static void *wake_seen_context = 0;
static void counting_wake(void *context)
{
	wake_calls++;
	wake_seen_context = context;
}

static void test_completion_routing(void)
{
	SparkAdapterCommonState common;
	SparkModelDriverCompletion dc;
	SparkAdapterPendingCore core;
	SparkModelServingCompletion completion;
	SparkModelDriverResidencyToken anchor;
	uint64_t tokens = 10u, delay = 20u, service = 30u;
	unsigned char *bytes;
	size_t i;
	int clean;

	memset(&common, 0, sizeof(common));
	SparkAdapterOrphanDriverCompletion(&common, 0);
	expect(common.orphan_completion_count == 1u,
		"orphan callback bumps counter");
	SparkAdapterOrphanDriverCompletion(0, 0);
	expect(common.orphan_completion_count == 1u,
		"orphan callback NULL-context safe");

	common.wake_function = counting_wake;
	common.wake_context = &common;
	wake_calls = 0;
	SparkAdapterDispatchWake(&common);
	expect(wake_calls == 1 && wake_seen_context == &common,
		"dispatch-wake forwards through route");
	common.wake_function = 0;
	SparkAdapterDispatchWake(&common);
	expect(wake_calls == 1, "dispatch-wake without route is a no-op");
	SparkAdapterDispatchWake(0);
	expect(wake_calls == 1, "dispatch-wake NULL-state safe");

	memset(&dc, 0, sizeof(dc));
	dc.request_id = 5u; dc.sequence_id = 6u;
	dc.sequence_position = 7u; dc.program_id = 8u;
	expect(SparkAdapterDriverCompletionMatches(&dc, 5u, 6u, 7u, 8u) == 1u,
		"matches on all-four agreement");
	expect(SparkAdapterDriverCompletionMatches(&dc, 9u, 6u, 7u, 8u) == 0u &&
		SparkAdapterDriverCompletionMatches(&dc, 5u, 9u, 7u, 8u) == 0u &&
		SparkAdapterDriverCompletionMatches(&dc, 5u, 6u, 9u, 8u) == 0u &&
		SparkAdapterDriverCompletionMatches(&dc, 5u, 6u, 7u, 9u) == 0u,
		"any single-field mismatch rejects");

	dc.accepted_token_count = 4u;
	dc.queue_delay_ns = 5u;
	dc.service_time_ns = 6u;
	SparkAdapterAccumulateFrameCounters(&tokens, &delay, &service, &dc);
	expect(tokens == 14u && delay == 25u && service == 36u,
		"frame counters accumulate into u64 accumulators");

	memset(&core, 0, sizeof(core));
	core.identity.submission_id = 1u; core.identity.request_id = 2u;
	core.identity.sequence_id = 3u; core.identity.sequence_position = 4u;
	core.identity.control_generation = 5u;
	core.identity.transaction_id = 6u;
	core.identity.dispatch_generation = 7u;
	core.identity.request_generation = 8u;
	core.identity.step_generation = 9u;
	memset(&completion, 0xA5, sizeof(completion));
	SparkAdapterBuildCompletionHeader(&completion, &core);
	expect(completion.abi_version ==
			SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION &&
		completion.descriptor_bytes ==
			SPARK_MODEL_SERVING_COMPLETION_BYTES,
		"header stamps ABI version + descriptor bytes (ABI LAW)");
	expect(completion.submission_id == 1u && completion.request_id == 2u &&
		completion.sequence_id == 3u &&
		completion.sequence_position == 4u &&
		completion.control_generation == 5u &&
		completion.transaction_id == 6u &&
		completion.dispatch_generation == 7u &&
		completion.request_generation == 8u &&
		completion.step_generation == 9u,
		"header echoes nine-field identity");
	clean = 1;
	for (i = offsetof_identity_tail(); i < sizeof(completion); i++)
		if (((unsigned char *)&completion)[i] != 0) { clean = 0; break; }
	expect(clean, "header zeroes everything past the echo");

	memset(&anchor, 0xB7, sizeof(anchor));
	SparkAdapterBuildCompletionHeaderWithResidency(&completion, &core,
		&anchor);
	expect(memcmp(&completion.residency, &anchor,
		sizeof(anchor)) == 0, "with-residency copies anchor token");
	SparkAdapterBuildCompletionHeaderWithResidency(&completion, &core, 0);
	bytes = (unsigned char *)&completion.residency;
	clean = 1;
	for (i = 0; i < sizeof(completion.residency); i++)
		if (bytes[i] != 0) { clean = 0; break; }
	expect(clean, "NULL residency keeps zeroed default (residency law)");

	expect(SparkAdapterClampAcceptedTokenCount(0u) == 0u &&
		SparkAdapterClampAcceptedTokenCount(4095u) == 4095u &&
		SparkAdapterClampAcceptedTokenCount(UINT32_MAX) == UINT32_MAX &&
		SparkAdapterClampAcceptedTokenCount((uint64_t)UINT32_MAX + 1u) ==
			UINT32_MAX &&
		SparkAdapterClampAcceptedTokenCount(~(uint64_t)0u) == UINT32_MAX,
		"clamp saturates u64 -> u32 at UINT32_MAX");
}

/* offset of the tail after step_generation in the completion struct */
static size_t offsetof_identity_tail(void)
{
	SparkModelServingCompletion c;
	return (size_t)((char *)&c.step_generation + sizeof(c.step_generation) -
		(char *)&c);
}

/* ---- env readers ----------------------------------------------------------- */

static void test_env_readers(void)
{
	const char *n = "SPARK_AC_TEST_ENV";
	unsetenv(n);
	expect(SparkAdapterEnvText(n) == 0, "env-text unset -> NULL");
	setenv(n, "hello", 1);
	expect(strcmp(SparkAdapterEnvText(n), "hello") == 0,
		"env-text set -> value");

	setenv(n, "", 1);
	expect(SparkAdapterEnvFlagDefaultOn(n) == 1u, "default-on empty ON");
	setenv(n, "0", 1);
	expect(SparkAdapterEnvFlagDefaultOn(n) == 0u, "default-on zero OFF");
	unsetenv(n);
	expect(SparkAdapterEnvFlagDefaultOn(n) == 1u, "default-on unset ON");
	setenv(n, "00", 1);
	expect(SparkAdapterEnvFlagDefaultOn(n) == 1u,
		"default-on double-zero ON (strcmp law)");

	setenv(n, "", 1);
	expect(SparkAdapterEnvFlagDefaultOffExactZero(n) == 0u,
		"exact-zero empty OFF");
	setenv(n, "0", 1);
	expect(SparkAdapterEnvFlagDefaultOffExactZero(n) == 0u,
		"exact-zero zero OFF");
	setenv(n, "01", 1);
	expect(SparkAdapterEnvFlagDefaultOffExactZero(n) == 1u,
		"exact-zero zero-one ON");
	unsetenv(n);
	expect(SparkAdapterEnvFlagDefaultOffExactZero(n) == 0u,
		"exact-zero unset OFF");

	setenv(n, "01", 1);
	expect(SparkAdapterEnvFlagDefaultOffTruthy(n) == 0u,
		"truthy zero-one OFF (first-byte law)");
	setenv(n, "10", 1);
	expect(SparkAdapterEnvFlagDefaultOffTruthy(n) == 1u, "truthy one-zero ON");
	setenv(n, "", 1);
	expect(SparkAdapterEnvFlagDefaultOffTruthy(n) == 0u, "truthy empty OFF");
	unsetenv(n);
	expect(SparkAdapterEnvFlagDefaultOffTruthy(n) == 0u, "truthy unset OFF");
	unsetenv(n);
}

/* ---- spinlock + staging ----------------------------------------------------- */

static void test_spinlocks_and_staging(void)
{
	_Atomic uint32_t lock = 0u;
	char buffer[8];

	SparkAdapterSpinLockAcquire(&lock);
	expect(atomic_load_explicit(&lock, memory_order_relaxed) == 1u,
		"spinlock acquire sets held");
	SparkAdapterSpinLockRelease(&lock);
	expect(atomic_load_explicit(&lock, memory_order_relaxed) == 0u,
		"spinlock release clears held");
	SparkAdapterSpinLockAcquire(&lock);
	SparkAdapterSpinLockRelease(&lock);
	expect(atomic_load_explicit(&lock, memory_order_relaxed) == 0u,
		"spinlock reacquire/release round-trip");

	expect(SparkAdapterSetEnvironmentText("SPARK_AC_STAGE", "v7") ==
		SPARK_STATUS_OK && strcmp(getenv("SPARK_AC_STAGE"), "v7") == 0,
		"set-environment-text stages value (overwrite mode)");
	expect(SparkAdapterSetEnvironmentText("bad=name", "x") ==
		SPARK_STATUS_INTERNAL_ERROR,
		"setenv failure maps to INTERNAL_ERROR (loud-fail law)");
	unsetenv("SPARK_AC_STAGE");

	SparkAdapterFormatEnvironmentUnsigned(buffer, sizeof(buffer), 0u);
	expect(strcmp(buffer, "0") == 0, "format unsigned zero");
	{
		char wide[16];
		SparkAdapterFormatEnvironmentUnsigned(wide, sizeof(wide),
			4294967295u);
		expect(strcmp(wide, "4294967295") == 0,
			"format unsigned u32-max");
	}
	memset(buffer, 'X', sizeof(buffer));
	SparkAdapterFormatEnvironmentUnsigned(buffer, 3u, 123456u);
	expect(buffer[2] == '\0' && buffer[0] == '1' && buffer[1] == '2',
		"truncated format still NUL-terminates");
}

/* ---- validate configuration ---------------------------------------------- */

static void test_validate_configuration(void)
{
	SparkModelServingAdapterDescriptor d = make_descriptor();
	SparkModelServingAdapterConfiguration c;
	memset(&c, 0, sizeof(c));
	c.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	c.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	c.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	c.runtime_limits.descriptor_bytes =
		SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	c.runtime_limits.max_inflight_submission_count = 2u;
	c.runtime_limits.max_active_sequence_count = 2u;
	c.runtime_limits.max_input_row_count = 2u;
	c.runtime_limits.resident_sequence_capacity = 2u;
	c.stage_index = 0u;
	c.runtime_root = "/tmp";
	c.node_id = "node-a";
	c.node_target = "cuda.sm121";
	c.adapter_configuration_path = "/tmp/cfg.json";
	c.driver_shared_object_path = "/tmp/driver.so";
	c.driver_program_name = "resident_decode";
	c.execution_stream = (void *)1;
	c.completion_function = (void *)1;

	CHECK_STATUS("validate-config happy", SparkAdapterValidateConfiguration(
		&d, &c, "resident_decode", 2u), SPARK_STATUS_OK);
	CHECK_STATUS("validate-config null config",
		SparkAdapterValidateConfiguration(&d, 0, "x", 2u),
		SPARK_STATUS_INVALID_ARGUMENT);
	{
		SparkModelServingAdapterConfiguration bad = c;
		bad.abi_version = 0u;
		CHECK_STATUS("validate-config abi mismatch",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_ABI_MISMATCH);
		bad = c;
		bad.descriptor_bytes = 1u;
		CHECK_STATUS("validate-config descriptor-bytes mismatch",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_ABI_MISMATCH);
		bad = c;
		bad.runtime_limits.max_inflight_submission_count =
			d.max_inflight_submission_count + 1u;
		CHECK_STATUS("validate-config limits exceed descriptor",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_INVALID_ARGUMENT);
		bad = c;
		bad.stage_index = 2u;
		CHECK_STATUS("validate-config stage-index bounds",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_INVALID_ARGUMENT);
		bad = c;
		bad.completion_function = 0;
		CHECK_STATUS("validate-config missing sink",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_INVALID_ARGUMENT);
		bad = c;
		bad.runtime_root = 0;
		CHECK_STATUS("validate-config missing runtime root",
			SparkAdapterValidateConfiguration(&d, &bad, "x", 2u),
			SPARK_STATUS_INVALID_ARGUMENT);
		bad = c;
		bad.driver_program_name = "other";
		CHECK_STATUS("validate-config program-name pin",
			SparkAdapterValidateConfiguration(&d, &bad,
				"resident_decode", 2u),
			SPARK_STATUS_INVALID_ARGUMENT);
	}
}

/* ---- prologue + open gate ------------------------------------------------ */

static void test_prologue_and_open_gate(void)
{
	SparkAdapterCommonState common;
	SparkAdapterPendingCore table[2];
	SparkModelServingRuntimeLimits lim;
	SparkModelServingSubmission s = make_submission();
	SparkModelServingAdapterDescriptor dd = make_descriptor();

	memset(table, 0, sizeof(table));
	memset(&common, 0, sizeof(common));
	memset(&lim, 0, sizeof(lim));
	lim.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	lim.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	lim.max_inflight_submission_count = 4u;
	lim.max_active_sequence_count = 2u;
	lim.max_input_row_count = 2u;
	lim.resident_sequence_capacity = 4u;
	common.pipeline_slot_count = 2u;
	common.pending = table;
	common.pending_element_bytes = sizeof(table[0]);

	{
		SparkModelServingAdapterConfiguration c;
		memset(&c, 0, sizeof(c));
		c.execution_stream = (void *)1234;
		c.completion_function = (void *)1;
		c.completion_context = (void *)2;
		c.wake_function = (void *)3;
		c.wake_context = (void *)4;
		c.runtime_limits = lim;
		expect(SparkAdapterInitializePrologue(&common, &c) ==
			SPARK_STATUS_OK, "prologue returns OK");
		expect(common.pipeline_slot_count == 4u &&
			common.max_active_sequence_count == 2u &&
			common.max_input_row_count == 2u &&
			common.resident_sequence_capacity == 4u &&
			common.execution_stream == (void *)1234 &&
			common.sink.function == (void *)1 &&
			common.sink.context == (void *)2 &&
			common.wake_function == (void *)3 &&
			common.wake_context == (void *)4 &&
			memcmp(&common.runtime_limits, &lim, sizeof(lim)) == 0,
			"prologue copies routes + limits + stream");
	}

	common.runtime_limits = lim;
	CHECK_STATUS("open-gate null state",
		SparkAdapterValidateSubmissionOpen(0, &dd, &s),
		SPARK_STATUS_INVALID_ARGUMENT);
	common.quiescing = 1u;
	CHECK_STATUS("open-gate quiescing BUSY",
		SparkAdapterValidateSubmissionOpen(&common, &dd, &s),
		SPARK_STATUS_BUSY);
	common.quiescing = 0u;
	CHECK_STATUS("open-gate delegates to runtime validator",
		SparkAdapterValidateSubmissionOpen(&common, &dd, &s),
		SPARK_STATUS_OK);
}

/* ---- quiesce / snapshot / destroy / teardown ------------------------------- */

static void test_quiesce_snapshot_destroy(void)
{
	FakeDriverState d;
	fake_driver_init(&d);
	{
		/* Driver-less state: only the argument gate may run safely. */
		SparkAdapterCommonState bare;
		memset(&bare, 0, sizeof(bare));
		expect(SparkAdapterQuiesce(&bare, 0u) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"quiesce zero-deadline gate fires before driver touch");
	}
	{
		SparkAdapterPendingCore table[2];
		SparkAdapterCommonState common;
		SparkModelServingAdapterSnapshot snap;
		memset(table, 0, sizeof(table));
		common = make_common(&d, table, sizeof(table[0]), 2u);

		expect(SparkAdapterQuiesce(0, 1u) ==
			SPARK_STATUS_INVALID_ARGUMENT, "quiesce null-state gate");
		expect(SparkAdapterQuiesce(&common, 0u) ==
			SPARK_STATUS_INVALID_ARGUMENT, "quiesce zero-deadline gate");

		table[0].active = 1u;
		common.quiescing = 0u;
		CHECK_STATUS("quiesce BUSY when slots claimed",
			SparkAdapterQuiesce(&common, 1u), SPARK_STATUS_BUSY);
		expect(common.quiescing == 1u, "quiesce latches even when BUSY");
		table[0].active = 0u;

		d.snapshot_mode = 2;
		CHECK_STATUS("quiesce propagates driver-poll error",
			SparkAdapterQuiesce(&common, 1u),
			SPARK_STATUS_INTERNAL_ERROR);
		expect(d.snapshot_calls == 1 && d.captured_program_id == 7u,
			"quiesce polls driver with program id");

		d.snapshot_mode = 1;
		CHECK_STATUS("quiesce BUSY on active submissions",
			SparkAdapterQuiesce(&common, 1u), SPARK_STATUS_BUSY);

		d.snapshot_mode = 0;
		CHECK_STATUS("quiesce OK when idle", SparkAdapterQuiesce(&common, 1u),
			SPARK_STATUS_OK);
		expect(d.snapshot_calls == 3, "quiesce polled three times total");

		common.quiescing = 0u;
		common.orphan_completion_count = 6u;
		memset(&snap, 0xEE, sizeof(snap));
		CHECK_STATUS("snapshot happy", SparkAdapterSnapshot(&common, &snap),
			SPARK_STATUS_OK);
		expect(snap.abi_version == SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION &&
			snap.descriptor_bytes ==
				SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES,
			"snapshot stamps ABI + bytes");
		expect(snap.available_submission_count == 2u,
			"snapshot availability min(host table, driver)");
		expect(snap.active_submission_count == 0u,
			"snapshot active = slots - free");
		expect(snap.submitted_count == 100u && snap.completed_count == 98u,
			"snapshot copies driver counters");
		expect(snap.rejected_count == 9u,
			"snapshot adds orphan count into rejected");
		expect(snap.resident_sequence_count == 5u &&
			snap.resident_token_count == 55u &&
			snap.kv_token_capacity == 4096u &&
			snap.device_memcpy_bytes_per_submit == 12u &&
			snap.host_staging_bytes_per_submit == 34u,
			"snapshot copies residency + traffic counters");
		common.quiescing = 1u;
		CHECK_STATUS("snapshot during quiesce",
			SparkAdapterSnapshot(&common, &snap), SPARK_STATUS_OK);
		expect(snap.available_submission_count == 0u,
			"quiesced snapshot reports zero available");
		CHECK_STATUS("snapshot null-out gate",
			SparkAdapterSnapshot(&common, 0),
			SPARK_STATUS_INVALID_ARGUMENT);
		CHECK_STATUS("snapshot null-state gate",
			SparkAdapterSnapshot(0, &snap),
			SPARK_STATUS_INVALID_ARGUMENT);

		table[0].active = 1u;
		expect(SparkAdapterDestroyReady(&common) == 0u,
			"destroy-ready refuses busy table");
		table[0].active = 0u;
		d.snapshot_mode = 2;
		expect(SparkAdapterDestroyReady(&common) == 0u,
			"destroy-ready refuses driver-poll error");
		d.snapshot_mode = 1;
		expect(SparkAdapterDestroyReady(&common) == 0u,
			"destroy-ready refuses active submissions");
		d.snapshot_mode = 0;
		expect(SparkAdapterDestroyReady(&common) == 1u,
			"destroy-ready accepts idle + drained driver");
		expect(d.destroy_calls == 0,
			"destroy-ready never destroys by itself");

		SparkAdapterTeardownDriver(&common);
		expect(d.destroy_calls == 1, "teardown destroys instance once");
		expect(common.driver.interface == 0,
			"teardown resets loaded-driver handle");
		SparkAdapterTeardownDriver(&common);
		expect(d.destroy_calls == 1, "teardown after reset is a no-op");
		{
			SparkAdapterCommonState empty;
			memset(&empty, 0, sizeof(empty));
			SparkAdapterTeardownDriver(&empty);
			expect(1, "teardown on never-loaded state is safe");
		}
	}
}

/* ---- load driver (error paths) --------------------------------------------- */

static void test_load_driver_error_paths(void)
{
	FakeDriverState d;
	fake_driver_init(&d);
	{
		SparkAdapterCommonState common;
		SparkAdapterDriverContract contract;
		SparkModelServingAdapterConfiguration c;
		SparkStatus st;
		memset(&common, 0, sizeof(common));
		memset(&contract, 0, sizeof(contract));
		memset(&c, 0, sizeof(c));
		common.pipeline_slot_count = 4u;
		common.max_active_sequence_count = 2u;
		common.max_input_row_count = 2u;
		common.resident_sequence_capacity = 2u;
		contract.model_id = "no-such-model";
		contract.model_revision = "r";
		contract.stage_name = "s";
		contract.description_sha256 = "d";
		contract.check_kind = 1u;
		c.driver_shared_object_path =
			"/nonexistent/spark-ac-missing-driver.so";
		c.node_target = "cuda.sm121";
		st = SparkAdapterLoadDriver(&common, &c, &contract);
		expect(st != SPARK_STATUS_OK, "missing .so fails loudly");
		expect(common.driver.interface == 0 &&
			common.driver_instance == 0,
			"failed load leaves handle reset for retry");
		expect(d.create_calls == 0, "no create without a successful load");
	}
}

/* ---- admit frame ------------------------------------------------------------ */

static int g_submits = 0;
static SparkStatus fake_submit_counter(void *driver_instance,
	SparkModelDriverFrame *frame)
{
	(void)driver_instance;
	(void)frame;
	g_submits++;
	return SPARK_STATUS_OK;
}

static void test_admit_frame(void)
{
	FakeDriverState d;
	fake_driver_init(&d);
	{
		SparkModelDriverFrame frame;
		SparkModelServingSubmission sub = make_submission();
		memset(&frame, 0, sizeof(frame));
		frame.program_id = 7u;

		CHECK_STATUS("admit-frame null-driver gate",
			SparkAdapterAdmitFrame(0, d.instance, &d.program, 0,
				&frame, 0u), SPARK_STATUS_INVALID_ARGUMENT);
		CHECK_STATUS("admit-frame null-program gate",
			SparkAdapterAdmitFrame(&d.handle, d.instance, 0, 0,
				&frame, 0u), SPARK_STATUS_INVALID_ARGUMENT);
		CHECK_STATUS("admit-frame null-frame gate",
			SparkAdapterAdmitFrame(&d.handle, d.instance, &d.program,
				0, 0, 0u), SPARK_STATUS_INVALID_ARGUMENT);

		CHECK_STATUS("admit-frame from-frame ok",
			SparkAdapterAdmitFrame(&d.handle, d.instance, &d.program,
				0, &frame, 0u), SPARK_STATUS_OK);
		expect(d.admit_calls == 1 && d.captured_program_id == 7u,
			"from-frame admission evaluated with program id");
		expect(d.admit_seen_instance == &d, "admit sees driver instance");

		attach_rows(&sub);
		CHECK_STATUS("admit-frame from-submission ok",
			SparkAdapterAdmitFrame(&d.handle, d.instance, &d.program,
				&sub, &frame, 0u), SPARK_STATUS_OK);
		expect(d.admit_calls == 2, "second admission evaluated");

		g_submits = 0;
		d.program.submit = fake_submit_counter;
		CHECK_STATUS("admit-frame submit-on-apply",
			SparkAdapterAdmitFrame(&d.handle, d.instance, &d.program,
				0, &frame, 1u), SPARK_STATUS_OK);
		expect(g_submits == 1, "submit_on_apply submitted exactly once");

		g_submits = 0;
		d.admit_status = SPARK_STATUS_BUSY;
		CHECK_STATUS("admission refusal propagates",
			SparkAdapterAdmitFrame(&d.handle, d.instance, &d.program,
				0, &frame, 1u), SPARK_STATUS_BUSY);
		expect(g_submits == 0, "refused frame is never submitted");
	}
}

/* ---- tp-collective stanza ----------------------------------------------------- */

static void write_parse(const char *text, SparkJsonDocument *doc, int *ok)
{
	memset(doc, 0, sizeof(*doc));
	*ok = SparkJsonParseText(text, strlen(text), doc) == SPARK_STATUS_OK;
}

static void test_load_tp_collective(void)
{
	const char *runtime_root = "/tmp";
	SparkJsonDocument doc;
	SparkAdapterTpCollectivePolicy policy;
	SparkAdapterTpCollectiveParsed parsed;
	char text[2048];
	int ok;

	memset(&policy, 0, sizeof(policy));
	CHECK_STATUS("tp arg-gate policy-null",
		SparkAdapterLoadTpCollective(0, -1, runtime_root, 2u, 0u,
			&policy, &parsed), SPARK_STATUS_INVALID_ARGUMENT);

	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"nccl\","
		"\"backend_module_path\":\"ac_backend.so\","
		"\"collective_identifier\":12345,\"listen_port\":62620,"
		"\"connect_timeout_milli\":5000,"
		"\"operation_timeout_milli\":30000,"
		"\"peer_hosts\":[\"h0\",\"h1\"],"
		"\"peer_ports\":[62620,62621]}}");
	write_parse(text, &doc, &ok);
	expect(ok, "tp nccl stanza parses");
	CHECK_STATUS("tp nccl happy", SparkAdapterLoadTpCollective(&doc,
		SparkJsonGetRootToken(&doc), runtime_root, 2u, 0u, &policy,
		&parsed), SPARK_STATUS_OK);
	expect(parsed.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL &&
		parsed.collective_identifier == 12345u &&
		parsed.listen_port == 62620u &&
		parsed.topology.rank_count == 2u &&
		strcmp(parsed.topology.rank_hosts[0], "h0") == 0 &&
		strcmp(parsed.topology.rank_hosts[1], "h1") == 0 &&
		parsed.peer_ports[0] == 62620u &&
		parsed.peer_ports[1] == 62621u,
		"tp nccl parsed fields land in output struct");
	SparkJsonDocumentDestroy(&doc);

	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"nccl\","
		"\"backend_module_path\":\"a.so\","
		"\"collective_identifier\":1,\"listen_port\":100,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"h0\",\"h2\"],"
		"\"peer_ports\":[100,150]}}");
	write_parse(text, &doc, &ok);
	expect(ok, "tp span stanza parses");
	CHECK_STATUS("tp span violation rejected under glm52 law",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 2u, 1u, &policy, &parsed),
		SPARK_STATUS_SCHEMA_ERROR);
	SparkJsonDocumentDestroy(&doc);

	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"nccl\","
		"\"backend_module_path\":\"a.so\","
		"\"collective_identifier\":0,\"listen_port\":100,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"h0\"],\"peer_ports\":[100]}}");
	write_parse(text, &doc, &ok);
	CHECK_STATUS("tp zero identifier rejected",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 1u, 0u, &policy, &parsed),
		SPARK_STATUS_SCHEMA_ERROR);
	SparkJsonDocumentDestroy(&doc);

	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"nccl\","
		"\"backend_module_path\":\"a.so\","
		"\"collective_identifier\":9,\"listen_port\":100,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"h0\",\"h1\",\"h2\"],"
		"\"peer_ports\":[100,101,102]}}");
	write_parse(text, &doc, &ok);
	CHECK_STATUS("tp rank-count mismatch rejected",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 2u, 0u, &policy, &parsed),
		SPARK_STATUS_SCHEMA_ERROR);
	SparkJsonDocumentDestroy(&doc);

	policy.adaptive_algorithm_mask =
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
	policy.adaptive_algorithm_count = 1u;
	policy.require_zero_thresholds = 1u;
	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"hidden_transport\","
		"\"backend_module_path\":\"adaptive.so\","
		"\"collective_identifier\":77,\"listen_port\":200,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"g0\",\"g1\"],"
		"\"peer_ports\":[200,201],"
		"\"algorithms\":[\"recursive_doubling\"],"
		"\"direct_all_to_all_max_payload_bytes\":0,"
		"\"split_ring_min_payload_bytes\":0,"
		"\"rail_peer_hosts\":[[\"g0\",\"g1\"],[\"g1\",\"g0\"]],"
		"\"step_rail_indices\":[0,1,0]}}");
	write_parse(text, &doc, &ok);
	expect(ok, "tp adaptive stanza parses");
	CHECK_STATUS("tp adaptive happy (glm52 knobs)",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 2u, 0u, &policy, &parsed), SPARK_STATUS_OK);
	expect(parsed.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT &&
		parsed.topology.algorithm_mask ==
			SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING &&
		parsed.topology.rail_count ==
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT &&
		parsed.topology.step_rail_indices[0] == 0u &&
		parsed.topology.step_rail_indices[2] == 0u,
		"tp adaptive parsed topology fields");
	SparkJsonDocumentDestroy(&doc);

	policy.adaptive_algorithm_mask =
		SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
	policy.adaptive_algorithm_count = 3u;
	policy.require_zero_thresholds = 0u;
	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"hidden_transport\","
		"\"backend_module_path\":\"adaptive.so\","
		"\"collective_identifier\":78,\"listen_port\":200,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"g0\",\"g1\"],"
		"\"peer_ports\":[200,201],"
		"\"algorithms\":[\"recursive_doubling\","
		"\"counter_rotating_split_ring\",\"direct_all_to_all\"],"
		"\"direct_all_to_all_max_payload_bytes\":100,"
		"\"split_ring_min_payload_bytes\":200,"
		"\"rail_peer_hosts\":[[\"g0\",\"g1\"],[\"g1\",\"g0\"]],"
		"\"step_rail_indices\":[1,0,1]}}");
	write_parse(text, &doc, &ok);
	CHECK_STATUS("tp dsv4 knobs happy (all-three + ordered thresholds)",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 2u, 0u, &policy, &parsed), SPARK_STATUS_OK);
	expect(parsed.topology.direct_all_to_all_max_payload_bytes == 100u &&
		parsed.topology.split_ring_min_payload_bytes == 200u,
		"tp dsv4 threshold values parsed");
	SparkJsonDocumentDestroy(&doc);

	snprintf(text, sizeof(text),
		"{\"tp_collective\":{\"backend\":\"carrier_pigeon\","
		"\"backend_module_path\":\"a.so\","
		"\"collective_identifier\":9,\"listen_port\":100,"
		"\"connect_timeout_milli\":5,\"operation_timeout_milli\":5,"
		"\"peer_hosts\":[\"h0\"],\"peer_ports\":[100]}}");
	write_parse(text, &doc, &ok);
	CHECK_STATUS("tp unknown backend rejected",
		SparkAdapterLoadTpCollective(&doc, SparkJsonGetRootToken(&doc),
			runtime_root, 1u, 0u, &policy, &parsed),
		SPARK_STATUS_SCHEMA_ERROR);
	SparkJsonDocumentDestroy(&doc);
}

int main(void)
{
	printf("== adapter_common: pending table ==\n");
	test_pending_table();
	test_padded_stride();
	printf("== adapter_common: row captures ==\n");
	test_row_captures();
	printf("== adapter_common: completion routing ==\n");
	test_completion_routing();
	printf("== adapter_common: env readers ==\n");
	test_env_readers();
	printf("== adapter_common: spinlock + staging ==\n");
	test_spinlocks_and_staging();
	printf("== adapter_common: validate configuration ==\n");
	test_validate_configuration();
	printf("== adapter_common: prologue + open gate ==\n");
	test_prologue_and_open_gate();
	printf("== adapter_common: quiesce/snapshot/destroy/teardown ==\n");
	test_quiesce_snapshot_destroy();
	printf("== adapter_common: load-driver error paths ==\n");
	test_load_driver_error_paths();
	printf("== adapter_common: admit frame ==\n");
	test_admit_frame();
	printf("== adapter_common: tp-collective stanzas ==\n");
	test_load_tp_collective();
	if (failures)
	{
		printf("\nFAIL (%d)\n", failures);
		return 1;
	}
	printf("\nadapter_common: every helper covered, gates green\n");
	return 0;
}
