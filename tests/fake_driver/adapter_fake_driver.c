/* Parameterized fake model driver for the adapter_common integration test.
 *
 * One source template, compiled once per family profile with -DPROF_*:
 * the four variants differ exactly the way real family drivers differ from
 * the shared skeleton's point of view - identity strings, program flag
 * mask, and nothing else. Exports SparkModelDriverGetInterface; instance
 * layout in adapter_fake_driver.h (shared with the harness so it can fire
 * the adapter-wired completion/wake routes and drive snapshot counters).
 *
 * Test infrastructure only. */
#include <stdlib.h>
#include <string.h>

#include "adapter_fake_driver.h"

#ifndef PROF_MODEL_ID
#error "PROF_MODEL_ID must name the fake model"
#endif
#ifndef PROF_MODEL_REVISION
#error "PROF_MODEL_REVISION must pin the fake revision"
#endif
#ifndef PROF_STAGE_NAME
#error "PROF_STAGE_NAME must name the fake stage"
#endif
#ifndef PROF_PROGRAM_NAME
#error "PROF_PROGRAM_NAME must name the fake program"
#endif
#ifndef PROF_PROGRAM_FLAGS
/* Every profile ships stream-ordered submit; the mask school in the harness
 * pins exactly this bit via its contract. */
#define PROF_PROGRAM_FLAGS SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED
#endif

/* 64 lowercase hex digits: passes every sha256-text validator. */
static const char kFakeSha256[] =
	"0123456789abcdef0123456789abcdef"
	"0123456789abcdef0123456789abcdef";

static SparkStatus FakeCreate(
	const SparkModelDriverCreateRequest *request,void **driver_instance)
{
	SparkFakeDriverInstance *instance;
	instance = calloc(1u,sizeof(*instance));
	if ( instance == 0 )
		return(SPARK_STATUS_IO_ERROR);
	instance->request = *request;
	instance->available_dispatch_slot_count = 64u;
	*driver_instance = instance;
	return(SPARK_STATUS_OK);
}

static void FakeDestroy(void *driver_instance)
{
	free(driver_instance);
}

static SparkStatus FakeAdmit(
	void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkFakeDriverInstance *instance = driver_instance;
	(void)request;
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes =
		(uint32_t)sizeof(SparkModelDriverAdmissionDecision);
	decision->accepted = 1u;
	/* An accepting driver hands out a REAL opaque slot: slot != INVALID
	 * plus a nonzero generation is exactly what SparkModelDriver-
	 * AdmissionDecisionIsValid demands once has_dispatch_slot holds. */
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = 3u;
	decision->driver_dispatch_generation =
		UINT64_C(0x5afe0000) + instance->submitted_count;
	decision->available_dispatch_slot_count =
		instance->available_dispatch_slot_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkFakeDriverInstance *instance = driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes =
		(uint32_t)sizeof(SparkModelDriverRuntimeSnapshot);
	snapshot->program_id = program_id;
	snapshot->active_submission_count = instance->active_submission_count;
	snapshot->available_dispatch_slot_count =
		instance->available_dispatch_slot_count;
	snapshot->submitted_count = instance->submitted_count;
	snapshot->completed_count = instance->completed_count;
	snapshot->rejected_count = instance->rejected_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeSubmit(void *driver_instance,SparkModelDriverFrame *frame)
{
	SparkFakeDriverInstance *instance = driver_instance;
	(void)frame;
	instance->submitted_count++;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverProgramProfile kFakeProfile =
{
	(uint32_t)sizeof(SparkModelDriverProgramProfile),
	PROF_PROGRAM_FLAGS, /* profile flags mirror the program mask (loader law) */
	64u,     /* max_inflight */
	1024u,   /* max_active_slots */
	65536u,  /* max_new_tokens */
	16384u,  /* max_resident_sequences */
	1048576ull, /* max_sequence_tokens */
	0ull, 0ull, /* target/validated latency */
	0ull, 0ull, 0ull, /* resident weights/workspaces */
	0ull, 0ull, /* memcpy/staging ceilings */
	0u, 0u
};

static const SparkModelDriverProgramDescriptor kFakePrograms[1] =
{
	{
		101u,                /* program_id */
		PROF_PROGRAM_FLAGS,
		64u,                 /* max_inflight */
		0u,
		PROF_PROGRAM_NAME,
		&kFakeProfile,
		FakeSubmit
	}
};

static const SparkModelDriverDescriptor kFakeDescriptor =
{
	SPARK_MODEL_DRIVER_ABI_VERSION,
	(uint32_t)sizeof(SparkModelDriverDescriptor),
	PROF_MODEL_ID,
	PROF_MODEL_REVISION,
	PROF_STAGE_NAME,
	"cpu.test",
	kFakeSha256,
	kFakeSha256,
	1u,   /* program_count */
	1u,   /* module_instance_count */
	kFakePrograms
};

static const SparkModelDriverInterface kFakeInterface =
{
	SPARK_MODEL_DRIVER_ABI_VERSION,
	(uint32_t)sizeof(SparkModelDriverInterface),
	&kFakeDescriptor,
	FakeCreate,
	FakeDestroy,
	FakeAdmit,
	FakeSnapshot
};

const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&kFakeInterface);
}
