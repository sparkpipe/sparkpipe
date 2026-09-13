 

































#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "spark_filesystem.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ADAPTER_COMMON_ABI_VERSION 1u

 




typedef struct SparkAdapterSubmissionIdentity
{
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
} SparkAdapterSubmissionIdentity;

 




/*
 * Head of every family's per-slot record. A family's pending array is an
 * array of its own struct whose FIRST member is this core, so table walks
 * here and payload accesses there share one allocation.
 *
 * ACTIVE-FLAG ORDERING (PENDING-SLOT EXCLUSIVITY, T1+T2): `active` is the
 * hand-off word between a completion path releasing a record and the next
 * submission-side claim of the same slot. The completing side clears it
 * with RELEASE ordering - strictly after its last read of the payload and
 * identity halves - and the claiming side treats observe-of-zero as
 * ACQUIRE, writing payload/identity and re-stamping active=1 only after
 * that observation. That pair is what makes a recycled record safe
 * without a lock under the async delivery school: the new claimant's
 * writes can neither become visible to the previous occupant's completion
 * path nor race it. Availability scans and snapshots may read `active`
 * stale; they count slots and never touch payloads. The flag is a plain
 * uint32_t because T1 keeps each side single-threaded; an adapter that
 * widens concurrency beyond T1+T2 must first widen the flag to _Atomic
 * with release/acquire on clear/claim.
 */
typedef struct SparkAdapterPendingCore
{
	uint32_t active;
	uint32_t row_count;
	uint32_t lane_count;
	uint32_t active_sequence_count;
	uint32_t work_kind;
	SparkAdapterSubmissionIdentity identity;
} SparkAdapterPendingCore;

 
typedef struct SparkAdapterCompletionSink
{
	SparkModelServingCompletionFunction function;
	void *context;
} SparkAdapterCompletionSink;

 




typedef struct SparkAdapterCommonState
{
	/* Orphan accounting first: the shared driver-completion callback
	 * receives this struct and bumps exactly this counter. ATOMIC
	 * (F-A2): under CAPABILITY_ASYNC_COMPLETION the bump runs on driver
	 * completion threads concurrent with snapshot reads, so writers use
	 * fetch_add and readers load; relaxed ordering suffices because it
	 * is a pure statistic - no other state is published through it. */
	_Atomic uint64_t orphan_completion_count;
	SparkAdapterCompletionSink sink;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	 


	SparkAdapterPendingCore *pending;
	size_t pending_element_bytes;
	uint32_t pipeline_slot_count;
	uint32_t quiescing;
	 
	SparkModelServingRuntimeLimits runtime_limits;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	void *execution_stream;
} SparkAdapterCommonState;

 














typedef struct SparkAdapterDriverContract
{
	const char *model_id;
	const char *model_revision;
	const char *stage_name;
	 




	const char *target;
	const char *description_sha256;
	uint32_t required_program_flags;
	uint32_t check_kind;
	void *node_context;
} SparkAdapterDriverContract;

 
typedef struct SparkAdapterEnvironmentSetting
{
	const char *name;
	const char *text;
} SparkAdapterEnvironmentSetting;

 

 






void SparkAdapterPendingCapture(
	SparkAdapterPendingCore *core,
	const SparkModelServingSubmission *submission);

 









/* The scan is the ACQUIRE half of the ACTIVE-FLAG ORDERING law on
 * SparkAdapterPendingCore: a zero observed here releases the slot to the
 * caller, which owns payload/identity writes up to its active=1 stamp. */
int32_t SparkAdapterPendingClaim(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count);

 
uint32_t SparkAdapterAvailableSubmissionCount(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count);

 

void SparkAdapterCaptureLastRowByLane(
	const SparkModelServingSubmission *submission,
	uint32_t *last_row_by_lane);

 
void SparkAdapterCaptureResidentSlotsPerLane(
	const SparkModelServingSubmission *submission,
	uint32_t lane_count,
	uint32_t *resident_slots);

 

void SparkAdapterCaptureResidentSlotsPerRow(
	const SparkModelServingSubmission *submission,
	uint32_t *slots_by_row);

 

 




void SparkAdapterOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion);

 




void SparkAdapterDispatchWake(void *wake_context);

 





uint32_t SparkAdapterDriverCompletionMatches(
	const SparkModelDriverCompletion *driver_completion,
	uint64_t request_id,
	uint64_t sequence_id,
	uint64_t sequence_position,
	uint32_t program_id);

 
void SparkAdapterAccumulateFrameCounters(
	uint64_t *accepted_token_count,
	uint64_t *queue_delay_ns,
	uint64_t *service_time_ns,
	const SparkModelDriverCompletion *driver_completion);

 





















void SparkAdapterBuildCompletionHeader(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core);

 






void SparkAdapterBuildCompletionHeaderWithResidency(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core,
	const SparkModelDriverResidencyToken *residency);

 
uint32_t SparkAdapterClampAcceptedTokenCount(uint64_t accepted_token_count);

 

 




SparkStatus SparkAdapterValidateConfiguration(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingAdapterConfiguration *configuration,
	const char *program_name,
	uint32_t stage_count);

 




SparkStatus SparkAdapterInitializePrologue(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration);

 





SparkStatus SparkAdapterLoadDriver(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration,
	const SparkAdapterDriverContract *contract);

 




SparkStatus SparkAdapterValidateSubmissionOpen(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission);

 




SparkStatus SparkAdapterQuiesce(
	SparkAdapterCommonState *common,
	uint64_t deadline_time_ns);

 
SparkStatus SparkAdapterSnapshot(
	SparkAdapterCommonState *common,
	SparkModelServingAdapterSnapshot *snapshot);

 



/*
 * Destroy guard: nonzero only when every slot is idle AND the driver
 * reports no active submissions (or has no snapshot to ask). F-A3 / T3:
 * it also requires closed admission - the quiescing latch a prior quiesce
 * round set - before reporting ready; the one escape is the
 * never-submitted instance (zero driver submissions), which has no
 * in-flight completion to drain and no open admission window.
 */
uint32_t SparkAdapterDestroyReady(SparkAdapterCommonState *common);

 
void SparkAdapterTeardownDriver(SparkAdapterCommonState *common);

 

 
const char *SparkAdapterEnvText(const char *name);

 





uint32_t SparkAdapterEnvFlagDefaultOn(const char *name);

 



uint32_t SparkAdapterEnvFlagDefaultOffExactZero(const char *name);

 





uint32_t SparkAdapterEnvFlagDefaultOffTruthy(const char *name);

 

 




SparkStatus SparkAdapterSetEnvironmentText(const char *name, const char *text);

 





void SparkAdapterSpinLockAcquire(_Atomic uint32_t *lock);
void SparkAdapterSpinLockRelease(_Atomic uint32_t *lock);

 
void SparkAdapterFormatEnvironmentUnsigned(
	char *buffer,
	size_t buffer_bytes,
	uint32_t value);

 

 








typedef struct SparkAdapterTpCollectivePolicy
{
	uint32_t adaptive_algorithm_mask;
	uint32_t adaptive_algorithm_count;
	uint32_t require_zero_thresholds;
} SparkAdapterTpCollectivePolicy;

typedef struct SparkAdapterTpCollectiveParsed
{
	uint32_t backend_kind;
	uint64_t collective_identifier;
	uint16_t listen_port;
	uint16_t control_port_base;
	uint32_t connect_timeout_milli;
	uint32_t operation_timeout_milli;
	uint16_t peer_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	char backend_module_path[SPARK_INTERNAL_PATH_BYTES];
	SparkTpDeviceCollectiveTopology topology;
} SparkAdapterTpCollectiveParsed;

 










SparkStatus SparkAdapterLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	uint32_t expected_rank_count,
	uint32_t validate_port_span,
	const SparkAdapterTpCollectivePolicy *policy,
	SparkAdapterTpCollectiveParsed *parsed);

 

 





SparkStatus SparkAdapterAdmitFrame(
	const SparkLoadedModelDriver *driver,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame,
	uint32_t submit_on_apply);
#ifdef __cplusplus
}
#endif
