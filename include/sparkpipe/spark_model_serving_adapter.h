#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weight_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION 12u
#define SPARK_MODEL_SERVING_ADAPTER_INTERFACE_SYMBOL \
	"SparkModelServingAdapterGetInterface"
#define SPARK_MODEL_SERVING_ADAPTER_ARTIFACT_SHA256_LENGTH 64u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES 512u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT 16u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_LAYER_COUNT 256u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT 1024u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT 65536u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT \
	SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT
#define SPARK_MODEL_SERVING_ADAPTER_MAX_INFLIGHT_SUBMISSION_COUNT 64u
#define SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT 4096u
#define SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT UINT32_MAX

/*
 * Mark the final prefill row for a lane as token-producing. Intermediate
 * prefill chunks leave this clear; every active decode lane sets it. The
 * completion remains lane-indexed; consumers ignore the zero placeholder for
 * unmarked prefill lanes.
 */
#define SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN UINT32_C(0x00000001)
#define SPARK_MODEL_SERVING_LANE_KNOWN_FLAGS \
	SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN

#define SPARK_MODEL_SERVING_SLOT_REUSE_NONE 0u
#define SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE 1u
#define SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO 2u

#define SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16 1u

#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL UINT32_C(0x00000001)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE UINT32_C(0x00000002)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE UINT32_C(0x00000004)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION \
	UINT32_C(0x00000008)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT \
	UINT32_C(0x00000010)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH UINT32_C(0x00000020)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESET UINT32_C(0x00000040)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV \
	UINT32_C(0x00000080)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV UINT32_C(0x00000100)
#define SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION UINT32_C(0x00000200)
#define SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESET | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION)

#define SPARK_MODEL_SERVING_WORK_KIND_PREFILL 1u
#define SPARK_MODEL_SERVING_WORK_KIND_DECODE 2u
#define SPARK_MODEL_SERVING_WORK_KIND_RELEASE 3u

#define SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS UINT32_C(0x00000001)
#define SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION \
	UINT32_C(0x00000002)
#define SPARK_MODEL_SERVING_COMPLETION_KNOWN_FLAGS \
	(SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS | \
	 SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION)

typedef struct SparkModelServingAdapterDescriptor
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t stage_count;
	uint32_t layer_count;
	uint32_t boundary_format;
	uint32_t boundary_element_count;
	uint32_t boundary_element_bytes;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t kv_cache_codec;
	uint32_t max_inflight_submission_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t max_resident_sequence_count;
	uint32_t max_output_token_count;
	uint32_t max_speculative_token_count;
	uint32_t resident_sequence_slot_reuse;
	const char *adapter_id;
	const char *model_id;
	const char *model_revision;
	const char *driver_program_name;
	const char *artifact_sha256;
	uint32_t stage_layer_counts[SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT];
	uint32_t boundary_sideband_kinds[SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT];
	uint32_t boundary_sideband_bytes_per_sequence[SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT];
	/* Target-owned scheduler floor; zero means every nonempty submission is efficient. */
	uint32_t minimum_efficient_submission_row_count;
} SparkModelServingAdapterDescriptor;

typedef struct SparkModelServingRuntimeLimits
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t max_inflight_submission_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t reserved[4];
} SparkModelServingRuntimeLimits;

typedef struct SparkModelServingLane
{
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t step_generation;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint32_t resident_sequence_slot;
	uint32_t context_token_count;
	uint32_t input_token_id;
	uint32_t flags;
} SparkModelServingLane;

typedef struct SparkModelServingSubmission
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t work_kind;
	uint32_t flags;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t deadline_time_ns;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	uint32_t priority;
	uint32_t active_sequence_count;
	uint32_t new_token_count;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t token_count;
	uint32_t model_extension_kind;
	uint32_t model_extension_bytes;
	const SparkModelServingLane *lanes;
	const uint32_t *token_ids;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
	const void *model_extension;
	const void *hidden_input_address;
	uint64_t hidden_input_bytes;
	const void *boundary_sideband_input_address;
	uint64_t boundary_sideband_input_bytes;
	void *hidden_output_address;
	uint64_t hidden_output_bytes;
	void *boundary_sideband_output_address;
	uint64_t boundary_sideband_output_bytes;
	SparkModelDriverResidencyToken residency;
} SparkModelServingSubmission;

typedef struct SparkModelServingCompletion
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t status;
	uint32_t completion_flags;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	SparkModelDriverResidencyToken residency;
	uint32_t accepted_token_count;
	uint32_t token_count;
	uint32_t token_ids[SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT];
	uint32_t model_extension_kind;
	uint32_t model_extension_bytes;
	uint8_t model_extension[SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES];
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint64_t device_memcpy_bytes;
	uint64_t host_staging_bytes;
} SparkModelServingCompletion;

typedef void (*SparkModelServingCompletionFunction)(
	void *completion_context,
	const SparkModelServingCompletion *completion);
typedef void (*SparkModelServingWakeFunction)(void *wake_context);
typedef struct SparkModelServingAdapterConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	SparkModelServingRuntimeLimits runtime_limits;
	const char *runtime_root;
	const char *node_id;
	const char *node_target;
	const char *adapter_configuration_path;
	const char *driver_shared_object_path;
	const char *driver_program_name;
	void *execution_stream;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
} SparkModelServingAdapterConfiguration;

typedef struct SparkModelServingAdapterSnapshot
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t active_submission_count;
	uint32_t available_submission_count;
	uint64_t submitted_count;
	uint64_t completed_count;
	uint64_t rejected_count;
	uint64_t resident_sequence_count;
	uint64_t resident_token_count;
	uint64_t kv_token_capacity;
	uint64_t device_memcpy_bytes_per_submit;
	uint64_t host_staging_bytes_per_submit;
} SparkModelServingAdapterSnapshot;

typedef SparkStatus (*SparkModelServingAdapterInitializeFunction)(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state);
typedef void (*SparkModelServingAdapterDestroyFunction)(void *adapter_state);
typedef SparkStatus (*SparkModelServingAdapterValidateSubmissionFunction)(
	void *adapter_state,
	const SparkModelServingSubmission *submission);
/*
 * validate_submission is the side-effect-free PREPARE hook. Resident-owned
 * hidden boundary pointers are intentionally absent until every rank commits.
 * submit begins execution and may complete synchronously through the callback.
 */
typedef SparkStatus (*SparkModelServingAdapterSubmitFunction)(
	void *adapter_state,
	const SparkModelServingSubmission *submission);
typedef SparkStatus (*SparkModelServingAdapterPrefetchFunction)(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count);
typedef SparkStatus (*SparkModelServingAdapterProgressFunction)(
	void *adapter_state,
	uint32_t maximum_step_count);
/*
 * Quiesce permanently closes admission for this adapter instance. It returns
 * OK only when no completion callback or model work can still reference
 * adapter-owned state. BUSY/PENDING asks the resident to progress and poll
 * again before the absolute CLOCK_MONOTONIC deadline.
 */
typedef SparkStatus (*SparkModelServingAdapterQuiesceFunction)(
	void *adapter_state,
	uint64_t deadline_time_ns);
typedef SparkStatus (*SparkModelServingAdapterSnapshotFunction)(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot);
typedef SparkStatus (*SparkModelServingAdapterResetFunction)(
	void *adapter_state,
	uint64_t control_generation);

typedef struct SparkModelServingAdapterInterface
{
	uint32_t abi_version;
	uint32_t interface_bytes;
	const SparkModelServingAdapterDescriptor *descriptor;
	SparkModelServingAdapterInitializeFunction initialize;
	SparkModelServingAdapterDestroyFunction destroy;
	SparkModelServingAdapterValidateSubmissionFunction validate_submission;
	SparkModelServingAdapterSubmitFunction submit;
	SparkModelServingAdapterPrefetchFunction prefetch;
	SparkModelServingAdapterProgressFunction progress;
	SparkModelServingAdapterQuiesceFunction quiesce;
	SparkModelServingAdapterSnapshotFunction snapshot;
	SparkModelServingAdapterResetFunction reset;
} SparkModelServingAdapterInterface;

typedef const SparkModelServingAdapterInterface *
	(*SparkModelServingAdapterGetInterfaceFunction)(void);

typedef struct SparkModelServingAdapterDynamicLibrary
{
	void *dynamic_library;
	SparkModelServingAdapterInterface adapter_interface;
} SparkModelServingAdapterDynamicLibrary;

#define SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkModelServingAdapterDescriptor))
#define SPARK_MODEL_SERVING_LANE_BYTES \
	((uint32_t)sizeof(SparkModelServingLane))
#define SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES \
	((uint32_t)sizeof(SparkModelServingRuntimeLimits))
#define SPARK_MODEL_SERVING_SUBMISSION_BYTES \
	((uint32_t)sizeof(SparkModelServingSubmission))
#define SPARK_MODEL_SERVING_COMPLETION_BYTES \
	((uint32_t)sizeof(SparkModelServingCompletion))
#define SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkModelServingAdapterConfiguration))
#define SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES \
	((uint32_t)sizeof(SparkModelServingAdapterSnapshot))
#define SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkModelServingAdapterInterface))

SparkStatus SparkModelServingAdapterValidateDescriptor(
	const SparkModelServingAdapterDescriptor *descriptor);
SparkStatus SparkModelServingAdapterValidateRuntimeLimits(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits);
SparkStatus SparkModelServingAdapterValidateInterface(
	const SparkModelServingAdapterInterface *adapter_interface,
	uint32_t required_capability_flags);
SparkStatus SparkModelServingAdapterValidateSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelServingAdapterValidateRuntimeSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelServingAdapterSelectEmitRows(
	const SparkModelServingSubmission *submission,
	uint32_t *emit_row_indices,
	uint32_t *emit_lane_indices,
	uint32_t emit_capacity,
	uint32_t *emit_count_out);
SparkStatus SparkModelServingAdapterValidateCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingCompletion *completion);
SparkStatus SparkModelServingAdapterValidateCompletionResidency(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion);
SparkStatus SparkModelServingAdapterValidateStageCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	uint32_t stage_index,
	uint32_t work_kind,
	uint32_t active_sequence_count,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion);
SparkStatus SparkModelServingAdapterLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkModelServingAdapterDynamicLibrary *library);
void SparkModelServingAdapterUnloadInterface(
	SparkModelServingAdapterDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
