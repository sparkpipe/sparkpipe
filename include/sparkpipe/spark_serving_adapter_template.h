#ifndef SPARKPIPE_SPARK_SERVING_ADAPTER_TEMPLATE_H
#define SPARKPIPE_SPARK_SERVING_ADAPTER_TEMPLATE_H

#include <stdint.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_memory_buffer.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared serving-adapter template. See the module-lifecycle header for the
 * paste doctrine this follows: every resident-stage family publishes a
 * SparkModelServingAdapterInterface and every one of them wrapped the same
 * plumbing - the descriptor's identity block, the capability chain, the
 * submission-slot reservation, the driver load/contract check, and the
 * tp_collective configuration parse. The paste was where the
 * admission-default-reject bug class was born twice; this header is the
 * single implementation. A family keeps only what is genuinely its own:
 * the geometry constants, the capability extras, the frame walk, and the
 * policy decisions this template exposes as explicit parameters.
 */

/*
 * Capability chain. Every serving adapter here declares prefill, decode,
 * and driver-owned-KV; the extras are the family's honest differences
 * (speculation, JIT-KV, fanout, lease, topology build knobs). The chain
 * composes them in one place so a capability can never silently go
 * missing from one family's paste.
 */
#define SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN_BASE \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV)

#define SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(family_extras) \
	(SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN_BASE | (family_extras))

/*
 * Descriptor identity block: the fields every adapter binds the same way
 * from its constants (the ABI version pair, then the five identity
 * strings). family values are the family's own macros.
 */
#define SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(adapter_id_value, \
	model_id_value, model_revision_value, program_name_value, \
	artifact_sha256_value) \
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION, \
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES, \
	.adapter_id = (adapter_id_value), \
	.model_id = (model_id_value), \
	.model_revision = (model_revision_value), \
	.driver_program_name = (program_name_value), \
	.artifact_sha256 = (artifact_sha256_value)

/* JSON member access, shared: the adapters pasted these two wrappers so a
 * schema read is one call. Missing member is a SCHEMA_ERROR, as pasted. */
int32_t SparkServingAdapterTemplateJsonMember(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name);

SparkStatus SparkServingAdapterTemplateJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint32_t *value);

/*
 * The one tp_collective configuration parser. The JSON contract (member
 * names, types, array shapes, range checks) is shared; the families'
 * genuinely different rules are explicit policy:
 *   - the peer count the peer_hosts/peer_ports arrays must match,
 *   - whether a zero collective_identifier is the degraded single-rank
 *     mode or a schema error,
 *   - whether peer_ports must be contiguous from peer_ports[0] (the
 *     control-port-base derivation; off = ports are free),
 *   - the adaptive-fabric algorithm set and the payload-threshold rule
 *     (rail-fabric builds require the full known set with ordered
 *     non-zero thresholds; single-algorithm builds require recursive
 *     doubling alone with both thresholds zero).
 */
typedef enum SparkTpCollectiveAlgorithmPolicy
{
	SPARK_TP_COLLECTIVE_ALGORITHMS_FULL_KNOWN_SET = 1,
	SPARK_TP_COLLECTIVE_ALGORITHMS_RECURSIVE_DOUBLING_ONLY = 2
} SparkTpCollectiveAlgorithmPolicy;

typedef enum SparkTpCollectiveThresholdPolicy
{
	SPARK_TP_COLLECTIVE_THRESHOLDS_ORDERED_NONZERO = 1,
	SPARK_TP_COLLECTIVE_THRESHOLDS_ZERO_REQUIRED = 2
} SparkTpCollectiveThresholdPolicy;

typedef struct SparkTpCollectiveConfigPolicy
{
	uint32_t peer_count;
	uint32_t allow_zero_collective_identifier;
	uint32_t require_contiguous_peer_ports;
	SparkTpCollectiveAlgorithmPolicy algorithms;
	SparkTpCollectiveThresholdPolicy thresholds;
} SparkTpCollectiveConfigPolicy;

/*
 * The parsed tp_collective block. Families copy what they keep in their
 * state layout (the template never embeds into the family state); set
 * backend_module_path_buffer/bytes to the family's destination buffer and
 * the runtime-relative backend path resolves directly into it, bounds
 * identical to the pasted parser.
 */
typedef struct SparkTpCollectiveAdapterConfig
{
	uint32_t backend_kind;
	uint64_t collective_identifier;
	uint16_t listen_port;
	uint16_t peer_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	uint32_t peer_count;
	uint32_t connect_timeout_milli;
	uint32_t operation_timeout_milli;
	uint32_t control_port_base;
	char *backend_module_path_buffer;
	uint32_t backend_module_path_bytes;
	SparkTpDeviceCollectiveTopology topology;
} SparkTpCollectiveAdapterConfig;

SparkStatus SparkServingAdapterTemplateLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config);

/*
 * Pending-submission reservation: the common spine of every adapter's
 * ReservePending. The family pending struct embeds
 * SparkServingAdapterPendingCommon (after any owner pointer the family
 * fills itself) and passes that embedding site as common_offset — the
 * template cannot know the family layout, so the offset is family data
 * exactly like last_row_by_lane_offset; the template finds the free
 * slot via the common's active flag at common_offset, zeroes the WHOLE
 * element (element_bytes covers the family's own fields), fills the
 * common submission view, walks last_row_by_lane at the family layout's
 * offsetof, and returns the element. It deliberately does NOT set the
 * active flag: the pasted reserves mark the slot live only after the
 * family's own fill steps succeed, and a failed fill must leave the slot
 * free. Returns 0 when every slot is active, exactly as pasted.
 */
typedef struct SparkServingAdapterPendingCommon
{
	uint32_t active;
	uint32_t row_count;
	uint32_t lane_count;
	uint32_t active_sequence_count;
	uint32_t work_kind;
	uint32_t tokens_per_sequence;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
} SparkServingAdapterPendingCommon;

void *SparkServingAdapterTemplateReservePending(
	void *pending_array,
	uint32_t element_bytes,
	uint32_t common_offset,
	uint32_t pipeline_slot_count,
	uint32_t last_row_by_lane_offset,
	const SparkModelServingSubmission *submission);

/*
 * Driver load + contract check. The shared spine: reset, load, the driver
 * descriptor identity compare, the program lookup, the admit/submit
 * presence check, the create request assembly, and the null-instance
 * guard. Family policy is explicit: the target pin (families that do not
 * pin the driver target leave driver_target 0), and the program
 * acceptance callback (the flag/profile contract differs per family and
 * per family build - the callback keeps those checks family-owned and
 * byte-equivalent, a shared approximation would change accept/reject
 * behavior on real descriptors).
 */
typedef struct SparkServingAdapterDriverContract
{
	const char *driver_model_id;
	const char *driver_model_revision;
	const char *driver_stage_name;
	const char *driver_target; /* 0 = the target is not pinned */
	const char *model_description_sha256;
} SparkServingAdapterDriverContract;

typedef SparkStatus (*SparkServingAdapterProgramAcceptFunction)(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context);

typedef struct SparkServingAdapterDriverRequest
{
	SparkServingAdapterDriverContract contract;
	void *node_context; /* 0 = omit on the create request */
	void *completion_context;
	SparkModelDriverCompletionFunction completion_function;
	SparkModelDriverWakeFunction wake_function;
} SparkServingAdapterDriverRequest;

SparkStatus SparkServingAdapterTemplateLoadDriver(
	const SparkServingAdapterDriverRequest *request,
	const SparkModelServingAdapterConfiguration *configuration,
	SparkLoadedModelDriver *driver,
	const SparkModelDriverProgramDescriptor **program_out,
	SparkServingAdapterProgramAcceptFunction program_accepts,
	void *accept_context,
	void **driver_instance_out);

#ifdef __cplusplus
}
#endif

#endif
