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


#define SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN_BASE \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV)

#define SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(family_extras) \
	(SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN_BASE | (family_extras))

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

int32_t SparkServingAdapterTemplateJsonMember(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name);

SparkStatus SparkServingAdapterTemplateJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint32_t *value);

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

typedef struct SparkServingAdapterDriverContract
{
	const char *driver_model_id;
	const char *driver_model_revision;
	const char *driver_stage_name;
	const char *driver_target;
	const char *model_description_sha256;
} SparkServingAdapterDriverContract;

typedef SparkStatus (*SparkServingAdapterProgramAcceptFunction)(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context);

typedef struct SparkServingAdapterDriverRequest
{
	SparkServingAdapterDriverContract contract;
	void *node_context;
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
