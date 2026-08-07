#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_model_resident_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION 7u
#define SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX UINT32_MAX
#define SPARK_MODEL_PIPELINE_STAGE_COMPLETION_FLAG_CLIENT_ELAPSED_VALID \
	UINT32_C(0x00000001)
#define SPARK_MODEL_PIPELINE_STAGE_COMPLETION_KNOWN_FLAGS \
	SPARK_MODEL_PIPELINE_STAGE_COMPLETION_FLAG_CLIENT_ELAPSED_VALID

typedef struct SparkModelPipelineClient SparkModelPipelineClient;

typedef struct SparkModelPipelineStageCompletion
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t stage_index;
	uint32_t work_kind;
	uint32_t active_sequence_count;
	uint32_t row_count;
	uint32_t status;
	uint32_t flags;
	uint64_t submission_id;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint64_t device_memcpy_bytes;
	uint64_t host_staging_bytes;
	uint64_t client_elapsed_ns;
} SparkModelPipelineStageCompletion;

typedef void (*SparkModelPipelineStageCompletionFunction)(
	void *completion_context,
	const SparkModelPipelineStageCompletion *completion);

/*
 * One event-loop thread owns submit/progress/destroy. Submit-result and final
 * completion callbacks run after the completed transaction is released and may
 * submit more work. They must not recursively call progress or destroy.
 *
 * Stage-completion callbacks are observation-only: they run while the
 * transaction is active, must copy any descriptor they retain, and must not
 * call a pipeline-client function.
 */

typedef struct SparkModelPipelineClientConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t connect_timeout_ms;
	uint32_t reserved0;
	const SparkModelResidentDeployment *deployment;
	const char *runtime_root;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelPipelineStageCompletionFunction stage_completion_function;
	void *stage_completion_context;
} SparkModelPipelineClientConfiguration;

typedef struct SparkModelPipelineClientView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_count;
	uint32_t connected_rank_count;
	uint32_t active_transaction_count;
	uint32_t transaction_capacity;
	uint32_t failed_status;
	/* First stage where the client observed a fatal error, not causal proof. */
	uint32_t failed_stage_index;
	uint64_t submitted_count;
	uint64_t admitted_count;
	uint64_t rejected_count;
	uint64_t completed_count;
} SparkModelPipelineClientView;

#define SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkModelPipelineClientConfiguration))
#define SPARK_MODEL_PIPELINE_STAGE_COMPLETION_BYTES \
	((uint32_t)sizeof(SparkModelPipelineStageCompletion))
#define SPARK_MODEL_PIPELINE_CLIENT_VIEW_BYTES \
	((uint32_t)sizeof(SparkModelPipelineClientView))

SparkStatus SparkModelPipelineClientConnect(
	const SparkModelPipelineClientConfiguration *configuration,
	SparkModelPipelineClient **pipeline_out);
void SparkModelPipelineClientDestroy(SparkModelPipelineClient *pipeline);
SparkStatus SparkModelPipelineClientSubmit(
	SparkModelPipelineClient *pipeline,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelPipelineClientProgress(
	SparkModelPipelineClient *pipeline,
	uint32_t maximum_message_count_per_rank);
SparkStatus SparkModelPipelineClientGetPollDescriptors(
	const SparkModelPipelineClient *pipeline,
	SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out);
SparkStatus SparkModelPipelineClientGetView(
	const SparkModelPipelineClient *pipeline,
	SparkModelPipelineClientView *view);
const SparkModelServingAdapterDescriptor *SparkModelPipelineClientGetAdapterDescriptor(
	const SparkModelPipelineClient *pipeline);

#ifdef __cplusplus
}
#endif
