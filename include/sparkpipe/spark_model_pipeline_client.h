#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_model_resident_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION 6u

typedef struct SparkModelPipelineClient SparkModelPipelineClient;

/*
 * One event-loop thread owns submit/progress/destroy. Callbacks observe queue
 * state after the completed transaction is released and may submit more work.
 * They must not recursively call progress or destroy the pipeline client.
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
	uint32_t reserved0;
	uint64_t submitted_count;
	uint64_t admitted_count;
	uint64_t rejected_count;
	uint64_t completed_count;
} SparkModelPipelineClientView;

#define SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkModelPipelineClientConfiguration))
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
