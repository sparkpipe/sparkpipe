#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PIPELINE_RUNTIME_ABI_VERSION 6u
#define SPARK_PIPELINE_RUNTIME_HOST_BYTES 64u
#define SPARK_PIPELINE_RUNTIME_ROUTE_BYTES 160u
#define SPARK_PIPELINE_RUNTIME_MODULE_ID_BYTES 160u
#define SPARK_PIPELINE_RUNTIME_NO_RANK UINT32_MAX

#define SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS UINT32_C(0x00000001)
#define SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT UINT32_C(0x00000002)
#define SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE UINT32_C(0x00000004)
#define SPARK_PIPELINE_RUNTIME_RANK_FLAG_PARALLEL_FANOUT UINT32_C(0x00000008)
#define SPARK_PIPELINE_RUNTIME_RANK_KNOWN_FLAGS \
	(SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS | \
	 SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT | \
	 SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE | \
	 SPARK_PIPELINE_RUNTIME_RANK_FLAG_PARALLEL_FANOUT)

typedef struct SparkPipelineRuntimeLinearNode
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t previous_rank_index;
	uint32_t next_rank_index;
	uint32_t reserved0;
	const char *host_name;
	const char *previous_host_name;
	const char *next_host_name;
} SparkPipelineRuntimeLinearNode;

typedef struct SparkPipelineRuntimeFanoutNode
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t reserved0;
	const char *host_name;
} SparkPipelineRuntimeFanoutNode;

typedef struct SparkPipelineRuntimeRankPlan
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t flags;
	uint32_t previous_rank_index;
	uint32_t next_rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t boundary_format;
	uint32_t boundary_element_count;
	uint32_t boundary_element_bytes;
	uint32_t input_sideband_kind;
	uint32_t input_sideband_bytes_per_sequence;
	uint32_t output_sideband_kind;
	uint32_t output_sideband_bytes_per_sequence;
	uint32_t transport_capability_flags;
	uint32_t transport_control_port_base;
	uint32_t reserved0;
	uint64_t boundary_bytes_per_sequence;
	uint64_t input_packet_bytes_per_sequence;
	uint64_t output_packet_bytes_per_sequence;
	uint64_t input_max_packet_bytes;
	uint64_t output_max_packet_bytes;
	char host_name[SPARK_PIPELINE_RUNTIME_HOST_BYTES];
	char previous_host_name[SPARK_PIPELINE_RUNTIME_HOST_BYTES];
	char next_host_name[SPARK_PIPELINE_RUNTIME_HOST_BYTES];
	char input_route_name[SPARK_PIPELINE_RUNTIME_ROUTE_BYTES];
	char output_route_name[SPARK_PIPELINE_RUNTIME_ROUTE_BYTES];
	char transport_module_id[SPARK_PIPELINE_RUNTIME_MODULE_ID_BYTES];
} SparkPipelineRuntimeRankPlan;

#define SPARK_PIPELINE_RUNTIME_RANK_PLAN_BYTES \
	((uint32_t)sizeof(SparkPipelineRuntimeRankPlan))
#define SPARK_PIPELINE_RUNTIME_LINEAR_NODE_BYTES \
	((uint32_t)sizeof(SparkPipelineRuntimeLinearNode))
#define SPARK_PIPELINE_RUNTIME_FANOUT_NODE_BYTES \
	((uint32_t)sizeof(SparkPipelineRuntimeFanoutNode))

SparkStatus SparkPipelineRuntimeBuildLinearRankPlan(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeLinearNode *node,
	uint32_t max_active_sequence_count,
	uint32_t max_input_row_count,
	uint32_t transport_capability_flags,
	uint32_t transport_control_port_base,
	const char *transport_module_id,
	SparkPipelineRuntimeRankPlan *rank_plan);
SparkStatus SparkPipelineRuntimeBuildFanoutRankPlan(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeFanoutNode *node,
	uint32_t max_active_sequence_count,
	uint32_t max_input_row_count,
	uint32_t transport_control_port_base,
	SparkPipelineRuntimeRankPlan *rank_plan);
SparkStatus SparkPipelineRuntimeValidateRankPlan(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan);
SparkStatus SparkPipelineRuntimeBuildInputEndpoint(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	SparkHiddenTransportEndpoint *endpoint);
SparkStatus SparkPipelineRuntimeBuildOutputEndpoint(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	SparkHiddenTransportEndpoint *endpoint);

#ifdef __cplusplus
}
#endif
