#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_pipeline_runtime.h"

static void TestBuildDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	static const uint32_t layers[13] = {3u,3u,3u,3u,3u,3u,3u,4u,4u,4u,4u,4u,2u};
	uint32_t index;
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT;
	descriptor->stage_count = 13u;
	descriptor->layer_count = 43u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 16384u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 128u;
	descriptor->max_input_row_count = 128u;
	descriptor->max_resident_sequence_count = 256u;
	descriptor->max_output_token_count = 128u;
	descriptor->adapter_id = "test.adapter";
	descriptor->model_id = "test/model";
	descriptor->model_revision = "revision";
	descriptor->driver_program_name = "decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	for (index=0u; index<13u; index++)
		descriptor->stage_layer_counts[index] = layers[index];
	descriptor->boundary_sideband_kinds[0] = 1u;
	descriptor->boundary_sideband_bytes_per_sequence[0] = 8192u;
}

static void TestBuildNode(
	SparkPipelineRuntimeLinearNode *node,
	uint32_t rank_index,
	uint32_t stage_index,
	uint32_t previous_rank_index,
	uint32_t next_rank_index,
	const char *host,
	const char *previous,
	const char *next)
{
	memset(node,0,sizeof(*node));
	node->abi_version = SPARK_PIPELINE_RUNTIME_ABI_VERSION;
	node->descriptor_bytes = SPARK_PIPELINE_RUNTIME_LINEAR_NODE_BYTES;
	node->rank_index = rank_index;
	node->stage_index = stage_index;
	node->stage_count = 13u;
	node->previous_rank_index = previous_rank_index;
	node->next_rank_index = next_rank_index;
	node->host_name = host;
	node->previous_host_name = previous;
	 node->next_host_name = next;
}

static void TestBuildFanoutDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	uint32_t index;
	TestBuildDescriptor(descriptor);
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT;
	descriptor->stage_count = 16u;
	memset(descriptor->stage_layer_counts,0,sizeof(descriptor->stage_layer_counts));
	for (index=0u; index<13u; index++)
		descriptor->stage_layer_counts[index] = 3u;
	descriptor->stage_layer_counts[13] = 2u;
	descriptor->stage_layer_counts[14] = 1u;
	descriptor->stage_layer_counts[15] = 1u;
	descriptor->boundary_sideband_kinds[0] = 0u;
	descriptor->boundary_sideband_bytes_per_sequence[0] = 0u;
}

int main(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingAdapterDescriptor fanout_descriptor;
	SparkHiddenTransportEndpoint endpoint;
	SparkPipelineRuntimeLinearNode node;
	SparkPipelineRuntimeFanoutNode fanout_node;
	SparkPipelineRuntimeRankPlan copied_plan,rank_plan;
	TestBuildDescriptor(&descriptor);
	TestBuildNode(&node,0u,0u,SPARK_PIPELINE_RUNTIME_NO_RANK,2u,"node-alpha",0,"node-beta");
	assert(SparkPipelineRuntimeBuildLinearRankPlan(&descriptor,&node,64u,128u,SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS,59000u,SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID,&rank_plan) == SPARK_STATUS_OK);
	assert(rank_plan.first_layer_index == 0u);
	assert(rank_plan.layer_count == 3u);
	assert(strcmp(rank_plan.host_name,"node-alpha") == 0);
	assert(strcmp(rank_plan.next_host_name,"node-beta") == 0);
	assert(rank_plan.boundary_bytes_per_sequence == 32768u);
	assert(rank_plan.output_sideband_kind == 1u);
	assert(rank_plan.output_sideband_bytes_per_sequence == 8192u);
	assert(rank_plan.output_packet_bytes_per_sequence == 40960u);
	assert(rank_plan.max_active_sequence_count == 64u);
	assert(rank_plan.max_input_row_count == 128u);
	assert(rank_plan.output_max_packet_bytes == 5242880u);
	assert(SparkPipelineRuntimeBuildOutputEndpoint(&descriptor,&rank_plan,&endpoint) == SPARK_STATUS_OK);
	assert(endpoint.max_active_sequence_count == 128u);
	assert(endpoint.bytes_per_sequence == 32768u);
	assert(endpoint.max_packet_bytes == 5242880u);
	assert(endpoint.configuration_flags == SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION);
	assert(endpoint.local_rank_index == 0u);
	assert(endpoint.source_rank_index == 0u);
	assert(endpoint.sink_rank_index == 2u);
	assert(strcmp(endpoint.source_host,"node-alpha") == 0);
	assert(strcmp(endpoint.sink_host,"node-beta") == 0);
	assert(strcmp(endpoint.route_name,"rank0_to_rank2_hidden") == 0);
	assert(endpoint.control_port_base == 59000u);
	assert(SparkPipelineRuntimeBuildInputEndpoint(&descriptor,&rank_plan,&endpoint) == SPARK_STATUS_NOT_FOUND);
	assert(strcmp(rank_plan.transport_module_id,SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID) == 0);
	assert((rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u);
	assert((rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u);
	TestBuildNode(&node,12u,12u,10u,SPARK_PIPELINE_RUNTIME_NO_RANK,"node-omega","node-psi",0);
	assert(SparkPipelineRuntimeBuildLinearRankPlan(&descriptor,&node,128u,128u,SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS,59000u,SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID,&rank_plan) == SPARK_STATUS_OK);
	assert(rank_plan.first_layer_index == 41u);
	assert(rank_plan.layer_count == 2u);
	assert(strcmp(rank_plan.host_name,"node-omega") == 0);
	assert(strcmp(rank_plan.previous_host_name,"node-psi") == 0);
	copied_plan = rank_plan;
	assert(SparkPipelineRuntimeBuildInputEndpoint(&descriptor,&copied_plan,&endpoint) == SPARK_STATUS_OK);
	assert(endpoint.source_rank_index == 10u);
	assert(endpoint.sink_rank_index == 12u);
	assert((rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u);
	assert(SparkPipelineRuntimeValidateRankPlan(&descriptor,&rank_plan) == SPARK_STATUS_OK);
	TestBuildNode(&node,0u,0u,SPARK_PIPELINE_RUNTIME_NO_RANK,2u,"node-alpha",0,"node-beta");
	assert(SparkPipelineRuntimeBuildLinearRankPlan(&descriptor,&node,64u,32u,SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS,59000u,SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID,&rank_plan) == SPARK_STATUS_INVALID_ARGUMENT);
	TestBuildFanoutDescriptor(&fanout_descriptor);
	memset(&fanout_node,0,sizeof(fanout_node));
	fanout_node.abi_version = SPARK_PIPELINE_RUNTIME_ABI_VERSION;
	fanout_node.descriptor_bytes = SPARK_PIPELINE_RUNTIME_FANOUT_NODE_BYTES;
	fanout_node.rank_index = 15u;
	fanout_node.stage_index = 15u;
	fanout_node.stage_count = 16u;
	fanout_node.host_name = "tp-rank-15";
	assert(SparkPipelineRuntimeBuildFanoutRankPlan(&fanout_descriptor,&fanout_node,1u,1u,0u,&rank_plan) == SPARK_STATUS_OK);
	assert((rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_PARALLEL_FANOUT) != 0u);
	assert((rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u);
	assert((rank_plan.flags & (SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS | SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT)) == 0u);
	assert(rank_plan.first_layer_index == 0u);
	assert(rank_plan.layer_count == 43u);
	assert(rank_plan.input_packet_bytes_per_sequence == 0u);
	assert(rank_plan.output_packet_bytes_per_sequence == 0u);
	assert(SparkPipelineRuntimeBuildInputEndpoint(&fanout_descriptor,&rank_plan,&endpoint) == SPARK_STATUS_NOT_FOUND);
	assert(SparkPipelineRuntimeBuildOutputEndpoint(&fanout_descriptor,&rank_plan,&endpoint) == SPARK_STATUS_NOT_FOUND);
	assert(SparkPipelineRuntimeValidateRankPlan(&fanout_descriptor,&rank_plan) == SPARK_STATUS_OK);
	return(0);
}
