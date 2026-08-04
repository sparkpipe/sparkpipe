#include "sparkpipe/spark_pipeline_runtime.h"

#include <stdio.h>
#include <string.h>

static SparkStatus SparkPipelineRuntimeFormatRoute(
	uint32_t source_rank_index,
	uint32_t sink_rank_index,
	char *route,
	uint32_t route_bytes)
{
	int32_t written;
	if ( source_rank_index == sink_rank_index || route == 0 || route_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	written = snprintf(route,route_bytes,"rank%u_to_rank%u_hidden",source_rank_index,sink_rank_index);
	return(written < 0 || (uint32_t)written >= route_bytes ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK);
}

static SparkStatus SparkPipelineRuntimeCopyText(
	char *destination,
	uint32_t destination_bytes,
	const char *source)
{
	uint32_t bytes;
	if ( destination == 0 || destination_bytes == 0u || source == 0 || source[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	bytes = (uint32_t)strlen(source) + 1u;
	if ( bytes > destination_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(destination,source,bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkPipelineRuntimeInitializeEndpoint(
	SparkHiddenTransportEndpoint *endpoint,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	uint32_t source_rank_index,
	uint32_t sink_rank_index,
	const char *source_host,
	const char *sink_host,
	const char *route_name)
{
	memset(endpoint,0,sizeof(*endpoint));
	endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
	endpoint->capability_flags = rank_plan->transport_capability_flags;
	endpoint->hidden_dimension = rank_plan->boundary_element_count;
	endpoint->bytes_per_sequence = (uint32_t)rank_plan->bytes_per_sequence;
	endpoint->max_active_sequence_count = rank_plan->max_input_row_count;
	endpoint->configuration_flags = SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION;
	endpoint->local_rank_index = rank_plan->rank_index;
	endpoint->source_rank_index = source_rank_index;
	endpoint->sink_rank_index = sink_rank_index;
	endpoint->control_port_base = rank_plan->transport_control_port_base;
	endpoint->max_packet_bytes = rank_plan->max_packet_bytes;
	endpoint->transport_module_id = rank_plan->transport_module_id;
	endpoint->route_name = route_name;
	endpoint->source_host = source_host;
	endpoint->sink_host = sink_host;
	return(SparkHiddenTransportValidateEndpoint(endpoint));
}

static uint32_t SparkPipelineRuntimeNeighborIsValid(
	uint32_t stage_index,
	uint32_t stage_count,
	uint32_t rank_index,
	uint32_t neighbor_rank_index,
	const char *neighbor_host,
	uint32_t previous)
{
	uint32_t has_neighbor;
	has_neighbor = previous != 0u ? stage_index != 0u : stage_index + 1u < stage_count;
	if ( has_neighbor == 0u )
		return(neighbor_rank_index == SPARK_PIPELINE_RUNTIME_NO_RANK && neighbor_host == 0 ? 1u : 0u);
	return(neighbor_rank_index < stage_count && neighbor_rank_index != rank_index && neighbor_host != 0 && neighbor_host[0] != '\0' ? 1u : 0u);
}

static uint32_t SparkPipelineRuntimeTextBufferIsValid(
	const char *text,
	uint32_t text_bytes,
	uint32_t required)
{
	const char *end;
	end = (const char *)memchr(text,'\0',text_bytes);
	return(end != 0 && (required == 0u || end != text) ? 1u : 0u);
}

static SparkStatus SparkPipelineRuntimeGetStageSlice(
	const SparkModelServingAdapterDescriptor *descriptor,
	uint32_t stage_index,
	uint32_t *first_layer_index,
	uint32_t *layer_count)
{
	uint32_t index,first;
	if ( descriptor == 0 || stage_index >= descriptor->stage_count || first_layer_index == 0 || layer_count == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	first = 0u;
	for (index=0u; index<stage_index; index++)
		first += descriptor->stage_layer_counts[index];
	*first_layer_index = first;
	*layer_count = descriptor->stage_layer_counts[stage_index];
	return(SPARK_STATUS_OK);
}

SparkStatus SparkPipelineRuntimeBuildLinearRankPlan(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeLinearNode *node,
	uint32_t max_active_sequence_count,
	uint32_t max_input_row_count,
	uint32_t transport_capability_flags,
	uint32_t transport_control_port_base,
	const char *transport_module_id,
	SparkPipelineRuntimeRankPlan *rank_plan)
{
	SparkStatus status;
	uint64_t bytes_per_sequence;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || node == 0 || rank_plan == 0 || transport_module_id == 0 || transport_module_id[0] == '\0' || node->abi_version != SPARK_PIPELINE_RUNTIME_ABI_VERSION || node->descriptor_bytes != SPARK_PIPELINE_RUNTIME_LINEAR_NODE_BYTES || node->rank_index >= descriptor->stage_count || node->stage_index >= descriptor->stage_count || node->stage_count != descriptor->stage_count || node->reserved0 != 0u || node->host_name == 0 || node->host_name[0] == '\0' || SparkPipelineRuntimeNeighborIsValid(node->stage_index,node->stage_count,node->rank_index,node->previous_rank_index,node->previous_host_name,1u) == 0u || SparkPipelineRuntimeNeighborIsValid(node->stage_index,node->stage_count,node->rank_index,node->next_rank_index,node->next_host_name,0u) == 0u || max_active_sequence_count == 0u || max_active_sequence_count > descriptor->max_active_sequence_count || max_input_row_count == 0u || max_input_row_count > descriptor->max_input_row_count || max_input_row_count < max_active_sequence_count || transport_control_port_base == 0u || transport_control_port_base > UINT16_MAX - (descriptor->stage_count - 1u) || descriptor->boundary_format != SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16 || descriptor->boundary_element_bytes != SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	bytes_per_sequence = (uint64_t)descriptor->boundary_element_count * descriptor->boundary_element_bytes;
	if ( bytes_per_sequence > UINT32_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(rank_plan,0,sizeof(*rank_plan));
	rank_plan->abi_version = SPARK_PIPELINE_RUNTIME_ABI_VERSION;
	rank_plan->descriptor_bytes = SPARK_PIPELINE_RUNTIME_RANK_PLAN_BYTES;
	rank_plan->rank_index = node->rank_index;
	rank_plan->stage_index = node->stage_index;
	rank_plan->stage_count = descriptor->stage_count;
	rank_plan->previous_rank_index = node->previous_rank_index;
	rank_plan->next_rank_index = node->next_rank_index;
	status = SparkPipelineRuntimeGetStageSlice(descriptor,node->stage_index,&rank_plan->first_layer_index,&rank_plan->layer_count);
	rank_plan->max_active_sequence_count = max_active_sequence_count;
	rank_plan->max_input_row_count = max_input_row_count;
	rank_plan->boundary_format = descriptor->boundary_format;
	rank_plan->boundary_element_count = descriptor->boundary_element_count;
	rank_plan->boundary_element_bytes = descriptor->boundary_element_bytes;
	rank_plan->transport_capability_flags = transport_capability_flags;
	rank_plan->transport_control_port_base = transport_control_port_base;
	rank_plan->bytes_per_sequence = bytes_per_sequence;
	rank_plan->max_packet_bytes = bytes_per_sequence * max_input_row_count;
	if ( status == SPARK_STATUS_OK )
		status = SparkPipelineRuntimeCopyText(rank_plan->transport_module_id,sizeof(rank_plan->transport_module_id),transport_module_id);
	if ( status == SPARK_STATUS_OK )
		status = SparkPipelineRuntimeCopyText(rank_plan->host_name,sizeof(rank_plan->host_name),node->host_name);
	if ( status == SPARK_STATUS_OK && node->stage_index != 0u )
	{
		rank_plan->flags |= SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS;
		status = SparkPipelineRuntimeCopyText(rank_plan->previous_host_name,sizeof(rank_plan->previous_host_name),node->previous_host_name);
		if ( status == SPARK_STATUS_OK )
			status = SparkPipelineRuntimeFormatRoute(rank_plan->previous_rank_index,rank_plan->rank_index,rank_plan->input_route_name,sizeof(rank_plan->input_route_name));
	}
	if ( status == SPARK_STATUS_OK && node->stage_index + 1u < descriptor->stage_count )
	{
		rank_plan->flags |= SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT;
		status = SparkPipelineRuntimeCopyText(rank_plan->next_host_name,sizeof(rank_plan->next_host_name),node->next_host_name);
		if ( status == SPARK_STATUS_OK )
			status = SparkPipelineRuntimeFormatRoute(rank_plan->rank_index,rank_plan->next_rank_index,rank_plan->output_route_name,sizeof(rank_plan->output_route_name));
	}
	if ( node->stage_index + 1u == descriptor->stage_count )
		rank_plan->flags |= SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE;
	return(status == SPARK_STATUS_OK ? SparkPipelineRuntimeValidateRankPlan(descriptor,rank_plan) : status);
}

SparkStatus SparkPipelineRuntimeValidateRankPlan(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan)
{
	SparkStatus status;
	uint32_t first_layer_index,layer_count;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || rank_plan == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( rank_plan->abi_version != SPARK_PIPELINE_RUNTIME_ABI_VERSION || rank_plan->descriptor_bytes != SPARK_PIPELINE_RUNTIME_RANK_PLAN_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( SparkPipelineRuntimeTextBufferIsValid(rank_plan->transport_module_id,sizeof(rank_plan->transport_module_id),1u) == 0u || SparkPipelineRuntimeTextBufferIsValid(rank_plan->host_name,sizeof(rank_plan->host_name),1u) == 0u || SparkPipelineRuntimeTextBufferIsValid(rank_plan->previous_host_name,sizeof(rank_plan->previous_host_name),rank_plan->stage_index != 0u) == 0u || SparkPipelineRuntimeTextBufferIsValid(rank_plan->next_host_name,sizeof(rank_plan->next_host_name),rank_plan->stage_index + 1u < rank_plan->stage_count) == 0u || SparkPipelineRuntimeTextBufferIsValid(rank_plan->input_route_name,sizeof(rank_plan->input_route_name),rank_plan->stage_index != 0u) == 0u || SparkPipelineRuntimeTextBufferIsValid(rank_plan->output_route_name,sizeof(rank_plan->output_route_name),rank_plan->stage_index + 1u < rank_plan->stage_count) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( rank_plan->rank_index >= descriptor->stage_count || rank_plan->stage_index >= descriptor->stage_count || rank_plan->stage_count != descriptor->stage_count || (rank_plan->flags & ~SPARK_PIPELINE_RUNTIME_RANK_KNOWN_FLAGS) != 0u || rank_plan->reserved0 != 0u || rank_plan->boundary_format != descriptor->boundary_format || rank_plan->boundary_element_count != descriptor->boundary_element_count || rank_plan->boundary_element_bytes != descriptor->boundary_element_bytes || rank_plan->max_active_sequence_count == 0u || rank_plan->max_active_sequence_count > descriptor->max_active_sequence_count || rank_plan->max_input_row_count < rank_plan->max_active_sequence_count || rank_plan->max_input_row_count > descriptor->max_input_row_count || rank_plan->transport_control_port_base == 0u || rank_plan->transport_control_port_base > UINT16_MAX - (rank_plan->stage_count - 1u) || rank_plan->bytes_per_sequence != (uint64_t)descriptor->boundary_element_count * descriptor->boundary_element_bytes || rank_plan->max_packet_bytes != rank_plan->bytes_per_sequence * rank_plan->max_input_row_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkPipelineRuntimeNeighborIsValid(rank_plan->stage_index,rank_plan->stage_count,rank_plan->rank_index,rank_plan->previous_rank_index,rank_plan->stage_index != 0u ? rank_plan->previous_host_name : 0,1u) == 0u || SparkPipelineRuntimeNeighborIsValid(rank_plan->stage_index,rank_plan->stage_count,rank_plan->rank_index,rank_plan->next_rank_index,rank_plan->stage_index + 1u < rank_plan->stage_count ? rank_plan->next_host_name : 0,0u) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkPipelineRuntimeGetStageSlice(descriptor,rank_plan->stage_index,&first_layer_index,&layer_count);
	if ( status != SPARK_STATUS_OK || rank_plan->first_layer_index != first_layer_index || rank_plan->layer_count != layer_count )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((rank_plan->flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u) != (rank_plan->stage_index != 0u) || ((rank_plan->flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u) != (rank_plan->stage_index + 1u < rank_plan->stage_count) || ((rank_plan->flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u) != (rank_plan->stage_index + 1u == rank_plan->stage_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkPipelineRuntimeBuildEndpoint(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	uint32_t input,
	SparkHiddenTransportEndpoint *endpoint)
{
	SparkStatus status;
	status = SparkPipelineRuntimeValidateRankPlan(descriptor,rank_plan);
	if ( status != SPARK_STATUS_OK || endpoint == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( input != 0u )
	{
		if ( (rank_plan->flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u )
			return(SPARK_STATUS_NOT_FOUND);
		return(SparkPipelineRuntimeInitializeEndpoint(endpoint,rank_plan,rank_plan->previous_rank_index,rank_plan->rank_index,rank_plan->previous_host_name,rank_plan->host_name,rank_plan->input_route_name));
	}
	if ( (rank_plan->flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	return(SparkPipelineRuntimeInitializeEndpoint(endpoint,rank_plan,rank_plan->rank_index,rank_plan->next_rank_index,rank_plan->host_name,rank_plan->next_host_name,rank_plan->output_route_name));
}

SparkStatus SparkPipelineRuntimeBuildInputEndpoint(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	SparkHiddenTransportEndpoint *endpoint)
{
	return(SparkPipelineRuntimeBuildEndpoint(descriptor,rank_plan,1u,endpoint));
}

SparkStatus SparkPipelineRuntimeBuildOutputEndpoint(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkPipelineRuntimeRankPlan *rank_plan,
	SparkHiddenTransportEndpoint *endpoint)
{
	return(SparkPipelineRuntimeBuildEndpoint(descriptor,rank_plan,0u,endpoint));
}
