/*
 * Shared serving-adapter template implementation. See
 * include/sparkpipe/spark_serving_adapter_template.h for the contract and
 * the paste this replaces (the tp_collective parse alone was re-pasted
 * per TP family, and the load/reserve spines per every family). The
 * family owns its policy; this file owns the walk.
 */

#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_serving_adapter_template.h"

int32_t SparkServingAdapterTemplateJsonMember(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,object,name));
}

SparkStatus SparkServingAdapterTemplateJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkServingAdapterTemplateJsonMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkTpCollectiveValidateMembers(
	const SparkJsonDocument *document,
	int32_t object,
	uint32_t backend_kind)
{
	static const char *const base_members[] =
	{
		"backend","backend_module_path","collective_identifier",
		"listen_port","connect_timeout_milli","operation_timeout_milli",
		"peer_hosts","peer_ports"
	};
	static const char *const adaptive_members[] =
	{
		"backend","backend_module_path","collective_identifier",
		"listen_port","connect_timeout_milli","operation_timeout_milli",
		"peer_hosts","peer_ports","algorithms",
		"direct_all_to_all_max_payload_bytes",
		"split_ring_min_payload_bytes","rail_peer_hosts",
		"step_rail_indices"
	};
	const char *const *members;
	uint32_t member_count;
	members = backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ?
		adaptive_members : base_members;
	member_count = backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ?
		(uint32_t)(sizeof(adaptive_members) / sizeof(adaptive_members[0])) :
		(uint32_t)(sizeof(base_members) / sizeof(base_members[0]));
	return(SparkJsonValidateObjectMembersExact(document,object,members,
		member_count));
}

static SparkStatus SparkTpCollectiveLoadAlgorithms(
	const SparkJsonDocument *document,
	int32_t object,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,mask;
	token = SparkServingAdapterTemplateJsonMember(document,object,"algorithms");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	mask = 0u;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		if ( SparkJsonStringEquals(document,element,"recursive_doubling") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
		else if ( SparkJsonStringEquals(document,element,
				"counter_rotating_split_ring") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
		else if ( SparkJsonStringEquals(document,element,"direct_all_to_all") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
		else
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( policy->algorithms == SPARK_TP_COLLECTIVE_ALGORITHMS_FULL_KNOWN_SET )
	{
		if ( count != 3u || mask != SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else
	{
		/* The collective implements split-ring and direct-all-to-all only
		 * at tp_degree 4, so the single-algorithm builds run recursive
		 * doubling alone. */
		if ( count != 1u ||
			mask != SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	topology->algorithm_mask = mask;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadStepRails(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,value;
	SparkStatus status;
	token = SparkServingAdapterTemplateJsonMember(document,object,
		"step_rail_indices");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonGetArrayElementCount(document,token) !=
			SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&value);
		if ( status != SPARK_STATUS_OK || value >=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
			return(SPARK_STATUS_SCHEMA_ERROR);
		topology->step_rail_indices[index] = value;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadRailHosts(
	const SparkJsonDocument *document,
	int32_t object,
	uint32_t peer_count,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,host_element,token;
	uint32_t host_count,index,rail;
	char *host;
	SparkStatus status;
	token = SparkServingAdapterTemplateJsonMember(document,object,
		"rail_peer_hosts");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonGetArrayElementCount(document,token) !=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	topology->rail_count = SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT;
	for (rail=0u; rail<topology->rail_count; rail++)
	{
		element = SparkJsonGetArrayElement(document,token,rail);
		if ( element < 0 ||
			!SparkJsonTokenIsType(document,element,SPARK_JSON_TOKEN_ARRAY) )
			return(SPARK_STATUS_SCHEMA_ERROR);
		host_count = SparkJsonGetArrayElementCount(document,element);
		if ( host_count != peer_count )
			return(SPARK_STATUS_SCHEMA_ERROR);
		for (index=0u; index<host_count; index++)
		{
			host_element = SparkJsonGetArrayElement(document,element,index);
			host = 0;
			status = host_element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
				SparkJsonCopyString(document,host_element,&host);
			if ( status == SPARK_STATUS_OK )
				status = SparkCopyString(
					topology->rail_rank_hosts[rail][index],
					SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
			free(host);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadThresholds(
	const SparkJsonDocument *document,
	int32_t object,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpDeviceCollectiveTopology *topology)
{
	SparkStatus status;
	status = SparkServingAdapterTemplateJsonUnsigned(document,object,
		"direct_all_to_all_max_payload_bytes",
		&topology->direct_all_to_all_max_payload_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(document,object,
			"split_ring_min_payload_bytes",
			&topology->split_ring_min_payload_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( policy->thresholds == SPARK_TP_COLLECTIVE_THRESHOLDS_ORDERED_NONZERO )
	{
		if ( topology->direct_all_to_all_max_payload_bytes == 0u ||
			topology->split_ring_min_payload_bytes == 0u ||
			topology->direct_all_to_all_max_payload_bytes >=
			topology->split_ring_min_payload_bytes )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else
	{
		if ( topology->direct_all_to_all_max_payload_bytes != 0u ||
			topology->split_ring_min_payload_bytes != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadAdaptiveFabric(
	const SparkJsonDocument *document,
	int32_t object,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config)
{
	SparkStatus status;
	if ( config->backend_kind !=
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		return(SPARK_STATUS_OK);
	status = SparkTpCollectiveLoadAlgorithms(document,object,policy,
		&config->topology);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadThresholds(document,object,policy,
			&config->topology);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadRailHosts(document,object,
			policy->peer_count,&config->topology);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadStepRails(document,object,
			&config->topology);
	return(status);
}

static SparkStatus SparkTpCollectiveLoadPeerHosts(
	const SparkJsonDocument *document,
	int32_t object,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config)
{
	int32_t element,token;
	uint32_t count,index;
	char *host;
	SparkStatus status;
	token = SparkServingAdapterTemplateJsonMember(document,object,"peer_hosts");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != policy->peer_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	config->peer_count = count;
	config->topology.rank_count = count;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		host = 0;
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonCopyString(document,element,&host);
		if ( status == SPARK_STATUS_OK )
			status = SparkCopyString(
				config->topology.rank_hosts[index],
				SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
		free(host);
		if ( status != SPARK_STATUS_OK ||
			config->topology.rank_hosts[index][0] == '\0' )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadPeerPorts(
	const SparkJsonDocument *document,
	int32_t object,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config)
{
	int32_t element,token;
	uint32_t count,index,port;
	SparkStatus status;
	token = SparkServingAdapterTemplateJsonMember(document,object,"peer_ports");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != policy->peer_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&port);
		if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
		config->peer_ports[index] = (uint16_t)port;
	}
	if ( policy->require_contiguous_peer_ports != 0u )
	{
		config->control_port_base = config->peer_ports[0];
		for (index=1u; index<count; index++)
		{
			if ( config->peer_ports[index] !=
				(uint16_t)(config->control_port_base + index) )
				return(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCollectiveLoadBackend(
	const SparkJsonDocument *document,
	int32_t object,
	const char *runtime_root,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config)
{
	int32_t token;
	uint32_t port;
	uint64_t collective_identifier;
	char *relative_backend_path;
	SparkStatus status;
	token = SparkServingAdapterTemplateJsonMember(document,object,"backend");
	if ( token < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkJsonStringEquals(document,token,"nccl") )
		config->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	else if ( SparkJsonStringEquals(document,token,"hidden_transport") )
		config->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	else
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkTpCollectiveValidateMembers(document,object,
		config->backend_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	relative_backend_path = 0;
	token = SparkServingAdapterTemplateJsonMember(document,object,
		"backend_module_path");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonCopyString(document,token,&relative_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_backend_path,
			config->backend_module_path_buffer,
			config->backend_module_path_bytes);
	free(relative_backend_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	token = SparkServingAdapterTemplateJsonMember(document,object,
		"collective_identifier");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonGetUInt64(document,token,&collective_identifier);
	if ( status != SPARK_STATUS_OK || (collective_identifier == 0u &&
		policy->allow_zero_collective_identifier == 0u) )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	config->collective_identifier = collective_identifier;
	status = SparkServingAdapterTemplateJsonUnsigned(document,object,
		"listen_port",&port);
	if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	config->listen_port = (uint16_t)port;
	status = SparkServingAdapterTemplateJsonUnsigned(document,object,
		"connect_timeout_milli",&config->connect_timeout_milli);
	if ( status != SPARK_STATUS_OK || config->connect_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	status = SparkServingAdapterTemplateJsonUnsigned(document,object,
		"operation_timeout_milli",&config->operation_timeout_milli);
	if ( status != SPARK_STATUS_OK || config->operation_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkServingAdapterTemplateLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	const SparkTpCollectiveConfigPolicy *policy,
	SparkTpCollectiveAdapterConfig *config)
{
	int32_t object;
	SparkStatus status;
	if ( document == 0 || runtime_root == 0 || policy == 0 || config == 0 ||
		policy->peer_count == 0u ||
		policy->peer_count > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE ||
		config->backend_module_path_buffer == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* Reset the outputs but keep the caller-set destination pointers. */
	memset(&config->topology,0,sizeof(config->topology));
	config->backend_kind = 0u;
	config->collective_identifier = 0u;
	config->listen_port = 0u;
	memset(config->peer_ports,0,sizeof(config->peer_ports));
	config->peer_count = 0u;
	config->connect_timeout_milli = 0u;
	config->operation_timeout_milli = 0u;
	config->control_port_base = 0u;
	config->topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	config->topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	object = SparkServingAdapterTemplateJsonMember(document,root,"tp_collective");
	if ( object < 0 ||
		!SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkTpCollectiveLoadBackend(document,object,runtime_root,
		policy,config);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadPeerHosts(document,object,policy,config);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadPeerPorts(document,object,policy,config);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCollectiveLoadAdaptiveFabric(document,object,policy,
			config);
	return(status);
}

SparkStatus SparkServingAdapterTemplateLoadDriver(
	const SparkServingAdapterDriverRequest *request,
	const SparkModelServingAdapterConfiguration *configuration,
	SparkLoadedModelDriver *driver,
	const SparkModelDriverProgramDescriptor **program_out,
	SparkServingAdapterProgramAcceptFunction program_accepts,
	void *accept_context,
	void **driver_instance_out)
{
	const SparkModelDriverDescriptor *descriptor;
	SparkModelDriverCreateRequest create_request;
	char error_buffer[512];
	SparkStatus status;
	if ( request == 0 || configuration == 0 || driver == 0 ||
		program_out == 0 || program_accepts == 0 ||
		driver_instance_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkLoadedModelDriverReset(driver);
	status = SparkLoadModelDriver(configuration->driver_shared_object_path,
		configuration->node_target,driver,error_buffer,sizeof(error_buffer));
	if ( status != SPARK_STATUS_OK )
		return(status);
	descriptor = driver->interface->descriptor;
	if ( descriptor == 0 ||
		strcmp(descriptor->model_id,request->contract.driver_model_id) != 0 ||
		strcmp(descriptor->model_revision,request->contract.driver_model_revision) != 0 ||
		strcmp(descriptor->stage_name,request->contract.driver_stage_name) != 0 ||
		(request->contract.driver_target != 0 &&
			strcmp(descriptor->target,request->contract.driver_target) != 0) ||
		strcmp(descriptor->model_description_sha256,request->contract.model_description_sha256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	*program_out = SparkFindLoadedModelDriverProgram(driver,
		configuration->driver_program_name);
	if ( *program_out == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( driver->interface->admit == 0 || (*program_out)->submit == 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	if ( program_accepts(*program_out,accept_context) != SPARK_STATUS_OK )
		return(SPARK_STATUS_TARGET_MISMATCH);
	SparkModelDriverInitializeCreateRequest(&create_request);
	create_request.node_id = configuration->node_id;
	create_request.node_target = configuration->node_target;
	if ( request->node_context != 0 )
		create_request.node_context = request->node_context;
	create_request.kv_logical_page_capacity =
		configuration->runtime_limits.kv_logical_page_capacity;
	create_request.kv_physical_page_capacity =
		configuration->runtime_limits.kv_physical_page_capacity;
	create_request.kv_backing_directory =
		configuration->kv_backing_directory;
	create_request.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	create_request.execution_stream = configuration->execution_stream;
	create_request.completion_function = request->completion_function;
	create_request.completion_context = request->completion_context;
	create_request.wake_function = request->wake_function;
	create_request.wake_context = request->completion_context;
	status = driver->interface->create(&create_request,driver_instance_out);
	return(status == SPARK_STATUS_OK && *driver_instance_out == 0 ?
		SPARK_STATUS_INVALID_ARGUMENT : status);
}

void *SparkServingAdapterTemplateReservePending(
	void *pending_array,
	uint32_t element_bytes,
	uint32_t common_offset,
	uint32_t pipeline_slot_count,
	uint32_t last_row_by_lane_offset,
	const SparkModelServingSubmission *submission)
{
	uint8_t *elements;
	SparkServingAdapterPendingCommon *common;
	uint32_t *last_row_by_lane;
	uint32_t index,row;
	if ( pending_array == 0 || element_bytes < sizeof(*common) ||
		common_offset > element_bytes - sizeof(*common) ||
		submission == 0 )
		return(0);
	elements = (uint8_t *)pending_array;
	for (index=0u; index<pipeline_slot_count; index++)
	{
		void *element;
		element = elements + ((size_t)index * element_bytes);
		/* The family struct embeds the common view after its own owner
		 * pointer, so the view lives at the family-supplied common_offset,
		 * not at the element base (same family-layout-as-data rule as
		 * last_row_by_lane_offset below). */
		common = (SparkServingAdapterPendingCommon *)(void *)
			((uint8_t *)element + common_offset);
		if ( common->active != 0u )
			continue;
		memset(element,0,element_bytes);
		common->row_count = submission->row_count;
		common->lane_count = submission->lane_count;
		common->active_sequence_count = submission->active_sequence_count;
		common->work_kind = submission->work_kind;
		common->tokens_per_sequence = submission->tokens_per_sequence;
		common->submission_id = submission->submission_id;
		common->request_id = submission->request_id;
		common->sequence_id = submission->sequence_id;
		common->sequence_position = submission->sequence_position;
		common->control_generation = submission->control_generation;
		common->transaction_id = submission->transaction_id;
		common->dispatch_generation = submission->dispatch_generation;
		common->request_generation = submission->request_generation;
		common->step_generation = submission->step_generation;
		last_row_by_lane = (uint32_t *)(void *)((uint8_t *)element +
			last_row_by_lane_offset);
		for (row=0u; row<submission->row_count; row++)
		{
			uint32_t lane;
			lane = submission->row_lane_indices[row];
			last_row_by_lane[lane] = row;
		}
		/* The ACTIVE flag is the family's to set: the pasted reserve marks
		 * the slot live only after the family's own fill steps (cache lanes,
		 * emit rows) succeed, and a failed fill must leave the slot free. */
		return(element);
	}
	return(0);
}
