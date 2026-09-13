/*
 * Shared serving-adapter skeleton implementation. See adapter_common.h for
 * the contract; every function body here is the statement sequence lifted
 * from the four family adapters (the four migrated adapters), with the
 * family name replaced by the shared state it now travels in.
 */

#include "adapter_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The fused-admission resolve helpers call the admission surface directly. */
#include "sparkpipe/spark_admission.h"

/* ---- slot claiming ----------------------------------------------------- */

void SparkAdapterPendingCapture(
	SparkAdapterPendingCore *core,
	const SparkModelServingSubmission *submission)
{
	core->row_count = submission->row_count;
	core->lane_count = submission->lane_count;
	core->active_sequence_count = submission->active_sequence_count;
	core->work_kind = submission->work_kind;
	core->identity.submission_id = submission->submission_id;
	core->identity.request_id = submission->request_id;
	core->identity.sequence_id = submission->sequence_id;
	core->identity.sequence_position = submission->sequence_position;
	core->identity.control_generation = submission->control_generation;
	core->identity.transaction_id = submission->transaction_id;
	core->identity.dispatch_generation = submission->dispatch_generation;
	core->identity.request_generation = submission->request_generation;
	core->identity.step_generation = submission->step_generation;
}

static const SparkAdapterPendingCore *SparkAdapterPendingCoreAt(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t index)
{
	return(const SparkAdapterPendingCore *)(
		(const char *)pending_table + index * pending_element_bytes);
}

int32_t SparkAdapterPendingClaim(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count)
{
	uint32_t index;
	for (index=0u; index<slot_count; index++)
		if ( SparkAdapterPendingCoreAt(pending_table,
			pending_element_bytes,index)->active == 0u )
			return((int32_t)index);
	return(-1);
}

uint32_t SparkAdapterAvailableSubmissionCount(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<slot_count; index++)
		available += SparkAdapterPendingCoreAt(pending_table,
			pending_element_bytes,index)->active == 0u ? 1u : 0u;
	return(available);
}

void SparkAdapterCaptureLastRowByLane(
	const SparkModelServingSubmission *submission,
	uint32_t *last_row_by_lane)
{
	uint32_t lane,row;
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		last_row_by_lane[lane] = row;
	}
}

void SparkAdapterCaptureResidentSlotsPerLane(
	const SparkModelServingSubmission *submission,
	uint32_t lane_count,
	uint32_t *resident_slots)
{
	uint32_t lane;
	for (lane=0u; lane<lane_count; lane++)
		resident_slots[lane] = submission->lanes[lane].resident_sequence_slot;
}

void SparkAdapterCaptureResidentSlotsPerRow(
	const SparkModelServingSubmission *submission,
	uint32_t *slots_by_row)
{
	uint32_t lane,row;
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		slots_by_row[row] = submission->lanes[lane].resident_sequence_slot;
	}
}

/* ---- completion routing ------------------------------------------------- */

void SparkAdapterOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkAdapterCommonState *common;
	(void)driver_completion;
	common = (SparkAdapterCommonState *)completion_context;
	if ( common != 0 )
		common->orphan_completion_count++;
}

void SparkAdapterDispatchWake(void *wake_context)
{
	SparkAdapterCommonState *common;
	common = (SparkAdapterCommonState *)wake_context;
	if ( common != 0 && common->wake_function != 0 )
		common->wake_function(common->wake_context);
}

uint32_t SparkAdapterDriverCompletionMatches(
	const SparkModelDriverCompletion *driver_completion,
	uint64_t request_id,
	uint64_t sequence_id,
	uint64_t sequence_position,
	uint32_t program_id)
{
	return(driver_completion->request_id == request_id &&
		driver_completion->sequence_id == sequence_id &&
		driver_completion->sequence_position == sequence_position &&
		driver_completion->program_id == program_id) ? 1u : 0u;
}

void SparkAdapterAccumulateFrameCounters(
	uint64_t *accepted_token_count,
	uint64_t *queue_delay_ns,
	uint64_t *service_time_ns,
	const SparkModelDriverCompletion *driver_completion)
{
	*accepted_token_count += driver_completion->accepted_token_count;
	*queue_delay_ns += driver_completion->queue_delay_ns;
	*service_time_ns += driver_completion->service_time_ns;
}

void SparkAdapterBuildCompletionHeader(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core)
{
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->submission_id = core->identity.submission_id;
	completion->request_id = core->identity.request_id;
	completion->sequence_id = core->identity.sequence_id;
	completion->sequence_position = core->identity.sequence_position;
	completion->control_generation = core->identity.control_generation;
	completion->transaction_id = core->identity.transaction_id;
	completion->dispatch_generation = core->identity.dispatch_generation;
	completion->request_generation = core->identity.request_generation;
	completion->step_generation = core->identity.step_generation;
}

void SparkAdapterBuildCompletionHeaderWithResidency(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core,
	const SparkModelDriverResidencyToken *residency)
{
	SparkAdapterBuildCompletionHeader(completion, core);
	if ( residency != 0 )
		completion->residency = *residency;
}

uint32_t SparkAdapterClampAcceptedTokenCount(uint64_t accepted_token_count)
{
	return((uint32_t)(accepted_token_count > UINT32_MAX ? UINT32_MAX :
		accepted_token_count));
}

/* ---- lifecycle ----------------------------------------------------------- */

SparkStatus SparkAdapterValidateConfiguration(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingAdapterConfiguration *configuration,
	const char *program_name,
	uint32_t stage_count)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,
		&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= stage_count ||
		configuration->runtime_root == 0 || configuration->node_id == 0 ||
		configuration->node_target == 0 ||
		configuration->adapter_configuration_path == 0 ||
		configuration->driver_shared_object_path == 0 ||
		configuration->driver_program_name == 0 ||
		strcmp(configuration->driver_program_name,program_name) != 0 ||
		configuration->execution_stream == 0 ||
		configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkAdapterInitializePrologue(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration)
{
	common->pipeline_slot_count =
		configuration->runtime_limits.max_inflight_submission_count;
	common->max_active_sequence_count =
		configuration->runtime_limits.max_active_sequence_count;
	common->max_input_row_count =
		configuration->runtime_limits.max_input_row_count;
	common->resident_sequence_capacity =
		configuration->runtime_limits.resident_sequence_capacity;
	common->runtime_limits = configuration->runtime_limits;
	common->sink.function = configuration->completion_function;
	common->sink.context = configuration->completion_context;
	common->wake_function = configuration->wake_function;
	common->wake_context = configuration->wake_context;
	common->execution_stream = configuration->execution_stream;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkAdapterLoadDriver(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration,
	const SparkAdapterDriverContract *contract)
{
	const SparkModelDriverDescriptor *descriptor;
	SparkModelDriverCreateRequest request;
	char error_buffer[512];
	SparkStatus status;
	SparkLoadedModelDriverReset(&common->driver);
	status = SparkLoadModelDriver(configuration->driver_shared_object_path,
		configuration->node_target,&common->driver,error_buffer,
		sizeof(error_buffer));
	if ( status != SPARK_STATUS_OK )
		return(status);
	descriptor = common->driver.interface->descriptor;
	if ( descriptor == 0 ||
		strcmp(descriptor->model_id,contract->model_id) != 0 ||
		strcmp(descriptor->model_revision,contract->model_revision) != 0 ||
		strcmp(descriptor->stage_name,contract->stage_name) != 0 ||
		(contract->target != 0 &&
			strcmp(descriptor->target,contract->target) != 0) ||
		strcmp(descriptor->model_description_sha256,
			contract->description_sha256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	common->program = SparkFindLoadedModelDriverProgram(&common->driver,
		configuration->driver_program_name);
	if ( common->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( contract->check_kind == 0u )
	{
		if ( common->driver.interface->admit == 0 ||
			common->program->submit == 0 ||
			SparkModelDriverProgramSupportsRuntimeLimits(common->program,
				contract->required_program_flags,
				common->pipeline_slot_count,
				common->max_active_sequence_count,
				common->max_input_row_count,
				common->resident_sequence_capacity) == 0u )
			return(SPARK_STATUS_TARGET_MISMATCH);
	}
	else
	{
		if ( common->driver.interface->admit == 0 ||
			common->program->submit == 0 ||
			(common->program->flags & contract->required_program_flags) !=
				contract->required_program_flags ||
			common->program->max_inflight < common->pipeline_slot_count ||
			common->program->profile == 0 ||
			common->program->profile->max_active_slots <
				common->max_active_sequence_count ||
			common->program->profile->max_new_tokens <
				common->max_input_row_count )
			return(SPARK_STATUS_TARGET_MISMATCH);
	}
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = configuration->node_id;
	request.node_target = configuration->node_target;
	request.node_context = contract->node_context;
	request.kv_logical_page_capacity =
		configuration->runtime_limits.kv_logical_page_capacity;
	request.kv_physical_page_capacity =
		configuration->runtime_limits.kv_physical_page_capacity;
	request.kv_backing_directory = configuration->kv_backing_directory;
	request.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	request.execution_stream = configuration->execution_stream;
	request.completion_function = SparkAdapterOrphanDriverCompletion;
	request.completion_context = common;
	request.wake_function = SparkAdapterDispatchWake;
	request.wake_context = common;
	status = common->driver.interface->create(&request,
		&common->driver_instance);
	return(status == SPARK_STATUS_OK && common->driver_instance == 0 ?
		SPARK_STATUS_INVALID_ARGUMENT : status);
}

SparkStatus SparkAdapterValidateSubmissionOpen(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission)
{
	if ( common == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( common->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	return(SparkModelServingAdapterValidateRuntimeSubmission(descriptor,
		&common->runtime_limits,submission));
}

/*
 * The complete quiesce body all four adapters ship: presence/deadline
 * gate, the quiescing latch, the idle-slot demand, then the driver
 * active-submission poll.
 */
SparkStatus SparkAdapterQuiesce(
	SparkAdapterCommonState *common,
	uint64_t deadline_time_ns)
{
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	if ( common == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	common->quiescing = 1u;
	if ( SparkAdapterAvailableSubmissionCount(common->pending,
		common->pending_element_bytes,
		common->pipeline_slot_count) != common->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = common->driver.interface->snapshot(common->driver_instance,
		common->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK :
		SPARK_STATUS_BUSY);
}

/*
 * Snapshot merge: driver counters + slot availability + orphans. Opens with
 * the argument gate every source adapter ships (state==0 or snapshot==0 is
 * INVALID_ARGUMENT) - migrating must not turn snapshot(NULL,...) into a
 * crash, so the gate moves WITH the body.
 */
SparkStatus SparkAdapterSnapshot(
	SparkAdapterCommonState *common,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	if ( common == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = common->driver.interface->snapshot(common->driver_instance,
		common->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkAdapterAvailableSubmissionCount(common->pending,
		common->pending_element_bytes,common->pipeline_slot_count);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count =
		common->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = common->pipeline_slot_count -
		SparkAdapterAvailableSubmissionCount(common->pending,
			common->pending_element_bytes,common->pipeline_slot_count);
	snapshot->submitted_count = driver_snapshot.submitted_count;
	snapshot->completed_count = driver_snapshot.completed_count;
	snapshot->rejected_count = driver_snapshot.rejected_count +
		common->orphan_completion_count;
	snapshot->resident_sequence_count = driver_snapshot.resident_sequence_count;
	snapshot->resident_token_count = driver_snapshot.resident_token_count;
	snapshot->kv_token_capacity = driver_snapshot.kv_token_capacity;
	snapshot->device_memcpy_bytes_per_submit =
		driver_snapshot.device_memcpy_bytes_per_submit;
	snapshot->host_staging_bytes_per_submit =
		driver_snapshot.host_staging_bytes_per_submit;
	return(SPARK_STATUS_OK);
}

uint32_t SparkAdapterDestroyReady(SparkAdapterCommonState *common)
{
	SparkModelDriverRuntimeSnapshot snapshot;
	if ( SparkAdapterAvailableSubmissionCount(common->pending,
		common->pending_element_bytes,
		common->pipeline_slot_count) != common->pipeline_slot_count )
		return(0u);
	if ( common->driver.interface != 0 &&
		common->driver.interface->snapshot != 0 &&
		common->driver_instance != 0 && common->program != 0 )
	{
		memset(&snapshot,0,sizeof(snapshot));
		if ( common->driver.interface->snapshot(common->driver_instance,
			common->program->program_id,&snapshot) != SPARK_STATUS_OK ||
			snapshot.active_submission_count != 0u )
			return(0u);
	}
	return(1u);
}

void SparkAdapterTeardownDriver(SparkAdapterCommonState *common)
{
	if ( common->driver.interface != 0 &&
		common->driver.interface->destroy != 0 &&
		common->driver_instance != 0 )
		common->driver.interface->destroy(common->driver_instance);
	SparkUnloadModelDriver(&common->driver);
}

/* ---- env parsing --------------------------------------------------------- */

const char *SparkAdapterEnvText(const char *name)
{
	return(getenv(name));
}

uint32_t SparkAdapterEnvFlagDefaultOn(const char *name)
{
	const char *value = getenv(name);
	return(value == 0 || value[0] == '\0' || strcmp(value,"0") != 0 ?
		1u : 0u);
}

uint32_t SparkAdapterEnvFlagDefaultOffExactZero(const char *name)
{
	const char *value = getenv(name);
	/* Set AND nonempty AND not exactly "0" - the qwen36
	 * SPARK_QWEN36_SERVING_SPECULATE (:578-583) and SPARK_QWEN36_SPEC_AUDIT
	 * (:191-195) discipline. Empty reads OFF (unlike a bare strcmp guard,
	 * which would light every A/B ledger for free). */
	return(value != 0 && value[0] != '\0' && strcmp(value,"0") != 0 ?
		1u : 0u);
}

uint32_t SparkAdapterEnvFlagDefaultOffTruthy(const char *name)
{
	const char *value = getenv(name);
	return(value != 0 && value[0] != '\0' && value[0] != '0' ? 1u : 0u);
}

/* ---- tp_collective stanza parsing ----------------------------------------- */

static SparkStatus SparkAdapterTpValidateMembers(
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

SparkStatus SparkAdapterLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	uint32_t expected_rank_count,
	uint32_t validate_port_span,
	const SparkAdapterTpCollectivePolicy *policy,
	SparkAdapterTpCollectiveParsed *parsed)
{
	int32_t object,token,element,host_element;
	uint32_t count,index,port,rail;
	uint64_t collective_identifier;
	char *host,*relative_backend_path;
	SparkStatus status;
	if ( document == 0 || runtime_root == 0 || policy == 0 || parsed == 0 ||
		expected_rank_count == 0u ||
		expected_rank_count > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(parsed,0,sizeof(*parsed));
	parsed->topology.abi_version =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	parsed->topology.descriptor_bytes =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	object = SparkJsonFindObjectMember(document,root,"tp_collective");
	if ( object < 0 || !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkJsonFindObjectMember(document,object,"backend");
	if ( token < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkJsonStringEquals(document,token,"nccl") )
		parsed->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	else if ( SparkJsonStringEquals(document,token,"hidden_transport") )
		parsed->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	else
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkAdapterTpValidateMembers(document,object,parsed->backend_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	relative_backend_path = 0;
	token = SparkJsonFindObjectMember(document,object,"backend_module_path");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonCopyString(document,token,&relative_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_backend_path,
			parsed->backend_module_path,sizeof(parsed->backend_module_path));
	free(relative_backend_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	token = SparkJsonFindObjectMember(document,object,"collective_identifier");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonGetUInt64(document,token,&collective_identifier);
	if ( status != SPARK_STATUS_OK || collective_identifier == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	parsed->collective_identifier = collective_identifier;
	status = SparkJsonGetUInt32Member(document,object,"listen_port",&port);
	if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	parsed->listen_port = (uint16_t)port;
	status = SparkJsonGetUInt32Member(document,object,
		"connect_timeout_milli",&parsed->connect_timeout_milli);
	if ( status != SPARK_STATUS_OK || parsed->connect_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	status = SparkJsonGetUInt32Member(document,object,
		"operation_timeout_milli",&parsed->operation_timeout_milli);
	if ( status != SPARK_STATUS_OK || parsed->operation_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	token = SparkJsonFindObjectMember(document,object,"peer_hosts");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != expected_rank_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	parsed->topology.rank_count = count;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		host = 0;
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonCopyString(document,element,&host);
		if ( status == SPARK_STATUS_OK )
			status = SparkCopyString(
				parsed->topology.rank_hosts[index],
				SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
		free(host);
		if ( status != SPARK_STATUS_OK ||
			parsed->topology.rank_hosts[index][0] == '\0' )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	}
	token = SparkJsonFindObjectMember(document,object,"peer_ports");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != expected_rank_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&port);
		if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
		parsed->peer_ports[index] = (uint16_t)port;
	}
	if ( validate_port_span != 0u )
	{
		parsed->control_port_base = parsed->peer_ports[0];
		for (index=1u; index<expected_rank_count; index++)
			if ( parsed->peer_ports[0] > (uint16_t)(UINT16_MAX - index) ||
				parsed->peer_ports[index] !=
					(uint16_t)(parsed->peer_ports[0] + index) )
				return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( parsed->backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		token = SparkJsonFindObjectMember(document,object,"algorithms");
		if ( token < 0 ||
			!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
			return(SPARK_STATUS_SCHEMA_ERROR);
		count = SparkJsonGetArrayElementCount(document,token);
		port = 0u;
		for (index=0u; index<count; index++)
		{
			element = SparkJsonGetArrayElement(document,token,index);
			if ( SparkJsonStringEquals(document,element,"recursive_doubling") )
				port |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
			else if ( SparkJsonStringEquals(document,element,
					"counter_rotating_split_ring") )
				port |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
			else if ( SparkJsonStringEquals(document,element,"direct_all_to_all") )
				port |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
			else
				return(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( count != policy->adaptive_algorithm_count ||
			port != policy->adaptive_algorithm_mask )
			return(SPARK_STATUS_SCHEMA_ERROR);
		parsed->topology.algorithm_mask = port;
		status = SparkJsonGetUInt32Member(document,object,
			"direct_all_to_all_max_payload_bytes",
			&parsed->topology.direct_all_to_all_max_payload_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkJsonGetUInt32Member(document,object,
				"split_ring_min_payload_bytes",
				&parsed->topology.split_ring_min_payload_bytes);
		if ( status == SPARK_STATUS_OK )
		{
			if ( policy->require_zero_thresholds != 0u &&
				(parsed->topology.direct_all_to_all_max_payload_bytes != 0u ||
				 parsed->topology.split_ring_min_payload_bytes != 0u) )
				return(SPARK_STATUS_SCHEMA_ERROR);
			if ( policy->require_zero_thresholds == 0u &&
				(parsed->topology.direct_all_to_all_max_payload_bytes == 0u ||
				 parsed->topology.split_ring_min_payload_bytes == 0u ||
				 parsed->topology.direct_all_to_all_max_payload_bytes >=
					parsed->topology.split_ring_min_payload_bytes) )
				return(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( status == SPARK_STATUS_OK )
		{
			token = SparkJsonFindObjectMember(document,object,"rail_peer_hosts");
			if ( token < 0 ||
				!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
				SparkJsonGetArrayElementCount(document,token) !=
					SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
				return(SPARK_STATUS_SCHEMA_ERROR);
			parsed->topology.rail_count =
				SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT;
			for (rail=0u; rail<parsed->topology.rail_count; rail++)
			{
				element = SparkJsonGetArrayElement(document,token,rail);
				if ( element < 0 ||
					!SparkJsonTokenIsType(document,element,SPARK_JSON_TOKEN_ARRAY) )
					return(SPARK_STATUS_SCHEMA_ERROR);
				count = SparkJsonGetArrayElementCount(document,element);
				if ( count != expected_rank_count )
					return(SPARK_STATUS_SCHEMA_ERROR);
				for (index=0u; index<count; index++)
				{
					host_element = SparkJsonGetArrayElement(document,element,index);
					host = 0;
					status = host_element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
						SparkJsonCopyString(document,host_element,&host);
					if ( status == SPARK_STATUS_OK )
						status = SparkCopyString(
							parsed->topology.rail_rank_hosts[rail][index],
							SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
					free(host);
					if ( status != SPARK_STATUS_OK )
						return(status);
				}
			}
			token = SparkJsonFindObjectMember(document,object,"step_rail_indices");
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
					SparkJsonGetUInt32(document,element,&port);
				if ( status != SPARK_STATUS_OK || port >=
					SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
					return(SPARK_STATUS_SCHEMA_ERROR);
				parsed->topology.step_rail_indices[index] = port;
			}
		}
	}
	return(status);
}

/* ---- shared admission glue ------------------------------------------------- */

SparkStatus SparkAdapterAdmitFrame(
	const SparkLoadedModelDriver *driver,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame,
	uint32_t submit_on_apply)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	if ( driver == 0 || program == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = submission != 0 ?
		SparkAdmissionRequestFromSubmission(program->program_id,submission,
			0,0u,&request) :
		SparkAdmissionRequestFromFrame(program->program_id,frame,0,0u,
			&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkAdmissionEvaluateAndApply(driver->interface,driver_instance,
		&request,frame,&decision);
	if ( status == SPARK_STATUS_OK && submit_on_apply != 0u )
		status = program->submit(driver_instance,frame);
	return(status);
}

/* ---- module environment staging ------------------------------------------ */

SparkStatus SparkAdapterSetEnvironmentText(const char *name, const char *text)
{
	return(setenv(name,text,1) == 0 ? SPARK_STATUS_OK :
		SPARK_STATUS_INTERNAL_ERROR);
}

void SparkAdapterFormatEnvironmentUnsigned(
	char *buffer,
	size_t buffer_bytes,
	uint32_t value)
{
	(void)snprintf(buffer,buffer_bytes,"%u",value);
}
