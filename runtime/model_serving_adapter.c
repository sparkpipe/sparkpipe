#include "sparkpipe/spark_model_serving_adapter.h"

#include <dlfcn.h>
#include <string.h>

static uint32_t SparkModelServingAdapterTextIsPresent(const char *text)
{
	return(text != 0 && text[0] != '\0');
}

static uint32_t SparkModelServingAdapterSha256IsValid(const char *sha256)
{
	uint32_t index;
	if ( sha256 == 0 )
		return(0u);
	for (index=0u; index<SPARK_MODEL_SERVING_ADAPTER_ARTIFACT_SHA256_LENGTH; index++)
		if ( !((sha256[index] >= '0' && sha256[index] <= '9') || (sha256[index] >= 'a' && sha256[index] <= 'f')) )
			return(0u);
	return(sha256[SPARK_MODEL_SERVING_ADAPTER_ARTIFACT_SHA256_LENGTH] == '\0');
}

SparkStatus SparkModelServingAdapterValidateDescriptor(
	const SparkModelServingAdapterDescriptor *descriptor)
{
	uint32_t index,total;
	if ( descriptor == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( descriptor->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || descriptor->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (descriptor->capability_flags & ~SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES) != 0u || descriptor->stage_count == 0u || descriptor->stage_count > SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT || descriptor->layer_count == 0u || descriptor->layer_count > SPARK_MODEL_SERVING_ADAPTER_MAX_LAYER_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( descriptor->boundary_format != SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16 || descriptor->boundary_element_count == 0u || descriptor->boundary_element_bytes != sizeof(uint16_t) || descriptor->max_inflight_submission_count == 0u || descriptor->max_inflight_submission_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INFLIGHT_SUBMISSION_COUNT || descriptor->max_active_sequence_count == 0u || descriptor->max_active_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || descriptor->max_input_row_count < descriptor->max_active_sequence_count || descriptor->max_input_row_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT || descriptor->max_resident_sequence_count < descriptor->max_active_sequence_count || descriptor->max_resident_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT || descriptor->max_output_token_count == 0u || descriptor->max_output_token_count > SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION) != 0u) != (descriptor->max_speculative_token_count != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( descriptor->resident_sequence_slot_reuse > SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV) != 0u) != (descriptor->resident_sequence_slot_reuse != SPARK_MODEL_SERVING_SLOT_REUSE_NONE) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE && (descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkModelServingAdapterTextIsPresent(descriptor->adapter_id) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->model_id) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->model_revision) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->driver_program_name) == 0u || SparkModelServingAdapterSha256IsValid(descriptor->artifact_sha256) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<4u; index++)
		if ( descriptor->reserved[index] != 0u )
			return(SPARK_STATUS_ABI_MISMATCH);
	total = 0u;
	for (index=0u; index<descriptor->stage_count; index++)
	{
		if ( descriptor->stage_layer_counts[index] == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		total += descriptor->stage_layer_counts[index];
	}
	for (; index<SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT; index++)
		if ( descriptor->stage_layer_counts[index] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	return(total == descriptor->layer_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkModelServingAdapterValidateRuntimeLimits(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits)
{
	SparkStatus status;
	uint32_t index;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || runtime_limits == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( runtime_limits->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || runtime_limits->descriptor_bytes != SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( runtime_limits->max_inflight_submission_count == 0u || runtime_limits->max_inflight_submission_count > descriptor->max_inflight_submission_count || runtime_limits->max_active_sequence_count == 0u || runtime_limits->max_active_sequence_count > descriptor->max_active_sequence_count || runtime_limits->max_input_row_count < runtime_limits->max_active_sequence_count || runtime_limits->max_input_row_count > descriptor->max_input_row_count || runtime_limits->resident_sequence_capacity < runtime_limits->max_active_sequence_count || runtime_limits->resident_sequence_capacity > descriptor->max_resident_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<4u; index++)
		if ( runtime_limits->reserved[index] != 0u )
			return(SPARK_STATUS_ABI_MISMATCH);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateInterface(
	const SparkModelServingAdapterInterface *adapter_interface,
	uint32_t required_capability_flags)
{
	SparkStatus status;
	if ( adapter_interface == 0 || (required_capability_flags & ~SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( adapter_interface->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || adapter_interface->interface_bytes != SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateDescriptor(adapter_interface->descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (adapter_interface->descriptor->capability_flags & required_capability_flags) != required_capability_flags || adapter_interface->initialize == 0 || adapter_interface->destroy == 0 || adapter_interface->validate_submission == 0 || adapter_interface->submit == 0 || adapter_interface->progress == 0 || adapter_interface->quiesce == 0 || adapter_interface->snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (adapter_interface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) != 0u && adapter_interface->prefetch == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (adapter_interface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESET) != 0u && adapter_interface->reset == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelServingAdapterValidateRows(
	const SparkModelServingSubmission *submission,
	uint32_t require_live_rows,
	uint32_t resident_sequence_capacity)
{
	uint8_t seen[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t seen_slots[SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT];
	uint32_t lane,row,slot;
	memset(seen,0,sizeof(seen));
	memset(seen_slots,0,sizeof(seen_slots));
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		if ( submission->lanes[lane].reserved0 != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( lane >= submission->active_sequence_count )
		{
			if ( submission->lanes[lane].request_id != 0u || submission->lanes[lane].request_generation != 0u || submission->lanes[lane].step_generation != 0u || submission->lanes[lane].sequence_id != 0u || submission->lanes[lane].sequence_position != 0u || submission->lanes[lane].resident_sequence_slot != SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT || submission->lanes[lane].context_token_count != 0u || submission->lanes[lane].input_token_id != 0u )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			continue;
		}
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( submission->lanes[lane].request_id == 0u || submission->lanes[lane].request_generation == 0u || submission->lanes[lane].step_generation == 0u || submission->lanes[lane].sequence_id == 0u || slot >= resident_sequence_capacity || seen_slots[slot] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen_slots[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_sequence_ids[row] != submission->lanes[lane].sequence_id )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
	}
	for (lane=0u; require_live_rows != 0u && lane<submission->active_sequence_count; lane++)
		if ( seen[lane] == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	uint32_t required_capability;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( submission->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || submission->descriptor_bytes != SPARK_MODEL_SERVING_SUBMISSION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( submission->flags != 0u || submission->submission_id == 0u || submission->control_generation == 0u || submission->transaction_id == 0u || submission->dispatch_generation == 0u || submission->request_generation == 0u || submission->step_generation == 0u || submission->work_kind < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || submission->work_kind > SPARK_MODEL_SERVING_WORK_KIND_RELEASE || submission->lane_count == 0u || submission->lane_count > descriptor->max_active_sequence_count || submission->active_sequence_count == 0u || submission->active_sequence_count > submission->lane_count || submission->lanes == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	required_capability = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL : submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE : SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	if ( (descriptor->capability_flags & required_capability) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( submission->model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES || (submission->model_extension_bytes != 0u) != (submission->model_extension != 0) || (submission->model_extension_bytes != 0u) != (submission->model_extension_kind != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (submission->hidden_input_address != 0) != (submission->hidden_input_bytes != 0u) || (submission->hidden_output_address != 0) != (submission->hidden_output_bytes != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		if ( submission->row_count != 0u || submission->token_count != 0u || submission->new_token_count != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		return(SparkModelServingAdapterValidateRows(submission,0u,descriptor->max_resident_sequence_count));
	}
	if ( submission->row_count == 0u || submission->token_count != submission->row_count || submission->new_token_count != submission->row_count || submission->token_count > descriptor->max_input_row_count || submission->token_ids == 0 || submission->row_lane_indices == 0 || submission->row_positions == 0 || submission->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkModelServingAdapterValidateRows(submission,1u,descriptor->max_resident_sequence_count));
}

SparkStatus SparkModelServingAdapterValidateRuntimeSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	uint32_t lane;
	status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,runtime_limits);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateSubmission(descriptor,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->lane_count > runtime_limits->max_active_sequence_count || submission->row_count > runtime_limits->max_input_row_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( submission->lanes[lane].resident_sequence_slot >= runtime_limits->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SparkModelServingAdapterValidateRows(submission,submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE,runtime_limits->resident_sequence_capacity));
}

SparkStatus SparkModelServingAdapterValidateCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	uint32_t has_tokens,has_extension;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || completion == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( completion->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || completion->descriptor_bytes != SPARK_MODEL_SERVING_COMPLETION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( completion->submission_id == 0u || completion->control_generation == 0u || completion->transaction_id == 0u || completion->dispatch_generation == 0u || completion->request_generation == 0u || completion->step_generation == 0u || completion->status > SPARK_STATUS_UNSUPPORTED || (completion->completion_flags & ~SPARK_MODEL_SERVING_COMPLETION_KNOWN_FLAGS) != 0u || completion->token_count > descriptor->max_output_token_count || completion->model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_tokens = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) != 0u;
	has_extension = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION) != 0u;
	if ( has_tokens != (completion->token_count != 0u) || has_extension != (completion->model_extension_bytes != 0u) || (completion->model_extension_bytes == 0u && completion->model_extension_kind != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( completion->status != SPARK_STATUS_OK && (completion->completion_flags != 0u || completion->accepted_token_count != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateCompletionResidency(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	status = SparkModelServingAdapterValidateCompletion(descriptor,completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( expected_residency == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(memcmp(expected_residency,&completion->residency,sizeof(*expected_residency)) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkModelServingAdapterValidateStageCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	uint32_t stage_index,
	uint32_t work_kind,
	uint32_t active_sequence_count,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	uint32_t final_stage,has_tokens;
	status = SparkModelServingAdapterValidateCompletionResidency(descriptor,expected_residency,completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( stage_index >= descriptor->stage_count || work_kind < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || work_kind > SPARK_MODEL_SERVING_WORK_KIND_RELEASE || active_sequence_count == 0u || active_sequence_count > descriptor->max_active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_tokens = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) != 0u;
	if ( completion->status != SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(has_tokens == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	final_stage = descriptor->stage_count - 1u;
	if ( stage_index != final_stage )
		return(has_tokens == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	return(has_tokens != 0u && completion->token_count == active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkModelServingAdapterLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkModelServingAdapterDynamicLibrary *library)
{
	SparkModelServingAdapterGetInterfaceFunction get_interface;
	const SparkModelServingAdapterInterface *adapter_interface;
	void *dynamic_library;
	SparkStatus status;
	if ( shared_object_path == 0 || library == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if ( dynamic_library == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	get_interface = (SparkModelServingAdapterGetInterfaceFunction)dlsym(dynamic_library,SPARK_MODEL_SERVING_ADAPTER_INTERFACE_SYMBOL);
	if ( get_interface == 0 )
	{
		dlclose(dynamic_library);
		return(SPARK_STATUS_NOT_FOUND);
	}
	adapter_interface = get_interface();
	status = SparkModelServingAdapterValidateInterface(adapter_interface,required_capability_flags);
	if ( status != SPARK_STATUS_OK )
	{
		dlclose(dynamic_library);
		return(status);
	}
	library->dynamic_library = dynamic_library;
	library->adapter_interface = *adapter_interface;
	return(SPARK_STATUS_OK);
}

void SparkModelServingAdapterUnloadInterface(
	SparkModelServingAdapterDynamicLibrary *library)
{
	if ( library == 0 )
		return;
	if ( library->dynamic_library != 0 )
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
