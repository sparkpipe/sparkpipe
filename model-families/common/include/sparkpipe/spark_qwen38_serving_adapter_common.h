#ifndef SPARKPIPE_SPARK_QWEN38_SERVING_ADAPTER_COMMON_H
#define SPARKPIPE_SPARK_QWEN38_SERVING_ADAPTER_COMMON_H

#include <stddef.h>

#include "sparkpipe/spark_serving_adapter_template.h"

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReleaseLane)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	uint32_t slot);

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingOrphanDriverCompletion)(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	(void)driver_completion;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDriverCompletion)(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending;
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	uint32_t matches;
	pending = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->common.active == 0u || driver_completion == 0 )
		return;
	matches = driver_completion->request_id == pending->common.request_id && driver_completion->sequence_id == pending->frame_sequence_id && driver_completion->sequence_position == pending->frame_sequence_position && driver_completion->program_id == state->program->program_id;
	if ( matches == 0u )
	{
		state->orphan_completion_count++;
		pending->frame_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	pending->frame_status = (SparkStatus)driver_completion->status;
	pending->residency = driver_completion->residency;
	pending->accepted_token_count += driver_completion->accepted_token_count;
	pending->queue_delay_ns += driver_completion->queue_delay_ns;
	pending->service_time_ns += driver_completion->service_time_ns;
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDriverWake)(void *wake_context)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAvailableSubmissionCount)(
	const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].common.active == 0u ? 1u : 0u;
	return(available);
}

static SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReservePending)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending;
	uint32_t lane;
	pending = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *)
		SparkServingAdapterTemplateReservePending(state->pending,
			sizeof(*pending),
			(uint32_t)offsetof(SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending),common),
			state->pipeline_slot_count,
			(uint32_t)offsetof(SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending),last_row_by_lane),
			submission);
	if ( pending == 0 )
		return(0);
	pending->owner = state;
	pending->common.active = 1u;
	pending->frame_status = SPARK_STATUS_OK;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		pending->resident_slots[lane] = submission->lanes[lane].resident_sequence_slot;
	return(pending);
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDropSubmission)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReleaseLane)(state,submission->lanes[lane].resident_sequence_slot);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAdmit)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	(void)submission;
	status = SparkAdmissionRequestFromFrame(
		state->program->program_id,frame,0,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluateAndApply(
		state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPostReceive)(
	SparkHiddenTransportSession *transport_session,
	SparkHiddenTransportPacket *packet)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingTransportShim) *shim;
	const void *source;
	uint32_t row;
	shim = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingTransportShim) *)transport_session;
	if ( shim == 0 || packet == 0 || shim->input_base == 0 || shim->input_rows == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	source = shim->input_base;
	if ( shim->input_row_map != 0 )
	{
		for (row=0u; row<shim->input_rows; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->input_scratch + ((uint64_t)row * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES)),(const uint8_t *)shim->input_base + ((uint64_t)shim->input_row_map[row] * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES)),SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES),cudaMemcpyDeviceToDevice,(cudaStream_t)shim->execution_stream) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
		if ( cudaStreamSynchronize((cudaStream_t)shim->execution_stream) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
		source = shim->input_scratch;
	}
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = shim->input_rows;
	packet->hidden_dimension = SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_DIMENSION);
	packet->bytes_per_sequence = SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES);
	packet->hidden_bf16 = source;
	packet->cuda_stream = shim->execution_stream;
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSend)(
	SparkHiddenTransportSession *transport_session,
	const SparkHiddenTransportPacket *packet)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingTransportShim) *shim;
	uint32_t row;
	shim = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingTransportShim) *)transport_session;
	if ( shim == 0 || packet == 0 || packet->hidden_bf16 == 0 || shim->output_base == 0 || packet->active_sequence_count == 0u || packet->hidden_dimension != SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_DIMENSION) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( shim->output_row_map != 0 )
	{
		for (row=0u; row<packet->active_sequence_count; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->output_base + ((uint64_t)shim->output_row_map[row] * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES)),(const uint8_t *)packet->hidden_bf16 + ((uint64_t)row * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES)),SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES),cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
	}
	else if ( cudaMemcpyAsync(shim->output_base,packet->hidden_bf16,(uint64_t)packet->active_sequence_count * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES),cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	return(cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingProgress)(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingQuiesce)(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAvailableSubmissionCount)(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSnapshot)(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAvailableSubmissionCount)(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAvailableSubmissionCount)(state);
	snapshot->submitted_count = driver_snapshot.submitted_count;
	snapshot->completed_count = driver_snapshot.completed_count;
	snapshot->rejected_count = driver_snapshot.rejected_count + state->orphan_completion_count;
	snapshot->resident_sequence_count = driver_snapshot.resident_sequence_count;
	snapshot->resident_token_count = driver_snapshot.resident_token_count;
	snapshot->kv_token_capacity = driver_snapshot.kv_token_capacity;
	snapshot->device_memcpy_bytes_per_submit = driver_snapshot.device_memcpy_bytes_per_submit;
	snapshot->host_staging_bytes_per_submit = driver_snapshot.host_staging_bytes_per_submit;
	return(SPARK_STATUS_OK);
}

static uint32_t SPARK_QWEN38_SERVING_ADAPTER_FN(ServingStageAttentionLayers)(uint32_t first_layer, uint32_t layer_count)
{
	uint32_t layer,count;
	count = 0u;
	for (layer=first_layer; layer<first_layer+layer_count; layer++)
		count += SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_LAYER_IS_GDN)(layer) == 0u ? 1u : 0u;
	return(count);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateConfiguration)(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDescriptor),&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_COUNT) || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_PROGRAM_NAME)) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}
#endif
