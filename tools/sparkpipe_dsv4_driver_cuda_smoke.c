#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

typedef struct SparkDsv4DriverCudaSmokeState
{
	uint16_t hidden_bf16[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint32_t send_count;
	uint32_t nonzero_hidden_count;
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4DriverCudaSmokeState;

static SparkDsv4DriverCudaSmokeState *SparkDsv4DriverCudaSmokeCurrent;

static void SparkDsv4DriverCudaSmokeCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkDsv4DriverCudaSmokeState *state = completion_context;
	if ( state == 0 || completion == 0 )
		return;
	state->completion = *completion;
	state->completion_count++;
}

static SparkStatus SparkDsv4DriverCudaSmokeSend(
	SparkHiddenTransportSession *transport_session,
	const SparkHiddenTransportPacket *packet)
{
	SparkDsv4DriverCudaSmokeState *state = SparkDsv4DriverCudaSmokeCurrent;
	uint32_t element;
	cudaError_t error;
	(void)transport_session;
	if ( state == 0 || packet == 0 || packet->hidden_bf16 == 0 ||
		packet->active_sequence_count != 1u ||
		packet->hidden_dimension != SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS ||
		packet->bytes_per_sequence !=
			SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
				SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpy(
		state->hidden_bf16,
		packet->hidden_bf16,
		packet->bytes_per_sequence,
		cudaMemcpyDeviceToHost);
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"dsv4_driver_smoke cuda_copy=%s\n",cudaGetErrorString(error));
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	state->send_count++;
	for (element = 0u; element < SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
	{
		if ( state->hidden_bf16[element] != 0u )
			state->nonzero_hidden_count++;
	}
	return(SPARK_STATUS_OK);
}

static int SparkDsv4DriverCudaSmokeRun(
	const char *driver_path,
	const char *expected_target)
{
	SparkDsv4DriverCudaSmokeState state;
	SparkLoadedModelDriver driver;
	SparkModelDriverCreateRequest create_request;
	void *driver_instance = 0;
	SparkModelDriverAdmissionRequest admission_request;
	SparkModelDriverAdmissionDecision admission_decision;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkDsv4PrefillBatchView prefill;
	SparkDsv4ResidentDecodeStageFrameContext context;
	SparkModelDriverRuntimeSnapshot snapshot;
	const SparkModelDriverProgramDescriptor *program;
	uint32_t token_id = 10397u;
	uint32_t lane_index = 0u;
	uint64_t sequence_id = 1u;
	uint64_t position = 0u;
	char error_buffer[1024];
	SparkStatus status;
	int result = 1;

	memset(&state,0,sizeof(state));
	SparkDsv4DriverCudaSmokeCurrent = &state;
	SparkLoadedModelDriverReset(&driver);
	status = SparkLoadModelDriver(driver_path,expected_target,&driver,error_buffer,sizeof(error_buffer));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_driver_smoke load=%s: %s\n",SparkStatusToString(status),error_buffer);
		goto done;
	}
	program = SparkFindLoadedModelDriverProgram(&driver,"resident_decode");
	if ( program == 0 )
	{
		fprintf(stderr,"dsv4_driver_smoke program_missing\n");
		goto unload;
	}
	memset(&create_request,0,sizeof(create_request));
	create_request.node_id = "spark0";
	create_request.node_target = expected_target;
	create_request.completion_function = SparkDsv4DriverCudaSmokeCompletion;
	create_request.completion_context = &state;
	status = driver.interface->create(&create_request,&driver_instance);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_driver_smoke create=%s\n",SparkStatusToString(status));
		goto unload;
	}
	memset(&admission_request,0,sizeof(admission_request));
	memset(&admission_decision,0,sizeof(admission_decision));
	admission_request.descriptor_bytes = sizeof(admission_request);
	admission_request.program_id = program->program_id;
	admission_request.request_id = 91u;
	admission_request.sequence_id = sequence_id;
	admission_request.active_slot_count = 1u;
	admission_request.new_token_count = 1u;
	admission_request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	status = driver.interface->admit(driver_instance,&admission_request,&admission_decision);
	if ( status != SPARK_STATUS_OK || admission_decision.accepted == 0u )
	{
		fprintf(stderr,"dsv4_driver_smoke admit=%s accepted=%u\n",SparkStatusToString(status),admission_decision.accepted);
		goto destroy;
	}
	memset(&prefill,0,sizeof(prefill));
	prefill.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION;
	prefill.descriptor_bytes = sizeof(prefill);
	prefill.row_count = 1u;
	prefill.lane_count = 1u;
	prefill.token_ids = &token_id;
	prefill.row_lane_indices = &lane_index;
	prefill.row_positions = &position;
	prefill.row_sequence_ids = &sequence_id;
	memset(&context,0,sizeof(context));
	context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context.descriptor_bytes = sizeof(context);
	context.flags = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
	context.prefill_batch = &prefill;
	context.hidden_output_transport_session =
		(SparkHiddenTransportSession *)(uintptr_t)1u;
	context.hidden_output_send_function = SparkDsv4DriverCudaSmokeSend;
	memset(&buffer,0,sizeof(buffer));
	buffer.slot = 0u;
	buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	buffer.address = &token_id;
	buffer.bytes = sizeof(token_id);
	memset(&frame,0,sizeof(frame));
	frame.request_id = 91u;
	frame.sequence_id = sequence_id;
	frame.sequence_position = position;
	frame.active_slot_count = 1u;
	frame.new_token_count = 1u;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame.driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame.program_id = program->program_id;
	frame.buffers = &buffer;
	frame.buffer_count = 1u;
	frame.user_context = &context;
	frame.completion_function = SparkDsv4DriverCudaSmokeCompletion;
	frame.completion_context = &state;
	status = program->submit(driver_instance,&frame);
	if ( status != SPARK_STATUS_OK || state.completion_count != 1u ||
		state.completion.status != SPARK_STATUS_OK || state.send_count != 1u ||
		state.nonzero_hidden_count == 0u )
	{
		fprintf(stderr,"dsv4_driver_smoke execute=%s completion=%u status=%s sends=%u nonzero=%u\n",SparkStatusToString(status),state.completion_count,SparkStatusToString(state.completion.status),state.send_count,state.nonzero_hidden_count);
		goto destroy;
	}
	status = driver.interface->snapshot(driver_instance,program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK || snapshot.submitted_count != 1u ||
		snapshot.completed_count != 1u || snapshot.active_submission_count != 0u )
	{
		fprintf(stderr,"dsv4_driver_smoke snapshot=%s submitted=%llu completed=%llu active=%u\n",SparkStatusToString(status),(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count,snapshot.active_submission_count);
		goto destroy;
	}
	printf("dsv4_driver_cuda_smoke PASS sends=%u nonzero_hidden=%u submitted=%llu completed=%llu\n",state.send_count,state.nonzero_hidden_count,(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count);
	result = 0;
destroy:
	driver.interface->destroy(driver_instance);
unload:
	SparkUnloadModelDriver(&driver);
done:
	SparkDsv4DriverCudaSmokeCurrent = 0;
	return(result);
}

int main(int argument_count,char **arguments)
{
	if ( argument_count != 3 )
	{
		fprintf(stderr,"usage: %s DRIVER_SO EXPECTED_TARGET\n",arguments[0]);
		return(2);
	}
	return(SparkDsv4DriverCudaSmokeRun(arguments[1],arguments[2]));
}
