#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_runner.h"

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
	SparkDsv4StageRunner runner;
	SparkDsv4StageRunnerConfiguration runner_configuration;
	SparkDsv4StageRunnerDispatch dispatch;
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
	memset(&runner_configuration,0,sizeof(runner_configuration));
	runner_configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	runner_configuration.descriptor_bytes =
		SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
	runner_configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION |
		SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	runner_configuration.stage_index = 0u;
	runner_configuration.stage_count = 13u;
	runner_configuration.max_active_sequence_count = 1u;
	runner_configuration.driver_interface = driver.interface;
	runner_configuration.driver_instance = driver_instance;
	runner_configuration.program = program;
	runner_configuration.hidden_output_send_function =
		SparkDsv4DriverCudaSmokeSend;
	status = SparkDsv4StageRunnerInitialize(&runner,&runner_configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_driver_smoke runner_initialize=%s\n",SparkStatusToString(status));
		goto destroy;
	}
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES;
	dispatch.flags = SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL;
	dispatch.request_id = 91u;
	dispatch.sequence_id = sequence_id;
	dispatch.active_sequence_count = 1u;
	dispatch.new_token_count = 1u;
	dispatch.row_count = 1u;
	dispatch.lane_count = 1u;
	dispatch.token_ids = &token_id;
	dispatch.row_lane_indices = &lane_index;
	dispatch.row_positions = &position;
	dispatch.row_sequence_ids = &sequence_id;
	dispatch.hidden_output_transport_session =
		(SparkHiddenTransportSession *)(uintptr_t)1u;
	dispatch.completion_function = SparkDsv4DriverCudaSmokeCompletion;
	dispatch.completion_context = &state;
	status = SparkDsv4StageRunnerSubmit(&runner,&dispatch);
	if ( status != SPARK_STATUS_OK || state.completion_count != 1u ||
		state.completion.status != SPARK_STATUS_OK || state.send_count != 1u ||
		state.nonzero_hidden_count == 0u || runner.stats.submitted_count != 1u )
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
