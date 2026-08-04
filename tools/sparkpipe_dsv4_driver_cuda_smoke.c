#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
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
	uint16_t *hidden_input_device;
	uint32_t receive_count;
	uint32_t send_count;
	uint32_t nonzero_hidden_count;
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4DriverCudaSmokeState;

static SparkDsv4DriverCudaSmokeState *SparkDsv4DriverCudaSmokeCurrent;

static SparkStatus SparkDsv4DriverCudaSmokeReadUnsigned(
	const char *name,
	uint32_t minimum,
	uint32_t maximum,
	uint32_t fallback,
	uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	if ( value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	text = getenv(name);
	if ( text == 0 || text[0] == '\0' )
	{
		*value = fallback;
		return(SPARK_STATUS_OK);
	}
	errno = 0;
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( errno != 0 || end == text || end[0] != '\0' ||
		parsed < (unsigned long)minimum || parsed > (unsigned long)maximum )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*value = (uint32_t)parsed;
	return(SPARK_STATUS_OK);
}

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

static SparkStatus SparkDsv4DriverCudaSmokeReceive(
	SparkHiddenTransportSession *transport_session,
	SparkHiddenTransportPacket *packet)
{
	cudaError_t error;
	if ( transport_session == 0 || packet == 0 || packet->hidden_bf16 == 0 ||
		packet->active_sequence_count != 1u ||
		packet->hidden_dimension != SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS ||
		packet->bytes_per_sequence !=
			SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
				SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemset(
		(void *)packet->hidden_bf16,0,packet->bytes_per_sequence);
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"dsv4_driver_smoke cuda_input=%s\n",cudaGetErrorString(error));
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	SparkDsv4DriverCudaSmokeCurrent->receive_count++;
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
	uint32_t stage_index;
	uint32_t stage_count;
	uint64_t hidden_input_bytes;
	uint64_t sequence_id = 1u;
	uint64_t position = 0u;
	cudaError_t input_error;
	char error_buffer[1024];
	SparkStatus status;
	int result = 1;

	memset(&state,0,sizeof(state));
	SparkDsv4DriverCudaSmokeCurrent = &state;
	status = SparkDsv4DriverCudaSmokeReadUnsigned(
		"SPARK_DSV4_STAGE_INDEX",0u,
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,0u,
		&stage_index);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_driver_smoke stage_index=invalid\n");
		goto done;
	}
	status = SparkDsv4DriverCudaSmokeReadUnsigned(
		"SPARK_DSV4_STAGE_COUNT",1u,
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,13u,
		&stage_count);
	if ( status != SPARK_STATUS_OK || stage_index >= stage_count )
	{
		fprintf(stderr,"dsv4_driver_smoke stage_count=invalid\n");
		goto done;
	}
	hidden_input_bytes = (uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( stage_index != 0u )
	{
		input_error = cudaMalloc(
			(void **)&state.hidden_input_device,hidden_input_bytes);
		if ( input_error != cudaSuccess )
		{
			fprintf(stderr,"dsv4_driver_smoke cuda_input_alloc=%s\n",cudaGetErrorString(input_error));
			status = SPARK_STATUS_INTERNAL_ERROR;
			goto done;
		}
	}
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
	runner_configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION;
	if ( stage_index != 0u )
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
	if ( stage_index + 1u < stage_count )
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	runner_configuration.stage_index = stage_index;
	runner_configuration.stage_count = stage_count;
	runner_configuration.max_active_sequence_count = 1u;
	runner_configuration.driver_interface = driver.interface;
	runner_configuration.driver_instance = driver_instance;
	runner_configuration.program = program;
	runner_configuration.hidden_input_post_receive_function =
		stage_index != 0u ? SparkDsv4DriverCudaSmokeReceive : 0;
	runner_configuration.hidden_output_send_function =
		stage_index + 1u < stage_count ? SparkDsv4DriverCudaSmokeSend : 0;
	runner_configuration.hidden_input_bf16 = state.hidden_input_device;
	runner_configuration.hidden_input_bytes = hidden_input_bytes;
	status = SparkDsv4StageRunnerInitialize(&runner,&runner_configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_driver_smoke runner_initialize=%s\n",SparkStatusToString(status));
		goto destroy;
	}
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES;
	dispatch.flags = stage_index == 0u ?
		SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL : 0u;
	dispatch.request_id = 91u;
	dispatch.sequence_id = sequence_id;
	dispatch.active_sequence_count = 1u;
	dispatch.new_token_count = 1u;
	dispatch.row_count = 1u;
	dispatch.lane_count = 1u;
	dispatch.token_ids = stage_index == 0u ? &token_id : 0;
	dispatch.row_lane_indices = &lane_index;
	dispatch.row_positions = &position;
	dispatch.row_sequence_ids = &sequence_id;
	dispatch.hidden_input_transport_session = stage_index != 0u ?
		(SparkHiddenTransportSession *)(uintptr_t)1u : 0;
	dispatch.hidden_output_transport_session = stage_index + 1u < stage_count ?
		(SparkHiddenTransportSession *)(uintptr_t)1u : 0;
	dispatch.completion_function = SparkDsv4DriverCudaSmokeCompletion;
	dispatch.completion_context = &state;
	status = SparkDsv4StageRunnerSubmit(&runner,&dispatch);
	if ( status != SPARK_STATUS_OK || state.completion_count != 1u ||
		state.completion.status != SPARK_STATUS_OK ||
		state.send_count != (stage_index + 1u < stage_count ? 1u : 0u) ||
		state.receive_count != (stage_index != 0u ? 1u : 0u) ||
		(stage_index == 0u && state.nonzero_hidden_count == 0u) ||
		runner.stats.submitted_count != 1u )
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
	if ( state.hidden_input_device != 0 )
		cudaFree(state.hidden_input_device);
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
