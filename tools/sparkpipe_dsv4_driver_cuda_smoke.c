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
#include "sparkpipe/spark_model_driver_support.h"

#define SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES 2u
#define SPARK_DSV4_DRIVER_CUDA_SMOKE_PROMPT_TOKENS 9u
#define SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS 18u

typedef struct SparkDsv4DriverCudaSmokeState
{
	cudaStream_t execution_stream;
	uint16_t hidden_bf16[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint16_t *hidden_input_device;
	uint16_t *hidden_output_device;
	uint32_t receive_count;
	uint32_t send_count;
	uint32_t nonzero_hidden_count;
	uint32_t mismatch_count;
	uint32_t output_token_ids[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS];
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4DriverCudaSmokeState;

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
	SparkDsv4ResidentDecodeStageNodeContext node_context;
	SparkModelDriverRuntimeSnapshot snapshot;
	const SparkModelDriverProgramDescriptor *program;
	const char *stage_pack_path;
	const uint32_t prompt_token_ids[SPARK_DSV4_DRIVER_CUDA_SMOKE_PROMPT_TOKENS] = {0u,128803u,19905u,418u,9045u,28u,20370u,128804u,128822u};
	uint32_t token_ids[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS],lane_indices[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS];
	uint32_t emit_rows[SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES] = {16u,17u};
	uint32_t emit_lanes[SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES] = {0u,1u};
	uint64_t positions[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS],sequence_ids[SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS];
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t element,lane,row,wave;
	uint64_t hidden_input_bytes;
	cudaError_t input_error;
	char error_buffer[1024];
	SparkStatus status;
	int result = 1;

	memset(&state,0,sizeof(state));
	for (row=0u; row<SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS; row++)
	{
		lane = row % SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
		wave = row / SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
		token_ids[row] = prompt_token_ids[wave];
		lane_indices[row] = lane;
		positions[row] = wave;
		sequence_ids[row] = lane + 1u;
		state.output_token_ids[row] = UINT32_MAX;
	}
	input_error = cudaStreamCreateWithFlags(
		&state.execution_stream,cudaStreamNonBlocking);
	if ( input_error != cudaSuccess )
	{
		fprintf(stderr,"dsv4_driver_smoke cuda_stream=%s\n",cudaGetErrorString(input_error));
		goto done;
	}
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
	status = SparkDsv4DriverCudaSmokeReadUnsigned("SPARK_DSV4_STAGE_FIRST_LAYER",0u,SPARK_DSV4_MODEL_LAYER_COUNT - 1u,0u,&first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4DriverCudaSmokeReadUnsigned("SPARK_DSV4_STAGE_LAYER_COUNT",1u,SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT,&layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4DriverCudaSmokeReadUnsigned("SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES",SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES,&max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4DriverCudaSmokeReadUnsigned("SPARK_DSV4_STAGE_PIPELINE_SLOTS",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,2u,&pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4DriverCudaSmokeReadUnsigned("SPARK_DSV4_STAGE_MAX_SEQ",SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO,SPARK_DSV4_MODEL_MAX_POSITIONS,4096u,&max_sequence_positions);
	stage_pack_path = getenv("SPARK_DSV4_STAGE_PACK_PATH");
	if ( status != SPARK_STATUS_OK || first_layer_index + layer_count > SPARK_DSV4_MODEL_LAYER_COUNT || stage_pack_path == 0 || stage_pack_path[0] == '\0' )
	{
		fprintf(stderr,"dsv4_driver_smoke node_context=invalid\n");
		goto done;
	}
	hidden_input_bytes = (uint64_t)SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( stage_index != 0u )
	{
		for (element=0u; element<SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
			state.hidden_bf16[element] = UINT16_C(0x3f80);
		input_error = cudaMalloc(
			(void **)&state.hidden_input_device,hidden_input_bytes);
		if ( input_error != cudaSuccess )
		{
			fprintf(stderr,"dsv4_driver_smoke cuda_input_alloc=%s\n",cudaGetErrorString(input_error));
			status = SPARK_STATUS_INTERNAL_ERROR;
			goto done;
		}
		input_error = cudaMemcpy(state.hidden_input_device,state.hidden_bf16,
			(size_t)hidden_input_bytes,cudaMemcpyHostToDevice);
		if ( input_error != cudaSuccess )
		{
			fprintf(stderr,"dsv4_driver_smoke cuda_input_copy=%s\n",cudaGetErrorString(input_error));
			status = SPARK_STATUS_INTERNAL_ERROR;
			goto done;
		}
		state.receive_count = 1u;
	}
	if ( stage_index + 1u < stage_count )
	{
		input_error = cudaMalloc((void **)&state.hidden_output_device,hidden_input_bytes);
		if ( input_error != cudaSuccess )
		{
			fprintf(stderr,"dsv4_driver_smoke cuda_output_alloc=%s\n",cudaGetErrorString(input_error));
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
	SparkModelDriverInitializeCreateRequest(&create_request);
	memset(&node_context,0,sizeof(node_context));
	node_context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node_context.descriptor_bytes = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	node_context.flags = 0u;
	node_context.stage_count = stage_count;
	node_context.stage_index = stage_index;
	node_context.first_layer_index = first_layer_index;
	node_context.layer_count = layer_count;
	node_context.resident_sequence_capacity = max_active_sequence_count;
	node_context.pipeline_slot_count = pipeline_slot_count;
	node_context.max_sequence_positions = max_sequence_positions;
	node_context.linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	node_context.expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	node_context.kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
	node_context.stage_pack_path = stage_pack_path;
	create_request.node_id = "spark0";
	create_request.node_target = expected_target;
	create_request.node_context = &node_context;
	create_request.execution_stream = state.execution_stream;
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
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY;
	if ( stage_index + 1u < stage_count )
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY;
	runner_configuration.stage_index = stage_index;
	runner_configuration.stage_count = stage_count;
	runner_configuration.max_active_sequence_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
	runner_configuration.max_input_row_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS;
	runner_configuration.driver_interface = driver.interface;
	runner_configuration.driver_instance = driver_instance;
	runner_configuration.program = program;
	runner_configuration.execution_stream = state.execution_stream;
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
	dispatch.sequence_id = sequence_ids[0];
	dispatch.active_sequence_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
	dispatch.new_token_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS;
	dispatch.row_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS;
	dispatch.lane_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
	dispatch.token_ids = token_ids;
	dispatch.row_lane_indices = lane_indices;
	dispatch.row_positions = positions;
	dispatch.row_sequence_ids = sequence_ids;
	dispatch.emit_count = SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES;
	dispatch.emit_row_indices = emit_rows;
	dispatch.emit_lane_indices = emit_lanes;
	dispatch.hidden_input_bf16 = stage_index != 0u ? state.hidden_input_device : 0;
	dispatch.hidden_input_bytes = stage_index != 0u ? hidden_input_bytes : 0u;
	dispatch.hidden_output_bf16 = stage_index + 1u < stage_count ? state.hidden_output_device : 0;
	dispatch.hidden_output_bytes = stage_index + 1u < stage_count ? hidden_input_bytes : 0u;
	dispatch.output_token_ids = stage_index + 1u == stage_count ? state.output_token_ids : 0;
	dispatch.completion_function = SparkDsv4DriverCudaSmokeCompletion;
	dispatch.completion_context = &state;
	status = SparkDsv4StageRunnerSubmit(&runner,&dispatch);
	if ( status == SPARK_STATUS_OK )
	{
		input_error = cudaStreamSynchronize(state.execution_stream);
		if ( input_error != cudaSuccess )
			status = SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( status == SPARK_STATUS_OK && stage_index + 1u < stage_count )
	{
		input_error = cudaMemcpy(state.hidden_bf16,state.hidden_output_device,(size_t)hidden_input_bytes,cudaMemcpyDeviceToHost);
		if ( input_error != cudaSuccess )
			status = SPARK_STATUS_INTERNAL_ERROR;
		else
		{
			state.send_count = 1u;
			for (element=0u; element<SPARK_DSV4_DRIVER_CUDA_SMOKE_ROWS * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
				if ( state.hidden_bf16[element] != 0u )
					state.nonzero_hidden_count++;
			for (wave=0u; wave<SPARK_DSV4_DRIVER_CUDA_SMOKE_PROMPT_TOKENS; wave++)
				for (element=0u; element<SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
					state.mismatch_count += state.hidden_bf16[((uint64_t)wave * SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS) + element] != state.hidden_bf16[((uint64_t)(wave * SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES + 1u) * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS) + element] ? 1u : 0u;
		}
	}
	if ( status == SPARK_STATUS_OK && stage_index + 1u == stage_count )
		for (wave=0u; wave<SPARK_DSV4_DRIVER_CUDA_SMOKE_PROMPT_TOKENS; wave++)
			state.mismatch_count += state.output_token_ids[wave * SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES] != state.output_token_ids[wave * SPARK_DSV4_DRIVER_CUDA_SMOKE_LANES + 1u] ? 1u : 0u;
	if ( status != SPARK_STATUS_OK || state.completion_count != 1u ||
		state.completion.status != SPARK_STATUS_OK ||
		state.send_count != (stage_index + 1u < stage_count ? 1u : 0u) ||
		state.receive_count != (stage_index != 0u ? 1u : 0u) ||
		(stage_index + 1u < stage_count && state.nonzero_hidden_count == 0u) ||
		state.mismatch_count != 0u ||
		(stage_index + 1u == stage_count && state.output_token_ids[0] >= SPARK_DSV4_MODEL_VOCAB_COUNT) ||
		runner.stats.submitted_count != 1u )
	{
		fprintf(stderr,"dsv4_driver_smoke execute=%s completion=%u status=%s sends=%u nonzero=%u mismatches=%u output_token=%u\n",SparkStatusToString(status),state.completion_count,SparkStatusToString(state.completion.status),state.send_count,state.nonzero_hidden_count,state.mismatch_count,state.output_token_ids[0]);
		goto destroy;
	}
	status = driver.interface->snapshot(driver_instance,program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK || snapshot.submitted_count != 1u ||
		snapshot.completed_count != 1u || snapshot.active_submission_count != 0u )
	{
		fprintf(stderr,"dsv4_driver_smoke snapshot=%s submitted=%llu completed=%llu active=%u\n",SparkStatusToString(status),(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count,snapshot.active_submission_count);
		goto destroy;
	}
	printf("dsv4_driver_cuda_smoke PASS sends=%u nonzero_hidden=%u mismatches=%u output_token=%u submitted=%llu completed=%llu\n",state.send_count,state.nonzero_hidden_count,state.mismatch_count,state.output_token_ids[0],(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count);
	result = 0;
destroy:
	driver.interface->destroy(driver_instance);
unload:
	SparkUnloadModelDriver(&driver);
done:
	if ( state.hidden_input_device != 0 )
		cudaFree(state.hidden_input_device);
	if ( state.hidden_output_device != 0 )
		cudaFree(state.hidden_output_device);
	if ( state.execution_stream != 0 )
		cudaStreamDestroy(state.execution_stream);
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
