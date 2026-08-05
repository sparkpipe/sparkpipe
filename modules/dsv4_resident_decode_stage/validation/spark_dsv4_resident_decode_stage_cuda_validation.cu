#include <cuda_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

typedef struct SparkDsv4ValidationCapture
{
	uint16_t hidden[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint32_t output_token_id;
	uint32_t output_count;
	uint32_t nonzero_count;
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4ValidationCapture;

typedef struct SparkDsv4ValidationFrame
{
	SparkDsv4DecodeBatchView batch;
	SparkDsv4ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	void *hidden_input_bf16;
	void *hidden_output_bf16;
	uint32_t token_id;
	uint32_t lane;
	uint64_t position;
	uint64_t sequence_id;
} SparkDsv4ValidationFrame;

static int SparkDsv4ValidationRequire(int condition,const char *message)
{
	if ( condition != 0 )
		return(0);
	fprintf(stderr,"dsv4_validation failure=%s\n",message);
	return(1);
}

static int SparkDsv4ValidationReadUnsigned(
	const char *name,
	uint32_t minimum,
	uint32_t maximum,
	uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	text = getenv(name);
	if ( text == 0 || text[0] == '\0' || value == 0 )
		return(1);
	errno = 0;
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( errno != 0 || end == text || end[0] != '\0' || parsed < minimum || parsed > maximum )
		return(1);
	*value = (uint32_t)parsed;
	return(0);
}

static int SparkDsv4ValidationLoadNodeContext(
	SparkDsv4ResidentDecodeStageNodeContext *context)
{
	uint32_t mtp_layer_count,cuda_graph_count;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	if ( SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_COUNT",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&context->stage_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_INDEX",0u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&context->stage_index) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_FIRST_LAYER",0u,SPARK_DSV4_MODEL_LAYER_COUNT - 1u,&context->first_layer_index) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_LAYER_COUNT",1u,SPARK_DSV4_MODEL_LAYER_COUNT,&context->layer_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&context->resident_sequence_capacity) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_PIPELINE_SLOTS",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&context->pipeline_slot_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MAX_SEQ",SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO,SPARK_DSV4_MODEL_MAX_POSITIONS,&context->max_sequence_positions) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MTP",0u,0u,&mtp_layer_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_GRAPHS",0u,0u,&cuda_graph_count) != 0 )
		return(1);
	context->linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	context->expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	context->kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
	context->stage_pack_path = getenv("SPARK_DSV4_STAGE_PACK_PATH");
	if ( context->stage_index >= context->stage_count || context->first_layer_index + context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' )
		return(1);
	return(0);
}

static void SparkDsv4ValidationInitializeConfiguration(
	SparkFirmwareModuleConfiguration *configuration,
	SparkFirmwareModuleHostServices *host_services,
	SparkDsv4ResidentDecodeStageNodeContext *node_context)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration->descriptor_bytes = sizeof(*configuration);
	configuration->model_id = SPARK_DSV4_MODEL_ID;
	configuration->model_revision = SPARK_DSV4_MODEL_SOURCE_REVISION;
	configuration->stage_name = "dsv4_resident_decode_stage";
	configuration->program_name = "resident_decode";
	configuration->operation_name = "dsv4_resident_decode_stage";
	configuration->configuration_json = "{}";
	configuration->configuration_json_bytes = 2u;
	memset(host_services,0,sizeof(*host_services));
	host_services->abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services->descriptor_bytes = sizeof(*host_services);
	host_services->node_id = "spark-dsv4-validator";
	host_services->node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	host_services->execution_stream = (void *)cudaStreamPerThread;
	host_services->node_context = node_context;
}

static void SparkDsv4ValidationCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkDsv4ValidationCapture *capture;
	capture = (SparkDsv4ValidationCapture *)completion_context;
	if ( capture == 0 || completion == 0 )
		return;
	capture->completion = *completion;
	capture->completion_count++;
}

static void SparkDsv4ValidationDestroyFrame(SparkDsv4ValidationFrame *frame)
{
	if ( frame->hidden_input_bf16 != 0 )
		cudaFree(frame->hidden_input_bf16);
	if ( frame->hidden_output_bf16 != 0 )
		cudaFree(frame->hidden_output_bf16);
	memset(frame,0,sizeof(*frame));
}

static int SparkDsv4ValidationInitializeBoundaries(
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture,
	SparkDsv4ValidationFrame *frame)
{
	uint32_t element;
	uint64_t hidden_bytes;
	cudaError_t error;
	hidden_bytes = sizeof(capture->hidden);
	if ( node_context->stage_index != 0u )
	{
		for (element=0u; element<SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
			capture->hidden[element] = UINT16_C(0x3f80);
		error = cudaMalloc(&frame->hidden_input_bf16,hidden_bytes);
		if ( error == cudaSuccess )
			error = cudaMemcpy(frame->hidden_input_bf16,capture->hidden,hidden_bytes,cudaMemcpyHostToDevice);
		if ( error != cudaSuccess )
			return(1);
	}
	if ( node_context->stage_index + 1u < node_context->stage_count )
	{
		error = cudaMalloc(&frame->hidden_output_bf16,hidden_bytes);
		if ( error != cudaSuccess )
			return(1);
	}
	return(0);
}

static void SparkDsv4ValidationBuildFrame(
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture,
	SparkDsv4ValidationFrame *frame)
{
	uint32_t buffer_count;
	uint64_t hidden_bytes;
	hidden_bytes = sizeof(capture->hidden);
	frame->token_id = 10397u;
	frame->sequence_id = 1u;
	frame->batch.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
	frame->batch.descriptor_bytes = sizeof(frame->batch);
	frame->batch.row_count = 1u;
	frame->batch.row_lane_indices = &frame->lane;
	frame->batch.row_positions = &frame->position;
	frame->batch.row_sequence_ids = &frame->sequence_id;
	frame->context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	frame->context.descriptor_bytes = sizeof(frame->context);
	frame->context.flags = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
	frame->context.decode_batch = &frame->batch;
	frame->context.hidden_input_bf16 = frame->hidden_input_bf16;
	frame->context.hidden_input_bytes = frame->hidden_input_bf16 != 0 ? hidden_bytes : 0u;
	frame->context.hidden_output_bf16 = frame->hidden_output_bf16;
	frame->context.hidden_output_bytes = frame->hidden_output_bf16 != 0 ? hidden_bytes : 0u;
	if ( frame->hidden_input_bf16 != 0 )
		frame->context.flags |= SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER;
	if ( frame->hidden_output_bf16 != 0 )
		frame->context.flags |= SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER;
	buffer_count = 0u;
	if ( node_context->first_layer_index == 0u )
	{
		frame->buffers[buffer_count].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		frame->buffers[buffer_count].address = &frame->token_id;
		frame->buffers[buffer_count++].bytes = sizeof(frame->token_id);
	}
	if ( node_context->first_layer_index + node_context->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		frame->buffers[buffer_count].slot = 1u;
		frame->buffers[buffer_count].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		frame->buffers[buffer_count].address = &capture->output_token_id;
		frame->buffers[buffer_count++].bytes = sizeof(capture->output_token_id);
	}
	frame->frame.request_id = 1u;
	frame->frame.sequence_id = frame->sequence_id;
	frame->frame.active_slot_count = 1u;
	frame->frame.new_token_count = 1u;
	frame->frame.program_id = 1u;
	frame->frame.execution_stream = (void *)cudaStreamPerThread;
	frame->frame.buffers = buffer_count != 0u ? frame->buffers : 0;
	frame->frame.buffer_count = buffer_count;
	frame->frame.user_context = &frame->context;
	frame->frame.completion_function = SparkDsv4ValidationCompletion;
	frame->frame.completion_context = capture;
}

static int SparkDsv4ValidationRunFrame(
	void *module_state,
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture)
{
	SparkDsv4ValidationFrame frame;
	uint32_t element;
	cudaError_t error;
	SparkStatus status;
	memset(&frame,0,sizeof(frame));
	memset(capture,0,sizeof(*capture));
	capture->output_token_id = UINT32_MAX;
	if ( SparkDsv4ValidationInitializeBoundaries(node_context,capture,&frame) != 0 )
	{
		SparkDsv4ValidationDestroyFrame(&frame);
		return(1);
	}
	SparkDsv4ValidationBuildFrame(node_context,capture,&frame);
	status = SparkDsv4ResidentDecodeStageExecute(module_state,&frame.frame);
	error = status == SPARK_STATUS_OK ? cudaStreamSynchronize(cudaStreamPerThread) : cudaErrorUnknown;
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && frame.hidden_output_bf16 != 0 )
		error = cudaMemcpy(capture->hidden,frame.hidden_output_bf16,sizeof(capture->hidden),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess && frame.hidden_output_bf16 != 0 )
		for (element=0u; element<SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
			capture->nonzero_count += capture->hidden[element] != 0u ? 1u : 0u;
	capture->output_count = error == cudaSuccess ? 1u : 0u;
	SparkDsv4ValidationDestroyFrame(&frame);
	if ( status != SPARK_STATUS_OK || error != cudaSuccess )
	{
		fprintf(stderr,"dsv4_validation execute=%s cuda=%s\n",SparkStatusToString(status),cudaGetErrorString(error));
		return(1);
	}
	if ( SparkDsv4ValidationRequire(capture->completion_count == 1u && capture->completion.status == SPARK_STATUS_OK,"completion") != 0 || SparkDsv4ValidationRequire(capture->output_count == 1u,"output_count") != 0 )
		return(1);
	if ( node_context->stage_index + 1u < node_context->stage_count )
		return(SparkDsv4ValidationRequire(capture->nonzero_count > 0u,"hidden_output_nonzero"));
	return(SparkDsv4ValidationRequire(capture->output_token_id < SPARK_DSV4_MODEL_VOCAB_COUNT,"output_token_range"));
}

static int SparkDsv4ValidationAdmit(void *module_state)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = 1u;
	request.sequence_id = 1u;
	request.active_slot_count = 1u;
	request.new_token_count = 1u;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkDsv4ResidentDecodeStageAdmit(module_state,&request,&decision);
	if ( status != SPARK_STATUS_OK || decision.accepted == 0u )
	{
		fprintf(stderr,"dsv4_validation admit=%s accepted=%u\n",SparkStatusToString(status),decision.accepted);
		return(1);
	}
	return(0);
}

int main(int argument_count,char **arguments)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkDsv4ResidentDecodeStageNodeContext node_context;
	SparkDsv4ValidationCapture capture;
	SparkModelDriverRuntimeSnapshot snapshot;
	void *module_state;
	SparkStatus status;
	if ( argument_count != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",arguments[0]);
		return(2);
	}
	if ( SparkDsv4ValidationLoadNodeContext(&node_context) != 0 )
	{
		fprintf(stderr,"dsv4_validation configuration=invalid\n");
		return(1);
	}
	SparkDsv4ValidationInitializeConfiguration(&configuration,&host_services,&node_context);
	module_state = 0;
	status = SparkDsv4ResidentDecodeStageInitialize(&configuration,&host_services,&module_state);
	if ( status != SPARK_STATUS_OK || module_state == 0 )
	{
		fprintf(stderr,"dsv4_validation initialize=%s\n",SparkStatusToString(status));
		return(1);
	}
	if ( SparkDsv4ValidationAdmit(module_state) != 0 || SparkDsv4ValidationRunFrame(module_state,&node_context,&capture) != 0 )
	{
		SparkDsv4ResidentDecodeStageDestroy(module_state);
		return(1);
	}
	memset(&snapshot,0,sizeof(snapshot));
	snapshot.descriptor_bytes = sizeof(snapshot);
	status = SparkDsv4ResidentDecodeStageSnapshot(module_state,1u,&snapshot);
	SparkDsv4ResidentDecodeStageDestroy(module_state);
	if ( status != SPARK_STATUS_OK || snapshot.submitted_count != 1u || snapshot.completed_count != 1u || snapshot.active_submission_count != 0u )
	{
		fprintf(stderr,"dsv4_validation snapshot=%s submitted=%llu completed=%llu active=%u\n",SparkStatusToString(status),(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count,snapshot.active_submission_count);
		return(1);
	}
	printf("dsv4_validation PASS config=%s stage=%u slice=%u+%u nonzero_hidden=%u output_token=%u\n",arguments[1],node_context.stage_index,node_context.first_layer_index,node_context.layer_count,capture.nonzero_count,capture.output_token_id);
	return(0);
}
