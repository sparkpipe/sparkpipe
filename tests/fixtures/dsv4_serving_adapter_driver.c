#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"

typedef struct TestDsv4ServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint64_t submitted_count;
	uint64_t completed_count;
} TestDsv4ServingDriver;

static SparkStatus TestDsv4ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestDsv4ServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = 4u,
	.max_active_slots = 128u,
	.max_new_tokens = 128u,
	.max_resident_sequences = 128u,
	.max_sequence_tokens = 1048576u
};

static const SparkModelDriverProgramDescriptor TestDsv4ServingDriverProgram =
{
	.program_id = 1u,
	.flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = 4u,
	.name = "resident_decode",
	.profile = &TestDsv4ServingDriverProfile,
	.submit = TestDsv4ServingDriverSubmit
};

static const SparkModelDriverDescriptor TestDsv4ServingDriverDescriptor =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.descriptor_bytes = sizeof(SparkModelDriverDescriptor),
	.model_id = SPARK_DSV4_MODEL_DRIVER_MODEL_ID,
	.model_revision = SPARK_DSV4_MODEL_DRIVER_REVISION,
	.stage_name = "dsv4_resident_decode_stage",
	.target = SPARK_DSV4_MODEL_MODULE_TARGET,
	.model_description_sha256 = SPARK_DSV4_MODEL_DESCRIPTION_SHA256,
	.compiled_program_sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
	.program_count = 1u,
	.module_instance_count = 1u,
	.programs = &TestDsv4ServingDriverProgram
};

static SparkStatus TestDsv4ServingDriverCreate(
	const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	const SparkDsv4ResidentDecodeStageNodeContext *context;
	TestDsv4ServingDriver *driver;
	if ( SparkModelDriverCreateRequestIsValid(request) == 0u || driver_instance == 0 || request->node_context == 0 || request->execution_stream == 0 || request->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageNodeContext *)request->node_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES || context->stage_count != 13u || context->stage_index >= context->stage_count || context->layer_count == 0u || context->stage_pack_path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestDsv4ServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	driver->stage_index = context->stage_index;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestDsv4ServingDriverDestroy(void *driver_instance)
{
	free(driver_instance);
}

static SparkStatus TestDsv4ServingDriverAdmit(
	void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	if ( driver_instance == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = sizeof(*decision);
	decision->accepted = 1u;
	decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	decision->available_dispatch_slot_count = 4u;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestDsv4ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestDsv4ServingDriver *driver;
	SparkDsv4ResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	const uint32_t *row_slots;
	const uint64_t *row_sequences;
	uint32_t *tokens,row,row_count;
	uint64_t hidden_bytes;
	driver = (TestDsv4ServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkDsv4ResidentDecodeStageFrameContext *)frame->user_context;
	row_count = context->decode_batch != 0 ? context->decode_batch->row_count : context->prefill_batch != 0 ? context->prefill_batch->row_count : 0u;
	if ( row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	row_slots = context->decode_batch != 0 ? context->decode_batch->row_lane_indices : context->prefill_batch->row_lane_indices;
	row_sequences = context->decode_batch != 0 ? context->decode_batch->row_sequence_ids : context->prefill_batch->row_sequence_ids;
	for (row=0u; row<row_count; row++)
	{
		uint32_t expected_slot;
		expected_slot = row_sequences[row] == 100u ? 7u : row_sequences[row] == 101u ? 3u : row_sequences[row] == 200u ? 5u : UINT32_MAX;
		if ( row_slots[row] != expected_slot )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	hidden_bytes = (uint64_t)row_count * 16384u * sizeof(uint16_t);
	if ( driver->stage_index != 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < hidden_bytes) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( driver->stage_index + 1u < 13u )
	{
		if ( context->hidden_output_bf16 == 0 || context->hidden_output_bytes < hidden_bytes )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( context->hidden_input_bf16 != 0 )
			memcpy(context->hidden_output_bf16,context->hidden_input_bf16,(size_t)hidden_bytes);
		else
			memset(context->hidden_output_bf16,0x5a,(size_t)hidden_bytes);
	}
	else
	{
		if ( frame->buffer_count != 2u || frame->buffers[0].slot != 0u || frame->buffers[0].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_READ || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)row_count * sizeof(uint32_t) || frame->buffers[1].slot != 1u || frame->buffers[1].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || frame->buffers[1].address == 0 || frame->buffers[1].bytes < (uint64_t)row_count * sizeof(uint32_t) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( context->prefill_batch != 0 && context->prefill_batch->token_ids != frame->buffers[0].address )
			return(SPARK_STATUS_SCHEMA_ERROR);
		tokens = (uint32_t *)frame->buffers[1].address;
		for (row=0u; row<row_count; row++)
			tokens[row] = 4200u + row;
	}
	driver->submitted_count++;
	driver->completed_count++;
	memset(&completion,0,sizeof(completion));
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.accepted_token_count = frame->new_token_count;
	completion.residency = frame->residency;
	completion.status = SPARK_STATUS_OK;
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static SparkStatus TestDsv4ServingDriverSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	TestDsv4ServingDriver *driver;
	if ( driver_instance == 0 || program_id != 1u || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestDsv4ServingDriver *)driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = 4u;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverInterface TestDsv4ServingDriverInterface =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = sizeof(SparkModelDriverInterface),
	.descriptor = &TestDsv4ServingDriverDescriptor,
	.create = TestDsv4ServingDriverCreate,
	.destroy = TestDsv4ServingDriverDestroy,
	.admit = TestDsv4ServingDriverAdmit,
	.snapshot = TestDsv4ServingDriverSnapshot
};

__attribute__((visibility("default")))
const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&TestDsv4ServingDriverInterface);
}
