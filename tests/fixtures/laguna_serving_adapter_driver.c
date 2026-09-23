#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"

#ifndef LAGUNA_MODEL_REVISION
#error "LAGUNA_MODEL_REVISION must match the adapter build"
#endif
#ifndef LAGUNA_CONTRACT_SHA256
#error "LAGUNA_CONTRACT_SHA256 must match the adapter build"
#endif
#ifndef LAGUNA_EXPERT_CODEC_NAME
#error "LAGUNA_EXPERT_CODEC_NAME must match the adapter build"
#endif

/* The boundary receipt tokens: the frame reaches the driver either as
   the raw wire form (no hidden transport bound yet - the residentd
   validates before the route bind) or as the route-bound form (exactly
   one hidden side wired: a first-stage frame ships hidden_output, a
   last-stage frame consumes hidden_input). The stage markers let the
   test pin that the bound form survives validation and execution - the
   attach-011d first-decode abort was the validator rejecting exactly
   that form. */
#define TEST_LAGUNA_STAGE0_TOKEN 5000u
#define TEST_LAGUNA_STAGE1_TOKEN 6000u
#define TEST_LAGUNA_RAW_TOKEN 7000u
#define TEST_LAGUNA_DRIVER_PROGRAM_ID 1u
#define TEST_LAGUNA_DRIVER_KV_BLOCKS 16u

typedef struct TestLagunaServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint64_t submitted_count;
	uint64_t completed_count;
} TestLagunaServingDriver;

static SparkStatus TestLagunaServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestLagunaServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL,
	.max_inflight = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_slots = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_new_tokens = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequences = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_sequence_tokens = SPARK_LAGUNA_MODEL_MAXIMUM_CONTEXT_TOKENS
};

static const SparkModelDriverProgramDescriptor TestLagunaServingDriverProgram =
{
	.program_id = TEST_LAGUNA_DRIVER_PROGRAM_ID,
	.flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL,
	.max_inflight = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.name = "resident_decode",
	.profile = &TestLagunaServingDriverProfile,
	.submit = TestLagunaServingDriverSubmit
};

static const SparkModelDriverDescriptor TestLagunaServingDriverDescriptor =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.descriptor_bytes = sizeof(SparkModelDriverDescriptor),
	.model_id = "laguna.laguna-s-2.1.resident-decode-stage-firmware",
	.model_revision = LAGUNA_MODEL_REVISION,
	.stage_name = "laguna_resident_decode_stage",
	.target = "cuda.sm121.laguna.resident_decode_stage.bf16.expert_" LAGUNA_EXPERT_CODEC_NAME,
	.model_description_sha256 = LAGUNA_CONTRACT_SHA256,
	.compiled_program_sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
	.program_count = 1u,
	.module_instance_count = 1u,
	.programs = &TestLagunaServingDriverProgram
};

static SparkStatus TestLagunaServingDriverCreate(
	const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	TestLagunaServingDriver *driver;
	if ( SparkModelDriverCreateRequestIsValid(request) == 0u || driver_instance == 0 || request->execution_stream == 0 || request->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestLagunaServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestLagunaServingDriverDestroy(void *driver_instance)
{
	TestLagunaServingDriver *driver;
	driver = (TestLagunaServingDriver *)driver_instance;
	if ( driver == 0 )
		return;
	free(driver);
}

static SparkStatus TestLagunaServingDriverAdmit(
	void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	if ( driver_instance == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->descriptor_bytes != (uint32_t)sizeof(*request) || request->program_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = sizeof(*decision);
	decision->accepted = 1u;
	decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	decision->available_dispatch_slot_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestLagunaServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestLagunaServingDriver *driver;
	SparkLagunaResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	uint32_t rows,row,token;
	uint32_t *tokens;
	driver = (TestLagunaServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 || frame->user_context == 0 || frame->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkLagunaResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes < (uint32_t)sizeof(*context) || context->batch == 0 )
		return(SPARK_STATUS_ABI_MISMATCH);
	rows = context->batch->row_count;
	if ( rows == 0u || rows != frame->new_token_count || frame->active_slot_count != context->batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* Two legitimate frame forms: the raw wire form (no hidden side
	   bound yet, token buffer present) and the route-bound form. A
	   bound frame ships hidden_output WITHOUT a token buffer (the final
	   stage alone materializes tokens - the module's ValidateFrame
	   rejects a WRITE buffer on a shipping stage); a consuming frame
	   carries hidden_input plus the WRITE buffer. */
	if ( (context->hidden_input_bytes == 0u) != (context->hidden_input_bf16 == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->hidden_output_bytes == 0u) != (context->hidden_output_bf16 == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->hidden_input_bytes != 0u && context->hidden_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT) != 0u != (context->hidden_input_bytes != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT) != 0u != (context->hidden_output_bytes != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count == 0u )
	{
		if ( frame->buffers != 0 || context->hidden_output_bytes == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		token = 0u;
	}
	else if ( frame->buffer_count == 1u && frame->buffers != 0 && frame->buffers[0].flags == SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE && frame->buffers[0].address != 0 && frame->buffers[0].bytes >= (uint64_t)rows * sizeof(uint32_t) && context->hidden_output_bytes == 0u )
	{
		if ( context->hidden_input_bytes != 0u )
			token = TEST_LAGUNA_STAGE1_TOKEN;
		else
			token = TEST_LAGUNA_RAW_TOKEN;
		tokens = (uint32_t *)frame->buffers[0].address;
		for (row=0u; row<rows; row++)
			tokens[row] = token + row;
	}
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver->submitted_count++;
	driver->completed_count++;
	memset(&completion,0,sizeof(completion));
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.accepted_token_count = frame->new_token_count;
	completion.status = SPARK_STATUS_OK;
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static SparkStatus TestLagunaServingDriverSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	TestLagunaServingDriver *driver;
	if ( driver_instance == 0 || program_id != TEST_LAGUNA_DRIVER_PROGRAM_ID || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestLagunaServingDriver *)driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	snapshot->kv_token_capacity = (uint64_t)TEST_LAGUNA_DRIVER_KV_BLOCKS * SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverInterface TestLagunaServingDriverInterface =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = sizeof(SparkModelDriverInterface),
	.descriptor = &TestLagunaServingDriverDescriptor,
	.create = TestLagunaServingDriverCreate,
	.destroy = TestLagunaServingDriverDestroy,
	.admit = TestLagunaServingDriverAdmit,
	.snapshot = TestLagunaServingDriverSnapshot
};

__attribute__((visibility("default")))
const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&TestLagunaServingDriverInterface);
}
