
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_muse_glimmer_model.h"
#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"

#ifndef MUSE_MODEL_REVISION
#error "MUSE_MODEL_REVISION must match the adapter build"
#endif
#ifndef MUSE_CONTRACT_SHA256
#error "MUSE_CONTRACT_SHA256 must match the adapter build"
#endif

#define TEST_MUSE_GLIMMER_DRIVER_STAGE_COUNT 1u
#define TEST_MUSE_GLIMMER_DRIVER_CAPTURE_ROWS 16u
#define TEST_MUSE_GLIMMER_DRIVER_PROGRAM_ID 1u

typedef struct TestMuseGlimmerServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t kv_block_count;
	uint64_t submitted_count;
	uint64_t completed_count;
} TestMuseGlimmerServingDriver;

static SparkStatus TestMuseGlimmerServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestMuseGlimmerServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_slots = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_new_tokens = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequences = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_sequence_tokens = SPARK_MUSE_GLIMMER_MODEL_MAXIMUM_CONTEXT_TOKENS
};

static const SparkModelDriverProgramDescriptor TestMuseGlimmerServingDriverProgram =
{
	.program_id = TEST_MUSE_GLIMMER_DRIVER_PROGRAM_ID,
	.flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.name = "resident_decode",
	.profile = &TestMuseGlimmerServingDriverProfile,
	.submit = TestMuseGlimmerServingDriverSubmit
};

static const SparkModelDriverDescriptor TestMuseGlimmerServingDriverDescriptor =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.descriptor_bytes = sizeof(SparkModelDriverDescriptor),
	.model_id = "muse_glimmer.30b.resident-decode-stage-firmware",
	.model_revision = MUSE_MODEL_REVISION,
	.stage_name = "muse_glimmer_resident_decode_stage",
	.target = "cuda.sm121.muse_glimmer.resident_decode_stage.bf16",
	.model_description_sha256 = MUSE_CONTRACT_SHA256,
	.compiled_program_sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
	.program_count = 1u,
	.module_instance_count = 1u,
	.programs = &TestMuseGlimmerServingDriverProgram
};

static uint32_t TestMuseGlimmerServingDriverEnvironmentUnsigned(
	const char *name,
	uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	text = getenv(name);
	if ( text == 0 || text[0] == '\0' )
		return(0u);
	parsed = strtoul(text,&end,10);
	if ( end == text || *end != '\0' || parsed > 0xfffffffful )
		return(0u);
	*value = (uint32_t)parsed;
	return(1u);
}

static SparkStatus TestMuseGlimmerServingDriverCreate(
	const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	TestMuseGlimmerServingDriver *driver;
	uint32_t stage_count,stage_index,kv_blocks,pipeline_slots;
	const char *pack_path;
	if ( SparkModelDriverCreateRequestIsValid(request) == 0u || driver_instance == 0 || request->execution_stream == 0 || request->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->node_context != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pack_path = getenv("SPARK_MUSE_GLIMMER_STAGE_PACK_PATH");
	if ( pack_path == 0 || strstr(pack_path,"muse") == 0 || getenv("SPARK_MUSE_GLIMMER_ALLOW_UNQUALIFIED_EXECUTION") == 0 || getenv("SPARK_MUSE_GLIMMER_ALLOW_UNQUALIFIED_EXECUTION")[0] != '1' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( TestMuseGlimmerServingDriverEnvironmentUnsigned("SPARK_MUSE_GLIMMER_STAGE_COUNT",&stage_count) == 0u || stage_count != TEST_MUSE_GLIMMER_DRIVER_STAGE_COUNT || TestMuseGlimmerServingDriverEnvironmentUnsigned("SPARK_MUSE_GLIMMER_STAGE_INDEX",&stage_index) == 0u || stage_index >= stage_count || TestMuseGlimmerServingDriverEnvironmentUnsigned("SPARK_MUSE_GLIMMER_STAGE_KV_BLOCKS",&kv_blocks) == 0u || kv_blocks == 0u || TestMuseGlimmerServingDriverEnvironmentUnsigned("SPARK_MUSE_GLIMMER_STAGE_PIPELINE_SLOTS",&pipeline_slots) == 0u || pipeline_slots == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestMuseGlimmerServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	driver->kv_block_count = kv_blocks;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestMuseGlimmerServingDriverDestroy(void *driver_instance)
{
	TestMuseGlimmerServingDriver *driver;
	driver = (TestMuseGlimmerServingDriver *)driver_instance;
	if ( driver == 0 )
		return;
	free(driver);
}

static SparkStatus TestMuseGlimmerServingDriverAdmit(
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
	decision->available_dispatch_slot_count = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestMuseGlimmerServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestMuseGlimmerServingDriver *driver;
	SparkMuseGlimmerResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	uint32_t prefill,rows,row;
	uint32_t *tokens;
	driver = (TestMuseGlimmerServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->tokens_per_sequence != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkMuseGlimmerResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes < (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_ABI_MISMATCH);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill != 0u )
	{
		if ( (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) == 0u || context->prefill_frame == 0 || context->decode_batch != 0 || frame->active_slot_count != 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->prefill_frame->token_count;
	}
	else
	{
		if ( (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u || context->decode_batch == 0 || context->prefill_frame != 0 || frame->active_slot_count != frame->new_token_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->decode_batch->row_count;
	}
	if ( rows == 0u || rows > TEST_MUSE_GLIMMER_DRIVER_CAPTURE_ROWS || rows != frame->new_token_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u || context->kv_block_table == 0 || context->kv_block_table->physical_block_indices == 0 || context->kv_block_table->host_physical_block_indices == 0 || context->kv_block_table->host_lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & (SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT | SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT)) != 0u || context->hidden_input_post_receive_function != 0 || context->hidden_output_send_function != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 2u || frame->buffers[0].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_READ || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)rows * sizeof(uint32_t) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffers[1].slot != 1u || frame->buffers[1].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || frame->buffers[1].address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	tokens = (uint32_t *)frame->buffers[1].address;
	if ( prefill != 0u )
		tokens[0] = 4242u;
	else
		for (row=0u; row<rows; row++)
			tokens[row] = 4200u + row;
	driver->submitted_count++;
	driver->completed_count++;
	memset(&completion,0,sizeof(completion));
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.accepted_token_count = frame->new_token_count;
	completion.tokens_per_sequence = frame->tokens_per_sequence;
	completion.residency = frame->residency;
	completion.status = SPARK_STATUS_OK;
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static SparkStatus TestMuseGlimmerServingDriverSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	TestMuseGlimmerServingDriver *driver;
	if ( driver_instance == 0 || program_id != TEST_MUSE_GLIMMER_DRIVER_PROGRAM_ID || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestMuseGlimmerServingDriver *)driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	snapshot->kv_token_capacity = (uint64_t)driver->kv_block_count * SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverInterface TestMuseGlimmerServingDriverInterface =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = sizeof(SparkModelDriverInterface),
	.descriptor = &TestMuseGlimmerServingDriverDescriptor,
	.create = TestMuseGlimmerServingDriverCreate,
	.destroy = TestMuseGlimmerServingDriverDestroy,
	.admit = TestMuseGlimmerServingDriverAdmit,
	.snapshot = TestMuseGlimmerServingDriverSnapshot
};

__attribute__((visibility("default")))
const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&TestMuseGlimmerServingDriverInterface);
}
