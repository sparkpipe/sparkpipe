/*
 * Fixture driver for the qwen36 serving adapter test. Emulates the compiled
 * model_driver.so shape: descriptor/flags/profile the adapter pins, create
 * reading the strict process environment the adapter must set (the qwen36
 * module's real configuration channel), submit_return completion, the
 * first-class hidden transport callbacks (post_receive gathers, send
 * scatters - echoed so the test sees a round trip), and the head stage's
 * token emission. kv_token_capacity doubles as the observation channel for
 * the KV pool size the adapter derived (blocks x block tokens).
 */

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"

#ifndef QWEN36_MODEL_REVISION
#error "QWEN36_MODEL_REVISION must match the adapter build"
#endif
#ifndef QWEN36_CONTRACT_SHA256
#error "QWEN36_CONTRACT_SHA256 must match the adapter build"
#endif

#define TEST_QWEN36_DRIVER_STAGE_COUNT 13u
#define TEST_QWEN36_DRIVER_CAPTURE_ROWS 16u

typedef struct TestQwen36ServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint32_t kv_block_count;
	uint64_t submitted_count;
	uint64_t completed_count;
	/* Device memory: on a CUDA host the adapter's transport shim moves
	 * hidden rows with real device-to-device copies, so the patterned
	 * stage-0 payload must live on device too. The cuda stub makes this a
	 * plain malloc on hosts without CUDA. */
	void *capture;
} TestQwen36ServingDriver;

static SparkStatus TestQwen36ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestQwen36ServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_slots = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_new_tokens = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequences = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_sequence_tokens = SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS
};

static const SparkModelDriverProgramDescriptor TestQwen36ServingDriverProgram =
{
	.program_id = 1u,
	.flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.name = "resident_decode",
	.profile = &TestQwen36ServingDriverProfile,
	.submit = TestQwen36ServingDriverSubmit
};

static const SparkModelDriverDescriptor TestQwen36ServingDriverDescriptor =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.descriptor_bytes = sizeof(SparkModelDriverDescriptor),
	.model_id = "alibaba.qwen3.6-27b.resident-decode-stage-firmware",
	.model_revision = QWEN36_MODEL_REVISION,
	.stage_name = "qwen36_resident_decode_stage",
	.target = "cuda.sm121.qwen36.resident_decode_stage.bf16",
	.model_description_sha256 = QWEN36_CONTRACT_SHA256,
	.compiled_program_sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
	.program_count = 1u,
	.module_instance_count = 1u,
	.programs = &TestQwen36ServingDriverProgram
};

static uint32_t TestQwen36ServingDriverEnvironmentUnsigned(
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

static SparkStatus TestQwen36ServingDriverCreate(
	const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	TestQwen36ServingDriver *driver;
	uint32_t stage_count,stage_index,kv_blocks,pipeline_slots;
	const char *pack_path;
	if ( SparkModelDriverCreateRequestIsValid(request) == 0u || driver_instance == 0 || request->execution_stream == 0 || request->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* The adapter configures the module through the strict environment and
	 * passes no node context; both are pinned here. */
	if ( request->node_context != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pack_path = getenv("SPARK_QWEN36_STAGE_PACK_PATH");
	if ( pack_path == 0 || strstr(pack_path,"qwen36-stage") == 0 || getenv("SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION") == 0 || getenv("SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION")[0] != '1' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( TestQwen36ServingDriverEnvironmentUnsigned("SPARK_QWEN36_STAGE_COUNT",&stage_count) == 0u || stage_count != TEST_QWEN36_DRIVER_STAGE_COUNT || TestQwen36ServingDriverEnvironmentUnsigned("SPARK_QWEN36_STAGE_INDEX",&stage_index) == 0u || stage_index >= stage_count || TestQwen36ServingDriverEnvironmentUnsigned("SPARK_QWEN36_STAGE_KV_BLOCKS",&kv_blocks) == 0u || kv_blocks == 0u || TestQwen36ServingDriverEnvironmentUnsigned("SPARK_QWEN36_STAGE_PIPELINE_SLOTS",&pipeline_slots) == 0u || pipeline_slots == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestQwen36ServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc(&driver->capture,(size_t)(TEST_QWEN36_DRIVER_CAPTURE_ROWS * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES)) != cudaSuccess )
	{
		free(driver);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	driver->stage_index = stage_index;
	driver->kv_block_count = kv_blocks;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestQwen36ServingDriverDestroy(void *driver_instance)
{
	TestQwen36ServingDriver *driver;
	driver = (TestQwen36ServingDriver *)driver_instance;
	if ( driver == 0 )
		return;
	(void)cudaFree(driver->capture);
	free(driver);
}

static SparkStatus TestQwen36ServingDriverAdmit(
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
	decision->available_dispatch_slot_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestQwen36ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestQwen36ServingDriver *driver;
	SparkQwen36ResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	SparkHiddenTransportPacket packet;
	uint32_t prefill,rows,row;
	uint64_t hidden_bytes;
	driver = (TestQwen36ServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkQwen36ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes < (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_ABI_MISMATCH);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill != 0u )
	{
		if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) == 0u || context->prefill_frame == 0 || context->decode_batch != 0 || frame->active_slot_count != 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->prefill_frame->token_count;
	}
	else
	{
		if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u || context->decode_batch == 0 || context->prefill_frame != 0 || frame->active_slot_count != frame->new_token_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->decode_batch->row_count;
	}
	if ( rows == 0u || rows > TEST_QWEN36_DRIVER_CAPTURE_ROWS || rows != frame->new_token_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u || context->kv_block_table == 0 || context->kv_block_table->physical_block_indices == 0 || context->kv_block_table->host_physical_block_indices == 0 || context->kv_block_table->host_lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	hidden_bytes = (uint64_t)rows * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES;
	if ( driver->stage_index != 0u )
	{
		if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) == 0u || context->hidden_input_post_receive_function == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet) != SPARK_STATUS_OK )
			return(SPARK_STATUS_IO_ERROR);
		if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION || context->hidden_input_packet.hidden_bf16 == 0 )
			return(SPARK_STATUS_VALIDATION_FAILED);
		/* Echo the received packet unchanged. The pointer is the adapter's
		 * device-side gather (or the submission boundary itself); a host
		 * copy out of it is exactly what segfaults a CUDA host. */
		if ( cudaMemcpy(driver->capture,context->hidden_input_packet.hidden_bf16,(size_t)hidden_bytes,cudaMemcpyDeviceToDevice) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	else
	{
		if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u || context->hidden_input_post_receive_function != 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( frame->buffer_count == 0u || frame->buffers[0].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_READ || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)rows * sizeof(uint32_t) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( cudaMemset(driver->capture,0x5a,(size_t)hidden_bytes) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	if ( driver->stage_index + 1u == TEST_QWEN36_DRIVER_STAGE_COUNT )
	{
		uint32_t out_index;
		uint32_t *tokens;
		out_index = driver->stage_index == 0u ? 1u : 0u;
		if ( frame->buffer_count != out_index + 1u || frame->buffers[out_index].slot != 1u || frame->buffers[out_index].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || frame->buffers[out_index].address == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		tokens = (uint32_t *)frame->buffers[out_index].address;
		if ( prefill != 0u )
			tokens[0] = 4242u;
		else
			for (row=0u; row<rows; row++)
				tokens[row] = 4200u + row;
	}
	else
	{
		if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) == 0u || context->hidden_output_send_function == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		memset(&packet,0,sizeof(packet));
		packet.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
		packet.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
		packet.active_sequence_count = rows;
		packet.hidden_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		packet.bytes_per_sequence = SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES;
		packet.hidden_bf16 = driver->capture;
		packet.cuda_stream = frame->execution_stream;
		if ( context->hidden_output_send_function(context->hidden_output_transport_session,&packet) != SPARK_STATUS_OK )
			return(SPARK_STATUS_IO_ERROR);
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

static SparkStatus TestQwen36ServingDriverSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	TestQwen36ServingDriver *driver;
	if ( driver_instance == 0 || program_id != 1u || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestQwen36ServingDriver *)driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	snapshot->kv_token_capacity = (uint64_t)driver->kv_block_count * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverInterface TestQwen36ServingDriverInterface =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = sizeof(SparkModelDriverInterface),
	.descriptor = &TestQwen36ServingDriverDescriptor,
	.create = TestQwen36ServingDriverCreate,
	.destroy = TestQwen36ServingDriverDestroy,
	.admit = TestQwen36ServingDriverAdmit,
	.snapshot = TestQwen36ServingDriverSnapshot
};

__attribute__((visibility("default")))
const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&TestQwen36ServingDriverInterface);
}
