/*
 * Fixture driver for the qwen38 serving adapter test. Emulates the compiled
 * model_driver.so shape: descriptor/flags/profile the adapter pins, create
 * reading the strict process environment the adapter must set (the qwen38
 * module's real configuration channel), submit_return completion, and the
 * head stage's token emission. TP4: every rank owns the embedding and the
 * head, so a frame carries token ids in (buffer 0) and head tokens out
 * (buffer 1) with no hidden transport. kv_token_capacity doubles as the
 * observation channel for the KV pool size the adapter derived (blocks x
 * block tokens).
 *
 * The prefix-cache gate needs the FRAMES the adapter built, so every
 * submitted frame is recorded - context flags, prefill geometry, the row
 * token ids, and the lane's block-table row from the host mirrors - into a
 * ring the test reads back through the exported query symbols (the test
 * dlopen()s this same library, so the ring is shared).
 */

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen38_max_model.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"

#ifndef QWEN38_MAX_MODEL_REVISION
#error "QWEN38_MAX_MODEL_REVISION must match the adapter build"
#endif
#ifndef QWEN38_MAX_CONTRACT_SHA256
#error "QWEN38_MAX_CONTRACT_SHA256 must match the adapter build"
#endif

/* The qwen38 adapter derives the 16 world stages from the firmware
 * contract and sets STAGE_INDEX = its own stage_index on every rank. */
#define TEST_QWEN38_DRIVER_STAGE_COUNT 16u
#define TEST_QWEN38_DRIVER_CAPTURE_ROWS 512u
#define TEST_QWEN38_DRIVER_RECORD_CAPACITY 4096u

/* One submitted frame, as the prefix gate observes it. */
typedef struct TestQwen38DriverFrameRecord
{
	uint32_t prefill;
	uint32_t rows;
	uint32_t context_flags;
	uint32_t snapshot_index_present;
	uint32_t snapshot_index;
	uint32_t lane_index;
	uint64_t base_position;
	uint64_t sequence_id;
	uint32_t block_count;
	uint32_t token_ids[TEST_QWEN38_DRIVER_CAPTURE_ROWS];
	uint32_t blocks[TEST_QWEN38_DRIVER_CAPTURE_ROWS];
}
TestQwen38DriverFrameRecord;

static TestQwen38DriverFrameRecord TestQwen38ServingDriverRecords[TEST_QWEN38_DRIVER_RECORD_CAPACITY];
static uint32_t TestQwen38ServingDriverRecordHead;
/* The KV pool size this driver instance was created with (the adapter's
 * SPARK_QWEN38_MAX_STAGE_KV_BLOCKS), for the deployment-wiring assertions. */
static uint32_t TestQwen38ServingDriverKvBlocks;

__attribute__((visibility("default")))
uint32_t TestQwen38ServingDriverKvBlockCount(void)
{
	return(TestQwen38ServingDriverKvBlocks);
}

__attribute__((visibility("default")))
uint32_t TestQwen38ServingDriverRecordCount(void)
{
	return(TestQwen38ServingDriverRecordHead);
}

__attribute__((visibility("default")))
const TestQwen38DriverFrameRecord *TestQwen38ServingDriverRecord(
	uint32_t index)
{
	if ( index >= TestQwen38ServingDriverRecordHead )
		return(0);
	return(&TestQwen38ServingDriverRecords[index]);
}

__attribute__((visibility("default")))
void TestQwen38ServingDriverResetRecords(void)
{
	TestQwen38ServingDriverRecordHead = 0u;
	memset(TestQwen38ServingDriverRecords,0,
		sizeof(TestQwen38ServingDriverRecords));
}

typedef struct TestQwen38ServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint32_t kv_block_count;
	uint64_t submitted_count;
	uint64_t completed_count;
} TestQwen38ServingDriver;

static SparkStatus TestQwen38ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestQwen38ServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_slots = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_new_tokens = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequences = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_sequence_tokens = SPARK_QWEN38_MAX_MODEL_MAXIMUM_CONTEXT_TOKENS
};

static const SparkModelDriverProgramDescriptor TestQwen38ServingDriverProgram =
{
	.program_id = 1u,
	.flags = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT,
	.max_inflight = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.name = "resident_decode",
	.profile = &TestQwen38ServingDriverProfile,
	.submit = TestQwen38ServingDriverSubmit
};

static const SparkModelDriverDescriptor TestQwen38ServingDriverDescriptor =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.descriptor_bytes = sizeof(SparkModelDriverDescriptor),
	.model_id = "qwen38_max.2.4t-a95b.resident-decode-stage-firmware",
	.model_revision = QWEN38_MAX_MODEL_REVISION,
	.stage_name = "qwen38_max_resident_decode_stage",
	.target = "cuda.sm121.qwen38_max.resident_decode_stage.fp8",
	.model_description_sha256 = QWEN38_MAX_CONTRACT_SHA256,
	.compiled_program_sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
	.program_count = 1u,
	.module_instance_count = 1u,
	.programs = &TestQwen38ServingDriverProgram
};

static uint32_t TestQwen38ServingDriverEnvironmentUnsigned(
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

static SparkStatus TestQwen38ServingDriverCreate(
	const SparkModelDriverCreateRequest *request,
	void **driver_instance)
{
	TestQwen38ServingDriver *driver;
	uint32_t stage_count,stage_index,kv_blocks,pipeline_slots;
	const char *pack_path;
	if ( SparkModelDriverCreateRequestIsValid(request) == 0u || driver_instance == 0 || request->execution_stream == 0 || request->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* The adapter configures the module through the strict environment and
	 * passes no node context; both are pinned here. */
	if ( request->node_context != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pack_path = getenv("SPARK_QWEN38_MAX_STAGE_PACK_PATH");
	if ( pack_path == 0 || strstr(pack_path,"qwen38-stage") == 0 || getenv("SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION") == 0 || getenv("SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION")[0] != '1' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( TestQwen38ServingDriverEnvironmentUnsigned("SPARK_QWEN38_MAX_STAGE_COUNT",&stage_count) == 0u || stage_count != TEST_QWEN38_DRIVER_STAGE_COUNT || TestQwen38ServingDriverEnvironmentUnsigned("SPARK_QWEN38_MAX_STAGE_INDEX",&stage_index) == 0u || stage_index >= stage_count || TestQwen38ServingDriverEnvironmentUnsigned("SPARK_QWEN38_MAX_STAGE_KV_BLOCKS",&kv_blocks) == 0u || kv_blocks == 0u || TestQwen38ServingDriverEnvironmentUnsigned("SPARK_QWEN38_MAX_STAGE_PIPELINE_SLOTS",&pipeline_slots) == 0u || pipeline_slots == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestQwen38ServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	driver->stage_index = stage_index;
	driver->kv_block_count = kv_blocks;
	TestQwen38ServingDriverKvBlocks = kv_blocks;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestQwen38ServingDriverDestroy(void *driver_instance)
{
	TestQwen38ServingDriver *driver;
	driver = (TestQwen38ServingDriver *)driver_instance;
	if ( driver == 0 )
		return;
	free(driver);
}

static SparkStatus TestQwen38ServingDriverAdmit(
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
	decision->available_dispatch_slot_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestQwen38ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestQwen38ServingDriver *driver;
	SparkQwen38MaxResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	uint32_t prefill,rows,row;
	uint32_t *tokens;
	driver = (TestQwen38ServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->tokens_per_sequence != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkQwen38MaxResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes < (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_ABI_MISMATCH);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill != 0u )
	{
		if ( (context->flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) == 0u || context->prefill_frame == 0 || context->decode_batch != 0 || frame->active_slot_count != 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->prefill_frame->token_count;
	}
	else
	{
		/* Plain batched decodes walk one row per active slot. A speculative
		 * MTP_DRAFT_AFTER frame walks 1 + D rows on ONE resident slot (the
		 * module contract's single-slot draft chain), so active_slot_count
		 * == 1 with more rows than slots is a legal decode shape here. */
		if ( (context->flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u || context->decode_batch == 0 || context->prefill_frame != 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( frame->active_slot_count != frame->new_token_count && frame->active_slot_count != 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		rows = context->decode_batch->row_count;
	}
	if ( rows == 0u || rows > TEST_QWEN38_DRIVER_CAPTURE_ROWS || rows != frame->new_token_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u || context->kv_block_table == 0 || context->kv_block_table->physical_block_indices == 0 || context->kv_block_table->host_physical_block_indices == 0 || context->kv_block_table->host_lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* TP4: every rank owns the embedding and the head, so a frame carries the
	 * token ids in (buffer 0) and the head tokens out (buffer 1) with no
	 * hidden transport. */
	if ( (context->flags & (SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT | SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT)) != 0u || context->hidden_input_post_receive_function != 0 || context->hidden_output_send_function != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 2u || frame->buffers[0].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_READ || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)rows * sizeof(uint32_t) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffers[1].slot != 1u || frame->buffers[1].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || frame->buffers[1].address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* Record the frame for the prefix gate BEFORE emitting: the input ids
	 * are read straight from buffer 0 (host memory), and the block-table
	 * row from the host mirrors the adapter must keep proven. */
	if ( TestQwen38ServingDriverRecordHead < TEST_QWEN38_DRIVER_RECORD_CAPACITY )
	{
		TestQwen38DriverFrameRecord *record =
			&TestQwen38ServingDriverRecords[TestQwen38ServingDriverRecordHead++];
		const uint32_t *input_ids = (const uint32_t *)frame->buffers[0].address;
		const SparkQwen38MaxKvBlockTableView *table = context->kv_block_table;
		uint32_t lane_index,ordinal;
		memset(record,0,sizeof(*record));
		record->prefill = prefill;
		record->rows = rows;
		record->context_flags = context->flags;
		if ( context->gdn_snapshot != 0 )
		{
			record->snapshot_index_present = 1u;
			record->snapshot_index = context->gdn_snapshot->snapshot_index;
		}
		if ( prefill != 0u )
		{
			record->lane_index = context->prefill_frame->lane_index;
			record->base_position = context->prefill_frame->base_position;
			record->sequence_id = context->prefill_frame->sequence_id;
		}
		else
		{
			lane_index = context->decode_batch->row_lane_indices[0];
			record->lane_index = lane_index;
			record->base_position = context->decode_batch->row_positions[0];
			record->sequence_id = context->decode_batch->row_sequence_ids[0];
		}
		lane_index = record->lane_index;
		for (row=0u; row<rows && row<TEST_QWEN38_DRIVER_CAPTURE_ROWS; row++)
			record->token_ids[row] = input_ids[row];
		if ( table != 0 && table->host_physical_block_indices != 0 &&
			table->host_lane_physical_block_counts != 0 &&
			lane_index < table->lane_count )
		{
			record->block_count = table->host_lane_physical_block_counts[lane_index];
			for (ordinal=0u; ordinal<record->block_count &&
				ordinal<TEST_QWEN38_DRIVER_CAPTURE_ROWS; ordinal++)
				record->blocks[ordinal] = table->host_physical_block_indices[
					((uint64_t)lane_index * table->lane_stride) + ordinal];
		}
	}
	tokens = (uint32_t *)frame->buffers[1].address;
	if ( prefill != 0u )
	{
		/* Every row of a prefill frame emits deterministically (a
		 * speculative verify frame reads ALL D rows back host-side). */
		for (row=0u; row<rows; row++)
			tokens[row] = 4242u;
	}
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

static SparkStatus TestQwen38ServingDriverSnapshot(
	void *driver_instance,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	TestQwen38ServingDriver *driver;
	if ( driver_instance == 0 || program_id != 1u || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestQwen38ServingDriver *)driver_instance;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	snapshot->kv_token_capacity = (uint64_t)driver->kv_block_count * SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverInterface TestQwen38ServingDriverInterface =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = sizeof(SparkModelDriverInterface),
	.descriptor = &TestQwen38ServingDriverDescriptor,
	.create = TestQwen38ServingDriverCreate,
	.destroy = TestQwen38ServingDriverDestroy,
	.admit = TestQwen38ServingDriverAdmit,
	.snapshot = TestQwen38ServingDriverSnapshot
};

__attribute__((visibility("default")))
const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
	return(&TestQwen38ServingDriverInterface);
}
