#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"

#if TEST_DSV4_SERVING_HYBRID
#define TEST_DSV4_SERVING_STAGE_COUNT 4u
#define TEST_DSV4_SERVING_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#elif TEST_DSV4_SERVING_TP16
#define TEST_DSV4_SERVING_STAGE_COUNT 16u
#define TEST_DSV4_SERVING_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#else
#define TEST_DSV4_SERVING_STAGE_COUNT 13u
#define TEST_DSV4_SERVING_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#endif

/* Prepared-cache state is intentionally owned by the driver fixture. */
#define TEST_DSV4_SERVING_CACHE_ADMISSION_FREE 0u
#define TEST_DSV4_SERVING_CACHE_ADMISSION_PREPARED 1u
#define TEST_DSV4_SERVING_CACHE_ADMISSION_COMMITTED 2u

typedef struct TestDsv4ServingCacheAdmission
{
	uint32_t state;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverCacheLane cache_lanes[
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
}
TestDsv4ServingCacheAdmission;

typedef struct TestDsv4ServingDriver
{
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t parallel;
	uint32_t needs_hidden_input;
	uint32_t needs_hidden_output;
	uint32_t owns_final_head;
	uint32_t resident_sequence_capacity;
	uint32_t cuda_graph_count;
	uint64_t submitted_count;
	uint64_t completed_count;
	TestDsv4ServingCacheAdmission cache_admissions[
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} TestDsv4ServingDriver;

static SparkStatus TestDsv4ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame);

static const SparkModelDriverProgramProfile TestDsv4ServingDriverProfile =
{
	.descriptor_bytes = sizeof(SparkModelDriverProgramProfile),
	.profile_flags = TEST_DSV4_SERVING_PROGRAM_FLAGS & ~SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION,
	.max_inflight = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_slots = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_new_tokens = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequences = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT,
	.max_sequence_tokens = 1048576u
};

static const SparkModelDriverProgramDescriptor TestDsv4ServingDriverProgram =
{
	.program_id = 1u,
	.flags = TEST_DSV4_SERVING_PROGRAM_FLAGS,
	.max_inflight = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
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
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES || context->stage_count != TEST_DSV4_SERVING_STAGE_COUNT || context->stage_index >= context->stage_count || context->layer_count == 0u || context->stage_pack_path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	driver = (TestDsv4ServingDriver *)calloc(1u,sizeof(*driver));
	if ( driver == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	driver->completion_function = request->completion_function;
	driver->completion_context = request->completion_context;
	driver->stage_index = context->stage_index;
	driver->stage_count = context->stage_count;
	driver->parallel = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_TENSOR_PARALLEL) != 0u ? 1u : 0u;
	driver->needs_hidden_input = context->pp_stage_index > 0u ? 1u : 0u;
	driver->needs_hidden_output = context->pp_stage_index + 1u < context->pp_stage_count ? 1u : 0u;
	driver->owns_final_head = context->first_layer_index + context->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT && context->tp_rank + 1u == context->tp_degree ? 1u : 0u;
	driver->resident_sequence_capacity = context->resident_sequence_capacity;
	driver->cuda_graph_count = context->cuda_graph_count;
	*driver_instance = driver;
	return(SPARK_STATUS_OK);
}

static void TestDsv4ServingDriverDestroy(void *driver_instance)
{
	free(driver_instance);
}

static uint32_t TestDsv4ServingCacheAdmissionIdentityMatches(
	const TestDsv4ServingCacheAdmission *prepared,
	const SparkModelDriverAdmissionRequest *request)
{
	return(prepared->state != TEST_DSV4_SERVING_CACHE_ADMISSION_FREE &&
		prepared->request.submission_id == request->submission_id &&
		prepared->request.control_generation == request->control_generation &&
		prepared->request.transaction_id == request->transaction_id &&
		prepared->request.request_generation == request->request_generation &&
		prepared->request.step_generation == request->step_generation ? 1u : 0u);
}

static uint32_t TestDsv4ServingCacheAdmissionRequestMatches(
	const TestDsv4ServingCacheAdmission *prepared,
	const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverAdmissionRequest *expected;
	expected = &prepared->request;
	return(TestDsv4ServingCacheAdmissionIdentityMatches(prepared,request) != 0u &&
		expected->program_id == request->program_id &&
		expected->request_id == request->request_id &&
		expected->sequence_id == request->sequence_id &&
		expected->sequence_position == request->sequence_position &&
		expected->deadline_time_ns == request->deadline_time_ns &&
		expected->active_slot_count == request->active_slot_count &&
		expected->new_token_count == request->new_token_count &&
		expected->priority == request->priority &&
		expected->frame_flags == request->frame_flags &&
		expected->cache_lane_count == request->cache_lane_count &&
		memcmp(&expected->residency,&request->residency,
			sizeof(expected->residency)) == 0 &&
		memcmp(prepared->cache_lanes,request->cache_lanes,
			(uint64_t)request->cache_lane_count *
			sizeof(prepared->cache_lanes[0])) == 0 ? 1u : 0u);
}

static TestDsv4ServingCacheAdmission *TestDsv4ServingFindCacheAdmission(
	TestDsv4ServingDriver *driver,
	const SparkModelDriverAdmissionRequest *request,
	TestDsv4ServingCacheAdmission **free_record_out)
{
	TestDsv4ServingCacheAdmission *free_record,*prepared;
	uint32_t index;
	free_record = 0;
	for (index=0u;
		index<SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; index++)
	{
		prepared = &driver->cache_admissions[index];
		if ( prepared->state == TEST_DSV4_SERVING_CACHE_ADMISSION_FREE )
		{
			if ( free_record == 0 )
				free_record = prepared;
		}
		else if ( TestDsv4ServingCacheAdmissionIdentityMatches(prepared,
			request) != 0u )
		{
			if ( free_record_out != 0 )
				*free_record_out = free_record;
			return(prepared);
		}
	}
	if ( free_record_out != 0 )
		*free_record_out = free_record;
	return(0);
}

static void TestDsv4ServingAcceptAdmission(
	SparkModelDriverAdmissionDecision *decision)
{
	SparkModelDriverInitializeAdmissionDecision(decision);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->available_dispatch_slot_count =
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
}

static SparkStatus TestDsv4ServingDriverAdmit(
	void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	TestDsv4ServingCacheAdmission *free_record,*prepared;
	TestDsv4ServingDriver *driver;
	uint32_t release;
	driver = (TestDsv4ServingDriver *)driver_instance;
	if ( driver == 0 || request == 0 || decision == 0 ||
		SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	release = (request->frame_flags &
		SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u ? 1u : 0u;
	if ( release != 0u )
	{
		if ( request->admission_flags != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		TestDsv4ServingAcceptAdmission(decision);
		return(SPARK_STATUS_OK);
	}
	prepared = TestDsv4ServingFindCacheAdmission(driver,request,&free_record);
	if ( request->admission_flags ==
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE )
	{
		if ( prepared != 0 )
		{
			if ( TestDsv4ServingCacheAdmissionRequestMatches(prepared,request) == 0u )
				return(SPARK_STATUS_VALIDATION_FAILED);
			if ( prepared->state !=
				TEST_DSV4_SERVING_CACHE_ADMISSION_PREPARED )
				return(SPARK_STATUS_DUPLICATE);
		}
		else
		{
			if ( free_record == 0 )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			memset(free_record,0,sizeof(*free_record));
			free_record->state = TEST_DSV4_SERVING_CACHE_ADMISSION_PREPARED;
			free_record->request = *request;
			memcpy(free_record->cache_lanes,request->cache_lanes,
				(uint64_t)request->cache_lane_count *
				sizeof(free_record->cache_lanes[0]));
			free_record->request.admission_flags = 0u;
			free_record->request.cache_lanes = free_record->cache_lanes;
		}
	}
	else if ( request->admission_flags ==
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT )
	{
		if ( prepared == 0 )
			return(SPARK_STATUS_NOT_FOUND);
		if ( TestDsv4ServingCacheAdmissionRequestMatches(prepared,request) == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		if ( prepared->state != TEST_DSV4_SERVING_CACHE_ADMISSION_PREPARED )
			return(SPARK_STATUS_DUPLICATE);
		prepared->state = TEST_DSV4_SERVING_CACHE_ADMISSION_COMMITTED;
	}
	else if ( request->admission_flags ==
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
	{
		if ( prepared == 0 )
			return(SPARK_STATUS_NOT_FOUND);
		if ( TestDsv4ServingCacheAdmissionRequestMatches(prepared,request) == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		if ( prepared->state != TEST_DSV4_SERVING_CACHE_ADMISSION_PREPARED )
			return(SPARK_STATUS_DUPLICATE);
		memset(prepared,0,sizeof(*prepared));
	}
	else
	{
		if ( request->admission_flags != 0u || prepared == 0 ||
			prepared->state != TEST_DSV4_SERVING_CACHE_ADMISSION_COMMITTED ||
			TestDsv4ServingCacheAdmissionRequestMatches(prepared,request) == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	TestDsv4ServingAcceptAdmission(decision);
	return(SPARK_STATUS_OK);
}

static TestDsv4ServingCacheAdmission *TestDsv4ServingFindFrameAdmission(
	TestDsv4ServingDriver *driver,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	SparkModelDriverAdmissionRequest request;
	TestDsv4ServingCacheAdmission *prepared;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = frame->program_id;
	request.submission_id = context->submission_id;
	request.control_generation = context->control_generation;
	request.transaction_id = context->transaction_id;
	request.request_generation = context->request_generation;
	request.step_generation = context->step_generation;
	request.request_id = frame->request_id;
	request.sequence_id = frame->sequence_id;
	request.sequence_position = frame->sequence_position;
	request.deadline_time_ns = frame->deadline_time_ns;
	request.active_slot_count = frame->active_slot_count;
	request.new_token_count = frame->new_token_count;
	request.priority = frame->priority;
	request.frame_flags = frame->flags;
	request.cache_lane_count = frame->cache_lane_count;
	request.cache_lanes = frame->cache_lanes;
	request.residency = frame->residency;
	prepared = TestDsv4ServingFindCacheAdmission(driver,&request,0);
	return(prepared != 0 &&
		prepared->state == TEST_DSV4_SERVING_CACHE_ADMISSION_COMMITTED &&
		TestDsv4ServingCacheAdmissionRequestMatches(prepared,&request) != 0u ?
		prepared : 0);
}

static SparkStatus TestDsv4ServingDriverSubmit(
	void *driver_instance,
	SparkModelDriverFrame *frame)
{
	TestDsv4ServingDriver *driver;
	TestDsv4ServingCacheAdmission *prepared;
	SparkDsv4ResidentDecodeStageFrameContext *context;
	SparkModelDriverCompletion completion;
	const uint32_t *row_slots;
	const uint64_t *row_sequences;
	uint32_t *tokens,emit,lane,row,row_count;
	uint64_t hidden_bytes;
	driver = (TestDsv4ServingDriver *)driver_instance;
	if ( driver == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		if ( frame->flags != SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE ||
			frame->new_token_count != 0u || frame->user_context != 0 ||
			frame->cache_lane_count != frame->active_slot_count ||
			frame->cache_lanes == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (lane=0u; lane<frame->cache_lane_count; lane++)
			if ( frame->cache_lanes[lane].flags !=
				SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE )
				return(SPARK_STATUS_INVALID_ARGUMENT);
		driver->submitted_count++;
		driver->completed_count++;
		memset(&completion,0,sizeof(completion));
		completion.request_id = frame->request_id;
		completion.sequence_id = frame->sequence_id;
		completion.sequence_position = frame->sequence_position;
		completion.program_id = frame->program_id;
		completion.residency = frame->residency;
		completion.status = SPARK_STATUS_OK;
		frame->completion_function(frame->completion_context,&completion);
		return(SPARK_STATUS_OK);
	}
	if ( frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (SparkDsv4ResidentDecodeStageFrameContext *)frame->user_context;
	prepared = TestDsv4ServingFindFrameAdmission(driver,frame,context);
	if ( prepared == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	row_count = context->decode_batch != 0 ? context->decode_batch->row_count : context->prefill_batch != 0 ? context->prefill_batch->row_count : 0u;
	if ( row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	row_slots = context->decode_batch != 0 ? context->decode_batch->row_lane_indices : context->prefill_batch->row_lane_indices;
	row_sequences = context->decode_batch != 0 ? context->decode_batch->row_sequence_ids : context->prefill_batch->row_sequence_ids;
	for (row=0u; row<row_count; row++)
	{
		uint32_t expected_slot;
		expected_slot = row_sequences[row] == 100u ? driver->resident_sequence_capacity - 1u : row_sequences[row] == 101u ? 3u : row_sequences[row] == 200u ? 5u : UINT32_MAX;
		if ( row_slots[row] != expected_slot )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	hidden_bytes = (uint64_t)row_count * 16384u * sizeof(uint16_t);
	if ( driver->needs_hidden_input != 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < hidden_bytes) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( driver->needs_hidden_output != 0u )
	{
		if ( context->hidden_output_bf16 == 0 || context->hidden_output_bytes < hidden_bytes )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( context->hidden_input_bf16 != 0 )
			memcpy(context->hidden_output_bf16,context->hidden_input_bf16,(size_t)hidden_bytes);
		else
			memset(context->hidden_output_bf16,0x5a,(size_t)hidden_bytes);
	}
	else if ( driver->owns_final_head != 0u )
	{
		if ( frame->buffer_count != 2u || frame->buffers[0].slot != 0u || frame->buffers[0].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_READ || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)row_count * sizeof(uint32_t) || frame->buffers[1].slot != 1u || frame->buffers[1].flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || frame->buffers[1].address == 0 || frame->buffers[1].bytes < (uint64_t)frame->active_slot_count * sizeof(uint32_t) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( context->prefill_batch != 0 && context->prefill_batch->token_ids != frame->buffers[0].address )
			return(SPARK_STATUS_SCHEMA_ERROR);
		tokens = (uint32_t *)frame->buffers[1].address;
		memset(tokens,0,(size_t)frame->active_slot_count * sizeof(uint32_t));
		if ( context->prefill_batch != 0 )
		{
			for (emit=0u; emit<context->prefill_batch->emit_count; emit++)
				tokens[context->prefill_batch->emit_lane_indices[emit]] = 4200u + context->prefill_batch->emit_row_indices[emit];
		}
		else
			for (row=0u; row<row_count; row++)
				tokens[row] = 4200u + row;
	}
	else if ( frame->buffer_count != 1u ||
		context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(prepared,0,sizeof(*prepared));
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
	snapshot->available_dispatch_slot_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
	snapshot->submitted_count = driver->submitted_count;
	snapshot->completed_count = driver->completed_count;
	/* Observation channel for the adapter test's graph opt-in assertions:
	 * kv_token_capacity is otherwise unused by this fixture. */
	snapshot->kv_token_capacity = driver->cuda_graph_count;
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
