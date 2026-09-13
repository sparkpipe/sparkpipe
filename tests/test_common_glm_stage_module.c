#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_head_screen.h"

#define SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 8u

typedef struct TestPackRange
{
	uint64_t offset;
	uint64_t bytes;
} TestPackRange;

typedef struct TestEntry
{
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} TestEntry;

typedef enum
{
	TEST_TENSOR_EMBEDDING = 0,
	TEST_TENSOR_FINAL_NORM = 1,
	TEST_TENSOR_FIRST = 2,
	TEST_TENSOR_LM_HEAD = 3,
	TEST_TENSOR_KIND_COUNT = 6
} TestTensorKind;

typedef struct
{
	uint32_t rows;
	uint32_t columns;
} TestTensorShape;

typedef struct TestState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_index;
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
	uint32_t resident_sequence_capacity;
	uint32_t execution_row_capacity;
	uint32_t pipeline_slot_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	atomic_uint lane_states[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	struct TestCompletion *completions;
	struct TestSlot *slots;
	const void *lm_head_bf16;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
} TestState;

typedef struct TestCompletion
{
	TestState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkModelDriverCompletion completion;
} TestCompletion;

typedef struct TestSlot
{
	void *stream;
	uint32_t *host_token_ids;
	uint32_t *host_resident_slots;
	uint32_t *host_positions;
	uint32_t *host_kv_access_error;
	uint32_t *kv_access_error;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
	uint64_t *head_maxloc_u64;
	void *head_certified_scratch;
	uint32_t *head_certified_candidates;
	uint32_t *head_screened_count;
} TestSlot;

struct TestSlot;

cudaError_t SparkGlmLaunchHeadCertifiedQuantize(cudaStream_t stream,const void *head_bf16,uint8_t *certified_payload,float *certified_scale_f32,float *certified_norm_f32,uint32_t vocabulary,uint32_t hidden_dimension);

typedef struct TestBatch
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t active_sequence_count;
	const uint32_t *token_ids;
	const uint32_t *row_resident_slots;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} TestBatch;

typedef struct TestFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const TestBatch *batch;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
	const void *sideband_input;
	uint64_t sideband_input_bytes;
	void *sideband_output;
	uint64_t sideband_output_bytes;
} TestFrameContext;

typedef struct TestChain
{
	SparkStatus retained_status;
} TestChain;

#define SPARK_GLM_STAGE_STATE TestState
#define SPARK_GLM_STAGE_SLOT TestSlot
#define SPARK_GLM_STAGE_COMPLETION TestCompletion
#define SPARK_GLM_STAGE_BATCH TestBatch
#define SPARK_GLM_STAGE_FRAME TestFrameContext
#define SPARK_GLM_STAGE_CHAIN TestChain
#define SPARK_GLM_STAGE_ENTRY TestEntry
#define SPARK_GLM_STAGE_TENSOR_SHAPE TestTensorShape
#define SPARK_GLM_STAGE_PACK_RANGE TestPackRange
#define SPARK_GLM_STAGE_TENSOR_FIRST TEST_TENSOR_FIRST
#define SPARK_GLM_STAGE_TENSOR_KIND_COUNT TEST_TENSOR_KIND_COUNT
#define SPARK_GLM_STAGE_TENSOR_EMBEDDING TEST_TENSOR_EMBEDDING
#define SPARK_GLM_STAGE_TENSOR_FINAL_NORM TEST_TENSOR_FINAL_NORM
#define SPARK_GLM_STAGE_TENSOR_LM_HEAD TEST_TENSOR_LM_HEAD
#define SPARK_GLM_STAGE_EXPECTED_SHAPE(kind,layer_index,codec,tp_degree,shape) \
	TestExpectedShape(kind,layer_index,codec,tp_degree,shape)
#define SPARK_GLM_STAGE_MODULE_TAG "test_glm_stage"
#define SPARK_GLM_STAGE_KV_ACCESS_ERROR_WORD_COUNT 6u
#define SPARK_GLM_STAGE_MAX_INPUT_ROW_COUNT 64u
#define SPARK_GLM_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_GLM_STAGE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_GLM_STAGE_FRAME_KNOWN_FLAGS 0x1Fu
#define SPARK_GLM_STAGE_FRAME_FLAG_PREFILL 0x01u
#define SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_INPUT 0x02u
#define SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_OUTPUT 0x04u
#define SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_INPUT 0x08u
#define SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT 0x10u
#define SPARK_GLM_STAGE_BOUNDARY_ELEMENT_COUNT 12288u
#define SPARK_GLM_STAGE_BOUNDARY_ELEMENT_BYTES 2u
#define SPARK_GLM_STAGE_DSA_SIDEBAND_BYTES_PER_ROW 8192u
#define SPARK_GLM_STAGE_REQUIRES_SIDEBAND_INPUT(stage_index) 0u
#define SPARK_GLM_STAGE_REQUIRES_SIDEBAND_OUTPUT(stage_index) 0u
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_HOST(slot) SPARK_STATUS_OK
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_METADATA(state,slot) SPARK_STATUS_OK
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_HIDDEN(state,slot) SPARK_STATUS_OK
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_MLP(state,slot) SPARK_STATUS_OK
#define SPARK_GLM_STAGE_COMPLETE_ASYNC 0
#define SPARK_GLM_STAGE_VALIDATE_FRAME_BUFFERS(state,frame,row_count) \
	SPARK_STATUS_OK
#define SPARK_GLM_STAGE_LAZY_RECOVER_LEASE(state,slot,out) SPARK_STATUS_IO_ERROR
#define SPARK_GLM_STAGE_TP_CHAIN_FAIL(chain,status) ((void)(chain),((void)(status)))

#define SPARK_LLM_OUTPUT_VOCAB_COUNT 154880u
#define SPARK_LLM_HEAD_TILE 1024u
#define SPARK_LLM_HIDDEN_DIMENSION 6144u

static int32_t TestExpectedShape(uint32_t kind,uint32_t layer_index,
	uint32_t codec,uint32_t tp_degree,TestTensorShape *shape)
{
	(void)layer_index;
	(void)codec;
	(void)tp_degree;
	if ( (kind & 1u) != 0u )
		return(-1);
	shape->rows = kind;
	shape->columns = 1u;
	return(0);
}

static void TestCompletionSink(void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	(void)completion_context;
	(void)completion;
}

static uint32_t failures;

#define CHECK(condition,label) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("  FAIL %s\n",label); \
			failures++; \
		} \
	} while (0)

#include "common/common_glm_stage_module/spark_glm_stage_module.h"

int main(void)
{
	TestEntry entries[2];
	TestState state;
	TestBatch batch;
	uint32_t tokens[4] = {7u,7u,7u,7u};
	uint32_t slots[4] = {0u,1u,2u,3u};
	uint64_t positions[4] = {0u,1u,2u,3u};
	uint64_t sequence_ids[4] = {11u,11u,11u,11u};
	TestSlot slot;
	TestCompletion completions[1];
	uint32_t host_token_ids[4],host_resident_slots[4],host_positions[4];
	uint32_t host_kv_access_error[6];
	uint8_t lane_bound[4] = {1u,1u,1u,1u};
	SparkModelDriverFrame frame;
	SparkModelDriverBuffer buffer;
	TestFrameContext context;
	const TestFrameContext *seen;
	void *unused_reference;
	memset(entries,0,sizeof(entries));
	memset(&state,0,sizeof(state));
	memset(&batch,0,sizeof(batch));
	memset(&slot,0,sizeof(slot));
	memset(completions,0,sizeof(completions));
	memset(&frame,0,sizeof(frame));
	memset(&buffer,0,sizeof(buffer));
	memset(&context,0,sizeof(context));
	(void)SparkGlmStageAllocateRows;
	(void)SparkGlmStageAllocateSlotHead;
	(void)SparkGlmStageAllocateSlots;
	(void)SparkGlmStageBuildHeadShadow;
	(void)SparkGlmStageEnqueueAsyncCompletion;
	(void)SparkGlmStageLazyRetryRetained;
	(void)SparkGlmStageRoundMajorWaveRows;

	entries[0].payload_offset = 0u;
	entries[0].payload_bytes = 100u;
	entries[0].scale_offset = 100u;
	entries[0].scale_bytes = 10u;
	entries[1] = entries[0];
	entries[1].payload_offset = 50u;
	CHECK(SparkGlmStagePackValidateRanges(entries,2) == SPARK_STATUS_SCHEMA_ERROR,"overlapping payload ranges must fail schema");
	entries[1].payload_offset = 110u;
	entries[1].scale_offset = 210u;
	entries[1].scale_bytes = 10u;
	CHECK(SparkGlmStagePackValidateRanges(entries,2) == SPARK_STATUS_OK,"disjoint ranges must pass");
	{
		TestPackRange left = {10u,20u};
		TestPackRange right = {20u,10u};
		TestPackRange far = {100u,10u};
		CHECK(SparkGlmStagePackRangesOverlap(&left,&right) == 1u,"adjacent ranges sharing a boundary overlap");
		CHECK(SparkGlmStagePackRangesOverlap(&left,&far) == 0u,"disjoint ranges never overlap");
	}

	state.expert_weight_codec = 5u;
	state.tp_degree = 4u;
	state.resident_sequence_capacity = 8u;
	state.owns_embedding = 1u;
	state.owns_final_head = 1u;
	CHECK(SparkGlmStageExpectedGlobalMask(&state) ==
		((UINT64_C(1) << TEST_TENSOR_EMBEDDING) |
		(UINT64_C(1) << TEST_TENSOR_FINAL_NORM) |
		(UINT64_C(1) << TEST_TENSOR_LM_HEAD)),"global mask covers owned globals");
	CHECK(SparkGlmStageExpectedLayerMask(&state,0u) == 0x14u,
		"layer mask covers even tensor kinds only");

	batch.abi_version = 1u;
	batch.descriptor_bytes = sizeof(batch);
	batch.row_count = 4u;
	batch.active_sequence_count = 4u;
	batch.token_ids = tokens;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequence_ids;
	slot.host_token_ids = host_token_ids;
	slot.host_resident_slots = host_resident_slots;
	slot.host_positions = host_positions;
	slot.host_kv_access_error = host_kv_access_error;
	host_kv_access_error[0] = 99u;
	state.owns_embedding = 1u;
	CHECK(SparkGlmStageStageHostBatch(&state,&slot,&batch) == SPARK_STATUS_OK,"host batch staging passes");
	CHECK(host_token_ids[3] == 7u && host_positions[3] == 3u && host_resident_slots[3] == 3u,"staged row fields match the batch");
	CHECK(host_kv_access_error[0] == 0u,"kv access error words are cleared");

	state.execution_row_capacity = 64u;
	state.pipeline_slot_count = 1u;
	state.completions = completions;
	frame.buffers = &buffer;
	frame.buffer_count = 1u;
	SparkGlmStagePrepareAsyncCompletion(&state,&frame,&batch,lane_bound,sequence_ids,positions,0u);
	CHECK(completions[0].lane_count == 4u && completions[0].row_count == 4u,"async completion carries the batch shape");
	CHECK(completions[0].lane_indices[2] == 2u && completions[0].lane_sequence_ids[0] == 11u && completions[0].lane_next_positions[2] == 2u,"async completion carries lane state");
	CHECK(completions[0].completion.host_staging_bytes ==
		(UINT64_C(4) * sizeof(uint32_t) * 4u),"host staging bytes count rows");
	CHECK(SparkGlmStageAllocateBytes(&state,0u,1u,1u,&unused_reference) == SPARK_STATUS_CAPACITY_EXCEEDED,"zero count allocation is rejected before device work");
	CHECK(SparkGlmStageAllocateBytes(&state,UINT64_MAX,2u,1u,&unused_reference) == SPARK_STATUS_CAPACITY_EXCEEDED,"overflowing allocation is rejected before device work");

	{
		TestBatch bad_batch = batch;
		CHECK(SparkGlmStageValidateRoundMajor(&state,&batch) == SPARK_STATUS_OK,"round major layout of the batch is valid");
		bad_batch.row_count = 2u;
		bad_batch.active_sequence_count = 4u;
		CHECK(SparkGlmStageValidateRoundMajor(&state,&bad_batch) == SPARK_STATUS_INVALID_ARGUMENT,"rows below active sequences must be rejected");
	}

	seen = NULL;
	state.owns_embedding = 1u;
	state.owns_final_head = 1u;
	state.stage_index = 0u;
	state.resident_sequence_capacity = 8u;
	state.execution_stream = (void *)0x2;
	frame.execution_stream = (void *)0x2;
	frame.completion_function = TestCompletionSink;
	frame.user_context = &context;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame.active_slot_count = 4u;
	frame.new_token_count = 4u;
	frame.buffers = &buffer;
	frame.buffer_count = 1u;
	context.abi_version = SPARK_GLM_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context.descriptor_bytes = sizeof(context);
	context.batch = &batch;
	context.flags = SPARK_GLM_STAGE_FRAME_FLAG_PREFILL;
	CHECK(SparkGlmStageValidateFrame(&state,&frame,&seen) == SPARK_STATUS_OK && seen == &context,"valid frame passes and is returned");
	context.hidden_output_bf16 = (void *)0x3;
	context.hidden_output_bytes = UINT64_C(4) * SPARK_GLM_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM_STAGE_BOUNDARY_ELEMENT_BYTES;
	seen = NULL;
	CHECK(SparkGlmStageValidateFrame(&state,&frame,&seen) == SPARK_STATUS_CAPACITY_EXCEEDED,"final head stage must not accept hidden output");
	context.hidden_output_bf16 = 0;
	context.hidden_output_bytes = 0u;
	context.abi_version = 99u;
	seen = NULL;
	CHECK(SparkGlmStageValidateFrame(&state,&frame,&seen) == SPARK_STATUS_ABI_MISMATCH && seen == NULL,"wrong frame abi must fail with ABI_MISMATCH");
	context.abi_version = SPARK_GLM_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context.flags = 0u;
	seen = NULL;
	CHECK(SparkGlmStageValidateFrame(&state,&frame,&seen) == SPARK_STATUS_SCHEMA_ERROR,"missing prefill flag must fail with SCHEMA_ERROR");

	if ( failures == 0u )
		printf("PASS common_glm_stage_module functions behave per contract\n");
	return failures == 0u ? 0 : 1;
}
