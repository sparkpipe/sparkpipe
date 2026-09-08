#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver_support.h"
#define GLM5_NEXT_EXPERT_CODEC_NAME "fp8"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"

#define PROBE_ROWS 3u
#define PROBE_STEPS 4u
#define PROBE_TARGET "cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_fp8"

typedef struct probe_state
{
	SparkLoadedModelDriver driver;
	const SparkModelDriverProgramDescriptor *program;
	void *instance;
	cudaStream_t stream;
	SparkGlm5NextResidentDecodeStageNodeContext node;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkModelDriverFrame frame;
	SparkModelDriverBuffer buffer;
	SparkModelDriverCompletion completion;
	atomic_uint completed;
	uint32_t tokens[PROBE_ROWS],slots[PROBE_ROWS],outputs[PROBE_ROWS];
	uint64_t positions[PROBE_ROWS],sequences[PROBE_ROWS];
} probe_state_t;

static uint64_t probe_time(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		_Exit(120);
	return(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
}

static void probe_complete(void *opaque,const SparkModelDriverCompletion *completion)
{
	probe_state_t *state = opaque;
	state->completion = *completion;
	atomic_store_explicit(&state->completed,1u,memory_order_release);
}

static int32_t probe_wait(probe_state_t *state)
{
	SparkModelDriverRuntimeSnapshot snapshot;
	struct timespec pause = {0,1000000};
	uint64_t deadline = (probe_time() + UINT64_C(120000000000));
	SparkStatus status;
	for (;;)
	{
		if ( atomic_load_explicit(&state->completed,memory_order_acquire) != 0u )
		{
			memset(&snapshot,0,sizeof(snapshot));
			status = state->driver.interface->snapshot(state->instance,state->program->program_id,&snapshot);
			if ( status != SPARK_STATUS_OK )
				return(-1);
			if ( snapshot.active_submission_count == 0u )
				return(state->completion.status == SPARK_STATUS_OK ? 0 : -2);
		}
		if ( probe_time() >= deadline )
			return(-3);
		(void)nanosleep(&pause,0);
	}
}

static void probe_node(probe_state_t *state,const char *pack,uint32_t rows)
{
	SparkGlm5NextResidentDecodeStageNodeContext *node = &state->node;
	node->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->descriptor_bytes = sizeof(*node);
	node->stage_count = 1u;
	node->layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	node->expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	node->resident_sequence_capacity = rows;
	node->pipeline_slot_count = 1u;
	node->max_sequence_positions = 64u;
	node->execution_row_capacity = rows;
	node->tp_degree = 16u;
	node->tp_rank = 0u;
	node->stage_pack_path = pack;
	node->model_revision = state->driver.interface->descriptor->model_revision;
	// Explicit local differential execution: no collective transport.
	node->tp_collective_identifier = 0u;
}

static int32_t probe_open(probe_state_t *state,const char *driver,const char *pack,uint32_t rows)
{
	SparkModelDriverCreateRequest request;
	char error[512];
	SparkStatus status;
	status = SparkLoadModelDriver(driver,PROBE_TARGET,&state->driver,error,sizeof(error));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"load status=%d error=%s\n",status,error);
		return(-4);
	}
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,"resident_decode");
	if ( state->program == 0 || state->program->submit == 0 || state->driver.interface->snapshot == 0 )
		return(-5);
	if ( cudaStreamCreateWithFlags(&state->stream,cudaStreamNonBlocking) != cudaSuccess )
		return(-6);
	probe_node(state,pack,rows);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = "glm-local-probe";
	request.node_target = PROBE_TARGET;
	request.node_context = &state->node;
	request.execution_stream = state->stream;
	request.completion_function = probe_complete;
	request.completion_context = state;
	status = state->driver.interface->create(&request,&state->instance);
	fprintf(stderr,"create status=%d revision=%s collective=disabled tp=16 rank=0\n",status,state->node.model_revision);
	return(status == SPARK_STATUS_OK && state->instance != 0 ? 0 : -7);
}

static void probe_batch(probe_state_t *state,uint32_t rows,uint32_t step)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		state->tokens[row] = (1u + (row * PROBE_STEPS) + step);
		state->slots[row] = row;
		state->positions[row] = step;
		state->sequences[row] = (row + 1u);
		state->outputs[row] = UINT32_MAX;
	}
	state->batch.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	state->batch.descriptor_bytes = sizeof(state->batch);
	state->batch.row_count = rows;
	state->batch.active_sequence_count = rows;
	state->batch.token_ids = state->tokens;
	state->batch.row_resident_slots = state->slots;
	state->batch.row_positions = state->positions;
	state->batch.row_sequence_ids = state->sequences;
	state->context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	state->context.descriptor_bytes = sizeof(state->context);
	state->context.flags = step == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	state->context.batch = &state->batch;
}

static void probe_frame(probe_state_t *state,uint32_t rows,uint32_t step)
{
	SparkModelDriverFrame *frame = &state->frame;
	memset(frame,0,sizeof(*frame));
	state->buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	state->buffer.address = state->outputs;
	state->buffer.bytes = ((uint64_t)rows * sizeof(uint32_t));
	frame->request_id = (step + 1u);
	frame->sequence_id = 1u;
	frame->sequence_position = step;
	frame->active_slot_count = rows;
	frame->new_token_count = rows;
	frame->tokens_per_sequence = 1u;
	frame->flags = step == 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->stream;
	frame->buffers = &state->buffer;
	frame->buffer_count = 1u;
	frame->user_context = &state->context;
	frame->completion_function = probe_complete;
	frame->completion_context = state;
}

static int32_t probe_step(probe_state_t *state,uint32_t rows,uint32_t step)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	uint32_t row;
	int32_t result;
	probe_batch(state,rows,step);
	probe_frame(state,rows,step);
	atomic_store_explicit(&state->completed,0u,memory_order_release);
	status = SparkAdmissionRequestFromFrame(state->program->program_id,&state->frame,0,0u,&request);
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionEvaluateAndApply(state->driver.interface,state->instance,&request,&state->frame,&decision);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->instance,&state->frame);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"submit step=%u status=%d\n",step,status);
		return(-8);
	}
	result = probe_wait(state);
	if ( result != 0 )
	{
		fprintf(stderr,"wait step=%u result=%d\n",step,result);
		return(-9);
	}
	if ( state->completion.request_id != state->frame.request_id || state->completion.program_id != state->frame.program_id || state->completion.sequence_id != state->frame.sequence_id || state->completion.sequence_position != state->frame.sequence_position )
		return(-11);
	for (row=0u; row<rows; row++)
	{
		if ( state->outputs[row] >= SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT )
			return(-10);
		printf("TOKEN step=%u row=%u input=%u output=%u\n",step,row,state->tokens[row],state->outputs[row]);
	}
	return(0);
}

int main(int argc,char **argv)
{
	probe_state_t state;
	uint32_t rows,step;
	int32_t result;
	const char *socket;
	if ( argc != 5 || (strcmp(argv[3],"resident") != 0 && strcmp(argv[3],"lazy") != 0) || (strcmp(argv[4],"1") != 0 && strcmp(argv[4],"3") != 0) )
	{
		fprintf(stderr,"usage: %s DRIVER TP16_RANK0_PACK resident|lazy 1|3\n",argv[0]);
		return(2);
	}
	socket = getenv("SPARK_WEIGHTD_SOCKET");
	if ( (strcmp(argv[3],"lazy") == 0) != (socket != 0 && socket[0] != '\0') )
	{
		fprintf(stderr,"mode disagrees with SPARK_WEIGHTD_SOCKET; refusing ambiguous loading\n");
		return(3);
	}
	rows = (uint32_t)(argv[4][0] - '0');
	memset(&state,0,sizeof(state));
	atomic_init(&state.completed,0u);
	result = probe_open(&state,argv[1],argv[2],rows);
	for (step=0u; result == 0 && step<PROBE_STEPS; step++)
		result = probe_step(&state,rows,step);
	if ( result != 0 )
	{
		fprintf(stderr,"probe failed result=%d; process exit preserves uncertain in-flight ownership\n",result);
		fflush(0);
		_Exit(4);
	}
	state.driver.interface->destroy(state.instance);
	if ( cudaStreamDestroy(state.stream) != cudaSuccess )
		return(5);
	SparkUnloadModelDriver(&state.driver);
	printf("PASS local-token-smoke mode=%s rows=%u steps=%u; not full-model numerical qualification\n",argv[3],rows,PROBE_STEPS);
	return(0);
}
