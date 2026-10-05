#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#define PROBE_LANES 2u
#define PROBE_PREFIX 63u
#define PROBE_CONTINUATION 4u
#define PROBE_TARGET "cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8"

typedef struct probe_state
{
	SparkLoadedModelDriver driver;
	const SparkModelDriverProgramDescriptor *program;
	void *instance;
	cudaStream_t stream;
	SparkGlm52ResidentDecodeStageNodeContext node;
	SparkGlm52ResidentDecodeStageBatchView batch;
	SparkGlm52ResidentDecodeStageFrameContext context;
	SparkModelDriverFrame frame;
	SparkModelDriverBuffer buffer;
	SparkModelDriverCompletion completion;
	SparkModelDriverCacheLane lanes[PROBE_LANES];
	atomic_uint completed;
	uint64_t next_request_id,control_generation;
	uint32_t tokens[PROBE_LANES],slots[PROBE_LANES],outputs[PROBE_LANES];
	uint64_t positions[PROBE_LANES],sequences[PROBE_LANES];
} probe_state_t;

static uint64_t probe_time(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		_Exit(120);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
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
	uint64_t deadline = probe_time() + UINT64_C(300000000000);
	while ( probe_time() < deadline )
	{
		if ( atomic_load_explicit(&state->completed,memory_order_acquire) != 0u )
		{
			memset(&snapshot,0,sizeof(snapshot));
			if ( state->driver.interface->snapshot(state->instance,state->program->program_id,&snapshot) != SPARK_STATUS_OK )
				return(-1);
			if ( snapshot.active_submission_count == 0u )
				return(state->completion.status == SPARK_STATUS_OK ? 0 : -2);
		}
		(void)nanosleep(&pause,0);
	}
	return(-3);
}

static int32_t probe_open(probe_state_t *state,const char *driver,const char *pack)
{
	SparkModelDriverCreateRequest request;
	SparkGlm52ResidentDecodeStageNodeContext *node = &state->node;
	char error[512];
	SparkStatus status = SparkLoadModelDriver(driver,PROBE_TARGET,&state->driver,error,sizeof(error));
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
	node->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->descriptor_bytes = sizeof(*node);
	node->stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT;
	node->first_layer_index = SparkGlm52ResidentDecodeStageFirstLayer(0u);
	node->layer_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE;
	node->expert_weight_codec = GLM_EXPERT_WEIGHT_CODEC;
	node->resident_sequence_capacity = 2u * PROBE_LANES;
	node->pipeline_slot_count = 1u;
	node->max_sequence_positions = 128u;
	node->execution_row_capacity = PROBE_LANES;
	node->tp_degree = 16u;
	node->tp_rank = 0u;
	node->stage_pack_path = pack;
	node->model_revision = state->driver.interface->descriptor->model_revision;
	node->tp_collective_identifier = 0u;
	node->kv_backing_directory = "/tmp/prefixprobe/kv";
	node->kv_snapshot_directory = "/tmp/prefixprobe/snapshot";
	node->kv_snapshot_maximum_bytes = 1ull << 30;
	(void)mkdir("/tmp/prefixprobe",0700);
	(void)mkdir(node->kv_backing_directory,0700);
	(void)mkdir(node->kv_snapshot_directory,0700);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = "glm52-local-prefix-probe";
	request.node_target = PROBE_TARGET;
	request.node_context = node;
	request.execution_stream = state->stream;
	request.kv_logical_page_capacity = node->resident_sequence_capacity * (node->max_sequence_positions / 64u);
	request.kv_physical_page_capacity = request.kv_logical_page_capacity;
	request.completion_function = probe_complete;
	request.completion_context = state;
	status = state->driver.interface->create(&request,&state->instance);
	fprintf(stderr,"create status=%d revision=%s collective=disabled tp=16 rank=0\n",status,node->model_revision);
	return(status == SPARK_STATUS_OK && state->instance != 0 ? 0 : -7);
}

static void probe_frame(probe_state_t *state,uint32_t step,uint32_t restored)
{
	SparkModelDriverFrame *frame = &state->frame;
	uint32_t lane;
	for (lane=0u; lane<PROBE_LANES; lane++)
	{
		state->tokens[lane] = 1u + lane * 1000u + step;
		state->slots[lane] = restored != 0u ? PROBE_LANES + lane : lane;
		state->positions[lane] = step;
		state->sequences[lane] = restored != 0u ? restored * PROBE_LANES + lane + 1u : lane + 1u;
		state->outputs[lane] = UINT32_MAX;
		state->lanes[lane] = (SparkModelDriverCacheLane){.sequence_id=state->sequences[lane],.sequence_position=step,.request_generation=1u,.step_generation=step + 1u,
			.resident_sequence_slot=state->slots[lane],.context_token_count=step + 1u};
		if ( (restored == 0u && step + 1u == PROBE_PREFIX) || step + 1u == 64u )
		{
			state->lanes[lane].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
			state->lanes[lane].publish_token_count = step + 1u;
			state->lanes[lane].publish_identity.sha256[0] = (uint8_t)(lane + 1u);
			state->lanes[lane].publish_identity.sha256[1] = (uint8_t)(step + 1u);
		}
		if ( restored != 0u && step == PROBE_PREFIX )
		{
			state->lanes[lane].flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
			state->lanes[lane].prefix_token_count = PROBE_PREFIX;
			state->lanes[lane].prefix_identity.sha256[0] = (uint8_t)((restored == 2u ? lane ^ 1u : lane) + 1u);
			state->lanes[lane].prefix_identity.sha256[1] = PROBE_PREFIX;
		}
	}
	state->batch = (SparkGlm52ResidentDecodeStageBatchView){.abi_version=SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION,.descriptor_bytes=sizeof(state->batch),
		.row_count=PROBE_LANES,.active_sequence_count=PROBE_LANES,.token_ids=state->tokens,.row_resident_slots=state->slots,.row_positions=state->positions,.row_sequence_ids=state->sequences};
	state->context = (SparkGlm52ResidentDecodeStageFrameContext){.abi_version=SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION,.descriptor_bytes=sizeof(state->context),
		.flags=step == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u,.batch=&state->batch};
	memset(frame,0,sizeof(*frame));
	state->buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	state->buffer.address = state->outputs;
	state->buffer.bytes = PROBE_LANES * sizeof(uint32_t);
	frame->request_id = ++state->next_request_id;
	frame->sequence_id = state->sequences[0];
	frame->sequence_position = step;
	frame->active_slot_count = PROBE_LANES;
	frame->new_token_count = PROBE_LANES;
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
	frame->cache_lanes = state->lanes;
	frame->cache_lane_count = PROBE_LANES;
}

static SparkStatus probe_request(probe_state_t *state,SparkModelDriverAdmissionRequest *request)
{
	SparkStatus status = SparkAdmissionRequestFromFrame(state->program->program_id,&state->frame,state->lanes,0u,request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request->submission_id = state->frame.request_id;
	request->control_generation = state->control_generation;
	request->transaction_id = state->frame.request_id;
	request->request_generation = 1u;
	request->step_generation = state->frame.request_id;
	return(SPARK_STATUS_OK);
}

static SparkStatus probe_submit(probe_state_t *state)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status = probe_request(state,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	request.admission_flags = 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionEvaluateAndApply(state->driver.interface,state->instance,&request,&state->frame,&decision);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->instance,&state->frame);
	if ( status != SPARK_STATUS_OK )
	{
		request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
		(void)SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	}
	return(status);
}

static int32_t probe_step(probe_state_t *state,uint32_t step,uint32_t restored)
{
	SparkStatus status;
	uint32_t lane;
	probe_frame(state,step,restored);
	atomic_store_explicit(&state->completed,0u,memory_order_release);
	status = probe_submit(state);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"submit step=%u restored=%u status=%d\n",step,restored,status);
		return(-8);
	}
	if ( probe_wait(state) != 0 )
	{
		fprintf(stderr,"wait step=%u restored=%u status=%d\n",step,restored,state->completion.status);
		return(-9);
	}
	for (lane=0u; lane<PROBE_LANES; lane++)
		if ( state->outputs[lane] >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT )
			return(-10);
	return(0);
}

static int32_t probe_release(probe_state_t *state,uint32_t restored,uint32_t position)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	uint32_t lane;
	probe_frame(state,position,restored);
	state->frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	state->frame.new_token_count = 0u;
	for (lane=0u; lane<PROBE_LANES; lane++)
	{
		state->lanes[lane].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
		state->lanes[lane].prefix_token_count = 0u;
		state->lanes[lane].publish_token_count = 0u;
		state->lanes[lane].sequence_position = position;
		state->lanes[lane].context_token_count = position;
	}
	if ( probe_request(state,&request) != SPARK_STATUS_OK ||
		SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision) != SPARK_STATUS_OK ||
		state->driver.interface->snapshot(state->instance,state->program->program_id,&snapshot) != SPARK_STATUS_OK ||
		snapshot.active_submission_count != 0u || snapshot.resident_sequence_count != 0u )
		return(-12);
	return(0);
}

static int32_t probe_reset(probe_state_t *state)
{
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	request.descriptor_bytes = sizeof(request);
	request.program_id = state->program->program_id;
	request.control_generation = state->control_generation + 1u;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	if ( SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision) != SPARK_STATUS_OK )
		return(-13);
	state->control_generation = request.control_generation;
	return(0);
}

static int32_t probe_run(probe_state_t *state)
{
	uint32_t expected[PROBE_CONTINUATION][PROBE_LANES],step,lane;
	int32_t result = 0;
	for (step=0u; result == 0 && step<PROBE_PREFIX + PROBE_CONTINUATION; step++)
	{
		result = probe_step(state,step,0u);
		if ( result == 0 && step >= PROBE_PREFIX )
			memcpy(expected[step - PROBE_PREFIX],state->outputs,sizeof(expected[0]));
	}
	if ( result == 0 )
		result = probe_release(state,0u,PROBE_PREFIX + PROBE_CONTINUATION);
	for (step=0u; result == 0 && step<PROBE_CONTINUATION; step++)
	{
		result = probe_step(state,PROBE_PREFIX + step,1u);
		for (lane=0u; result == 0 && lane<PROBE_LANES; lane++)
		{
			printf("TOKEN position=%u lane=%u uninterrupted=%u restored=%u\n",PROBE_PREFIX + step,lane,expected[step][lane],state->outputs[lane]);
			if ( expected[step][lane] != state->outputs[lane] )
				result = -14;
		}
	}
	if ( result == 0 )
		result = probe_release(state,1u,PROBE_PREFIX + PROBE_CONTINUATION);
	if ( result == 0 )
	{
		uint32_t differ = 0u;
		result = probe_step(state,PROBE_PREFIX,2u);
		for (lane=0u; result == 0 && lane<PROBE_LANES; lane++)
		{
			printf("SWAPPED position=%u lane=%u uninterrupted=%u with-other-lane-prefix=%u\n",PROBE_PREFIX,lane,expected[0][lane],state->outputs[lane]);
			differ += expected[0][lane] != state->outputs[lane] ? 1u : 0u;
		}
		printf("SENSITIVITY swapped-prefix lanes-changed=%u of %u\n",differ,PROBE_LANES);
		if ( result == 0 )
			result = probe_release(state,2u,PROBE_PREFIX + 1u);
	}
	if ( result == 0 )
		result = probe_reset(state);
	if ( result != 0 )
		return(result);
	probe_frame(state,PROBE_PREFIX,1u);
	if ( probe_submit(state) != SPARK_STATUS_NOT_FOUND )
		return(-15);
	return(0);
}

int main(int argc,char **argv)
{
	probe_state_t state;
	int32_t result;
	if ( argc != 3 )
	{
		fprintf(stderr,"usage: %s DRIVER TP16_RANK0_PACK (with SPARK_WEIGHTD_SOCKET set)\n",argv[0]);
		return(2);
	}
	memset(&state,0,sizeof(state));
	state.control_generation = 1u;
	atomic_init(&state.completed,0u);
	result = probe_open(&state,argv[1],argv[2]);
	if ( result == 0 )
		result = probe_run(&state);
	if ( result != 0 )
	{
		fprintf(stderr,"probe failed result=%d\n",result);
		fflush(0);
		_Exit(4);
	}
	state.driver.interface->destroy(state.instance);
	(void)cudaStreamDestroy(state.stream);
	SparkUnloadModelDriver(&state.driver);
	printf("PASS glm52 prefix restore lanes=%u prefix=%u continuation=%u restored-slot tokens=exact reset=prefix-gone; rank-local computation only\n",PROBE_LANES,PROBE_PREFIX,PROBE_CONTINUATION);
	return(0);
}
