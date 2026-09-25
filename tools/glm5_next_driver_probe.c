#include <stdint.h>
#include <math.h>
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

#define PROBE_ROWS 5u
#define PROBE_STEPS 4u
#define PROBE_INPUT_ROWS (PROBE_ROWS * PROBE_STEPS)
#define PROBE_PREFIX_TOKENS 63u
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
	SparkModelDriverCacheLane cache_lanes[PROBE_ROWS];
	atomic_uint completed;
	uint32_t prefix_probe;
	SparkGlm5NextStateCapture capture;
	SparkGlm5NextStateCaptureLane captured_lanes[PROBE_ROWS];
	uint32_t logical_pages[PROBE_ROWS * 2u],physical_pages[PROBE_ROWS * 2u];
	uint64_t next_request_id,control_generation;
	uint32_t tokens[PROBE_INPUT_ROWS],slots[PROBE_INPUT_ROWS],outputs[PROBE_INPUT_ROWS];
	uint64_t positions[PROBE_INPUT_ROWS],sequences[PROBE_INPUT_ROWS];
	SparkRowSampling sampling[PROBE_INPUT_ROWS];
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
	node->resident_sequence_capacity = state->prefix_probe != 0u ? 2u * rows : rows;
	node->pipeline_slot_count = 1u;
	node->max_sequence_positions = state->prefix_probe != 0u ? 128u : 64u;
	node->execution_row_capacity = state->prefix_probe != 0u ? rows * PROBE_STEPS : rows;
	node->tp_degree = 16u;
	node->tp_rank = 0u;
	node->stage_pack_path = pack;
	node->model_revision = state->driver.interface->descriptor->model_revision;
	// Explicit local differential execution: no collective transport.
	node->tp_collective_identifier = 0u;
}


static int32_t probe_capture_open(probe_state_t *state,uint32_t rows)
{
	uint32_t layer,kda = 0u;
	uint64_t page_bytes,recurrent_bytes,hidden_bytes;
	SparkGlm5NextStateCapture *capture = &state->capture;
	for (layer=0u; layer<SPARK_GLM5_NEXT_MODEL_LAYER_COUNT; layer++)
		kda += SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) != 0u ? 1u : 0u;
	page_bytes = (uint64_t)(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT - kda) * 64u * (SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES + SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
	recurrent_bytes = (uint64_t)kda * (SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER + SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER) / state->node.tp_degree;
	hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	*capture = (SparkGlm5NextStateCapture){.abi_version=SPARK_GLM5_NEXT_STATE_CAPTURE_ABI_VERSION,.descriptor_bytes=sizeof(*capture),.lane_capacity=rows,.pages_per_lane_capacity=2u,.payload_capacity=rows * (2u * page_bytes + recurrent_bytes + hidden_bytes),.lanes=state->captured_lanes,.logical_pages=state->logical_pages,.physical_pages=state->physical_pages};
	capture->payload = malloc(capture->payload_capacity);
	return(capture->payload != 0 ? 0 : -17);
}

static uint64_t probe_payload_hash(const uint8_t *data,uint64_t bytes)
{
	uint64_t hash = UINT64_C(14695981039346656037),i;
	for (i=0u; i<bytes; i++)
		hash = (hash ^ data[i]) * UINT64_C(1099511628211);
	return(hash);
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
	if ( state->prefix_probe != 0u && probe_capture_open(state,rows) != 0 )
		return(-17);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = "glm-local-probe";
	request.node_target = PROBE_TARGET;
	request.node_context = &state->node;
	request.execution_stream = state->stream;
	request.kv_logical_page_capacity = state->node.resident_sequence_capacity * (state->node.max_sequence_positions / 64u);
	request.kv_physical_page_capacity = state->prefix_probe != 0u ? 2u * rows : rows;
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
		state->sampling[row] = SparkSamplingRule(0.0f,0u);
		state->outputs[row] = UINT32_MAX;
		state->cache_lanes[row] = (SparkModelDriverCacheLane){.sequence_id=(row + 1u),.sequence_position=step,.request_generation=1u,.step_generation=(step + 1u),.resident_sequence_slot=row,.context_token_count=(step + 1u)};
	}
	state->batch.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	state->batch.descriptor_bytes = sizeof(state->batch);
	state->batch.row_count = rows;
	state->batch.active_sequence_count = rows;
	state->batch.token_ids = state->tokens;
	state->batch.row_resident_slots = state->slots;
	state->batch.row_positions = state->positions;
	state->batch.row_sequence_ids = state->sequences;
	state->batch.row_sampling = state->sampling;
	state->context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	state->context.descriptor_bytes = sizeof(state->context);
	state->context.flags = step == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	state->context.batch = &state->batch;
	state->context.state_capture = state->prefix_probe != 0u && (step < PROBE_STEPS || step >= PROBE_PREFIX_TOKENS) ? &state->capture : 0;
	state->capture.payload_bytes = 0u;
}

static void probe_frame(probe_state_t *state,uint32_t rows,uint32_t step)
{
	SparkModelDriverFrame *frame = &state->frame;
	memset(frame,0,sizeof(*frame));
	state->buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	state->buffer.address = state->outputs;
	state->buffer.bytes = ((uint64_t)rows * sizeof(uint32_t));
	frame->request_id = ++state->next_request_id;
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
	frame->cache_lanes = state->cache_lanes;
	frame->cache_lane_count = rows;
}

static SparkStatus probe_request(probe_state_t *state,SparkModelDriverAdmissionRequest *request)
{
	SparkStatus status = SparkAdmissionRequestFromFrame(state->program->program_id,&state->frame,state->cache_lanes,0u,request);
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
	SparkStatus status,rollback;
	status = probe_request(state,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	{
		uint64_t deadline = probe_time() + UINT64_C(120000000000);
		struct timespec pause = {0,1000000};
		do
		{
			status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
			if ( status == SPARK_STATUS_BUSY )
				(void)nanosleep(&pause,0);
		} while ( status == SPARK_STATUS_BUSY && probe_time() < deadline );
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	request.admission_flags = 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionEvaluateAndApply(state->driver.interface,state->instance,&request,&state->frame,&decision);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->instance,&state->frame);
	if ( status == SPARK_STATUS_OK )
		return(status);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	rollback = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	return(rollback == SPARK_STATUS_OK ? status : rollback);
}

static int32_t probe_execute(probe_state_t *state,uint32_t rows,uint32_t step)
{
	SparkStatus status;
	uint32_t row;
	int32_t result;
	atomic_store_explicit(&state->completed,0u,memory_order_release);
	status = probe_submit(state);
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
	for (row=0u; row<state->batch.row_count; row++)
	{
		if ( state->outputs[row] >= SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT )
			return(-10);
		printf("TOKEN step=%u row=%u input=%u output=%u\n",step,row,state->tokens[row],state->outputs[row]);
	}
	for (row=0u; row<rows; row++)
	{
		if ( state->context.state_capture != 0 )
		{
			SparkGlm5NextStateCaptureLane *lane = &state->captured_lanes[row];
			uint32_t score;
			if ( !isfinite(lane->output_score) || state->capture.payload_bytes == 0u || state->capture.payload_bytes > state->capture.payload_capacity || lane->sequence_id != state->sequences[row] || lane->next_position != state->cache_lanes[row].context_token_count || lane->resident_slot != state->slots[row] || lane->payload_bytes == 0u || lane->payload_offset > state->capture.payload_bytes || lane->payload_bytes > state->capture.payload_bytes - lane->payload_offset )
				return(-18);
			memcpy(&score,&lane->output_score,sizeof(score));
			printf("STATE step=%u row=%u bytes=%llu hash=%016llx score=%08x\n",step,row,(unsigned long long)lane->payload_bytes,(unsigned long long)probe_payload_hash(state->capture.payload + lane->payload_offset,lane->payload_bytes),score);
		}
	}
	return(0);
}

static int32_t probe_step(probe_state_t *state,uint32_t rows,uint32_t step)
{
	probe_batch(state,rows,step);
	probe_frame(state,rows,step);
	return(probe_execute(state,rows,step));
}

static int32_t probe_release_remaining(probe_state_t *state,uint32_t rows,uint32_t position,uint32_t remaining)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	uint32_t row;
	state->frame.request_id = ++state->next_request_id;
	state->frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	state->frame.new_token_count = 0u;
	state->frame.sequence_position = position == UINT32_MAX ? state->cache_lanes[0].context_token_count : position;
	for (row=0u; row<rows; row++)
	{
		state->cache_lanes[row].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
		state->cache_lanes[row].sequence_position = position == UINT32_MAX ? state->cache_lanes[row].context_token_count : position;
		state->cache_lanes[row].step_generation = state->cache_lanes[row].sequence_position + 1u;
	}
	status = probe_request(state,&request);
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	if ( status == SPARK_STATUS_OK )
		status = state->driver.interface->snapshot(state->instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK || snapshot.active_submission_count != 0u || snapshot.resident_sequence_count != remaining )
		return(-12);
	return(0);
}

static int32_t probe_release(probe_state_t *state,uint32_t rows,uint32_t position)
{
	return(probe_release_remaining(state,rows,position,0u));
}

static void probe_checkpoint_frame(probe_state_t *state,uint32_t rows,uint32_t step,uint32_t replay)
{
	SparkModelDriverCacheLane *lane;
	uint32_t row;
	probe_batch(state,rows,step);
	probe_frame(state,rows,step);
	for (row=0u; row<rows; row++)
	{
		lane = &state->cache_lanes[row];
		if ( replay == 1u || replay == 3u )
		{
			state->slots[row] = lane->resident_sequence_slot = rows + row;
			state->sequences[row] = lane->sequence_id = rows + row + 1u;
		}
		// Fixed-input test identities distinguish the three complete prompts.
		if ( step + 1u == PROBE_PREFIX_TOKENS || step + 1u == 64u )
		{
			lane->flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
			lane->publish_token_count = step + 1u;
			lane->publish_identity.sha256[0] = (uint8_t)(row + 1u);
			lane->publish_identity.sha256[1] = (uint8_t)(step + 1u);
		}
		if ( replay == 1u && step == PROBE_PREFIX_TOKENS )
		{
			lane->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
			lane->prefix_token_count = PROBE_PREFIX_TOKENS;
			lane->prefix_identity.sha256[0] = (uint8_t)(row + 1u);
			lane->prefix_identity.sha256[1] = PROBE_PREFIX_TOKENS;
		}
	}
	if ( replay >= 2u )
	{
		state->context.state_capture = 0;
		for (row=0u; row<rows; row++)
			state->tokens[row] += replay * 1000u;
	}
	state->frame.sequence_id = state->sequences[0];
}


static void probe_join_frame(probe_state_t *state,uint32_t rows,uint32_t existing)
{
	uint32_t row;
	probe_checkpoint_frame(state,rows,0u,2u);
	for (row=0u; row<existing; row++)
	{
		state->tokens[row]++;
		state->positions[row] = state->cache_lanes[row].sequence_position = 1u;
		state->cache_lanes[row].context_token_count = state->cache_lanes[row].step_generation = 2u;
	}
	state->frame.sequence_position = state->positions[0];
}

static int32_t probe_reset(probe_state_t *state)
{
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	request.descriptor_bytes = sizeof(request);
	request.program_id = state->program->program_id;
	request.control_generation = state->control_generation + 1u;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	status = SparkAdmissionEvaluate(state->driver.interface,state->instance,&request,&decision);
	if ( status != SPARK_STATUS_OK )
		return(-13);
	state->control_generation = request.control_generation;
	return(0);
}


static void probe_temporal_frame(probe_state_t *state,uint32_t lanes)
{
	uint32_t lane,step,row = 0u;
	probe_batch(state,lanes,0u);
	probe_frame(state,lanes,0u);
	for (lane=0u; lane<lanes; lane++)
		state->cache_lanes[lane].context_token_count = lane == 0u ? PROBE_STEPS : 1u + lane % PROBE_STEPS;
	for (step=0u; step<PROBE_STEPS; step++)
		for (lane=0u; lane<lanes; lane++)
			if (step < state->cache_lanes[lane].context_token_count)
			{
				state->tokens[row] = 1u + lane * PROBE_STEPS + step;
				state->slots[row] = lane;
				state->positions[row] = step;
				state->sequences[row] = lane + 1u;
				state->outputs[row++] = UINT32_MAX;
			}
	state->batch.row_count = state->frame.new_token_count = row;
	state->buffer.bytes = row * sizeof(uint32_t);
	state->frame.tokens_per_sequence = 0u;
}

static int32_t probe_prefix(probe_state_t *state,uint32_t rows)
{
	uint32_t start[PROBE_STEPS][PROBE_ROWS],continuation[PROBE_STEPS][PROBE_ROWS];
	uint32_t maps[PROBE_ROWS],step,row,moved = 0u,existing = rows == 5u ? 3u : 1u;
	uint8_t *expected[PROBE_STEPS * 2u] = {0};
	uint64_t sizes[PROBE_STEPS * 2u] = {0};
	float scores[PROBE_STEPS * 2u][PROBE_ROWS];
	int32_t result;
	for (step=0u; step<PROBE_PREFIX_TOKENS + PROBE_STEPS; step++)
	{
		probe_checkpoint_frame(state,rows,step,0u);
		result = probe_execute(state,rows,step);
		if ( result != 0 )
			return(result);
		if ( step < PROBE_STEPS || step >= PROBE_PREFIX_TOKENS )
		{
			uint32_t index = step < PROBE_STEPS ? step : PROBE_STEPS + step - PROBE_PREFIX_TOKENS;
			sizes[index] = state->capture.payload_bytes;
			expected[index] = malloc(sizes[index]);
			if ( expected[index] == 0 )
				return(-17);
			memcpy(expected[index],state->capture.payload,sizes[index]);
			for (row=0u; row<rows; row++)
				scores[index][row] = state->captured_lanes[row].output_score;
		}
		if ( step < PROBE_STEPS )
			memcpy(start[step],state->outputs,rows * sizeof(uint32_t));
		if ( step >= PROBE_PREFIX_TOKENS )
		{
			memcpy(continuation[step - PROBE_PREFIX_TOKENS],state->outputs,rows * sizeof(uint32_t));
			for (row=0u; row<rows; row++)
				maps[row] = state->physical_pages[row * 2u];
		}
	}
	result = probe_release(state,rows,PROBE_PREFIX_TOKENS + PROBE_STEPS);
	probe_checkpoint_frame(state,existing,0u,2u);
	if ( result == 0 )
		result = probe_execute(state,existing,1000u);
	probe_join_frame(state,rows,existing);
	if ( result == 0 )
		result = probe_execute(state,rows,1001u);
	probe_checkpoint_frame(state,rows,0u,3u);
	if ( result == 0 )
		result = probe_execute(state,rows,1002u);
	if ( result == 0 )
		result = probe_release_remaining(state,rows,1u,rows);
	probe_join_frame(state,rows,existing);
	if ( result == 0 )
		result = probe_release(state,rows,UINT32_MAX);
	if ( result == 0 )
		printf("JOIN existing=%u new=%u launched=%u unequal_positions=%u\n",existing,rows - existing,rows,rows > 1u ? 1u : 0u);
	for (step=0u; result == 0 && step<PROBE_STEPS; step++)
	{
		probe_checkpoint_frame(state,rows,PROBE_PREFIX_TOKENS + step,1u);
		result = probe_execute(state,rows,PROBE_PREFIX_TOKENS + step);
		if ( result == 0 && (memcmp(continuation[step],state->outputs,rows * sizeof(uint32_t)) != 0 || state->capture.payload_bytes != sizes[PROBE_STEPS + step] || memcmp(expected[PROBE_STEPS + step],state->capture.payload,sizes[PROBE_STEPS + step]) != 0) )
			return(-14);
		for (row=0u; result == 0 && row<rows; row++)
		{
			if ( memcmp(&scores[PROBE_STEPS + step][row],&state->captured_lanes[row].output_score,sizeof(float)) != 0 )
				return(-19);
			if ( step == 0u && maps[row] != state->physical_pages[row * 2u] )
				moved++;
		}
	}
	if ( result == 0 && (moved != rows || state->capture.backing_read_count < rows || state->capture.backing_write_count < rows || state->capture.prefix_hit_count < rows) )
		return(-20);
	if ( result == 0 )
		result = probe_release(state,rows,PROBE_PREFIX_TOKENS + PROBE_STEPS);
	if ( result == 0 )
		result = probe_reset(state);
	if ( result != 0 )
		return(result);
	probe_checkpoint_frame(state,rows,PROBE_PREFIX_TOKENS,1u);
	if ( probe_submit(state) != SPARK_STATUS_NOT_FOUND )
		return(-15);
	for (step=0u; step<PROBE_STEPS; step++)
	{
		result = probe_step(state,rows,step);
		if ( result != 0 || memcmp(start[step],state->outputs,rows * sizeof(uint32_t)) != 0 || state->capture.payload_bytes != sizes[step] || memcmp(expected[step],state->capture.payload,sizes[step]) != 0 )
			return(result != 0 ? result : -16);
	}
	result = probe_release(state,rows,PROBE_STEPS);
	if ( result == 0 ) result = probe_reset(state);
	if ( result != 0 ) return(result);
	probe_temporal_frame(state,rows);
	result = probe_execute(state,rows,2000u);
	if ( result != 0 ) return(result);
	for (row=0u; row<state->batch.row_count; row++)
		if ( state->outputs[row] != start[state->positions[row]][state->slots[row]] )
			return(-21);
	for (row=0u; row<rows; row++)
	{
		SparkGlm5NextStateCaptureLane *lane = &state->captured_lanes[row];
		step = state->cache_lanes[row].context_token_count - 1u;
		if ( sizes[step] % rows != 0u || lane->payload_bytes != sizes[step] / rows || memcmp(expected[step] + row * lane->payload_bytes,state->capture.payload + lane->payload_offset,lane->payload_bytes) != 0 || memcmp(&scores[step][row],&lane->output_score,sizeof(float)) != 0 )
			return(-22);
	}
	printf("TEMPORAL lanes=%u rows=%u unequal_lengths=%u state=exact selected-logit=exact tokens=exact\n",rows,state->batch.row_count,rows > 1u ? 1u : 0u);
	for (step=0u; step<PROBE_STEPS * 2u; step++)
		free(expected[step]);
	printf("RESTORE rows=%u moved=%u state=exact selected-logit=exact full-vocabulary-logits=unavailable\n",rows,moved);
	return(probe_release(state,rows,UINT32_MAX));
}

int main(int argc,char **argv)
{
	probe_state_t state;
	uint32_t rows,step;
	int32_t result;
	const char *socket;
	if ( (argc != 5 && argc != 6) || (argc == 6 && strcmp(argv[5],"prefix") != 0) || (strcmp(argv[3],"resident") != 0 && strcmp(argv[3],"lazy") != 0) || (strcmp(argv[4],"1") != 0 && strcmp(argv[4],"3") != 0 && strcmp(argv[4],"5") != 0) )
	{
		fprintf(stderr,"usage: %s DRIVER TP16_RANK0_PACK resident|lazy 1|3|5 [prefix]\n",argv[0]);
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
	state.prefix_probe = argc == 6 ? 1u : 0u;
	state.control_generation = 1u;
	atomic_init(&state.completed,0u);
	result = probe_open(&state,argv[1],argv[2],rows);
	if ( result == 0 && state.prefix_probe != 0u )
		result = probe_prefix(&state,rows);
	for (step=0u; state.prefix_probe == 0u && result == 0 && step<PROBE_STEPS; step++)
		result = probe_step(&state,rows,step);
	if ( result == 0 && state.prefix_probe == 0u )
		result = probe_release(&state,rows,PROBE_STEPS);
	if ( result != 0 )
	{
		fprintf(stderr,"probe failed result=%d; process exit preserves uncertain in-flight ownership\n",result);
		fflush(0);
		_Exit(4);
	}
	free(state.capture.payload);
	state.driver.interface->destroy(state.instance);
	if ( cudaStreamDestroy(state.stream) != cudaSuccess )
		return(5);
	SparkUnloadModelDriver(&state.driver);
	if ( state.prefix_probe != 0u )
		printf("PASS local-prefix-reuse mode=%s rows=%u prefix=63 continuation=4 cow=required boundary64=published eviction=verified movement=verified state=exact reset=verified; rank-local computation only\n",argv[3],rows);
	else
		printf("PASS local-token-smoke mode=%s rows=%u steps=%u; not full-model numerical qualification\n",argv[3],rows,PROBE_STEPS);
	return(0);
}
