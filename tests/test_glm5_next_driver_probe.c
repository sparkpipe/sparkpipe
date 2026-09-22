#define main probe_main
#include "../tools/glm5_next_driver_probe.c"
#undef main
#include <assert.h>

static uint32_t submitted,destroyed,snapshots,expected_rows,phase,released,aborted,snapshot_pending;
static uint32_t prefix_mode,reset_count,prefix_saved[PROBE_ROWS],prefix_positions[2u * PROBE_ROWS],prefix_live[2u * PROBE_ROWS];

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,unsigned int flags)
{
	assert(flags == cudaStreamNonBlocking);
	*stream = (cudaStream_t)(uintptr_t)1u;
	return(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	assert(stream == (cudaStream_t)(uintptr_t)1u);
	return(cudaSuccess);
}

static SparkStatus fake_create(const SparkModelDriverCreateRequest *request,void **instance)
{
	SparkGlm5NextResidentDecodeStageNodeContext *node = request->node_context;
	assert(node->tp_degree == 16u && node->tp_rank == 0u && node->tp_collective_identifier == 0u);
	assert(node->resident_sequence_capacity == expected_rows * (prefix_mode != 0u ? 2u : 1u) && node->flags == 0u);
	assert(node->max_sequence_positions == (prefix_mode != 0u ? 128u : 64u));
	assert(request->kv_logical_page_capacity == expected_rows * (prefix_mode != 0u ? 4u : 1u));
	assert(request->kv_physical_page_capacity == expected_rows * (prefix_mode != 0u ? 2u : 1u));
	assert(node->execution_row_capacity == expected_rows * (prefix_mode != 0u ? PROBE_STEPS : 1u));
	*instance = node;
	return(SPARK_STATUS_OK);
}

static void fake_destroy(void *instance)
{
	assert(instance != 0 && submitted == (prefix_mode != 0u ? PROBE_PREFIX_TOKENS + 3u * PROBE_STEPS + 4u : PROBE_STEPS));
	destroyed++;
}

static SparkStatus fake_snapshot(void *instance,uint32_t program,SparkModelDriverRuntimeSnapshot *snapshot)
{
	assert(instance != 0 && program == 7u);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->active_submission_count = snapshot_pending;
	snapshot_pending = 0u;
	snapshot->resident_sequence_count = released == 0u ? expected_rows : 0u;
	if ( prefix_mode != 0u )
	{
		uint32_t slot,live = 0u;
		for (slot=0u; slot<2u * expected_rows; slot++)
			live += prefix_live[slot];
		snapshot->resident_sequence_count = live;

	}
	snapshots++;
	return(SPARK_STATUS_OK);
}

static SparkStatus fake_transaction(const SparkModelDriverAdmissionRequest *request)
{
	uint32_t row;
	assert(request->cache_lane_count == expected_rows && request->cache_lanes != 0);
	assert(request->submission_id == request->request_id && request->transaction_id == request->request_id);
	assert(request->control_generation == 1u && request->request_generation == 1u && request->step_generation == request->request_id);
	for (row=0u; row<expected_rows; row++)
	{
		assert(request->cache_lanes[row].sequence_id == row + 1u && request->cache_lanes[row].resident_sequence_slot == row);
		assert(request->cache_lanes[row].sequence_position == submitted && request->cache_lanes[row].step_generation == submitted + 1u);
	}
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		assert(phase == 0u && submitted == PROBE_STEPS && request->new_token_count == 0u);
		for (row=0u; row<expected_rows; row++)
			assert(request->cache_lanes[row].flags == SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE);
		released++;
		return(SPARK_STATUS_OK);
	}
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
	{
		assert(phase > 0u && phase <= 3u);
		aborted++;
		phase = 0u;
		return(SPARK_STATUS_OK);
	}
	assert(request->admission_flags == (phase == 0u ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE : phase == 1u ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : 0u));
	if ( getenv("PROBE_FAIL_PHASE") != 0 && strtoul(getenv("PROBE_FAIL_PHASE"),0,10) == phase + 1u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	phase++;
	return(SPARK_STATUS_OK);
}

static SparkStatus fake_prefix_admit(const SparkModelDriverAdmissionRequest *request)
{
	uint32_t row,slot;
	assert(SparkModelDriverAdmissionRequestIsValid(request) != 0u);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		assert(phase == 0u && request->control_generation == reset_count + 2u);
		for (slot=0u; slot<2u * expected_rows; slot++)
			assert(prefix_live[slot] == 0u);
		memset(prefix_saved,0,sizeof(prefix_saved));
		reset_count++;
		return(SPARK_STATUS_OK);
	}
	assert(request->cache_lane_count <= expected_rows && request->cache_lane_count != 0u && request->control_generation == reset_count + 1u);
	for (row=0u; row<request->cache_lane_count; row++)
	{
		const SparkModelDriverCacheLane *lane = &request->cache_lanes[row];
		slot = lane->resident_sequence_slot;
		assert(slot == row || slot == expected_rows + row);
		assert(lane->sequence_id == slot + 1u);
		if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u )
		{
			assert(phase == 0u || phase == 1u || phase == 2u);
			assert(lane->prefix_token_count == PROBE_PREFIX_TOKENS && lane->prefix_identity.sha256[0] == row + 1u);
			if ( prefix_saved[row] == 0u )
				return(SPARK_STATUS_NOT_FOUND);
		}
		else
			assert(lane->sequence_position == prefix_positions[slot]);
		if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		{
			assert(phase == 0u && request->new_token_count == 0u);
			prefix_live[slot] = prefix_positions[slot] = 0u;
		}
	}
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		released++;
		return(SPARK_STATUS_OK);
	}
	assert(request->admission_flags == (phase == 0u ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE : phase == 1u ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : 0u));
	phase++;
	return(SPARK_STATUS_OK);
}

static SparkStatus fake_admit(void *instance,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	if ( prefix_mode != 0u )
	{
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		return(fake_prefix_admit(request));
	}
	assert(instance != 0 && request->active_slot_count == expected_rows);
	assert(request->sequence_position == submitted && request->request_id == (submitted + 1u));
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(fake_transaction(request));
}

static SparkStatus fake_submit(void *instance,SparkModelDriverFrame *frame)
{
	SparkGlm5NextResidentDecodeStageFrameContext *context = frame->user_context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkModelDriverCompletion completion = {0};
	uint32_t row,*output = frame->buffers[0].address;
	assert(phase == 3u && frame->cache_lane_count == batch->active_sequence_count);
	if ( getenv("PROBE_FAIL_PHASE") != 0 && strcmp(getenv("PROBE_FAIL_PHASE"),"4") == 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	phase = 0u;
	assert(instance != 0 && batch->row_count <= expected_rows * (prefix_mode != 0u ? PROBE_STEPS : 1u) && batch->row_count != 0u);
	assert(context->flags == (((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u) ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u));
	for (row=0u; row<batch->row_count; row++)
	{
		output[row] = (batch->token_ids[row] + 5u);
		if ( prefix_mode != 0u )
		{
			uint32_t slot = batch->row_resident_slots[row];
			assert(batch->row_sequence_ids[row] == slot + 1u);
			prefix_live[slot] = 1u;
			prefix_positions[slot] = (uint32_t)batch->row_positions[row] + 1u;
			if ( row < batch->active_sequence_count && (frame->cache_lanes[row].flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u )
			{
				assert(prefix_positions[slot] == PROBE_PREFIX_TOKENS || prefix_positions[slot] == 64u);
				assert(frame->cache_lanes[row].publish_identity.sha256[1] == prefix_positions[slot]);
				if ( prefix_positions[slot] == PROBE_PREFIX_TOKENS )
					prefix_saved[row] = 1u;
			}
			if ( (slot >= expected_rows && getenv("PROBE_BAD_PREFIX") != 0) || (reset_count != 0u && getenv("PROBE_BAD_RESET") != 0) )
				output[row]++;
		}
		else
		{
			assert(batch->row_positions[row] == submitted && batch->row_sequence_ids[row] == (row + 1u));
			assert(batch->row_resident_slots[row] == row);
		}
	}
	if ( context->state_capture != 0 )
	{
		SparkGlm5NextStateCapture *capture = context->state_capture;
		assert(capture->lane_capacity >= expected_rows && capture->payload_capacity >= 32u * expected_rows);
		capture->payload_bytes = 32u * expected_rows;
		capture->backing_write_count = released >= 3u ? expected_rows : 0u;
		capture->backing_read_count = released >= 3u ? expected_rows : 0u;
		capture->prefix_hit_count = released >= 3u ? expected_rows : 0u;
		for (row=0u; row<batch->active_sequence_count; row++)
		{
			uint32_t slot = batch->row_resident_slots[row],last = batch->row_count - 1u;
			while (batch->row_resident_slots[last] != slot) last--;
			capture->lanes[row] = (SparkGlm5NextStateCaptureLane){.sequence_id=batch->row_sequence_ids[row],.next_position=batch->row_positions[last] + 1u,.resident_slot=slot,.page_count=1u,.payload_offset=row * 32u,.payload_bytes=32u,.output_score=(float)batch->token_ids[last]};
			memset(capture->payload + row * 32u,(int)batch->token_ids[last],32u);
			capture->logical_pages[row * 2u] = row;
			capture->physical_pages[row * 2u] = row + (slot >= expected_rows ? expected_rows : 0u);
			if ( batch->row_count > batch->active_sequence_count && getenv("PROBE_BAD_TEMPORAL") != 0 )
				capture->payload[row * 32u]++;
			if ( slot >= expected_rows && getenv("PROBE_BAD_STATE") != 0 )
				capture->payload[row * 32u]++;
			if ( slot >= expected_rows && getenv("PROBE_BAD_SCORE") != 0 )
				capture->lanes[row].output_score++;
			if ( getenv("PROBE_NO_MOVE") != 0 )
				capture->physical_pages[row * 2u] = row;
		}
		if ( getenv("PROBE_NO_READBACK") != 0 )
			capture->backing_read_count = 0u;
	}
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.status = SPARK_STATUS_OK;
	if ( getenv("PROBE_BAD_COMPLETION") != 0 )
		completion.request_id++;
	submitted++;
	snapshot_pending = 1u;
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static const SparkModelDriverProgramDescriptor fake_program = {.program_id = 7u,.submit = fake_submit};
static const SparkModelDriverDescriptor fake_descriptor = {.model_revision = "fixture"};
static const SparkModelDriverInterface fake_interface = {.descriptor = &fake_descriptor,.create = fake_create,.destroy = fake_destroy,.admit = fake_admit,.snapshot = fake_snapshot};

SparkStatus SparkLoadModelDriver(const char *path,const char *target,SparkLoadedModelDriver *driver,char *error,uint32_t bytes)
{
	(void)path;
	(void)error;
	(void)bytes;
	assert(strcmp(target,PROBE_TARGET) == 0);
	driver->interface = &fake_interface;
	return(SPARK_STATUS_OK);
}

void SparkUnloadModelDriver(SparkLoadedModelDriver *driver)
{
	assert(driver->interface == &fake_interface && destroyed == 1u);
}

const SparkModelDriverProgramDescriptor *SparkFindLoadedModelDriverProgram(const SparkLoadedModelDriver *driver,const char *name)
{
	assert(driver->interface == &fake_interface && strcmp(name,"resident_decode") == 0);
	return(&fake_program);
}

int main(int argc,char **argv)
{
	int32_t result;
	expected_rows = argc >= 5 ? (uint32_t)strtoul(argv[4],0,10) : 1u;
	prefix_mode = argc == 6 && strcmp(argv[5],"prefix") == 0 ? 1u : 0u;
	result = probe_main(argc,argv);
	if ( result == 0 && prefix_mode != 0u )
		assert(submitted == PROBE_PREFIX_TOKENS + 3u * PROBE_STEPS + 4u && destroyed == 1u && released == 6u && reset_count == 2u && phase == 0u);
	if ( result == 0 && prefix_mode == 0u )
		assert(submitted == PROBE_STEPS && snapshots == (PROBE_STEPS * 2u) + 1u && destroyed == 1u && released == 1u && aborted == 0u);
	if ( getenv("PROBE_FAIL_PHASE") != 0 )
		assert(result == 4 && submitted == 0u && phase == 0u && aborted == (strcmp(getenv("PROBE_FAIL_PHASE"),"1") == 0 ? 0u : 1u));
	return(result);
}
