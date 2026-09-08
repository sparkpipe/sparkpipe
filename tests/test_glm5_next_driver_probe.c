#define main probe_main
#include "../tools/glm5_next_driver_probe.c"
#undef main
#include <assert.h>

static uint32_t submitted,destroyed,snapshots,expected_rows;

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
	assert(node->resident_sequence_capacity == expected_rows && node->flags == 0u);
	*instance = node;
	return(SPARK_STATUS_OK);
}

static void fake_destroy(void *instance)
{
	assert(instance != 0 && submitted == PROBE_STEPS);
	destroyed++;
}

static SparkStatus fake_snapshot(void *instance,uint32_t program,SparkModelDriverRuntimeSnapshot *snapshot)
{
	assert(instance != 0 && program == 7u);
	snapshot->active_submission_count = (snapshots++ % 2u) == 0u ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus fake_admit(void *instance,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	assert(instance != 0 && request->active_slot_count == expected_rows);
	assert(request->sequence_position == submitted && request->request_id == (submitted + 1u));
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(SPARK_STATUS_OK);
}

static SparkStatus fake_submit(void *instance,SparkModelDriverFrame *frame)
{
	SparkGlm5NextResidentDecodeStageFrameContext *context = frame->user_context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkModelDriverCompletion completion = {0};
	uint32_t row,*output = frame->buffers[0].address;
	assert(instance != 0 && batch->row_count == expected_rows);
	assert(context->flags == (submitted == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u));
	for (row=0u; row<expected_rows; row++)
	{
		assert(batch->row_positions[row] == submitted && batch->row_sequence_ids[row] == (row + 1u));
		assert(batch->row_resident_slots[row] == row);
		output[row] = (batch->token_ids[row] + 5u);
	}
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.status = SPARK_STATUS_OK;
	if ( getenv("PROBE_BAD_COMPLETION") != 0 )
		completion.request_id++;
	submitted++;
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
	expected_rows = argc == 5 && strcmp(argv[4],"3") == 0 ? 3u : 1u;
	result = probe_main(argc,argv);
	if ( result == 0 )
		assert(submitted == PROBE_STEPS && snapshots == (PROBE_STEPS * 2u) && destroyed == 1u);
	return(result);
}
