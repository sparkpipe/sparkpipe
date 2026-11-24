#include "inference/runner/stage_runner.cu"

#include <stdio.h>

#define FAILED_ROWS 4u
#define FAILED_HIDDEN 2048u
#define FAILED_SPIN_NS 200000000ull
#define FAILED_VOCAB 16u

static int failures;
static uint32_t failing_steps;

extern "C" {

void SparkTpDeviceCollectiveRegisterLaunchers(const SparkTpMeshLaunchers *launchers)
{
	(void)launchers;
}

SparkStatus SparkTpDeviceCollectiveCreate(const SparkTpDeviceCollectiveConfig *config, SparkTpDeviceCollective *collective_out)
{
	(void)config; (void)collective_out;
	return SPARK_STATUS_UNSUPPORTED;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
	(void)collective;
}

SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(SparkTpDeviceCollective *collective, void *receive_device,
	uint32_t active_sequence_count, uint32_t hidden_dimension, uint32_t step_index, void *cuda_stream)
{
	(void)collective; (void)receive_device; (void)active_sequence_count; (void)hidden_dimension; (void)step_index; (void)cuda_stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission,
	uint32_t operation_kind)
{
	(void)collective; (void)submission; (void)operation_kind;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveBegin(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission,
	uint32_t operation_kind)
{
	(void)collective; (void)submission; (void)operation_kind;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveFinish(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

uint32_t SparkTpDeviceCollectiveSupportsPairAllReduce(const SparkTpDeviceCollective *collective, uint32_t rows, uint32_t row_elements)
{
	(void)collective; (void)rows; (void)row_elements;
	return 0u;
}

uint32_t SparkTpDeviceCollectivePublished(const SparkTpDeviceCollective *collective)
{
	(void)collective;
	return 0u;
}

SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveHardwareStats(SparkTpDeviceCollective *collective, SparkTpDeviceCollectiveHardwareTiming *timing_out)
{
	(void)collective; (void)timing_out;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphSettle(SparkTpDeviceCollective *collective, void *stream, uint64_t *error_out)
{
	(void)collective; (void)stream; (void)error_out;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveEndChain(SparkTpDeviceCollective *collective, void *cuda_stream)
{
	(void)collective; (void)cuda_stream;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveChainKey(SparkTpDeviceCollective *collective, uint64_t request_id)
{
	(void)collective; (void)request_id;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveDisarmCapture(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveSetOperationTimeout(SparkTpDeviceCollective *collective, uint32_t timeout_milli, uint32_t *previous_milli)
{
	(void)collective; (void)timeout_milli;
	if ( previous_milli != 0 )
		*previous_milli = 0u;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveReady(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

}

__global__ static void FailedSpinKernel(uint64_t nanoseconds)
{
	uint64_t start, now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < nanoseconds );
}

static SparkStatus FailingStep(void *model, const SparkStageRunnerStep *step, void *stream)
{
	(void)model;
	(void)step;
	(void)stream;
	failing_steps++;
	return SPARK_STATUS_INTERNAL_ERROR;
}

static void Expect(int condition, const char *name)
{
	if ( condition )
		return;
	printf("FAIL %s\n", name);
	failures++;
}

static void Fill(uint16_t *rows, uint16_t value, size_t count)
{
	for ( size_t index = 0u; index < count; ++index )
		rows[index] = (uint16_t)(value + index);
}

static size_t Matching(const uint16_t *rows, uint16_t value, size_t count)
{
	size_t same = 0u;
	for ( size_t index = 0u; index < count; ++index )
		same += rows[index] == (uint16_t)(value + index) ? 1u : 0u;
	return same;
}

static size_t EmbeddedRows(const uint16_t *hidden, const uint16_t *table, const uint32_t *tokens, uint32_t rows)
{
	size_t same = 0u;
	for ( uint32_t row = 0u; row < rows; ++row )
		same += memcmp(hidden + (size_t)row * FAILED_HIDDEN, table + (size_t)tokens[row] * FAILED_HIDDEN, FAILED_HIDDEN * sizeof(uint16_t)) == 0 ? 1u : 0u;
	return same;
}

int main(void)
{
	const size_t elements = (size_t)FAILED_ROWS * FAILED_HIDDEN, bytes = elements * sizeof(uint16_t);
	const size_t table_elements = (size_t)FAILED_VOCAB * FAILED_HIDDEN;
	const uint32_t first[FAILED_ROWS] = {1u, 2u, 3u, 4u}, second[FAILED_ROWS] = {9u, 10u, 11u, 12u};
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunnerModelInterface model_interface;
	SparkStageRunnerDispatch dispatch;
	SparkStageRunner runner;
	uint16_t *input = 0, *table = new uint16_t[table_elements], *seen = new uint16_t[elements], *table_device = 0;
	memset(state, 0, sizeof(*state));
	memset(&model_interface, 0, sizeof(model_interface));
	memset(&dispatch, 0, sizeof(dispatch));
	memset(&runner, 0, sizeof(runner));
	for ( size_t index = 0u; index < table_elements; ++index )
		table[index] = (uint16_t)(0x3c00u + index % 0x3ffu + index / FAILED_HIDDEN * 7u);
	if ( cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaMalloc((void **)&state->hidden, bytes) != cudaSuccess ||
		cudaMalloc((void **)&state->chain_step_device, sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc((void **)&state->token_ids_device, FAILED_ROWS * sizeof(uint32_t)) != cudaSuccess ||
		cudaHostAlloc((void **)&state->token_ids_host, FAILED_ROWS * sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess ||
		cudaMalloc((void **)&table_device, table_elements * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemcpy(table_device, table, table_elements * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaHostAlloc((void **)&input, bytes, cudaHostAllocDefault) != cudaSuccess )
	{
		printf("FAIL stage_runner_failed_dispatch_cuda: no CUDA device\n");
		return 1;
	}
	model_interface.tag = "failed-dispatch";
	model_interface.step = FailingStep;
	state->model_interface = &model_interface;
	state->max_rows = FAILED_ROWS;
	state->geometry.hidden = FAILED_HIDDEN;
	state->tap_final_entry = UINT32_MAX;
	state->embed_weight = table_device;
	state->vocab = FAILED_VOCAB;
	state->vocab_slice_rows = FAILED_VOCAB;
	state->tp_degree = 1u;
	runner.private_state = state;
	runner.tp_degree = 1u;
	runner.owns_embedding = 1u;
	dispatch.row_count = FAILED_ROWS;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = first;

	Fill(input, 0x1000u, elements);
	FailedSpinKernel<<<1, 1, 0, state->stream>>>(FAILED_SPIN_NS);
	Expect(cudaMemcpyAsync(state->hidden, input, bytes, cudaMemcpyHostToDevice, state->stream) == cudaSuccess, "the control copy is queued");
	Fill(input, 0x5000u, elements);
	Expect(cudaStreamSynchronize(state->stream) == cudaSuccess && cudaMemcpy(seen, state->hidden, bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
		"the control copy completes");
	Expect(Matching(seen, 0x5000u, elements) == elements,
		"control: an upload queued behind other work reads its pinned source when it runs, so a host write before then changes what lands");

	FailedSpinKernel<<<1, 1, 0, state->stream>>>(FAILED_SPIN_NS);
	Expect(SparkStageRunnerSubmit(&runner, &dispatch) == SPARK_STATUS_INTERNAL_ERROR && failing_steps == 1u, "the model step fails the dispatch");
	Expect(cudaStreamQuery(state->stream) == cudaSuccess, "a failed dispatch returns only after the work it queued has completed");
	Expect(state->dispatch_queued == 0u, "a failed dispatch leaves nothing marked queued");
	memcpy(state->token_ids_host, second, sizeof(second));
	Expect(cudaStreamSynchronize(state->stream) == cudaSuccess && cudaMemcpy(seen, state->hidden, bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
		"the runner's stream is idle");
	Expect(EmbeddedRows(seen, table, first, FAILED_ROWS) == FAILED_ROWS,
		"the next dispatch may rewrite the token staging as soon as a failed dispatch returns: the failed one embedded its own tokens");

	cudaFreeHost(input);
	cudaFreeHost(state->token_ids_host);
	cudaFree(state->hidden);
	cudaFree(state->chain_step_device);
	cudaFree(state->token_ids_device);
	cudaFree(table_device);
	cudaStreamDestroy(state->stream);
	delete[] seen;
	delete[] table;
	delete state;
	if ( failures != 0 )
	{
		printf("FAIL stage_runner_failed_dispatch_cuda: %d failures\n", failures);
		return 1;
	}
	printf("PASS stage_runner_failed_dispatch_cuda: a queued upload reads its pinned source when it runs (control); a failed dispatch completes its queued work before it returns\n");
	return 0;
}
