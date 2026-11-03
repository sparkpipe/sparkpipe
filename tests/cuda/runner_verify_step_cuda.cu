#include "inference/runner/stage_runner.cu"

#include <stdio.h>

static int failures = 0;

static void Expect(SparkStageRunnerState *state, const SparkStageRunner *runner, uint32_t flags, uint64_t recurrent_bytes, uint32_t commit,
	const char *what)
{
	SparkStageRunnerDispatch dispatch;
	SparkStageRunnerStep in;
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.flags = flags;
	dispatch.active_sequence_count = 1u;
	state->geometry.recurrent_bytes = recurrent_bytes;
	StageRunnerModelStep(state, runner, &dispatch, 7u, &in);
	if ( in.commit != commit || in.rows != 7u || in.sequences != 1u )
	{
		printf("FAIL %s (flags 0x%x recurrent bytes %llu: commit %u rows %u sequences %u)\n", what, flags, (unsigned long long)recurrent_bytes, in.commit,
			in.rows, in.sequences);
		failures++;
	}
}

int main(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunnerModelInterface model;
	SparkStageRunner runner;
	memset(state, 0, sizeof(*state));
	memset(&model, 0, sizeof(model));
	memset(&runner, 0, sizeof(runner));
	state->model_interface = &model;
	Expect(state, &runner, SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY, 0u, 0u, "a verify step of a model without recurrent state does not commit");
	Expect(state, &runner, SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY, 4096u, 0u, "a verify step of a model with recurrent state does not commit");
	Expect(state, &runner, 0u, 0u, 1u, "a decode step commits");
	Expect(state, &runner, 0u, 4096u, 1u, "a decode step of a model with recurrent state commits");
	Expect(state, &runner, SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL, 0u, 1u, "a prefill step commits");
	delete state;
	if ( failures != 0 )
	{
		printf("FAIL runner_verify_step_cuda: %d failures\n", failures);
		return 1;
	}
	printf("PASS runner_verify_step_cuda: every verify dispatch reaches the model as a step that does not commit, whatever state the model keeps\n");
	return 0;
}
