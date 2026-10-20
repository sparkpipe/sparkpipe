#include "inference/runner/stage_runner.cu"

#include <stdio.h>

int main(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerStep in;
	int failures = 0;
	memset(state, 0, sizeof(*state));
	memset(&runner, 0, sizeof(runner));
	memset(&in, 0, sizeof(in));
	in.graph_key = 1u;
	in.rows = 1u;
	state->copy_failed = 1u;
	state->tp_collective_failed = 1u;
	StageRunnerGraphCapture(state, &runner, &in, 0);
	if ( state->graph_exec[0] != 0 || state->graph_refused[0] != 1u )
	{
		printf("FAIL a capture without a collective is refused for its row count (exec %p refused %u)\n", (void *)state->graph_exec[0], state->graph_refused[0]);
		failures++;
	}
	if ( state->copy_failed != 1u || state->tp_collective_failed != 1u )
	{
		printf("FAIL a refused capture keeps the copy and collective failures recorded before it (copy %u collective %u)\n",
			state->copy_failed, state->tp_collective_failed);
		failures++;
	}
	state->copy_failed = 0u;
	state->tp_collective_failed = 0u;
	state->graph_refused[1] = 0u;
	in.rows = 2u;
	StageRunnerGraphCapture(state, &runner, &in, 0);
	if ( state->copy_failed != 0u || state->tp_collective_failed != 0u )
	{
		printf("FAIL a refused capture of a clean step leaves no failure behind (copy %u collective %u)\n", state->copy_failed, state->tp_collective_failed);
		failures++;
	}
	StageRunnerGraphPrefetchFree(state);
	delete state;
	if ( failures != 0 )
	{
		printf("FAIL runner_capture_flags_cuda: %d failures\n", failures);
		return 1;
	}
	printf("PASS runner_capture_flags_cuda: a refused graph capture keeps the failures recorded before it and adds none\n");
	return 0;
}
