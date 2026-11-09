#include "inference/runner/stage_runner.cu"

#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerStep in;
	int status = 0;
	memset(state, 0, sizeof(*state));
	memset(&runner, 0, sizeof(runner));
	memset(&in, 0, sizeof(in));
	in.graph_key = 1u;
	in.rows = 1u;
	const pid_t child = fork();
	if ( child == 0 )
	{
		StageRunnerGraphCapture(state, &runner, &in, 0);
		_exit(0);
	}
	if ( child < 0 || waitpid(child, &status, 0) != child || !WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT )
	{
		printf("FAIL a decode graph that cannot be captured aborts the rank (wait status %d)\n", status);
		return 1;
	}
	delete state;
	printf("PASS runner_capture_flags_cuda: a decode graph that cannot be captured aborts the rank\n");
	return 0;
}
