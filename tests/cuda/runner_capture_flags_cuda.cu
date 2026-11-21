#include "inference/runner/stage_runner.cu"

#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

static int IdentitiesMatch(const SparkStageRunnerStep *first, const SparkStageRunnerStep *second, uint32_t ignores_gather)
{
	SparkStageRunnerStep one, two;
	StageRunnerGraphIdentity(first, ignores_gather, &one);
	StageRunnerGraphIdentity(second, ignores_gather, &two);
	return memcmp(&one, &two, sizeof(one)) == 0 ? 1 : 0;
}

static int GraphIdentityRules(void)
{
	SparkStageRunnerStep prompt_end, decode, other_prompt, wider, verify, verify_other;
	static const uint32_t positions_a[1] = { 8191u }, positions_b[1] = { 32767u };
	memset(&prompt_end, 0, sizeof(prompt_end));
	prompt_end.graph_key = 1u;
	prompt_end.rows = 1u;
	prompt_end.sequences = 1u;
	prompt_end.commit = 1u;
	prompt_end.context = 262144u;
	prompt_end.gather_context = 8191u;
	prompt_end.gather_sequence = 3u;
	prompt_end.host_positions = positions_a;
	memcpy(&decode, &prompt_end, sizeof(decode));
	decode.gather_context = 0u;
	decode.gather_sequence = 0u;
	memcpy(&other_prompt, &prompt_end, sizeof(other_prompt));
	other_prompt.gather_context = 32767u;
	other_prompt.host_positions = positions_b;
	memcpy(&wider, &decode, sizeof(wider));
	wider.rows = 2u;
	wider.sequences = 2u;
	memcpy(&verify, &prompt_end, sizeof(verify));
	verify.graph_key = 2u;
	verify.rows = 4u;
	verify.commit = 0u;
	memcpy(&verify_other, &verify, sizeof(verify_other));
	verify_other.gather_context = 511u;
	if ( IdentitiesMatch(&prompt_end, &decode, 0u) || IdentitiesMatch(&prompt_end, &other_prompt, 0u) )
	{
		printf("FAIL a model that reads the gather fields must recapture when the prompt length changes\n");
		return 1;
	}
	if ( !IdentitiesMatch(&prompt_end, &decode, 1u) || !IdentitiesMatch(&prompt_end, &other_prompt, 1u) )
	{
		printf("FAIL a model that ignores the gather fields must replay one graph for every prompt length and for decode\n");
		return 1;
	}
	if ( IdentitiesMatch(&decode, &wider, 1u) )
	{
		printf("FAIL a step of another row count must not replay a graph captured for one row\n");
		return 1;
	}
	if ( !IdentitiesMatch(&verify, &verify_other, 0u) )
	{
		printf("FAIL a verify step must replay its graph whatever its gather fields\n");
		return 1;
	}
	return 0;
}

static int GraphSlotRules(void)
{
	static uint8_t seen[STAGE_RUNNER_GRAPH_SLOTS];
	memset(seen, 0, sizeof(seen));
	for ( uint32_t key = 1u; key <= STAGE_RUNNER_GRAPH_KEYS; ++key )
		for ( uint32_t rows = 1u; rows <= STAGE_RUNNER_GRAPH_ROWS; ++rows )
			for ( uint32_t sampled = 0u; sampled < 2u; ++sampled )
				for ( uint32_t last_rows = 0u; last_rows < 2u; ++last_rows )
				{
					const uint32_t slot = StageRunnerGraphSlot(key, rows, sampled, last_rows);
					if ( slot >= STAGE_RUNNER_GRAPH_SLOTS || seen[slot] != 0u )
					{
						printf("FAIL graph key %u rows %u sampled %u last rows %u has no slot of its own (%u)\n", key, rows, sampled, last_rows, slot);
						return 1;
					}
					seen[slot] = 1u;
				}
	return 0;
}

int main(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerStep in;
	int status = 0;
	if ( GraphIdentityRules() != 0 || GraphSlotRules() != 0 )
		return 1;
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
	printf("PASS runner_capture_flags_cuda: graph replay follows the gather rule, a prompt's last step and a decode step keep separate graphs, and a decode graph that cannot be captured aborts the rank\n");
	return 0;
}
