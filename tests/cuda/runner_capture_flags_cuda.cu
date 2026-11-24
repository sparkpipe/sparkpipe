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

static uint32_t mock_verify_steps = 0u;

static SparkStatus MockVerifyStep(void *model, const SparkStageRunnerStep *step, void *stream)
{
	(void)model;
	(void)step;
	(void)stream;
	mock_verify_steps++;
	return SPARK_STATUS_OK;
}

static uint32_t MockVerifyGraphKey(void *model, const SparkStageRunnerStep *step)
{
	(void)model;
	return step->commit == 0u && step->sequences == 1u && step->rows > 1u ? 2u : 1u;
}

static int GraphIdentityRules(void)
{
	SparkStageRunnerStep prompt_end, decode, other_prompt, wider, verify, verify_other, tree_verify, tree_verify_other;
	static const uint32_t positions_a[1] = { 8191u }, positions_b[1] = { 32767u };
	static const uint32_t slots_a[1] = { 10u }, slots_b[1] = { 20u };
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
	verify.rows = 24u;
	verify.commit = 0u;
	memcpy(&verify_other, &verify, sizeof(verify_other));
	verify_other.gather_context = 511u;
	memcpy(&tree_verify, &verify, sizeof(tree_verify));
	tree_verify.tree = 1u;
	tree_verify.host_slots = slots_a;
	tree_verify.host_parent_rows = slots_a;
	memcpy(&tree_verify_other, &tree_verify, sizeof(tree_verify_other));
	tree_verify_other.gather_context = 1023u;
	tree_verify_other.host_slots = slots_b;
	tree_verify_other.host_parent_rows = slots_b;
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
	if ( !IdentitiesMatch(&tree_verify, &tree_verify_other, 0u) )
	{
		printf("FAIL a tree verify step must replay its graph whatever its gather or host metadata pointers\n");
		return 1;
	}
	if ( IdentitiesMatch(&verify, &tree_verify, 0u) )
	{
		printf("FAIL a chain verify step and a tree verify step must not share identity\n");
		return 1;
	}
	if ( StageRunnerGraphSlotKey(&verify) != 2u || StageRunnerGraphSlotKey(&tree_verify) != 3u ||
		StageRunnerGraphSlot(StageRunnerGraphSlotKey(&verify), 24u, 0u, 0u) ==
		StageRunnerGraphSlot(StageRunnerGraphSlotKey(&tree_verify), 24u, 0u, 0u) )
	{
		printf("FAIL chain verify and tree verify must map to distinct graph slots\n");
		return 1;
	}
	return 0;
}

static int GraphSlotRules(void)
{
	static uint8_t seen[STAGE_RUNNER_GRAPH_SLOTS];
	SparkStageRunnerState state;
	SparkStageRunner runner;
	SparkStageRunnerModelInterface iface;
	uint16_t dummy_head = 0u;
	if ( STAGE_RUNNER_GRAPH_KEYS != 10u || STAGE_RUNNER_GRAPH_ROWS != 64u )
	{
		printf("FAIL expected 10 graph keys and 64 graph rows (got keys=%u rows=%u)\n", STAGE_RUNNER_GRAPH_KEYS, STAGE_RUNNER_GRAPH_ROWS);
		return 1;
	}
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
	memset(&state, 0, sizeof(state));
	memset(&runner, 0, sizeof(runner));
	memset(&iface, 0, sizeof(iface));
	state.model_interface = &iface;
	state.programs = 1u;
	state.tp_degree = 1u;
	state.head_weight = &dummy_head;
	runner.owns_embedding = 1u;
	runner.owns_final_head = 1u;
	if ( StageRunnerGraphEligible(&state, &runner, 1u, 32u, 32u) == 0u ||
		StageRunnerGraphEligible(&state, &runner, 1u, 64u, 64u) == 0u ||
		StageRunnerGraphEligible(&state, &runner, 1u, 65u, 65u) != 0u )
	{
		printf("FAIL decode graphs must cover B=32 and B=64 and stop past STAGE_RUNNER_GRAPH_ROWS\n");
		return 1;
	}
	return 0;
}

static uint16_t *UploadBf16(const uint16_t *host, size_t bytes)
{
	uint16_t *device = 0;
	if ( SparkGpuMalloc((void **)&device, bytes) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(device, host, bytes, SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return 0;
	return device;
}

static int AlternatingTreeGraphReplayRules(void)
{
	const uint32_t hidden = 256u, vocab = 256u, rows = 24u;
	static uint16_t embed[256u * 256u], head[256u * 256u], norm[256u];
	static const uint32_t host_a[24] = { 0u }, host_b[24] = { 1u };
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerModelInterface iface;
	SparkStageRunnerConfiguration config;
	SparkStageRunnerDispatch dispatch;
	SparkStageRunnerStep chain_step, tree_step;
	SparkDeviceProgram *chain_prog = 0, *tree_prog = 0;
	uint32_t *dummy_a = 0, *dummy_b = 0;
	uint32_t chain_slot, tree_slot;
	int failed = 0;
	memset(state, 0, sizeof(*state));
	memset(&runner, 0, sizeof(runner));
	memset(&iface, 0, sizeof(iface));
	memset(&config, 0, sizeof(config));
	memset(&dispatch, 0, sizeof(dispatch));
	memset(embed, 0, sizeof(embed));
	memset(head, 0, sizeof(head));
	for ( uint32_t token = 0u; token < vocab; ++token )
	{
		embed[(size_t)token * hidden + token] = 0x3f80u;
		head[(size_t)token * hidden + token] = 0x3f80u;
	}
	for ( uint32_t index = 0u; index < hidden; ++index )
		norm[index] = 0x3f80u;
	iface.tag = "verify-tree-graph";
	iface.step = MockVerifyStep;
	iface.graph_key = MockVerifyGraphKey;
	config.max_input_row_count = rows;
	config.max_active_sequence_count = rows;
	config.resident_sequence_capacity = rows;
	state->model_interface = &iface;
	state->tp_degree = 1u;
	state->max_rows = rows;
	state->head_capacity = rows;
	state->vocab = vocab;
	state->output_vocab = vocab;
	state->vocab_slice_rows = vocab;
	state->head_output_rows = vocab;
	state->sequence_capacity = rows;
	state->tap_final_entry = UINT32_MAX;
	state->geometry.hidden = hidden;
	state->geometry.vocab = vocab;
	state->geometry.rms_epsilon = 1e-6f;
	runner.private_state = state;
	runner.tp_degree = 1u;
	runner.owns_embedding = 1u;
	runner.owns_final_head = 1u;
	state->embed_weight = UploadBf16(embed, sizeof(embed));
	state->head_weight = UploadBf16(head, sizeof(head));
	state->head_norm_weight = UploadBf16(norm, sizeof(norm));
	if ( SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		state->embed_weight == 0 || state->head_weight == 0 || state->head_norm_weight == 0 ||
		SparkGpuMalloc(&state->head_certified_fp8_payload, (size_t)vocab * hidden) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_fp8_scale_f32, (size_t)vocab * (hidden / 32u) * sizeof(float)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_fp8_norm_f32, (size_t)vocab * (hidden / 32u) * sizeof(float)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_scratch, SparkHeadCertifiedFp8ScratchBytes(vocab, hidden) * rows) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_candidates, SparkHeadCertifiedFp8CandidateBytes(vocab) * rows) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_screened_count, sizeof(uint32_t) * rows) != SPARK_GPU_SUCCESS ||
		SparkLmHostLaunchHeadCertifiedFp8Quantize(0, state->head_weight, state->head_certified_fp8_payload,
			state->head_certified_fp8_scale_f32, state->head_certified_fp8_norm_f32, vocab, hidden) != SPARK_GPU_SUCCESS ||
		StageRunnerCreateBuffers(state, &runner, &config) != SPARK_STATUS_OK ||
		StageRunnerAdoptQueue(state) != SPARK_STATUS_OK ||
		SparkGpuMalloc((void **)&dummy_a, rows * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&dummy_b, rows * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemsetAsync(state->token_ids_device, 0, rows * sizeof(uint32_t), state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemsetAsync(state->chain_step_device, 0, sizeof(uint32_t), state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL setup for alternating tree verify graph replay test\n");
		delete state;
		return 1;
	}
	if ( StageRunnerGraphEligible(state, &runner, 2u, 24u, 1u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 2u, 32u, 1u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 2u, 64u, 1u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 2u, 65u, 1u) != 0u )
	{
		printf("FAIL graph row eligibility bounds (expected 24, 32 and 64 eligible, 65 ineligible)\n");
		failed = 1;
	}
	state->step_dispatch = &dispatch;
	memset(&chain_step, 0, sizeof(chain_step));
	chain_step.hidden_bf16 = state->hidden;
	chain_step.token_ids = state->token_ids_device;
	chain_step.positions = dummy_a;
	chain_step.slots = dummy_a;
	chain_step.rows = rows;
	chain_step.sequences = 1u;
	chain_step.commit = 0u;
	chain_step.graph_key = 2u;
	chain_step.tree = 0u;
	chain_step.gather_context = 100u;
	chain_step.host_positions = host_a;
	memcpy(&tree_step, &chain_step, sizeof(tree_step));
	tree_step.tree = 1u;
	tree_step.slots = dummy_b;
	tree_step.parent_rows = dummy_b;
	tree_step.host_slots = host_a;
	tree_step.host_parent_rows = host_a;
	chain_slot = StageRunnerGraphSlot(StageRunnerGraphSlotKey(&chain_step), rows, 0u, 0u);
	tree_slot = StageRunnerGraphSlot(StageRunnerGraphSlotKey(&tree_step), rows, 0u, 0u);
	mock_verify_steps = 0u;
	if ( failed == 0 && (StageRunnerStep(state, &runner, &chain_step, 0u, state->stream) != SPARK_STATUS_OK ||
		SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
	{
		printf("FAIL first chain verify graph capture at rows=%u\n", rows);
		failed = 1;
	}
	chain_prog = state->graph_program[chain_slot];
	if ( failed == 0 && (chain_prog == 0 || mock_verify_steps != 1u || state->graph_launches != 1u) )
	{
		printf("FAIL first chain verify did not capture one program (steps=%u launches=%llu)\n",
			mock_verify_steps, (unsigned long long)state->graph_launches);
		failed = 1;
	}
	if ( failed == 0 && (StageRunnerStep(state, &runner, &tree_step, 0u, state->stream) != SPARK_STATUS_OK ||
		SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
	{
		printf("FAIL first tree verify graph capture at rows=%u\n", rows);
		failed = 1;
	}
	tree_prog = state->graph_program[tree_slot];
	if ( failed == 0 && (tree_prog == 0 || state->graph_program[chain_slot] != chain_prog ||
		mock_verify_steps != 2u || state->graph_launches != 2u) )
	{
		printf("FAIL first tree verify collided with chain verify slot (steps=%u launches=%llu)\n",
			mock_verify_steps, (unsigned long long)state->graph_launches);
		failed = 1;
	}
	chain_step.gather_context = 200u;
	chain_step.host_positions = host_b;
	if ( failed == 0 && (StageRunnerStep(state, &runner, &chain_step, 0u, state->stream) != SPARK_STATUS_OK ||
		SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
	{
		printf("FAIL second chain verify graph replay at rows=%u\n", rows);
		failed = 1;
	}
	if ( failed == 0 && (state->graph_program[chain_slot] != chain_prog || state->graph_program[tree_slot] != tree_prog ||
		mock_verify_steps != 2u || state->graph_launches != 3u) )
	{
		printf("FAIL alternating chain verify triggered recapture (steps=%u launches=%llu)\n",
			mock_verify_steps, (unsigned long long)state->graph_launches);
		failed = 1;
	}
	tree_step.gather_context = 300u;
	tree_step.host_slots = host_b;
	tree_step.host_parent_rows = host_b;
	if ( failed == 0 && (StageRunnerStep(state, &runner, &tree_step, 0u, state->stream) != SPARK_STATUS_OK ||
		SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
	{
		printf("FAIL second tree verify graph replay at rows=%u\n", rows);
		failed = 1;
	}
	if ( failed == 0 && (state->graph_program[chain_slot] != chain_prog || state->graph_program[tree_slot] != tree_prog ||
		mock_verify_steps != 2u || state->graph_launches != 4u) )
	{
		printf("FAIL alternating tree verify triggered recapture (steps=%u launches=%llu)\n",
			mock_verify_steps, (unsigned long long)state->graph_launches);
		failed = 1;
	}
	if ( failed == 0 && (StageRunnerGraphEligible(state, &runner, 1u, 16u, 4u, 0u, 0u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 2u, 16u, 4u, 0u, 0u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 1u, 24u, 8u, 0u, 0u) != 1u ||
		StageRunnerGraphEligible(state, &runner, 1u, 16u, 4u, 1u, 0u) != 0u ||
		StageRunnerGraphEligible(state, &runner, 1u, 16u, 4u, 0u, 1u) != 0u ||
		StageRunnerGraphEligible(state, &runner, 0u, 16u, 4u, 0u, 0u) != 0u) )
	{
		printf("FAIL multi-lane verify graph eligibility rules\n");
		failed = 1;
	}
	{
		SparkStageRunnerStep lane4_step, lane8_step, lane1_step;
		SparkDeviceProgram *lane4_prog = 0, *lane8_prog = 0, *lane1_prog = 0;
		uint32_t lane4_slot, lane8_slot, lane1_slot;
		uint32_t tokens_a[24], tokens_b[24], out_tokens[24];
		float out_scores[24];
		for ( uint32_t index = 0u; index < 24u; ++index )
		{
			tokens_a[index] = (index * 7u + 3u) % vocab;
			tokens_b[index] = (index * 11u + 5u) % vocab;
		}
		memcpy(&lane4_step, &chain_step, sizeof(lane4_step));
		lane4_step.rows = 16u;
		lane4_step.sequences = 4u;
		lane4_step.graph_key = 1u;
		lane4_step.gather_context = 0u;
		memcpy(&lane8_step, &lane4_step, sizeof(lane8_step));
		lane8_step.rows = 24u;
		lane8_step.sequences = 8u;
		memcpy(&lane1_step, &lane4_step, sizeof(lane1_step));
		lane1_step.rows = 16u;
		lane1_step.sequences = 1u;
		lane1_step.graph_key = 2u;
		lane4_slot = StageRunnerGraphSlot(StageRunnerGraphSlotKey(&lane4_step), 16u, 0u, 0u);
		lane8_slot = StageRunnerGraphSlot(StageRunnerGraphSlotKey(&lane8_step), 24u, 0u, 0u);
		lane1_slot = StageRunnerGraphSlot(StageRunnerGraphSlotKey(&lane1_step), 16u, 0u, 0u);
		if ( failed == 0 && (StageRunnerGraphSlotKey(&lane4_step) != 6u || StageRunnerGraphSlotKey(&lane8_step) != 10u ||
			lane4_slot == lane1_slot || lane4_slot == lane8_slot) )
		{
			printf("FAIL multi-lane verify slot keys collided (k4=%u k8=%u s1=%u s4=%u s8=%u)\n",
				StageRunnerGraphSlotKey(&lane4_step), StageRunnerGraphSlotKey(&lane8_step), lane1_slot, lane4_slot, lane8_slot);
			failed = 1;
		}
		if ( failed == 0 && (SparkGpuMemcpyAsync(state->token_ids_device, tokens_a, 16u * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
			StageRunnerStep(state, &runner, &lane4_step, 0u, state->stream) != SPARK_STATUS_OK ||
			SparkGpuMemcpyAsync(out_tokens, state->output_token, 16u * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpyAsync(out_scores, state->output_score, 16u * sizeof(float), SPARK_GPU_COPY_DEVICE_TO_HOST, state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		{
			printf("FAIL multi-lane verify (B=4, rows=16) capture\n");
			failed = 1;
		}
		lane4_prog = state->graph_program[lane4_slot];
		if ( failed == 0 )
			for ( uint32_t index = 0u; index < 16u; ++index )
				if ( out_tokens[index] != tokens_a[index] || !(out_scores[index] > 0.0f) )
				{
					printf("FAIL multi-lane verify (B=4, rows=16) capture output mismatch row=%u got=%u want=%u score=%.3f\n",
						index, out_tokens[index], tokens_a[index], out_scores[index]);
					failed = 1;
					break;
				}
		if ( failed == 0 && (lane4_prog == 0 || mock_verify_steps != 3u || state->graph_launches != 5u) )
		{
			printf("FAIL multi-lane verify (B=4, rows=16) did not capture one program (steps=%u launches=%llu)\n",
				mock_verify_steps, (unsigned long long)state->graph_launches);
			failed = 1;
		}
		if ( failed == 0 && (SparkGpuMemcpyAsync(state->token_ids_device, tokens_a, 24u * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
			StageRunnerStep(state, &runner, &lane8_step, 0u, state->stream) != SPARK_STATUS_OK ||
			SparkGpuMemcpyAsync(out_tokens, state->output_token, 24u * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		{
			printf("FAIL multi-lane verify (B=8, rows=24) capture\n");
			failed = 1;
		}
		lane8_prog = state->graph_program[lane8_slot];
		if ( failed == 0 )
			for ( uint32_t index = 0u; index < 24u; ++index )
				if ( out_tokens[index] != tokens_a[index] )
				{
					printf("FAIL multi-lane verify (B=8, rows=24) capture output mismatch row=%u got=%u want=%u\n",
						index, out_tokens[index], tokens_a[index]);
					failed = 1;
					break;
				}
		if ( failed == 0 && (StageRunnerStep(state, &runner, &lane1_step, 0u, state->stream) != SPARK_STATUS_OK ||
			SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		{
			printf("FAIL single-lane verify (B=1, rows=16) capture alongside multi-lane\n");
			failed = 1;
		}
		lane1_prog = state->graph_program[lane1_slot];
		if ( failed == 0 && (lane8_prog == 0 || lane1_prog == 0 || state->graph_program[lane4_slot] != lane4_prog ||
			mock_verify_steps != 5u || state->graph_launches != 7u) )
		{
			printf("FAIL multi-lane verify collided with single-lane slot (steps=%u launches=%llu)\n",
				mock_verify_steps, (unsigned long long)state->graph_launches);
			failed = 1;
		}
		lane4_step.host_positions = host_b;
		if ( failed == 0 && (SparkGpuMemcpyAsync(state->token_ids_device, tokens_b, 16u * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
			StageRunnerStep(state, &runner, &lane4_step, 0u, state->stream) != SPARK_STATUS_OK ||
			SparkGpuMemcpyAsync(out_tokens, state->output_token, 16u * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		{
			printf("FAIL multi-lane verify (B=4, rows=16) replay\n");
			failed = 1;
		}
		if ( failed == 0 )
			for ( uint32_t index = 0u; index < 16u; ++index )
				if ( out_tokens[index] != tokens_b[index] )
				{
					printf("FAIL multi-lane verify (B=4, rows=16) replay output mismatch row=%u got=%u want=%u\n",
						index, out_tokens[index], tokens_b[index]);
					failed = 1;
					break;
				}
		lane8_step.host_positions = host_b;
		if ( failed == 0 && (SparkGpuMemcpyAsync(state->token_ids_device, tokens_b, 24u * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
			StageRunnerStep(state, &runner, &lane8_step, 0u, state->stream) != SPARK_STATUS_OK ||
			SparkGpuMemcpyAsync(out_tokens, state->output_token, 24u * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		{
			printf("FAIL multi-lane verify (B=8, rows=24) replay\n");
			failed = 1;
		}
		if ( failed == 0 )
			for ( uint32_t index = 0u; index < 24u; ++index )
				if ( out_tokens[index] != tokens_b[index] )
				{
					printf("FAIL multi-lane verify (B=8, rows=24) replay output mismatch row=%u got=%u want=%u\n",
						index, out_tokens[index], tokens_b[index]);
					failed = 1;
					break;
				}
		if ( failed == 0 && (state->graph_program[lane4_slot] != lane4_prog || state->graph_program[lane8_slot] != lane8_prog ||
			state->graph_program[lane1_slot] != lane1_prog || mock_verify_steps != 5u || state->graph_launches != 9u) )
		{
			printf("FAIL multi-lane verify replay triggered recapture (steps=%u launches=%llu)\n",
				mock_verify_steps, (unsigned long long)state->graph_launches);
			failed = 1;
		}
	}
	StageRunnerGraphDrop(state);
	StageRunnerGraphPrefetchFree(state);
	(void)SparkGpuFree(dummy_a);
	(void)SparkGpuFree(dummy_b);
	delete state;
	return failed;
}

static SparkStatus MockKeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream)
{
	(void)model;
	(void)step;
	(void)kept;
	(void)stream;
	return SPARK_STATUS_OK;
}

static int TreeSlotGrainRules(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerModelInterface iface;
	uint32_t dummy_path = 0u;
	memset(state, 0, sizeof(*state));
	memset(&runner, 0, sizeof(runner));
	memset(&iface, 0, sizeof(iface));
	iface.keep_rows = MockKeepRows;
	state->model_interface = &iface;
	state->geometry.tree_verify = 1u;
	state->geometry.kv_shard.degree = 16u;
	state->geometry.kv_shard.grain = 8u;
	state->verify_path_rows = &dummy_path;
	runner.private_state = state;
	const uint32_t fallback = SparkStageRunnerTreeSlotGrain(&runner);
	state->geometry.tree_slot_grain = 256u;
	const uint32_t explicit_grain = SparkStageRunnerTreeSlotGrain(&runner);
	delete state;
	if ( fallback != 128u || explicit_grain != 256u )
	{
		printf("FAIL tree slot grain rules: fallback=%u explicit=%u\n", fallback, explicit_grain);
		return 1;
	}
	return 0;
}

int main(void)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunner runner;
	SparkStageRunnerStep in;
	int status = 0;
	if ( GraphIdentityRules() != 0 || GraphSlotRules() != 0 || TreeSlotGrainRules() != 0 )
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
	if ( AlternatingTreeGraphReplayRules() != 0 )
		return 1;
	printf("PASS runner_capture_flags_cuda: graph replay follows the gather rule, alternating chain and tree verify graphs up to 64 rows keep separate slots without recapture, and a decode graph that cannot be captured aborts the rank\n");
	return 0;
}
