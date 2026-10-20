#include "modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu"

#include <stdio.h>
#include <stdlib.h>

#define PROBE_POOL 3u

static uint32_t probe_rounds,probe_fails;
static int probe_failures;

static SparkStatus ProbeRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements, const void *local, void *full)
{
	(void)context; (void)stream; (void)operation; (void)stage; (void)rows; (void)row_elements; (void)local; (void)full;
	probe_rounds++;
	return SPARK_STATUS_OK;
}

static void ProbeFail(void *context, SparkStatus status)
{
	(void)context; (void)status;
	probe_fails++;
}

static uint32_t ProbeFree(const SparkK3ModelState *state)
{
	uint32_t count = 0u;
	for (const SparkK3RunnerTpContext *context = state->tp_context_free_head; context != 0; context = context->pool_next)
		count++;
	return count;
}

static void ProbeCheck(int condition, const char *message)
{
	if ( condition == 0 )
	{
		printf("FAIL %s (rounds %u fails %u)\n", message, probe_rounds, probe_fails);
		probe_failures++;
	}
}

static void *ProbeDevice(uint64_t bytes)
{
	void *pointer = 0;
	if ( cudaMalloc(&pointer, bytes) != cudaSuccess || cudaMemset(pointer, 0, bytes) != cudaSuccess )
	{
		printf("FAIL cuda allocation of %llu bytes\n", (unsigned long long)bytes);
		exit(1);
	}
	return pointer;
}

int main(void)
{
	static SparkStageRunnerServices services;
	SparkK3ModelState *state = (SparkK3ModelState *)calloc(1u, sizeof(SparkK3ModelState));
	K3LayerBuffers *buffers = (K3LayerBuffers *)calloc(1u, sizeof(K3LayerBuffers));
	const uint64_t bytes = (uint64_t)4u * K3_HIDDEN * sizeof(uint16_t);
	uint32_t index;
	if ( state == 0 || buffers == 0 )
		return 1;
	services.round = ProbeRound;
	services.fail = ProbeFail;
	state->services = &services;
	state->has_collective = 1u;
	state->rows = 1u;
	state->fused_device = (uint16_t *)ProbeDevice(bytes);
	buffers->tp_sharded = 1u;
	buffers->hidden_bf16 = (uint16_t *)ProbeDevice(bytes);
	buffers->attention_out_bf16 = (uint16_t *)ProbeDevice(bytes);
	buffers->shared_out_bf16 = (uint16_t *)ProbeDevice(bytes);
	buffers->gate_up_bf16 = (uint16_t *)ProbeDevice(bytes);
	state->dispatch.buffers = buffers;
	state->lanes[0].state = state;
	for (index = 0u; index < PROBE_POOL; ++index)
	{
		state->tp_context_pool[index].owner = state;
		state->tp_context_pool[index].pool_next = index + 1u < PROBE_POOL ? &state->tp_context_pool[index + 1u] : 0;
	}
	state->tp_context_free_head = &state->tp_context_pool[0];

	K3RunnerLayerCollective(&state->lanes[0], 0, 1u, 0u | K3_COLLECTIVE_BEGIN);
	ProbeCheck(probe_rounds == 1u && probe_fails == 0u && state->pending_tp != 0 && ProbeFree(state) == PROBE_POOL - 1u, "a begun round holds one context");
	K3RunnerLayerCollective(&state->lanes[0], 0, 1u, 0u | K3_COLLECTIVE_BEGIN);
	ProbeCheck(probe_rounds == 2u && probe_fails == 1u && ProbeFree(state) == PROBE_POOL - 1u,
		"a second begin before the finish fails loudly and returns the first context instead of leaking it");
	K3RunnerTpStepReset(state);
	ProbeCheck(state->pending_tp == 0 && ProbeFree(state) == PROBE_POOL && state->tp_context_overflow == 0u,
		"a new step returns a context left pending by an aborted step");

	probe_rounds = probe_fails = 0u;
	K3RunnerLayerCollective(&state->lanes[0], 0, 1u, 0u | K3_COLLECTIVE_FINISH);
	ProbeCheck(probe_rounds == 1u && probe_fails == 1u, "a finish without its begin fails loudly and still takes part in the round");

	probe_rounds = probe_fails = 0u;
	state->tp_context_free_head = 0;
	K3RunnerLayerCollective(&state->lanes[0], 0, 1u, 0u);
	ProbeCheck(probe_rounds == 1u && probe_fails == 1u, "an exhausted context pool fails loudly and still takes part in the round, so peers are not left waiting");
	K3RunnerLayerCollective(&state->lanes[0], 0, 1u, 2u);
	ProbeCheck(probe_rounds == 2u && probe_fails == 2u, "the gate-up round also takes part when no context is left");
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("FAIL device error after the rounds\n");
		return 1;
	}
	if ( probe_failures != 0 )
	{
		printf("FAIL k3_tp_context_cuda: %d failures\n", probe_failures);
		return 1;
	}
	printf("PASS k3_tp_context_cuda: K3 never skips a collective round it owes, fails loudly when it cannot apply one, and a new step reclaims a stranded context\n");
	return 0;
}
