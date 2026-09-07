
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/linear_attn.cuh"

#define HEADS 2u
#define KEY_DIM 4u
#define VALUE_DIM 4u
#define CONV_KERNEL 4u
#define STEPS 6u
#define THREADS 1u
#define SLOT_BYTES (HEADS * KEY_DIM * VALUE_DIM * 4u)

static uint16_t bf16(float value) { return LmFloatToBf16(value); }
static float f32(uint16_t value) { return LmBf16ToFloat(value); }

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return (float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

int main(void)
{
	static uint16_t query[STEPS][HEADS * KEY_DIM];
	static uint16_t key[STEPS][HEADS * KEY_DIM];
	static uint16_t value[STEPS][HEADS * VALUE_DIM];
	static uint16_t decay_logit[STEPS][HEADS * KEY_DIM];
	static uint16_t output[STEPS][HEADS * VALUE_DIM];
	static float retention[HEADS * KEY_DIM];
	static float write_gate[HEADS];
	static float channel_bias[HEADS * KEY_DIM];
	static float head_log_scale[HEADS];
	static uint8_t state_pool[HEADS * KEY_DIM * VALUE_DIM * 4u];
	static uint32_t state_index[1] = { 0u };
	uint32_t seed = 12345u, step, index, head;

	memset(state_pool, 0, sizeof(state_pool));
	for (index = 0u; index < HEADS * KEY_DIM; ++index)
		channel_bias[index] = NextRandom(&seed) * 0.5f;
	for (head = 0u; head < HEADS; ++head)
		head_log_scale[head] = NextRandom(&seed) * 0.3f;
	for (step = 0u; step < STEPS; ++step)
	{
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
		{
			query[step][index] = bf16(NextRandom(&seed));
			key[step][index] = bf16(NextRandom(&seed));
			decay_logit[step][index] = bf16(NextRandom(&seed) * 2.0f);
		}
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			value[step][index] = bf16(NextRandom(&seed));
		for (head = 0u; head < HEADS; ++head)
			write_gate[head] = 0.5f + 0.25f * NextRandom(&seed);
	}

	printf("HEADS %u KEY_DIM %u VALUE_DIM %u STEPS %u GMIN %.9g\n",
		HEADS, KEY_DIM, VALUE_DIM, STEPS, (double)-5.0f);
	for (index = 0u; index < HEADS * KEY_DIM; ++index)
		printf("bias %.9g\n", (double)channel_bias[index]);
	for (head = 0u; head < HEADS; ++head)
		printf("scale %.9g\n", (double)head_log_scale[head]);
	for (step = 0u; step < STEPS; ++step)
	{
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
			printf("q %.9g\nk %.9g\nz %.9g\n",
				(double)f32(query[step][index]), (double)f32(key[step][index]),
				(double)f32(decay_logit[step][index]));
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			printf("v %.9g\n", (double)f32(value[step][index]));
		for (head = 0u; head < HEADS; ++head)
			printf("beta %.9g\n", (double)write_gate[head]);
	}

	for (step = 0u; step < STEPS; ++step)
	{
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmBoundedDecayKernel<THREADS, KEY_DIM>(
				decay_logit[step], channel_bias, head_log_scale,
				retention, HEADS, -5.0f, 1u)));
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM>(
				state_pool, SLOT_BYTES, state_index, 0, 0, query[step], key[step], value[step],
				retention, write_gate, output[step], HEADS, 1u, 1u, 1u)));
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			printf("out %.9g\n", (double)f32(output[step][index]));
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
			printf("ret %.9g\n", (double)retention[index]);
	}
	{
		static uint8_t replay_pool[sizeof(state_pool)];
		static LmReplayStep steps[STEPS];
		static float replay_retention[STEPS][HEADS * KEY_DIM];
		static uint32_t accepted[1];
		uint32_t mismatch = 0u;
		memset(replay_pool, 0, sizeof(replay_pool));
		for (step = 0u; step < STEPS; ++step)
		{
			LM_HOST_LAUNCH(dim3(1u, HEADS),
				(LmBoundedDecayKernel<THREADS, KEY_DIM>(
					decay_logit[step], channel_bias, head_log_scale,
					replay_retention[step], HEADS, -5.0f, 1u)));
			steps[step].key_bf16 = key[step];
			steps[step].value_bf16 = value[step];
			steps[step].retention = replay_retention[step];
			steps[step].write_gate = write_gate;
		}
		accepted[0] = STEPS;
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmReplayFoldKernel<THREADS, KEY_DIM, VALUE_DIM>(
				replay_pool, SLOT_BYTES, state_index, steps, accepted, HEADS, 1u, 1u)));
		for (index = 0u; index < sizeof(state_pool); ++index)
			if (replay_pool[index] != state_pool[index])
				++mismatch;
		printf("replay_mismatch %u of %u bytes\n",
			mismatch, (unsigned)sizeof(state_pool));
	}
	{
		static uint8_t run_pool[sizeof(state_pool)];
		static uint16_t run_output[STEPS][HEADS * VALUE_DIM];
		static uint16_t run_query[STEPS][HEADS * KEY_DIM];
		static uint16_t run_key[STEPS][HEADS * KEY_DIM];
		static uint16_t run_value[STEPS][HEADS * VALUE_DIM];
		static float run_retention[STEPS][HEADS * KEY_DIM];
		static float run_beta[STEPS][HEADS];
		static uint32_t run_begin[2];
		uint32_t mismatch = 0u;
		memset(run_pool, 0, sizeof(run_pool));
		for (step = 0u; step < STEPS; ++step)
		{
			memcpy(run_query[step], query[step], sizeof(run_query[step]));
			memcpy(run_key[step], key[step], sizeof(run_key[step]));
			memcpy(run_value[step], value[step], sizeof(run_value[step]));
			memcpy(run_beta[step], write_gate, sizeof(run_beta[step]));
			LM_HOST_LAUNCH(dim3(1u, HEADS),
				(LmBoundedDecayKernel<THREADS, KEY_DIM>(
					decay_logit[step], channel_bias, head_log_scale,
					run_retention[step], HEADS, -5.0f, 1u)));
		}
		run_begin[0] = 0u; run_begin[1] = STEPS;
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM>(
				run_pool, SLOT_BYTES, state_index, run_begin, 0,
				run_query[0], run_key[0], run_value[0],
				run_retention[0], run_beta[0], run_output[0],
				HEADS, 1u, 1u, 1u)));
		for (step = 0u; step < STEPS; ++step)
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				if (run_output[step][index] != output[step][index])
					++mismatch;
		for (index = 0u; index < sizeof(state_pool); ++index)
			if (run_pool[index] != state_pool[index])
				++mismatch;
		printf("run_mismatch %u\n", mismatch);
		memset(run_pool, 0, sizeof(run_pool));
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM>(
				run_pool, SLOT_BYTES, state_index, run_begin, 0,
				run_query[0], run_key[0], run_value[0],
				run_retention[0], run_beta[0], run_output[0],
				HEADS, 1u, 1u, 0u)));
		mismatch = 0u;
		for (index = 0u; index < sizeof(run_pool); ++index)
			if (run_pool[index] != 0u)
				++mismatch;
		for (step = 0u; step < STEPS; ++step)
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				if (run_output[step][index] != output[step][index])
					++mismatch;
		printf("verify_mismatch %u\n", mismatch);
	}
	return 0;
}
