
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
#define THREADS 1u
#define ELEMENTS (HEADS * KEY_DIM * VALUE_DIM)
#define SLOT_F32 (ELEMENTS * 4u)
#define SLOT_BF16 (ELEMENTS * 2u)

static uint16_t bf16(float value) { return LmFloatToBf16(value); }
static float f32(uint16_t value) { return LmBf16ToFloat(value); }

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return (float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static float Bf16UlpsAtScale(float scale, float reference, float rounded)
{
	int exponent;
	if ( reference == rounded || !(scale > 0.0f) )
		return 0.0f;
	frexpf(scale,&exponent);
	return fabsf(reference - rounded) / ldexpf(1.0f,exponent - 8);
}

static void StepBoth(uint8_t *pool_f32, uint8_t *pool_bf16, const uint16_t *query, const uint16_t *key, const uint16_t *value, const float *retention, const float *write_gate, uint16_t *out_f32, uint16_t *out_bf16)
{
	uint32_t state_index[1] = { 0u };
	LM_HOST_LAUNCH(dim3(1u, HEADS),
		(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM, float>(
			pool_f32, SLOT_F32, state_index, 0, 0, query, key, value,
			retention, write_gate, out_f32, HEADS, 1u, 1u, 1u)));
	LM_HOST_LAUNCH(dim3(1u, HEADS),
		(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM, uint16_t>(
			pool_bf16, SLOT_BF16, state_index, 0, 0, query, key, value,
			retention, write_gate, out_bf16, HEADS, 1u, 1u, 1u)));
}

int main(void)
{
	uint32_t seed = 777u,index,step,trial;

	{
		static uint8_t pool_f32[SLOT_F32],pool_bf16[SLOT_BF16];
		static uint16_t query[HEADS * KEY_DIM],key[HEADS * KEY_DIM],value[HEADS * VALUE_DIM];
		static uint16_t out_f32[HEADS * VALUE_DIM],out_bf16[HEADS * VALUE_DIM];
		static float retention[HEADS * KEY_DIM],write_gate[HEADS];
		uint32_t output_mismatch = 0u,state_mismatch = 0u;
		for (index = 0u; index < ELEMENTS; ++index)
		{
			float exact = f32(bf16(NextRandom(&seed)));
			((float *)pool_f32)[index] = exact;
			((uint16_t *)pool_bf16)[index] = bf16(exact);
		}
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
			retention[index] = 0.5f;
		for (index = 0u; index < HEADS; ++index)
			write_gate[index] = 1.0f;
		memset(key,0,sizeof(key));
		for (step = 0u; step < 8u; ++step)
		{
			for (index = 0u; index < HEADS * KEY_DIM; ++index)
				query[index] = bf16(NextRandom(&seed));
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				value[index] = bf16(NextRandom(&seed));
			StepBoth(pool_f32,pool_bf16,query,key,value,retention,write_gate,
				out_f32,out_bf16);
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				if ( out_f32[index] != out_bf16[index] )
					++output_mismatch;
			for (index = 0u; index < ELEMENTS; ++index)
				if ( ((uint16_t *)pool_bf16)[index]
					!= bf16(((float *)pool_f32)[index]) )
					++state_mismatch;
		}
		printf("exact_output_mismatch %u\n",output_mismatch);
		printf("exact_state_mismatch %u\n",state_mismatch);
	}

	{
		static uint8_t pool_f32[SLOT_F32],pool_bf16[SLOT_BF16];
		static uint16_t query[HEADS * KEY_DIM],key[HEADS * KEY_DIM],value[HEADS * VALUE_DIM];
		static uint16_t out_f32[HEADS * VALUE_DIM],out_bf16[HEADS * VALUE_DIM];
		static float retention[HEADS * KEY_DIM],write_gate[HEADS];
		uint32_t output_mismatch = 0u,store_mismatch = 0u;
		for (trial = 0u; trial < 64u; ++trial)
		{
			for (index = 0u; index < ELEMENTS; ++index)
			{
				float exact = f32(bf16(NextRandom(&seed)));
				((float *)pool_f32)[index] = exact;
				((uint16_t *)pool_bf16)[index] = bf16(exact);
			}
			for (index = 0u; index < HEADS * KEY_DIM; ++index)
			{
				query[index] = bf16(NextRandom(&seed));
				key[index] = bf16(NextRandom(&seed));
				retention[index] = 0.5f
					+ 0.25f * (NextRandom(&seed) + 1.0f);
			}
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				value[index] = bf16(NextRandom(&seed));
			for (index = 0u; index < HEADS; ++index)
				write_gate[index] = 0.25f
					+ 0.25f * (NextRandom(&seed) + 1.0f);
			StepBoth(pool_f32,pool_bf16,query,key,value,retention,write_gate,
				out_f32,out_bf16);
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				if ( out_f32[index] != out_bf16[index] )
					++output_mismatch;
			for (index = 0u; index < ELEMENTS; ++index)
				if ( ((uint16_t *)pool_bf16)[index]
					!= bf16(((float *)pool_f32)[index]) )
					++store_mismatch;
		}
		printf("step_trials %u\n",64u);
		printf("step_output_mismatch %u\n",output_mismatch);
		printf("step_store_mismatch %u\n",store_mismatch);
	}

	{
		static uint8_t pool_f32[SLOT_F32],pool_bf16[SLOT_BF16];
		static uint16_t query[HEADS * KEY_DIM],key[HEADS * KEY_DIM],value[HEADS * VALUE_DIM];
		static uint16_t out_f32[HEADS * VALUE_DIM],out_bf16[HEADS * VALUE_DIM];
		static float retention[HEADS * KEY_DIM],write_gate[HEADS];
		float alpha_max = 0.0f,worst = 0.0f,worst_early = 0.0f,worst_late = 0.0f,envelope;
		for (index = 0u; index < ELEMENTS; ++index)
		{
			float raw = NextRandom(&seed);
			((float *)pool_f32)[index] = raw;
			((uint16_t *)pool_bf16)[index] = bf16(raw);
		}
		for (step = 0u; step < 64u; ++step)
		{
			for (index = 0u; index < HEADS * KEY_DIM; ++index)
			{
				query[index] = bf16(NextRandom(&seed));
				key[index] = bf16(NextRandom(&seed));
				retention[index] = 0.5f
					+ 0.225f * (NextRandom(&seed) + 1.0f);
				if ( retention[index] > alpha_max )
					alpha_max = retention[index];
			}
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				value[index] = bf16(NextRandom(&seed));
			for (index = 0u; index < HEADS; ++index)
				write_gate[index] = 0.25f
					+ 0.25f * (NextRandom(&seed) + 1.0f);
			StepBoth(pool_f32,pool_bf16,query,key,value,retention,write_gate,
				out_f32,out_bf16);
			{
				float scale = 0.0f;
				for (index = 0u; index < ELEMENTS; ++index)
					if ( fabsf(((float *)pool_f32)[index]) > scale )
						scale = fabsf(((float *)pool_f32)[index]);
				for (index = 0u; index < ELEMENTS; ++index)
				{
					float ulps = Bf16UlpsAtScale(scale,
						((float *)pool_f32)[index],
						f32(((uint16_t *)pool_bf16)[index]));
				if ( ulps > worst )
					worst = ulps;
				if ( step < 32u )
				{
					if ( ulps > worst_early )
						worst_early = ulps;
				}
					else if ( ulps > worst_late )
						worst_late = ulps;
				}
			}
		}
		envelope = (0.5f / (1.0f - alpha_max)) + 1.0f;
		printf("ulp_steps %u\n",64u);
		printf("ulp_alpha_max %.6g\n",(double)alpha_max);
		printf("ulp_max %.6g\n",(double)worst);
		printf("ulp_envelope %.6g\n",(double)envelope);
		printf("ulp_first_half %.6g\n",(double)worst_early);
		printf("ulp_second_half %.6g\n",(double)worst_late);
	}

	{
		static uint8_t decode_pool[SLOT_BF16],fold_pool[SLOT_BF16],prefix_pool[SLOT_BF16],checkpoint[SLOT_BF16];
		static uint16_t query[8u][HEADS * KEY_DIM],key[8u][HEADS * KEY_DIM],value[8u][HEADS * VALUE_DIM];
		static uint16_t out[HEADS * VALUE_DIM];
		static float retention[8u][HEADS * KEY_DIM],write_gate[HEADS];
		static LmReplayStep steps[8u];
		static uint32_t accepted[1];
		uint32_t state_index[1] = { 0u };
		uint32_t mismatch = 0u,prefix_mismatch = 0u;
		for (index = 0u; index < ELEMENTS; ++index)
		{
			uint16_t initial = bf16(NextRandom(&seed));
			((uint16_t *)decode_pool)[index] = initial;
			((uint16_t *)fold_pool)[index] = initial;
		}
		for (step = 0u; step < 8u; ++step)
		{
			for (index = 0u; index < HEADS * KEY_DIM; ++index)
			{
				query[step][index] = bf16(NextRandom(&seed));
				key[step][index] = bf16(NextRandom(&seed));
				retention[step][index] = 0.5f
					+ 0.25f * (NextRandom(&seed) + 1.0f);
			}
			for (index = 0u; index < HEADS * VALUE_DIM; ++index)
				value[step][index] = bf16(NextRandom(&seed));
		}
		for (index = 0u; index < HEADS; ++index)
			write_gate[index] = 0.25f + 0.25f * (NextRandom(&seed) + 1.0f);
		for (step = 0u; step < 8u; ++step)
		{
			steps[step].key_bf16 = key[step];
			steps[step].value_bf16 = value[step];
			steps[step].retention = retention[step];
			steps[step].write_gate = write_gate;
			LM_HOST_LAUNCH(dim3(1u, HEADS),
				(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM, uint16_t>(
					decode_pool, SLOT_BF16, state_index, 0, 0,
					query[step], key[step], value[step],
					retention[step], write_gate, out, HEADS, 1u, 1u, 1u)));
			if ( step == 2u )
				memcpy(checkpoint,decode_pool,sizeof(checkpoint));
		}
		accepted[0] = 8u;
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmReplayFoldKernel<THREADS, KEY_DIM, VALUE_DIM, uint16_t>(
				fold_pool, SLOT_BF16, state_index, steps, accepted,
				HEADS, 1u, 1u)));
		accepted[0] = 5u;
		memcpy(prefix_pool,checkpoint,sizeof(prefix_pool));
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmReplayFoldKernel<THREADS, KEY_DIM, VALUE_DIM, uint16_t>(
				prefix_pool, SLOT_BF16, state_index, &steps[3u], accepted,
				HEADS, 1u, 1u)));
		for (index = 0u; index < sizeof(decode_pool); ++index)
			if ( fold_pool[index] != decode_pool[index] )
				++mismatch;
		for (index = 0u; index < sizeof(decode_pool); ++index)
			if ( prefix_pool[index] != decode_pool[index] )
				++prefix_mismatch;
		printf("fold_mismatch %u of %u bytes\n",mismatch,(unsigned)sizeof(decode_pool));
		printf("fold_prefix_mismatch %u\n",prefix_mismatch);
	}
	return 0;
}
