// Prove the bf16-state delta-rule variant keeps the fp32 kernel's contracts.
//
// linear_attn.cuh's LmDeltaRuleKernel and LmReplayFoldKernel take the state
// pool's element type as a template parameter: float for the contract slot,
// uint16_t (bf16) for the admission-time half-width option of kimi_k3's
// K3_KDA_STATE_SLOT_BYTES_BF16. The variant upcasts the slot into the same
// fp32 shared tile, runs the recurrence in fp32 exactly as the fp32 kernel,
// and rounds only on the commit store. This harness runs BOTH instantiations
// of the real kernels - the code the device compiles, given a grid by the
// shim - and measures the three properties the option signs:
//
//   1. GIVEN THE SAME FP32 INPUT STATE, ONE STEP IS BIT-IDENTICAL. Over many
//      random trials the two kernels must produce the same bf16 outputs bit
//      for bit, and the bf16 pool must hold exactly LmFloatToBf16 of the
//      fp32 pool - the commit rounding and nothing else. A multi-step run
//      whose states all round-trip losslessly (an exact dyadic decay) must
//      agree on every output AND every committed state element.
//   2. WITH ARBITRARY FP32 STATE THE DIVERGENCE IS BOUNDED. Each commit
//      re-rounds every element (at most half a bf16 ulp, round-to-nearest-
//      even), and the recurrence contracts that rounding by at most the
//      retention factor alpha per step - the error map is
//      Diag(alpha) (I - beta k k^T), spectral norm at most alpha_max - so
//      the divergence from the fp32 kernel follows a geometric series
//      bounded by 0.5 / (1 - alpha_max) bf16 ulps at the state's operating
//      magnitude. Measured over 64 synthetic commits, not assumed.
//   3. THE REPLAY FOLD STAYS BYTE-EXACT. Folding the accepted prefix under
//      bf16 state must land on the state serial decode committed, byte for
//      byte - the ReplaySSM property, now across a rounding boundary,
//      holding because the fold upcasts and re-rounds at the same two
//      points once per committed step.
//
// tests/test_kda_bf16_state.py drives this and owns the assertions.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

// Backing for the kernels' dynamic shared memory, as in kda_host.cu: one
// fixed buffer per declared name, because one block runs at a time.
uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

// One lane per warp at THREADS == 1, same override as kda_host.cu.
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/linear_attn.cuh"

#define HEADS 2u
#define KEY_DIM 4u
#define VALUE_DIM 4u
#define THREADS 1u
#define ELEMENTS (HEADS * KEY_DIM * VALUE_DIM)
// Both pool strides, named once and passed to the kernels - the harness
// allocates each pool at the stride the instantiation addresses.
#define SLOT_F32 (ELEMENTS * 4u)
#define SLOT_BF16 (ELEMENTS * 2u)

static uint16_t bf16(float value) { return LmFloatToBf16(value); }
static float f32(uint16_t value) { return LmBf16ToFloat(value); }

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return (float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

// The distance between an fp32 reference state element and the bf16 kernel's,
// in bf16 ulps at the STATE'S OPERATING MAGNITUDE, not at the element's own
// exponent. The per-commit rounding of each element is at most half a ulp of
// that element's magnitude - bounded by half a ulp of the scale - and the
// error map Diag(alpha)(I - beta k k^T) contracts in absolute terms, which is
// what the next step's fp32 math sees. Measuring each element against its own
// exponent denominates the error in a yardstick that vanishes near
// cancellation: an element the recurrence drives through zero shows
// thousands of "ulps" of a magnitude nothing reads.
static float Bf16UlpsAtScale(float scale, float reference, float rounded)
{
	int exponent;
	if ( reference == rounded || !(scale > 0.0f) )
		return 0.0f;
	// scale = m * 2^exponent with m in [0.5, 1): a bf16 ulp at the scale is
	// 2^(exponent - 1 - 7), seven stored mantissa bits.
	frexpf(scale,&exponent);
	return fabsf(reference - rounded) / ldexpf(1.0f,exponent - 8);
}

// One committed decode step of both instantiations over the same inputs:
// the fp32 pool against the fp32 kernel, the bf16 pool against the bf16
// kernel. query/key/value/forget/write_gate are shared verbatim.
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

	// -- 1a: a lossless round-trip is bit-identical over a multi-step run ---
	//
	// Retention 0.5 per channel and a zero key: the update is S = 0.5 * S, an
	// exact scaling in both fp32 and bf16, so a bf16-exact initial state
	// round-trips losslessly at every commit. The outputs still exercise the
	// whole per-step path - the q normalisation, the S^T q read of the
	// UPDATED state, the commit store - so any divergence in the load, the
	// math or the store shows up here as a flipped bit.
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

	// -- 1b: one step from the same fp32 state is bit-identical -------------
	//
	// Random bf16-exact states, random nonzero keys, random gates: the bf16
	// kernel's outputs must equal the fp32 kernel's bit for bit (the
	// recurrence math is identical given the same fp32 input state), and its
	// committed state must be exactly LmFloatToBf16 of the fp32 kernel's -
	// the commit rounding and nothing else.
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
				// retention in [0.5, 1): the contraction half of the envelope
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

	// -- 2: arbitrary fp32 state, divergence bounded over 64 commits --------
	//
	// The fp32 pool starts from arbitrary fp32 values; the bf16 pool starts
	// from their round-to-nearest-even - the one admission rounding the
	// option costs. Every commit then adds at most half a bf16 ulp per
	// element and the recurrence contracts the accumulated error by at most
	// alpha_max per step, so the divergence stays under
	// 0.5 / (1 - alpha_max) ulps at the state's operating magnitude:
	// bounded, never compounding. The harness knows the retention factors it
	// dealt, so the envelope is computed from them, and the divergence is
	// measured after EVERY step.
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
				// retention in [0.5, 0.95): near-1 is where the geometric
				// envelope is loosest, so the test lives there
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
		// the geometric series 0.5 * (1 + a + a^2 + ...) plus the admission
		// rounding, in bf16 ulps at the state's operating magnitude
		envelope = (0.5f / (1.0f - alpha_max)) + 1.0f;
		printf("ulp_steps %u\n",64u);
		printf("ulp_alpha_max %.6g\n",(double)alpha_max);
		printf("ulp_max %.6g\n",(double)worst);
		printf("ulp_envelope %.6g\n",(double)envelope);
		printf("ulp_first_half %.6g\n",(double)worst_early);
		printf("ulp_second_half %.6g\n",(double)worst_late);
	}

	// -- 3: the replay fold stays byte-exact under bf16 state ---------------
	//
	// Serial decode commits each of the eight steps into a bf16 pool. The
	// fold replays the same steps from the same initial state - once over
	// the whole prefix, and once over the tail from a mid-run checkpoint,
	// which is the DSpark shape: the committed checkpoint plus the accepted
	// prefix of a verify run. Both folds must land on the decode pool byte
	// for byte, because the fold upcasts and re-rounds at the same two
	// points once per step, exactly as a decode commit does.
	{
		static uint8_t decode_pool[SLOT_BF16],fold_pool[SLOT_BF16],prefix_pool[SLOT_BF16],checkpoint[SLOT_BF16];
		static uint16_t query[8u][HEADS * KEY_DIM],key[8u][HEADS * KEY_DIM],value[8u][HEADS * VALUE_DIM];
		static uint16_t out[HEADS * VALUE_DIM];
		static float retention[8u][HEADS * KEY_DIM],write_gate[HEADS];
		static LmReplayStep steps[8u];
		static uint32_t accepted[1];
		uint32_t state_index[1] = { 0u };
		uint32_t mismatch = 0u,prefix_mismatch = 0u;
		// fold_pool keeps the initial state while decode advances its copy;
		// the full fold replays from where decode STARTED.
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
		// the whole prefix from the initial state
		accepted[0] = 8u;
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmReplayFoldKernel<THREADS, KEY_DIM, VALUE_DIM, uint16_t>(
				fold_pool, SLOT_BF16, state_index, steps, accepted,
				HEADS, 1u, 1u)));
		// the accepted tail from the mid-run checkpoint - the DSpark shape
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
