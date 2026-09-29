#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#define LM_KV_QUANT_SIM_LANES LM_HOST_WARP_LANES

#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/kv_quant_sim.cuh"

#define HEADS 2u
#define KEY_DIM 4u
#define VALUE_DIM 4u
#define THREADS 1u
#define STEPS 6u
#define ELEMENTS (HEADS * KEY_DIM * VALUE_DIM)
#define SLOT_F32 (ELEMENTS * 4u)
#define SLOT_BF16 (ELEMENTS * 2u)
#define MAX_ROW_ELEMENTS (1u << 20u)

static uint16_t rows_bf16[MAX_ROW_ELEMENTS];

static int RunRows(void)
{
	uint32_t header[5];
	uint64_t elements;
	int32_t status;
	if ( fread(header,sizeof(uint32_t),5,stdin) != 5 )
		return(1);
	elements = (uint64_t)header[0] * header[2];
	if ( elements > MAX_ROW_ELEMENTS )
		return(2);
	if ( fread(rows_bf16,sizeof(uint16_t),elements,stdin) != elements )
		return(3);
	status = LmKvQuantSimRowsLaunch(header[0] != 0u ? rows_bf16 : 0,header[2],header[0],header[1],header[3],header[4],0);
	if ( fwrite(&status,sizeof(status),1,stdout) != 1 )
		return(4);
	return(fwrite(rows_bf16,sizeof(uint16_t),elements,stdout) == elements ? 0 : 5);
}

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return((float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f);
}

template<class State>
static void Step(uint8_t *pool, uint32_t slot_bytes, const uint32_t *begin, uint32_t rows, const uint16_t *query, const uint16_t *key, const uint16_t *value, const float *retention, const float *write_gate, uint16_t *out)
{
	uint32_t state_index[1] = { 0u };
	uint32_t count[1] = { rows };
	LM_HOST_LAUNCH(dim3(1u, HEADS),
		(LmDeltaRuleKernel<THREADS, KEY_DIM, VALUE_DIM, State>(
			pool, slot_bytes, state_index, begin, count, query, key, value,
			retention, write_gate, out, HEADS, 1u, 1u, 1u)));
}

static int RunState(void)
{
	static uint16_t query[STEPS * HEADS * KEY_DIM],key[STEPS * HEADS * KEY_DIM],value[STEPS * HEADS * VALUE_DIM];
	static float retention[STEPS * HEADS * KEY_DIM],write_gate[STEPS * HEADS];
	static uint16_t out_wave[STEPS * HEADS * VALUE_DIM],out_steps[STEPS * HEADS * VALUE_DIM],out_float[STEPS * HEADS * VALUE_DIM];
	static uint16_t out_bf16_wave[STEPS * HEADS * VALUE_DIM],out_bf16_steps[STEPS * HEADS * VALUE_DIM];
	static float pool_wave[ELEMENTS],pool_steps[ELEMENTS],pool_float[ELEMENTS],pool_reference[ELEMENTS];
	static uint16_t pool_bf16_wave[ELEMENTS],pool_bf16_steps[ELEMENTS];
	uint32_t seed = 2929u,index,step,begin[2] = { 0u, STEPS },single[2] = { 0u, 1u };
	uint32_t wave_mismatch = 0u,reference_mismatch = 0u,off_grid = 0u,float_differs = 0u,bf16_wave_mismatch = 0u;
	for (index = 0u; index < STEPS * HEADS * KEY_DIM; ++index)
	{
		query[index] = LmFloatToBf16(NextRandom(&seed));
		key[index] = LmFloatToBf16(NextRandom(&seed));
		retention[index] = 0.75f + 0.2f * NextRandom(&seed);
	}
	for (index = 0u; index < STEPS * HEADS * VALUE_DIM; ++index)
		value[index] = LmFloatToBf16(3.0f * NextRandom(&seed));
	for (index = 0u; index < STEPS * HEADS; ++index)
		write_gate[index] = 0.5f + 0.4f * NextRandom(&seed);
	for (index = 0u; index < ELEMENTS; ++index)
	{
		pool_wave[index] = LmBf16ToFloat(LmFloatToBf16(NextRandom(&seed)));
		pool_steps[index] = pool_float[index] = pool_reference[index] = pool_wave[index];
		pool_bf16_wave[index] = pool_bf16_steps[index] = LmFloatToBf16(pool_wave[index]);
	}
	Step<LmKvStateBf16Grid>((uint8_t *)pool_wave,SLOT_F32,begin,STEPS,query,key,value,retention,write_gate,out_wave);
	Step<uint16_t>((uint8_t *)pool_bf16_wave,SLOT_BF16,begin,STEPS,query,key,value,retention,write_gate,out_bf16_wave);
	for (step = 0u; step < STEPS; ++step)
	{
		uint64_t k = (uint64_t)step * HEADS * KEY_DIM,v = (uint64_t)step * HEADS * VALUE_DIM;
		static uint16_t scratch[HEADS * VALUE_DIM];
		Step<LmKvStateBf16Grid>((uint8_t *)pool_steps,SLOT_F32,single,1u,query + k,key + k,value + v,retention + k,write_gate + step * HEADS,out_steps + v);
		Step<float>((uint8_t *)pool_float,SLOT_F32,single,1u,query + k,key + k,value + v,retention + k,write_gate + step * HEADS,out_float + v);
		Step<uint16_t>((uint8_t *)pool_bf16_steps,SLOT_BF16,single,1u,query + k,key + k,value + v,retention + k,write_gate + step * HEADS,out_bf16_steps + v);
		Step<float>((uint8_t *)pool_reference,SLOT_F32,single,1u,query + k,key + k,value + v,retention + k,write_gate + step * HEADS,scratch);
		for (index = 0u; index < ELEMENTS; ++index)
		{
			pool_reference[index] = LmBf16ToFloat(LmFloatToBf16(pool_reference[index]));
			if ( memcmp(&pool_reference[index],&pool_steps[index],sizeof(float)) != 0 )
				++reference_mismatch;
		}
	}
	for (index = 0u; index < STEPS * HEADS * VALUE_DIM; ++index)
	{
		if ( out_wave[index] != out_steps[index] )
			++wave_mismatch;
		if ( out_wave[index] != out_float[index] )
			++float_differs;
		if ( out_bf16_wave[index] != out_bf16_steps[index] )
			++bf16_wave_mismatch;
	}
	for (index = 0u; index < ELEMENTS; ++index)
	{
		uint32_t bits;
		memcpy(&bits,&pool_wave[index],sizeof(bits));
		if ( (bits & 0xffffu) != 0u )
			++off_grid;
		if ( memcmp(&pool_wave[index],&pool_steps[index],sizeof(float)) != 0 )
			++wave_mismatch;
		if ( memcmp(&pool_wave[index],&pool_float[index],sizeof(float)) != 0 )
			++float_differs;
		if ( pool_bf16_wave[index] != pool_bf16_steps[index] )
			++bf16_wave_mismatch;
	}
	printf("grid_wave_vs_steps_mismatch %u\n",wave_mismatch);
	printf("grid_vs_rounded_reference_mismatch %u\n",reference_mismatch);
	printf("grid_state_off_bf16_grid %u\n",off_grid);
	printf("grid_vs_fp32_differs %u\n",float_differs);
	printf("writeback_bf16_wave_vs_steps_mismatch %u\n",bf16_wave_mismatch);
	return(0);
}

int main(int argc, char **argv)
{
	if ( argc == 2 && strcmp(argv[1],"rows") == 0 )
		return(RunRows());
	if ( argc == 2 && strcmp(argv[1],"state") == 0 )
		return(RunState());
	fprintf(stderr,"usage: %s rows|state\n",argv[0]);
	return(9);
}
