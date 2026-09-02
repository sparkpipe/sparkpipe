
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/norm.cuh"
#include "inference/kernels/linear_attn.cuh"

#define THREADS 1u
#define ROWS 3u
#define HEADS 2u
#define HEAD_DIM 4u
#define CHANNELS (HEADS * HEAD_DIM)
#define CONV_KERNEL 4u
#define INTERMEDIATE 5u
#define TOP_K 2u
#define SOURCES 4u
#define EXPERT_ROWS (ROWS * TOP_K)

static uint32_t seed = 24680u;
static float NextRandom(void)
{
	seed = (seed * 1664525u) + 1013904223u;
	return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static void Emit(const char *tag, const uint16_t *values, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; ++index)
		printf("%s %.9g\n", tag, (double)LmBf16ToFloat(values[index]));
}

static void EmitF32(const char *tag, const float *values, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; ++index)
		printf("%s %.9g\n", tag, (double)values[index]);
}

int main(void)
{
	static uint16_t window[CHANNELS * CONV_KERNEL];
	static uint16_t conv_in[ROWS * CHANNELS];
	static float conv_w[CHANNELS * CONV_KERNEL];
	static uint16_t conv_out[ROWS * CHANNELS];
	static uint16_t l2_in[ROWS * CHANNELS];
	static uint16_t gate_up[ROWS * INTERMEDIATE * 2u], situ_out[ROWS * INTERMEDIATE];
	static uint16_t norm_in[ROWS * CHANNELS], norm_res[ROWS * CHANNELS];
	static float norm_w[CHANNELS];
	static uint16_t norm_res_out[ROWS * CHANNELS];
	static uint16_t norm_out[ROWS * CHANNELS];
	static uint16_t gate_val[ROWS * CHANNELS], gated[ROWS * CHANNELS];
	static uint16_t expert_out[EXPERT_ROWS * CHANNELS], finalized[ROWS * CHANNELS];
	static uint16_t bank[ROWS * (SOURCES - 1u) * CHANNELS];
	static uint16_t partial[ROWS * CHANNELS], attnres_w[CHANNELS];
	static uint16_t attnres_out[ROWS * CHANNELS];
	static uint32_t state_index[1] = { 0u };
	static uint32_t packed_row[ROWS * TOP_K];
	static float route_weight[ROWS * TOP_K];
	uint32_t index, row, slot;

	printf("ROWS %u HEADS %u HEAD_DIM %u CHANNELS %u CONV %u INTER %u TOPK %u\n",
		ROWS, HEADS, HEAD_DIM, CHANNELS, CONV_KERNEL, INTERMEDIATE, TOP_K);

	for (index = 0u; index < CHANNELS * CONV_KERNEL; ++index)
	{
		window[index] = LmFloatToBf16(NextRandom());
		conv_w[index] = NextRandom() * 0.5f;
	}
	for (index = 0u; index < ROWS * CHANNELS; ++index)
	{
		conv_in[index] = LmFloatToBf16(NextRandom());
		l2_in[index] = LmFloatToBf16(NextRandom());
		norm_in[index] = LmFloatToBf16(NextRandom());
		norm_res[index] = LmFloatToBf16(NextRandom());
		gate_val[index] = LmFloatToBf16(NextRandom());
		gated[index] = LmFloatToBf16(NextRandom());
	}
	for (index = 0u; index < CHANNELS; ++index)
		norm_w[index] = 1.0f + 0.2f * NextRandom();
	for (index = 0u; index < ROWS * INTERMEDIATE * 2u; ++index)
		gate_up[index] = LmFloatToBf16(NextRandom() * 4.0f);
	for (index = 0u; index < EXPERT_ROWS * CHANNELS; ++index)
		expert_out[index] = LmFloatToBf16(NextRandom());
	for (row = 0u; row < ROWS; ++row)
		for (slot = 0u; slot < TOP_K; ++slot)
		{
			packed_row[(row * TOP_K) + slot] = (row * TOP_K) + slot;
			route_weight[(row * TOP_K) + slot] = 0.3f + 0.2f * NextRandom();
		}

	Emit("window", window, CHANNELS * CONV_KERNEL);
	EmitF32("convw", conv_w, CHANNELS * CONV_KERNEL);
	Emit("convin", conv_in, ROWS * CHANNELS);
	Emit("l2in", l2_in, ROWS * CHANNELS);
	Emit("gateup", gate_up, ROWS * INTERMEDIATE * 2u);
	Emit("normin", norm_in, ROWS * CHANNELS);
	Emit("normres", norm_res, ROWS * CHANNELS);
	EmitF32("normw", norm_w, CHANNELS);
	Emit("gateval", gate_val, ROWS * CHANNELS);
	Emit("gated", gated, ROWS * CHANNELS);
	for (index = 0u; index < ROWS * (SOURCES - 1u) * CHANNELS; ++index)
		bank[index] = LmFloatToBf16(NextRandom() * 2.0f);
	for (index = 0u; index < ROWS * CHANNELS; ++index)
		partial[index] = LmFloatToBf16(NextRandom());
	for (index = 0u; index < CHANNELS; ++index)
		attnres_w[index] = LmFloatToBf16(NextRandom());
	Emit("bank", bank, ROWS * (SOURCES - 1u) * CHANNELS);
	Emit("partial", partial, ROWS * CHANNELS);
	Emit("attnresw", attnres_w, CHANNELS);
	Emit("expert", expert_out, EXPERT_ROWS * CHANNELS);
	for (index = 0u; index < ROWS * TOP_K; ++index)
		printf("rweight %.9g\n", (double)route_weight[index]);

	static uint32_t conv_run_begin[2] = { 0u, ROWS };
	LM_HOST_LAUNCH(dim3(1u, CHANNELS),
		(LmCausalConvKernel<THREADS, CONV_KERNEL, LM_CONV_SWISH,float>(
			window, state_index, conv_run_begin, 0, conv_in, conv_w, conv_out, CHANNELS, 1u, 1u)));
	Emit("convout", conv_out, ROWS * CHANNELS);

	LM_HOST_LAUNCH(dim3(ROWS, HEADS),
		(LmL2NormalisePerHeadKernel<THREADS, HEAD_DIM>(
			l2_in, HEADS, ROWS, 1e-6f)));
	Emit("l2out", l2_in, ROWS * CHANNELS);

	LM_HOST_LAUNCH(dim3(ROWS),
		(LmSituMulKernel<THREADS>(gate_up, situ_out, INTERMEDIATE, 4.0f, 25.0f)));
	Emit("situout", situ_out, ROWS * INTERMEDIATE);

	LM_HOST_LAUNCH(dim3(ROWS),
		(LmFusedResidualRmsNormKernel<THREADS,float>(
			norm_in, norm_res, norm_w, norm_res_out, norm_out, CHANNELS,CHANNELS, 1e-5f)));
	Emit("normout", norm_out, ROWS * CHANNELS);
	Emit("normresout", norm_res_out, ROWS * CHANNELS);

	LM_HOST_LAUNCH(dim3(ROWS),
		(LmOutputGateKernel<THREADS>(gated, gate_val, CHANNELS)));
	Emit("gateout", gated, ROWS * CHANNELS);

	LM_HOST_LAUNCH(dim3(CHANNELS, ROWS),
		(LmMoeFinalizeKernel<THREADS>(
			expert_out, packed_row, route_weight, finalized,
			ROWS, TOP_K, CHANNELS)));
	Emit("finalout", finalized, ROWS * CHANNELS);

	LM_HOST_LAUNCH(dim3(ROWS),
		(LmAttnResKernel<THREADS, 16u>(
			bank, partial, attnres_w, attnres_out, SOURCES, ROWS, CHANNELS, 1e-5f)));
	Emit("attnresout", attnres_out, ROWS * CHANNELS);
	return 0;
}
