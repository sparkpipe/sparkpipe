// Numerical gate for the interleaved expert GEMM (pack V2 mxfp4_ws_interleaved_v1).
//
// Builds a tiny interleaved weight tensor exactly as tools/k3_pack.py lays it
// out - per expert, per 128-element pack k-tile, per 16-neuron cell: 16 rows
// of 64 payload bytes (128 E2M1 nibble pairs) and one 64-byte scale row
// (16 neurons x 4 E8M0 group scales) - and runs BOTH interleaved launches the
// MoE layer uses (the w1 indirect path over the route map and the w2 direct
// path) against a CPU fp32 reference.
//
// The weight pattern is chosen so every addressing mistake changes the answer:
// all payload is 1.0, expert 0 scales are 2^0 in cell 0 and 2^-1 elsewhere,
// expert 1 scales are all 2^-1. Correct output per (packed row, neuron) is the
// row's BF16 sum times the neuron's scale; a wrong cell offset, wrong expert
// (z), wrong scale row, or a mis-staged sector moves whole columns.
//
// Payload 1.0 encodes to E2M1 code 2 (s0 e01 m0) in BOTH nibbles, so the test
// is immune to the pair's nibble order; the scales are exact powers of two.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/gemm.cuh"
#include "inference/kernels/formats/mxfp4.cuh"

// Host-side BF16 round-trips (round-to-nearest-even), matching the device
// conversion the GEMM pipeline performs; LmFloatToBf16 is __device__-only.
static uint16_t HostFloatToBf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	uint32_t lsb = (u >> 16u) & 1u;
	uint32_t rounded = u + 0x7fffu + lsb;
	return((uint16_t)(rounded >> 16u));
}
static float HostBf16ToFloat(uint16_t h)
{
	uint32_t u = ((uint32_t)h) << 16u;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

#define TEST_EXPERTS 2u
#define TEST_TOKENS 2u
#define TEST_TOP_K 2u
#define TEST_PACKED (TEST_TOKENS * TEST_TOP_K)
#define TEST_IN 256u     // two 128-element pack k-tiles
#define TEST_OUT 128u    // 8 cells, one TILE_N-wide neuron tile
#define TEST_CELLS (TEST_OUT / 16u)
#define TEST_K_TILES (TEST_IN / 128u)
#define TEST_ROWS_PER_EXPERT (TEST_K_TILES * TEST_CELLS * 17u)
#define TEST_EXPERT_BYTES ((size_t)TEST_ROWS_PER_EXPERT * 64u)
#define TEST_TILE_N 128u
#define TEST_STAGES 2u
#define TEST_WARPS 8u

// The three interleaved direct instantiations this TU launches.
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
// THE TILE_K 32 VARIANTS (TP16 grid): 16-byte cell rows, SWIZZLE_NONE maps.
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, TEST_TILE_N, 32u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, TEST_TILE_N, 128u, TEST_STAGES, TEST_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);

static void BuildInterleavedWeights(uint8_t *bytes)
{
	// Per expert: k-tile t, cell c, rows 0..15 payload, row 16 scales.
	for ( uint32_t e = 0u; e < TEST_EXPERTS; ++e )
	{
		for ( uint32_t t = 0u; t < TEST_K_TILES; ++t )
		{
			for ( uint32_t c = 0u; c < TEST_CELLS; ++c )
			{
				uint8_t *cell = bytes + (size_t)e * TEST_EXPERT_BYTES
					+ ((size_t)t * TEST_CELLS + c) * 17u * 64u;
				for ( uint32_t r = 0u; r < 16u; ++r )
					for ( uint32_t b = 0u; b < 64u; ++b )
						cell[r * 64u + b] = 0x22u; /* 1.0, 1.0 */
				for ( uint32_t n = 0u; n < 16u; ++n )
				{
					uint8_t code;
					if ( e == 1u )
						code = 126u;               /* 2^-1 */
					else if ( c == 0u )
						code = 127u;               /* 2^0  */
					else
						code = 126u;               /* 2^-1 */
					for ( uint32_t g = 0u; g < 4u; ++g )
						cell[16u * 64u + n * 4u + g] = code;
				}
			}
		}
	}
}

#define TEST_K_TILES_32 (TEST_IN / 32u)
#define TEST_ROWS_PER_EXPERT_32 (TEST_K_TILES_32 * TEST_CELLS * 17u)
#define TEST_EXPERT_BYTES_32 ((size_t)TEST_ROWS_PER_EXPERT_32 * 16u)

static void BuildInterleavedWeights32(uint8_t *bytes)
{
	// The same logical weights on the 32-element grid: per expert, per
	// 32-element pack k-tile, per 16-neuron cell, 16 rows of 16 payload
	// bytes (32 E2M1 nibble pairs) and one 16-byte scale row (one E8M0
	// byte per neuron - a 32-element tile is exactly one scale group).
	for ( uint32_t e = 0u; e < TEST_EXPERTS; ++e )
	{
		for ( uint32_t t = 0u; t < TEST_K_TILES_32; ++t )
		{
			for ( uint32_t c = 0u; c < TEST_CELLS; ++c )
			{
				uint8_t *cell = bytes + (size_t)e * TEST_EXPERT_BYTES_32
					+ ((size_t)t * TEST_CELLS + c) * 17u * 16u;
				for ( uint32_t r = 0u; r < 16u; ++r )
					for ( uint32_t b = 0u; b < 16u; ++b )
						cell[r * 16u + b] = 0x22u; /* 1.0, 1.0 */
				for ( uint32_t n = 0u; n < 16u; ++n )
				{
					uint8_t code;
					if ( e == 1u )
						code = 126u;
					else if ( c == 0u )
						code = 127u;
					else
						code = 126u;
					cell[16u * 16u + n] = code;
				}
			}
		}
	}
}

static float ExpectedValue(const float *a_row, uint32_t neuron)
{
	// a_row is the fp32 source row; the GEMM input is its BF16 rounding, so
	// the reference sums the rounded values - the tolerance is the fp32
	// accumulation-order slack, not the rounding.
	float sum = 0.0f;
	for ( uint32_t k = 0u; k < TEST_IN; ++k )
		sum += (double)HostBf16ToFloat(HostFloatToBf16(a_row[k]));
	sum *= (neuron < 16u) ? 1.0f : 0.5f;
	return sum;
}

int main(void)
{
	cudaError_t err;
	int32_t status;
	int device = 0, multiprocessors = 0;
	uint16_t *d_activation = 0, *d_packed = 0, *d_out = 0;
	uint8_t *d_weight = 0;
	uint32_t *d_group_offset = 0, *d_group_prefix = 0, *d_source_map = 0;
	float *h_activation = 0;
	uint16_t *h_out = 0;
	uint32_t h_group_offset[TEST_EXPERTS + 1u] = { 0u, 2u, 4u };
	uint32_t h_group_prefix[TEST_EXPERTS + 1u] = { 0u, 1u, 2u };
	uint32_t h_source_map[TEST_PACKED] = { 0u, 1u, 0u, 1u };
	uint8_t *h_weight = 0;
	uint32_t p, k, n, failures = 0u;

	err = cudaSetDevice(device);
	if ( err != cudaSuccess ) { printf("FAIL setdevice %d\n", (int)err); return 1; }
	cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);

	h_activation = (float *)malloc(TEST_TOKENS * TEST_IN * sizeof(float));
	h_out = (uint16_t *)malloc(TEST_PACKED * TEST_OUT * sizeof(uint16_t));
	h_weight = (uint8_t *)malloc(TEST_EXPERT_BYTES * TEST_EXPERTS);
	for ( k = 0u; k < TEST_TOKENS * TEST_IN; ++k )
		h_activation[k] = (float)((int32_t)((k * 37u) % 11u) - 5) * 0.25f;
	BuildInterleavedWeights(h_weight);

	cudaMalloc(&d_activation, TEST_TOKENS * TEST_IN * sizeof(uint16_t));
	cudaMalloc(&d_packed, TEST_PACKED * TEST_IN * sizeof(uint16_t));
	cudaMalloc(&d_out, TEST_PACKED * TEST_OUT * sizeof(uint16_t));
	cudaMalloc(&d_weight, TEST_EXPERT_BYTES * TEST_EXPERTS);
	cudaMalloc(&d_group_offset, sizeof(h_group_offset));
	cudaMalloc(&d_group_prefix, sizeof(h_group_prefix));
	cudaMalloc(&d_source_map, sizeof(h_source_map));
	{
		uint16_t *tmp = (uint16_t *)malloc(TEST_TOKENS * TEST_IN * sizeof(uint16_t));
		for ( k = 0u; k < TEST_TOKENS * TEST_IN; ++k )
			tmp[k] = HostFloatToBf16(h_activation[k]);
		cudaMemcpy(d_activation, tmp, TEST_TOKENS * TEST_IN * sizeof(uint16_t), cudaMemcpyHostToDevice);
		free(tmp);
	}
	cudaMemcpy(d_weight, h_weight, TEST_EXPERT_BYTES * TEST_EXPERTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_group_offset, h_group_offset, sizeof(h_group_offset), cudaMemcpyHostToDevice);
	cudaMemcpy(d_group_prefix, h_group_prefix, sizeof(h_group_prefix), cudaMemcpyHostToDevice);
	cudaMemcpy(d_source_map, h_source_map, sizeof(h_source_map), cudaMemcpyHostToDevice);
	// The packed activation for the direct (w2-style) test is the un-gathered
	// tensor expanded expert-major: row p duplicates token source_row_map[p].
	{
		uint16_t *tmp = (uint16_t *)malloc(TEST_PACKED * TEST_IN * sizeof(uint16_t));
		for ( p = 0u; p < TEST_PACKED; ++p )
			for ( k = 0u; k < TEST_IN; ++k )
				tmp[p * TEST_IN + k] = HostFloatToBf16(h_activation[h_source_map[p] * TEST_IN + k]);
		cudaMemcpy(d_packed, tmp, TEST_PACKED * TEST_IN * sizeof(uint16_t), cudaMemcpyHostToDevice);
		free(tmp);
	}

	// --- map encode diagnostics ---
	{
		alignas(64) CUtensorMap act_map, wgt_map;
		int32_t s1 = LmGemmEncodeActivationMap(&act_map, d_packed, TEST_PACKED,
			TEST_IN, 16u, 64u, LmBf16Format::kStoredBits);
		int32_t s2 = LmGemmEncodeWeightMapInterleaved(&wgt_map, d_weight,
			TEST_IN, TEST_OUT, TEST_EXPERTS, TEST_TILE_N);
		printf("encode act=%d interleaved-weight=%d\n", s1, s2);
	}

	// --- w2-style: direct interleaved over the packed activation ---
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = LmScaleTensorNone();
		gemm.group_row_offset = d_group_offset;
		gemm.group_tile_prefix = d_group_prefix;
		gemm.prefix_built = 1u;
		gemm.output_bf16 = d_out;
		status = LmGemmWeightOnlyInterleavedLaunch<
			LmMxfp4, TEST_TILE_N, TEST_STAGES, TEST_WARPS>(
			&gemm, d_packed, d_weight, TEST_PACKED, TEST_TOKENS, TEST_TOP_K,
			TEST_EXPERTS, TEST_IN, TEST_OUT, (uint32_t)multiprocessors,
			true, (cudaStream_t)0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL direct launch %d\n", status); return 1; }
		err = cudaDeviceSynchronize();
		if ( err != cudaSuccess ) { printf("FAIL direct sync %d\n", (int)err); return 1; }
		cudaMemcpy(h_out, d_out, TEST_PACKED * TEST_OUT * sizeof(uint16_t), cudaMemcpyDeviceToHost);

		for ( p = 0u; p < TEST_PACKED; ++p )
		{
			uint32_t expert = p / TEST_TOP_K;
			uint32_t token = h_source_map[p];
			float base = ExpectedValue(&h_activation[token * TEST_IN], 0u);
			for ( n = 0u; n < TEST_OUT; ++n )
			{
				float scale = (expert == 1u) ? 0.5f : (n < 16u ? 1.0f : 0.5f);
				float expect = base * scale;
				float got = (double)HostBf16ToFloat(h_out[p * TEST_OUT + n]);
				if ( fabsf(got - expect) > 0.03f * fabsf(expect) + 1e-3f )
				{
					if ( failures < 8u )
						printf("DIRECT mismatch p=%u n=%u got=%g expect=%g\n", p, n, got, expect);
					++failures;
				}
			}
		}
	}

	// --- w1-style: indirect interleaved over the route map ---
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = LmScaleTensorNone();
		gemm.group_row_offset = d_group_offset;
		gemm.group_tile_prefix = d_group_prefix;
		gemm.prefix_built = 1u;
		gemm.output_bf16 = d_out;
		gemm.source_row_map = d_source_map;
		gemm.source_row_count = TEST_TOKENS;
		status = LmGemmWeightOnlyIndirectInterleavedLaunch<
			LmMxfp4, TEST_TILE_N, TEST_STAGES, TEST_WARPS>(
			&gemm, d_activation, d_weight, TEST_PACKED, TEST_TOKENS, TEST_TOP_K,
			TEST_EXPERTS, TEST_IN, TEST_OUT, (uint32_t)multiprocessors,
			(cudaStream_t)0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL indirect launch %d\n", status); return 1; }
		err = cudaDeviceSynchronize();
		if ( err != cudaSuccess ) { printf("FAIL indirect sync %d\n", (int)err); return 1; }
		cudaMemcpy(h_out, d_out, TEST_PACKED * TEST_OUT * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		for ( p = 0u; p < TEST_PACKED; ++p )
		{
			uint32_t expert = p / TEST_TOP_K;
			uint32_t token = h_source_map[p];
			float base = ExpectedValue(&h_activation[token * TEST_IN], 0u);
			for ( n = 0u; n < TEST_OUT; ++n )
			{
				float scale = (expert == 1u) ? 0.5f : (n < 16u ? 1.0f : 0.5f);
				float expect = base * scale;
				float got = (double)HostBf16ToFloat(h_out[p * TEST_OUT + n]);
				if ( fabsf(got - expect) > 0.03f * fabsf(expect) + 1e-3f )
				{
					if ( failures < 8u )
						printf("INDIRECT mismatch p=%u n=%u got=%g expect=%g\n", p, n, got, expect);
					++failures;
				}
			}
		}
	}

	// --- the TILE_K 32 grid: same weights, same expected values ---
	{
		uint8_t *h_weight32 = (uint8_t *)malloc(TEST_EXPERT_BYTES_32 * TEST_EXPERTS);
		uint8_t *d_weight32 = 0;
		BuildInterleavedWeights32(h_weight32);
		cudaMalloc(&d_weight32, TEST_EXPERT_BYTES_32 * TEST_EXPERTS);
		cudaMemcpy(d_weight32, h_weight32, TEST_EXPERT_BYTES_32 * TEST_EXPERTS,
			cudaMemcpyHostToDevice);
		{
			alignas(64) CUtensorMap wgt_map32;
			int32_t s3 = LmGemmEncodeWeightMapInterleaved(&wgt_map32, d_weight32,
				TEST_IN, TEST_OUT, TEST_EXPERTS, TEST_TILE_N, 32u);
			printf("encode interleaved-weight-32=%d\n", s3);
		}
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = LmScaleTensorNone();
		gemm.group_row_offset = d_group_offset;
		gemm.group_tile_prefix = d_group_prefix;
		gemm.prefix_built = 1u;
		gemm.output_bf16 = d_out;
		status = LmGemmWeightOnlyInterleavedLaunch<
			LmMxfp4, TEST_TILE_N, TEST_STAGES, TEST_WARPS, 32u>(
			&gemm, d_packed, d_weight32, TEST_PACKED, TEST_TOKENS, TEST_TOP_K,
			TEST_EXPERTS, TEST_IN, TEST_OUT, (uint32_t)multiprocessors,
			true, (cudaStream_t)0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL direct-32 launch %d\n", status); return 1; }
		err = cudaDeviceSynchronize();
		if ( err != cudaSuccess ) { printf("FAIL direct-32 sync %d\n", (int)err); return 1; }
		cudaMemcpy(h_out, d_out, TEST_PACKED * TEST_OUT * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		for ( p = 0u; p < TEST_PACKED; ++p )
		{
			uint32_t expert = p / TEST_TOP_K;
			uint32_t token = h_source_map[p];
			float base = ExpectedValue(&h_activation[token * TEST_IN], 0u);
			for ( n = 0u; n < TEST_OUT; ++n )
			{
				float scale = (expert == 1u) ? 0.5f : (n < 16u ? 1.0f : 0.5f);
				float expect = base * scale;
				float got = (double)HostBf16ToFloat(h_out[p * TEST_OUT + n]);
				if ( fabsf(got - expect) > 0.03f * fabsf(expect) + 1e-3f )
				{
					if ( failures < 8u )
						printf("DIRECT32 mismatch p=%u n=%u got=%g expect=%g\n", p, n, got, expect);
					++failures;
				}
			}
		}
		cudaFree(d_weight32);
		free(h_weight32);
	}

	if ( failures == 0u )
	{
		printf("k3 interleave gemm gate PASS (direct + indirect + tile_k 32)\n");
		return 0;
	}
	printf("k3 interleave gemm gate FAIL: %u mismatches\n", failures);
	return 1;
}
