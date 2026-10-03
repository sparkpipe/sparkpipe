#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "runtime/gemm.cuh"
#include "inference/kernels/formats/mxfp4.cuh"

#define TP16_EXPERTS 2u
#define TP16_TOKENS 2u
#define TP16_TOP_K 2u
#define TP16_PACKED (TP16_TOKENS * TP16_TOP_K)
#define TP16_TILE_K 32u
#define TP16_TILE_N 128u
#define TP16_STAGES 2u
#define TP16_WARPS 8u
#define TP16_CELL_BYTES (17u * (TP16_TILE_K / 2u))
#define TP16_MAP_ALL 1u
#define TP16_MAP_ROUTED_AND_BASE 2u

template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, TP16_TILE_N, TP16_TILE_K, TP16_STAGES, TP16_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, TP16_TILE_N, TP16_TILE_K, TP16_STAGES, TP16_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, TP16_TILE_N, TP16_TILE_K, TP16_STAGES, TP16_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);

static uint16_t HostFloatToBf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	return((uint16_t)((u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u));
}

static float HostBf16ToFloat(uint16_t h)
{
	uint32_t u = ((uint32_t)h) << 16u;
	float f;
	memcpy(&f, &u, sizeof(f));
	return(f);
}

static float CellScale(uint32_t expert, uint32_t cell)
{
	return(expert == 1u ? 0.5f : (cell == 0u ? 1.0f : 0.25f));
}

static void BuildWeights(uint8_t *bytes, uint32_t input, uint32_t output)
{
	const uint32_t tiles = input / TP16_TILE_K;
	const uint32_t cells = output / 16u;
	const size_t expert_bytes = (size_t)tiles * cells * TP16_CELL_BYTES;
	for ( uint32_t e = 0u; e < TP16_EXPERTS; ++e )
		for ( uint32_t t = 0u; t < tiles; ++t )
			for ( uint32_t c = 0u; c < cells; ++c )
			{
				uint8_t *cell = bytes + (size_t)e * expert_bytes + ((size_t)t * cells + c) * TP16_CELL_BYTES;
				float scale = CellScale(e, c);
				uint8_t code = scale == 1.0f ? 127u : (scale == 0.5f ? 126u : 125u);
				memset(cell, 0x22, 16u * (TP16_TILE_K / 2u));
				memset(cell + 16u * (TP16_TILE_K / 2u), code, TP16_TILE_K / 2u);
			}
}

static uint8_t *MapAtEnd(size_t bytes, CUdeviceptr *reservation, CUmemGenericAllocationHandle *handle, size_t *mapped, size_t *reserved)
{
	CUmemAllocationProp prop;
	size_t granularity = 0u;
	memset(&prop, 0, sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	if ( cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS )
		return(0);
	*mapped = ((bytes + granularity - 1u) / granularity) * granularity;
	*reserved = *mapped + 2u * granularity;
	if ( cuMemAddressReserve(reservation, *reserved, 0u, 0u, 0u) != CUDA_SUCCESS ||
		cuMemCreate(handle, *mapped, &prop, 0u) != CUDA_SUCCESS ||
		cuMemMap(*reservation, *mapped, 0u, *handle, 0u) != CUDA_SUCCESS )
		return(0);
	CUmemAccessDesc access;
	memset(&access, 0, sizeof(access));
	access.location = prop.location;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if ( cuMemSetAccess(*reservation, *mapped, &access, 1u) != CUDA_SUCCESS )
		return(0);
	return((uint8_t *)*reservation + (*mapped - bytes));
}

static uint32_t RunShape(uint32_t input, uint32_t output, uint32_t multiprocessors)
{
	const uint32_t tiles_per_expert = (output + TP16_TILE_N - 1u) / TP16_TILE_N;
	const size_t expert_bytes = (size_t)(input / TP16_TILE_K) * (output / 16u) * TP16_CELL_BYTES;
	uint32_t group_offset[TP16_EXPERTS + 1u] = { 0u, 2u, 4u };
	uint32_t group_prefix[TP16_EXPERTS + 1u] = { 0u, tiles_per_expert, 2u * tiles_per_expert };
	uint32_t source[TP16_PACKED] = { 0u, 1u, 0u, 1u };
	float *activation = (float *)malloc((size_t)TP16_TOKENS * input * sizeof(float));
	uint16_t *packed = (uint16_t *)malloc((size_t)TP16_PACKED * input * sizeof(uint16_t));
	uint16_t *out = (uint16_t *)malloc((size_t)TP16_PACKED * output * sizeof(uint16_t));
	uint8_t *weights = (uint8_t *)malloc(expert_bytes * TP16_EXPERTS);
	uint16_t *d_packed = 0, *d_out = 0;
	uint8_t *d_weights = 0;
	uint32_t *d_offset = 0, *d_prefix = 0;
	uint32_t failures = 0u;
	LmGemmArguments gemm;
	int32_t status;
	cudaError_t err;
	for ( uint32_t k = 0u; k < TP16_TOKENS * input; ++k )
		activation[k] = (float)((int32_t)((k * 37u) % 11u) - 5) * 0.25f;
	for ( uint32_t p = 0u; p < TP16_PACKED; ++p )
		for ( uint32_t k = 0u; k < input; ++k )
			packed[(size_t)p * input + k] = HostFloatToBf16(activation[(size_t)source[p] * input + k]);
	BuildWeights(weights, input, output);
	cudaMalloc(&d_packed, (size_t)TP16_PACKED * input * sizeof(uint16_t));
	cudaMalloc(&d_out, (size_t)TP16_PACKED * output * sizeof(uint16_t));
	CUdeviceptr reservation = 0u;
	CUmemGenericAllocationHandle handle = 0u;
	size_t mapped = 0u, reserved = 0u;
	d_weights = MapAtEnd(expert_bytes * TP16_EXPERTS, &reservation, &handle, &mapped, &reserved);
	if ( d_weights == 0 )
	{
		printf("FAIL weights could not be mapped at the end of a reservation\n");
		return(1u);
	}
	cudaMalloc(&d_offset, sizeof(group_offset));
	cudaMalloc(&d_prefix, sizeof(group_prefix));
	cudaMemcpy(d_packed, packed, (size_t)TP16_PACKED * input * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, expert_bytes * TP16_EXPERTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, group_offset, sizeof(group_offset), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prefix, group_prefix, sizeof(group_prefix), cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, (size_t)TP16_PACKED * output * sizeof(uint16_t));
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = d_offset;
	gemm.group_tile_prefix = d_prefix;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = d_out;
	status = LmGemmWeightOnlyInterleavedLaunch<LmMxfp4, TP16_TILE_N, TP16_STAGES, TP16_WARPS, TP16_TILE_K>(
		&gemm, d_packed, d_weights, TP16_PACKED, TP16_TOKENS, TP16_TOP_K, TP16_EXPERTS,
		input, output, multiprocessors, true, (cudaStream_t)0);
	err = cudaDeviceSynchronize();
	printf("shape in=%u out=%u tile_k=%u: launch=%d sync=%s\n", input, output, TP16_TILE_K,
		(int)status, cudaGetErrorString(err));
	if ( status != LM_LAUNCH_OK || err != cudaSuccess )
		return(1u);
	cudaMemcpy(out, d_out, (size_t)TP16_PACKED * output * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	for ( uint32_t p = 0u; p < TP16_PACKED; ++p )
	{
		float base = 0.0f;
		for ( uint32_t k = 0u; k < input; ++k )
			base += HostBf16ToFloat(packed[(size_t)p * input + k]);
		for ( uint32_t n = 0u; n < output; ++n )
		{
			float expect = base * CellScale(p / TP16_TOP_K, n / 16u);
			float got = HostBf16ToFloat(out[(size_t)p * output + n]);
			if ( fabsf(got - expect) > 0.03f * fabsf(expect) + 1e-2f )
			{
				if ( failures < 6u )
					printf("mismatch in=%u out=%u p=%u n=%u got=%g expect=%g\n", input, output, p, n, got, expect);
				failures++;
			}
		}
	}
	cuMemUnmap(reservation, mapped);
	cuMemRelease(handle);
	cuMemAddressFree(reservation, reserved);
	cudaFree(d_packed); cudaFree(d_out); cudaFree(d_offset); cudaFree(d_prefix);
	free(activation); free(packed); free(out); free(weights);
	return(failures);
}

static uint32_t RunLeased(uint32_t input, uint32_t output, uint32_t experts, uint32_t multiprocessors, uint32_t mapping)
{
	static const uint32_t routed[16] = { 79u, 156u, 210u, 220u, 353u, 468u, 485u, 553u, 564u, 592u, 656u, 679u, 731u, 767u, 788u, 801u };
	const uint32_t routes = 16u;
	const uint32_t tiles_per_expert = (output + TP16_TILE_N - 1u) / TP16_TILE_N;
	const size_t expert_bytes = (size_t)(input / TP16_TILE_K) * (output / 16u) * TP16_CELL_BYTES;
	const size_t total = expert_bytes * experts;
	CUmemAllocationProp prop;
	CUmemAccessDesc access;
	size_t granularity = 0u, reserved;
	CUdeviceptr base = 0u;
	uint32_t *offset = (uint32_t *)calloc(experts + 1u, sizeof(uint32_t));
	uint32_t *prefix = (uint32_t *)calloc(experts + 1u, sizeof(uint32_t));
	uint8_t *mapped_chunk;
	uint8_t *one = (uint8_t *)malloc(expert_bytes * TP16_EXPERTS);
	uint16_t *packed = (uint16_t *)malloc((size_t)routes * input * sizeof(uint16_t));
	uint16_t *d_packed = 0, *d_out = 0;
	uint32_t *d_offset = 0, *d_prefix = 0;
	LmGemmArguments gemm;
	int32_t status;
	cudaError_t err;
	memset(&prop, 0, sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	reserved = ((total + granularity - 1u) / granularity + 1u) * granularity;
	if ( cuMemAddressReserve(&base, reserved, granularity, 0u, 0u) != CUDA_SUCCESS )
		return(1u);
	mapped_chunk = (uint8_t *)calloc(reserved / granularity, 1u);
	memset(&access, 0, sizeof(access));
	access.location = prop.location;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	BuildWeights(one, input, output);
	if ( mapping == TP16_MAP_ROUTED_AND_BASE )
	{
		CUmemGenericAllocationHandle handle;
		if ( cuMemCreate(&handle, granularity, &prop, 0u) != CUDA_SUCCESS ||
			cuMemMap(base, granularity, 0u, handle, 0u) != CUDA_SUCCESS ||
			cuMemSetAccess(base, granularity, &access, 1u) != CUDA_SUCCESS )
			return(1u);
		mapped_chunk[0] = 1u;
	}
	for ( uint32_t r = 0u; r < routes; ++r )
	{
		size_t begin = mapping == TP16_MAP_ALL ? 0u : (size_t)routed[r] * expert_bytes;
		size_t end = mapping == TP16_MAP_ALL ? total : begin + expert_bytes;
		for ( size_t c = begin / granularity; c <= (end - 1u) / granularity; ++c )
			if ( mapped_chunk[c] == 0u )
			{
				CUmemGenericAllocationHandle handle;
				if ( cuMemCreate(&handle, granularity, &prop, 0u) != CUDA_SUCCESS ||
					cuMemMap(base + c * granularity, granularity, 0u, handle, 0u) != CUDA_SUCCESS ||
					cuMemSetAccess(base + c * granularity, granularity, &access, 1u) != CUDA_SUCCESS )
					return(1u);
				mapped_chunk[c] = 1u;
			}
		cudaMemcpy((uint8_t *)base + (size_t)routed[r] * expert_bytes, one, expert_bytes, cudaMemcpyHostToDevice);
	}
	for ( uint32_t e = 0u, r = 0u; e < experts; ++e )
	{
		offset[e] = r;
		if ( r < routes && routed[r] == e )
			r++;
		offset[e + 1u] = r;
		prefix[e + 1u] = prefix[e] + (offset[e + 1u] - offset[e]) * tiles_per_expert;
	}
	for ( uint32_t p = 0u; p < routes; ++p )
		for ( uint32_t k = 0u; k < input; ++k )
			packed[(size_t)p * input + k] = HostFloatToBf16((float)((int32_t)((k * 37u) % 11u) - 5) * 0.25f);
	cudaMalloc(&d_packed, (size_t)routes * input * sizeof(uint16_t));
	cudaMalloc(&d_out, (size_t)routes * output * sizeof(uint16_t));
	cudaMalloc(&d_offset, (experts + 1u) * sizeof(uint32_t));
	cudaMalloc(&d_prefix, (experts + 1u) * sizeof(uint32_t));
	cudaMemcpy(d_packed, packed, (size_t)routes * input * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, offset, (experts + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prefix, prefix, (experts + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice);
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = d_offset;
	gemm.group_tile_prefix = d_prefix;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = d_out;
	status = LmGemmWeightOnlyInterleavedLaunch<LmMxfp4, TP16_TILE_N, TP16_STAGES, TP16_WARPS, TP16_TILE_K>(
		&gemm, d_packed, (const void *)base, routes, 1u, routes, experts,
		input, output, multiprocessors, true, (cudaStream_t)0);
	err = cudaDeviceSynchronize();
	printf("leased experts=%u in=%u out=%u mapping=%s: launch=%d sync=%s\n", experts, input, output, mapping == TP16_MAP_ALL ? "all" : "routed+base", (int)status, cudaGetErrorString(err));
	if ( status != LM_LAUNCH_OK || err != cudaSuccess )
		return(1u);
	{
		uint16_t *out = (uint16_t *)malloc((size_t)routes * output * sizeof(uint16_t));
		uint32_t wrong = 0u;
		cudaMemcpy(out, d_out, (size_t)routes * output * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		for ( uint32_t p = 0u; p < routes; ++p )
		{
			float base_sum = 0.0f;
			for ( uint32_t k = 0u; k < input; ++k )
				base_sum += HostBf16ToFloat(packed[(size_t)p * input + k]);
			for ( uint32_t n = 0u; n < output; ++n )
			{
				float expect = base_sum * CellScale(0u, n / 16u);
				float got = HostBf16ToFloat(out[(size_t)p * output + n]);
				if ( fabsf(got - expect) > 0.03f * fabsf(expect) + 1e-2f )
				{
					if ( wrong < 4u )
						printf("leased mismatch p=%u n=%u got=%g expect=%g\n", p, n, got, expect);
					wrong++;
				}
			}
		}
		free(out);
		return(wrong);
	}
}

int main(void)
{
	int multiprocessors = 0;
	uint32_t failures = 0u;
	if ( cudaSetDevice(0) != cudaSuccess )
		return(1);
	cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
	failures += RunShape(256u, 128u, (uint32_t)multiprocessors);
	failures += RunShape(3072u, 256u, (uint32_t)multiprocessors);
	failures += RunShape(3072u, 224u, (uint32_t)multiprocessors);
	failures += RunLeased(3072u, 224u, 896u, (uint32_t)multiprocessors, TP16_MAP_ALL);
	failures += RunLeased(256u, 224u, 896u, (uint32_t)multiprocessors, TP16_MAP_ROUTED_AND_BASE);
	failures += RunLeased(3072u, 224u, 896u, (uint32_t)multiprocessors, TP16_MAP_ROUTED_AND_BASE);
	printf("test_k3_tp16_expert_gemm: %u failures\n", failures);
	return(failures != 0u ? 1 : 0);
}
