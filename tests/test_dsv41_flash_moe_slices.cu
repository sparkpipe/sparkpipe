#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/cell_pack.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define EXPERTS 64u
#define DEGREE 16u
#define LOCAL (EXPERTS / DEGREE)
#define TOPK 6u
#define HIDDEN 5120u
#define INTER 2304u
#define SLICE (INTER / DEGREE)
#define HIDDEN_SLICE (HIDDEN / DEGREE)
#define ROW_ELEMENTS (TOPK * SLICE + HIDDEN_SLICE)
#define LIMIT 10.0f
#define THREADS 256u
#define SKINNY_ROWS 16u

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static __global__ void FillBytes(uint8_t *data, uint64_t count, uint64_t seed, uint32_t low, uint32_t span)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t x = (index + 1u) * 0x9E3779B97F4A7C15ull ^ seed;
		x ^= x >> 31u; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29u;
		data[index] = (uint8_t)(low + (uint32_t)(x % span));
	}
}

static __global__ void FillBf16(uint16_t *data, uint64_t count, uint64_t seed, float scale)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t x = (index + 7u) * 0xD1B54A32D192ED03ull ^ seed;
		x ^= x >> 33u; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33u;
		data[index] = LmFloatToBf16(((float)(x % 20001u) / 10000.0f - 1.0f) * scale);
	}
}

static __global__ void SumRanks(const uint16_t *partials, uint16_t *output, uint32_t ranks, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t rank = 0u; rank < ranks; ++rank )
			sum += LmBf16ToFloat(partials[(uint64_t)rank * elements + index]);
		output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
	}
}

static void *Device(uint64_t bytes)
{
	void *pointer = 0;
	CUDA(cudaMalloc(&pointer, bytes));
	return pointer;
}

static int32_t Experts(const void *cells, const uint16_t *activation, uint16_t *output, const uint32_t *offsets, const uint32_t *source,
	uint32_t groups, uint32_t rows, uint32_t routes, uint32_t packed, uint32_t input, uint32_t outputs)
{
	if ( rows <= SKINNY_ROWS )
		return LmSkinnyCellExperts(cells, activation, output, offsets, packed != 0u ? 0 : source, groups, routes, rows, packed, input, outputs,
			LM_SKINNY_CELL_TILE_K, 0);
	return LmCellMmaExperts(cells, activation, output, offsets, packed != 0u ? 0 : source, groups, routes, packed, input, outputs,
		LM_SKINNY_CELL_TILE_K, 0);
}

int main(void)
{
	const uint64_t up_bytes = (uint64_t)INTER * (HIDDEN / 2u), up_scales = (uint64_t)INTER * (HIDDEN / 32u);
	const uint64_t down_bytes = (uint64_t)HIDDEN * (INTER / 2u), down_scales = (uint64_t)HIDDEN * (INTER / 32u);
	const uint32_t row_counts[] = {1u, 3u, 16u, 17u, 64u};
	uint8_t *w1 = (uint8_t *)Device(EXPERTS * up_bytes), *w3 = (uint8_t *)Device(EXPERTS * up_bytes), *w2 = (uint8_t *)Device(EXPERTS * down_bytes);
	uint8_t *s1 = (uint8_t *)Device(EXPERTS * up_scales), *s3 = (uint8_t *)Device(EXPERTS * up_scales), *s2 = (uint8_t *)Device(EXPERTS * down_scales);
	std::vector<uint8_t *> local_up(DEGREE), local_gate(DEGREE), local_down(DEGREE), slice_up(DEGREE), slice_down(DEGREE);
	uint64_t failures = 0u;
	FillBytes<<<1024, 256>>>(w1, EXPERTS * up_bytes, 11u, 0u, 256u);
	FillBytes<<<1024, 256>>>(w3, EXPERTS * up_bytes, 13u, 0u, 256u);
	FillBytes<<<1024, 256>>>(w2, EXPERTS * down_bytes, 17u, 0u, 256u);
	FillBytes<<<1024, 256>>>(s1, EXPERTS * up_scales, 19u, 119u, 6u);
	FillBytes<<<1024, 256>>>(s3, EXPERTS * up_scales, 23u, 119u, 6u);
	FillBytes<<<1024, 256>>>(s2, EXPERTS * down_scales, 29u, 117u, 6u);
	CUDA(cudaDeviceSynchronize());
	{
		uint8_t *payload = (uint8_t *)Device(EXPERTS * 2u * SLICE * (HIDDEN / 2u)), *scales = (uint8_t *)Device(EXPERTS * 2u * SLICE * (HIDDEN / 32u));
		uint8_t *down_payload = (uint8_t *)Device(EXPERTS * HIDDEN_SLICE * (INTER / 2u)), *down_scale = (uint8_t *)Device(EXPERTS * HIDDEN_SLICE * (INTER / 32u));
		for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		{
			local_gate[rank] = (uint8_t *)Device(LmCellPackBytes(LOCAL, INTER, HIDDEN));
			local_up[rank] = (uint8_t *)Device(LmCellPackBytes(LOCAL, INTER, HIDDEN));
			local_down[rank] = (uint8_t *)Device(LmCellPackBytes(LOCAL, HIDDEN, INTER));
			REQUIRE(LmCellPackMxfp4(w1 + rank * LOCAL * up_bytes, s1 + rank * LOCAL * up_scales, local_gate[rank], LOCAL, INTER, HIDDEN, 0) == LM_LAUNCH_OK);
			REQUIRE(LmCellPackMxfp4(w3 + rank * LOCAL * up_bytes, s3 + rank * LOCAL * up_scales, local_up[rank], LOCAL, INTER, HIDDEN, 0) == LM_LAUNCH_OK);
			REQUIRE(LmCellPackMxfp4(w2 + rank * LOCAL * down_bytes, s2 + rank * LOCAL * down_scales, local_down[rank], LOCAL, HIDDEN, INTER, 0) == LM_LAUNCH_OK);
			for ( uint32_t part = 0u; part < 2u; ++part )
			{
				const uint8_t *weights = part == 0u ? w1 : w3, *weight_scales = part == 0u ? s1 : s3;
				CUDA(cudaMemcpy2D(payload + part * SLICE * (HIDDEN / 2u), 2u * SLICE * (HIDDEN / 2u), weights + rank * SLICE * (HIDDEN / 2u), up_bytes,
					SLICE * (HIDDEN / 2u), EXPERTS, cudaMemcpyDeviceToDevice));
				CUDA(cudaMemcpy2D(scales + part * SLICE * (HIDDEN / 32u), 2u * SLICE * (HIDDEN / 32u), weight_scales + rank * SLICE * (HIDDEN / 32u),
					up_scales, SLICE * (HIDDEN / 32u), EXPERTS, cudaMemcpyDeviceToDevice));
			}
			CUDA(cudaMemcpy2D(down_payload, HIDDEN_SLICE * (INTER / 2u), w2 + rank * HIDDEN_SLICE * (INTER / 2u), down_bytes, HIDDEN_SLICE * (INTER / 2u),
				EXPERTS, cudaMemcpyDeviceToDevice));
			CUDA(cudaMemcpy2D(down_scale, HIDDEN_SLICE * (INTER / 32u), s2 + rank * HIDDEN_SLICE * (INTER / 32u), down_scales, HIDDEN_SLICE * (INTER / 32u),
				EXPERTS, cudaMemcpyDeviceToDevice));
			slice_up[rank] = (uint8_t *)Device(LmCellPackBytes(EXPERTS, 2u * SLICE, HIDDEN));
			slice_down[rank] = (uint8_t *)Device(LmCellPackBytes(EXPERTS, HIDDEN_SLICE, INTER));
			REQUIRE(LmCellPackMxfp4(payload, scales, slice_up[rank], EXPERTS, 2u * SLICE, HIDDEN, 0) == LM_LAUNCH_OK);
			REQUIRE(LmCellPackMxfp4(down_payload, down_scale, slice_down[rank], EXPERTS, HIDDEN_SLICE, INTER, 0) == LM_LAUNCH_OK);
			CUDA(cudaDeviceSynchronize());
		}
		CUDA(cudaFree(payload)); CUDA(cudaFree(scales)); CUDA(cudaFree(down_payload)); CUDA(cudaFree(down_scale));
	}
	for ( uint32_t trial = 0u; trial < sizeof(row_counts) / sizeof(row_counts[0]); ++trial )
	{
		const uint32_t rows = row_counts[trial], routes = rows * TOPK;
		const uint64_t stride = (uint64_t)rows * ROW_ELEMENTS;
		std::vector<uint32_t> host_expert(routes);
		std::vector<float> host_weight(routes);
		srand(1234u + rows);
		for ( uint32_t row = 0u; row < rows; ++row )
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				uint32_t expert, earlier;
				do
				{
					expert = (uint32_t)rand() % EXPERTS;
					for ( earlier = 0u; earlier < k && host_expert[row * TOPK + earlier] != expert; ++earlier )
						;
				} while ( earlier < k );
				host_expert[row * TOPK + k] = expert;
				host_weight[row * TOPK + k] = 0.05f + (float)(rand() % 1000) / 2000.0f;
			}
		uint16_t *xq = (uint16_t *)Device((uint64_t)rows * HIDDEN * 2u), *shared = (uint16_t *)Device((uint64_t)DEGREE * rows * HIDDEN * 2u);
		uint32_t *global = (uint32_t *)Device(routes * 4u), *expert = (uint32_t *)Device(routes * 4u), *global_copy = (uint32_t *)Device(routes * 4u);
		float *weight = (float *)Device(routes * 4u), *packed_weight = (float *)Device(routes * 4u), *weight_copy = (float *)Device(routes * 4u);
		uint32_t *offsets = (uint32_t *)Device((EXPERTS + 2u) * 4u), *packed_row = (uint32_t *)Device(routes * 4u), *source = (uint32_t *)Device(routes * 4u);
		uint32_t *tiles_up = (uint32_t *)Device((EXPERTS + 2u) * 4u), *tiles_down = (uint32_t *)Device((EXPERTS + 2u) * 4u);
		uint16_t *gate = (uint16_t *)Device((uint64_t)routes * INTER * 2u), *up = (uint16_t *)Device((uint64_t)routes * INTER * 2u);
		uint16_t *act = (uint16_t *)Device((uint64_t)routes * INTER * 2u), *out = (uint16_t *)Device((uint64_t)routes * HIDDEN * 2u);
		uint16_t *partials = (uint16_t *)Device((uint64_t)DEGREE * rows * HIDDEN * 2u), *reference = (uint16_t *)Device((uint64_t)rows * HIDDEN * 2u);
		uint16_t *gate_up = (uint16_t *)Device((uint64_t)routes * 2u * SLICE * 2u), *slice_out = (uint16_t *)Device((uint64_t)routes * HIDDEN_SLICE * 2u);
		uint16_t *send = (uint16_t *)Device((uint64_t)DEGREE * DEGREE * stride * 2u), *received = (uint16_t *)Device((uint64_t)DEGREE * stride * 2u);
		uint16_t *final_slices = (uint16_t *)Device((uint64_t)DEGREE * rows * HIDDEN_SLICE * 2u), *candidate = (uint16_t *)Device((uint64_t)rows * HIDDEN * 2u);
		FillBf16<<<256, 256>>>(xq, (uint64_t)rows * HIDDEN, 31u + rows, 2.0f);
		FillBf16<<<256, 256>>>(shared, (uint64_t)DEGREE * rows * HIDDEN, 37u + rows, 0.25f);
		SparkDsv41FlashLaunchFp8Qdq(0, xq, (uint64_t)rows * HIDDEN);
		CUDA(cudaMemcpy(global, host_expert.data(), routes * 4u, cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(weight, host_weight.data(), routes * 4u, cudaMemcpyHostToDevice));
		for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		{
			CUDA(cudaMemcpy(expert, global, routes * 4u, cudaMemcpyDeviceToDevice));
			CUDA(cudaMemcpy(weight_copy, weight, routes * 4u, cudaMemcpyDeviceToDevice));
			LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u>>>(expert, weight_copy, global_copy, routes, rank * LOCAL, LOCAL);
			REQUIRE((LmRouteBuild<THREADS,LOCAL + 1u>(expert, rows, routes, TOPK, offsets, packed_row, source, INTER, HIDDEN, 128u, tiles_up, tiles_down, 0)) == LM_LAUNCH_OK);
			LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u>>>(packed_row, weight_copy, packed_weight, routes);
			REQUIRE(Experts(local_gate[rank], xq, gate, offsets, source, LOCAL, rows, routes, 0u, HIDDEN, INTER) == LM_LAUNCH_OK);
			REQUIRE(Experts(local_up[rank], xq, up, offsets, source, LOCAL, rows, routes, 0u, HIDDEN, INTER) == LM_LAUNCH_OK);
			LmSwigluPackedKernel<<<dim3(INTER / THREADS, routes), THREADS>>>(gate, up, packed_weight, act, INTER, LIMIT);
			SparkDsv41FlashLaunchFp8Qdq(0, act, (uint64_t)routes * INTER);
			REQUIRE(Experts(local_down[rank], act, out, offsets, source, LOCAL, rows, routes, 1u, INTER, HIDDEN) == LM_LAUNCH_OK);
			LmMoeLocalFinalizeKernel<<<dim3(HIDDEN / THREADS, rows), THREADS>>>(out, packed_row, expert, shared + (uint64_t)rank * rows * HIDDEN,
				partials + (uint64_t)rank * rows * HIDDEN, TOPK, LOCAL, HIDDEN);
			CUDA(cudaDeviceSynchronize());
		}
		SumRanks<<<256, 256>>>(partials, reference, DEGREE, (uint64_t)rows * HIDDEN);
		REQUIRE((LmRouteBuild<THREADS,EXPERTS>(global, rows, routes, TOPK, offsets, packed_row, source, 2u * SLICE, HIDDEN_SLICE, 128u, tiles_up, tiles_down, 0)) ==
			LM_LAUNCH_OK);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u>>>(packed_row, weight, packed_weight, routes);
		for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		{
			REQUIRE(Experts(slice_up[rank], xq, gate_up, offsets, source, EXPERTS, rows, routes, 0u, HIDDEN, 2u * SLICE) == LM_LAUNCH_OK);
			D41MoeSendKernel<<<dim3(HIDDEN / THREADS, routes + rows), THREADS>>>(gate_up, packed_weight, shared + (uint64_t)rank * rows * HIDDEN,
				send + (uint64_t)rank * DEGREE * stride, routes, SLICE, HIDDEN_SLICE, DEGREE, stride, LIMIT);
		}
		for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		{
			for ( uint32_t peer = 0u; peer < DEGREE; ++peer )
				CUDA(cudaMemcpy(received + (uint64_t)peer * stride, send + ((uint64_t)peer * DEGREE + rank) * stride, stride * 2u, cudaMemcpyDeviceToDevice));
			D41ActAssembleQdqKernel<<<dim3(INTER / THREADS, routes), THREADS>>>(received, act, SLICE, INTER, stride);
			REQUIRE(Experts(slice_down[rank], act, slice_out, offsets, source, EXPERTS, rows, routes, 1u, INTER, HIDDEN_SLICE) == LM_LAUNCH_OK);
			D41FinalizeSliceKernel<<<dim3((HIDDEN_SLICE + THREADS - 1u) / THREADS, rows), THREADS>>>(slice_out, received, global, packed_row,
				final_slices + (uint64_t)rank * rows * HIDDEN_SLICE, TOPK, LOCAL, DEGREE, HIDDEN_SLICE, stride, (uint64_t)routes * SLICE);
		}
		D41SliceGatherKernel<<<dim3(HIDDEN / THREADS, rows), THREADS>>>(final_slices, candidate, rows, HIDDEN_SLICE, DEGREE);
		CUDA(cudaDeviceSynchronize());
		{
			std::vector<uint16_t> want((uint64_t)rows * HIDDEN), got((uint64_t)rows * HIDDEN);
			uint64_t differ = 0u, nonzero = 0u;
			CUDA(cudaMemcpy(want.data(), reference, want.size() * 2u, cudaMemcpyDeviceToHost));
			CUDA(cudaMemcpy(got.data(), candidate, got.size() * 2u, cudaMemcpyDeviceToHost));
			for ( uint64_t index = 0u; index < want.size(); ++index )
			{
				differ += want[index] != got[index] ? 1u : 0u;
				nonzero += (want[index] & 0x7FFFu) != 0u ? 1u : 0u;
			}
			printf("%s rows %u: %llu of %llu outputs differ (%llu nonzero, %s path)\n", differ == 0u && nonzero > want.size() / 2u ? "PASS" : "FAIL", rows,
				(unsigned long long)differ, (unsigned long long)want.size(), (unsigned long long)nonzero, rows <= SKINNY_ROWS ? "skinny" : "tensor-core");
			failures += differ != 0u || nonzero <= want.size() / 2u ? 1u : 0u;
		}
		void *buffers[] = {xq, shared, global, expert, global_copy, weight, packed_weight, weight_copy, offsets, packed_row, source, tiles_up, tiles_down,
			gate, up, act, out, partials, reference, gate_up, slice_out, send, received, final_slices, candidate};
		for ( void *buffer : buffers )
			CUDA(cudaFree(buffer));
	}
	if ( failures != 0u )
		return 1;
	printf("PASS dsv41 expert slices: the sliced MoE equals the per-rank-expert MoE plus the rank-ordered all-reduce bit for bit\n");
	return 0;
}
