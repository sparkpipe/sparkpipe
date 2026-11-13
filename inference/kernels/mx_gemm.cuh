#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/mma.cuh"
#include "runtime/launch.h"
#include <atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define LM_MX_THREADS 256u
#define LM_MX_WARPS 8u
#define LM_MX_TILE_K 128u
#define LM_MX_BLOCK 32u
#define LM_MX_BAND_ROW_TILES 8u
#define LM_MX_SHARED_LIMIT 101376u
#define LM_MX_MAX_STAGES 4u
#define LM_MX_MAX_DEVICES 64u
#define LM_MX_QUANTIZE_GRID 4096u

typedef struct LmMxGemmArguments
{
	const uint8_t *activation;
	const uint8_t *activation_scale;
	const uint8_t *weight;
	const uint8_t *weight_scale;
	uint16_t *output_bf16;
	uint32_t rows;
	uint32_t input_dimension,output_dimension;
	uint32_t output_row_stride,output_column_offset;
}
LmMxGemmArguments;

__global__ __launch_bounds__(LM_MX_THREADS) void LmMxQuantizeKernel(const uint16_t *__restrict__ source, uint8_t *__restrict__ codes,
	uint8_t *__restrict__ scales, uint64_t octets)
{
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	for (uint64_t base = (uint64_t)blockIdx.x * blockDim.x + (threadIdx.x & ~31u); base < octets; base += stride)
	{
		const uint64_t octet = base + (threadIdx.x & 31u);
		const uint32_t live = octet < octets ? 1u : 0u;
		const uint4 raw = live != 0u ? *(const uint4 *)(source + octet * 8u) : make_uint4(0u,0u,0u,0u);
		const uint32_t words[4] = { raw.x, raw.y, raw.z, raw.w };
		float value[8],amax = 0.0f,scale;
		uint32_t low = 0u,high = 0u,e;
#pragma unroll
		for (e = 0u; e < 4u; e++)
		{
			value[2u * e] = __uint_as_float(words[e] << 16u);
			value[2u * e + 1u] = __uint_as_float(words[e] & 0xffff0000u);
			amax = fmaxf(amax,fmaxf(fabsf(value[2u * e]),fabsf(value[2u * e + 1u])));
		}
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,1u));
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,2u));
		scale = SparkMathFp8QdqScale(fmaxf(0.0f,amax));
#pragma unroll
		for (e = 0u; e < 4u; e++)
		{
			low |= (uint32_t)LmFloatToE4m3(fminf(fmaxf(value[e] / scale,-LM_E4M3_MAX),LM_E4M3_MAX)) << (8u * e);
			high |= (uint32_t)LmFloatToE4m3(fminf(fmaxf(value[4u + e] / scale,-LM_E4M3_MAX),LM_E4M3_MAX)) << (8u * e);
		}
		if ( live != 0u )
		{
			*(uint2 *)(codes + octet * 8u) = make_uint2(low,high);
			if ( (octet & 3u) == 0u )
				scales[octet >> 2u] = (uint8_t)((__float_as_uint(scale) >> 23u) & 0xffu);
		}
	}
}

static int32_t LmMxQuantize(const uint16_t *source, uint8_t *codes, uint8_t *scales, uint64_t elements, cudaStream_t stream)
{
	uint64_t blocks;
	if ( source == 0 || codes == 0 || scales == 0 || elements == 0u || elements % LM_MX_BLOCK != 0u || ((uintptr_t)source & 15u) != 0u ||
		((uintptr_t)codes & 7u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	blocks = (elements / 8u + LM_MX_THREADS - 1u) / LM_MX_THREADS;
	LmMxQuantizeKernel<<<(uint32_t)(blocks < LM_MX_QUANTIZE_GRID ? blocks : LM_MX_QUANTIZE_GRID),LM_MX_THREADS,0,stream>>>(source,codes,scales,elements / 8u);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ void LmMxMatrix4(uint32_t address, uint32_t value[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3]) : "r"(address));
}

static __device__ __forceinline__ void LmMxCopy16(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmMxCopy4(uint32_t destination, const void *source)
{
	asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(destination), "l"(source) : "memory");
}

static __device__ __forceinline__ void LmMxCommit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<uint32_t PENDING>
static __device__ __forceinline__ void LmMxWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

static __device__ __forceinline__ uint32_t LmMxSwizzle(uint32_t row, uint32_t chunk)
{
	return(row * LM_MX_TILE_K + ((chunk ^ (row & 7u)) << 4u));
}

template<uint32_t TILE_M, uint32_t TILE_N>
static __host__ __device__ constexpr uint32_t LmMxStageBytes(void)
{
	return((TILE_M + TILE_N) * (LM_MX_TILE_K + LM_MX_TILE_K / LM_MX_BLOCK));
}

template<uint32_t TILE_M, uint32_t TILE_N>
static __host__ __device__ constexpr uint32_t LmMxStages(void)
{
	return(LM_MX_SHARED_LIMIT / LmMxStageBytes<TILE_M,TILE_N>() < LM_MX_MAX_STAGES ? LM_MX_SHARED_LIMIT / LmMxStageBytes<TILE_M,TILE_N>() : LM_MX_MAX_STAGES);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M>
__global__ __launch_bounds__(LM_MX_THREADS,1) void LmMxGemmKernel(const __grid_constant__ LmMxGemmArguments args)
{
	constexpr uint32_t STAGES = LmMxStages<TILE_M,TILE_N>();
	constexpr uint32_t WARPS_N = LM_MX_WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M;
	constexpr uint32_t WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u;
	constexpr uint32_t NF = WARP_N / 8u;
	constexpr uint32_t STAGE_BYTES = LmMxStageBytes<TILE_M,TILE_N>();
	constexpr uint32_t A_BYTES = TILE_M * LM_MX_TILE_K;
	constexpr uint32_t B_BYTES = TILE_N * LM_MX_TILE_K;
	constexpr uint32_t A_PER_THREAD = TILE_M * (LM_MX_TILE_K / 16u) / LM_MX_THREADS;
	constexpr uint32_t B_PER_THREAD = TILE_N * (LM_MX_TILE_K / 16u) / LM_MX_THREADS;
	static_assert(STAGES >= 3u,"the tile leaves no room for a third pipeline stage");
	static_assert(WARPS_M * WARPS_N == LM_MX_WARPS && MF >= 1u && NF >= 2u && NF % 2u == 0u && WARP_M % 16u == 0u,"the tile is not a whole number of fragments per warp");
	static_assert(TILE_M * (LM_MX_TILE_K / 16u) % LM_MX_THREADS == 0u && TILE_N * (LM_MX_TILE_K / 16u) % LM_MX_THREADS == 0u,"a stage must divide across the block");
	static_assert(TILE_M + TILE_N <= 2u * LM_MX_THREADS,"the scale words must load in two passes");
	extern __shared__ __align__(128) uint8_t lm_mx_shared[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(lm_mx_shared);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u,group = lane >> 2u,quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N,warp_n = warp % WARPS_N;
	const uint32_t scale_columns = args.input_dimension / LM_MX_BLOCK;
	const uint32_t row_tiles = (args.rows + TILE_M - 1u) / TILE_M;
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t total_tiles = row_tiles * neuron_tiles;
	const uint32_t k_tiles = args.input_dimension / LM_MX_TILE_K;
	const uint32_t own_tiles = blockIdx.x < total_tiles ? (total_tiles - blockIdx.x + gridDim.x - 1u) / gridDim.x : 0u;
	const uint32_t steps = own_tiles * k_tiles;
	float total[MF][NF][4];
	uint32_t load_row_base = 0u,load_neuron_base = 0u,work_row_base = 0u,work_neuron_base = 0u,produced,consumed,i,j,e;

	auto place = [&](uint32_t tile, uint32_t &row_base, uint32_t &neuron_base)
	{
		const uint32_t band = tile / (LM_MX_BAND_ROW_TILES * neuron_tiles);
		const uint32_t band_first = band * LM_MX_BAND_ROW_TILES;
		const uint32_t band_rows = row_tiles - band_first < LM_MX_BAND_ROW_TILES ? row_tiles - band_first : LM_MX_BAND_ROW_TILES;
		const uint32_t within = tile - band * LM_MX_BAND_ROW_TILES * neuron_tiles;
		row_base = (band_first + within % band_rows) * TILE_M;
		neuron_base = (within / band_rows) * TILE_N;
	};
	auto issue = [&](uint32_t step)
	{
		const uint32_t stage = shared_base + (step % STAGES) * STAGE_BYTES,k = step % k_tiles;
		uint32_t q,chunk,row,valid,source;
		if ( k == 0u )
			place(blockIdx.x + (step / k_tiles) * gridDim.x,load_row_base,load_neuron_base);
#pragma unroll
		for (q = 0u; q < A_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_MX_THREADS;
			row = chunk >> 3u;
			valid = load_row_base + row < args.rows ? 1u : 0u;
			source = valid != 0u ? load_row_base + row : 0u;
			LmMxCopy16(stage + LmMxSwizzle(row,chunk & 7u),args.activation + (uint64_t)source * args.input_dimension + (uint64_t)k * LM_MX_TILE_K + (chunk & 7u) * 16u,valid * 16u);
		}
#pragma unroll
		for (q = 0u; q < B_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_MX_THREADS;
			row = chunk >> 3u;
			valid = load_neuron_base + row < args.output_dimension ? 1u : 0u;
			source = valid != 0u ? load_neuron_base + row : 0u;
			LmMxCopy16(stage + A_BYTES + LmMxSwizzle(row,chunk & 7u),args.weight + (uint64_t)source * args.input_dimension + (uint64_t)k * LM_MX_TILE_K + (chunk & 7u) * 16u,valid * 16u);
		}
#pragma unroll
		for (q = 0u; q < 2u; q++)
		{
			row = threadIdx.x + q * LM_MX_THREADS;
			if ( row < TILE_M )
			{
				source = load_row_base + row < args.rows ? load_row_base + row : args.rows - 1u;
				LmMxCopy4(stage + A_BYTES + B_BYTES + row * 4u,args.activation_scale + (uint64_t)source * scale_columns + (uint64_t)k * (LM_MX_TILE_K / LM_MX_BLOCK));
			}
			else if ( row < TILE_M + TILE_N )
			{
				source = load_neuron_base + row - TILE_M < args.output_dimension ? load_neuron_base + row - TILE_M : args.output_dimension - 1u;
				LmMxCopy4(stage + A_BYTES + B_BYTES + row * 4u,args.weight_scale + (uint64_t)source * scale_columns + (uint64_t)k * (LM_MX_TILE_K / LM_MX_BLOCK));
			}
		}
	};

	for (produced = 0u; produced + 1u < STAGES; produced++)
	{
		if ( produced < steps )
			issue(produced);
		LmMxCommit();
	}
	for (consumed = 0u; consumed < steps; consumed++)
	{
		const uint32_t kt = consumed % k_tiles;
		const uint32_t *words;
		uint32_t stage_a,stage_b,scale_a[MF],scale_b[NF];
		if ( kt == 0u )
		{
			place(blockIdx.x + (consumed / k_tiles) * gridDim.x,work_row_base,work_neuron_base);
#pragma unroll
			for (i = 0u; i < MF; i++)
#pragma unroll
				for (j = 0u; j < NF; j++)
#pragma unroll
					for (e = 0u; e < 4u; e++)
						total[i][j][e] = 0.0f;
		}
		LmMxWait<STAGES - 2u>();
		__syncthreads();
		if ( produced < steps )
			issue(produced);
		produced++;
		LmMxCommit();
		stage_a = shared_base + (consumed % STAGES) * STAGE_BYTES;
		stage_b = stage_a + A_BYTES;
		words = (const uint32_t *)(lm_mx_shared + (consumed % STAGES) * STAGE_BYTES + A_BYTES + B_BYTES);
#pragma unroll
		for (i = 0u; i < MF; i++)
			scale_a[i] = words[warp_m * WARP_M + i * 16u + group + ((lane & 1u) << 3u)];
#pragma unroll
		for (j = 0u; j < NF; j++)
			scale_b[j] = words[TILE_M + warp_n * WARP_N + j * 8u + group];
#pragma unroll
		for (uint32_t step = 0u; step < LM_MX_TILE_K / LM_MX_BLOCK; step++)
		{
			uint32_t a[MF][4],b[NF][2];
#pragma unroll
			for (i = 0u; i < MF; i++)
				LmMxMatrix4(stage_a + LmMxSwizzle(warp_m * WARP_M + i * 16u + ((lane >> 3u) & 1u) * 8u + (lane & 7u),step * 2u + (lane >> 4u)),a[i]);
#pragma unroll
			for (j = 0u; j < NF; j += 2u)
			{
				uint32_t value[4];
				LmMxMatrix4(stage_b + LmMxSwizzle(warp_n * WARP_N + j * 8u + (lane >> 4u) * 8u + (lane & 7u),step * 2u + ((lane >> 3u) & 1u)),value);
				b[j][0] = value[0];
				b[j][1] = value[1];
				b[j + 1u][0] = value[2];
				b[j + 1u][1] = value[3];
			}
#pragma unroll
			for (i = 0u; i < MF; i++)
#pragma unroll
				for (j = 0u; j < NF; j++)
				{
					float block[4];
					LmMmaMxf8Block(block,a[i],b[j],scale_a[i],(uint16_t)step,scale_b[j],(uint16_t)step);
#pragma unroll
					for (e = 0u; e < 4u; e++)
						total[i][j][e] += block[e];
				}
		}
		if ( kt + 1u == k_tiles )
#pragma unroll
			for (i = 0u; i < MF; i++)
#pragma unroll
				for (j = 0u; j < NF; j++)
#pragma unroll
					for (e = 0u; e < 4u; e += 2u)
					{
						const uint32_t row = work_row_base + warp_m * WARP_M + i * 16u + group + (e >> 1u) * 8u;
						const uint32_t column = work_neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
						if ( row < args.rows && column < args.output_dimension )
							*(uint32_t *)(args.output_bf16 + (uint64_t)row * args.output_row_stride + args.output_column_offset + column) =
								(uint32_t)LmFloatToBf16(total[i][j][e]) | ((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
					}
	}
	LmMxWait<0u>();
}

static int32_t LmMxGemmValidate(const LmMxGemmArguments *args, uint32_t multiprocessors)
{
	if ( args == 0 || args->activation == 0 || args->activation_scale == 0 || args->weight == 0 || args->weight_scale == 0 || args->output_bf16 == 0 ||
		multiprocessors == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || args->input_dimension == 0u || args->input_dimension % LM_MX_TILE_K != 0u || args->output_dimension == 0u ||
		(args->output_dimension & 1u) != 0u || (args->output_column_offset & 1u) != 0u || (args->output_row_stride & 1u) != 0u ||
		args->output_row_stride < args->output_column_offset + args->output_dimension )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( (((uintptr_t)args->activation | (uintptr_t)args->weight) & 15u) != 0u || (((uintptr_t)args->activation_scale | (uintptr_t)args->weight_scale) & 3u) != 0u ||
		((uintptr_t)args->output_bf16 & 3u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M>
static int32_t LmMxGemmLaunchTile(const LmMxGemmArguments *args, uint32_t multiprocessors, cudaStream_t stream)
{
	static std::atomic<uint64_t> granted(0u);
	const uint32_t shared = LmMxStages<TILE_M,TILE_N>() * LmMxStageBytes<TILE_M,TILE_N>();
	const uint64_t tiles = (uint64_t)((args->rows + TILE_M - 1u) / TILE_M) * ((args->output_dimension + TILE_N - 1u) / TILE_N);
	int device = 0;
	if ( cudaGetDevice(&device) != cudaSuccess || device < 0 || (uint32_t)device >= LM_MX_MAX_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( cudaFuncSetAttribute((const void *)LmMxGemmKernel<TILE_M,TILE_N,WARPS_M>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)LM_MX_SHARED_LIMIT) != cudaSuccess )
			return(LM_LAUNCH_ERR_ATTRIBUTE);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	LmMxGemmKernel<TILE_M,TILE_N,WARPS_M><<<(uint32_t)(tiles < multiprocessors ? tiles : multiprocessors),LM_MX_THREADS,shared,stream>>>(*args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmMxGemmDense(const uint8_t *activation, const uint8_t *activation_scale, const uint8_t *weight, const uint8_t *weight_scale,
	uint16_t *output_bf16, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset,
	uint32_t multiprocessors, cudaStream_t stream)
{
	LmMxGemmArguments args;
	uint64_t neuron_tiles;
	int32_t status;
	memset(&args,0,sizeof(args));
	args.activation = activation;
	args.activation_scale = activation_scale;
	args.weight = weight;
	args.weight_scale = weight_scale;
	args.output_bf16 = output_bf16;
	args.rows = rows;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
	args.output_column_offset = output_column_offset;
	status = LmMxGemmValidate(&args,multiprocessors);
	if ( status != LM_LAUNCH_OK )
		return(status);
	neuron_tiles = (output_dimension + 127u) / 128u;
	if ( (uint64_t)((rows + 127u) / 128u) * neuron_tiles >= 2u * (uint64_t)multiprocessors )
		return(LmMxGemmLaunchTile<128u,128u,2u>(&args,multiprocessors,stream));
	if ( (uint64_t)((rows + 63u) / 64u) * neuron_tiles >= (uint64_t)multiprocessors )
		return(LmMxGemmLaunchTile<64u,128u,2u>(&args,multiprocessors,stream));
	return(LmMxGemmLaunchTile<32u,128u,2u>(&args,multiprocessors,stream));
}
