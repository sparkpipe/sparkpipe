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
#include "sparkpipe/spark_gpu.h"

#define LM_MX_THREADS 256u
#define LM_MX_WARPS 8u
#define LM_MX_TILE_K 128u
#define LM_MX_BLOCK 32u
#define LM_MX_BAND_ROW_TILES 8u
#define LM_MX_SHARED_LIMIT 101376u
#define LM_MX_MAX_STAGES 4u
#define LM_MX_MAX_DEVICES 64u
#define LM_MX_QUANTIZE_GRID 4096u
#define LM_MX_MAX_GROUPS 8u

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

typedef struct LmMxGemmGroups
{
	const uint8_t *activation[LM_MX_MAX_GROUPS];
	const uint8_t *activation_scale[LM_MX_MAX_GROUPS];
	const uint8_t *weight[LM_MX_MAX_GROUPS];
	const uint8_t *weight_scale[LM_MX_MAX_GROUPS];
	uint32_t output_column_offset[LM_MX_MAX_GROUPS];
	uint32_t count;
}
LmMxGemmGroups;

__global__ __launch_bounds__(LM_MX_THREADS) void LmMxQuantizeKernel(const uint16_t *__restrict__ source, uint8_t *__restrict__ codes,
	uint8_t *__restrict__ scales, uint64_t octets, uint64_t row_octets, uint64_t row_stride)
{
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	for (uint64_t base = (uint64_t)blockIdx.x * blockDim.x + (threadIdx.x & ~31u); base < octets; base += stride)
	{
		const uint64_t octet = base + (threadIdx.x & 31u);
		const uint32_t live = octet < octets ? 1u : 0u;
		const uint64_t row = row_stride != 0u ? octet / row_octets : 0u;
		const uint4 raw = live != 0u ? *(const uint4 *)(source + row * row_stride + (octet - row * row_octets) * 8u) : make_uint4(0u,0u,0u,0u);
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
		const uint32_t scale_bits = __float_as_uint(scale);
		const float inv_scale = __uint_as_float(0x7f000000u - scale_bits);
		const uint32_t p0 = (uint32_t)LmFloatPairToE4m3(value[0] * inv_scale,value[1] * inv_scale);
		const uint32_t p1 = (uint32_t)LmFloatPairToE4m3(value[2] * inv_scale,value[3] * inv_scale);
		const uint32_t p2 = (uint32_t)LmFloatPairToE4m3(value[4] * inv_scale,value[5] * inv_scale);
		const uint32_t p3 = (uint32_t)LmFloatPairToE4m3(value[6] * inv_scale,value[7] * inv_scale);
		low = p0 | (p1 << 16u);
		high = p2 | (p3 << 16u);
		if ( live != 0u )
		{
			*(uint2 *)(codes + octet * 8u) = make_uint2(low,high);
			if ( (octet & 3u) == 0u )
				scales[octet >> 2u] = (uint8_t)((scale_bits >> 23u) & 0xffu);
		}
	}
}

static int32_t LmMxQuantize(const uint16_t *source, uint8_t *codes, uint8_t *scales, uint64_t elements, SparkGpuStream stream)
{
	uint64_t blocks;
	if ( source == 0 || codes == 0 || scales == 0 || elements == 0u || elements % LM_MX_BLOCK != 0u || ((uintptr_t)source & 15u) != 0u ||
		((uintptr_t)codes & 7u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	blocks = (elements / 8u + LM_MX_THREADS - 1u) / LM_MX_THREADS;
	LM_LAUNCH((LmMxQuantizeKernel),(uint32_t)(blocks < LM_MX_QUANTIZE_GRID ? blocks : LM_MX_QUANTIZE_GRID),LM_MX_THREADS,0,stream,source,codes,scales,elements / 8u,
		elements / 8u,0u);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmMxQuantizeRows(const uint16_t *source, uint64_t row_stride, uint8_t *codes, uint8_t *scales, uint32_t rows, uint32_t width, SparkGpuStream stream)
{
	const uint64_t octets = (uint64_t)rows * width / 8u;
	uint64_t blocks;
	if ( source == 0 || codes == 0 || scales == 0 || rows == 0u || width == 0u || width % LM_MX_BLOCK != 0u || row_stride < width || row_stride % 8u != 0u ||
		((uintptr_t)source & 15u) != 0u || ((uintptr_t)codes & 7u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	blocks = (octets + LM_MX_THREADS - 1u) / LM_MX_THREADS;
	LM_LAUNCH((LmMxQuantizeKernel),(uint32_t)(blocks < LM_MX_QUANTIZE_GRID ? blocks : LM_MX_QUANTIZE_GRID),LM_MX_THREADS,0,stream,source,codes,scales,octets,
		width / 8u,row_stride);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ void LmMxMatrix4(uint32_t address, uint32_t value[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3]) : "r"(address));
}

static __device__ __forceinline__ void LmMxCopy16(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
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
	return((TILE_M + TILE_N) * LM_MX_TILE_K);
}

template<uint32_t TILE_M, uint32_t TILE_N>
static __host__ __device__ constexpr uint32_t LmMxStages(void)
{
	return(3u);
}

template<uint32_t TILE_M, uint32_t TILE_N>
static __host__ __device__ constexpr uint32_t LmMxSharedBytes(void)
{
	return(LmMxStages<TILE_M,TILE_N>() * LmMxStageBytes<TILE_M,TILE_N>() + (TILE_M + TILE_N) * 8u);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M>
__global__ __launch_bounds__(LM_MX_THREADS,1) void LmMxGemmKernel(const __grid_constant__ LmMxGemmArguments args, const __grid_constant__ LmMxGemmGroups groups)
{
	constexpr uint32_t STAGES = LmMxStages<TILE_M,TILE_N>();
	constexpr uint32_t WARPS_N = LM_MX_WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M;
	constexpr uint32_t WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u;
	constexpr uint32_t NF = WARP_N / 8u;
	constexpr uint32_t STAGE_BYTES = LmMxStageBytes<TILE_M,TILE_N>();
	constexpr uint32_t A_BYTES = TILE_M * LM_MX_TILE_K;
	constexpr uint32_t A_PER_THREAD = TILE_M * (LM_MX_TILE_K / 16u) / LM_MX_THREADS;
	constexpr uint32_t B_PER_THREAD = TILE_N * (LM_MX_TILE_K / 16u) / LM_MX_THREADS;
	static_assert(STAGES == 3u && LmMxSharedBytes<TILE_M,TILE_N>() <= LM_MX_SHARED_LIMIT,"three stages plus scale scratch fit in SM shared memory");
	static_assert(WARPS_M * WARPS_N == LM_MX_WARPS && MF >= 1u && NF >= 2u && NF % 2u == 0u && WARP_M % 16u == 0u,"the tile is not a whole number of fragments per warp");
	static_assert(TILE_M * (LM_MX_TILE_K / 16u) % LM_MX_THREADS == 0u && TILE_N * (LM_MX_TILE_K / 16u) % LM_MX_THREADS == 0u,"a stage must divide across the block");
	static_assert(TILE_M + TILE_N <= LM_MX_THREADS,"scale scratch loads one word per thread");
	extern __shared__ __align__(128) uint8_t lm_mx_shared[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(lm_mx_shared);
	uint32_t *s_scratch = (uint32_t *)(lm_mx_shared + STAGES * STAGE_BYTES);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u,group = lane >> 2u,quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N,warp_n = warp % WARPS_N;
	const uint32_t scale_columns = args.input_dimension / LM_MX_BLOCK;
	const uint32_t row_tiles = (args.rows + TILE_M - 1u) / TILE_M;
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t group_tiles = row_tiles * neuron_tiles,total_tiles = group_tiles * groups.count;
	const uint32_t k_tiles = args.input_dimension / LM_MX_TILE_K;
	const uint32_t col0 = threadIdx.x & 7u,row0 = threadIdx.x >> 3u;
	const uint64_t rstride = 32ull * args.input_dimension;

	for (uint32_t whole = blockIdx.x; whole < total_tiles; whole += gridDim.x)
	{
		const uint32_t member = whole / group_tiles,tile = whole % group_tiles;
		const uint32_t band = tile / (LM_MX_BAND_ROW_TILES * neuron_tiles);
		const uint32_t band_first = band * LM_MX_BAND_ROW_TILES;
		const uint32_t band_rows = row_tiles - band_first < LM_MX_BAND_ROW_TILES ? row_tiles - band_first : LM_MX_BAND_ROW_TILES;
		const uint32_t within = tile - band * LM_MX_BAND_ROW_TILES * neuron_tiles;
		const uint32_t rn = (band & 1u) != 0u ? neuron_tiles - 1u - (within / band_rows) : (within / band_rows);
		const uint32_t row_base = (band_first + within % band_rows) * TILE_M;
		const uint32_t neuron_base = rn * TILE_N;
		const uint8_t *a_group = groups.activation[member],*w_group = groups.weight[member];
		const uint8_t *a_base = a_group + (uint64_t)(row_base + row0) * args.input_dimension + col0 * 16u;
		const uint8_t *b_base = w_group + (uint64_t)(neuron_base + row0) * args.input_dimension + col0 * 16u;
		const uint8_t *s_ptr = (const uint8_t *)0;
		uint32_t ab_mask = 0u,kp = 0u,sp_off = 0u,sc_off = 0u,i,j,e;
		float total[MF][NF][4];
		uint32_t a_reg[2][MF][4],b_reg[2][NF][2],scale_a[MF],scale_b[NF];
#pragma unroll
		for (uint32_t q = 0u; q < A_PER_THREAD; q++)
			ab_mask |= (row_base + row0 + q * 32u < args.rows ? 1u : 0u) << q;
#pragma unroll
		for (uint32_t q = 0u; q < B_PER_THREAD; q++)
			ab_mask |= (neuron_base + row0 + q * 32u < args.output_dimension ? 1u : 0u) << (q + 8u);
		if ( threadIdx.x < TILE_M )
		{
			const uint32_t src = row_base + threadIdx.x < args.rows ? row_base + threadIdx.x : args.rows - 1u;
			s_ptr = groups.activation_scale[member] + (uint64_t)src * scale_columns;
		}
		else if ( threadIdx.x < TILE_M + TILE_N )
		{
			const uint32_t src = neuron_base + threadIdx.x - TILE_M < args.output_dimension ? neuron_base + threadIdx.x - TILE_M : args.output_dimension - 1u;
			s_ptr = groups.weight_scale[member] + (uint64_t)src * scale_columns;
		}
		auto load_s4 = [&](uint32_t kt_block) -> uint4
		{
			uint4 v = make_uint4(0u,0u,0u,0u);
			if ( s_ptr != 0 )
			{
				const uint32_t k_start = kt_block * 4u;
				const uint8_t *p = s_ptr + (uint64_t)k_start * 4u;
				if ( k_start + 4u <= k_tiles && (((uintptr_t)p & 15u) == 0u) )
					v = __ldg((const uint4 *)p);
				else
				{
					if ( k_start < k_tiles ) v.x = __ldg((const uint32_t *)p);
					if ( k_start + 1u < k_tiles ) v.y = __ldg((const uint32_t *)(p + 4u));
					if ( k_start + 2u < k_tiles ) v.z = __ldg((const uint32_t *)(p + 8u));
					if ( k_start + 3u < k_tiles ) v.w = __ldg((const uint32_t *)(p + 12u));
				}
			}
			return(v);
		};
		auto issue_a = [&](uint32_t st_off)
		{
			const uint32_t st = shared_base + st_off;
#pragma unroll
			for (uint32_t q = 0u; q < A_PER_THREAD; q++)
			{
				const uint32_t valid = (ab_mask >> q) & 1u;
				LmMxCopy16(st + LmMxSwizzle(row0 + q * 32u,col0),valid != 0u ? a_base + q * rstride : a_group,valid << 4u);
			}
			a_base += LM_MX_TILE_K;
		};
		auto issue_b = [&](uint32_t st_off)
		{
			const uint32_t st = shared_base + st_off + A_BYTES;
#pragma unroll
			for (uint32_t q = 0u; q < B_PER_THREAD; q++)
			{
				const uint32_t valid = (ab_mask >> (q + 8u)) & 1u;
				LmMxCopy16(st + LmMxSwizzle(row0 + q * 32u,col0),valid != 0u ? b_base + q * rstride : w_group,valid << 4u);
			}
			b_base += LM_MX_TILE_K;
		};
		auto load_frag = [&](uint32_t buf, uint32_t step, uint32_t stage_s)
		{
#pragma unroll
			for (i = 0u; i < MF; i++)
				LmMxMatrix4(stage_s + LmMxSwizzle(warp_m * WARP_M + i * 16u + ((lane >> 3u) & 1u) * 8u + (lane & 7u),step * 2u + (lane >> 4u)),a_reg[buf][i]);
#pragma unroll
			for (j = 0u; j < NF; j += 2u)
			{
				uint32_t value[4];
				LmMxMatrix4(stage_s + A_BYTES + LmMxSwizzle(warp_n * WARP_N + j * 8u + (lane >> 4u) * 8u + (lane & 7u),step * 2u + ((lane >> 3u) & 1u)),value);
				b_reg[buf][j][0] = value[0]; b_reg[buf][j][1] = value[1];
				b_reg[buf][j + 1u][0] = value[2]; b_reg[buf][j + 1u][1] = value[3];
			}
		};
		auto advance = [](uint32_t &off) { off += STAGE_BYTES; if ( off == STAGES * STAGE_BYTES ) off = 0u; };
#pragma unroll
		for (i = 0u; i < MF; i++)
#pragma unroll
			for (j = 0u; j < NF; j++)
				total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
		uint4 s_cur = load_s4(0u);
#pragma unroll
		for (uint32_t s = 0u; s + 1u < STAGES; s++)
		{
			if ( kp < k_tiles )
			{
				issue_a(sp_off);
				issue_b(sp_off);
			}
			LmMxCommit();
			kp++;
			advance(sp_off);
		}
		if ( threadIdx.x < TILE_M + TILE_N )
			s_scratch[threadIdx.x] = s_cur.x;
		LmMxWait<STAGES - 2u>();
		__syncthreads();
#pragma unroll
		for (i = 0u; i < MF; i++)
			scale_a[i] = s_scratch[warp_m * WARP_M + i * 16u + group + ((lane & 1u) << 3u)];
#pragma unroll
		for (j = 0u; j < NF; j++)
			scale_b[j] = s_scratch[TILE_M + warp_n * WARP_N + j * 8u + group];
		load_frag(0u,0u,shared_base + sc_off);
		for (uint32_t kt = 0u; kt < k_tiles; kt++)
		{
			const uint32_t stage_s = shared_base + sc_off,next_kt = kt + 1u,next_sub = next_kt & 3u,ns_off = (next_kt & 1u) * (TILE_M + TILE_N);
			if ( next_kt < k_tiles && next_sub == 0u )
				s_cur = load_s4(next_kt >> 2u);
			if ( kp < k_tiles )
				issue_a(sp_off);
#pragma unroll 1
			for (uint32_t pair = 0u; pair < 2u; pair++)
			{
				const uint32_t s0 = pair * 2u;
				load_frag(1u,s0 + 1u,stage_s);
#pragma unroll
				for (i = 0u; i < MF; i++)
#pragma unroll
					for (j = 0u; j < NF; j++)
					{
						float partial[4];
						LmMmaMxf8Block(partial,a_reg[0][i],b_reg[0][j],scale_a[i],0u,scale_b[j],0u);
#pragma unroll
						for (e = 0u; e < 4u; e++)
							total[i][j][e] += partial[e];
					}
				if ( pair == 0u )
				{
					if ( kp < k_tiles )
					{
						issue_b(sp_off);
						kp++;
						advance(sp_off);
					}
					LmMxCommit();
					load_frag(0u,2u,stage_s);
				}
				else if ( next_kt < k_tiles )
				{
					if ( threadIdx.x < TILE_M + TILE_N )
						s_scratch[ns_off + threadIdx.x] = next_sub == 0u ? s_cur.x : next_sub == 1u ? s_cur.y : next_sub == 2u ? s_cur.z : s_cur.w;
					LmMxWait<STAGES - 2u>();
					__syncthreads();
					advance(sc_off);
					load_frag(0u,0u,shared_base + sc_off);
				}
#pragma unroll
				for (i = 0u; i < MF; i++)
#pragma unroll
					for (j = 0u; j < NF; j++)
					{
						float partial[4];
						LmMmaMxf8Block(partial,a_reg[1][i],b_reg[1][j],scale_a[i],1u,scale_b[j],1u);
#pragma unroll
						for (e = 0u; e < 4u; e++)
							total[i][j][e] += partial[e];
					}
#pragma unroll
				for (i = 0u; i < MF; i++)
					scale_a[i] >>= 16u;
#pragma unroll
				for (j = 0u; j < NF; j++)
					scale_b[j] >>= 16u;
			}
			if ( next_kt < k_tiles )
			{
#pragma unroll
				for (i = 0u; i < MF; i++)
					scale_a[i] = s_scratch[ns_off + warp_m * WARP_M + i * 16u + group + ((lane & 1u) << 3u)];
#pragma unroll
				for (j = 0u; j < NF; j++)
					scale_b[j] = s_scratch[ns_off + TILE_M + warp_n * WARP_N + j * 8u + group];
			}
		}
		LmMxWait<0u>();
#pragma unroll
		for (i = 0u; i < MF; i++)
#pragma unroll
			for (j = 0u; j < NF; j++)
#pragma unroll
				for (e = 0u; e < 4u; e += 2u)
				{
					const uint32_t row = row_base + warp_m * WARP_M + i * 16u + group + (e >> 1u) * 8u;
					const uint32_t column = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( row < args.rows && column < args.output_dimension )
						*(uint32_t *)(args.output_bf16 + (uint64_t)row * args.output_row_stride + groups.output_column_offset[member] + column) =
							(uint32_t)LmFloatToBf16(total[i][j][e]) | ((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
				}
		if ( whole + gridDim.x < total_tiles )
			__syncthreads();
	}
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
static int32_t LmMxGemmLaunchGroups(const LmMxGemmArguments *args, const LmMxGemmGroups *groups, uint32_t multiprocessors, SparkGpuStream stream)
{
	static std::atomic<uint64_t> granted(0u);
	const uint32_t shared = LmMxSharedBytes<TILE_M,TILE_N>();
	const uint64_t tiles = (uint64_t)((args->rows + TILE_M - 1u) / TILE_M) * ((args->output_dimension + TILE_N - 1u) / TILE_N) * groups->count;
	int device = 0;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_MX_MAX_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( SparkGpuFuncSetAttribute((const void *)LmMxGemmKernel<TILE_M,TILE_N,WARPS_M>,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)LM_MX_SHARED_LIMIT) != SPARK_GPU_SUCCESS )
			return(LM_LAUNCH_ERR_ATTRIBUTE);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	LM_LAUNCH((LmMxGemmKernel<TILE_M,TILE_N,WARPS_M>),(uint32_t)(tiles < multiprocessors ? tiles : multiprocessors),LM_MX_THREADS,shared,stream,*args,*groups);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static LmMxGemmGroups LmMxGemmSingle(const LmMxGemmArguments *args)
{
	LmMxGemmGroups groups;
	memset(&groups,0,sizeof(groups));
	groups.activation[0] = args->activation;
	groups.activation_scale[0] = args->activation_scale;
	groups.weight[0] = args->weight;
	groups.weight_scale[0] = args->weight_scale;
	groups.output_column_offset[0] = args->output_column_offset;
	groups.count = 1u;
	return(groups);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M>
static int32_t LmMxGemmLaunchTile(const LmMxGemmArguments *args, uint32_t multiprocessors, SparkGpuStream stream)
{
	const LmMxGemmGroups groups = LmMxGemmSingle(args);
	return(LmMxGemmLaunchGroups<TILE_M,TILE_N,WARPS_M>(args,&groups,multiprocessors,stream));
}

static int32_t LmMxGemmDenseGrid(const uint8_t *activation, const uint8_t *activation_scale, const uint8_t *weight, const uint8_t *weight_scale,
	uint16_t *output_bf16, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset,
	uint32_t multiprocessors, uint32_t grid_limit, SparkGpuStream stream)
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
	if ( grid_limit == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	const uint32_t grid = grid_limit < multiprocessors ? grid_limit : UINT32_MAX;
	neuron_tiles = (output_dimension + 127u) / 128u;
	if ( (uint64_t)((rows + 127u) / 128u) * neuron_tiles >= 2u * (uint64_t)multiprocessors )
		return(LmMxGemmLaunchTile<128u,128u,2u>(&args,grid,stream));
	if ( (uint64_t)((rows + 63u) / 64u) * neuron_tiles >= (uint64_t)multiprocessors )
		return(LmMxGemmLaunchTile<64u,128u,2u>(&args,grid,stream));
	return(LmMxGemmLaunchTile<32u,128u,2u>(&args,grid,stream));
}

static int32_t LmMxGemmDense(const uint8_t *activation, const uint8_t *activation_scale, const uint8_t *weight, const uint8_t *weight_scale,
	uint16_t *output_bf16, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset,
	uint32_t multiprocessors, SparkGpuStream stream)
{
	return(LmMxGemmDenseGrid(activation,activation_scale,weight,weight_scale,output_bf16,rows,input_dimension,output_dimension,output_row_stride,
		output_column_offset,multiprocessors,multiprocessors,stream));
}

static int32_t LmMxGemmGrouped(const uint8_t *const *activation, const uint8_t *const *activation_scale, const uint8_t *const *weight,
	const uint8_t *const *weight_scale, const uint32_t *output_column_offset, uint32_t count, uint16_t *output_bf16, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t output_row_stride, uint32_t multiprocessors, SparkGpuStream stream)
{
	LmMxGemmArguments args;
	LmMxGemmGroups groups;
	uint64_t tiles;
	if ( count == 0u || count > LM_MX_MAX_GROUPS || activation == 0 || activation_scale == 0 || weight == 0 || weight_scale == 0 || output_column_offset == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&groups,0,sizeof(groups));
	for (uint32_t member = count; member-- > 0u;)
	{
		int32_t status;
		memset(&args,0,sizeof(args));
		args.activation = activation[member];
		args.activation_scale = activation_scale[member];
		args.weight = weight[member];
		args.weight_scale = weight_scale[member];
		args.output_bf16 = output_bf16;
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
		args.output_column_offset = output_column_offset[member];
		status = LmMxGemmValidate(&args,multiprocessors);
		if ( status != LM_LAUNCH_OK )
			return(status);
		groups.activation[member] = args.activation;
		groups.activation_scale[member] = args.activation_scale;
		groups.weight[member] = args.weight;
		groups.weight_scale[member] = args.weight_scale;
		groups.output_column_offset[member] = args.output_column_offset;
	}
	groups.count = count;
	tiles = (uint64_t)((output_dimension + 127u) / 128u) * count;
	if ( (uint64_t)((rows + 127u) / 128u) * tiles >= 2u * (uint64_t)multiprocessors )
		return(LmMxGemmLaunchGroups<128u,128u,2u>(&args,&groups,UINT32_MAX,stream));
	if ( (uint64_t)((rows + 63u) / 64u) * tiles >= (uint64_t)multiprocessors )
		return(LmMxGemmLaunchGroups<64u,128u,2u>(&args,&groups,UINT32_MAX,stream));
	return(LmMxGemmLaunchGroups<32u,128u,2u>(&args,&groups,UINT32_MAX,stream));
}
