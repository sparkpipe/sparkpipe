#pragma once

#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include <atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define LM_STREAM_THREADS 256u
#define LM_STREAM_WARPS 8u
#define LM_STREAM_SCALE_K 128u
#define LM_STREAM_BAND_ROW_TILES 8u
#define LM_STREAM_SHARED_LIMIT 101376u
#define LM_STREAM_STAGE_BUDGET 98304u
#define LM_STREAM_MAX_STAGES 4u
#define LM_STREAM_MAX_DEVICES 64u

typedef struct LmStreamGemmArguments
{
	const uint16_t *activation;
	const uint8_t *weight;
	const uint32_t *group_row_offset;
	const uint32_t *source_row_map;
	uint16_t *output_bf16;
	float *output_f32;
	LmScaleTensor scale;
	uint64_t weight_group_bytes;
	uint32_t rows,groups;
	uint32_t input_dimension,output_dimension;
	uint32_t output_row_stride,output_column_offset;
}
LmStreamGemmArguments;

static __device__ __forceinline__ uint32_t LmStreamSwizzle8(uint32_t row, uint32_t chunk)
{
	return(chunk ^ (((row & 1u) << 2u) | ((row >> 1u) & 3u)));
}

static __device__ __forceinline__ uint32_t LmStreamSwizzlePairs(uint32_t row, uint32_t chunk)
{
	return(chunk ^ ((row & 3u) << 1u));
}

static __device__ __forceinline__ uint32_t LmStreamE4m3PairBf16(uint32_t word, uint32_t selector)
{
	uint32_t placed = __byte_perm(word,0u,selector),bits,product;
	bits = ((placed >> 4u) & 0x07f007f0u) | (placed & 0x80008000u);
	asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(product) : "r"(bits), "r"(0x7b807b80u));
	return(product);
}

template<class Format>
struct LmStreamWeight
{
	static constexpr bool kSupported = false;
};

template<>
struct LmStreamWeight<LmBf16Format>
{
	static constexpr bool kSupported = true;
	static constexpr bool kScaled = false;
	static constexpr uint32_t kElementBytes = 2u;
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ uint32_t Offset(uint32_t row, uint32_t chunk)
	{
		return(row * ROW_BYTES + LmStreamSwizzle8(row,chunk) * 16u);
	}
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ void Fragment(const uint8_t *stage, uint32_t row, uint32_t step, uint32_t quad, uint32_t b[4])
	{
		uint4 value = *(const uint4 *)(stage + Offset<ROW_BYTES>(row,step * 4u + quad));
		b[0] = value.x;
		b[1] = value.y;
		b[2] = value.z;
		b[3] = value.w;
	}
};

template<>
struct LmStreamWeight<LmFp8>
{
	static constexpr bool kSupported = true;
	static constexpr bool kScaled = true;
	static constexpr uint32_t kElementBytes = 1u;
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ uint32_t Offset(uint32_t row, uint32_t chunk)
	{
		return(row * ROW_BYTES + LmStreamSwizzlePairs(row,chunk) * 16u);
	}
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ void Fragment(const uint8_t *stage, uint32_t row, uint32_t step, uint32_t quad, uint32_t b[4])
	{
		uint2 value = *(const uint2 *)(stage + Offset<ROW_BYTES>(row,step * 2u + (quad >> 1u)) + (quad & 1u) * 8u);
		b[0] = LmStreamE4m3PairBf16(value.x,0x1404u);
		b[1] = LmStreamE4m3PairBf16(value.x,0x3424u);
		b[2] = LmStreamE4m3PairBf16(value.y,0x1404u);
		b[3] = LmStreamE4m3PairBf16(value.y,0x3424u);
	}
};

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamStageBytes(void)
{
	return(TILE_M * TILE_K * 2u + TILE_N * TILE_K * LmStreamWeight<Format>::kElementBytes);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamStages(void)
{
	return(LM_STREAM_STAGE_BUDGET / LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() < LM_STREAM_MAX_STAGES ? LM_STREAM_STAGE_BUDGET / LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() : LM_STREAM_MAX_STAGES);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamSharedBytes(uint32_t groups)
{
	return(LmStreamStages<Format,TILE_M,TILE_N,TILE_K>() * LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() + (groups + 1u) * 4u);
}

static __device__ __forceinline__ void LmStreamCopy(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmStreamCommit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<uint32_t PENDING>
static __device__ __forceinline__ void LmStreamWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

typedef struct LmStreamTile
{
	uint32_t group,row_base,row_limit,neuron_base;
}
LmStreamTile;

static __device__ void LmStreamTilePrefix(const LmStreamGemmArguments &args, uint32_t *prefix, uint32_t groups, uint32_t tile_m, uint32_t neuron_tiles)
{
	uint32_t lane = threadIdx.x & 31u,per_lane,first,group,rows,own,running,other,offset;
	if ( threadIdx.x < 32u )
	{
		per_lane = (groups + 31u) / 32u;
		first = lane * per_lane;
		own = 0u;
		for (group = first; group < first + per_lane && group < groups; group++)
		{
			rows = args.group_row_offset == 0 ? args.rows : args.group_row_offset[group + 1u] - args.group_row_offset[group];
			own += (rows + tile_m - 1u) / tile_m * neuron_tiles;
		}
		running = own;
		for (offset = 1u; offset < 32u; offset <<= 1u)
		{
			other = __shfl_up_sync(0xffffffffu,running,offset);
			if ( lane >= offset )
				running += other;
		}
		offset = running - own;
		for (group = first; group < first + per_lane && group < groups; group++)
		{
			prefix[group] = offset;
			rows = args.group_row_offset == 0 ? args.rows : args.group_row_offset[group + 1u] - args.group_row_offset[group];
			offset += (rows + tile_m - 1u) / tile_m * neuron_tiles;
		}
		if ( lane == 31u )
			prefix[groups] = running;
	}
	__syncthreads();
}

static __device__ __forceinline__ LmStreamTile LmStreamTileAt(const LmStreamGemmArguments &args, const uint32_t *prefix, uint32_t groups, uint32_t tile_m, uint32_t tile_n, uint32_t neuron_tiles, uint32_t tile)
{
	LmStreamTile at;
	uint32_t low = 0u,high = groups,middle,first,row_tiles,band,band_first,band_rows,within;
	while ( low + 1u < high )
	{
		middle = (low + high) >> 1u;
		if ( prefix[middle] <= tile )
			low = middle;
		else
			high = middle;
	}
	at.group = low;
	first = args.group_row_offset == 0 ? 0u : args.group_row_offset[low];
	at.row_limit = args.group_row_offset == 0 ? args.rows : args.group_row_offset[low + 1u];
	row_tiles = (at.row_limit - first + tile_m - 1u) / tile_m;
	within = tile - prefix[low];
	band = within / (LM_STREAM_BAND_ROW_TILES * neuron_tiles);
	band_first = band * LM_STREAM_BAND_ROW_TILES;
	band_rows = row_tiles - band_first < LM_STREAM_BAND_ROW_TILES ? row_tiles - band_first : LM_STREAM_BAND_ROW_TILES;
	within -= band * LM_STREAM_BAND_ROW_TILES * neuron_tiles;
	at.row_base = first + (band_first + within % band_rows) * tile_m;
	at.neuron_base = (within / band_rows) * tile_n;
	return(at);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
__global__ __launch_bounds__(LM_STREAM_THREADS,1) void LmStreamGemmKernel(const __grid_constant__ LmStreamGemmArguments args)
{
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t STAGES = LmStreamStages<Format,TILE_M,TILE_N,TILE_K>();
	constexpr uint32_t B_ROW_BYTES = TILE_K * Weight::kElementBytes;
	constexpr uint32_t B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t SCALE_BLOCKS = Weight::kScaled ? TILE_K / LM_STREAM_SCALE_K : 1u;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u;
	constexpr uint32_t A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t WARPS_M = TILE_M <= 32u ? 1u : TILE_M / 32u;
	constexpr uint32_t WARPS_N = LM_STREAM_WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M;
	constexpr uint32_t WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u;
	constexpr uint32_t NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>();
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS;
	constexpr uint32_t A_PER_THREAD = (A_CHUNKS + LM_STREAM_THREADS - 1u) / LM_STREAM_THREADS;
	constexpr uint32_t B_CHUNKS = TILE_N * B_ROW_CHUNKS;
	constexpr uint32_t B_PER_THREAD = B_CHUNKS / LM_STREAM_THREADS;
	static_assert(Weight::kSupported,"the stream GEMM has no path for this weight format");
	static_assert(MF >= 1u && NF >= 1u && WARPS_M * WARPS_N == LM_STREAM_WARPS && WARP_M % 16u == 0u,"the tile is not a whole number of MMA fragments per warp");
	static_assert(B_CHUNKS % LM_STREAM_THREADS == 0u,"the weight stage must divide across the block");
	static_assert(STAGES >= 2u,"the tile leaves no room for a second pipeline stage");
	static_assert(!Weight::kScaled || TILE_K % LM_STREAM_SCALE_K == 0u,"a k-tile must hold whole weight scale blocks");
	extern __shared__ __align__(128) uint8_t lm_stream_shared[];
	uint32_t *prefix = (uint32_t *)(lm_stream_shared + STAGES * STAGE_BYTES);
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(lm_stream_shared);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u,quad_row = lane >> 2u,quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N,warp_n = warp % WARPS_N;
	const uint32_t groups = args.group_row_offset == 0 ? 1u : args.groups;
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = args.input_dimension / TILE_K;
	const uint64_t weight_row_bytes = (uint64_t)args.input_dimension * Weight::kElementBytes;
	float total[MF][NF][4],partial[MF][NF][4],scale[SCALE_BLOCKS][NF][2];
	const uint16_t *a_source[A_PER_THREAD];
	const uint8_t *b_source[B_PER_THREAD];
	uint32_t a_bytes[A_PER_THREAD],b_bytes[B_PER_THREAD];
	uint32_t total_tiles,own_tiles,steps,produced,consumed,kt,i,j,e;
	LmStreamTile load,work;

	LmStreamTilePrefix(args,prefix,groups,TILE_M,neuron_tiles);
	total_tiles = prefix[groups];
	own_tiles = blockIdx.x < total_tiles ? (total_tiles - blockIdx.x + gridDim.x - 1u) / gridDim.x : 0u;
	steps = own_tiles * k_tiles;
	auto locate = [&](uint32_t step)
	{
		uint32_t q,chunk,row,packed,valid,source,neuron;
		load = LmStreamTileAt(args,prefix,groups,TILE_M,TILE_N,neuron_tiles,blockIdx.x + (step / k_tiles) * gridDim.x);
		for (q = 0u; q < A_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_STREAM_THREADS;
			row = chunk / A_ROW_CHUNKS;
			packed = load.row_base + row;
			valid = chunk < A_CHUNKS && packed < load.row_limit ? 1u : 0u;
			source = valid == 0u ? 0u : (args.source_row_map != 0 ? args.source_row_map[packed] : packed);
			a_source[q] = args.activation + (uint64_t)source * args.input_dimension + (chunk % A_ROW_CHUNKS) * 8u;
			a_bytes[q] = valid != 0u ? 16u : 0u;
		}
		for (q = 0u; q < B_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_STREAM_THREADS;
			neuron = load.neuron_base + chunk / B_ROW_CHUNKS;
			valid = neuron < args.output_dimension ? 1u : 0u;
			b_source[q] = args.weight + (uint64_t)load.group * args.weight_group_bytes + (uint64_t)(valid != 0u ? neuron : 0u) * weight_row_bytes + (chunk % B_ROW_CHUNKS) * 16u;
			b_bytes[q] = valid != 0u ? 16u : 0u;
		}
	};
	auto issue = [&](uint32_t step)
	{
		uint32_t stage = shared_base + (step % STAGES) * STAGE_BYTES,k = step % k_tiles,q,chunk,row;
		if ( k == 0u )
			locate(step);
		for (q = 0u; q < A_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_STREAM_THREADS;
			if ( chunk < A_CHUNKS )
			{
				row = chunk / A_ROW_CHUNKS;
				LmStreamCopy(stage + row * A_ROW_BYTES + LmStreamSwizzle8(row,chunk % A_ROW_CHUNKS) * 16u,a_source[q] + (uint64_t)k * TILE_K,a_bytes[q]);
			}
		}
		for (q = 0u; q < B_PER_THREAD; q++)
		{
			chunk = threadIdx.x + q * LM_STREAM_THREADS;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<B_ROW_BYTES>(chunk / B_ROW_CHUNKS,chunk % B_ROW_CHUNKS),b_source[q] + (uint64_t)k * TILE_K * Weight::kElementBytes,b_bytes[q]);
		}
	};

	for (produced = 0u; produced + 1u < STAGES; produced++)
	{
		if ( produced < steps )
			issue(produced);
		LmStreamCommit();
	}
	for (consumed = 0u; consumed < steps; consumed++)
	{
		const uint8_t *stage_a,*stage_b;
		kt = consumed % k_tiles;
		if ( kt == 0u )
		{
			work = LmStreamTileAt(args,prefix,groups,TILE_M,TILE_N,neuron_tiles,blockIdx.x + (consumed / k_tiles) * gridDim.x);
			for (i = 0u; i < MF; i++)
				for (j = 0u; j < NF; j++)
					for (e = 0u; e < 4u; e++)
					{
						total[i][j][e] = 0.0f;
						partial[i][j][e] = 0.0f;
					}
		}
		if constexpr ( Weight::kScaled )
			for (uint32_t block = 0u; block < SCALE_BLOCKS; block++)
				for (j = 0u; j < NF; j++)
				{
					uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
					scale[block][j][0] = column < args.output_dimension ? LmScaleTensorLoad(&args.scale,work.group,column,kt * TILE_K + block * LM_STREAM_SCALE_K) : 0.0f;
					scale[block][j][1] = column + 1u < args.output_dimension ? LmScaleTensorLoad(&args.scale,work.group,column + 1u,kt * TILE_K + block * LM_STREAM_SCALE_K) : 0.0f;
				}
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		if ( produced < steps )
			issue(produced);
		produced++;
		LmStreamCommit();
		stage_a = lm_stream_shared + (consumed % STAGES) * STAGE_BYTES;
		stage_b = stage_a + A_BYTES;
#pragma unroll
		for (uint32_t step = 0u; step < TILE_K / 32u; step++)
		{
			uint32_t b[NF][4];
#pragma unroll
			for (j = 0u; j < NF; j++)
				Weight::template Fragment<B_ROW_BYTES>(stage_b,warp_n * WARP_N + j * 8u + quad_row,step,quad,b[j]);
#pragma unroll
			for (i = 0u; i < MF; i++)
			{
				uint32_t row = warp_m * WARP_M + i * 16u + quad_row,first[4],second[4];
				uint4 low = *(const uint4 *)(stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + quad) * 16u);
				uint4 high = *(const uint4 *)(stage_a + (row + 8u) * A_ROW_BYTES + LmStreamSwizzle8(row + 8u,step * 4u + quad) * 16u);
				first[0] = low.x; first[1] = high.x; first[2] = low.y; first[3] = high.y;
				second[0] = low.z; second[1] = high.z; second[2] = low.w; second[3] = high.w;
#pragma unroll
				for (j = 0u; j < NF; j++)
				{
					uint32_t b_first[2] = { b[j][0], b[j][1] },b_second[2] = { b[j][2], b[j][3] };
					if constexpr ( Weight::kScaled )
					{
						LmMmaBf16(partial[i][j],first,b_first);
						LmMmaBf16(partial[i][j],second,b_second);
					}
					else
					{
						LmMmaBf16(total[i][j],first,b_first);
						LmMmaBf16(total[i][j],second,b_second);
					}
				}
			}
			if constexpr ( Weight::kScaled )
			{
				if ( step % (LM_STREAM_SCALE_K / 32u) == LM_STREAM_SCALE_K / 32u - 1u )
#pragma unroll
					for (i = 0u; i < MF; i++)
#pragma unroll
						for (j = 0u; j < NF; j++)
#pragma unroll
							for (e = 0u; e < 4u; e++)
							{
								total[i][j][e] = fmaf(partial[i][j][e],scale[step / (LM_STREAM_SCALE_K / 32u)][j][e & 1u],total[i][j][e]);
								partial[i][j][e] = 0.0f;
							}
			}
		}
		if ( kt + 1u == k_tiles )
			for (i = 0u; i < MF; i++)
				for (j = 0u; j < NF; j++)
					for (e = 0u; e < 4u; e += 2u)
					{
						uint32_t row = work.row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
						uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
						uint64_t at;
						if ( row >= work.row_limit || column >= args.output_dimension )
							continue;
						at = (uint64_t)row * args.output_row_stride + args.output_column_offset + column;
						if ( args.output_f32 != 0 )
							*(float2 *)(args.output_f32 + at) = make_float2(total[i][j][e],total[i][j][e + 1u]);
						else
							*(uint32_t *)(args.output_bf16 + at) = (uint32_t)LmFloatToBf16(total[i][j][e]) | ((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
					}
	}
	LmStreamWait<0u>();
}

template<class Format>
static uint32_t LmStreamScaleFits(const LmScaleTensor *scale, uint32_t groups, uint32_t output, uint32_t input)
{
	if constexpr ( !LmStreamWeight<Format>::kScaled )
		return(scale->encoding == LM_SCALE_ENCODING_NONE ? 1u : 0u);
	else
		return(scale->encoding == LM_SCALE_ENCODING_F32 && scale->reserved == 0u && scale->data != 0 && scale->row_group_size == 1u &&
			scale->k_group_size == LM_STREAM_SCALE_K && scale->group_count >= groups &&
			scale->row_count >= output && scale->input_dimension >= input ? 1u : 0u);
}

template<class Format>
static int32_t LmStreamGemmValidate(const LmStreamGemmArguments *args, uint32_t multiprocessors)
{
	uint32_t groups;
	if ( args == 0 || args->activation == 0 || args->weight == 0 || (args->output_bf16 == 0) == (args->output_f32 == 0) || multiprocessors == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || args->input_dimension == 0u || args->output_dimension == 0u ||
		args->input_dimension % (LmStreamWeight<Format>::kScaled ? LM_STREAM_SCALE_K : 64u) != 0u ||
		(args->output_dimension & 1u) != 0u || (args->output_column_offset & 1u) != 0u || (args->output_row_stride & 1u) != 0u ||
		args->output_row_stride < args->output_column_offset + args->output_dimension )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( (((uintptr_t)args->activation | (uintptr_t)args->weight | (uintptr_t)args->weight_group_bytes) & 15u) != 0u ||
		(((uintptr_t)args->output_bf16 | (uintptr_t)args->output_f32) & 7u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->group_row_offset == 0 ? args->groups > 1u || args->source_row_map != 0 : args->groups == 0u ||
		(args->groups > 1u && args->weight_group_bytes < (uint64_t)args->output_dimension * args->input_dimension * LmStreamWeight<Format>::kElementBytes) )
		return(LM_LAUNCH_ERR_SHAPE);
	groups = args->group_row_offset == 0 ? 1u : args->groups;
	if ( (groups + 1u) * 4u > LM_STREAM_SHARED_LIMIT - LM_STREAM_STAGE_BUDGET || LmStreamScaleFits<Format>(&args->scale,groups,args->output_dimension,args->input_dimension) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static int32_t LmStreamGemmLaunchTile(const LmStreamGemmArguments *args, uint64_t tile_bound, uint32_t multiprocessors, cudaStream_t stream)
{
	static std::atomic<uint64_t> granted(0u);
	static std::atomic<int32_t> resident(0);
	const uint32_t shared = LmStreamSharedBytes<Format,TILE_M,TILE_N,TILE_K>(args->group_row_offset == 0 ? 1u : args->groups);
	uint64_t blocks;
	int device = 0,per_multiprocessor;
	if ( cudaGetDevice(&device) != cudaSuccess || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( cudaFuncSetAttribute((const void *)LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)LM_STREAM_SHARED_LIMIT) != cudaSuccess )
			return(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor,LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K>,LM_STREAM_THREADS,LM_STREAM_SHARED_LIMIT) != cudaSuccess || per_multiprocessor < 1 )
			return(LM_LAUNCH_ERR_ATTRIBUTE);
		resident.store(per_multiprocessor);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	blocks = (uint64_t)multiprocessors * (uint64_t)resident.load();
	if ( tile_bound < blocks )
		blocks = tile_bound;
	LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K><<<(uint32_t)blocks,LM_STREAM_THREADS,shared,stream>>>(*args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format, uint32_t TILE_N, uint32_t TILE_K>
static int32_t LmStreamGemmLaunchShape(const LmStreamGemmArguments *args, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t groups = args->group_row_offset == 0 ? 1u : args->groups;
	const uint32_t neuron_tiles = (args->output_dimension + TILE_N - 1u) / TILE_N;
	uint32_t mean_rows = (args->rows + groups - 1u) / groups,tile_m;
	tile_m = mean_rows > 64u && LmStreamStages<Format,128u,TILE_N,TILE_K>() >= 3u ? 128u
		: mean_rows > 32u && LmStreamStages<Format,64u,TILE_N,TILE_K>() >= 3u ? 64u
		: mean_rows > 16u ? 32u : 16u;
	while ( args->group_row_offset == 0 && tile_m > 16u && (uint64_t)((args->rows + tile_m - 1u) / tile_m) * neuron_tiles < multiprocessors )
		tile_m >>= 1u;
	const uint64_t tile_bound = ((uint64_t)args->rows / tile_m + groups) * neuron_tiles;
	switch ( tile_m )
	{
		case 16u: return(LmStreamGemmLaunchTile<Format,16u,TILE_N,TILE_K>(args,tile_bound,multiprocessors,stream));
		case 32u: return(LmStreamGemmLaunchTile<Format,32u,TILE_N,TILE_K>(args,tile_bound,multiprocessors,stream));
		case 64u:
			if constexpr ( LmStreamStages<Format,64u,TILE_N,TILE_K>() >= 3u )
				return(LmStreamGemmLaunchTile<Format,64u,TILE_N,TILE_K>(args,tile_bound,multiprocessors,stream));
			return(LM_LAUNCH_ERR_TILE);
		default:
			if constexpr ( LmStreamStages<Format,128u,TILE_N,TILE_K>() >= 3u )
				return(LmStreamGemmLaunchTile<Format,128u,TILE_N,TILE_K>(args,tile_bound,multiprocessors,stream));
			return(LM_LAUNCH_ERR_TILE);
	}
}

template<class Format>
static int32_t LmStreamGemmLaunch(const LmStreamGemmArguments *args, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status = LmStreamGemmValidate<Format>(args,multiprocessors);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if constexpr ( LmStreamWeight<Format>::kScaled )
	{
		if ( args->input_dimension % 256u == 0u )
			return(LmStreamGemmLaunchShape<Format,64u,256u>(args,multiprocessors,stream));
		return(LmStreamGemmLaunchShape<Format,128u,128u>(args,multiprocessors,stream));
	}
	else
		return(LmStreamGemmLaunchShape<Format,128u,64u>(args,multiprocessors,stream));
}

template<class Format>
static int32_t LmStreamGemmGrouped(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *source_row_map, uint32_t groups, uint32_t packed_rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmStreamGemmArguments args;
	memset(&args,0,sizeof(args));
	args.activation = activation;
	args.weight = (const uint8_t *)weight;
	args.group_row_offset = group_row_offset;
	args.source_row_map = source_row_map;
	args.output_bf16 = output_bf16;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * LmStreamWeight<Format>::kElementBytes;
	args.rows = packed_rows;
	args.groups = groups;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_dimension;
	return(LmStreamGemmLaunch<Format>(&args,multiprocessors,stream));
}

template<class Format>
static int32_t LmStreamGemmDense(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, uint32_t multiprocessors, cudaStream_t stream)
{
	LmStreamGemmArguments args;
	memset(&args,0,sizeof(args));
	args.activation = activation;
	args.weight = (const uint8_t *)weight;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * LmStreamWeight<Format>::kElementBytes;
	args.rows = rows;
	args.groups = 1u;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
	args.output_column_offset = output_column_offset;
	return(LmStreamGemmLaunch<Format>(&args,multiprocessors,stream));
}
