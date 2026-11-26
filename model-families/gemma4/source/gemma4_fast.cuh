#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

static __device__ __forceinline__ float GemmaFastHalfWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 8u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return value;
}

static __device__ __forceinline__ float GemmaBf16Dot8Exact(const float *wf, const float *af)
{
	float a0 = fmaf(wf[0], af[0], 0.0f);
	float a1 = wf[1] * af[1];
	a0 = fmaf(wf[2], af[2], a0);
	a1 = fmaf(wf[3], af[3], a1);
	a0 = fmaf(wf[4], af[4], a0);
	a1 = fmaf(wf[5], af[5], a1);
	a0 = fmaf(wf[6], af[6], a0);
	a1 = fmaf(wf[7], af[7], a1);
	return a0 + a1;
}

template<uint32_t ACT_MODE>
static __device__ __forceinline__ uint16_t GemmaMmaActValue(float g_acc, float u_acc, float wt, float limit)
{
	const float g = SparkMathRoundBf16(g_acc);
	const float u = SparkMathRoundBf16(u_acc);
	if constexpr ( ACT_MODE == 3u )
		return LmFloatToBf16(SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(g)) * u * wt);
	else
		return LmFloatToBf16(SparkMathGateLimitedGlu(g, u, limit) * wt);
}

template<uint32_t ACT_MODE>
static __device__ __forceinline__ uint16_t GemmaDenseActValue(float g_acc, float u_acc, float limit)
{
	const float g = SparkMathRoundBf16(g_acc);
	const float u = SparkMathRoundBf16(u_acc);
	if constexpr ( ACT_MODE == 0u )
		return LmFloatToBf16(SparkMathRoundBf16(SparkMathGeluTanh(g)) * u);
	else if constexpr ( ACT_MODE == 1u )
		return LmFloatToBf16(SparkMathRoundBf16(SparkMathSilu(g)) * u);
	else
		return LmFloatToBf16(SparkMathGateLimitedGlu(g, u, limit));
}

static __device__ __forceinline__ void GemmaGroupedTileLookupGlobal(const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ prefix, uint32_t groups, uint32_t tile, uint32_t tile_m, uint32_t n_tiles,
	uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
{
	uint32_t lo = 0u, hi = groups;
	while ( lo + 1u < hi )
	{
		const uint32_t mid = (lo + hi) >> 1u;
		if ( __ldg(prefix + mid) <= tile )
			lo = mid;
		else
			hi = mid;
	}
	grp = lo;
	const uint32_t first = __ldg(group_row_offset + grp);
	row_limit = __ldg(group_row_offset + grp + 1u);
	const uint32_t within = tile - __ldg(prefix + grp);
	row_base = first + (within / n_tiles) * tile_m;
	nt = within % n_tiles;
}

__global__ __launch_bounds__(32u, 4)
static void GemmaBuildTilePrefixKernel(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ prefix,
	uint32_t groups, uint32_t tile_m, uint32_t n_tiles)
{
	const uint32_t lane = threadIdx.x & 31u;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t per_lane = (groups + 31u) / 32u, first = lane * per_lane;
	uint32_t own = 0u;
	for ( uint32_t g = first; g < first + per_lane && g < groups; ++g )
	{
		const uint32_t r = __ldg(group_row_offset + g + 1u) - __ldg(group_row_offset + g);
		own += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	uint32_t running = own;
	#pragma unroll
	for ( uint32_t offset = 1u; offset < 32u; offset <<= 1u )
	{
		const uint32_t other = __shfl_up_sync(0xffffffffu, running, offset);
		if ( lane >= offset )
			running += other;
	}
	uint32_t off = running - own;
	for ( uint32_t g = first; g < first + per_lane && g < groups; ++g )
	{
		prefix[g] = off;
		const uint32_t r = __ldg(group_row_offset + g + 1u) - __ldg(group_row_offset + g);
		off += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	if ( lane == 31u )
		prefix[groups] = running;
}

template<uint32_t TAG, class Kernel>
static int32_t GemmaDynSmemSetup(Kernel kernel, uint32_t shared_bytes, uint32_t threads, uint32_t sms, uint32_t *blocks_out)
{
	static std::atomic<uint64_t> granted(0u);
	static std::atomic<int32_t> resident(0);
	int device = 0, per_sm = 1;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( shared_bytes > 48000u &&
			SparkGpuFuncSetAttribute((const void *)kernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, (int)threads, (size_t)shared_bytes) != SPARK_GPU_SUCCESS || per_sm < 1 )
			per_sm = 1;
		resident.store(per_sm);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	if ( blocks_out != 0 )
		*blocks_out = sms * (uint32_t)resident.load();
	return LM_LAUNCH_OK;
}

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, bool USE_SW, uint32_t MIN_BLK>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32u, MIN_BLK)
static void GemmaExactDenseTileKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out_bf16, float *__restrict__ out_f32, uint32_t rows, uint32_t K, uint32_t N)
{
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t TILE_M = WARPS_M * RPG, TILE_N = WARPS_N * NPG;
	constexpr uint32_t TOTAL_A = TILE_M * 32u;
	constexpr uint32_t TOTAL_W = USE_SW ? TILE_N * 32u : 1u;
	__shared__ __align__(16) uint4 s_a[2][TILE_M][32u];
	__shared__ __align__(16) uint4 s_w[USE_SW ? 2u : 1u][USE_SW ? TILE_N : 1u][USE_SW ? 32u : 1u];
	const uint32_t n_blk = blockIdx.x * TILE_N, m_blk = blockIdx.y * TILE_M;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t wm = warp / WARPS_N, wn = warp % WARPS_N;
	const uint32_t warp_r0 = wm * RPG, warp_n0 = wn * NPG;
	const uint32_t global_r0 = m_blk + warp_r0, global_n0 = n_blk + warp_n0;
	const uint32_t row_cnt = global_r0 < rows ? (rows - global_r0 < RPG ? rows - global_r0 : RPG) : 0u;
	const uint32_t chunks = K >> 3u;
	const uint4 *w4 = (const uint4 *)weight, *a4 = (const uint4 *)act;
	LmDependentWait();
	LmDependentRelease();
	auto stage = [&](uint32_t st, uint32_t k_base)
	{
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_A; idx += THREADS )
		{
			const uint32_t tm = idx >> 5u, ac = idx & 31u, row = m_blk + tm, c = k_base + ac;
			const bool ok = row < rows && c < chunks;
			LmSkinnyCpAsync16(&s_a[st][tm][ac], ok ? (a4 + (uint64_t)row * chunks + c) : a4, ok ? 16u : 0u);
		}
		if constexpr ( USE_SW )
		{
			#pragma unroll
			for ( uint32_t idx = threadIdx.x; idx < TOTAL_W; idx += THREADS )
			{
				const uint32_t tn = idx >> 5u, wc = idx & 31u, neuron = n_blk + tn, c = k_base + wc;
				const bool ok = neuron < N && c < chunks;
				LmSkinnyCpAsync16(&s_w[st][tn][wc], ok ? (w4 + (uint64_t)neuron * chunks + c) : w4, ok ? 16u : 0u);
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	stage(0u, 0u);
	float acc[NPG][RPG];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < RPG; ++r )
			acc[n][r] = 0.0f;
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += 32u )
	{
		const uint32_t cur = (k_base >> 5u) & 1u;
		if ( k_base + 32u < chunks )
		{
			stage(cur ^ 1u, k_base + 32u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t c = k_base + sub;
		if ( row_cnt != 0u && c < chunks )
		{
			float wf[NPG][8];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
			{
				if constexpr ( USE_SW )
					LmSkinnyUnpackBf16x8(s_w[cur][warp_n0 + n][sub], wf[n]);
				else
				{
					const uint32_t neuron = global_n0 + n < N ? global_n0 + n : 0u;
					LmSkinnyUnpackBf16x8(__ldcs(w4 + (uint64_t)neuron * chunks + c), wf[n]);
				}
			}
			if ( row_cnt == RPG )
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + r][sub], af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[n], af), acc[n][r]);
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + (r < row_cnt ? r : 0u)][sub], af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[n], af), acc[n][r]);
				}
			}
		}
		if ( k_base + 32u < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
	{
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
		{
			#pragma unroll
			for ( uint32_t r = 0u; r < RPG; ++r )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], step);
			}
			if ( sub == 0u && global_n0 + n < N )
			{
				const uint32_t col = global_n0 + n;
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
					if ( r < row_cnt )
					{
						const uint64_t idx = (uint64_t)(global_r0 + r) * N + col;
						if ( out_f32 != 0 )
							out_f32[idx] = acc[n][r];
						else
							out_bf16[idx] = LmFloatToBf16(acc[n][r]);
					}
			}
		}
	}
}

template<uint32_t LANES, uint32_t RG, uint32_t NPG>
__global__ __launch_bounds__(128u, 4)
static void GemmaExactDenseSmallKKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out_bf16, float *__restrict__ out_f32, uint32_t rows, uint32_t N)
{
	constexpr uint32_t GROUPS = 128u / LANES, TILE_N = GROUPS * NPG;
	__shared__ __align__(16) uint4 s_a[RG][LANES];
	const uint32_t m_blk = blockIdx.y * RG, n_blk = blockIdx.x * TILE_N;
	const uint32_t count = m_blk < rows ? (rows - m_blk < RG ? rows - m_blk : RG) : 0u;
	const uint32_t grp = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	const uint32_t col0 = n_blk + grp * NPG;
	const uint4 *a4 = (const uint4 *)act, *w4 = (const uint4 *)weight;
	LmDependentWait();
	LmDependentRelease();
	if ( count == 0u )
		return;
	for ( uint32_t idx = threadIdx.x; idx < count * LANES; idx += 128u )
	{
		const uint32_t r = idx / LANES, c = idx & (LANES - 1u);
		s_a[r][c] = __ldg(a4 + (uint64_t)(m_blk + r) * LANES + c);
	}
	__syncthreads();
	if ( col0 < N )
	{
		float wf[NPG][8];
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
		{
			const uint32_t col = col0 + n < N ? col0 + n : 0u;
			const uint4 vw = rows <= RG ? __ldcs(w4 + (uint64_t)col * LANES + sub) : __ldg(w4 + (uint64_t)col * LANES + sub);
			LmSkinnyUnpackBf16x8(vw, wf[n]);
		}
		for ( uint32_t r = 0u; r < count; ++r )
		{
			float af[8];
			LmSkinnyUnpackBf16x8(s_a[r][sub], af);
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
			{
				float dot = fmaf(1.0f, GemmaBf16Dot8Exact(wf[n], af), 0.0f);
				#pragma unroll
				for ( uint32_t step = LANES / 2u; step > 0u; step >>= 1u )
					dot += __shfl_xor_sync(0xffffffffu, dot, step);
				if ( sub == 0u && col0 + n < N )
				{
					const uint64_t idx = (uint64_t)(m_blk + r) * N + (col0 + n);
					if ( out_f32 != 0 )
						out_f32[idx] = dot;
					else
						out_bf16[idx] = LmFloatToBf16(dot);
				}
			}
		}
	}
}

template<bool DUAL, uint32_t RT, uint32_t BN>
__global__ __launch_bounds__(256u, 2)
static void GemmaExactSplitDenseV2Kernel(const uint16_t *__restrict__ w0, const uint16_t *__restrict__ w1,
	const uint16_t *__restrict__ act, uint16_t *__restrict__ out0, float *__restrict__ out0_f32,
	uint16_t *__restrict__ out1, float *__restrict__ out1_f32, uint32_t rows, uint32_t K, uint32_t n0, uint32_t n1)
{
	constexpr uint32_t SPLIT_W = 8u;
	__shared__ float partial[SPLIT_W][BN][RT];
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t n0_tiles = (n0 + BN - 1u) / BN;
	const uint32_t nt = blockIdx.x, r0 = blockIdx.y * RT;
	const uint32_t count = rows - r0 < RT ? rows - r0 : RT;
	const bool second = DUAL && nt >= n0_tiles;
	const uint32_t col0 = (second ? nt - n0_tiles : nt) * BN;
	const uint32_t cur_n = second ? n1 : n0;
	const uint16_t *w_ptr = second ? w1 : w0;
	uint16_t *o_ptr = second ? out1 : out0;
	float *o_f32 = second ? out1_f32 : out0_f32;
	const uint32_t chunks = K >> 3u;
	const uint32_t first = chunks * warp / SPLIT_W, last = chunks * (warp + 1u) / SPLIT_W, seg = last - first;
	const uint4 *w_rows[BN];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		w_rows[b] = (const uint4 *)(w_ptr + (uint64_t)(col0 + b < cur_n ? col0 + b : 0u) * K) + first;
	const bool one_pass = rows <= RT;
	float acc[BN][RT];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
			acc[b][r] = 0.0f;
	LmDependentWait();
	LmDependentRelease();
	for ( uint32_t c = sub; c < seg; c += 32u )
	{
		float wf[BN][8];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN; ++b )
			LmSkinnyUnpackBf16x8(one_pass ? __ldcs(w_rows[b] + c) : __ldg(w_rows[b] + c), wf[b]);
		const uint32_t col_elem = (first + c) * 8u;
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
		{
			const uint32_t rr = r0 + (r < count ? r : 0u);
			float af[8];
			LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)rr * K + col_elem)), af);
			#pragma unroll
			for ( uint32_t b = 0u; b < BN; ++b )
				acc[b][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[b], af), acc[b][r]);
		}
	}
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
		{
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				acc[b][r] += __shfl_xor_sync(0xffffffffu, acc[b][r], step);
			if ( sub == 0u )
				partial[warp][b][r] = acc[b][r];
		}
	__syncthreads();
	if ( warp == 0u && sub < BN * RT )
	{
		const uint32_t b = sub / RT, r = sub % RT;
		if ( col0 + b < cur_n && r < count )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < SPLIT_W; ++w )
				total += partial[w][b][r];
			const uint64_t idx = (uint64_t)(r0 + r) * cur_n + (col0 + b);
			if ( o_f32 != 0 )
				o_f32[idx] = total;
			else
				o_ptr[idx] = LmFloatToBf16(total);
		}
	}
}

template<uint32_t RG, uint32_t ACT_MODE>
__global__ __launch_bounds__(128u, RG <= 8u ? 4 : 3)
static void GemmaExactGateUpActKernel(const uint16_t *__restrict__ gate_up_w, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out_inter, uint32_t rows, uint32_t K, uint32_t ffn_rows, float limit)
{
	constexpr uint32_t WARPS = 4u, THREADS = 128u;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t col = blockIdx.x * WARPS + warp;
	const uint32_t r0 = blockIdx.y * RG;
	const uint32_t count = r0 < rows ? (rows - r0 < RG ? rows - r0 : RG) : 0u;
	const uint32_t chunks = K >> 3u;
	const uint4 *w4 = (const uint4 *)gate_up_w, *a4 = (const uint4 *)act;
	LmDependentWait();
	LmDependentRelease();
	if ( count == 0u )
		return;
	if constexpr ( RG == 1u )
	{
		if ( col < ffn_rows )
		{
			const uint4 *wg = w4 + (uint64_t)col * chunks;
			const uint4 *wu = w4 + (uint64_t)(ffn_rows + col) * chunks;
			const uint4 *ar = a4 + (uint64_t)r0 * chunks;
			float ag = 0.0f, au = 0.0f;
			#pragma unroll 4
			for ( uint32_t c = sub; c < chunks; c += 32u )
			{
				float af[8];
				LmSkinnyUnpackBf16x8(__ldg(ar + c), af);
				ag = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(wg + c), af, 0.0f), ag);
				au = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(wu + c), af, 0.0f), au);
			}
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			{
				ag += __shfl_xor_sync(0xffffffffu, ag, step);
				au += __shfl_xor_sync(0xffffffffu, au, step);
			}
			if ( sub == 0u )
				out_inter[(uint64_t)r0 * ffn_rows + col] = GemmaDenseActValue<ACT_MODE>(ag, au, limit);
		}
		return;
	}
	else
	{
		__shared__ __align__(16) uint4 s_a[2][RG][32u];
		constexpr uint32_t TOTAL_A = RG * 32u;
		auto stage_a = [&](uint32_t st, uint32_t k_base)
		{
			#pragma unroll
			for ( uint32_t idx = threadIdx.x; idx < TOTAL_A; idx += THREADS )
			{
				const uint32_t tr = idx >> 5u, ac = idx & 31u, row = r0 + tr, c = k_base + ac;
				const bool ok = row < rows && c < chunks;
				LmSkinnyCpAsync16(&s_a[st][tr][ac], ok ? (a4 + (uint64_t)row * chunks + c) : a4, ok ? 16u : 0u);
			}
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		};
		stage_a(0u, 0u);
		const uint32_t safe_col = col < ffn_rows ? col : 0u;
		const uint4 *wg = w4 + (uint64_t)safe_col * chunks;
		const uint4 *wu = w4 + (uint64_t)(ffn_rows + safe_col) * chunks;
		float ag[RG], au[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
		{
			ag[r] = 0.0f;
			au[r] = 0.0f;
		}
		for ( uint32_t k_base = 0u; k_base < chunks; k_base += 32u )
		{
			const uint32_t cur = (k_base >> 5u) & 1u;
			if ( k_base + 32u < chunks )
			{
				stage_a(cur ^ 1u, k_base + 32u);
				asm volatile("cp.async.wait_group 1;\n" ::: "memory");
			}
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			const uint32_t c = k_base + sub;
			if ( col < ffn_rows && c < chunks )
			{
				float wfg[8], wfu[8];
				LmSkinnyUnpackBf16x8(__ldcs(wg + c), wfg);
				LmSkinnyUnpackBf16x8(__ldcs(wu + c), wfu);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][r < count ? r : 0u][sub], af);
					ag[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wfg, af), ag[r]);
					au[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wfu, af), au[r]);
				}
			}
			if ( k_base + 32u < chunks )
				__syncthreads();
		}
		if ( col < ffn_rows )
		{
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				{
					ag[r] += __shfl_xor_sync(0xffffffffu, ag[r], step);
					au[r] += __shfl_xor_sync(0xffffffffu, au[r], step);
				}
				if ( sub == 0u && r < count )
					out_inter[(uint64_t)(r0 + r) * ffn_rows + col] = GemmaDenseActValue<ACT_MODE>(ag[r], au[r], limit);
			}
		}
	}
}

template<bool HAS_ROUTER, bool ROUTER_F32, uint32_t RT, uint32_t ACT_MODE>
__global__ __launch_bounds__(256u, RT == 1u ? 4 : 2)
static void GemmaExactSplitGateUpRouterKernel(const uint16_t *__restrict__ gate_up_w, const uint16_t *__restrict__ router_w,
	const uint16_t *__restrict__ act, uint16_t *__restrict__ out_inter, uint16_t *__restrict__ router_bf16,
	float *__restrict__ router_f32, uint32_t rows, uint32_t K, uint32_t ffn_rows, uint32_t experts, float limit)
{
	constexpr uint32_t SPLIT_W = 8u;
	constexpr uint32_t BN_GU = (RT == 1u || RT > 8u) ? 1u : 2u;
	constexpr uint32_t BN_RT = (RT == 1u || RT > 8u) ? 2u : 4u;
	__shared__ float partial[SPLIT_W][BN_RT][RT];
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t gu_blocks = (ffn_rows + BN_GU - 1u) / BN_GU;
	const uint32_t nt = blockIdx.x, r0 = blockIdx.y * RT;
	const uint32_t count = rows - r0 < RT ? rows - r0 : RT;
	const bool is_router = HAS_ROUTER && nt >= gu_blocks;
	const uint32_t chunks = K >> 3u;
	const uint32_t first = chunks * warp / SPLIT_W, last = chunks * (warp + 1u) / SPLIT_W, seg = last - first;
	const bool one_pass = rows <= RT;
	LmDependentWait();
	LmDependentRelease();
	if ( !is_router )
	{
		const uint32_t col0 = nt * BN_GU;
		const uint4 *wg[BN_GU], *wu[BN_GU];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_GU; ++b )
		{
			const uint32_t cc = col0 + b < ffn_rows ? col0 + b : 0u;
			wg[b] = (const uint4 *)(gate_up_w + (uint64_t)cc * K) + first;
			wu[b] = (const uint4 *)(gate_up_w + (uint64_t)(ffn_rows + cc) * K) + first;
		}
		float ag[BN_GU][RT], au[BN_GU][RT];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_GU; ++b )
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				ag[b][r] = 0.0f;
				au[b][r] = 0.0f;
			}
		for ( uint32_t c = sub; c < seg; c += 32u )
		{
			float wfg[BN_GU][8], wfu[BN_GU][8];
			#pragma unroll
			for ( uint32_t b = 0u; b < BN_GU; ++b )
			{
				LmSkinnyUnpackBf16x8(one_pass ? __ldcs(wg[b] + c) : __ldg(wg[b] + c), wfg[b]);
				LmSkinnyUnpackBf16x8(one_pass ? __ldcs(wu[b] + c) : __ldg(wu[b] + c), wfu[b]);
			}
			const uint32_t col_elem = (first + c) * 8u;
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				const uint32_t rr = r0 + (r < count ? r : 0u);
				float af[8];
				LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)rr * K + col_elem)), af);
				#pragma unroll
				for ( uint32_t b = 0u; b < BN_GU; ++b )
				{
					ag[b][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wfg[b], af), ag[b][r]);
					au[b][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wfu[b], af), au[b][r]);
				}
			}
		}
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_GU; ++b )
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				{
					ag[b][r] += __shfl_xor_sync(0xffffffffu, ag[b][r], step);
					au[b][r] += __shfl_xor_sync(0xffffffffu, au[b][r], step);
				}
				if ( sub == 0u )
				{
					partial[warp][2u * b][r] = ag[b][r];
					partial[warp][2u * b + 1u][r] = au[b][r];
				}
			}
		__syncthreads();
		if ( warp == 0u && sub < BN_GU * RT )
		{
			const uint32_t b = sub / RT, r = sub % RT;
			if ( col0 + b < ffn_rows && r < count )
			{
				float tg = 0.0f, tu = 0.0f;
				#pragma unroll
				for ( uint32_t w = 0u; w < SPLIT_W; ++w )
				{
					tg += partial[w][2u * b][r];
					tu += partial[w][2u * b + 1u][r];
				}
				out_inter[(uint64_t)(r0 + r) * ffn_rows + (col0 + b)] = GemmaDenseActValue<ACT_MODE>(tg, tu, limit);
			}
		}
	}
	else if constexpr ( HAS_ROUTER )
	{
		const uint32_t col0 = (nt - gu_blocks) * BN_RT;
		const uint4 *wr[BN_RT];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_RT; ++b )
			wr[b] = (const uint4 *)(router_w + (uint64_t)(col0 + b < experts ? col0 + b : 0u) * K) + first;
		float acc[BN_RT][RT];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_RT; ++b )
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
				acc[b][r] = 0.0f;
		for ( uint32_t c = sub; c < seg; c += 32u )
		{
			float wf[BN_RT][8];
			#pragma unroll
			for ( uint32_t b = 0u; b < BN_RT; ++b )
				LmSkinnyUnpackBf16x8(one_pass ? __ldcs(wr[b] + c) : __ldg(wr[b] + c), wf[b]);
			const uint32_t col_elem = (first + c) * 8u;
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				const uint32_t rr = r0 + (r < count ? r : 0u);
				float af[8];
				LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)rr * K + col_elem)), af);
				#pragma unroll
				for ( uint32_t b = 0u; b < BN_RT; ++b )
					acc[b][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[b], af), acc[b][r]);
			}
		}
		#pragma unroll
		for ( uint32_t b = 0u; b < BN_RT; ++b )
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					acc[b][r] += __shfl_xor_sync(0xffffffffu, acc[b][r], step);
				if ( sub == 0u )
					partial[warp][b][r] = acc[b][r];
			}
		__syncthreads();
		if ( warp == 0u && sub < BN_RT * RT )
		{
			const uint32_t b = sub / RT, r = sub % RT;
			if ( col0 + b < experts && r < count )
			{
				float total = 0.0f;
				#pragma unroll
				for ( uint32_t w = 0u; w < SPLIT_W; ++w )
					total += partial[w][b][r];
				const uint64_t idx = (uint64_t)(r0 + r) * experts + (col0 + b);
				if constexpr ( ROUTER_F32 )
					router_f32[idx] = total;
				else
					router_bf16[idx] = LmFloatToBf16(total);
			}
		}
	}
}

template<uint32_t RG, bool HEAD_GATE>
__global__ __launch_bounds__(128u, RG <= 8u ? 4 : 3)
static void GemmaExactGateOutKernel(const uint16_t *__restrict__ merged, const uint16_t *__restrict__ gate_logits,
	const uint16_t *__restrict__ out_weight, uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t N,
	uint32_t head_dim, uint32_t gate_stride)
{
	constexpr uint32_t WARPS = 4u, NPG = RG <= 8u ? 4u : 2u, TILE_N = WARPS * NPG;
	extern __shared__ __align__(16) uint4 gm_gate_smem[];
	__shared__ float s_hg[HEAD_GATE ? RG : 1u][HEAD_GATE ? 8u : 1u];
	const uint32_t r0 = blockIdx.y * RG;
	const uint32_t count = r0 < rows ? (rows - r0 < RG ? rows - r0 : RG) : 0u;
	const uint32_t chunks = K >> 3u;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t col0 = blockIdx.x * TILE_N + warp * NPG;
	LmDependentWait();
	LmDependentRelease();
	if ( count == 0u )
		return;
	if constexpr ( HEAD_GATE )
	{
		const uint32_t num_heads = K / head_dim;
		if ( threadIdx.x < count * num_heads )
		{
			const uint32_t hr = threadIdx.x / num_heads, hh = threadIdx.x % num_heads;
			s_hg[hr][hh] = SparkMathSoftplusGate(LmBf16ToFloat(__ldg(gate_logits + (uint64_t)(r0 + hr) * gate_stride + hh)));
		}
		__syncthreads();
	}
	const uint4 *m4 = (const uint4 *)merged;
	const uint4 *g4 = (const uint4 *)gate_logits;
	for ( uint32_t idx = threadIdx.x; idx < count * chunks; idx += 128u )
	{
		const uint32_t r = idx / chunks, c = idx % chunks;
		float v[8];
		LmLoadBf16x8AsFloat(__ldg(m4 + (uint64_t)(r0 + r) * chunks + c), v);
		if constexpr ( HEAD_GATE )
		{
			const uint32_t head = (c << 3u) / head_dim;
			const float g = s_hg[r][head];
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				v[k] *= g;
		}
		else
		{
			float g[8];
			LmLoadBf16x8AsFloat(__ldg(g4 + (uint64_t)(r0 + r) * chunks + c), g);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				v[k] = v[k] * SparkMathRoundBf16(SparkMathSigmoid(g[k]));
		}
		gm_gate_smem[r * chunks + c] = LmPackUint4FloatToBf16(v);
	}
	__syncthreads();
	if ( col0 < N )
	{
		const uint4 *w4 = (const uint4 *)out_weight;
		float acc[NPG][RG];
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[n][r] = 0.0f;
		for ( uint32_t c = sub; c < chunks; c += 32u )
		{
			float wf[NPG][8];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
			{
				const uint32_t col = col0 + n < N ? col0 + n : 0u;
				LmSkinnyUnpackBf16x8(__ldcs(w4 + (uint64_t)col * chunks + c), wf[n]);
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(gm_gate_smem[r * chunks + c], af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[n], af), acc[n][r]);
				}
		}
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
		{
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], step);
			}
			if ( sub == 0u && col0 + n < N )
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
						output[(uint64_t)(r0 + r) * N + (col0 + n)] = LmFloatToBf16(acc[n][r]);
		}
	}
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t ACT_MODE, bool HAS_ROUTER, bool ROUTER_F32>
__global__ __launch_bounds__(THREADS, 1)
static void GemmaDenseGateUpActMmaKernel(const uint16_t *__restrict__ gate_up_w, const uint16_t *__restrict__ router_w,
	const uint16_t *__restrict__ act, uint16_t *__restrict__ out_inter, uint16_t *__restrict__ router_bf16,
	float *__restrict__ router_f32, uint32_t rows, uint32_t K, uint32_t ffn_rows, uint32_t experts, float limit)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + THREADS - 1u) / THREADS;
	constexpr uint32_t B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS, ROW_STEP = THREADS / ROW_CHUNKS;
	static_assert(WARPS_M * WARPS_N == WARPS && (TILE_N * ROW_CHUNKS) % THREADS == 0u, "valid dense gate-up MMA tile");
	extern __shared__ __align__(128) uint8_t gm_dgu_smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(gm_dgu_smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t gu_n_tiles = (ffn_rows + TILE_N - 1u) / TILE_N;
	const uint32_t nt = blockIdx.x, row_base = blockIdx.y * TILE_M;
	const bool is_router = HAS_ROUTER && nt >= gu_n_tiles;
	const uint32_t col_base = (is_router ? nt - gu_n_tiles : nt) * TILE_N;
	const uint32_t cur_n = is_router ? experts : ffn_rows;
	LmDependentWait();
	LmDependentRelease();
	const uint16_t *a_src[A_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * THREADS;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = chunk < A_CHUNKS && row_base + r < rows ? 1u : 0u;
		a_src[q] = act + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
		b_mask |= (col_base + row0 + q * ROW_STEP < cur_n ? 1u : 0u) << q;
	const uint64_t b_step = (uint64_t)ROW_STEP * K;
	const uint16_t *bg_base = is_router ? (router_w + (uint64_t)(col_base + row0) * K + col0 * 8u) :
		(gate_up_w + (uint64_t)(col_base + row0) * K + col0 * 8u);
	const uint16_t *bu_base = is_router ? bg_base : (gate_up_w + (uint64_t)(ffn_rows + col_base + row0) * K + col0 * 8u);
	const uint16_t *w_safe = is_router ? router_w : gate_up_w;
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS;
			if ( chunk < A_CHUNKS )
			{
				const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
				const uint32_t valid = (a_mask >> q) & 1u;
				LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], valid << 4u);
				a_src[q] += TILE_K;
			}
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
			const uint32_t valid = (b_mask >> q) & 1u;
			LmStreamCopy(stage + A_BYTES + sw, valid != 0u ? bg_base + q * b_step : w_safe, valid << 4u);
			if ( !is_router )
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, valid != 0u ? bu_base + q * b_step : w_safe, valid << 4u);
		}
		bg_base += TILE_K;
		if ( !is_router )
			bu_base += TILE_K;
		LmStreamCommit();
	};
	uint32_t a_reg[2][MF][4], bg_reg[2][NF][2], bu_reg[2][NF][2];
	float tg[MF][NF][4], tu[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
				tg[i][j][e] = tu[i][j][e] = 0.0f;
	auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
	{
		#pragma unroll
		for ( uint32_t ii = 0u; ii < MF; ++ii )
		{
			const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
			const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
		}
		if constexpr ( NF == 1u )
		{
			const uint32_t r = warp_n * WARP_N + (lane & 7u);
			const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
			const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
			LmStreamLdMatrix2(bg_reg[buf][0], stage_s + A_BYTES + off);
			if ( !is_router )
				LmStreamLdMatrix2(bu_reg[buf][0], stage_s + A_BYTES + B_BYTES + off);
		}
		else
		{
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + off);
				bg_reg[buf][jj][0] = tmp[0]; bg_reg[buf][jj][1] = tmp[1];
				bg_reg[buf][jj + 1u][0] = tmp[2]; bg_reg[buf][jj + 1u][1] = tmp[3];
				if ( !is_router )
				{
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
					bu_reg[buf][jj][0] = tmp[0]; bu_reg[buf][jj][1] = tmp[1];
					bu_reg[buf][jj + 1u][0] = tmp[2]; bu_reg[buf][jj + 1u][1] = tmp[3];
				}
			}
		}
	};
	uint32_t sp_off = 0u, sc_off = 0u;
	issue_k(sp_off);
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_s = shared_base + sc_off;
		if ( kt + 1u < k_tiles )
		{
			issue_k(sp_off);
			sp_off ^= STAGE_BYTES;
		}
		load_frag(0u, 0u, stage_s);
		#pragma unroll
		for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
		{
			const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
			if ( ks + 1u < KSTEPS )
				load_frag(nxt, ks + 1u, stage_s);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
					if ( !is_router )
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
				}
		}
		sc_off ^= STAGE_BYTES;
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
				const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( r < rows && c < cur_n )
				{
					if ( !is_router )
					{
						const uint16_t lo = GemmaDenseActValue<ACT_MODE>(tg[i][j][e], tu[i][j][e], limit);
						const uint16_t hi = GemmaDenseActValue<ACT_MODE>(tg[i][j][e + 1u], tu[i][j][e + 1u], limit);
						*(uint32_t *)(out_inter + (uint64_t)r * ffn_rows + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
					else if constexpr ( ROUTER_F32 )
						*(float2 *)(router_f32 + (uint64_t)r * experts + c) = make_float2(tg[i][j][e], tg[i][j][e + 1u]);
					else
						*(uint32_t *)(router_bf16 + (uint64_t)r * experts + c) = LmPackBf16Pair(tg[i][j][e], tg[i][j][e + 1u]);
				}
			}
}

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t ACT_MODE, bool LDCS_B = true>
__global__ __launch_bounds__(256u, 1)
static void GemmaMoeMmaW1ActLdcsKernel(const uint16_t *__restrict__ w_gate, const uint16_t *__restrict__ w_up,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ tile_prefix, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ out,
	uint32_t groups, uint32_t K, uint32_t N, float limit)
{
	constexpr uint32_t TILE_N = 64u, TILE_K = 64u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u;
	constexpr uint32_t B_PER_T = (TILE_N * ROW_CHUNKS) / 256u, ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K, n_tiles = N / TILE_N;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t total_tiles = __ldg(tile_prefix + groups);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		GemmaGroupedTileLookupGlobal(group_row_offset, tile_prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			const uint32_t packed = row_base + r;
			const uint32_t valid = chunk < A_CHUNKS && packed < row_limit ? 1u : 0u;
			const uint32_t sr = valid != 0u ? __ldg(source_row_map + packed) : 0u;
			a_src[q] = act + (uint64_t)sr * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint64_t g_off = ((uint64_t)grp * N + col_base + row0) * K + col0 * 8u;
		const uint16_t *bg_base = w_gate + g_off, *bu_base = w_up + g_off;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		uint4 bg_w[LDCS_B ? B_PER_T : 1u], bu_w[LDCS_B ? B_PER_T : 1u];
		auto issue_a = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * 256u;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
					const uint32_t valid = (a_mask >> q) & 1u;
					LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], valid << 4u);
					a_src[q] += TILE_K;
				}
			}
			if constexpr ( !LDCS_B )
			{
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					const uint32_t r = row0 + q * ROW_STEP;
					const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
					LmStreamCopy(stage + A_BYTES + sw, bg_base + q * b_step, 16u);
					LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bu_base + q * b_step, 16u);
				}
				bg_base += TILE_K;
				bu_base += TILE_K;
			}
			LmStreamCommit();
		};
		auto prefetch_b = [&]()
		{
			if constexpr ( LDCS_B )
			{
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					bg_w[q] = __ldcs((const uint4 *)(bg_base + q * b_step));
					bu_w[q] = __ldcs((const uint4 *)(bu_base + q * b_step));
				}
				bg_base += TILE_K;
				bu_base += TILE_K;
			}
		};
		auto store_b = [&](uint32_t st_off)
		{
			if constexpr ( LDCS_B )
			{
				uint8_t *dst = smem + st_off;
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					const uint32_t r = row0 + q * ROW_STEP;
					const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
					*(uint4 *)(dst + A_BYTES + sw) = bg_w[q];
					*(uint4 *)(dst + A_BYTES + B_BYTES + sw) = bu_w[q];
				}
			}
		};
		uint32_t a_reg[2][MF][4], bg_reg[2][NF][2], bu_reg[2][NF][2];
		float tg[MF][NF][4], tu[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tg[i][j][e] = tu[i][j][e] = 0.0f;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			if constexpr ( NF == 1u )
			{
				const uint32_t r = warp_n * WARP_N + (lane & 7u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
				LmStreamLdMatrix2(bg_reg[buf][0], stage_s + A_BYTES + off);
				LmStreamLdMatrix2(bu_reg[buf][0], stage_s + A_BYTES + B_BYTES + off);
			}
			else
			{
				#pragma unroll
				for ( uint32_t jj = 0u; jj < NF; jj += 2u )
				{
					const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
					const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
					const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
					uint32_t tmp[4];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + off);
					bg_reg[buf][jj][0] = tmp[0]; bg_reg[buf][jj][1] = tmp[1];
					bg_reg[buf][jj + 1u][0] = tmp[2]; bg_reg[buf][jj + 1u][1] = tmp[3];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
					bu_reg[buf][jj][0] = tmp[0]; bu_reg[buf][jj][1] = tmp[1];
					bu_reg[buf][jj + 1u][0] = tmp[2]; bu_reg[buf][jj + 1u][1] = tmp[3];
				}
			}
		};
		uint32_t sp_off = 0u, sc_off = 0u;
		issue_a(sp_off);
		prefetch_b();
		store_b(sp_off);
		sp_off ^= STAGE_BYTES;
		for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
		{
			LmStreamWait<0u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + sc_off;
			if ( kt + 1u < k_tiles )
			{
				issue_a(sp_off);
				prefetch_b();
			}
			load_frag(0u, 0u, stage_s);
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
				if ( ks + 1u < 4u )
					load_frag(nxt, ks + 1u, stage_s);
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
			}
			if ( kt + 1u < k_tiles )
			{
				store_b(sp_off);
				sp_off ^= STAGE_BYTES;
			}
			sc_off ^= STAGE_BYTES;
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit )
					{
						const float wt = __ldg(packed_weight + r);
						const uint16_t lo = GemmaMmaActValue<ACT_MODE>(tg[i][j][e], tu[i][j][e], wt, limit);
						const uint16_t hi = GemmaMmaActValue<ACT_MODE>(tg[i][j][e + 1u], tu[i][j][e + 1u], wt, limit);
						*(uint32_t *)(out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
}

template<uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(256u, 1)
static void GemmaMoeMmaW2K128Kernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	uint16_t *__restrict__ out_bf16, uint32_t groups, uint32_t N)
{
	constexpr uint32_t K = 128u, TILE_N = 128u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u;
	constexpr uint32_t B_PER_T = (TILE_N * ROW_CHUNKS) / 256u, ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t n_tiles = N / TILE_N;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t total_tiles = __ldg(tile_prefix + groups);
	if ( blockIdx.x >= total_tiles )
		return;
	auto issue_half = [&](uint32_t grp, uint32_t rb, uint32_t rl, uint32_t nt, uint32_t k_off, uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off, col_base = nt * TILE_N;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			if ( chunk < A_CHUNKS )
			{
				const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
				const uint32_t packed = rb + r, valid = packed < rl ? 1u : 0u;
				LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u,
					valid != 0u ? act + (uint64_t)packed * K + k_off + c * 8u : act, valid << 4u);
			}
		}
		const uint16_t *b_ptr = w2 + ((uint64_t)grp * N + col_base + row0) * K + k_off + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, b_ptr + q * b_step, 16u);
		}
		LmStreamCommit();
	};
	uint32_t cur_grp, cur_rb, cur_rl, cur_nt;
	GemmaGroupedTileLookupGlobal(group_row_offset, tile_prefix, groups, blockIdx.x, TILE_M, n_tiles, cur_grp, cur_rb, cur_rl, cur_nt);
	issue_half(cur_grp, cur_rb, cur_rl, cur_nt, 0u, 0u);
	uint32_t a_reg[2][MF][4], b_reg[2][NF][2];
	auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
	{
		#pragma unroll
		for ( uint32_t ii = 0u; ii < MF; ++ii )
		{
			const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
			const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
		}
		#pragma unroll
		for ( uint32_t jj = 0u; jj < NF; jj += 2u )
		{
			const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
			const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
			uint32_t tmp[4];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
			b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
		}
	};
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		LmStreamWait<0u>();
		__syncthreads();
		issue_half(cur_grp, cur_rb, cur_rl, cur_nt, 64u, STAGE_BYTES);
		load_frag(0u, 0u, shared_base);
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
			if ( ks + 1u < 4u )
				load_frag(nxt, ks + 1u, shared_base);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
		}
		uint32_t nxt_grp = 0u, nxt_rb = 0u, nxt_rl = 0u, nxt_nt = 0u;
		const uint32_t has_next = tile + gridDim.x < total_tiles ? 1u : 0u;
		LmStreamWait<0u>();
		__syncthreads();
		if ( has_next != 0u )
		{
			GemmaGroupedTileLookupGlobal(group_row_offset, tile_prefix, groups, tile + gridDim.x, TILE_M, n_tiles, nxt_grp, nxt_rb, nxt_rl, nxt_nt);
			issue_half(nxt_grp, nxt_rb, nxt_rl, nxt_nt, 0u, 0u);
		}
		load_frag(0u, 0u, shared_base + STAGE_BYTES);
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
			if ( ks + 1u < 4u )
				load_frag(nxt, ks + 1u, shared_base + STAGE_BYTES);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
		}
		const uint32_t col_base = cur_nt * TILE_N;
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = cur_rb + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < cur_rl )
						__stcs((uint32_t *)(out_bf16 + (uint64_t)r * N + c), LmPackBf16Pair(acc[i][j][e], acc[i][j][e + 1u]));
				}
		cur_grp = nxt_grp;
		cur_rb = nxt_rb;
		cur_rl = nxt_rl;
		cur_nt = nxt_nt;
	}
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t THREADS = 256u, uint32_t STAGES = 2u>
__global__ __launch_bounds__(THREADS, STAGES >= 3u ? 2 : 1)
static void GemmaMoeMmaW2GeneralKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	uint16_t *__restrict__ out_bf16, uint32_t groups, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_K = 64u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + THREADS - 1u) / THREADS;
	constexpr uint32_t B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS, ROW_STEP = THREADS / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K, n_tiles = N / TILE_N;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t total_tiles = __ldg(tile_prefix + groups);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		GemmaGroupedTileLookupGlobal(group_row_offset, tile_prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			const uint32_t packed = row_base + r, valid = chunk < A_CHUNKS && packed < row_limit ? 1u : 0u;
			a_src[q] = act + (uint64_t)(valid != 0u ? packed : 0u) * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint16_t *b_base = w2 + ((uint64_t)grp * N + col_base + row0) * K + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * THREADS;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
					const uint32_t valid = (a_mask >> q) & 1u;
					LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], valid << 4u);
					a_src[q] += TILE_K;
				}
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, b_base + q * b_step, 16u);
			}
			b_base += TILE_K;
			LmStreamCommit();
		};
		auto advance = [](uint32_t &off)
		{
			if constexpr ( STAGES == 2u )
				off ^= STAGE_BYTES;
			else
			{
				off += STAGE_BYTES;
				if ( off == STAGES * STAGE_BYTES )
					off = 0u;
			}
		};
		uint32_t a_reg[2][MF][4], b_reg[2][NF][2];
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
				b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
				b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
			}
		};
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < k_tiles )
				issue_k(sp_off);
			else
				LmStreamCommit();
			kp++;
			advance(sp_off);
		}
		if constexpr ( STAGES >= 3u )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			load_frag(0u, 0u, shared_base + sc_off);
			for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
			{
				const uint32_t stage_s = shared_base + sc_off;
				#pragma unroll
				for ( uint32_t ks = 0u; ks < 4u; ++ks )
				{
					const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
					if ( ks == 0u )
					{
						if ( kp < k_tiles )
						{
							issue_k(sp_off);
							kp++;
							advance(sp_off);
						}
						else
							LmStreamCommit();
					}
					if ( ks + 1u < 4u )
						load_frag(nxt, ks + 1u, stage_s);
					else if ( kt + 1u < k_tiles )
					{
						LmStreamWait<STAGES - 2u>();
						__syncthreads();
						advance(sc_off);
						load_frag(nxt, 0u, shared_base + sc_off);
					}
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
						#pragma unroll
						for ( uint32_t j = 0u; j < NF; ++j )
							LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
			}
		}
		else
		{
			for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
			{
				LmStreamWait<0u>();
				__syncthreads();
				const uint32_t stage_s = shared_base + sc_off;
				if ( kp < k_tiles )
				{
					issue_k(sp_off);
					kp++;
					advance(sp_off);
				}
				load_frag(0u, 0u, stage_s);
				#pragma unroll
				for ( uint32_t ks = 0u; ks < 4u; ++ks )
				{
					const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
					if ( ks + 1u < 4u )
						load_frag(nxt, ks + 1u, stage_s);
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
						#pragma unroll
						for ( uint32_t j = 0u; j < NF; ++j )
							LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
				advance(sc_off);
			}
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit )
						__stcs((uint32_t *)(out_bf16 + (uint64_t)r * N + c), LmPackBf16Pair(acc[i][j][e], acc[i][j][e + 1u]));
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
}

template<uint32_t R, uint32_t K_V, uint32_t N, bool GELU>
static __device__ __forceinline__ void GemmaMoeExactW1W3Pass(const uint4 *w1_0, const uint4 *w3_0, const uint4 *w1_1, const uint4 *w3_1,
	bool has1, const uint4 *act4, const float *packed_weight, const uint32_t *route_source_token, uint16_t *out,
	uint32_t r0, uint32_t count, uint32_t col0, uint32_t lane, float limit)
{
	const uint4 *a_row[R];
	#pragma unroll
	for ( uint32_t r = 0u; r < R; ++r )
		a_row[r] = act4 + (uint64_t)__ldg(route_source_token + r0 + (r < count ? r : 0u)) * K_V;
	float g0[R], u0[R], g1[R], u1[R];
	#pragma unroll
	for ( uint32_t r = 0u; r < R; ++r )
	{
		g0[r] = 0.0f;
		u0[r] = 0.0f;
		g1[r] = 0.0f;
		u1[r] = 0.0f;
	}
	if constexpr ( R == 1u )
	{
		#pragma unroll 4
		for ( uint32_t c = lane; c < K_V; c += 32u )
		{
			float af[8];
			LmSkinnyUnpackBf16x8(__ldg(a_row[0] + c), af);
			g0[0] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(w1_0 + c), af, 0.0f), g0[0]);
			u0[0] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(w3_0 + c), af, 0.0f), u0[0]);
			g1[0] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(w1_1 + c), af, 0.0f), g1[0]);
			u1[0] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(__ldcs(w3_1 + c), af, 0.0f), u1[0]);
		}
	}
	else
	{
		#pragma unroll 2
		for ( uint32_t c = lane; c < K_V; c += 32u )
		{
			float wf1_0[8], wf3_0[8], wf1_1[8], wf3_1[8];
			LmSkinnyUnpackBf16x8(__ldcs(w1_0 + c), wf1_0);
			LmSkinnyUnpackBf16x8(__ldcs(w3_0 + c), wf3_0);
			LmSkinnyUnpackBf16x8(__ldcs(w1_1 + c), wf1_1);
			LmSkinnyUnpackBf16x8(__ldcs(w3_1 + c), wf3_1);
			#pragma unroll
			for ( uint32_t r = 0u; r < R; ++r )
				if ( r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg(a_row[r] + c), af);
					g0[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf1_0, af), g0[r]);
					u0[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf3_0, af), u0[r]);
					g1[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf1_1, af), g1[r]);
					u1[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf3_1, af), u1[r]);
				}
		}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < R; ++r )
		if ( r < count )
		{
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			{
				g0[r] += __shfl_down_sync(0xffffffffu, g0[r], offset);
				u0[r] += __shfl_down_sync(0xffffffffu, u0[r], offset);
				g1[r] += __shfl_down_sync(0xffffffffu, g1[r], offset);
				u1[r] += __shfl_down_sync(0xffffffffu, u1[r], offset);
			}
			if ( lane == 0u )
			{
				const float wt = __ldg(packed_weight + r0 + r);
				out[(uint64_t)(r0 + r) * N + col0] = GemmaMmaActValue<GELU ? 3u : 4u>(g0[r], u0[r], wt, limit);
				if ( has1 )
					out[(uint64_t)(r0 + r) * N + col0 + 1u] = GemmaMmaActValue<GELU ? 3u : 4u>(g1[r], u1[r], wt, limit);
			}
		}
}

template<class G, bool GELU, uint32_t MAX_R = 8u>
__global__ __launch_bounds__(128u, MAX_R <= 4u ? 4 : 3)
static void GemmaMoeExactW1W3ActKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ out, float limit)
{
	constexpr uint32_t WARPS = 4u, NPG = 2u, K = G::kHidden, N = G::kMoeInter, K_V = K >> 3u;
	const uint32_t grp = blockIdx.y, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t col0 = (blockIdx.x * WARPS + warp) * NPG;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint64_t w_off0 = ((uint64_t)grp * N + col0) * K_V;
			const uint4 *w1_0 = (const uint4 *)w1 + w_off0;
			const uint4 *w3_0 = (const uint4 *)w3 + w_off0;
			const uint4 *w1_1 = has1 ? w1_0 + K_V : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + K_V : w3_0;
			const uint4 *act4 = (const uint4 *)act;
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
			{
				const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
				if ( count == 1u )
					GemmaMoeExactW1W3Pass<1u, K_V, N, GELU>(w1_0, w3_0, w1_1, w3_1, has1, act4, packed_weight, route_source_token, out, r0, 1u, col0, lane, limit);
				else if ( count == 2u )
					GemmaMoeExactW1W3Pass<2u, K_V, N, GELU>(w1_0, w3_0, w1_1, w3_1, has1, act4, packed_weight, route_source_token, out, r0, 2u, col0, lane, limit);
				else if ( count == 3u )
					GemmaMoeExactW1W3Pass<3u, K_V, N, GELU>(w1_0, w3_0, w1_1, w3_1, has1, act4, packed_weight, route_source_token, out, r0, 3u, col0, lane, limit);
				else if ( count == 4u || MAX_R <= 4u )
					GemmaMoeExactW1W3Pass<4u, K_V, N, GELU>(w1_0, w3_0, w1_1, w3_1, has1, act4, packed_weight, route_source_token, out, r0, count, col0, lane, limit);
				else
					GemmaMoeExactW1W3Pass<MAX_R, K_V, N, GELU>(w1_0, w3_0, w1_1, w3_1, has1, act4, packed_weight, route_source_token, out, r0, count, col0, lane, limit);
			}
		}
	}
}

template<uint32_t RG, uint32_t N>
static __device__ __forceinline__ void GemmaMoeW2K128Apply(const uint4 (*s_act)[16u], uint16_t *__restrict__ out_bf16,
	uint32_t col, uint32_t r0, uint32_t count, uint32_t lane, uint4 w_bf16)
{
	if ( col >= N )
		return;
	float wf[8];
	LmSkinnyUnpackBf16x8(w_bf16, wf);
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
		if ( r < count )
		{
			float af[8];
			LmSkinnyUnpackBf16x8(s_act[r][lane], af);
			const float dot = GemmaFastHalfWarpSum(GemmaBf16Dot8Exact(wf, af));
			if ( lane == 0u )
				out_bf16[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(dot);
		}
}

template<class G, uint32_t RG, uint32_t C_ITERS>
__global__ __launch_bounds__(256u, 4)
static void GemmaMoeExactW2K128Kernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t K_V = 16u, N = G::kHidden, LANES = 16u, COLS_PER_BLK = 256u / LANES;
	__shared__ __align__(16) uint4 s_act[RG][K_V];
	const uint32_t grp = blockIdx.y, group = threadIdx.x / LANES, lane = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	LmDependentRelease();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t col_base = blockIdx.x * (COLS_PER_BLK * C_ITERS) + group, n_rows = r_end - r_begin;
		const uint4 *w4 = (const uint4 *)w2, *a4 = (const uint4 *)act;
		if ( n_rows <= RG )
		{
			if ( threadIdx.x < n_rows * K_V )
				((uint4 *)s_act)[threadIdx.x] = __ldg(a4 + (uint64_t)r_begin * K_V + threadIdx.x);
			__syncthreads();
			uint4 w_cur = col_base < N ? __ldcs(w4 + ((uint64_t)grp * N + col_base) * K_V + lane) : make_uint4(0u, 0u, 0u, 0u);
			#pragma unroll
			for ( uint32_t ci = 0u; ci < C_ITERS; ++ci )
			{
				const uint32_t col = col_base + ci * COLS_PER_BLK;
				uint4 w_nxt{};
				if ( ci + 1u < C_ITERS )
				{
					const uint32_t ncol = col + COLS_PER_BLK;
					w_nxt = ncol < N ? __ldcs(w4 + ((uint64_t)grp * N + ncol) * K_V + lane) : make_uint4(0u, 0u, 0u, 0u);
				}
				GemmaMoeW2K128Apply<RG, N>(s_act, out_bf16, col, r_begin, n_rows, lane, w_cur);
				w_cur = w_nxt;
			}
		}
		else
		{
			#pragma unroll
			for ( uint32_t ci = 0u; ci < C_ITERS; ++ci )
			{
				const uint32_t col = col_base + ci * COLS_PER_BLK;
				const uint4 w_bf16 = col < N ? __ldcs(w4 + ((uint64_t)grp * N + col) * K_V + lane) : make_uint4(0u, 0u, 0u, 0u);
				for ( uint32_t r0 = r_begin; r0 < r_end; r0 += RG )
				{
					const uint32_t count = r_end - r0 < RG ? r_end - r0 : RG;
					if ( threadIdx.x < count * K_V )
						((uint4 *)s_act)[threadIdx.x] = __ldg(a4 + (uint64_t)r0 * K_V + threadIdx.x);
					__syncthreads();
					GemmaMoeW2K128Apply<RG, N>(s_act, out_bf16, col, r0, count, lane, w_bf16);
					if ( r0 + RG < r_end || ci + 1u < C_ITERS )
						__syncthreads();
				}
			}
		}
	}
}

template<uint32_t R, uint32_t K_V, uint32_t N, bool STREAM>
static __device__ __forceinline__ void GemmaMoeW2GeneralPass(const uint4 *w0, const uint4 *w1, const uint4 *s_act,
	uint16_t *out, uint32_t r0, uint32_t count, uint32_t col0, uint32_t lane)
{
	float acc0[R], acc1[R];
	#pragma unroll
	for ( uint32_t r = 0u; r < R; ++r )
	{
		acc0[r] = 0.0f;
		acc1[r] = 0.0f;
	}
	#pragma unroll 4
	for ( uint32_t c = lane; c < K_V; c += 32u )
	{
		float wf0[8], wf1[8];
		LmSkinnyUnpackBf16x8(STREAM ? __ldcs(w0 + c) : __ldg(w0 + c), wf0);
		LmSkinnyUnpackBf16x8(STREAM ? __ldcs(w1 + c) : __ldg(w1 + c), wf1);
		#pragma unroll
		for ( uint32_t r = 0u; r < R; ++r )
			if ( r < count )
			{
				float af[8];
				LmSkinnyUnpackBf16x8(s_act[r * K_V + c], af);
				acc0[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf0, af), acc0[r]);
				acc1[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf1, af), acc1[r]);
			}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < R; ++r )
		if ( r < count )
		{
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			{
				acc0[r] += __shfl_down_sync(0xffffffffu, acc0[r], offset);
				acc1[r] += __shfl_down_sync(0xffffffffu, acc1[r], offset);
			}
			if ( lane == 0u )
				*(uint32_t *)(out + (uint64_t)(r0 + r) * N + col0) = LmPackBf16Pair(acc0[r], acc1[r]);
		}
}

template<class G, uint32_t RG>
__global__ __launch_bounds__(128u, 3)
static void GemmaMoeExactW2GeneralKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t K = G::kMoeInter, N = G::kHidden, K_V = K >> 3u;
	constexpr uint32_t WARPS = 4u, NPG = 2u;
	constexpr bool STREAM = G::kLocalExperts > 8u;
	extern __shared__ __align__(16) uint4 gm_w2_smem[];
	const uint32_t grp = blockIdx.y, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t col0 = (blockIdx.x * WARPS + warp) * NPG;
		const uint4 *a4 = (const uint4 *)act;
		const uint4 *w0 = (const uint4 *)w2 + ((uint64_t)grp * N + (col0 < N ? col0 : 0u)) * K_V;
		const uint4 *w1 = w0 + K_V;
		for ( uint32_t r0 = r_begin; r0 < r_end; r0 += RG )
		{
			const uint32_t count = r_end - r0 < RG ? r_end - r0 : RG;
			for ( uint32_t idx = threadIdx.x; idx < count * K_V; idx += 128u )
				gm_w2_smem[idx] = __ldg(a4 + (uint64_t)r0 * K_V + idx);
			__syncthreads();
			if ( col0 < N )
			{
				if ( count == 1u )
					GemmaMoeW2GeneralPass<1u, K_V, N, STREAM>(w0, w1, gm_w2_smem, out_bf16, r0, 1u, col0, lane);
				else if ( count == 2u )
					GemmaMoeW2GeneralPass<2u, K_V, N, STREAM>(w0, w1, gm_w2_smem, out_bf16, r0, 2u, col0, lane);
				else if ( count == 3u )
					GemmaMoeW2GeneralPass<3u, K_V, N, STREAM>(w0, w1, gm_w2_smem, out_bf16, r0, 3u, col0, lane);
				else if ( count == 4u )
					GemmaMoeW2GeneralPass<4u, K_V, N, STREAM>(w0, w1, gm_w2_smem, out_bf16, r0, 4u, col0, lane);
				else
					GemmaMoeW2GeneralPass<RG, K_V, N, STREAM>(w0, w1, gm_w2_smem, out_bf16, r0, count, col0, lane);
			}
			if ( r0 + RG < r_end )
				__syncthreads();
		}
	}
}
