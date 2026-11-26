#pragma once

static __global__ __launch_bounds__(32u, 1)
void Glm5NextBuildTilePrefixKernel(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ prefix,
	uint32_t groups, uint32_t tile_m, uint32_t n_tiles)
{
	const uint32_t lane = threadIdx.x, per = (groups + 31u) / 32u, first = lane * per;
	uint32_t own = 0u;
	LmDependentWait();
	for ( uint32_t g = first; g < first + per && g < groups; ++g )
	{
		const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
		own += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	uint32_t run = own;
	#pragma unroll
	for ( uint32_t s = 1u; s < 32u; s <<= 1u )
	{
		const uint32_t other = __shfl_up_sync(0xffffffffu, run, s);
		if ( lane >= s )
			run += other;
	}
	uint32_t off = run - own;
	for ( uint32_t g = first; g < first + per && g < groups; ++g )
	{
		prefix[g] = off;
		const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
		off += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	if ( lane == 31u )
		prefix[groups] = run;
	LmDependentRelease();
}

static __device__ __forceinline__ void Glm5NextGroupedTileLookupGlobal(const uint32_t *__restrict__ group_row_offset,
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
	const uint32_t first = __ldg(group_row_offset + lo);
	row_limit = __ldg(group_row_offset + lo + 1u);
	const uint32_t within = tile - __ldg(prefix + lo);
	row_base = first + (within / n_tiles) * tile_m;
	nt = within % n_tiles;
}

static __device__ __forceinline__ void Glm5NextTilePrefixScan(const uint32_t *__restrict__ group_row_offset, uint32_t *prefix,
	uint32_t groups, uint32_t tile_m, uint32_t n_tiles)
{
	if ( threadIdx.x < 32u )
	{
		const uint32_t lane = threadIdx.x, per = (groups + 31u) / 32u, first = lane * per;
		uint32_t own = 0u;
		for ( uint32_t g = first; g < first + per && g < groups; ++g )
		{
			const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
			own += ((r + tile_m - 1u) / tile_m) * n_tiles;
		}
		uint32_t run = own;
		#pragma unroll
		for ( uint32_t s = 1u; s < 32u; s <<= 1u )
		{
			const uint32_t other = __shfl_up_sync(0xffffffffu, run, s);
			if ( lane >= s )
				run += other;
		}
		uint32_t off = run - own;
		for ( uint32_t g = first; g < first + per && g < groups; ++g )
		{
			prefix[g] = off;
			const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
			off += ((r + tile_m - 1u) / tile_m) * n_tiles;
		}
		if ( lane == 31u )
			prefix[groups] = run;
	}
	__syncthreads();
}

static __device__ __forceinline__ void Glm5NextGroupedTileLookup(const uint32_t *__restrict__ group_row_offset, const uint32_t *prefix,
	uint32_t groups, uint32_t tile, uint32_t tile_m, uint32_t n_tiles, uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
{
	uint32_t lo = 0u, hi = groups;
	while ( lo + 1u < hi )
	{
		const uint32_t mid = (lo + hi) >> 1u;
		if ( prefix[mid] <= tile )
			lo = mid;
		else
			hi = mid;
	}
	grp = lo;
	const uint32_t first = group_row_offset[lo];
	row_limit = group_row_offset[lo + 1u];
	const uint32_t within = tile - prefix[lo];
	row_base = first + (within / n_tiles) * tile_m;
	nt = within % n_tiles;
}

template<uint32_t ExpertCodec, uint32_t ROWS>
static __device__ __forceinline__ void Glm5NextMoeW1ActPass(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out,
	uint64_t exp_base, uint32_t ch, uint32_t r0, uint32_t count, uint32_t lane)
{
	constexpr uint32_t K = G5N_HIDDEN, INTER = 128u, FP8_CHUNKS = K / 16u, BF16_CHUNKS = K / 8u, SCALES = K / 128u;
	uint32_t src_r[ROWS];
	float au[ROWS], ag[ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		src_r[r] = (ROWS <= 4u || r < count) ? __ldg(route_source_token + r0 + r) : 0u;
		au[r] = 0.0f;
		ag[r] = 0.0f;
	}
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		const uint4 *w_up = (const uint4 *)(exp_w1 + (exp_base + ch) * K);
		const uint4 *w_gt = (const uint4 *)(exp_w1 + (exp_base + INTER + ch) * K);
		const float *s_up = exp_s1 + (exp_base + ch) * SCALES;
		const float *s_gt = exp_s1 + (exp_base + INTER + ch) * SCALES;
		for ( uint32_t c = lane; c < FP8_CHUNKS; c += 32u )
		{
			const uint32_t sg = c >> 3u;
			const float su = __ldg(s_up + sg), sgs = __ldg(s_gt + sg);
			if constexpr ( ROWS == 1u )
			{
				const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(act + (uint64_t)src_r[0] * K + c * 16u);
				au[0] = fmaf(su, LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_up + c), a), au[0]);
				ag[0] = fmaf(sgs, LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_gt + c), a), ag[0]);
			}
			else
			{
				const LmSkinnyFormat<LmFp8>::Decoded du = LmSkinnyFormat<LmFp8>::Decode(__ldcs(w_up + c));
				const LmSkinnyFormat<LmFp8>::Decoded dg = LmSkinnyFormat<LmFp8>::Decode(__ldcs(w_gt + c));
				#pragma unroll
				for ( uint32_t r = 0u; r < ROWS; ++r )
					if ( ROWS <= 4u || r < count )
					{
						const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(act + (uint64_t)src_r[r] * K + c * 16u);
						au[r] = fmaf(su, LmSkinnyFormat<LmFp8>::ChunkDecoded(du, a), au[r]);
						ag[r] = fmaf(sgs, LmSkinnyFormat<LmFp8>::ChunkDecoded(dg, a), ag[r]);
					}
			}
		}
	}
	else
	{
		const uint4 *w_up = (const uint4 *)((const uint16_t *)exp_w1 + (exp_base + ch) * K);
		const uint4 *w_gt = (const uint4 *)((const uint16_t *)exp_w1 + (exp_base + INTER + ch) * K);
		for ( uint32_t c = lane; c < BF16_CHUNKS; c += 32u )
		{
			const uint4 wu = __ldcs(w_up + c), wg = __ldcs(w_gt + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS; ++r )
				if ( ROWS <= 4u || r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)src_r[r] * K) + c), af);
					au[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(wu, af, 0.0f), au[r]);
					ag[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(wg, af, 0.0f), ag[r]);
				}
		}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
		if ( ROWS <= 4u || r < count )
		{
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			{
				au[r] += __shfl_xor_sync(0xffffffffu, au[r], step);
				ag[r] += __shfl_xor_sync(0xffffffffu, ag[r], step);
			}
			if ( lane == 0u )
				inter_out[(uint64_t)(r0 + r) * INTER + ch] = LmFloatToBf16(LmClampedUpGateOp{G5N_SWIGLU_LIMIT}(
					LmBf16ToFloat(LmFloatToBf16(ag[r])), LmBf16ToFloat(LmFloatToBf16(au[r]))));
		}
}

template<uint32_t ExpertCodec>
__global__ __launch_bounds__(128u, 4)
static void Glm5NextMoeExactW1ActKernel(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token,
	uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t WARPS = 4u, INTER = 128u;
	const uint32_t grp = blockIdx.y, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t ch = blockIdx.x * WARPS + warp;
		const uint64_t exp_base = (uint64_t)grp * (2u * INTER);
		for ( uint32_t r0 = r_begin; r0 < r_end; r0 += 8u )
		{
			const uint32_t count = r_end - r0 < 8u ? r_end - r0 : 8u;
			if ( count == 1u )
				Glm5NextMoeW1ActPass<ExpertCodec, 1u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 1u, lane);
			else if ( count == 2u )
				Glm5NextMoeW1ActPass<ExpertCodec, 2u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 2u, lane);
			else if ( count == 3u )
				Glm5NextMoeW1ActPass<ExpertCodec, 3u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 3u, lane);
			else if ( count == 4u )
				Glm5NextMoeW1ActPass<ExpertCodec, 4u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 4u, lane);
			else
				Glm5NextMoeW1ActPass<ExpertCodec, 8u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, count, lane);
		}
	}
	LmDependentRelease();
}

template<uint32_t ExpertCodec, uint32_t RG>
__global__ __launch_bounds__(256u, 4)
static void Glm5NextMoeExactW2Kernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ inter_in, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t INTER = 128u, N = G5N_HIDDEN;
	constexpr bool IS_FP8 = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t LANES = IS_FP8 ? 8u : 16u, COLS_PER_BLK = 256u / LANES;
	__shared__ __align__(16) uint16_t s_act[RG][INTER];
	const uint32_t grp = blockIdx.y, group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
	LmDependentWait();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t col = blockIdx.x * COLS_PER_BLK + group;
		const uint64_t w_row = (uint64_t)grp * N + col;
		if constexpr ( IS_FP8 )
		{
			const float sc = __ldg(exp_s2 + w_row);
			const LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(__ldcs((const uint4 *)(exp_w2 + w_row * INTER) + lane));
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += RG )
			{
				const uint32_t count = r_end - r0 < RG ? r_end - r0 : RG;
				const uint32_t total_v = count * (INTER / 8u);
				if ( threadIdx.x < total_v )
					((uint4 *)s_act)[threadIdx.x] = __ldg((const uint4 *)(inter_in + (uint64_t)r0 * INTER) + threadIdx.x);
				__syncthreads();
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						LmSkinnyFormat<LmFp8>::Activation a;
						a.low = ((const uint4 *)s_act[r])[2u * lane];
						a.high = ((const uint4 *)s_act[r])[2u * lane + 1u];
						float dot = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, a), 0.0f);
						dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
						dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
						dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
						if ( lane == 0u )
							out_bf16[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(dot);
					}
				if ( r0 + RG < r_end )
					__syncthreads();
			}
		}
		else
		{
			const uint4 w = __ldcs((const uint4 *)((const uint16_t *)exp_w2 + w_row * INTER) + lane);
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += RG )
			{
				const uint32_t count = r_end - r0 < RG ? r_end - r0 : RG;
				const uint32_t total_v = count * (INTER / 8u);
				if ( threadIdx.x < total_v )
					((uint4 *)s_act)[threadIdx.x] = __ldg((const uint4 *)(inter_in + (uint64_t)r0 * INTER) + threadIdx.x);
				__syncthreads();
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						float dot = fmaf(1.0f, LmSkinnyBf16Chunk(w, ((const uint4 *)s_act[r])[lane], 0.0f), 0.0f);
						dot += __shfl_xor_sync(0xffffffffu, dot, 8u);
						dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
						dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
						dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
						if ( lane == 0u )
							out_bf16[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(dot);
					}
				if ( r0 + RG < r_end )
					__syncthreads();
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t MAX_R>
__global__ __launch_bounds__(LM_SKINNY_THREADS)
static void Glm5NextSharedW1ActSmallKernel(const uint16_t *__restrict__ shared_w1, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ shared_inter_out, uint32_t rows)
{
	constexpr uint32_t K = G5N_HIDDEN, INTER = 128u, CHUNKS = K / 8u, WARPS = LM_SKINNY_SPLIT_WARPS;
	__shared__ float partial[4][MAX_R][WARPS];
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t ch0 = blockIdx.x * 2u;
	const uint32_t first = CHUNKS * warp / WARPS, last = CHUNKS * (warp + 1u) / WARPS;
	const uint4 *wu0 = (const uint4 *)(shared_w1 + (uint64_t)ch0 * K);
	const uint4 *wg0 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0) * K);
	const uint4 *wu1 = (const uint4 *)(shared_w1 + (uint64_t)(ch0 + 1u) * K);
	const uint4 *wg1 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0 + 1u) * K);
	float u0[MAX_R], g0[MAX_R], u1[MAX_R], g1[MAX_R];
	#pragma unroll
	for ( uint32_t r = 0u; r < MAX_R; ++r )
		u0[r] = g0[r] = u1[r] = g1[r] = 0.0f;
	LmDependentWait();
	for ( uint32_t c = first + lane; c < last; c += LM_WARP_LANES )
	{
		const uint4 w_u0 = __ldcs(wu0 + c), w_g0 = __ldcs(wg0 + c);
		const uint4 w_u1 = __ldcs(wu1 + c), w_g1 = __ldcs(wg1 + c);
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
			if ( r < rows )
			{
				const uint4 a = __ldg((const uint4 *)(act + (uint64_t)r * K) + c);
				u0[r] = fmaf(1.0f, LmSkinnyBf16Chunk(w_u0, a, 0.0f), u0[r]);
				g0[r] = fmaf(1.0f, LmSkinnyBf16Chunk(w_g0, a, 0.0f), g0[r]);
				u1[r] = fmaf(1.0f, LmSkinnyBf16Chunk(w_u1, a, 0.0f), u1[r]);
				g1[r] = fmaf(1.0f, LmSkinnyBf16Chunk(w_g1, a, 0.0f), g1[r]);
			}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < MAX_R; ++r )
		if ( r < rows )
		{
			#pragma unroll
			for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
			{
				u0[r] += __shfl_xor_sync(0xffffffffu, u0[r], step);
				g0[r] += __shfl_xor_sync(0xffffffffu, g0[r], step);
				u1[r] += __shfl_xor_sync(0xffffffffu, u1[r], step);
				g1[r] += __shfl_xor_sync(0xffffffffu, g1[r], step);
			}
			if ( lane == 0u )
			{
				partial[0][r][warp] = u0[r];
				partial[1][r][warp] = g0[r];
				partial[2][r][warp] = u1[r];
				partial[3][r][warp] = g1[r];
			}
		}
	__syncthreads();
	if ( threadIdx.x < rows * 2u )
	{
		const uint32_t r = threadIdx.x >> 1u, k = threadIdx.x & 1u;
		float tu = 0.0f, tg = 0.0f;
		#pragma unroll
		for ( uint32_t w = 0u; w < WARPS; ++w )
		{
			tu += partial[k * 2u][r][w];
			tg += partial[k * 2u + 1u][r][w];
		}
		shared_inter_out[(uint64_t)r * INTER + ch0 + k] = LmFloatToBf16(LmClampedUpGateOp{G5N_SWIGLU_LIMIT}(
			LmBf16ToFloat(LmFloatToBf16(tg)), LmBf16ToFloat(LmFloatToBf16(tu))));
	}
	LmDependentRelease();
}

template<uint32_t ExpertCodec>
__global__ __launch_bounds__(256u)
static void Glm5NextMoeSmallW2FinalizeKernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ shared_w2, const uint16_t *__restrict__ inter_in, const uint16_t *__restrict__ shared_inter_in,
	const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight,
	uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t INTER = 128u;
	constexpr bool IS_FP8 = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t LANES = IS_FP8 ? 8u : 16u, ROWS_PER_BLOCK = 256u / LANES;
	__shared__ __align__(16) uint16_t s_inter[G5N_TOP_K + 1u][INTER];
	__shared__ uint32_t s_expert[G5N_TOP_K];
	__shared__ uint32_t s_packed[G5N_TOP_K];
	__shared__ float s_weight[G5N_TOP_K];
	const uint32_t token = blockIdx.y, group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
	const uint32_t out_row = blockIdx.x * ROWS_PER_BLOCK + group;
	LmDependentWait();
	if ( threadIdx.x < G5N_TOP_K )
	{
		const uint64_t idx = (uint64_t)token * G5N_TOP_K + threadIdx.x;
		s_expert[threadIdx.x] = route_expert[idx];
		s_packed[threadIdx.x] = route_packed_row[idx];
		s_weight[threadIdx.x] = route_weight[idx];
	}
	__syncthreads();
	for ( uint32_t v = threadIdx.x; v < G5N_TOP_K * (INTER / 8u); v += 256u )
	{
		const uint32_t slot = v / (INTER / 8u), col = v % (INTER / 8u);
		((uint4 *)s_inter[slot])[col] = __ldg((const uint4 *)(inter_in + (uint64_t)s_packed[slot] * INTER) + col);
	}
	if ( threadIdx.x < INTER / 8u )
		((uint4 *)s_inter[G5N_TOP_K])[threadIdx.x] = __ldg((const uint4 *)(shared_inter_in + (uint64_t)token * INTER) + threadIdx.x);
	__syncthreads();
	float acc = 0.0f;
	if constexpr ( IS_FP8 )
	{
		const uint64_t row0 = (uint64_t)s_expert[0] * G5N_HIDDEN + out_row;
		uint4 w_cur = __ldcs((const uint4 *)(exp_w2 + row0 * INTER) + lane);
		float sc_cur = __ldg(exp_s2 + row0);
		for ( uint32_t slot = 0u; slot < G5N_TOP_K; ++slot )
		{
			uint4 w_next;
			float sc_next;
			if ( slot + 1u < G5N_TOP_K )
			{
				const uint64_t nr = (uint64_t)s_expert[slot + 1u] * G5N_HIDDEN + out_row;
				w_next = __ldcs((const uint4 *)(exp_w2 + nr * INTER) + lane);
				sc_next = __ldg(exp_s2 + nr);
			}
			LmSkinnyFormat<LmFp8>::Activation a;
			a.low = ((const uint4 *)s_inter[slot])[2u * lane];
			a.high = ((const uint4 *)s_inter[slot])[2u * lane + 1u];
			float dot = fmaf(sc_cur, LmSkinnyFormat<LmFp8>::Chunk(w_cur, a), 0.0f);
			dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
			if ( lane == 0u )
				acc = fmaf(LmBf16ToFloat(LmFloatToBf16(dot)), s_weight[slot], acc);
			if ( slot + 1u < G5N_TOP_K )
			{
				w_cur = w_next;
				sc_cur = sc_next;
			}
		}
		const uint4 *sw = (const uint4 *)(shared_w2 + (uint64_t)out_row * INTER);
		const uint4 sa0 = ((const uint4 *)s_inter[G5N_TOP_K])[lane], sa1 = ((const uint4 *)s_inter[G5N_TOP_K])[lane + 8u];
		float sdot = LmSkinnyBf16Chunk(__ldcs(sw + lane), sa0, 0.0f) + LmSkinnyBf16Chunk(__ldcs(sw + lane + 8u), sa1, 0.0f);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 4u);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 2u);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 1u);
		if ( lane == 0u )
			out_bf16[(uint64_t)token * G5N_HIDDEN + out_row] = LmFloatToBf16(SparkMathRoundBf16(acc) + LmBf16ToFloat(LmFloatToBf16(sdot)));
	}
	else
	{
		const uint16_t *ew2 = (const uint16_t *)exp_w2;
		const uint64_t row0 = (uint64_t)s_expert[0] * G5N_HIDDEN + out_row;
		uint4 w_cur = __ldcs((const uint4 *)(ew2 + row0 * INTER) + lane);
		for ( uint32_t slot = 0u; slot < G5N_TOP_K; ++slot )
		{
			uint4 w_next;
			if ( slot + 1u < G5N_TOP_K )
			{
				const uint64_t nr = (uint64_t)s_expert[slot + 1u] * G5N_HIDDEN + out_row;
				w_next = __ldcs((const uint4 *)(ew2 + nr * INTER) + lane);
			}
			float dot = LmSkinnyBf16Chunk(w_cur, ((const uint4 *)s_inter[slot])[lane], 0.0f);
			dot += __shfl_xor_sync(0xffffffffu, dot, 8u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
			if ( lane == 0u )
				acc = fmaf(LmBf16ToFloat(LmFloatToBf16(dot)), s_weight[slot], acc);
			if ( slot + 1u < G5N_TOP_K )
				w_cur = w_next;
		}
		const uint4 *sw = (const uint4 *)(shared_w2 + (uint64_t)out_row * INTER);
		float sdot = LmSkinnyBf16Chunk(__ldcs(sw + lane), ((const uint4 *)s_inter[G5N_TOP_K])[lane], 0.0f);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 8u);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 4u);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 2u);
		sdot += __shfl_xor_sync(0xffffffffu, sdot, 1u);
		if ( lane == 0u )
			out_bf16[(uint64_t)token * G5N_HIDDEN + out_row] = LmFloatToBf16(SparkMathRoundBf16(acc) + LmBf16ToFloat(LmFloatToBf16(sdot)));
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextMoeMmaW1ActFp8Kernel(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t K = G5N_HIDDEN, INTER = 128u, TILE_N = 64u, TILE_K = 128u, N_TILES = INTER / TILE_N, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u, B_ROW_BYTES = TILE_K, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * A_ROW_CHUNKS) / 256u, B_PER_T = (TILE_N * B_ROW_CHUNKS) / 256u;
	constexpr uint32_t A_STEP = 256u / A_ROW_CHUNKS, B_STEP = 256u / B_ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t a_row0 = threadIdx.x / A_ROW_CHUNKS, a_col0 = threadIdx.x % A_ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G5N_EXPERTS);
	auto adv = [](uint32_t &off)
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
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		Glm5NextGroupedTileLookupGlobal(group_row_offset, tile_prefix, G5N_EXPERTS, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t packed = row_base + a_row0 + q * A_STEP;
			const uint32_t valid = packed < row_limit ? 1u : 0u;
			const uint32_t sr = valid != 0u ? route_source_token[packed] : 0u;
			a_src[q] = act + (uint64_t)sr * K + a_col0 * 8u;
			a_mask |= valid << q;
		}
		const uint64_t exp_base = (uint64_t)grp * (2u * INTER);
		const uint8_t *bu_base = exp_w1 + (exp_base + col_base + b_row0) * K + b_col0 * 16u;
		const uint8_t *bg_base = exp_w1 + (exp_base + INTER + col_base + b_row0) * K + b_col0 * 16u;
		const uint64_t b_ptr_step = (uint64_t)B_STEP * K;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t r = a_row0 + q * A_STEP;
				const uint32_t valid = (a_mask >> q) & 1u;
				LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u, a_src[q], valid << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = b_row0 + q * B_STEP;
				const uint32_t sw = LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0);
				LmStreamCopy(stage + A_BYTES + sw, bu_base + q * b_ptr_step, 16u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bg_base + q * b_ptr_step, 16u);
			}
			bu_base += TILE_K;
			bg_base += TILE_K;
		};
		float tu[MF][NF][4], tg[MF][NF][4], pu[MF][NF][4], pg[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tu[i][j][e] = tg[i][j][e] = pu[i][j][e] = pg[i][j][e] = 0.0f;
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
			}
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			if constexpr ( STAGES > 2u )
			{
				if ( kp < K_TILES )
					LmStreamWait<STAGES - 2u>();
				else
					LmStreamWait<0u>();
			}
			else
				LmStreamWait<0u>();
			__syncthreads();
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
			}
			float su[NF][2], sg_s[NF][2];
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
				su[j][0] = __ldg(exp_s1 + (exp_base + col) * K_TILES + kt);
				su[j][1] = __ldg(exp_s1 + (exp_base + col + 1u) * K_TILES + kt);
				sg_s[j][0] = __ldg(exp_s1 + (exp_base + INTER + col) * K_TILES + kt);
				sg_s[j][1] = __ldg(exp_s1 + (exp_base + INTER + col + 1u) * K_TILES + kt);
			}
			const uint32_t stage_a = shared_base + sc_off;
			const uint32_t stage_bu = stage_a + A_BYTES;
			const uint32_t stage_bg = stage_bu + B_BYTES;
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				uint32_t bu[NF][4], bg[NF][4];
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bu, warp_n * WARP_N + j * 8u, step, lane, bu[j]);
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bg, warp_n * WARP_N + j * 8u, step, lane, bg[j]);
				}
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					const uint32_t row = ldm_a_row + i * 16u;
					uint32_t af[4];
					LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						uint32_t bu0[2] = { bu[j][0], bu[j][1] }, bg0[2] = { bg[j][0], bg[j][1] };
						LmMmaBf16(pu[i][j], af, bu0);
						LmMmaBf16(pg[i][j], af, bg0);
					}
					LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						uint32_t bu1[2] = { bu[j][2], bu[j][3] }, bg1[2] = { bg[j][2], bg[j][3] };
						LmMmaBf16(pu[i][j], af, bu1);
						LmMmaBf16(pg[i][j], af, bg1);
					}
				}
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
					{
						tu[i][j][e] = fmaf(pu[i][j][e], su[j][e & 1u], tu[i][j][e]);
						tg[i][j][e] = fmaf(pg[i][j][e], sg_s[j][e & 1u], tg[i][j][e]);
						pu[i][j][e] = 0.0f;
						pg[i][j][e] = 0.0f;
					}
			adv(sc_off);
		}
		LmStreamWait<0u>();
		const LmClampedUpGateOp op{G5N_SWIGLU_LIMIT};
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
						const uint16_t lo = LmFloatToBf16(op(LmBf16ToFloat(LmFloatToBf16(tg[i][j][e])), LmBf16ToFloat(LmFloatToBf16(tu[i][j][e]))));
						const uint16_t hi = LmFloatToBf16(op(LmBf16ToFloat(LmFloatToBf16(tg[i][j][e + 1u])), LmBf16ToFloat(LmFloatToBf16(tu[i][j][e + 1u]))));
						*(uint32_t *)(inter_out + (uint64_t)r * INTER + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextMoeMmaW1ActBf16Kernel(const uint16_t *__restrict__ exp_w1, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t K = G5N_HIDDEN, INTER = 128u, TILE_N = 64u, TILE_K = 64u, N_TILES = INTER / TILE_N, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	constexpr uint32_t ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G5N_EXPERTS);
	auto adv = [](uint32_t &off)
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
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		Glm5NextGroupedTileLookupGlobal(group_row_offset, tile_prefix, G5N_EXPERTS, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
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
			const uint32_t sr = valid != 0u ? route_source_token[packed] : 0u;
			a_src[q] = act + (uint64_t)sr * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint64_t exp_base = (uint64_t)grp * (2u * INTER);
		const uint16_t *bu_base = exp_w1 + (exp_base + col_base + row0) * K + col0 * 8u;
		const uint16_t *bg_base = exp_w1 + (exp_base + INTER + col_base + row0) * K + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		auto issue_k = [&](uint32_t st_off)
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
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
				LmStreamCopy(stage + A_BYTES + sw, bu_base + q * b_step, 16u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bg_base + q * b_step, 16u);
			}
			bu_base += TILE_K;
			bg_base += TILE_K;
		};
		uint32_t a_reg[2][MF][4], bu_reg[2][NF][2], bg_reg[2][NF][2];
		float tu[MF][NF][4], tg[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tu[i][j][e] = tg[i][j][e] = 0.0f;
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
				LmStreamLdMatrix2(bu_reg[buf][0], stage_s + A_BYTES + off);
				LmStreamLdMatrix2(bg_reg[buf][0], stage_s + A_BYTES + B_BYTES + off);
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
					bu_reg[buf][jj][0] = tmp[0]; bu_reg[buf][jj][1] = tmp[1];
					bu_reg[buf][jj + 1u][0] = tmp[2]; bu_reg[buf][jj + 1u][1] = tmp[3];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
					bg_reg[buf][jj][0] = tmp[0]; bg_reg[buf][jj][1] = tmp[1];
					bg_reg[buf][jj + 1u][0] = tmp[2]; bg_reg[buf][jj + 1u][1] = tmp[3];
				}
			}
		};
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
			}
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			if constexpr ( STAGES > 2u )
			{
				if ( kp < K_TILES )
					LmStreamWait<STAGES - 2u>();
				else
					LmStreamWait<0u>();
			}
			else
				LmStreamWait<0u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + sc_off;
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
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
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
						LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
					}
			}
			adv(sc_off);
		}
		LmStreamWait<0u>();
		const LmClampedUpGateOp op{G5N_SWIGLU_LIMIT};
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
						const uint16_t lo = LmFloatToBf16(op(LmBf16ToFloat(LmFloatToBf16(tg[i][j][e])), LmBf16ToFloat(LmFloatToBf16(tu[i][j][e]))));
						const uint16_t hi = LmFloatToBf16(op(LmBf16ToFloat(LmFloatToBf16(tg[i][j][e + 1u])), LmBf16ToFloat(LmFloatToBf16(tu[i][j][e + 1u]))));
						*(uint32_t *)(inter_out + (uint64_t)r * INTER + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextMoeMmaW2K128Fp8Kernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ inter_in, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t K = 128u, N = G5N_HIDDEN, TILE_N = 128u, N_TILES = N / TILE_N, STAGES = 2u;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u, B_ROW_BYTES = K, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * A_ROW_CHUNKS) / 256u, B_PER_T = (TILE_N * B_ROW_CHUNKS) / 256u;
	constexpr uint32_t A_STEP = 256u / A_ROW_CHUNKS, B_STEP = 256u / B_ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t a_row0 = threadIdx.x / A_ROW_CHUNKS, a_col0 = threadIdx.x % A_ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G5N_EXPERTS);
	auto issue_tile = [&](uint32_t t_idx, uint32_t st_off, uint32_t &grp_o, uint32_t &rb_o, uint32_t &rl_o, uint32_t &nt_o)
	{
		Glm5NextGroupedTileLookupGlobal(group_row_offset, tile_prefix, G5N_EXPERTS, t_idx, TILE_M, N_TILES, grp_o, rb_o, rl_o, nt_o);
		const uint32_t stage = shared_base + st_off;
		const uint32_t col_base = nt_o * TILE_N;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = a_row0 + q * A_STEP;
			const uint32_t packed = rb_o + r;
			const uint32_t valid = packed < rl_o ? 1u : 0u;
			LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u,
				valid != 0u ? inter_in + (uint64_t)packed * K + a_col0 * 8u : inter_in, valid << 4u);
		}
		const uint8_t *b_ptr = exp_w2 + ((uint64_t)grp_o * N + col_base + b_row0) * K + b_col0 * 16u;
		const uint64_t b_ptr_step = (uint64_t)B_STEP * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = b_row0 + q * B_STEP;
			LmStreamCopy(stage + A_BYTES + LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0), b_ptr + q * b_ptr_step, 16u);
		}
		LmStreamCommit();
	};
	if ( blockIdx.x >= total_tiles )
	{
		LmDependentRelease();
		return;
	}
	uint32_t cur_grp, cur_rb, cur_rl, cur_nt;
	uint32_t nxt_grp = 0u, nxt_rb = 0u, nxt_rl = 0u, nxt_nt = 0u;
	uint32_t cur_off = 0u;
	issue_tile(blockIdx.x, cur_off, cur_grp, cur_rb, cur_rl, cur_nt);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_a = shared_base + cur_off, stage_b = stage_a + A_BYTES;
		const uint32_t has_next = tile + gridDim.x < total_tiles ? 1u : 0u;
		if ( has_next != 0u )
			issue_tile(tile + gridDim.x, cur_off ^ STAGE_BYTES, nxt_grp, nxt_rb, nxt_rl, nxt_nt);
		const uint32_t col_base = cur_nt * TILE_N;
		float sc[NF][2];
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
			sc[j][0] = __ldg(exp_s2 + (uint64_t)cur_grp * N + col);
			sc[j][1] = __ldg(exp_s2 + (uint64_t)cur_grp * N + col + 1u);
		}
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t b[NF][4];
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b[j]);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				uint32_t af[4];
				LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b0[2] = { b[j][0], b[j][1] };
					LmMmaBf16(acc[i][j], af, b0);
				}
				LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b1[2] = { b[j][2], b[j][3] };
					LmMmaBf16(acc[i][j], af, b1);
				}
			}
		}
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
					{
						const uint16_t lo = LmFloatToBf16(fmaf(acc[i][j][e], sc[j][0], 0.0f));
						const uint16_t hi = LmFloatToBf16(fmaf(acc[i][j][e + 1u], sc[j][1], 0.0f));
						*(uint32_t *)(out_bf16 + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		cur_grp = nxt_grp;
		cur_rb = nxt_rb;
		cur_rl = nxt_rl;
		cur_nt = nxt_nt;
		cur_off ^= STAGE_BYTES;
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextMoeMmaW2K128Bf16Kernel(const uint16_t *__restrict__ exp_w2, const uint16_t *__restrict__ inter_in,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t K = 128u, N = G5N_HIDDEN, TILE_N = 128u, TILE_K = 64u, N_TILES = N / TILE_N, STAGES = 2u;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	constexpr uint32_t ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G5N_EXPERTS);
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
					valid != 0u ? inter_in + (uint64_t)packed * K + k_off + c * 8u : inter_in, valid << 4u);
			}
		}
		const uint16_t *b_ptr = exp_w2 + ((uint64_t)grp * N + col_base + row0) * K + k_off + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, b_ptr + q * b_step, 16u);
		}
		LmStreamCommit();
	};
	if ( blockIdx.x >= total_tiles )
	{
		LmDependentRelease();
		return;
	}
	uint32_t cur_grp, cur_rb, cur_rl, cur_nt;
	Glm5NextGroupedTileLookupGlobal(group_row_offset, tile_prefix, G5N_EXPERTS, blockIdx.x, TILE_M, N_TILES, cur_grp, cur_rb, cur_rl, cur_nt);
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
			Glm5NextGroupedTileLookupGlobal(group_row_offset, tile_prefix, G5N_EXPERTS, tile + gridDim.x, TILE_M, N_TILES, nxt_grp, nxt_rb, nxt_rl, nxt_nt);
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
						*(uint32_t *)(out_bf16 + (uint64_t)r * N + c) = (uint32_t)LmFloatToBf16(acc[i][j][e]) | ((uint32_t)LmFloatToBf16(acc[i][j][e + 1u]) << 16u);
				}
		cur_grp = nxt_grp;
		cur_rb = nxt_rb;
		cur_rl = nxt_rl;
		cur_nt = nxt_nt;
	}
	LmDependentRelease();
}

template<uint32_t TAG, class Kernel>
static int32_t Glm5NextMoeMmaSetup(Kernel kernel, uint32_t shared_bytes, uint32_t sms, uint32_t *blocks_out)
{
	static std::atomic<uint64_t> granted(0u);
	static std::atomic<int32_t> resident(0);
	int device = 0, per_sm = 1;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( shared_bytes > 49152u &&
			SparkGpuFuncSetAttribute((const void *)kernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, (size_t)shared_bytes) != SPARK_GPU_SUCCESS || per_sm < 1 )
			per_sm = 1;
		resident.store(per_sm);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	*blocks_out = sms * (uint32_t)resident.load();
	return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeMmaW1ActLaunch(const Glm5NextLayerBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	const uint32_t mean = (packed_rows + G5N_EXPERTS - 1u) / G5N_EXPERTS;
	const uint32_t tile_m = mean > 32u ? 64u : mean > 16u ? 32u : 16u;
	uint32_t blocks = sms;
	LM_LAUNCH_DEPENDENT((Glm5NextBuildTilePrefixKernel), 1u, 32u, 0, stream, b->group_row_offset, b->group_tile_prefix_w1, G5N_EXPERTS, tile_m, 2u);
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		if ( tile_m == 64u )
		{
			constexpr uint32_t sh = 2u * (64u * 256u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<1u>(Glm5NextMoeMmaW1ActFp8Kernel<64u, 4u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActFp8Kernel<64u, 4u, 2u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset,
				b->group_tile_prefix_w1, b->route_source_token, b->intermediate_bf16);
		}
		else if ( tile_m == 32u )
		{
			constexpr uint32_t sh = 2u * (32u * 256u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<2u>(Glm5NextMoeMmaW1ActFp8Kernel<32u, 2u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActFp8Kernel<32u, 2u, 2u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset,
				b->group_tile_prefix_w1, b->route_source_token, b->intermediate_bf16);
		}
		else
		{
			constexpr uint32_t sh = 2u * (16u * 256u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<3u>(Glm5NextMoeMmaW1ActFp8Kernel<16u, 1u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActFp8Kernel<16u, 1u, 2u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset,
				b->group_tile_prefix_w1, b->route_source_token, b->intermediate_bf16);
		}
	}
	else
	{
		if ( tile_m == 64u )
		{
			constexpr uint32_t sh = 2u * (64u * 128u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<4u>(Glm5NextMoeMmaW1ActBf16Kernel<64u, 4u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActBf16Kernel<64u, 4u, 2u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w1_weight, b->normed_bf16, b->group_row_offset, b->group_tile_prefix_w1,
				b->route_source_token, b->intermediate_bf16);
		}
		else if ( tile_m == 32u )
		{
			constexpr uint32_t sh = 2u * (32u * 128u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<5u>(Glm5NextMoeMmaW1ActBf16Kernel<32u, 2u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActBf16Kernel<32u, 2u, 2u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w1_weight, b->normed_bf16, b->group_row_offset, b->group_tile_prefix_w1,
				b->route_source_token, b->intermediate_bf16);
		}
		else
		{
			constexpr uint32_t sh = 2u * (16u * 128u + 2u * 64u * 128u);
			if ( Glm5NextMoeMmaSetup<6u>(Glm5NextMoeMmaW1ActBf16Kernel<16u, 1u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW1ActBf16Kernel<16u, 1u, 2u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w1_weight, b->normed_bf16, b->group_row_offset, b->group_tile_prefix_w1,
				b->route_source_token, b->intermediate_bf16);
		}
	}
	return Glm5NextLaunched();
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeMmaW2K128Launch(const Glm5NextLayerBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	const uint32_t mean = (packed_rows + G5N_EXPERTS - 1u) / G5N_EXPERTS;
	const uint32_t tile_m = mean > 32u ? 64u : mean > 16u ? 32u : 16u;
	uint32_t blocks = sms;
	LM_LAUNCH_DEPENDENT((Glm5NextBuildTilePrefixKernel), 1u, 32u, 0, stream, b->group_row_offset, b->group_tile_prefix_w2, G5N_EXPERTS, tile_m, G5N_HIDDEN / 128u);
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		if ( tile_m == 64u )
		{
			constexpr uint32_t sh = 2u * (64u * 256u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<11u>(Glm5NextMoeMmaW2K128Fp8Kernel<64u, 4u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Fp8Kernel<64u, 4u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
				b->group_tile_prefix_w2, b->expert_out_bf16);
		}
		else if ( tile_m == 32u )
		{
			constexpr uint32_t sh = 2u * (32u * 256u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<12u>(Glm5NextMoeMmaW2K128Fp8Kernel<32u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Fp8Kernel<32u, 2u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
				b->group_tile_prefix_w2, b->expert_out_bf16);
		}
		else
		{
			constexpr uint32_t sh = 2u * (16u * 256u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<13u>(Glm5NextMoeMmaW2K128Fp8Kernel<16u, 1u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Fp8Kernel<16u, 1u>), blocks, 256u, sh, stream,
				(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
				b->group_tile_prefix_w2, b->expert_out_bf16);
		}
	}
	else
	{
		if ( tile_m == 64u )
		{
			constexpr uint32_t sh = 2u * (64u * 128u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<14u>(Glm5NextMoeMmaW2K128Bf16Kernel<64u, 4u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Bf16Kernel<64u, 4u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2, b->expert_out_bf16);
		}
		else if ( tile_m == 32u )
		{
			constexpr uint32_t sh = 2u * (32u * 128u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<15u>(Glm5NextMoeMmaW2K128Bf16Kernel<32u, 2u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Bf16Kernel<32u, 2u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2, b->expert_out_bf16);
		}
		else if ( b->prefill_unified != 0u )
		{
			constexpr uint32_t sh = 2u * (16u * 128u + 128u * 128u);
			if ( Glm5NextMoeMmaSetup<16u>(Glm5NextMoeMmaW2K128Bf16Kernel<16u, 1u>, sh, sms, &blocks) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			LM_LAUNCH_DEPENDENT((Glm5NextMoeMmaW2K128Bf16Kernel<16u, 1u>), blocks, 256u, sh, stream,
				(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2, b->expert_out_bf16);
		}
		else
		{
			return LmStreamGemmGrouped<LmBf16Format>(b->expert_w2_weight, LmScaleTensorNone(), b->intermediate_bf16, b->expert_out_bf16,
				b->group_row_offset, 0, G5N_EXPERTS, packed_rows, b->expert_intermediate, G5N_HIDDEN, sms, stream);
		}
	}
	return Glm5NextLaunched();
}
