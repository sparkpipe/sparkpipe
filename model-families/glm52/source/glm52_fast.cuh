#pragma once

static __device__ __forceinline__ void GlmCpAsync16(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

template<class Format, uint32_t LANES, uint32_t ROWS_PER_GROUP, uint32_t NPG>
__global__ __launch_bounds__(GLM_THREADS, 2)
static void GlmExactLinearKernel(const uint8_t *__restrict__ weight, const float *__restrict__ scale, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, float *__restrict__ output_f32, uint32_t rows, uint32_t input_dim, uint32_t output_dim,
	uint32_t output_row_stride, uint32_t output_col_offset)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMENTS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t GROUPS = GLM_THREADS / LANES;
	constexpr uint32_t TILE_ROWS = GROUPS * ROWS_PER_GROUP;
	constexpr uint32_t TOTAL_W = NPG * LANES;
	__shared__ __align__(16) uint4 s_w[2][NPG][LANES];
	const uint32_t n0 = blockIdx.x * NPG;
	const uint32_t grp = threadIdx.x / LANES;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const uint32_t warp_first_row = blockIdx.y * TILE_ROWS + grp * ROWS_PER_GROUP;
	const uint32_t row_cnt = warp_first_row < rows ? (rows - warp_first_row < ROWS_PER_GROUP ? rows - warp_first_row : ROWS_PER_GROUP) : 0u;
	const uint32_t chunks = input_dim / ELEMENTS;
	const uint32_t scale_cols = (input_dim + 127u) >> 7u;
	const uint4 *w4 = (const uint4 *)weight;
	LmDependentWait();
	auto stage_w = [&](uint32_t st, uint32_t k_base)
	{
		if ( threadIdx.x < TOTAL_W )
		{
			const uint32_t wn = threadIdx.x / LANES, wc = threadIdx.x & (LANES - 1u);
			const uint32_t neuron = n0 + wn, c = k_base + wc;
			const bool ok = neuron < output_dim && c < chunks;
			const uint4 *src = ok ? (w4 + (uint64_t)neuron * chunks + c) : w4;
			GlmCpAsync16(&s_w[st][wn][wc], src, ok ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	stage_w(0u, 0u);
	float acc[NPG][ROWS_PER_GROUP];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
			acc[n][r] = 0.0f;
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += LANES )
	{
		const uint32_t cur = (k_base / LANES) & 1u;
		if ( k_base + LANES < chunks )
			stage_w(cur ^ 1u, k_base + LANES);
		if ( k_base + LANES < chunks )
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t c = k_base + sub;
		if ( row_cnt != 0u && c < chunks )
		{
			if constexpr ( IS_FP8 )
			{
				const uint32_t sg = c >> 3u;
				LmSkinnyFormat<LmFp8>::Decoded dec[NPG];
				float sc[NPG];
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
				{
					const uint32_t neuron = n0 + n < output_dim ? n0 + n : 0u;
					dec[n] = LmSkinnyFormat<LmFp8>::Decode(s_w[cur][n][sub]);
					sc[n] = __ldg(scale + (uint64_t)neuron * scale_cols + sg);
				}
				#pragma unroll
				for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
				{
					const uint32_t row = r < row_cnt ? warp_first_row + r : 0u;
					const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(activation + (uint64_t)row * input_dim + c * 16u);
					float af[16];
					LmSkinnyUnpackBf16x8(a.low, af);
					LmSkinnyUnpackBf16x8(a.high, af + 8);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(sc[n], LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(dec[n], af), acc[n][r]);
				}
			}
			else
			{
				uint4 w[NPG];
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
					w[n] = s_w[cur][n][sub];
				#pragma unroll
				for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
				{
					const uint32_t row = r < row_cnt ? warp_first_row + r : 0u;
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)row * input_dim) + c), af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w[n], af, 0.0f), acc[n][r]);
				}
			}
		}
		if ( k_base + LANES < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
	{
		#pragma unroll
		for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				#pragma unroll
				for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
					acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], step, LANES);
		if ( sub == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
				if ( r < row_cnt )
				{
					const uint64_t row_off = (uint64_t)(warp_first_row + r) * output_row_stride + output_col_offset + n0;
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						if ( n0 + n < output_dim )
						{
							if ( output_f32 != 0 )
								output_f32[row_off + n] = acc[n][r];
							else
								output_bf16[row_off + n] = LmFloatToBf16(acc[n][r]);
						}
				}
	}
	LmDependentRelease();
}

template<class Format>
__global__ __launch_bounds__(GLM_THREADS, 2)
static void GlmExactSplitLinearKernel(const uint8_t *__restrict__ weight, const float *__restrict__ scale, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, float *__restrict__ output_f32, uint32_t rows, uint32_t input_dim, uint32_t output_dim,
	uint32_t output_row_stride, uint32_t output_col_offset)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMENTS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t SPLIT_W = 8u, BN = IS_FP8 ? 2u : 4u, RT = 8u;
	__shared__ float partial[SPLIT_W][BN][RT];
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t col0 = blockIdx.x * BN, r0 = blockIdx.y * RT;
	const uint32_t count = rows - r0 < RT ? rows - r0 : RT;
	const uint32_t chunks = input_dim / ELEMENTS;
	const uint32_t scale_cols = (input_dim + 127u) >> 7u;
	const uint32_t first = chunks * warp / SPLIT_W, last = chunks * (warp + 1u) / SPLIT_W, seg = last - first;
	const uint4 *w_rows[BN];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		w_rows[b] = (const uint4 *)weight + (uint64_t)(col0 + b < output_dim ? col0 + b : 0u) * chunks + first;
	float acc[BN][RT];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
			acc[b][r] = 0.0f;
	LmDependentWait();
	for ( uint32_t c = sub; c < seg; c += 32u )
	{
		const uint32_t gc = first + c;
		if constexpr ( IS_FP8 )
		{
			const uint32_t sg = gc >> 3u;
			LmSkinnyFormat<LmFp8>::Decoded dec[BN];
			float sc[BN];
			#pragma unroll
			for ( uint32_t b = 0u; b < BN; ++b )
			{
				const uint32_t neuron = col0 + b < output_dim ? col0 + b : 0u;
				dec[b] = LmSkinnyFormat<LmFp8>::Decode(__ldg(w_rows[b] + c));
				sc[b] = __ldg(scale + (uint64_t)neuron * scale_cols + sg);
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				const uint32_t rr = r0 + (r < count ? r : 0u);
				const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(activation + (uint64_t)rr * input_dim + gc * 16u);
				float af[16];
				LmSkinnyUnpackBf16x8(a.low, af);
				LmSkinnyUnpackBf16x8(a.high, af + 8);
				#pragma unroll
				for ( uint32_t b = 0u; b < BN; ++b )
					acc[b][r] = fmaf(sc[b], LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(dec[b], af), acc[b][r]);
			}
		}
		else
		{
			uint4 wv[BN];
			#pragma unroll
			for ( uint32_t b = 0u; b < BN; ++b )
				wv[b] = __ldg(w_rows[b] + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < RT; ++r )
			{
				const uint32_t rr = r0 + (r < count ? r : 0u);
				float af[8];
				LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)rr * input_dim + gc * 8u)), af);
				#pragma unroll
				for ( uint32_t b = 0u; b < BN; ++b )
					acc[b][r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(wv[b], af, 0.0f), acc[b][r]);
			}
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
		const uint32_t b = sub >> 3u, r = sub & 7u;
		if ( col0 + b < output_dim && r < count )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < SPLIT_W; ++w )
				total += partial[w][b][r];
			const uint64_t idx = (uint64_t)(r0 + r) * output_row_stride + output_col_offset + (col0 + b);
			if ( output_f32 != 0 )
				output_f32[idx] = total;
			else
				output_bf16[idx] = LmFloatToBf16(total);
		}
	}
	LmDependentRelease();
}

template<class Format>
static int32_t GlmExactLinearLaunch(const uint16_t *activation, const void *weight, const float *scale, uint16_t *output_bf16, float *output_f32,
	uint32_t rows, uint32_t input_dim, uint32_t output_dim, uint32_t output_row_stride, uint32_t output_col_offset, uint32_t allow_split,
	SparkGpuStream stream)
{
	constexpr uint32_t ELEMENTS = LmSkinnyFormat<Format>::kElements;
	if ( !LmIsAligned16(activation) || !LmIsAligned16(weight) || (input_dim % ELEMENTS) != 0u )
		return LM_LAUNCH_ERR_SHAPE;
	const uint32_t stride = output_row_stride != 0u ? output_row_stride : output_dim;
	const uint32_t chunks = input_dim / ELEMENTS;
	if ( allow_split != 0u && output_dim <= LM_SKINNY_SPLIT_MAX_OUTPUTS && chunks >= LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES )
	{
		constexpr uint32_t SPLIT_BN = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 ? 2u : 4u;
		LM_LAUNCH_DEPENDENT((GlmExactSplitLinearKernel<Format>), dim3((output_dim + SPLIT_BN - 1u) / SPLIT_BN, (rows + 7u) / 8u), GLM_THREADS, 0, stream,
			(const uint8_t *)weight, scale, activation, output_bf16, output_f32, rows, input_dim, output_dim, stride, output_col_offset);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	if ( chunks >= 32u )
		LM_LAUNCH_DEPENDENT((GlmExactLinearKernel<Format, 32u, 8u, 4u>), dim3((output_dim + 3u) / 4u, (rows + 63u) / 64u), GLM_THREADS, 0, stream,
			(const uint8_t *)weight, scale, activation, output_bf16, output_f32, rows, input_dim, output_dim, stride, output_col_offset);
	else if ( chunks >= 16u )
		LM_LAUNCH_DEPENDENT((GlmExactLinearKernel<Format, 16u, 4u, 4u>), dim3((output_dim + 3u) / 4u, (rows + 63u) / 64u), GLM_THREADS, 0, stream,
			(const uint8_t *)weight, scale, activation, output_bf16, output_f32, rows, input_dim, output_dim, stride, output_col_offset);
	else if ( chunks >= 8u )
		LM_LAUNCH_DEPENDENT((GlmExactLinearKernel<Format, 8u, 2u, 4u>), dim3((output_dim + 3u) / 4u, (rows + 63u) / 64u), GLM_THREADS, 0, stream,
			(const uint8_t *)weight, scale, activation, output_bf16, output_f32, rows, input_dim, output_dim, stride, output_col_offset);
	else
		return LM_LAUNCH_ERR_SHAPE;
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static __global__ __launch_bounds__(GLM_THREADS, 4)
void GlmIndexPackDirectKernel(SparkKvShard shard, const float *__restrict__ scores, uint32_t score_stride, uint32_t rows,
	uint2 *__restrict__ candidates)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y;
	if ( row < rows )
	{
		const uint64_t base = (uint64_t)row * score_stride;
		for ( uint32_t local = blockIdx.x * GLM_THREADS + threadIdx.x; local < score_stride; local += gridDim.x * GLM_THREADS )
		{
			const float score = scores[base + local];
			candidates[base + local] = make_uint2(__float_as_uint(score),
				score > -INFINITY ? SparkKvShardLocalPosition(shard, local) : LM_INDEX_SHARD_NO_POSITION);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t RANK, uint32_t TILE_R>
__global__ __launch_bounds__(THREADS, 2)
static void GlmLowRankGateDownFastKernel(const uint16_t *__restrict__ rows_bf16, const uint16_t *__restrict__ down_bf16,
	float *__restrict__ bottleneck, uint32_t rows, uint32_t dimension)
{
	__shared__ float partial[LmBlockWarps<THREADS>()];
	const uint32_t rank = blockIdx.x, row_base = blockIdx.y * TILE_R;
	const uint32_t cnt = row_base < rows ? (rows - row_base < TILE_R ? rows - row_base : TILE_R) : 0u;
	const uint32_t vecs = dimension >> 3u;
	LmDependentWait();
	if ( cnt != 0u && rank < RANK )
	{
		const uint4 *d4 = (const uint4 *)(down_bf16 + (uint64_t)rank * dimension);
		const bool reuse = rows > TILE_R;
		float total[TILE_R];
		#pragma unroll
		for ( uint32_t r = 0u; r < TILE_R; ++r )
			total[r] = 0.0f;
		for ( uint32_t v = threadIdx.x; v < vecs; v += THREADS )
		{
			float dw[8];
			LmLoadBf16x8AsFloat(reuse ? __ldg(d4 + v) : __ldcs(d4 + v), dw);
			#pragma unroll
			for ( uint32_t r = 0u; r < TILE_R; ++r )
				if ( TILE_R == 1u || r < cnt )
				{
					float xv[8];
					LmLoadBf16x8AsFloat(__ldg((const uint4 *)(rows_bf16 + (uint64_t)(row_base + r) * dimension) + v), xv);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
						total[r] = fmaf(xv[k], dw[k], total[r]);
				}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < TILE_R; ++r )
			if ( TILE_R == 1u || r < cnt )
			{
				const float sum = LmBlockSum<THREADS>(total[r], partial);
				if ( threadIdx.x == 0u )
					bottleneck[(uint64_t)(row_base + r) * RANK + rank] = SparkMathRoundBf16(SparkMathSiluFast(SparkMathRoundBf16(sum)));
			}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t RANK, uint32_t TILE_R>
__global__ __launch_bounds__(THREADS, 2)
static void GlmLowRankGateApplyFastKernel(uint16_t *__restrict__ rows_bf16, const float *__restrict__ bottleneck,
	const uint16_t *__restrict__ up_bf16, uint32_t rows, uint32_t dimension)
{
	__shared__ float s_factors[TILE_R][RANK];
	const uint32_t index = blockIdx.x * THREADS + threadIdx.x, row_base = blockIdx.y * TILE_R;
	const uint32_t cnt = row_base < rows ? (rows - row_base < TILE_R ? rows - row_base : TILE_R) : 0u;
	LmDependentWait();
	if ( cnt != 0u )
	{
		if ( threadIdx.x < cnt * RANK )
		{
			const uint32_t r = threadIdx.x / RANK, rk = threadIdx.x % RANK;
			s_factors[r][rk] = bottleneck[(uint64_t)(row_base + r) * RANK + rk];
		}
		__syncthreads();
		if ( index < dimension )
		{
			const bool reuse = rows > TILE_R;
			if constexpr ( RANK == 16u )
			{
				const uint4 *u4 = (const uint4 *)(up_bf16 + (uint64_t)index * 16u);
				float u0[8], u1[8];
				LmLoadBf16x8AsFloat(reuse ? __ldg(u4) : __ldcs(u4), u0);
				LmLoadBf16x8AsFloat(reuse ? __ldg(u4 + 1) : __ldcs(u4 + 1), u1);
				#pragma unroll
				for ( uint32_t r = 0u; r < TILE_R; ++r )
					if ( TILE_R == 1u || r < cnt )
					{
						float gate = 0.0f;
						#pragma unroll
						for ( uint32_t k = 0u; k < 8u; ++k )
							gate += s_factors[r][k] * u0[k];
						#pragma unroll
						for ( uint32_t k = 0u; k < 8u; ++k )
							gate += s_factors[r][8u + k] * u1[k];
						const uint64_t pos = (uint64_t)(row_base + r) * dimension + index;
						rows_bf16[pos] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[pos]) * SparkMathSigmoidFast(SparkMathRoundBf16(gate)));
					}
			}
			else
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < TILE_R; ++r )
					if ( TILE_R == 1u || r < cnt )
					{
						float gate = 0.0f;
						for ( uint32_t rank = 0u; rank < RANK; ++rank )
							gate += s_factors[r][rank] * LmBf16ToFloat(__ldg(up_bf16 + (uint64_t)index * RANK + rank));
						const uint64_t pos = (uint64_t)(row_base + r) * dimension + index;
						rows_bf16[pos] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[pos]) * SparkMathSigmoidFast(SparkMathRoundBf16(gate)));
					}
			}
		}
	}
	LmDependentRelease();
}

template<class S>
__global__ __launch_bounds__(GLM_THREADS, 2)
static void GlmIndexHeadSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	float *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr uint32_t K = S::kHidden, N = S::kIndexHeads, CHUNKS = K / 8u, WARPS = GLM_THREADS / 32u, RG = 8u;
	constexpr bool USE_SPLIT = CHUNKS >= WARPS * 32u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	__shared__ float partial[WARPS][RG];
	const uint32_t head = blockIdx.x, row_base = blockIdx.y * RG, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t cnt = row_base < rows ? (rows - row_base < RG ? rows - row_base : RG) : 0u;
	LmDependentWait();
	if ( warp == 0u )
	{
		uint32_t sparse = 0u;
		if ( lane < cnt )
		{
			const uint32_t r = row_base + lane;
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( positions != 0 && positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			sparse = bound > dense_limit ? 1u : 0u;
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
		if ( lane == 0u )
		{
			uint32_t n = 0u, m = mask;
			while ( m != 0u )
			{
				const uint32_t bit = (uint32_t)(__ffs((int)m) - 1);
				s_rows[n++] = row_base + bit;
				m &= m - 1u;
			}
			s_cnt = n;
		}
	}
	__syncthreads();
	const uint32_t act_cnt = s_cnt;
	if ( act_cnt != 0u && head < N )
	{
		const uint4 *w_row = (const uint4 *)(weight + (uint64_t)head * K);
		const bool reuse = rows > RG;
		if constexpr ( USE_SPLIT )
		{
			const uint32_t first = CHUNKS * warp / WARPS, last = CHUNKS * (warp + 1u) / WARPS;
			float acc[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[r] = 0.0f;
			for ( uint32_t c = first + lane; c < last; c += 32u )
			{
				const uint4 vw = reuse ? __ldg(w_row + c) : __ldcs(w_row + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < act_cnt )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)s_rows[r] * K) + c), af);
						acc[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw, af, 0.0f), acc[r]);
					}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < act_cnt )
				{
					#pragma unroll
					for ( uint32_t step = 16u; step > 0u; step >>= 1u )
						acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], step);
					if ( lane == 0u )
						partial[warp][r] = acc[r];
				}
			__syncthreads();
			if ( warp == 0u && lane < act_cnt )
			{
				float total = 0.0f;
				#pragma unroll
				for ( uint32_t w = 0u; w < WARPS; ++w )
					total += partial[w][lane];
				output[(uint64_t)s_rows[lane] * N + head] = total;
			}
		}
		else
		{
			if ( warp < act_cnt )
			{
				const uint32_t r = s_rows[warp];
				float acc = 0.0f;
				for ( uint32_t c = lane; c < CHUNKS; c += 32u )
				{
					const uint4 vw = reuse ? __ldg(w_row + c) : __ldcs(w_row + c);
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)r * K) + c), af);
					acc = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw, af, 0.0f), acc);
				}
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					acc += __shfl_xor_sync(0xffffffffu, acc, step);
				if ( lane == 0u )
					output[(uint64_t)r * N + head] = acc;
			}
		}
	}
	LmDependentRelease();
}

template<class S>
__global__ __launch_bounds__(128u, 4)
static void GlmIndexQExactSkipKernel(const uint8_t *__restrict__ weight, const float *__restrict__ scale,
	const uint16_t *__restrict__ activation, uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr bool IS_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t K = S::kQueryA, N = S::kIndexQuery, ELEM = IS_FP8 ? 16u : 8u, CHUNKS = K / ELEM, SCALES = (K + 127u) >> 7u, RG = 8u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	const uint32_t row_base = blockIdx.y * RG, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t cnt = row_base < rows ? (rows - row_base < RG ? rows - row_base : RG) : 0u;
	LmDependentWait();
	if ( warp == 0u )
	{
		uint32_t sparse = 0u;
		if ( lane < cnt )
		{
			const uint32_t r = row_base + lane;
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( positions != 0 && positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			sparse = bound > dense_limit ? 1u : 0u;
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
		if ( lane == 0u )
		{
			uint32_t n = 0u, m = mask;
			while ( m != 0u )
			{
				const uint32_t bit = (uint32_t)(__ffs((int)m) - 1);
				s_rows[n++] = row_base + bit;
				m &= m - 1u;
			}
			s_cnt = n;
		}
	}
	__syncthreads();
	const uint32_t act_cnt = s_cnt;
	const uint32_t col0 = (blockIdx.x * 4u + warp) * 2u;
	if ( act_cnt != 0u && col0 < N )
	{
		const bool reuse = rows > RG;
		float a0[RG], a1[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			a0[r] = a1[r] = 0.0f;
		if constexpr ( IS_FP8 )
		{
			const uint4 *w0 = (const uint4 *)(weight + (uint64_t)col0 * K);
			const uint4 *w1 = w0 + CHUNKS;
			const float *s0 = scale + (uint64_t)col0 * SCALES;
			const float *s1 = s0 + SCALES;
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				const uint32_t sg = c >> 3u;
				const LmSkinnyFormat<LmFp8>::Decoded d0 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(w0 + c) : __ldcs(w0 + c));
				const LmSkinnyFormat<LmFp8>::Decoded d1 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(w1 + c) : __ldcs(w1 + c));
				const float sc0 = __ldg(s0 + sg), sc1 = __ldg(s1 + sg);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < act_cnt )
					{
						const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(activation + (uint64_t)s_rows[r] * K + c * 16u);
						float af[16];
						LmSkinnyUnpackBf16x8(a.low, af);
						LmSkinnyUnpackBf16x8(a.high, af + 8);
						a0[r] = fmaf(sc0, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(d0, af), a0[r]);
						a1[r] = fmaf(sc1, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(d1, af), a1[r]);
					}
			}
		}
		else
		{
			const uint4 *w0 = (const uint4 *)((const uint16_t *)weight + (uint64_t)col0 * K);
			const uint4 *w1 = w0 + CHUNKS;
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
				const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < act_cnt )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)s_rows[r] * K) + c), af);
						a0[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw0, af, 0.0f), a0[r]);
						a1[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw1, af, 0.0f), a1[r]);
					}
			}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			if ( r < act_cnt )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				{
					a0[r] += __shfl_xor_sync(0xffffffffu, a0[r], step);
					a1[r] += __shfl_xor_sync(0xffffffffu, a1[r], step);
				}
				if ( lane == 0u )
					*(uint32_t *)(output + (uint64_t)s_rows[r] * N + col0) = LmPackBf16Pair(a0[r], a1[r]);
			}
	}
	LmDependentRelease();
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(256u, 2)
static void GlmIndexQMmaSkipKernel(const uint8_t *__restrict__ weight, const float *__restrict__ scale,
	const uint16_t *__restrict__ activation, uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr bool IS_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t K = S::kQueryA, N = S::kIndexQuery, TILE_N = 64u, TILE_K = IS_FP8 ? 128u : 64u, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t B_ROW_BYTES = IS_FP8 ? TILE_K : TILE_K * 2u, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u;
	constexpr uint32_t B_PER_T = (TILE_N * B_ROW_CHUNKS) / 256u, B_STEP = 256u / B_ROW_CHUNKS;
	__shared__ __align__(128) uint8_t smem[2u * STAGE_BYTES];
	__shared__ uint32_t s_any;
	const uint32_t row_base = blockIdx.y * TILE_M, row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	LmDependentWait();
	if ( threadIdx.x == 0u )
		s_any = 0u;
	__syncthreads();
	if ( threadIdx.x < TILE_M )
	{
		const uint32_t r = row_base + threadIdx.x;
		if ( r < row_limit )
		{
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( positions != 0 && positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			if ( bound > dense_limit )
				s_any = 1u;
		}
	}
	__syncthreads();
	if ( s_any != 0u )
	{
		const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
		const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
		const uint32_t col_base = blockIdx.x * TILE_N;
		const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
		const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
			const uint32_t valid = chunk < A_CHUNKS && row_base + r < row_limit ? 1u : 0u;
			a_src[q] = activation + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint8_t *b_src = weight + (uint64_t)(col_base + b_row0) * (IS_FP8 ? K : K * 2u) + b_col0 * 16u;
		const uint64_t b_step = (uint64_t)B_STEP * (IS_FP8 ? K : K * 2u);
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * 256u;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
					LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
					a_src[q] += TILE_K;
				}
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = b_row0 + q * B_STEP;
				const uint32_t sw = IS_FP8 ? LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0) :
					(r * B_ROW_BYTES + LmStreamSwizzle8(r, b_col0) * 16u);
				LmStreamCopy(stage + A_BYTES + sw, b_src + q * b_step, 16u);
			}
			b_src += IS_FP8 ? TILE_K : TILE_K * 2u;
		};
		float acc[MF][NF][4], part[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					acc[i][j][e] = part[i][j][e] = 0.0f;
		uint32_t sp_off = 0u, sc_off = 0u;
		issue_k(sp_off);
		LmStreamCommit();
		sp_off ^= STAGE_BYTES;
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<0u>();
			__syncthreads();
			if ( kt + 1u < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				sp_off ^= STAGE_BYTES;
			}
			const uint32_t stage_a = shared_base + sc_off, stage_b = stage_a + A_BYTES;
			if constexpr ( IS_FP8 )
			{
				float sc[NF][2];
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					sc[j][0] = __ldg(scale + (uint64_t)col * K_TILES + kt);
					sc[j][1] = __ldg(scale + (uint64_t)(col + 1u) * K_TILES + kt);
				}
				#pragma unroll
				for ( uint32_t step = 0u; step < 4u; ++step )
				{
					uint32_t af0[MF][4], af1[MF][4];
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
					{
						const uint32_t row = ldm_a_row + i * 16u;
						LmStreamLdMatrix4(af0[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
						LmStreamLdMatrix4(af1[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
					}
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; j += 2u )
					{
						uint32_t b0[4], b1[4];
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b0, b1);
						uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
						uint32_t b1l[2] = { b1[0], b1[1] }, b1h[2] = { b1[2], b1[3] };
						#pragma unroll
						for ( uint32_t i = 0u; i < MF; ++i )
						{
							LmMmaBf16(part[i][j], af0[i], b0l);
							LmMmaBf16(part[i][j + 1u], af0[i], b1l);
							LmMmaBf16(part[i][j], af1[i], b0h);
							LmMmaBf16(part[i][j + 1u], af1[i], b1h);
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
							acc[i][j][e] = fmaf(part[i][j][e], sc[j][e & 1u], acc[i][j][e]);
							part[i][j][e] = 0.0f;
						}
			}
			else
			{
				#pragma unroll
				for ( uint32_t ks = 0u; ks < 4u; ++ks )
				{
					uint32_t af[MF][4], bf[NF][2];
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
					{
						const uint32_t ar = ldm_a_row + i * 16u, ac = ks * 2u + ldm_a_col;
						LmStreamLdMatrix4(af[i], stage_a + ar * A_ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
					}
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						const uint32_t br = warp_n * WARP_N + j * 8u + (lane & 7u), bc = ks * 2u + ((lane >> 3u) & 1u);
						LmStreamLdMatrix2(bf[j], stage_b + br * B_ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
					}
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
						#pragma unroll
						for ( uint32_t j = 0u; j < NF; ++j )
							LmMmaBf16(acc[i][j], af[i], bf[j]);
				}
			}
			sc_off ^= STAGE_BYTES;
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t r0 = row_base + warp_m * WARP_M + i * 16u + quad_row, r1 = r0 + 8u;
				const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( r0 < row_limit )
					*(uint32_t *)(output + (uint64_t)r0 * N + col) = LmPackBf16Pair(acc[i][j][0], acc[i][j][1]);
				if ( r1 < row_limit )
					*(uint32_t *)(output + (uint64_t)r1 * N + col) = LmPackBf16Pair(acc[i][j][2], acc[i][j][3]);
			}
	}
	LmDependentRelease();
}

template<class S, uint32_t ROWS_PER_BLOCK>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
static void GlmIndexScoreSkipKernel(const uint16_t *__restrict__ index_query_bf16, const float *__restrict__ head_weight,
	LmKvShardView index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores,
	const uint32_t *__restrict__ fork_positions, const uint32_t *__restrict__ slot_offsets, uint32_t dense_limit)
{
	constexpr uint32_t HEAD_TILES = S::kIndexHeads / 16u, STEPS = S::kIndexDim / 16u;
	constexpr uint32_t WARPS = LM_INDEX_SCORE_WARPS;
	LmDependentWait();
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint32_t group = lane >> 2u, pair = lane & 3u;
	const uint32_t row = ROWS_PER_BLOCK == 1u ? blockIdx.y : (blockIdx.y * WARPS + warp);
	if ( scores == 0 || row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	const uint32_t valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	const uint32_t sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
	if ( valid == 0u || sequence >= index_cache.sequence_count )
	{
		if ( lane == 0u && blockIdx.x == 0u && (ROWS_PER_BLOCK != 1u || warp == 0u) )
			LmKvReportRequiredAccessFailure(index_cache, valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
				LM_KV_ACCESS_READ, row, sequence, 0u, 0xffffffffu);
		LmDependentRelease();
		return;
	}
	uint32_t bound = context_length[sequence];
	if ( row_position != 0 && row_position[row] + 1u < bound )
		bound = row_position[row] + 1u;
	if ( dense_limit != 0u && bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	bound = LmIndexScoreKeys(index_view, bound);
	const uint32_t limit = bound < score_stride ? bound : score_stride;
	const uint32_t total_tiles = (limit + LM_INDEX_SCORE_TILE_POSITIONS - 1u) / LM_INDEX_SCORE_TILE_POSITIONS;
	const uint32_t tile0 = ROWS_PER_BLOCK == 1u ? (blockIdx.x * WARPS + warp) : blockIdx.x;
	const uint32_t tile_step = ROWS_PER_BLOCK == 1u ? (gridDim.x * WARPS) : gridDim.x;
	if ( tile0 >= total_tiles )
	{
		LmDependentRelease();
		return;
	}
	uint32_t a[HEAD_TILES][STEPS][4];
	float weight[2u * HEAD_TILES];
	const uint16_t *query = index_query_bf16 + (uint64_t)row * S::kIndexHeads * S::kIndexDim;
	#pragma unroll
	for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			const uint16_t *top = query + (uint64_t)(ht * 16u + group) * S::kIndexDim + step * 16u + pair * 2u;
			const uint16_t *bottom = top + 8u * S::kIndexDim;
			a[ht][step][0] = __ldg((const uint32_t *)top);
			a[ht][step][1] = __ldg((const uint32_t *)bottom);
			a[ht][step][2] = __ldg((const uint32_t *)(top + 8u));
			a[ht][step][3] = __ldg((const uint32_t *)(bottom + 8u));
		}
	#pragma unroll
	for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
	{
		weight[2u * ht] = LmIndexScoreWeight(head_weight[(uint64_t)row * S::kIndexHeads + ht * 16u + group]);
		weight[2u * ht + 1u] = LmIndexScoreWeight(head_weight[(uint64_t)row * S::kIndexHeads + ht * 16u + group + 8u]);
	}
	const uint32_t fork = fork_positions != 0 ? fork_positions[row] : 0xffffffffu;
	const uint32_t offset = slot_offsets != 0 ? slot_offsets[row] : 0u;
	uint32_t b[STEPS][2], next[STEPS][2];
	const uint32_t first0 = tile0 * LM_INDEX_SCORE_TILE_POSITIONS;
	LmIndexScoreKeyFragments<STEPS>(first0 + group < limit ?
		LmIndexScoreSlot<GlmIndexKv<S>>(index_view, sequence, first0 + group, row, fork, offset) : 0, pair, next);
	for ( uint32_t tile = tile0; tile < total_tiles; tile += tile_step )
	{
		const uint32_t first = tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t next_tile = tile + tile_step;
		const uint32_t ahead = next_tile * LM_INDEX_SCORE_TILE_POSITIONS + group;
		float accumulator[HEAD_TILES][4] = {};
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			b[step][0] = next[step][0];
			b[step][1] = next[step][1];
		}
		if ( next_tile < total_tiles )
			LmIndexScoreKeyFragments<STEPS>(ahead < limit ?
				LmIndexScoreSlot<GlmIndexKv<S>>(index_view, sequence, ahead, row, fork, offset) : 0, pair, next);
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
			#pragma unroll
			for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
				LmMmaBf16(accumulator[ht], a[ht][step], b[step]);
		LmIndexScoreStore<HEAD_TILES, false>(accumulator, weight, 1u, bound, first + pair * 2u, group, row, score_stride, qk_scale, scores);
	}
	LmDependentRelease();
}

template<class S>
__global__ __launch_bounds__(32u)
static void GlmKvBValueB1GateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16,
	const uint16_t *__restrict__ q_bf16, uint16_t *__restrict__ output_bf16)
{
	constexpr uint32_t IN_DIM = S::kLatent, OUT_DIM = S::kValue, OUTPUTS = 16u, K_TILE = 64u;
	constexpr uint32_t VECTORS = IN_DIM / 8u, TILE_VECS = K_TILE / 8u, STRIDE = K_TILE + 8u;
	__shared__ __align__(16) uint16_t input_tile[IN_DIM];
	__shared__ __align__(16) uint16_t weight_tile[2][OUTPUTS][STRIDE];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint16_t *weight_base = weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	LmDependentWait();
	for ( uint32_t i = threadIdx.x; i < VECTORS; i += 32u )
		((uint4 *)input_tile)[i] = __ldg((const uint4 *)(input_bf16 + (uint64_t)head * IN_DIM) + i);
	const uint32_t w_base_s = (uint32_t)__cvta_generic_to_shared(weight_tile);
	constexpr uint32_t stage_bytes = OUTPUTS * STRIDE * 2u;
	auto issue_tile = [&](uint32_t stage, uint32_t k0)
	{
		const uint32_t dst_stage = w_base_s + stage * stage_bytes;
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < OUTPUTS * TILE_VECS; idx += 32u )
		{
			const uint32_t r = idx / TILE_VECS, c = idx % TILE_VECS;
			LmProjectCpAsync16(dst_stage + (r * STRIDE + c * 8u) * 2u, (const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_tile(0u, 0u);
	float total = 0.0f;
	uint32_t stage = 0u;
	for ( uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE )
	{
		if ( k0 + K_TILE < IN_DIM )
			issue_tile(stage ^ 1u, k0 + K_TILE);
		if ( k0 + K_TILE < IN_DIM )
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		__syncwarp();
		if ( threadIdx.x < OUTPUTS )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < K_TILE; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[stage][threadIdx.x] + k), total);
		}
		stage ^= 1u;
		if ( k0 + K_TILE < IN_DIM )
			__syncwarp();
	}
	if ( threadIdx.x < OUTPUTS )
	{
		const uint32_t col = first_output + threadIdx.x;
		float val = LmBf16ToFloat(LmFloatToBf16(total));
		if constexpr ( S::kQueryGate != 0u )
		{
			const float gate = LmBf16ToFloat(q_bf16[(uint64_t)head * S::kQueryHead + S::kNope + S::kRope + col]);
			val *= SparkMathSigmoidFast(gate);
		}
		output_bf16[(uint64_t)head * OUT_DIM + col] = LmFloatToBf16(val);
	}
	LmDependentRelease();
}

template<class S, uint32_t OUTPUTS, uint32_t ROW_TILE, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GlmKvBValueChainKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16,
	const uint16_t *__restrict__ q_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t IN_DIM = S::kLatent, OUT_DIM = S::kValue, VECTORS = IN_DIM / 8u, STRIDE = IN_DIM + 8u;
	__shared__ __align__(16) uint16_t weight_tile[OUTPUTS][STRIDE];
	__shared__ __align__(16) uint16_t input_tile[ROW_TILE][STRIDE];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint4 *weight = (const uint4 *)(weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM);
	LmDependentRelease();
	for ( uint32_t index = threadIdx.x; index < OUTPUTS * VECTORS; index += THREADS )
		*(uint4 *)(weight_tile[index / VECTORS] + (index % VECTORS) * 8u) = __ldg(weight + index);
	LmDependentWait();
	const uint32_t output = threadIdx.x % OUTPUTS, row_in_tile = threadIdx.x / OUTPUTS;
	for ( uint32_t r0 = 0u; r0 < rows; r0 += ROW_TILE )
	{
		const uint32_t cnt = rows - r0 < ROW_TILE ? rows - r0 : ROW_TILE;
		for ( uint32_t index = threadIdx.x; index < cnt * VECTORS; index += THREADS )
		{
			const uint32_t r = index / VECTORS, k = (index % VECTORS) * 8u;
			*(uint4 *)(input_tile[r] + k) = *(const uint4 *)(input_bf16 + ((uint64_t)(r0 + r) * heads + head) * IN_DIM + k);
		}
		__syncthreads();
		if ( row_in_tile < cnt )
		{
			float total = 0.0f;
			#pragma unroll 4
			for ( uint32_t k = 0u; k < IN_DIM; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile[row_in_tile] + k), *(const uint4 *)(weight_tile[output] + k), total);
			const uint32_t col = first_output + output;
			const uint32_t gr = r0 + row_in_tile;
			float val = LmBf16ToFloat(LmFloatToBf16(total));
			if constexpr ( S::kQueryGate != 0u )
			{
				const float gate = LmBf16ToFloat(q_bf16[((uint64_t)gr * heads + head) * S::kQueryHead + S::kNope + S::kRope + col]);
				val *= SparkMathSigmoidFast(gate);
			}
			output_bf16[((uint64_t)gr * heads + head) * OUT_DIM + col] = LmFloatToBf16(val);
		}
		if ( r0 + ROW_TILE < rows )
			__syncthreads();
	}
}

template<class S, uint32_t OUT_HEAD_STRIDE, uint32_t OUT_OFFSET>
__global__ __launch_bounds__(GLM_THREADS, 4)
static void GlmPrefillExtractRopeKernel(const uint16_t *__restrict__ q_bf16, uint16_t *__restrict__ out_bf16,
	const uint32_t *__restrict__ positions, const float *__restrict__ rope_table, uint32_t rows)
{
	static_assert(S::kRope == 64u, "extract rope packs 32 frequency pairs per warp");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t r = blockIdx.x * 2u + (warp >> 2u), head = warp & 3u;
	LmDependentWait();
	if ( r < rows )
	{
		const uint64_t in_base = ((uint64_t)r * 4u + head) * S::kQueryHead + S::kNope;
		const uint64_t out_base = ((uint64_t)r * 4u + head) * OUT_HEAD_STRIDE + OUT_OFFSET;
		const uint32_t lo = lane * 2u;
		float a = LmBf16ToFloat(q_bf16[in_base + lo]);
		float b = LmBf16ToFloat(q_bf16[in_base + lo + 1u]);
		const float angle = (float)positions[r] * (S::kRopeTable ? rope_table[lane] : __powf(S::kRopeTheta, -2.0f * (float)lane / (float)S::kRope));
		LmRopePairRounded<S::kRopeRoundedTerms ? LM_ROPE_ROUNDED_TERMS : LM_ROPE_FP32>(&a, &b, angle);
		*(uint32_t *)(out_bf16 + out_base + lo) = LmPackBf16Pair(a, b);
	}
	LmDependentRelease();
}

template<class S>
__global__ __launch_bounds__(128u, 2)
static void GlmKvBValueMmaGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16,
	const uint16_t *__restrict__ q_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t IN_DIM = S::kLatent, OUT_DIM = S::kValue, TILE_M = 64u, TILE_N = 64u, TILE_K = 64u;
	constexpr uint32_t WORDS = TILE_K / 8u, TILES = TILE_N / 8u, ROW_BYTES = TILE_K * 2u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	static_assert(IN_DIM % TILE_K == 0u && OUT_DIM % TILE_N == 0u, "value MMA gate tiles divide latent and value");
	__shared__ __align__(128) uint8_t smem[2u * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t first_output = blockIdx.x * TILE_N, head = blockIdx.y, first_row = blockIdx.z * TILE_M;
	const uint32_t lane = threadIdx.x & 31u, warp_row = (threadIdx.x >> 5u) * 16u;
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t ldm_a_row = warp_row + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint32_t ldm_b_row = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), ldm_b_col = (lane >> 3u) & 1u;
	const uint16_t *weight = weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	float acc[TILES][4];
	#pragma unroll
	for ( uint32_t t = 0u; t < TILES; ++t )
		acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
	LmDependentWait();
	auto issue_k = [&](uint32_t st_off, uint32_t k0)
	{
		const uint32_t stage_a = shared_base + st_off, stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TILE_M * WORDS; idx += 128u )
		{
			const uint32_t r = idx / WORDS, c = idx % WORDS;
			const uint32_t bytes = first_row + r < rows ? 16u : 0u;
			const uint16_t *src = bytes != 0u ? input_bf16 + ((uint64_t)(first_row + r) * heads + head) * IN_DIM + k0 + c * 8u : input_bf16;
			LmStreamCopy(stage_a + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, src, bytes);
		}
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TILE_N * WORDS; idx += 128u )
		{
			const uint32_t r = idx / WORDS, c = idx % WORDS;
			LmStreamCopy(stage_b + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, weight + (uint64_t)r * IN_DIM + k0 + c * 8u, 16u);
		}
		LmStreamCommit();
	};
	constexpr uint32_t K_TILES = IN_DIM / TILE_K;
	issue_k(0u, 0u);
	uint32_t cur_off = 0u;
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		if ( kt + 1u < K_TILES )
		{
			issue_k(cur_off ^ STAGE_BYTES, (kt + 1u) * TILE_K);
			LmStreamWait<1u>();
		}
		else
			LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_a = shared_base + cur_off, stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t a[4], bt[4];
			LmStreamLdMatrix4(a, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, ks * 2u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t t = 0u; t < TILES; t += 2u )
			{
				const uint32_t br = t * 8u + ldm_b_row;
				LmStreamLdMatrix4(bt, stage_b + br * ROW_BYTES + LmStreamSwizzle8(br, ks * 2u + ldm_b_col) * 16u);
				uint32_t b0[2] = { bt[0], bt[1] }, b1[2] = { bt[2], bt[3] };
				LmMmaBf16(acc[t], a, b0);
				LmMmaBf16(acc[t + 1u], a, b1);
			}
		}
		cur_off ^= STAGE_BYTES;
		if ( kt + 1u < K_TILES )
			__syncthreads();
	}
	#pragma unroll
	for ( uint32_t t = 0u; t < TILES; ++t )
		#pragma unroll
		for ( uint32_t reg = 0u; reg < 4u; reg += 2u )
		{
			const uint32_t r = first_row + warp_row + quad_row + (reg >> 1u) * 8u;
			const uint32_t col = first_output + t * 8u + quad * 2u;
			if ( r < rows )
			{
				float v0 = LmBf16ToFloat(LmFloatToBf16(acc[t][reg]));
				float v1 = LmBf16ToFloat(LmFloatToBf16(acc[t][reg + 1u]));
				if constexpr ( S::kQueryGate != 0u )
				{
					const uint32_t gp = *(const uint32_t *)(q_bf16 + ((uint64_t)r * heads + head) * S::kQueryHead + S::kNope + S::kRope + col);
					v0 *= SparkMathSigmoidFast(__uint_as_float(gp << 16u));
					v1 *= SparkMathSigmoidFast(__uint_as_float(gp & 0xffff0000u));
				}
				*(uint32_t *)(output_bf16 + ((uint64_t)r * heads + head) * OUT_DIM + col) = LmPackBf16Pair(v0, v1);
			}
		}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t K, uint32_t GROUPS, uint32_t TOP_GROUPS>
__global__ __launch_bounds__(THREADS, 1)
static void GlmGroupedTopkRouteSmallKernel(const float *__restrict__ scores, const float *__restrict__ selection_bias,
	float mixture_scale, uint32_t *__restrict__ fault, uint32_t rows, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t keys[LM_TOPK_SMALL_LIMIT], slots[LM_TOPK_SMALL_LIMIT];
	__shared__ uint32_t group_key[LM_TOPK_MAX_GROUPS];
	__shared__ uint32_t sel[LM_ROUTE_SMALL_MAX];
	LmDependentWait();
	const uint32_t per_group = EXPERTS / GROUPS;
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const uint64_t base = (uint64_t)row * EXPERTS;
		for ( uint32_t index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS )
		{
			keys[index] = index < EXPERTS ? LmTopkRouteKey(LmTopkScore<LM_TOPK_SCORE_SIGMOID>(scores, (const uint16_t *)0, base + index) +
				(selection_bias != 0 ? selection_bias[index] : 0.0f)) : 0u;
			slots[index] = index;
		}
		__syncthreads();
		for ( uint32_t group = threadIdx.x; group < GROUPS; group += THREADS )
		{
			uint32_t best = 0u, second = 0u;
			for ( uint32_t member = 0u; member < per_group; ++member )
			{
				const uint32_t value = keys[group * per_group + member];
				if ( value > best ) { second = best; best = value; }
				else if ( value > second ) second = value;
			}
			group_key[group] = LmTopkRouteKey(LmTopkValue(best) + LmTopkValue(second));
		}
		__syncthreads();
		for ( uint32_t group = threadIdx.x; group < GROUPS; group += THREADS )
		{
			uint32_t better = 0u;
			for ( uint32_t other = 0u; other < GROUPS; ++other )
				if ( group_key[other] > group_key[group] || (group_key[other] == group_key[group] && other < group) )
					++better;
			if ( better >= TOP_GROUPS )
				group_key[group] = 0xffffffffu;
		}
		__syncthreads();
		for ( uint32_t index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS )
			if ( index < EXPERTS && group_key[index / per_group] == 0xffffffffu )
				keys[index] = 0u;
		__syncthreads();
		LmTopkSelectRounds<THREADS, K>(keys, slots, EXPERTS);
		if ( threadIdx.x == 0u )
			LmTopkRouteFault(fault, keys[K - 1u]);
		if ( threadIdx.x < 32u )
		{
			const uint32_t picked = threadIdx.x < K ? slots[threadIdx.x] : 0u;
			const float value = threadIdx.x < K ? LmTopkScore<LM_TOPK_SCORE_SIGMOID>(scores, (const uint16_t *)0, base + picked) : 0.0f;
			if ( threadIdx.x < K )
			{
				route_expert[(uint64_t)row * K + threadIdx.x] = picked;
				sel[row * K + threadIdx.x] = picked;
			}
			float tree = value;
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			{
				const float other = __shfl_down_sync(0xffffffffu, tree, step);
				tree = threadIdx.x < step ? tree + other : tree;
			}
			const float total = __shfl_sync(0xffffffffu, tree, 0) + 1e-20f;
			if ( threadIdx.x < K )
				route_weight[(uint64_t)row * K + threadIdx.x] = (value / total) * mixture_scale;
		}
		__syncthreads();
	}
	LmRouteBuildSmallBody<THREADS, EXPERTS>(sel, rows * K, K, group_row_offset, route_packed_row, route_source_token,
		tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	LmDependentRelease();
}

static __global__ __launch_bounds__(32u, 1)
void GlmBuildTilePrefixKernel(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ prefix,
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

static __device__ __forceinline__ void GlmGroupedTileLookupGlobal(const uint32_t *__restrict__ group_row_offset,
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

template<class S, uint32_t ROWS>
static __device__ __forceinline__ void GlmMoeW1ActFp8Pass(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out,
	uint64_t exp_base, uint32_t ch, uint32_t r0, uint32_t count, uint32_t lane)
{
	constexpr uint32_t K = S::kHidden, INTER = 128u, FP8_CHUNKS = K / 16u, SCALES = K / 128u;
	uint32_t src_r[ROWS];
	float au[ROWS], ag[ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		src_r[r] = (ROWS <= 4u || r < count) ? __ldg(route_source_token + r0 + r) : 0u;
		au[r] = 0.0f;
		ag[r] = 0.0f;
	}
	const uint4 *w_up = (const uint4 *)(exp_w1 + (exp_base + ch) * K);
	const uint4 *w_gt = (const uint4 *)(exp_w1 + (exp_base + INTER + ch) * K);
	const float *s_up = exp_s1 + (exp_base + ch) * SCALES;
	const float *s_gt = exp_s1 + (exp_base + INTER + ch) * SCALES;
	if constexpr ( ROWS == 1u )
	{
		const uint16_t *ar = act + (uint64_t)src_r[0] * K;
		uint32_t c = lane;
		for ( ; c + 3u * 32u < FP8_CHUNKS; c += 4u * 32u )
		{
			uint4 wu[4], wg[4];
			float su[4], sgs[4];
			LmSkinnyFormat<LmFp8>::Activation a[4];
			#pragma unroll
			for ( uint32_t u = 0u; u < 4u; ++u )
			{
				const uint32_t ci = c + u * 32u, sgi = ci >> 3u;
				wu[u] = __ldcs(w_up + ci);
				wg[u] = __ldcs(w_gt + ci);
				su[u] = __ldg(s_up + sgi);
				sgs[u] = __ldg(s_gt + sgi);
				a[u] = LmSkinnyFormat<LmFp8>::Load(ar + ci * 16u);
			}
			#pragma unroll
			for ( uint32_t u = 0u; u < 4u; ++u )
			{
				au[0] = fmaf(su[u], LmSkinnyFormat<LmFp8>::Chunk(wu[u], a[u]), au[0]);
				ag[0] = fmaf(sgs[u], LmSkinnyFormat<LmFp8>::Chunk(wg[u], a[u]), ag[0]);
			}
		}
		for ( ; c < FP8_CHUNKS; c += 32u )
		{
			const uint32_t sg = c >> 3u;
			const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(ar + c * 16u);
			au[0] = fmaf(__ldg(s_up + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_up + c), a), au[0]);
			ag[0] = fmaf(__ldg(s_gt + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_gt + c), a), ag[0]);
		}
	}
	else
	{
		uint4 wu_cur = __ldcs(w_up + lane), wg_cur = __ldcs(w_gt + lane);
		float su_cur = __ldg(s_up + (lane >> 3u)), sg_cur = __ldg(s_gt + (lane >> 3u));
		for ( uint32_t c = lane; c < FP8_CHUNKS; c += 32u )
		{
			uint4 wu_nxt{}, wg_nxt{};
			float su_nxt = 0.0f, sg_nxt = 0.0f;
			if ( c + 32u < FP8_CHUNKS )
			{
				const uint32_t cn = c + 32u, sgn = cn >> 3u;
				wu_nxt = __ldcs(w_up + cn);
				wg_nxt = __ldcs(w_gt + cn);
				su_nxt = __ldg(s_up + sgn);
				sg_nxt = __ldg(s_gt + sgn);
			}
			const LmSkinnyFormat<LmFp8>::Decoded du = LmSkinnyFormat<LmFp8>::Decode(wu_cur);
			const LmSkinnyFormat<LmFp8>::Decoded dg = LmSkinnyFormat<LmFp8>::Decode(wg_cur);
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS; ++r )
				if ( ROWS <= 4u || r < count )
				{
					const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(act + (uint64_t)src_r[r] * K + c * 16u);
					float af[16];
					LmSkinnyUnpackBf16x8(a.low, af);
					LmSkinnyUnpackBf16x8(a.high, af + 8);
					au[r] = fmaf(su_cur, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(du, af), au[r]);
					ag[r] = fmaf(sg_cur, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(dg, af), ag[r]);
				}
			wu_cur = wu_nxt;
			wg_cur = wg_nxt;
			su_cur = su_nxt;
			sg_cur = sg_nxt;
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
				inter_out[(uint64_t)(r0 + r) * INTER + ch] = LmFloatToBf16(LmSiluOp{}(
					LmBf16ToFloat(LmFloatToBf16(ag[r])), LmBf16ToFloat(LmFloatToBf16(au[r]))));
		}
}

template<class S, uint32_t ROWS>
static __device__ __forceinline__ void GlmMoeW1ActBf16Pass(const uint8_t *__restrict__ exp_w1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out,
	uint64_t exp_base, uint32_t ch, uint32_t r0, uint32_t count, uint32_t lane)
{
	constexpr uint32_t K = S::kHidden, INTER = 128u, BF16_CHUNKS = K / 8u;
	uint32_t src_r[ROWS];
	float au[ROWS], ag[ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		src_r[r] = (ROWS <= 4u || r < count) ? __ldg(route_source_token + r0 + r) : 0u;
		au[r] = 0.0f;
		ag[r] = 0.0f;
	}
	const uint4 *w_up = (const uint4 *)(exp_w1 + (exp_base + ch) * K * 2u);
	const uint4 *w_gt = (const uint4 *)(exp_w1 + (exp_base + INTER + ch) * K * 2u);
	if constexpr ( ROWS == 1u )
	{
		const uint4 *ar = (const uint4 *)(act + (uint64_t)src_r[0] * K);
		uint32_t c = lane;
		for ( ; c + 3u * 32u < BF16_CHUNKS; c += 4u * 32u )
		{
			uint4 wu[4], wg[4], a[4];
			#pragma unroll
			for ( uint32_t u = 0u; u < 4u; ++u )
			{
				const uint32_t ci = c + u * 32u;
				wu[u] = __ldcs(w_up + ci);
				wg[u] = __ldcs(w_gt + ci);
				a[u] = __ldg(ar + ci);
			}
			#pragma unroll
			for ( uint32_t u = 0u; u < 4u; ++u )
			{
				au[0] = fmaf(1.0f, LmSkinnyBf16Chunk(wu[u], a[u], 0.0f), au[0]);
				ag[0] = fmaf(1.0f, LmSkinnyBf16Chunk(wg[u], a[u], 0.0f), ag[0]);
			}
		}
		for ( ; c < BF16_CHUNKS; c += 32u )
		{
			const uint4 a = __ldg(ar + c);
			au[0] = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w_up + c), a, 0.0f), au[0]);
			ag[0] = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w_gt + c), a, 0.0f), ag[0]);
		}
	}
	else
	{
		uint4 wu_cur = __ldcs(w_up + lane), wg_cur = __ldcs(w_gt + lane);
		for ( uint32_t c = lane; c < BF16_CHUNKS; c += 32u )
		{
			uint4 wu_nxt{}, wg_nxt{};
			if ( c + 32u < BF16_CHUNKS )
			{
				wu_nxt = __ldcs(w_up + c + 32u);
				wg_nxt = __ldcs(w_gt + c + 32u);
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS; ++r )
				if ( ROWS <= 4u || r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)src_r[r] * K) + c), af);
					au[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(wu_cur, af, 0.0f), au[r]);
					ag[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(wg_cur, af, 0.0f), ag[r]);
				}
			wu_cur = wu_nxt;
			wg_cur = wg_nxt;
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
				inter_out[(uint64_t)(r0 + r) * INTER + ch] = LmFloatToBf16(LmSiluOp{}(
					LmBf16ToFloat(LmFloatToBf16(ag[r])), LmBf16ToFloat(LmFloatToBf16(au[r]))));
		}
}

template<class S>
__global__ __launch_bounds__(256u, 2)
static void GlmMoeExactW1ActKernel(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token,
	uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t WARPS = 8u, INTER = 128u;
	constexpr bool IS_FP8 = S::kExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
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
			if constexpr ( IS_FP8 )
			{
				if ( count == 1u )
					GlmMoeW1ActFp8Pass<S, 1u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 1u, lane);
				else if ( count == 2u )
					GlmMoeW1ActFp8Pass<S, 2u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 2u, lane);
				else if ( count == 3u )
					GlmMoeW1ActFp8Pass<S, 3u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 3u, lane);
				else if ( count == 4u )
					GlmMoeW1ActFp8Pass<S, 4u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, 4u, lane);
				else
					GlmMoeW1ActFp8Pass<S, 8u>(exp_w1, exp_s1, act, route_source_token, inter_out, exp_base, ch, r0, count, lane);
			}
			else
			{
				if ( count == 1u )
					GlmMoeW1ActBf16Pass<S, 1u>(exp_w1, act, route_source_token, inter_out, exp_base, ch, r0, 1u, lane);
				else if ( count == 2u )
					GlmMoeW1ActBf16Pass<S, 2u>(exp_w1, act, route_source_token, inter_out, exp_base, ch, r0, 2u, lane);
				else if ( count == 3u )
					GlmMoeW1ActBf16Pass<S, 3u>(exp_w1, act, route_source_token, inter_out, exp_base, ch, r0, 3u, lane);
				else if ( count == 4u )
					GlmMoeW1ActBf16Pass<S, 4u>(exp_w1, act, route_source_token, inter_out, exp_base, ch, r0, 4u, lane);
				else
					GlmMoeW1ActBf16Pass<S, 8u>(exp_w1, act, route_source_token, inter_out, exp_base, ch, r0, count, lane);
			}
		}
	}
	LmDependentRelease();
}

template<class S, uint32_t RG>
static __device__ __forceinline__ void GlmMoeW2Apply(const uint16_t (*s_act)[128u], uint16_t *__restrict__ out_bf16,
	uint32_t col, uint32_t r0, uint32_t count, uint32_t lane, float sc,
	const typename LmSkinnyFormat<LmFp8>::Decoded &dec, uint4 w_bf16)
{
	constexpr uint32_t N = S::kHidden;
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
		if ( r < count )
		{
			float dot;
			if constexpr ( S::kExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
			{
				LmSkinnyFormat<LmFp8>::Activation a;
				a.low = ((const uint4 *)s_act[r])[2u * lane];
				a.high = ((const uint4 *)s_act[r])[2u * lane + 1u];
				dot = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, a), 0.0f);
			}
			else
			{
				dot = fmaf(1.0f, LmSkinnyBf16Chunk(w_bf16, ((const uint4 *)s_act[r])[lane], 0.0f), 0.0f);
				dot += __shfl_xor_sync(0xffffffffu, dot, 8u);
			}
			dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
			dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
			if ( lane == 0u )
				out_bf16[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(dot);
		}
}

template<class S, uint32_t RG, uint32_t C_ITERS>
__global__ __launch_bounds__(256u, 4)
static void GlmMoeExactW2Kernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ inter_in, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t INTER = 128u, N = S::kHidden;
	constexpr bool IS_FP8 = S::kExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t LANES = IS_FP8 ? 8u : 16u, COLS_PER_BLK = 256u / LANES;
	__shared__ __align__(16) uint16_t s_act[RG][INTER];
	const uint32_t grp = blockIdx.y, group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
	LmDependentWait();
	const uint32_t r_begin = __ldg(group_row_offset + grp), r_end = __ldg(group_row_offset + grp + 1u);
	if ( r_begin < r_end )
	{
		const uint32_t col_base = blockIdx.x * (COLS_PER_BLK * C_ITERS) + group, n_rows = r_end - r_begin;
		if ( n_rows <= RG )
		{
			if ( threadIdx.x < n_rows * (INTER / 8u) )
				((uint4 *)s_act)[threadIdx.x] = __ldg((const uint4 *)(inter_in + (uint64_t)r_begin * INTER) + threadIdx.x);
			__syncthreads();
			const uint64_t wr0 = (uint64_t)grp * N + col_base;
			uint4 w_cur = IS_FP8 ? __ldcs((const uint4 *)(exp_w2 + wr0 * INTER) + lane) :
				__ldcs((const uint4 *)((const uint16_t *)exp_w2 + wr0 * INTER) + lane);
			float sc_cur = IS_FP8 ? __ldg(exp_s2 + wr0) : 1.0f;
			#pragma unroll
			for ( uint32_t ci = 0u; ci < C_ITERS; ++ci )
			{
				const uint32_t col = col_base + ci * COLS_PER_BLK;
				uint4 w_nxt{};
				float sc_nxt = 1.0f;
				if ( ci + 1u < C_ITERS )
				{
					const uint64_t wrn = (uint64_t)grp * N + col + COLS_PER_BLK;
					w_nxt = IS_FP8 ? __ldcs((const uint4 *)(exp_w2 + wrn * INTER) + lane) :
						__ldcs((const uint4 *)((const uint16_t *)exp_w2 + wrn * INTER) + lane);
					if constexpr ( IS_FP8 )
						sc_nxt = __ldg(exp_s2 + wrn);
				}
				LmSkinnyFormat<LmFp8>::Decoded dec{};
				if constexpr ( IS_FP8 )
					dec = LmSkinnyFormat<LmFp8>::Decode(w_cur);
				GlmMoeW2Apply<S, RG>(s_act, out_bf16, col, r_begin, n_rows, lane, sc_cur, dec, w_cur);
				w_cur = w_nxt;
				sc_cur = sc_nxt;
			}
		}
		else
		{
			#pragma unroll
			for ( uint32_t ci = 0u; ci < C_ITERS; ++ci )
			{
				const uint32_t col = col_base + ci * COLS_PER_BLK;
				const uint64_t w_row = (uint64_t)grp * N + col;
				float sc = 1.0f;
				LmSkinnyFormat<LmFp8>::Decoded dec{};
				uint4 w_bf16{};
				if constexpr ( IS_FP8 )
				{
					sc = __ldg(exp_s2 + w_row);
					dec = LmSkinnyFormat<LmFp8>::Decode(__ldcs((const uint4 *)(exp_w2 + w_row * INTER) + lane));
				}
				else
					w_bf16 = __ldcs((const uint4 *)((const uint16_t *)exp_w2 + w_row * INTER) + lane);
				for ( uint32_t r0 = r_begin; r0 < r_end; r0 += RG )
				{
					const uint32_t count = r_end - r0 < RG ? r_end - r0 : RG;
					if ( threadIdx.x < count * (INTER / 8u) )
						((uint4 *)s_act)[threadIdx.x] = __ldg((const uint4 *)(inter_in + (uint64_t)r0 * INTER) + threadIdx.x);
					__syncthreads();
					GlmMoeW2Apply<S, RG>(s_act, out_bf16, col, r0, count, lane, sc, dec, w_bf16);
					if ( r0 + RG < r_end || ci + 1u < C_ITERS )
						__syncthreads();
				}
			}
		}
	}
	LmDependentRelease();
}

template<class S, uint32_t MAX_R>
__global__ __launch_bounds__(LM_SKINNY_THREADS)
static void GlmSharedW1ActTileKernel(const uint8_t *__restrict__ shared_w1, const float *__restrict__ shared_s1,
	const uint16_t *__restrict__ act, uint16_t *__restrict__ shared_inter_out, uint32_t rows)
{
	constexpr bool LIN_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t K = S::kHidden, INTER = 128u, WARPS = LM_SKINNY_SPLIT_WARPS;
	constexpr uint32_t CHUNKS = LIN_FP8 ? (K / 16u) : (K / 8u), SCALES = K / 128u;
	__shared__ float partial[4][MAX_R][WARPS];
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t ch0 = blockIdx.x * 2u, row_base = blockIdx.y * MAX_R;
	const uint32_t cnt = row_base < rows ? (rows - row_base < MAX_R ? rows - row_base : MAX_R) : 0u;
	const uint32_t first = CHUNKS * warp / WARPS, last = CHUNKS * (warp + 1u) / WARPS;
	float u0[MAX_R], g0[MAX_R], u1[MAX_R], g1[MAX_R];
	#pragma unroll
	for ( uint32_t r = 0u; r < MAX_R; ++r )
		u0[r] = g0[r] = u1[r] = g1[r] = 0.0f;
	LmDependentWait();
	if ( cnt != 0u )
	{
		const bool reuse = rows > MAX_R;
		if constexpr ( LIN_FP8 )
		{
			const uint4 *wu0 = (const uint4 *)(shared_w1 + (uint64_t)ch0 * K), *wg0 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0) * K);
			const uint4 *wu1 = (const uint4 *)(shared_w1 + (uint64_t)(ch0 + 1u) * K), *wg1 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0 + 1u) * K);
			const float *su0 = shared_s1 + (uint64_t)ch0 * SCALES, *sg0 = shared_s1 + (uint64_t)(INTER + ch0) * SCALES;
			const float *su1 = shared_s1 + (uint64_t)(ch0 + 1u) * SCALES, *sg1 = shared_s1 + (uint64_t)(INTER + ch0 + 1u) * SCALES;
			for ( uint32_t c = first + lane; c < last; c += LM_WARP_LANES )
			{
				const uint32_t sg = c >> 3u;
				const LmSkinnyFormat<LmFp8>::Decoded du0 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(wu0 + c) : __ldcs(wu0 + c));
				const LmSkinnyFormat<LmFp8>::Decoded dg0 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(wg0 + c) : __ldcs(wg0 + c));
				const LmSkinnyFormat<LmFp8>::Decoded du1 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(wu1 + c) : __ldcs(wu1 + c));
				const LmSkinnyFormat<LmFp8>::Decoded dg1 = LmSkinnyFormat<LmFp8>::Decode(reuse ? __ldg(wg1 + c) : __ldcs(wg1 + c));
				const float fsu0 = __ldg(su0 + sg), fsg0 = __ldg(sg0 + sg), fsu1 = __ldg(su1 + sg), fsg1 = __ldg(sg1 + sg);
				#pragma unroll
				for ( uint32_t r = 0u; r < MAX_R; ++r )
					if ( r < cnt )
					{
						const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(act + (uint64_t)(row_base + r) * K + c * 16u);
						float af[16];
						LmSkinnyUnpackBf16x8(a.low, af);
						LmSkinnyUnpackBf16x8(a.high, af + 8);
						u0[r] = fmaf(fsu0, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(du0, af), u0[r]);
						g0[r] = fmaf(fsg0, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(dg0, af), g0[r]);
						u1[r] = fmaf(fsu1, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(du1, af), u1[r]);
						g1[r] = fmaf(fsg1, LmSkinnyFormat<LmFp8>::ChunkDecodedUnpacked(dg1, af), g1[r]);
					}
			}
		}
		else
		{
			const uint4 *wu0 = (const uint4 *)(shared_w1 + (uint64_t)ch0 * K * 2u), *wg0 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0) * K * 2u);
			const uint4 *wu1 = (const uint4 *)(shared_w1 + (uint64_t)(ch0 + 1u) * K * 2u), *wg1 = (const uint4 *)(shared_w1 + (uint64_t)(INTER + ch0 + 1u) * K * 2u);
			for ( uint32_t c = first + lane; c < last; c += LM_WARP_LANES )
			{
				const uint4 w_u0 = reuse ? __ldg(wu0 + c) : __ldcs(wu0 + c);
				const uint4 w_g0 = reuse ? __ldg(wg0 + c) : __ldcs(wg0 + c);
				const uint4 w_u1 = reuse ? __ldg(wu1 + c) : __ldcs(wu1 + c);
				const uint4 w_g1 = reuse ? __ldg(wg1 + c) : __ldcs(wg1 + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < MAX_R; ++r )
					if ( r < cnt )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)(row_base + r) * K) + c), af);
						u0[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w_u0, af, 0.0f), u0[r]);
						g0[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w_g0, af, 0.0f), g0[r]);
						u1[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w_u1, af, 0.0f), u1[r]);
						g1[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w_g1, af, 0.0f), g1[r]);
					}
			}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
			if ( r < cnt )
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
		if ( threadIdx.x < cnt * 2u )
		{
			const uint32_t r = threadIdx.x >> 1u, k = threadIdx.x & 1u;
			float tu = 0.0f, tg = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < WARPS; ++w )
			{
				tu += partial[k * 2u][r][w];
				tg += partial[k * 2u + 1u][r][w];
			}
			shared_inter_out[(uint64_t)(row_base + r) * INTER + ch0 + k] = LmFloatToBf16(LmSiluOp{}(
				LmBf16ToFloat(LmFloatToBf16(tg)), LmBf16ToFloat(LmFloatToBf16(tu))));
		}
	}
	LmDependentRelease();
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void GlmMoeMmaW1ActFp8Kernel(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ act, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t K = S::kHidden, INTER = 128u, TILE_N = 64u, TILE_K = 128u, N_TILES = INTER / TILE_N, K_TILES = K / TILE_K;
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
	const uint32_t total_tiles = __ldg(tile_prefix + S::kExperts);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		GlmGroupedTileLookupGlobal(group_row_offset, tile_prefix, S::kExperts, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
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
		float tu[MF][NF][4], tg[MF][NF][4], pu[MF][NF][4], pg[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tu[i][j][e] = tg[i][j][e] = pu[i][j][e] = pg[i][j][e] = 0.0f;
		auto run_mma_k = [&](uint32_t kt, uint32_t cur_off)
		{
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
			const uint32_t stage_a = shared_base + cur_off;
			const uint32_t stage_bu = stage_a + A_BYTES;
			const uint32_t stage_bg = stage_bu + B_BYTES;
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				uint32_t af0[MF][4], af1[MF][4];
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					const uint32_t row = ldm_a_row + i * 16u;
					LmStreamLdMatrix4(af0[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
					LmStreamLdMatrix4(af1[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				}
				if constexpr ( NF >= 2u )
				{
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; j += 2u )
					{
						uint32_t b0[4], b1[4];
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_bu, warp_n * WARP_N + j * 8u, step, lane, b0, b1);
						uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
						uint32_t b1l[2] = { b1[0], b1[1] }, b1h[2] = { b1[2], b1[3] };
						#pragma unroll
						for ( uint32_t i = 0u; i < MF; ++i )
						{
							LmMmaBf16(pu[i][j], af0[i], b0l);
							LmMmaBf16(pu[i][j + 1u], af0[i], b1l);
							LmMmaBf16(pu[i][j], af1[i], b0h);
							LmMmaBf16(pu[i][j + 1u], af1[i], b1h);
						}
					}
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; j += 2u )
					{
						uint32_t b0[4], b1[4];
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_bg, warp_n * WARP_N + j * 8u, step, lane, b0, b1);
						uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
						uint32_t b1l[2] = { b1[0], b1[1] }, b1h[2] = { b1[2], b1[3] };
						#pragma unroll
						for ( uint32_t i = 0u; i < MF; ++i )
						{
							LmMmaBf16(pg[i][j], af0[i], b0l);
							LmMmaBf16(pg[i][j + 1u], af0[i], b1l);
							LmMmaBf16(pg[i][j], af1[i], b0h);
							LmMmaBf16(pg[i][j + 1u], af1[i], b1h);
						}
					}
				}
				else
				{
					uint32_t bu[4], bg[4];
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bu, warp_n * WARP_N, step, lane, bu);
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bg, warp_n * WARP_N, step, lane, bg);
					uint32_t bu0[2] = { bu[0], bu[1] }, bu1[2] = { bu[2], bu[3] };
					uint32_t bg0[2] = { bg[0], bg[1] }, bg1[2] = { bg[2], bg[3] };
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
					{
						LmMmaBf16(pu[i][0], af0[i], bu0);
						LmMmaBf16(pg[i][0], af0[i], bg0);
						LmMmaBf16(pu[i][0], af1[i], bu1);
						LmMmaBf16(pg[i][0], af1[i], bg1);
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
		};
		if constexpr ( STAGES == 2u )
		{
			uint4 bu_reg[B_PER_T], bg_reg[B_PER_T];
			auto issue_a = [&](uint32_t st_off)
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
				LmStreamCommit();
			};
			auto prefetch_b = [&]()
			{
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					bu_reg[q] = __ldcs((const uint4 *)(bu_base + q * b_ptr_step));
					bg_reg[q] = __ldcs((const uint4 *)(bg_base + q * b_ptr_step));
				}
				bu_base += TILE_K;
				bg_base += TILE_K;
			};
			auto store_b = [&](uint32_t st_off)
			{
				uint8_t *dst = smem + st_off;
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					const uint32_t r = b_row0 + q * B_STEP;
					const uint32_t sw = LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0);
					*(uint4 *)(dst + A_BYTES + sw) = bu_reg[q];
					*(uint4 *)(dst + A_BYTES + B_BYTES + sw) = bg_reg[q];
				}
			};
			uint32_t sp_off = 0u, sc_off = 0u;
			issue_a(sp_off);
			prefetch_b();
			store_b(sp_off);
			sp_off ^= STAGE_BYTES;
			for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
			{
				LmStreamWait<0u>();
				__syncthreads();
				if ( kt + 1u < K_TILES )
				{
					issue_a(sp_off);
					prefetch_b();
				}
				run_mma_k(kt, sc_off);
				if ( kt + 1u < K_TILES )
				{
					store_b(sp_off);
					sp_off ^= STAGE_BYTES;
				}
				sc_off ^= STAGE_BYTES;
			}
		}
		else
		{
			auto issue_k = [&](uint32_t st_off)
			{
				const uint32_t stage = shared_base + st_off;
				#pragma unroll
				for ( uint32_t q = 0u; q < A_PER_T; ++q )
				{
					const uint32_t r = a_row0 + q * A_STEP;
					LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
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
			uint32_t sp_off = 0u, sc_off = 0u, kp = 0u;
			#pragma unroll
			for ( uint32_t p = 0u; p < STAGES - 1u; ++p )
			{
				if ( kp < K_TILES )
				{
					issue_k(sp_off);
					++kp;
					sp_off += STAGE_BYTES;
					if ( sp_off == STAGES * STAGE_BYTES )
						sp_off = 0u;
				}
				LmStreamCommit();
			}
			for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
			{
				LmStreamWait<STAGES - 2u>();
				__syncthreads();
				if ( kp < K_TILES )
				{
					issue_k(sp_off);
					++kp;
					sp_off += STAGE_BYTES;
					if ( sp_off == STAGES * STAGE_BYTES )
						sp_off = 0u;
				}
				LmStreamCommit();
				run_mma_k(kt, sc_off);
				sc_off += STAGE_BYTES;
				if ( sc_off == STAGES * STAGE_BYTES )
					sc_off = 0u;
			}
		}
		LmStreamWait<0u>();
		const LmSiluOp op{};
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

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void GlmMoeMmaW1ActBf16Kernel(const uint16_t *__restrict__ exp_w1, const uint16_t *__restrict__ act,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out)
{
	constexpr uint32_t K = S::kHidden, INTER = 128u, TILE_N = 64u, TILE_K = 64u, N_TILES = INTER / TILE_N, K_TILES = K / TILE_K;
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
	const uint32_t total_tiles = __ldg(tile_prefix + S::kExperts);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		GlmGroupedTileLookupGlobal(group_row_offset, tile_prefix, S::kExperts, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
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
		auto run_mma_k = [&](uint32_t stage_s)
		{
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
		};
		if constexpr ( STAGES == 2u )
		{
			uint4 bu_w[B_PER_T], bg_w[B_PER_T];
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
				LmStreamCommit();
			};
			auto prefetch_b = [&]()
			{
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					bu_w[q] = __ldcs((const uint4 *)(bu_base + q * b_step));
					bg_w[q] = __ldcs((const uint4 *)(bg_base + q * b_step));
				}
				bu_base += TILE_K;
				bg_base += TILE_K;
			};
			auto store_b = [&](uint32_t st_off)
			{
				uint8_t *dst = smem + st_off;
				#pragma unroll
				for ( uint32_t q = 0u; q < B_PER_T; ++q )
				{
					const uint32_t r = row0 + q * ROW_STEP;
					const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
					*(uint4 *)(dst + A_BYTES + sw) = bu_w[q];
					*(uint4 *)(dst + A_BYTES + B_BYTES + sw) = bg_w[q];
				}
			};
			uint32_t sp_off = 0u, sc_off = 0u;
			issue_a(sp_off);
			prefetch_b();
			store_b(sp_off);
			sp_off ^= STAGE_BYTES;
			for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
			{
				LmStreamWait<0u>();
				__syncthreads();
				const uint32_t stage_s = shared_base + sc_off;
				if ( kt + 1u < K_TILES )
				{
					issue_a(sp_off);
					prefetch_b();
				}
				run_mma_k(stage_s);
				if ( kt + 1u < K_TILES )
				{
					store_b(sp_off);
					sp_off ^= STAGE_BYTES;
				}
				sc_off ^= STAGE_BYTES;
			}
		}
		else
		{
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
						LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
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
			uint32_t sp_off = 0u, sc_off = 0u, kp = 0u;
			#pragma unroll
			for ( uint32_t p = 0u; p < STAGES - 1u; ++p )
			{
				if ( kp < K_TILES )
				{
					issue_k(sp_off);
					++kp;
					sp_off += STAGE_BYTES;
					if ( sp_off == STAGES * STAGE_BYTES )
						sp_off = 0u;
				}
				LmStreamCommit();
			}
			for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
			{
				LmStreamWait<STAGES - 2u>();
				__syncthreads();
				if ( kp < K_TILES )
				{
					issue_k(sp_off);
					++kp;
					sp_off += STAGE_BYTES;
					if ( sp_off == STAGES * STAGE_BYTES )
						sp_off = 0u;
				}
				LmStreamCommit();
				run_mma_k(shared_base + sc_off);
				sc_off += STAGE_BYTES;
				if ( sc_off == STAGES * STAGE_BYTES )
					sc_off = 0u;
			}
		}
		LmStreamWait<0u>();
		const LmSiluOp op{};
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

template<class S, uint32_t TILE_M, uint32_t WARPS_M, bool STREAM_OUT = true>
__global__ __launch_bounds__(256u, 1)
static void GlmMoeMmaW2K128Fp8Kernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ inter_in, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ row_tile_prefix,
	uint16_t *__restrict__ out_bf16, uint32_t col_tile_begin, uint32_t slice_n_tiles)
{
	constexpr uint32_t K = 128u, N = S::kHidden, TILE_N = 128u;
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
	const uint32_t slice_cols = slice_n_tiles * TILE_N;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(row_tile_prefix + S::kExperts) * slice_n_tiles;
	auto issue_tile = [&](uint32_t t_idx, uint32_t st_off, uint32_t &grp_o, uint32_t &rb_o, uint32_t &rl_o, uint32_t &nt_o)
	{
		uint32_t dummy_nt = 0u;
		GlmGroupedTileLookupGlobal(group_row_offset, row_tile_prefix, S::kExperts, t_idx / slice_n_tiles, TILE_M, 1u, grp_o, rb_o, rl_o, dummy_nt);
		nt_o = t_idx % slice_n_tiles;
		const uint32_t stage = shared_base + st_off, col_base = (col_tile_begin + nt_o) * TILE_N;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = a_row0 + q * A_STEP;
			const uint32_t packed = rb_o + r, valid = packed < rl_o ? 1u : 0u;
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
		const uint32_t col_base = (col_tile_begin + cur_nt) * TILE_N;
		const uint32_t out_col_base = cur_nt * TILE_N;
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
			uint32_t af0[MF][4], af1[MF][4];
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				LmStreamLdMatrix4(af0[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
				LmStreamLdMatrix4(af1[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
			}
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; j += 2u )
			{
				uint32_t b0[4], b1[4];
				LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b0, b1);
				uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
				uint32_t b1l[2] = { b1[0], b1[1] }, b1h[2] = { b1[2], b1[3] };
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					LmMmaBf16(acc[i][j], af0[i], b0l);
					LmMmaBf16(acc[i][j + 1u], af0[i], b1l);
					LmMmaBf16(acc[i][j], af1[i], b0h);
					LmMmaBf16(acc[i][j + 1u], af1[i], b1h);
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
					const uint32_t c = out_col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < cur_rl )
					{
						const uint16_t lo = LmFloatToBf16(fmaf(acc[i][j][e], sc[j][0], 0.0f));
						const uint16_t hi = LmFloatToBf16(fmaf(acc[i][j][e + 1u], sc[j][1], 0.0f));
						const uint32_t packed_pair = (uint32_t)lo | ((uint32_t)hi << 16u);
						uint32_t *dst = (uint32_t *)(out_bf16 + (uint64_t)r * slice_cols + c);
						if constexpr ( STREAM_OUT )
							__stcs(dst, packed_pair);
						else
							*dst = packed_pair;
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

template<class S, uint32_t TILE_M, uint32_t WARPS_M, bool STREAM_OUT = true>
__global__ __launch_bounds__(256u, 1)
static void GlmMoeMmaW2K128Bf16Kernel(const uint16_t *__restrict__ exp_w2, const uint16_t *__restrict__ inter_in,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ row_tile_prefix, uint16_t *__restrict__ out_bf16,
	uint32_t col_tile_begin, uint32_t slice_n_tiles)
{
	constexpr uint32_t K = 128u, N = S::kHidden, TILE_N = 128u, TILE_K = 64u;
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
	const uint32_t slice_cols = slice_n_tiles * TILE_N;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(row_tile_prefix + S::kExperts) * slice_n_tiles;
	auto issue_half = [&](uint32_t grp, uint32_t rb, uint32_t rl, uint32_t nt, uint32_t k_off, uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off, col_base = (col_tile_begin + nt) * TILE_N;
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
	uint32_t cur_grp, cur_rb, cur_rl, cur_nt, dummy_nt = 0u;
	GlmGroupedTileLookupGlobal(group_row_offset, row_tile_prefix, S::kExperts, blockIdx.x / slice_n_tiles, TILE_M, 1u, cur_grp, cur_rb, cur_rl, dummy_nt);
	cur_nt = blockIdx.x % slice_n_tiles;
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
			const uint32_t nt_idx = tile + gridDim.x;
			GlmGroupedTileLookupGlobal(group_row_offset, row_tile_prefix, S::kExperts, nt_idx / slice_n_tiles, TILE_M, 1u, nxt_grp, nxt_rb, nxt_rl, dummy_nt);
			nxt_nt = nt_idx % slice_n_tiles;
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
		const uint32_t out_col_base = cur_nt * TILE_N;
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = cur_rb + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = out_col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < cur_rl )
					{
						const uint32_t packed_pair = (uint32_t)LmFloatToBf16(acc[i][j][e]) | ((uint32_t)LmFloatToBf16(acc[i][j][e + 1u]) << 16u);
						uint32_t *dst = (uint32_t *)(out_bf16 + (uint64_t)r * slice_cols + c);
						if constexpr ( STREAM_OUT )
							__stcs(dst, packed_pair);
						else
							*dst = packed_pair;
					}
				}
		cur_grp = nxt_grp;
		cur_rb = nxt_rb;
		cur_rl = nxt_rl;
		cur_nt = nxt_nt;
	}
	LmDependentRelease();
}

template<uint32_t TOPK>
__global__ __launch_bounds__(128u, 4)
static void GlmMoeFinalizeAddSliceKernel(const uint16_t *__restrict__ packed_slice_bf16,
	const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight,
	const uint16_t *__restrict__ addend_bf16, uint16_t *__restrict__ output_bf16,
	uint32_t tokens, uint32_t col_offset, uint32_t slice_cols, uint32_t total_cols)
{
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t token = blockIdx.x * 4u + warp;
	if ( token >= tokens )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t my_pr = lane < TOPK ? __ldg(packed_row_of_token_route + (uint64_t)token * TOPK + lane) : 0u;
	const float my_wt = lane < TOPK ? __ldg(route_weight + (uint64_t)token * TOPK + lane) : 0.0f;
	for ( uint32_t elem = lane * 8u; elem < slice_cols; elem += 32u * 8u )
	{
		float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, add[8];
		#pragma unroll
		for ( uint32_t route = 0u; route < TOPK; ++route )
		{
			const uint32_t pr = __shfl_sync(0xffffffffu, my_pr, route);
			const float wt = __shfl_sync(0xffffffffu, my_wt, route);
			LmFmaBf16x8(total, __ldcs((const uint4 *)(packed_slice_bf16 + (uint64_t)pr * slice_cols + elem)), wt);
		}
		LmLoadBf16x8AsFloat(__ldcs((const uint4 *)(addend_bf16 + (uint64_t)token * total_cols + col_offset + elem)), add);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			total[k] = SparkMathRoundBf16(total[k]) + add[k];
		*(uint4 *)(output_bf16 + (uint64_t)token * total_cols + col_offset + elem) = LmPackUint4FloatToBf16(total);
	}
	LmDependentRelease();
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void GlmPrefillQkvIndexMmaKernel(const uint16_t *__restrict__ act,
	const uint8_t *__restrict__ qa_w, const float *__restrict__ qa_s, uint32_t q_count,
	const uint8_t *__restrict__ kva_w, const float *__restrict__ kva_s, uint32_t kv_count,
	const uint8_t *__restrict__ ik_w, const float *__restrict__ ik_s, uint32_t with_ik,
	uint16_t *__restrict__ proj_local, uint32_t proj_width, uint32_t query_max,
	uint16_t *__restrict__ index_key, uint32_t rows)
{
	constexpr bool IS_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t K = S::kHidden, TILE_N = 64u, TILE_K = IS_FP8 ? 128u : 64u, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t B_ROW_BYTES = IS_FP8 ? TILE_K : TILE_K * 2u, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u;
	constexpr uint32_t B_PER_T = (TILE_N * B_ROW_CHUNKS) / 256u, B_STEP = 256u / B_ROW_CHUNKS;
	constexpr uint64_t W_ROW_BYTES = IS_FP8 ? (uint64_t)K : (uint64_t)K * 2u;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t qt = (q_count + 63u) / 64u, kvt = (kv_count + 63u) / 64u;
	const uint8_t *w_base;
	const float *s_base;
	uint16_t *dst;
	uint32_t col_base, col_limit, dst_stride, dst_off;
	if ( blockIdx.x < qt )
	{
		w_base = qa_w; s_base = qa_s; dst = proj_local;
		col_base = blockIdx.x * 64u; col_limit = q_count; dst_stride = proj_width; dst_off = 0u;
	}
	else if ( blockIdx.x < qt + kvt )
	{
		w_base = kva_w; s_base = kva_s; dst = proj_local;
		col_base = (blockIdx.x - qt) * 64u; col_limit = kv_count; dst_stride = proj_width; dst_off = query_max;
	}
	else
	{
		w_base = ik_w; s_base = ik_s; dst = index_key;
		col_base = (blockIdx.x - qt - kvt) * 64u; col_limit = S::kIndexDim; dst_stride = S::kIndexDim; dst_off = 0u;
	}
	const uint32_t row_base = blockIdx.y * TILE_M, row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	if ( with_ik == 0u && blockIdx.x >= qt + kvt )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *a_src[A_PER_T];
	uint32_t a_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
		const uint32_t valid = chunk < A_CHUNKS && row_base + r < row_limit ? 1u : 0u;
		a_src[q] = act + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	const uint8_t *b_src[B_PER_T];
	uint32_t b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t bc = col_base + b_row0 + q * B_STEP;
		const uint32_t valid = bc < col_limit ? 1u : 0u;
		b_src[q] = w_base + (uint64_t)(valid != 0u ? bc : 0u) * W_ROW_BYTES + b_col0 * 16u;
		b_mask |= valid << q;
	}
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			if ( chunk < A_CHUNKS )
			{
				const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
				LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
				a_src[q] += TILE_K;
			}
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = b_row0 + q * B_STEP;
			const uint32_t sw = IS_FP8 ? LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0) :
				(r * B_ROW_BYTES + LmStreamSwizzle8(r, b_col0) * 16u);
			LmStreamCopy(stage + A_BYTES + sw, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += IS_FP8 ? TILE_K : TILE_K * 2u;
		}
	};
	float acc[MF][NF][4], part[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
				acc[i][j][e] = part[i][j][e] = 0.0f;
	uint32_t sp_off = 0u, sc_off = 0u, kp = 0u;
	#pragma unroll
	for ( uint32_t p = 0u; p < STAGES - 1u; ++p )
	{
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			++kp;
			sp_off += STAGE_BYTES;
			if ( sp_off == STAGES * STAGE_BYTES )
				sp_off = 0u;
		}
		LmStreamCommit();
	}
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			++kp;
			sp_off += STAGE_BYTES;
			if ( sp_off == STAGES * STAGE_BYTES )
				sp_off = 0u;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off, stage_b = stage_a + A_BYTES;
		if constexpr ( IS_FP8 )
		{
			float sc[NF][2];
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
				const uint32_t c0 = col < col_limit ? col : 0u, c1 = col + 1u < col_limit ? col + 1u : 0u;
				sc[j][0] = __ldg(s_base + (uint64_t)c0 * K_TILES + kt);
				sc[j][1] = __ldg(s_base + (uint64_t)c1 * K_TILES + kt);
			}
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				uint32_t af0[MF][4], af1[MF][4];
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					const uint32_t row = ldm_a_row + i * 16u;
					LmStreamLdMatrix4(af0[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
					LmStreamLdMatrix4(af1[i], stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				}
				if constexpr ( NF >= 2u )
				{
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; j += 2u )
					{
						uint32_t b0[4], b1[4];
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b0, b1);
						uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
						uint32_t b1l[2] = { b1[0], b1[1] }, b1h[2] = { b1[2], b1[3] };
						#pragma unroll
						for ( uint32_t i = 0u; i < MF; ++i )
						{
							LmMmaBf16(part[i][j], af0[i], b0l);
							LmMmaBf16(part[i][j + 1u], af0[i], b1l);
							LmMmaBf16(part[i][j], af1[i], b0h);
							LmMmaBf16(part[i][j + 1u], af1[i], b1h);
						}
					}
				}
				else
				{
					uint32_t b0[4];
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_b, warp_n * WARP_N, step, lane, b0);
					uint32_t b0l[2] = { b0[0], b0[1] }, b0h[2] = { b0[2], b0[3] };
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
					{
						LmMmaBf16(part[i][0], af0[i], b0l);
						LmMmaBf16(part[i][0], af1[i], b0h);
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
						acc[i][j][e] = fmaf(part[i][j][e], sc[j][e & 1u], acc[i][j][e]);
						part[i][j][e] = 0.0f;
					}
		}
		else
		{
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				uint32_t af[MF][4], bf[NF][2];
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					const uint32_t ar = ldm_a_row + i * 16u, ac = ks * 2u + ldm_a_col;
					LmStreamLdMatrix4(af[i], stage_a + ar * A_ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
				}
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					const uint32_t br = warp_n * WARP_N + j * 8u + (lane & 7u), bc = ks * 2u + ((lane >> 3u) & 1u);
					LmStreamLdMatrix2(bf[j], stage_b + br * B_ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
				}
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
						LmMmaBf16(acc[i][j], af[i], bf[j]);
			}
		}
		sc_off += STAGE_BYTES;
		if ( sc_off == STAGES * STAGE_BYTES )
			sc_off = 0u;
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t r0 = row_base + warp_m * WARP_M + i * 16u + quad_row, r1 = r0 + 8u;
			const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
			if ( col < col_limit )
			{
				if ( r0 < row_limit )
					*(uint32_t *)(dst + (uint64_t)r0 * dst_stride + dst_off + col) = LmPackBf16Pair(acc[i][j][0], acc[i][j][1]);
				if ( r1 < row_limit )
					*(uint32_t *)(dst + (uint64_t)r1 * dst_stride + dst_off + col) = LmPackBf16Pair(acc[i][j][2], acc[i][j][3]);
			}
		}
	LmDependentRelease();
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(128u, 2)
static void GlmPrefillIndexHeadMmaKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ act,
	float *__restrict__ output_f32, uint32_t rows)
{
	constexpr uint32_t K = S::kHidden, N = S::kIndexHeads, TILE_K = 64u, K_TILES = K / TILE_K, STAGES = 3u;
	constexpr uint32_t WARPS_N = 4u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / 128u, B_PER_T = (N * ROW_CHUNKS) / 128u;
	constexpr uint32_t ROW_STEP = 128u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t row_base = blockIdx.x * TILE_M, row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	const uint16_t *a_src[A_PER_T];
	uint32_t a_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t r = row0 + q * ROW_STEP;
		const uint32_t valid = row_base + r < row_limit ? 1u : 0u;
		a_src[q] = act + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + col0 * 8u;
		a_mask |= valid << q;
	}
	const uint16_t *b_base = weight + (uint64_t)row0 * K + col0 * 8u;
	const uint64_t b_step = (uint64_t)ROW_STEP * K;
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, b_base + q * b_step, 16u);
		}
		b_base += TILE_K;
	};
	float acc[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	uint32_t sp_off = 0u, sc_off = 0u, kp = 0u;
	#pragma unroll
	for ( uint32_t p = 0u; p < STAGES - 1u; ++p )
	{
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			++kp;
			sp_off += STAGE_BYTES;
			if ( sp_off == STAGES * STAGE_BYTES )
				sp_off = 0u;
		}
		LmStreamCommit();
	}
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			++kp;
			sp_off += STAGE_BYTES;
			if ( sp_off == STAGES * STAGE_BYTES )
				sp_off = 0u;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off, stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t af[MF][4], bf[NF][2];
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t ar = ldm_a_row + i * 16u, ac = ks * 2u + ldm_a_col;
				LmStreamLdMatrix4(af[i], stage_a + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			}
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t br = warp_n * WARP_N + j * 8u + (lane & 7u), bc = ks * 2u + ((lane >> 3u) & 1u);
				LmStreamLdMatrix2(bf[j], stage_b + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					LmMmaBf16(acc[i][j], af[i], bf[j]);
		}
		sc_off += STAGE_BYTES;
		if ( sc_off == STAGES * STAGE_BYTES )
			sc_off = 0u;
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t r0 = row_base + warp_m * WARP_M + i * 16u + quad_row, r1 = r0 + 8u;
			const uint32_t col = warp_n * WARP_N + j * 8u + quad * 2u;
			if ( r0 < row_limit )
				*(float2 *)(output_f32 + (uint64_t)r0 * N + col) = make_float2(acc[i][j][0], acc[i][j][1]);
			if ( r1 < row_limit )
				*(float2 *)(output_f32 + (uint64_t)r1 * N + col) = make_float2(acc[i][j][2], acc[i][j][3]);
		}
	LmDependentRelease();
}

template<class S, uint32_t TAG, uint32_t THREADS = 256u, class Kernel>
static int32_t GlmMoeMmaSetup(Kernel kernel, uint32_t shared_bytes, uint32_t sms, uint32_t *blocks_out)
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
		if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, (int)THREADS, (size_t)shared_bytes) != SPARK_GPU_SUCCESS || per_sm < 1 )
			per_sm = 1;
		resident.store(per_sm);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	if ( blocks_out != 0 )
		*blocks_out = sms * (uint32_t)resident.load();
	return LM_LAUNCH_OK;
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t FP8_STAGES, uint32_t BF16_STAGES, uint32_t TAG>
static int32_t GlmMoeMmaW1LaunchTile(const GlmBuffers *b, uint32_t sms, SparkGpuStream stream)
{
	uint32_t blocks = sms;
	if constexpr ( S::kExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		constexpr uint32_t sh = FP8_STAGES * (TILE_M * 256u + 2u * 64u * 128u);
		if ( GlmMoeMmaSetup<S, TAG>(GlmMoeMmaW1ActFp8Kernel<S, TILE_M, WARPS_M, FP8_STAGES>, sh, sms, &blocks) != LM_LAUNCH_OK )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		LM_LAUNCH_DEPENDENT((GlmMoeMmaW1ActFp8Kernel<S, TILE_M, WARPS_M, FP8_STAGES>), blocks, 256u, sh, stream,
			(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset,
			b->group_tile_prefix_w1, b->route_source_token, b->intermediate_bf16);
	}
	else
	{
		constexpr uint32_t sh = BF16_STAGES * (TILE_M * 128u + 2u * 64u * 128u);
		if ( GlmMoeMmaSetup<S, TAG + 3u>(GlmMoeMmaW1ActBf16Kernel<S, TILE_M, WARPS_M, BF16_STAGES>, sh, sms, &blocks) != LM_LAUNCH_OK )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		LM_LAUNCH_DEPENDENT((GlmMoeMmaW1ActBf16Kernel<S, TILE_M, WARPS_M, BF16_STAGES>), blocks, 256u, sh, stream,
			(const uint16_t *)b->expert_w1_weight, b->normed_bf16, b->group_row_offset, b->group_tile_prefix_w1,
			b->route_source_token, b->intermediate_bf16);
	}
	return GlmLaunched();
}

template<class S>
static int32_t GlmMoeMmaW1ActLaunch(const GlmBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	const uint32_t mean = (packed_rows + S::kExperts - 1u) / S::kExperts;
	const uint32_t tile_m = mean >= 22u ? 64u : mean >= 10u ? 32u : 16u;
	LM_LAUNCH_DEPENDENT((GlmBuildTilePrefixKernel), 1u, 32u, 0, stream, b->group_row_offset, b->group_tile_prefix_w1, S::kExperts, tile_m, 2u);
	if ( tile_m == 64u )
		return GlmMoeMmaW1LaunchTile<S, 64u, 2u, 2u, 4u, 1u>(b, sms, stream);
	if ( tile_m == 32u )
		return GlmMoeMmaW1LaunchTile<S, 32u, 2u, 2u, 2u, 2u>(b, sms, stream);
	return GlmMoeMmaW1LaunchTile<S, 16u, 1u, 2u, 2u, 3u>(b, sms, stream);
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t TAG, bool STREAM_OUT = true>
static int32_t GlmMoeMmaW2LaunchTile(const GlmBuffers *b, uint32_t sms, uint32_t col_tile_begin, uint32_t slice_n_tiles, SparkGpuStream stream)
{
	uint32_t blocks = sms;
	if constexpr ( S::kExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		constexpr uint32_t sh = 2u * (TILE_M * 256u + 128u * 128u);
		if ( GlmMoeMmaSetup<S, TAG>(GlmMoeMmaW2K128Fp8Kernel<S, TILE_M, WARPS_M, STREAM_OUT>, sh, sms, &blocks) != LM_LAUNCH_OK )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		LM_LAUNCH_DEPENDENT((GlmMoeMmaW2K128Fp8Kernel<S, TILE_M, WARPS_M, STREAM_OUT>), blocks, 256u, sh, stream,
			(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
			b->group_tile_prefix_w2, b->expert_out_bf16, col_tile_begin, slice_n_tiles);
	}
	else
	{
		constexpr uint32_t sh = 2u * (TILE_M * 128u + 128u * 128u);
		if ( GlmMoeMmaSetup<S, TAG + 3u>(GlmMoeMmaW2K128Bf16Kernel<S, TILE_M, WARPS_M, STREAM_OUT>, sh, sms, &blocks) != LM_LAUNCH_OK )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		LM_LAUNCH_DEPENDENT((GlmMoeMmaW2K128Bf16Kernel<S, TILE_M, WARPS_M, STREAM_OUT>), blocks, 256u, sh, stream,
			(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2,
			b->expert_out_bf16, col_tile_begin, slice_n_tiles);
	}
	return GlmLaunched();
}

template<class S>
static int32_t GlmMoeMmaW2K128Launch(const GlmBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr uint32_t TOTAL_N_TILES = S::kHidden / 128u;
	const uint32_t mean = (packed_rows + S::kExperts - 1u) / S::kExperts;
	const uint32_t tile_m = mean >= 22u ? 64u : mean >= 10u ? 32u : 16u;
	LM_LAUNCH_DEPENDENT((GlmBuildTilePrefixKernel), 1u, 32u, 0, stream, b->group_row_offset, b->group_tile_prefix_w2, S::kExperts, tile_m, 1u);
	if ( tile_m == 64u )
		return GlmMoeMmaW2LaunchTile<S, 64u, 2u, 11u, true>(b, sms, 0u, TOTAL_N_TILES, stream);
	if ( tile_m == 32u )
		return GlmMoeMmaW2LaunchTile<S, 32u, 2u, 12u, true>(b, sms, 0u, TOTAL_N_TILES, stream);
	return GlmMoeMmaW2LaunchTile<S, 16u, 1u, 13u, true>(b, sms, 0u, TOTAL_N_TILES, stream);
}

template<class S>
static int32_t GlmMoeMmaW2K128SlicedFinalizeLaunch(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr uint32_t TOTAL_N_TILES = S::kHidden / 128u;
	const uint32_t mean = (packed_rows + S::kExperts - 1u) / S::kExperts;
	const uint32_t tile_m = mean >= 22u ? 64u : mean >= 10u ? 32u : 16u;
	const uint64_t target_bytes = 12u * 1024u * 1024u;
	uint32_t slice_n_tiles = (uint32_t)(target_bytes / ((uint64_t)packed_rows * 128u * sizeof(uint16_t)));
	if ( slice_n_tiles < 1u )
		slice_n_tiles = 1u;
	if ( slice_n_tiles > TOTAL_N_TILES )
		slice_n_tiles = TOTAL_N_TILES;
	LM_LAUNCH_DEPENDENT((GlmBuildTilePrefixKernel), 1u, 32u, 0, stream, b->group_row_offset, b->group_tile_prefix_w2, S::kExperts, tile_m, 1u);
	for ( uint32_t col_tile = 0u; col_tile < TOTAL_N_TILES; col_tile += slice_n_tiles )
	{
		const uint32_t cur_n_tiles = TOTAL_N_TILES - col_tile < slice_n_tiles ? TOTAL_N_TILES - col_tile : slice_n_tiles;
		int32_t status;
		if ( tile_m == 64u )
			status = GlmMoeMmaW2LaunchTile<S, 64u, 2u, 51u, false>(b, sms, col_tile, cur_n_tiles, stream);
		else if ( tile_m == 32u )
			status = GlmMoeMmaW2LaunchTile<S, 32u, 2u, 52u, false>(b, sms, col_tile, cur_n_tiles, stream);
		else
			status = GlmMoeMmaW2LaunchTile<S, 16u, 1u, 53u, false>(b, sms, col_tile, cur_n_tiles, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
		LM_LAUNCH_DEPENDENT((GlmMoeFinalizeAddSliceKernel<S::kTopK>), (rows + 3u) / 4u, 128u, 0, stream,
			b->expert_out_bf16, b->route_packed_row, b->route_weight, b->shared_out_bf16, b->hidden_bf16,
			rows, col_tile * 128u, cur_n_tiles * 128u, S::kHidden);
	}
	return GlmLaunched();
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES, uint32_t TAG>
static int32_t GlmPrefillQkvIndexLaunchTile(const GlmBuffers *b, GlmLinearWeight q_a, uint32_t q_count, GlmLinearWeight kv_a, uint32_t kv_count,
	uint32_t width, uint32_t query_max, uint32_t with_ik, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr bool IS_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	constexpr uint32_t sh = STAGES * (TILE_M * (IS_FP8 ? 256u : 128u) + 64u * 128u);
	const uint32_t nt = (q_count + 63u) / 64u + (kv_count + 63u) / 64u + (with_ik != 0u ? (S::kIndexDim + 63u) / 64u : 0u);
	if ( GlmMoeMmaSetup<S, TAG>(GlmPrefillQkvIndexMmaKernel<S, TILE_M, WARPS_M, STAGES>, sh, sms, (uint32_t *)0) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	LM_LAUNCH_DEPENDENT((GlmPrefillQkvIndexMmaKernel<S, TILE_M, WARPS_M, STAGES>), dim3(nt, (rows + TILE_M - 1u) / TILE_M), 256u, sh, stream,
		b->normed_bf16, q_a.weight, q_a.scale, q_count, kv_a.weight, kv_a.scale, kv_count,
		with_ik != 0u ? b->weights->index_k.weight : (const uint8_t *)0, with_ik != 0u ? b->weights->index_k.scale : (const float *)0,
		with_ik, b->projection_local_bf16, width, query_max, b->index_key_bf16, rows);
	return GlmLaunched();
}

template<class S>
static int32_t GlmPrefillQkvIndexMmaLaunch(const GlmBuffers *b, GlmLinearWeight q_a, uint32_t q_count, GlmLinearWeight kv_a, uint32_t kv_count,
	uint32_t width, uint32_t query_max, uint32_t with_ik, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr bool IS_FP8 = S::kLinearCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	if ( rows >= 48u )
		return GlmPrefillQkvIndexLaunchTile<S, 64u, 2u, IS_FP8 ? 3u : 4u, 21u>(b, q_a, q_count, kv_a, kv_count, width, query_max, with_ik, rows, sms, stream);
	if ( rows >= 20u )
		return GlmPrefillQkvIndexLaunchTile<S, 32u, 2u, 4u, 22u>(b, q_a, q_count, kv_a, kv_count, width, query_max, with_ik, rows, sms, stream);
	return GlmPrefillQkvIndexLaunchTile<S, 16u, 1u, 4u, 23u>(b, q_a, q_count, kv_a, kv_count, width, query_max, with_ik, rows, sms, stream);
}

template<class S, uint32_t TILE_M, uint32_t WARPS_M, uint32_t TAG>
static int32_t GlmPrefillIndexHeadLaunchTile(const uint16_t *weight, const uint16_t *act, float *output_f32, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr uint32_t sh = 3u * (TILE_M + S::kIndexHeads) * 128u;
	if ( GlmMoeMmaSetup<S, TAG, 128u>(GlmPrefillIndexHeadMmaKernel<S, TILE_M, WARPS_M>, sh, sms, (uint32_t *)0) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	LM_LAUNCH_DEPENDENT((GlmPrefillIndexHeadMmaKernel<S, TILE_M, WARPS_M>), (rows + TILE_M - 1u) / TILE_M, 128u, sh, stream,
		weight, act, output_f32, rows);
	return GlmLaunched();
}

template<class S>
static int32_t GlmPrefillIndexHeadMmaLaunch(const uint16_t *weight, const uint16_t *act, float *output_f32, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	if ( rows >= 48u )
		return GlmPrefillIndexHeadLaunchTile<S, 64u, 4u, 41u>(weight, act, output_f32, rows, sms, stream);
	if ( rows >= 20u )
		return GlmPrefillIndexHeadLaunchTile<S, 32u, 2u, 42u>(weight, act, output_f32, rows, sms, stream);
	return GlmPrefillIndexHeadLaunchTile<S, 16u, 1u, 43u>(weight, act, output_f32, rows, sms, stream);
}
