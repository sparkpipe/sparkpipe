#include "model-families/hy4/source/hy4_stage.cuh"
#include "model-families/mimo26/source/mimo26_stage.cuh"
#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

static int g_failures = 0;

static void Check(int ok, const char *tag, const char *detail, double a = 0.0, double b = 0.0)
{
	if ( ok == 0 )
	{
		fprintf(stderr, "FAIL [%s] %s (a=%.6g, b=%.6g)\n", tag, detail, a, b);
		g_failures++;
	}
}

#define FILL_BF16 0u
#define FILL_F32 1u
#define FILL_FP8 2u
#define FILL_E2M1 3u
#define FILL_E8M0 4u

__global__ static void BenchFillKernel(void *dst, uint64_t count, uint32_t kind, uint32_t seed, float scale, float offset)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint32_t bits = Mix(seed * 0x9e3779b9u + (uint32_t)i);
		const float val = offset + scale * Uniform(seed, (uint32_t)i);
		if ( kind == FILL_BF16 )
			((uint16_t *)dst)[i] = Bf16(val);
		else if ( kind == FILL_F32 )
			((float *)dst)[i] = val;
		else if ( kind == FILL_FP8 )
			((uint8_t *)dst)[i] = (uint8_t)((bits & 0x80u) | ((4u + ((bits >> 3u) & 3u)) << 3u) | (bits & 7u));
		else if ( kind == FILL_E2M1 )
			((uint8_t *)dst)[i] = (uint8_t)(bits >> 8u);
		else
			((uint8_t *)dst)[i] = (uint8_t)((uint32_t)offset + ((bits >> 8u) % 5u));
	}
}

struct DevPool
{
	std::vector<void *> ptrs;
	uint32_t seed;
	cudaStream_t stream;

	DevPool(cudaStream_t s) : seed(77u), stream(s) {}
	~DevPool(void)
	{
		for ( void *p : ptrs )
			cudaFree(p);
	}

	void *Alloc(uint64_t bytes)
	{
		void *p = 0;
		const uint64_t aligned = ((bytes + 127u) / 128u) * 128u;
		if ( cudaMalloc(&p, aligned > 0u ? aligned : 128u) != cudaSuccess )
		{
			fprintf(stderr, "cudaMalloc(%llu) failed\n", (unsigned long long)aligned);
			std::exit(2);
		}
		ptrs.push_back(p);
		return p;
	}

	void *Fill(uint64_t count, uint32_t elem_bytes, uint32_t kind, float scale, float offset)
	{
		void *p = Alloc(count * elem_bytes);
		const uint32_t blocks = (uint32_t)((count + 255u) / 256u);
		const uint32_t capped = blocks > 4096u ? 4096u : (blocks == 0u ? 1u : blocks);
		BenchFillKernel<<<capped, 256u, 0, stream>>>(p, count, kind, seed++, scale, offset);
		return p;
	}
};

static float Bf16ToF32(uint16_t b)
{
	const uint32_t u = (uint32_t)b << 16u;
	float f = 0.0f;
	memcpy(&f, &u, sizeof(float));
	return f;
}

static double RelL2(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, uint64_t count, uint32_t *out_max_ulp)
{
	double num = 0.0, den = 0.0;
	uint32_t max_ulp = 0u;
	for ( uint64_t i = 0u; i < count; ++i )
	{
		const float fa = Bf16ToF32(a[i]), fb = Bf16ToF32(b[i]);
		if ( !std::isfinite(fa) || !std::isfinite(fb) )
			return 1e9;
		const double d = (double)fa - (double)fb;
		num += d * d;
		den += (double)fb * (double)fb;
		if ( a[i] != b[i] )
		{
			const int32_t ia = (a[i] & 0x8000u) ? -(int32_t)(a[i] & 0x7fffu) : (int32_t)a[i];
			const int32_t ib = (b[i] & 0x8000u) ? -(int32_t)(b[i] & 0x7fffu) : (int32_t)b[i];
			const uint32_t ulp = (uint32_t)std::abs(ia - ib);
			if ( ulp > max_ulp )
				max_ulp = ulp;
		}
	}
	if ( out_max_ulp != 0 )
		*out_max_ulp = max_ulp;
	return std::sqrt(num / (den > 1e-12 ? den : 1.0));
}

static uint64_t DiffCount(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, uint64_t count)
{
	uint64_t diffs = 0u;
	for ( uint64_t i = 0u; i < count; ++i )
		if ( a[i] != b[i] )
			diffs++;
	return diffs;
}

template<class Fn>
static double TimeUs(cudaStream_t stream, int warmup, int iters, Fn &&fn)
{
	for ( int i = 0; i < warmup; ++i )
		fn();
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);
	cudaStreamSynchronize(stream);
	cudaEventRecord(t0, stream);
	for ( int i = 0; i < iters; ++i )
		fn();
	cudaEventRecord(t1, stream);
	cudaEventSynchronize(t1);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, t0, t1);
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	return (double)ms * 1000.0 / (double)iters;
}

static void BuildSyntheticRoutes(uint32_t local_experts, uint32_t rows, uint32_t top_k, uint32_t total_experts,
	std::vector<uint32_t> &offsets, std::vector<uint32_t> &src_tok, std::vector<float> &packed_wt)
{
	offsets.assign(local_experts + 1u, 0u);
	std::vector<std::vector<uint32_t>> buckets(local_experts);
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t ge = Mix(r * 131u + k * 17u + 99u) % total_experts;
			if ( ge < local_experts )
				buckets[ge].push_back(r);
		}
	}
	uint32_t cur = 0u;
	src_tok.clear();
	packed_wt.clear();
	for ( uint32_t e = 0u; e < local_experts; ++e )
	{
		offsets[e] = cur;
		for ( uint32_t tok : buckets[e] )
		{
			src_tok.push_back(tok);
			packed_wt.push_back(0.125f + 0.05f * Uniform(777u, cur));
			cur++;
		}
	}
	offsets[local_experts] = cur;
}

/* BASE e17c8fce1 MiMo MXFP4 expert kernel for direct paired A/B timing */
static __device__ __forceinline__ uint32_t BaseE8m0ToBf16x2(uint8_t sc)
{
	const uint32_t s = (uint32_t)sc;
	if ( s >= 2u && s <= 142u )
	{
		const uint32_t b = (s + 112u) << 7u;
		return b | (b << 16u);
	}
	return 0u;
}

static __device__ __forceinline__ void BaseE2m1Word4(uint32_t word, uint8_t sc, uint32_t scale_bf16x2, uint32_t lo[2], uint32_t hi[2])
{
	if ( scale_bf16x2 != 0u )
	{
		lo[0] = MimoE2m1ByteFast(word, scale_bf16x2);
		lo[1] = MimoE2m1ByteFast(word >> 8u, scale_bf16x2);
		hi[0] = MimoE2m1ByteFast(word >> 16u, scale_bf16x2);
		hi[1] = MimoE2m1ByteFast(word >> 24u, scale_bf16x2);
	}
	else
	{
		const float scale = LmUe8m0ToFloat(sc);
		const float2 p0 = LmSkinnyE2m1Pair(word, 0u);
		const float2 p1 = LmSkinnyE2m1Pair(word, 1u);
		const float2 p2 = LmSkinnyE2m1Pair(word, 2u);
		const float2 p3 = LmSkinnyE2m1Pair(word, 3u);
		lo[0] = LmPackBf16Pair(p0.x * scale, p0.y * scale);
		lo[1] = LmPackBf16Pair(p1.x * scale, p1.y * scale);
		hi[0] = LmPackBf16Pair(p2.x * scale, p2.y * scale);
		hi[1] = LmPackBf16Pair(p3.x * scale, p3.y * scale);
	}
}

static __device__ __forceinline__ void BaseCpAsync16(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

template<uint32_t MAX_M_TILES, uint32_t STAGES, bool FUSED_GATE_UP>
static __host__ __device__ constexpr uint32_t BaseMxfp4SharedBytes(uint32_t groups)
{
	constexpr uint32_t TILE_M = MAX_M_TILES * 16u;
	constexpr uint32_t W_ROWS_TOTAL = (FUSED_GATE_UP ? 2u : 1u) * (FUSED_GATE_UP ? 64u : 128u);
	return STAGES * TILE_M * 136u * 2u + STAGES * W_ROWS_TOTAL * 64u + 2u * W_ROWS_TOTAL * 16u + (groups + 1u) * 4u;
}

template<uint32_t MAX_M_TILES, uint32_t STAGES, bool ACTIVATION_PACKED, bool FUSED_GATE_UP>
__global__ __launch_bounds__(256u, MAX_M_TILES <= 2u ? 2 : 1)
static void BaseMimoMxfp4ExpertsKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1, const uint8_t *__restrict__ w3,
	const uint8_t *__restrict__ s3, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token, uint32_t groups, uint32_t input_dim,
	uint32_t output_dim)
{
	constexpr uint32_t THREADS = 256u, TILE_M = MAX_M_TILES * 16u, TILE_K = 128u;
	constexpr uint32_t TILE_N = FUSED_GATE_UP ? 64u : 128u, W_ROWS_TOTAL = (FUSED_GATE_UP ? 2u : 1u) * TILE_N;
	constexpr uint32_t A_PITCH = 136u, B_ROW_BYTES = 64u;
	constexpr uint32_t A_STAGE_BYTES = TILE_M * A_PITCH * 2u, B_STAGE_BYTES = W_ROWS_TOTAL * B_ROW_BYTES;
	extern __shared__ __align__(128) uint8_t base_mx4_smem[];
	uint16_t (*s_a)[TILE_M][A_PITCH] = (uint16_t (*)[TILE_M][A_PITCH])base_mx4_smem;
	uint8_t (*s_b)[W_ROWS_TOTAL][B_ROW_BYTES] = (uint8_t (*)[W_ROWS_TOTAL][B_ROW_BYTES])(base_mx4_smem + STAGES * A_STAGE_BYTES);
	uint8_t (*s_sc)[W_ROWS_TOTAL][16u] = (uint8_t (*)[W_ROWS_TOTAL][16u])(base_mx4_smem + STAGES * A_STAGE_BYTES + STAGES * B_STAGE_BYTES);
	uint32_t *prefix = (uint32_t *)(base_mx4_smem + STAGES * A_STAGE_BYTES + STAGES * B_STAGE_BYTES + 2u * W_ROWS_TOTAL * 16u);
	const uint32_t n_blocks = (output_dim + TILE_N - 1u) / TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, g = lane >> 2u, c = lane & 3u;
	const uint32_t k_tiles = input_dim / TILE_K;
	const uint64_t payload_pitch = (uint64_t)(input_dim >> 1u), scale_pitch = (uint64_t)(input_dim >> 5u);
	const uint64_t scale_group_stride = (uint64_t)output_dim * scale_pitch;
	const uint32_t b_row0 = FUSED_GATE_UP ? (warp * 8u + g) : (warp * 16u + g);
	const uint32_t b_row1 = FUSED_GATE_UP ? (TILE_N + warp * 8u + g) : (warp * 16u + 8u + g);
	constexpr uint32_t A_VEC_THREADS = 16u, A_ROW_STEP = 16u, A_COPIES = (TILE_M + 15u) / 16u;
	const uint32_t a_vec = threadIdx.x % A_VEC_THREADS, a_row0 = threadIdx.x / A_VEC_THREADS;
	LmStreamGemmArguments targs;
	targs.group_row_offset = group_row_offset;
	targs.groups = groups;
	targs.rows = 0u;
	LmStreamTilePrefix(targs, prefix, groups, TILE_M, n_blocks);
	const uint32_t total_tiles = prefix[groups];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		const LmStreamTile work = LmStreamTileAt(targs, prefix, groups, TILE_M, TILE_N, n_blocks, tile);
		const uint32_t group = work.group, first = work.row_base, row_end = work.row_limit, n_base = work.neuron_base;
		const uint32_t count = row_end - first < TILE_M ? row_end - first : TILE_M;
		const uint32_t m_tiles_live = (count + 15u) >> 4u;
		const uint16_t *a_ptrs[A_COPIES];
		#pragma unroll
		for ( uint32_t i = 0u; i < A_COPIES; ++i )
		{
			const uint32_t lr = a_row0 + i * A_ROW_STEP;
			if ( lr < count )
			{
				const uint32_t pr = first + lr;
				a_ptrs[i] = activation + (uint64_t)(ACTIVATION_PACKED ? pr : route_source_token[pr]) * input_dim + (uint64_t)a_vec * 8u;
			}
			else
				a_ptrs[i] = 0;
		}
		auto stage_tile = [&](uint32_t st, uint32_t kt) {
			#pragma unroll
			for ( uint32_t i = 0u; i < A_COPIES; ++i )
			{
				const uint32_t lr = a_row0 + i * A_ROW_STEP;
				if ( lr < TILE_M && a_ptrs[i] != 0 )
					BaseCpAsync16(&s_a[st][lr][a_vec * 8u], a_ptrs[i] + (uint64_t)kt * TILE_K, 16u);
			}
			#pragma unroll
			for ( uint32_t idx = threadIdx.x; idx < W_ROWS_TOTAL * 4u; idx += THREADS )
			{
				const uint32_t wr = idx / 4u, bv = idx % 4u;
				const uint32_t mat = FUSED_GATE_UP ? (wr / TILE_N) : 0u;
				const uint32_t neuron = n_base + (FUSED_GATE_UP ? (wr % TILE_N) : wr);
				const bool ok = neuron < output_dim;
				const uint8_t *wbase = (mat == 0u ? w1 : w3);
				const uint8_t *src = ok ? wbase + ((uint64_t)group * output_dim + neuron) * payload_pitch + (uint64_t)kt * B_ROW_BYTES + (uint64_t)bv * 16u : wbase;
				BaseCpAsync16(&s_b[st][wr][bv * 16u], src, ok ? 16u : 0u);
			}
			if ( (kt & 3u) == 0u )
			{
				const uint32_t sc_off = kt * 4u, avail = sc_off < scale_pitch ? (uint32_t)(scale_pitch - sc_off) : 0u;
				const uint32_t sb_slot = (kt >> 2u) & 1u;
				#pragma unroll
				for ( uint32_t wr = threadIdx.x; wr < W_ROWS_TOTAL; wr += THREADS )
				{
					const uint32_t mat = FUSED_GATE_UP ? (wr / TILE_N) : 0u;
					const uint32_t neuron = n_base + (FUSED_GATE_UP ? (wr % TILE_N) : wr);
					const uint32_t bytes = (neuron < output_dim && avail != 0u) ? (avail >= 16u ? 16u : avail) : 0u;
					const uint8_t *sbase = (mat == 0u ? s1 : s3);
					const uint8_t *src = bytes != 0u ? sbase + (uint64_t)group * scale_group_stride + (uint64_t)neuron * scale_pitch + sc_off : sbase;
					BaseCpAsync16(&s_sc[sb_slot][wr][0], src, bytes);
				}
			}
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		};
		float acc0[MAX_M_TILES][4], acc1[MAX_M_TILES][4];
		#pragma unroll
		for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
			acc0[m][0] = acc0[m][1] = acc0[m][2] = acc0[m][3] = acc1[m][0] = acc1[m][1] = acc1[m][2] = acc1[m][3] = 0.0f;
		uint32_t kp = 0u, sp = 0u, sc = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < k_tiles )
			{
				stage_tile(sp, kp);
				kp++;
				sp = (sp + 1u == STAGES) ? 0u : (sp + 1u);
			}
		}
		for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
		{
			if constexpr ( STAGES == 3u )
			{
				if ( kt + 1u < k_tiles )
					asm volatile("cp.async.wait_group 1;\n" ::: "memory");
				else
					asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			}
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			if ( kp < k_tiles )
			{
				stage_tile(sp, kp);
				kp++;
				sp = (sp + 1u == STAGES) ? 0u : (sp + 1u);
			}
			const uint32_t sb_slot = (kt >> 2u) & 1u, sb_sub = (kt & 3u) * 4u;
			const uint32_t sc0_all = *(const uint32_t *)&s_sc[sb_slot][b_row0][sb_sub];
			const uint32_t sc1_all = *(const uint32_t *)&s_sc[sb_slot][b_row1][sb_sub];
			#pragma unroll
			for ( uint32_t sg = 0u; sg < 4u; ++sg )
			{
				const uint8_t raw0 = (uint8_t)(sc0_all >> (sg * 8u));
				const uint8_t raw1 = (uint8_t)(sc1_all >> (sg * 8u));
				const uint32_t w0 = *(const uint32_t *)&s_b[sc][b_row0][sg * 16u + c * 4u];
				const uint32_t w1_w = *(const uint32_t *)&s_b[sc][b_row1][sg * 16u + c * 4u];
				uint32_t b0_lo[2], b0_hi[2], b1_lo[2], b1_hi[2];
				BaseE2m1Word4(w0, raw0, BaseE8m0ToBf16x2(raw0), b0_lo, b0_hi);
				BaseE2m1Word4(w1_w, raw1, BaseE8m0ToBf16x2(raw1), b1_lo, b1_hi);
				const uint32_t k_off = sg * 32u + c * 8u;
				#pragma unroll
				for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
				{
					if ( m < m_tiles_live )
					{
						const uint4 low = *(const uint4 *)&s_a[sc][m * 16u + g][k_off];
						const uint4 high = *(const uint4 *)&s_a[sc][m * 16u + g + 8u][k_off];
						const uint32_t a0[4] = { low.x, high.x, low.y, high.y };
						const uint32_t a1[4] = { low.z, high.z, low.w, high.w };
						LmMmaBf16(acc0[m], a0, b0_lo);
						LmMmaBf16(acc0[m], a1, b0_hi);
						LmMmaBf16(acc1[m], a0, b1_lo);
						LmMmaBf16(acc1[m], a1, b1_hi);
					}
				}
			}
			sc = (sc + 1u == STAGES) ? 0u : (sc + 1u);
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		if constexpr ( FUSED_GATE_UP )
		{
			const uint32_t col = n_base + warp * 8u + c * 2u;
			if ( col < output_dim )
				#pragma unroll
				for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
					if ( m < m_tiles_live )
						#pragma unroll
						for ( uint32_t h = 0u; h < 2u; ++h )
						{
							const uint32_t pr = first + m * 16u + g + h * 8u;
							if ( pr < row_end )
							{
								const float v0 = SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(acc0[m][h * 2u]))) * SparkMathRoundBf16(acc1[m][h * 2u]);
								const float v1 = SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(acc0[m][h * 2u + 1u]))) * SparkMathRoundBf16(acc1[m][h * 2u + 1u]);
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col) = LmPackBf16Pair(v0, v1);
							}
						}
		}
		else
		{
			const uint32_t col0 = n_base + warp * 16u + c * 2u, col1 = col0 + 8u;
			#pragma unroll
			for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
				if ( m < m_tiles_live )
					#pragma unroll
					for ( uint32_t h = 0u; h < 2u; ++h )
					{
						const uint32_t pr = first + m * 16u + g + h * 8u;
						if ( pr < row_end )
						{
							if ( col0 < output_dim )
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col0) = LmPackBf16Pair(acc0[m][h * 2u], acc0[m][h * 2u + 1u]);
							if ( col1 < output_dim )
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col1) = LmPackBf16Pair(acc1[m][h * 2u], acc1[m][h * 2u + 1u]);
						}
					}
		}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
}

static void RunHyMoeTests(uint32_t sms, cudaStream_t stream)
{
	using G = Hy4Geometry<16u>;
	DevPool pool(stream);
	constexpr uint32_t E = G::kLocalExperts, H = G::kHidden, I = G::kMoeInter;
	const uint32_t max_rows = 2048u, max_packed = max_rows * 2u;

	uint8_t *w1 = (uint8_t *)pool.Fill((uint64_t)E * I * H, 1u, FILL_FP8, 0.0f, 0.0f);
	uint8_t *w3 = (uint8_t *)pool.Fill((uint64_t)E * I * H, 1u, FILL_FP8, 0.0f, 0.0f);
	uint8_t *w2 = (uint8_t *)pool.Fill((uint64_t)E * H * I, 1u, FILL_FP8, 0.0f, 0.0f);
	uint8_t *s1_mx = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
	uint8_t *s3_mx = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
	uint8_t *s2_mx = (uint8_t *)pool.Fill((uint64_t)E * H * (I / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
	float *s1_f8 = (float *)pool.Fill((uint64_t)E * I * (H / 128u), sizeof(float), FILL_F32, 0.04f, 0.05f);
	float *s3_f8 = (float *)pool.Fill((uint64_t)E * I * (H / 128u), sizeof(float), FILL_F32, 0.04f, 0.05f);
	float *s2_f8 = (float *)pool.Fill((uint64_t)E * H * (I / 128u), sizeof(float), FILL_F32, 0.04f, 0.05f);

	uint16_t *act = (uint16_t *)pool.Fill((uint64_t)max_rows * H, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *gate_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *up_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_96 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_96 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint32_t *d_off = (uint32_t *)pool.Alloc((E + 1u) * sizeof(uint32_t));
	uint32_t *d_src = (uint32_t *)pool.Alloc(max_packed * sizeof(uint32_t));
	float *d_pwt = (float *)pool.Alloc(max_packed * sizeof(float));

	const LmScaleTensor t1_mx = LmScaleTensorBuild(s1_mx, LM_SCALE_ENCODING_UE8M0, E, I, H, 1u, 32u);
	const LmScaleTensor t3_mx = LmScaleTensorBuild(s3_mx, LM_SCALE_ENCODING_UE8M0, E, I, H, 1u, 32u);
	const LmScaleTensor t2_mx = LmScaleTensorBuild(s2_mx, LM_SCALE_ENCODING_UE8M0, E, H, I, 1u, 32u);
	const LmScaleTensor t1_f8 = LmScaleTensorBuild(s1_f8, LM_SCALE_ENCODING_F32, E, I, H, 1u, 128u);
	const LmScaleTensor t3_f8 = LmScaleTensorBuild(s3_f8, LM_SCALE_ENCODING_F32, E, I, H, 1u, 128u);
	const LmScaleTensor t2_f8 = LmScaleTensorBuild(s2_f8, LM_SCALE_ENCODING_F32, E, H, I, 1u, 128u);

	auto set_smem = [](const void *fn, uint32_t bytes) -> uint32_t {
		if ( bytes > 49152u )
			cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
		int occ = 1;
		if ( cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, (int)bytes) != cudaSuccess || occ < 1 )
			occ = 1;
		return (uint32_t)occ;
	};
	constexpr uint32_t smem_u16 = 2u * (16u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_u32 = 2u * (32u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_u64 = 3u * (64u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_u96 = 2u * (96u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d16 = 2u * (16u * 256u + 128u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d32 = 2u * (32u * 256u + 128u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d64 = 3u * (64u * 256u + 128u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d96 = 3u * (96u * 256u + 64u * 128u) + (E + 1u) * 4u;
	const uint32_t occ_u16 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 16u, 1u, 8u, 2u>, smem_u16);
	const uint32_t occ_u32 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 2u>, smem_u32);
	const uint32_t occ_u64 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u>, smem_u64);
	const uint32_t occ_u96 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 96u, 2u, 4u, 2u>, smem_u96);
	const uint32_t occ_d16 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 16u, 128u, 1u, 8u, 2u>, smem_d16);
	const uint32_t occ_d32 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 128u, 2u, 4u, 2u>, smem_d32);
	const uint32_t occ_d64 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 128u, 2u, 4u, 3u>, smem_d64);
	const uint32_t occ_d96 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 96u, 64u, 2u, 4u, 3u>, smem_d96);
	const uint32_t occ_fu16 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 16u, 1u, 8u, 2u>, smem_u16);
	const uint32_t occ_fu32 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 32u, 2u, 4u, 2u>, smem_u32);
	const uint32_t occ_fu64 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 64u, 2u, 4u, 3u>, smem_u64);
	const uint32_t occ_fu96 = set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 96u, 2u, 4u, 2u>, smem_u96);
	const uint32_t occ_fd16 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 16u, 128u, 1u, 8u, 2u>, smem_d16);
	const uint32_t occ_fd32 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 32u, 128u, 2u, 4u, 2u>, smem_d32);
	const uint32_t occ_fd64 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 64u, 128u, 2u, 4u, 3u>, smem_d64);
	const uint32_t occ_fd96 = set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 96u, 64u, 2u, 4u, 3u>, smem_d96);

	printf("\n=== 1. Hy4 MoE W1W3+SwiGLU & W2 (MXFP8 & FP8, occ: u16=%u u32=%u u64=%u u96=%u d16=%u d32=%u d64=%u d96=%u) ===\n",
		occ_u16, occ_u32, occ_u64, occ_u96, occ_d16, occ_d32, occ_d64, occ_d96);
	{
		std::vector<uint32_t> off, src;
		std::vector<float> pwt;
		BuildSyntheticRoutes(E, 512u, G::kTopK, G::kExperts, off, src, pwt);
		const uint32_t packed = off[E];
		cudaMemcpy(d_off, off.data(), off.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pwt, pwt.data(), pwt.size() * sizeof(float), cudaMemcpyHostToDevice);

		LmStreamGemmGrouped<LmMxfp8>(w1, t1_mx, act, gate_ref, d_off, d_src, E, packed, H, I, sms, stream);
		LmStreamGemmGrouped<LmMxfp8>(w3, t3_mx, act, up_ref, d_off, d_src, E, packed, H, I, sms, stream);
		HyMoeSwigluPackedKernel<<<dim3(LmVecBlocks<HY_THREADS>(I), packed), HY_THREADS, 0, stream>>>(
			gate_ref, up_ref, d_pwt, d_off, E, act_ref, I, G::kSwigluLimit);
		LmStreamGemmGrouped<LmMxfp8>(w2, t2_mx, act_ref, out_ref, d_off, (const uint32_t *)0, E, packed, I, H, sms, stream);

		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 16u, 1u, 8u, 2u><<<sms * occ_u16, 256u, smem_u16, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_16);
		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 2u><<<sms * occ_u32, 256u, smem_u32, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_32);
		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms * occ_u64, 256u, smem_u64, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_64);
		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 96u, 2u, 4u, 2u><<<sms * occ_u96, 256u, smem_u96, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_96);
		HyMoeMmaW2Kernel<G, LmMxfp8, 16u, 128u, 1u, 8u, 2u><<<sms * occ_d16, 256u, smem_d16, stream>>>(w2, t2_mx, act_16, d_off, out_16);
		HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 128u, 2u, 4u, 2u><<<sms * occ_d32, 256u, smem_d32, stream>>>(w2, t2_mx, act_32, d_off, out_32);
		HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 128u, 2u, 4u, 3u><<<sms * occ_d64, 256u, smem_d64, stream>>>(w2, t2_mx, act_64, d_off, out_64);
		HyMoeMmaW2Kernel<G, LmMxfp8, 96u, 64u, 2u, 4u, 3u><<<sms * occ_d96, 256u, smem_d96, stream>>>(w2, t2_mx, act_96, d_off, out_96);
		cudaStreamSynchronize(stream);

		std::vector<uint16_t> ha_ref((uint64_t)packed * I), ha16((uint64_t)packed * I), ha32((uint64_t)packed * I), ha64((uint64_t)packed * I), ha96((uint64_t)packed * I);
		std::vector<uint16_t> ho_ref((uint64_t)packed * H), ho16((uint64_t)packed * H), ho32((uint64_t)packed * H), ho64((uint64_t)packed * H), ho96((uint64_t)packed * H);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha32.data(), act_32, ha32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha64.data(), act_64, ha64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha96.data(), act_96, ha96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho32.data(), out_32, ho32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho64.data(), out_64, ho64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho96.data(), out_96, ho96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

		uint32_t ulp_u = 0u, ulp_d = 0u;
		Check(RelL2(ha96, ha_ref, ha96.size(), &ulp_u) < 2e-3, "hy4_moe_mxfp8", "W1W3 vs ref", RelL2(ha96, ha_ref, ha96.size(), 0), (double)ulp_u);
		Check(DiffCount(ha16, ha32, ha16.size()) == 0u && DiffCount(ha32, ha64, ha32.size()) == 0u && DiffCount(ha64, ha96, ha64.size()) == 0u,
			"hy4_moe_mxfp8", "W1W3 0-ULP M16==M32==M64==M96");
		Check(RelL2(ho96, ho_ref, ho96.size(), &ulp_d) < 2e-3, "hy4_moe_mxfp8", "W2 vs ref", RelL2(ho96, ho_ref, ho96.size(), 0), (double)ulp_d);
		Check(DiffCount(ho16, ho32, ho16.size()) == 0u && DiffCount(ho32, ho64, ho32.size()) == 0u && DiffCount(ho64, ho96, ho64.size()) == 0u,
			"hy4_moe_mxfp8", "W2 0-ULP M16==M32==M64==M96");

		LmStreamGemmGrouped<LmFp8>(w1, t1_f8, act, gate_ref, d_off, d_src, E, packed, H, I, sms, stream);
		LmStreamGemmGrouped<LmFp8>(w3, t3_f8, act, up_ref, d_off, d_src, E, packed, H, I, sms, stream);
		HyMoeSwigluPackedKernel<<<dim3(LmVecBlocks<HY_THREADS>(I), packed), HY_THREADS, 0, stream>>>(
			gate_ref, up_ref, d_pwt, d_off, E, act_ref, I, G::kSwigluLimit);
		LmStreamGemmGrouped<LmFp8>(w2, t2_f8, act_ref, out_ref, d_off, (const uint32_t *)0, E, packed, I, H, sms, stream);

		HyMoeMmaW1W3SwigluKernel<G, LmFp8, 16u, 1u, 8u, 2u><<<sms * occ_fu16, 256u, smem_u16, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_16);
		HyMoeMmaW1W3SwigluKernel<G, LmFp8, 96u, 2u, 4u, 2u><<<sms * occ_fu96, 256u, smem_u96, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_96);
		HyMoeMmaW2Kernel<G, LmFp8, 16u, 128u, 1u, 8u, 2u><<<sms * occ_fd16, 256u, smem_d16, stream>>>(w2, t2_f8, act_16, d_off, out_16);
		HyMoeMmaW2Kernel<G, LmFp8, 96u, 64u, 2u, 4u, 3u><<<sms * occ_fd96, 256u, smem_d96, stream>>>(w2, t2_f8, act_96, d_off, out_96);
		cudaStreamSynchronize(stream);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha96.data(), act_96, ha96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho96.data(), out_96, ho96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		Check(DiffCount(ha96, ha_ref, ha96.size()) == 0u && DiffCount(ha16, ha96, ha16.size()) == 0u, "hy4_moe_fp8", "W1W3 0-ULP exact vs ref");
		Check(DiffCount(ho96, ho_ref, ho96.size()) == 0u && DiffCount(ho16, ho96, ho16.size()) == 0u, "hy4_moe_fp8", "W2 0-ULP exact vs ref");
	}

	for ( uint32_t rows : { 512u, 1024u, 2048u } )
	{
		std::vector<uint32_t> off, src;
		std::vector<float> pwt;
		BuildSyntheticRoutes(E, rows, G::kTopK, G::kExperts, off, src, pwt);
		const uint32_t packed = off[E];
		cudaMemcpy(d_off, off.data(), off.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pwt, pwt.data(), pwt.size() * sizeof(float), cudaMemcpyHostToDevice);
		const uint32_t mean = (packed + E - 1u) / E;

		const double t_ref_mx = TimeUs(stream, 2, 10, [&]() {
			LmStreamGemmGrouped<LmMxfp8>(w1, t1_mx, act, gate_ref, d_off, d_src, E, rows * G::kTopK, H, I, sms, stream);
			LmStreamGemmGrouped<LmMxfp8>(w3, t3_mx, act, up_ref, d_off, d_src, E, rows * G::kTopK, H, I, sms, stream);
			HyMoeSwigluPackedKernel<<<dim3(LmVecBlocks<HY_THREADS>(I), packed), HY_THREADS, 0, stream>>>(
				gate_ref, up_ref, d_pwt, d_off, E, act_ref, I, G::kSwigluLimit);
			LmStreamGemmGrouped<LmMxfp8>(w2, t2_mx, act_ref, out_ref, d_off, (const uint32_t *)0, E, rows * G::kTopK, I, H, sms, stream);
		});
		const double t_new_mx = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 40u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 96u, 2u, 4u, 2u><<<sms * occ_u96, 256u, smem_u96, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_96);
				HyMoeMmaW2Kernel<G, LmMxfp8, 96u, 64u, 2u, 4u, 3u><<<sms * occ_d96, 256u, smem_d96, stream>>>(w2, t2_mx, act_96, d_off, out_96);
			}
			else if ( mean > 24u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms * occ_u64, 256u, smem_u64, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_64);
				HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 128u, 2u, 4u, 3u><<<sms * occ_d64, 256u, smem_d64, stream>>>(w2, t2_mx, act_64, d_off, out_64);
			}
			else
			{
				HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 2u><<<sms * occ_u32, 256u, smem_u32, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_32);
				HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 128u, 2u, 4u, 2u><<<sms * occ_d32, 256u, smem_d32, stream>>>(w2, t2_mx, act_32, d_off, out_32);
			}
		});
		const double t_new_f8 = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 40u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmFp8, 96u, 2u, 4u, 2u><<<sms * occ_fu96, 256u, smem_u96, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_96);
				HyMoeMmaW2Kernel<G, LmFp8, 96u, 64u, 2u, 4u, 3u><<<sms * occ_fd96, 256u, smem_d96, stream>>>(w2, t2_f8, act_96, d_off, out_96);
			}
			else if ( mean > 24u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmFp8, 64u, 2u, 4u, 3u><<<sms * occ_fu64, 256u, smem_u64, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_64);
				HyMoeMmaW2Kernel<G, LmFp8, 64u, 128u, 2u, 4u, 3u><<<sms * occ_fd64, 256u, smem_d64, stream>>>(w2, t2_f8, act_64, d_off, out_64);
			}
			else
			{
				HyMoeMmaW1W3SwigluKernel<G, LmFp8, 32u, 2u, 4u, 2u><<<sms * occ_fu32, 256u, smem_u32, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_32);
				HyMoeMmaW2Kernel<G, LmFp8, 32u, 128u, 2u, 4u, 2u><<<sms * occ_fd32, 256u, smem_d32, stream>>>(w2, t2_f8, act_32, d_off, out_32);
			}
		});
		printf("  hy4_moe        M=%4u (packed=%4u, mean=%2u) | MXFP8: ref=%7.1f us  new=%7.1f us (%5.2fx) | FP8 new=%7.1f us\n",
			rows, packed, mean, t_ref_mx, t_new_mx, t_ref_mx / t_new_mx, t_new_f8);
	}
}

template<class G>
static void RunMimoMxfp4Tests(const char *tag, uint32_t sms, cudaStream_t stream)
{
	using Model = MimoModel<G>;
	DevPool pool(stream);
	constexpr uint32_t E = G::kLocalExperts, H = G::kHidden, I = G::kMoeInter;
	const uint32_t max_rows = 2048u, max_packed = max_rows * 2u;

	uint8_t *w1 = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 2u), 1u, FILL_E2M1, 0.0f, 0.0f);
	uint8_t *s1 = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
	uint8_t *w3 = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 2u), 1u, FILL_E2M1, 0.0f, 0.0f);
	uint8_t *s3 = (uint8_t *)pool.Fill((uint64_t)E * I * (H / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
	uint8_t *w2 = (uint8_t *)pool.Fill((uint64_t)E * H * (I / 2u), 1u, FILL_E2M1, 0.0f, 0.0f);
	uint8_t *s2 = (uint8_t *)pool.Fill((uint64_t)E * H * (I / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);

	uint16_t *act = (uint16_t *)pool.Fill((uint64_t)max_rows * H, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *act_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_96 = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *act_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *out_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_96 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint32_t *d_off = (uint32_t *)pool.Alloc((E + 1u) * sizeof(uint32_t));
	uint32_t *d_src = (uint32_t *)pool.Alloc(max_packed * sizeof(uint32_t));

	auto prep_base = [](const void *fn, uint32_t bytes) -> uint32_t {
		if ( bytes > 49152u )
			cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
		int occ = 1;
		if ( cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, (int)bytes) != cudaSuccess || occ < 1 )
			occ = 1;
		return (uint32_t)occ;
	};
	constexpr uint32_t bsm_u1 = BaseMxfp4SharedBytes<1u, 2u, true>(E);
	constexpr uint32_t bsm_u2 = BaseMxfp4SharedBytes<2u, 2u, true>(E);
	constexpr uint32_t bsm_u4 = BaseMxfp4SharedBytes<4u, 3u, true>(E);
	constexpr uint32_t bsm_d1 = BaseMxfp4SharedBytes<1u, 2u, false>(E);
	constexpr uint32_t bsm_d2 = BaseMxfp4SharedBytes<2u, 2u, false>(E);
	constexpr uint32_t bsm_d4 = BaseMxfp4SharedBytes<4u, 3u, false>(E);
	const uint32_t bocc_u1 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<1u, 2u, false, true>, bsm_u1);
	const uint32_t bocc_u2 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<2u, 2u, false, true>, bsm_u2);
	const uint32_t bocc_u4 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<4u, 3u, false, true>, bsm_u4);
	const uint32_t bocc_d1 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<1u, 2u, true, false>, bsm_d1);
	const uint32_t bocc_d2 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<2u, 2u, true, false>, bsm_d2);
	const uint32_t bocc_d4 = prep_base((const void *)BaseMimoMxfp4ExpertsKernel<4u, 3u, true, false>, bsm_d4);

	{
		std::vector<uint32_t> off, src;
		std::vector<float> pwt;
		BuildSyntheticRoutes(E, 512u, G::kTopK, G::kExperts, off, src, pwt);
		const uint32_t packed = off[E];
		cudaMemcpy(d_off, off.data(), off.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

		MimoMoeFastW1W3SiluKernel<G><<<sms * 4u, 128u, 0, stream>>>(w1, s1, w3, s3, act, d_src, d_off, act_ref, 512u);
		MimoMoeFastW2Kernel<G><<<sms * 4u, 128u, 0, stream>>>(w2, s2, act_ref, d_off, out_ref);

		Check(Model::template Mxfp4Launch<16u, 64u, 1u, 8u, 2u, false, true>(w1, s1, w3, s3, act, act_16, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u16");
		Check(Model::template Mxfp4Launch<32u, 64u, 2u, 4u, 2u, false, true>(w1, s1, w3, s3, act, act_32, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u32");
		Check(Model::template Mxfp4Launch<64u, 64u, 2u, 4u, 3u, false, true>(w1, s1, w3, s3, act, act_64, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u64");
		Check(Model::template Mxfp4Launch<96u, 64u, 2u, 4u, 2u, false, true>(w1, s1, w3, s3, act, act_96, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u96");
		Check(Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 2u, true, false>(w2, s2, 0, 0, act_16, out_16, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d16");
		Check(Model::template Mxfp4Launch<32u, 128u, 2u, 4u, 2u, true, false>(w2, s2, 0, 0, act_32, out_32, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d32");
		Check(Model::template Mxfp4Launch<64u, 128u, 2u, 4u, 3u, true, false>(w2, s2, 0, 0, act_64, out_64, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d64");
		Check(Model::template Mxfp4Launch<96u, 128u, 2u, 4u, 2u, true, false>(w2, s2, 0, 0, act_96, out_96, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d96");
		cudaStreamSynchronize(stream);

		std::vector<uint16_t> ha_ref((uint64_t)packed * I), ha16((uint64_t)packed * I), ha32((uint64_t)packed * I), ha64((uint64_t)packed * I), ha96((uint64_t)packed * I);
		std::vector<uint16_t> ho_ref((uint64_t)packed * H), ho16((uint64_t)packed * H), ho32((uint64_t)packed * H), ho64((uint64_t)packed * H), ho96((uint64_t)packed * H);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha32.data(), act_32, ha32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha64.data(), act_64, ha64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha96.data(), act_96, ha96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho32.data(), out_32, ho32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho64.data(), out_64, ho64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho96.data(), out_96, ho96.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

		uint32_t ulp_u = 0u, ulp_d = 0u;
		Check(RelL2(ha96, ha_ref, ha96.size(), &ulp_u) < 2e-3, tag, "W1W3 vs fast scalar ref", RelL2(ha96, ha_ref, ha96.size(), 0), (double)ulp_u);
		Check(DiffCount(ha16, ha32, ha16.size()) == 0u && DiffCount(ha32, ha64, ha32.size()) == 0u && DiffCount(ha64, ha96, ha64.size()) == 0u,
			tag, "W1W3 0-ULP M16==M32==M64==M96");
		Check(RelL2(ho96, ho_ref, ho96.size(), &ulp_d) < 2e-3, tag, "W2 vs fast scalar ref", RelL2(ho96, ho_ref, ho96.size(), 0), (double)ulp_d);
		Check(DiffCount(ho16, ho32, ho16.size()) == 0u && DiffCount(ho32, ho64, ho32.size()) == 0u && DiffCount(ho64, ho96, ho64.size()) == 0u,
			tag, "W2 0-ULP M16==M32==M64==M96");
	}

	for ( uint32_t rows : { 512u, 1024u, 2048u } )
	{
		std::vector<uint32_t> off, src;
		std::vector<float> pwt;
		BuildSyntheticRoutes(E, rows, G::kTopK, G::kExperts, off, src, pwt);
		const uint32_t packed = off[E];
		cudaMemcpy(d_off, off.data(), off.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		const uint32_t mean = (packed + E - 1u) / E;

		const double t_base = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 32u )
			{
				BaseMimoMxfp4ExpertsKernel<4u, 3u, false, true><<<sms * bocc_u4, 256u, bsm_u4, stream>>>(w1, s1, w3, s3, act, act_ref, d_off, d_src, E, H, I);
				BaseMimoMxfp4ExpertsKernel<4u, 3u, true, false><<<sms * bocc_d4, 256u, bsm_d4, stream>>>(w2, s2, 0, 0, act_ref, out_ref, d_off, 0, E, I, H);
			}
			else if ( mean > 16u )
			{
				BaseMimoMxfp4ExpertsKernel<2u, 2u, false, true><<<sms * bocc_u2, 256u, bsm_u2, stream>>>(w1, s1, w3, s3, act, act_ref, d_off, d_src, E, H, I);
				BaseMimoMxfp4ExpertsKernel<2u, 2u, true, false><<<sms * bocc_d2, 256u, bsm_d2, stream>>>(w2, s2, 0, 0, act_ref, out_ref, d_off, 0, E, I, H);
			}
			else
			{
				BaseMimoMxfp4ExpertsKernel<1u, 2u, false, true><<<sms * bocc_u1, 256u, bsm_u1, stream>>>(w1, s1, w3, s3, act, act_ref, d_off, d_src, E, H, I);
				BaseMimoMxfp4ExpertsKernel<1u, 2u, true, false><<<sms * bocc_d1, 256u, bsm_d1, stream>>>(w2, s2, 0, 0, act_ref, out_ref, d_off, 0, E, I, H);
			}
		});
		const double t_new = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 40u )
			{
				Model::template Mxfp4Launch<96u, 64u, 2u, 4u, 2u, false, true>(w1, s1, w3, s3, act, act_96, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<96u, 128u, 2u, 4u, 2u, true, false>(w2, s2, 0, 0, act_96, out_96, d_off, 0, packed, I, H, sms, stream);
			}
			else if ( mean > 24u )
			{
				Model::template Mxfp4Launch<64u, 64u, 2u, 4u, 3u, false, true>(w1, s1, w3, s3, act, act_64, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<64u, 128u, 2u, 4u, 3u, true, false>(w2, s2, 0, 0, act_64, out_64, d_off, 0, packed, I, H, sms, stream);
			}
			else if ( mean > 10u )
			{
				Model::template Mxfp4Launch<32u, 64u, 2u, 4u, 2u, false, true>(w1, s1, w3, s3, act, act_32, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<32u, 128u, 2u, 4u, 2u, true, false>(w2, s2, 0, 0, act_32, out_32, d_off, 0, packed, I, H, sms, stream);
			}
			else
			{
				Model::template Mxfp4Launch<16u, 64u, 1u, 8u, 2u, false, true>(w1, s1, w3, s3, act, act_16, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 2u, true, false>(w2, s2, 0, 0, act_16, out_16, d_off, 0, packed, I, H, sms, stream);
			}
		});
		const double flops = 6.0 * (double)packed * (double)H * (double)I;
		printf("  %-14s M=%4u (packed=%4u, mean=%2u) | BASE=%7.1f us  NEW=%7.1f us (%5.2fx, %5.1f TF/s)\n",
			tag, rows, packed, mean, t_base, t_new, t_base / t_new, flops / (t_new * 1e6));
	}
}

int main(void)
{
	cudaDeviceProp prop;
	if ( cudaGetDeviceProperties(&prop, 0) != cudaSuccess )
	{
		fprintf(stderr, "No CUDA device available\n");
		return 1;
	}
	const uint32_t sms = (uint32_t)prop.multiProcessorCount;
	printf("Device: %s (SM %d.%d, %u SMs, %zu B max optin SMEM)\n",
		prop.name, prop.major, prop.minor, sms, prop.sharedMemPerBlockOptin);
	cudaStream_t stream = 0;
	cudaStreamCreate(&stream);

	RunHyMoeTests(sms, stream);
	printf("\n=== 2. MiMo26 MXFP4 MoE W1W3+SiLU & W2 (Paired BASE vs NEW) ===\n");
	RunMimoMxfp4Tests<Mimo26ProGeometry<16u>>("mimo_pro_mx4", sms, stream);
	RunMimoMxfp4Tests<Mimo26FlashGeometry<16u>>("mimo_flash_mx4", sms, stream);

	cudaStreamDestroy(stream);
	if ( g_failures != 0 )
	{
		fprintf(stderr, "\nFAILED: %d check(s) failed\n", g_failures);
		return 1;
	}
	printf("\nALL EXACTNESS & 0-ULP CHECKS PASSED!\n");
	return 0;
}
