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

static void RunHyScaledDenseTests(uint32_t sms, cudaStream_t stream)
{
	using HyProd = HyModel<Hy4Geometry<16u>>;
	DevPool pool(stream);
	struct ShapeSpec { const char *name; uint32_t n; uint32_t k; };
	const ShapeSpec shapes[] = {
		{ "hy4_q_a", 2048u, 6144u },
		{ "hy4_kv_a", 576u, 6144u },
		{ "hy4_out", 6144u, 1024u },
		{ "hy4_index_q", 4096u, 2048u },
		{ "hy4_shared_gu", 256u, 6144u },
		{ "hy4_shared_dn", 6144u, 128u },
	};
	const uint32_t max_rows = 2048u;
	uint16_t *act = (uint16_t *)pool.Fill((uint64_t)max_rows * 6144u, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *out_new = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));
	uint16_t *out_sub = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));

	printf("\n=== 1. Hy4 Scaled Dense (MXFP8 fused-E8M0 & FP8 block-128) ===\n");
	for ( const ShapeSpec &sp : shapes )
	{
		uint8_t *w = (uint8_t *)pool.Fill((uint64_t)sp.n * sp.k, 1u, FILL_FP8, 0.0f, 0.0f);
		uint8_t *s_mx = (uint8_t *)pool.Fill((uint64_t)sp.n * (sp.k / 32u), 1u, FILL_E8M0, 0.0f, 123.0f);
		float *s_fp8 = (float *)pool.Fill((uint64_t)sp.n * (sp.k / 128u), sizeof(float), FILL_F32, 0.05f, 0.06f);
		HyScaled w_mx = { w, s_mx, HY_CODEC_MXFP8 };
		HyScaled w_f8 = { w, s_fp8, HY_CODEC_FP8_BLOCK128 };
		const LmScaleTensor sc_mx = LmScaleTensorBuild(s_mx, LM_SCALE_ENCODING_UE8M0, 1u, sp.n, sp.k, 1u, 32u);
		const LmScaleTensor sc_f8 = LmScaleTensorBuild(s_fp8, LM_SCALE_ENCODING_F32, 1u, sp.n, sp.k, 1u, 128u);

		const uint32_t check_rows = 512u;
		Check(HyProd::LaunchScaledDense<LmMxfp8>(w_mx, act, out_new, check_rows, sp.k, sp.n, sms, stream) == LM_LAUNCH_OK, sp.name, "mxfp8 new launch");
		Check(LmStreamGemmDense<LmMxfp8>(w, sc_mx, act, out_ref, (float *)0, check_rows, sp.k, sp.n, 0u, 0u, sms, stream) == LM_LAUNCH_OK, sp.name, "mxfp8 ref launch");
		cudaStreamSynchronize(stream);
		std::vector<uint16_t> h_new((uint64_t)check_rows * sp.n), h_ref((uint64_t)check_rows * sp.n);
		cudaMemcpy(h_new.data(), out_new, h_new.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ref.data(), out_ref, h_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		uint32_t max_ulp = 0u;
		const double rel_mx = RelL2(h_new, h_ref, h_new.size(), &max_ulp);
		Check(rel_mx < 1e-3 && max_ulp <= 1u, sp.name, "mxfp8 vs ref exactness", rel_mx, (double)max_ulp);

		const uint32_t sub_rows_list[] = { 1u, 16u, 17u, 32u, 33u, 64u };
		for ( uint32_t sr : sub_rows_list )
		{
			Check(HyProd::LaunchScaledDense<LmMxfp8>(w_mx, act, out_sub, sr, sp.k, sp.n, sms, stream) == LM_LAUNCH_OK, sp.name, "mxfp8 sub launch");
			cudaStreamSynchronize(stream);
			std::vector<uint16_t> h_sub((uint64_t)sr * sp.n);
			cudaMemcpy(h_sub.data(), out_sub, h_sub.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
			Check(DiffCount(h_sub, h_new, h_sub.size()) == 0u, sp.name, "mxfp8 0-ULP warm==cold", (double)sr, (double)DiffCount(h_sub, h_new, h_sub.size()));
		}

		Check(HyProd::LaunchScaledDense<LmFp8>(w_f8, act, out_new, check_rows, sp.k, sp.n, sms, stream) == LM_LAUNCH_OK, sp.name, "fp8 new launch");
		Check(LmStreamGemmDense<LmFp8>(w, sc_f8, act, out_ref, (float *)0, check_rows, sp.k, sp.n, 0u, 0u, sms, stream) == LM_LAUNCH_OK, sp.name, "fp8 ref launch");
		cudaStreamSynchronize(stream);
		cudaMemcpy(h_new.data(), out_new, h_new.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ref.data(), out_ref, h_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		Check(DiffCount(h_new, h_ref, h_new.size()) == 0u, sp.name, "fp8 0-ULP vs LmStreamGemmDense", (double)DiffCount(h_new, h_ref, h_new.size()), 0.0);

		for ( uint32_t rows : { 512u, 1024u, 2048u } )
		{
			const double t_mx_ref = TimeUs(stream, 4, 20, [&]() {
				LmStreamGemmDense<LmMxfp8>(w, sc_mx, act, out_ref, (float *)0, rows, sp.k, sp.n, 0u, 0u, sms, stream);
			});
			const double t_mx_new = TimeUs(stream, 4, 20, [&]() {
				HyProd::LaunchScaledDense<LmMxfp8>(w_mx, act, out_new, rows, sp.k, sp.n, sms, stream);
			});
			const double t_f8_ref = TimeUs(stream, 4, 20, [&]() {
				LmStreamGemmDense<LmFp8>(w, sc_f8, act, out_ref, (float *)0, rows, sp.k, sp.n, 0u, 0u, sms, stream);
			});
			const double t_f8_new = TimeUs(stream, 4, 20, [&]() {
				HyProd::LaunchScaledDense<LmFp8>(w_f8, act, out_new, rows, sp.k, sp.n, sms, stream);
			});
			const double flops = 2.0 * (double)rows * (double)sp.n * (double)sp.k;
			printf("  %-14s M=%4u | MXFP8: ref=%7.1f us  new=%7.1f us (%5.2fx, %5.1f TF/s) | FP8: ref=%7.1f us  new=%7.1f us (%5.2fx, %5.1f TF/s)\n",
				sp.name, rows, t_mx_ref, t_mx_new, t_mx_ref / t_mx_new, flops / (t_mx_new * 1e6),
				t_f8_ref, t_f8_new, t_f8_ref / t_f8_new, flops / (t_f8_new * 1e6));
		}
	}

	uint8_t *w_hi = (uint8_t *)pool.Fill(256ull * 128ull, 1u, FILL_FP8, 0.0f, 0.0f);
	uint8_t *s_hi = (uint8_t *)pool.Fill(256ull * 4ull, 1u, FILL_E8M0, 0.0f, 136.0f);
	HyScaled w_mx_hi = { w_hi, s_hi, HY_CODEC_MXFP8 };
	const LmScaleTensor sc_hi = LmScaleTensorBuild(s_hi, LM_SCALE_ENCODING_UE8M0, 1u, 256u, 128u, 1u, 32u);
	Check(HyProd::LaunchScaledDense<LmMxfp8>(w_mx_hi, act, out_new, 64u, 128u, 256u, sms, stream) == LM_LAUNCH_OK, "hy4_hi_e8m0", "new launch");
	Check(LmStreamGemmDense<LmMxfp8>(w_hi, sc_hi, act, out_ref, (float *)0, 64u, 128u, 256u, 0u, 0u, sms, stream) == LM_LAUNCH_OK, "hy4_hi_e8m0", "ref launch");
	cudaStreamSynchronize(stream);
	std::vector<uint16_t> h_hi_new(64u * 256u), h_hi_ref(64u * 256u);
	cudaMemcpy(h_hi_new.data(), out_new, h_hi_new.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_hi_ref.data(), out_ref, h_hi_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	uint32_t hi_ulp = 0u;
	const double hi_rel = RelL2(h_hi_new, h_hi_ref, h_hi_new.size(), &hi_ulp);
	Check(hi_rel < 1e-3 && hi_ulp <= 1u, "hy4_hi_e8m0", "e8m0>134 cold branch exactness", hi_rel, (double)hi_ulp);
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
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint32_t *d_off = (uint32_t *)pool.Alloc((E + 1u) * sizeof(uint32_t));
	uint32_t *d_src = (uint32_t *)pool.Alloc(max_packed * sizeof(uint32_t));
	float *d_pwt = (float *)pool.Alloc(max_packed * sizeof(float));

	const LmScaleTensor t1_mx = LmScaleTensorBuild(s1_mx, LM_SCALE_ENCODING_UE8M0, E, I, H, 1u, 32u);
	const LmScaleTensor t3_mx = LmScaleTensorBuild(s3_mx, LM_SCALE_ENCODING_UE8M0, E, I, H, 1u, 32u);
	const LmScaleTensor t2_mx = LmScaleTensorBuild(s2_mx, LM_SCALE_ENCODING_UE8M0, E, H, I, 1u, 32u);
	const LmScaleTensor t1_f8 = LmScaleTensorBuild(s1_f8, LM_SCALE_ENCODING_F32, E, I, H, 1u, 128u);
	const LmScaleTensor t3_f8 = LmScaleTensorBuild(s3_f8, LM_SCALE_ENCODING_F32, E, I, H, 1u, 128u);
	const LmScaleTensor t2_f8 = LmScaleTensorBuild(s2_f8, LM_SCALE_ENCODING_F32, E, H, I, 1u, 128u);

	auto set_smem = [](const void *fn, uint32_t bytes) {
		if ( bytes > 49152u )
			cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
	};
	constexpr uint32_t smem_u16 = 3u * (16u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_u32 = 3u * (32u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_u64 = 3u * (64u * 256u + 2u * 64u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d16 = 3u * (16u * 256u + 128u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d32 = 3u * (32u * 256u + 128u * 128u) + (E + 1u) * 4u;
	constexpr uint32_t smem_d64 = 3u * (64u * 256u + 128u * 128u) + (E + 1u) * 4u;
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 16u, 1u, 8u, 3u>, smem_u16);
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 3u>, smem_u32);
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u>, smem_u64);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 16u, 1u, 8u, 3u>, smem_d16);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 2u, 4u, 3u>, smem_d32);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 2u, 4u, 3u>, smem_d64);
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 16u, 1u, 8u, 3u>, smem_u16);
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 32u, 2u, 4u, 3u>, smem_u32);
	set_smem((const void *)HyMoeMmaW1W3SwigluKernel<G, LmFp8, 64u, 2u, 4u, 3u>, smem_u64);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 16u, 1u, 8u, 3u>, smem_d16);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 32u, 2u, 4u, 3u>, smem_d32);
	set_smem((const void *)HyMoeMmaW2Kernel<G, LmFp8, 64u, 2u, 4u, 3u>, smem_d64);

	printf("\n=== 2. Hy4 MoE W1W3+SwiGLU & W2 (MXFP8 & FP8) ===\n");
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

		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 16u, 1u, 8u, 3u><<<sms, 256u, smem_u16, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_16);
		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_u32, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_32);
		HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_u64, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_64);
		HyMoeMmaW2Kernel<G, LmMxfp8, 16u, 1u, 8u, 3u><<<sms, 256u, smem_d16, stream>>>(w2, t2_mx, act_16, d_off, out_16);
		HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_d32, stream>>>(w2, t2_mx, act_32, d_off, out_32);
		HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_d64, stream>>>(w2, t2_mx, act_64, d_off, out_64);
		cudaStreamSynchronize(stream);

		std::vector<uint16_t> ha_ref((uint64_t)packed * I), ha16((uint64_t)packed * I), ha32((uint64_t)packed * I), ha64((uint64_t)packed * I);
		std::vector<uint16_t> ho_ref((uint64_t)packed * H), ho16((uint64_t)packed * H), ho32((uint64_t)packed * H), ho64((uint64_t)packed * H);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha32.data(), act_32, ha32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha64.data(), act_64, ha64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho32.data(), out_32, ho32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho64.data(), out_64, ho64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

		uint32_t ulp_u = 0u, ulp_d = 0u;
		Check(RelL2(ha64, ha_ref, ha64.size(), &ulp_u) < 2e-3, "hy4_moe_mxfp8", "W1W3 vs ref", RelL2(ha64, ha_ref, ha64.size(), 0), (double)ulp_u);
		Check(DiffCount(ha16, ha32, ha16.size()) == 0u && DiffCount(ha32, ha64, ha32.size()) == 0u, "hy4_moe_mxfp8", "W1W3 0-ULP M16==M32==M64");
		Check(RelL2(ho64, ho_ref, ho64.size(), &ulp_d) < 2e-3, "hy4_moe_mxfp8", "W2 vs ref", RelL2(ho64, ho_ref, ho64.size(), 0), (double)ulp_d);
		Check(DiffCount(ho16, ho32, ho16.size()) == 0u && DiffCount(ho32, ho64, ho32.size()) == 0u, "hy4_moe_mxfp8", "W2 0-ULP M16==M32==M64");

		LmStreamGemmGrouped<LmFp8>(w1, t1_f8, act, gate_ref, d_off, d_src, E, packed, H, I, sms, stream);
		LmStreamGemmGrouped<LmFp8>(w3, t3_f8, act, up_ref, d_off, d_src, E, packed, H, I, sms, stream);
		HyMoeSwigluPackedKernel<<<dim3(LmVecBlocks<HY_THREADS>(I), packed), HY_THREADS, 0, stream>>>(
			gate_ref, up_ref, d_pwt, d_off, E, act_ref, I, G::kSwigluLimit);
		LmStreamGemmGrouped<LmFp8>(w2, t2_f8, act_ref, out_ref, d_off, (const uint32_t *)0, E, packed, I, H, sms, stream);

		HyMoeMmaW1W3SwigluKernel<G, LmFp8, 16u, 1u, 8u, 3u><<<sms, 256u, smem_u16, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_16);
		HyMoeMmaW1W3SwigluKernel<G, LmFp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_u64, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_64);
		HyMoeMmaW2Kernel<G, LmFp8, 16u, 1u, 8u, 3u><<<sms, 256u, smem_d16, stream>>>(w2, t2_f8, act_16, d_off, out_16);
		HyMoeMmaW2Kernel<G, LmFp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_d64, stream>>>(w2, t2_f8, act_64, d_off, out_64);
		cudaStreamSynchronize(stream);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha64.data(), act_64, ha64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho64.data(), out_64, ho64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		Check(DiffCount(ha64, ha_ref, ha64.size()) == 0u && DiffCount(ha16, ha64, ha16.size()) == 0u, "hy4_moe_fp8", "W1W3 0-ULP exact vs ref");
		Check(DiffCount(ho64, ho_ref, ho64.size()) == 0u && DiffCount(ho16, ho64, ho16.size()) == 0u, "hy4_moe_fp8", "W2 0-ULP exact vs ref");
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
			if ( mean > 24u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_u64, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_64);
				HyMoeMmaW2Kernel<G, LmMxfp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_d64, stream>>>(w2, t2_mx, act_64, d_off, out_64);
			}
			else
			{
				HyMoeMmaW1W3SwigluKernel<G, LmMxfp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_u32, stream>>>(w1, t1_mx, w3, t3_mx, act, d_pwt, d_src, d_off, act_32);
				HyMoeMmaW2Kernel<G, LmMxfp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_d32, stream>>>(w2, t2_mx, act_32, d_off, out_32);
			}
		});
		const double t_new_f8 = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 24u )
			{
				HyMoeMmaW1W3SwigluKernel<G, LmFp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_u64, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_64);
				HyMoeMmaW2Kernel<G, LmFp8, 64u, 2u, 4u, 3u><<<sms, 256u, smem_d64, stream>>>(w2, t2_f8, act_64, d_off, out_64);
			}
			else
			{
				HyMoeMmaW1W3SwigluKernel<G, LmFp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_u32, stream>>>(w1, t1_f8, w3, t3_f8, act, d_pwt, d_src, d_off, act_32);
				HyMoeMmaW2Kernel<G, LmFp8, 32u, 2u, 4u, 3u><<<sms, 256u, smem_d32, stream>>>(w2, t2_f8, act_32, d_off, out_32);
			}
		});
		printf("  hy4_moe        M=%4u (packed=%4u, mean=%2u) | MXFP8: ref=%7.1f us  new=%7.1f us (%5.2fx) | FP8 new=%7.1f us\n",
			rows, packed, mean, t_ref_mx, t_new_mx, t_ref_mx / t_new_mx, t_new_f8);
	}
}

static void RunMimoFp8DenseTests(uint32_t sms, cudaStream_t stream)
{
	using ProModel = MimoModel<Mimo26ProGeometry<16u>>;
	DevPool pool(stream);
	struct ShapeSpec { const char *name; uint32_t n; uint32_t k; };
	const ShapeSpec shapes[] = {
		{ "pro_qkv", 1696u, 6144u },
		{ "pro_dense_gu", 2304u, 6144u },
		{ "pro_dense_dn", 6144u, 1152u },
		{ "flash_qkv_s", 928u, 4096u },
		{ "flash_qkv_f", 848u, 4096u },
	};
	const uint32_t max_rows = 2048u;
	uint16_t *act = (uint16_t *)pool.Fill((uint64_t)max_rows * 6144u, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *out_new = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));
	uint16_t *out_sub = (uint16_t *)pool.Alloc((uint64_t)max_rows * 6144u * sizeof(uint16_t));

	printf("\n=== 3. MiMo26 FP8 Block-128 Dense ===\n");
	for ( const ShapeSpec &sp : shapes )
	{
		uint8_t *w = (uint8_t *)pool.Fill((uint64_t)sp.n * sp.k, 1u, FILL_FP8, 0.0f, 0.0f);
		float *s = (float *)pool.Fill((uint64_t)sp.n * (sp.k / 128u), sizeof(float), FILL_F32, 0.05f, 0.06f);
		MimoScaled ws = { w, s };
		const LmScaleTensor sc = LmScaleTensorBuild(s, LM_SCALE_ENCODING_F32, 1u, sp.n, sp.k, 1u, 128u);

		const uint32_t check_rows = 512u;
		Check(ProModel::LaunchFp8Dense(ws, sc, act, out_new, check_rows, sp.k, sp.n, sms, stream) == LM_LAUNCH_OK, sp.name, "fp8 new launch");
		Check(LmStreamGemmDense<LmFp8>(w, sc, act, out_ref, (float *)0, check_rows, sp.k, sp.n, 0u, 0u, sms, stream) == LM_LAUNCH_OK, sp.name, "fp8 ref launch");
		cudaStreamSynchronize(stream);
		std::vector<uint16_t> h_new((uint64_t)check_rows * sp.n), h_ref((uint64_t)check_rows * sp.n);
		cudaMemcpy(h_new.data(), out_new, h_new.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ref.data(), out_ref, h_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		Check(DiffCount(h_new, h_ref, h_new.size()) == 0u, sp.name, "0-ULP vs LmStreamGemmDense", (double)DiffCount(h_new, h_ref, h_new.size()), 0.0);

		for ( uint32_t sr : { 1u, 16u, 17u, 32u, 64u } )
		{
			Check(ProModel::LaunchFp8Dense(ws, sc, act, out_sub, sr, sp.k, sp.n, sms, stream) == LM_LAUNCH_OK, sp.name, "fp8 sub launch");
			cudaStreamSynchronize(stream);
			std::vector<uint16_t> h_sub((uint64_t)sr * sp.n);
			cudaMemcpy(h_sub.data(), out_sub, h_sub.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
			Check(DiffCount(h_sub, h_new, h_sub.size()) == 0u, sp.name, "0-ULP warm==cold", (double)sr, (double)DiffCount(h_sub, h_new, h_sub.size()));
		}

		for ( uint32_t rows : { 512u, 1024u, 2048u } )
		{
			const double t_ref = TimeUs(stream, 4, 20, [&]() {
				LmStreamGemmDense<LmFp8>(w, sc, act, out_ref, (float *)0, rows, sp.k, sp.n, 0u, 0u, sms, stream);
			});
			const double t_new = TimeUs(stream, 4, 20, [&]() {
				ProModel::LaunchFp8Dense(ws, sc, act, out_new, rows, sp.k, sp.n, sms, stream);
			});
			const double flops = 2.0 * (double)rows * (double)sp.n * (double)sp.k;
			printf("  %-14s M=%4u | ref=%7.1f us  new=%7.1f us (%5.2fx, %5.1f TF/s)\n",
				sp.name, rows, t_ref, t_new, t_ref / t_new, flops / (t_new * 1e6));
		}
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
	uint16_t *act_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * I * sizeof(uint16_t));
	uint16_t *out_16 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_32 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_64 = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint16_t *out_ref = (uint16_t *)pool.Alloc((uint64_t)max_packed * H * sizeof(uint16_t));
	uint32_t *d_off = (uint32_t *)pool.Alloc((E + 1u) * sizeof(uint32_t));
	uint32_t *d_src = (uint32_t *)pool.Alloc(max_packed * sizeof(uint32_t));

	{
		std::vector<uint32_t> off, src;
		std::vector<float> pwt;
		BuildSyntheticRoutes(E, 512u, G::kTopK, G::kExperts, off, src, pwt);
		const uint32_t packed = off[E];
		cudaMemcpy(d_off, off.data(), off.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

		MimoMoeFastW1W3SiluKernel<G><<<sms * 4u, 128u, 0, stream>>>(w1, s1, w3, s3, act, d_src, d_off, act_ref, 512u);
		MimoMoeFastW2Kernel<G><<<sms * 4u, 128u, 0, stream>>>(w2, s2, act_ref, d_off, out_ref);

		Check(Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 3u, false, true>(w1, s1, w3, s3, act, act_16, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u16");
		Check(Model::template Mxfp4Launch<32u, 128u, 1u, 8u, 3u, false, true>(w1, s1, w3, s3, act, act_32, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u32");
		Check(Model::template Mxfp4Launch<64u, 64u, 2u, 4u, 3u, false, true>(w1, s1, w3, s3, act, act_64, d_off, d_src, packed, H, I, sms, stream) == LM_LAUNCH_OK, tag, "u64");
		Check(Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 3u, true, false>(w2, s2, 0, 0, act_16, out_16, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d16");
		Check(Model::template Mxfp4Launch<32u, 256u, 1u, 8u, 3u, true, false>(w2, s2, 0, 0, act_32, out_32, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d32");
		Check(Model::template Mxfp4Launch<64u, 128u, 2u, 4u, 3u, true, false>(w2, s2, 0, 0, act_64, out_64, d_off, 0, packed, I, H, sms, stream) == LM_LAUNCH_OK, tag, "d64");
		cudaStreamSynchronize(stream);

		std::vector<uint16_t> ha_ref((uint64_t)packed * I), ha16((uint64_t)packed * I), ha32((uint64_t)packed * I), ha64((uint64_t)packed * I);
		std::vector<uint16_t> ho_ref((uint64_t)packed * H), ho16((uint64_t)packed * H), ho32((uint64_t)packed * H), ho64((uint64_t)packed * H);
		cudaMemcpy(ha_ref.data(), act_ref, ha_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha16.data(), act_16, ha16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha32.data(), act_32, ha32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha64.data(), act_64, ha64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho_ref.data(), out_ref, ho_ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho16.data(), out_16, ho16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho32.data(), out_32, ho32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ho64.data(), out_64, ho64.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

		uint32_t ulp_u = 0u, ulp_d = 0u;
		Check(RelL2(ha64, ha_ref, ha64.size(), &ulp_u) < 2e-3, tag, "W1W3 vs fast scalar ref", RelL2(ha64, ha_ref, ha64.size(), 0), (double)ulp_u);
		Check(DiffCount(ha16, ha32, ha16.size()) == 0u && DiffCount(ha32, ha64, ha32.size()) == 0u, tag, "W1W3 0-ULP M16==M32==M64");
		Check(RelL2(ho64, ho_ref, ho64.size(), &ulp_d) < 2e-3, tag, "W2 vs fast scalar ref", RelL2(ho64, ho_ref, ho64.size(), 0), (double)ulp_d);
		Check(DiffCount(ho16, ho32, ho16.size()) == 0u && DiffCount(ho32, ho64, ho32.size()) == 0u, tag, "W2 0-ULP M16==M32==M64");
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
		const double t_new = TimeUs(stream, 2, 10, [&]() {
			if ( mean > 24u )
			{
				Model::template Mxfp4Launch<64u, 64u, 2u, 4u, 3u, false, true>(w1, s1, w3, s3, act, act_64, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<64u, 128u, 2u, 4u, 3u, true, false>(w2, s2, 0, 0, act_64, out_64, d_off, 0, packed, I, H, sms, stream);
			}
			else if ( mean > 10u )
			{
				Model::template Mxfp4Launch<32u, 128u, 1u, 8u, 3u, false, true>(w1, s1, w3, s3, act, act_32, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<32u, 256u, 1u, 8u, 3u, true, false>(w2, s2, 0, 0, act_32, out_32, d_off, 0, packed, I, H, sms, stream);
			}
			else
			{
				Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 3u, false, true>(w1, s1, w3, s3, act, act_16, d_off, d_src, packed, H, I, sms, stream);
				Model::template Mxfp4Launch<16u, 128u, 1u, 8u, 3u, true, false>(w2, s2, 0, 0, act_16, out_16, d_off, 0, packed, I, H, sms, stream);
			}
		});
		const double flops = 6.0 * (double)packed * (double)H * (double)I;
		printf("  %-14s M=%4u (packed=%4u, mean=%2u) | new=%7.1f us (%5.1f TF/s)\n", tag, rows, packed, mean, t_new, flops / (t_new * 1e6));
	}
}

static void RunMimoLocalAttnTests(cudaStream_t stream)
{
	DevPool pool(stream);
	constexpr uint32_t HD = 192u, VD = 128u, SLOT = HD + VD;
	const uint32_t rows = 512u, prior = 3584u, rank_heads = 4u, group = 16u;
	uint16_t *q = (uint16_t *)pool.Fill((uint64_t)rows * rank_heads * HD, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *kv_p = (uint16_t *)pool.Fill((uint64_t)prior * SLOT, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	uint16_t *kv_c = (uint16_t *)pool.Fill((uint64_t)rows * SLOT, sizeof(uint16_t), FILL_BF16, 0.25f, 0.0f);
	float *sinks = (float *)pool.Fill(rank_heads, sizeof(float), FILL_F32, 0.5f, 0.2f);
	uint16_t *out4 = (uint16_t *)pool.Alloc((uint64_t)rows * rank_heads * VD * sizeof(uint16_t));
	uint16_t *out2 = (uint16_t *)pool.Alloc((uint64_t)rows * rank_heads * VD * sizeof(uint16_t));
	std::vector<uint32_t> h_pos(rows);
	for ( uint32_t i = 0u; i < rows; ++i )
		h_pos[i] = prior + i;
	uint32_t *d_pos = (uint32_t *)pool.Alloc(rows * sizeof(uint32_t));
	cudaMemcpy(d_pos, h_pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);

	printf("\n=== 5. MiMo26 Local Prefill Attn (Q_WARPS=2 vs Q_WARPS=4 0-ULP) ===\n");
	const float scale = 1.0f / std::sqrt((float)HD);
	MimoLocalPrefillAttnKernel<HD, VD, true, 4u><<<dim3(rank_heads, (rows + 63u) / 64u), 128u, 0, stream>>>(
		q, kv_p, kv_c, d_pos, sinks, rows, rank_heads, 0u, group, 0u, 0u, prior, 128u, scale, out4);
	MimoLocalPrefillAttnKernel<HD, VD, true, 2u><<<dim3(rank_heads, (rows + 31u) / 32u), 64u, 0, stream>>>(
		q, kv_p, kv_c, d_pos, sinks, rows, rank_heads, 0u, group, 0u, 0u, prior, 128u, scale, out2);
	cudaStreamSynchronize(stream);
	std::vector<uint16_t> h4((uint64_t)rows * rank_heads * VD), h2((uint64_t)rows * rank_heads * VD);
	cudaMemcpy(h4.data(), out4, h4.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h2.data(), out2, h2.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	Check(DiffCount(h2, h4, h4.size()) == 0u, "mimo_attn_swa", "Q_WARPS=2 == Q_WARPS=4 0-ULP");

	MimoLocalPrefillAttnKernel<HD, VD, false, 4u><<<dim3(rank_heads, (rows + 63u) / 64u), 128u, 0, stream>>>(
		q, kv_p, kv_c, d_pos, (const float *)0, rows, rank_heads, 0u, group, 0u, 0u, prior, 0u, scale, out4);
	MimoLocalPrefillAttnKernel<HD, VD, false, 2u><<<dim3(rank_heads, (rows + 31u) / 32u), 64u, 0, stream>>>(
		q, kv_p, kv_c, d_pos, (const float *)0, rows, rank_heads, 0u, group, 0u, 0u, prior, 0u, scale, out2);
	cudaStreamSynchronize(stream);
	cudaMemcpy(h4.data(), out4, h4.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h2.data(), out2, h2.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	Check(DiffCount(h2, h4, h4.size()) == 0u, "mimo_attn_full", "Q_WARPS=2 == Q_WARPS=4 0-ULP");

	const double t4 = TimeUs(stream, 4, 20, [&]() {
		MimoLocalPrefillAttnKernel<HD, VD, false, 4u><<<dim3(rank_heads, (rows + 63u) / 64u), 128u, 0, stream>>>(
			q, kv_p, kv_c, d_pos, (const float *)0, rows, rank_heads, 0u, group, 0u, 0u, prior, 0u, scale, out4);
	});
	const double t2 = TimeUs(stream, 4, 20, [&]() {
		MimoLocalPrefillAttnKernel<HD, VD, false, 2u><<<dim3(rank_heads, (rows + 31u) / 32u), 64u, 0, stream>>>(
			q, kv_p, kv_c, d_pos, (const float *)0, rows, rank_heads, 0u, group, 0u, 0u, prior, 0u, scale, out2);
	});
	printf("  flash_full_attn M=512 ctx=4096 | Q_WARPS=4: %7.1f us  Q_WARPS=2: %7.1f us (%5.2fx)\n", t4, t2, t4 / t2);
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

	RunHyScaledDenseTests(sms, stream);
	RunHyMoeTests(sms, stream);
	RunMimoFp8DenseTests(sms, stream);
	printf("\n=== 4. MiMo26 MXFP4 MoE W1W3+SiLU & W2 (0-bank-conflict ldmatrix) ===\n");
	RunMimoMxfp4Tests<Mimo26ProGeometry<16u>>("mimo_pro_mx4", sms, stream);
	RunMimoMxfp4Tests<Mimo26FlashGeometry<16u>>("mimo_flash_mx4", sms, stream);
	RunMimoLocalAttnTests(stream);

	cudaStreamDestroy(stream);
	if ( g_failures != 0 )
	{
		fprintf(stderr, "\nFAILED: %d check(s) failed\n", g_failures);
		return 1;
	}
	printf("\nALL EXACTNESS & 0-ULP CHECKS PASSED!\n");
	return 0;
}
