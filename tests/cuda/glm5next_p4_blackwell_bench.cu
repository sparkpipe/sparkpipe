#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "glm5_next_stage.cuh"

#define BENCH_WARMUP 5u
#define BENCH_ITERS 20u

#define CHECK_CUDA(...) do { cudaError_t err_ = (__VA_ARGS__); if ( err_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #__VA_ARGS__, cudaGetErrorString(err_)); exit(1); } } while (0)
#define CHECK_GPU(...) do { SparkGpuError err_ = (__VA_ARGS__); if ( err_ != SPARK_GPU_SUCCESS ) { printf("FAIL gpu %s: %d\n", #__VA_ARGS__, (int)err_); exit(1); } } while (0)
#define CHECK_OK(...) do { int32_t st_ = (__VA_ARGS__); if ( st_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #__VA_ARGS__, st_); exit(1); } } while (0)

static int g_failures = 0;
static uint32_t g_sms = 48u;

static __host__ __device__ __forceinline__ uint64_t Mix64(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static __global__ void FillBf16Kernel(uint16_t *dst, uint64_t count, uint64_t seed, float offset, float scale)
{
	for ( uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += (uint64_t)gridDim.x * blockDim.x )
	{
		const float u = (float)((int64_t)(Mix64(idx + seed) % 2001u) - 1000) * 0.001f;
		dst[idx] = LmFloatToBf16(offset + scale * u);
	}
}

static __global__ void FillF32Kernel(float *dst, uint64_t count, uint64_t seed, float offset, float scale)
{
	for ( uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += (uint64_t)gridDim.x * blockDim.x )
	{
		const float u = (float)((int64_t)(Mix64(idx + seed) % 2001u) - 1000) * 0.001f;
		dst[idx] = offset + scale * u;
	}
}

static __global__ void FillFp8Kernel(uint8_t *dst, uint64_t count, uint64_t seed)
{
	for ( uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint64_t z = Mix64(idx * 3u + seed);
		dst[idx] = (uint8_t)(((z & 1u) << 7u) | ((3u + (z >> 1u) % 7u) << 3u) | ((z >> 8u) & 7u));
	}
}

static void FillBf16(uint16_t *dst, uint64_t count, uint64_t seed, float offset, float scale, cudaStream_t stream)
{
	FillBf16Kernel<<<1024u, 256u, 0u, stream>>>(dst, count, seed, offset, scale);
	CHECK_CUDA(cudaGetLastError());
}

static void FillF32(float *dst, uint64_t count, uint64_t seed, float offset, float scale, cudaStream_t stream)
{
	FillF32Kernel<<<1024u, 256u, 0u, stream>>>(dst, count, seed, offset, scale);
	CHECK_CUDA(cudaGetLastError());
}

static void FillFp8(uint8_t *dst, uint64_t count, uint64_t seed, cudaStream_t stream)
{
	FillFp8Kernel<<<1024u, 256u, 0u, stream>>>(dst, count, seed);
	CHECK_CUDA(cudaGetLastError());
}

static void *DevAlloc(uint64_t bytes)
{
	void *p = 0;
	CHECK_CUDA(cudaMalloc(&p, (size_t)bytes));
	CHECK_CUDA(cudaMemset(p, 0, (size_t)bytes));
	return p;
}

static float Bf16ToHost(uint16_t bits)
{
	uint32_t u = (uint32_t)bits << 16u;
	float f = 0.0f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

struct ErrStats
{
	double max_abs;
	double max_rel;
	uint64_t exact_diffs;
	uint64_t nan_inf_mismatch;
};

static ErrStats CompareBf16(const uint16_t *d_a, const uint16_t *d_b, uint64_t count, cudaStream_t stream)
{
	std::vector<uint16_t> ha(count), hb(count);
	CHECK_CUDA(cudaMemcpyAsync(ha.data(), d_a, count * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaMemcpyAsync(hb.data(), d_b, count * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaStreamSynchronize(stream));
	ErrStats st = {0.0, 0.0, 0u, 0u};
	for ( uint64_t i = 0u; i < count; ++i )
	{
		if ( ha[i] != hb[i] )
			st.exact_diffs++;
		const float fa = Bf16ToHost(ha[i]), fb = Bf16ToHost(hb[i]);
		if ( !isfinite(fa) || !isfinite(fb) )
		{
			if ( ha[i] != hb[i] )
				st.nan_inf_mismatch++;
			continue;
		}
		const double ad = fabs((double)fa - (double)fb);
		const double rd = ad / fmax(fabs((double)fb), 1e-3);
		if ( ad > st.max_abs ) st.max_abs = ad;
		if ( rd > st.max_rel ) st.max_rel = rd;
	}
	return st;
}

static ErrStats CompareF32(const float *d_a, const float *d_b, uint64_t count, cudaStream_t stream)
{
	std::vector<float> ha(count), hb(count);
	CHECK_CUDA(cudaMemcpyAsync(ha.data(), d_a, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaMemcpyAsync(hb.data(), d_b, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaStreamSynchronize(stream));
	ErrStats st = {0.0, 0.0, 0u, 0u};
	for ( uint64_t i = 0u; i < count; ++i )
	{
		uint32_t ua = 0u, ub = 0u;
		memcpy(&ua, &ha[i], 4u);
		memcpy(&ub, &hb[i], 4u);
		if ( ua != ub )
			st.exact_diffs++;
		if ( !isfinite(ha[i]) || !isfinite(hb[i]) )
		{
			if ( ua != ub )
				st.nan_inf_mismatch++;
			continue;
		}
		const double ad = fabs((double)ha[i] - (double)hb[i]);
		const double rd = ad / fmax(fabs((double)hb[i]), 1e-3);
		if ( ad > st.max_abs ) st.max_abs = ad;
		if ( rd > st.max_rel ) st.max_rel = rd;
	}
	return st;
}

template<class Fn>
static float TimeMs(Fn fn, cudaStream_t stream)
{
	for ( uint32_t i = 0u; i < BENCH_WARMUP; ++i )
		fn();
	CHECK_CUDA(cudaStreamSynchronize(stream));
	cudaEvent_t t0, t1;
	CHECK_CUDA(cudaEventCreate(&t0));
	CHECK_CUDA(cudaEventCreate(&t1));
	CHECK_CUDA(cudaEventRecord(t0, stream));
	for ( uint32_t i = 0u; i < BENCH_ITERS; ++i )
		fn();
	CHECK_CUDA(cudaEventRecord(t1, stream));
	CHECK_CUDA(cudaEventSynchronize(t1));
	float ms = 0.0f;
	CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
	CHECK_CUDA(cudaEventDestroy(t0));
	CHECK_CUDA(cudaEventDestroy(t1));
	return ms / (float)BENCH_ITERS;
}

static void CheckTol(const char *name, const ErrStats &st, double abs_tol, double rel_tol, bool require_zero_ulp = false)
{
	const bool ok = st.nan_inf_mismatch == 0u &&
		(!require_zero_ulp || st.exact_diffs == 0u) &&
		(st.max_abs <= abs_tol || st.max_rel <= rel_tol);
	printf("  [%s] %-42s max_abs=%.3e max_rel=%.3e diffs=%llu nan_mis=%llu\n",
		ok ? "PASS" : "FAIL", name, st.max_abs, st.max_rel,
		(unsigned long long)st.exact_diffs, (unsigned long long)st.nan_inf_mismatch);
	if ( !ok )
		g_failures++;
}

static void BenchHcPostDotMma(cudaStream_t stream)
{
	printf("\n=== 1. HC Post+Dot+Norm MMA (90 sites/step) ===\n");
	const uint32_t max_rows = 1024u;
	uint16_t *streams_in = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HC_FLAT * 2u);
	uint16_t *post_out = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint16_t *streams_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HC_FLAT * 2u);
	uint16_t *streams_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HC_FLAT * 2u);
	uint16_t *norm_w = (uint16_t *)DevAlloc((uint64_t)G5N_HIDDEN * 2u);
	uint16_t *normed_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint16_t *normed_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	float *fn_f32 = (float *)DevAlloc((uint64_t)G5N_HC_MIX * G5N_HC_FLAT * 4u);
	float *base_f32 = (float *)DevAlloc((uint64_t)G5N_HC_MIX * 4u);
	float *scale_f32 = (float *)DevAlloc(3u * 4u);
	float *post_init = (float *)DevAlloc((uint64_t)max_rows * G5N_HC * 4u);
	float *comb_init = (float *)DevAlloc((uint64_t)max_rows * 16u * 4u);
	float *post_ref = (float *)DevAlloc((uint64_t)max_rows * G5N_HC * 4u);
	float *comb_ref = (float *)DevAlloc((uint64_t)max_rows * 16u * 4u);
	float *post_opt = (float *)DevAlloc((uint64_t)max_rows * G5N_HC * 4u);
	float *comb_opt = (float *)DevAlloc((uint64_t)max_rows * 16u * 4u);
	float *partials = (float *)DevAlloc((uint64_t)max_rows * LM_HC_SITE_CLUSTER * (LM_HC_SITE_MIX + 1u) * 4u);

	FillBf16(streams_in, (uint64_t)max_rows * G5N_HC_FLAT, 101u, 0.0f, 0.5f, stream);
	FillBf16(post_out, (uint64_t)max_rows * G5N_HIDDEN, 102u, 0.0f, 0.5f, stream);
	FillBf16(norm_w, G5N_HIDDEN, 103u, 1.0f, 0.1f, stream);
	FillF32(fn_f32, (uint64_t)G5N_HC_MIX * G5N_HC_FLAT, 104u, 0.0f, 0.02f, stream);
	FillF32(base_f32, G5N_HC_MIX, 105u, 0.0f, 0.2f, stream);
	FillF32(scale_f32, 3u, 106u, 0.25f, 0.05f, stream);
	FillF32(post_init, (uint64_t)max_rows * G5N_HC, 107u, 1.0f, 0.2f, stream);
	FillF32(comb_init, (uint64_t)max_rows * 16u, 108u, 0.25f, 0.05f, stream);

	const uint32_t row_cases[3] = {64u, 256u, 1024u};
	for ( uint32_t ci = 0u; ci < 3u; ++ci )
	{
		const uint32_t rows = row_cases[ci];
		auto run_ref = [&]()
		{
			CHECK_CUDA(cudaMemcpyAsync(post_ref, post_init, (uint64_t)rows * G5N_HC * 4u, cudaMemcpyDeviceToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(comb_ref, comb_init, (uint64_t)rows * 16u * 4u, cudaMemcpyDeviceToDevice, stream));
			Glm5NextLayerBuffers b;
			memset(&b, 0, sizeof(b));
			b.prefill_unified = 0u;
			b.hc_collapsed_bf16 = (uint16_t *)partials;
			b.hc_post_f32 = post_ref;
			b.hc_comb_f32 = comb_ref;
			b.normed_bf16 = normed_ref;
			CHECK_OK(Glm5NextHcPostNorm(&b, streams_in, post_out, streams_ref, fn_f32, base_f32, scale_f32, norm_w, rows, (SparkGpuStream)stream));
		};
		auto run_opt = [&]()
		{
			CHECK_CUDA(cudaMemcpyAsync(post_opt, post_init, (uint64_t)rows * G5N_HC * 4u, cudaMemcpyDeviceToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(comb_opt, comb_init, (uint64_t)rows * 16u * 4u, cudaMemcpyDeviceToDevice, stream));
			Glm5NextLayerBuffers b;
			memset(&b, 0, sizeof(b));
			b.prefill_unified = 1u;
			b.hc_collapsed_bf16 = (uint16_t *)partials;
			b.hc_post_f32 = post_opt;
			b.hc_comb_f32 = comb_opt;
			b.normed_bf16 = normed_opt;
			CHECK_OK(Glm5NextHcPostNorm(&b, streams_in, post_out, streams_opt, fn_f32, base_f32, scale_f32, norm_w, rows, (SparkGpuStream)stream));
		};
		run_ref();
		run_opt();
		char label[64];
		snprintf(label, sizeof(label), "HC streams_out 0-ULP (rows=%u)", rows);
		CheckTol(label, CompareBf16(streams_opt, streams_ref, (uint64_t)rows * G5N_HC_FLAT, stream), 0.0, 0.0, true);
		snprintf(label, sizeof(label), "HC normed_bf16 (rows=%u)", rows);
		CheckTol(label, CompareBf16(normed_opt, normed_ref, (uint64_t)rows * G5N_HIDDEN, stream), 2e-2, 2e-2);
		snprintf(label, sizeof(label), "HC post_f32 (rows=%u)", rows);
		CheckTol(label, CompareF32(post_opt, post_ref, (uint64_t)rows * G5N_HC, stream), 5e-4, 5e-4);

		auto time_ref = [&]()
		{
			Glm5NextLayerBuffers b;
			memset(&b, 0, sizeof(b));
			b.prefill_unified = 0u;
			b.hc_collapsed_bf16 = (uint16_t *)partials;
			b.hc_post_f32 = post_ref;
			b.hc_comb_f32 = comb_ref;
			b.normed_bf16 = normed_ref;
			CHECK_OK(Glm5NextHcPostNorm(&b, streams_in, post_out, streams_ref, fn_f32, base_f32, scale_f32, norm_w, rows, (SparkGpuStream)stream));
		};
		auto time_opt = [&]()
		{
			Glm5NextLayerBuffers b;
			memset(&b, 0, sizeof(b));
			b.prefill_unified = 1u;
			b.hc_collapsed_bf16 = (uint16_t *)partials;
			b.hc_post_f32 = post_opt;
			b.hc_comb_f32 = comb_opt;
			b.normed_bf16 = normed_opt;
			CHECK_OK(Glm5NextHcPostNorm(&b, streams_in, post_out, streams_opt, fn_f32, base_f32, scale_f32, norm_w, rows, (SparkGpuStream)stream));
		};
		const float ms_ref = TimeMs(time_ref, stream), ms_opt = TimeMs(time_opt, stream);
		printf("  [PERF] HC PostNorm rows=%4u: ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (90 sites: %.1f -> %.1f ms)\n",
			rows, ms_ref, ms_opt, ms_ref / ms_opt, ms_ref * 90.0f, ms_opt * 90.0f);
	}

	cudaFree(streams_in); cudaFree(post_out); cudaFree(streams_ref); cudaFree(streams_opt);
	cudaFree(norm_w); cudaFree(normed_ref); cudaFree(normed_opt); cudaFree(fn_f32);
	cudaFree(base_f32); cudaFree(scale_f32); cudaFree(post_init); cudaFree(comb_init);
	cudaFree(post_ref); cudaFree(comb_ref); cudaFree(post_opt); cudaFree(comb_opt); cudaFree(partials);
}

static void BenchKdaFusedAndWy(cudaStream_t stream)
{
	printf("\n=== 2. KDA UpDecayBeta MMA + NormGate + WY C=32 (34 layers/step) ===\n");
	const uint32_t max_rows = 1024u, heads = 4u, rank_qk = heads * G5N_KDA_KEY, fused = 3u * rank_qk + heads;
	uint16_t *fused_dg = (uint16_t *)DevAlloc((uint64_t)max_rows * 2u * G5N_KDA_LOW_RANK * 2u);
	uint16_t *decay_lat = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_KDA_LOW_RANK * 2u);
	uint16_t *gate_lat = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_KDA_LOW_RANK * 2u);
	uint16_t *decay_up_w = (uint16_t *)DevAlloc((uint64_t)rank_qk * G5N_KDA_LOW_RANK * 2u);
	uint16_t *gate_up_w = (uint16_t *)DevAlloc((uint64_t)rank_qk * G5N_KDA_LOW_RANK * 2u);
	uint16_t *fused_qkvb = (uint16_t *)DevAlloc((uint64_t)max_rows * fused * 2u);
	uint16_t *beta_only = (uint16_t *)DevAlloc((uint64_t)max_rows * heads * 2u);
	uint16_t *decay_logit = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *gate_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *gate_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	float *decay_bias = (float *)DevAlloc(rank_qk * 4u);
	float *head_log = (float *)DevAlloc(heads * 4u);
	float *ret_ref = (float *)DevAlloc((uint64_t)max_rows * rank_qk * 4u);
	float *ret_opt = (float *)DevAlloc((uint64_t)max_rows * rank_qk * 4u);
	float *wg_ref = (float *)DevAlloc((uint64_t)max_rows * heads * 4u);
	float *wg_opt = (float *)DevAlloc((uint64_t)max_rows * heads * 4u);

	FillBf16(fused_dg, (uint64_t)max_rows * 2u * G5N_KDA_LOW_RANK, 201u, 0.0f, 0.5f, stream);
	FillBf16(decay_up_w, (uint64_t)rank_qk * G5N_KDA_LOW_RANK, 202u, 0.0f, 0.1f, stream);
	FillBf16(gate_up_w, (uint64_t)rank_qk * G5N_KDA_LOW_RANK, 203u, 0.0f, 0.1f, stream);
	FillBf16(fused_qkvb, (uint64_t)max_rows * fused, 204u, 0.0f, 1.0f, stream);
	FillF32(decay_bias, rank_qk, 205u, -1.0f, 0.5f, stream);
	FillF32(head_log, heads, 206u, 0.0f, 0.3f, stream);

	uint16_t *q_tmp = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *k_tmp = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *v_tmp = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	const uint32_t rows = 1024u;
	LM_LAUNCH((LmSplitQkvbKernel<G5N_THREADS>), rows, G5N_THREADS, 0, (SparkGpuStream)stream,
		fused_qkvb, q_tmp, k_tmp, v_tmp, beta_only, rows, rank_qk, rank_qk, heads, fused);

	auto run_up_ref = [&]()
	{
		LM_LAUNCH((LmSplit2RowsKernel<G5N_THREADS>), rows, G5N_THREADS, 0, (SparkGpuStream)stream, fused_dg, decay_lat, gate_lat, rows, G5N_KDA_LOW_RANK);
		CHECK_OK(LmStreamGemmDense<LmBf16Format>(decay_up_w, LmScaleTensorNone(), decay_lat, decay_logit, (float *)0, rows, G5N_KDA_LOW_RANK, rank_qk, 0u, 0u, g_sms, (SparkGpuStream)stream));
		CHECK_OK(LmStreamGemmDense<LmBf16Format>(gate_up_w, LmScaleTensorNone(), gate_lat, gate_ref, (float *)0, rows, G5N_KDA_LOW_RANK, rank_qk, 0u, 0u, g_sms, (SparkGpuStream)stream));
		LM_LAUNCH_DEPENDENT((Glm5NextKdaDecayBetaKernel), rows, G5N_THREADS, 0, (SparkGpuStream)stream, decay_logit, decay_bias, head_log, beta_only, ret_ref, wg_ref, heads, rows);
	};
	auto run_up_opt = [&]()
	{
		LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaMmaKernel), dim3((rank_qk + 63u) / 64u, (rows + 63u) / 64u), 256u, 0, (SparkGpuStream)stream,
			fused_dg, decay_up_w, gate_up_w, decay_bias, head_log, fused_qkvb, fused, 3u * rank_qk, ret_opt, gate_opt, wg_opt, rank_qk, heads, rows);
	};
	run_up_ref();
	run_up_opt();
	CheckTol("KDA UpDecayBeta write_gate 0-ULP", CompareF32(wg_opt, wg_ref, (uint64_t)rows * heads, stream), 0.0, 0.0, true);
	CheckTol("KDA UpDecayBeta retention", CompareF32(ret_opt, ret_ref, (uint64_t)rows * rank_qk, stream), 2e-2, 2e-2);
	CheckTol("KDA UpDecayBeta gate_bf16", CompareBf16(gate_opt, gate_ref, (uint64_t)rows * rank_qk, stream), 2e-2, 2e-2);
	const float ms_up_ref = TimeMs(run_up_ref, stream), ms_up_opt = TimeMs(run_up_opt, stream);
	printf("  [PERF] KDA UpDecayBeta rows=1024: ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (34 layers: %.1f -> %.1f ms)\n",
		ms_up_ref, ms_up_opt, ms_up_ref / ms_up_opt, ms_up_ref * 34.0f, ms_up_opt * 34.0f);

	float *kda_norm_w = (float *)DevAlloc(G5N_KDA_VALUE * 4u);
	uint16_t *ng_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *ng_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	FillF32(kda_norm_w, G5N_KDA_VALUE, 207u, 1.0f, 0.1f, stream);
	auto run_ng_ref = [&]()
	{
		LM_LAUNCH((LmRmsNormSigmoidGateKernel<G5N_THREADS>), dim3((uint64_t)rows * heads), G5N_THREADS, 0, (SparkGpuStream)stream,
			q_tmp, gate_ref, kda_norm_w, ng_ref, G5N_KDA_VALUE, G5N_EPS);
	};
	auto run_ng_opt = [&]()
	{
		LM_LAUNCH_DEPENDENT((Glm5NextKdaNormGatePrefillKernel), (rows * heads + 7u) / 8u, 256u, 0, (SparkGpuStream)stream,
			q_tmp, gate_ref, kda_norm_w, ng_opt, rows * heads);
	};
	run_ng_ref();
	run_ng_opt();
	CheckTol("KDA NormGatePrefill rows=1024", CompareBf16(ng_opt, ng_ref, (uint64_t)rows * rank_qk, stream), 1e-2, 1e-2);
	const float ms_ng_ref = TimeMs(run_ng_ref, stream), ms_ng_opt = TimeMs(run_ng_opt, stream);
	printf("  [PERF] KDA NormGate rows=1024:    ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (34 layers: %.1f -> %.1f ms)\n",
		ms_ng_ref, ms_ng_opt, ms_ng_ref / ms_ng_opt, ms_ng_ref * 34.0f, ms_ng_opt * 34.0f);

	const uint32_t slot_bytes = rank_qk * G5N_KDA_VALUE * sizeof(float);
	uint8_t *state_init = (uint8_t *)DevAlloc(slot_bytes);
	uint8_t *state_ref = (uint8_t *)DevAlloc(slot_bytes);
	uint8_t *state_opt = (uint8_t *)DevAlloc(slot_bytes);
	uint32_t *state_idx = (uint32_t *)DevAlloc(4u);
	uint32_t *seq_begin = (uint32_t *)DevAlloc(8u);
	uint16_t *out_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *out_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * rank_qk * 2u);
	uint16_t *wy_ws = (uint16_t *)DevAlloc((uint64_t)max_rows * 8u * G5N_HIDDEN * 2u);
	FillF32((float *)state_init, rank_qk * G5N_KDA_VALUE, 208u, 0.0f, 0.1f, stream);
	FillBf16(q_tmp, (uint64_t)max_rows * rank_qk, 209u, 0.0f, 0.5f, stream);
	FillBf16(k_tmp, (uint64_t)max_rows * rank_qk, 210u, 0.0f, 0.5f, stream);
	FillBf16(v_tmp, (uint64_t)max_rows * rank_qk, 211u, 0.0f, 0.5f, stream);
	FillF32(ret_ref, (uint64_t)max_rows * rank_qk, 212u, 0.5f, 0.49f, stream);
	FillF32(wg_ref, (uint64_t)max_rows * heads, 213u, 0.5f, 0.45f, stream);

	const uint32_t wy_cases[3] = {100u, 256u, 1024u};
	for ( uint32_t ci = 0u; ci < 3u; ++ci )
	{
		const uint32_t r = wy_cases[ci];
		const uint32_t h_sb[2] = {0u, r};
		CHECK_CUDA(cudaMemcpyAsync(seq_begin, h_sb, sizeof(h_sb), cudaMemcpyHostToDevice, stream));
		Glm5NextLayerBuffers b;
		memset(&b, 0, sizeof(b));
		b.kda_heads = heads;
		b.kda_state_pool = state_opt;
		b.kda_state_slot_bytes = slot_bytes;
		b.kda_state_index = state_idx;
		b.sequence_row_begin = seq_begin;
		b.q_bf16 = q_tmp;
		b.kv_slot_bf16 = k_tmp;
		b.gate_up_bf16 = v_tmp;
		b.kda_retention = ret_ref;
		b.kda_write_gate = wg_ref;
		b.attention_out_bf16 = out_opt;
		b.expert_out_bf16 = wy_ws;

		CHECK_CUDA(cudaMemcpyAsync(state_ref, state_init, slot_bytes, cudaMemcpyDeviceToDevice, stream));
		CHECK_CUDA(cudaMemcpyAsync(state_opt, state_init, slot_bytes, cudaMemcpyDeviceToDevice, stream));
		CHECK_GPU((LmDeltaRuleChunkLaunch<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float>(state_ref, slot_bytes, state_idx,
			seq_begin, (const uint32_t *)0, q_tmp, k_tmp, v_tmp, ret_ref, wg_ref, out_ref, heads, 1u, 1u, 1u, (const uint32_t *)0, (SparkGpuStream)stream)));
		CHECK_OK(Glm5NextKdaWyLaunch(&b, r, 1u, (SparkGpuStream)stream));
		char label[64];
		snprintf(label, sizeof(label), "KDA WY C=32 output (rows=%u)", r);
		CheckTol(label, CompareBf16(out_opt, out_ref, (uint64_t)r * rank_qk, stream), 2e-2, 2e-2);
		snprintf(label, sizeof(label), "KDA WY C=32 final state (rows=%u)", r);
		CheckTol(label, CompareF32((const float *)state_opt, (const float *)state_ref, rank_qk * G5N_KDA_VALUE, stream), 2e-4, 2e-4);

		auto time_chunk = [&]()
		{
			CHECK_GPU((LmDeltaRuleChunkLaunch<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float>(state_ref, slot_bytes, state_idx,
				seq_begin, (const uint32_t *)0, q_tmp, k_tmp, v_tmp, ret_ref, wg_ref, out_ref, heads, 1u, 1u, 1u, (const uint32_t *)0, (SparkGpuStream)stream)));
		};
		auto time_wy = [&]()
		{
			CHECK_OK(Glm5NextKdaWyLaunch(&b, r, 1u, (SparkGpuStream)stream));
		};
		const float ms_ref = TimeMs(time_chunk, stream), ms_opt = TimeMs(time_wy, stream);
		printf("  [PERF] KDA Recur rows=%4u:        ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (34 layers: %.1f -> %.1f ms)\n",
			r, ms_ref, ms_opt, ms_ref / ms_opt, ms_ref * 34.0f, ms_opt * 34.0f);
	}

	cudaFree(fused_dg); cudaFree(decay_lat); cudaFree(gate_lat); cudaFree(decay_up_w); cudaFree(gate_up_w);
	cudaFree(fused_qkvb); cudaFree(beta_only); cudaFree(decay_logit); cudaFree(gate_ref); cudaFree(gate_opt);
	cudaFree(decay_bias); cudaFree(head_log); cudaFree(ret_ref); cudaFree(ret_opt); cudaFree(wg_ref); cudaFree(wg_opt);
	cudaFree(q_tmp); cudaFree(k_tmp); cudaFree(v_tmp); cudaFree(kda_norm_w); cudaFree(ng_ref); cudaFree(ng_opt);
	cudaFree(state_init); cudaFree(state_ref); cudaFree(state_opt); cudaFree(state_idx); cudaFree(seq_begin);
	cudaFree(out_ref); cudaFree(out_opt); cudaFree(wy_ws);
}

static void BenchDsaPoolScoreAndProj(cudaStream_t stream)
{
	printf("\n=== 3. DSA PoolScore MMA + Fused IndexProj + IndexQ 3-Stage + SmallN MMA ===\n");
	const uint32_t rows = 1024u, degree = 16u, owner_rank = 3u;
	uint16_t *query_bf16 = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_HEADS * G5N_INDEX_DIM * 2u);
	uint16_t *hw_bf16 = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_HEADS * 2u);
	uint32_t *seq_of_row = (uint32_t *)DevAlloc((uint64_t)rows * 4u);
	uint32_t *ctx_len = (uint32_t *)DevAlloc(4u);
	uint32_t *row_pos = (uint32_t *)DevAlloc((uint64_t)rows * 4u);
	FillBf16(query_bf16, (uint64_t)rows * G5N_INDEX_HEADS * G5N_INDEX_DIM, 301u, 0.0f, 0.5f, stream);
	FillBf16(hw_bf16, (uint64_t)rows * G5N_INDEX_HEADS, 302u, 0.1f, 0.3f, stream);

	const uint32_t ctx_cases[2] = {8192u, 32768u};
	for ( uint32_t ci = 0u; ci < 2u; ++ci )
	{
		const uint32_t ctx = ctx_cases[ci];
		const uint32_t pools = SparkGlm5NextIndexCpPools(ctx);
		const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
		float *pool_keys = (float *)DevAlloc((uint64_t)local_stride * G5N_INDEX_DIM * 4u);
		uint32_t *pool_ready = (uint32_t *)DevAlloc((uint64_t)local_stride * 4u);
		float *scores_ref = (float *)DevAlloc((uint64_t)rows * local_stride * 4u);
		float *scores_opt = (float *)DevAlloc((uint64_t)rows * local_stride * 4u);
		FillF32(pool_keys, (uint64_t)local_stride * G5N_INDEX_DIM, 303u + ci, 0.0f, 0.5f, stream);
		std::vector<uint32_t> h_ready(local_stride, 1u), h_pos(rows);
		for ( uint32_t r = 0u; r < rows; ++r )
			h_pos[r] = ctx - rows + r;
		CHECK_CUDA(cudaMemcpyAsync(pool_ready, h_ready.data(), local_stride * 4u, cudaMemcpyHostToDevice, stream));
		CHECK_CUDA(cudaMemcpyAsync(ctx_len, &ctx, 4u, cudaMemcpyHostToDevice, stream));
		CHECK_CUDA(cudaMemcpyAsync(row_pos, h_pos.data(), rows * 4u, cudaMemcpyHostToDevice, stream));

		auto run_ref = [&]()
		{
			LM_LAUNCH((Glm5NextPoolScoreCachedKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, G5N_POOL_BLOCK_POOLS, G5N_POOL_BLOCK_ROWS>),
				dim3((local_stride + G5N_POOL_BLOCK_POOLS - 1u) / G5N_POOL_BLOCK_POOLS, (rows + G5N_POOL_BLOCK_ROWS - 1u) / G5N_POOL_BLOCK_ROWS),
				G5N_THREADS, 0, (SparkGpuStream)stream, query_bf16, hw_bf16, pool_keys, pool_ready, seq_of_row, ctx_len, row_pos,
				pools, owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE,
				scores_ref, rows);
		};
		auto run_opt = [&]()
		{
			CHECK_OK(Glm5NextPoolScoreMmaLaunch(query_bf16, hw_bf16, pool_keys, pool_ready, seq_of_row, ctx_len, row_pos,
				pools, owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE,
				scores_opt, rows, (SparkGpuStream)stream));
		};
		run_ref();
		run_opt();
		char label[64];
		snprintf(label, sizeof(label), "PoolScore MMA (ctx=%u, ls=%u)", ctx, local_stride);
		CheckTol(label, CompareF32(scores_opt, scores_ref, (uint64_t)rows * local_stride, stream), 5e-4, 5e-4);
		const float ms_ref = TimeMs(run_ref, stream), ms_opt = TimeMs(run_opt, stream);
		printf("  [PERF] PoolScore ctx=%5u rows=1024: ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (11 layers: %.1f -> %.1f ms)\n",
			ctx, ms_ref, ms_opt, ms_ref / ms_opt, ms_ref * 11.0f, ms_opt * 11.0f);
		cudaFree(pool_keys); cudaFree(pool_ready); cudaFree(scores_ref); cudaFree(scores_opt);
	}

	uint16_t *normed = (uint16_t *)DevAlloc((uint64_t)rows * G5N_HIDDEN * 2u);
	uint16_t *w_k = (uint16_t *)DevAlloc((uint64_t)G5N_INDEX_DIM * G5N_HIDDEN * 2u);
	uint16_t *w_g = (uint16_t *)DevAlloc((uint64_t)G5N_INDEX_DIM * G5N_HIDDEN * 2u);
	uint16_t *w_h = (uint16_t *)DevAlloc((uint64_t)G5N_INDEX_HEADS * G5N_HIDDEN * 2u);
	uint16_t *w_r = (uint16_t *)DevAlloc((uint64_t)G5N_EXPERTS * G5N_HIDDEN * 2u);
	uint16_t *ok_ref = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_DIM * 2u);
	uint16_t *og_ref = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_DIM * 2u);
	uint16_t *oh_ref = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_HEADS * 2u);
	uint16_t *ok_opt = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_DIM * 2u);
	uint16_t *og_opt = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_DIM * 2u);
	uint16_t *oh_opt = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_HEADS * 2u);
	float *rlog_ref = (float *)DevAlloc((uint64_t)rows * G5N_EXPERTS * 4u);
	float *rlog_opt = (float *)DevAlloc((uint64_t)rows * G5N_EXPERTS * 4u);
	FillBf16(normed, (uint64_t)rows * G5N_HIDDEN, 310u, 0.0f, 0.3f, stream);
	FillBf16(w_k, (uint64_t)G5N_INDEX_DIM * G5N_HIDDEN, 311u, 0.0f, 0.05f, stream);
	FillBf16(w_g, (uint64_t)G5N_INDEX_DIM * G5N_HIDDEN, 312u, 0.0f, 0.05f, stream);
	FillBf16(w_h, (uint64_t)G5N_INDEX_HEADS * G5N_HIDDEN, 313u, 0.0f, 0.05f, stream);
	FillBf16(w_r, (uint64_t)G5N_EXPERTS * G5N_HIDDEN, 314u, 0.0f, 0.05f, stream);

	Glm5NextLayerWeights lw;
	memset(&lw, 0, sizeof(lw));
	lw.index_k = w_k;
	lw.index_compress_gate = w_g;
	lw.index_head = w_h;
	Glm5NextLayerBuffers lb;
	memset(&lb, 0, sizeof(lb));
	lb.w = &lw;
	lb.normed_bf16 = normed;
	lb.index_key_bf16 = ok_opt;
	lb.index_gate_bf16 = og_opt;
	lb.index_head_weight_bf16 = oh_opt;

	auto run_idx_ref = [&]()
	{
		CHECK_OK(LmStreamGemmDense<LmBf16Format>(w_k, LmScaleTensorNone(), normed, ok_ref, (float *)0, rows, G5N_HIDDEN, G5N_INDEX_DIM, 0u, 0u, g_sms, (SparkGpuStream)stream));
		CHECK_OK(LmStreamGemmDense<LmBf16Format>(w_g, LmScaleTensorNone(), normed, og_ref, (float *)0, rows, G5N_HIDDEN, G5N_INDEX_DIM, 0u, 0u, g_sms, (SparkGpuStream)stream));
		LM_LAUNCH_DEPENDENT((Glm5NextIndexHeadSkipKernel), dim3(G5N_INDEX_HEADS, (rows + 7u) / 8u), 32u, 0, (SparkGpuStream)stream,
			w_h, normed, oh_ref, seq_of_row, ctx_len, row_pos, rows, G5N_SELECTED);
	};
	auto run_idx_opt = [&]()
	{
		CHECK_OK(Glm5NextDsaIndexProjMmaLaunch(&lb, rows, 1u, (SparkGpuStream)stream));
	};
	run_idx_ref();
	run_idx_opt();
	CheckTol("DsaIndexProjMma index_k", CompareBf16(ok_opt, ok_ref, (uint64_t)rows * G5N_INDEX_DIM, stream), 2e-2, 2e-2);
	CheckTol("DsaIndexProjMma index_gate", CompareBf16(og_opt, og_ref, (uint64_t)rows * G5N_INDEX_DIM, stream), 2e-2, 2e-2);
	CheckTol("DsaIndexProjMma index_head", CompareBf16(oh_opt, oh_ref, (uint64_t)rows * G5N_INDEX_HEADS, stream), 2e-2, 2e-2);
	const float ms_idx_ref = TimeMs(run_idx_ref, stream), ms_idx_opt = TimeMs(run_idx_opt, stream);
	printf("  [PERF] DsaIndexProj rows=1024:        ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (11 layers: %.1f -> %.1f ms)\n",
		ms_idx_ref, ms_idx_opt, ms_idx_ref / ms_idx_opt, ms_idx_ref * 11.0f, ms_idx_opt * 11.0f);

	auto run_rtr_ref = [&]()
	{
		CHECK_OK(LmStreamGemmDense<LmBf16Format>(w_r, LmScaleTensorNone(), normed, (uint16_t *)0, rlog_ref, rows, G5N_HIDDEN, G5N_EXPERTS, G5N_EXPERTS, 0u, g_sms, (SparkGpuStream)stream));
	};
	auto run_rtr_opt = [&]()
	{
		CHECK_OK(Glm5NextSmallNMmaLaunch(normed, w_r, (uint16_t *)0, rlog_opt, rows, G5N_HIDDEN, G5N_EXPERTS, (SparkGpuStream)stream));
	};
	run_rtr_ref();
	run_rtr_opt();
	CheckTol("SmallNMma router N=288 f32", CompareF32(rlog_opt, rlog_ref, (uint64_t)rows * G5N_EXPERTS, stream), 5e-4, 5e-4);
	const float ms_rtr_ref = TimeMs(run_rtr_ref, stream), ms_rtr_opt = TimeMs(run_rtr_opt, stream);
	printf("  [PERF] SmallNMma router rows=1024:    ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (42 layers: %.1f -> %.1f ms)\n",
		ms_rtr_ref, ms_rtr_opt, ms_rtr_ref / ms_rtr_opt, ms_rtr_ref * 42.0f, ms_rtr_opt * 42.0f);

	uint16_t *q_comp = (uint16_t *)DevAlloc((uint64_t)rows * G5N_QA * 2u);
	uint16_t *w_iq = (uint16_t *)DevAlloc((uint64_t)G5N_INDEX_QUERY * G5N_QA * 2u);
	uint16_t *iq_ref = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_QUERY * 2u);
	uint16_t *iq_opt = (uint16_t *)DevAlloc((uint64_t)rows * G5N_INDEX_QUERY * 2u);
	FillBf16(q_comp, (uint64_t)rows * G5N_QA, 315u, 0.0f, 0.3f, stream);
	FillBf16(w_iq, (uint64_t)G5N_INDEX_QUERY * G5N_QA, 316u, 0.0f, 0.05f, stream);
	constexpr uint32_t sh_iq_old = 2u * (32u * 128u + 64u * 128u);
	constexpr uint32_t sh_iq_new = 3u * (64u * 128u + 128u * 128u);
	CHECK_GPU(SparkGpuFuncSetAttribute((const void *)Glm5NextIndexQMmaSkipKernel<64u, 4u, 128u, 3u>,
		SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh_iq_new));
	auto run_iq_ref = [&]()
	{
		LM_LAUNCH_DEPENDENT((Glm5NextIndexQMmaSkipKernel<32u, 2u, 64u, 2u>), dim3(G5N_INDEX_QUERY / 64u, (rows + 31u) / 32u), 256u, sh_iq_old, (SparkGpuStream)stream,
			w_iq, q_comp, iq_ref, seq_of_row, ctx_len, row_pos, rows, G5N_SELECTED);
	};
	auto run_iq_opt = [&]()
	{
		LM_LAUNCH_DEPENDENT((Glm5NextIndexQMmaSkipKernel<64u, 4u, 128u, 3u>), dim3(G5N_INDEX_QUERY / 128u, (rows + 63u) / 64u), 256u, sh_iq_new, (SparkGpuStream)stream,
			w_iq, q_comp, iq_opt, seq_of_row, ctx_len, row_pos, rows, G5N_SELECTED);
	};
	run_iq_ref();
	run_iq_opt();
	CheckTol("IndexQMma 64x128x3 vs 32x64x2", CompareBf16(iq_opt, iq_ref, (uint64_t)rows * G5N_INDEX_QUERY, stream), 2e-2, 2e-2);
	const float ms_iq_ref = TimeMs(run_iq_ref, stream), ms_iq_opt = TimeMs(run_iq_opt, stream);
	printf("  [PERF] IndexQ MMA rows=1024:          ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (11 layers: %.1f -> %.1f ms)\n",
		ms_iq_ref, ms_iq_opt, ms_iq_ref / ms_iq_opt, ms_iq_ref * 11.0f, ms_iq_opt * 11.0f);
	cudaFree(q_comp); cudaFree(w_iq); cudaFree(iq_ref); cudaFree(iq_opt);

	cudaFree(query_bf16); cudaFree(hw_bf16); cudaFree(seq_of_row); cudaFree(ctx_len); cudaFree(row_pos);
	cudaFree(normed); cudaFree(w_k); cudaFree(w_g); cudaFree(w_h); cudaFree(w_r);
	cudaFree(ok_ref); cudaFree(og_ref); cudaFree(oh_ref); cudaFree(ok_opt); cudaFree(og_opt); cudaFree(oh_opt);
	cudaFree(rlog_ref); cudaFree(rlog_opt);
}

static void BuildSyntheticRouting(uint32_t rows, bool skewed, uint32_t *d_group_off, uint32_t *d_src_tok,
	uint32_t *d_packed_row, uint32_t *d_route_exp, float *d_route_w, cudaStream_t stream)
{
	const uint32_t packed = rows * G5N_TOP_K;
	std::vector<uint32_t> h_exp(packed), h_off(G5N_EXPERTS + 1u, 0u), h_src(packed), h_pr(packed), h_cur(G5N_EXPERTS, 0u);
	std::vector<float> h_w(packed);
	for ( uint32_t t = 0u; t < rows; ++t )
	{
		uint32_t used[G5N_TOP_K];
		for ( uint32_t k = 0u; k < G5N_TOP_K; ++k )
		{
			uint32_t e = 0u;
			for ( uint32_t tries = 0u; tries < 64u; ++tries )
			{
				const uint64_t z = Mix64((uint64_t)t * 97u + k * 13u + tries * 7u + (skewed ? 999u : 111u));
				if ( skewed && (z % 100u) < 40u )
					e = (uint32_t)(z % 32u);
				else
					e = (uint32_t)(z % G5N_EXPERTS);
				bool dup = false;
				for ( uint32_t p = 0u; p < k; ++p )
					if ( used[p] == e ) dup = true;
				if ( !dup ) break;
			}
			used[k] = e;
			h_exp[t * G5N_TOP_K + k] = e;
			h_w[t * G5N_TOP_K + k] = 0.25f + 0.05f * (float)k;
			h_off[e + 1u]++;
		}
	}
	for ( uint32_t e = 0u; e < G5N_EXPERTS; ++e )
	{
		h_off[e + 1u] += h_off[e];
		h_cur[e] = h_off[e];
	}
	for ( uint32_t t = 0u; t < rows; ++t )
		for ( uint32_t k = 0u; k < G5N_TOP_K; ++k )
		{
			const uint32_t e = h_exp[t * G5N_TOP_K + k];
			const uint32_t p = h_cur[e]++;
			h_src[p] = t;
			h_pr[t * G5N_TOP_K + k] = p;
		}
	CHECK_CUDA(cudaMemcpyAsync(d_group_off, h_off.data(), (G5N_EXPERTS + 1u) * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_src_tok, h_src.data(), packed * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_packed_row, h_pr.data(), packed * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_route_exp, h_exp.data(), packed * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_route_w, h_w.data(), packed * 4u, cudaMemcpyHostToDevice, stream));
}

template<uint32_t ExpertCodec>
static void BenchMoeCodec(const char *tag, cudaStream_t stream)
{
	printf("\n=== 4. MoE %s (TILE_M=64 W1 + TILE_M=128 W2 + Fused Shared, 42 layers/step) ===\n", tag);
	using Format = typename LmWeightCodec<ExpertCodec>::Format;
	const bool is_fp8 = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
	const uint32_t max_rows = 1024u, max_packed = max_rows * G5N_TOP_K, inter = 128u;
	const uint64_t w1_elems = (uint64_t)G5N_EXPERTS * (2u * inter) * G5N_HIDDEN;
	const uint64_t w2_elems = (uint64_t)G5N_EXPERTS * G5N_HIDDEN * inter;
	void *exp_w1 = DevAlloc(w1_elems * (is_fp8 ? 1u : 2u));
	void *exp_w2 = DevAlloc(w2_elems * (is_fp8 ? 1u : 2u));
	float *exp_s1 = is_fp8 ? (float *)DevAlloc((uint64_t)G5N_EXPERTS * (2u * inter) * (G5N_HIDDEN / 128u) * 4u) : 0;
	float *exp_s2 = is_fp8 ? (float *)DevAlloc((uint64_t)G5N_EXPERTS * G5N_HIDDEN * 4u) : 0;
	uint16_t *shared_w1 = (uint16_t *)DevAlloc((uint64_t)(2u * inter) * G5N_HIDDEN * 2u);
	uint16_t *shared_w2 = (uint16_t *)DevAlloc((uint64_t)G5N_HIDDEN * inter * 2u);
	uint16_t *normed = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint16_t *gate_up = (uint16_t *)DevAlloc((uint64_t)max_packed * (2u * inter) * 2u);
	uint16_t *inter_ref = (uint16_t *)DevAlloc((uint64_t)max_packed * inter * 2u);
	uint16_t *inter_opt = (uint16_t *)DevAlloc((uint64_t)max_packed * inter * 2u);
	uint16_t *eout_ref = (uint16_t *)DevAlloc((uint64_t)max_packed * G5N_HIDDEN * 2u);
	uint16_t *eout_opt = (uint16_t *)DevAlloc((uint64_t)max_packed * G5N_HIDDEN * 2u);
	uint16_t *sout_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint16_t *out_ref = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint16_t *out_opt = (uint16_t *)DevAlloc((uint64_t)max_rows * G5N_HIDDEN * 2u);
	uint32_t *group_off = (uint32_t *)DevAlloc((G5N_EXPERTS + 1u) * 4u);
	uint32_t *tp_w1 = (uint32_t *)DevAlloc((G5N_EXPERTS + 1u) * 4u);
	uint32_t *tp_w2 = (uint32_t *)DevAlloc((G5N_EXPERTS + 1u) * 4u);
	uint32_t *src_tok = (uint32_t *)DevAlloc(max_packed * 4u);
	uint32_t *packed_row = (uint32_t *)DevAlloc(max_packed * 4u);
	uint32_t *route_exp = (uint32_t *)DevAlloc(max_packed * 4u);
	float *route_w = (float *)DevAlloc(max_packed * 4u);

	if ( is_fp8 )
	{
		FillFp8((uint8_t *)exp_w1, w1_elems, 401u, stream);
		FillFp8((uint8_t *)exp_w2, w2_elems, 402u, stream);
		FillF32(exp_s1, (uint64_t)G5N_EXPERTS * (2u * inter) * (G5N_HIDDEN / 128u), 403u, 0.05f, 0.02f, stream);
		FillF32(exp_s2, (uint64_t)G5N_EXPERTS * G5N_HIDDEN, 404u, 0.05f, 0.02f, stream);
	}
	else
	{
		FillBf16((uint16_t *)exp_w1, w1_elems, 401u, 0.0f, 0.05f, stream);
		FillBf16((uint16_t *)exp_w2, w2_elems, 402u, 0.0f, 0.05f, stream);
	}
	FillBf16(shared_w1, (uint64_t)(2u * inter) * G5N_HIDDEN, 405u, 0.0f, 0.05f, stream);
	FillBf16(shared_w2, (uint64_t)G5N_HIDDEN * inter, 406u, 0.0f, 0.05f, stream);
	FillBf16(normed, (uint64_t)max_rows * G5N_HIDDEN, 407u, 0.0f, 0.3f, stream);

	Glm5NextLayerWeights lw;
	memset(&lw, 0, sizeof(lw));
	lw.shared_gate_up = shared_w1;
	lw.shared_down = shared_w2;

	for ( uint32_t skew = 0u; skew < 2u; ++skew )
	{
		const uint32_t rows = 1024u, packed = rows * G5N_TOP_K;
		BuildSyntheticRouting(rows, skew != 0u, group_off, src_tok, packed_row, route_exp, route_w, stream);

		Glm5NextLayerBuffers b;
		memset(&b, 0, sizeof(b));
		b.w = &lw;
		b.prefill_unified = 1u;
		b.expert_w1_rows = 2u * inter;
		b.expert_intermediate = inter;
		b.shared_gate_up_rows = 2u * inter;
		b.shared_intermediate = inter;
		b.expert_w1_weight = exp_w1;
		b.expert_w1_scale = exp_s1;
		b.expert_w2_weight = exp_w2;
		b.expert_w2_scale = exp_s2;
		b.normed_bf16 = normed;
		b.gate_up_bf16 = gate_up;
		b.intermediate_bf16 = inter_opt;
		b.expert_out_bf16 = eout_opt;
		b.shared_out_bf16 = sout_ref;
		b.attention_out_bf16 = out_opt;
		b.group_row_offset = group_off;
		b.group_tile_prefix_w1 = tp_w1;
		b.group_tile_prefix_w2 = tp_w2;
		b.route_source_token = src_tok;
		b.route_packed_row = packed_row;
		b.route_expert = route_exp;
		b.route_weight = route_w;

		auto run_ref = [&]()
		{
			LmScaleTensor s1 = LmWeightCodecScaleTensor<ExpertCodec>(exp_s1, G5N_EXPERTS, 2u * inter, G5N_HIDDEN);
			LmScaleTensor s2 = LmWeightCodecScaleTensor<ExpertCodec>(exp_s2, G5N_EXPERTS, G5N_HIDDEN, inter);
			CHECK_OK(LmStreamGemmGrouped<Format>(exp_w1, s1, normed, gate_up, group_off, src_tok, G5N_EXPERTS, packed, G5N_HIDDEN, 2u * inter, g_sms, (SparkGpuStream)stream));
			LM_LAUNCH_DEPENDENT((Glm5NextClampedUpGatePacked128Kernel), (packed + 15u) / 16u, 256u, 0, (SparkGpuStream)stream, gate_up, inter_ref, packed);
			CHECK_OK(LmStreamGemmGrouped<Format>(exp_w2, s2, inter_ref, eout_ref, group_off, 0, G5N_EXPERTS, packed, inter, G5N_HIDDEN, g_sms, (SparkGpuStream)stream));
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(shared_w1, LmScaleTensorNone(), normed, gate_up, (float *)0, rows, G5N_HIDDEN, 2u * inter, 0u, 0u, g_sms, (SparkGpuStream)stream));
			LM_LAUNCH_DEPENDENT((Glm5NextClampedUpGatePacked128Kernel), (rows + 15u) / 16u, 256u, 0, (SparkGpuStream)stream, gate_up, inter_ref, rows);
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(shared_w2, LmScaleTensorNone(), inter_ref, sout_ref, (float *)0, rows, inter, G5N_HIDDEN, 0u, 0u, g_sms, (SparkGpuStream)stream));
			LM_LAUNCH_DEPENDENT((Glm5NextMoeFinalizeAddKernel), dim3((G5N_HIDDEN / 8u + 255u) / 256u, rows), 256u, 0, (SparkGpuStream)stream,
				eout_ref, packed_row, route_w, sout_ref, out_ref, rows);
		};
		auto run_opt = [&]()
		{
			CHECK_OK(Glm5NextMoeExperts<ExpertCodec>(&b, rows, g_sms, (SparkGpuStream)stream));
		};
		run_ref();
		run_opt();
		char label[64];
		snprintf(label, sizeof(label), "MoE %s %s rows=1024", tag, skew != 0u ? "skewed" : "uniform");
		CheckTol(label, CompareBf16(out_opt, out_ref, (uint64_t)rows * G5N_HIDDEN, stream), 3e-2, 3e-2);
		const float ms_ref = TimeMs(run_ref, stream), ms_opt = TimeMs(run_opt, stream);
		printf("  [PERF] MoE %s %-7s rows=1024: ref=%.3f ms  opt=%.3f ms  speedup=%.2fx  (42 layers: %.1f -> %.1f ms)\n",
			tag, skew != 0u ? "skewed" : "uniform", ms_ref, ms_opt, ms_ref / ms_opt, ms_ref * 42.0f, ms_opt * 42.0f);
	}

	cudaFree(exp_w1); cudaFree(exp_w2); if ( exp_s1 ) cudaFree(exp_s1); if ( exp_s2 ) cudaFree(exp_s2);
	cudaFree(shared_w1); cudaFree(shared_w2); cudaFree(normed); cudaFree(gate_up);
	cudaFree(inter_ref); cudaFree(inter_opt); cudaFree(eout_ref); cudaFree(eout_opt);
	cudaFree(sout_ref); cudaFree(out_ref); cudaFree(out_opt);
	cudaFree(group_off); cudaFree(tp_w1); cudaFree(tp_w2); cudaFree(src_tok);
	cudaFree(packed_row); cudaFree(route_exp); cudaFree(route_w);
}

static void Check5cBatchInvariance(cudaStream_t stream)
{
	printf("\n=== 5. 5c Batch Invariance (B1 == B4 == B8, 0-ULP, row_invariant=1) ===\n");
	const uint32_t max_b = 8u;
	uint16_t *act = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HIDDEN * 2u);
	uint16_t *w_lin = (uint16_t *)DevAlloc((uint64_t)1536u * G5N_HIDDEN * 2u);
	uint16_t *o_b1 = (uint16_t *)DevAlloc((uint64_t)max_b * 1536u * 2u);
	uint16_t *o_b4 = (uint16_t *)DevAlloc((uint64_t)max_b * 1536u * 2u);
	uint16_t *o_b8 = (uint16_t *)DevAlloc((uint64_t)max_b * 1536u * 2u);
	FillBf16(act, (uint64_t)max_b * G5N_HIDDEN, 501u, 0.0f, 0.3f, stream);
	FillBf16(w_lin, (uint64_t)1536u * G5N_HIDDEN, 502u, 0.0f, 0.05f, stream);

	Glm5NextLayerBuffers b;
	memset(&b, 0, sizeof(b));
	b.row_invariant = 1u;
	b.prefill_unified = 0u;
	for ( uint32_t r = 0u; r < max_b; ++r )
		CHECK_OK(Glm5NextLinear(&b, act + (uint64_t)r * G5N_HIDDEN, w_lin, o_b1 + (uint64_t)r * 1536u, 1u, G5N_HIDDEN, 1536u, g_sms, (SparkGpuStream)stream));
	CHECK_OK(Glm5NextLinear(&b, act, w_lin, o_b4, 4u, G5N_HIDDEN, 1536u, g_sms, (SparkGpuStream)stream));
	CHECK_OK(Glm5NextLinear(&b, act, w_lin, o_b8, 8u, G5N_HIDDEN, 1536u, g_sms, (SparkGpuStream)stream));
	CheckTol("5c Glm5NextLinear B1==B4 0-ULP", CompareBf16(o_b4, o_b1, 4u * 1536u, stream), 0.0, 0.0, true);
	CheckTol("5c Glm5NextLinear B1==B8 0-ULP", CompareBf16(o_b8, o_b1, 8u * 1536u, stream), 0.0, 0.0, true);

	uint16_t *s_in = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HC_FLAT * 2u);
	uint16_t *p_out = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HIDDEN * 2u);
	uint16_t *s_b1 = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HC_FLAT * 2u);
	uint16_t *s_b8 = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HC_FLAT * 2u);
	uint16_t *nw = (uint16_t *)DevAlloc(G5N_HIDDEN * 2u);
	uint16_t *n_b1 = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HIDDEN * 2u);
	uint16_t *n_b8 = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HIDDEN * 2u);
	float *fn = (float *)DevAlloc((uint64_t)G5N_HC_MIX * G5N_HC_FLAT * 4u);
	float *base = (float *)DevAlloc(G5N_HC_MIX * 4u);
	float *sc = (float *)DevAlloc(3u * 4u);
	float *pi = (float *)DevAlloc((uint64_t)max_b * G5N_HC * 4u);
	float *ci = (float *)DevAlloc((uint64_t)max_b * 16u * 4u);
	float *p1 = (float *)DevAlloc((uint64_t)max_b * G5N_HC * 4u);
	float *c1 = (float *)DevAlloc((uint64_t)max_b * 16u * 4u);
	float *p8 = (float *)DevAlloc((uint64_t)max_b * G5N_HC * 4u);
	float *c8 = (float *)DevAlloc((uint64_t)max_b * 16u * 4u);
	uint16_t *hc_col = (uint16_t *)DevAlloc((uint64_t)max_b * G5N_HIDDEN * 2u);
	FillBf16(s_in, (uint64_t)max_b * G5N_HC_FLAT, 503u, 0.0f, 0.5f, stream);
	FillBf16(p_out, (uint64_t)max_b * G5N_HIDDEN, 504u, 0.0f, 0.5f, stream);
	FillBf16(nw, G5N_HIDDEN, 505u, 1.0f, 0.1f, stream);
	FillF32(fn, (uint64_t)G5N_HC_MIX * G5N_HC_FLAT, 506u, 0.0f, 0.02f, stream);
	FillF32(base, G5N_HC_MIX, 507u, 0.0f, 0.2f, stream);
	FillF32(sc, 3u, 508u, 0.25f, 0.05f, stream);
	FillF32(pi, (uint64_t)max_b * G5N_HC, 509u, 1.0f, 0.2f, stream);
	FillF32(ci, (uint64_t)max_b * 16u, 510u, 0.25f, 0.05f, stream);
	CHECK_CUDA(cudaMemcpyAsync(p1, pi, (uint64_t)max_b * G5N_HC * 4u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(c1, ci, (uint64_t)max_b * 16u * 4u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(p8, pi, (uint64_t)max_b * G5N_HC * 4u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(c8, ci, (uint64_t)max_b * 16u * 4u, cudaMemcpyDeviceToDevice, stream));

	for ( uint32_t r = 0u; r < max_b; ++r )
	{
		Glm5NextLayerBuffers b1;
		memset(&b1, 0, sizeof(b1));
		b1.row_invariant = 1u;
		b1.hc_collapsed_bf16 = hc_col;
		b1.hc_post_f32 = p1 + (uint64_t)r * G5N_HC;
		b1.hc_comb_f32 = c1 + (uint64_t)r * 16u;
		b1.normed_bf16 = n_b1 + (uint64_t)r * G5N_HIDDEN;
		CHECK_OK(Glm5NextHcPostNorm(&b1, s_in + (uint64_t)r * G5N_HC_FLAT, p_out + (uint64_t)r * G5N_HIDDEN,
			s_b1 + (uint64_t)r * G5N_HC_FLAT, fn, base, sc, nw, 1u, (SparkGpuStream)stream));
	}
	Glm5NextLayerBuffers b8;
	memset(&b8, 0, sizeof(b8));
	b8.row_invariant = 1u;
	b8.hc_collapsed_bf16 = hc_col;
	b8.hc_post_f32 = p8;
	b8.hc_comb_f32 = c8;
	b8.normed_bf16 = n_b8;
	CHECK_OK(Glm5NextHcPostNorm(&b8, s_in, p_out, s_b8, fn, base, sc, nw, max_b, (SparkGpuStream)stream));
	CheckTol("5c HcPostNorm streams B1==B8 0-ULP", CompareBf16(s_b8, s_b1, (uint64_t)max_b * G5N_HC_FLAT, stream), 0.0, 0.0, true);
	CheckTol("5c HcPostNorm normed B1==B8 0-ULP", CompareBf16(n_b8, n_b1, (uint64_t)max_b * G5N_HIDDEN, stream), 0.0, 0.0, true);

	cudaFree(act); cudaFree(w_lin); cudaFree(o_b1); cudaFree(o_b4); cudaFree(o_b8);
	cudaFree(s_in); cudaFree(p_out); cudaFree(s_b1); cudaFree(s_b8); cudaFree(nw);
	cudaFree(n_b1); cudaFree(n_b8); cudaFree(fn); cudaFree(base); cudaFree(sc);
	cudaFree(pi); cudaFree(ci); cudaFree(p1); cudaFree(c1); cudaFree(p8); cudaFree(c8); cudaFree(hc_col);
}

int main(void)
{
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
	g_sms = (uint32_t)prop.multiProcessorCount;
	printf("glm5next_p4_blackwell_bench on %s (sm_%d%d, %u SMs)\n", prop.name, prop.major, prop.minor, g_sms);
	cudaStream_t stream = 0;
	CHECK_CUDA(cudaStreamCreate(&stream));

	BenchHcPostDotMma(stream);
	BenchKdaFusedAndWy(stream);
	BenchDsaPoolScoreAndProj(stream);
	BenchMoeCodec<SPARK_WEIGHT_CODEC_FP8_E4M3>("FP8", stream);
	BenchMoeCodec<SPARK_WEIGHT_CODEC_BF16>("BF16", stream);
	Check5cBatchInvariance(stream);

	CHECK_CUDA(cudaStreamDestroy(stream));
	if ( g_failures != 0 )
	{
		printf("\nFAIL glm5next_p4_blackwell_bench (%d failures)\n", g_failures);
		return 1;
	}
	printf("\nPASS glm5next_p4_blackwell_bench\n");
	return 0;
}
