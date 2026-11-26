#include "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu"
#include "modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_model.cu"
#include "modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

static constexpr uint32_t kWarmup = 6u;
static constexpr uint32_t kIters = 24u;

static int g_bench_fail = 0;

#define BENCH_CHECK(expr) \
	do { \
		cudaError_t _e = (expr); \
		if ( _e != cudaSuccess ) { \
			fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(_e)); \
			exit(1); \
		} \
	} while ( 0 )

template<class Fn>
static float TimeMs(cudaStream_t stream, Fn fn)
{
	for ( uint32_t i = 0u; i < kWarmup; ++i )
		fn();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	cudaEvent_t ev0, ev1;
	BENCH_CHECK(cudaEventCreate(&ev0));
	BENCH_CHECK(cudaEventCreate(&ev1));
	BENCH_CHECK(cudaEventRecord(ev0, stream));
	for ( uint32_t i = 0u; i < kIters; ++i )
		fn();
	BENCH_CHECK(cudaEventRecord(ev1, stream));
	BENCH_CHECK(cudaEventSynchronize(ev1));
	float ms = 0.0f;
	BENCH_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
	BENCH_CHECK(cudaEventDestroy(ev0));
	BENCH_CHECK(cudaEventDestroy(ev1));
	return ms / (float)kIters;
}

static void CheckBf16(const char *label, const uint16_t *d_ref, const uint16_t *d_got, uint64_t n, uint32_t max_ulp_allowed, float max_abs_allowed)
{
	std::vector<uint16_t> h_ref(n), h_got(n);
	BENCH_CHECK(cudaMemcpy(h_ref.data(), d_ref, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	BENCH_CHECK(cudaMemcpy(h_got.data(), d_got, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	uint32_t max_ulp = 0u;
	uint64_t mismatches = 0u;
	float max_abs = 0.0f;
	for ( uint64_t i = 0u; i < n; ++i )
	{
		if ( h_ref[i] != h_got[i] )
		{
			mismatches++;
			const float a = VfFloat(h_ref[i]), b = VfFloat(h_got[i]);
			const float ad = fabsf(a - b);
			if ( ad > max_abs )
				max_abs = ad;
			const uint32_t ua = h_ref[i], ub = h_got[i];
			const uint32_t d = ua > ub ? (ua - ub) : (ub - ua);
			if ( d > max_ulp )
				max_ulp = d;
		}
	}
	const bool ok = (max_ulp <= max_ulp_allowed) || (max_abs_allowed > 0.0f && max_abs <= max_abs_allowed);
	if ( !ok )
	{
		printf("  FAIL %-44s mismatches=%llu/%llu max_ulp=%u max_abs=%.6f\n",
			label, (unsigned long long)mismatches, (unsigned long long)n, max_ulp, max_abs);
		g_bench_fail++;
	}
	else
		printf("  PASS %-44s mismatches=%llu max_ulp=%u max_abs=%.6f\n",
			label, (unsigned long long)mismatches, max_ulp, max_abs);
}

static void CheckF32(const char *label, const float *d_ref, const float *d_got, uint64_t n, float tol)
{
	std::vector<float> h_ref(n), h_got(n);
	BENCH_CHECK(cudaMemcpy(h_ref.data(), d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
	BENCH_CHECK(cudaMemcpy(h_got.data(), d_got, n * sizeof(float), cudaMemcpyDeviceToHost));
	uint64_t mismatches = 0u;
	float max_abs = 0.0f;
	for ( uint64_t i = 0u; i < n; ++i )
	{
		const float ad = fabsf(h_ref[i] - h_got[i]);
		if ( ad > tol )
			mismatches++;
		if ( ad > max_abs )
			max_abs = ad;
	}
	if ( mismatches != 0u )
	{
		printf("  FAIL %-44s mismatches=%llu/%llu max_abs=%.7f\n",
			label, (unsigned long long)mismatches, (unsigned long long)n, max_abs);
		g_bench_fail++;
	}
	else
		printf("  PASS %-44s mismatches=0 max_abs=%.7f\n", label, max_abs);
}

template<uint32_t HD, uint32_t NEW_QW>
static void BenchGemmaPrefillAttn(const char *tag, uint32_t rank_heads, uint32_t window, uint32_t rows, uint32_t total_ctx,
	uint32_t kv_chunk, uint32_t sms, cudaStream_t stream)
{
	constexpr uint32_t KV_TILE = GemmaLocalPrefillKvTile<HD>();
	constexpr uint32_t V_WARPS = HD <= 128u ? 1u : (HD / 128u);
	constexpr uint32_t BASE_QW = GemmaLocalPrefillQWarps<HD, 0u>();
	constexpr uint32_t ACT_NEW_QW = GemmaLocalPrefillQWarps<HD, NEW_QW>();
	constexpr uint32_t BASE_THREADS = BASE_QW * V_WARPS * 32u;
	constexpr uint32_t NEW_THREADS = ACT_NEW_QW * V_WARPS * 32u;
	constexpr uint32_t BASE_QT = BASE_QW * 16u;
	constexpr uint32_t NEW_QT = ACT_NEW_QW * 16u;
	constexpr uint32_t BASE_SMEM = GemmaLocalPrefillSharedBytes<HD, 0u>();
	constexpr uint32_t NEW_SMEM = GemmaLocalPrefillSharedBytes<HD, NEW_QW>();

	const uint32_t old_ctx = total_ctx - rows;
	const uint32_t raw_win = (window != 0u && old_ctx + 1u > window) ? (old_ctx + 1u - window) : 0u;
	const uint32_t win_begin = raw_win & ~(KV_TILE - 1u);
	const uint32_t prior_keys = old_ctx > win_begin ? (old_ctx - win_begin) : 0u;
	const uint32_t c_base = win_begin / kv_chunk;
	const uint32_t c_end = (total_ctx - 1u) / kv_chunk + 1u;
	const uint32_t num_chunks = c_end - c_base;

	GemmaPrefillSeqPlan plan;
	memset(&plan, 0, sizeof(plan));
	plan.count = 1u;
	plan.row_begin[0] = 0u;
	plan.row_len[0] = rows;
	plan.old_ctx[0] = old_ctx;
	plan.win_begin[0] = win_begin;
	plan.prior_off[0] = 0u;
	plan.prior_keys[0] = prior_keys;
	plan.total_prior = prior_keys;

	const uint16_t *d_q = Tensor<uint16_t>((uint64_t)rows * rank_heads * HD, 0.5f, 0.0f);
	const uint16_t *d_kv_prior = Tensor<uint16_t>((uint64_t)(prior_keys > 0u ? prior_keys : 1u) * 2u * HD, 0.5f, 0.0f);
	const uint16_t *d_kv_cur = Tensor<uint16_t>((uint64_t)rows * 2u * HD, 0.5f, 0.0f);
	std::vector<uint32_t> h_pos(rows);
	for ( uint32_t r = 0u; r < rows; ++r )
		h_pos[r] = old_ctx + r;
	uint32_t *d_pos = 0;
	uint16_t *d_out_base = 0, *d_out_new = 0;
	float *d_partials = 0;
	BENCH_CHECK(cudaMalloc(&d_pos, rows * sizeof(uint32_t)));
	BENCH_CHECK(cudaMemcpy(d_pos, h_pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMalloc(&d_out_base, (uint64_t)rows * rank_heads * HD * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_out_new, (uint64_t)rows * rank_heads * HD * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_partials, (uint64_t)num_chunks * rows * rank_heads * (HD + 2u) * sizeof(float)));

	if constexpr ( BASE_SMEM > 48000u )
		BENCH_CHECK(cudaFuncSetAttribute(GemmaLocalPrefillAttnKernel<HD, 0u, 0u, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, BASE_SMEM));
	if constexpr ( NEW_SMEM > 48000u )
		BENCH_CHECK(cudaFuncSetAttribute(GemmaLocalPrefillAttnKernel<HD, NEW_QW, 0u, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, NEW_SMEM));

	const float qk_scale = 1.0f / sqrtf((float)HD);
	auto run_base = [&]()
	{
		GemmaLocalPrefillAttnKernel<HD, 0u, 0u, false><<<dim3(rank_heads, (rows + BASE_QT - 1u) / BASE_QT, 1u), BASE_THREADS, BASE_SMEM, stream>>>(
			d_q, d_kv_prior, d_kv_cur, d_pos, plan, rows, rank_heads, 0u, rank_heads, 0u, window, qk_scale,
			(const uint16_t *)0, 0u, d_out_base, 0u, 1u, (float *)0);
	};
	auto run_new = [&]()
	{
		GemmaLocalPrefillAttnKernel<HD, NEW_QW, 0u, true><<<dim3(rank_heads, (rows + NEW_QT - 1u) / NEW_QT, num_chunks), NEW_THREADS, NEW_SMEM, stream>>>(
			d_q, d_kv_prior, d_kv_cur, d_pos, plan, rows, rank_heads, 0u, rank_heads, 0u, window, qk_scale,
			(const uint16_t *)0, 0u, (uint16_t *)0, kv_chunk / KV_TILE, num_chunks, d_partials);
		GemmaLocalPrefillMergeKernel<HD, 0u><<<dim3(rank_heads, rows, 1u), HD / 2u, 0, stream>>>(
			d_partials, d_pos, plan, (const uint16_t *)0, rows, rank_heads, window, kv_chunk, num_chunks, qk_scale, 0u, d_out_new);
	};
	run_base();
	run_new();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	const float ms_base = TimeMs(stream, run_base);
	const float ms_new = TimeMs(stream, run_new);
	printf("[gemma4 attn] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx (chunks=%u)\n",
		tag, ms_base, ms_new, ms_base / ms_new, num_chunks);
	CheckBf16(tag, d_out_base, d_out_new, (uint64_t)rows * rank_heads * HD, 1u, 1e-2f);

	if ( rows >= 256u )
	{
		const uint32_t warm_rows = 128u;
		GemmaPrefillSeqPlan warm_plan = plan;
		warm_plan.row_len[0] = warm_rows;
		const uint32_t wc_end = (old_ctx + warm_rows - 1u) / kv_chunk + 1u;
		const uint32_t w_chunks = wc_end - c_base;
		uint16_t *d_out_warm = 0;
		BENCH_CHECK(cudaMalloc(&d_out_warm, (uint64_t)warm_rows * rank_heads * HD * sizeof(uint16_t)));
		GemmaLocalPrefillAttnKernel<HD, NEW_QW, 0u, true><<<dim3(rank_heads, (warm_rows + NEW_QT - 1u) / NEW_QT, w_chunks), NEW_THREADS, NEW_SMEM, stream>>>(
			d_q, d_kv_prior, d_kv_cur, d_pos, warm_plan, warm_rows, rank_heads, 0u, rank_heads, 0u, window, qk_scale,
			(const uint16_t *)0, 0u, (uint16_t *)0, kv_chunk / KV_TILE, w_chunks, d_partials);
		GemmaLocalPrefillMergeKernel<HD, 0u><<<dim3(rank_heads, warm_rows, 1u), HD / 2u, 0, stream>>>(
			d_partials, d_pos, warm_plan, (const uint16_t *)0, warm_rows, rank_heads, window, kv_chunk, w_chunks, qk_scale, 0u, d_out_warm);
		BENCH_CHECK(cudaStreamSynchronize(stream));
		char buf[96];
		snprintf(buf, sizeof(buf), "%s 4c-warm==cold 0-ULP", tag);
		CheckBf16(buf, d_out_new, d_out_warm, (uint64_t)warm_rows * rank_heads * HD, 0u, 0.0f);
		BENCH_CHECK(cudaFree(d_out_warm));
	}
	(void)sms;
	BENCH_CHECK(cudaFree((void *)d_q));
	BENCH_CHECK(cudaFree((void *)d_kv_prior));
	BENCH_CHECK(cudaFree((void *)d_kv_cur));
	BENCH_CHECK(cudaFree(d_pos));
	BENCH_CHECK(cudaFree(d_out_base));
	BENCH_CHECK(cudaFree(d_out_new));
	BENCH_CHECK(cudaFree(d_partials));
}

static void BenchGemma26bFfnAndMoe(uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	using G = Gemma4_26bGeometry<16u>;
	constexpr uint32_t K = G::kHidden;
	constexpr uint32_t FFN = 128u;
	constexpr uint32_t MOE_N = G::kMoeInter;
	constexpr uint32_t LOCAL_E = G::kLocalExperts;
	constexpr uint32_t TOPK = G::kTopK;
	const uint32_t routes = rows * TOPK;

	const uint16_t *d_act = Tensor<uint16_t>((uint64_t)rows * K, 0.5f, 0.0f);
	const uint16_t *d_gu_w = Matrix(2u * FFN, K);
	uint16_t *d_gu_tmp = 0, *d_inter_base = 0, *d_inter_new = 0;
	BENCH_CHECK(cudaMalloc(&d_gu_tmp, (uint64_t)rows * 2u * FFN * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_inter_base, (uint64_t)rows * FFN * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_inter_new, (uint64_t)rows * FFN * sizeof(uint16_t)));

	auto run_gu_base = [&]()
	{
		GemmaMmaKernel<64u, 32u, 64u, 128u, 2u, 3u, false, false><<<((rows + 63u) / 64u) * ((2u * FFN + 31u) / 32u), 128u, 0, stream>>>(
			d_gu_w, (const uint16_t *)0, d_act, d_gu_tmp, (uint16_t *)0, rows, K, 2u * FFN, 0u);
		GemmaGeluTanhMulKernel<GM_THREADS><<<dim3(LmVecBlocks<GM_THREADS>(FFN), rows), GM_THREADS, 0, stream>>>(
			d_gu_tmp, d_inter_base, FFN);
	};
	auto run_gu_new = [&]()
	{
		constexpr uint32_t smem = 3u * (64u + 32u) * 128u;
		GemmaDenseGateUpActMmaKernel<64u, 16u, 128u, 2u, 0u, false, false, 3u><<<dim3(FFN / 16u, (rows + 63u) / 64u), 128u, smem, stream>>>(
			d_gu_w, (const uint16_t *)0, d_act, d_inter_new, (uint16_t *)0, (float *)0, rows, K, FFN, 0u, INFINITY);
	};
	run_gu_base();
	run_gu_new();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	char label[96];
	snprintf(label, sizeof(label), "26B dense gate_up+GELU rows=%u", rows);
	const float ms_gu_b = TimeMs(stream, run_gu_base);
	const float ms_gu_n = TimeMs(stream, run_gu_new);
	printf("[gemma4 ffn ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_gu_b, ms_gu_n, ms_gu_b / ms_gu_n);
	CheckBf16(label, d_inter_base, d_inter_new, (uint64_t)rows * FFN, 0u, 0.0f);

	std::vector<uint32_t> h_re(routes), h_gro(LOCAL_E + 2u, 0u), h_rpr(routes), h_src(routes);
	std::vector<float> h_pw(routes);
	std::vector<uint32_t> cnt(LOCAL_E + 1u, 0u);
	for ( uint32_t i = 0u; i < routes; ++i )
	{
		const uint32_t ge = Mix(1000u + i) % G::kExperts;
		const uint32_t le = ge < LOCAL_E ? ge : LOCAL_E;
		h_re[i] = le;
		cnt[le]++;
	}
	for ( uint32_t e = 0u; e <= LOCAL_E; ++e )
		h_gro[e + 1u] = h_gro[e] + cnt[e];
	std::vector<uint32_t> cur = h_gro;
	for ( uint32_t i = 0u; i < routes; ++i )
	{
		const uint32_t p = cur[h_re[i]]++;
		h_rpr[i] = p;
		h_src[p] = i / TOPK;
		h_pw[p] = 0.25f + 0.5f * ((float)(i & 15u) / 15.0f);
	}
	uint32_t *d_re = 0, *d_gro = 0, *d_rpr = 0, *d_src = 0, *d_pfx_b = 0, *d_pfx_n = 0;
	float *d_pw = 0;
	BENCH_CHECK(cudaMalloc(&d_re, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_gro, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_rpr, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_src, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_pfx_b, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_pfx_n, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_pw, routes * sizeof(float)));
	BENCH_CHECK(cudaMemcpy(d_re, h_re.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_gro, h_gro.data(), (LOCAL_E + 2u) * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_rpr, h_rpr.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_src, h_src.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_pw, h_pw.data(), routes * sizeof(float), cudaMemcpyHostToDevice));

	const uint32_t local_rows = h_gro[LOCAL_E];
	const uint16_t *d_w1 = Matrix((uint64_t)LOCAL_E * MOE_N, K);
	const uint16_t *d_w3 = Matrix((uint64_t)LOCAL_E * MOE_N, K);
	const uint16_t *d_w2 = Matrix((uint64_t)LOCAL_E * K, MOE_N);
	uint16_t *d_ea_b = 0, *d_ea_n = 0, *d_eo_b = 0, *d_eo_n = 0, *d_fin_b = 0, *d_fin_n = 0;
	BENCH_CHECK(cudaMalloc(&d_ea_b, (uint64_t)routes * MOE_N * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_ea_n, (uint64_t)routes * MOE_N * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_eo_b, (uint64_t)routes * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_eo_n, (uint64_t)routes * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_fin_b, (uint64_t)rows * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_fin_n, (uint64_t)rows * K * sizeof(uint16_t)));

	const uint32_t local_mean = (routes + G::kExperts - 1u) / G::kExperts;
	const bool use128 = local_mean >= 50u;
	constexpr uint32_t smem_w1_b = 2u * (64u + 128u) * 128u;
	constexpr uint32_t smem_w1_n64 = 4u * (64u + 128u) * 128u;
	constexpr uint32_t smem_w1_n128 = 3u * (128u + 128u) * 128u;
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW1ActLdcsKernel<64u, 2u, 3u, false, 2u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w1_b));
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW1ActLdcsKernel<64u, 2u, 3u, false, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w1_n64));
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW1ActLdcsKernel<128u, 4u, 3u, false, 3u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w1_n128));
	auto run_w1_base = [&]()
	{
		GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_b, LOCAL_E, 64u, MOE_N / 64u);
		GemmaMoeMmaW1ActLdcsKernel<64u, 2u, 3u, false, 2u><<<sms, 256u, smem_w1_b, stream>>>(
			d_w1, d_w3, d_act, d_pw, d_gro, d_pfx_b, d_src, d_ea_b, LOCAL_E, K, MOE_N, INFINITY);
	};
	auto run_w1_new = [&]()
	{
		if ( use128 )
		{
			GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_n, LOCAL_E, 128u, MOE_N / 64u);
			GemmaMoeMmaW1ActLdcsKernel<128u, 4u, 3u, false, 3u><<<sms, 256u, smem_w1_n128, stream>>>(
				d_w1, d_w3, d_act, d_pw, d_gro, d_pfx_n, d_src, d_ea_n, LOCAL_E, K, MOE_N, INFINITY);
		}
		else
		{
			GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_n, LOCAL_E, 64u, MOE_N / 64u);
			GemmaMoeMmaW1ActLdcsKernel<64u, 2u, 3u, false, 4u><<<sms, 256u, smem_w1_n64, stream>>>(
				d_w1, d_w3, d_act, d_pw, d_gro, d_pfx_n, d_src, d_ea_n, LOCAL_E, K, MOE_N, INFINITY);
		}
	};
	run_w1_base();
	run_w1_new();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "26B MoE W1W3+GELU rows=%u (%s)", rows, use128 ? "M128-S3" : "M64-S4");
	const float ms_w1_b = TimeMs(stream, run_w1_base);
	const float ms_w1_n = TimeMs(stream, run_w1_new);
	printf("[gemma4 moe ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_w1_b, ms_w1_n, ms_w1_b / ms_w1_n);
	CheckBf16(label, d_ea_b, d_ea_n, (uint64_t)local_rows * MOE_N, 0u, 0.0f);

	constexpr uint32_t smem_w2_b = 2u * (64u + 128u) * 128u;
	constexpr uint32_t smem_w2_n64 = 4u * (64u + 128u) * 128u;
	constexpr uint32_t smem_w2_n128 = 3u * (128u + 128u) * 128u;
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW2GeneralKernel<64u, 128u, 2u, 256u, 2u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w2_b));
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW2GeneralKernel<64u, 128u, 2u, 256u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w2_n64));
	BENCH_CHECK(cudaFuncSetAttribute(GemmaMoeMmaW2GeneralKernel<128u, 128u, 4u, 256u, 3u>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_w2_n128));
	auto run_w2_base = [&]()
	{
		GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_b, LOCAL_E, 64u, K / 128u);
		GemmaMoeMmaW2GeneralKernel<64u, 128u, 2u, 256u, 2u><<<sms, 256u, smem_w2_b, stream>>>(
			d_w2, d_ea_b, d_gro, d_pfx_b, d_eo_b, LOCAL_E, MOE_N, K);
	};
	auto run_w2_new = [&]()
	{
		if ( use128 )
		{
			GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_n, LOCAL_E, 128u, K / 128u);
			GemmaMoeMmaW2GeneralKernel<128u, 128u, 4u, 256u, 3u><<<sms, 256u, smem_w2_n128, stream>>>(
				d_w2, d_ea_n, d_gro, d_pfx_n, d_eo_n, LOCAL_E, MOE_N, K);
		}
		else
		{
			GemmaBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_n, LOCAL_E, 64u, K / 128u);
			GemmaMoeMmaW2GeneralKernel<64u, 128u, 2u, 256u, 4u><<<sms, 256u, smem_w2_n64, stream>>>(
				d_w2, d_ea_n, d_gro, d_pfx_n, d_eo_n, LOCAL_E, MOE_N, K);
		}
	};
	run_w2_base();
	run_w2_new();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "26B MoE W2 rows=%u (%s)", rows, use128 ? "M128-S3" : "M64-S4");
	const float ms_w2_b = TimeMs(stream, run_w2_base);
	const float ms_w2_n = TimeMs(stream, run_w2_new);
	printf("[gemma4 moe ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_w2_b, ms_w2_n, ms_w2_b / ms_w2_n);
	CheckBf16(label, d_eo_b, d_eo_n, (uint64_t)local_rows * K, 0u, 0.0f);

	auto run_fin_base = [&]()
	{
		GemmaMoeLocalFinalizeKernel<false, false, TOPK><<<dim3(1u, rows), GM_THREADS, 0, stream>>>(
			d_eo_b, d_rpr, d_re, (const uint16_t *)0, d_fin_b, TOPK, LOCAL_E, K);
	};
	auto run_fin_new = [&]()
	{
		GemmaMoeLocalFinalizeWarpKernel<false, false, TOPK><<<(rows + 3u) / 4u, 128u, 0, stream>>>(
			d_eo_n, d_rpr, d_re, (const uint16_t *)0, d_fin_n, rows, LOCAL_E, K);
	};
	run_fin_base();
	run_fin_new();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "26B MoE Finalize rows=%u", rows);
	const float ms_fin_b = TimeMs(stream, run_fin_base);
	const float ms_fin_n = TimeMs(stream, run_fin_new);
	printf("[gemma4 moe ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_fin_b, ms_fin_n, ms_fin_b / ms_fin_n);
	CheckBf16(label, d_fin_b, d_fin_n, (uint64_t)rows * K, 0u, 0.0f);

	BENCH_CHECK(cudaFree((void *)d_act));
	BENCH_CHECK(cudaFree((void *)d_gu_w));
	BENCH_CHECK(cudaFree(d_gu_tmp));
	BENCH_CHECK(cudaFree(d_inter_base));
	BENCH_CHECK(cudaFree(d_inter_new));
	BENCH_CHECK(cudaFree(d_re));
	BENCH_CHECK(cudaFree(d_gro));
	BENCH_CHECK(cudaFree(d_rpr));
	BENCH_CHECK(cudaFree(d_src));
	BENCH_CHECK(cudaFree(d_pfx_b));
	BENCH_CHECK(cudaFree(d_pfx_n));
	BENCH_CHECK(cudaFree(d_pw));
	BENCH_CHECK(cudaFree((void *)d_w1));
	BENCH_CHECK(cudaFree((void *)d_w3));
	BENCH_CHECK(cudaFree((void *)d_w2));
	BENCH_CHECK(cudaFree(d_ea_b));
	BENCH_CHECK(cudaFree(d_ea_n));
	BENCH_CHECK(cudaFree(d_eo_b));
	BENCH_CHECK(cudaFree(d_eo_n));
	BENCH_CHECK(cudaFree(d_fin_b));
	BENCH_CHECK(cudaFree(d_fin_n));
}

static void BenchLing3Prefill(uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	using G = Ling3Geometry<16u>;
	using S = LingShape<G>;
	constexpr uint32_t K = G::kHidden;
	constexpr uint32_t CH = S::kKdaChannels;
	constexpr uint32_t W = S::kKdaWidth;
	constexpr uint32_t H = S::kKdaRankHeads;
	constexpr uint32_t SH = G::kSharedRows;
	constexpr uint32_t MOE_N = G::kMoeInter;
	constexpr uint32_t LOCAL_E = G::kLocalExperts;
	constexpr uint32_t TOPK = G::kTopK;
	char label[96];

	const uint16_t *d_normed = Tensor<uint16_t>((uint64_t)rows * K, 0.5f, 0.0f);
	const uint16_t *d_w_qkv = Matrix(CH, K);
	const uint16_t *d_w_f = Matrix(W, K);
	const uint16_t *d_w_g = Matrix(W, K);
	uint16_t *d_qkv_b = 0, *d_f_b = 0, *d_g_b = 0;
	uint16_t *d_qkv_n = 0, *d_f_n = 0, *d_g_n = 0;
	BENCH_CHECK(cudaMalloc(&d_qkv_b, (uint64_t)rows * CH * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_f_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_g_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_qkv_n, (uint64_t)rows * CH * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_f_n, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_g_n, (uint64_t)rows * W * sizeof(uint16_t)));

	LingMultiDensePlan plan;
	memset(&plan, 0, sizeof(plan));
	plan.count = 3u;
	plan.weight[0] = d_w_qkv; plan.out_bf16[0] = d_qkv_n; plan.out_dim[0] = CH; plan.tile_prefix[1] = CH / 128u;
	plan.weight[1] = d_w_f; plan.out_bf16[1] = d_f_n; plan.out_dim[1] = W; plan.tile_prefix[2] = plan.tile_prefix[1] + W / 128u;
	plan.weight[2] = d_w_g; plan.out_bf16[2] = d_g_n; plan.out_dim[2] = W; plan.tile_prefix[3] = plan.tile_prefix[2] + W / 128u;
	constexpr uint32_t md_smem = 4u * (64u + 128u) * 128u;
	BENCH_CHECK(cudaFuncSetAttribute(LingMultiDenseMmaKernel<64u, 128u, 2u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, md_smem));

	auto run_kda_proj_b = [&]()
	{
		LmStreamGemmDense<LmBf16Format>(d_w_qkv, LmScaleTensorNone(), d_normed, d_qkv_b, (float *)0, rows, K, CH, 0u, 0u, sms, (SparkGpuStream)stream);
		LmStreamGemmDense<LmBf16Format>(d_w_f, LmScaleTensorNone(), d_normed, d_f_b, (float *)0, rows, K, W, 0u, 0u, sms, (SparkGpuStream)stream);
		LmStreamGemmDense<LmBf16Format>(d_w_g, LmScaleTensorNone(), d_normed, d_g_b, (float *)0, rows, K, W, 0u, 0u, sms, (SparkGpuStream)stream);
	};
	auto run_kda_proj_n = [&]()
	{
		LingMultiDenseMmaKernel<64u, 128u, 2u, 4u><<<sms, 256u, md_smem, stream>>>(plan, d_normed, rows, K);
	};
	run_kda_proj_b();
	run_kda_proj_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "Ling3 KDA QKV+F+G rows=%u", rows);
	const float ms_kp_b = TimeMs(stream, run_kda_proj_b);
	const float ms_kp_n = TimeMs(stream, run_kda_proj_n);
	printf("[ling3 kda  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_kp_b, ms_kp_n, ms_kp_b / ms_kp_n);
	CheckBf16("Ling3 KDA QKV 0-ULP", d_qkv_b, d_qkv_n, (uint64_t)rows * CH, 0u, 0.0f);
	CheckBf16("Ling3 KDA F 0-ULP", d_f_b, d_f_n, (uint64_t)rows * W, 0u, 0.0f);
	CheckBf16("Ling3 KDA G 0-ULP", d_g_b, d_g_n, (uint64_t)rows * W, 0u, 0.0f);

	const uint16_t *d_conv_w = Tensor<uint16_t>((uint64_t)CH * G::kConv, 0.5f, 0.0f);
	const uint16_t *d_beta = Tensor<uint16_t>((uint64_t)rows * H, 0.5f, 0.0f);
	const float *d_dt_bias = Norm(W);
	const float *d_a_log = Norm(H);
	uint16_t *d_win_init = (uint16_t *)Tensor<uint16_t>((uint64_t)CH * G::kConv, 0.5f, 0.0f);
	uint16_t *d_win_b = 0, *d_win_n = 0;
	uint16_t *d_q_b = 0, *d_k_b = 0, *d_v_b = 0, *d_q_n = 0, *d_k_n = 0, *d_v_n = 0;
	float *d_ret_b = 0, *d_wg_b = 0, *d_ret_n = 0, *d_wg_n = 0;
	uint32_t *d_sidx = 0, *d_srb = 0;
	const uint32_t h_sidx = 0u;
	const uint32_t h_srb[2] = { 0u, rows };
	BENCH_CHECK(cudaMalloc(&d_win_b, (uint64_t)CH * G::kConv * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_win_n, (uint64_t)CH * G::kConv * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_q_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_k_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_v_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_q_n, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_k_n, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_v_n, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_ret_b, (uint64_t)rows * W * sizeof(float)));
	BENCH_CHECK(cudaMalloc(&d_wg_b, (uint64_t)rows * H * sizeof(float)));
	BENCH_CHECK(cudaMalloc(&d_ret_n, (uint64_t)rows * W * sizeof(float)));
	BENCH_CHECK(cudaMalloc(&d_wg_n, (uint64_t)rows * H * sizeof(float)));
	BENCH_CHECK(cudaMalloc(&d_sidx, sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_srb, 2u * sizeof(uint32_t)));
	BENCH_CHECK(cudaMemcpy(d_sidx, &h_sidx, sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_srb, h_srb, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_win_b, d_win_init, (uint64_t)CH * G::kConv * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	BENCH_CHECK(cudaMemcpy(d_win_n, d_win_init, (uint64_t)CH * G::kConv * sizeof(uint16_t), cudaMemcpyDeviceToDevice));

	auto run_conv_b = [&]()
	{
		LingKdaConvSplitTiledKernel<G, false><<<dim3(1u, (CH + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS), LM_CAUSAL_CONV_TILE_THREADS, 0, stream>>>(
			d_win_b, d_sidx, d_srb, (const uint32_t *)0, d_qkv_b, d_conv_w, d_q_b, d_k_b, d_v_b, (uint16_t *)0, 1u, 1u, (const uint32_t *)0);
		LingKdaGatesKernel<G, false, false><<<rows, 64u, 0, stream>>>(
			d_f_b, d_dt_bias, d_a_log, d_beta, (const uint16_t *)0, d_normed, d_ret_b, d_wg_b, (float *)0, (float *)0, rows);
	};
	auto run_conv_n = [&]()
	{
		constexpr uint32_t CONV_BLKS = (CH + LG_THREADS - 1u) / LG_THREADS;
		LingKdaConvGatesPrefillKernel<G><<<dim3((rows + 31u) / 32u, CONV_BLKS + 1u), LG_THREADS, 0, stream>>>(
			d_win_n, d_sidx, d_qkv_b, d_conv_w, d_f_b, d_dt_bias, d_a_log, d_beta, d_q_n, d_k_n, d_v_n, d_ret_n, d_wg_n, rows);
		LingKdaWindowCommitKernel<G><<<CONV_BLKS, LG_THREADS, 0, stream>>>(d_win_n, d_sidx, d_qkv_b, rows);
	};
	run_conv_b();
	run_conv_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	CheckBf16("Ling3 KDA conv Q 0-ULP", d_q_b, d_q_n, (uint64_t)rows * W, 0u, 0.0f);
	CheckBf16("Ling3 KDA conv V 0-ULP", d_v_b, d_v_n, (uint64_t)rows * W, 0u, 0.0f);
	CheckBf16("Ling3 KDA conv window 0-ULP", d_win_b, d_win_n, (uint64_t)CH * G::kConv, 0u, 0.0f);
	CheckF32("Ling3 KDA retention 0-ULP", d_ret_b, d_ret_n, (uint64_t)rows * W, 0.0f);
	CheckF32("Ling3 KDA write_gate 0-ULP", d_wg_b, d_wg_n, (uint64_t)rows * H, 0.0f);
	snprintf(label, sizeof(label), "Ling3 KDA Conv+Gates rows=%u", rows);
	const float ms_cv_b = TimeMs(stream, run_conv_b);
	const float ms_cv_n = TimeMs(stream, run_conv_n);
	printf("[ling3 kda  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_cv_b, ms_cv_n, ms_cv_b / ms_cv_n);

	const uint16_t *d_knorm_w = Tensor<uint16_t>(G::kKdaDim, 0.2f, 1.0f);
	uint16_t *d_mix_b = 0, *d_mix_n = 0;
	BENCH_CHECK(cudaMalloc(&d_mix_b, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_mix_n, (uint64_t)rows * W * sizeof(uint16_t)));
	BENCH_CHECK(cudaMemcpy(d_mix_b, d_q_b, (uint64_t)rows * W * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	BENCH_CHECK(cudaMemcpy(d_mix_n, d_q_b, (uint64_t)rows * W * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	LingKdaHeadNormGateKernel<G::kKdaDim, G::kKdaNormRoundOnce><<<dim3(H, rows), 128u, 0, stream>>>(d_mix_b, d_g_b, d_knorm_w, H, G::kNormEpsilon);
	LingKdaHeadNormGateWarpKernel<G::kKdaDim, G::kKdaNormRoundOnce><<<(rows * H + 3u) / 4u, 128u, 0, stream>>>(d_mix_n, d_g_b, d_knorm_w, rows * H, G::kNormEpsilon);
	BENCH_CHECK(cudaStreamSynchronize(stream));
	CheckBf16("Ling3 KDA HeadNorm+Gate 0-ULP", d_mix_b, d_mix_n, (uint64_t)rows * W, 0u, 0.0f);
	snprintf(label, sizeof(label), "Ling3 KDA HeadNorm+Gate rows=%u", rows);
	const float ms_hn_b = TimeMs(stream, [&]() {
		LingKdaHeadNormGateKernel<G::kKdaDim, G::kKdaNormRoundOnce><<<dim3(H, rows), 128u, 0, stream>>>(d_mix_b, d_g_b, d_knorm_w, H, G::kNormEpsilon);
	});
	const float ms_hn_n = TimeMs(stream, [&]() {
		LingKdaHeadNormGateWarpKernel<G::kKdaDim, G::kKdaNormRoundOnce><<<(rows * H + 3u) / 4u, 128u, 0, stream>>>(d_mix_n, d_g_b, d_knorm_w, rows * H, G::kNormEpsilon);
	});
	printf("[ling3 kda  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_hn_b, ms_hn_n, ms_hn_b / ms_hn_n);

	const uint16_t *d_sh_gu = Tensor<uint16_t>((uint64_t)rows * 2u * SH, 0.5f, 0.0f);
	const uint16_t *d_sh_dn = Matrix(K, SH);
	uint16_t *d_sh_act = 0, *d_sh_out_b = 0, *d_sh_out_n = 0;
	BENCH_CHECK(cudaMalloc(&d_sh_act, (uint64_t)rows * SH * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_sh_out_b, (uint64_t)rows * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_sh_out_n, (uint64_t)rows * K * sizeof(uint16_t)));
	auto run_sh_b = [&]()
	{
		LingSiluMulLimitKernel<G::kFfnSiluRounded><<<rows, LG_THREADS, 0, stream>>>(d_sh_gu, d_sh_act, SH, INFINITY);
		LmStreamGemmDense<LmBf16Format>(d_sh_dn, LmScaleTensorNone(), d_sh_act, d_sh_out_b, (float *)0, rows, SH, K, 0u, 0u, sms, (SparkGpuStream)stream);
	};
	auto run_sh_n = [&]()
	{
		LingSharedActDownMmaKernel<G><<<dim3(K / 64u, (rows + 63u) / 64u), 128u, 0, stream>>>(d_sh_gu, d_sh_dn, d_sh_out_n, rows, INFINITY);
	};
	run_sh_b();
	run_sh_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "Ling3 Shared Act+Down rows=%u", rows);
	const float ms_sh_b = TimeMs(stream, run_sh_b);
	const float ms_sh_n = TimeMs(stream, run_sh_n);
	printf("[ling3 moe  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_sh_b, ms_sh_n, ms_sh_b / ms_sh_n);
	CheckBf16(label, d_sh_out_b, d_sh_out_n, (uint64_t)rows * K, 0u, 0.0f);

	const uint32_t routes = rows * TOPK;
	std::vector<uint32_t> h_re(routes), h_gro(LOCAL_E + 2u, 0u), h_rpr(routes), h_src(routes);
	std::vector<float> h_rw(routes), h_pw(routes);
	std::vector<uint32_t> cnt(LOCAL_E + 1u, 0u);
	for ( uint32_t i = 0u; i < routes; ++i )
	{
		const uint32_t ge = Mix(5000u + i) % G::kExperts;
		const uint32_t le = ge < LOCAL_E ? ge : LOCAL_E;
		h_re[i] = le;
		h_rw[i] = le < LOCAL_E ? (0.1f + 0.3f * ((float)(i & 7u) / 7.0f)) : 0.0f;
		cnt[le]++;
	}
	for ( uint32_t e = 0u; e <= LOCAL_E; ++e )
		h_gro[e + 1u] = h_gro[e] + cnt[e];
	std::vector<uint32_t> cur = h_gro;
	for ( uint32_t i = 0u; i < routes; ++i )
	{
		const uint32_t p = cur[h_re[i]]++;
		h_rpr[i] = p;
		h_src[p] = i / TOPK;
		h_pw[p] = h_rw[i];
	}
	const uint32_t local_rows = h_gro[LOCAL_E];
	uint32_t *d_re = 0, *d_gro = 0, *d_rpr = 0, *d_src = 0, *d_pfx_b = 0, *d_pfx_n = 0;
	float *d_rw = 0, *d_pw = 0;
	BENCH_CHECK(cudaMalloc(&d_re, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_gro, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_rpr, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_src, routes * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_pfx_b, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_pfx_n, (LOCAL_E + 2u) * sizeof(uint32_t)));
	BENCH_CHECK(cudaMalloc(&d_rw, routes * sizeof(float)));
	BENCH_CHECK(cudaMalloc(&d_pw, routes * sizeof(float)));
	BENCH_CHECK(cudaMemcpy(d_re, h_re.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_gro, h_gro.data(), (LOCAL_E + 2u) * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_rpr, h_rpr.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_src, h_src.data(), routes * sizeof(uint32_t), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_rw, h_rw.data(), routes * sizeof(float), cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(d_pw, h_pw.data(), routes * sizeof(float), cudaMemcpyHostToDevice));

	const uint16_t *d_w1 = Matrix((uint64_t)LOCAL_E * MOE_N, K);
	const uint16_t *d_w3 = Matrix((uint64_t)LOCAL_E * MOE_N, K);
	const uint16_t *d_w2 = Matrix((uint64_t)LOCAL_E * K, MOE_N);
	uint16_t *d_ea_b = 0, *d_ea_n = 0, *d_eo_b = 0, *d_eo_n = 0, *d_fin_b = 0, *d_fin_n = 0;
	BENCH_CHECK(cudaMalloc(&d_ea_b, (uint64_t)routes * MOE_N * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_ea_n, (uint64_t)routes * MOE_N * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_eo_b, (uint64_t)routes * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_eo_n, (uint64_t)routes * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_fin_b, (uint64_t)rows * K * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&d_fin_n, (uint64_t)rows * K * sizeof(uint16_t)));

	const uint32_t expected_local = (uint32_t)(((uint64_t)routes * LOCAL_E + G::kExperts - 1u) / G::kExperts);
	const uint32_t mean = (expected_local + LOCAL_E - 1u) / LOCAL_E;
	const uint32_t base_tm = mean > 32u ? 64u : (mean > 16u ? 32u : 16u);
	const uint32_t new_tm = mean >= 50u ? 128u : mean >= 22u ? 64u : mean >= 10u ? 32u : 16u;
	constexpr uint32_t sh_w1_16 = 4u * (16u + 128u) * 128u;
	constexpr uint32_t sh_w1_32 = 4u * (32u + 128u) * 128u;
	constexpr uint32_t sh_w1_64 = 4u * (64u + 128u) * 128u;
	constexpr uint32_t sh_w2_16 = 4u * (16u + 128u) * 128u;
	constexpr uint32_t sh_w2_32 = 4u * (32u + 128u) * 128u;
	constexpr uint32_t sh_w2_64 = 4u * (64u + 128u) * 128u;
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW1W3SiluBf16Kernel<G, 16u, 1u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w1_16));
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW1W3SiluBf16Kernel<G, 32u, 2u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w1_32));
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW1W3SiluBf16Kernel<G, 64u, 4u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w1_64));
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW2Bf16Kernel<G, 16u, 128u, 1u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w2_16));
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW2Bf16Kernel<G, 32u, 128u, 2u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w2_32));
	BENCH_CHECK(cudaFuncSetAttribute(LingMoeMmaW2Bf16Kernel<G, 64u, 128u, 2u, 4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_w2_64));

	auto launch_w1_tm = [&](uint32_t tm, uint32_t *pfx, uint16_t *out)
	{
		LingBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, pfx, LOCAL_E, tm, MOE_N / 64u);
		if ( tm == 64u )
			LingMoeMmaW1W3SiluBf16Kernel<G, 64u, 4u, 4u><<<sms, 256u, sh_w1_64, stream>>>(d_w1, d_w3, d_normed, d_pw, d_gro, pfx, d_src, out, INFINITY);
		else if ( tm == 32u )
			LingMoeMmaW1W3SiluBf16Kernel<G, 32u, 2u, 4u><<<sms, 256u, sh_w1_32, stream>>>(d_w1, d_w3, d_normed, d_pw, d_gro, pfx, d_src, out, INFINITY);
		else
			LingMoeMmaW1W3SiluBf16Kernel<G, 16u, 1u, 4u><<<sms, 256u, sh_w1_16, stream>>>(d_w1, d_w3, d_normed, d_pw, d_gro, pfx, d_src, out, INFINITY);
	};
	auto run_moe_w1_b = [&]() { launch_w1_tm(base_tm, d_pfx_b, d_ea_b); };
	auto run_moe_w1_n = [&]() { launch_w1_tm(new_tm, d_pfx_n, d_ea_n); };
	run_moe_w1_b();
	run_moe_w1_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "Ling3 MoE W1W3 rows=%u (M%u->M%u)", rows, base_tm, new_tm);
	const float ms_lw1_b = TimeMs(stream, run_moe_w1_b);
	const float ms_lw1_n = TimeMs(stream, run_moe_w1_n);
	printf("[ling3 moe  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_lw1_b, ms_lw1_n, ms_lw1_b / ms_lw1_n);
	CheckBf16(label, d_ea_b, d_ea_n, (uint64_t)local_rows * MOE_N, 0u, 0.0f);

	auto run_moe_w2_b = [&]()
	{
		LmStreamGemmGrouped<LmBf16Format>(d_w2, LmScaleTensorNone(), d_ea_b, d_eo_b, d_gro, (const uint32_t *)0,
			LOCAL_E, expected_local, MOE_N, K, sms, (SparkGpuStream)stream);
	};
	auto run_moe_w2_n = [&]()
	{
		LingBuildTilePrefixKernel<<<1u, 32u, 0, stream>>>(d_gro, d_pfx_n, LOCAL_E, new_tm, K / 128u);
		if ( new_tm == 64u )
			LingMoeMmaW2Bf16Kernel<G, 64u, 128u, 2u, 4u><<<sms, 256u, sh_w2_64, stream>>>(d_w2, d_ea_n, d_gro, d_pfx_n, d_eo_n);
		else if ( new_tm == 32u )
			LingMoeMmaW2Bf16Kernel<G, 32u, 128u, 2u, 4u><<<sms, 256u, sh_w2_32, stream>>>(d_w2, d_ea_n, d_gro, d_pfx_n, d_eo_n);
		else
			LingMoeMmaW2Bf16Kernel<G, 16u, 128u, 1u, 4u><<<sms, 256u, sh_w2_16, stream>>>(d_w2, d_ea_n, d_gro, d_pfx_n, d_eo_n);
	};
	run_moe_w2_b();
	run_moe_w2_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "Ling3 MoE W2 rows=%u (M%u)", rows, new_tm);
	const float ms_lw2_b = TimeMs(stream, run_moe_w2_b);
	const float ms_lw2_n = TimeMs(stream, run_moe_w2_n);
	printf("[ling3 moe  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_lw2_b, ms_lw2_n, ms_lw2_b / ms_lw2_n);
	CheckBf16(label, d_eo_b, d_eo_n, (uint64_t)local_rows * K, 0u, 0.0f);

	auto run_lfin_b = [&]()
	{
		LingMoeLocalFinalizeKernel<G::kExpertWeightAfterDown><<<dim3(LmVecBlocks<LG_THREADS>(K), rows), LG_THREADS, 0, stream>>>(
			d_eo_b, d_rpr, d_re, d_rw, d_sh_out_b, d_fin_b, TOPK, LOCAL_E, K);
	};
	auto run_lfin_n = [&]()
	{
		LingMoeFinalizeWarpSliceKernel<G::kExpertWeightAfterDown, TOPK><<<(rows + 3u) / 4u, 128u, 0, stream>>>(
			d_eo_n, d_rpr, d_re, d_rw, d_sh_out_n, d_fin_n, rows, LOCAL_E, 0u, K, K);
	};
	run_lfin_b();
	run_lfin_n();
	BENCH_CHECK(cudaStreamSynchronize(stream));
	snprintf(label, sizeof(label), "Ling3 MoE Finalize rows=%u", rows);
	const float ms_lf_b = TimeMs(stream, run_lfin_b);
	const float ms_lf_n = TimeMs(stream, run_lfin_n);
	printf("[ling3 moe  ] %-38s BASE=%.3f ms NEW=%.3f ms speedup=%.2fx\n", label, ms_lf_b, ms_lf_n, ms_lf_b / ms_lf_n);
	CheckBf16(label, d_fin_b, d_fin_n, (uint64_t)rows * K, 0u, 0.0f);

	BENCH_CHECK(cudaFree((void *)d_normed));
	BENCH_CHECK(cudaFree((void *)d_w_qkv));
	BENCH_CHECK(cudaFree((void *)d_w_f));
	BENCH_CHECK(cudaFree((void *)d_w_g));
	BENCH_CHECK(cudaFree(d_qkv_b));
	BENCH_CHECK(cudaFree(d_f_b));
	BENCH_CHECK(cudaFree(d_g_b));
	BENCH_CHECK(cudaFree(d_qkv_n));
	BENCH_CHECK(cudaFree(d_f_n));
	BENCH_CHECK(cudaFree(d_g_n));
	BENCH_CHECK(cudaFree((void *)d_conv_w));
	BENCH_CHECK(cudaFree((void *)d_beta));
	BENCH_CHECK(cudaFree((void *)d_dt_bias));
	BENCH_CHECK(cudaFree((void *)d_a_log));
	BENCH_CHECK(cudaFree(d_win_init));
	BENCH_CHECK(cudaFree(d_win_b));
	BENCH_CHECK(cudaFree(d_win_n));
	BENCH_CHECK(cudaFree(d_q_b));
	BENCH_CHECK(cudaFree(d_k_b));
	BENCH_CHECK(cudaFree(d_v_b));
	BENCH_CHECK(cudaFree(d_q_n));
	BENCH_CHECK(cudaFree(d_k_n));
	BENCH_CHECK(cudaFree(d_v_n));
	BENCH_CHECK(cudaFree(d_ret_b));
	BENCH_CHECK(cudaFree(d_wg_b));
	BENCH_CHECK(cudaFree(d_ret_n));
	BENCH_CHECK(cudaFree(d_wg_n));
	BENCH_CHECK(cudaFree(d_sidx));
	BENCH_CHECK(cudaFree(d_srb));
	BENCH_CHECK(cudaFree((void *)d_knorm_w));
	BENCH_CHECK(cudaFree(d_mix_b));
	BENCH_CHECK(cudaFree(d_mix_n));
	BENCH_CHECK(cudaFree((void *)d_sh_gu));
	BENCH_CHECK(cudaFree((void *)d_sh_dn));
	BENCH_CHECK(cudaFree(d_sh_act));
	BENCH_CHECK(cudaFree(d_sh_out_b));
	BENCH_CHECK(cudaFree(d_sh_out_n));
	BENCH_CHECK(cudaFree(d_re));
	BENCH_CHECK(cudaFree(d_gro));
	BENCH_CHECK(cudaFree(d_rpr));
	BENCH_CHECK(cudaFree(d_src));
	BENCH_CHECK(cudaFree(d_pfx_b));
	BENCH_CHECK(cudaFree(d_pfx_n));
	BENCH_CHECK(cudaFree(d_rw));
	BENCH_CHECK(cudaFree(d_pw));
	BENCH_CHECK(cudaFree((void *)d_w1));
	BENCH_CHECK(cudaFree((void *)d_w3));
	BENCH_CHECK(cudaFree((void *)d_w2));
	BENCH_CHECK(cudaFree(d_ea_b));
	BENCH_CHECK(cudaFree(d_ea_n));
	BENCH_CHECK(cudaFree(d_eo_b));
	BENCH_CHECK(cudaFree(d_eo_n));
	BENCH_CHECK(cudaFree(d_fin_b));
	BENCH_CHECK(cudaFree(d_fin_n));
}

int main(void)
{
	cudaDeviceProp prop;
	BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
	const uint32_t sms = (uint32_t)prop.multiProcessorCount;
	printf("Device: %s (sm_%d%d, %u SMs, maxSmemPerBlockOptin=%zu)\n",
		prop.name, prop.major, prop.minor, sms, prop.sharedMemPerBlockOptin);
	cudaStream_t stream = 0;
	BENCH_CHECK(cudaStreamCreate(&stream));

	printf("\n=== GEMMA-4 P4 PREFILL ATTENTION (BASE vs NEW + 4c 0-ULP) ===\n");
	BenchGemmaPrefillAttn<512u, 0u>("gemma4-26b full HD=512 r512 ctx8K", 1u, 0u, 512u, 8192u, 2048u, sms, stream);
	BenchGemmaPrefillAttn<512u, 0u>("gemma4-26b full HD=512 r512 ctx32K", 1u, 0u, 512u, 32768u, 2048u, sms, stream);
	BenchGemmaPrefillAttn<512u, 1u>("gemma4-26b full HD=512 r128 ctx8K 4c", 1u, 0u, 128u, 8192u, 2048u, sms, stream);
	BenchGemmaPrefillAttn<512u, 0u>("gemma4-31b full HD=512 r512 ctx8K", 2u, 0u, 512u, 8192u, 2048u, sms, stream);
	BenchGemmaPrefillAttn<512u, 0u>("gemma4-31b full HD=512 r512 ctx32K", 2u, 0u, 512u, 32768u, 2048u, sms, stream);
	BenchGemmaPrefillAttn<256u, 4u>("gemma4-26b slid HD=256 r512 ctx8K", 1u, 1024u, 512u, 8192u, 256u, sms, stream);
	BenchGemmaPrefillAttn<256u, 4u>("gemma4-31b slid HD=256 r512 ctx8K", 2u, 1024u, 512u, 8192u, 256u, sms, stream);

	printf("\n=== GEMMA-4-26B-A4B FFN + MOE (BASE vs NEW + 0-ULP) ===\n");
	BenchGemma26bFfnAndMoe(512u, sms, stream);
	BenchGemma26bFfnAndMoe(1024u, sms, stream);
	BenchGemma26bFfnAndMoe(2048u, sms, stream);

	printf("\n=== LING-3.0-FLASH KDA + MOE (BASE vs NEW + 0-ULP) ===\n");
	BenchLing3Prefill(512u, sms, stream);
	BenchLing3Prefill(1024u, sms, stream);
	BenchLing3Prefill(2048u, sms, stream);

	BENCH_CHECK(cudaStreamDestroy(stream));
	if ( g_bench_fail != 0 )
	{
		printf("\nFAILED: %d checks failed\n", g_bench_fail);
		return 1;
	}
	printf("\nALL PAIRED CHECKS PASSED (0-ULP verified)\n");
	return 0;
}
