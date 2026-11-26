#define QWEN38_27B_TP_DEGREE 1u
#define QWEN38_FLASH_EXPERT_CODEC SPARK_WEIGHT_CODEC_NVFP4_E2M1
#define QWEN38_FLASH_PLE_TABLE_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3

#include "modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stage_model.cu"
#include "modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stage_model.cu"
#include "modules/qwen38_flash_resident_decode_stage/source/spark_qwen38_flash_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

static constexpr uint32_t kBenchWarmup = 6u;
static constexpr uint32_t kBenchIters = 20u;

template<typename Fn>
static float TimeMs(cudaStream_t stream, Fn fn, uint32_t warmup = kBenchWarmup, uint32_t iters = kBenchIters)
{
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);
	for ( uint32_t w = 0u; w < warmup; ++w )
		fn();
	const cudaError_t werr = cudaStreamSynchronize(stream);
	if ( werr != cudaSuccess )
	{
		printf("FAIL cuda warmup error %d (%s)\n", (int)werr, cudaGetErrorString(werr));
		vf_failures++;
		cudaEventDestroy(t0);
		cudaEventDestroy(t1);
		return 0.0f;
	}
	cudaEventRecord(t0, stream);
	for ( uint32_t it = 0u; it < iters; ++it )
		fn();
	cudaEventRecord(t1, stream);
	const cudaError_t rerr = cudaEventSynchronize(t1);
	if ( rerr != cudaSuccess )
	{
		printf("FAIL cuda bench error %d (%s)\n", (int)rerr, cudaGetErrorString(rerr));
		vf_failures++;
		cudaEventDestroy(t0);
		cudaEventDestroy(t1);
		return 0.0f;
	}
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, t0, t1);
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	return ms / (float)iters;
}

static uint32_t CountBf16Diffs(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, float *max_abs_out = 0)
{
	uint32_t diffs = 0u;
	float max_abs = 0.0f;
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for ( size_t i = 0u; i < n; ++i )
	{
		if ( a[i] != b[i] )
		{
			diffs++;
			const float d = fabsf(VfFloat(a[i]) - VfFloat(b[i]));
			if ( d > max_abs )
				max_abs = d;
		}
	}
	if ( max_abs_out != 0 )
		*max_abs_out = max_abs;
	return diffs;
}

static double RelL2Bf16(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b)
{
	double num = 0.0, den = 0.0;
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for ( size_t i = 0u; i < n; ++i )
	{
		const double va = (double)VfFloat(a[i]), vb = (double)VfFloat(b[i]);
		const double d = va - vb;
		num += d * d;
		den += va * va;
	}
	return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static double RelL2F32(const std::vector<float> &a, const std::vector<float> &b, uint32_t *bit_diffs = 0)
{
	double num = 0.0, den = 0.0;
	uint32_t diffs = 0u;
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for ( size_t i = 0u; i < n; ++i )
	{
		uint32_t ba, bb;
		memcpy(&ba, &a[i], 4u);
		memcpy(&bb, &b[i], 4u);
		if ( ba != bb )
			diffs++;
		const double d = (double)a[i] - (double)b[i];
		num += d * d;
		den += (double)a[i] * (double)a[i];
	}
	if ( bit_diffs != 0 )
		*bit_diffs = diffs;
	return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static void BenchGdnPrefill(cudaStream_t stream, const char *label, uint32_t qk_heads, uint32_t v_heads, uint32_t rows)
{
	constexpr uint32_t KEY_DIM = 128u, VAL_DIM = 128u, CONV = 4u;
	const uint32_t gdn_qk = qk_heads * KEY_DIM, gdn_v = v_heads * VAL_DIM, channels = 2u * gdn_qk + gdn_v;
	const uint32_t qkvz_stride = channels + gdn_v;
	const uint32_t slot_bytes = v_heads * KEY_DIM * VAL_DIM * (uint32_t)sizeof(float);
	const uint32_t win_elems = channels * CONV;

	const uint16_t *qkvz = Tensor<uint16_t>((uint64_t)rows * qkvz_stride, 0.5f, 0.0f);
	const uint16_t *conv_w = Tensor<uint16_t>((uint64_t)channels * CONV, 0.25f, 0.0f);
	const uint16_t *ba = Tensor<uint16_t>((uint64_t)rows * 2u * v_heads, 0.4f, 0.0f);
	const float *a_log = Tensor<float>(v_heads, 0.2f, -1.5f);
	const float *dt_bias = Tensor<float>(v_heads, 0.2f, -1.0f);

	uint16_t *win_init = (uint16_t *)Tensor<uint16_t>(win_elems, 0.2f, 0.0f);
	float *st_init = (float *)Tensor<float>(v_heads * KEY_DIM * VAL_DIM, 0.05f, 0.0f);

	uint16_t *win_base = 0, *win_new = 0, *q_buf = 0, *k_buf = 0, *v_buf = 0, *out_base = 0, *out_new = 0;
	float *st_base = 0, *st_new = 0, *ret_buf = 0, *wg_buf = 0;
	uint32_t *d_state_idx = 0, *d_seq_begin = 0;
	cudaMalloc(&win_base, (size_t)win_elems * sizeof(uint16_t));
	cudaMalloc(&win_new, (size_t)win_elems * sizeof(uint16_t));
	cudaMalloc(&q_buf, (size_t)rows * v_heads * KEY_DIM * sizeof(uint16_t));
	cudaMalloc(&k_buf, (size_t)rows * v_heads * KEY_DIM * sizeof(uint16_t));
	cudaMalloc(&v_buf, (size_t)rows * gdn_v * sizeof(uint16_t));
	cudaMalloc(&out_base, (size_t)rows * gdn_v * sizeof(uint16_t));
	cudaMalloc(&out_new, (size_t)rows * gdn_v * sizeof(uint16_t));
	cudaMalloc(&st_base, slot_bytes);
	cudaMalloc(&st_new, slot_bytes);
	cudaMalloc(&ret_buf, (size_t)rows * v_heads * KEY_DIM * sizeof(float));
	cudaMalloc(&wg_buf, (size_t)rows * v_heads * sizeof(float));
	const uint32_t h_idx[1] = { 0u }, h_beg[2] = { 0u, rows };
	cudaMalloc(&d_state_idx, sizeof(uint32_t));
	cudaMalloc(&d_seq_begin, 2u * sizeof(uint32_t));
	cudaMemcpy(d_state_idx, h_idx, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_seq_begin, h_beg, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice);

	const uint32_t ch_tiles = (channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t gate_blocks = (v_heads + (QH_THREADS / LM_WARP_LANES) - 1u) / (QH_THREADS / LM_WARP_LANES);
	const uint32_t row_tiles = (rows + LM_CAUSAL_CONV_TILE_ROWS - 1u) / LM_CAUSAL_CONV_TILE_ROWS;

	auto run_base = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenGdnConvSplitTiledKernel<CONV, KEY_DIM>),
			dim3(1u, ch_tiles), LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
			win_base, d_state_idx, d_seq_begin, (const uint32_t *)0, qkvz, conv_w,
			q_buf, k_buf, v_buf, (uint16_t *)0, gdn_qk, gdn_v, v_heads, qkvz_stride, 1u, 1u, (const uint32_t *)0);
		(void)LM_LAUNCH_DEPENDENT((QwenGdnGateOnlyKernel<KEY_DIM>),
			dim3(gate_blocks, rows), QH_THREADS, 0, stream,
			ba, a_log, dt_bias, ret_buf, wg_buf, v_heads, rows);
		(void)LmDeltaRuleChunkLaunch<QH_THREADS, KEY_DIM, VAL_DIM, float>((uint8_t *)st_base, slot_bytes, d_state_idx,
			d_seq_begin, (const uint32_t *)0, q_buf, k_buf, v_buf, ret_buf, wg_buf, out_base,
			v_heads, 1u, 1u, 1u, (const uint32_t *)0, stream);
	};
	auto run_new = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenGdnConvPrefill2DKernel<CONV, KEY_DIM>),
			dim3(1u, ch_tiles, row_tiles), LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
			win_new, d_state_idx, d_seq_begin, (const uint32_t *)0, qkvz, conv_w,
			q_buf, k_buf, v_buf, gdn_qk, gdn_v, qkvz_stride, 1u, row_tiles == 1u ? 1u : 0u, (const uint32_t *)0);
		if ( row_tiles > 1u )
			(void)LM_LAUNCH_DEPENDENT((QwenGdnConvWindowCommitKernel<CONV>),
				dim3(1u, (channels + QH_THREADS - 1u) / QH_THREADS), QH_THREADS, 0, stream,
				win_new, d_state_idx, d_seq_begin, (const uint32_t *)0, qkvz, channels, qkvz_stride, 1u, (const uint32_t *)0);
		(void)QwenGdnScalarChunkLaunch<KEY_DIM, VAL_DIM>((uint8_t *)st_new, slot_bytes, d_state_idx,
			d_seq_begin, (const uint32_t *)0, q_buf, k_buf, v_buf, ba, a_log, dt_bias, out_new,
			qk_heads, v_heads, 1u, 1u, (const uint32_t *)0, stream);
	};

	cudaMemcpyAsync(win_base, win_init, (size_t)win_elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(st_base, st_init, slot_bytes, cudaMemcpyDeviceToDevice, stream);
	run_base();
	cudaMemcpyAsync(win_new, win_init, (size_t)win_elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(st_new, st_init, slot_bytes, cudaMemcpyDeviceToDevice, stream);
	run_new();
	cudaStreamSynchronize(stream);

	std::vector<uint16_t> hb(rows * gdn_v), hn(rows * gdn_v), wb(win_elems), wn(win_elems);
	std::vector<float> sb(v_heads * KEY_DIM * VAL_DIM), sn(v_heads * KEY_DIM * VAL_DIM);
	cudaMemcpy(hb.data(), out_base, hb.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(hn.data(), out_new, hn.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(wb.data(), win_base, wb.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(wn.data(), win_new, wn.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(sb.data(), st_base, slot_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sn.data(), st_new, slot_bytes, cudaMemcpyDeviceToHost);

	const double out_rel = RelL2Bf16(hb, hn);
	const double st_rel = RelL2F32(sb, sn);
	const uint32_t win_diffs = CountBf16Diffs(wb, wn);
	Require(win_diffs == 0u && out_rel < 5e-3 && st_rel < 5e-3, label, "GDN accuracy vs BASE", win_diffs, (uint32_t)(out_rel * 1e6), (uint32_t)(st_rel * 1e6));

	uint32_t gx_out_diffs = 0u, gx_st_diffs = 0u, gx_win_diffs = 0u;
	if ( rows >= 192u )
	{
		const uint32_t cached = 128u, delta = 23u, total = cached + delta;
		uint16_t *out_cold = 0, *out_warm = 0;
		cudaMalloc(&out_cold, (size_t)total * gdn_v * sizeof(uint16_t));
		cudaMalloc(&out_warm, (size_t)delta * gdn_v * sizeof(uint16_t));
		auto run_new_range = [&](const uint16_t *in_qkvz, const uint16_t *in_ba, uint32_t n_rows, uint16_t *dst_win, float *dst_st, uint16_t *dst_out) {
			const uint32_t hb2[2] = { 0u, n_rows };
			cudaMemcpyAsync(d_seq_begin, hb2, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
			const uint32_t rt = (n_rows + LM_CAUSAL_CONV_TILE_ROWS - 1u) / LM_CAUSAL_CONV_TILE_ROWS;
			(void)LM_LAUNCH_DEPENDENT((QwenGdnConvPrefill2DKernel<CONV, KEY_DIM>),
				dim3(1u, ch_tiles, rt), LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
				dst_win, d_state_idx, d_seq_begin, (const uint32_t *)0, in_qkvz, conv_w,
				q_buf, k_buf, v_buf, gdn_qk, gdn_v, qkvz_stride, 1u, rt == 1u ? 1u : 0u, (const uint32_t *)0);
			if ( rt > 1u )
				(void)LM_LAUNCH_DEPENDENT((QwenGdnConvWindowCommitKernel<CONV>),
					dim3(1u, (channels + QH_THREADS - 1u) / QH_THREADS), QH_THREADS, 0, stream,
					dst_win, d_state_idx, d_seq_begin, (const uint32_t *)0, in_qkvz, channels, qkvz_stride, 1u, (const uint32_t *)0);
			(void)QwenGdnScalarChunkLaunch<KEY_DIM, VAL_DIM>((uint8_t *)dst_st, slot_bytes, d_state_idx,
				d_seq_begin, (const uint32_t *)0, q_buf, k_buf, v_buf, in_ba, a_log, dt_bias, dst_out,
				qk_heads, v_heads, 1u, 1u, (const uint32_t *)0, stream);
		};
		cudaMemcpyAsync(win_base, win_init, (size_t)win_elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
		cudaMemcpyAsync(st_base, st_init, slot_bytes, cudaMemcpyDeviceToDevice, stream);
		run_new_range(qkvz, ba, total, win_base, st_base, out_cold);
		cudaMemcpyAsync(win_new, win_init, (size_t)win_elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
		cudaMemcpyAsync(st_new, st_init, slot_bytes, cudaMemcpyDeviceToDevice, stream);
		run_new_range(qkvz, ba, cached, win_new, st_new, out_new);
		run_new_range(qkvz + (uint64_t)cached * qkvz_stride, ba + (uint64_t)cached * 2u * v_heads, delta, win_new, st_new, out_warm);
		cudaStreamSynchronize(stream);
		std::vector<uint16_t> hc(delta * gdn_v), hw(delta * gdn_v);
		cudaMemcpy(hc.data(), out_cold + (uint64_t)cached * gdn_v, hc.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(hw.data(), out_warm, hw.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(wb.data(), win_base, wb.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(wn.data(), win_new, wn.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(sb.data(), st_base, slot_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(sn.data(), st_new, slot_bytes, cudaMemcpyDeviceToHost);
		gx_out_diffs = CountBf16Diffs(hc, hw);
		gx_win_diffs = CountBf16Diffs(wb, wn);
		(void)RelL2F32(sb, sn, &gx_st_diffs);
		Require(gx_out_diffs == 0u && gx_win_diffs == 0u && gx_st_diffs == 0u, label, "GDN Gate GX 0-ULP", gx_out_diffs, gx_win_diffs, gx_st_diffs);
		cudaMemcpy(d_seq_begin, h_beg, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaFree(out_cold);
		cudaFree(out_warm);
	}

	const float base_ms = TimeMs(stream, run_base);
	const float new_ms = TimeMs(stream, run_new);
	printf("  GDN %-18s rows=%4u qk=%2u v=%2u | BASE=%7.3f ms NEW=%7.3f ms speedup=%5.2fx | relL2=%.2e GX_diff=(%u,%u,%u)\n",
		label, rows, qk_heads, v_heads, base_ms, new_ms, new_ms > 0.0f ? base_ms / new_ms : 0.0f, out_rel, gx_out_diffs, gx_win_diffs, gx_st_diffs);

	cudaFree((void *)qkvz); cudaFree((void *)conv_w); cudaFree((void *)ba); cudaFree((void *)a_log); cudaFree((void *)dt_bias);
	cudaFree(win_init); cudaFree(st_init); cudaFree(win_base); cudaFree(win_new);
	cudaFree(q_buf); cudaFree(k_buf); cudaFree(v_buf); cudaFree(out_base); cudaFree(out_new);
	cudaFree(st_base); cudaFree(st_new); cudaFree(ret_buf); cudaFree(wg_buf);
	cudaFree(d_state_idx); cudaFree(d_seq_begin);
}

static void BenchSharedAndSmallN(cudaStream_t stream, const char *label, uint32_t hidden, uint32_t shared_rows, uint32_t ba_n, uint32_t rows)
{
	const uint16_t *act = Tensor<uint16_t>((uint64_t)rows * hidden, 0.5f, 0.0f);
	const uint16_t *w_sh = Tensor<uint16_t>((uint64_t)2u * shared_rows * hidden, 0.05f, 0.0f);
	const uint16_t *w_ba = Tensor<uint16_t>((uint64_t)ba_n * hidden, 0.05f, 0.0f);
	uint16_t *gu_tmp = 0, *sh_base = 0, *sh_new = 0, *ba_base = 0, *ba_new = 0;
	cudaMalloc(&gu_tmp, (size_t)rows * 2u * shared_rows * sizeof(uint16_t));
	cudaMalloc(&sh_base, (size_t)rows * shared_rows * sizeof(uint16_t));
	cudaMalloc(&sh_new, (size_t)rows * shared_rows * sizeof(uint16_t));
	cudaMalloc(&ba_base, (size_t)rows * ba_n * sizeof(uint16_t));
	cudaMalloc(&ba_new, (size_t)rows * ba_n * sizeof(uint16_t));

	constexpr uint32_t sm_small32 = QwenBf16SmallNMmaWideSharedBytes<32u>();
	constexpr uint32_t sm_sh32 = QwenSharedGateUpSwigluWideSharedBytes<32u>();
	SparkGpuFuncSetAttribute((const void *)QwenBf16SmallNMmaWideKernel<32u>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sm_small32);
	SparkGpuFuncSetAttribute((const void *)QwenSharedGateUpSwigluWideKernel<32u>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sm_sh32);

	const bool use_wide_ba = rows > 16u && (ba_n >= 16u || rows >= 1024u);
	auto run_sh_base = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaKernel), ((rows + 15u) / 16u) * ((2u * shared_rows + 15u) / 16u), 128u, 0, stream,
			w_sh, act, gu_tmp, rows, hidden, 2u * shared_rows);
		(void)LM_LAUNCH_DEPENDENT((QwenSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, gu_tmp, sh_base, shared_rows);
	};
	auto run_sh_new = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenSharedGateUpSwigluWideKernel<32u>), ((rows + 31u) / 32u) * ((shared_rows + 15u) / 16u), 128u, sm_sh32, stream,
			w_sh, act, sh_new, rows, hidden, shared_rows);
	};
	auto run_ba_base = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaKernel), ((rows + 15u) / 16u) * ((ba_n + 15u) / 16u), 128u, 0, stream,
			w_ba, act, ba_base, rows, hidden, ba_n);
	};
	auto run_ba_new = [&]() {
		if ( use_wide_ba )
			(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaWideKernel<32u>), ((rows + 31u) / 32u) * ((ba_n + 15u) / 16u), 128u, sm_small32, stream,
				w_ba, act, ba_new, rows, hidden, ba_n);
		else
			(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaKernel), ((rows + 15u) / 16u) * ((ba_n + 15u) / 16u), 128u, 0, stream,
				w_ba, act, ba_new, rows, hidden, ba_n);
	};

	run_sh_base(); run_sh_new(); run_ba_base(); run_ba_new();
	cudaStreamSynchronize(stream);
	std::vector<uint16_t> sb(rows * shared_rows), sn(rows * shared_rows), bb(rows * ba_n), bn(rows * ba_n);
	cudaMemcpy(sb.data(), sh_base, sb.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(sn.data(), sh_new, sn.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(bb.data(), ba_base, bb.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(bn.data(), ba_new, bn.size() * 2u, cudaMemcpyDeviceToHost);
	const uint32_t sh_d = CountBf16Diffs(sb, sn), ba_d = CountBf16Diffs(bb, bn);
	Require(sh_d == 0u && ba_d == 0u, label, "Shared/SmallN 0-ULP vs BASE", sh_d, ba_d, rows);

	const float sh_bms = TimeMs(stream, run_sh_base), sh_nms = TimeMs(stream, run_sh_new);
	const float ba_bms = TimeMs(stream, run_ba_base), ba_nms = TimeMs(stream, run_ba_new);
	printf("  SharedGU  %-12s rows=%4u H=%4u S=%3u | BASE=%7.3f ms NEW=%7.3f ms speedup=%5.2fx | diffs=%u\n",
		label, rows, hidden, shared_rows, sh_bms, sh_nms, sh_nms > 0.0f ? sh_bms / sh_nms : 0.0f, sh_d);
	printf("  SmallN    %-12s rows=%4u H=%4u N=%3u | BASE=%7.3f ms NEW=%7.3f ms speedup=%5.2fx | diffs=%u%s\n",
		label, rows, hidden, ba_n, ba_bms, ba_nms, ba_nms > 0.0f ? ba_bms / ba_nms : 0.0f, ba_d, use_wide_ba ? "" : " (gated->base)");

	cudaFree((void *)act); cudaFree((void *)w_sh); cudaFree((void *)w_ba);
	cudaFree(gu_tmp); cudaFree(sh_base); cudaFree(sh_new); cudaFree(ba_base); cudaFree(ba_new);
}

static void BenchIndexScoreMma(cudaStream_t stream, uint32_t rows, uint32_t total_blocks)
{
	using G = Qwen38FlashGeometry<16u, SPARK_WEIGHT_CODEC_NVFP4_E2M1, SPARK_WEIGHT_CODEC_FP8_E4M3>;
	constexpr uint32_t D = G::kIndexDim, H = G::kIndexHeads, DEGREE = 16u;
	const uint32_t max_local = (total_blocks + DEGREE - 1u) / DEGREE;
	const uint16_t *keys = Tensor<uint16_t>((uint64_t)DEGREE * max_local * D, 0.3f, 0.0f);
	const uint16_t *query = Tensor<uint16_t>((uint64_t)rows * H * D, 0.3f, 0.0f);
	std::vector<uint32_t> h_pos(rows);
	const uint32_t start_pos = total_blocks * G::kIndexRatio - rows;
	for ( uint32_t r = 0u; r < rows; ++r )
		h_pos[r] = start_pos + r;
	uint32_t *d_pos = 0;
	float *sc_base = 0, *sc_new = 0;
	cudaMalloc(&d_pos, rows * sizeof(uint32_t));
	cudaMalloc(&sc_base, (size_t)rows * total_blocks * sizeof(float));
	cudaMalloc(&sc_new, (size_t)rows * total_blocks * sizeof(float));
	cudaMemcpy(d_pos, h_pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemset(sc_base, 0, (size_t)rows * total_blocks * sizeof(float));
	cudaMemset(sc_new, 0, (size_t)rows * total_blocks * sizeof(float));

	const float scale = 1.0f / sqrtf((float)D);
	constexpr uint32_t idx_smem = 2u * (64u + H * 32u) * 128u;
	SparkGpuFuncSetAttribute((const void *)QwenIndexGatheredScoreMmaKernel<G>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)idx_smem);

	auto run_base = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenIndexGatheredScoreKernel<G>), dim3((total_blocks + QH_THREADS - 1u) / QH_THREADS, rows), QH_THREADS, 0, stream,
			keys, max_local, DEGREE, query, d_pos, 0u, rows, total_blocks, sc_base, scale);
	};
	auto run_new = [&]() {
		(void)LM_LAUNCH_DEPENDENT((QwenIndexGatheredScoreMmaKernel<G>), dim3((total_blocks + 63u) / 64u, (rows + 31u) / 32u), 256u, idx_smem, stream,
			keys, max_local, DEGREE, query, d_pos, 0u, rows, total_blocks, sc_new, scale);
	};
	run_base();
	run_new();
	cudaStreamSynchronize(stream);
	std::vector<float> hb((size_t)rows * total_blocks), hn((size_t)rows * total_blocks);
	cudaMemcpy(hb.data(), sc_base, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hn.data(), sc_new, hn.size() * sizeof(float), cudaMemcpyDeviceToHost);
	float max_abs = 0.0f;
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		const uint32_t blocks = (h_pos[r] + 1u) / G::kIndexRatio;
		if ( blocks <= QwenHybridShape<G>::kIndexBlocks )
			continue;
		for ( uint32_t b = 0u; b < blocks; ++b )
		{
			const float d = fabsf(hb[(size_t)r * total_blocks + b] - hn[(size_t)r * total_blocks + b]);
			if ( d > max_abs )
				max_abs = d;
		}
	}
	Require(max_abs < 5e-3f, "flash-indexer", "IndexScore MMA accuracy vs BASE", rows, total_blocks, (uint32_t)(max_abs * 1e6f));

	const float base_ms = TimeMs(stream, run_base);
	const float new_ms = TimeMs(stream, run_new);
	printf("  IdxScore  flash-tp16   rows=%4u blks=%4u | BASE=%7.3f ms NEW=%7.3f ms speedup=%5.2fx | maxAbs=%.2e\n",
		rows, total_blocks, base_ms, new_ms, new_ms > 0.0f ? base_ms / new_ms : 0.0f, max_abs);

	cudaFree((void *)keys); cudaFree((void *)query); cudaFree(d_pos); cudaFree(sc_base); cudaFree(sc_new);
}

int main(void)
{
	int dev = 0;
	cudaDeviceProp prop;
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&prop, dev);
	const uint32_t sms = (uint32_t)prop.multiProcessorCount;
	cudaStream_t stream = 0;
	cudaStreamCreate(&stream);

	printf("=== QWEN38 BLACKWELL P4 PREFILL PAIRED BENCHMARK (GPU=%s, SM=%d.%d, SMs=%u) ===\n",
		prop.name, prop.major, prop.minor, sms);

	printf("\n--- [1] GDN Prefill (Conv2D + Scalar Chunk-32 vs SplitTiled + GateOnly + Chunk-16) ---\n");
	BenchGdnPrefill(stream, "qwen3.8-27b-tp1", 16u, 48u, 512u);
	BenchGdnPrefill(stream, "qwen3.8-27b-tp1", 16u, 48u, 1024u);
	BenchGdnPrefill(stream, "qwen3.8-max-tp16", 1u, 8u, 128u);
	BenchGdnPrefill(stream, "qwen3.8-max-tp16", 1u, 8u, 512u);
	BenchGdnPrefill(stream, "qwen3.8-max-tp16", 1u, 8u, 1024u);
	BenchGdnPrefill(stream, "qwen3.8-flash-tp16", 1u, 3u, 128u);
	BenchGdnPrefill(stream, "qwen3.8-flash-tp16", 1u, 3u, 512u);
	BenchGdnPrefill(stream, "qwen3.8-flash-tp16", 1u, 3u, 1024u);

	printf("\n--- [2] Shared GateUp SwiGLU & Wide Small-N MMA (0-ULP Exact) ---\n");
	BenchSharedAndSmallN(stream, "max-tp16", 8192u, 128u, 16u, 512u);
	BenchSharedAndSmallN(stream, "max-tp16", 8192u, 128u, 16u, 1024u);
	BenchSharedAndSmallN(stream, "flash-tp16", 2560u, 64u, 6u, 512u);
	BenchSharedAndSmallN(stream, "flash-tp16", 2560u, 64u, 6u, 1024u);

	printf("\n--- [3] Compact Indexer Score (Tensor-Core MMA vs Scalar) ---\n");
	BenchIndexScoreMma(stream, 1024u, 2048u);
	BenchIndexScoreMma(stream, 1024u, 8192u);

	cudaStreamDestroy(stream);
	if ( vf_failures != 0 )
	{
		printf("\nFAIL qwen38 P4 Blackwell bench: %d failures\n", vf_failures);
		return 1;
	}
	printf("\nPASS qwen38 P4 Blackwell bench: all paired BASE vs NEW correctness, 0-ULP, and Gate GX checks passed!\n");
	return 0;
}
