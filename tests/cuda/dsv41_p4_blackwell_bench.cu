#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/cell_mx.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mx_gemm.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_kernels.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_attend.cuh"

#define BENCH_CHECK(call) do { \
	cudaError_t _e = (call); \
	if ( _e != cudaSuccess ) { \
		fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
		exit(1); \
	} \
} while ( 0 )

#define BENCH_REQ(cond, msg) do { \
	if ( !(cond) ) { \
		fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
		exit(1); \
	} \
} while ( 0 )

static __global__ void BenchFillKernel(uint8_t *dst, uint64_t bytes, uint64_t seed, uint32_t kind)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t h = (seed ^ (i >> (kind == 1u ? 1u : 0u))) + 0x9e3779b97f4a7c15ull;
		h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
		h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
		h ^= h >> 31;
		if ( kind == 1u )
		{
			uint16_t v = (uint16_t)((h & 0x8000u) | ((120u + (uint32_t)((h >> 16) % 7u)) << 7) | ((h >> 24) & 0x7fu));
			dst[i] = (uint8_t)(v >> (8u * (i & 1u)));
		}
		else if ( kind == 2u )
			dst[i] = (uint8_t)((h & 0x3eu) | 0x20u);
		else if ( kind == 3u )
			dst[i] = (uint8_t)(124u + (h & 3u));
		else
			dst[i] = (i % 272u) >= 256u ? (uint8_t)(124u + (h & 3u)) : (uint8_t)(h & 0xffu);
	}
}

static void *BenchAllocFilled(uint64_t bytes, uint32_t kind, uint64_t seed)
{
	void *p = 0;
	BENCH_CHECK(cudaMalloc(&p, bytes));
	BenchFillKernel<<<256, 256>>>((uint8_t *)p, bytes, seed, kind);
	BENCH_CHECK(cudaGetLastError());
	return p;
}

template<class FnA, class FnB>
static void BenchPairedMedianUs(FnA &&fn_a, FnB &&fn_b, int warmup, int iters, double *med_a, double *med_b)
{
	for ( int i = 0; i < warmup; ++i )
	{
		fn_a();
		fn_b();
	}
	BENCH_CHECK(cudaDeviceSynchronize());
	std::vector<double> sa((size_t)iters), sb((size_t)iters);
	cudaEvent_t e0, e1, e2;
	BENCH_CHECK(cudaEventCreate(&e0));
	BENCH_CHECK(cudaEventCreate(&e1));
	BENCH_CHECK(cudaEventCreate(&e2));
	for ( int i = 0; i < iters; ++i )
	{
		float ms_a = 0.0f, ms_b = 0.0f;
		if ( (i & 1) == 0 )
		{
			BENCH_CHECK(cudaEventRecord(e0, 0));
			fn_a();
			BENCH_CHECK(cudaEventRecord(e1, 0));
			fn_b();
			BENCH_CHECK(cudaEventRecord(e2, 0));
			BENCH_CHECK(cudaEventSynchronize(e2));
			BENCH_CHECK(cudaEventElapsedTime(&ms_a, e0, e1));
			BENCH_CHECK(cudaEventElapsedTime(&ms_b, e1, e2));
		}
		else
		{
			BENCH_CHECK(cudaEventRecord(e0, 0));
			fn_b();
			BENCH_CHECK(cudaEventRecord(e1, 0));
			fn_a();
			BENCH_CHECK(cudaEventRecord(e2, 0));
			BENCH_CHECK(cudaEventSynchronize(e2));
			BENCH_CHECK(cudaEventElapsedTime(&ms_b, e0, e1));
			BENCH_CHECK(cudaEventElapsedTime(&ms_a, e1, e2));
		}
		sa[(size_t)i] = (double)ms_a * 1000.0;
		sb[(size_t)i] = (double)ms_b * 1000.0;
	}
	BENCH_CHECK(cudaEventDestroy(e0));
	BENCH_CHECK(cudaEventDestroy(e1));
	BENCH_CHECK(cudaEventDestroy(e2));
	std::sort(sa.begin(), sa.end());
	std::sort(sb.begin(), sb.end());
	*med_a = sa[(size_t)(iters / 2)];
	*med_b = sb[(size_t)(iters / 2)];
}

static void BenchAttend(void)
{
	using Kv2 = LmKvGeometry<1280u, 256u, true>;
	const uint32_t dim = 512u, part_heads = 16u, group_heads = 8u, window = 128u, topk = 2048u;
	const uint32_t max_rows = 2048u, ctx_keys = 4096u, pages_per_seq = (ctx_keys + 255u) / 256u;
	const float qk_scale = 1.0f / sqrtf((float)dim);
	float *dfreq = 0;
	BENCH_CHECK(cudaMalloc(&dfreq, 2u * D41K_ROPE_PAIRS * sizeof(float)));
	BENCH_REQ(D41LaunchFrequencies(dfreq, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0) == SPARK_GPU_SUCCESS, "freq");
	const float *dfreq_c = dfreq + D41K_ROPE_PAIRS;

	uint8_t *dkv = (uint8_t *)BenchAllocFilled((uint64_t)pages_per_seq * 256u * 1280u, 1u, 11ull);
	std::vector<uint32_t> hpt(pages_per_seq);
	for ( uint32_t p = 0u; p < pages_per_seq; ++p ) hpt[p] = p;
	uint32_t *dpt = 0;
	BENCH_CHECK(cudaMalloc(&dpt, hpt.size() * sizeof(uint32_t)));
	BENCH_CHECK(cudaMemcpy(dpt, hpt.data(), hpt.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	LmKvShardView view;
	memset(&view, 0, sizeof(view));
	view.pages.pool = dkv;
	view.pages.page_table = dpt;
	view.pages.page_table_stride = pages_per_seq;
	view.pages.sequence_count = 1u;
	view.pages.pool_page_count = pages_per_seq;
	view.shard.degree = 1u;
	view.shard.rank = 0u;
	view.shard.grain = 8u;

	const uint32_t parts = 4u;
	const uint64_t part_width = (uint64_t)part_heads * dim;
	const uint64_t max_q_elems = (uint64_t)parts * max_rows * part_width;
	uint16_t *dq = (uint16_t *)BenchAllocFilled(max_q_elems * sizeof(uint16_t), 1u, 22ull);
	uint16_t *dwkv = (uint16_t *)BenchAllocFilled((uint64_t)max_rows * dim * sizeof(uint16_t), 1u, 33ull);
	uint16_t *dring = (uint16_t *)BenchAllocFilled((uint64_t)window * dim * sizeof(uint16_t), 1u, 44ull);
	std::vector<float> hsink(parts * part_heads);
	for ( uint32_t i = 0u; i < hsink.size(); ++i ) hsink[i] = -1.0f + 0.05f * (float)(i % 32u);
	float *dsink = 0;
	BENCH_CHECK(cudaMalloc(&dsink, hsink.size() * sizeof(float)));
	BENCH_CHECK(cudaMemcpy(dsink, hsink.data(), hsink.size() * sizeof(float), cudaMemcpyHostToDevice));

	std::vector<uint32_t> hpos(max_rows), hseq(max_rows, 0u), hlen(1u, ctx_keys), hbound(max_rows, ctx_keys - 1u);
	std::vector<uint32_t> hlane(max_rows, 0u), hbeg = { 0u, max_rows }, hidx(max_rows), hslot = { 0u };
	std::vector<uint32_t> hsel((size_t)max_rows * topk);
	for ( uint32_t r = 0u; r < max_rows; ++r )
	{
		hpos[r] = 5632u + r;
		hidx[r] = r;
		for ( uint32_t k = 0u; k < topk; ++k )
			hsel[(size_t)r * topk + k] = (k * 2u + (r & 1u)) % ctx_keys;
	}
	uint32_t *dpos = 0, *dseq = 0, *dlen = 0, *dbnd = 0, *dlane = 0, *dbeg = 0, *didx = 0, *dslot = 0, *dsel = 0;
	BENCH_CHECK(cudaMalloc(&dpos, max_rows * 4u));
	BENCH_CHECK(cudaMalloc(&dseq, max_rows * 4u));
	BENCH_CHECK(cudaMalloc(&dlen, 4u));
	BENCH_CHECK(cudaMalloc(&dbnd, max_rows * 4u));
	BENCH_CHECK(cudaMalloc(&dlane, max_rows * 4u));
	BENCH_CHECK(cudaMalloc(&dbeg, 2u * 4u));
	BENCH_CHECK(cudaMalloc(&didx, max_rows * 4u));
	BENCH_CHECK(cudaMalloc(&dslot, 4u));
	BENCH_CHECK(cudaMalloc(&dsel, (size_t)max_rows * topk * 4u));
	BENCH_CHECK(cudaMemcpy(dpos, hpos.data(), max_rows * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dseq, hseq.data(), max_rows * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dlen, hlen.data(), 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dbnd, hbound.data(), max_rows * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dlane, hlane.data(), max_rows * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dbeg, hbeg.data(), 2u * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(didx, hidx.data(), max_rows * 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dslot, hslot.data(), 4u, cudaMemcpyHostToDevice));
	BENCH_CHECK(cudaMemcpy(dsel, hsel.data(), (size_t)max_rows * topk * 4u, cudaMemcpyHostToDevice));

	float2 *drsc_f = 0, *drsc_c = 0;
	BENCH_CHECK(cudaMalloc(&drsc_f, max_rows * D41K_ROPE_PAIRS * sizeof(float2)));
	BENCH_CHECK(cudaMalloc(&drsc_c, max_rows * D41K_ROPE_PAIRS * sizeof(float2)));
	D41RopeTableKernel<<<max_rows, D41K_ROPE_PAIRS>>>(dpos, dfreq, drsc_f, drsc_c, max_rows);
	BENCH_CHECK(cudaDeviceSynchronize());

	const uint64_t max_part_codes = (uint64_t)max_rows * part_heads * dim;
	const uint64_t max_part_bytes = max_part_codes + max_part_codes / 32u;
	uint8_t *dc_base = 0, *dc_new = 0;
	BENCH_CHECK(cudaMalloc(&dc_base, (size_t)parts * max_part_bytes));
	BENCH_CHECK(cudaMalloc(&dc_new, (size_t)parts * max_part_bytes));

	printf("=== BENCH 1: D41AttendKernel TP4 (16 heads, QTILES=1) ===\n");
	for ( uint32_t rows : { 512u, 2048u } )
	{
		const uint64_t bytes = (uint64_t)rows * part_heads * dim + (uint64_t)rows * part_heads * dim / 32u;
		BENCH_CHECK(cudaMemset(dc_base, 0, bytes));
		BENCH_CHECK(cudaMemset(dc_new, 0, bytes));
		auto run_base = [&]() {
			BENCH_REQ((D41AttendLaunch<Kv2, 512u, 1u, false>(view, 1u, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring,
				(uint64_t)window * dim, window, dpos, dlane, dbeg, didx, dslot, dsink, qk_scale, (uint16_t *)0, dfreq_c,
				dim - 64u, group_heads, dc_base, rows, rows, 0u, 0)) == SPARK_GPU_SUCCESS, "tp4 base");
		};
		auto run_new = [&]() {
			BENCH_REQ((D41AttendLaunch<Kv2, 512u, 1u, true>(view, 1u, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring,
				(uint64_t)window * dim, window, dpos, dlane, dbeg, didx, dslot, dsink, qk_scale, (uint16_t *)0, dfreq_c,
				dim - 64u, group_heads, dc_new, rows, rows, 0u, 0, 0u, 0u, 0u, 0, 0, 0, drsc_c, 1u)) == SPARK_GPU_SUCCESS, "tp4 new");
		};
		run_base();
		run_new();
		BENCH_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> hb(bytes), hn(bytes);
		BENCH_CHECK(cudaMemcpy(hb.data(), dc_base, bytes, cudaMemcpyDeviceToHost));
		BENCH_CHECK(cudaMemcpy(hn.data(), dc_new, bytes, cudaMemcpyDeviceToHost));
		uint64_t diffs = 0u;
		for ( uint64_t i = 0u; i < bytes; ++i ) diffs += hb[i] != hn[i] ? 1u : 0u;
		BENCH_REQ(diffs == 0u, "TP4 D41AttendKernel byte mismatch");
		double base_us = 0.0, new_us = 0.0;
		BenchPairedMedianUs(run_base, run_new, 3, 21, &base_us, &new_us);
		printf("TP4 rows=%4u: BASE=%.1f us  NEW=%.1f us  speedup=%.3fx  diffs=%llu\n",
			rows, base_us, new_us, base_us / new_us, (unsigned long long)diffs);
	}

	printf("=== BENCH 2: D41AttendKernel ROWS16 (64 heads, 4 parts: warm-8K & cold prefill) ===\n");
	for ( uint32_t live : { 8u, 16u, 24u, 128u } )
	{
		const uint32_t own = live;
		const uint64_t part_codes = (uint64_t)own * part_heads * dim;
		const uint64_t part_bytes = part_codes + part_codes / 32u;
		const uint64_t total_bytes = (uint64_t)parts * part_bytes;
		const uint64_t q_stride = (uint64_t)own * part_width;
		BENCH_CHECK(cudaMemset(dc_base, 0, total_bytes));
		BENCH_CHECK(cudaMemset(dc_new, 0, total_bytes));
		auto run_base = [&]() {
			for ( uint32_t p = 0u; p < parts; p += 2u )
				BENCH_REQ((D41AttendLaunch<Kv2, 512u, 2u, false>(view, 1u, dq + (uint64_t)p * q_stride, dseq, dlen, dbnd, dsel, topk,
					dwkv, dring, (uint64_t)window * dim, window, dpos, dlane, dbeg, didx, dslot, dsink + p * part_heads, qk_scale,
					(uint16_t *)0, dfreq_c, dim - 64u, group_heads, dc_base + (uint64_t)p * part_bytes, own, live, 0u, 0,
					q_stride, 0u, part_bytes)) == SPARK_GPU_SUCCESS, "r16 base");
		};
		auto run_new = [&]() {
			if ( live <= 12u )
				BENCH_REQ((D41AttendLaunch<Kv2, 512u, 1u, true>(view, 1u, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring,
					(uint64_t)window * dim, window, dpos, dlane, dbeg, didx, dslot, dsink, qk_scale, (uint16_t *)0, dfreq_c,
					dim - 64u, group_heads, dc_new, own, live, 0u, 0, q_stride, 0u, part_bytes, 0, 0, 0, drsc_c, parts)) == SPARK_GPU_SUCCESS, "r16 new q1");
			else
				BENCH_REQ((D41AttendLaunch<Kv2, 512u, 2u, true>(view, 1u, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring,
					(uint64_t)window * dim, window, dpos, dlane, dbeg, didx, dslot, dsink, qk_scale, (uint16_t *)0, dfreq_c,
					dim - 64u, group_heads, dc_new, own, live, 0u, 0, q_stride, 0u, part_bytes, 0, 0, 0, drsc_c, 2u)) == SPARK_GPU_SUCCESS, "r16 new q2");
		};
		run_base();
		run_new();
		BENCH_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> hb(total_bytes), hn(total_bytes);
		BENCH_CHECK(cudaMemcpy(hb.data(), dc_base, total_bytes, cudaMemcpyDeviceToHost));
		BENCH_CHECK(cudaMemcpy(hn.data(), dc_new, total_bytes, cudaMemcpyDeviceToHost));
		uint64_t diffs = 0u;
		for ( uint64_t i = 0u; i < total_bytes; ++i ) diffs += hb[i] != hn[i] ? 1u : 0u;
		BENCH_REQ(diffs == 0u, "ROWS16 D41AttendKernel byte mismatch");
		double base_us = 0.0, new_us = 0.0;
		BenchPairedMedianUs(run_base, run_new, 3, 25, &base_us, &new_us);
		printf("ROWS16 live=%3u (wave~%4u): BASE=%.1f us  NEW=%.1f us  speedup=%.3fx  diffs=%llu\n",
			live, live * 16u, base_us, new_us, base_us / new_us, (unsigned long long)diffs);
	}

	cudaFree(dfreq); cudaFree(dkv); cudaFree(dpt); cudaFree(dq); cudaFree(dwkv); cudaFree(dring); cudaFree(dsink);
	cudaFree(dpos); cudaFree(dseq); cudaFree(dlen); cudaFree(dbnd); cudaFree(dlane); cudaFree(dbeg); cudaFree(didx);
	cudaFree(dslot); cudaFree(dsel); cudaFree(drsc_f); cudaFree(drsc_c); cudaFree(dc_base); cudaFree(dc_new);
}

static void BenchTp4AndRows16Moe(void)
{
	const uint32_t in_dim = 5120u, tp4_w13_out = 1024u, r16_w13_out = 4096u;
	const uint32_t tp4_experts = 384u, r16_groups = 48u;
	const uint64_t tp4_group_bytes = (uint64_t)(in_dim / 32u) * (tp4_w13_out / 16u) * 272u;
	const uint64_t r16_group_bytes = (uint64_t)(in_dim / 32u) * (r16_w13_out / 16u) * 272u;

	uint8_t *w_tp4 = (uint8_t *)BenchAllocFilled((uint64_t)tp4_experts * tp4_group_bytes, 0u, 101ull);
	uint8_t *act_codes = (uint8_t *)BenchAllocFilled((uint64_t)2048u * in_dim, 2u, 102ull);
	uint8_t *act_scales = (uint8_t *)BenchAllocFilled((uint64_t)2048u * (in_dim / 32u), 3u, 103ull);
	uint16_t *out_a = 0, *out_b = 0;
	BENCH_CHECK(cudaMalloc(&out_a, (uint64_t)12288u * r16_w13_out * sizeof(uint16_t)));
	BENCH_CHECK(cudaMalloc(&out_b, (uint64_t)12288u * r16_w13_out * sizeof(uint16_t)));
	uint32_t *doff = 0, *dsrc = 0, *dact = 0;
	BENCH_CHECK(cudaMalloc(&doff, (tp4_experts + 1u) * 4u));
	BENCH_CHECK(cudaMalloc(&dsrc, 12288u * 4u));
	BENCH_CHECK(cudaMalloc(&dact, (tp4_experts + 1u) * 4u));

	printf("=== BENCH 3: TP4 Prefill W13 Forensic (025f1c373 Direct vs LmCellMxExperts) ===\n");
	for ( uint32_t rows : { 512u, 2048u } )
	{
		const uint32_t routes = rows * 6u;
		std::vector<uint32_t> hoff(tp4_experts + 1u, 0u), hsrc(routes);
		for ( uint32_t e = 0u; e <= tp4_experts; ++e ) hoff[e] = (uint32_t)((uint64_t)e * routes / tp4_experts);
		for ( uint32_t i = 0u; i < routes; ++i ) hsrc[i] = (i * 131u) % rows;
		BENCH_CHECK(cudaMemcpy(doff, hoff.data(), (tp4_experts + 1u) * 4u, cudaMemcpyHostToDevice));
		BENCH_CHECK(cudaMemcpy(dsrc, hsrc.data(), routes * 4u, cudaMemcpyHostToDevice));
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff, dact, tp4_experts);
		BENCH_CHECK(cudaDeviceSynchronize());
		auto run_direct = [&]() {
			D41CellMxW13DirectKernel<2u, 2u, 384u, 5120u, 1024u><<<384u * (1024u / 128u), 256u>>>(
				w_tp4, act_codes, act_scales, out_a, doff, dsrc, dact, routes);
		};
		auto run_cellmx = [&]() {
			BENCH_REQ(LmCellMxExperts(w_tp4, act_codes, act_scales, out_b, doff, dsrc, tp4_experts, routes, 0u, in_dim, tp4_w13_out, 0) == LM_LAUNCH_OK, "tp4 cellmx");
		};
		double dir_us = 0.0, cmx_us = 0.0;
		BenchPairedMedianUs(run_direct, run_cellmx, 3, 21, &dir_us, &cmx_us);
		printf("TP4 W13 rows=%4u routes=%5u: Direct<2,2>=%.1f us  LmCellMxExperts=%.1f us  LmCellMx_gain=%.3fx\n",
			rows, routes, dir_us, cmx_us, dir_us / cmx_us);
	}
	cudaFree(w_tp4);

	uint8_t *w_r16 = (uint8_t *)BenchAllocFilled((uint64_t)r16_groups * r16_group_bytes, 0u, 201ull);
	printf("=== BENCH 4: ROWS16 MoE Split W13 (groups=48: total routes vs local_routes) ===\n");
	for ( uint32_t rows : { 64u, 129u, 244u } )
	{
		const uint32_t total_routes = rows * 6u;
		const uint32_t local_routes = (total_routes + 15u) / 16u;
		std::vector<uint32_t> hoff(r16_groups + 1u, 0u), hsrc(local_routes);
		for ( uint32_t g = 0u; g <= r16_groups; ++g ) hoff[g] = (uint32_t)((uint64_t)g * local_routes / r16_groups);
		for ( uint32_t i = 0u; i < local_routes; ++i ) hsrc[i] = (i * 37u) % rows;
		BENCH_CHECK(cudaMemcpy(doff, hoff.data(), (r16_groups + 1u) * 4u, cudaMemcpyHostToDevice));
		BENCH_CHECK(cudaMemcpy(dsrc, hsrc.data(), local_routes * 4u, cudaMemcpyHostToDevice));
		const LmCellMxCodes mx = LmCellMxPacked(act_codes, act_scales, in_dim);
		auto run_base = [&]() {
			BENCH_REQ(LmCellMxExpertsSplit(w_r16, 0, 0u, mx, out_a, doff, dsrc, r16_groups, total_routes, 0u, in_dim, r16_w13_out, 0) == LM_LAUNCH_OK, "r16 moe base");
		};
		auto run_new = [&]() {
			BENCH_REQ(LmCellMxExpertsSplit(w_r16, 0, 0u, mx, out_b, doff, dsrc, r16_groups, local_routes, 0u, in_dim, r16_w13_out, 0) == LM_LAUNCH_OK, "r16 moe new");
		};
		run_base();
		run_new();
		BENCH_CHECK(cudaDeviceSynchronize());
		const size_t elems = (size_t)local_routes * r16_w13_out;
		std::vector<uint16_t> ha(elems), hb(elems);
		BENCH_CHECK(cudaMemcpy(ha.data(), out_a, elems * 2u, cudaMemcpyDeviceToHost));
		BENCH_CHECK(cudaMemcpy(hb.data(), out_b, elems * 2u, cudaMemcpyDeviceToHost));
		uint64_t diffs = 0u;
		for ( size_t i = 0u; i < elems; ++i ) diffs += ha[i] != hb[i] ? 1u : 0u;
		BENCH_REQ(diffs == 0u, "ROWS16 MoE wide vs standard mismatch");
		double base_us = 0.0, new_us = 0.0;
		BenchPairedMedianUs(run_base, run_new, 3, 25, &base_us, &new_us);
		printf("ROWS16 W13 rows=%3u local_routes=%3u: BASE(routes=%4u)=%.1f us  NEW(local=%3u)=%.1f us  speedup=%.3fx  diffs=%llu\n",
			rows, local_routes, total_routes, base_us, local_routes, new_us, base_us / new_us, (unsigned long long)diffs);
	}
	cudaFree(w_r16); cudaFree(act_codes); cudaFree(act_scales); cudaFree(out_a); cudaFree(out_b);
	cudaFree(doff); cudaFree(dsrc); cudaFree(dact);
}

int main(void)
{
	BenchAttend();
	BenchTp4AndRows16Moe();
	printf("PASS dsv41_p4_blackwell_bench\n");
	return 0;
}
