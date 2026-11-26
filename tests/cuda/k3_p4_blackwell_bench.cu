#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "model-families/k3/source/layer.cuh"

#define BENCH_WARMUP 5u
#define BENCH_ITERS 20u

#define CHECK_CUDA(...) do { cudaError_t err_ = (__VA_ARGS__); if ( err_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #__VA_ARGS__, cudaGetErrorString(err_)); exit(1); } } while (0)
#define CHECK_GPU(...) do { SparkGpuError err_ = (__VA_ARGS__); if ( err_ != SPARK_GPU_SUCCESS ) { printf("FAIL gpu %s: %d\n", #__VA_ARGS__, (int)err_); exit(1); } } while (0)
#define CHECK_OK(...) do { int32_t st_ = (__VA_ARGS__); if ( st_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #__VA_ARGS__, st_); exit(1); } } while (0)

static int g_failures;
static uint32_t g_sms;

static __host__ __device__ __forceinline__ uint64_t Hash64(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static __global__ void FillBf16Kernel(uint16_t *dst, uint64_t count, uint64_t seed, float scale)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
	{
		const float v = (float)((int64_t)(Hash64(idx + seed) % 2001u) - 1000) * scale;
		dst[idx] = LmFloatToBf16(v);
	}
}

static __global__ void FillFloatKernel(float *dst, uint64_t count, uint64_t seed, float scale)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
		dst[idx] = (float)((int64_t)(Hash64(idx + seed) % 2001u) - 1000) * scale;
}

static __global__ void FillCellWeightKernel(uint8_t *dst, uint64_t cells_total, uint64_t seed)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t total_bytes = cells_total * LM_CELL_MMA_CELL_BYTES;
	if ( idx >= total_bytes )
		return;
	const uint64_t cell = idx / LM_CELL_MMA_CELL_BYTES;
	const uint32_t off = (uint32_t)(idx % LM_CELL_MMA_CELL_BYTES);
	const uint64_t h = Hash64(cell * 313u + off + seed);
	dst[idx] = off < 256u ? (uint8_t)h : (uint8_t)(122u + (uint32_t)(h % 7u));
}

static void FillBf16(uint16_t *d_ptr, uint64_t count, uint64_t seed, float scale)
{
	const uint32_t blocks = (uint32_t)((count + 255u) / 256u);
	FillBf16Kernel<<<blocks, 256u>>>(d_ptr, count, seed, scale);
	CHECK_CUDA(cudaGetLastError());
}

static void FillFloat(float *d_ptr, uint64_t count, uint64_t seed, float scale)
{
	const uint32_t blocks = (uint32_t)((count + 255u) / 256u);
	FillFloatKernel<<<blocks, 256u>>>(d_ptr, count, seed, scale);
	CHECK_CUDA(cudaGetLastError());
}

static void FillCellWeight(uint8_t *d_ptr, uint64_t cells_total, uint64_t seed)
{
	const uint64_t bytes = cells_total * LM_CELL_MMA_CELL_BYTES;
	const uint32_t blocks = (uint32_t)((bytes + 255u) / 256u);
	FillCellWeightKernel<<<blocks, 256u>>>(d_ptr, cells_total, seed);
	CHECK_CUDA(cudaGetLastError());
}

static float Bf16Host(uint16_t bits)
{
	uint32_t u = (uint32_t)bits << 16u;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static uint32_t Bf16UlpDiff(uint16_t a, uint16_t b)
{
	if ( a == b )
		return 0u;
	if ( ((a & 0x7fffu) == 0u) && ((b & 0x7fffu) == 0u) )
		return 0u;
	int32_t ia = (a & 0x8000u) ? -(int32_t)(a & 0x7fffu) : (int32_t)(a & 0x7fffu);
	int32_t ib = (b & 0x8000u) ? -(int32_t)(b & 0x7fffu) : (int32_t)(b & 0x7fffu);
	return (uint32_t)(ia > ib ? ia - ib : ib - ia);
}

static void CheckBf16Close(const char *label, const uint16_t *d_ref, const uint16_t *d_got, uint64_t count, float abs_tol, uint32_t max_ulp_allowed)
{
	std::vector<uint16_t> h_ref(count), h_got(count);
	CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_got.data(), d_got, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	float max_abs = 0.0f;
	uint32_t max_ulp = 0u, mismatches = 0u;
	for ( uint64_t i = 0u; i < count; ++i )
	{
		const float r = Bf16Host(h_ref[i]), g = Bf16Host(h_got[i]);
		const float d = fabsf(r - g);
		const uint32_t ulp = Bf16UlpDiff(h_ref[i], h_got[i]);
		if ( !isfinite(g) || (d > abs_tol && ulp > max_ulp_allowed) )
			mismatches++;
		if ( d > max_abs )
			max_abs = d;
		if ( ulp > max_ulp )
			max_ulp = ulp;
	}
	if ( mismatches != 0u )
	{
		printf("FAIL %s: mismatches=%u/%llu max_abs=%.6g max_ulp=%u (tol=%.6g, ulp<=%u)\n",
			label, mismatches, (unsigned long long)count, max_abs, max_ulp, abs_tol, max_ulp_allowed);
		g_failures++;
	}
	else
		printf("PASS %s: max_abs=%.6g max_ulp=%u\n", label, max_abs, max_ulp);
}

static void CheckFloatClose(const char *label, const float *d_ref, const float *d_got, uint64_t count, float abs_tol)
{
	std::vector<float> h_ref(count), h_got(count);
	CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, count * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_got.data(), d_got, count * sizeof(float), cudaMemcpyDeviceToHost));
	float max_abs = 0.0f;
	uint32_t mismatches = 0u;
	for ( uint64_t i = 0u; i < count; ++i )
	{
		const float d = fabsf(h_ref[i] - h_got[i]);
		if ( !isfinite(h_got[i]) || d > abs_tol )
			mismatches++;
		if ( d > max_abs )
			max_abs = d;
	}
	if ( mismatches != 0u )
	{
		printf("FAIL %s: mismatches=%u/%llu max_abs=%.6g (tol=%.6g)\n",
			label, mismatches, (unsigned long long)count, max_abs, abs_tol);
		g_failures++;
	}
	else
		printf("PASS %s: max_abs=%.6g\n", label, max_abs);
}

static void CheckBf16Exact0Ulp(const char *label, const uint16_t *d_a, const uint16_t *d_b, uint64_t count)
{
	std::vector<uint16_t> ha(count), hb(count);
	CHECK_CUDA(cudaMemcpy(ha.data(), d_a, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(hb.data(), d_b, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	uint32_t diffs = 0u;
	for ( uint64_t i = 0u; i < count; ++i )
		if ( ha[i] != hb[i] )
			diffs++;
	if ( diffs != 0u )
	{
		printf("FAIL %s (5c 0-ULP): diffs=%u/%llu\n", label, diffs, (unsigned long long)count);
		g_failures++;
	}
	else
		printf("PASS %s (5c 0-ULP bit-exact across %llu bf16)\n", label, (unsigned long long)count);
}

template<class Fn>
static double TimeUs(Fn &&fn, cudaStream_t stream)
{
	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	for ( uint32_t i = 0u; i < BENCH_WARMUP; ++i )
		fn();
	CHECK_CUDA(cudaStreamSynchronize(stream));
	std::vector<float> samples(BENCH_ITERS);
	for ( uint32_t i = 0u; i < BENCH_ITERS; ++i )
	{
		CHECK_CUDA(cudaEventRecord(ev0, stream));
		fn();
		CHECK_CUDA(cudaEventRecord(ev1, stream));
		CHECK_CUDA(cudaEventSynchronize(ev1));
		CHECK_CUDA(cudaEventElapsedTime(&samples[i], ev0, ev1));
	}
	std::sort(samples.begin(), samples.end());
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
	return (double)samples[BENCH_ITERS / 2u] * 1000.0;
}

static void BuildSyntheticRoutes(uint32_t rows, std::vector<uint32_t> &route_expert, std::vector<float> &route_weight,
	std::vector<uint32_t> &group_row_offset, std::vector<uint32_t> &route_packed_row,
	std::vector<uint32_t> &route_source_token, std::vector<uint32_t> &active_experts)
{
	const uint32_t packed = rows * K3_TOP_K;
	route_expert.resize(packed);
	route_weight.resize(packed);
	group_row_offset.assign(K3_EXPERTS + 1u, 0u);
	route_packed_row.resize(packed);
	route_source_token.resize(packed);
	std::vector<uint32_t> counts(K3_EXPERTS, 0u);
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		uint32_t chosen[K3_TOP_K];
		for ( uint32_t k = 0u; k < K3_TOP_K; ++k )
		{
			uint32_t e = (uint32_t)(Hash64((uint64_t)r * 131u + k * 17u + 19u) % K3_EXPERTS);
			bool dup = true;
			while ( dup )
			{
				dup = false;
				for ( uint32_t j = 0u; j < k; ++j )
					if ( chosen[j] == e ) { e = (e + 1u) % K3_EXPERTS; dup = true; break; }
			}
			chosen[k] = e;
			route_expert[r * K3_TOP_K + k] = e;
			route_weight[r * K3_TOP_K + k] = (float)(1u + (k & 7u)) * 0.04f;
			counts[e]++;
		}
	}
	active_experts.clear();
	active_experts.push_back(0u);
	uint32_t cur = 0u;
	for ( uint32_t e = 0u; e < K3_EXPERTS; ++e )
	{
		group_row_offset[e] = cur;
		cur += counts[e];
		if ( counts[e] > 0u )
			active_experts.push_back(e);
	}
	group_row_offset[K3_EXPERTS] = cur;
	active_experts[0u] = (uint32_t)active_experts.size() - 1u;
	std::vector<uint32_t> cursor = group_row_offset;
	for ( uint32_t i = 0u; i < packed; ++i )
	{
		const uint32_t e = route_expert[i];
		const uint32_t p = cursor[e]++;
		route_packed_row[i] = p;
		route_source_token[p] = i / K3_TOP_K;
	}
}

static void BenchSuite1Moe(cudaStream_t stream)
{
	printf("\n=== Suite 1: Routed MoE MXFP4 (896 experts, H=3584, ch=192, topk=16) ===\n");
	constexpr uint32_t CH = 192u, HALF_CELLS = CH / LM_SKINNY_CELL_NEURONS;
	constexpr uint32_t W1_KT = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_TILE_K;
	constexpr uint32_t W2_KT = CH / LM_SKINNY_CELL_TILE_K;
	constexpr uint32_t W2_CELLS = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_NEURONS;
	constexpr uint64_t W1_GROUP_BYTES = (uint64_t)W1_KT * (HALF_CELLS * 2u) * LM_CELL_MMA_CELL_BYTES;
	constexpr uint64_t W2_GROUP_BYTES = (uint64_t)W2_KT * W2_CELLS * LM_CELL_MMA_CELL_BYTES;
	constexpr uint32_t MAX_ROWS = 2048u, MAX_PACKED = MAX_ROWS * K3_TOP_K;

	uint8_t *d_w1 = 0, *d_w2 = 0;
	uint16_t *d_act = 0, *d_mid_ref = 0, *d_mid_new = 0, *d_gu_ref = 0, *d_gu_new = 0, *d_out_ref = 0, *d_out_new = 0;
	uint32_t *d_goff = 0, *d_re = 0, *d_rsrc = 0, *d_rpack = 0, *d_act_exp = 0;
	float *d_rw = 0;
	CHECK_CUDA(cudaMalloc(&d_w1, (uint64_t)K3_EXPERTS * W1_GROUP_BYTES));
	CHECK_CUDA(cudaMalloc(&d_w2, (uint64_t)K3_EXPERTS * W2_GROUP_BYTES));
	CHECK_CUDA(cudaMalloc(&d_act, (uint64_t)MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_mid_ref, (uint64_t)MAX_PACKED * CH * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_mid_new, (uint64_t)MAX_PACKED * CH * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_gu_ref, (uint64_t)MAX_PACKED * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_gu_new, (uint64_t)MAX_PACKED * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_out_ref, (uint64_t)MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_out_new, (uint64_t)MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
	CHECK_CUDA(cudaMalloc(&d_goff, (K3_EXPERTS + 1u) * sizeof(uint32_t)));
	CHECK_CUDA(cudaMalloc(&d_re, MAX_PACKED * sizeof(uint32_t)));
	CHECK_CUDA(cudaMalloc(&d_rsrc, MAX_PACKED * sizeof(uint32_t)));
	CHECK_CUDA(cudaMalloc(&d_rpack, MAX_PACKED * sizeof(uint32_t)));
	CHECK_CUDA(cudaMalloc(&d_act_exp, (K3_EXPERTS + 1u) * sizeof(uint32_t)));
	CHECK_CUDA(cudaMalloc(&d_rw, MAX_PACKED * sizeof(float)));

	FillCellWeight(d_w1, (uint64_t)K3_EXPERTS * W1_KT * HALF_CELLS * 2u, 11u);
	FillCellWeight(d_w2, (uint64_t)K3_EXPERTS * W2_KT * W2_CELLS, 23u);
	FillBf16(d_act, (uint64_t)MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN, 37u, 0.0005f);
	CHECK_CUDA(cudaDeviceSynchronize());

	{
		std::vector<uint32_t> re1, goff1, rpack1, rsrc1, ae1;
		std::vector<float> rw1;
		BuildSyntheticRoutes(1u, re1, rw1, goff1, rpack1, rsrc1, ae1);
		CHECK_CUDA(cudaMemcpy(d_re, re1.data(), re1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rw, rw1.data(), rw1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_goff, goff1.data(), goff1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rpack, rpack1.data(), rpack1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rsrc, rsrc1.data(), rsrc1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_act_exp, ae1.data(), ae1.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_OK(K3MoeCellsSkinnyFastLaunch(d_w1, d_w2, d_act, d_gu_new, d_mid_new, d_out_new,
			d_goff, d_re, d_rsrc, d_rpack, d_rw, d_act_exp, 1u, K3_TOP_K, CH, (SparkGpuStream)stream));
		CHECK_CUDA(cudaStreamSynchronize(stream));
		uint16_t *d_b1_saved = 0;
		CHECK_CUDA(cudaMalloc(&d_b1_saved, K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t)));
		CHECK_CUDA(cudaMemcpy(d_b1_saved, d_out_new, K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToDevice));

		const uint32_t inv_rows[2] = {4u, 8u};
		for ( uint32_t r : inv_rows )
		{
			std::vector<uint32_t> re, goff, rpack, rsrc, ae;
			std::vector<float> rw;
			BuildSyntheticRoutes(r, re, rw, goff, rpack, rsrc, ae);
			CHECK_CUDA(cudaMemcpy(d_re, re.data(), re.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(d_rw, rw.data(), rw.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(d_goff, goff.data(), goff.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(d_rpack, rpack.data(), rpack.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(d_rsrc, rsrc.data(), rsrc.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(d_act_exp, ae.data(), ae.size() * 4u, cudaMemcpyHostToDevice));
			CHECK_OK(K3MoeCellsSkinnyFastLaunch(d_w1, d_w2, d_act, d_gu_new, d_mid_new, d_out_new,
				d_goff, d_re, d_rsrc, d_rpack, d_rw, d_act_exp, r, r * K3_TOP_K, CH, (SparkGpuStream)stream));
			CHECK_CUDA(cudaStreamSynchronize(stream));
			char tag[64];
			snprintf(tag, sizeof(tag), "MoE Skinny B1==B%u row0", r);
			CheckBf16Exact0Ulp(tag, d_b1_saved, d_out_new, K3_ROUTED_EXPERT_HIDDEN);
		}
		CHECK_CUDA(cudaFree(d_b1_saved));
		const double b1_us = TimeUs([&]() {
			K3CellSkinnyW2FinalizeRow1Kernel<<<W2_CELLS, 128u, 0u, stream>>>(
				d_w2, d_mid_new, d_re, d_rpack, d_rw, d_out_new, W2_GROUP_BYTES, W2_KT, W2_CELLS, CH);
		}, stream);
		printf("BENCH MoE B1 W2FinalizeRow1 (224 CTAs): %.1f us\n", b1_us);
	}

	const uint32_t pf_rows[2] = {1024u, 2048u};
	for ( uint32_t r : pf_rows )
	{
		const uint32_t packed = r * K3_TOP_K;
		const uint32_t max_act = K3_EXPERTS;
		std::vector<uint32_t> re, goff, rpack, rsrc, ae;
		std::vector<float> rw;
		BuildSyntheticRoutes(r, re, rw, goff, rpack, rsrc, ae);
		CHECK_CUDA(cudaMemcpy(d_re, re.data(), re.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rw, rw.data(), rw.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_goff, goff.data(), goff.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rpack, rpack.data(), rpack.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_rsrc, rsrc.data(), rsrc.size() * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_act_exp, ae.data(), ae.size() * 4u, cudaMemcpyHostToDevice));

		(void)SparkGpuFuncSetAttribute((const void *)K3CellMmaW1SituKernel<8u>,
			SPARK_GPU_FUNCTION_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
		auto run_old_all = [&]() {
			K3CellMmaW1SituKernel<8u><<<max_act * (HALF_CELLS / 4u), LM_CELL_MMA_THREADS, 0u, stream>>>(
				d_w1, d_act, d_mid_ref, d_goff, d_rsrc, d_act_exp, W1_GROUP_BYTES, W1_KT, HALF_CELLS, K3_ROUTED_EXPERT_HIDDEN, CH);
			K3CellMmaW2SliceKernel<6u, 3u, true><<<max_act * (W2_CELLS / LM_CELL_MMA_BLOCK_CELLS), LM_CELL_MMA_THREADS, 0u, stream>>>(
				d_w2, d_mid_ref, d_gu_ref, d_goff, d_act_exp, W2_GROUP_BYTES, W2_KT, W2_CELLS, 0u, W2_CELLS, CH);
			K3MoeFinalizeBlockKernel<<<dim3(((K3_ROUTED_EXPERT_HIDDEN / 8u) + 127u) / 128u, r), 128u, 0u, stream>>>(
				d_gu_ref, d_rpack, d_rw, d_out_ref, r, K3_ROUTED_EXPERT_HIDDEN);
		};
		auto run_old_w2 = [&]() {
			K3CellMmaW2SliceKernel<6u, 3u, true><<<max_act * (W2_CELLS / LM_CELL_MMA_BLOCK_CELLS), LM_CELL_MMA_THREADS, 0u, stream>>>(
				d_w2, d_mid_ref, d_gu_ref, d_goff, d_act_exp, W2_GROUP_BYTES, W2_KT, W2_CELLS, 0u, W2_CELLS, CH);
			K3MoeFinalizeBlockKernel<<<dim3(((K3_ROUTED_EXPERT_HIDDEN / 8u) + 127u) / 128u, r), 128u, 0u, stream>>>(
				d_gu_ref, d_rpack, d_rw, d_out_ref, r, K3_ROUTED_EXPERT_HIDDEN);
		};
		auto run_new_w2 = [&]() {
			K3CellMmaW2SliceKernel<4u, 3u, true><<<max_act * (W2_CELLS / LM_CELL_MMA_BLOCK_CELLS), LM_CELL_MMA_THREADS, 0u, stream>>>(
				d_w2, d_mid_ref, d_gu_new, d_goff, d_act_exp, W2_GROUP_BYTES, W2_KT, W2_CELLS, 0u, W2_CELLS, CH);
			K3MoeFinalizeToken224Kernel<<<r, 224u, 0u, stream>>>(d_gu_new, d_rpack, d_rw, d_out_new, r);
		};
		auto run_new_full = [&]() {
			CHECK_OK(K3MoeCellsMmaFastLaunch(d_w1, d_w2, d_act, d_gu_new, d_mid_new, d_out_new,
				d_goff, d_rsrc, d_rpack, d_rw, d_act_exp, r, packed, CH, (SparkGpuStream)stream));
		};
		run_old_all();
		run_new_full();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "MoE PF%u full W1+W2+Finalize", r);
		CheckBf16Exact0Ulp(tag, d_out_ref, d_out_new, (uint64_t)r * K3_ROUTED_EXPERT_HIDDEN);
		const double old_w2 = TimeUs(run_old_w2, stream);
		const double new_w2 = TimeUs(run_new_w2, stream);
		const double old_all = TimeUs(run_old_all, stream);
		const double new_all = TimeUs(run_new_full, stream);
		printf("BENCH MoE PF%u W2+Fin: old_6u3s=%.1f us  new_4u3s_tok224=%.1f us  speedup=%.2fx  (full: old=%.1f us  new=%.1f us  total=%.2fx)\n",
			r, old_w2, new_w2, old_w2 / new_w2, old_all, new_all, old_all / new_all);
	}

	CHECK_CUDA(cudaFree(d_w1)); CHECK_CUDA(cudaFree(d_w2)); CHECK_CUDA(cudaFree(d_act));
	CHECK_CUDA(cudaFree(d_mid_ref)); CHECK_CUDA(cudaFree(d_mid_new));
	CHECK_CUDA(cudaFree(d_gu_ref)); CHECK_CUDA(cudaFree(d_gu_new));
	CHECK_CUDA(cudaFree(d_out_ref)); CHECK_CUDA(cudaFree(d_out_new));
	CHECK_CUDA(cudaFree(d_goff)); CHECK_CUDA(cudaFree(d_re)); CHECK_CUDA(cudaFree(d_rsrc));
	CHECK_CUDA(cudaFree(d_rpack)); CHECK_CUDA(cudaFree(d_act_exp)); CHECK_CUDA(cudaFree(d_rw));
}

static void BenchSuite2Indexer(cudaStream_t stream)
{
	printf("\n=== Suite 2: Indexer Projections (K3IndexHeadMmaSkip & K3IndexQMmaWideSkip) ===\n");
	constexpr uint32_t MAX_ROWS = 2048u;
	uint16_t *d_w_head = 0, *d_w_q = 0, *d_norm = 0, *d_lat = 0;
	uint16_t *d_oh_ref = 0, *d_oh_new = 0, *d_oq_ref = 0, *d_oq_new = 0;
	uint32_t *d_seq = 0, *d_ctx = 0, *d_pos = 0;
	CHECK_CUDA(cudaMalloc(&d_w_head, (uint64_t)K3_INDEX_HEADS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_w_q, (uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK * 2u));
	CHECK_CUDA(cudaMalloc(&d_norm, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_lat, (uint64_t)MAX_ROWS * K3_Q_LORA_RANK * 2u));
	CHECK_CUDA(cudaMalloc(&d_oh_ref, (uint64_t)MAX_ROWS * K3_INDEX_HEADS * 2u));
	CHECK_CUDA(cudaMalloc(&d_oh_new, (uint64_t)MAX_ROWS * K3_INDEX_HEADS * 2u));
	CHECK_CUDA(cudaMalloc(&d_oq_ref, (uint64_t)MAX_ROWS * K3_INDEX_QUERY * 2u));
	CHECK_CUDA(cudaMalloc(&d_oq_new, (uint64_t)MAX_ROWS * K3_INDEX_QUERY * 2u));
	CHECK_CUDA(cudaMalloc(&d_seq, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMalloc(&d_ctx, 4u));
	CHECK_CUDA(cudaMalloc(&d_pos, MAX_ROWS * 4u));

	FillBf16(d_w_head, (uint64_t)K3_INDEX_HEADS * K3_HIDDEN, 101u, 0.0005f);
	FillBf16(d_w_q, (uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK, 103u, 0.0005f);
	FillBf16(d_norm, (uint64_t)MAX_ROWS * K3_HIDDEN, 107u, 0.0005f);
	FillBf16(d_lat, (uint64_t)MAX_ROWS * K3_Q_LORA_RANK, 109u, 0.0005f);
	std::vector<uint32_t> h_seq(MAX_ROWS, 0u), h_pos(MAX_ROWS);
	uint32_t ctx_sparse = 8192u;
	for ( uint32_t i = 0u; i < MAX_ROWS; ++i )
		h_pos[i] = 4096u + i;
	CHECK_CUDA(cudaMemcpy(d_seq, h_seq.data(), MAX_ROWS * 4u, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_ctx, &ctx_sparse, 4u, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_pos, h_pos.data(), MAX_ROWS * 4u, cudaMemcpyHostToDevice));

	const uint32_t test_rows[3] = {64u, 512u, 2048u};
	for ( uint32_t r : test_rows )
	{
		char tag[64];
		if ( r > 64u )
		{
			CHECK_CUDA(cudaMemsetAsync(d_oh_ref, 0, (uint64_t)r * K3_INDEX_HEADS * 2u, stream));
			CHECK_CUDA(cudaMemsetAsync(d_oh_new, 0, (uint64_t)r * K3_INDEX_HEADS * 2u, stream));
			auto old_h = [&]() {
				K3IndexHeadSkipKernel<<<dim3((K3_INDEX_HEADS + 3u) / 4u, r), 128u, 0u, stream>>>(
					d_w_head, d_norm, d_oh_ref, d_seq, d_ctx, d_pos, r);
			};
			auto new_h = [&]() {
				K3IndexHeadMmaSkipKernel<<<(r + 31u) / 32u, 128u, 0u, stream>>>(
					(const uint8_t *)d_w_head, d_norm, d_oh_new, d_seq, d_ctx, d_pos, r);
			};
			old_h(); new_h();
			CHECK_CUDA(cudaStreamSynchronize(stream));
			snprintf(tag, sizeof(tag), "IndexHead MMA vs Scalar R=%u", r);
			CheckBf16Close(tag, d_oh_ref, d_oh_new, (uint64_t)r * K3_INDEX_HEADS, 0.02f, 2u);
			const double t_old_h = TimeUs(old_h, stream);
			const double t_new_h = TimeUs(new_h, stream);
			printf("BENCH IndexHead R=%u: old_scalar=%.1f us  new_mma=%.1f us  speedup=%.2fx\n", r, t_old_h, t_new_h, t_old_h / t_new_h);
		}

		CHECK_CUDA(cudaMemsetAsync(d_oq_ref, 0, (uint64_t)r * K3_INDEX_QUERY * 2u, stream));
		CHECK_CUDA(cudaMemsetAsync(d_oq_new, 0, (uint64_t)r * K3_INDEX_QUERY * 2u, stream));
		auto old_q = [&]() {
			K3IndexQMmaSkipKernel<<<dim3(K3_INDEX_QUERY / 64u, (r + 15u) / 16u), LM_STREAM_THREADS, 0u, stream>>>(
				(const uint8_t *)d_w_q, d_lat, d_oq_ref, d_seq, d_ctx, d_pos, r);
		};
		auto new_q = [&]() {
			CHECK_OK(K3IndexQMmaWideSkipLaunch(d_w_q, d_lat, d_oq_new, d_seq, d_ctx, d_pos, r, (SparkGpuStream)stream));
		};
		old_q(); new_q();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		snprintf(tag, sizeof(tag), "IndexQ Wide vs Narrow R=%u", r);
		CheckBf16Close(tag, d_oq_ref, d_oq_new, (uint64_t)r * K3_INDEX_QUERY, 0.02f, 2u);
		const double t_old_q = TimeUs(old_q, stream);
		const double t_new_q = TimeUs(new_q, stream);
		printf("BENCH IndexQ R=%u: old_16x64=%.1f us  new_wide=%.1f us  speedup=%.2fx\n", r, t_old_q, t_new_q, t_old_q / t_new_q);
	}

	uint32_t ctx_dense = 1024u;
	for ( uint32_t i = 0u; i < MAX_ROWS; ++i )
		h_pos[i] = i < 1024u ? i : 1023u;
	CHECK_CUDA(cudaMemcpy(d_ctx, &ctx_dense, 4u, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_pos, h_pos.data(), MAX_ROWS * 4u, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemsetAsync(d_oh_ref, 0x5a, (uint64_t)MAX_ROWS * K3_INDEX_HEADS * 2u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_oh_new, 0x5a, (uint64_t)MAX_ROWS * K3_INDEX_HEADS * 2u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_oq_ref, 0x5a, (uint64_t)MAX_ROWS * K3_INDEX_QUERY * 2u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_oq_new, 0x5a, (uint64_t)MAX_ROWS * K3_INDEX_QUERY * 2u, stream));
	K3IndexHeadMmaSkipKernel<<<MAX_ROWS / 32u, 128u, 0u, stream>>>((const uint8_t *)d_w_head, d_norm, d_oh_new, d_seq, d_ctx, d_pos, MAX_ROWS);
	CHECK_OK(K3IndexQMmaWideSkipLaunch(d_w_q, d_lat, d_oq_new, d_seq, d_ctx, d_pos, MAX_ROWS, (SparkGpuStream)stream));
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CheckBf16Exact0Ulp("IndexHead dense-skip untouched", d_oh_ref, d_oh_new, (uint64_t)MAX_ROWS * K3_INDEX_HEADS);
	CheckBf16Exact0Ulp("IndexQ dense-skip untouched", d_oq_ref, d_oq_new, (uint64_t)MAX_ROWS * K3_INDEX_QUERY);

	CHECK_CUDA(cudaFree(d_w_head)); CHECK_CUDA(cudaFree(d_w_q)); CHECK_CUDA(cudaFree(d_norm)); CHECK_CUDA(cudaFree(d_lat));
	CHECK_CUDA(cudaFree(d_oh_ref)); CHECK_CUDA(cudaFree(d_oh_new)); CHECK_CUDA(cudaFree(d_oq_ref)); CHECK_CUDA(cudaFree(d_oq_new));
	CHECK_CUDA(cudaFree(d_seq)); CHECK_CUDA(cudaFree(d_ctx)); CHECK_CUDA(cudaFree(d_pos));
}

static __global__ void RefAddBf16Kernel(const uint16_t *a, const uint16_t *b, uint16_t *out, uint64_t count)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
		out[idx] = LmFloatToBf16(LmBf16ToFloat(a[idx]) + LmBf16ToFloat(b[idx]));
}

static void BenchSuite3FusedPrefillGemm(cudaStream_t stream)
{
	printf("\n=== Suite 3: Fused Prefill BF16 MMA Kernels (KDA, MoeDown+Route, SharedW1+SiTU, RoutedUp+Add) ===\n");
	constexpr uint32_t MAX_ROWS = 2048u;
	constexpr uint32_t QKVB_N = K3_KDA_QKVB_FUSED_ROWS / 16u;
	constexpr uint32_t GATE_N = K3_KDA_V_DIM / 16u;
	constexpr uint32_t DECAY_N = K3_KDA_KEY_DIM;
	constexpr uint32_t RANK_HEADS = K3_KDA_HEADS / 16u;
	constexpr uint32_t RANK_QK = RANK_HEADS * K3_KDA_KEY_DIM;
	constexpr uint32_t RANK_V = RANK_HEADS * K3_KDA_VALUE_DIM;
	constexpr uint32_t DOWN_N = K3_ROUTED_EXPERT_HIDDEN / 16u;
	constexpr uint32_t ROUTE_N = K3_EXPERTS / 16u;
	constexpr uint32_t SH_CH = K3_SHARED_INTERMEDIATE / 16u;

	uint16_t *d_act = 0, *d_w_qkvb = 0, *d_w_gate = 0, *d_w_dec = 0, *d_fused = 0;
	uint16_t *d_q_ref = 0, *d_k_ref = 0, *d_v_ref = 0, *d_b_ref = 0, *d_g_ref = 0, *d_d_ref = 0;
	uint16_t *d_q_new = 0, *d_k_new = 0, *d_v_new = 0, *d_b_new = 0, *d_g_new = 0, *d_d_new = 0;
	CHECK_CUDA(cudaMalloc(&d_act, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_w_qkvb, (uint64_t)QKVB_N * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_w_gate, (uint64_t)GATE_N * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_w_dec, (uint64_t)DECAY_N * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_fused, (uint64_t)MAX_ROWS * QKVB_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_q_ref, (uint64_t)MAX_ROWS * RANK_QK * 2u));
	CHECK_CUDA(cudaMalloc(&d_k_ref, (uint64_t)MAX_ROWS * RANK_QK * 2u));
	CHECK_CUDA(cudaMalloc(&d_v_ref, (uint64_t)MAX_ROWS * RANK_V * 2u));
	CHECK_CUDA(cudaMalloc(&d_b_ref, (uint64_t)MAX_ROWS * RANK_HEADS * 2u));
	CHECK_CUDA(cudaMalloc(&d_g_ref, (uint64_t)MAX_ROWS * GATE_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_d_ref, (uint64_t)MAX_ROWS * DECAY_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_q_new, (uint64_t)MAX_ROWS * RANK_QK * 2u));
	CHECK_CUDA(cudaMalloc(&d_k_new, (uint64_t)MAX_ROWS * RANK_QK * 2u));
	CHECK_CUDA(cudaMalloc(&d_v_new, (uint64_t)MAX_ROWS * RANK_V * 2u));
	CHECK_CUDA(cudaMalloc(&d_b_new, (uint64_t)MAX_ROWS * RANK_HEADS * 2u));
	CHECK_CUDA(cudaMalloc(&d_g_new, (uint64_t)MAX_ROWS * GATE_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_d_new, (uint64_t)MAX_ROWS * DECAY_N * 2u));
	FillBf16(d_act, (uint64_t)MAX_ROWS * K3_HIDDEN, 211u, 0.0005f);
	FillBf16(d_w_qkvb, (uint64_t)QKVB_N * K3_HIDDEN, 223u, 0.0005f);
	FillBf16(d_w_gate, (uint64_t)GATE_N * K3_HIDDEN, 227u, 0.0005f);
	FillBf16(d_w_dec, (uint64_t)DECAY_N * K3_HIDDEN, 229u, 0.0005f);

	const uint32_t sizes[3] = {512u, 1024u, 2048u};
	for ( uint32_t r : sizes )
	{
		K3KdaProjMmaArgs mma = {};
		mma.w_qkvb = (const uint8_t *)d_w_qkvb; mma.w_gate = (const uint8_t *)d_w_gate; mma.w_decay = (const uint8_t *)d_w_dec;
		mma.activation = d_act;
		mma.q = d_q_new; mma.k = d_k_new; mma.v = d_v_new; mma.beta = d_b_new; mma.gate = d_g_new; mma.latent = d_d_new;
		mma.rows = r; mma.qkvb_n = QKVB_N; mma.gate_n = GATE_N; mma.decay_n = DECAY_N;
		mma.rank_qk = RANK_QK; mma.rank_v = RANK_V; mma.rank_heads = RANK_HEADS;
		mma.tiles_qkvb = (QKVB_N + 127u) / 128u; mma.tiles_gate = (GATE_N + 127u) / 128u;
		auto old_fn = [&]() {
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_qkvb, LmScaleTensorNone(), d_act, d_fused, (float *)0, r, K3_HIDDEN, QKVB_N, 0u, 0u, g_sms, (SparkGpuStream)stream));
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_dec, LmScaleTensorNone(), d_act, d_d_ref, (float *)0, r, K3_HIDDEN, DECAY_N, 0u, 0u, g_sms, (SparkGpuStream)stream));
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_gate, LmScaleTensorNone(), d_act, d_g_ref, (float *)0, r, K3_HIDDEN, GATE_N, 0u, 0u, g_sms, (SparkGpuStream)stream));
			K3SplitFusedProjectionsKernel<K3_LAYER_THREADS><<<r, K3_LAYER_THREADS, 0u, stream>>>(
				d_fused, d_q_ref, d_k_ref, d_v_ref, d_b_ref, (uint16_t *)0, (uint16_t *)0, (uint16_t *)0, r, RANK_QK, RANK_V, RANK_HEADS, QKVB_N);
		};
		auto new_fn = [&]() { CHECK_OK(K3KdaProjPrefillMmaLaunch(mma, (SparkGpuStream)stream)); };
		old_fn(); new_fn();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "KdaProjPrefill Q R=%u", r);
		CheckBf16Close(tag, d_q_ref, d_q_new, (uint64_t)r * RANK_QK, 0.02f, 2u);
		snprintf(tag, sizeof(tag), "KdaProjPrefill Beta R=%u", r);
		CheckBf16Close(tag, d_b_ref, d_b_new, (uint64_t)r * RANK_HEADS, 0.02f, 2u);
		snprintf(tag, sizeof(tag), "KdaProjPrefill Gate R=%u", r);
		CheckBf16Close(tag, d_g_ref, d_g_new, (uint64_t)r * GATE_N, 0.02f, 2u);
		snprintf(tag, sizeof(tag), "KdaProjPrefill Decay R=%u", r);
		CheckBf16Close(tag, d_d_ref, d_d_new, (uint64_t)r * DECAY_N, 0.02f, 2u);
		const double t_old = TimeUs(old_fn, stream), t_new = TimeUs(new_fn, stream);
		printf("BENCH KdaProj R=%u: old_4kern=%.1f us  new_fused=%.1f us  speedup=%.2fx\n", r, t_old, t_new, t_old / t_new);
	}
	CHECK_CUDA(cudaFree(d_w_qkvb)); CHECK_CUDA(cudaFree(d_w_gate)); CHECK_CUDA(cudaFree(d_w_dec)); CHECK_CUDA(cudaFree(d_fused));
	CHECK_CUDA(cudaFree(d_q_ref)); CHECK_CUDA(cudaFree(d_k_ref)); CHECK_CUDA(cudaFree(d_v_ref)); CHECK_CUDA(cudaFree(d_b_ref)); CHECK_CUDA(cudaFree(d_g_ref)); CHECK_CUDA(cudaFree(d_d_ref));
	CHECK_CUDA(cudaFree(d_q_new)); CHECK_CUDA(cudaFree(d_k_new)); CHECK_CUDA(cudaFree(d_v_new)); CHECK_CUDA(cudaFree(d_b_new)); CHECK_CUDA(cudaFree(d_g_new)); CHECK_CUDA(cudaFree(d_d_new));

	uint16_t *d_w_down = 0, *d_w_route = 0, *d_lat_ref = 0, *d_lat_new = 0;
	float *d_rt_ref = 0, *d_rt_new = 0;
	CHECK_CUDA(cudaMalloc(&d_w_down, (uint64_t)DOWN_N * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_w_route, (uint64_t)ROUTE_N * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_lat_ref, (uint64_t)MAX_ROWS * DOWN_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_lat_new, (uint64_t)MAX_ROWS * DOWN_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_rt_ref, (uint64_t)MAX_ROWS * ROUTE_N * 4u));
	CHECK_CUDA(cudaMalloc(&d_rt_new, (uint64_t)MAX_ROWS * ROUTE_N * 4u));
	FillBf16(d_w_down, (uint64_t)DOWN_N * K3_HIDDEN, 233u, 0.0005f);
	FillBf16(d_w_route, (uint64_t)ROUTE_N * K3_HIDDEN, 239u, 0.0005f);
	for ( uint32_t r : sizes )
	{
		auto old_fn = [&]() {
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_route, LmScaleTensorNone(), d_act, (uint16_t *)0, d_rt_ref, r, K3_HIDDEN, ROUTE_N, 0u, 0u, g_sms, (SparkGpuStream)stream));
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_down, LmScaleTensorNone(), d_act, d_lat_ref, (float *)0, r, K3_HIDDEN, DOWN_N, 0u, 0u, g_sms, (SparkGpuStream)stream));
		};
		auto new_fn = [&]() {
			CHECK_OK(K3MoeDownRoutePrefillMmaLaunch(d_w_down, d_w_route, d_act, d_lat_new, d_rt_new, r, DOWN_N, ROUTE_N, (SparkGpuStream)stream));
		};
		old_fn(); new_fn();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "MoeDownRoute latent R=%u", r);
		CheckBf16Close(tag, d_lat_ref, d_lat_new, (uint64_t)r * DOWN_N, 0.02f, 2u);
		snprintf(tag, sizeof(tag), "MoeDownRoute router R=%u", r);
		CheckFloatClose(tag, d_rt_ref, d_rt_new, (uint64_t)r * ROUTE_N, 0.02f);
		const double t_old = TimeUs(old_fn, stream), t_new = TimeUs(new_fn, stream);
		printf("BENCH MoeDown+Route R=%u: old_2kern=%.1f us  new_fused=%.1f us  speedup=%.2fx\n", r, t_old, t_new, t_old / t_new);
	}
	CHECK_CUDA(cudaFree(d_w_down)); CHECK_CUDA(cudaFree(d_w_route)); CHECK_CUDA(cudaFree(d_lat_ref));
	CHECK_CUDA(cudaFree(d_lat_new)); CHECK_CUDA(cudaFree(d_rt_ref)); CHECK_CUDA(cudaFree(d_rt_new));

	uint16_t *d_w_sh = 0, *d_gu = 0, *d_sh_ref = 0, *d_sh_new = 0;
	CHECK_CUDA(cudaMalloc(&d_w_sh, (uint64_t)SH_CH * 2u * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_gu, (uint64_t)MAX_ROWS * SH_CH * 2u * 2u));
	CHECK_CUDA(cudaMalloc(&d_sh_ref, (uint64_t)MAX_ROWS * SH_CH * 2u));
	CHECK_CUDA(cudaMalloc(&d_sh_new, (uint64_t)MAX_ROWS * SH_CH * 2u));
	FillBf16(d_w_sh, (uint64_t)SH_CH * 2u * K3_HIDDEN, 241u, 0.0005f);
	for ( uint32_t r : sizes )
	{
		auto old_fn = [&]() {
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_sh, LmScaleTensorNone(), d_act, d_gu, (float *)0, r, K3_HIDDEN, SH_CH * 2u, 0u, 0u, g_sms, (SparkGpuStream)stream));
			LmSituMulKernel<K3_LAYER_THREADS><<<r, K3_LAYER_THREADS, 0u, stream>>>(d_gu, d_sh_ref, SH_CH, (uint64_t)SH_CH, K3_SITU_BETA, K3_SITU_LINEAR_BETA, 1u);
		};
		auto new_fn = [&]() {
			CHECK_OK(K3SharedW1SituPrefillMmaLaunch(d_w_sh, d_act, d_sh_new, r, SH_CH, (SparkGpuStream)stream));
		};
		old_fn(); new_fn();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "SharedW1+SiTU R=%u", r);
		CheckBf16Close(tag, d_sh_ref, d_sh_new, (uint64_t)r * SH_CH, 0.02f, 2u);
		const double t_old = TimeUs(old_fn, stream), t_new = TimeUs(new_fn, stream);
		printf("BENCH SharedW1+SiTU R=%u: old_2kern=%.1f us  new_fused=%.1f us  speedup=%.2fx\n", r, t_old, t_new, t_old / t_new);
	}
	CHECK_CUDA(cudaFree(d_w_sh)); CHECK_CUDA(cudaFree(d_gu)); CHECK_CUDA(cudaFree(d_sh_ref)); CHECK_CUDA(cudaFree(d_sh_new));

	uint16_t *d_w_up = 0, *d_in_up = 0, *d_add = 0, *d_up_tmp = 0, *d_up_ref = 0, *d_up_new = 0;
	CHECK_CUDA(cudaMalloc(&d_w_up, (uint64_t)K3_HIDDEN * DOWN_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_in_up, (uint64_t)MAX_ROWS * DOWN_N * 2u));
	CHECK_CUDA(cudaMalloc(&d_add, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_up_tmp, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_up_ref, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	CHECK_CUDA(cudaMalloc(&d_up_new, (uint64_t)MAX_ROWS * K3_HIDDEN * 2u));
	FillBf16(d_w_up, (uint64_t)K3_HIDDEN * DOWN_N, 251u, 0.001f);
	FillBf16(d_in_up, (uint64_t)MAX_ROWS * DOWN_N, 257u, 0.001f);
	FillBf16(d_add, (uint64_t)MAX_ROWS * K3_HIDDEN, 263u, 0.001f);
	for ( uint32_t r : sizes )
	{
		const uint64_t elems = (uint64_t)r * K3_HIDDEN;
		auto old_fn = [&]() {
			CHECK_OK(LmStreamGemmDense<LmBf16Format>(d_w_up, LmScaleTensorNone(), d_in_up, d_up_tmp, (float *)0, r, DOWN_N, K3_HIDDEN, 0u, 0u, g_sms, (SparkGpuStream)stream));
			RefAddBf16Kernel<<<(uint32_t)((elems + 255u) / 256u), 256u, 0u, stream>>>(d_up_tmp, d_add, d_up_ref, elems);
		};
		auto new_fn = [&]() {
			CHECK_OK(K3RoutedUpAddPrefillMmaLaunch(d_w_up, d_in_up, d_add, d_up_new, r, DOWN_N, K3_HIDDEN, (SparkGpuStream)stream));
		};
		old_fn(); new_fn();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "RoutedUp(K=224)+Add R=%u", r);
		CheckBf16Close(tag, d_up_ref, d_up_new, elems, 0.02f, 2u);
		const double t_old = TimeUs(old_fn, stream), t_new = TimeUs(new_fn, stream);
		printf("BENCH RoutedUp(K=224)+Add R=%u: old_tail+pack=%.1f us  new_k32_fused=%.1f us  speedup=%.2fx\n", r, t_old, t_new, t_old / t_new);
	}
	CHECK_CUDA(cudaFree(d_w_up)); CHECK_CUDA(cudaFree(d_in_up)); CHECK_CUDA(cudaFree(d_add));
	CHECK_CUDA(cudaFree(d_up_tmp)); CHECK_CUDA(cudaFree(d_up_ref)); CHECK_CUDA(cudaFree(d_up_new)); CHECK_CUDA(cudaFree(d_act));
}

static void BenchSuite4MlaPrefill(cudaStream_t stream)
{
	printf("\n=== Suite 4: MLA Prefill Attention (6-Head Sparse & Direct Expanded-V) ===\n");
	constexpr uint32_t HEADS = 6u, MAX_CTX = 32768u, MAX_ROWS = 2048u;
	constexpr uint32_t ROWS_PER_BLOCK = 8u, UNION_CAP = ROWS_PER_BLOCK * K3_INDEX_TOPK;
	constexpr uint32_t MAX_BLOCKS = MAX_ROWS / ROWS_PER_BLOCK;

	uint16_t *d_q = 0, *d_kv = 0, *d_wv = 0, *d_out_ref = 0, *d_out_new = 0, *d_val_ref = 0, *d_val_new = 0;
	uint8_t *d_exp_v = 0, *d_umasks = 0, *d_dummy_pool = 0;
	uint32_t *d_seq = 0, *d_ctx = 0, *d_pos = 0, *d_sel = 0, *d_upos = 0, *d_ucnt = 0, *d_goff = 0, *d_pt = 0;
	float *d_part = 0;
	LmKvAccessError *d_err = 0;

	CHECK_CUDA(cudaMalloc(&d_q, (uint64_t)MAX_ROWS * HEADS * K3_MLA_KV_A_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_kv, (uint64_t)MAX_CTX * K3_MLA_KV_A_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_wv, (uint64_t)HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK * 2u));
	CHECK_CUDA(cudaMalloc(&d_out_ref, (uint64_t)MAX_ROWS * HEADS * K3_KV_LORA_RANK * 2u));
	CHECK_CUDA(cudaMalloc(&d_out_new, (uint64_t)MAX_ROWS * HEADS * K3_KV_LORA_RANK * 2u));
	CHECK_CUDA(cudaMalloc(&d_val_ref, (uint64_t)MAX_ROWS * HEADS * K3_V_HEAD_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_val_new, (uint64_t)MAX_ROWS * HEADS * K3_V_HEAD_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_exp_v, (uint64_t)MAX_CTX * HEADS * K3_V_HEAD_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_seq, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMalloc(&d_ctx, 4u));
	CHECK_CUDA(cudaMalloc(&d_pos, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMalloc(&d_sel, (uint64_t)MAX_ROWS * K3_INDEX_TOPK * 4u));
	CHECK_CUDA(cudaMalloc(&d_upos, (uint64_t)MAX_BLOCKS * UNION_CAP * 4u));
	CHECK_CUDA(cudaMalloc(&d_umasks, (uint64_t)MAX_BLOCKS * UNION_CAP));
	CHECK_CUDA(cudaMalloc(&d_ucnt, MAX_BLOCKS * 4u));
	CHECK_CUDA(cudaMalloc(&d_goff, 4u));
	CHECK_CUDA(cudaMalloc(&d_pt, 64u * 4u));
	CHECK_CUDA(cudaMalloc(&d_dummy_pool, K3GlobalKv::kPageBytes));
	CHECK_CUDA(cudaMalloc(&d_err, sizeof(LmKvAccessError)));
	const uint64_t part_stride = (uint64_t)MAX_ROWS * HEADS * LM_LATENT_SHARD_RECORD_FLOATS(K3_KV_LORA_RANK);
	CHECK_CUDA(cudaMalloc(&d_part, part_stride * sizeof(float)));

	FillBf16(d_q, (uint64_t)MAX_ROWS * HEADS * K3_MLA_KV_A_DIM, 307u, 0.0005f);
	FillBf16(d_kv, (uint64_t)MAX_CTX * K3_MLA_KV_A_DIM, 311u, 0.0005f);
	FillBf16(d_wv, (uint64_t)HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK, 313u, 0.001f);
	CHECK_CUDA(cudaMemset(d_seq, 0, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMemset(d_goff, 0, 4u));
	CHECK_CUDA(cudaMemset(d_pt, 0, 64u * 4u));
	CHECK_CUDA(cudaMemset(d_err, 0, sizeof(LmKvAccessError)));

	constexpr uint32_t SHARED2 = LmPrefillAttnSharedBytes<K3_MLA_KV_A_DIM, LM_PREFILL_ATTN_QUERIES>();
	constexpr uint32_t SHARED6 = K3Prefill6HeadSharedBytes();
	CHECK_GPU(SparkGpuFuncSetAttribute((const void *)K3MlaGatherPrefillKernel<true>,
		SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED2));
	CHECK_GPU(SparkGpuFuncSetAttribute((const void *)K3MlaGatherPrefill6HeadKernel,
		SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED6));

	const uint32_t ctxs[2] = {8192u, 32768u};
	for ( uint32_t ctx : ctxs )
	{
		const uint32_t rows = 2048u, blocks = rows / ROWS_PER_BLOCK;
		std::vector<uint32_t> h_pos(rows), h_sel((uint64_t)rows * K3_INDEX_TOPK);
		for ( uint32_t r = 0u; r < rows; ++r )
		{
			h_pos[r] = ctx - rows + r;
			const uint32_t stride = (h_pos[r] + 1u) / K3_INDEX_TOPK;
			for ( uint32_t k = 0u; k < K3_INDEX_TOPK; ++k )
			{
				const uint32_t jitter = (uint32_t)(Hash64((uint64_t)(r / 8u) * 7919u + k * 13u) % (stride > 1u ? stride : 1u));
				const uint32_t p = k * stride + jitter;
				h_sel[(uint64_t)r * K3_INDEX_TOPK + k] = p <= h_pos[r] ? p : h_pos[r];
			}
			std::sort(h_sel.begin() + (ptrdiff_t)((uint64_t)r * K3_INDEX_TOPK),
				h_sel.begin() + (ptrdiff_t)((uint64_t)(r + 1u) * K3_INDEX_TOPK));
		}
		CHECK_CUDA(cudaMemcpy(d_ctx, &ctx, 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_pos, h_pos.data(), rows * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4u, cudaMemcpyHostToDevice));
		LmPrefillSparseUnionKernel<ROWS_PER_BLOCK, K3_INDEX_TOPK><<<blocks, LM_PREFILL_UNION_THREADS, 0u, stream>>>(
			d_sel, d_pos, rows, (const uint32_t *)0, UNION_CAP, d_upos, d_umasks, d_ucnt);
		CHECK_CUDA(cudaStreamSynchronize(stream));

		auto old_s = [&]() {
			K3MlaGatherPrefillKernel<true><<<dim3(blocks, HEADS / 2u), LM_PREFILL_ATTN_THREADS, SHARED2, stream>>>(
				d_q, d_kv, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE, d_out_ref, rows, HEADS, ctx, d_upos, d_umasks, d_ucnt, UNION_CAP);
		};
		auto new_s = [&]() {
			K3MlaGatherPrefill6HeadKernel<<<blocks, LM_PREFILL_ATTN_THREADS, SHARED6, stream>>>(
				d_q, d_kv, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE, d_out_new, rows, ctx, d_upos, d_umasks, d_ucnt, UNION_CAP);
		};
		old_s(); new_s();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "MLA Sparse 6Head vs 2Head ctx=%u", ctx);
		CheckBf16Close(tag, d_out_ref, d_out_new, (uint64_t)rows * HEADS * K3_KV_LORA_RANK, 0.01f, 2u);
		const double t_old = TimeUs(old_s, stream), t_new = TimeUs(new_s, stream);
		printf("BENCH MLA Sparse R=2048 ctx=%u: old_2head=%.1f us  new_6head=%.1f us  speedup=%.2fx\n", ctx, t_old, t_new, t_old / t_new);
	}

	LmKvView pages = {};
	LmKvViewInitialize(&pages, d_dummy_pool, d_pt, 64u, 1u, 1u, d_err);
	SparkKvShard flat_owner = {1u, 0u, 1u};
	LmKvShardGatherAllView all_view = {};
	LmKvShardGatherView flat_view = {};
	LmKvShardGatherAllViewInitialize<K3GlobalKv>(&all_view, pages, flat_owner, (const uint8_t *)d_kv, 0u, d_goff, d_ctx);
	LmKvShardGatherViewInitialize<K3GlobalKv>(&flat_view, pages, flat_owner, (const uint8_t *)d_kv, d_goff, d_ctx);

	for ( uint32_t ctx : ctxs )
	{
		const uint32_t rows = 1024u;
		std::vector<uint32_t> h_pos(rows);
		for ( uint32_t r = 0u; r < rows; ++r )
			h_pos[r] = ctx - rows + r;
		CHECK_CUDA(cudaMemcpy(d_ctx, &ctx, 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_pos, h_pos.data(), rows * 4u, cudaMemcpyHostToDevice));
		auto old_dense = [&]() {
			CHECK_GPU(LmLatentGatherPrefillLaunch<K3GlobalKv, LmKvShardGatherAllView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
				all_view, d_q, HEADS, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE, d_part, rows, (SparkGpuStream)stream));
			CHECK_GPU(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(d_part, part_stride, 1u, HEADS, d_out_ref, rows, (SparkGpuStream)stream));
			CHECK_GPU(LmPerHeadProjectChainLaunch<K3_LAYER_THREADS, K3_KV_LORA_RANK, K3_V_HEAD_DIM>(
				d_out_ref, d_wv, d_val_ref, HEADS, rows, (SparkGpuStream)stream, 0u));
		};
		auto new_exp = [&]() {
			CHECK_GPU(LmLatentValueExpandLaunch<K3_KV_LORA_RANK, K3_MLA_KV_A_DIM, K3_V_HEAD_DIM>(
				(const uint8_t *)d_kv, 0u, 1u, ctx, d_wv, HEADS, d_exp_v, 0u, (SparkGpuStream)stream));
			CHECK_GPU(LmExpandedGatherPrefillLaunch<K3GlobalKv, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_V_HEAD_DIM>(
				flat_view, d_exp_v, d_q, HEADS, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE, d_part, rows, (SparkGpuStream)stream));
			CHECK_GPU(LmLatentShardMergeLaunch<K3_V_HEAD_DIM>(d_part, part_stride, 1u, HEADS, d_val_new, rows, (SparkGpuStream)stream));
		};
		old_dense(); new_exp();
		CHECK_CUDA(cudaStreamSynchronize(stream));
		char tag[64];
		snprintf(tag, sizeof(tag), "MLA Direct Expanded-V vs Latent ctx=%u", ctx);
		CheckBf16Close(tag, d_val_ref, d_val_new, (uint64_t)rows * HEADS * K3_V_HEAD_DIM, 0.02f, 2u);
		const double t_old = TimeUs(old_dense, stream), t_new = TimeUs(new_exp, stream);
		printf("BENCH MLA Direct Dense R=1024 ctx=%u: old_latent512=%.1f us  new_expanded128=%.1f us  speedup=%.2fx\n",
			ctx, t_old, t_new, t_old / t_new);
	}

	CHECK_CUDA(cudaFree(d_q)); CHECK_CUDA(cudaFree(d_kv)); CHECK_CUDA(cudaFree(d_wv));
	CHECK_CUDA(cudaFree(d_out_ref)); CHECK_CUDA(cudaFree(d_out_new));
	CHECK_CUDA(cudaFree(d_val_ref)); CHECK_CUDA(cudaFree(d_val_new)); CHECK_CUDA(cudaFree(d_exp_v));
	CHECK_CUDA(cudaFree(d_seq)); CHECK_CUDA(cudaFree(d_ctx)); CHECK_CUDA(cudaFree(d_pos));
	CHECK_CUDA(cudaFree(d_sel)); CHECK_CUDA(cudaFree(d_upos)); CHECK_CUDA(cudaFree(d_umasks)); CHECK_CUDA(cudaFree(d_ucnt));
	CHECK_CUDA(cudaFree(d_goff)); CHECK_CUDA(cudaFree(d_pt)); CHECK_CUDA(cudaFree(d_dummy_pool)); CHECK_CUDA(cudaFree(d_err)); CHECK_CUDA(cudaFree(d_part));
}

static void BenchSuite5Elementwise(cudaStream_t stream)
{
	printf("\n=== Suite 5: AttnRes Slice Fusion & Vectorized Elementwise + 5c 0-ULP ===\n");
	constexpr uint32_t ROWS = 2048u, HEADS = 6u;

	uint16_t *d_q_orig = 0, *d_k_orig = 0, *d_q_ref = 0, *d_k_ref = 0, *d_q_new = 0, *d_k_new = 0;
	uint16_t *d_dec = 0, *d_beta = 0, *d_gate = 0, *d_ao_orig = 0, *d_ao_ref = 0, *d_ao_new = 0;
	float *d_cbias = 0, *d_hls = 0, *d_ret_ref = 0, *d_ret_new = 0, *d_wg_ref = 0, *d_wg_new = 0, *d_nw = 0;
	CHECK_CUDA(cudaMalloc(&d_q_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_k_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_q_ref, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_k_ref, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_q_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_k_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_dec, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_beta, (uint64_t)ROWS * HEADS * 2u));
	CHECK_CUDA(cudaMalloc(&d_gate, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_ao_orig, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_ao_ref, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_ao_new, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u));
	CHECK_CUDA(cudaMalloc(&d_cbias, (uint64_t)HEADS * K3_KDA_KEY_DIM * 4u));
	CHECK_CUDA(cudaMalloc(&d_hls, HEADS * 4u));
	CHECK_CUDA(cudaMalloc(&d_ret_ref, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 4u));
	CHECK_CUDA(cudaMalloc(&d_ret_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 4u));
	CHECK_CUDA(cudaMalloc(&d_wg_ref, (uint64_t)ROWS * HEADS * 4u));
	CHECK_CUDA(cudaMalloc(&d_wg_new, (uint64_t)ROWS * HEADS * 4u));
	CHECK_CUDA(cudaMalloc(&d_nw, K3_KDA_VALUE_DIM * 4u));

	FillBf16(d_q_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM, 401u, 0.002f);
	FillBf16(d_k_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM, 409u, 0.002f);
	FillBf16(d_dec, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM, 419u, 0.002f);
	FillBf16(d_beta, (uint64_t)ROWS * HEADS, 421u, 0.002f);
	FillBf16(d_gate, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM, 431u, 0.002f);
	FillBf16(d_ao_orig, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM, 433u, 0.002f);
	FillFloat(d_cbias, (uint64_t)HEADS * K3_KDA_KEY_DIM, 439u, 0.001f);
	FillFloat(d_hls, HEADS, 443u, 0.001f);
	FillFloat(d_nw, K3_KDA_VALUE_DIM, 449u, 0.001f);

	CHECK_CUDA(cudaMemcpyAsync(d_q_ref, d_q_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_k_ref, d_k_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_q_new, d_q_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_k_new, d_k_orig, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	K3KdaPrepFinishKernel<128u, K3_KDA_KEY_DIM><<<dim3(ROWS, HEADS), 128u, 0u, stream>>>(
		d_q_ref, d_k_ref, d_dec, d_cbias, d_hls, d_beta, d_ret_ref, d_wg_ref, HEADS, K3_KDA_GATE_LOWER_BOUND, K3_RMS_EPSILON, ROWS);
	K3KdaPrepFinishVecKernel<<<(ROWS * HEADS + 3u) / 4u, 128u, 0u, stream>>>(
		d_q_new, d_k_new, d_dec, d_cbias, d_hls, d_beta, d_ret_new, d_wg_new, HEADS, K3_KDA_GATE_LOWER_BOUND, K3_RMS_EPSILON, ROWS);
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CheckBf16Exact0Ulp("KdaPrepFinishVec Q (0-ULP)", d_q_ref, d_q_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM);
	CheckBf16Exact0Ulp("KdaPrepFinishVec K (0-ULP)", d_k_ref, d_k_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM);
	CheckFloatClose("KdaPrepFinishVec retention", d_ret_ref, d_ret_new, (uint64_t)ROWS * HEADS * K3_KDA_KEY_DIM, 0.0f);
	CheckFloatClose("KdaPrepFinishVec write_gate", d_wg_ref, d_wg_new, (uint64_t)ROWS * HEADS, 0.0f);

	const double t_prep_old = TimeUs([&]() {
		K3KdaPrepFinishKernel<128u, K3_KDA_KEY_DIM><<<dim3(ROWS, HEADS), 128u, 0u, stream>>>(
			d_q_ref, d_k_ref, d_dec, d_cbias, d_hls, d_beta, d_ret_ref, d_wg_ref, HEADS, K3_KDA_GATE_LOWER_BOUND, K3_RMS_EPSILON, ROWS);
	}, stream);
	const double t_prep_new = TimeUs([&]() {
		K3KdaPrepFinishVecKernel<<<(ROWS * HEADS + 3u) / 4u, 128u, 0u, stream>>>(
			d_q_new, d_k_new, d_dec, d_cbias, d_hls, d_beta, d_ret_new, d_wg_new, HEADS, K3_KDA_GATE_LOWER_BOUND, K3_RMS_EPSILON, ROWS);
	}, stream);
	printf("BENCH KdaPrepFinish R=2048: old_scalar=%.1f us  new_vec=%.1f us  speedup=%.2fx\n", t_prep_old, t_prep_new, t_prep_old / t_prep_new);

	CHECK_CUDA(cudaMemcpyAsync(d_ao_ref, d_ao_orig, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_ao_new, d_ao_orig, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM * 2u, cudaMemcpyDeviceToDevice, stream));
	K3KdaNormGateKernel<128u, K3_KDA_VALUE_DIM><<<dim3(ROWS, HEADS), 128u, 0u, stream>>>(d_ao_ref, d_gate, d_nw, HEADS, K3_RMS_EPSILON, ROWS);
	K3KdaNormGateVecKernel<<<(ROWS * HEADS + 3u) / 4u, 128u, 0u, stream>>>(d_ao_new, d_gate, d_nw, HEADS, K3_RMS_EPSILON, ROWS);
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CheckBf16Exact0Ulp("KdaNormGateVec (0-ULP)", d_ao_ref, d_ao_new, (uint64_t)ROWS * HEADS * K3_KDA_VALUE_DIM);
	const double t_ng_old = TimeUs([&]() {
		K3KdaNormGateKernel<128u, K3_KDA_VALUE_DIM><<<dim3(ROWS, HEADS), 128u, 0u, stream>>>(d_ao_ref, d_gate, d_nw, HEADS, K3_RMS_EPSILON, ROWS);
	}, stream);
	const double t_ng_new = TimeUs([&]() {
		K3KdaNormGateVecKernel<<<(ROWS * HEADS + 3u) / 4u, 128u, 0u, stream>>>(d_ao_new, d_gate, d_nw, HEADS, K3_RMS_EPSILON, ROWS);
	}, stream);
	printf("BENCH KdaNormGate R=2048: old_scalar=%.1f us  new_vec=%.1f us  speedup=%.2fx\n", t_ng_old, t_ng_new, t_ng_old / t_ng_new);

	{
		constexpr uint32_t TOT_ROWS = 1024u, COUNT = 64u, FIRST = 128u, SOURCES = 9u;
		uint16_t *d_bank = 0, *d_p_orig = 0, *d_p_ref = 0, *d_p_new = 0, *d_io_orig = 0, *d_io_ref = 0, *d_io_new = 0, *d_sw = 0;
		CHECK_CUDA(cudaMalloc(&d_bank, (uint64_t)(SOURCES - 1u) * TOT_ROWS * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_p_orig, (uint64_t)TOT_ROWS * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_p_ref, (uint64_t)TOT_ROWS * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_p_new, (uint64_t)TOT_ROWS * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_io_orig, (uint64_t)COUNT * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_io_ref, (uint64_t)COUNT * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_io_new, (uint64_t)COUNT * K3_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_sw, K3_HIDDEN * 2u));
		FillBf16(d_bank, (uint64_t)(SOURCES - 1u) * TOT_ROWS * K3_HIDDEN, 501u, 0.001f);
		FillBf16(d_p_orig, (uint64_t)TOT_ROWS * K3_HIDDEN, 503u, 0.001f);
		FillBf16(d_io_orig, (uint64_t)COUNT * K3_HIDDEN, 509u, 0.001f);
		FillBf16(d_sw, K3_HIDDEN, 521u, 0.001f);

		for ( uint32_t set_p = 0u; set_p <= 1u; ++set_p )
		{
			CHECK_CUDA(cudaMemcpyAsync(d_p_ref, d_p_orig, (uint64_t)TOT_ROWS * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(d_p_new, d_p_orig, (uint64_t)TOT_ROWS * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(d_io_ref, d_io_orig, (uint64_t)COUNT * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(d_io_new, d_io_orig, (uint64_t)COUNT * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice, stream));
			K3LayerBuffers bref = {};
			bref.attnres_bank_bf16 = d_bank; bref.attnres_partial_bf16 = d_p_ref;
			K3PartialRows(&bref, d_io_ref, FIRST, COUNT, set_p, (SparkGpuStream)stream);
			K3AttnResRows(&bref, d_sw, SOURCES, TOT_ROWS, FIRST, COUNT, d_io_ref, (SparkGpuStream)stream);
			K3PartialAddAttnResSliceKernel<<<COUNT, 256u, 0u, stream>>>(
				d_bank, d_p_new, d_io_new, d_sw, SOURCES, TOT_ROWS, FIRST, COUNT, set_p, K3_RMS_EPSILON);
			CHECK_CUDA(cudaStreamSynchronize(stream));
			char tag[64];
			snprintf(tag, sizeof(tag), "AttnResSlice partial set=%u", set_p);
			CheckBf16Exact0Ulp(tag, d_p_ref + (uint64_t)FIRST * K3_HIDDEN, d_p_new + (uint64_t)FIRST * K3_HIDDEN, (uint64_t)COUNT * K3_HIDDEN);
			snprintf(tag, sizeof(tag), "AttnResSlice mix set=%u (0-ULP)", set_p);
			CheckBf16Exact0Ulp(tag, d_io_ref, d_io_new, (uint64_t)COUNT * K3_HIDDEN);
		}
		K3LayerBuffers bref = {};
		bref.attnres_bank_bf16 = d_bank; bref.attnres_partial_bf16 = d_p_ref;
		const double t_ar_old = TimeUs([&]() {
			K3PartialRows(&bref, d_io_ref, FIRST, COUNT, 0u, (SparkGpuStream)stream);
			K3AttnResRows(&bref, d_sw, SOURCES, TOT_ROWS, FIRST, COUNT, d_io_ref, (SparkGpuStream)stream);
		}, stream);
		const double t_ar_new = TimeUs([&]() {
			K3PartialAddAttnResSliceKernel<<<COUNT, 256u, 0u, stream>>>(
				d_bank, d_p_new, d_io_new, d_sw, SOURCES, TOT_ROWS, FIRST, COUNT, 0u, K3_RMS_EPSILON);
		}, stream);
		printf("BENCH AttnResSlice count=64 src=9: old_2kern=%.1f us  new_fused=%.1f us  speedup=%.2fx\n", t_ar_old, t_ar_new, t_ar_old / t_ar_new);

		CHECK_CUDA(cudaFree(d_bank)); CHECK_CUDA(cudaFree(d_p_orig)); CHECK_CUDA(cudaFree(d_p_ref)); CHECK_CUDA(cudaFree(d_p_new));
		CHECK_CUDA(cudaFree(d_io_orig)); CHECK_CUDA(cudaFree(d_io_ref)); CHECK_CUDA(cudaFree(d_io_new)); CHECK_CUDA(cudaFree(d_sw));
	}

	{
		uint16_t *d_in = 0, *d_w = 0, *d_out1 = 0, *d_out8 = 0;
		CHECK_CUDA(cudaMalloc(&d_in, (uint64_t)ROWS * K3_ROUTED_EXPERT_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_w, K3_ROUTED_EXPERT_HIDDEN * 2u));
		CHECK_CUDA(cudaMalloc(&d_out1, 224u * 2u));
		CHECK_CUDA(cudaMalloc(&d_out8, (uint64_t)ROWS * 224u * 2u));
		FillBf16(d_in, (uint64_t)ROWS * K3_ROUTED_EXPERT_HIDDEN, 601u, 0.002f);
		FillBf16(d_w, K3_ROUTED_EXPERT_HIDDEN, 607u, 0.002f);
		K3RmsNormSliceKernel<256u, K3_ROUTED_EXPERT_HIDDEN><<<1u, 256u, 0u, stream>>>(d_in, d_w, d_out1, 224u, 224u, K3_RMS_EPSILON);
		K3RmsNormSliceKernel<256u, K3_ROUTED_EXPERT_HIDDEN><<<8u, 256u, 0u, stream>>>(d_in, d_w, d_out8, 224u, 224u, K3_RMS_EPSILON);
		CHECK_CUDA(cudaStreamSynchronize(stream));
		CheckBf16Exact0Ulp("RmsNormSlice B1==B8", d_out1, d_out8, 224u);
		const double t_rms = TimeUs([&]() {
			K3RmsNormSliceKernel<256u, K3_ROUTED_EXPERT_HIDDEN><<<ROWS, 256u, 0u, stream>>>(d_in, d_w, d_out8, 224u, 224u, K3_RMS_EPSILON);
		}, stream);
		printf("BENCH RmsNormSlice (0-SMEM vec) R=2048: %.1f us\n", t_rms);
		CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_w)); CHECK_CUDA(cudaFree(d_out1)); CHECK_CUDA(cudaFree(d_out8));
	}

	{
		uint16_t *d_src = 0, *d_w = 0, *d_all_ref = 0, *d_all_new = 0;
		uint8_t *d_pool_ref = 0, *d_pool_new = 0;
		uint32_t *d_seq = 0, *d_pos = 0, *d_pt = 0;
		LmKvAccessError *d_err = 0;
		const uint32_t pages_n = 32u;
		CHECK_CUDA(cudaMalloc(&d_src, (uint64_t)ROWS * K3_MLA_KV_A_DIM * 2u));
		CHECK_CUDA(cudaMalloc(&d_w, K3_KV_LORA_RANK * 2u));
		CHECK_CUDA(cudaMalloc(&d_all_ref, (uint64_t)ROWS * K3_MLA_KV_A_DIM * 2u));
		CHECK_CUDA(cudaMalloc(&d_all_new, (uint64_t)ROWS * K3_MLA_KV_A_DIM * 2u));
		CHECK_CUDA(cudaMalloc(&d_pool_ref, (uint64_t)pages_n * K3GlobalKv::kPageBytes));
		CHECK_CUDA(cudaMalloc(&d_pool_new, (uint64_t)pages_n * K3GlobalKv::kPageBytes));
		CHECK_CUDA(cudaMalloc(&d_seq, ROWS * 4u));
		CHECK_CUDA(cudaMalloc(&d_pos, ROWS * 4u));
		CHECK_CUDA(cudaMalloc(&d_pt, pages_n * 4u));
		CHECK_CUDA(cudaMalloc(&d_err, sizeof(LmKvAccessError)));
		FillBf16(d_src, (uint64_t)ROWS * K3_MLA_KV_A_DIM, 701u, 0.002f);
		FillBf16(d_w, K3_KV_LORA_RANK, 709u, 0.002f);
		std::vector<uint32_t> h_pos(ROWS), h_pt(pages_n);
		for ( uint32_t i = 0u; i < ROWS; ++i ) h_pos[i] = i;
		for ( uint32_t i = 0u; i < pages_n; ++i ) h_pt[i] = i;
		CHECK_CUDA(cudaMemset(d_seq, 0, ROWS * 4u));
		CHECK_CUDA(cudaMemcpy(d_pos, h_pos.data(), ROWS * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_pt, h_pt.data(), pages_n * 4u, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemset(d_err, 0, sizeof(LmKvAccessError)));
		LmKvShardView vref = {}, vnew = {};
		const SparkKvShard sh = {16u, 0u, 1u};
		LmKvShardViewInitialize<K3GlobalKv>(&vref, d_pool_ref, d_pt, pages_n, 1u, pages_n, d_err, sh);
		LmKvShardViewInitialize<K3GlobalKv>(&vnew, d_pool_new, d_pt, pages_n, 1u, pages_n, d_err, sh);
		K3RmsNormKvShardStoreKernel<K3GlobalKv, 256u><<<ROWS, 256u, 0u, stream>>>(
			vref, d_src, d_w, d_seq, d_pos, ROWS, K3_LORA_RMS_EPSILON, d_all_ref, ROWS);
		K3RmsNormKvShardStoreKernel<K3GlobalKv, 64u><<<ROWS, 64u, 0u, stream>>>(
			vnew, d_src, d_w, d_seq, d_pos, ROWS, K3_LORA_RMS_EPSILON, d_all_new, ROWS);
		CHECK_CUDA(cudaStreamSynchronize(stream));
		CheckBf16Close("RmsNormKvShardStore64 kv_all", d_all_ref, d_all_new, (uint64_t)ROWS * K3_MLA_KV_A_DIM, 1e-3f, 1u);
		const double t_kv_old = TimeUs([&]() {
			K3RmsNormKvShardStoreKernel<K3GlobalKv, 256u><<<ROWS, 256u, 0u, stream>>>(
				vref, d_src, d_w, d_seq, d_pos, ROWS, K3_LORA_RMS_EPSILON, d_all_ref, ROWS);
		}, stream);
		const double t_kv_new = TimeUs([&]() {
			K3RmsNormKvShardStoreKernel<K3GlobalKv, 64u><<<ROWS, 64u, 0u, stream>>>(
				vnew, d_src, d_w, d_seq, d_pos, ROWS, K3_LORA_RMS_EPSILON, d_all_new, ROWS);
		}, stream);
		printf("BENCH RmsNormKvShardStore R=2048: old_256scalar=%.1f us  new_64vec=%.1f us  speedup=%.2fx\n", t_kv_old, t_kv_new, t_kv_old / t_kv_new);
		CHECK_CUDA(cudaFree(d_src)); CHECK_CUDA(cudaFree(d_w)); CHECK_CUDA(cudaFree(d_all_ref)); CHECK_CUDA(cudaFree(d_all_new));
		CHECK_CUDA(cudaFree(d_pool_ref)); CHECK_CUDA(cudaFree(d_pool_new)); CHECK_CUDA(cudaFree(d_seq)); CHECK_CUDA(cudaFree(d_pos));
		CHECK_CUDA(cudaFree(d_pt)); CHECK_CUDA(cudaFree(d_err));
	}

	CHECK_CUDA(cudaFree(d_q_orig)); CHECK_CUDA(cudaFree(d_k_orig)); CHECK_CUDA(cudaFree(d_q_ref)); CHECK_CUDA(cudaFree(d_k_ref));
	CHECK_CUDA(cudaFree(d_q_new)); CHECK_CUDA(cudaFree(d_k_new)); CHECK_CUDA(cudaFree(d_dec)); CHECK_CUDA(cudaFree(d_beta));
	CHECK_CUDA(cudaFree(d_gate)); CHECK_CUDA(cudaFree(d_ao_orig)); CHECK_CUDA(cudaFree(d_ao_ref)); CHECK_CUDA(cudaFree(d_ao_new));
	CHECK_CUDA(cudaFree(d_cbias)); CHECK_CUDA(cudaFree(d_hls)); CHECK_CUDA(cudaFree(d_ret_ref)); CHECK_CUDA(cudaFree(d_ret_new));
	CHECK_CUDA(cudaFree(d_wg_ref)); CHECK_CUDA(cudaFree(d_wg_new)); CHECK_CUDA(cudaFree(d_nw));
}

int main(void)
{
	int device = 0, sms = 0;
	cudaDeviceProp prop = {};
	CHECK_CUDA(cudaGetDevice(&device));
	CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
	CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
	g_sms = (uint32_t)sms;
	printf("k3_p4_blackwell_bench: %s (sm_%d%d, %u SMs)\n", prop.name, prop.major, prop.minor, g_sms);

	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreate(&stream));
	BenchSuite1Moe(stream);
	BenchSuite2Indexer(stream);
	BenchSuite3FusedPrefillGemm(stream);
	BenchSuite4MlaPrefill(stream);
	BenchSuite5Elementwise(stream);
	CHECK_CUDA(cudaStreamDestroy(stream));

	if ( g_failures != 0 )
	{
		printf("\nFAIL total failures=%d\n", g_failures);
		return 1;
	}
	printf("\nALL PASS (0 failures)\n");
	return 0;
}
