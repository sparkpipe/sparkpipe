#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mx_gemm.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_kernels.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"

#define TEST_CHECK(...) do { \
	cudaError_t _err = (__VA_ARGS__); \
	if ( _err != cudaSuccess ) { \
		fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(_err)); \
		exit(1); \
	} \
} while ( 0 )

static uint32_t TestNext(uint32_t *state)
{
	uint32_t x = *state;
	x ^= x << 13u;
	x ^= x >> 17u;
	x ^= x << 5u;
	*state = x;
	return x;
}

static uint16_t TestRandBf16(uint32_t *state, float scale)
{
	const float u = ((float)(TestNext(state) & 0xffffu) / 32768.0f) - 1.0f;
	const float v = u * scale;
	uint32_t bits;
	memcpy(&bits, &v, sizeof(bits));
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static __global__ void TestExpertActRefKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ group_offset, uint32_t groups, uint8_t *__restrict__ codes, uint16_t *__restrict__ act,
	uint32_t routes, uint32_t width, float limit)
{
	const uint32_t total = group_offset[groups];
	for ( uint32_t route = blockIdx.y; route < total; route += gridDim.y )
	{
		const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
		if ( element >= width ) return;
		const uint16_t value = LmSwigluPackedValue(gate_up[(uint64_t)route * 2u * width + element],
			gate_up[(uint64_t)route * 2u * width + width + element], packed_weight[route], limit);
		if ( act != 0 )
		{
			act[(uint64_t)route * width + element] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(value));
			return;
		}
		uint8_t exponent;
		const uint8_t code = D41QuantFp8Block(value, &exponent);
		codes[(uint64_t)route * width + element] = code;
		if ( element % SPARK_DSV41_FLASH_FP8_BLOCK == 0u )
			codes[(uint64_t)routes * width + (uint64_t)route * (width / SPARK_DSV41_FLASH_FP8_BLOCK) + element / SPARK_DSV41_FLASH_FP8_BLOCK] = exponent;
	}
}

template<uint32_t THREADS>
static __global__ void TestRmsNormMxKernel(const uint16_t *input, const uint16_t *streams, const float *pre,
	const uint16_t *weight, uint16_t *output, uint16_t *quantized, uint32_t dimension, float epsilon,
	uint8_t *mx_codes, uint8_t *mx_scales)
{
	extern __shared__ __align__(16) float norm_shared[];
	const uint32_t row = blockIdx.x;
	const uint64_t base = (uint64_t)row * dimension;
	for ( uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u )
	{
		float v[8];
		if ( streams != 0 )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k ) v[k] = 0.0f;
			#pragma unroll
			for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
			{
				const float pw = pre[(uint64_t)row * D41K_HC + copy];
				float s[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(streams + ((uint64_t)row * D41K_HC + copy) * dimension + offset), s);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k ) v[k] += pw * s[k];
			}
			LmLoadBf16x8AsFloat(LmPackUint4FloatToBf16(v), v);
		}
		else
			LmLoadBf16x8AsFloat(*(const uint4 *)(input + base + offset), v);
		LmStoreFloat8(norm_shared + offset, v);
	}
	__syncthreads();
	float total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS ) { const float val = norm_shared[index]; total += val * val; }
	const float scale = SparkMathRmsScale(LmBlockSum<THREADS>(total, norm_shared + dimension), (float)dimension, epsilon);
	for ( uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u )
	{
		float v[8], w[8];
		LmLoadFloat8(norm_shared + offset, v);
		LmLoadBf16x8AsFloat(*(const uint4 *)(weight + offset), w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) v[k] = v[k] * scale * w[k];
		const uint4 normed = LmPackUint4FloatToBf16(v);
		if ( output != 0 ) *(uint4 *)(output + base + offset) = normed;
		LmLoadBf16x8AsFloat(normed, v);
		if ( mx_codes != 0 )
		{
			uint2 packed;
			uint8_t sc;
			D41Fp8Quant8(v, &packed, &sc);
			*(uint2 *)(mx_codes + base + offset) = packed;
			if ( (threadIdx.x & 3u) == 0u ) mx_scales[(base + offset) >> 5u] = sc;
		}
		else
		{
			for ( uint32_t k = 0u; k < 8u; ++k ) quantized[base + offset + k] = LmFloatToBf16(v[k]);
		}
	}
}

using TestKvGeo = LmKvGeometry<1280u, 128u, true>;

template<bool ROUND_BF16>
static void TestIndexMxfp4Case(uint32_t rows, uint32_t stride, uint32_t degree, uint32_t rank, bool use_mask, uint32_t *rng)
{
	const uint32_t heads = 32u, dim = 128u, grain = 8u;
	const uint32_t block_stride = stride / grain;
	const uint32_t pages = (stride + 127u) / 128u;
	const uint64_t slot_bytes = 1280u;
	const uint64_t kv_bytes = (uint64_t)pages * 128u * slot_bytes;
	std::vector<uint16_t> h_q((size_t)rows * heads * dim), h_w((size_t)rows * heads);
	std::vector<uint8_t> h_kv(kv_bytes, 0u);
	for ( size_t i = 0; i < h_q.size(); ++i ) h_q[i] = TestRandBf16(rng, 2.5f);
	for ( size_t i = 0; i < h_w.size(); ++i ) h_w[i] = TestRandBf16(rng, 0.5f);
	for ( uint32_t p = 0u; p < stride; ++p )
	{
		uint16_t *slot = (uint16_t *)(h_kv.data() + (uint64_t)p * slot_bytes + 1024u);
		for ( uint32_t d = 0u; d < dim; ++d ) slot[d] = TestRandBf16(rng, 2.5f);
	}
	uint16_t *d_q = 0, *d_w = 0;
	uint8_t *d_kv = 0, *d_mask = 0;
	uint32_t *d_pt = 0, *d_seq = 0, *d_len = 0, *d_bound = 0;
	uint32_t *d_kc = 0, *d_ks = 0, *d_qc = 0;
	uint8_t *d_qs = 0;
	float *d_ref = 0, *d_got = 0;
	TEST_CHECK(cudaMalloc(&d_q, h_q.size() * 2u));
	TEST_CHECK(cudaMalloc(&d_w, h_w.size() * 2u));
	TEST_CHECK(cudaMalloc(&d_kv, kv_bytes));
	TEST_CHECK(cudaMalloc(&d_pt, pages * 4u));
	TEST_CHECK(cudaMalloc(&d_seq, rows * 4u));
	TEST_CHECK(cudaMalloc(&d_len, 4u));
	TEST_CHECK(cudaMalloc(&d_bound, rows * 4u));
	TEST_CHECK(cudaMalloc(&d_kc, (size_t)stride * 64u));
	TEST_CHECK(cudaMalloc(&d_ks, (size_t)stride * 4u));
	TEST_CHECK(cudaMalloc(&d_qc, (size_t)rows * heads * 64u));
	TEST_CHECK(cudaMalloc(&d_qs, (size_t)rows * heads * 4u));
	TEST_CHECK(cudaMalloc(&d_ref, (size_t)rows * stride * 4u));
	TEST_CHECK(cudaMalloc(&d_got, (size_t)rows * stride * 4u));
	TEST_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 2u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * 2u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_kv, h_kv.data(), kv_bytes, cudaMemcpyHostToDevice));
	SparkDsv41FlashLaunchFp4Pow2Qdq(0, d_q, (uint64_t)rows * heads * (dim / 32u), 32u);
	for ( uint32_t p = 0u; p < stride; ++p )
		SparkDsv41FlashLaunchFp4Pow2Qdq(0, (uint16_t *)(d_kv + (uint64_t)p * slot_bytes + 1024u), dim / 32u, 32u);
	std::vector<uint32_t> h_pt(pages), h_seq(rows, 0u), h_bound(rows);
	for ( uint32_t i = 0u; i < pages; ++i ) h_pt[i] = i;
	const uint32_t full_len = stride * degree;
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		if ( r == 0u ) h_bound[r] = 0u;
		else if ( r == 1u ) h_bound[r] = 17u * degree;
		else if ( r == 2u ) h_bound[r] = D41K_NO_ROW;
		else if ( r == 3u ) h_bound[r] = (stride - 5u) * degree;
		else h_bound[r] = (uint32_t)((uint64_t)(r + 1u) * (full_len - 1u) / rows);
	}
	TEST_CHECK(cudaMemcpy(d_pt, h_pt.data(), pages * 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_seq, h_seq.data(), rows * 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_len, &full_len, 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_bound, h_bound.data(), rows * 4u, cudaMemcpyHostToDevice));
	if ( use_mask )
	{
		std::vector<uint8_t> h_mask((size_t)rows * block_stride);
		for ( size_t i = 0; i < h_mask.size(); ++i ) h_mask[i] = (TestNext(rng) & 3u) == 0u ? 1u : 0u;
		TEST_CHECK(cudaMalloc(&d_mask, h_mask.size()));
		TEST_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size(), cudaMemcpyHostToDevice));
	}
	LmKvAccessError *d_err = 0;
	TEST_CHECK(cudaMalloc(&d_err, sizeof(LmKvAccessError)));
	TEST_CHECK(cudaMemset(d_err, 0, sizeof(LmKvAccessError)));
	LmKvShardView view;
	memset(&view, 0, sizeof(view));
	view.pages.pool = d_kv + 1024u;
	view.pages.page_table = d_pt;
	view.pages.page_table_stride = pages;
	view.pages.sequence_count = 1u;
	view.pages.pool_page_count = pages;
	view.pages.access_error = d_err;
	view.shard.degree = degree;
	view.shard.rank = rank;
	view.shard.grain = grain;
	TEST_CHECK(LmWeightedSparseScoreLaunch<TestKvGeo, 32u, 128u, LmKvShardView, ROUND_BF16>(
		d_q, d_w, view, d_seq, d_len, d_bound, rows, stride, 1.0f, d_ref, 0));
	if ( use_mask )
		TEST_CHECK(SparkDsv41FlashLaunchShardBlockApply(0, d_ref, stride, grain, d_mask, block_stride, rows));
	D41PackIndexKeysMxfp4Kernel<TestKvGeo><<<(uint32_t)(((uint64_t)stride * 16u + 255u) / 256u), 256u>>>(view, d_seq, d_len, stride, d_kc, d_ks);
	const uint64_t q_octets = (uint64_t)rows * heads * (dim / 8u);
	D41PackIndexQueriesMxfp4Kernel<<<(uint32_t)((q_octets + 255u) / 256u), 256u>>>(d_q, d_qc, d_qs, q_octets);
	D41IndexScoreMxfp4Kernel<ROUND_BF16><<<dim3((stride + 255u) / 256u, (rows + 7u) / 8u), 256u>>>(
		d_qc, (const uint32_t *)d_qs, d_w, d_kc, d_ks, view.shard, d_seq, d_len, d_bound, d_mask, block_stride, rows, stride, d_got);
	TEST_CHECK(cudaDeviceSynchronize());
	std::vector<uint32_t> hr((size_t)rows * stride), hg((size_t)rows * stride);
	TEST_CHECK(cudaMemcpy(hr.data(), d_ref, hr.size() * 4u, cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(hg.data(), d_got, hg.size() * 4u, cudaMemcpyDeviceToHost));
	uint64_t diffs = 0u;
	for ( size_t i = 0; i < hr.size(); ++i )
		if ( hr[i] != hg[i] )
		{
			if ( diffs < 3u )
			{
				float fr, fg;
				memcpy(&fr, &hr[i], 4u);
				memcpy(&fg, &hg[i], 4u);
				fprintf(stderr, "MISMATCH index deg=%u rbf16=%d mask=%d at %zu: ref=%g (0x%08x) got=%g (0x%08x)\n",
					degree, (int)ROUND_BF16, (int)use_mask, i, fr, hr[i], fg, hg[i]);
			}
			diffs++;
		}
	if ( diffs != 0u ) exit(1);
	printf("PASS D41IndexScoreMxfp4Kernel<rbf16=%d> rows=%u stride=%u deg=%u mask=%d: 0 of %zu floats differ\n",
		(int)ROUND_BF16, rows, stride, degree, (int)use_mask, hr.size());
	cudaFree(d_q); cudaFree(d_w); cudaFree(d_kv); cudaFree(d_pt); cudaFree(d_seq); cudaFree(d_len); cudaFree(d_bound);
	cudaFree(d_kc); cudaFree(d_ks); cudaFree(d_qc); cudaFree(d_qs); cudaFree(d_ref); cudaFree(d_got); cudaFree(d_err);
	if ( d_mask ) cudaFree(d_mask);
}

static void TestWholeBlockMaskAndDirectTopk(uint32_t *rng)
{
	const uint32_t rows = 64u, stride = 4096u, k = 2048u, bstride = stride / 8u, keep = 128u;
	std::vector<float> h_scores((size_t)rows * stride);
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		const uint32_t valid = r == 0u ? 13u : r == 1u ? 0u : r == 2u ? 1900u : r == 3u ? 2048u : stride;
		for ( uint32_t c = 0u; c < stride; ++c )
		{
			if ( c >= valid || (r == 4u && (c % 3u) != 0u) ) h_scores[(size_t)r * stride + c] = -INFINITY;
			else if ( r == 5u ) h_scores[(size_t)r * stride + c] = (float)((c / 16u) % 5u);
			else h_scores[(size_t)r * stride + c] = ((float)(TestNext(rng) & 0xffffu) / 1000.0f) - 32.0f;
		}
	}
	float *d_scores = 0, *d_bscores = 0, *d_tv = 0;
	uint32_t *d_tp = 0, *d_sel_ref = 0, *d_sel_got = 0, *d_local = 0, *d_bchosen = 0, *d_bsel = 0;
	uint2 *d_cand = 0;
	uint8_t *d_mask_ref = 0, *d_mask_got = 0;
	const uint64_t scratch_entries = (uint64_t)rows * 4096u * 2u;
	TEST_CHECK(cudaMalloc(&d_scores, h_scores.size() * 4u));
	TEST_CHECK(cudaMalloc(&d_bscores, (size_t)rows * bstride * 4u));
	TEST_CHECK(cudaMalloc(&d_tv, scratch_entries * 4u));
	TEST_CHECK(cudaMalloc(&d_tp, scratch_entries * 4u));
	TEST_CHECK(cudaMalloc(&d_sel_ref, (size_t)rows * k * 4u));
	TEST_CHECK(cudaMalloc(&d_sel_got, (size_t)rows * k * 4u));
	TEST_CHECK(cudaMalloc(&d_local, (size_t)rows * k * 4u));
	TEST_CHECK(cudaMalloc(&d_bchosen, (size_t)rows * keep * 4u));
	TEST_CHECK(cudaMalloc(&d_bsel, (size_t)rows * keep * 4u));
	TEST_CHECK(cudaMalloc(&d_cand, (size_t)rows * k * sizeof(uint2)));
	TEST_CHECK(cudaMalloc(&d_mask_ref, (size_t)rows * bstride));
	TEST_CHECK(cudaMalloc(&d_mask_got, (size_t)rows * bstride));
	TEST_CHECK(cudaMemcpy(d_scores, h_scores.data(), h_scores.size() * 4u, cudaMemcpyHostToDevice));
	SparkKvShard shard = { 1u, 0u, 8u };
	TEST_CHECK(LmTopkExactLaunch<256u>(d_scores, rows, stride, k, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, d_tv, d_tp, scratch_entries, d_local, 0));
	TEST_CHECK(LmIndexShardCandidatePackLaunch<256u>(shard, d_scores, stride, d_local, k, rows, d_cand, 0));
	TEST_CHECK(LmIndexShardCandidateTopkLaunch<256u, 2048u>(d_cand, (uint64_t)rows * k, 1u, k, rows, stride, k, 1u, d_sel_ref, 0));
	TEST_CHECK(LmTopkExactLaunch<256u>(d_scores, rows, stride, k, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, d_tv, d_tp, scratch_entries, d_sel_got, 0));
	std::vector<float> h_bscores((size_t)rows * bstride);
	for ( size_t i = 0; i < h_bscores.size(); ++i ) h_bscores[i] = (i % 7u == 0u) ? -INFINITY : ((float)(TestNext(rng) & 0xfffu) / 100.0f);
	TEST_CHECK(cudaMemcpy(d_bscores, h_bscores.data(), h_bscores.size() * 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(LmTopkExactLaunch<256u>(d_bscores, rows, bstride, keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, d_tv, d_tp, scratch_entries, d_bchosen, 0));
	SparkKvShard bshard = { 1u, 0u, 1u };
	TEST_CHECK(LmIndexShardCandidatePackLaunch<256u>(bshard, d_bscores, bstride, d_bchosen, keep, rows, d_cand, 0));
	TEST_CHECK(LmIndexShardCandidateTopkLaunch<256u, 2048u>(d_cand, (uint64_t)rows * keep, 1u, keep, rows, bstride, keep, 0u, d_bsel, 0));
	TEST_CHECK(SparkDsv41FlashLaunchShardBlockMask(0, bstride, d_bsel, keep, bshard, d_mask_ref, bstride, rows));
	D41WholeBlockMaskKernel<<<rows, 256u>>>(d_bscores, d_bchosen, keep, d_mask_got, bstride);
	TEST_CHECK(cudaDeviceSynchronize());
	std::vector<uint32_t> sr((size_t)rows * k), sg((size_t)rows * k);
	std::vector<uint8_t> mr((size_t)rows * bstride), mg((size_t)rows * bstride);
	TEST_CHECK(cudaMemcpy(sr.data(), d_sel_ref, sr.size() * 4u, cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(sg.data(), d_sel_got, sg.size() * 4u, cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(mr.data(), d_mask_ref, mr.size(), cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(mg.data(), d_mask_got, mg.size(), cudaMemcpyDeviceToHost));
	for ( size_t i = 0; i < sr.size(); ++i )
		if ( sr[i] != sg[i] ) { fprintf(stderr, "MISMATCH direct topk at %zu: ref=%u got=%u\n", i, sr[i], sg[i]); exit(1); }
	for ( size_t i = 0; i < mr.size(); ++i )
		if ( mr[i] != mg[i] ) { fprintf(stderr, "MISMATCH whole block mask at %zu: ref=%u got=%u\n", i, mr[i], mg[i]); exit(1); }
	printf("PASS D41SelectWhole direct topk (%zu) + D41WholeBlockMaskKernel (%zu): 0 diffs\n", sr.size(), mr.size());
	cudaFree(d_scores); cudaFree(d_bscores); cudaFree(d_tv); cudaFree(d_tp); cudaFree(d_sel_ref); cudaFree(d_sel_got);
	cudaFree(d_local); cudaFree(d_bchosen); cudaFree(d_bsel); cudaFree(d_cand); cudaFree(d_mask_ref); cudaFree(d_mask_got);
}

static void TestFusedMxKernels(uint32_t *rng)
{
	const uint32_t rows = 64u, hidden = 5120u, qlora = 1536u, inter = 2304u, sw = 640u, lw = 2048u, routes = 256u;
	std::vector<uint16_t> h_fp4((size_t)rows * 8192u);
	for ( size_t i = 0; i < h_fp4.size(); ++i ) h_fp4[i] = TestRandBf16(rng, 4.0f);
	uint16_t *d_a = 0, *d_b = 0;
	TEST_CHECK(cudaMalloc(&d_a, h_fp4.size() * 2u));
	TEST_CHECK(cudaMalloc(&d_b, h_fp4.size() * 2u));
	TEST_CHECK(cudaMemcpy(d_a, h_fp4.data(), h_fp4.size() * 2u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(d_b, h_fp4.data(), h_fp4.size() * 2u, cudaMemcpyHostToDevice));
	const uint64_t groups = h_fp4.size() / 32u;
	SparkDsv41FlashFp4Pow2QdqKernel<<<(uint32_t)((groups + 255u) / 256u), 256u>>>(d_a, groups, 32u);
	SparkDsv41FlashLaunchFp4Pow2Qdq(0, d_b, groups, 32u);
	TEST_CHECK(cudaDeviceSynchronize());
	std::vector<uint16_t> r1(h_fp4.size()), g1(h_fp4.size());
	TEST_CHECK(cudaMemcpy(r1.data(), d_a, r1.size() * 2u, cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(g1.data(), d_b, g1.size() * 2u, cudaMemcpyDeviceToHost));
	if ( memcmp(r1.data(), g1.data(), r1.size() * 2u) != 0 ) { fprintf(stderr, "MISMATCH Fp4Pow2Qdq\n"); exit(1); }
	printf("PASS SparkDsv41FlashLaunchFp4Pow2Qdq: 0 of %zu values differ\n", r1.size());
	cudaFree(d_a); cudaFree(d_b);

	{
		std::vector<uint16_t> h_in((size_t)rows * 4u * hidden), h_w(hidden);
		std::vector<float> h_pre((size_t)rows * 4u);
		for ( size_t i = 0; i < h_in.size(); ++i ) h_in[i] = TestRandBf16(rng, 2.0f);
		for ( size_t i = 0; i < h_w.size(); ++i ) h_w[i] = TestRandBf16(rng, 1.5f);
		for ( size_t i = 0; i < h_pre.size(); ++i ) h_pre[i] = ((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f;
		uint16_t *d_in = 0, *d_w = 0, *d_out1 = 0, *d_out2 = 0, *d_q = 0;
		float *d_pre = 0;
		uint8_t *d_c1 = 0, *d_s1 = 0, *d_c2 = 0, *d_s2 = 0;
		TEST_CHECK(cudaMalloc(&d_in, h_in.size() * 2u));
		TEST_CHECK(cudaMalloc(&d_w, hidden * 2u));
		TEST_CHECK(cudaMalloc(&d_pre, h_pre.size() * 4u));
		TEST_CHECK(cudaMalloc(&d_out1, (size_t)rows * hidden * 2u));
		TEST_CHECK(cudaMalloc(&d_out2, (size_t)rows * hidden * 2u));
		TEST_CHECK(cudaMalloc(&d_q, (size_t)rows * hidden * 2u));
		TEST_CHECK(cudaMalloc(&d_c1, (size_t)rows * hidden));
		TEST_CHECK(cudaMalloc(&d_s1, (size_t)rows * (hidden / 32u)));
		TEST_CHECK(cudaMalloc(&d_c2, (size_t)rows * hidden));
		TEST_CHECK(cudaMalloc(&d_s2, (size_t)rows * (hidden / 32u)));
		TEST_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(d_w, h_w.data(), hidden * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(d_pre, h_pre.data(), h_pre.size() * 4u, cudaMemcpyHostToDevice));
		TestRmsNormMxKernel<256u><<<rows, 256u, (hidden + 8u) * 4u>>>(0, d_in, d_pre, d_w, d_out1, d_q, hidden, 1e-6f, 0, 0);
		if ( LmMxQuantize(d_q, d_c1, d_s1, (uint64_t)rows * hidden, 0) != LM_LAUNCH_OK ) exit(1);
		TestRmsNormMxKernel<256u><<<rows, 256u, (hidden + 8u) * 4u>>>(0, d_in, d_pre, d_w, d_out2, 0, hidden, 1e-6f, d_c2, d_s2);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> c1((size_t)rows * hidden), c2((size_t)rows * hidden), s1((size_t)rows * hidden / 32u), s2((size_t)rows * hidden / 32u);
		std::vector<uint16_t> o1((size_t)rows * hidden), o2((size_t)rows * hidden);
		TEST_CHECK(cudaMemcpy(c1.data(), d_c1, c1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), d_c2, c2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), d_s1, s1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), d_s2, s2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(o1.data(), d_out1, o1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(o2.data(), d_out2, o2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( memcmp(o1.data(), o2.data(), o1.size() * 2u) != 0 || memcmp(c1.data(), c2.data(), c1.size()) != 0 || memcmp(s1.data(), s2.data(), s1.size()) != 0 )
		{ fprintf(stderr, "MISMATCH CollapseNormMx\n"); exit(1); }
		TestRmsNormMxKernel<256u><<<rows, 256u, (qlora + 8u) * 4u>>>(d_in, 0, 0, d_w, 0, d_q, qlora, 1e-6f, 0, 0);
		if ( LmMxQuantize(d_q, d_c1, d_s1, (uint64_t)rows * qlora, 0) != LM_LAUNCH_OK ) exit(1);
		TestRmsNormMxKernel<256u><<<rows, 256u, (qlora + 8u) * 4u>>>(d_in, 0, 0, d_w, 0, 0, qlora, 1e-6f, d_c2, d_s2);
		TEST_CHECK(cudaDeviceSynchronize());
		TEST_CHECK(cudaMemcpy(c1.data(), d_c1, (size_t)rows * qlora, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), d_c2, (size_t)rows * qlora, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), d_s1, (size_t)rows * qlora / 32u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), d_s2, (size_t)rows * qlora / 32u, cudaMemcpyDeviceToHost));
		if ( memcmp(c1.data(), c2.data(), (size_t)rows * qlora) != 0 || memcmp(s1.data(), s2.data(), (size_t)rows * qlora / 32u) != 0 )
		{ fprintf(stderr, "MISMATCH RmsNormMx qlora\n"); exit(1); }
		printf("PASS Fused CollapseNorm+MXFP8 (5120) and RmsNorm+MXFP8 (1536): 0 diffs\n");
		cudaFree(d_in); cudaFree(d_w); cudaFree(d_pre); cudaFree(d_out1); cudaFree(d_out2); cudaFree(d_q);
		cudaFree(d_c1); cudaFree(d_s1); cudaFree(d_c2); cudaFree(d_s2);
	}

	{
		for ( uint32_t w : { inter, sw } )
		{
			std::vector<uint16_t> hg((size_t)rows * w), hu((size_t)rows * w);
			for ( size_t i = 0; i < hg.size(); ++i ) { hg[i] = TestRandBf16(rng, 3.0f); hu[i] = TestRandBf16(rng, 3.0f); }
			uint16_t *dg = 0, *du = 0, *da = 0;
			uint8_t *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
			TEST_CHECK(cudaMalloc(&dg, hg.size() * 2u)); TEST_CHECK(cudaMalloc(&du, hu.size() * 2u)); TEST_CHECK(cudaMalloc(&da, hg.size() * 2u));
			TEST_CHECK(cudaMalloc(&dc1, hg.size())); TEST_CHECK(cudaMalloc(&ds1, hg.size() / 32u));
			TEST_CHECK(cudaMalloc(&dc2, hg.size())); TEST_CHECK(cudaMalloc(&ds2, hg.size() / 32u));
			TEST_CHECK(cudaMemcpy(dg, hg.data(), hg.size() * 2u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(du, hu.data(), hu.size() * 2u, cudaMemcpyHostToDevice));
			D41SwigluRowsKernel<<<dim3((w + 255u) / 256u, rows), 256u>>>(dg, du, da, w, 7.0f);
			if ( LmMxQuantize(da, dc1, ds1, (uint64_t)rows * w, 0) != LM_LAUNCH_OK ) exit(1);
			D41SwigluMxKernel<<<dim3(((w / 8u) + 255u) / 256u, rows), 256u>>>(dg, du, dc2, ds2, w, 7.0f);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint8_t> c1(hg.size()), c2(hg.size()), s1(hg.size() / 32u), s2(hg.size() / 32u);
			TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
			if ( memcmp(c1.data(), c2.data(), c1.size()) != 0 || memcmp(s1.data(), s2.data(), s1.size()) != 0 )
			{ fprintf(stderr, "MISMATCH D41SwigluMxKernel w=%u\n", w); exit(1); }
			cudaFree(dg); cudaFree(du); cudaFree(da); cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2);
		}
		printf("PASS D41SwigluMxKernel (2304 & 640): 0 diffs\n");
	}

	{
		const uint32_t ranks = 6u;
		std::vector<uint16_t> h_g((size_t)ranks * rows * lw);
		for ( size_t i = 0; i < h_g.size(); ++i ) h_g[i] = TestRandBf16(rng, 2.0f);
		uint16_t *dg = 0, *drm = 0;
		uint8_t *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
		TEST_CHECK(cudaMalloc(&dg, h_g.size() * 2u));
		TEST_CHECK(cudaMalloc(&drm, (size_t)rows * 2u * lw * 2u));
		TEST_CHECK(cudaMalloc(&dc1, (size_t)rows * 2u * lw)); TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * 2u * lw / 32u));
		TEST_CHECK(cudaMalloc(&dc2, (size_t)rows * 2u * lw)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * 2u * lw / 32u));
		TEST_CHECK(cudaMemcpy(dg, h_g.data(), h_g.size() * 2u, cudaMemcpyHostToDevice));
		LmRankMajorRowsKernel<256u><<<dim3(((2u * lw / 8u) + 255u) / 256u, rows), 256u>>>(dg, drm, rows, lw, 2u, 4u);
		if ( LmMxQuantize(drm, dc1, ds1, (uint64_t)rows * 2u * lw, 0) != LM_LAUNCH_OK ) exit(1);
		D41RankMajorMxKernel<<<dim3(((2u * lw / 8u) + 255u) / 256u, rows), 256u>>>(dg, dc2, ds2, rows, lw, 2u, 4u);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> c1((size_t)rows * 2u * lw), c2((size_t)rows * 2u * lw), s1((size_t)rows * 2u * lw / 32u), s2((size_t)rows * 2u * lw / 32u);
		TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
		if ( memcmp(c1.data(), c2.data(), c1.size()) != 0 || memcmp(s1.data(), s2.data(), s1.size()) != 0 )
		{ fprintf(stderr, "MISMATCH D41RankMajorMxKernel\n"); exit(1); }
		printf("PASS D41RankMajorMxKernel: 0 diffs\n");
		cudaFree(dg); cudaFree(drm); cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2);
	}

	{
		const uint32_t deg = 16u, slice = inter / deg, hslice = hidden / deg;
		const uint64_t stride = (((uint64_t)routes * slice + (uint64_t)rows * hslice) + 7u) / 8u * 8u;
		std::vector<uint16_t> hgu((size_t)routes * 2u * slice), hsh((size_t)rows * hidden);
		std::vector<float> hpw(routes);
		for ( size_t i = 0; i < hgu.size(); ++i ) hgu[i] = TestRandBf16(rng, 2.5f);
		for ( size_t i = 0; i < hsh.size(); ++i ) hsh[i] = TestRandBf16(rng, 2.5f);
		for ( size_t i = 0; i < hpw.size(); ++i ) hpw[i] = 0.25f + (float)(i % 5u) * 0.1f;
		uint16_t *dgu = 0, *dsh = 0, *ds1 = 0, *dact1 = 0, *dact2 = 0;
		float *dpw = 0;
		uint8_t *dc2 = 0;
		TEST_CHECK(cudaMalloc(&dgu, hgu.size() * 2u)); TEST_CHECK(cudaMalloc(&dsh, hsh.size() * 2u));
		TEST_CHECK(cudaMalloc(&dpw, routes * 4u)); TEST_CHECK(cudaMalloc(&ds1, (size_t)deg * stride * 2u));
		TEST_CHECK(cudaMalloc(&dact1, (size_t)routes * inter * 2u)); TEST_CHECK(cudaMalloc(&dact2, (size_t)routes * inter * 2u));
		TEST_CHECK(cudaMalloc(&dc2, (size_t)routes * (inter + inter / 32u)));
		TEST_CHECK(cudaMemset(ds1, 0, (size_t)deg * stride * 2u));
		TEST_CHECK(cudaMemcpy(dgu, hgu.data(), hgu.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsh, hsh.data(), hsh.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpw, hpw.data(), routes * 4u, cudaMemcpyHostToDevice));
		D41MoeSendKernel<<<D41ItemGrid((hidden + 255u) / 256u, routes + rows), 256u>>>(
			dgu, dpw, dsh, ds1, routes, rows, slice, hslice, deg, stride, 7.0f, slice, 0u);
		D41ActAssembleQdqKernel<<<D41ItemGrid(inter / 256u, routes), 256u>>>(ds1, dact1, slice, inter, stride, routes, 0u);
		D41ActAssembleMxKernel<<<D41ItemGrid(((inter / 8u) + 255u) / 256u, routes), 256u>>>(ds1, dc2, slice, inter, stride, routes);
		D41SharedDequantKernel<<<dim3(1u, routes), 256u>>>(dc2, dact2, routes, inter);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> a1((size_t)routes * inter), a2((size_t)routes * inter);
		TEST_CHECK(cudaMemcpy(a1.data(), dact1, a1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(a2.data(), dact2, a2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( memcmp(a1.data(), a2.data(), a1.size() * 2u) != 0 ) { fprintf(stderr, "MISMATCH D41ActAssembleMxKernel\n"); exit(1); }
		printf("PASS D41ActAssembleMxKernel (dequant == QDQ): 0 of %zu bf16 values differ\n", a1.size());
		cudaFree(dgu); cudaFree(dsh); cudaFree(dpw); cudaFree(ds1); cudaFree(dact1); cudaFree(dact2); cudaFree(dc2);
	}

	{
		std::vector<uint16_t> hgu((size_t)routes * 2u * inter);
		std::vector<float> hpw(routes);
		uint32_t go[2] = { 0u, routes };
		for ( size_t i = 0; i < hgu.size(); ++i ) hgu[i] = TestRandBf16(rng, 2.5f);
		for ( size_t i = 0; i < hpw.size(); ++i ) hpw[i] = 0.3f + (float)(i % 7u) * 0.05f;
		uint16_t *dgu = 0, *da1 = 0, *da2 = 0;
		float *dpw = 0;
		uint32_t *dgo = 0;
		uint8_t *dc1 = 0, *dc2 = 0;
		const size_t cbytes = (size_t)routes * (inter + inter / 32u);
		TEST_CHECK(cudaMalloc(&dgu, hgu.size() * 2u)); TEST_CHECK(cudaMalloc(&dpw, routes * 4u)); TEST_CHECK(cudaMalloc(&dgo, 8u));
		TEST_CHECK(cudaMalloc(&dc1, cbytes)); TEST_CHECK(cudaMalloc(&dc2, cbytes));
		TEST_CHECK(cudaMalloc(&da1, (size_t)routes * inter * 2u)); TEST_CHECK(cudaMalloc(&da2, (size_t)routes * inter * 2u));
		TEST_CHECK(cudaMemcpy(dgu, hgu.data(), hgu.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpw, hpw.data(), routes * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dgo, go, 8u, cudaMemcpyHostToDevice));
		TestExpertActRefKernel<<<dim3((inter + 255u) / 256u, routes), 256u>>>(dgu, dpw, dgo, 1u, dc1, 0, routes, inter, 7.0f);
		D41ExpertActKernel<<<dim3(((inter / 8u) + 255u) / 256u, routes), 256u>>>(dgu, dpw, dgo, 1u, dc2, 0, routes, inter, 7.0f);
		TestExpertActRefKernel<<<dim3((inter + 255u) / 256u, routes), 256u>>>(dgu, dpw, dgo, 1u, 0, da1, routes, inter, 7.0f);
		D41ExpertActKernel<<<dim3(((inter / 8u) + 255u) / 256u, routes), 256u>>>(dgu, dpw, dgo, 1u, 0, da2, routes, inter, 7.0f);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> c1(cbytes), c2(cbytes);
		std::vector<uint16_t> a1((size_t)routes * inter), a2((size_t)routes * inter);
		TEST_CHECK(cudaMemcpy(c1.data(), dc1, cbytes, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), dc2, cbytes, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(a1.data(), da1, a1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(a2.data(), da2, a2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( memcmp(c1.data(), c2.data(), cbytes) != 0 || memcmp(a1.data(), a2.data(), a1.size() * 2u) != 0 )
		{ fprintf(stderr, "MISMATCH D41ExpertActKernel\n"); exit(1); }
		printf("PASS D41ExpertActKernel (MX & BF16): 0 diffs\n");
		cudaFree(dgu); cudaFree(dpw); cudaFree(dgo); cudaFree(dc1); cudaFree(dc2); cudaFree(da1); cudaFree(da2);
	}

	{
		const uint32_t own = 16u, topk = 8u;
		const uint64_t bbytes = ((uint64_t)own * (hidden + hidden / 32u + 8u * topk) + 15u) / 16u * 16u;
		std::vector<uint16_t> hx((size_t)rows * hidden);
		std::vector<uint32_t> hre((size_t)rows * topk);
		std::vector<float> hrw((size_t)rows * topk);
		for ( size_t i = 0; i < hx.size(); ++i ) hx[i] = TestRandBf16(rng, 2.0f);
		for ( size_t i = 0; i < hre.size(); ++i ) { hre[i] = TestNext(rng) % 384u; hrw[i] = 0.125f; }
		uint16_t *dx = 0, *dxq = 0;
		uint32_t *dre_in = 0, *dre1 = 0, *dre2 = 0;
		float *drw_in = 0, *drw1 = 0, *drw2 = 0;
		uint8_t *drec = 0, *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
		TEST_CHECK(cudaMalloc(&dx, hx.size() * 2u)); TEST_CHECK(cudaMalloc(&dxq, hx.size() * 2u));
		TEST_CHECK(cudaMalloc(&dre_in, hre.size() * 4u)); TEST_CHECK(cudaMalloc(&drw_in, hrw.size() * 4u));
		TEST_CHECK(cudaMalloc(&dre1, hre.size() * 4u)); TEST_CHECK(cudaMalloc(&drw1, hrw.size() * 4u));
		TEST_CHECK(cudaMalloc(&dre2, hre.size() * 4u)); TEST_CHECK(cudaMalloc(&drw2, hrw.size() * 4u));
		TEST_CHECK(cudaMalloc(&drec, 4u * bbytes));
		TEST_CHECK(cudaMalloc(&dc1, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * hidden / 32u));
		TEST_CHECK(cudaMalloc(&dc2, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * hidden / 32u));
		TEST_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dre_in, hre.data(), hre.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(drw_in, hrw.data(), hrw.size() * 4u, cudaMemcpyHostToDevice));
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			uint8_t *blk = drec + s * bbytes;
			if ( LmMxQuantize(dx + (uint64_t)s * own * hidden, blk, blk + (uint64_t)own * hidden, (uint64_t)own * hidden, 0) != LM_LAUNCH_OK ) exit(1);
			D41FfnPackKernel<<<(own * topk + 255u) / 256u, 256u>>>(dre_in + s * own * topk, drw_in + s * own * topk, blk, own, own, hidden, topk);
		}
		if ( LmMxQuantize(dx, dc1, ds1, (uint64_t)rows * hidden, 0) != LM_LAUNCH_OK ) exit(1);
		D41FfnUnpackKernel<<<dim3((hidden / 8u + 255u) / 256u, rows), 256u>>>(drec, dxq, 0, 0, dre1, drw1, own, hidden, topk, bbytes);
		D41FfnUnpackKernel<<<dim3((hidden / 16u + 255u) / 256u, rows), 256u>>>(drec, 0, dc2, ds2, dre2, drw2, own, hidden, topk, bbytes);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> c1((size_t)rows * hidden), c2((size_t)rows * hidden), s1((size_t)rows * hidden / 32u), s2((size_t)rows * hidden / 32u);
		std::vector<uint32_t> re1(hre.size()), re2(hre.size());
		TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(re1.data(), dre1, re1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(re2.data(), dre2, re2.size() * 4u, cudaMemcpyDeviceToHost));
		if ( memcmp(c1.data(), c2.data(), c1.size()) != 0 || memcmp(s1.data(), s2.data(), s1.size()) != 0 || memcmp(re1.data(), re2.data(), re1.size() * 4u) != 0 )
		{ fprintf(stderr, "MISMATCH D41FfnUnpackKernel MX mode\n"); exit(1); }
		printf("PASS D41FfnUnpackKernel MX direct copy: 0 diffs\n");
		cudaFree(dx); cudaFree(dxq); cudaFree(dre_in); cudaFree(drw_in); cudaFree(dre1); cudaFree(drw1);
		cudaFree(dre2); cudaFree(drw2); cudaFree(drec); cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2);
	}

	{
		const uint32_t deg = 4u;
		const uint64_t stride = (uint64_t)rows * (sw / 2u + sw / 64u);
		std::vector<uint16_t> hg((size_t)rows * sw), hu((size_t)rows * sw);
		for ( size_t i = 0; i < hg.size(); ++i ) { hg[i] = TestRandBf16(rng, 2.5f); hu[i] = TestRandBf16(rng, 2.5f); }
		uint16_t *dg = 0, *du = 0, *da = 0, *ds = 0;
		uint8_t *dc = 0, *dsc = 0;
		TEST_CHECK(cudaMalloc(&dg, hg.size() * 2u)); TEST_CHECK(cudaMalloc(&du, hu.size() * 2u));
		TEST_CHECK(cudaMalloc(&da, hg.size() * 2u)); TEST_CHECK(cudaMalloc(&ds, (size_t)deg * stride * 2u));
		TEST_CHECK(cudaMalloc(&dc, hg.size())); TEST_CHECK(cudaMalloc(&dsc, hg.size() / 32u));
		TEST_CHECK(cudaMemcpy(dg, hg.data(), hg.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(du, hu.data(), hu.size() * 2u, cudaMemcpyHostToDevice));
		D41SwigluRowsKernel<<<dim3((sw + 255u) / 256u, rows), 256u>>>(dg, du, da, sw, 7.0f);
		if ( LmMxQuantize(da, dc, dsc, (uint64_t)rows * sw, 0) != LM_LAUNCH_OK ) exit(1);
		D41SharedSendKernel<<<dim3(((sw / 8u) + 255u) / 256u, rows), 256u>>>(dg, du, ds, rows, sw, deg, stride, 7.0f);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> rc(hg.size()), rs(hg.size() / 32u), got((size_t)deg * stride * 2u);
		TEST_CHECK(cudaMemcpy(rc.data(), dc, rc.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(rs.data(), dsc, rs.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(got.data(), ds, got.size(), cudaMemcpyDeviceToHost));
		for ( uint32_t d = 0u; d < deg; ++d )
		{
			const uint8_t *blk = got.data() + (size_t)d * stride * 2u;
			if ( memcmp(blk, rc.data(), rc.size()) != 0 || memcmp(blk + rc.size(), rs.data(), rs.size()) != 0 )
			{ fprintf(stderr, "MISMATCH D41SharedSendKernel rank=%u\n", d); exit(1); }
		}
		printf("PASS D41SharedSendKernel (fused SwiGLU+MXFP8 x 4 ranks): 0 diffs\n");
		cudaFree(dg); cudaFree(du); cudaFree(da); cudaFree(ds); cudaFree(dc); cudaFree(dsc);
	}
}

static void TestSpreadOutputAndSplitSkinny(uint32_t *rng)
{
	const uint32_t row_cases[] = { 1u, 8u, 17u, 32u, 64u };
	const uint32_t heads = 4u;
	std::vector<float> hfreq(D41K_ROPE_PAIRS);
	for ( uint32_t i = 0u; i < D41K_ROPE_PAIRS; ++i )
		hfreq[i] = powf(10000.0f, -2.0f * (float)i / (float)(2u * D41K_ROPE_PAIRS));
	float *dfreq = 0;
	TEST_CHECK(cudaMalloc(&dfreq, hfreq.size() * 4u));
	TEST_CHECK(cudaMemcpy(dfreq, hfreq.data(), hfreq.size() * 4u, cudaMemcpyHostToDevice));
	for ( uint32_t rows : row_cases )
	{
		const uint64_t stride = (uint64_t)rows * heads * D41K_RECORD;
		std::vector<float> hrec(stride), hsink(heads);
		std::vector<uint32_t> hpos(rows);
		for ( uint32_t r = 0u; r < rows; ++r )
		{
			hpos[r] = 50u + r * 3u;
			for ( uint32_t h = 0u; h < heads; ++h )
			{
				float *rec = hrec.data() + ((uint64_t)r * heads + h) * D41K_RECORD;
				rec[0] = ((float)(TestNext(rng) % 400u) / 100.0f) - 2.0f;
				rec[1] = 0.5f + (float)(TestNext(rng) % 500u) / 100.0f;
				for ( uint32_t e = 0u; e < D41K_LATENT; ++e )
					rec[2u + e] = ((float)(TestNext(rng) & 0xffffu) / 32768.0f - 1.0f) * 3.0f;
			}
		}
		for ( uint32_t h = 0u; h < heads; ++h )
			hsink[h] = ((float)(TestNext(rng) % 400u) / 100.0f) - 2.0f;
		float *drec = 0, *dsink = 0;
		uint32_t *dpos = 0;
		uint16_t *dref = 0, *dgot = 0;
		const size_t out_n = (size_t)rows * heads * D41K_LATENT;
		TEST_CHECK(cudaMalloc(&drec, stride * 4u)); TEST_CHECK(cudaMalloc(&dsink, heads * 4u));
		TEST_CHECK(cudaMalloc(&dpos, rows * 4u));
		TEST_CHECK(cudaMalloc(&dref, out_n * 2u)); TEST_CHECK(cudaMalloc(&dgot, out_n * 2u));
		TEST_CHECK(cudaMemcpy(drec, hrec.data(), stride * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsink, hsink.data(), heads * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpos, hpos.data(), rows * 4u, cudaMemcpyHostToDevice));
		for ( uint32_t qdq = 0u; qdq < 2u; ++qdq )
		{
			D41MergeSinkKernel<<<dim3(rows, heads), 256u>>>(drec, stride, 1u, heads, dsink, dref);
			D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS>>>(dref, heads, D41K_LATENT, D41K_LATENT - 2u * D41K_ROPE_PAIRS, dpos, 0, dfreq, 1u, 0);
			if ( qdq != 0u )
				TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchFp8Qdq(0, dref, out_n));
			D41SpreadOutputKernel<256u><<<dim3(rows, heads), 256u>>>(drec, stride, heads, dsink, dpos, dfreq, dgot, qdq);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint16_t> r1(out_n), r2(out_n);
			TEST_CHECK(cudaMemcpy(r1.data(), dref, out_n * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(r2.data(), dgot, out_n * 2u, cudaMemcpyDeviceToHost));
			if ( memcmp(r1.data(), r2.data(), out_n * 2u) != 0 )
			{ fprintf(stderr, "MISMATCH D41SpreadOutputKernel rows=%u qdq=%u\n", rows, qdq); exit(1); }
		}
		cudaFree(drec); cudaFree(dsink); cudaFree(dpos); cudaFree(dref); cudaFree(dgot);
	}
	cudaFree(dfreq);
	printf("PASS D41SpreadOutputKernel (rows=1..64, qdq=0/1): 0 diffs\n");

	const uint32_t primary = 24u, backup = 48u, groups = primary + backup, in_dim = 512u, out_dim = 256u;
	const uint64_t gbytes = (uint64_t)(in_dim / LM_SKINNY_CELL_TILE_K) * (out_dim / LM_SKINNY_CELL_NEURONS) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	std::vector<uint8_t> w1((size_t)primary * gbytes), w2((size_t)backup * gbytes);
	for ( auto &b : w1 ) b = (uint8_t)(TestNext(rng) & 0xffu);
	for ( auto &b : w2 ) b = (uint8_t)(TestNext(rng) & 0xffu);
	for ( uint32_t i = 0u; i < primary * (in_dim / 32u) * (out_dim / 16u); ++i )
		for ( uint32_t j = 0u; j < 16u; ++j )
			w1[(size_t)i * LM_SKINNY_CELL_ROWS * 16u + 16u * 16u + j] = (uint8_t)(124u + (TestNext(rng) % 7u));
	for ( uint32_t i = 0u; i < backup * (in_dim / 32u) * (out_dim / 16u); ++i )
		for ( uint32_t j = 0u; j < 16u; ++j )
			w2[(size_t)i * LM_SKINNY_CELL_ROWS * 16u + 16u * 16u + j] = (uint8_t)(124u + (TestNext(rng) % 7u));
	uint8_t *dw1 = 0, *dw2 = 0;
	TEST_CHECK(cudaMalloc(&dw1, w1.size())); TEST_CHECK(cudaMalloc(&dw2, w2.size()));
	TEST_CHECK(cudaMemcpy(dw1, w1.data(), w1.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dw2, w2.data(), w2.size(), cudaMemcpyHostToDevice));
	for ( uint32_t rows : { 1u, 8u, 32u } )
	{
		const uint32_t routes = rows * 6u;
		std::vector<uint16_t> hact((size_t)rows * in_dim);
		for ( auto &v : hact ) v = TestRandBf16(rng, 2.0f);
		std::vector<uint32_t> cnt(groups, 0u), off(groups + 1u, 0u), src(routes, 0u);
		for ( uint32_t r = 0u; r < routes; ++r ) cnt[TestNext(rng) % groups]++;
		for ( uint32_t g = 0u; g < groups; ++g ) off[g + 1u] = off[g] + cnt[g];
		for ( uint32_t r = 0u; r < routes; ++r ) src[r] = TestNext(rng) % rows;
		uint16_t *dact = 0, *dref = 0, *dgot = 0;
		uint32_t *doff = 0, *dsrc = 0;
		TEST_CHECK(cudaMalloc(&dact, hact.size() * 2u));
		TEST_CHECK(cudaMalloc(&dref, (size_t)routes * out_dim * 2u));
		TEST_CHECK(cudaMalloc(&dgot, (size_t)routes * out_dim * 2u));
		TEST_CHECK(cudaMalloc(&doff, off.size() * 4u)); TEST_CHECK(cudaMalloc(&dsrc, src.size() * 4u));
		TEST_CHECK(cudaMemcpy(dact, hact.data(), hact.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(doff, off.data(), off.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsrc, src.data(), src.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemset(dref, 0, (size_t)routes * out_dim * 2u));
		TEST_CHECK(cudaMemset(dgot, 0, (size_t)routes * out_dim * 2u));
		if ( LmSkinnyCellExperts(dw1, dact, dref, doff, dsrc, primary, routes, rows, 0u, in_dim, out_dim, LM_SKINNY_CELL_TILE_K, 0) != LM_LAUNCH_OK ||
			LmSkinnyCellExperts(dw2, dact, dref, doff + primary, dsrc, backup, routes, rows, 0u, in_dim, out_dim, LM_SKINNY_CELL_TILE_K, 0) != LM_LAUNCH_OK ||
			LmSkinnyCellExpertsSplit(dw1, dw2, primary, dact, dgot, doff, dsrc, groups, routes, rows, 0u, in_dim, out_dim, LM_SKINNY_CELL_TILE_K, 0) != LM_LAUNCH_OK )
		{ fprintf(stderr, "LAUNCH FAIL LmSkinnyCellExpertsSplit rows=%u\n", rows); exit(1); }
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> r1((size_t)routes * out_dim), r2((size_t)routes * out_dim);
		TEST_CHECK(cudaMemcpy(r1.data(), dref, r1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(r2.data(), dgot, r2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( memcmp(r1.data(), r2.data(), r1.size() * 2u) != 0 )
		{ fprintf(stderr, "MISMATCH LmSkinnyCellExpertsSplit rows=%u\n", rows); exit(1); }
		cudaFree(dact); cudaFree(dref); cudaFree(dgot); cudaFree(doff); cudaFree(dsrc);
	}
	cudaFree(dw1); cudaFree(dw2);
	printf("PASS LmSkinnyCellExpertsSplit (rows=1/8/32): 0 diffs\n");
}

int main(void)
{
	uint32_t rng = 0x51a7f00du;
	TestIndexMxfp4Case<false>(128u, 4096u, 1u, 0u, false, &rng);
	TestIndexMxfp4Case<false>(128u, 4096u, 1u, 0u, true, &rng);
	TestIndexMxfp4Case<true>(96u, 1024u, 16u, 5u, false, &rng);
	TestIndexMxfp4Case<true>(96u, 1024u, 16u, 5u, true, &rng);
	TestWholeBlockMaskAndDirectTopk(&rng);
	TestFusedMxKernels(&rng);
	TestSpreadOutputAndSplitSkinny(&rng);
	printf("PASS dsv41 prefill opt parity\n");
	return 0;
}
