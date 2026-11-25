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
#include "model-families/dsv41_flash/source/dsv41_flash_attend.cuh"

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

static void TestRows16HillclimbParity(uint32_t *rng)
{
	{
		const uint32_t hidden = 5120u, hc = 4u, flat = hc * hidden, mix = 24u, tasks = 25u;
		std::vector<float> h_fn((size_t)mix * flat), h_sc(3u), h_bs(mix);
		std::vector<uint16_t> h_nw(hidden);
		for ( auto &v : h_fn ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.05f;
		for ( auto &v : h_sc ) v = 0.25f + ((float)(TestNext(rng) & 0xffu) / 512.0f);
		for ( auto &v : h_bs ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.5f;
		for ( auto &v : h_nw ) v = TestRandBf16(rng, 1.5f);
		float *dfn = 0, *dsc = 0, *dbs = 0;
		uint16_t *dnw = 0;
		TEST_CHECK(cudaMalloc(&dfn, h_fn.size() * 4u));
		TEST_CHECK(cudaMalloc(&dsc, 12u));
		TEST_CHECK(cudaMalloc(&dbs, mix * 4u));
		TEST_CHECK(cudaMalloc(&dnw, hidden * 2u));
		TEST_CHECK(cudaMemcpy(dfn, h_fn.data(), h_fn.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsc, h_sc.data(), 12u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbs, h_bs.data(), mix * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dnw, h_nw.data(), hidden * 2u, cudaMemcpyHostToDevice));
		for ( uint32_t rows : { 1u, 6u, 8u, 16u, 32u, 48u, 64u } )
			for ( uint32_t use_mx : { 0u, 1u } )
			{
				std::vector<uint16_t> h_st((size_t)rows * flat);
				std::vector<float> h_cpre((size_t)rows * hc);
				for ( auto &v : h_st ) v = TestRandBf16(rng, 1.5f);
				for ( auto &v : h_cpre ) v = 0.25f + (((float)(TestNext(rng) & 0xffu) / 256.0f) - 0.5f) * 0.4f;
				uint16_t *dst = 0, *dx1 = 0, *dx2 = 0, *dxq1 = 0, *dxq2 = 0;
				uint8_t *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
				float *dcpre = 0, *dpart = 0, *dpr1 = 0, *dpo1 = 0, *dcb1 = 0, *dpr2 = 0, *dpo2 = 0, *dcb2 = 0;
				uint32_t *dcnt = 0;
				TEST_CHECK(cudaMalloc(&dst, h_st.size() * 2u));
				TEST_CHECK(cudaMalloc(&dcpre, h_cpre.size() * 4u));
				TEST_CHECK(cudaMalloc(&dpart, (size_t)rows * tasks * 8u * 4u));
				TEST_CHECK(cudaMalloc(&dpr1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb1, (size_t)rows * hc * hc * 4u));
				TEST_CHECK(cudaMalloc(&dpr2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb2, (size_t)rows * hc * hc * 4u));
				TEST_CHECK(cudaMalloc(&dx1, (size_t)rows * hidden * 2u)); TEST_CHECK(cudaMalloc(&dx2, (size_t)rows * hidden * 2u));
				TEST_CHECK(cudaMalloc(&dxq1, (size_t)rows * hidden * 2u)); TEST_CHECK(cudaMalloc(&dxq2, (size_t)rows * hidden * 2u));
				TEST_CHECK(cudaMalloc(&dc1, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * hidden / 32u));
				TEST_CHECK(cudaMalloc(&dc2, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * hidden / 32u));
				TEST_CHECK(cudaMalloc(&dcnt, (size_t)rows * 4u));
				TEST_CHECK(cudaMemset(dcnt, 0, (size_t)rows * 4u));
				TEST_CHECK(cudaMemcpy(dst, h_st.data(), h_st.size() * 2u, cudaMemcpyHostToDevice));
				TEST_CHECK(cudaMemcpy(dcpre, h_cpre.data(), h_cpre.size() * 4u, cudaMemcpyHostToDevice));
				if ( LmHcSinkhornMixes<4u>(dst, dfn, dsc, dbs, dpart, dpr1, dpo1, dcb1, rows, hidden, 1e-20f, 20u, 1e-6f, 0) != LM_LAUNCH_OK ) exit(1);
				TestRmsNormMxKernel<256u><<<rows, 256u, (hidden + 8u) * 4u>>>(0, dst, dcpre, dnw, dx1, use_mx != 0u ? (uint16_t *)0 : dxq1, hidden, 1e-20f,
					use_mx != 0u ? dc1 : (uint8_t *)0, use_mx != 0u ? ds1 : (uint8_t *)0);
				if ( use_mx == 0u ) TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchFp8Qdq(0, dxq1, (uint64_t)rows * hidden));
				if ( rows == 1u )
					D41HcMixNormKernel<1u, 8u, 2u, 5120u><<<2u * tasks + 1u, 128u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
						use_mx != 0u ? (uint16_t *)0 : dxq2, use_mx != 0u ? dc2 : (uint8_t *)0, use_mx != 0u ? ds2 : (uint8_t *)0, rows, 1e-20f, 20u, 1e-6f);
				else if ( rows <= 8u )
					D41HcMixNormKernel<8u, 4u, 2u, 5120u><<<2u * tasks + rows, 128u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
						use_mx != 0u ? (uint16_t *)0 : dxq2, use_mx != 0u ? dc2 : (uint8_t *)0, use_mx != 0u ? ds2 : (uint8_t *)0, rows, 1e-20f, 20u, 1e-6f);
				else if ( rows <= 32u )
					D41HcMixNormStagedKernel<4u, 5120u><<<((rows + 3u) / 4u) * 8u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
						use_mx != 0u ? (uint16_t *)0 : dxq2, use_mx != 0u ? dc2 : (uint8_t *)0, use_mx != 0u ? ds2 : (uint8_t *)0, rows, 1e-20f, 20u, 1e-6f);
				else
					D41HcMixNormStagedKernel<8u, 5120u><<<((rows + 7u) / 8u) * 8u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
						use_mx != 0u ? (uint16_t *)0 : dxq2, use_mx != 0u ? dc2 : (uint8_t *)0, use_mx != 0u ? ds2 : (uint8_t *)0, rows, 1e-20f, 20u, 1e-6f);
				TEST_CHECK(cudaDeviceSynchronize());
				std::vector<float> pr1((size_t)rows * hc), pr2((size_t)rows * hc), po1((size_t)rows * hc), po2((size_t)rows * hc), cb1((size_t)rows * hc * hc), cb2((size_t)rows * hc * hc);
				std::vector<uint16_t> x1((size_t)rows * hidden), x2((size_t)rows * hidden), xq1((size_t)rows * hidden), xq2((size_t)rows * hidden);
				std::vector<uint8_t> c1((size_t)rows * hidden), c2((size_t)rows * hidden), s1((size_t)rows * hidden / 32u), s2((size_t)rows * hidden / 32u);
				TEST_CHECK(cudaMemcpy(pr1.data(), dpr1, pr1.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(pr2.data(), dpr2, pr2.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(po1.data(), dpo1, po1.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(po2.data(), dpo2, po2.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(cb1.data(), dcb1, cb1.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(cb2.data(), dcb2, cb2.size() * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(x1.data(), dx1, x1.size() * 2u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(x2.data(), dx2, x2.size() * 2u, cudaMemcpyDeviceToHost));
				if ( memcmp(pr1.data(), pr2.data(), pr1.size() * 4u) != 0 || memcmp(po1.data(), po2.data(), po1.size() * 4u) != 0 ||
					memcmp(cb1.data(), cb2.data(), cb1.size() * 4u) != 0 || memcmp(x1.data(), x2.data(), x1.size() * 2u) != 0 )
				{
					fprintf(stderr, "MISMATCH HcMixNorm rows=%u mx=%u pr=%d po=%d cb=%d x=%d\n", rows, use_mx,
						memcmp(pr1.data(), pr2.data(), pr1.size() * 4u) != 0, memcmp(po1.data(), po2.data(), po1.size() * 4u) != 0,
						memcmp(cb1.data(), cb2.data(), cb1.size() * 4u) != 0, memcmp(x1.data(), x2.data(), x1.size() * 2u) != 0);
					for ( size_t i = 0; i < pr1.size(); ++i ) if ( pr1[i] != pr2[i] ) { fprintf(stderr, "  pr[%zu] %.9g vs %.9g\n", i, pr1[i], pr2[i]); break; }
					for ( size_t i = 0; i < x1.size(); ++i ) if ( x1[i] != x2[i] ) { fprintf(stderr, "  x[%zu] 0x%04x vs 0x%04x\n", i, x1[i], x2[i]); break; }
					exit(1);
				}
				if ( use_mx != 0u )
				{
					TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
					if ( memcmp(c1.data(), c2.data(), c1.size()) != 0 || memcmp(s1.data(), s2.data(), s1.size()) != 0 )
					{ fprintf(stderr, "MISMATCH HcMixNorm mx codes/scales rows=%u\n", rows); exit(1); }
				}
				else
				{
					TEST_CHECK(cudaMemcpy(xq1.data(), dxq1, xq1.size() * 2u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(xq2.data(), dxq2, xq2.size() * 2u, cudaMemcpyDeviceToHost));
					if ( memcmp(xq1.data(), xq2.data(), xq1.size() * 2u) != 0 )
					{ fprintf(stderr, "MISMATCH HcMixNorm xq rows=%u\n", rows); exit(1); }
				}
				cudaFree(dst); cudaFree(dcpre); cudaFree(dpart); cudaFree(dpr1); cudaFree(dpo1); cudaFree(dcb1);
				cudaFree(dpr2); cudaFree(dpo2); cudaFree(dcb2); cudaFree(dx1); cudaFree(dx2); cudaFree(dxq1); cudaFree(dxq2);
				cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2); cudaFree(dcnt);
			}
		cudaFree(dfn); cudaFree(dsc); cudaFree(dbs); cudaFree(dnw);
		printf("PASS D41HcMixNormKernel & D41HcMixNormStagedKernel (rows=1..64, qdq/mx): 0 diffs\n");
	}

	{
		float *dfreq = 0;
		TEST_CHECK(cudaMalloc(&dfreq, 2u * D41K_ROPE_PAIRS * sizeof(float)));
		TEST_CHECK((cudaError_t)D41LaunchFrequencies(dfreq, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0));
		const uint32_t heads = 4u, window = 128u;
		const float qk_scale = 1.0f / sqrtf(512.0f);
		for ( uint32_t rows : { 1u, 6u, 8u, 32u, 64u } )
			for ( uint32_t has_c : { 0u, 1u } )
				for ( uint32_t qdq : { 0u, 1u } )
				{
					const uint64_t rstride = (uint64_t)window * D41K_LATENT;
					std::vector<float> hrec((size_t)rows * heads * D41K_RECORD), hsink(heads);
					std::vector<uint16_t> hq((size_t)rows * heads * D41K_LATENT), hwkv((size_t)rows * D41K_LATENT), hring((size_t)rows * rstride);
					std::vector<uint32_t> hpos(rows), hlane(rows), hord(rows), hbeg(rows + 1u), hidx(rows), hslot(rows);
					for ( uint32_t r = 0u; r < rows; ++r )
					{
						hpos[r] = 5u + (r * 37u) % 240u; hlane[r] = r; hord[r] = 0u; hbeg[r] = r; hidx[r] = r; hslot[r] = r;
						for ( uint32_t h = 0u; h < heads; ++h )
						{
							float *rp = hrec.data() + ((size_t)r * heads + h) * D41K_RECORD;
							rp[0] = has_c != 0u ? (((float)(TestNext(rng) & 0xffu) / 64.0f) - 2.0f) : -INFINITY;
							rp[1] = has_c != 0u ? (0.5f + (float)(TestNext(rng) & 0xffu) / 64.0f) : 0.0f;
							for ( uint32_t c = 0u; c < D41K_LATENT; ++c ) rp[2u + c] = has_c != 0u ? (((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f) : 0.0f;
						}
					}
					hbeg[rows] = rows;
					for ( auto &v : hsink ) v = (((float)(TestNext(rng) & 0xffu) / 64.0f) - 2.0f);
					for ( auto &v : hq ) v = TestRandBf16(rng, 1.0f);
					for ( auto &v : hwkv ) v = TestRandBf16(rng, 1.0f);
					for ( auto &v : hring ) v = TestRandBf16(rng, 1.0f);
					float *drec1 = 0, *drec2 = 0, *dsink = 0;
					float2 *drsc = 0, *drsc_c = 0;
					uint16_t *dq = 0, *dwkv = 0, *dring1 = 0, *dring2 = 0, *dout1 = 0, *dout2 = 0;
					uint32_t *dpos = 0, *dlane = 0, *dord = 0, *dbeg = 0, *didx = 0, *dslot = 0;
					TEST_CHECK(cudaMalloc(&drec1, hrec.size() * 4u)); TEST_CHECK(cudaMalloc(&drec2, hrec.size() * 4u));
					TEST_CHECK(cudaMalloc(&dsink, heads * 4u));
					TEST_CHECK(cudaMalloc(&drsc, rows * D41K_ROPE_PAIRS * sizeof(float2))); TEST_CHECK(cudaMalloc(&drsc_c, rows * D41K_ROPE_PAIRS * sizeof(float2)));
					TEST_CHECK(cudaMalloc(&dq, hq.size() * 2u)); TEST_CHECK(cudaMalloc(&dwkv, hwkv.size() * 2u));
					TEST_CHECK(cudaMalloc(&dring1, hring.size() * 2u)); TEST_CHECK(cudaMalloc(&dring2, hring.size() * 2u));
					TEST_CHECK(cudaMalloc(&dout1, hq.size() * 2u)); TEST_CHECK(cudaMalloc(&dout2, hq.size() * 2u));
					TEST_CHECK(cudaMalloc(&dpos, rows * 4u)); TEST_CHECK(cudaMalloc(&dlane, rows * 4u)); TEST_CHECK(cudaMalloc(&dord, rows * 4u));
					TEST_CHECK(cudaMalloc(&dbeg, (rows + 1u) * 4u)); TEST_CHECK(cudaMalloc(&didx, rows * 4u)); TEST_CHECK(cudaMalloc(&dslot, rows * 4u));
					TEST_CHECK(cudaMemcpy(drec1, hrec.data(), hrec.size() * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(drec2, hrec.data(), hrec.size() * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dsink, hsink.data(), heads * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dwkv, hwkv.data(), hwkv.size() * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dring1, hring.data(), hring.size() * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dring2, hring.data(), hring.size() * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dpos, hpos.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dlane, hlane.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dord, hord.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dbeg, hbeg.data(), (rows + 1u) * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(didx, hidx.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dslot, hslot.data(), rows * 4u, cudaMemcpyHostToDevice));
					D41RopeTableKernel<<<rows, D41K_ROPE_PAIRS>>>(dpos, dfreq, drsc, drsc_c, rows);
					D41WindowFoldWholeKernel<<<dim3(rows, heads), D41K_THREADS>>>(drec1, heads, dq, dwkv, dring1, dpos, dlane, dord, dbeg, didx, dslot, rstride, window, qk_scale);
					D41SpreadOutputRingKernel<256u><<<rows * (heads + 1u), 256u>>>(drec1, (uint64_t)rows * heads * D41K_RECORD, heads, dsink, dpos, dfreq, dout1, qdq,
						dwkv, dring1, dlane, dord, dbeg, dslot, rstride, window, window, rows, 1u);
					if ( rows <= 8u )
						D41WindowFoldSpreadClusterKernel<4u><<<dim3(D41K_WINDOW_PARTS, rows), D41K_THREADS>>>(drec2, has_c, dq, dwkv, dring2, dpos, dlane, dord,
							dbeg, didx, dslot, rstride, window, window, qk_scale, dsink, drsc, dout2, qdq, rows, 1u);
					else
						D41WindowFoldSpreadKernel<4u, 2u><<<dim3(rows, 2u), D41K_THREADS>>>(drec2, has_c, dq, dwkv, dring2, dpos, dlane, dord,
							dbeg, didx, dslot, rstride, window, window, qk_scale, dsink, drsc, dout2, qdq, rows, 1u);
					TEST_CHECK(cudaDeviceSynchronize());
					std::vector<uint16_t> o1(hq.size()), o2(hq.size()), g1(hring.size()), g2(hring.size());
					TEST_CHECK(cudaMemcpy(o1.data(), dout1, o1.size() * 2u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(o2.data(), dout2, o2.size() * 2u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(g1.data(), dring1, g1.size() * 2u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(g2.data(), dring2, g2.size() * 2u, cudaMemcpyDeviceToHost));
					if ( memcmp(o1.data(), o2.data(), o1.size() * 2u) != 0 || memcmp(g1.data(), g2.data(), g1.size() * 2u) != 0 )
					{ fprintf(stderr, "MISMATCH WindowFoldSpread rows=%u carried=%u qdq=%u\n", rows, has_c, qdq); exit(1); }
					cudaFree(drec1); cudaFree(drec2); cudaFree(dsink); cudaFree(drsc); cudaFree(drsc_c); cudaFree(dq); cudaFree(dwkv);
					cudaFree(dring1); cudaFree(dring2); cudaFree(dout1); cudaFree(dout2);
					cudaFree(dpos); cudaFree(dlane); cudaFree(dord); cudaFree(dbeg); cudaFree(didx); cudaFree(dslot);
				}
		cudaFree(dfreq);
		printf("PASS D41WindowFoldSpreadKernel & ClusterKernel (rows=1..64, carried=0/1, qdq=0/1): 0 diffs\n");
	}

	{
		const uint32_t ranks = 16u, experts = 384u, local = 24u, topk = 6u, held1 = 73u;
		std::vector<uint8_t> h_own(experts), h_bak(2u * experts), h_grp(experts);
		for ( uint32_t e = 0u; e < experts; ++e )
		{
			h_own[e] = (uint8_t)(e / local);
			h_bak[e] = (uint8_t)((h_own[e] + 1u) % ranks);
			h_bak[experts + e] = (uint8_t)((h_own[e] + 2u) % ranks);
			h_grp[e] = (uint8_t)((e % local) + (h_own[e] == 3u ? 0u : h_bak[e] == 3u ? local : h_bak[experts + e] == 3u ? 2u * local : 3u * local));
		}
		std::vector<float> h_bias(experts);
		for ( auto &v : h_bias ) v = (((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f) * 0.2f;
		uint8_t *down = 0, *dbak = 0, *dgrp = 0;
		float *dbias = 0;
		TEST_CHECK(cudaMalloc(&down, experts)); TEST_CHECK(cudaMalloc(&dbak, 2u * experts));
		TEST_CHECK(cudaMalloc(&dgrp, experts)); TEST_CHECK(cudaMalloc(&dbias, experts * 4u));
		TEST_CHECK(cudaMemcpy(down, h_own.data(), experts, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbak, h_bak.data(), 2u * experts, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dgrp, h_grp.data(), experts, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbias, h_bias.data(), experts * 4u, cudaMemcpyHostToDevice));
		for ( uint32_t rows : { 1u, 8u, 32u, 48u, 64u } )
			for ( uint32_t ri : { 0u, 1u } )
				for ( uint32_t copies : { 1u, 2u } )
				{
					const uint32_t routes = rows * topk;
					std::vector<float> h_log((size_t)rows * experts);
					for ( auto &v : h_log ) v = (((float)(TestNext(rng) & 0xffffu) / 32768.0f) - 1.0f) * 3.0f;
					float *dlog = 0, *dw1 = 0, *dw2 = 0, *dpw1 = 0, *dpw2 = 0;
					uint32_t *de1 = 0, *de2 = 0, *dg1 = 0, *dg2 = 0, *do1 = 0, *do2 = 0, *dp1 = 0, *dp2 = 0, *ds1 = 0, *ds2 = 0, *df1 = 0, *df2 = 0, *dt = 0;
					TEST_CHECK(cudaMalloc(&dlog, h_log.size() * 4u));
					TEST_CHECK(cudaMalloc(&de1, routes * 4u)); TEST_CHECK(cudaMalloc(&de2, routes * 4u));
					TEST_CHECK(cudaMalloc(&dw1, routes * 4u)); TEST_CHECK(cudaMalloc(&dw2, routes * 4u));
					TEST_CHECK(cudaMalloc(&dg1, routes * 4u)); TEST_CHECK(cudaMalloc(&dg2, routes * 4u));
					TEST_CHECK(cudaMalloc(&do1, (held1 + 1u) * 4u)); TEST_CHECK(cudaMalloc(&do2, (held1 + 1u) * 4u));
					TEST_CHECK(cudaMalloc(&dp1, routes * 4u)); TEST_CHECK(cudaMalloc(&dp2, routes * 4u));
					TEST_CHECK(cudaMalloc(&ds1, routes * 4u)); TEST_CHECK(cudaMalloc(&ds2, routes * 4u));
					TEST_CHECK(cudaMalloc(&dpw1, routes * 4u)); TEST_CHECK(cudaMalloc(&dpw2, routes * 4u));
					TEST_CHECK(cudaMalloc(&df1, 4u)); TEST_CHECK(cudaMalloc(&df2, 4u)); TEST_CHECK(cudaMalloc(&dt, (held1 + 1u) * 4u));
					TEST_CHECK(cudaMemset(df1, 0, 4u)); TEST_CHECK(cudaMemset(df2, 0, 4u));
					TEST_CHECK(cudaMemcpy(dlog, h_log.data(), h_log.size() * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK((cudaError_t)LmTopkRouteLaunch<256u, 6u, true, LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, dlog, experts, de1, dw1, dbias, 0, 1.5f, df1, 0));
					D41BalanceRouteKernel<16u, 384u, 24u, 256u, 73u><<<1u, 256u>>>(de1, dw1, dg1, rows, topk, 3u, copies, ri, down, dbak, dgrp, do1, dp1, ds1, 16u, 36u, dt, 40u, dt, dpw1);
					D41TopkBalanceRouteKernel<16u, 384u, 24u, 256u, 73u, 6u, 2u><<<1u, 256u>>>(dlog, dbias, 1.5f, df2, de2, dw2, dg2, rows, 3u, copies, ri, down, dbak, dgrp, do2, dp2, ds2, 16u, 36u, dt, 40u, dt, dpw2);
					TEST_CHECK(cudaDeviceSynchronize());
					std::vector<uint32_t> o1(held1 + 1u), o2(held1 + 1u), p1(routes), p2(routes), s1(routes), s2(routes), e1(routes), e2(routes), g1(routes), g2(routes);
					std::vector<float> pw1(routes), pw2(routes);
					TEST_CHECK(cudaMemcpy(o1.data(), do1, o1.size() * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(o2.data(), do2, o2.size() * 4u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(p1.data(), dp1, routes * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(p2.data(), dp2, routes * 4u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(s1.data(), ds1, routes * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(s2.data(), ds2, routes * 4u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(e1.data(), de1, routes * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(e2.data(), de2, routes * 4u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(g1.data(), dg1, routes * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(g2.data(), dg2, routes * 4u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(pw1.data(), dpw1, routes * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(pw2.data(), dpw2, routes * 4u, cudaMemcpyDeviceToHost));
					if ( o1 != o2 || p1 != p2 || s1 != s2 || e1 != e2 || g1 != g2 || memcmp(pw1.data(), pw2.data(), routes * 4u) != 0 )
					{ fprintf(stderr, "MISMATCH D41TopkBalanceRouteKernel rows=%u ri=%u copies=%u\n", rows, ri, copies); exit(1); }
					cudaFree(dlog); cudaFree(de1); cudaFree(de2); cudaFree(dw1); cudaFree(dw2); cudaFree(dg1); cudaFree(dg2);
					cudaFree(do1); cudaFree(do2); cudaFree(dp1); cudaFree(dp2); cudaFree(ds1); cudaFree(ds2);
					cudaFree(dpw1); cudaFree(dpw2); cudaFree(df1); cudaFree(df2); cudaFree(dt);
				}
		cudaFree(down); cudaFree(dbak); cudaFree(dgrp); cudaFree(dbias);
		printf("PASS D41TopkBalanceRouteKernel (rows=1..64, ri=0/1, copies=1/2): 0 diffs\n");
	}

	{
		const uint32_t rows = 64u, kdim = 512u, n0 = 320u, n1 = 512u;
		std::vector<uint16_t> ha((size_t)rows * kdim);
		std::vector<uint8_t> hw0((size_t)n0 * kdim), hs0((size_t)n0 * kdim / 32u), hw1((size_t)n1 * kdim), hs1((size_t)n1 * kdim / 32u);
		for ( auto &v : ha ) v = TestRandBf16(rng, 2.0f);
		for ( auto &v : hw0 ) v = (uint8_t)(TestNext(rng) & 0x7eu);
		for ( auto &v : hw1 ) v = (uint8_t)(TestNext(rng) & 0x7eu);
		for ( auto &v : hs0 ) v = (uint8_t)(124u + (TestNext(rng) % 5u));
		for ( auto &v : hs1 ) v = (uint8_t)(124u + (TestNext(rng) % 5u));
		uint16_t *da = 0, *dr0 = 0, *dr1 = 0, *dg0 = 0, *dg1 = 0;
		uint8_t *dac = 0, *das = 0, *dw0 = 0, *ds0 = 0, *dw1 = 0, *ds1 = 0;
		float *df32 = 0;
		TEST_CHECK(cudaMalloc(&da, ha.size() * 2u)); TEST_CHECK(cudaMalloc(&dac, ha.size())); TEST_CHECK(cudaMalloc(&das, ha.size() / 32u));
		TEST_CHECK(cudaMalloc(&dw0, hw0.size())); TEST_CHECK(cudaMalloc(&ds0, hs0.size()));
		TEST_CHECK(cudaMalloc(&dw1, hw1.size())); TEST_CHECK(cudaMalloc(&ds1, hs1.size()));
		TEST_CHECK(cudaMalloc(&dr0, (size_t)rows * n0 * 2u)); TEST_CHECK(cudaMalloc(&dr1, (size_t)rows * n1 * 2u));
		TEST_CHECK(cudaMalloc(&dg0, (size_t)rows * n0 * 2u)); TEST_CHECK(cudaMalloc(&dg1, (size_t)rows * n1 * 2u));
		TEST_CHECK(cudaMalloc(&df32, (size_t)rows * n0 * 4u));
		TEST_CHECK(cudaMemcpy(da, ha.data(), ha.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw0, hw0.data(), hw0.size(), cudaMemcpyHostToDevice)); TEST_CHECK(cudaMemcpy(ds0, hs0.data(), hs0.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw1, hw1.data(), hw1.size(), cudaMemcpyHostToDevice)); TEST_CHECK(cudaMemcpy(ds1, hs1.data(), hs1.size(), cudaMemcpyHostToDevice));
		if ( LmMxQuantize(da, dac, das, (uint64_t)rows * kdim, 0) != LM_LAUNCH_OK ||
			LmMxGemmDense(dac, das, dw0, ds0, dr0, rows, kdim, n0, 0u, 0u, 48u, 0) != LM_LAUNCH_OK ||
			LmMxGemmDense(dac, das, dw1, ds1, dr1, rows, kdim, n1, 0u, 0u, 48u, 0) != LM_LAUNCH_OK ) exit(1);
		D41MxGemmSmallKernel<16u, 32u, 1u, 2u, 4u, false><<<dim3(n0 / 32u + n1 / 32u, rows / 16u), 64u>>>(dac, das, dw0, ds0, dg0, n0, dw1, ds1, dg1, n1, rows, kdim);
		D41MxGemmSmallKernel<32u, 64u, 2u, 2u, 3u, true><<<dim3(n0 / 64u, rows / 32u), 128u>>>(dac, das, dw0, ds0, df32, n0, 0, 0, 0, 0u, rows, kdim);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> r0((size_t)rows * n0), g0((size_t)rows * n0), r1((size_t)rows * n1), g1((size_t)rows * n1);
		std::vector<float> gf((size_t)rows * n0);
		TEST_CHECK(cudaMemcpy(r0.data(), dr0, r0.size() * 2u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(g0.data(), dg0, g0.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(r1.data(), dr1, r1.size() * 2u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(g1.data(), dg1, g1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(gf.data(), df32, gf.size() * 4u, cudaMemcpyDeviceToHost));
		if ( r0 != g0 || r1 != g1 ) { fprintf(stderr, "MISMATCH D41MxGemmSmallKernel BF16\n"); exit(1); }
		for ( size_t i = 0; i < r0.size(); ++i )
		{
			uint32_t bits; memcpy(&bits, &gf[i], 4u);
			const uint16_t bf = (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
			if ( bf != r0[i] ) { fprintf(stderr, "MISMATCH D41MxGemmSmallKernel F32 at %zu\n", i); exit(1); }
		}
		cudaFree(da); cudaFree(dac); cudaFree(das); cudaFree(dw0); cudaFree(ds0); cudaFree(dw1); cudaFree(ds1);
		cudaFree(dr0); cudaFree(dr1); cudaFree(dg0); cudaFree(dg1); cudaFree(df32);
		printf("PASS D41MxGemmSmallKernel (16x32 pair BF16 & 32x64 F32): 0 diffs\n");
	}

	{
		const uint32_t rows = 16u, width = 5120u, degree = 16u, sw = width / degree;
		std::vector<uint16_t> hst((size_t)rows * 4u * width), hgb((size_t)degree * rows * width);
		std::vector<float> hgf((size_t)degree * rows * width), hpost((size_t)rows * 4u), hcomb((size_t)rows * 16u);
		for ( auto &v : hst ) v = TestRandBf16(rng, 1.5f);
		for ( auto &v : hgb ) v = TestRandBf16(rng, 1.0f);
		for ( auto &v : hgf ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.25f;
		for ( auto &v : hpost ) v = (((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f);
		for ( auto &v : hcomb ) v = (((float)(TestNext(rng) & 0xffu) / 256.0f) - 0.5f);
		uint16_t *dst1 = 0, *dst2 = 0, *dgb = 0, *dpart = 0, *dq1 = 0, *dq2 = 0;
		float *dgf = 0, *dpost = 0, *dcomb = 0;
		TEST_CHECK(cudaMalloc(&dst1, hst.size() * 2u)); TEST_CHECK(cudaMalloc(&dst2, hst.size() * 2u));
		TEST_CHECK(cudaMalloc(&dgb, hgb.size() * 2u)); TEST_CHECK(cudaMalloc(&dgf, hgf.size() * 4u));
		TEST_CHECK(cudaMalloc(&dpart, (size_t)rows * width * 2u));
		TEST_CHECK(cudaMalloc(&dq1, (size_t)rows * width * 2u)); TEST_CHECK(cudaMalloc(&dq2, (size_t)rows * width * 2u));
		TEST_CHECK(cudaMalloc(&dpost, hpost.size() * 4u)); TEST_CHECK(cudaMalloc(&dcomb, hcomb.size() * 4u));
		TEST_CHECK(cudaMemcpy(dgb, hgb.data(), hgb.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dgf, hgf.data(), hgf.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpost, hpost.data(), hpost.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dcomb, hcomb.data(), hcomb.size() * 4u, cudaMemcpyHostToDevice));
		LmRankMajorRowsKernel<256u><<<dim3(3u, rows), 256u>>>(dgb, dq1, rows, sw, degree, 0u);
		TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchFp8Qdq(0, dq1, (uint64_t)rows * width));
		D41RankMajorQdqKernel<<<dim3(3u, rows), 256u>>>(dgb, dq2, rows, sw, degree, 0u);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> rq1((size_t)rows * width), rq2((size_t)rows * width);
		TEST_CHECK(cudaMemcpy(rq1.data(), dq1, rq1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(rq2.data(), dq2, rq2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( rq1 != rq2 ) { fprintf(stderr, "MISMATCH D41RankMajorQdqKernel\n"); exit(1); }
		for ( uint32_t mode : { 1u, 2u, 3u } )
		{
			TEST_CHECK(cudaMemcpy(dst1, hst.data(), hst.size() * 2u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(dst2, hst.data(), hst.size() * 2u, cudaMemcpyHostToDevice));
			if ( mode == 1u )
			{
				LmRankMajorRowsKernel<256u><<<dim3(3u, rows), 256u>>>(dgb, dpart, rows, sw, degree, 0u);
				LmHcPostF32SourceMajorKernel<4u><<<dim3(3u, rows), 256u>>>(dpart, dpost, dcomb, dst1, width);
				D41HcPostFusedKernel<4u, 1u><<<dim3(3u, rows), 256u>>>(dgb, dpost, dcomb, dst2, rows, width, degree, sw);
			}
			else if ( mode == 2u )
			{
				D41SumSourcesF32Kernel<<<64u, 256u>>>(dgf, dpart, (uint64_t)rows * width, degree);
				LmHcPostF32SourceMajorKernel<4u><<<dim3(3u, rows), 256u>>>(dpart, dpost, dcomb, dst1, width);
				D41HcPostFusedKernel<4u, 2u><<<dim3(3u, rows), 256u>>>(dgf, dpost, dcomb, dst2, rows, width, degree, 0u);
			}
			else
			{
				D41SumGatheredKernel<<<64u, 256u>>>(dgb, dpart, (uint64_t)rows * width, degree);
				LmHcPostF32SourceMajorKernel<4u><<<dim3(3u, rows), 256u>>>(dpart, dpost, dcomb, dst1, width);
				D41HcPostFusedKernel<4u, 3u><<<dim3(3u, rows), 256u>>>(dgb, dpost, dcomb, dst2, rows, width, degree, 0u);
			}
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint16_t> s1(hst.size()), s2(hst.size());
			TEST_CHECK(cudaMemcpy(s1.data(), dst1, s1.size() * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s2.data(), dst2, s2.size() * 2u, cudaMemcpyDeviceToHost));
			if ( s1 != s2 ) { fprintf(stderr, "MISMATCH D41HcPostFusedKernel mode=%u\n", mode); exit(1); }
		}
		cudaFree(dst1); cudaFree(dst2); cudaFree(dgb); cudaFree(dgf); cudaFree(dpart);
		cudaFree(dq1); cudaFree(dq2); cudaFree(dpost); cudaFree(dcomb);
		printf("PASS D41RankMajorQdqKernel & D41HcPostFusedKernel (modes 1/2/3): 0 diffs\n");
	}

	{
		const uint32_t rows = 8u, heads = 32u, dim = 128u;
		float *dfreq = 0;
		TEST_CHECK(cudaMalloc(&dfreq, 2u * D41K_ROPE_PAIRS * sizeof(float)));
		TEST_CHECK((cudaError_t)D41LaunchFrequencies(dfreq, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0));
		std::vector<uint16_t> hq((size_t)rows * heads * dim), hw((size_t)rows * heads);
		std::vector<uint32_t> hpos(rows);
		for ( auto &v : hq ) v = TestRandBf16(rng, 2.0f);
		for ( auto &v : hw ) v = TestRandBf16(rng, 1.0f);
		for ( uint32_t r = 0u; r < rows; ++r ) hpos[r] = 17u + r * 53u;
		uint16_t *dq1 = 0, *dq2 = 0, *dw1 = 0, *dw2 = 0;
		uint32_t *dpos = 0;
		TEST_CHECK(cudaMalloc(&dq1, hq.size() * 2u)); TEST_CHECK(cudaMalloc(&dq2, hq.size() * 2u));
		TEST_CHECK(cudaMalloc(&dw1, hw.size() * 2u)); TEST_CHECK(cudaMalloc(&dw2, hw.size() * 2u));
		TEST_CHECK(cudaMalloc(&dpos, rows * 4u));
		TEST_CHECK(cudaMemcpy(dq1, hq.data(), hq.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dq2, hq.data(), hq.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw1, hw.data(), hw.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw2, hw.data(), hw.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpos, hpos.data(), rows * 4u, cudaMemcpyHostToDevice));
		const float wscale = (float)(1.0 / sqrt(128.0) / sqrt(32.0));
		D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS>>>(dq1, heads, dim, dim - 2u * D41K_ROPE_PAIRS, dpos, 0, dfreq + D41K_ROPE_PAIRS, 0u, 0);
		SparkDsv41FlashLaunchFp4Pow2Qdq(0, dq1, (uint64_t)rows * heads * (dim / 32u), 32u);
		LmScaleElementsKernel<uint16_t><<<16u, 256u>>>(dw1, (uint64_t)rows * heads, wscale);
		D41IndexQueryPrepKernel<<<dim3(rows, heads), 64u>>>(dq2, dw2, dpos, dfreq + D41K_ROPE_PAIRS, heads, dim, wscale, 0, 0, 0, 0u);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> q1(hq.size()), q2(hq.size()), w1(hw.size()), w2(hw.size());
		TEST_CHECK(cudaMemcpy(q1.data(), dq1, q1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(q2.data(), dq2, q2.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(w1.data(), dw1, w1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(w2.data(), dw2, w2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( q1 != q2 || w1 != w2 ) { fprintf(stderr, "MISMATCH D41IndexQueryPrepKernel\n"); exit(1); }
		cudaFree(dfreq); cudaFree(dq1); cudaFree(dq2); cudaFree(dw1); cudaFree(dw2); cudaFree(dpos);
		printf("PASS D41IndexQueryPrepKernel: 0 diffs\n");
	}

	{
		using KvGeo = LmKvGeometry<1280u, 128u, true>;
		const uint32_t rows = 4u, lat = 512u, idim = 128u, pages_per_seq = 4u;
		float *dfreq = 0;
		TEST_CHECK(cudaMalloc(&dfreq, 2u * D41K_ROPE_PAIRS * sizeof(float)));
		TEST_CHECK((cudaError_t)D41LaunchFrequencies(dfreq, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0));
		std::vector<float> hckv((size_t)rows * lat), hcsc((size_t)rows * lat), hst((size_t)rows * 2u * lat);
		std::vector<uint16_t> hcn(lat), hik((size_t)rows * idim), hikn(idim);
		std::vector<uint32_t> hpos = { 1u, 2u, 3u, 5u }, hlane = { 0u, 1u, 2u, 3u }, hord = { 0u, 0u, 0u, 0u };
		std::vector<uint32_t> hbeg = { 0u, 1u, 2u, 3u, 4u }, hidx = { 0u, 1u, 2u, 3u }, hslot = { 0u, 1u, 2u, 3u }, hseq = { 0u, 1u, 2u, 3u };
		for ( auto &v : hckv ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 2.0f;
		for ( auto &v : hcsc ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 2.0f;
		for ( auto &v : hst ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 2.0f;
		for ( auto &v : hcn ) v = TestRandBf16(rng, 1.5f);
		for ( auto &v : hik ) v = TestRandBf16(rng, 2.0f);
		for ( auto &v : hikn ) v = TestRandBf16(rng, 1.5f);
		const size_t pool_bytes = (size_t)rows * pages_per_seq * 128u * 1280u;
		uint8_t *dpool1 = 0, *dpool2 = 0;
		uint32_t *dpt = 0;
		LmKvAccessError *derr = 0;
		std::vector<uint32_t> hpt((size_t)rows * pages_per_seq);
		for ( size_t i = 0; i < hpt.size(); ++i ) hpt[i] = (uint32_t)i;
		float *dckv = 0, *dcsc = 0, *dst1 = 0, *dst2 = 0;
		uint16_t *dcn = 0, *dp1 = 0, *dp2 = 0, *dls = 0, *dik1 = 0, *dik2 = 0, *dikn = 0;
		uint32_t *dpos = 0, *dlane = 0, *dord = 0, *dbeg = 0, *didx = 0, *dslot = 0, *dseq = 0, *dem1 = 0, *dem2 = 0;
		TEST_CHECK(cudaMalloc(&dpool1, pool_bytes)); TEST_CHECK(cudaMalloc(&dpool2, pool_bytes));
		TEST_CHECK(cudaMemset(dpool1, 0, pool_bytes)); TEST_CHECK(cudaMemset(dpool2, 0, pool_bytes));
		TEST_CHECK(cudaMalloc(&derr, sizeof(LmKvAccessError))); TEST_CHECK(cudaMemset(derr, 0, sizeof(LmKvAccessError)));
		TEST_CHECK(cudaMalloc(&dpt, hpt.size() * 4u)); TEST_CHECK(cudaMemcpy(dpt, hpt.data(), hpt.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMalloc(&dckv, hckv.size() * 4u)); TEST_CHECK(cudaMalloc(&dcsc, hcsc.size() * 4u));
		TEST_CHECK(cudaMalloc(&dst1, hst.size() * 4u)); TEST_CHECK(cudaMalloc(&dst2, hst.size() * 4u));
		TEST_CHECK(cudaMalloc(&dcn, lat * 2u)); TEST_CHECK(cudaMalloc(&dp1, (size_t)rows * lat * 2u)); TEST_CHECK(cudaMalloc(&dp2, (size_t)rows * lat * 2u));
		TEST_CHECK(cudaMalloc(&dls, (size_t)rows * lat * 2u)); TEST_CHECK(cudaMalloc(&dik1, hik.size() * 2u)); TEST_CHECK(cudaMalloc(&dik2, hik.size() * 2u));
		TEST_CHECK(cudaMalloc(&dikn, idim * 2u));
		TEST_CHECK(cudaMalloc(&dpos, 16u)); TEST_CHECK(cudaMalloc(&dlane, 16u)); TEST_CHECK(cudaMalloc(&dord, 16u));
		TEST_CHECK(cudaMalloc(&dbeg, 20u)); TEST_CHECK(cudaMalloc(&didx, 16u)); TEST_CHECK(cudaMalloc(&dslot, 16u)); TEST_CHECK(cudaMalloc(&dseq, 16u));
		TEST_CHECK(cudaMalloc(&dem1, 16u)); TEST_CHECK(cudaMalloc(&dem2, 16u));
		TEST_CHECK(cudaMemcpy(dckv, hckv.data(), hckv.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dcsc, hcsc.data(), hcsc.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dst1, hst.data(), hst.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dst2, hst.data(), hst.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dcn, hcn.data(), lat * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dik1, hik.data(), hik.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dik2, hik.data(), hik.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dikn, hikn.data(), idim * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpos, hpos.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dlane, hlane.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dord, hord.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbeg, hbeg.data(), 20u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(didx, hidx.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dslot, hslot.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dseq, hseq.data(), 16u, cudaMemcpyHostToDevice));
		LmKvShardView v1{}, v2{};
		v1.pages.pool = dpool1; v1.pages.page_table = dpt; v1.pages.page_table_stride = pages_per_seq;
		v1.pages.sequence_count = rows; v1.pages.pool_page_count = rows * pages_per_seq; v1.pages.access_error = derr;
		v1.shard = { 1u, 0u, 8u };
		v2 = v1; v2.pages.pool = dpool2;
		D41CompressorRowsKernel<<<rows, 256u>>>(dckv, dcsc, dp1, dem1, dst1, dpos, dlane, dord, dbeg, didx, dslot, 2u, 2u * lat, lat);
		D41CompressorSaveKernel<<<rows, 256u>>>(dckv, dcsc, dst1, dpos, dlane, dord, dbeg, dslot, rows, 2u, 2u * lat, lat, 1u, 0);
		LmFusedResidualRmsNormKernel<256u, uint16_t><<<rows, 256u, (lat + 8u) * 4u>>>(dp1, 0, dcn, 0, dp1, lat, lat, 1e-20f);
		LmFusedResidualRmsNormKernel<256u, uint16_t><<<rows, 256u, (idim + 8u) * 4u>>>(dik1, 0, dikn, 0, dik1, idim, idim, 1e-20f);
		D41RopeRowsKernel<<<dim3(rows, 1u), D41K_ROPE_PAIRS>>>(dik1, 1u, idim, idim - 2u * D41K_ROPE_PAIRS, dpos, -1, dfreq + D41K_ROPE_PAIRS, 0u, dem1);
		SparkDsv41FlashLaunchFp4Pow2Qdq(0, dik1, (uint64_t)rows * (idim / 32u), 32u);
		TEST_CHECK(cudaMemcpy(dls, dp1, (size_t)rows * lat * 2u, cudaMemcpyDeviceToDevice));
		D41RopeRowsKernel<<<dim3(rows, 1u), D41K_ROPE_PAIRS>>>(dls, 1u, lat, lat - 2u * D41K_ROPE_PAIRS, dpos, -1, dfreq + D41K_ROPE_PAIRS, 0u, dem1);
		TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchKvFp4Qdq(0, dls, lat, rows, lat));
		D41StoreCompressedKernel<KvGeo><<<rows, 256u>>>(v1, dls, dik1, dem1, dseq, idim);
		D41CompressorRowsKernel<<<rows, 256u>>>(dckv, dcsc, dp2, dem2, dst2, dpos, dlane, dord, dbeg, didx, dslot, 2u, 2u * lat, lat, 0, 0, dcn, 1e-20f, 1u);
		D41CompressFinishStoreKernel<KvGeo><<<rows, 256u>>>(v2, dp2, dik2, dikn, dpos, -1, dfreq + D41K_ROPE_PAIRS, dem2, dseq, idim, 1e-20f);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> p1(pool_bytes), p2(pool_bytes);
		std::vector<float> s1(hst.size()), s2(hst.size());
		TEST_CHECK(cudaMemcpy(p1.data(), dpool1, pool_bytes, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(p2.data(), dpool2, pool_bytes, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), dst1, s1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), dst2, s2.size() * 4u, cudaMemcpyDeviceToHost));
		if ( p1 != p2 || memcmp(s1.data(), s2.data(), s1.size() * 4u) != 0 )
		{ fprintf(stderr, "MISMATCH D41CompressFinishStoreKernel\n"); exit(1); }
		cudaFree(dfreq); cudaFree(dpool1); cudaFree(dpool2); cudaFree(derr); cudaFree(dpt); cudaFree(dckv); cudaFree(dcsc);
		cudaFree(dst1); cudaFree(dst2); cudaFree(dcn); cudaFree(dp1); cudaFree(dp2); cudaFree(dls);
		cudaFree(dik1); cudaFree(dik2); cudaFree(dikn); cudaFree(dpos); cudaFree(dlane); cudaFree(dord);
		cudaFree(dbeg); cudaFree(didx); cudaFree(dslot); cudaFree(dseq); cudaFree(dem1); cudaFree(dem2);
		printf("PASS D41CompressorRowsKernel(norm+save) & D41CompressFinishStoreKernel: 0 diffs\n");
	}

	{
		const uint32_t rows = 4u, stride = 20480u, bstride = stride / 8u, keep = 2048u;
		std::vector<float> hsc((size_t)rows * stride, -INFINITY);
		std::vector<uint32_t> hseq = { 0u, 1u, 2u, 3u }, hlen = { 128u, 4096u, 16384u, 20000u }, hbound = { 127u, 4095u, 16383u, 19999u };
		for ( uint32_t r = 0u; r < rows; ++r )
			for ( uint32_t i = 0u; i < hlen[r]; ++i )
				hsc[(size_t)r * stride + i] = ((float)(TestNext(rng) & 0xfffu) / 100.0f) - 20.0f;
		float *dsc1 = 0, *dsc2 = 0, *db1 = 0, *db2 = 0, *dtv = 0;
		uint32_t *dseq = 0, *dlen = 0, *dbnd = 0, *dch1 = 0, *dch2 = 0, *dtp = 0;
		uint8_t *dm1 = 0, *dm2 = 0;
		const uint64_t entries = (uint64_t)rows * 4096u * 2u;
		TEST_CHECK(cudaMalloc(&dsc1, hsc.size() * 4u)); TEST_CHECK(cudaMalloc(&dsc2, hsc.size() * 4u));
		TEST_CHECK(cudaMalloc(&db1, (size_t)rows * bstride * 4u)); TEST_CHECK(cudaMalloc(&db2, (size_t)rows * bstride * 4u));
		TEST_CHECK(cudaMalloc(&dtv, entries * 4u)); TEST_CHECK(cudaMalloc(&dtp, entries * 4u));
		TEST_CHECK(cudaMalloc(&dseq, 16u)); TEST_CHECK(cudaMalloc(&dlen, 16u)); TEST_CHECK(cudaMalloc(&dbnd, 16u));
		TEST_CHECK(cudaMalloc(&dch1, (size_t)rows * keep * 4u)); TEST_CHECK(cudaMalloc(&dch2, (size_t)rows * keep * 4u));
		TEST_CHECK(cudaMalloc(&dm1, (size_t)rows * bstride)); TEST_CHECK(cudaMalloc(&dm2, (size_t)rows * bstride));
		TEST_CHECK(cudaMemcpy(dsc1, hsc.data(), hsc.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsc2, hsc.data(), hsc.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dseq, hseq.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dlen, hlen.data(), 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbnd, hbound.data(), 16u, cudaMemcpyHostToDevice));
		SparkKvShard sh = { 1u, 0u, 8u };
		TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchShardBlockMax(0, dsc1, stride, sh, dseq, dlen, dbnd, db1, bstride, rows));
		TEST_CHECK((cudaError_t)LmTopkExactLaunch<256u>(db1, rows, bstride, keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, dtv, dtp, entries, dch1, 0));
		D41WholeBlockMaskKernel<<<rows, 256u>>>(db1, dch1, keep, dm1, bstride);
		TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchShardBlockApply(0, dsc1, stride, 8u, dm1, bstride, rows));
		D41WholeBlockMaxKernel<<<dim3((bstride + 255u) / 256u, rows), 256u>>>(dsc2, stride, dseq, dlen, dbnd, db2, bstride, 8u, 16384u);
		TEST_CHECK((cudaError_t)LmTopkExactLaunch<256u>(db2, rows, bstride, keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, dtv, dtp, entries, dch2, 0, dseq, dlen, dbnd, 16384u, 8u));
		D41WholeBlockMaskKernel<<<rows, 256u>>>(db2, dch2, keep, dm2, bstride, dseq, dlen, dbnd, 8u, 16384u);
		D41WholeBlockApplyKernel<<<dim3((stride + 255u) / 256u, rows), 256u>>>(dsc2, stride, 8u, dm2, bstride, dseq, dlen, dbnd, 16384u);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> m1((size_t)rows * bstride), m2((size_t)rows * bstride);
		std::vector<float> s1(hsc.size()), s2(hsc.size());
		TEST_CHECK(cudaMemcpy(m1.data(), dm1, m1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(m2.data(), dm2, m2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), dsc1, s1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), dsc2, s2.size() * 4u, cudaMemcpyDeviceToHost));
		if ( m1 != m2 || memcmp(s1.data(), s2.data(), s1.size() * 4u) != 0 )
		{ fprintf(stderr, "MISMATCH D41WholeBlockMax/Mask/Apply\n"); exit(1); }
		cudaFree(dsc1); cudaFree(dsc2); cudaFree(db1); cudaFree(db2); cudaFree(dtv); cudaFree(dtp);
		cudaFree(dseq); cudaFree(dlen); cudaFree(dbnd); cudaFree(dch1); cudaFree(dch2); cudaFree(dm1); cudaFree(dm2);
		printf("PASS D41WholeBlockMax/Mask/Apply (ctx=128/4096/16384/20000): 0 diffs\n");
	}
}

static void TestDirectIndexScoreAndFastTopk(uint32_t *rng)
{
	{
		using Geo = LmKvGeometry<1280u, 256u, true>;
		const uint32_t rows = 16u, stride = 4096u, bstride = stride / 8u, seqs = 4u, pages_per_seq = 16u;
		std::vector<uint16_t> hq((size_t)rows * 32u * 128u), hw((size_t)rows * 32u);
		std::vector<uint8_t> hpages((size_t)seqs * pages_per_seq * 256u * 1280u, 0u);
		std::vector<uint32_t> hseq(rows), hlen(seqs), hbnd(rows);
		std::vector<uint8_t> hmask((size_t)rows * bstride, 0u);
		for ( auto &v : hq ) v = TestRandBf16(rng, 1.5f);
		for ( auto &v : hw ) v = TestRandBf16(rng, 0.5f);
		for ( uint32_t s = 0u; s < seqs; ++s ) hlen[s] = s == 0u ? 256u : s == 1u ? 1234u : s == 2u ? 2048u : 3800u;
		for ( uint32_t r = 0u; r < rows; ++r ) { hseq[r] = r % seqs; hbnd[r] = hlen[hseq[r]] - 1u - (r % 5u); }
		for ( size_t i = 0; i < hmask.size(); ++i ) hmask[i] = (TestNext(rng) & 3u) != 0u ? 1u : 0u;
		for ( uint32_t s = 0u; s < seqs; ++s )
			for ( uint32_t p = 0u; p < hlen[s]; ++p )
			{
				uint16_t *slot = (uint16_t *)(hpages.data() + ((size_t)s * pages_per_seq * 256u + p) * 1280u + 512u * 2u);
				for ( uint32_t d = 0u; d < 128u; ++d ) slot[d] = TestRandBf16(rng, 1.5f);
			}
		uint16_t *dq = 0, *dw = 0; uint8_t *dpages = 0, *dmask = 0;
		uint32_t *dseq = 0, *dlen = 0, *dbnd = 0, *dpt = 0;
		float *ds1 = 0, *ds2 = 0;
		LmKvAccessError *derr = 0;
		std::vector<uint32_t> hpt((size_t)seqs * pages_per_seq);
		for ( size_t i = 0; i < hpt.size(); ++i ) hpt[i] = (uint32_t)i;
		TEST_CHECK(cudaMalloc(&dq, hq.size() * 2u)); TEST_CHECK(cudaMalloc(&dw, hw.size() * 2u));
		TEST_CHECK(cudaMalloc(&dpages, hpages.size())); TEST_CHECK(cudaMalloc(&dpt, hpt.size() * 4u));
		TEST_CHECK(cudaMalloc(&dseq, rows * 4u)); TEST_CHECK(cudaMalloc(&dlen, seqs * 4u)); TEST_CHECK(cudaMalloc(&dbnd, rows * 4u));
		TEST_CHECK(cudaMalloc(&dmask, hmask.size())); TEST_CHECK(cudaMalloc(&derr, sizeof(LmKvAccessError)));
		TEST_CHECK(cudaMemset(derr, 0, sizeof(LmKvAccessError)));
		TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * stride * 4u)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * stride * 4u));
		TEST_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpages, hpages.data(), hpages.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpt, hpt.data(), hpt.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dseq, hseq.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dlen, hlen.data(), seqs * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbnd, hbnd.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dmask, hmask.data(), hmask.size(), cudaMemcpyHostToDevice));
		LmKvShardView view = {};
		view.pages.pool = dpages + 1024u; view.pages.page_table = dpt; view.pages.page_table_stride = pages_per_seq;
		view.pages.sequence_count = seqs; view.pages.pool_page_count = seqs * pages_per_seq;
		view.pages.access_error = derr;
		view.shard = { 1u, 0u, 8u };
		for ( uint32_t use_mask : { 0u, 1u } )
		{
			TEST_CHECK(cudaMemset(ds1, 0, (size_t)rows * stride * 4u));
			TEST_CHECK(cudaMemset(ds2, 0, (size_t)rows * stride * 4u));
			TEST_CHECK((cudaError_t)LmWeightedSparseScoreLaunch<Geo, 32u, 128u, LmKvShardView>(dq, dw, view, dseq, dlen, dbnd, rows, stride, 1.0f, ds1, 0, 0, 0, 512u));
			if ( use_mask != 0u )
				D41WholeBlockApplyKernel<<<dim3((stride + 255u) / 256u, rows), 256u>>>(ds1, stride, 8u, dmask, bstride, dseq, dlen, dbnd, 0u);
			D41IndexScoreDirectKernel<Geo, false><<<dim3((stride + 255u) / 256u, rows), 256u>>>(dq, dw, view, dseq, dlen, dbnd,
				use_mask != 0u ? dmask : (const uint8_t *)0, bstride, rows, stride, 1.0f, ds2, 0, 0, 512u);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<float> s1((size_t)rows * stride), s2((size_t)rows * stride);
			TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size() * 4u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size() * 4u, cudaMemcpyDeviceToHost));
			for ( uint32_t r = 0u; r < rows; ++r )
			{
				const uint32_t act = hbnd[r] + 1u < hlen[hseq[r]] ? hbnd[r] + 1u : hlen[hseq[r]];
				if ( act <= 512u ) continue;
				for ( uint32_t k = 0u; k < act; ++k )
					if ( memcmp(&s1[(size_t)r * stride + k], &s2[(size_t)r * stride + k], 4u) != 0 )
					{ fprintf(stderr, "MISMATCH D41IndexScoreDirect mask=%u r=%u k=%u ref=%g got=%g\n", use_mask, r, k, s1[(size_t)r * stride + k], s2[(size_t)r * stride + k]); exit(1); }
			}
		}
		cudaFree(dq); cudaFree(dw); cudaFree(dpages); cudaFree(dpt); cudaFree(dseq); cudaFree(dlen); cudaFree(dbnd);
		cudaFree(dmask); cudaFree(derr); cudaFree(ds1); cudaFree(ds2);
		printf("PASS D41IndexScoreDirectKernel (multi-seq, mask=0/1, dense_limit=512): 0 diffs\n");
	}
	{
		struct Case { uint32_t rows, n, k, ties; };
		const Case cases[] = {
			{ 1u, 600u, 512u, 0u }, { 1u, 4093u, 512u, 1u }, { 1u, 4096u, 512u, 0u },
			{ 1u, 24576u, 512u, 0u }, { 1u, 24576u, 512u, 1u }, { 1u, 65536u, 512u, 0u },
			{ 4u, 8192u, 2048u, 1u }, { 16u, 4096u, 512u, 1u }, { 32u, 8192u, 512u, 0u }
		};
		for ( const auto &c : cases )
		{
			std::vector<float> hsc((size_t)c.rows * c.n);
			for ( size_t i = 0; i < hsc.size(); ++i )
			{
				const uint32_t x = TestNext(rng);
				if ( c.ties != 0u ) hsc[i] = (x & 7u) == 0u ? -INFINITY : (float)((int)(x & 15u) - 4) * 0.25f;
				else hsc[i] = (x & 31u) == 0u ? -INFINITY : ((float)(x & 0xffffu) / 1024.0f - 16.0f);
			}
			float *dsc = 0, *dtv = 0; uint32_t *dtp = 0, *do1 = 0, *do2 = 0;
			const uint64_t entries = (uint64_t)c.rows * LmTopkExactCandidateEntries(c.n, c.k, LM_TOPK_EXACT_CHUNK);
			TEST_CHECK(cudaMalloc(&dsc, hsc.size() * 4u));
			TEST_CHECK(cudaMalloc(&dtv, (size_t)entries * 4u)); TEST_CHECK(cudaMalloc(&dtp, (size_t)entries * 4u));
			TEST_CHECK(cudaMalloc(&do1, (size_t)c.rows * c.k * 4u)); TEST_CHECK(cudaMalloc(&do2, (size_t)c.rows * c.k * 4u));
			TEST_CHECK(cudaMemcpy(dsc, hsc.data(), hsc.size() * 4u, cudaMemcpyHostToDevice));
			TEST_CHECK((cudaError_t)LmTopkExactLaunch<256u>(dsc, c.rows, c.n, c.k, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, dtv, dtp, entries, do1, 0));
			const uint64_t chunk_entries = (uint64_t)c.rows * ((c.n + 4095u) / 4096u) * c.k;
			float *dtv2 = 0; uint32_t *dtp2 = 0;
			TEST_CHECK(cudaMalloc(&dtv2, (size_t)chunk_entries * 4u)); TEST_CHECK(cudaMalloc(&dtp2, (size_t)chunk_entries * 4u));
			TEST_CHECK((cudaError_t)D41TopkExactLaunch(dsc, c.rows, c.n, c.k, do2, 0, 0, 0, 0, 0u, 1u, dtv2, dtp2, chunk_entries));
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint32_t> o1((size_t)c.rows * c.k), o2((size_t)c.rows * c.k);
			TEST_CHECK(cudaMemcpy(o1.data(), do1, o1.size() * 4u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(o2.data(), do2, o2.size() * 4u, cudaMemcpyDeviceToHost));
			if ( o1 != o2 )
			{ fprintf(stderr, "MISMATCH D41TopkExactLaunch rows=%u n=%u k=%u ties=%u\n", c.rows, c.n, c.k, c.ties); exit(1); }
			cudaFree(dsc); cudaFree(dtv); cudaFree(dtp); cudaFree(dtv2); cudaFree(dtp2); cudaFree(do1); cudaFree(do2);
		}
		printf("PASS D41TopkExactLaunch (FastKernel + ChunkRegKernel, n=600..65536, ties=0/1): 0 diffs\n");
	}
	{
		const uint32_t heads = 16u, window = 128u, rows = 8u;
		const float qk_scale = 1.0f / sqrtf(512.0f);
		const uint64_t rstride = (uint64_t)window * D41K_LATENT;
		std::vector<uint16_t> hq((size_t)rows * heads * D41K_LATENT), hwkv((size_t)rows * D41K_LATENT), hring((size_t)rows * rstride);
		std::vector<float> hrec((size_t)rows * heads * D41K_RECORD);
		std::vector<uint32_t> hpos(rows), hlane(rows), hord(rows), hbeg(rows + 1u), hidx(rows), hslot(rows);
		for ( auto &v : hq ) v = TestRandBf16(rng, 0.5f);
		for ( auto &v : hwkv ) v = TestRandBf16(rng, 0.5f);
		for ( auto &v : hring ) v = TestRandBf16(rng, 0.5f);
		for ( size_t i = 0; i < hrec.size(); ++i ) hrec[i] = ((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f;
		for ( uint32_t r = 0u; r < rows; ++r )
		{
			hpos[r] = 200u + r * 37u; hlane[r] = r; hord[r] = 0u; hbeg[r] = r; hidx[r] = r; hslot[r] = r;
			for ( uint32_t h = 0u; h < heads; ++h )
			{
				hrec[((size_t)r * heads + h) * D41K_RECORD + 0u] = 0.25f;
				hrec[((size_t)r * heads + h) * D41K_RECORD + 1u] = 2.0f;
			}
		}
		hbeg[rows] = rows;
		float *drec1 = 0, *drec2 = 0; uint16_t *dq = 0, *dwkv = 0, *dring = 0;
		uint32_t *dpos = 0, *dlane = 0, *dord = 0, *dbeg = 0, *didx = 0, *dslot = 0;
		TEST_CHECK(cudaMalloc(&drec1, hrec.size() * 4u)); TEST_CHECK(cudaMalloc(&drec2, hrec.size() * 4u));
		TEST_CHECK(cudaMalloc(&dq, hq.size() * 2u)); TEST_CHECK(cudaMalloc(&dwkv, hwkv.size() * 2u)); TEST_CHECK(cudaMalloc(&dring, hring.size() * 2u));
		TEST_CHECK(cudaMalloc(&dpos, rows * 4u)); TEST_CHECK(cudaMalloc(&dlane, rows * 4u)); TEST_CHECK(cudaMalloc(&dord, rows * 4u));
		TEST_CHECK(cudaMalloc(&dbeg, (rows + 1u) * 4u)); TEST_CHECK(cudaMalloc(&didx, rows * 4u)); TEST_CHECK(cudaMalloc(&dslot, rows * 4u));
		TEST_CHECK(cudaMemcpy(drec1, hrec.data(), hrec.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(drec2, hrec.data(), hrec.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dwkv, hwkv.data(), hwkv.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dring, hring.data(), hring.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpos, hpos.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dlane, hlane.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dord, hord.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbeg, hbeg.data(), (rows + 1u) * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(didx, hidx.data(), rows * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dslot, hslot.data(), rows * 4u, cudaMemcpyHostToDevice));
		D41WindowFoldWholeKernel<<<dim3(rows, heads), D41K_THREADS>>>(drec1, heads, dq, dwkv, dring, dpos, dlane, dord, dbeg, didx, dslot, rstride, window, qk_scale);
		D41WindowFoldWholeTiledKernel<4u><<<dim3(rows, heads / 4u), D41K_THREADS>>>(drec2, heads, dq, dwkv, dring, dpos, dlane, dord, dbeg, didx, dslot, rstride, window, qk_scale);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<float> r1(hrec.size()), r2(hrec.size());
		TEST_CHECK(cudaMemcpy(r1.data(), drec1, r1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(r2.data(), drec2, r2.size() * 4u, cudaMemcpyDeviceToHost));
		for ( size_t i = 0; i < r1.size(); ++i )
			if ( fabsf(r1[i] - r2[i]) > 1e-5f ) { fprintf(stderr, "MISMATCH WindowFoldWholeTiled idx=%zu ref=%g got=%g\n", i, r1[i], r2[i]); exit(1); }
		cudaFree(drec1); cudaFree(drec2); cudaFree(dq); cudaFree(dwkv); cudaFree(dring);
		cudaFree(dpos); cudaFree(dlane); cudaFree(dord); cudaFree(dbeg); cudaFree(didx); cudaFree(dslot);
		printf("PASS D41WindowFoldWholeTiledKernel<4u> (rows=8, heads=16): 0 diffs\n");
	}
	{
		const uint32_t experts = 384u, topk = 6u;
		for ( uint32_t rows : { 1u, 4u, 8u } )
		{
			const uint32_t routes = rows * topk;
			std::vector<float> hlog((size_t)rows * experts), hbias(experts);
			for ( auto &v : hlog ) v = ((float)(TestNext(rng) & 0xfffu) / 512.0f) - 4.0f;
			for ( auto &v : hbias ) v = ((float)(TestNext(rng) & 0xffu) / 512.0f) - 0.25f;
			float *dlog = 0, *dbias = 0, *drw1 = 0, *drw2 = 0, *dpw1 = 0, *dpw2 = 0;
			uint32_t *df1 = 0, *df2 = 0, *dre1 = 0, *dre2 = 0, *doff1 = 0, *doff2 = 0, *dpr1 = 0, *dpr2 = 0;
			uint32_t *dst1 = 0, *dst2 = 0, *dtu1 = 0, *dtu2 = 0, *dtd1 = 0, *dtd2 = 0;
			TEST_CHECK(cudaMalloc(&dlog, hlog.size() * 4u)); TEST_CHECK(cudaMalloc(&dbias, hbias.size() * 4u));
			TEST_CHECK(cudaMalloc(&drw1, routes * 4u)); TEST_CHECK(cudaMalloc(&drw2, routes * 4u));
			TEST_CHECK(cudaMalloc(&dpw1, routes * 4u)); TEST_CHECK(cudaMalloc(&dpw2, routes * 4u));
			TEST_CHECK(cudaMalloc(&df1, 4u)); TEST_CHECK(cudaMalloc(&df2, 4u));
			TEST_CHECK(cudaMalloc(&dre1, routes * 4u)); TEST_CHECK(cudaMalloc(&dre2, routes * 4u));
			TEST_CHECK(cudaMalloc(&doff1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&doff2, (experts + 1u) * 4u));
			TEST_CHECK(cudaMalloc(&dpr1, routes * 4u)); TEST_CHECK(cudaMalloc(&dpr2, routes * 4u));
			TEST_CHECK(cudaMalloc(&dst1, routes * 4u)); TEST_CHECK(cudaMalloc(&dst2, routes * 4u));
			TEST_CHECK(cudaMalloc(&dtu1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtu2, (experts + 1u) * 4u));
			TEST_CHECK(cudaMalloc(&dtd1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtd2, (experts + 1u) * 4u));
			TEST_CHECK(cudaMemcpy(dlog, hlog.data(), hlog.size() * 4u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(dbias, hbias.data(), hbias.size() * 4u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemset(df1, 0, 4u)); TEST_CHECK(cudaMemset(df2, 0, 4u));
			TEST_CHECK((cudaError_t)LmTopkRouteLaunch<256u, topk, true, LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, dlog, experts, dre1, drw1, dbias, 0, 2.5f, df1, 0));
			if ( LmRouteBuild<256u, experts>(dre1, rows, routes, topk, doff1, dpr1, dst1, 640u, 1280u, 128u, dtu1, dtd1, 0) != LM_LAUNCH_OK ) exit(1);
			LmPackedRouteWeightKernel<<<1, 256>>>(dpr1, drw1, dpw1, routes);
			D41TopkRoutePackedKernel<experts, topk><<<1, 256>>>(dlog, dbias, 2.5f, df2, dre2, drw2, doff2, dpr2, dst2,
				(uint32_t)LmLaunchGroupedTileM(rows, topk, experts), 5u, dtu2, 10u, dtd2, dpw2, rows);
			TEST_CHECK(cudaDeviceSynchronize());
			auto eq_u32 = [&](const uint32_t *a, const uint32_t *b, size_t n, const char *tag) {
				std::vector<uint32_t> va(n), vb(n);
				TEST_CHECK(cudaMemcpy(va.data(), a, n * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(vb.data(), b, n * 4u, cudaMemcpyDeviceToHost));
				if ( va != vb ) { fprintf(stderr, "MISMATCH TopkRoutePacked %s rows=%u\n", tag, rows); exit(1); }
			};
			eq_u32(dre1, dre2, routes, "re"); eq_u32(doff1, doff2, experts + 1u, "off");
			eq_u32(dpr1, dpr2, routes, "pr"); eq_u32(dst1, dst2, routes, "st");
			eq_u32(dtu1, dtu2, experts + 1u, "tu"); eq_u32(dtd1, dtd2, experts + 1u, "td");
			eq_u32((const uint32_t *)drw1, (const uint32_t *)drw2, routes, "rw");
			eq_u32((const uint32_t *)dpw1, (const uint32_t *)dpw2, routes, "pw");
			cudaFree(dlog); cudaFree(dbias); cudaFree(drw1); cudaFree(drw2); cudaFree(dpw1); cudaFree(dpw2);
			cudaFree(df1); cudaFree(df2); cudaFree(dre1); cudaFree(dre2); cudaFree(doff1); cudaFree(doff2);
			cudaFree(dpr1); cudaFree(dpr2); cudaFree(dst1); cudaFree(dst2); cudaFree(dtu1); cudaFree(dtu2); cudaFree(dtd1); cudaFree(dtd2);
		}
		printf("PASS D41TopkRoutePackedKernel<384,6> (rows=1/4/8): 0 diffs\n");
	}
}

static void TestE5LeapfrogKernels(uint32_t *rng)
{
	{
		const uint32_t experts = 384u, topk = 6u, hidden = 5120u;
		std::vector<uint16_t> hw((size_t)experts * hidden), hx(8u * hidden);
		std::vector<float> hbias(experts);
		for ( auto &v : hw ) v = TestRandBf16(rng, 0.25f);
		for ( auto &v : hx ) v = TestRandBf16(rng, 1.0f);
		for ( auto &v : hbias ) v = ((float)(TestNext(rng) & 0xffu) / 512.0f) - 0.25f;
		uint16_t *dw = 0, *dx = 0;
		float *dbias = 0, *dlog1 = 0, *dlog2 = 0, *drw1 = 0, *drw2 = 0, *dpw1 = 0, *dpw2 = 0;
		uint32_t *df1 = 0, *df2 = 0, *dre1 = 0, *dre2 = 0, *doff1 = 0, *doff2 = 0, *dpr1 = 0, *dpr2 = 0;
		uint32_t *dst1 = 0, *dst2 = 0, *dtu1 = 0, *dtu2 = 0, *dtd1 = 0, *dtd2 = 0, *dcnt = 0, *dvalid = 0;
		TEST_CHECK(cudaMalloc(&dw, hw.size() * 2u)); TEST_CHECK(cudaMalloc(&dx, hx.size() * 2u));
		TEST_CHECK(cudaMalloc(&dbias, experts * 4u));
		TEST_CHECK(cudaMalloc(&dlog1, 8u * experts * 4u)); TEST_CHECK(cudaMalloc(&dlog2, 8u * experts * 4u));
		TEST_CHECK(cudaMalloc(&drw1, 48u * 4u)); TEST_CHECK(cudaMalloc(&drw2, 48u * 4u));
		TEST_CHECK(cudaMalloc(&dpw1, 48u * 4u)); TEST_CHECK(cudaMalloc(&dpw2, 48u * 4u));
		TEST_CHECK(cudaMalloc(&df1, 4u)); TEST_CHECK(cudaMalloc(&df2, 4u));
		TEST_CHECK(cudaMalloc(&dre1, 48u * 4u)); TEST_CHECK(cudaMalloc(&dre2, 48u * 4u));
		TEST_CHECK(cudaMalloc(&doff1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&doff2, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dpr1, 48u * 4u)); TEST_CHECK(cudaMalloc(&dpr2, 48u * 4u));
		TEST_CHECK(cudaMalloc(&dst1, 48u * 4u)); TEST_CHECK(cudaMalloc(&dst2, 48u * 4u));
		TEST_CHECK(cudaMalloc(&dtu1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtu2, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dtd1, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtd2, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dcnt, 4u)); TEST_CHECK(cudaMalloc(&dvalid, 8u * 4u));
		TEST_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbias, hbias.data(), experts * 4u, cudaMemcpyHostToDevice));
		auto run_case = [&](uint32_t rows) {
			const uint32_t routes = rows * topk, tile_m = (uint32_t)LmLaunchGroupedTileM(rows, topk, experts);
			TEST_CHECK(cudaMemset(df1, 0, 4u)); TEST_CHECK(cudaMemset(df2, 0, 4u)); TEST_CHECK(cudaMemset(dcnt, 0, 4u));
			LmSkinnyArguments sa;
			memset(&sa, 0, sizeof(sa));
			sa.weight = (const uint8_t *)dw; sa.activation = dx; sa.output_f32 = dlog1;
			sa.scale = LmScaleTensorNone(); sa.rows = rows; sa.input_dimension = hidden; sa.output_dimension = experts;
			if ( LmSkinnyLaunch<LmBf16Format>(&sa, 0, 1u) != LM_LAUNCH_OK ) exit(1);
			D41TopkRoutePackedKernel<experts, topk><<<1u, 256u>>>(dlog1, dbias, 2.5f, df1, dre1, drw1, doff1, dpr1, dst1,
				tile_m, 5u, dtu1, 10u, dtd1, dpw1, rows);
			if ( rows == 1u )
				D41FusedRouterPackedKernel<experts, topk, hidden, 1u><<<experts, 256u>>>(dw, dx, dlog2, dbias, 2.5f, df2, dre2, drw2,
					doff2, dpr2, dst2, tile_m, 5u, dtu2, 10u, dtd2, dpw2, rows, dcnt);
			else if ( rows == 4u )
				D41FusedRouterPackedKernel<experts, topk, hidden, 4u><<<experts, 256u>>>(dw, dx, dlog2, dbias, 2.5f, df2, dre2, drw2,
					doff2, dpr2, dst2, tile_m, 5u, dtu2, 10u, dtd2, dpw2, rows, dcnt);
			else
				D41FusedRouterPackedKernel<experts, topk, hidden, 8u><<<experts, 256u>>>(dw, dx, dlog2, dbias, 2.5f, df2, dre2, drw2,
					doff2, dpr2, dst2, tile_m, 5u, dtu2, 10u, dtd2, dpw2, rows, dcnt);
			TEST_CHECK(cudaDeviceSynchronize());
			auto eq = [&](const void *a, const void *b, size_t n, const char *tag) {
				std::vector<uint32_t> va(n), vb(n);
				TEST_CHECK(cudaMemcpy(va.data(), a, n * 4u, cudaMemcpyDeviceToHost));
				TEST_CHECK(cudaMemcpy(vb.data(), b, n * 4u, cudaMemcpyDeviceToHost));
				if ( va != vb ) { fprintf(stderr, "MISMATCH FusedRouterPacked %s rows=%u\n", tag, rows); exit(1); }
			};
			eq(dlog1, dlog2, (size_t)rows * experts, "logits");
			eq(dre1, dre2, routes, "re"); eq(drw1, drw2, routes, "rw");
			eq(doff1, doff2, experts + 1u, "off"); eq(dpr1, dpr2, routes, "pr");
			eq(dst1, dst2, routes, "st"); eq(dtu1, dtu2, experts + 1u, "tu");
			eq(dtd1, dtd2, experts + 1u, "td"); eq(dpw1, dpw2, routes, "pw");
		};
		run_case(1u); run_case(4u); run_case(8u);
		const uint32_t hv[4] = { 1u, 0u, 1u, 0u };
		TEST_CHECK(cudaMemcpy(dvalid, hv, 16u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemset(dcnt, 0, 4u)); TEST_CHECK(cudaMemset(df2, 0, 4u));
		D41FusedRouterPackedKernel<experts, topk, hidden, 4u><<<experts, 256u>>>(dw, dx, dlog2, dbias, 2.5f, df2, dre2, drw2,
			doff2, dpr2, dst2, (uint32_t)LmLaunchGroupedTileM(4u, topk, experts), 5u, dtu2, 10u, dtd2, dpw2, 4u, dcnt, dvalid);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint32_t> hre(24u), hoff(experts + 1u), hpr(24u);
		std::vector<float> hrw(24u);
		TEST_CHECK(cudaMemcpy(hre.data(), dre2, 24u * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(hoff.data(), doff2, (experts + 1u) * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(hpr.data(), dpr2, 24u * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(hrw.data(), drw2, 24u * 4u, cudaMemcpyDeviceToHost));
		if ( hoff[experts] != 12u ) { fprintf(stderr, "MISMATCH dead-row total=%u expected 12\n", hoff[experts]); exit(1); }
		for ( uint32_t r = 0u; r < 4u; ++r )
			for ( uint32_t k = 0u; k < topk; ++k )
			{
				const uint32_t i = r * topk + k;
				if ( hv[r] == 0u && (hre[i] != 0xffffffffu || hrw[i] != 0.0f || hpr[i] != 24u) )
				{ fprintf(stderr, "MISMATCH dead-row slot %u\n", i); exit(1); }
				if ( hv[r] != 0u && (hre[i] >= experts || hpr[i] >= 12u) )
				{ fprintf(stderr, "MISMATCH live-row slot %u\n", i); exit(1); }
			}
		D41L2PrefetchKernel<<<2u, 64u>>>((const uint8_t *)dw, 65536u, (const uint8_t *)dx, 16384u);
		D41L2PrefetchKernel<<<2u, 64u>>>((const uint8_t *)dw, 32768u, (const uint8_t *)0, 0u);
		TEST_CHECK(cudaDeviceSynchronize());
		printf("PASS D41FusedRouterPackedKernel (rows=1/4/8 + dead-row skip) & D41L2PrefetchKernel: 0 diffs\n");
		cudaFree(dw); cudaFree(dx); cudaFree(dbias); cudaFree(dlog1); cudaFree(dlog2);
		cudaFree(drw1); cudaFree(drw2); cudaFree(dpw1); cudaFree(dpw2); cudaFree(df1); cudaFree(df2);
		cudaFree(dre1); cudaFree(dre2); cudaFree(doff1); cudaFree(doff2); cudaFree(dpr1); cudaFree(dpr2);
		cudaFree(dst1); cudaFree(dst2); cudaFree(dtu1); cudaFree(dtu2); cudaFree(dtd1); cudaFree(dtd2);
		cudaFree(dcnt); cudaFree(dvalid);
	}
	{
		const uint32_t ranks = 16u, experts = 384u, local = 24u, held1 = 73u, topk = 6u, copies = 2u, rows = 512u, routes = rows * topk;
		std::vector<uint8_t> hown(experts), hbk(copies * experts), hgrp(experts, 255u);
		uint32_t b1 = 0u, b2 = 0u;
		for ( uint32_t e = 0u; e < experts; ++e )
		{
			const uint32_t ow = e / local, k0 = (ow + 1u) % ranks, k1 = (ow + 2u) % ranks;
			hown[e] = (uint8_t)ow; hbk[e] = (uint8_t)k0; hbk[experts + e] = (uint8_t)k1;
			if ( ow == 0u ) hgrp[e] = (uint8_t)(e % local);
			else if ( k0 == 0u ) hgrp[e] = (uint8_t)(local + b1++);
			else if ( k1 == 0u ) hgrp[e] = (uint8_t)(2u * local + b2++);
		}
		std::vector<uint32_t> hre_in(routes);
		std::vector<float> hrw_in(routes);
		for ( uint32_t r = 0u; r < routes; ++r ) { hre_in[r] = TestNext(rng) % experts; hrw_in[r] = 0.1f + (float)(r % 11u) * 0.03f; }
		uint8_t *down = 0, *dbk = 0, *dgrp = 0;
		uint32_t *dre = 0, *drg = 0, *doff = 0, *dpr = 0, *dst = 0, *dtu = 0, *dtd = 0;
		float *drw = 0, *dpw = 0;
		TEST_CHECK(cudaMalloc(&down, experts)); TEST_CHECK(cudaMalloc(&dbk, copies * experts)); TEST_CHECK(cudaMalloc(&dgrp, experts));
		TEST_CHECK(cudaMalloc(&dre, routes * 4u)); TEST_CHECK(cudaMalloc(&drg, routes * 4u)); TEST_CHECK(cudaMalloc(&drw, routes * 4u));
		TEST_CHECK(cudaMalloc(&doff, (held1 + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dpr, routes * 4u)); TEST_CHECK(cudaMalloc(&dst, routes * 4u));
		TEST_CHECK(cudaMalloc(&dtu, (held1 + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtd, (held1 + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dpw, routes * 4u));
		TEST_CHECK(cudaMemcpy(down, hown.data(), experts, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbk, hbk.data(), copies * experts, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dgrp, hgrp.data(), experts, cudaMemcpyHostToDevice));
		for ( uint32_t row_inv = 0u; row_inv < 2u; ++row_inv )
		{
			TEST_CHECK(cudaMemcpy(dre, hre_in.data(), routes * 4u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(drw, hrw_in.data(), routes * 4u, cudaMemcpyHostToDevice));
			D41BalanceRouteKernel<ranks, experts, local, 256u, held1><<<1u, 256u>>>(dre, drw, drg, rows, topk, 0u, copies, row_inv,
				down, dbk, dgrp, doff, dpr, dst, 16u, 36u, dtu, 40u, dtd, dpw);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint32_t> ref_re(routes), ref_rg(routes), ref_off(held1 + 1u, 0u), ref_pr(routes), ref_st(routes), ref_tu(held1 + 1u, 0u), ref_td(held1 + 1u, 0u);
			std::vector<float> ref_rw(routes), ref_pw(routes);
			std::vector<uint8_t> erank(experts, 255u);
			uint32_t rload[ranks] = {};
			const uint32_t groups = local + copies * local;
			for ( uint32_t i = 0u; i < routes; ++i )
			{
				if ( row_inv != 0u && (i % topk) == 0u ) memset(rload, 0, sizeof(rload));
				const uint32_t e = hre_in[i];
				ref_rg[i] = e;
				uint32_t chosen = ranks;
				if ( row_inv == 0u && erank[e] < ranks ) { chosen = erank[e]; rload[chosen] += 1u; }
				else
				{
					chosen = hown[e];
					uint32_t cl = rload[chosen];
					for ( uint32_t c = 0u; c < copies; ++c )
					{
						const uint32_t cand = hbk[c * experts + e];
						if ( cand < ranks && rload[cand] < cl ) { chosen = cand; cl = rload[cand]; }
					}
					if ( row_inv == 0u ) erank[e] = (uint8_t)chosen;
					rload[chosen] = cl + (row_inv != 0u ? 1u : 17u);
				}
				if ( chosen == 0u && hgrp[e] < groups ) { ref_re[i] = hgrp[e]; ref_rw[i] = hrw_in[i]; }
				else { ref_re[i] = groups; ref_rw[i] = 0.0f; }
			}
			std::vector<uint32_t> cnt(held1, 0u), cur(held1, 0u);
			for ( uint32_t i = 0u; i < routes; ++i ) cnt[ref_re[i]]++;
			for ( uint32_t g = 0u; g < held1; ++g )
			{
				ref_off[g + 1u] = ref_off[g] + cnt[g];
				const uint32_t mt = cnt[g] != 0u ? (cnt[g] + 15u) / 16u : 0u;
				ref_tu[g + 1u] = ref_tu[g] + mt * 36u;
				ref_td[g + 1u] = ref_td[g] + mt * 40u;
				cur[g] = ref_off[g];
			}
			for ( uint32_t i = 0u; i < routes; ++i )
			{
				const uint32_t p = cur[ref_re[i]]++;
				ref_pr[i] = p; ref_st[p] = i / topk; ref_pw[p] = ref_rw[i];
			}
			auto cmp = [&](const void *d, const void *h, size_t n, const char *tag) {
				std::vector<uint32_t> got(n);
				TEST_CHECK(cudaMemcpy(got.data(), d, n * 4u, cudaMemcpyDeviceToHost));
				if ( memcmp(got.data(), h, n * 4u) != 0 ) { fprintf(stderr, "MISMATCH BalanceRoute %s row_inv=%u\n", tag, row_inv); exit(1); }
			};
			cmp(dre, ref_re.data(), routes, "re"); cmp(drg, ref_rg.data(), routes, "rg");
			cmp(drw, ref_rw.data(), routes, "rw"); cmp(doff, ref_off.data(), held1 + 1u, "off");
			cmp(dpr, ref_pr.data(), routes, "pr"); cmp(dst, ref_st.data(), routes, "st");
			cmp(dtu, ref_tu.data(), held1 + 1u, "tu"); cmp(dtd, ref_td.data(), held1 + 1u, "td");
			cmp(dpw, ref_pw.data(), routes, "pw");
		}
		printf("PASS D41BalanceRouteKernel (warp-32 speculative rows=512, row_inv=0/1): 0 diffs\n");
		cudaFree(down); cudaFree(dbk); cudaFree(dgrp); cudaFree(dre); cudaFree(drg); cudaFree(drw);
		cudaFree(doff); cudaFree(dpr); cudaFree(dst); cudaFree(dtu); cudaFree(dtd); cudaFree(dpw);
	}
	{
		const uint32_t routes = 96u, rows = 16u, slice = 1152u, inter = 4608u, deg = 4u, hslice = 1280u;
		const uint32_t route_el = slice / 2u + slice / 64u;
		const uint64_t stride = (uint64_t)routes * slice + (uint64_t)rows * hslice;
		std::vector<uint16_t> hgu((size_t)routes * 2u * slice), hsh((size_t)rows * hslice * deg);
		std::vector<float> hpw(routes);
		for ( auto &v : hgu ) v = TestRandBf16(rng, 2.0f);
		for ( auto &v : hsh ) v = TestRandBf16(rng, 2.0f);
		for ( uint32_t i = 0u; i < routes; ++i ) hpw[i] = 0.25f + (float)(i % 5u) * 0.1f;
		uint16_t *dgu = 0, *dsh = 0, *ds1 = 0, *ds2 = 0, *da1 = 0, *da2 = 0;
		float *dpw = 0;
		TEST_CHECK(cudaMalloc(&dgu, hgu.size() * 2u)); TEST_CHECK(cudaMalloc(&dsh, hsh.size() * 2u));
		TEST_CHECK(cudaMalloc(&dpw, routes * 4u));
		TEST_CHECK(cudaMalloc(&ds1, (size_t)deg * stride * 2u)); TEST_CHECK(cudaMalloc(&ds2, (size_t)deg * stride * 2u));
		TEST_CHECK(cudaMalloc(&da1, (size_t)routes * inter * 2u)); TEST_CHECK(cudaMalloc(&da2, (size_t)routes * inter * 2u));
		TEST_CHECK(cudaMemcpy(dgu, hgu.data(), hgu.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsh, hsh.data(), hsh.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dpw, hpw.data(), routes * 4u, cudaMemcpyHostToDevice));
		for ( uint32_t codes = 0u; codes < 2u; ++codes )
		{
			const uint32_t rel = codes != 0u ? route_el : slice;
			TEST_CHECK(cudaMemset(ds1, 0, (size_t)deg * stride * 2u)); TEST_CHECK(cudaMemset(ds2, 0, (size_t)deg * stride * 2u));
			D41MoeSendKernel<<<D41ItemGrid((hslice * deg + 255u) / 256u, routes + rows), 256u>>>(dgu, dpw, dsh, ds1, routes, rows, slice, hslice, deg, stride, 7.0f, rel, codes);
			D41MoeSendVecKernel<<<D41ItemGrid((hslice * deg / 8u + 255u) / 256u, routes + rows), 256u>>>(dgu, dpw, dsh, ds2, routes, rows, slice, hslice, deg, stride, 7.0f, rel, codes);
			D41ActAssembleQdqKernel<<<D41ItemGrid(inter / 256u, routes), 256u>>>(ds1, da1, slice, inter, stride, routes, codes);
			D41ActAssembleQdqVecKernel<<<D41ItemGrid((inter / 8u + 255u) / 256u, routes), 256u>>>(ds2, da2, slice, inter, stride, routes, codes);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<uint16_t> s1((size_t)deg * stride), s2((size_t)deg * stride), a1((size_t)routes * inter), a2((size_t)routes * inter);
			TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size() * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size() * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(a1.data(), da1, a1.size() * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(a2.data(), da2, a2.size() * 2u, cudaMemcpyDeviceToHost));
			if ( s1 != s2 || a1 != a2 ) { fprintf(stderr, "MISMATCH MoeSendVec/ActAssembleQdqVec codes=%u\n", codes); exit(1); }
		}
		printf("PASS D41MoeSendVecKernel & D41ActAssembleQdqVecKernel (codes=0/1): 0 diffs\n");
		cudaFree(dgu); cudaFree(dsh); cudaFree(dpw); cudaFree(ds1); cudaFree(ds2); cudaFree(da1); cudaFree(da2);
	}
	{
		const uint32_t rows = 8u, deg = 4u, total_gran = 18u, gran = 128u, max_w = 640u, inter = 2304u;
		const uint64_t stride = (uint64_t)rows * (max_w / 2u + max_w / 64u);
		std::vector<uint8_t> hrec((size_t)deg * stride * 2u);
		for ( uint32_t s = 0u; s < deg; ++s )
		{
			const uint32_t base = total_gran / deg, extra = total_gran % deg;
			const uint32_t width = (base + (s < extra ? 1u : 0u)) * gran;
			uint8_t *p = hrec.data() + (size_t)s * stride * 2u;
			for ( uint32_t i = 0u; i < rows * width; ++i ) p[i] = (uint8_t)(TestNext(rng) & 0x7eu);
			for ( uint32_t i = 0u; i < rows * (width / 32u); ++i ) p[rows * width + i] = (uint8_t)(120u + (TestNext(rng) & 7u));
		}
		uint16_t *drec = 0, *da1 = 0, *da2 = 0;
		TEST_CHECK(cudaMalloc(&drec, hrec.size()));
		TEST_CHECK(cudaMalloc(&da1, (size_t)rows * inter * 2u)); TEST_CHECK(cudaMalloc(&da2, (size_t)rows * inter * 2u));
		TEST_CHECK(cudaMemcpy(drec, hrec.data(), hrec.size(), cudaMemcpyHostToDevice));
		for ( uint32_t s = 0u; s < deg; ++s )
		{
			const uint32_t base = total_gran / deg, extra = total_gran % deg;
			const uint32_t first = (s * base + (s < extra ? s : extra)) * gran;
			const uint32_t width = (base + (s < extra ? 1u : 0u)) * gran;
			D41SharedDequantKernel<<<dim3((width / 8u + 255u) / 256u, rows), 256u>>>((const uint8_t *)(drec + (uint64_t)s * stride),
				da1 + (uint64_t)first * rows, rows, width);
		}
		D41SharedDequantAllKernel<<<dim3((max_w / 8u + 255u) / 256u, rows, deg), 256u>>>(drec, da2, rows, stride, deg, total_gran, gran);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> a1((size_t)rows * inter), a2((size_t)rows * inter);
		TEST_CHECK(cudaMemcpy(a1.data(), da1, a1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(a2.data(), da2, a2.size() * 2u, cudaMemcpyDeviceToHost));
		if ( a1 != a2 ) { fprintf(stderr, "MISMATCH D41SharedDequantAllKernel\n"); exit(1); }
		printf("PASS D41SharedDequantAllKernel (rows=8, deg=4): 0 diffs\n");
		cudaFree(drec); cudaFree(da1); cudaFree(da2);
	}
}

static void TestTp4UnpackAndCompact(uint32_t *rng)
{
	{
		const uint32_t qlora = 1280u, own = 16u, ranks = 4u, rows = own * ranks;
		const uint64_t seg = (uint64_t)own * (qlora + qlora / 32u);
		std::vector<uint8_t> hrec((size_t)ranks * seg), ref_c((size_t)rows * qlora), ref_s((size_t)rows * (qlora / 32u));
		for ( uint32_t s = 0u; s < ranks; ++s )
			for ( uint32_t l = 0u; l < own; ++l )
			{
				const uint32_t row = s * own + l;
				for ( uint32_t i = 0u; i < qlora; ++i )
				{
					const uint8_t v = (uint8_t)(TestNext(rng) & 0x7eu);
					hrec[(size_t)s * seg + (size_t)l * qlora + i] = v;
					ref_c[(size_t)row * qlora + i] = v;
				}
				for ( uint32_t i = 0u; i < qlora / 32u; ++i )
				{
					const uint8_t v = (uint8_t)(120u + (TestNext(rng) & 7u));
					hrec[(size_t)s * seg + (size_t)own * qlora + (size_t)l * (qlora / 32u) + i] = v;
					ref_s[(size_t)row * (qlora / 32u) + i] = v;
				}
			}
		uint8_t *drec = 0, *dc = 0, *ds = 0;
		TEST_CHECK(cudaMalloc(&drec, hrec.size()));
		TEST_CHECK(cudaMalloc(&dc, ref_c.size()));
		TEST_CHECK(cudaMalloc(&ds, ref_s.size()));
		TEST_CHECK(cudaMemcpy(drec, hrec.data(), hrec.size(), cudaMemcpyHostToDevice));
		D41UnpackQCodesKernel<1280u><<<dim3(1u, rows), 128u>>>(drec, dc, ds, own, seg);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint8_t> gc(ref_c.size()), gs(ref_s.size());
		TEST_CHECK(cudaMemcpy(gc.data(), dc, gc.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(gs.data(), ds, gs.size(), cudaMemcpyDeviceToHost));
		if ( gc != ref_c || gs != ref_s ) { fprintf(stderr, "MISMATCH D41UnpackQCodesKernel\n"); exit(1); }
		printf("PASS D41UnpackQCodesKernel (rows=%u): 0 diffs\n", rows);
		cudaFree(drec); cudaFree(dc); cudaFree(ds);
	}
	{
		const uint32_t experts = 384u;
		std::vector<uint32_t> off(experts + 1u, 0u), ref_act;
		for ( uint32_t e = 0u; e < experts; ++e )
		{
			const uint32_t c = (TestNext(rng) % 7u == 0u) ? (1u + (TestNext(rng) % 5u)) : 0u;
			off[e + 1u] = off[e] + c;
			if ( c != 0u ) ref_act.push_back(e);
		}
		uint32_t *doff = 0, *dact = 0;
		TEST_CHECK(cudaMalloc(&doff, off.size() * 4u));
		TEST_CHECK(cudaMalloc(&dact, (experts + 1u) * 4u));
		TEST_CHECK(cudaMemcpy(doff, off.data(), off.size() * 4u, cudaMemcpyHostToDevice));
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff, dact);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint32_t> got(experts + 1u);
		TEST_CHECK(cudaMemcpy(got.data(), dact, got.size() * 4u, cudaMemcpyDeviceToHost));
		if ( got[experts] != (uint32_t)ref_act.size() ) { fprintf(stderr, "MISMATCH D41CompactActiveExpertsKernel count\n"); exit(1); }
		for ( size_t i = 0; i < ref_act.size(); ++i )
			if ( got[i] != ref_act[i] ) { fprintf(stderr, "MISMATCH D41CompactActiveExpertsKernel idx=%zu\n", i); exit(1); }
		printf("PASS D41CompactActiveExpertsKernel (active=%zu/%u): 0 diffs\n", ref_act.size(), experts);
		cudaFree(doff); cudaFree(dact);
	}
}

static void TestTp4HcMixK32Parity(uint32_t *rng)
{
	const uint32_t hidden = 5120u, hc = 4u, flat = hc * hidden, mix = 24u, tasks = 25u;
	std::vector<float> h_fn((size_t)mix * flat), h_sc(3u), h_bs(mix);
	std::vector<uint16_t> h_nw(hidden);
	for ( auto &v : h_fn ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.05f;
	for ( auto &v : h_sc ) v = 0.25f + ((float)(TestNext(rng) & 0xffu) / 512.0f);
	for ( auto &v : h_bs ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.5f;
	for ( auto &v : h_nw ) v = TestRandBf16(rng, 1.5f);
	float *dfn = 0, *dsc = 0, *dbs = 0;
	uint16_t *dnw = 0;
	TEST_CHECK(cudaMalloc(&dfn, h_fn.size() * 4u));
	TEST_CHECK(cudaMalloc(&dsc, 12u));
	TEST_CHECK(cudaMalloc(&dbs, mix * 4u));
	TEST_CHECK(cudaMalloc(&dnw, hidden * 2u));
	TEST_CHECK(cudaMemcpy(dfn, h_fn.data(), h_fn.size() * 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dsc, h_sc.data(), 12u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dbs, h_bs.data(), mix * 4u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dnw, h_nw.data(), hidden * 2u, cudaMemcpyHostToDevice));
	for ( uint32_t rows : { 4u, 8u, 16u, 32u } )
	{
		std::vector<uint16_t> h_st((size_t)rows * flat);
		std::vector<float> h_cpre((size_t)rows * hc);
		for ( auto &v : h_st ) v = TestRandBf16(rng, 1.5f);
		for ( auto &v : h_cpre ) v = 0.25f + (((float)(TestNext(rng) & 0xffu) / 256.0f) - 0.5f) * 0.4f;
		uint16_t *dst = 0, *dx1 = 0, *dx2 = 0;
		uint8_t *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
		float *dcpre = 0, *dpart1 = 0, *dpart2 = 0, *dpr1 = 0, *dpo1 = 0, *dcb1 = 0, *dpr2 = 0, *dpo2 = 0, *dcb2 = 0;
		uint32_t *dcnt = 0;
		TEST_CHECK(cudaMalloc(&dst, h_st.size() * 2u));
		TEST_CHECK(cudaMalloc(&dcpre, h_cpre.size() * 4u));
		TEST_CHECK(cudaMalloc(&dpart1, (size_t)rows * tasks * 8u * 4u));
		TEST_CHECK(cudaMalloc(&dpart2, (size_t)rows * tasks * 32u * 4u));
		TEST_CHECK(cudaMalloc(&dpr1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb1, (size_t)rows * hc * hc * 4u));
		TEST_CHECK(cudaMalloc(&dpr2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb2, (size_t)rows * hc * hc * 4u));
		TEST_CHECK(cudaMalloc(&dx1, (size_t)rows * hidden * 2u)); TEST_CHECK(cudaMalloc(&dx2, (size_t)rows * hidden * 2u));
		TEST_CHECK(cudaMalloc(&dc1, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * hidden / 32u));
		TEST_CHECK(cudaMalloc(&dc2, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * hidden / 32u));
		TEST_CHECK(cudaMalloc(&dcnt, (size_t)rows * 4u));
		TEST_CHECK(cudaMemset(dcnt, 0, (size_t)rows * 4u));
		TEST_CHECK(cudaMemcpy(dst, h_st.data(), h_st.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dcpre, h_cpre.data(), h_cpre.size() * 4u, cudaMemcpyHostToDevice));
		if ( LmHcSinkhornMixes<4u>(dst, dfn, dsc, dbs, dpart1, dpr1, dpo1, dcb1, rows, hidden, 1e-20f, 20u, 1e-6f, 0) != LM_LAUNCH_OK ) exit(1);
		TestRmsNormMxKernel<256u><<<rows, 256u, (hidden + 8u) * 4u>>>(0, dst, dcpre, dnw, dx1, (uint16_t *)0, hidden, 1e-20f, dc1, ds1);
		if ( rows <= 4u )
			D41HcMixNormK32Kernel<4u, 5120u><<<32u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart2, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
				(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
		else if ( rows <= 8u )
			D41HcMixNormK32Kernel<8u, 5120u><<<32u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart2, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
				(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
		else
			D41HcMixNormK32Kernel<16u, 5120u><<<((rows + 15u) / 16u) * 32u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart2, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
				(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<float> pr1((size_t)rows * hc), pr2((size_t)rows * hc), po1((size_t)rows * hc), po2((size_t)rows * hc), cb1((size_t)rows * hc * hc), cb2((size_t)rows * hc * hc);
		std::vector<uint16_t> x1((size_t)rows * hidden), x2((size_t)rows * hidden);
		std::vector<uint8_t> c1((size_t)rows * hidden), c2((size_t)rows * hidden), s1((size_t)rows * hidden / 32u), s2((size_t)rows * hidden / 32u);
		TEST_CHECK(cudaMemcpy(pr1.data(), dpr1, pr1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(pr2.data(), dpr2, pr2.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(po1.data(), dpo1, po1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(po2.data(), dpo2, po2.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(cb1.data(), dcb1, cb1.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(cb2.data(), dcb2, cb2.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(x1.data(), dx1, x1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(x2.data(), dx2, x2.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
		float max_d = 0.0f;
		for ( size_t i = 0; i < pr1.size(); ++i ) max_d = fmaxf(max_d, fabsf(pr1[i] - pr2[i]));
		for ( size_t i = 0; i < po1.size(); ++i ) max_d = fmaxf(max_d, fabsf(po1[i] - po2[i]));
		for ( size_t i = 0; i < cb1.size(); ++i ) max_d = fmaxf(max_d, fabsf(cb1[i] - cb2[i]));
		if ( max_d > 1e-5f || x1 != x2 || c1 != c2 || s1 != s2 )
		{ fprintf(stderr, "MISMATCH D41HcMixNormK32Kernel rows=%u max_d=%.9g\n", rows, max_d); exit(1); }
		cudaFree(dst); cudaFree(dcpre); cudaFree(dpart1); cudaFree(dpart2); cudaFree(dpr1); cudaFree(dpo1); cudaFree(dcb1);
		cudaFree(dpr2); cudaFree(dpo2); cudaFree(dcb2); cudaFree(dx1); cudaFree(dx2); cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2); cudaFree(dcnt);
	}
	cudaFree(dfn); cudaFree(dsc); cudaFree(dbs); cudaFree(dnw);
	printf("PASS D41HcMixNormK32Kernel (rows=4/8/16/32): exact collapse+mx, max sinkhorn diff <= 1e-5\n");
}

static void TestTp4Batch16AndSkip(uint32_t *rng)
{
	const uint32_t rows = 8u, k0 = 640u, k1 = 512u, n0 = 1280u, n1 = 1280u;
	std::vector<uint8_t> ha0((size_t)rows * k0), has0((size_t)rows * (k0 / 32u));
	std::vector<uint8_t> ha1((size_t)rows * k1), has1((size_t)rows * (k1 / 32u));
	std::vector<uint8_t> hw0((size_t)n0 * k0), hws0((size_t)n0 * (k0 / 32u));
	std::vector<uint8_t> hw1((size_t)n1 * k1), hws1((size_t)n1 * (k1 / 32u));
	for ( auto &v : ha0 ) v = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
	for ( auto &v : ha1 ) v = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
	for ( auto &v : hw0 ) v = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
	for ( auto &v : hw1 ) v = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
	for ( auto &v : has0 ) v = (uint8_t)(124u + (TestNext(rng) & 3u));
	for ( auto &v : has1 ) v = (uint8_t)(124u + (TestNext(rng) & 3u));
	for ( auto &v : hws0 ) v = (uint8_t)(124u + (TestNext(rng) & 3u));
	for ( auto &v : hws1 ) v = (uint8_t)(124u + (TestNext(rng) & 3u));
	uint8_t *da0 = 0, *das0 = 0, *da1 = 0, *das1 = 0, *dw0 = 0, *dws0 = 0, *dw1 = 0, *dws1 = 0;
	uint16_t *dr = 0, *dg = 0;
	const uint32_t stride = n0 + n1;
	TEST_CHECK(cudaMalloc(&da0, ha0.size())); TEST_CHECK(cudaMalloc(&das0, has0.size()));
	TEST_CHECK(cudaMalloc(&da1, ha1.size())); TEST_CHECK(cudaMalloc(&das1, has1.size()));
	TEST_CHECK(cudaMalloc(&dw0, hw0.size())); TEST_CHECK(cudaMalloc(&dws0, hws0.size()));
	TEST_CHECK(cudaMalloc(&dw1, hw1.size())); TEST_CHECK(cudaMalloc(&dws1, hws1.size()));
	TEST_CHECK(cudaMalloc(&dr, (size_t)rows * stride * 2u)); TEST_CHECK(cudaMalloc(&dg, (size_t)rows * stride * 2u));
	TEST_CHECK(cudaMemcpy(da0, ha0.data(), ha0.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(das0, has0.data(), has0.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(da1, ha1.data(), ha1.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(das1, has1.data(), has1.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dw0, hw0.data(), hw0.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dws0, hws0.data(), hws0.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dw1, hw1.data(), hw1.size(), cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dws1, hws1.data(), hws1.size(), cudaMemcpyHostToDevice));
	D41MxGemmSmallKernel<16u, 32u, 1u, 2u, 4u, false><<<dim3(n0 / 32u, 1u), 64u>>>(da0, das0, dw0, dws0, dr, n0, 0, 0, 0, 0u, rows, k0, stride, 0u);
	D41MxGemmSmallKernel<16u, 32u, 1u, 2u, 4u, false><<<dim3(n1 / 32u, 1u), 64u>>>(da1, das1, dw1, dws1, dr, n1, 0, 0, 0, 0u, rows, k1, stride, n0);
	D41MxBatchDesc desc;
	memset(&desc, 0, sizeof(desc));
	desc.act[0] = da0; desc.act_scale[0] = das0; desc.w[0] = dw0; desc.ws[0] = dws0; desc.out[0] = dg;
	desc.in_dim[0] = k0; desc.out_dim[0] = n0; desc.out_stride[0] = stride; desc.out_col[0] = 0u; desc.tile_end[0] = n0 / 16u;
	desc.act[1] = da1; desc.act_scale[1] = das1; desc.w[1] = dw1; desc.ws[1] = dws1; desc.out[1] = dg;
	desc.in_dim[1] = k1; desc.out_dim[1] = n1; desc.out_stride[1] = stride; desc.out_col[1] = n0; desc.tile_end[1] = (n0 + n1) / 16u;
	desc.tile_end[2] = desc.tile_end[1]; desc.tile_end[3] = desc.tile_end[1];
	D41MxGemmBatch16Kernel<3u><<<dim3(desc.tile_end[1], 1u, 1u), 128u>>>(desc, rows);
	TEST_CHECK(cudaDeviceSynchronize());
	std::vector<uint16_t> vr((size_t)rows * stride), vg((size_t)rows * stride);
	TEST_CHECK(cudaMemcpy(vr.data(), dr, vr.size() * 2u, cudaMemcpyDeviceToHost));
	TEST_CHECK(cudaMemcpy(vg.data(), dg, vg.size() * 2u, cudaMemcpyDeviceToHost));
	float max_rel = 0.0f;
	for ( size_t i = 0; i < vr.size(); ++i )
	{
		const uint32_t ua = (uint32_t)vr[i] << 16u, ub = (uint32_t)vg[i] << 16u;
		float a, b;
		memcpy(&a, &ua, 4u); memcpy(&b, &ub, 4u);
		const float denom = fmaxf(1.0f, fabsf(a));
		max_rel = fmaxf(max_rel, fabsf(a - b) / denom);
	}
	if ( max_rel > 0.01f ) { fprintf(stderr, "MISMATCH D41MxGemmBatch16Kernel max_rel=%.6g\n", max_rel); exit(1); }
	uint32_t hseq[8] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
	uint32_t hlen_lo[8] = { 100u, 200u, 300u, 400u, 500u, 600u, 700u, 2048u };
	uint32_t hlen_hi[8] = { 100u, 200u, 300u, 400u, 500u, 600u, 2049u, 100u };
	uint32_t *dseq = 0, *dlen = 0;
	uint16_t *dout0 = 0, *dout1 = 0;
	TEST_CHECK(cudaMalloc(&dseq, 32u)); TEST_CHECK(cudaMalloc(&dlen, 32u));
	TEST_CHECK(cudaMalloc(&dout0, (size_t)rows * n0 * 2u)); TEST_CHECK(cudaMalloc(&dout1, (size_t)rows * n1 * 2u));
	TEST_CHECK(cudaMemcpy(dseq, hseq, 32u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemcpy(dlen, hlen_lo, 32u, cudaMemcpyHostToDevice));
	TEST_CHECK(cudaMemset(dout1, 0x5a, (size_t)rows * n1 * 2u));
	D41MxGemmSmallKernel<16u, 32u, 1u, 2u, 4u, false><<<dim3(n0 / 32u + n1 / 32u, 1u), 64u>>>(da0, das0, dw0, dws0, dout0, n0, dw0, dws0, dout1, n1,
		rows, k0, 0u, 0u, dseq, dlen, (const uint32_t *)0, 2048u);
	TEST_CHECK(cudaDeviceSynchronize());
	std::vector<uint16_t> sk((size_t)rows * n1);
	TEST_CHECK(cudaMemcpy(sk.data(), dout1, sk.size() * 2u, cudaMemcpyDeviceToHost));
	for ( auto v : sk ) if ( v != 0x5a5au ) { fprintf(stderr, "MISMATCH idx_skip did not skip\n"); exit(1); }
	TEST_CHECK(cudaMemcpy(dlen, hlen_hi, 32u, cudaMemcpyHostToDevice));
	D41MxGemmSmallKernel<16u, 32u, 1u, 2u, 4u, false><<<dim3(n0 / 32u + n1 / 32u, 1u), 64u>>>(da0, das0, dw0, dws0, dout0, n0, dw0, dws0, dout1, n1,
		rows, k0, 0u, 0u, dseq, dlen, (const uint32_t *)0, 2048u);
	TEST_CHECK(cudaDeviceSynchronize());
	TEST_CHECK(cudaMemcpy(sk.data(), dout1, sk.size() * 2u, cudaMemcpyDeviceToHost));
	uint32_t changed = 0u;
	for ( auto v : sk ) if ( v != 0x5a5au ) ++changed;
	if ( changed == 0u ) { fprintf(stderr, "MISMATCH idx_skip skipped when row exceeded bound\n"); exit(1); }
	printf("PASS D41MxGemmBatch16Kernel (max_rel=%.6g) & multi-row idx_skip\n", max_rel);
	cudaFree(da0); cudaFree(das0); cudaFree(da1); cudaFree(das1); cudaFree(dw0); cudaFree(dws0); cudaFree(dw1); cudaFree(dws1);
	cudaFree(dr); cudaFree(dg); cudaFree(dseq); cudaFree(dlen); cudaFree(dout0); cudaFree(dout1);
}

static void TestTp4W2DirectParity(uint32_t *rng)
{
	const uint32_t experts = 384u, slice = 576u, inter = 2304u, out_dim = 1280u, deg = 4u;
	const uint64_t gbytes = (uint64_t)(inter / LM_SKINNY_CELL_TILE_K) * (out_dim / LM_SKINNY_CELL_NEURONS) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	const uint32_t active_e = 6u;
	std::vector<uint8_t> hw((size_t)experts * gbytes, 0u);
	for ( uint32_t e = 0u; e < active_e; ++e )
	{
		uint8_t *ep = hw.data() + (size_t)e * gbytes;
		for ( size_t i = 0; i < gbytes; ++i ) ep[i] = (uint8_t)(TestNext(rng) & 0xffu);
		for ( uint32_t c = 0u; c < (inter / 32u) * (out_dim / 16u); ++c )
			for ( uint32_t j = 0u; j < 16u; ++j )
				ep[(size_t)c * 272u + 256u + j] = (uint8_t)(124u + (TestNext(rng) & 3u));
	}
	uint8_t *dw = 0;
	TEST_CHECK(cudaMalloc(&dw, hw.size()));
	TEST_CHECK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
	for ( uint32_t routes : { 24u, 96u } )
	{
		const uint64_t rstride = (uint64_t)routes * (slice / 2u + slice / 64u);
		std::vector<uint8_t> hrecv((size_t)deg * rstride * 2u);
		for ( uint32_t s = 0u; s < deg; ++s )
		{
			uint8_t *p = hrecv.data() + (size_t)s * rstride * 2u;
			for ( uint32_t i = 0u; i < routes * slice; ++i ) p[i] = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
			for ( uint32_t i = 0u; i < routes * (slice / 32u); ++i ) p[routes * slice + i] = (uint8_t)(124u + (TestNext(rng) & 3u));
		}
		std::vector<uint32_t> off(experts + 1u, routes);
		for ( uint32_t e = 0u; e < active_e; ++e ) off[e] = e * (routes / active_e);
		uint16_t *drecv = 0, *dr = 0, *dg = 0;
		uint8_t *dact = 0;
		uint32_t *doff = 0, *dact_e = 0;
		TEST_CHECK(cudaMalloc(&drecv, hrecv.size()));
		TEST_CHECK(cudaMalloc(&dact, (size_t)routes * (inter + inter / 32u)));
		TEST_CHECK(cudaMalloc(&dr, (size_t)routes * out_dim * 2u));
		TEST_CHECK(cudaMalloc(&dg, (size_t)routes * out_dim * 2u));
		TEST_CHECK(cudaMalloc(&doff, off.size() * 4u));
		TEST_CHECK(cudaMalloc(&dact_e, (experts + 1u) * 4u));
		TEST_CHECK(cudaMemcpy(drecv, hrecv.data(), hrecv.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(doff, off.data(), off.size() * 4u, cudaMemcpyHostToDevice));
		D41ActAssembleCodesKernel<<<D41ItemGrid((inter / 16u + inter / 64u + 255u) / 256u, routes), 256u>>>(drecv, dact, slice, inter, rstride, routes);
		if ( LmCellMxExperts(dw, dact, dact + (uint64_t)routes * inter, dr, doff, (const uint32_t *)0, experts, routes, 1u, inter, out_dim, 0) != LM_LAUNCH_OK )
			exit(1);
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff, dact_e);
		if ( routes <= 32u )
			D41CellMxW2DirectKernel<1u, 3u, 384u, 576u, 2304u, 1280u><<<routes * (out_dim / 128u), 256u>>>(dw, drecv, dg, doff, dact_e, routes, rstride);
		else
			D41CellMxW2DirectKernel<2u, 2u, 384u, 576u, 2304u, 1280u><<<routes * (out_dim / 128u), 256u>>>(dw, drecv, dg, doff, dact_e, routes, rstride);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> vr((size_t)routes * out_dim), vg((size_t)routes * out_dim);
		TEST_CHECK(cudaMemcpy(vr.data(), dr, vr.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(vg.data(), dg, vg.size() * 2u, cudaMemcpyDeviceToHost));
		if ( vr != vg ) { fprintf(stderr, "MISMATCH D41CellMxW2DirectKernel routes=%u\n", routes); exit(1); }
		cudaFree(drecv); cudaFree(dact); cudaFree(dr); cudaFree(dg); cudaFree(doff); cudaFree(dact_e);
	}
	cudaFree(dw);
	printf("PASS D41CellMxW2DirectKernel (<1,3> & <2,2>): 0 diffs\n");
}

static float TestBf16ToFloat(uint16_t v)
{
	const uint32_t u = (uint32_t)v << 16u;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static float TestE4m3ToFloatHost(uint8_t c)
{
	const uint32_t s = (c >> 7u) & 1u, e = (c >> 3u) & 15u, m = c & 7u;
	float val;
	if ( e == 0u ) val = ldexpf((float)m, -9);
	else val = ldexpf((float)(8u + m), (int)e - 10);
	return s != 0u ? -val : val;
}

static float TestUe8m0ToFloatHost(uint8_t e)
{
	const uint32_t u = (uint32_t)e << 23u;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static void TestTp4BaseRLeapfrogParity(uint32_t *rng)
{
	{
		const uint32_t hidden = 5120u, hc = 4u, flat = hc * hidden, mix = 24u;
		std::vector<float> h_fn((size_t)mix * flat), h_sc(3u), h_bs(mix);
		std::vector<uint16_t> h_nw(hidden);
		for ( auto &v : h_fn ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.05f;
		for ( auto &v : h_sc ) v = 0.25f + ((float)(TestNext(rng) & 0xffu) / 512.0f);
		for ( auto &v : h_bs ) v = (((float)(TestNext(rng) & 0xfffu) / 2048.0f) - 1.0f) * 0.5f;
		for ( auto &v : h_nw ) v = TestRandBf16(rng, 1.5f);
		float *dfn = 0, *dsc = 0, *dbs = 0;
		uint16_t *dnw = 0;
		TEST_CHECK(cudaMalloc(&dfn, h_fn.size() * 4u)); TEST_CHECK(cudaMalloc(&dsc, 12u));
		TEST_CHECK(cudaMalloc(&dbs, mix * 4u)); TEST_CHECK(cudaMalloc(&dnw, hidden * 2u));
		TEST_CHECK(cudaMemcpy(dfn, h_fn.data(), h_fn.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsc, h_sc.data(), 12u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbs, h_bs.data(), mix * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dnw, h_nw.data(), hidden * 2u, cudaMemcpyHostToDevice));
		for ( uint32_t rows : { 4u, 8u, 16u, 64u } )
		{
			std::vector<uint16_t> h_st((size_t)rows * flat);
			std::vector<float> h_cpre((size_t)rows * hc);
			for ( auto &v : h_st ) v = TestRandBf16(rng, 1.5f);
			for ( auto &v : h_cpre ) v = 0.25f + (((float)(TestNext(rng) & 0xffu) / 256.0f) - 0.5f) * 0.4f;
			uint16_t *dst = 0, *dx1 = 0, *dx2 = 0;
			uint8_t *dc1 = 0, *ds1 = 0, *dc2 = 0, *ds2 = 0;
			float *dcpre = 0, *dpart = 0, *dpr1 = 0, *dpo1 = 0, *dcb1 = 0, *dpr2 = 0, *dpo2 = 0, *dcb2 = 0;
			uint32_t *dcnt = 0;
			TEST_CHECK(cudaMalloc(&dst, h_st.size() * 2u)); TEST_CHECK(cudaMalloc(&dcpre, h_cpre.size() * 4u));
			TEST_CHECK(cudaMalloc(&dpart, (size_t)rows * 40u * 25u * 4u));
			TEST_CHECK(cudaMalloc(&dpr1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo1, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb1, (size_t)rows * hc * hc * 4u));
			TEST_CHECK(cudaMalloc(&dpr2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dpo2, (size_t)rows * hc * 4u)); TEST_CHECK(cudaMalloc(&dcb2, (size_t)rows * hc * hc * 4u));
			TEST_CHECK(cudaMalloc(&dx1, (size_t)rows * hidden * 2u)); TEST_CHECK(cudaMalloc(&dx2, (size_t)rows * hidden * 2u));
			TEST_CHECK(cudaMalloc(&dc1, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds1, (size_t)rows * hidden / 32u));
			TEST_CHECK(cudaMalloc(&dc2, (size_t)rows * hidden)); TEST_CHECK(cudaMalloc(&ds2, (size_t)rows * hidden / 32u));
			TEST_CHECK(cudaMalloc(&dcnt, (size_t)rows * 4u)); TEST_CHECK(cudaMemset(dcnt, 0, (size_t)rows * 4u));
			TEST_CHECK(cudaMemcpy(dst, h_st.data(), h_st.size() * 2u, cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(dcpre, h_cpre.data(), h_cpre.size() * 4u, cudaMemcpyHostToDevice));
			if ( LmHcSinkhornMixes<4u>(dst, dfn, dsc, dbs, dpart, dpr1, dpo1, dcb1, rows, hidden, 1e-20f, 20u, 1e-6f, 0) != LM_LAUNCH_OK ) exit(1);
			TestRmsNormMxKernel<256u><<<rows, 256u, (hidden + 8u) * 4u>>>(0, dst, dcpre, dnw, dx1, (uint16_t *)0, hidden, 1e-20f, dc1, ds1);
			if ( rows <= 4u )
				D41HcMixNormK40Kernel<4u, 5120u><<<40u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
					(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
			else if ( rows <= 8u )
				D41HcMixNormK40Kernel<8u, 5120u><<<40u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
					(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
			else
				D41HcMixNormK40Kernel<16u, 5120u><<<((rows + 15u) / 16u) * 40u + rows, 256u>>>(dst, dfn, dsc, dbs, dpart, dpr2, dpo2, dcb2, dcnt, dcpre, dnw, dx2,
					(uint16_t *)0, dc2, ds2, rows, 1e-20f, 20u, 1e-6f);
			TEST_CHECK(cudaDeviceSynchronize());
			std::vector<float> pr1((size_t)rows * hc), pr2((size_t)rows * hc), po1((size_t)rows * hc), po2((size_t)rows * hc), cb1((size_t)rows * hc * hc), cb2((size_t)rows * hc * hc);
			std::vector<uint16_t> x1((size_t)rows * hidden), x2((size_t)rows * hidden);
			std::vector<uint8_t> c1((size_t)rows * hidden), c2((size_t)rows * hidden), s1((size_t)rows * hidden / 32u), s2((size_t)rows * hidden / 32u);
			TEST_CHECK(cudaMemcpy(pr1.data(), dpr1, pr1.size() * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(pr2.data(), dpr2, pr2.size() * 4u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(po1.data(), dpo1, po1.size() * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(po2.data(), dpo2, po2.size() * 4u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(cb1.data(), dcb1, cb1.size() * 4u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(cb2.data(), dcb2, cb2.size() * 4u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(x1.data(), dx1, x1.size() * 2u, cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(x2.data(), dx2, x2.size() * 2u, cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(c1.data(), dc1, c1.size(), cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(c2.data(), dc2, c2.size(), cudaMemcpyDeviceToHost));
			TEST_CHECK(cudaMemcpy(s1.data(), ds1, s1.size(), cudaMemcpyDeviceToHost)); TEST_CHECK(cudaMemcpy(s2.data(), ds2, s2.size(), cudaMemcpyDeviceToHost));
			float md = 0.0f;
			for ( size_t i = 0; i < pr1.size(); ++i ) md = fmaxf(md, fabsf(pr1[i] - pr2[i]));
			for ( size_t i = 0; i < po1.size(); ++i ) md = fmaxf(md, fabsf(po1[i] - po2[i]));
			for ( size_t i = 0; i < cb1.size(); ++i ) md = fmaxf(md, fabsf(cb1[i] - cb2[i]));
			if ( md > 1e-4f || x1 != x2 || c1 != c2 || s1 != s2 )
			{ fprintf(stderr, "MISMATCH D41HcMixNormK40Kernel rows=%u md=%.6g\n", rows, md); exit(1); }
			cudaFree(dst); cudaFree(dcpre); cudaFree(dpart); cudaFree(dpr1); cudaFree(dpo1); cudaFree(dcb1);
			cudaFree(dpr2); cudaFree(dpo2); cudaFree(dcb2); cudaFree(dx1); cudaFree(dx2);
			cudaFree(dc1); cudaFree(ds1); cudaFree(dc2); cudaFree(ds2); cudaFree(dcnt);
		}
		cudaFree(dfn); cudaFree(dsc); cudaFree(dbs); cudaFree(dnw);
		printf("PASS D41HcMixNormK40Kernel (rows=4/8/16/64): exact\n");
	}

	{
		const uint32_t out_dim = 1280u, w = 576u, sb = w / 32u;
		uint8_t *dw[4], *dws[4];
		std::vector<std::vector<uint8_t>> hw(4), hws(4);
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			hw[s].resize((size_t)out_dim * w); hws[s].resize((size_t)out_dim * sb);
			for ( auto &b : hw[s] ) b = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
			for ( auto &b : hws[s] ) b = (uint8_t)(124u + (TestNext(rng) & 3u));
			TEST_CHECK(cudaMalloc(&dw[s], hw[s].size())); TEST_CHECK(cudaMalloc(&dws[s], hws[s].size()));
			TEST_CHECK(cudaMemcpy(dw[s], hw[s].data(), hw[s].size(), cudaMemcpyHostToDevice));
			TEST_CHECK(cudaMemcpy(dws[s], hws[s].data(), hws[s].size(), cudaMemcpyHostToDevice));
		}
		const uint32_t rows6 = 6u;
		const uint64_t stride6 = (uint64_t)rows6 * (w / 2u + sb / 2u) + 64u;
		const uint64_t stride1 = (uint64_t)1u * (w / 2u + sb / 2u) + 64u;
		std::vector<uint8_t> hrecv6((size_t)4u * stride6 * 2u, 0u), hrecv1((size_t)4u * stride1 * 2u, 0u);
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			uint8_t *p6 = hrecv6.data() + (size_t)s * stride6 * 2u;
			uint8_t *p1 = hrecv1.data() + (size_t)s * stride1 * 2u;
			for ( uint32_t r = 0u; r < rows6; ++r )
			{
				for ( uint32_t k = 0u; k < w; ++k ) p6[r * w + k] = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
				for ( uint32_t k = 0u; k < sb; ++k ) p6[rows6 * w + r * sb + k] = (uint8_t)(124u + (TestNext(rng) & 3u));
			}
			memcpy(p1, p6, w);
			memcpy(p1 + w, p6 + rows6 * w, sb);
		}
		uint16_t *drecv6 = 0, *drecv1 = 0, *dout6 = 0, *dout1 = 0;
		TEST_CHECK(cudaMalloc(&drecv6, hrecv6.size())); TEST_CHECK(cudaMalloc(&drecv1, hrecv1.size()));
		TEST_CHECK(cudaMalloc(&dout6, (size_t)4u * rows6 * out_dim * 2u)); TEST_CHECK(cudaMalloc(&dout1, (size_t)4u * out_dim * 2u));
		TEST_CHECK(cudaMemcpy(drecv6, hrecv6.data(), hrecv6.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(drecv1, hrecv1.data(), hrecv1.size(), cudaMemcpyHostToDevice));
		D41SharedPartialsDesc desc6, desc1;
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			desc6.w[s] = desc1.w[s] = dw[s]; desc6.ws[s] = desc1.ws[s] = dws[s];
			desc6.width[s] = desc1.width[s] = w;
			desc6.out[s] = dout6 + (uint64_t)s * rows6 * out_dim;
			desc1.out[s] = dout1 + (uint64_t)s * out_dim;
		}
		D41SharedPartialsFusedKernel<8u, 1280u, 640u><<<4u * (out_dim / 16u), 256u>>>(desc6, drecv6, stride6, rows6);
		D41SharedPartialsFusedKernel<1u, 1280u, 640u><<<4u * (out_dim / 16u), 256u>>>(desc1, drecv1, stride1, 1u);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> o6((size_t)4u * rows6 * out_dim), o1((size_t)4u * out_dim);
		TEST_CHECK(cudaMemcpy(o6.data(), dout6, o6.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(o1.data(), dout1, o1.size() * 2u, cudaMemcpyDeviceToHost));
		float max_rel = 0.0f;
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			for ( uint32_t o = 0u; o < out_dim; ++o )
				if ( o1[s * out_dim + o] != o6[(s * rows6 + 0u) * out_dim + o] )
				{ fprintf(stderr, "MISMATCH D41SharedPartialsFusedKernel row-invariant s=%u o=%u\n", s, o); exit(1); }
			const uint8_t *p6 = hrecv6.data() + (size_t)s * stride6 * 2u;
			for ( uint32_t r = 0u; r < rows6; ++r )
				for ( uint32_t o = 0u; o < 32u; ++o )
				{
					double acc = 0.0;
					for ( uint32_t k = 0u; k < w; ++k )
					{
						const float av = TestE4m3ToFloatHost(p6[r * w + k]) * TestUe8m0ToFloatHost(p6[rows6 * w + r * sb + (k >> 5u)]);
						uint32_t abits; memcpy(&abits, &av, 4u); abits += 0x7fffu + ((abits >> 16u) & 1u);
						const float abf = TestBf16ToFloat((uint16_t)(abits >> 16u));
						const float wv = TestE4m3ToFloatHost(hw[s][o * w + k]) * TestUe8m0ToFloatHost(hws[s][o * sb + (k >> 5u)]);
						acc += (double)abf * (double)wv;
					}
					const float got = TestBf16ToFloat(o6[(s * rows6 + r) * out_dim + o]);
					max_rel = fmaxf(max_rel, fabsf(got - (float)acc) / fmaxf(1.0f, fabsf((float)acc)));
				}
		}
		if ( max_rel > 0.01f ) { fprintf(stderr, "MISMATCH D41SharedPartialsFusedKernel max_rel=%.6g\n", max_rel); exit(1); }
		for ( uint32_t s = 0u; s < 4u; ++s ) { cudaFree(dw[s]); cudaFree(dws[s]); }
		cudaFree(drecv6); cudaFree(drecv1); cudaFree(dout6); cudaFree(dout1);
		printf("PASS D41SharedPartialsFusedKernel (rows=1/6, 0 bit diffs row-invariant, max_rel=%.6g)\n", max_rel);
	}

	{
		const uint32_t experts = 384u, topk = 6u, hidden = 5120u, rows = 8u, routes = rows * topk;
		std::vector<uint16_t> hw((size_t)experts * hidden), hx((size_t)rows * hidden);
		std::vector<float> hbias(experts);
		for ( auto &v : hw ) v = TestRandBf16(rng, 0.25f);
		for ( auto &v : hx ) v = TestRandBf16(rng, 0.25f);
		for ( auto &v : hbias ) v = (((float)(TestNext(rng) & 0xffu) / 128.0f) - 1.0f) * 0.2f;
		uint16_t *dw = 0, *dx = 0;
		float *dbias = 0, *dlog_m = 0, *drw_m = 0, *dpw_m = 0;
		uint32_t *dfault = 0, *dre_m = 0, *doff_m = 0, *dpr_m = 0, *dsrc_m = 0;
		uint32_t *dact_m = 0, *dact_c = 0, *dtu = 0, *dtd = 0, *dcnt = 0;
		TEST_CHECK(cudaMalloc(&dw, hw.size() * 2u)); TEST_CHECK(cudaMalloc(&dx, hx.size() * 2u));
		TEST_CHECK(cudaMalloc(&dbias, experts * 4u)); TEST_CHECK(cudaMalloc(&dlog_m, (size_t)rows * experts * 4u));
		TEST_CHECK(cudaMalloc(&drw_m, routes * 4u)); TEST_CHECK(cudaMalloc(&dpw_m, routes * 4u));
		TEST_CHECK(cudaMalloc(&dfault, 4u)); TEST_CHECK(cudaMemset(dfault, 0, 4u));
		TEST_CHECK(cudaMalloc(&dre_m, routes * 4u)); TEST_CHECK(cudaMalloc(&doff_m, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dpr_m, routes * 4u)); TEST_CHECK(cudaMalloc(&dsrc_m, routes * 4u));
		TEST_CHECK(cudaMalloc(&dact_m, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dact_c, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dtu, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dtd, (experts + 1u) * 4u));
		TEST_CHECK(cudaMalloc(&dcnt, 4u)); TEST_CHECK(cudaMemset(dcnt, 0, 4u));
		TEST_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dbias, hbias.data(), experts * 4u, cudaMemcpyHostToDevice));
		D41FusedRouterPackedKernel<384u, 6u, 5120u, 8u><<<experts, 256u>>>(dw, dx, dlog_m, dbias, 2.5f, dfault, dre_m, drw_m,
			doff_m, dpr_m, dsrc_m, 16u, 18u, dtu, 20u, dtd, dpw_m, rows, dcnt, (const uint32_t *)0, dact_m);
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff_m, dact_c);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint32_t> am(experts + 1u), ac(experts + 1u);
		TEST_CHECK(cudaMemcpy(am.data(), dact_m, am.size() * 4u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(ac.data(), dact_c, ac.size() * 4u, cudaMemcpyDeviceToHost));
		if ( am[experts] != ac[experts] || memcmp(am.data(), ac.data(), am[experts] * 4u) != 0 )
		{ fprintf(stderr, "MISMATCH fused active_out act=%u/%u\n", am[experts], ac[experts]); exit(1); }
		cudaFree(dw); cudaFree(dx); cudaFree(dbias); cudaFree(dlog_m); cudaFree(drw_m); cudaFree(dpw_m);
		cudaFree(dfault); cudaFree(dre_m); cudaFree(doff_m); cudaFree(dpr_m); cudaFree(dsrc_m);
		cudaFree(dact_m); cudaFree(dact_c); cudaFree(dtu); cudaFree(dtd); cudaFree(dcnt);
		printf("PASS D41FusedRouterPackedKernel fused active_out (0 diffs)\n");
	}

	{
		const uint32_t experts = 384u, slice = 576u, in13 = 5120u, out13 = 1152u, in2 = 2304u, out2 = 1280u, active_e = 6u;
		const uint64_t g13 = (uint64_t)(in13 / 32u) * (out13 / 16u) * 272u;
		const uint64_t g2 = (uint64_t)(in2 / 32u) * (out2 / 16u) * 272u;
		std::vector<uint8_t> hw13((size_t)experts * g13, 0u), hw2((size_t)experts * g2, 0u);
		for ( uint32_t e = 0u; e < active_e; ++e )
		{
			uint8_t *p13 = hw13.data() + (size_t)e * g13, *p2 = hw2.data() + (size_t)e * g2;
			for ( size_t i = 0; i < g13; ++i ) p13[i] = (uint8_t)(TestNext(rng) & 0xffu);
			for ( uint32_t c = 0u; c < (in13 / 32u) * (out13 / 16u); ++c )
				for ( uint32_t j = 0u; j < 16u; ++j ) p13[(size_t)c * 272u + 256u + j] = (uint8_t)(124u + (TestNext(rng) & 3u));
			for ( size_t i = 0; i < g2; ++i ) p2[i] = (uint8_t)(TestNext(rng) & 0xffu);
			for ( uint32_t c = 0u; c < (in2 / 32u) * (out2 / 16u); ++c )
				for ( uint32_t j = 0u; j < 16u; ++j ) p2[(size_t)c * 272u + 256u + j] = (uint8_t)(124u + (TestNext(rng) & 3u));
		}
		uint8_t *dw13 = 0, *dw2 = 0;
		TEST_CHECK(cudaMalloc(&dw13, hw13.size())); TEST_CHECK(cudaMalloc(&dw2, hw2.size()));
		TEST_CHECK(cudaMemcpy(dw13, hw13.data(), hw13.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dw2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice));
		const uint32_t rows6 = 6u, routes6 = 36u, routes1 = 6u;
		std::vector<uint16_t> hx6((size_t)rows6 * in13);
		for ( auto &v : hx6 ) v = TestRandBf16(rng, 1.0f);
		std::vector<uint32_t> off6(experts + 1u, routes6), off1(experts + 1u, routes1), src6(routes6), src1(routes1);
		for ( uint32_t e = 0u; e < active_e; ++e )
		{
			off6[e] = e * 6u;
			off1[e] = e;
			src1[e] = 0u;
			for ( uint32_t k = 0u; k < 6u; ++k ) src6[e * 6u + k] = k;
		}
		const uint64_t rstride6 = (uint64_t)routes6 * (slice / 2u + slice / 64u) + 32u;
		const uint64_t rstride1 = (uint64_t)routes1 * (slice / 2u + slice / 64u) + 32u;
		std::vector<uint8_t> hrecv6((size_t)4u * rstride6 * 2u, 0u), hrecv1((size_t)4u * rstride1 * 2u, 0u);
		for ( uint32_t s = 0u; s < 4u; ++s )
		{
			uint8_t *p6 = hrecv6.data() + (size_t)s * rstride6 * 2u;
			uint8_t *p1 = hrecv1.data() + (size_t)s * rstride1 * 2u;
			for ( uint32_t r = 0u; r < routes6; ++r )
			{
				for ( uint32_t k = 0u; k < slice; ++k ) p6[r * slice + k] = (uint8_t)((TestNext(rng) & 0x3eu) | 0x20u);
				for ( uint32_t k = 0u; k < slice / 32u; ++k ) p6[routes6 * slice + r * (slice / 32u) + k] = (uint8_t)(124u + (TestNext(rng) & 3u));
			}
			for ( uint32_t e = 0u; e < active_e; ++e )
			{
				memcpy(p1 + e * slice, p6 + (e * 6u) * slice, slice);
				memcpy(p1 + routes1 * slice + e * (slice / 32u), p6 + routes6 * slice + (e * 6u) * (slice / 32u), slice / 32u);
			}
		}
		uint16_t *dx = 0, *dref13 = 0, *dg13_6 = 0, *dg13_1 = 0, *drecv6 = 0, *drecv1 = 0, *dact6 = 0, *dref2 = 0, *dg2_6 = 0, *dg2_1 = 0;
		uint32_t *doff6 = 0, *doff1 = 0, *dsrc6 = 0, *dsrc1 = 0, *dae6 = 0, *dae1 = 0;
		TEST_CHECK(cudaMalloc(&dx, hx6.size() * 2u));
		TEST_CHECK(cudaMalloc(&dref13, (size_t)routes6 * out13 * 2u)); TEST_CHECK(cudaMalloc(&dg13_6, (size_t)routes6 * out13 * 2u));
		TEST_CHECK(cudaMalloc(&dg13_1, (size_t)routes1 * out13 * 2u));
		TEST_CHECK(cudaMalloc(&drecv6, hrecv6.size())); TEST_CHECK(cudaMalloc(&drecv1, hrecv1.size()));
		TEST_CHECK(cudaMalloc(&dact6, (size_t)routes6 * in2 * 2u));
		TEST_CHECK(cudaMalloc(&dref2, (size_t)routes6 * out2 * 2u)); TEST_CHECK(cudaMalloc(&dg2_6, (size_t)routes6 * out2 * 2u));
		TEST_CHECK(cudaMalloc(&dg2_1, (size_t)routes1 * out2 * 2u));
		TEST_CHECK(cudaMalloc(&doff6, off6.size() * 4u)); TEST_CHECK(cudaMalloc(&doff1, off1.size() * 4u));
		TEST_CHECK(cudaMalloc(&dsrc6, src6.size() * 4u)); TEST_CHECK(cudaMalloc(&dsrc1, src1.size() * 4u));
		TEST_CHECK(cudaMalloc(&dae6, (experts + 1u) * 4u)); TEST_CHECK(cudaMalloc(&dae1, (experts + 1u) * 4u));
		TEST_CHECK(cudaMemcpy(dx, hx6.data(), hx6.size() * 2u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(doff6, off6.data(), off6.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(doff1, off1.data(), off1.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsrc6, src6.data(), src6.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(dsrc1, src1.data(), src1.size() * 4u, cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(drecv6, hrecv6.data(), hrecv6.size(), cudaMemcpyHostToDevice));
		TEST_CHECK(cudaMemcpy(drecv1, hrecv1.data(), hrecv1.size(), cudaMemcpyHostToDevice));
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff6, dae6);
		D41CompactActiveExpertsKernel<384u><<<1u, 256u>>>(doff1, dae1);
		if ( LmSkinnyCellExperts(dw13, dx, dref13, doff6, dsrc6, experts, routes6, rows6, 0u, in13, out13, LM_SKINNY_CELL_TILE_K, 0) != LM_LAUNCH_OK ) exit(1);
		D41CellFp4GemvKernel<384u, 5120u, 1152u, 0u, 576u, 8u, 3u><<<routes6 * (out13 / 128u), 256u>>>(dw13, dx, (const uint16_t *)0, dg13_6, doff6, dsrc6, dae6, routes6, 0ull);
		D41CellFp4GemvKernel<384u, 5120u, 1152u, 0u, 576u, 4u, 3u><<<routes1 * (out13 / 64u), 128u>>>(dw13, dx, (const uint16_t *)0, dg13_1, doff1, dsrc1, dae1, routes1, 0ull);
		D41ActAssembleQdqKernel<<<D41ItemGrid(in2 / 256u, routes6), 256u>>>(drecv6, dact6, slice, in2, rstride6, routes6, 1u);
		if ( LmSkinnyCellExperts(dw2, dact6, dref2, doff6, (const uint32_t *)0, experts, routes6, rows6, 1u, in2, out2, LM_SKINNY_CELL_TILE_K, 0) != LM_LAUNCH_OK ) exit(1);
		D41CellFp4GemvKernel<384u, 2304u, 1280u, 1u, 576u, 8u, 3u><<<routes6 * (out2 / 128u), 256u>>>(dw2, (const uint16_t *)0, drecv6, dg2_6, doff6, (const uint32_t *)0, dae6, routes6, rstride6);
		D41CellFp4GemvKernel<384u, 2304u, 1280u, 1u, 576u, 4u, 3u><<<routes1 * (out2 / 64u), 128u>>>(dw2, (const uint16_t *)0, drecv1, dg2_1, doff1, (const uint32_t *)0, dae1, routes1, rstride1);
		TEST_CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> r13((size_t)routes6 * out13), g13_6((size_t)routes6 * out13), g13_1((size_t)routes1 * out13);
		std::vector<uint16_t> r2((size_t)routes6 * out2), g2_6((size_t)routes6 * out2), g2_1((size_t)routes1 * out2);
		TEST_CHECK(cudaMemcpy(r13.data(), dref13, r13.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(g13_6.data(), dg13_6, g13_6.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(g13_1.data(), dg13_1, g13_1.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(r2.data(), dref2, r2.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(g2_6.data(), dg2_6, g2_6.size() * 2u, cudaMemcpyDeviceToHost));
		TEST_CHECK(cudaMemcpy(g2_1.data(), dg2_1, g2_1.size() * 2u, cudaMemcpyDeviceToHost));
		for ( uint32_t e = 0u; e < active_e; ++e )
		{
			if ( memcmp(g13_1.data() + e * out13, g13_6.data() + (e * 6u) * out13, out13 * 2u) != 0 )
			{ fprintf(stderr, "MISMATCH D41CellFp4GemvKernel W13 row-invariant e=%u\n", e); exit(1); }
			if ( memcmp(g2_1.data() + e * out2, g2_6.data() + (e * 6u) * out2, out2 * 2u) != 0 )
			{ fprintf(stderr, "MISMATCH D41CellFp4GemvKernel W2 row-invariant e=%u\n", e); exit(1); }
		}
		float mr13 = 0.0f, mr2 = 0.0f;
		for ( size_t i = 0; i < r13.size(); ++i )
		{
			const float a = TestBf16ToFloat(r13[i]), b = TestBf16ToFloat(g13_6[i]);
			mr13 = fmaxf(mr13, fabsf(a - b) / fmaxf(1.0f, fabsf(a)));
		}
		for ( size_t i = 0; i < r2.size(); ++i )
		{
			const float a = TestBf16ToFloat(r2[i]), b = TestBf16ToFloat(g2_6[i]);
			mr2 = fmaxf(mr2, fabsf(a - b) / fmaxf(1.0f, fabsf(a)));
		}
		if ( mr13 > 0.01f || mr2 > 0.01f ) { fprintf(stderr, "MISMATCH D41CellFp4GemvKernel mr13=%.6g mr2=%.6g\n", mr13, mr2); exit(1); }
		cudaFree(dw13); cudaFree(dw2); cudaFree(dx); cudaFree(dref13); cudaFree(dg13_6); cudaFree(dg13_1);
		cudaFree(drecv6); cudaFree(drecv1); cudaFree(dact6); cudaFree(dref2); cudaFree(dg2_6); cudaFree(dg2_1);
		cudaFree(doff6); cudaFree(doff1); cudaFree(dsrc6); cudaFree(dsrc1); cudaFree(dae6); cudaFree(dae1);
		printf("PASS D41CellFp4GemvKernel (W13 mr=%.6g, W2 mr=%.6g, 4-cell vs 8-cell 0 bit diffs)\n", mr13, mr2);
	}

	{
		using KvGeo = LmKvGeometry<1280u, 256u, true>;
		const uint32_t dim = 512u, heads = 16u, group_heads = 8u, window = 128u, topk = 64u, seq_keys = 160u;
		const float qk_scale = 1.0f / sqrtf((float)dim);
		float *dfreq = 0;
		TEST_CHECK(cudaMalloc(&dfreq, 2u * D41K_ROPE_PAIRS * sizeof(float)));
		TEST_CHECK((cudaError_t)D41LaunchFrequencies(dfreq, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0));
		const uint32_t max_rows = 16u, page_bytes = 256u * 1280u;
		uint8_t *dkv_pool = 0;
		uint32_t *dpt = 0;
		TEST_CHECK(cudaMalloc(&dkv_pool, (size_t)max_rows * page_bytes));
		std::vector<uint16_t> hkv_pool((size_t)max_rows * page_bytes / 2u);
		for ( auto &v : hkv_pool ) v = TestRandBf16(rng, 0.5f);
		TEST_CHECK(cudaMemcpy(dkv_pool, hkv_pool.data(), hkv_pool.size() * 2u, cudaMemcpyHostToDevice));
		std::vector<uint32_t> hpt(max_rows);
		for ( uint32_t s = 0u; s < max_rows; ++s ) hpt[s] = s;
		TEST_CHECK(cudaMalloc(&dpt, max_rows * 4u));
		TEST_CHECK(cudaMemcpy(dpt, hpt.data(), max_rows * 4u, cudaMemcpyHostToDevice));
		LmKvShardView view;
		memset(&view, 0, sizeof(view));
		view.pages.pool = dkv_pool;
		view.pages.page_table = dpt;
		view.pages.page_table_stride = 1u;
		view.pages.sequence_count = max_rows;
		view.pages.pool_page_count = max_rows;
		view.shard.degree = 1u;
		view.shard.rank = 0u;
		view.shard.grain = 1u;
		std::vector<float> hsink(heads);
		for ( auto &v : hsink ) v = (((float)(TestNext(rng) & 0xffu) / 64.0f) - 2.0f);
		float *dsink = 0;
		TEST_CHECK(cudaMalloc(&dsink, heads * 4u));
		TEST_CHECK(cudaMemcpy(dsink, hsink.data(), heads * 4u, cudaMemcpyHostToDevice));
		std::vector<uint16_t> hq16((size_t)max_rows * heads * dim), hwkv16((size_t)max_rows * dim);
		const uint64_t rstride = (uint64_t)window * dim;
		std::vector<uint16_t> hring16((size_t)max_rows * rstride);
		for ( auto &v : hq16 ) v = TestRandBf16(rng, 0.5f);
		for ( auto &v : hwkv16 ) v = TestRandBf16(rng, 0.5f);
		for ( auto &v : hring16 ) v = TestRandBf16(rng, 0.5f);
		std::vector<uint16_t> mode0_row0_r1, mode0_row0_r6;
		for ( uint32_t rows : { 1u, 6u, 8u, 16u } )
			for ( uint32_t compressed : { 0u, 1u } )
				for ( uint32_t mode : { 0u, 1u } )
				{
					const uint32_t splits = rows <= 8u ? 8u : 4u;
					std::vector<uint32_t> hpos(rows), hseq(rows), hlen(max_rows, seq_keys), hbound(rows, seq_keys - 1u);
					std::vector<uint32_t> hlane(rows), hord(rows, 0u), hbeg(rows + 1u), hidx(rows), hslot(rows), hsel((size_t)rows * topk);
					for ( uint32_t r = 0u; r < rows; ++r )
					{
						hpos[r] = 30u + (r * 19u) % 180u; hseq[r] = r; hlane[r] = r; hbeg[r] = r; hidx[r] = r; hslot[r] = r;
						for ( uint32_t k = 0u; k < topk; ++k ) hsel[r * topk + k] = (k * 2u + (r & 1u)) % seq_keys;
					}
					hbeg[rows] = rows;
					uint16_t *dq = 0, *dwkv = 0, *dring_ref = 0, *dring_sp = 0, *dref_out = 0, *dgi_sp = 0;
					uint8_t *dc_ref = 0, *dc_sp = 0;
					float *dpart = 0;
					float2 *drsc = 0, *drsc_c = 0;
					uint32_t *dpos = 0, *dseq = 0, *dlen = 0, *dbnd = 0, *dlane = 0, *dord = 0, *dbeg = 0, *didx = 0, *dslot = 0, *dsel = 0, *dcnt = 0;
					const size_t code_bytes = (size_t)2u * rows * group_heads * dim + (size_t)2u * rows * (group_heads * dim / 32u);
					TEST_CHECK(cudaMalloc(&dq, (size_t)rows * heads * dim * 2u));
					TEST_CHECK(cudaMalloc(&dwkv, (size_t)rows * dim * 2u));
					TEST_CHECK(cudaMalloc(&dring_ref, (size_t)rows * rstride * 2u));
					TEST_CHECK(cudaMalloc(&dring_sp, (size_t)rows * rstride * 2u));
					TEST_CHECK(cudaMalloc(&dref_out, (size_t)rows * heads * dim * 2u));
					TEST_CHECK(cudaMalloc(&dgi_sp, (size_t)rows * heads * dim * 2u));
					TEST_CHECK(cudaMalloc(&dc_ref, code_bytes)); TEST_CHECK(cudaMalloc(&dc_sp, code_bytes));
					TEST_CHECK(cudaMalloc(&dpart, (size_t)rows * splits * heads * D41K_RECORD * 4u));
					TEST_CHECK(cudaMalloc(&drsc, rows * D41K_ROPE_PAIRS * sizeof(float2)));
					TEST_CHECK(cudaMalloc(&drsc_c, rows * D41K_ROPE_PAIRS * sizeof(float2)));
					TEST_CHECK(cudaMalloc(&dpos, rows * 4u)); TEST_CHECK(cudaMalloc(&dseq, rows * 4u));
					TEST_CHECK(cudaMalloc(&dlen, max_rows * 4u)); TEST_CHECK(cudaMalloc(&dbnd, rows * 4u));
					TEST_CHECK(cudaMalloc(&dlane, rows * 4u)); TEST_CHECK(cudaMalloc(&dord, rows * 4u));
					TEST_CHECK(cudaMalloc(&dbeg, (rows + 1u) * 4u)); TEST_CHECK(cudaMalloc(&didx, rows * 4u));
					TEST_CHECK(cudaMalloc(&dslot, rows * 4u)); TEST_CHECK(cudaMalloc(&dsel, (size_t)rows * topk * 4u));
					TEST_CHECK(cudaMalloc(&dcnt, rows * 4u)); TEST_CHECK(cudaMemset(dcnt, 0, rows * 4u));
					TEST_CHECK(cudaMemcpy(dq, hq16.data(), (size_t)rows * heads * dim * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dwkv, hwkv16.data(), (size_t)rows * dim * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dring_ref, hring16.data(), (size_t)rows * rstride * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dring_sp, hring16.data(), (size_t)rows * rstride * 2u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dpos, hpos.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dseq, hseq.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dlen, hlen.data(), max_rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dbnd, hbound.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dlane, hlane.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dord, hord.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dbeg, hbeg.data(), (rows + 1u) * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(didx, hidx.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dslot, hslot.data(), rows * 4u, cudaMemcpyHostToDevice));
					TEST_CHECK(cudaMemcpy(dsel, hsel.data(), (size_t)rows * topk * 4u, cudaMemcpyHostToDevice));
					D41RopeTableKernel<<<rows, D41K_ROPE_PAIRS>>>(dpos, dfreq, drsc, drsc_c, rows);
					const float *freq_use = compressed != 0u ? dfreq + D41K_ROPE_PAIRS : dfreq;
					const float2 *rsc_use = compressed != 0u ? drsc_c : drsc;
					if ( D41AttendLaunch<KvGeo, 512u>(view, compressed, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring_ref, rstride, window,
						dpos, dlane, dbeg, didx, dslot, dsink, qk_scale, mode == 0u ? dref_out : (uint16_t *)0, freq_use, dim - 64u,
						group_heads, mode == 1u ? dc_ref : (uint8_t *)0, rows, rows, 0u, 0) != SPARK_GPU_SUCCESS ) exit(1);
					D41WindowRingStoreKernel<<<rows, 256u>>>(dwkv, dring_ref, dpos, dlane, dord, dbeg, dslot, rstride, window, window, 1u, 0u, rows, 1u, (const uint32_t *)0);
					if ( mode == 0u )
					{
						D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS>>>(dref_out, heads, dim, dim - 64u, dpos, 0, freq_use, 1u, (const uint32_t *)0, rsc_use);
						TEST_CHECK((cudaError_t)SparkDsv41FlashLaunchFp8Qdq(0, dref_out, (uint64_t)rows * heads * dim));
						if ( D41AttendSplitDecodeLaunch<KvGeo, 512u, 0u>(view, compressed, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring_sp, rstride, window,
							dpos, dlane, dord, dbeg, didx, dslot, dsink, qk_scale, rsc_use, group_heads, dgi_sp, (uint8_t *)0, dpart, dcnt, rows, splits, 1u, 0) != SPARK_GPU_SUCCESS )
							exit(1);
					}
					else
					{
						if ( D41AttendSplitDecodeLaunch<KvGeo, 512u, 1u>(view, compressed, dq, dseq, dlen, dbnd, dsel, topk, dwkv, dring_sp, rstride, window,
							dpos, dlane, dord, dbeg, didx, dslot, dsink, qk_scale, rsc_use, group_heads, (uint16_t *)0, dc_sp, dpart, dcnt, rows, splits, 1u, 0) != SPARK_GPU_SUCCESS )
							exit(1);
					}
					TEST_CHECK(cudaDeviceSynchronize());
					std::vector<uint16_t> gr1((size_t)rows * rstride), gr2((size_t)rows * rstride);
					TEST_CHECK(cudaMemcpy(gr1.data(), dring_ref, gr1.size() * 2u, cudaMemcpyDeviceToHost));
					TEST_CHECK(cudaMemcpy(gr2.data(), dring_sp, gr2.size() * 2u, cudaMemcpyDeviceToHost));
					if ( gr1 != gr2 ) { fprintf(stderr, "MISMATCH D41AttendSplitDecodeKernel ring store rows=%u c=%u m=%u\n", rows, compressed, mode); exit(1); }
					if ( mode == 0u )
					{
						std::vector<uint16_t> oref((size_t)rows * heads * dim), osp((size_t)rows * heads * dim);
						TEST_CHECK(cudaMemcpy(oref.data(), dref_out, oref.size() * 2u, cudaMemcpyDeviceToHost));
						TEST_CHECK(cudaMemcpy(osp.data(), dgi_sp, osp.size() * 2u, cudaMemcpyDeviceToHost));
						float mad = 0.0f;
						for ( uint32_t r = 0u; r < rows; ++r )
							for ( uint32_t h = 0u; h < heads; ++h )
								for ( uint32_t c = 0u; c < dim; ++c )
								{
									const uint16_t a = oref[(r * heads + h) * dim + c];
									const uint16_t b = osp[((h / group_heads) * rows + r) * group_heads * dim + (h % group_heads) * dim + c];
									mad = fmaxf(mad, fabsf(TestBf16ToFloat(a) - TestBf16ToFloat(b)));
								}
						if ( mad > 0.05f ) { fprintf(stderr, "MISMATCH D41AttendSplitDecodeKernel MODE=0 rows=%u c=%u mad=%.6g\n", rows, compressed, mad); exit(1); }
						if ( compressed == 1u && (rows == 1u || rows == 6u) )
						{
							std::vector<uint16_t> r0(heads * dim);
							for ( uint32_t h = 0u; h < heads; ++h )
								memcpy(r0.data() + h * dim, osp.data() + ((h / group_heads) * rows + 0u) * group_heads * dim + (h % group_heads) * dim, dim * 2u);
							if ( rows == 1u ) mode0_row0_r1 = r0;
							else mode0_row0_r6 = r0;
						}
					}
					else
					{
						std::vector<uint8_t> c1(code_bytes), c2(code_bytes);
						TEST_CHECK(cudaMemcpy(c1.data(), dc_ref, code_bytes, cudaMemcpyDeviceToHost));
						TEST_CHECK(cudaMemcpy(c2.data(), dc_sp, code_bytes, cudaMemcpyDeviceToHost));
						const size_t ncodes = (size_t)2u * rows * group_heads * dim;
						float mad = 0.0f;
						for ( size_t i = 0; i < ncodes; ++i )
						{
							const float a = TestE4m3ToFloatHost(c1[i]) * TestUe8m0ToFloatHost(c1[ncodes + (i >> 5u)]);
							const float b = TestE4m3ToFloatHost(c2[i]) * TestUe8m0ToFloatHost(c2[ncodes + (i >> 5u)]);
							mad = fmaxf(mad, fabsf(a - b));
						}
						if ( mad > 0.05f ) { fprintf(stderr, "MISMATCH D41AttendSplitDecodeKernel MODE=1 rows=%u c=%u mad=%.6g\n", rows, compressed, mad); exit(1); }
					}
					cudaFree(dq); cudaFree(dwkv); cudaFree(dring_ref); cudaFree(dring_sp); cudaFree(dref_out); cudaFree(dgi_sp);
					cudaFree(dc_ref); cudaFree(dc_sp); cudaFree(dpart); cudaFree(drsc); cudaFree(drsc_c);
					cudaFree(dpos); cudaFree(dseq); cudaFree(dlen); cudaFree(dbnd); cudaFree(dlane); cudaFree(dord);
					cudaFree(dbeg); cudaFree(didx); cudaFree(dslot); cudaFree(dsel); cudaFree(dcnt);
				}
		if ( mode0_row0_r1 != mode0_row0_r6 )
		{ fprintf(stderr, "MISMATCH D41AttendSplitDecodeKernel rows=1 vs rows=6 row 0\n"); exit(1); }
		cudaFree(dfreq); cudaFree(dkv_pool); cudaFree(dpt); cudaFree(dsink);
		printf("PASS D41AttendSplitDecodeKernel (rows=1/6/8/16, comp=0/1, MODE=0/1, ring 0 diffs, row1==row6 0 bit diffs)\n");
	}
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
	TestRows16HillclimbParity(&rng);
	TestDirectIndexScoreAndFastTopk(&rng);
	TestE5LeapfrogKernels(&rng);
	TestTp4UnpackAndCompact(&rng);
	TestTp4HcMixK32Parity(&rng);
	TestTp4Batch16AndSkip(&rng);
	TestTp4W2DirectParity(&rng);
	TestTp4BaseRLeapfrogParity(&rng);
	printf("PASS dsv41 prefill opt parity\n");
	return 0;
}
