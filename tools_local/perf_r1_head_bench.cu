#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>

#include "sparkpipe/spark_lm_kernels.cuh"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kVocabFull = 248320u;
constexpr uint32_t kVocabShard = kVocabFull / 4u;
constexpr uint32_t kTrials = 64u;
constexpr uint32_t kWarmup = 8u;
constexpr uint32_t kIters = 40u;

__global__ void FillRandomBf16Kernel(uint16_t *values, uint64_t *state,
                                     uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;
	uint64_t x = state[index % 1024u];
	x ^= x << 13u;
	x ^= x >> 7u;
	x ^= x << 17u;
	state[index % 1024u] = x;
	uint32_t bits = 0x3f800000u | ((uint32_t)(x & 0xffffu) << 7u);
	float unit = __uint_as_float(bits);
	uint16_t bf16 = __bfloat16_as_ushort(__float2bfloat16(unit * 0.002f));
	values[index] = bf16;
}

__global__ void FillSmallBf16Kernel(uint16_t *values, float scale,
                                    uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;
	float unit = (float)((int32_t)(index % 17u) - 8u) * scale;
	values[index] = __bfloat16_as_ushort(__float2bfloat16(unit));
}

__global__ void ScaleRowsKernel(uint16_t *values, uint32_t hot_rows,
                                uint32_t hidden_dimension, float factor)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t total = (uint64_t)hot_rows * hidden_dimension;
	if (index >= total)
		return;
	float value = __bfloat162float(__ushort_as_bfloat16(values[index]));
	values[index] = __bfloat16_as_ushort(__float2bfloat16(value * factor));
}

float ElapsedMs(const cudaEvent_t *start, const cudaEvent_t *stop)
{
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, *start, *stop);
	return ms;
}

int RunGeometry(uint32_t candidate_count)
{
	cudaError_t error;
	uint16_t *head_bf16 = 0;
	uint16_t *hidden_bf16 = 0;
	uint8_t *fp8_payload = 0;
	float *fp8_scale = 0;
	float *fp8_norm = 0;
	void *direct_scratch = 0;
	void *certified_scratch = 0;
	uint32_t *candidate_ids = 0;
	uint32_t *candidate_count_dev = 0;
	uint32_t *direct_ids = 0;
	float *direct_scores = 0;
	float *certified_scores = 0;
	uint64_t *rng_state = 0;
	cudaEvent_t start, stop;
	int failures = 0;
	const uint64_t head_elements = (uint64_t)candidate_count * kHidden;
	const uint64_t head_bytes = head_elements * 2u;
	const uint64_t groups = (uint64_t)candidate_count * (kHidden / 32u);

	error = cudaMalloc(&head_bf16, head_bytes);
	if (error != cudaSuccess) { printf("alloc head failed\n"); return 1; }
	error = cudaMalloc(&hidden_bf16, kHidden * sizeof(uint16_t));
	error = cudaMalloc(&fp8_payload, head_bytes);
	error = cudaMalloc(&fp8_scale, groups * sizeof(float));
	error = cudaMalloc(&fp8_norm, groups * sizeof(float));
	error = cudaMalloc(&direct_scratch,
		((uint64_t)SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT * (sizeof(float) + sizeof(uint32_t))));
	error = cudaMalloc(&certified_scratch,
		((2ull * candidate_count) + kHidden / 32u +
		 2ull * SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT) * sizeof(float));
	error = cudaMalloc(&candidate_ids,
		(uint64_t)candidate_count * sizeof(uint32_t));
	error = cudaMalloc(&candidate_count_dev, sizeof(uint32_t));
	error = cudaMalloc(&direct_ids, 2u * sizeof(uint32_t));
	error = cudaMalloc(&direct_scores, sizeof(float));
	error = cudaMalloc(&certified_scores, sizeof(float));
	error = cudaMalloc(&rng_state, 1024u * sizeof(uint64_t));
	error = cudaEventCreate(&start);
	error = cudaEventCreate(&stop);
	if (error != cudaSuccess) { printf("alloc phase failed\n"); return 1; }

	{
		uint64_t host_state[1024];
		std::mt19937_64 rng(0x50455246ull);
		for (uint32_t i = 0; i < 1024u; i++)
			host_state[i] = rng();
		cudaMemcpy(rng_state, host_state, sizeof(host_state),
			cudaMemcpyHostToDevice);
	}
	FillRandomBf16Kernel<<<(uint32_t)((head_elements + 255u) / 256u), 256u>>>(
		head_bf16, rng_state, (uint32_t)head_elements);
	cudaDeviceSynchronize();

	SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head_bf16, fp8_payload,
		fp8_scale, fp8_norm, candidate_count, kHidden);
	cudaDeviceSynchronize();

	for (uint32_t trial = 0; trial < kTrials; trial++)
	{
		FillRandomBf16Kernel<<<(kHidden + 255u) / 256u, 256u>>>(hidden_bf16,
			rng_state, kHidden);
		if (trial == kTrials - 1u)
		{
			FillSmallBf16Kernel<<<(uint32_t)((head_elements + 255u) / 256u),
				256u>>>(head_bf16, 1.0f / 4096.0f, (uint32_t)head_elements);
			SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head_bf16,
				fp8_payload, fp8_scale, fp8_norm, candidate_count, kHidden);
			FillSmallBf16Kernel<<<(kHidden + 255u) / 256u, 256u>>>(hidden_bf16,
				1.0f, kHidden);
		}
		cudaDeviceSynchronize();

		error = SparkLmHostLaunchHeadDirectArgmaxWithScore(0, hidden_bf16,
			head_bf16, direct_scratch, candidate_count_dev, direct_ids,
			direct_scores, 0u, 1u, candidate_count, kHidden);
		if (error != cudaSuccess) { printf("direct launch failed\n"); return 1; }
		error = SparkLmHostLaunchHeadCertifiedFp8B1WithScore(0, hidden_bf16,
			head_bf16, fp8_payload, fp8_scale, fp8_norm, certified_scratch,
			candidate_ids, candidate_count_dev, direct_ids + 1u,
			certified_scores, 0u, 1u, candidate_count, kHidden);
		if (error != cudaSuccess) { printf("certified launch failed\n"); return 1; }
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess) { printf("parity sync failed\n"); return 1; }

		uint32_t id_host[2];
		float score_host[2];
		cudaMemcpy(id_host, direct_ids, sizeof(id_host), cudaMemcpyDeviceToHost);
		cudaMemcpy(&score_host[0], direct_scores, sizeof(float),
			cudaMemcpyDeviceToHost);
		cudaMemcpy(&score_host[1], certified_scores, sizeof(float),
			cudaMemcpyDeviceToHost);
		if (id_host[0] != id_host[1])
		{
			printf("PARITY FAIL geometry=%u trial=%u direct=%u certified=%u\n",
				candidate_count, trial, id_host[0], id_host[1]);
			failures++;
		}
		else if (memcmp(&score_host[0], &score_host[1], sizeof(float)) != 0)
		{
			printf("SCORE MISMATCH geometry=%u trial=%u id=%u %.9g vs %.9g\n",
				candidate_count, trial, id_host[0], score_host[0],
				score_host[1]);
			failures++;
		}
		if (trial == kTrials - 1u)
		{
			FillRandomBf16Kernel<<<(uint32_t)((head_elements + 255u) / 256u),
				256u>>>(head_bf16, rng_state, (uint32_t)head_elements);
			SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head_bf16,
				fp8_payload, fp8_scale, fp8_norm, candidate_count, kHidden);
			cudaDeviceSynchronize();
		}
	}

	ScaleRowsKernel<<<(uint32_t)((32ull * kHidden + 255u) / 256u), 256u>>>(
		head_bf16, 32u, kHidden, 64.0f);
	SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head_bf16, fp8_payload,
		fp8_scale, fp8_norm, candidate_count, kHidden);
	cudaDeviceSynchronize();

	float direct_ms[kIters];
	float certified_ms[kIters];
	for (uint32_t i = 0; i < kWarmup; i++)
	{
		SparkLmHostLaunchHeadDirectArgmaxWithScore(0, hidden_bf16, head_bf16,
			direct_scratch, candidate_count_dev, direct_ids, direct_scores,
			0u, 1u, candidate_count, kHidden);
		SparkLmHostLaunchHeadCertifiedFp8B1WithScore(0, hidden_bf16,
			head_bf16, fp8_payload, fp8_scale, fp8_norm, certified_scratch,
			candidate_ids, candidate_count_dev, direct_ids + 1u,
			certified_scores, 0u, 1u, candidate_count, kHidden);
	}
	cudaDeviceSynchronize();
	for (uint32_t i = 0; i < kIters; i++)
	{
		cudaEventRecord(start);
		SparkLmHostLaunchHeadDirectArgmaxWithScore(0, hidden_bf16, head_bf16,
			direct_scratch, candidate_count_dev, direct_ids, direct_scores,
			0u, 1u, candidate_count, kHidden);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		direct_ms[i] = ElapsedMs(&start, &stop);
	}
	for (uint32_t i = 0; i < kIters; i++)
	{
		cudaEventRecord(start);
		SparkLmHostLaunchHeadCertifiedFp8B1WithScore(0, hidden_bf16,
			head_bf16, fp8_payload, fp8_scale, fp8_norm, certified_scratch,
			candidate_ids, candidate_count_dev, direct_ids + 1u,
			certified_scores, 0u, 1u, candidate_count, kHidden);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		certified_ms[i] = ElapsedMs(&start, &stop);
	}
	float direct_mean = 0.0f, certified_mean = 0.0f;
	float direct_min = 1e30f, certified_min = 1e30f;
	for (uint32_t i = 0; i < kIters; i++)
	{
		direct_mean += direct_ms[i];
		certified_mean += certified_ms[i];
		direct_min = direct_ms[i] < direct_min ? direct_ms[i] : direct_min;
		certified_min = certified_ms[i] < certified_min ? certified_ms[i] : certified_min;
	}
	direct_mean /= kIters;
	certified_mean /= kIters;
	double gb_direct = (double)head_bytes / 1e9;
	printf("RESULT candidates=%u hidden=%u\n", candidate_count, kHidden);
	printf("RESULT direct    mean=%.3f ms min=%.3f ms (%.2f GB/token bf16)\n",
		direct_mean, direct_min, gb_direct);
	printf("RESULT certified mean=%.3f ms min=%.3f ms speedup=%.2fx\n",
		certified_mean, certified_min, direct_mean / certified_mean);
	printf("PARITY %s (trials=%u)\n", failures == 0 ? "PASS" : "FAIL", kTrials);
	printf("SCREENED candidates last trial: ");
	{
		uint32_t screened = 0;
		cudaMemcpy(&screened, candidate_count_dev, sizeof(screened),
			cudaMemcpyDeviceToHost);
		printf("%u of %u (%.2f%%)\n", screened, candidate_count,
			100.0 * screened / candidate_count);
	}

	cudaFree(head_bf16); cudaFree(hidden_bf16); cudaFree(fp8_payload);
	cudaFree(fp8_scale); cudaFree(fp8_norm); cudaFree(direct_scratch);
	cudaFree(certified_scratch); cudaFree(candidate_ids);
	cudaFree(candidate_count_dev); cudaFree(direct_ids);
	cudaFree(direct_scores); cudaFree(certified_scores);
	cudaFree(rng_state);
	return failures;
}

}

int main()
{
	printf("=== perf_r1_head_bench (R1 receipt) ===\n");
	int failures = RunGeometry(kVocabShard);
	failures += RunGeometry(kVocabFull);
	printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
	return failures == 0 ? 0 : 1;
}
