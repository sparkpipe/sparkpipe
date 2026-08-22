// Native SM121 block-scaled fp8 MMA micro-benchmark: effective GB/s at the
// verify-frame shapes, and the reference number for any B-load experiment.
// Build on spark2 from the repo root:
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
//     -I . -I include -I model-families/common/include \
//     -I modules/qwen36_resident_decode_stage/include tools/qwen36_native_linear_bench.cu \
//     -o /tmp/nb_bench -lcudart
// 2026-08-22 baseline: M=8 K=5120 N=17408 -> 0.72ms, ~116-126 GB/s (vs ~245
// GB/s D2D ceiling). Load-width variants measured: byte-by-byte __ldg WINS
// (125.6), __ldg-4B 91.7, plain-4B 83.7 - the byte loads' outstanding
// transactions are the memory-level parallelism this access pattern needs.
// K-split (grid.z partitioning, bit-exact) does NOT help: the fabric, not
// occupancy, caps this pattern. The path past ~125 GB/s is a cp.async
// shared-staged B tile (coalesced wide loads, CUTLASS-style).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static double now_s(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void)
{
	const uint32_t M = 8u, K = 5120u, N = 17408u;
	const int iters = 200;
	uint8_t *payload, *scale;
	__nv_bfloat16 *in, *out;
	const uint64_t payload_bytes = (uint64_t)N * K;
	const uint64_t scale_bytes = (uint64_t)N * (K / 128u);
	CHECK(cudaMalloc(&payload, payload_bytes));
	CHECK(cudaMalloc(&scale, scale_bytes));
	CHECK(cudaMalloc(&in, (uint64_t)M * K * 2u));
	CHECK(cudaMalloc(&out, (uint64_t)M * N * 2u));
	CHECK(cudaMemset(payload, 0x30, payload_bytes));
	CHECK(cudaMemset(scale, 127, scale_bytes));
	CHECK(cudaMemset(in, 0, (uint64_t)M * K * 2u));
	CHECK(cudaMemset(out, 0, (uint64_t)M * N * 2u));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	// warm
	for (int i = 0; i < 8; i++)
		CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
			stream, payload, scale, payload_bytes, scale_bytes,
			in, K, 0u, 0u, out, N, 0u, 0u, 1u, M, K, N));
	CHECK(cudaStreamSynchronize(stream));
	double t0 = now_s();
	for (int i = 0; i < iters; i++)
		CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
			stream, payload, scale, payload_bytes, scale_bytes,
			in, K, 0u, 0u, out, N, 0u, 0u, 1u, M, K, N));
	CHECK(cudaStreamSynchronize(stream));
	double dt = (now_s() - t0) / iters;
	double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
	printf("native M=%u K=%u N=%u: %.3f ms/iter, effective %.1f GB/s\n", M, K, N, dt * 1e3, bytes / dt / 1e9);
	// M=1 for reference
	t0 = now_s();
	for (int i = 0; i < iters; i++)
		CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
			stream, payload, scale, payload_bytes, scale_bytes,
			in, K, 0u, 0u, out, N, 0u, 0u, 1u, 1u, K, N));
	CHECK(cudaStreamSynchronize(stream));
	dt = (now_s() - t0) / iters;
	printf("native M=1: %.3f ms/iter, effective %.1f GB/s\n", dt * 1e3, bytes / dt / 1e9);
	return 0;
}
