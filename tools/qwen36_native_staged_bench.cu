// cp.async shared-staged B-tile variant of the native block-scaled fp8 MMA
// linear - the scoped FFN kernel project (see qwen36_native_linear_bench.cu
// for the baselines: byte-load 125.6 GB/s wins width tweaks; K-split null).
// This harness: correctness vs the library kernel on RANDOM data + GB/s.
//
// Build on spark2 from the repo root:
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
//     -I . -I include -I model-families/common/include \
//     -I modules/qwen36_resident_decode_stage/include \
//     tools/qwen36_native_staged_bench.cu -o /tmp/staged_bench -lcudart
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

#define ST_TILE_N 128u
#define ST_CHUNK_K 128u
#define ST_B_STRIDE (ST_CHUNK_K + 16u)  /* pad to 16B: cp.async alignment AND breaks the 32-word bank period */
/* PIPELINE-DEPTH LAW (measured 2026-08-22): this kernel family's effective
 * bandwidth ~= (in-flight staging bytes per CTA / DRAM latency) x resident
 * CTAs/SM x SMs. The depth-2 ring below holds ~16KB in flight -> ~113 GB/s
 * measured (16KB/600ns x ~4.5 CTAs x 10 SMs ~= 120). Pure coalesced reads
 * reach 266+. DEEPENING the ring (4+ stages, 64KB+ in flight) should
 * saturate DRAM - the sketch had a ring-indexing bug at session end; fix it
 * here and re-measure. Shared for depth D = D x 128 x 144 bytes. */

static __device__ __forceinline__ void StCpAsync16(void *shmem, const void *gmem)
{
	uint32_t s = (uint32_t)__cvta_generic_to_shared(shmem);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}

static __device__ __forceinline__ void StCpCommit(void)
{
	asm volatile("cp.async.commit_group;\n");
}

template <int Groups>
static __device__ __forceinline__ void StCpWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(Groups));
}

static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS, 1)
void SparkQwen36StagedLinearKernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	uint64_t input_row_stride,
	void *output_bf16,
	uint64_t output_row_stride,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	extern __shared__ uint8_t staged_b[];          /* [2][ST_TILE_N][ST_B_STRIDE] double buffer */
	__shared__ uint8_t activation_e4m3[16u * ST_CHUNK_K];   /* 4 steps staged per chunk */
	__shared__ uint8_t activation_scale_e8m0[16u * (ST_CHUNK_K / 32u)];
	float total[2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t row_base = blockIdx.x * 16u;
	uint32_t neuron_base = blockIdx.y * ST_TILE_N + warp * (2u * 8u);
	uint32_t chunks = input_dimension / ST_CHUNK_K;
	uint32_t chunk, step, ni, entry, t;

	/* stage one chunk: 128 neurons x 128B = 1024 x 16B, two per thread,
	 * fully coalesced along K (8 threads span one neuron's 128B line) */
	#define ST_STAGE(buf, chunk_index) do { \
		for ( t = threadIdx.x; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += SPARK_LM_CTA_THREADS ) { \
			uint32_t neuron = t / (ST_CHUNK_K / 16u); \
			uint32_t k16 = t % (ST_CHUNK_K / 16u); \
			StCpAsync16((buf) + neuron * ST_B_STRIDE + k16 * 16u, \
				weight_payload + (uint64_t)(blockIdx.y * ST_TILE_N + neuron) * input_dimension + \
				(uint64_t)(chunk_index) * ST_CHUNK_K + k16 * 16u); \
		} \
		StCpCommit(); \
	} while (0)

	/* VERIFIED depth-2 ring (bit-exact vs the library kernel, ~113 GB/s at
	 * M=8 K=5120 N=17408; the byte-load direct kernel measures 125.6):
	 * stage the next chunk into the OTHER buffer, wait for the current. */
	ST_STAGE(staged_b, 0u);   /* chunk 0 into the low buffer FIRST */
	if ( 1u < chunks ) {}
	for ( chunk = 0u; chunk < chunks; ++chunk )
	{
		uint8_t *cur = (chunk & 1u) != 0u ?
			staged_b + ST_TILE_N * ST_B_STRIDE : staged_b;
		if ( chunk + 1u < chunks )
		{
			ST_STAGE((chunk & 1u) != 0u ? staged_b :
				staged_b + ST_TILE_N * ST_B_STRIDE, chunk + 1u);
			StCpWait<1>();
		}
		else
			StCpWait<0>();
		__syncthreads();
		/* stage ALL the chunk's A blocks (4 steps) under ONE barrier */
		for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
				row_count, row_base, row_count, chunk * ST_CHUNK_K + step * 32u,
				activation_e4m3 + step * 16u * 32u,
				activation_scale_e8m0 + step * 16u);
		__syncthreads();
		for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
		{
			uint32_t k_base = chunk * ST_CHUNK_K + step * 32u;
			uint32_t a[4], scale_a, scale_b;
			SparkLmSm121LoadMxf8A(activation_e4m3 + step * 16u * 32u, 0u, lane, a);
			scale_a = SparkLmSm121ScaleA(activation_scale_e8m0 + step * 16u, 0u, lane);
			#pragma unroll
			for ( ni = 0u; ni < 2u; ++ni )
			{
				uint32_t fragment_neuron = neuron_base + ni * 8u;
				uint32_t b[2], reg;
				scale_b = SparkLmSm121ScaleB<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
					weight_scale_e8m0, fragment_neuron, input_dimension, k_base, lane);
				{
					uint32_t tile_neuron = fragment_neuron - blockIdx.y * ST_TILE_N +
						LmMma8OperandBRow(lane);
					const uint8_t *brow = cur + tile_neuron * ST_B_STRIDE + step * 32u;
					#pragma unroll
					for ( reg = 0u; reg < 2u; ++reg )
						b[reg] = *(const uint32_t *)(brow + LmMma8OperandBByte(lane, reg));
				}
				SparkLmSm121Mma<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(total[ni], a, b, scale_a, scale_b);
			}
		}
		__syncthreads();
	}
	#pragma unroll
	for ( ni = 0u; ni < 2u; ++ni )
		#pragma unroll
		for ( entry = 0u; entry < 4u; ++entry )
		{
			uint32_t row = row_base + LmMmaAccumulatorRow(lane, entry);
			uint32_t column = neuron_base + ni * 8u + LmMmaAccumulatorColumn(lane, entry);
			if ( row < row_count && column < output_dimension )
				SparkLmFloatToBf16(output_bf16,
					(uint64_t)row * output_row_stride + column, total[ni][entry]);
		}
}

static cudaError_t LaunchStaged(cudaStream_t stream,
	const void *payload, const uint8_t *scale, const void *in, uint64_t in_stride,
	void *out, uint64_t out_stride, uint32_t rows, uint32_t K, uint32_t N)
{
	dim3 grid((rows + 15u) / 16u, N / ST_TILE_N);
	size_t shared = 2u * ST_TILE_N * ST_B_STRIDE;
	cudaFuncSetAttribute(SparkQwen36StagedLinearKernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
	SparkQwen36StagedLinearKernel<<<grid, SPARK_LM_CTA_THREADS, shared, stream>>>(
		(const uint8_t *)payload, scale, in, in_stride, out, out_stride, rows, K, N);
	return cudaGetLastError();
}

int main(int argc, char **argv)
{
	const uint32_t M = argc > 1 ? (uint32_t)atoi(argv[1]) : 8u;
	const uint32_t K = argc > 2 ? (uint32_t)atoi(argv[2]) : 5120u;
	const uint32_t N = argc > 3 ? (uint32_t)atoi(argv[3]) : 17408u;
	const int iters = 200;
	uint8_t *payload, *scale;
	__nv_bfloat16 *in, *out, *ref;
	const uint64_t payload_bytes = (uint64_t)N * K;
	const uint64_t scale_bytes = (uint64_t)N * (K / 128u);
	CHECK(cudaMalloc(&payload, payload_bytes));
	CHECK(cudaMalloc(&scale, scale_bytes));
	CHECK(cudaMalloc(&in, (uint64_t)M * K * 2u));
	CHECK(cudaMalloc(&out, (uint64_t)M * N * 2u));
	CHECK(cudaMalloc(&ref, (uint64_t)M * N * 2u));
	/* RANDOM data so the comparison is meaningful */
	{
		uint8_t *h = (uint8_t *)malloc(payload_bytes);
		srand(12345);
		for (uint64_t i = 0; i < payload_bytes; i++) h[i] = (uint8_t)(rand() & 0x7e);
		CHECK(cudaMemcpy(payload, h, payload_bytes, cudaMemcpyHostToDevice));
		free(h);
		h = (uint8_t *)malloc(scale_bytes);
		for (uint64_t i = 0; i < scale_bytes; i++) h[i] = 120 + (uint8_t)(rand() % 12);
		CHECK(cudaMemcpy(scale, h, scale_bytes, cudaMemcpyHostToDevice));
		free(h);
		h = (uint8_t *)malloc((size_t)M * K * 2u);
		{
			/* valid bf16 values in [-2, 2): raw random bits are NaN patterns */
			uint16_t *h16 = (uint16_t *)h;
			for (uint64_t i = 0; i < (size_t)M * K; i++)
			{
				__nv_bfloat16 b = __float2bfloat16((float)(rand() % 2000 - 1000) / 500.0f);
				h16[i] = *(uint16_t *)&b;
			}
		}
		CHECK(cudaMemcpy(in, h, (size_t)M * K * 2u, cudaMemcpyHostToDevice));
		free(h);
	}
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	/* correctness vs the library kernel */
	CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
		stream, payload, scale, payload_bytes, scale_bytes, in, K, 0u, 0u,
		ref, N, 0u, 0u, 1u, M, K, N));
	CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N));
	CHECK(cudaStreamSynchronize(stream));
	{
		uint16_t *h_ref = (uint16_t *)malloc((size_t)M * N * 2u);
		uint16_t *h_out = (uint16_t *)malloc((size_t)M * N * 2u);
		CHECK(cudaMemcpy(h_ref, ref, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_out, out, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		int exact = 0, close = 0;
		double maxrel = 0;
		for (uint32_t i = 0; i < M * N; i++)
		{
			float a = __bfloat162float(*(__nv_bfloat16 *)&h_ref[i]);
			float b = __bfloat162float(*(__nv_bfloat16 *)&h_out[i]);
			if (a == b) exact++;
			else {
				double rel = fabs((double)a - b) / (fabs((double)a) + 1e-9);
				if (rel < maxrel) {} else maxrel = rel;
				if (rel < 0.05) close++;
			}
		}
		printf("correctness: exact=%d/%u close(<5%%)=%d maxrel=%.4f\n", exact, M * N, close, maxrel);
		{
			/* pattern: which columns of row 0 are exact */
			printf("  row0 exact cols:");
			for (uint32_t c = 0; c < (N < 64u ? N : 64u); c++)
				if (h_ref[c] == h_out[c]) printf(" %u", c);
			printf("\n");
			int shown = 0;
			for (uint32_t i = 0; i < M * N && shown < 8; i++)
			{
				float a = __bfloat162float(*(__nv_bfloat16 *)&h_ref[i]);
				float b = __bfloat162float(*(__nv_bfloat16 *)&h_out[i]);
				if (a != b)
				{
					printf("  mismatch row=%u col=%u ref=%f got=%f\n",
						i / N, i % N, a, b);
					shown++;
				}
			}
		}
		free(h_ref); free(h_out);
	}
	for (int i = 0; i < 8; i++)
		CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N));
	CHECK(cudaStreamSynchronize(stream));
	double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
	double t0 = now_s();
	for (int i = 0; i < iters; i++)
		CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N));
	CHECK(cudaStreamSynchronize(stream));
	double dt = (now_s() - t0) / iters;
	printf("staged M=%u K=%u N=%u: %.3f ms/iter, effective %.1f GB/s (baseline byte-load: 0.72ms / 125.6)\n",
		M, K, N, dt * 1e3, bytes / dt / 1e9);
	return 0;
}
