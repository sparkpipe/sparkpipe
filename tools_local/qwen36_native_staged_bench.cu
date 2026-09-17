#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

#define ST_TILE_N 128u
#define ST_CHUNK_K 128u
#define ST_B_STRIDE (ST_CHUNK_K + 16u)

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

template <uint32_t D>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS, 1)
void SparkQwen38_27bStagedLinearKernel(
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
	extern __shared__ uint8_t staged_b[];
	__shared__ uint8_t activation_e4m3[2u * 16u * ST_CHUNK_K];
	__shared__ uint8_t activation_scale_e8m0[2u * 16u * (ST_CHUNK_K / 32u)];
	float total[2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t row_base = blockIdx.x * 16u;
	uint32_t neuron_base = blockIdx.y * ST_TILE_N + warp * (2u * 8u);
	uint32_t chunks = input_dimension / ST_CHUNK_K;
	uint32_t chunk, step, ni, entry, t;

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

	{
		uint32_t pc;
		for ( pc = 0u; pc + 1u < D && pc < chunks; ++pc )
			ST_STAGE(staged_b + (pc % D) * ST_TILE_N * ST_B_STRIDE, pc);
		for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
				row_count, row_base, row_count, step * 32u,
				activation_e4m3 + step * 16u * 32u,
				activation_scale_e8m0 + step * 16u);
		__syncthreads();
	}
	for ( chunk = 0u; chunk < chunks; ++chunk )
	{
		uint8_t *cur = staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE;
		if ( chunk + D - 1u < chunks )
		{
			ST_STAGE(staged_b + ((chunk + D - 1u) % D) * ST_TILE_N * ST_B_STRIDE,
				chunk + D - 1u);
			StCpWait<D - 1u>();
		}
		else
			StCpWait<0>();
		__syncthreads();
		{
			uint32_t aslot = ((chunk + 1u) & 1u) * 16u * ST_CHUNK_K;
			uint32_t sscale_slot = ((chunk + 1u) & 1u) * 16u * (ST_CHUNK_K / 32u);
			uint32_t achunk = chunk + 1u;
			if ( achunk < chunks )
				for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
					SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
						row_count, row_base, row_count, achunk * ST_CHUNK_K + step * 32u,
						activation_e4m3 + aslot + step * 16u * 32u,
						activation_scale_e8m0 + sscale_slot + step * 16u);
		}
		for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
		{
			uint32_t k_base = chunk * ST_CHUNK_K + step * 32u;
			uint32_t a[4], scale_a, scale_b;
			SparkLmSm121LoadMxf8A(activation_e4m3 + (chunk & 1u) * 16u * ST_CHUNK_K + step * 16u * 32u, 0u, lane, a);
			scale_a = SparkLmSm121ScaleA(activation_scale_e8m0 + (chunk & 1u) * 16u * (ST_CHUNK_K / 32u) + step * 16u, 0u, lane);
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
	void *out, uint64_t out_stride, uint32_t rows, uint32_t K, uint32_t N,
	uint32_t depth)
{
	dim3 grid((rows + 15u) / 16u, N / ST_TILE_N);
	size_t shared = (size_t)depth * ST_TILE_N * ST_B_STRIDE;
	cudaError_t error;
	#define ST_LAUNCH(D) do { \
		error = cudaFuncSetAttribute(SparkQwen38_27bStagedLinearKernel<D>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared); \
		if ( error != cudaSuccess ) return(error); \
		SparkQwen38_27bStagedLinearKernel<D><<<grid, SPARK_LM_CTA_THREADS, shared, stream>>>( \
			(const uint8_t *)payload, scale, in, in_stride, out, out_stride, rows, K, N); \
	} while (0)
	switch ( depth )
	{
	case 2u: ST_LAUNCH(2u); break;
	case 4u: ST_LAUNCH(4u); break;
	case 6u: ST_LAUNCH(6u); break;
	case 8u: ST_LAUNCH(8u); break;
	default: return(cudaErrorInvalidValue);
	}
	#undef ST_LAUNCH
	return(cudaGetLastError());
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
	CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
		stream, payload, scale, payload_bytes, scale_bytes, in, K, 0u, 0u,
		ref, N, 0u, 0u, 1u, M, K, N));
	for ( uint32_t depth = 2u; depth <= 4u; depth += 2u )
	{
		CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, depth));
		CHECK(cudaStreamSynchronize(stream));
		uint16_t *h_ref = (uint16_t *)malloc((size_t)M * N * 2u);
		uint16_t *h_out = (uint16_t *)malloc((size_t)M * N * 2u);
		CHECK(cudaMemcpy(h_ref, ref, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_out, out, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		int exact = 0;
		for (uint32_t i = 0; i < M * N; i++)
			if (h_ref[i] == h_out[i]) exact++;
		free(h_ref); free(h_out);
		printf("depth=%u correctness: exact=%d/%u\n", depth, exact, M * N);
	}
	for (uint32_t depth = 2u; depth <= 4u; depth += 2u)
	{
		for (int i = 0; i < 8; i++)
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, depth));
		CHECK(cudaStreamSynchronize(stream));
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for (int i = 0; i < iters; i++)
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, depth));
		CHECK(cudaStreamSynchronize(stream));
		double dt = (now_s() - t0) / iters;
		printf("staged M=%u K=%u N=%u depth=%u: %.3f ms/iter, effective %.1f GB/s (direct byte-load: 125.6; pure read: 266-276)\n",
			M, K, N, depth, dt * 1e3, bytes / dt / 1e9);
	}
	return 0;
}
