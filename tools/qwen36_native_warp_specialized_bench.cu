// WARP-SPECIALIZED SKETCH - DEADLOCKS (2026-08-22 session end; do not run
// as-is). The DESIGN is the escape from the shared-budget tension the
// measured map hit: warps 8-15 continuously cp.async the B ring (the
// bandwidth path never blocks on compute), warps 0-7 mma; A quantization
// stays collective per chunk (reusing the library's StageMxf8 bit-exactly)
// via a third named barrier. The named-barrier choreography (bar 1 =
// B-ready arrive/sync, bar 2 = B-consumed arrive/sync, bar 3 = collective
// A) has an arrival-count mismatch - arrivals must match EXACTLY per
// phase; re-derive the per-chunk arrival matrix (prologue vs steady state
// vs tail) before rerunning. Fix that and this is the remaining FFN kernel
// path (prize -79ms/round at DRAM saturation).
// Build: same includes as tools/qwen36_native_staged_bench.cu.
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
/* THE COMPLETE MEASURED MAP (2026-08-22, M=8 K=5120 N=17408, all bit-exact
 * vs the library kernel):
 *   direct byte-load (production)          125.6 GB/s
 *   staged ring D=2 + A double-buffered    127.6 GB/s  <- best, marginal
 *   staged D=4                             112    (72KB shared -> 1 CTA/SM)
 *   staged CHUNK_K=256                     105    (same occupancy loss)
 *   K-split (grid.z)                       113-116
 *   4-byte loads (direct)                   83-92
 *   pure coalesced reads                   266-276 GB/s
 * THE LAW AND THE TENSION: effective bandwidth ~= (in-flight/latency) x
 * CTAs, and the 101376B/SM shared budget caps in-flight x occupancy - every
 * extra ring stage steals a resident CTA. D=2 x 2 CTAs (72KB in flight/SM)
 * is the budget's sweet spot and it lands at ~127: past this, the design
 * must ESCAPE the shared budget - warp-specialized producer/consumer stages
 * with minimal per-stage shared, or TMA bulk copies. That is the remaining
 * FFN kernel project (prize: -79ms/round, 149->~70ms FFN). */

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

/* Warp-specialized: warps 8-15 PRODUCE (cp.async the B ring continuously),
 * warps 0-7 CONSUME (mma the 128-neuron tile). A quantization stays
 * COLLECTIVE (all 16 warps, one rendezvous per chunk) so the library's
 * StageMxf8 is reused bit-exactly. Named barriers: 1 = B-stage-ready
 * (producer arrive / consumer sync), 2 = B-stage-consumed (consumer arrive
 * / producer sync), 3 = the collective A rendezvous (all threads). */
template <uint32_t D>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS, 1)
void SparkQwen36WarpSpecializedKernel(
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
	extern __shared__ uint8_t staged_b[];            /* [D][128][144] */
	__shared__ uint8_t a_e4m3[2u * 16u * ST_CHUNK_K];
	__shared__ uint8_t a_scale[2u * 16u * (ST_CHUNK_K / 32u)];
	const uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	const uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	const uint32_t chunks = input_dimension / ST_CHUNK_K;
	const uint32_t tile_n_base = blockIdx.y * ST_TILE_N;
	const uint32_t row_base = blockIdx.x * 16u;
	uint32_t chunk, step;

	/* prologue (all threads): A[0] into slot 0, B[0..D-1] staged by the
	 * producer subset below, one rendezvous to line up */
	if ( warp >= 8u )
	{
		const uint32_t ptid = (warp - 8u) * SPARK_LM_WARP_LANES + lane;
		const uint32_t pthreads = 8u * SPARK_LM_WARP_LANES;
		for ( chunk = 0u; chunk < D && chunk < chunks; ++chunk )
		{
			for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
			{
				uint32_t neuron = t / (ST_CHUNK_K / 16u);
				uint32_t k16 = t % (ST_CHUNK_K / 16u);
				StCpAsync16(staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE +
					neuron * ST_B_STRIDE + k16 * 16u,
					weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
					(uint64_t)chunk * ST_CHUNK_K + k16 * 16u);
			}
			StCpCommit();
		}
	}
	if ( warp == 0u || warp == 8u )
	{
		/* rows/warps both start at 0; every thread runs the A stage
		 * collectively right after, so all threads execute this prologue */
	}
	{
		for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
				row_count, row_base, row_count, step * 32u,
				a_e4m3 + step * 16u * 32u, a_scale + step * 16u);
	}
	if ( warp >= 8u )
		asm volatile("bar.arrive 1, 512;");
	else
		asm volatile("bar.sync 1, 512;");

	if ( warp >= 8u )
	{
		/* producers: continuous B stream; rendezvous per chunk for A */
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			/* chunk's B copy complete -> signal ready */
			StCpWait<D - 1u>();
			asm volatile("bar.arrive 1, 512;");
			/* A rendezvous: stage chunk+1's A with everyone */
			if ( chunk + 1u < chunks )
			{
				asm volatile("bar.sync 3, 512;");
				for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
					SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
						row_count, row_base, row_count, (chunk + 1u) * ST_CHUNK_K + step * 32u,
						a_e4m3 + ((chunk + 1u) & 1u) * 16u * ST_CHUNK_K + step * 16u * 32u,
						a_scale + ((chunk + 1u) & 1u) * 16u * (ST_CHUNK_K / 32u) + step * 16u);
				asm volatile("bar.sync 3, 512;");
			}
			/* restage slot (chunk+D)%D after the consumer frees it */
			if ( chunk + D < chunks )
			{
				asm volatile("bar.sync 2, 512;");
				{
					const uint32_t ptid = (warp - 8u) * SPARK_LM_WARP_LANES + lane;
					const uint32_t pthreads = 8u * SPARK_LM_WARP_LANES;
					uint32_t rchunk = chunk + D;
					for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
					{
						uint32_t neuron = t / (ST_CHUNK_K / 16u);
						uint32_t k16 = t % (ST_CHUNK_K / 16u);
						StCpAsync16(staged_b + (rchunk % D) * ST_TILE_N * ST_B_STRIDE +
							neuron * ST_B_STRIDE + k16 * 16u,
							weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
							(uint64_t)rchunk * ST_CHUNK_K + k16 * 16u);
					}
					StCpCommit();
				}
			}
		}
		return;
	}

	/* consumers */
	{
		float total[2][4] = {};
		const uint32_t neuron_base = tile_n_base + warp * (2u * 8u);
		uint32_t ni, entry;
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			uint8_t *cur = staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE;
			if ( chunk > 0u )
				asm volatile("bar.sync 1, 512;");
			/* A rendezvous: stage chunk+1's A with everyone */
			if ( chunk + 1u < chunks )
			{
				asm volatile("bar.sync 3, 512;");
				for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
					SparkLmSm121StageMxf8<16u>(input_bf16, input_row_stride, 0u, 0u,
						row_count, row_base, row_count, (chunk + 1u) * ST_CHUNK_K + step * 32u,
						a_e4m3 + ((chunk + 1u) & 1u) * 16u * ST_CHUNK_K + step * 16u * 32u,
						a_scale + ((chunk + 1u) & 1u) * 16u * (ST_CHUNK_K / 32u) + step * 16u);
				asm volatile("bar.sync 3, 512;");
			}
			for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			{
				uint32_t k_base = chunk * ST_CHUNK_K + step * 32u;
				uint32_t a[4], scale_a, scale_b, b[2], reg;
				SparkLmSm121LoadMxf8A(a_e4m3 + (chunk & 1u) * 16u * ST_CHUNK_K + step * 16u * 32u, 0u, lane, a);
				scale_a = SparkLmSm121ScaleA(a_scale + (chunk & 1u) * 16u * (ST_CHUNK_K / 32u) + step * 16u, 0u, lane);
				#pragma unroll
				for ( ni = 0u; ni < 2u; ++ni )
				{
					uint32_t fragment_neuron = neuron_base + ni * 8u;
					scale_b = SparkLmSm121ScaleB<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
						weight_scale_e8m0, fragment_neuron, input_dimension, k_base, lane);
					{
						uint32_t tile_neuron = fragment_neuron - tile_n_base + LmMma8OperandBRow(lane);
						const uint8_t *brow = cur + tile_neuron * ST_B_STRIDE + step * 32u;
						#pragma unroll
						for ( reg = 0u; reg < 2u; ++reg )
							b[reg] = *(const uint32_t *)(brow + LmMma8OperandBByte(lane, reg));
					}
					SparkLmSm121Mma<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(total[ni], a, b, scale_a, scale_b);
				}
			}
			asm volatile("bar.arrive 2, 512;");
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
		error = cudaFuncSetAttribute(SparkQwen36WarpSpecializedKernel<D>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared); \
		if ( error != cudaSuccess ) return(error); \
		SparkQwen36WarpSpecializedKernel<D><<<grid, SPARK_LM_CTA_THREADS, shared, stream>>>( \
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
	/* correctness vs the library kernel at every depth */
	CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
		stream, payload, scale, payload_bytes, scale_bytes, in, K, 0u, 0u,
		ref, N, 0u, 0u, 1u, M, K, N));
	for ( uint32_t depth = 2u; depth <= 4u; depth += 2u )  /* K must be % CHUNK_K; depth 6+ exceeds the 101376B shared opt-in cap */
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
