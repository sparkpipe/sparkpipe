/* P3 BATCHED SMALL-ROWS BENCH (2026-08-28) - PERF_PROGRAM item 3.
 * The knee-sweep's small-B "r-law" is a dispatch issue: the shared gate
 * routed every row_count < 16 to the scalar GEMV whose grid streams the
 * weight set once PER ROW, so B2/B4 aggregate stayed in the B1 rate class
 * (the measured 27B FP8 B1==B2==8.31 flat spot). SparkLmBatchedLinearKernel
 * streams the weights ONCE for the row group.
 *
 * This bench times the SHIPPED kernels (not variants) on the 27B decode
 * weight class - K=5120 -> N=17408, the 89 MB FP8 E8M0B128 verify shape -
 * carved into an arena sized like the 29.9 GB family pack, so one "step"
 * streams the working set once and aggregate tok/s = B / step_time maps
 * directly onto decode throughput.
 *
 *   scalar route  : SparkLmLinearKernel, grid.x = rows (the OLD B2..B15
 *                   behavior; identical to the production B1 route at B=1)
 *   batched route : SparkLmBatchedLinearKernel, one weight stream
 *
 * Win condition: batched aggregate B2/B1 > 1.5x (a per-row route cannot
 * exceed 1.0x by construction: B rows cost B streams).
 *
 * MEASURED LEDGER (2026-08-29, spark5, GB10, CUDA 13.0, GPU idle,
 * fleet_status: spark5 free):
 *   route           B  ms/step   GB/s   agg tok/s  vs B1
 *   scalar-per-row  1   130.7   228.4      7.65    1.00x
 *   scalar-per-row  2   128.6   232.1     15.55    2.03x
 *   scalar-per-row  4   155.6   191.8     25.70    3.36x
 *   scalar-per-row  8   271.0   110.2     29.52    3.86x
 *   batched-once    1   130.3   229.2      7.68    1.00x
 *   batched-once    2   169.8   175.8     11.78    1.53x
 *   batched-once    4   202.3   147.6     19.77    2.58x
 *   batched-once    8   271.5   110.0     29.47    3.84x
 *   exactness: batched == per-row scalar, B=4, 69632 neurons bit-exact.
 * CONCLUSION (negative as a route change): the per-row scalar route's
 * concurrent row streams OVERLAP in GB10's memory system (B2 costs ~0ms
 * over B1), so the one-pass batched kernel - paying the shared-staging
 * round trip, the same scalar-beats-tile gap the tile path measured at
 * M=9 - is BEHIND through B4 and equal at B8. The knee-sweep premise
 * "B1==B2==8.31" was a misread of its CSV (8.31 is the B1 row; B2 measured
 * 16.63 = 2.00x, matching the scalar 2.03x here). Verdict: keep the
 * scalar route for rows < 16; SparkLmBatchedLinearKernel stays compiled,
 * oracle-proven, and available for a bandwidth profile where per-row
 * overlap does not absorb the rows.
 *
 * Also spot-checks exactness on the first matrix: batched vs per-row scalar
 * (bit-compare), because the host oracle proves it and this confirms the
 * device build agrees.
 *
 * Build: nvcc -O3 -arch=sm_121a -I<repo> -I<repo>/include \
 *        -I<repo>/model-families/common/include \
 *        -o /tmp/p3bench tools/p3_batched_small_rows_bench.cu
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

#define K_DIM 5120u
#define N_DIM 17408u
#define MAX_ROWS 8u
#define WARMUP_STEPS 1u
#define TIMED_STEPS 3u
/* ~29.9 GB family working set carved into 89 MB matrices. */
#define ARENA_BYTES_TARGET 29900000000ull

static uint32_t bench_seed = 0x1234567u;
static uint32_t bench_next(void) { bench_seed = bench_seed * 1664525u + 1013904223u; return bench_seed; }

int main(void)
{
	uint64_t matrix_bytes = (uint64_t)K_DIM * N_DIM;
	uint64_t arena_bytes = ARENA_BYTES_TARGET - (ARENA_BYTES_TARGET % matrix_bytes);
	uint32_t matrix_count = (uint32_t)(arena_bytes / matrix_bytes);
	uint32_t scale_bytes = N_DIM * (K_DIM / 128u);
	uint8_t *arena = 0,*scales = 0;
	uint16_t *input = 0,*output_scalar = 0,*output_batched = 0;
	uint32_t matrix,step,path,B;
	double per_path[2][9] = {{0}};
	const char *path_name[2] = {"scalar-per-row","batched-once"};

	printf("P3 batched small-rows bench: K=%u N=%u matrix_bytes=%llu matrices=%u arena_bytes=%llu (%.1f GB)\n",
		(unsigned)K_DIM,(unsigned)N_DIM,(unsigned long long)matrix_bytes,
		(unsigned)matrix_count,(unsigned long long)arena_bytes,
		(double)arena_bytes / 1e9);

	CHECK(cudaMalloc((void **)&arena,arena_bytes));
	CHECK(cudaMalloc((void **)&scales,scale_bytes));
	CHECK(cudaMalloc((void **)&input,(uint64_t)MAX_ROWS * K_DIM * sizeof(uint16_t)));
	CHECK(cudaMalloc((void **)&output_scalar,(uint64_t)MAX_ROWS * N_DIM * sizeof(uint16_t)));
	CHECK(cudaMalloc((void **)&output_batched,(uint64_t)MAX_ROWS * N_DIM * sizeof(uint16_t)));

	/* Host-fill the first matrix + input with a pattern, then let the GPU
	 * replicate the first matrix across the arena (one-time cost, untimed). */
	{
		uint8_t *host_matrix = (uint8_t *)malloc(matrix_bytes);
		uint16_t *host_input = (uint16_t *)malloc((uint64_t)MAX_ROWS * K_DIM * sizeof(uint16_t));
		uint8_t *host_scales = (uint8_t *)malloc(scale_bytes);
		uint64_t index;
		uint32_t row;
		if ( !host_matrix || !host_input || !host_scales )
		{
			printf("ERR host fill alloc\n");
			return 1;
		}
		for (index = 0u; index < matrix_bytes; index++)
			host_matrix[index] = (uint8_t)(bench_next() >> 24u);
		for (row = 0u; row < MAX_ROWS * K_DIM; row++)
		{
			uint32_t bits = bench_next();
			host_input[row] = (uint16_t)(((bits & 1u) << 15u) | ((118u + ((bits >> 1u) % 5u)) << 7u) | ((bits >> 3u) & 0x7fu));
		}
		for (index = 0u; index < scale_bytes; index++)
			host_scales[index] = (uint8_t)(120u + (bench_next() % 15u));
		CHECK(cudaMemcpy(arena,host_matrix,matrix_bytes,cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(input,host_input,(uint64_t)MAX_ROWS * K_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(scales,host_scales,scale_bytes,cudaMemcpyHostToDevice));
		/* Replicate matrix 0 across the arena (the kernels only stream the
		 * bytes; every matrix holding the same pattern is fine and the
		 * copy is one-time, untimed). */
		for (index = matrix_bytes; index < arena_bytes; index += matrix_bytes)
			CHECK(cudaMemcpy((uint8_t *)arena + index,arena,matrix_bytes,cudaMemcpyDeviceToDevice));
		free(host_matrix);
		free(host_input);
		free(host_scales);
	}

	for (path = 0u; path < 2u; path++)
	{
		for (B = 1u; B <= 8u; B *= 2u)
		{
			double best = 0.0;
			for (step = 0u; step < WARMUP_STEPS + TIMED_STEPS; step++)
			{
				double begin,end;
				CHECK(cudaDeviceSynchronize());
				begin = now_s();
				for (matrix = 0u; matrix < matrix_count; matrix++)
				{
					const void *weights = (const uint8_t *)arena + (uint64_t)matrix * matrix_bytes;
					if ( path == 0u )
					{
						SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u><<<
							dim3(B,(N_DIM + 7u) / 8u),SPARK_LM_CTA_THREADS,
							K_DIM * sizeof(float)>>>(
							SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,weights,
							scales,input,output_scalar,B,K_DIM,N_DIM);
					}
					else if ( B == 1u )
					{
						SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u><<<
							dim3(1u,(N_DIM + 7u) / 8u),SPARK_LM_CTA_THREADS,
							K_DIM * sizeof(float)>>>(
							SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,weights,
							scales,input,output_batched,1u,K_DIM,N_DIM);
					}
					else
					{
						SparkLmBatchedLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,
							SPARK_LM_TILE,SPARK_LM_CTA_WARPS><<<
							dim3((N_DIM + 7u) / 8u),SPARK_LM_CTA_THREADS,
							B * (SPARK_LM_BATCHED_LINEAR_CHUNK * (uint32_t)sizeof(float))>>>(
							SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,weights,
							scales,input,output_batched,B,K_DIM,N_DIM);
					}
				}
				CHECK(cudaDeviceSynchronize());
				end = now_s();
				if ( step >= WARMUP_STEPS )
				{
					double seconds = end - begin;
					if ( best == 0.0 || seconds < best )
						best = seconds;
				}
			}
			per_path[path][B] = best;
		}
	}

	/* Exactness spot check on matrix 0: per-row scalar vs batched at B=4. */
	{
		const void *weights = arena;
		uint32_t row;
		uint64_t mismatch = 0u;
		for (row = 0u; row < 4u; row++)
			SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u><<<
				dim3(1u,(N_DIM + 7u) / 8u),SPARK_LM_CTA_THREADS,
				K_DIM * sizeof(float)>>>(
				SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,weights,scales,
				input + ((uint64_t)row * K_DIM),
				output_scalar + ((uint64_t)row * N_DIM),1u,K_DIM,N_DIM);
		SparkLmBatchedLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,
			SPARK_LM_TILE,SPARK_LM_CTA_WARPS><<<
			dim3((N_DIM + 7u) / 8u),SPARK_LM_CTA_THREADS,
			4u * (SPARK_LM_BATCHED_LINEAR_CHUNK * (uint32_t)sizeof(float))>>>(
			SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,weights,scales,input,
			output_batched,4u,K_DIM,N_DIM);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		{
			uint16_t *host_scalar = (uint16_t *)malloc(4ull * N_DIM * sizeof(uint16_t));
			uint16_t *host_batched = (uint16_t *)malloc(4ull * N_DIM * sizeof(uint16_t));
			CHECK(cudaMemcpy(host_scalar,output_scalar,4ull * N_DIM * sizeof(uint16_t),cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(host_batched,output_batched,4ull * N_DIM * sizeof(uint16_t),cudaMemcpyDeviceToHost));
			for (uint64_t index = 0u; index < 4ull * N_DIM; index++)
				if ( host_scalar[index] != host_batched[index] )
					mismatch++;
			if ( mismatch == 0u )
				printf("exactness: batched == per-row scalar, B=4, %llu neurons bit-exact\n",
					(unsigned long long)(4ull * N_DIM));
			else
				printf("exactness: %llu MISMATCHES of %llu\n",
					(unsigned long long)mismatch,(unsigned long long)(4ull * N_DIM));
			free(host_scalar);
			free(host_batched);
		}
	}

	printf("\n%-16s %2s %10s %10s %12s %8s\n","route","B","ms/step","GB/s","agg tok/s","vs B1");
	for (path = 0u; path < 2u; path++)
	{
		double base = 0.0;
		for (B = 1u; B <= 8u; B *= 2u)
		{
			double seconds = per_path[path][B];
			double gb_s = (double)arena_bytes / seconds / 1e9;
			double agg = (double)B / seconds;
			if ( B == 1u )
				base = agg;
			printf("%-16s %2u %10.1f %10.1f %12.2f %7.2fx\n",path_name[path],
				(unsigned)B,seconds * 1e3,gb_s,agg,agg / base);
		}
	}
	printf("\nwin condition: batched B2/B1 aggregate ratio = %.2fx (need > 1.5x)\n",
		(2.0 / per_path[1][2]) / (1.0 / per_path[1][1]));
	CHECK(cudaFree(arena));
	CHECK(cudaFree(scales));
	CHECK(cudaFree(input));
	CHECK(cudaFree(output_scalar));
	CHECK(cudaFree(output_batched));
	return 0;
}
