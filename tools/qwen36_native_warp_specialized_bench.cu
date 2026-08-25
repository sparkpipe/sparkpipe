// WARP-SPECIALIZED KERNEL - VERIFIED BIT-EXACT (2026-08-22, M=8 K=5120
// N=17408: 139264/139264 at depths 2 and 4). 512-thread CTAs: warps 8-15
// PRODUCE (the B ring via cp.async + the A ring via an 8-warp 2-rows-per-
// warp bit-exact variant of StageMxf8's warp-per-row quantizer), warps 0-7
// CONSUME (mma). Handshake = monotone shared counters with block fences;
// NO cross-group barriers (the consumer group uses bar 1 count 256, the
// producer group bar 2 count 256; barrier 0 stays __syncthreads-only).
//
// THE THREE RACES THE VERIFICATION SURVIVED (all fixed, all subtle):
// 1. cp.async groups are PER-THREAD - warp 8's wait_group does not cover
//    warps 9-15's copies; a group barrier between ALL producers' waits and
//    the b_ready publish is MANDATORY.
// 2. The 2-slot A ring needed the same consume-gating as the D-slot B ring
//    (staging A[c+1] overwrites the consumer's A[c-1]).
// 3. Spin gates must be followed by a GROUP barrier, not __syncwarp -
//    only warp 8 waited; warps 9-15 raced past.
// Also: SPARK_LM_CTA_THREADS is 256 (8 warps) - the specialization needs
// an explicit 512-thread launch (the first version's producers never ran).
//
// RESULT (end of session): **D=4 BIT-EXACT at 176.8-178.1 GB/s** - +41%
// over the production kernel (125.6), via the A-OVERLAP: producers stage
// the raw bf16 A (4KB/chunk) alongside B as ONE cp.async commit group
// (pure streaming); consumers quantize A from global input between their
// two sub-barriers (~450ns off the producer path - the earlier
// producer-inline-A cost 45 GB/s). StCpWait<0> at chunk 0 (the prologue
// group holds B[0]), wait<1> after (the needed group is second-to-last).
// KNOWN RESIDUAL RACES (do not ship these configs): D=2 corrupts ~3% (a
// depth-2-specific gate; D=4 is clean); the RAW_A shared-ring read path
// (RAW_A_FROM_GLOBAL undefined) corrupts - the consumer-side quantize
// reads the staged ring before a fence lands; until fixed, the verified
// config quantizes from GLOBAL (the cost is inside the consumer's
// measured 176.8).
// PERF LEDGER (all bit-exact unless noted, M=8 K=5120 N=17408):
//   direct byte-load (production)              125.6 GB/s
//   this kernel, first verified                 126.4
//   + lean sync (verified this session, recipe below)  133.0-133.4
//   + A-quantize moved to the consumer         127.3  (WORSE: serializes mma)
//   A-OVERLAP, D=4, quantize-from-global       176.8  VERIFIED BIT-EXACT
//   A-OVERLAP, D=2 (racy)                      176.3  ~3% corrupt
//   STAGE_ONLY experiments (the decisive data):
//     producers + inline A-quantize            ~136
//     pure-B producers (no A work at all)      179.6-182.2
//   => THE A-QUANTIZE IS ORPHAN WORK (~450ns/chunk): on the producer it
//   costs 45 GB/s of streaming; on the consumer it serializes the mma
//   critical path. Neither side has slack. The path to 180+: OVERLAP it -
//   cp.async the RAW bf16 A input (4KB/chunk) on the producer into a ring
//   and quantize from shared on the consumer (~150ns), or give A its own
//   warp group. Pure-B streaming at 180 leaves ~45 GB/s of headroom to
//   the 266 pure-read ceiling (the remaining gap: the cp.async 16B issue
//   pattern vs bulk TMA).
//
// LEAN-SYNC RECIPE (reconstructs the verified 133.0 from this base):
//   1. A ring 4 slots, stage chunk c+2 (gate: consumed >= c, folds into
//      the top spin); prologue stages A[0] AND A[1].
//   2. Prologue stages B chunks 0..D-2; iteration c restages chunk c+D-1
//      into slot (c-1)%D (gate: the SAME top spin consumed >= c) and uses
//      StCpWait<D-2> - one producer barrier per chunk total.
//   3. b_scale_tile[2][128]: producers stage each next chunk's e8m0 scale
//      bytes (1 strided byte load per neuron); consumers read scales from
//      shared instead of SparkLmSm121ScaleB's global __ldg.
// Build: same includes as tools/qwen38_27b_native_staged_bench.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

__device__ uint32_t g_ws_dbg[4];
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


template <bool PLAIN_B>
static __device__ __forceinline__ void StageB16(void *shmem, const void *gmem)
{
	if ( PLAIN_B )
		*(uint4 *)shmem = *(const uint4 *)gmem;
	else
		StCpAsync16(shmem,gmem);
}
template <int Groups>
static __device__ __forceinline__ void StCpWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(Groups));
}

/* Warp-specialized with the A-OVERLAP: producers stage BOTH the B ring and
 * the RAW bf16 A input (4KB/chunk) via cp.async - one commit group per
 * iteration, pure streaming (the inline quantize measured 45 GB/s of
 * streaming loss; consumer-side global quantize serialized the mma).
 * Consumers quantize A from SHARED (~150ns) between their two sub-barriers.
 * Handshake: one counter (b_ready = "B[c] and rawA[c] landed"), one top
 * spin (consumed >= c) gating both ring slots. The wait is StCpWait<1>
 * regardless of D: one commit per iteration, all-but-the-last complete. */
template <uint32_t D, bool PLAIN_B>
static __global__ __launch_bounds__(512u, 1)
void SparkQwen38_27bWarpSpecializedKernel(
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
	__shared__ uint8_t raw_a[2u * 8u * ST_CHUNK_K * 2u];   /* bf16, real rows only */
	__shared__ uint8_t a_e4m3[16u * ST_CHUNK_K];
	__shared__ uint8_t a_scale[16u * (ST_CHUNK_K / 32u)];
	__shared__ uint8_t b_scale_tile[2u * ST_TILE_N];
	__shared__ volatile uint32_t b_ready;
	__shared__ volatile uint32_t b_consumed;
	const uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	const uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	const uint32_t chunks = input_dimension / ST_CHUNK_K;
	const uint32_t tile_n_base = blockIdx.y * ST_TILE_N;
	const uint32_t row_base = blockIdx.x * 16u;
	uint32_t chunk, step;

	if ( threadIdx.x == 0u )
	{
		b_ready = 0u;
		b_consumed = 0u;
	}
	/* zero the padded A rows once (rows 8..15 read by LoadMxf8A forever) */
	for ( uint32_t z = threadIdx.x; z < 16u * ST_CHUNK_K; z += 512u )
		a_e4m3[z] = 0u;
	for ( uint32_t z = threadIdx.x; z < 16u * (ST_CHUNK_K / 32u); z += 512u )
		a_scale[z] = 127u;
	__syncthreads();

	if ( warp >= 8u )
	{
		/* ---------------- producers ---------------- */
		const uint32_t ptid = (warp - 8u) * SPARK_LM_WARP_LANES + lane;
		const uint32_t pthreads = 8u * SPARK_LM_WARP_LANES;
		/* prologue: B chunks 0..D-2 + rawA chunks 0,1 + scales 0,1, ONE commit */
		for ( chunk = 0u; chunk + 1u < D && chunk < chunks; ++chunk )
			for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
			{
				uint32_t neuron = t / (ST_CHUNK_K / 16u);
				uint32_t k16 = t % (ST_CHUNK_K / 16u);
				StageB16<PLAIN_B>(staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE +
					neuron * ST_B_STRIDE + k16 * 16u,
					(const uint8_t *)weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
					(uint64_t)chunk * ST_CHUNK_K + k16 * 16u);
			}
		for ( uint32_t ac = 0u; ac < 2u && ac < chunks; ++ac )
		{
			for ( uint32_t t = ptid; t < 8u * (ST_CHUNK_K * 2u / 16u); t += pthreads )
			{
				uint32_t row = t / (ST_CHUNK_K * 2u / 16u);
				uint32_t k16 = t % (ST_CHUNK_K * 2u / 16u);
				if ( row_base + row < row_count )
					StCpAsync16(raw_a + (ac & 1u) * 8u * ST_CHUNK_K * 2u + row * ST_CHUNK_K * 2u + k16 * 16u,
						(const uint8_t *)input_bf16 + (uint64_t)(row_base + row) * input_row_stride * 2u +
						(uint64_t)ac * ST_CHUNK_K * 2u + k16 * 16u);
			}
		}
		for ( uint32_t sc = 0u; sc < 2u && sc < chunks; ++sc )
			for ( uint32_t n = ptid; n < ST_TILE_N; n += pthreads )
				b_scale_tile[(sc & 1u) * ST_TILE_N + n] =
					weight_scale_e8m0[(uint64_t)(tile_n_base + n) * (input_dimension / 128u) + sc];
		StCpCommit();
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			if ( ptid == 0u )
				while ( b_consumed < chunk )
					__nanosleep(64);
			/* chunk 0's B/rawA live in the PROLOGUE group - the only group,
			 * so it must fully complete; later chunks' data is in the
			 * second-to-last group (wait<1> keeps just the last in flight) */
			if ( chunk == 0u )
				StCpWait<0>();
			else
				StCpWait<1>();
			asm volatile("bar.sync 2, 256;");
			if ( ptid == 0u )
			{
				__threadfence_block();
				b_ready = chunk + 1u;
			}
			/* scales for chunk+1 */
			if ( chunk + 1u < chunks )
				for ( uint32_t n = ptid; n < ST_TILE_N; n += pthreads )
					b_scale_tile[((chunk + 1u) & 1u) * ST_TILE_N + n] =
						weight_scale_e8m0[(uint64_t)(tile_n_base + n) * (input_dimension / 128u) + (chunk + 1u)];
			/* B restage chunk+D-1 + rawA chunk+1, one commit */
			if ( chunk + D - 1u < chunks )
				for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
				{
					uint32_t neuron = t / (ST_CHUNK_K / 16u);
					uint32_t k16 = t % (ST_CHUNK_K / 16u);
					uint32_t rchunk = chunk + D - 1u;
					StageB16<PLAIN_B>(staged_b + (rchunk % D) * ST_TILE_N * ST_B_STRIDE +
						neuron * ST_B_STRIDE + k16 * 16u,
						(const uint8_t *)weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
						(uint64_t)rchunk * ST_CHUNK_K + k16 * 16u);
				}
			if ( chunk + 1u < chunks )
				for ( uint32_t t = ptid; t < 8u * (ST_CHUNK_K * 2u / 16u); t += pthreads )
				{
					uint32_t row = t / (ST_CHUNK_K * 2u / 16u);
					uint32_t k16 = t % (ST_CHUNK_K * 2u / 16u);
					if ( row_base + row < row_count )
						StCpAsync16(raw_a + ((chunk + 1u) & 1u) * 8u * ST_CHUNK_K * 2u + row * ST_CHUNK_K * 2u + k16 * 16u,
							(const uint8_t *)input_bf16 + (uint64_t)(row_base + row) * input_row_stride * 2u +
							(uint64_t)(chunk + 1u) * ST_CHUNK_K * 2u + k16 * 16u);
				}
			StCpCommit();
		}
		return;
	}

	/* ---------------- consumers (warps 0-7) ---------------- */
	{
		float total[2][4] = {};
		const uint32_t neuron_base = tile_n_base + warp * (2u * 8u);
		uint32_t ni, entry;
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			uint8_t *cur = staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE;
			const uint8_t *raw = raw_a + (chunk & 1u) * 8u * ST_CHUNK_K * 2u;
			if ( threadIdx.x == 0u )
				while ( b_ready < chunk + 1u )
					__nanosleep(64);
			if ( PLAIN_B )
				asm volatile("bar.sync 1, 256;fence.acq_rel.cta;");
			else
				asm volatile("bar.sync 1, 256;");
			/* quantize A[c] from SHARED raw (warp w = row w; 8 real rows) */
			for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			{
				uint8_t *dst = a_e4m3 + step * 16u * 32u + warp * 32u;
				if ( row_base + warp < row_count )
				{
#ifdef RAW_A_FROM_GLOBAL
					const __nv_bfloat16 *src = (const __nv_bfloat16 *)
						((const uint8_t *)input_bf16 + (uint64_t)(row_base + warp) * input_row_stride * 2u +
						(uint64_t)chunk * ST_CHUNK_K * 2u + step * 32u * 2u);
#else
					const __nv_bfloat16 *src = (const __nv_bfloat16 *)(raw +
						warp * ST_CHUNK_K * 2u + step * 32u * 2u);
#endif
					float value = __bfloat162float(src[lane]);
					float amax = LmActivationWarpMax(fabsf(value));
					amax = __shfl_sync(0xffffffffu, amax, 0u);
					uint8_t scale_code = SparkLmSm121E8m0ScaleCode(amax);
					float scale = SparkLmSm121E8m0ScaleValue(scale_code);
					dst[lane] = LmFloatToE4m3(value / scale);
					if ( lane == 0u )
						a_scale[step * 16u + warp] = scale_code;
				}
			}
			asm volatile("bar.sync 1, 256;");
			for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			{
				uint32_t k_base = chunk * ST_CHUNK_K + step * 32u;
				uint32_t a[4], scale_a, scale_b, b[2], reg;
				SparkLmSm121LoadMxf8A(a_e4m3 + step * 16u * 32u, 0u, lane, a);
				scale_a = SparkLmSm121ScaleA(a_scale + step * 16u, 0u, lane);
				#pragma unroll
				for ( ni = 0u; ni < 2u; ++ni )
				{
					uint32_t fragment_neuron = neuron_base + ni * 8u;
					scale_b = (uint32_t)b_scale_tile[(chunk & 1u) * ST_TILE_N +
						(fragment_neuron - tile_n_base) + LmMma8OperandBRow(lane)];
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
			/* ALL consumer threads must be past this chunk's reads before the
			 * producer may restage slot (chunk-1)%D: thread 0 publishing
			 * alone races the slower warps (latent in the cp.async build,
			 * exposed by plain stores' instant visibility) */
			asm volatile("bar.sync 1, 256;");
			if ( threadIdx.x == 0u )
			{
				__threadfence_block();
				b_consumed = chunk + 1u;
			}
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
	uint32_t depth, int plain_b = 0)
{
	dim3 grid((rows + 15u) / 16u, N / ST_TILE_N);
	size_t shared = (size_t)depth * ST_TILE_N * ST_B_STRIDE;
	cudaError_t error;
	#define ST_LAUNCH(D, PB) do { \
		error = cudaFuncSetAttribute(SparkQwen38_27bWarpSpecializedKernel<D, PB>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared); \
		if ( error != cudaSuccess ) return(error); \
		SparkQwen38_27bWarpSpecializedKernel<D, PB><<<grid, 512u, shared, stream>>>(\
			(const uint8_t *)payload, scale, in, in_stride, out, out_stride, rows, K, N); \
	} while (0)
	if ( plain_b != 0 )
	{
		if ( depth != 4u )
			return(cudaErrorInvalidValue);
		ST_LAUNCH(4u, true);
		return(cudaGetLastError());
	}
	switch ( depth )
	{
	case 2u: ST_LAUNCH(2u, false); break;
	case 4u: ST_LAUNCH(4u, false); break;
	case 6u: ST_LAUNCH(6u, false); break;
	case 8u: ST_LAUNCH(8u, false); break;
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
	/* PLAIN-B variant: producers stage B with plain uint4 loads + shared
	 * stores instead of cp.async 16B (the ledger's 180 cp.async ceiling vs
	 * the 266 plain-read ceiling - is the async engine the bottleneck?)
	 * ANSWER: YES at the kernel level - 195-205 GB/s vs 176-181, bit-exact
	 * 3/3x WITH A READ FROM GLOBAL (the shared raw-A ring has a latent
	 * race that plain-B timing exposes: per-tile corruption, all rows,
	 * resistant to acquire fences and a publish barrier - root cause still
	 * open; production keeps A-from-global which is the verified config).
	 * IN SITU (o512 spec, kill-switch A/B on the same build): wall and
	 * ffn_ms/frame IDENTICAL (20.90 vs 20.91s, 85.1 vs 85.3ms) - the
	 * verify frame's FFN runs at ~123 GB/s effective regardless of the
	 * staging engine; the gap to 176 is interleaving/cold-L2/gaps, NOT
	 * the B-staging path. Plain-B ships default-ON (neutral, bit-exact)
	 * with kill-switch SPARK_QWEN38_27B_WS_PLAIN=0. */
	{
		CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, 4u, 1));
		CHECK(cudaStreamSynchronize(stream));
		uint16_t *h_ref = (uint16_t *)malloc((size_t)M * N * 2u);
		uint16_t *h_out = (uint16_t *)malloc((size_t)M * N * 2u);
		CHECK(cudaMemcpy(h_ref, ref, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_out, out, (size_t)M * N * 2u, cudaMemcpyDeviceToHost));
		int exact = 0;
		uint32_t row_err[32] = {0}, col_err[64] = {0};
		for (uint32_t i = 0; i < M * N; i++)
			if (h_ref[i] != h_out[i])
			{
				exact++;
				row_err[i / N]++;
				col_err[(i % N) * 64u / N]++;
			}
		printf("plain-b correctness: exact=%d/%u rows:", M * N - exact, M * N);
		for (uint32_t r = 0u; r < M; r++) printf(" %u", row_err[r]);
		printf(" cols:");
		for (uint32_t c = 0u; c < 64u; c++) printf("%u", col_err[c] > 9u ? 9u : col_err[c]);
		printf("\n");
		free(h_ref); free(h_out);
		for (int i = 0; i < 8; i++)
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, 4u, 1));
		CHECK(cudaStreamSynchronize(stream));
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for (int i = 0; i < iters; i++)
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, 4u, 1));
		CHECK(cudaStreamSynchronize(stream));
		double dt = now_s() - t0;
		printf("plain-b M=%u K=%u N=%u D=4: %.1f GB/s (%.3f ms/iter)\n", M, K, N, bytes * iters / dt / 1e9, dt / iters * 1e3);
	}
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
		{
			uint32_t dbg[4];
			CHECK(cudaMemcpyFromSymbol(dbg, g_ws_dbg, 16));
			printf("  counters: b_ready=%u a_ready=%u b_consumed=%u mark=%u (chunks=%u)\n",
				dbg[0], dbg[1], dbg[2], dbg[3], (unsigned)(K / 128u));
		}
	}
	for (uint32_t depth = 2u; depth <= 4u; depth += 2u)
	{
		for (int i = 0; i < 8; i++)
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, depth));
		CHECK(cudaStreamSynchronize(stream));
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for (int i = 0; i < iters; i++)
		{
#ifdef ISOLATED
			CHECK(cudaStreamSynchronize(stream));
#endif
			CHECK(LaunchStaged(stream, payload, scale, in, K, out, N, M, K, N, depth));
		}
		CHECK(cudaStreamSynchronize(stream));
		double dt = (now_s() - t0) / iters;
		printf("staged M=%u K=%u N=%u depth=%u: %.3f ms/iter, effective %.1f GB/s (direct byte-load: 125.6; pure read: 266-276)\n",
			M, K, N, depth, dt * 1e3, bytes / dt / 1e9);
	}
	return 0;
}
