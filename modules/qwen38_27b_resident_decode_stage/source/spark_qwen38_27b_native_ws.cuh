#pragma once
#include "sparkpipe/spark_lm_kernels.cuh"

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

template <bool PLAIN_B>
static __device__ __forceinline__ void StageB16(void *shmem, const void *gmem)
{
	if ( PLAIN_B )
		*(uint4 *)shmem = *(const uint4 *)gmem;
	else
		StCpAsync16(shmem,gmem);
}

#define ST_TILE_N 128u
#define ST_CHUNK_K 128u
#define ST_B_STRIDE (ST_CHUNK_K + 16u)

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
	extern __shared__ uint8_t staged_b[];
	__shared__ uint8_t raw_a[2u * 8u * ST_CHUNK_K * 2u];
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
	for ( uint32_t z = threadIdx.x; z < 16u * ST_CHUNK_K; z += 512u )
		a_e4m3[z] = 0u;
	for ( uint32_t z = threadIdx.x; z < 16u * (ST_CHUNK_K / 32u); z += 512u )
		a_scale[z] = 127u;
	__syncthreads();

	if ( warp >= 8u )
	{
		const uint32_t ptid = (warp - 8u) * SPARK_LM_WARP_LANES + lane;
		const uint32_t pthreads = 8u * SPARK_LM_WARP_LANES;
		for ( chunk = 0u; chunk + 1u < D && chunk < chunks; ++chunk )
			for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
			{
				uint32_t neuron = t / (ST_CHUNK_K / 16u);
				uint32_t k16 = t % (ST_CHUNK_K / 16u);
				StCpAsync16(staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE +
					neuron * ST_B_STRIDE + k16 * 16u,
					weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
					(uint64_t)chunk * ST_CHUNK_K + k16 * 16u);
			}
		for ( uint32_t sc = 0u; sc < 2u && sc < chunks; ++sc )
			for ( uint32_t n = ptid; n < ST_TILE_N; n += pthreads )
				b_scale_tile[(sc & 1u) * ST_TILE_N + n] =
					weight_scale_e8m0[(uint64_t)(tile_n_base + n) * (input_dimension / 128u) + sc];
		for ( uint32_t ac = 0u; ac < 2u && ac < chunks; ++ac )
			for ( uint32_t t = ptid; t < 8u * (ST_CHUNK_K * 2u / 16u); t += pthreads )
			{
				uint32_t row = t / (ST_CHUNK_K * 2u / 16u);
				uint32_t k16 = t % (ST_CHUNK_K * 2u / 16u);
				if ( row_base + row < row_count )
					StCpAsync16(raw_a + (ac & 1u) * 8u * ST_CHUNK_K * 2u + row * ST_CHUNK_K * 2u + k16 * 16u,
						(const uint8_t *)input_bf16 + (uint64_t)(row_base + row) * input_row_stride * 2u +
						(uint64_t)ac * ST_CHUNK_K * 2u + k16 * 16u);
			}
		StCpCommit();
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			if ( ptid == 0u )
				while ( b_consumed < chunk )
					__nanosleep(64);
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
			if ( chunk + 1u < chunks )
				for ( uint32_t n = ptid; n < ST_TILE_N; n += pthreads )
					b_scale_tile[((chunk + 1u) & 1u) * ST_TILE_N + n] =
						weight_scale_e8m0[(uint64_t)(tile_n_base + n) * (input_dimension / 128u) + (chunk + 1u)];
			if ( chunk + D - 1u < chunks )
				for ( uint32_t t = ptid; t < ST_TILE_N * (ST_CHUNK_K / 16u); t += pthreads )
				{
					uint32_t neuron = t / (ST_CHUNK_K / 16u);
					uint32_t k16 = t % (ST_CHUNK_K / 16u);
					uint32_t rchunk = chunk + D - 1u;
					StCpAsync16(staged_b + (rchunk % D) * ST_TILE_N * ST_B_STRIDE +
						neuron * ST_B_STRIDE + k16 * 16u,
						weight_payload + (uint64_t)(tile_n_base + neuron) * input_dimension +
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

	{
		float total[2][4] = {};
		const uint32_t neuron_base = tile_n_base + warp * (2u * 8u);
		uint32_t ni, entry;
		for ( chunk = 0u; chunk < chunks; ++chunk )
		{
			uint8_t *cur = staged_b + (chunk % D) * ST_TILE_N * ST_B_STRIDE;
			if ( threadIdx.x == 0u )
				while ( b_ready < chunk + 1u )
					__nanosleep(64);
			asm volatile("bar.sync 1, 256;");
			if ( PLAIN_B )
				__threadfence_block();
			for ( step = 0u; step < ST_CHUNK_K / 32u; ++step )
			{
				uint8_t *dst = a_e4m3 + step * 16u * 32u + warp * 32u;
				if ( row_base + warp < row_count )
				{
					const __nv_bfloat16 *src = (const __nv_bfloat16 *)
						((const uint8_t *)input_bf16 + (uint64_t)(row_base + warp) * input_row_stride * 2u +
						(uint64_t)chunk * ST_CHUNK_K * 2u + step * 32u * 2u);
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


static inline cudaError_t SparkQwen38_27bLaunchWsLinear(
	cudaStream_t stream,
	const void *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	uint64_t input_row_stride,
	void *output_bf16,
	uint64_t output_row_stride,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	dim3 grid((row_count + 15u) / 16u, output_dimension / ST_TILE_N);
	size_t shared = 4u * ST_TILE_N * ST_B_STRIDE +
		16u * ST_CHUNK_K +
		16u * (ST_CHUNK_K / 32u) +
		2u * ST_TILE_N;
	static bool ws_shared_ready = false;
	if ( !ws_shared_ready )
	{
		cudaError_t error = cudaFuncSetAttribute(SparkQwen38_27bWarpSpecializedKernel<4u,true>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if ( error != cudaSuccess )
			return(error);
		error = cudaFuncSetAttribute(SparkQwen38_27bWarpSpecializedKernel<4u,false>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if ( error != cudaSuccess )
			return(error);
		ws_shared_ready = true;
	}
	{
		const char *plain_env = getenv("SPARK_QWEN38_27B_WS_PLAIN");
		if ( plain_env != 0 && plain_env[0] == '0' )
			SparkQwen38_27bWarpSpecializedKernel<4u,false><<<grid, 512u, shared, stream>>>(
				(const uint8_t *)weight_payload, weight_scale_e8m0,
				input_bf16, input_row_stride, output_bf16, output_row_stride,
				row_count, input_dimension, output_dimension);
		else
			SparkQwen38_27bWarpSpecializedKernel<4u,true><<<grid, 512u, shared, stream>>>(
				(const uint8_t *)weight_payload, weight_scale_e8m0,
				input_bf16, input_row_stride, output_bf16, output_row_stride,
				row_count, input_dimension, output_dimension);
	}
	return(cudaGetLastError());
}
