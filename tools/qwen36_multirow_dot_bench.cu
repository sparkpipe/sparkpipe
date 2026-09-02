#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s = %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

#define MRD_MAX_M 16u
#define MRD_WARPS 8u

static __global__ void MrdFp32Kernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const __nv_bfloat16 *input_bf16,
	__nv_bfloat16 *output_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t neuron = blockIdx.x * MRD_WARPS + warp;
	if ( neuron >= output_dimension )
		return;
	const uint32_t groups = input_dimension >> 7u;
	const uint64_t wrow = (uint64_t)neuron * input_dimension;
	const uint64_t srow = (uint64_t)neuron * groups;
	float acc[MRD_MAX_M];
	uint32_t m,g;
	#pragma unroll
	for ( m = 0u; m < MRD_MAX_M; m++ )
		acc[m] = 0.0f;
	for ( g = 0u; g < groups; g++ )
	{
		float gacc[MRD_MAX_M];
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
			gacc[m] = 0.0f;
		const uint32_t packed = __ldg(((const uint32_t *)weight_payload) + ((wrow + (g << 7u)) >> 2u) + lane);
		uint32_t decoded[2];
		float wv[4];
		SparkLmDecodeE4m3x4Half2(packed,decoded);
		{
			SparkLmHalf2Bits b0,b1;
			b0.bits = decoded[0]; b1.bits = decoded[1];
			float2 v0 = __half22float2(b0.values), v1 = __half22float2(b1.values);
			wv[0] = v0.x; wv[1] = v0.y; wv[2] = v1.x; wv[3] = v1.y;
		}
		const uint32_t k_base = (g << 7u) + (lane << 2u);
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
		{
			if ( m < row_count )
			{
				const __nv_bfloat16 *arow = input_bf16 + (uint64_t)m * input_dimension + k_base;
				gacc[m] = fmaf(wv[0],__bfloat162float(arow[0]),gacc[m]);
				gacc[m] = fmaf(wv[1],__bfloat162float(arow[1]),gacc[m]);
				gacc[m] = fmaf(wv[2],__bfloat162float(arow[2]),gacc[m]);
				gacc[m] = fmaf(wv[3],__bfloat162float(arow[3]),gacc[m]);
			}
		}
		const float sv = __uint_as_float((uint32_t)__ldg(weight_scale_e8m0 + srow + g) << 23u);
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
			if ( m < row_count )
				acc[m] = fmaf(gacc[m],sv,acc[m]);
	}
	#pragma unroll
	for ( m = 0u; m < MRD_MAX_M; m++ )
	{
		if ( m >= row_count )
			break;
		float v = acc[m];
		#pragma unroll
		for ( uint32_t off = 16u; off != 0u; off >>= 1u )
			v += __shfl_down_sync(0xffffffffu,v,off);
		if ( lane == 0u )
			output_bf16[(uint64_t)m * output_dimension + neuron] = __float2bfloat16(v);
	}
}


static __global__ void MrdFp32x2Kernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const __nv_bfloat16 *input_bf16,
	__nv_bfloat16 *output_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t neuron0 = (blockIdx.x * MRD_WARPS + warp) * 2u;
	if ( neuron0 >= output_dimension )
		return;
	const uint32_t groups = input_dimension >> 7u;
	float acc0[MRD_MAX_M],acc1[MRD_MAX_M];
	uint32_t m,g;
	#pragma unroll
	for ( m = 0u; m < MRD_MAX_M; m++ )
	{
		acc0[m] = 0.0f;
		acc1[m] = 0.0f;
	}
	const uint32_t *w32 = (const uint32_t *)weight_payload;
	for ( g = 0u; g < groups; g++ )
	{
		float a[MRD_MAX_M][4];
		const uint32_t k_base = (g << 7u) + (lane << 2u);
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
		{
			if ( m < row_count )
			{
				uint2 pair = *(((const uint2 *)input_bf16) + ((uint64_t)m * input_dimension + k_base) / 4u);
				__nv_bfloat16 b0 = *(const __nv_bfloat16 *)&pair.x;
				__nv_bfloat16 b1 = *((const __nv_bfloat16 *)&pair.x + 1);
				__nv_bfloat16 b2 = *(const __nv_bfloat16 *)&pair.y;
				__nv_bfloat16 b3 = *((const __nv_bfloat16 *)&pair.y + 1);
				a[m][0] = __bfloat162float(b0);
				a[m][1] = __bfloat162float(b1);
				a[m][2] = __bfloat162float(b2);
				a[m][3] = __bfloat162float(b3);
			}
		}
		float wv0[4],wv1[4];
		{
			uint32_t dec[2];
			SparkLmDecodeE4m3x4Half2(__ldg(w32 + ((uint64_t)neuron0 * input_dimension + (g << 7u)) / 4u + lane),dec);
			SparkLmHalf2Bits b;
			b.bits = dec[0]; float2 v = __half22float2(b.values); wv0[0] = v.x; wv0[1] = v.y;
			b.bits = dec[1]; v = __half22float2(b.values); wv0[2] = v.x; wv0[3] = v.y;
			SparkLmDecodeE4m3x4Half2(__ldg(w32 + ((uint64_t)(neuron0 + 1u) * input_dimension + (g << 7u)) / 4u + lane),dec);
			b.bits = dec[0]; v = __half22float2(b.values); wv1[0] = v.x; wv1[1] = v.y;
			b.bits = dec[1]; v = __half22float2(b.values); wv1[2] = v.x; wv1[3] = v.y;
		}
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
		{
			if ( m < row_count )
			{
				acc0[m] = fmaf(wv0[0],a[m][0],acc0[m]);
				acc0[m] = fmaf(wv0[1],a[m][1],acc0[m]);
				acc0[m] = fmaf(wv0[2],a[m][2],acc0[m]);
				acc0[m] = fmaf(wv0[3],a[m][3],acc0[m]);
				acc1[m] = fmaf(wv1[0],a[m][0],acc1[m]);
				acc1[m] = fmaf(wv1[1],a[m][1],acc1[m]);
				acc1[m] = fmaf(wv1[2],a[m][2],acc1[m]);
				acc1[m] = fmaf(wv1[3],a[m][3],acc1[m]);
			}
		}
		const float sv0 = __uint_as_float((uint32_t)__ldg(weight_scale_e8m0 + (uint64_t)neuron0 * groups + g) << 23u);
		const float sv1 = __uint_as_float((uint32_t)__ldg(weight_scale_e8m0 + (uint64_t)(neuron0 + 1u) * groups + g) << 23u);
		#pragma unroll
		for ( uint32_t e = 0u; e < 4u; e++ )
		{
			wv0[e] *= sv0;
			wv1[e] *= sv1;
		}
	}
	#pragma unroll
	for ( m = 0u; m < MRD_MAX_M; m++ )
	{
		if ( m >= row_count )
			break;
		float v0 = acc0[m],v1 = acc1[m];
		#pragma unroll
		for ( uint32_t off = 16u; off != 0u; off >>= 1u )
		{
			v0 += __shfl_down_sync(0xffffffffu,v0,off);
			v1 += __shfl_down_sync(0xffffffffu,v1,off);
		}
		if ( lane == 0u )
		{
			output_bf16[(uint64_t)m * output_dimension + neuron0] = __float2bfloat16(v0);
			output_bf16[(uint64_t)m * output_dimension + neuron0 + 1u] = __float2bfloat16(v1);
		}
	}
}

template <uint32_t UNROLL, uint32_t MRD_M>
static __global__ void MrdFp32UnrollKernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const __nv_bfloat16 *input_bf16,
	__nv_bfloat16 *output_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t neuron = blockIdx.x * MRD_WARPS + warp;
	if ( neuron >= output_dimension )
		return;
	const uint32_t groups = input_dimension >> 7u;
	const uint32_t *w32 = (const uint32_t *)weight_payload;
	const uint32_t *a32 = (const uint32_t *)input_bf16;
	float acc[UNROLL][MRD_M];
	uint32_t m,g;
	#pragma unroll
	for ( uint32_t u = 0u; u < 4u; u++ )
		#pragma unroll
		for ( m = 0u; m < MRD_MAX_M; m++ )
			acc[u][m] = 0.0f;
	for ( g = 0u; g + UNROLL <= groups; g += UNROLL )
	{
		#pragma unroll
		for ( uint32_t u = 0u; u < UNROLL; u++ )
		{
			const uint32_t gg = g + u;
			float wv[4];
			uint32_t dec[2];
			SparkLmDecodeE4m3x4Half2(__ldg(w32 + (((uint64_t)neuron * input_dimension + (gg << 7u)) >> 2u) + lane),dec);
			{
				SparkLmHalf2Bits b;
				b.bits = dec[0]; float2 v = __half22float2(b.values); wv[0] = v.x; wv[1] = v.y;
				b.bits = dec[1]; v = __half22float2(b.values); wv[2] = v.x; wv[3] = v.y;
			}
			const float sv = __uint_as_float((uint32_t)__ldg(weight_scale_e8m0 + (uint64_t)neuron * groups + gg) << 23u);
			wv[0] *= sv; wv[1] *= sv; wv[2] *= sv; wv[3] *= sv;
			const uint32_t k4 = (gg << 5u) + lane;
			#pragma unroll
			for ( m = 0u; m < MRD_M; m++ )
			{
				if ( m < row_count )
				{
					const uint32_t kq = k4 + ((uint32_t)m * (input_dimension >> 2u));
					uint2 pair = __ldg(((const uint2 *)a32) + (uint64_t)kq);
					float2 p01 = __bfloat1622float2(*(__nv_bfloat162 *)&pair.x);
					float2 p23 = __bfloat1622float2(*(__nv_bfloat162 *)&pair.y);
					acc[u][m] = fmaf(wv[0],p01.x,acc[u][m]);
					acc[u][m] = fmaf(wv[1],p01.y,acc[u][m]);
					acc[u][m] = fmaf(wv[2],p23.x,acc[u][m]);
					acc[u][m] = fmaf(wv[3],p23.y,acc[u][m]);
				}
			}
		}
	}
	for ( ; g < groups; g++ )
	{
		float wv[4];
		uint32_t dec[2];
		SparkLmDecodeE4m3x4Half2(__ldg(w32 + (((uint64_t)neuron * input_dimension + (g << 7u)) >> 2u) + lane),dec);
		{
			SparkLmHalf2Bits b;
			b.bits = dec[0]; float2 v = __half22float2(b.values); wv[0] = v.x; wv[1] = v.y;
			b.bits = dec[1]; v = __half22float2(b.values); wv[2] = v.x; wv[3] = v.y;
		}
		const float sv = __uint_as_float((uint32_t)__ldg(weight_scale_e8m0 + (uint64_t)neuron * groups + g) << 23u);
		wv[0] *= sv; wv[1] *= sv; wv[2] *= sv; wv[3] *= sv;
		#pragma unroll
		for ( m = 0u; m < MRD_M; m++ )
		{
			if ( m < row_count )
			{
				uint2 pair = __ldg(((const uint2 *)a32) + (uint64_t)(((g << 5u) + lane) + m * (input_dimension >> 2u)));
				float2 p01 = __bfloat1622float2(*(__nv_bfloat162 *)&pair.x);
				float2 p23 = __bfloat1622float2(*(__nv_bfloat162 *)&pair.y);
				acc[0][m] = fmaf(wv[0],p01.x,acc[0][m]);
				acc[0][m] = fmaf(wv[1],p01.y,acc[0][m]);
				acc[0][m] = fmaf(wv[2],p23.x,acc[0][m]);
				acc[0][m] = fmaf(wv[3],p23.y,acc[0][m]);
			}
		}
	}
	#pragma unroll
	for ( m = 0u; m < MRD_M; m++ )
	{
		if ( m >= row_count )
			break;
		float v = 0.0f;
		#pragma unroll
		for ( uint32_t u = 0u; u < UNROLL; u++ )
			v += acc[u][m];
		#pragma unroll
		for ( uint32_t off = 16u; off != 0u; off >>= 1u )
			v += __shfl_down_sync(0xffffffffu,v,off);
		if ( lane == 0u )
			output_bf16[(uint64_t)m * output_dimension + neuron] = __float2bfloat16(v);
	}
}
int main(int argc, char **argv)
{
	const uint32_t M = argc > 1 ? (uint32_t)atoi(argv[1]) : 9u;
	const uint32_t K = argc > 2 ? (uint32_t)atoi(argv[2]) : 5120u;
	const uint32_t N = argc > 3 ? (uint32_t)atoi(argv[3]) : 17408u;
	const int iters = 200;
	uint8_t *payload,*scale;
	__nv_bfloat16 *in,*out,*ref;
	const uint64_t payload_bytes = (uint64_t)N * K;
	const uint64_t scale_bytes = (uint64_t)N * (K / 128u);
	CHECK(cudaMalloc(&payload,payload_bytes));
	CHECK(cudaMalloc(&scale,scale_bytes));
	CHECK(cudaMalloc(&in,(uint64_t)M * K * 2u));
	CHECK(cudaMalloc(&out,(uint64_t)M * N * 2u));
	CHECK(cudaMalloc(&ref,(uint64_t)M * N * 2u));
	{
		uint8_t *h = (uint8_t *)malloc(payload_bytes);
		srand(12345);
		for ( uint64_t i = 0u; i < payload_bytes; i++ ) h[i] = (uint8_t)(rand() & 0x7e);
		CHECK(cudaMemcpy(payload,h,payload_bytes,cudaMemcpyHostToDevice));
		free(h);
		h = (uint8_t *)malloc(scale_bytes);
		for ( uint64_t i = 0u; i < scale_bytes; i++ ) h[i] = 120u + (uint8_t)(rand() % 12u);
		CHECK(cudaMemcpy(scale,h,scale_bytes,cudaMemcpyHostToDevice));
		free(h);
		h = (uint8_t *)malloc((size_t)M * K * 2u);
		uint16_t *h16 = (uint16_t *)h;
		for ( uint64_t i = 0u; i < (uint64_t)M * K; i++ )
		{
			__nv_bfloat16 b = __float2bfloat16((float)(rand() % 2000 - 1000) / 500.0f);
			h16[i] = *(uint16_t *)&b;
		}
		CHECK(cudaMemcpy(in,h,(size_t)M * K * 2u,cudaMemcpyHostToDevice));
		free(h);
	}
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	int have_native = SparkLmSm121NativeDecodeShape(M) != 0u || M == 8u;
	CHECK(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(
		stream,payload,scale,payload_bytes,scale_bytes,in,K,0u,0u,
		ref,N,0u,0u,1u,M,K,N));
	if ( !have_native )
		printf("(native ref failed for M=%u - expected; CPU check below)\n",M);
	const uint32_t grid = (N + MRD_WARPS - 1u) / MRD_WARPS;
	MrdFp32Kernel<<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
	CHECK(cudaGetLastError());
	CHECK(cudaStreamSynchronize(stream));
	{
		uint16_t *hr = (uint16_t *)malloc((size_t)M * N * 2u);
		uint16_t *ho = (uint16_t *)malloc((size_t)M * N * 2u);
		CHECK(cudaMemcpy(hr,ref,(size_t)M * N * 2u,cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(ho,out,(size_t)M * N * 2u,cudaMemcpyDeviceToHost));
		double max_rel = 0.0;
		for ( uint64_t i = 0u; i < (uint64_t)M * N; i++ )
		{
			float r = __bfloat162float(*(__nv_bfloat16 *)&hr[i]);
			float o = __bfloat162float(*(__nv_bfloat16 *)&ho[i]);
			double denom = r != 0.0f ? (r < 0.0 ? -r : r) : 1.0;
			double rel = (o - r) / denom;
			if ( rel < 0.0 ) rel = -rel;
			if ( rel > max_rel ) max_rel = rel;
		}
		printf("fp32 variant max_rel_diff vs native = %.5f\n",max_rel);
		free(hr); free(ho);
	}
	{
		uint8_t *hp = (uint8_t *)malloc(payload_bytes);
		uint8_t *hs = (uint8_t *)malloc(scale_bytes);
		uint16_t *hi = (uint16_t *)malloc((size_t)M * K * 2u);
		uint16_t *ho = (uint16_t *)malloc((size_t)M * N * 2u);
		CHECK(cudaMemcpy(hp,payload,payload_bytes,cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(hs,scale,scale_bytes,cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(hi,in,(size_t)M * K * 2u,cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(ho,out,(size_t)M * N * 2u,cudaMemcpyDeviceToHost));
		double worst = 0.0;
		for ( uint32_t neuron = 0u; neuron < 4u; neuron++ )
			for ( uint32_t m = 0u; m < M; m++ )
			{
				float cpu = 0.0f;
				for ( uint32_t g = 0u; g < K / 128u; g++ )
				{
					float gsum = 0.0f;
					for ( uint32_t e = 0u; e < 128u; e++ )
					{
						uint8_t b = hp[(uint64_t)neuron * K + g * 128u + e];
						__nv_bfloat16 raw = __ushort_as_bfloat16((unsigned short)0u);
						(void)raw;
						int sign = (b & 0x80) ? -1 : 1;
						int expo = (b >> 3) & 0xF;
						int mant = b & 0x7;
						float wv;
						if ( expo == 0u )
							wv = (float)sign * (float)mant / 8.0f / 16.0f;
						else if ( expo == 15u && mant == 7u )
							wv = 0.0f;
						else if ( expo >= 8u )
							wv = (float)sign * (float)(8u + (unsigned)mant) / 8.0f * (float)(1u << (expo - 7u));
						else
							wv = (float)sign * (float)(8u + (unsigned)mant) / 8.0f / (float)(1u << (7u - expo));
						float av = __bfloat162float(*(__nv_bfloat16 *)&hi[(uint64_t)m * K + g * 128u + e]);
						gsum += wv * av;
					}
					float sv = (float)(1u << (hs[(uint64_t)neuron * (K / 128u) + g] - 127u));
					cpu += gsum * sv;
				}
				float got = __bfloat162float(*(__nv_bfloat16 *)&ho[(uint64_t)m * N + neuron]);
				double rel = cpu != 0.0f ? (got - cpu) / (cpu < 0 ? -cpu : cpu) : got;
				if ( rel < 0 ) rel = -rel;
				if ( rel > worst ) worst = rel;
			}
		{
			float cpu0 = 0.0f;
			for ( uint32_t g = 0u; g < K / 128u; g++ )
			{
				float gsum = 0.0f;
				for ( uint32_t e = 0u; e < 128u; e++ )
				{
					uint8_t b = hp[(uint64_t)e];
					int sign = (b & 0x80) ? -1 : 1;
					int expo = (b >> 3) & 0xF, mant = b & 0x7;
					float wv;
					if ( expo == 0u ) wv = (float)sign * (float)mant / 128.0f;
					else if ( expo >= 8u ) wv = (float)sign * (float)(8u + mant) / 8.0f * (float)(1u << (expo - 7u));
					else wv = (float)sign * (float)(8u + mant) / 8.0f / (float)(1u << (7u - expo));
					gsum += wv * __bfloat162float(*(__nv_bfloat16 *)&hi[e]);
				}
				cpu0 += gsum * (float)(1u << (hs[g] - 127u));
			}
			uint16_t ho0,hr0;
			CHECK(cudaMemcpy(&ho0,out,2u,cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&hr0,ref,2u,cudaMemcpyDeviceToHost));
			printf("n0r0: cpu=%.4f kernel=%.4f native=%.4f\n",cpu0,
				__bfloat162float(*(__nv_bfloat16 *)&ho0),
				__bfloat162float(*(__nv_bfloat16 *)&hr0));
		}
		printf("CPU-check (4 neurons x %u rows): max_rel = %.6f\n",M,worst);
		free(hp); free(hs); free(hi); free(ho);
	}
	{
		const uint32_t grid2 = (N + MRD_WARPS * 2u - 1u) / (MRD_WARPS * 2u);
		MrdFp32x2Kernel<<<grid2,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaGetLastError());
		CHECK(cudaStreamSynchronize(stream));
		for ( int i = 0; i < 8; i++ )
			MrdFp32x2Kernel<<<grid2,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaStreamSynchronize(stream));
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for ( int i = 0; i < iters; i++ )
			MrdFp32x2Kernel<<<grid2,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaStreamSynchronize(stream));
		double dt = now_s() - t0;
		printf("x2   M=%u K=%u N=%u: %.1f GB/s (%.3f ms/iter)\n",M,K,N,bytes * iters / dt / 1e9,dt / iters * 1e3);
	}
	{
		if ( M <= 10u )
			MrdFp32UnrollKernel<8u,10u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		else
			MrdFp32UnrollKernel<4u,16u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaGetLastError());
		CHECK(cudaStreamSynchronize(stream));
		for ( int i = 0; i < 8; i++ )
			if ( M <= 10u )
				MrdFp32UnrollKernel<8u,10u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
			else
				MrdFp32UnrollKernel<4u,16u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaStreamSynchronize(stream));
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for ( int i = 0; i < iters; i++ )
			if ( M <= 10u )
				MrdFp32UnrollKernel<8u,10u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
			else
				MrdFp32UnrollKernel<4u,16u><<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaStreamSynchronize(stream));
		double dt = now_s() - t0;
		printf("unrl M=%u K=%u N=%u: %.1f GB/s (%.3f ms/iter)\n",M,K,N,bytes * iters / dt / 1e9,dt / iters * 1e3);
	}
	for ( int i = 0; i < 8; i++ )
		MrdFp32Kernel<<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
	CHECK(cudaStreamSynchronize(stream));
	{
		double bytes = (double)payload_bytes + (double)scale_bytes + (double)M * K * 2u + (double)M * N * 2u;
		double t0 = now_s();
		for ( int i = 0; i < iters; i++ )
			MrdFp32Kernel<<<grid,MRD_WARPS * 32u,0,stream>>>(payload,scale,in,out,M,K,N);
		CHECK(cudaStreamSynchronize(stream));
		double dt = now_s() - t0;
		printf("fp32 M=%u K=%u N=%u: %.1f GB/s (%.3f ms/iter)\n",M,K,N,bytes * iters / dt / 1e9,dt / iters * 1e3);
	}
	return 0;
}
