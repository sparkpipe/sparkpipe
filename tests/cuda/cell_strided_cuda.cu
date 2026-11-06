#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/gemm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/moe_experts.cuh"

#define CHECK(call) do { cudaError_t status_ = (call); if ( status_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(status_)); exit(1); } } while (0)
#define LAUNCH(call) do { int32_t s_ = (call); if ( s_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #call, s_); exit(1); } } while (0)

static int failures;

static uint64_t Hash(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static uint16_t Bf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, 4);
	return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float FromBf16(uint16_t v)
{
	const uint32_t u = (uint32_t)v << 16;
	float f;
	memcpy(&f, &u, 4);
	return f;
}

static float Elapsed(cudaEvent_t a, cudaEvent_t b)
{
	float ms;
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventElapsedTime(&ms, a, b));
	return ms;
}

template<uint32_t Codec>
static void Run(const char *name, uint32_t experts_total, uint32_t local, uint32_t top_k, uint32_t hidden, uint32_t inter, uint32_t tokens, uint32_t sms)
{
	using Format = typename LmWeightCodec<Codec>::Format;
	const uint32_t routes = tokens * top_k;
	const uint64_t up_bytes = (uint64_t)local * inter * hidden / 2u, down_bytes = (uint64_t)local * hidden * inter / 2u;
	const uint64_t global_bytes = (Codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 ? (uint64_t)local * sizeof(float) : 0u);
	const uint64_t scale_bytes = global_bytes + (uint64_t)local * inter * hidden / Format::kScaleGroup;
	std::vector<uint8_t> payload(up_bytes), scales(scale_bytes);
	for ( uint64_t i = 0; i < up_bytes; ++i ) payload[i] = (uint8_t)Hash(11u + i);
	for ( uint64_t i = 0; i < scale_bytes; ++i ) scales[i] = Codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 ? (uint8_t)(40u + Hash(77u + i) % 60u) : (uint8_t)(118u + Hash(77u + i) % 9u);
	if ( global_bytes != 0u )
		for ( uint32_t g = 0; g < local; ++g )
			((float *)scales.data())[g] = 0.5f + (float)(Hash(3u + g) % 1000u) / 1000.0f;
	uint8_t *weight, *scale;
	CHECK(cudaMalloc(&weight, up_bytes + down_bytes));
	CHECK(cudaMalloc(&scale, scale_bytes));
	CHECK(cudaMemcpy(weight, payload.data(), up_bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(weight + up_bytes, payload.data(), down_bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(scale, scales.data(), scale_bytes, cudaMemcpyHostToDevice));
	const LmScaleTensor up_scale = LmWeightCodecScaleTensor<Codec>(scale, local, inter, hidden);
	const LmScaleTensor down_scale = LmWeightCodecScaleTensor<Codec>(scale, local, hidden, inter);
	std::vector<uint32_t> experts(routes);
	for ( uint32_t t = 0; t < tokens; ++t )
		for ( uint32_t j = 0; j < top_k; ++j )
			experts[t * top_k + j] = (uint32_t)((Hash(1000u + t * 31u + j) % (experts_total / top_k)) * top_k + j) % experts_total;
	std::vector<uint16_t> act((uint64_t)tokens * hidden), mid((uint64_t)routes * inter);
	for ( uint64_t i = 0; i < act.size(); ++i ) act[i] = Bf16((float)((int64_t)(Hash(5u + i) % 2001u) - 1000) / 400.0f);
	for ( uint64_t i = 0; i < mid.size(); ++i ) mid[i] = Bf16((float)((int64_t)(Hash(9u + i) % 2001u) - 1000) / 400.0f);
	uint32_t *route_expert, *route_global, *offsets, *packed_row, *source, *prefix_up, *prefix_down;
	float *route_weight;
	uint16_t *d_act, *d_mid, *out_a, *out_b;
	CHECK(cudaMalloc(&route_expert, routes * 4u));
	CHECK(cudaMalloc(&route_global, routes * 4u));
	CHECK(cudaMalloc(&route_weight, routes * 4u));
	CHECK(cudaMalloc(&offsets, 4096u * 4u));
	CHECK(cudaMalloc(&packed_row, routes * 4u));
	CHECK(cudaMalloc(&source, routes * 4u));
	CHECK(cudaMalloc(&prefix_up, 4096u * 4u));
	CHECK(cudaMalloc(&prefix_down, 4096u * 4u));
	CHECK(cudaMalloc(&d_act, act.size() * 2u));
	CHECK(cudaMalloc(&d_mid, mid.size() * 2u));
	CHECK(cudaMalloc(&out_a, (uint64_t)routes * hidden * 2u));
	CHECK(cudaMalloc(&out_b, (uint64_t)routes * hidden * 2u));
	CHECK(cudaMemcpy(route_expert, experts.data(), routes * 4u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_act, act.data(), act.size() * 2u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mid, mid.data(), mid.size() * 2u, cudaMemcpyHostToDevice));
	CHECK(cudaMemset(route_weight, 0, routes * 4u));
	LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u>>>(route_expert, route_weight, route_global, routes, 0u, local);
	LAUNCH((LmRouteBuild<256u, 1025u>(route_expert, tokens, routes, top_k, offsets, packed_row, source, inter, hidden, 128u, prefix_up, prefix_down, 0)));
	CHECK(cudaDeviceSynchronize());
	std::vector<uint32_t> off(local + 1u);
	CHECK(cudaMemcpy(off.data(), offsets, (local + 1u) * 4u, cudaMemcpyDeviceToHost));
	const uint32_t rows = off[local];
	for ( uint32_t down = 0u; down < 2u; ++down )
	{
		const uint32_t width = down != 0u ? hidden : inter;
		LmGemmArguments gemm;
		cudaEvent_t e[3];
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = down != 0u ? down_scale : up_scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = offsets;
		gemm.group_tile_prefix = down != 0u ? prefix_down : prefix_up;
		gemm.output_bf16 = out_a;
		for ( int i = 0; i < 3; ++i ) CHECK(cudaEventCreate(&e[i]));
		CHECK(cudaMemset(out_a, 0, (uint64_t)routes * width * 2u));
		CHECK(cudaMemset(out_b, 0, (uint64_t)routes * width * 2u));
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaEventRecord(e[0]));
		if ( down != 0u )
			LAUNCH((LmGemmWeightOnlyLaunch<Format,128u,2u,8u>(&gemm, d_mid, weight + up_bytes, routes, tokens, top_k, local, inter, hidden, sms, true, 0)));
		else
		{
			gemm.source_row_map = source;
			gemm.source_row_count = tokens;
			LAUNCH((LmGemmWeightOnlyIndirectLaunch<Format,128u,2u,8u>(&gemm, d_act, weight, routes, tokens, top_k, local, hidden, inter, sms, 0)));
		}
		CHECK(cudaEventRecord(e[1]));
		LAUNCH(LmCellMmaStridedExperts<Format>(down != 0u ? (const void *)(weight + up_bytes) : (const void *)weight, down != 0u ? down_scale : up_scale, down != 0u ? d_mid : d_act,
			out_b, offsets, down != 0u ? (const uint32_t *)0 : source, local, routes, down, down != 0u ? inter : hidden, width, 0));
		CHECK(cudaEventRecord(e[2]));
		const float ms_gemm = Elapsed(e[0], e[1]), ms_strided = Elapsed(e[1], e[2]);
		const uint64_t elements = (uint64_t)rows * width;
		std::vector<uint16_t> a(elements), b(elements);
		CHECK(cudaMemcpy(a.data(), out_a, elements * 2u, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b.data(), out_b, elements * 2u, cudaMemcpyDeviceToHost));
		double worst = 0.0, peak = 0.0, square = 0.0, difference = 0.0;
		uint64_t bad = 0u;
		for ( uint64_t i = 0; i < elements; ++i )
		{
			const double x = FromBf16(a[i]), y = FromBf16(b[i]);
			bad += !isfinite(x) || !isfinite(y);
			peak = fmax(peak, fabs(x));
			worst = fmax(worst, fabs(x - y));
			square += x * x;
			difference += (x - y) * (x - y);
		}
		const double rms = sqrt(difference / (square + 1e-30));
		const int ok = bad == 0u && rms < 8e-3;
		failures += !ok;
		printf("%s %s %s tokens=%u local_rows=%u %ux%u: rel rms %.2e worst %.3g of %.3g nonfinite %llu | gemm %.2f ms strided %.2f ms %.1fx\n", ok ? "PASS" : "FAIL", name, down != 0u ? "down" : "up",
			tokens, rows, width, down != 0u ? inter : hidden, rms, worst, peak, (unsigned long long)bad, ms_gemm, ms_strided, ms_gemm / ms_strided);
	}
	cudaFree(weight); cudaFree(scale); cudaFree(route_expert); cudaFree(route_global); cudaFree(route_weight); cudaFree(offsets); cudaFree(packed_row); cudaFree(source);
	cudaFree(prefix_up); cudaFree(prefix_down); cudaFree(d_act); cudaFree(d_mid); cudaFree(out_a); cudaFree(out_b);
}

int main(void)
{
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	const uint32_t sms = (uint32_t)prop.multiProcessorCount;
	const uint32_t tokens[] = {48u, 256u, 2048u};
	for ( uint32_t i = 0; i < 3u; ++i )
	{
		Run<SPARK_WEIGHT_CODEC_MXFP4_E2M1>("mxfp4", 128u, 8u, 6u, 1024u, 768u, tokens[i], sms);
		Run<SPARK_WEIGHT_CODEC_NVFP4_E2M1>("nvfp4", 128u, 8u, 6u, 1024u, 768u, tokens[i], sms);
	}
	Run<SPARK_WEIGHT_CODEC_MXFP4_E2M1>("mxfp4-wide", 512u, 32u, 10u, 8192u, 2048u, 2048u, sms);
	Run<SPARK_WEIGHT_CODEC_NVFP4_E2M1>("nvfp4-wide", 512u, 32u, 10u, 8192u, 2048u, 2048u, sms);
	printf("%s strided cell experts match the verbatim GEMM failures=%d\n", failures == 0 ? "PASS" : "FAIL", failures);
	return failures != 0;
}
