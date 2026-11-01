#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "inference/kernels/diff_attn.cuh"
#include "inference/kernels/poly_norm.cuh"

static float HostBf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16;
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

static uint16_t HostFloatToBf16(float value)
{
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float Round(float value)
{
	return HostBf16ToFloat(HostFloatToBf16(value));
}

static float Sigmoid(float value)
{
	return 1.0f / (1.0f + std::exp(-value));
}

static float Clamp(float value, float limit)
{
	return std::fmin(std::fmax(value, -limit), limit);
}

static uint32_t Distance(uint16_t a, uint16_t b)
{
	const int32_t left = (a & 0x8000u) ? -(int32_t)(a & 0x7fffu) : (int32_t)a, right = (b & 0x8000u) ? -(int32_t)(b & 0x7fffu) : (int32_t)b;
	return (uint32_t)std::abs(left - right);
}

static int CheckPoly(bool routed, uint32_t rows, uint32_t width, uint32_t experts)
{
	std::vector<uint16_t> gate_up((size_t)rows * 2u * width), out((size_t)rows * width);
	std::vector<float> params((size_t)experts * 4u);
	std::vector<uint32_t> param_of_row(rows);
	uint32_t seed = 12345u;
	auto random = [&]() { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xffffu) / 32768.0f - 1.0f; };
	for ( uint16_t &value : gate_up )
		value = HostFloatToBf16(random() * 3.0f);
	for ( float &value : params )
		value = Round(random() * 2.0f);
	for ( uint32_t row = 0u; row < rows; ++row )
		param_of_row[row] = routed ? row % experts : 0u;
	uint16_t *d_gate_up, *d_out;
	float *d_params;
	uint32_t *d_rows;
	cudaMalloc(&d_gate_up, gate_up.size() * 2u);
	cudaMalloc(&d_out, out.size() * 2u);
	cudaMalloc(&d_params, params.size() * 4u);
	cudaMalloc(&d_rows, rows * 4u);
	cudaMemcpy(d_gate_up, gate_up.data(), gate_up.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_params, params.data(), params.size() * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rows, param_of_row.data(), rows * 4u, cudaMemcpyHostToDevice);
	const float clamp = 1.0e6f, bias_clamp = 0.5f, scale = 0.5f, epsilon = 1e-6f;
	cudaError_t error = routed ? LmPolyNormMulLaunch<true>(d_gate_up, d_params, d_rows, d_out, rows, width, clamp, bias_clamp, scale, epsilon, 0)
		: LmPolyNormMulLaunch<false>(d_gate_up, d_params, 0, d_out, rows, width, clamp, bias_clamp, scale, epsilon, 0);
	if ( error != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		std::printf("FAIL poly launch %s\n", cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
	cudaMemcpy(out.data(), d_out, out.size() * 2u, cudaMemcpyDeviceToHost);
	uint32_t worst = 0u, exact = 0u;
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const float *p = params.data() + 4u * param_of_row[row];
		double s2 = 0.0, s4 = 0.0, s6 = 0.0;
		for ( uint32_t i = 0u; i < width; ++i )
		{
			const float g = Clamp(HostBf16ToFloat(gate_up[(size_t)row * 2u * width + i]), clamp);
			s2 += (double)g * g;
			s4 += (double)(g * g) * (g * g);
			s6 += (double)(g * g * g) * (g * g * g);
		}
		const float d1 = std::sqrt((float)(s2 / width) + epsilon), d2 = std::sqrt((float)(s4 / width) + epsilon), d3 = std::sqrt((float)(s6 / width) + epsilon);
		const float w0 = Sigmoid(p[0]), w1 = Sigmoid(p[1]), w2 = Sigmoid(p[2]), bias = routed ? Clamp(p[3], bias_clamp) : p[3];
		for ( uint32_t i = 0u; i < width; ++i )
		{
			const float g = Clamp(HostBf16ToFloat(gate_up[(size_t)row * 2u * width + i]), clamp);
			const float u = Clamp(HostBf16ToFloat(gate_up[(size_t)row * 2u * width + width + i]), clamp);
			const float value = (w0 * (g * g * g / d3) + w1 * (g * g / d2) + w2 * (g / d1) + bias) * u;
			const uint16_t expected = routed ? HostFloatToBf16(Clamp(value, clamp) * scale) : HostFloatToBf16(Round(value) * scale);
			const uint32_t distance = Distance(expected, out[(size_t)row * width + i]);
			worst = distance > worst ? distance : worst;
			exact += distance == 0u ? 1u : 0u;
		}
	}
	std::printf("%s poly rows %u width %u: worst %u ulp, %u of %u exact\n", routed ? "routed" : "dense", rows, width, worst, exact, rows * width);
	cudaFree(d_gate_up);
	cudaFree(d_out);
	cudaFree(d_params);
	cudaFree(d_rows);
	return worst <= 1u && exact * 100ull >= (uint64_t)rows * width * 99ull ? 0 : 1;
}

static int CheckDifferential(uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim)
{
	const uint32_t heads = groups * (signal + 1u), width = groups * signal * dim;
	std::vector<uint16_t> head((size_t)rows * heads * dim), lambda((size_t)rows * groups * signal), gate((size_t)rows * width), out((size_t)rows * width);
	uint32_t seed = 777u;
	auto random = [&]() { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xffffu) / 32768.0f - 1.0f; };
	for ( uint16_t &value : head )
		value = HostFloatToBf16(random() * 2.0f);
	for ( uint16_t &value : lambda )
		value = HostFloatToBf16(random() * 4.0f);
	for ( uint16_t &value : gate )
		value = HostFloatToBf16(random() * 4.0f);
	uint16_t *d_head, *d_lambda, *d_gate, *d_out;
	cudaMalloc(&d_head, head.size() * 2u);
	cudaMalloc(&d_lambda, lambda.size() * 2u);
	cudaMalloc(&d_gate, gate.size() * 2u);
	cudaMalloc(&d_out, out.size() * 2u);
	cudaMemcpy(d_head, head.data(), head.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda, lambda.data(), lambda.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate, gate.data(), gate.size() * 2u, cudaMemcpyHostToDevice);
	if ( LmDifferentialGateLaunch(d_head, d_lambda, d_gate, d_out, rows, groups, signal, dim, 0) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		std::printf("FAIL differential launch\n");
		return 1;
	}
	cudaMemcpy(out.data(), d_out, out.size() * 2u, cudaMemcpyDeviceToHost);
	uint32_t worst = 0u, exact = 0u;
	for ( uint32_t row = 0u; row < rows; ++row )
		for ( uint32_t g = 0u; g < groups; ++g )
			for ( uint32_t s = 0u; s < signal; ++s )
				for ( uint32_t d = 0u; d < dim; ++d )
				{
					const size_t base = ((size_t)row * heads + (size_t)g * (signal + 1u)) * dim;
					const float sig = HostBf16ToFloat(head[base + (size_t)s * dim + d]), noise = HostBf16ToFloat(head[base + (size_t)signal * dim + d]);
					const float lam = Round(Sigmoid(HostBf16ToFloat(lambda[(size_t)row * groups * signal + g * signal + s])));
					const float combined = Round(sig - Round(lam * noise));
					const size_t index = (size_t)row * width + ((size_t)g * signal + s) * dim + d;
					const uint16_t expected = HostFloatToBf16(combined * Round(Sigmoid(HostBf16ToFloat(gate[index]))));
					const uint32_t distance = Distance(expected, out[index]);
					worst = distance > worst ? distance : worst;
					exact += distance == 0u ? 1u : 0u;
				}
	std::printf("differential rows %u groups %u: worst %u ulp, %u of %u exact\n", rows, groups, worst, exact, rows * width);
	cudaFree(d_head);
	cudaFree(d_lambda);
	cudaFree(d_gate);
	cudaFree(d_out);
	return worst <= 1u && exact * 1000ull >= (uint64_t)rows * width * 999ull ? 0 : 1;
}

int main()
{
	int failed = 0;
	failed |= CheckPoly(true, 37u, 1280u, 24u);
	failed |= CheckPoly(false, 5u, 12288u, 1u);
	failed |= CheckDifferential(19u, 1u, 4u, 128u);
	failed |= CheckDifferential(3u, 16u, 4u, 128u);
	std::printf(failed ? "FAIL poly_norm_diff_cuda\n" : "PASS poly_norm_diff_cuda\n");
	return failed;
}
