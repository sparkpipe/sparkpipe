#include "inference/kernels/cell_mx.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t state = 2463534242u;

static uint32_t Next(void)
{
	state ^= state << 13u;
	state ^= state >> 17u;
	state ^= state << 5u;
	return state;
}

static double E2m1(uint32_t nibble)
{
	static const double magnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
	return (nibble & 8u) != 0u ? -magnitude[nibble & 7u] : magnitude[nibble & 7u];
}

static double E4m3(uint32_t code)
{
	const uint32_t exponent = (code >> 3u) & 15u, mantissa = code & 7u;
	const double magnitude = exponent == 0u ? std::ldexp((double)mantissa, -9) : std::ldexp(1.0 + mantissa / 8.0, (int)exponent - 7);
	return (code & 0x80u) != 0u ? -magnitude : magnitude;
}

static uint16_t Bf16(double value)
{
	float narrow = (float)value;
	uint32_t bits;
	memcpy(&bits, &narrow, 4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static double FromBf16(uint16_t bits)
{
	uint32_t wide = (uint32_t)bits << 16u;
	float value;
	memcpy(&value, &wide, 4u);
	return value;
}

static uint64_t CellByte(uint32_t neuron, uint32_t k)
{
	const uint32_t half = neuron / 8u, g = neuron % 8u, c = k / 8u;
	return half * 128u + (g * 4u + c) * 4u + (k % 8u) / 2u;
}

static int Run(uint32_t groups, uint32_t tokens, uint32_t per_token, uint32_t input, uint32_t output, uint32_t packed_input)
{
	const uint32_t tiles = input / 32u, cells = output / 16u, packed = tokens * per_token;
	std::vector<uint8_t> weight((uint64_t)groups * tiles * cells * 272u);
	std::vector<double> dense((uint64_t)groups * output * input);
	for ( uint32_t group = 0u; group < groups; group++ )
		for ( uint32_t t = 0u; t < tiles; t++ )
			for ( uint32_t cell = 0u; cell < cells; cell++ )
			{
				uint8_t *base = &weight[(((uint64_t)group * tiles + t) * cells + cell) * 272u];
				for ( uint32_t neuron = 0u; neuron < 16u; neuron++ )
				{
					const uint32_t scale = 122u + Next() % 9u;
					base[256u + neuron] = (uint8_t)scale;
					for ( uint32_t k = 0u; k < 32u; k++ )
					{
						const uint32_t nibble = Next() & 15u;
						base[CellByte(neuron, k)] |= (uint8_t)(nibble << ((k & 1u) * 4u));
						dense[((uint64_t)group * output + cell * 16u + neuron) * input + t * 32u + k] = E2m1(nibble) * std::ldexp(1.0, (int)scale - 127);
					}
				}
			}
	const uint32_t rows = packed_input != 0u ? packed : tokens;
	std::vector<uint8_t> codes((uint64_t)rows * input), scales((uint64_t)rows * tiles);
	std::vector<uint16_t> wide((uint64_t)rows * input);
	std::vector<double> value((uint64_t)rows * input);
	for ( uint32_t row = 0u; row < rows; row++ )
		for ( uint32_t t = 0u; t < tiles; t++ )
		{
			const uint32_t scale = 118u + Next() % 17u;
			scales[(uint64_t)row * tiles + t] = (uint8_t)scale;
			for ( uint32_t k = 0u; k < 32u; k++ )
			{
				uint32_t code = Next() & 0xffu;
				if ( (code & 0x7fu) == 0x7fu )
					code &= 0xfeu;
				codes[(uint64_t)row * input + t * 32u + k] = (uint8_t)code;
				value[(uint64_t)row * input + t * 32u + k] = E4m3(code) * std::ldexp(1.0, (int)scale - 127);
				wide[(uint64_t)row * input + t * 32u + k] = Bf16(value[(uint64_t)row * input + t * 32u + k]);
			}
		}
	std::vector<uint32_t> offset(groups + 1u, 0u), source(packed), owner(packed);
	std::vector<std::vector<uint32_t>> lists(groups);
	for ( uint32_t token = 0u; token < tokens; token++ )
		for ( uint32_t route = 0u; route < per_token; route++ )
			lists[(token * 7u + route * 3u + token / 5u) % groups].push_back(token);
	for ( uint32_t group = 0u, at = 0u; group < groups; group++ )
	{
		offset[group] = at;
		for ( uint32_t token : lists[group] )
		{
			source[at] = token;
			owner[at] = group;
			at++;
		}
		offset[group + 1u] = at;
	}
	uint8_t *d_weight, *d_codes, *d_scales;
	uint16_t *d_wide, *d_mx, *d_bf;
	uint32_t *d_offset, *d_source;
	cudaMalloc(&d_weight, weight.size());
	cudaMalloc(&d_codes, codes.size());
	cudaMalloc(&d_scales, scales.size());
	cudaMalloc(&d_wide, wide.size() * 2u);
	cudaMalloc(&d_mx, (uint64_t)packed * output * 2u);
	cudaMalloc(&d_bf, (uint64_t)packed * output * 2u);
	cudaMalloc(&d_offset, offset.size() * 4u);
	cudaMalloc(&d_source, source.size() * 4u);
	cudaMemcpy(d_weight, weight.data(), weight.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wide, wide.data(), wide.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, offset.data(), offset.size() * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_source, source.data(), source.size() * 4u, cudaMemcpyHostToDevice);
	const int32_t mx = LmCellMxExperts(d_weight, d_codes, d_scales, d_mx, d_offset, packed_input != 0u ? (const uint32_t *)0 : d_source, groups, packed,
		packed_input, input, output, 0);
	const int32_t bf = LmCellMmaExperts(d_weight, d_wide, d_bf, d_offset, packed_input != 0u ? (const uint32_t *)0 : d_source, groups, packed, packed_input,
		input, output, LM_SKINNY_CELL_TILE_K, 0);
	const cudaError_t error = cudaDeviceSynchronize();
	if ( mx != LM_LAUNCH_OK || bf != LM_LAUNCH_OK || error != cudaSuccess )
	{
		printf("FAIL launch mx=%d bf16=%d cuda=%s\n", mx, bf, cudaGetErrorString(error));
		return 1;
	}
	std::vector<uint16_t> out_mx((uint64_t)packed * output), out_bf((uint64_t)packed * output);
	cudaMemcpy(out_mx.data(), d_mx, out_mx.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_bf.data(), d_bf, out_bf.size() * 2u, cudaMemcpyDeviceToHost);
	double worst_mx = 0.0, worst_bf = 0.0;
	uint64_t same = 0u;
	for ( uint32_t row = 0u; row < packed; row++ )
	{
		const uint32_t input_row = packed_input != 0u ? row : source[row];
		for ( uint32_t neuron = 0u; neuron < output; neuron++ )
		{
			double sum = 0.0, magnitude = 0.0;
			for ( uint32_t k = 0u; k < input; k++ )
			{
				const double product = value[(uint64_t)input_row * input + k] * dense[((uint64_t)owner[row] * output + neuron) * input + k];
				sum += product;
				magnitude += std::fabs(product);
			}
			const double ulp = std::ldexp(1.0, (int)std::floor(std::log2(std::fabs(sum) > 1e-30 ? std::fabs(sum) : 1e-30)) - 7);
			const double allowed = ulp + magnitude * std::ldexp(1.0, -18);
			const double error_mx = std::fabs(FromBf16(out_mx[(uint64_t)row * output + neuron]) - sum) / allowed;
			const double error_bf = std::fabs(FromBf16(out_bf[(uint64_t)row * output + neuron]) - sum) / allowed;
			worst_mx = error_mx > worst_mx ? error_mx : worst_mx;
			worst_bf = error_bf > worst_bf ? error_bf : worst_bf;
			same += out_mx[(uint64_t)row * output + neuron] == out_bf[(uint64_t)row * output + neuron] ? 1u : 0u;
		}
	}
	printf("%s groups %u rows %u input %u output %u: worst error mx %.3f bf16 %.3f (units of one BF16 ulp plus magnitude x 2^-18), same bits %.4f%%\n",
		packed_input != 0u ? "packed" : "gathered", groups, packed, input, output, worst_mx, worst_bf, 100.0 * (double)same / (double)out_mx.size());
	cudaFree(d_weight);
	cudaFree(d_codes);
	cudaFree(d_scales);
	cudaFree(d_wide);
	cudaFree(d_mx);
	cudaFree(d_bf);
	cudaFree(d_offset);
	cudaFree(d_source);
	return worst_mx <= 1.0 && worst_bf <= 1.0 ? 0 : 1;
}

int main(void)
{
	int failures = 0;
	failures += Run(8u, 40u, 3u, 512u, 256u, 0u);
	failures += Run(16u, 300u, 6u, 1024u, 512u, 0u);
	failures += Run(8u, 50u, 4u, 2048u, 384u, 1u);
	failures += Run(4u, 3u, 2u, 256u, 128u, 0u);
	if ( failures != 0 )
	{
		printf("FAIL cell_mx_cuda: %d of 4 cases\n", failures);
		return 1;
	}
	printf("PASS cell_mx_cuda: the fp8 by fp4 block-scaled expert kernel and the BF16 expert kernel stay within one BF16 ulp of fp64 on gathered and packed rows\n");
	return 0;
}
