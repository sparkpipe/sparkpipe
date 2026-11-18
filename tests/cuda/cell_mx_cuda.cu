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

static uint8_t E4m3Code(double value)
{
	for ( uint32_t code = 0u; code < 256u; code++ )
		if ( (code & 0x7fu) != 0x7fu && E4m3(code) == value && (value != 0.0 || code == (std::signbit(value) ? 0x80u : 0u)) )
			return (uint8_t)code;
	return 0x7fu;
}

static __global__ void PublisherFp4GemmKernel(const uint8_t *codes, const uint8_t *code_scales, const uint8_t *weight_codes, const uint8_t *weight_scales,
	const uint32_t *offset, const uint32_t *source, uint32_t input, uint32_t output, uint16_t *out)
{
	const uint32_t group = blockIdx.z, lane = threadIdx.x, g = lane / 4u, t = lane % 4u, blocks = input / 32u;
	const uint32_t first = offset[group] + blockIdx.y * 16u, end = offset[group + 1u], column = blockIdx.x * 8u;
	float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	if ( first >= end )
		return;
	for ( uint32_t block = 0u; block < blocks; block++ )
	{
		uint32_t a[4], b[2];
		float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		for ( uint32_t r = 0u; r < 4u; r++ )
		{
			const uint32_t row = first + g + 8u * (r & 1u), k = block * 32u + t * 4u + 16u * (r >> 1u);
			a[r] = row < end ? *(const uint32_t *)(codes + (uint64_t)(source != 0 ? source[row] : row) * input + k) : 0u;
		}
		for ( uint32_t r = 0u; r < 2u; r++ )
			b[r] = *(const uint32_t *)(weight_codes + ((uint64_t)group * output + column + g) * input + block * 32u + t * 4u + 16u * r);
		LmMmaE4m3(part, a, b);
		for ( uint32_t e = 0u; e < 4u; e++ )
		{
			const uint32_t row = first + g + 8u * (e >> 1u), neuron = column + t * 2u + (e & 1u);
			const float scale_a = row < end ? __uint_as_float((uint32_t)code_scales[(uint64_t)(source != 0 ? source[row] : row) * blocks + block] << 23u) : 0.0f;
			const float scale_b = __uint_as_float((uint32_t)weight_scales[((uint64_t)group * output + neuron) * blocks + block] << 23u);
			total[e] += part[e] * scale_a * scale_b;
		}
	}
	for ( uint32_t e = 0u; e < 4u; e++ )
	{
		const uint32_t row = first + g + 8u * (e >> 1u), neuron = column + t * 2u + (e & 1u);
		if ( row < end )
			out[(uint64_t)row * output + neuron] = LmFloatToBf16(total[e]);
	}
}

static int Run(uint32_t groups, uint32_t tokens, uint32_t per_token, uint32_t input, uint32_t output, uint32_t packed_input)
{
	const uint32_t tiles = input / 32u, cells = output / 16u, packed = tokens * per_token;
	std::vector<uint8_t> weight((uint64_t)groups * tiles * cells * 272u), weight_codes((uint64_t)groups * output * input),
		weight_scales((uint64_t)groups * output * tiles);
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
					weight_scales[((uint64_t)group * output + cell * 16u + neuron) * tiles + t] = (uint8_t)scale;
					for ( uint32_t k = 0u; k < 32u; k++ )
					{
						const uint32_t nibble = Next() & 15u;
						base[CellByte(neuron, k)] |= (uint8_t)(nibble << ((k & 1u) * 4u));
						weight_codes[((uint64_t)group * output + cell * 16u + neuron) * input + t * 32u + k] = E4m3Code(E2m1(nibble));
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
	uint8_t *d_weight, *d_codes, *d_scales, *d_weight_codes, *d_weight_scales;
	uint16_t *d_wide, *d_mx, *d_bf, *d_publisher;
	uint32_t *d_offset, *d_source;
	cudaMalloc(&d_weight, weight.size());
	cudaMalloc(&d_codes, codes.size());
	cudaMalloc(&d_scales, scales.size());
	cudaMalloc(&d_wide, wide.size() * 2u);
	cudaMalloc(&d_mx, (uint64_t)packed * output * 2u);
	cudaMalloc(&d_bf, (uint64_t)packed * output * 2u);
	cudaMalloc(&d_publisher, (uint64_t)packed * output * 2u);
	cudaMalloc(&d_weight_codes, weight_codes.size());
	cudaMalloc(&d_weight_scales, weight_scales.size());
	cudaMemcpy(d_weight_codes, weight_codes.data(), weight_codes.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_scales, weight_scales.data(), weight_scales.size(), cudaMemcpyHostToDevice);
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
	uint32_t longest = 0u;
	for ( uint32_t group = 0u; group < groups; group++ )
		longest = offset[group + 1u] - offset[group] > longest ? offset[group + 1u] - offset[group] : longest;
	PublisherFp4GemmKernel<<<dim3(output / 8u, (longest + 15u) / 16u, groups), 32u>>>(d_codes, d_scales, d_weight_codes, d_weight_scales, d_offset,
		packed_input != 0u ? (const uint32_t *)0 : d_source, input, output, d_publisher);
	const cudaError_t error = cudaDeviceSynchronize();
	if ( mx != LM_LAUNCH_OK || bf != LM_LAUNCH_OK || error != cudaSuccess )
	{
		printf("FAIL launch mx=%d bf16=%d cuda=%s\n", mx, bf, cudaGetErrorString(error));
		return 1;
	}
	std::vector<uint16_t> out_mx((uint64_t)packed * output), out_bf((uint64_t)packed * output), out_publisher((uint64_t)packed * output);
	cudaMemcpy(out_mx.data(), d_mx, out_mx.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_bf.data(), d_bf, out_bf.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_publisher.data(), d_publisher, out_publisher.size() * 2u, cudaMemcpyDeviceToHost);
	const uint32_t split = groups / 2u;
	const uint64_t group_bytes = (uint64_t)(input / 32u) * (output / 16u) * 272u;
	uint8_t *d_upper = 0;
	cudaMalloc(&d_upper, (groups - split) * group_bytes);
	cudaMemcpy(d_upper, d_weight + split * group_bytes, (groups - split) * group_bytes, cudaMemcpyDeviceToDevice);
	cudaMemset(d_weight + split * group_bytes, 0x5a, (groups - split) * group_bytes);
	cudaMemset(d_mx, 0, out_mx.size() * 2u);
	const int32_t split_launch = LmCellMxExpertsSplit(d_weight, d_upper, split, d_codes, d_scales, d_mx, d_offset,
		packed_input != 0u ? (const uint32_t *)0 : d_source, groups, packed, packed_input, input, output, 0);
	std::vector<uint16_t> out_split((uint64_t)packed * output);
	if ( split_launch != LM_LAUNCH_OK || cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("FAIL split launch %d\n", split_launch);
		return 1;
	}
	cudaMemcpy(out_split.data(), d_mx, out_split.size() * 2u, cudaMemcpyDeviceToHost);
	cudaFree(d_upper);
	uint64_t mx_differ = 0u, bf_differ = 0u, split_differ = 0u;
	for ( uint64_t index = 0u; index < out_publisher.size(); index++ )
	{
		mx_differ += out_mx[index] != out_publisher[index] ? 1u : 0u;
		bf_differ += out_bf[index] != out_publisher[index] ? 1u : 0u;
		split_differ += out_split[index] != out_publisher[index] ? 1u : 0u;
	}
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
	printf("%s groups %u rows %u input %u output %u: worst error mx %.3f bf16 %.3f (units of one BF16 ulp plus magnitude x 2^-18), same bits %.4f%%; "
		"against the publisher's fp4 GEMM (each 32-block an fp8 MMA from zero, scaled, added in FP32) mx differs in %llu, bf16 in %llu of %llu; "
		"with groups from %u on read from a separate copy, %llu differ\n",
		packed_input != 0u ? "packed" : "gathered", groups, packed, input, output, worst_mx, worst_bf, 100.0 * (double)same / (double)out_mx.size(),
		(unsigned long long)mx_differ, (unsigned long long)bf_differ, (unsigned long long)out_publisher.size(), split, (unsigned long long)split_differ);
	cudaFree(d_weight);
	cudaFree(d_codes);
	cudaFree(d_scales);
	cudaFree(d_wide);
	cudaFree(d_mx);
	cudaFree(d_bf);
	cudaFree(d_publisher);
	cudaFree(d_weight_codes);
	cudaFree(d_weight_scales);
	cudaFree(d_offset);
	cudaFree(d_source);
	return worst_mx <= 1.0 && worst_bf <= 1.0 && mx_differ == 0u && split_differ == 0u ? 0 : 1;
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
	printf("PASS cell_mx_cuda: the fp8 by fp4 block-scaled expert kernel gives the publisher's fp4 GEMM bits, and both expert kernels stay within one BF16 ulp of fp64 on gathered and packed rows, and gives the same bits when the groups past a split are read from a separate copy\n");
	return 0;
}
