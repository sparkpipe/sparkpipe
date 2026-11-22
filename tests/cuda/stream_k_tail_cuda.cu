#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "inference/kernels/stream_gemm.cuh"

#define TAIL_CUDA(call) do { cudaError_t tail_error = (call); if ( tail_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(tail_error)); exit(1); } } while (0)
#define TAIL_SAMPLES 256u

static uint32_t failures, multiprocessors, state = 20261122u;
static uint64_t order_samples, order_sensitive;

static float Random(void)
{
	state ^= state << 13u;
	state ^= state >> 17u;
	state ^= state << 5u;
	return (float)(state >> 8u) * 0x1p-23f - 1.0f;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static double FromBf16(uint16_t value)
{
	const uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

template<class T>
static T *Upload(const std::vector<T> &host)
{
	T *device;
	TAIL_CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	TAIL_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template<class T>
static std::vector<T> Download(const T *device, size_t count)
{
	std::vector<T> host(count);
	TAIL_CUDA(cudaDeviceSynchronize());
	TAIL_CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

static uint32_t Padded(uint32_t input)
{
	return (input + 63u) / 64u * 64u;
}

static std::vector<uint16_t> Matrix(uint32_t rows, uint32_t input, float scale, uint32_t zero_row)
{
	std::vector<uint16_t> values((size_t)rows * input);
	for ( uint32_t row = 0u; row < rows; row++ )
		for ( uint32_t k = 0u; k < input; k++ )
			values[(size_t)row * input + k] = row == zero_row ? 0u : Bf16(scale * Random());
	return values;
}

static std::vector<uint16_t> Pad(const std::vector<uint16_t> &values, uint32_t rows, uint32_t input)
{
	const uint32_t padded = Padded(input);
	std::vector<uint16_t> result((size_t)rows * padded, 0u);
	for ( uint32_t row = 0u; row < rows; row++ )
		memcpy(&result[(size_t)row * padded], &values[(size_t)row * input], (size_t)input * sizeof(uint16_t));
	return result;
}

static void Reference(const char *name, const std::vector<uint16_t> &weight, const std::vector<uint16_t> &activation, const std::vector<uint16_t> &output,
	const std::vector<uint32_t> &group_of_row, const std::vector<uint32_t> &source_of_row, uint32_t rows, uint32_t input, uint32_t output_dimension)
{
	double worst = 0.0;
	for ( uint32_t sample = 0u; sample < TAIL_SAMPLES; sample++ )
	{
		const uint32_t row = (uint32_t)(((uint64_t)sample * 2654435761u) % rows), column = (uint32_t)(((uint64_t)sample * 40503u + 7u) % output_dimension);
		const uint32_t group = group_of_row.empty() ? 0u : group_of_row[row], source = source_of_row.empty() ? row : source_of_row[row];
		double sum = 0.0, magnitude = 0.0;
		float forward = 0.0f, backward = 0.0f;
		for ( uint32_t k = 0u; k < input; k++ )
		{
			const double product = FromBf16(weight[((size_t)group * output_dimension + column) * input + k]) * FromBf16(activation[(size_t)source * input + k]);
			const double mirrored = FromBf16(weight[((size_t)group * output_dimension + column) * input + input - 1u - k]) *
				FromBf16(activation[(size_t)source * input + input - 1u - k]);
			sum += product;
			magnitude += fabs(product);
			forward += (float)product;
			backward += (float)mirrored;
		}
		order_samples++;
		order_sensitive += memcmp(&forward, &backward, sizeof(float)) != 0 ? 1u : 0u;
		const double got = FromBf16(output[(size_t)row * output_dimension + column]);
		const double bound = fabs(sum) * 0x1p-8 + magnitude * input * 0x1p-24 + 0x1p-60;
		if ( fabs(got - sum) > bound )
		{
			printf("FAIL %s: row %u column %u gives %.9g against the fp64 reference %.9g (bound %.3g)\n", name, row, column, got, sum, bound);
			failures++;
			return;
		}
		worst = fabs(got - sum) / bound > worst ? fabs(got - sum) / bound : worst;
	}
	printf("PASS %s: %u sampled outputs within the bf16 rounding and fp32 accumulation bound of the fp64 reference (worst %.2f of the bound)\n", name,
		TAIL_SAMPLES, worst);
}

template<class Format>
static void Dense(const char *format, uint32_t rows, uint32_t input, uint32_t output)
{
	char name[160];
	const uint32_t padded = Padded(input);
	const std::vector<uint16_t> weight = Matrix(output, input, 0.0625f, 3u), activation = Matrix(rows, input, 4.0f, rows / 2u);
	uint16_t *weight_device = Upload(weight), *activation_device = Upload(activation), *weight_padded = Upload(Pad(weight, output, input)),
		*activation_padded = Upload(Pad(activation, rows, input)), *tail_bf16, *padded_bf16;
	float *tail_f32, *padded_f32;
	snprintf(name, sizeof(name), "%s dense K%u (padded %u) N%u rows%u", format, input, padded, output, rows);
	TAIL_CUDA(cudaMalloc((void **)&tail_bf16, (size_t)rows * output * sizeof(uint16_t)));
	TAIL_CUDA(cudaMalloc((void **)&padded_bf16, (size_t)rows * output * sizeof(uint16_t)));
	TAIL_CUDA(cudaMalloc((void **)&tail_f32, (size_t)rows * output * sizeof(float)));
	TAIL_CUDA(cudaMalloc((void **)&padded_f32, (size_t)rows * output * sizeof(float)));
	TAIL_CUDA(cudaMemset(tail_bf16, 0xff, (size_t)rows * output * sizeof(uint16_t)));
	TAIL_CUDA(cudaMemset(tail_f32, 0xff, (size_t)rows * output * sizeof(float)));
	const int32_t statuses[4] = {
		LmStreamGemmDense<Format>(weight_device, LmScaleTensorNone(), activation_device, tail_bf16, (float *)0, rows, input, output, 0u, 0u, multiprocessors, 0),
		LmStreamGemmDense<Format>(weight_padded, LmScaleTensorNone(), activation_padded, padded_bf16, (float *)0, rows, padded, output, 0u, 0u, multiprocessors, 0),
		LmStreamGemmDense<Format>(weight_device, LmScaleTensorNone(), activation_device, (uint16_t *)0, tail_f32, rows, input, output, 0u, 0u, multiprocessors, 0),
		LmStreamGemmDense<Format>(weight_padded, LmScaleTensorNone(), activation_padded, (uint16_t *)0, padded_f32, rows, padded, output, 0u, 0u, multiprocessors, 0)};
	if ( statuses[0] != LM_LAUNCH_OK || statuses[1] != LM_LAUNCH_OK || statuses[2] != LM_LAUNCH_OK || statuses[3] != LM_LAUNCH_OK )
	{
		printf("FAIL %s: launches returned %d %d %d %d\n", name, statuses[0], statuses[1], statuses[2], statuses[3]);
		failures++;
	}
	else
	{
		const std::vector<uint16_t> tail = Download(tail_bf16, (size_t)rows * output), reference = Download(padded_bf16, (size_t)rows * output);
		const std::vector<float> tail_wide = Download(tail_f32, (size_t)rows * output), reference_wide = Download(padded_f32, (size_t)rows * output);
		size_t differ = 0u;
		for ( size_t index = 0u; index < tail.size(); index++ )
			differ += tail[index] != reference[index] || memcmp(&tail_wide[index], &reference_wide[index], sizeof(float)) != 0 ? 1u : 0u;
		if ( differ != 0u )
		{
			printf("FAIL %s: %zu of %zu outputs differ from the zero-padded wave in bf16 or fp32\n", name, differ, tail.size());
			failures++;
		}
		else
			printf("PASS %s: bf16 and fp32 outputs bit-identical to the zero-padded wave over %zu outputs\n", name, tail.size());
		Reference(name, weight, activation, tail, std::vector<uint32_t>(), std::vector<uint32_t>(), rows, input, output);
	}
	TAIL_CUDA(cudaFree(weight_device));
	TAIL_CUDA(cudaFree(activation_device));
	TAIL_CUDA(cudaFree(weight_padded));
	TAIL_CUDA(cudaFree(activation_padded));
	TAIL_CUDA(cudaFree(tail_bf16));
	TAIL_CUDA(cudaFree(padded_bf16));
	TAIL_CUDA(cudaFree(tail_f32));
	TAIL_CUDA(cudaFree(padded_f32));
}

template<class Format>
static void Grouped(const char *format, uint32_t tokens, uint32_t groups, uint32_t input, uint32_t output)
{
	char name[160];
	const uint32_t padded = Padded(input);
	std::vector<uint32_t> offsets(groups + 1u, 0u), group_of_row, source_of_row;
	for ( uint32_t group = 0u; group < groups; group++ )
	{
		const uint32_t count = group % 3u == 1u ? 0u : 1u + (uint32_t)((uint64_t)group * 977u % 151u);
		offsets[group + 1u] = offsets[group] + count;
		for ( uint32_t index = 0u; index < count; index++ )
		{
			group_of_row.push_back(group);
			source_of_row.push_back((uint32_t)(((uint64_t)group_of_row.size() * 7919u) % tokens));
		}
	}
	const uint32_t packed = offsets[groups];
	const std::vector<uint16_t> weight = Matrix(groups * output, input, 0.0625f, 5u), activation = Matrix(tokens, input, 4.0f, 1u);
	uint16_t *weight_device = Upload(weight), *activation_device = Upload(activation), *weight_padded = Upload(Pad(weight, groups * output, input)),
		*activation_padded = Upload(Pad(activation, tokens, input)), *tail_out, *padded_out;
	uint32_t *offsets_device = Upload(offsets), *sources_device = Upload(source_of_row);
	snprintf(name, sizeof(name), "%s grouped K%u (padded %u) N%u groups%u packed rows%u", format, input, padded, output, groups, packed);
	TAIL_CUDA(cudaMalloc((void **)&tail_out, (size_t)packed * output * sizeof(uint16_t)));
	TAIL_CUDA(cudaMalloc((void **)&padded_out, (size_t)packed * output * sizeof(uint16_t)));
	TAIL_CUDA(cudaMemset(tail_out, 0xff, (size_t)packed * output * sizeof(uint16_t)));
	const int32_t tail_status = LmStreamGemmGrouped<Format>(weight_device, LmScaleTensorNone(), activation_device, tail_out, offsets_device, sources_device, groups, packed,
		input, output, multiprocessors, 0);
	const int32_t padded_status = LmStreamGemmGrouped<Format>(weight_padded, LmScaleTensorNone(), activation_padded, padded_out, offsets_device, sources_device, groups,
		packed, padded, output, multiprocessors, 0);
	if ( tail_status != LM_LAUNCH_OK || padded_status != LM_LAUNCH_OK )
	{
		printf("FAIL %s: launches returned %d %d\n", name, tail_status, padded_status);
		failures++;
	}
	else
	{
		const std::vector<uint16_t> tail = Download(tail_out, (size_t)packed * output), reference = Download(padded_out, (size_t)packed * output);
		const size_t differ = (size_t)(memcmp(tail.data(), reference.data(), tail.size() * sizeof(uint16_t)) != 0 ? 1u : 0u);
		if ( differ != 0u )
		{
			size_t count = 0u;
			for ( size_t index = 0u; index < tail.size(); index++ )
				count += tail[index] != reference[index] ? 1u : 0u;
			printf("FAIL %s: %zu of %zu outputs differ from the zero-padded wave\n", name, count, tail.size());
			failures++;
		}
		else
			printf("PASS %s: outputs bit-identical to the zero-padded wave over %zu outputs\n", name, tail.size());
		Reference(name, weight, activation, tail, group_of_row, source_of_row, packed, input, output);
	}
	TAIL_CUDA(cudaFree(weight_device));
	TAIL_CUDA(cudaFree(activation_device));
	TAIL_CUDA(cudaFree(weight_padded));
	TAIL_CUDA(cudaFree(activation_padded));
	TAIL_CUDA(cudaFree(offsets_device));
	TAIL_CUDA(cudaFree(sources_device));
	TAIL_CUDA(cudaFree(tail_out));
	TAIL_CUDA(cudaFree(padded_out));
}

static void Refuses(uint32_t input)
{
	char chunk[512];
	size_t count;
	std::string text;
	uint16_t *buffer;
	TAIL_CUDA(cudaMalloc((void **)&buffer, (size_t)64u * 1024u * sizeof(uint16_t)));
	FILE *sink = tmpfile();
	fflush(stderr);
	const int saved = dup(fileno(stderr));
	dup2(fileno(sink), fileno(stderr));
	spark_last_error_site.file = 0;
	const int32_t status = LmStreamGemmDense<LmBf16Format>(buffer, LmScaleTensorNone(), buffer, buffer + 32768u, (float *)0, 32u, input, 64u, 0u, 0u, multiprocessors, 0);
	fflush(stderr);
	dup2(saved, fileno(stderr));
	close(saved);
	rewind(sink);
	while ( (count = fread(chunk, 1u, sizeof(chunk), sink)) > 0u )
		text.append(chunk, count);
	fclose(sink);
	const char *file = spark_last_error_site.file != 0 ? spark_last_error_site.file : "";
	if ( status != LM_LAUNCH_ERR_SHAPE || strstr(file, "inference/kernels/stream_gemm.cuh") == 0 || text.find(std::string("ERRSITE ") + file) == std::string::npos )
	{
		printf("FAIL an input of %u: status %d, site %s, stderr: %s\n", input, status, file, text.c_str());
		failures++;
	}
	else
		printf("PASS an unscaled input of %u, not a multiple of 8, refuses with LM_LAUNCH_ERR_SHAPE through SPARK_FAIL at %s:%u\n", input, file,
			spark_last_error_site.line);
	TAIL_CUDA(cudaFree(buffer));
}

int main(void)
{
	int device = 0, value = 0;
	TAIL_CUDA(cudaGetDevice(&device));
	TAIL_CUDA(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
	multiprocessors = (uint32_t)value;
	Dense<LmBf16Format>("bf16", 2048u, 224u, 7168u);
	Dense<LmBf16Format>("bf16", 17u, 224u, 384u);
	Dense<LmBf16Format>("bf16", 40u, 200u, 1030u);
	Dense<LmBf16Format>("bf16", 9u, 8u, 64u);
	Dense<LmBf16Format>("bf16", 100u, 456u, 512u);
	Dense<LmBf16Format>("bf16", 300u, 40u, 2050u);
	Dense<LmBf16Block32>("bf16 block-32", 300u, 224u, 1024u);
	Dense<LmBf16Block32>("bf16 block-32", 64u, 200u, 2048u);
	Dense<LmBf16Block32>("bf16 block-32", 40u, 224u, 256u);
	Grouped<LmBf16Format>("bf16", 1000u, 12u, 224u, 512u);
	Grouped<LmBf16Block32>("bf16 block-32", 600u, 9u, 200u, 640u);
	Refuses(228u);
	if ( order_sensitive * 2u < order_samples )
	{
		printf("FAIL the operands are too regular to tell summation orders apart: reversing the fp32 sum changes only %llu of %llu sampled outputs\n",
			(unsigned long long)order_sensitive, (unsigned long long)order_samples);
		failures++;
	}
	else
		printf("PASS the operands tell summation orders apart: reversing the fp32 sum changes %llu of %llu sampled outputs, so a tail that summed in "
			"another order than the zero-padded wave would differ from it\n", (unsigned long long)order_sensitive, (unsigned long long)order_samples);
	if ( failures != 0u )
	{
		printf("FAIL stream K tail: %u cases\n", failures);
		return 1;
	}
	printf("PASS stream K tail: an unscaled input that is a multiple of 8 but not of the k-tile runs with its last tile zero-filled, bit-identical to the "
		"zero-padded wave for dense and grouped waves at every tile height, and other inputs refuse\n");
	return 0;
}
