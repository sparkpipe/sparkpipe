#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "sparkpipe/spark_lm_kernels.cuh"

static int probe_failures;

static uint64_t ProbeMix(uint64_t key)
{
	uint64_t z = key * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static float ProbeFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

static float ProbeUniform(uint64_t key)
{
	return (float)((double)(ProbeMix(key) >> 11u) / 9007199254740992.0 * 2.0 - 1.0);
}

template<typename T> static T *ProbeAllocate(uint64_t count)
{
	T *pointer = 0;
	if ( cudaMallocManaged(&pointer, count * sizeof(T)) != cudaSuccess || cudaMemset(pointer, 0, count * sizeof(T)) != cudaSuccess ||
		cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "allocation of %llu elements failed\n", (unsigned long long)count);
		exit(2);
	}
	return pointer;
}

static void ProbeShape(uint32_t vocabulary, uint32_t hidden, uint32_t rows, uint32_t seed)
{
	uint16_t *head = ProbeAllocate<uint16_t>((uint64_t)vocabulary * hidden);
	uint16_t *input = ProbeAllocate<uint16_t>((uint64_t)rows * hidden);
	uint8_t *payload = ProbeAllocate<uint8_t>((uint64_t)vocabulary * hidden);
	float *scale = ProbeAllocate<float>((uint64_t)vocabulary * hidden / 32u);
	float *norm = ProbeAllocate<float>((uint64_t)vocabulary * hidden / 32u);
	uint8_t *scratch = ProbeAllocate<uint8_t>(SparkHeadCertifiedFp8ScratchBytes(vocabulary, hidden) * rows);
	uint32_t *candidates = ProbeAllocate<uint32_t>((uint64_t)vocabulary * rows);
	uint32_t *counts = ProbeAllocate<uint32_t>(rows);
	uint32_t *tokens = ProbeAllocate<uint32_t>(rows);
	float *scores = ProbeAllocate<float>(rows);
	const uint32_t offset = 1000u;
	for ( uint64_t index = 0u; index < (uint64_t)vocabulary * hidden; ++index )
		head[index] = ProbeBf16(ProbeUniform(((uint64_t)seed << 40u) + index) * 0.05f);
	for ( uint64_t index = 0u; index < (uint64_t)rows * hidden; ++index )
		input[index] = ProbeBf16(ProbeUniform(((uint64_t)(seed + 7u) << 40u) + index) * 4.0f);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const uint32_t planted = (seed * 977u + row * 4099u + 3u) % vocabulary;
		const uint32_t rival = (planted + vocabulary / 2u + row) % vocabulary;
		for ( uint32_t element = 0u; element < hidden; ++element )
		{
			const float sign = ProbeFloat(input[(uint64_t)row * hidden + element]) >= 0.0f ? 1.0f : -1.0f;
			if ( row % 2u == 0u && element >= hidden - 256u )
				head[(uint64_t)planted * hidden + element] = ProbeBf16(sign * 0.25f);
			if ( row % 2u == 1u )
				head[(uint64_t)rival * hidden + element] = ProbeBf16(ProbeFloat(head[(uint64_t)rival * hidden + element]) + sign * 0.002f);
		}
	}
	if ( SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head, payload, scale, norm, vocabulary, hidden) != cudaSuccess ||
		SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(0, input, head, payload, scale, norm, scratch, candidates, counts, tokens, scores,
			offset, rows, vocabulary, hidden) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "FAIL vocab=%u hidden=%u rows=%u: certified head launch failed\n", vocabulary, hidden, rows);
		++probe_failures;
		return;
	}
	std::vector<uint32_t> rows_tokens(tokens, tokens + rows);
	std::vector<float> rows_scores(scores, scores + rows);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		double best = -1.0e300, second = -1.0e300;
		uint32_t best_token = 0u;
		for ( uint32_t candidate = 0u; candidate < vocabulary; ++candidate )
		{
			double total = 0.0;
			for ( uint32_t element = 0u; element < hidden; ++element )
				total += (double)ProbeFloat(input[(uint64_t)row * hidden + element]) * (double)ProbeFloat(head[(uint64_t)candidate * hidden + element]);
			if ( total > best )
			{
				second = best;
				best = total;
				best_token = candidate;
			}
			else if ( total > second )
				second = total;
		}
		if ( SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(0, input + (uint64_t)row * hidden, head, payload, scale, norm, scratch, candidates,
				counts, tokens, scores, offset, 1u, vocabulary, hidden) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
		{
			fprintf(stderr, "FAIL vocab=%u hidden=%u row=%u: B1 certified head launch failed\n", vocabulary, hidden, row);
			++probe_failures;
			continue;
		}
		const bool ok = rows_tokens[row] == best_token + offset && tokens[0] == best_token + offset &&
			fabs((double)rows_scores[row] - best) <= 1.0e-4 * fmax(1.0, fabs(best)) && fabs((double)scores[0] - best) <= 1.0e-4 * fmax(1.0, fabs(best));
		printf("%s vocab=%u hidden=%u row=%u exact=%u margin=%.6f rows_path=%u B1=%u score=%.6f candidates=%u\n", ok ? "ok" : "FAIL",
			vocabulary, hidden, row, best_token + offset, best - second, rows_tokens[row], tokens[0], rows_scores[row], counts[0]);
		if ( !ok )
			++probe_failures;
	}
	cudaFree(head);
	cudaFree(input);
	cudaFree(payload);
	cudaFree(scale);
	cudaFree(norm);
	cudaFree(scratch);
	cudaFree(candidates);
	cudaFree(counts);
	cudaFree(tokens);
	cudaFree(scores);
}

static void ProbeOutputVocabulary(uint32_t vocabulary, uint32_t valid, uint32_t hidden, uint32_t rows, uint32_t seed)
{
	uint16_t *head = ProbeAllocate<uint16_t>((uint64_t)vocabulary * hidden);
	uint16_t *input = ProbeAllocate<uint16_t>((uint64_t)rows * hidden);
	uint8_t *payload = ProbeAllocate<uint8_t>((uint64_t)vocabulary * hidden);
	float *scale = ProbeAllocate<float>((uint64_t)vocabulary * hidden / 32u);
	float *norm = ProbeAllocate<float>((uint64_t)vocabulary * hidden / 32u);
	uint8_t *scratch = ProbeAllocate<uint8_t>(SparkHeadCertifiedFp8ScratchBytes(vocabulary, hidden) * rows);
	uint32_t *candidates = ProbeAllocate<uint32_t>((uint64_t)vocabulary * rows);
	uint32_t *counts = ProbeAllocate<uint32_t>(rows);
	uint32_t *tokens = ProbeAllocate<uint32_t>(rows);
	float *scores = ProbeAllocate<float>(rows);
	const uint32_t offset = 7000u;
	for ( uint64_t index = 0u; index < (uint64_t)vocabulary * hidden; ++index )
		head[index] = ProbeBf16(ProbeUniform(((uint64_t)seed << 40u) + index) * 0.05f);
	for ( uint64_t index = 0u; index < (uint64_t)rows * hidden; ++index )
		input[index] = ProbeBf16(ProbeUniform(((uint64_t)(seed + 7u) << 40u) + index) * 4.0f);
	for ( uint32_t candidate = valid; candidate < vocabulary; ++candidate )
		for ( uint32_t element = 0u; element < hidden; ++element )
			head[(uint64_t)candidate * hidden + element] = ProbeBf16(ProbeFloat(input[element]) >= 0.0f ? 0.5f : -0.5f);
	if ( SparkLmHostLaunchHeadCertifiedFp8Quantize(0, head, payload, scale, norm, vocabulary, hidden) != cudaSuccess ||
		SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(0, input, head, payload, scale, norm, scratch, candidates, counts, tokens, scores,
			offset, rows, vocabulary, hidden) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess || tokens[0] < valid + offset )
	{
		fprintf(stderr, "FAIL vocab=%u valid=%u: the planted rows past the output vocabulary do not win the unclipped head (token %u)\n", vocabulary, valid, tokens[0]);
		++probe_failures;
		return;
	}
	if ( SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(0, input, head, payload, scale, norm, scratch, candidates, counts, tokens, scores,
			offset, rows, valid, hidden) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "FAIL vocab=%u valid=%u: clipped certified head launch failed\n", vocabulary, valid);
		++probe_failures;
		return;
	}
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		double best = -1.0e300;
		uint32_t best_token = 0u;
		for ( uint32_t candidate = 0u; candidate < valid; ++candidate )
		{
			double total = 0.0;
			for ( uint32_t element = 0u; element < hidden; ++element )
				total += (double)ProbeFloat(input[(uint64_t)row * hidden + element]) * (double)ProbeFloat(head[(uint64_t)candidate * hidden + element]);
			if ( total > best )
			{
				best = total;
				best_token = candidate;
			}
		}
		const bool ok = tokens[row] == best_token + offset && fabs((double)scores[row] - best) <= 1.0e-4 * fmax(1.0, fabs(best));
		printf("%s vocab=%u valid=%u row=%u exact=%u clipped=%u\n", ok ? "ok" : "FAIL", vocabulary, valid, row, best_token + offset, tokens[row]);
		if ( !ok )
			++probe_failures;
	}
	cudaFree(head);
	cudaFree(input);
	cudaFree(payload);
	cudaFree(scale);
	cudaFree(norm);
	cudaFree(scratch);
	cudaFree(candidates);
	cudaFree(counts);
	cudaFree(tokens);
	cudaFree(scores);
}

int main(void)
{
	ProbeShape(4096u, 5120u, 3u, 1u);
	ProbeShape(4096u, 5376u, 10u, 2u);
	ProbeShape(4096u, 2816u, 9u, 3u);
	ProbeShape(3000u, 6656u, 4u, 4u);
	ProbeShape(16384u, 5376u, 2u, 5u);
	ProbeShape(2048u, 768u, 5u, 6u);
	ProbeOutputVocabulary(9680u, 9656u, 5120u, 3u, 7u);
	ProbeOutputVocabulary(9536u, 8635u, 2816u, 4u, 8u);
	if ( probe_failures != 0 )
	{
		printf("FAIL head_certified_cuda: %d failures\n", probe_failures);
		return 1;
	}
	printf("PASS head_certified_cuda\n");
	return 0;
}
