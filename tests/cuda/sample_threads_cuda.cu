#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/launch.h"
#include "inference/kernels/sample.cuh"

#define PROBE_THREADS 1024u
#define PROBE_VOCAB 163840u
#define PROBE_SHARDS 16u
#define PROBE_SLICE (PROBE_VOCAB / PROBE_SHARDS)
#define PROBE_ROWS 7u
#define PROBE_SEEDS 6u

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

typedef struct ProbeRule
{
	float temperature;
	float top_p;
	uint32_t top_k;
	uint32_t logprobs;
	const char *name;
}
ProbeRule;

static const ProbeRule probe_rules[] = {
	{ 1.0f, 0.95f, 0u, 0u, "temperature 1 top-p 0.95" },
	{ 1.0f, 0.95f, 0u, SPARK_SAMPLING_MAX_LOGPROBS, "temperature 1 top-p 0.95 with every logprob" },
	{ 0.6f, 0.9f, 40u, 5u, "temperature 0.6 top-p 0.9 top-k 40 with 5 logprobs" },
	{ 1.0f, 1.0f, 0u, 0u, "temperature 1 untruncated" },
	{ 0.0f, 1.0f, 0u, SPARK_SAMPLING_MAX_LOGPROBS, "greedy with every logprob" },
	{ 0.3f, 0.5f, 1000u, 3u, "temperature 0.3 top-p 0.5 top-k 1000 with 3 logprobs" },
};

static uint64_t probe_state = 0x9e3779b97f4a7c15ull;

static double ProbeUniform(void)
{
	probe_state ^= probe_state << 13u;
	probe_state ^= probe_state >> 7u;
	probe_state ^= probe_state << 17u;
	return ((double)(probe_state >> 11u) + 0.5) * 0x1p-53;
}

static double ProbeNormal(void)
{
	return sqrt(-2.0 * log(ProbeUniform())) * cos(6.283185307179586 * ProbeUniform());
}

static float ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
	memcpy(&value, &bits, 4u);
	return value;
}

static float ProbeLogit(uint32_t shape, uint32_t token)
{
	switch ( shape )
	{
	case 0u: return (float)(ProbeNormal() * 2.5);
	case 1u: return 0.0f;
	case 2u: return ProbeBf16((float)(ProbeNormal() * 3.0));
	case 3u: return ProbeUniform() < 0.3 ? -INFINITY : (float)(ProbeNormal() * 2.5);
	case 4u: return token == 777u || token == 150000u ? 10.0f : (float)(ProbeNormal() * 1.5 - 5.0);
	case 5u: return (float)(ProbeNormal() * 8.0);
	default: return token == 12345u ? 25.0f : (float)ProbeNormal();
	}
}

static size_t ProbeIndex(uint32_t row, uint32_t token)
{
	return ((size_t)(token / PROBE_SLICE) * PROBE_ROWS + row) * PROBE_SLICE + token % PROBE_SLICE;
}

static void ProbeLaunch(uint32_t threads, const LmSampleVocab &vocab, const LmSampleRows &rows)
{
	if ( threads == 1u )
		LM_LAUNCH((LmSampleRowsKernel<1u>), rows.rows, 1u, 0, 0, vocab, rows);
	else
		LM_LAUNCH((LmSampleRowsKernel<PROBE_THREADS>), rows.rows, PROBE_THREADS, 0, 0, vocab, rows);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaDeviceSynchronize());
}

int main(void)
{
	const size_t logit_count = (size_t)PROBE_SHARDS * PROBE_ROWS * PROBE_SLICE;
	float *host_logits = (float *)malloc(logit_count * sizeof(float)), *logits;
	SparkRowSampling host_rules[PROBE_ROWS], *rules;
	uint32_t host_positions[PROBE_ROWS], *positions, *tokens[2], host_tokens[2][PROBE_ROWS];
	float *picked[2], host_picked[2][PROBE_ROWS];
	SparkSamplingLogprob *logprobs[2], host_logprobs[2][PROBE_ROWS * SPARK_SAMPLING_MAX_LOGPROBS];
	uint32_t row, token, rule, seed, side, index, failures = 0u, checks = 0u;
	double lse[PROBE_ROWS];
	LmSampleVocab vocab;
	LmSampleRows rows[2];
	for (row = 0u; row < PROBE_ROWS; ++row)
	{
		double maximum = -INFINITY, sum = 0.0;
		for (token = 0u; token < PROBE_VOCAB; ++token)
			host_logits[ProbeIndex(row, token)] = ProbeLogit(row, token);
		for (token = 0u; token < PROBE_VOCAB; ++token)
			maximum = fmax(maximum, (double)host_logits[ProbeIndex(row, token)]);
		for (token = 0u; token < PROBE_VOCAB; ++token)
			sum += exp((double)host_logits[ProbeIndex(row, token)] - maximum);
		lse[row] = maximum + log(sum);
	}
	PROBE_CUDA(cudaMalloc(&logits, logit_count * sizeof(float)));
	PROBE_CUDA(cudaMemcpy(logits, host_logits, logit_count * sizeof(float), cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMalloc(&rules, sizeof(host_rules)));
	PROBE_CUDA(cudaMalloc(&positions, sizeof(host_positions)));
	vocab.logits = logits;
	vocab.shard_stride = (uint64_t)PROBE_ROWS * PROBE_SLICE;
	vocab.row_stride = PROBE_SLICE;
	vocab.shard_tokens = PROBE_SLICE;
	vocab.vocabulary = PROBE_VOCAB;
	for (side = 0u; side < 2u; ++side)
	{
		PROBE_CUDA(cudaMalloc(&tokens[side], sizeof(host_tokens[side])));
		PROBE_CUDA(cudaMalloc(&picked[side], sizeof(host_picked[side])));
		PROBE_CUDA(cudaMalloc(&logprobs[side], sizeof(host_logprobs[side])));
		memset(&rows[side], 0, sizeof(rows[side]));
		rows[side].rules = rules;
		rows[side].positions = positions;
		rows[side].token_out = tokens[side];
		rows[side].logit_out = picked[side];
		rows[side].logprobs_out = logprobs[side];
		rows[side].rows = PROBE_ROWS;
	}
	for (rule = 0u; rule < sizeof(probe_rules) / sizeof(probe_rules[0]); ++rule)
		for (seed = 0u; seed < PROBE_SEEDS; ++seed)
		{
			for (row = 0u; row < PROBE_ROWS; ++row)
			{
				host_rules[row] = SparkSamplingRuleWith(probe_rules[rule].temperature, 1000003ull * (seed + 1u) + 7919ull * row + rule,
					probe_rules[rule].top_p, probe_rules[rule].top_k, probe_rules[rule].logprobs);
				host_positions[row] = 137000u + 3u * seed + row;
			}
			PROBE_CUDA(cudaMemcpy(rules, host_rules, sizeof(host_rules), cudaMemcpyHostToDevice));
			PROBE_CUDA(cudaMemcpy(positions, host_positions, sizeof(host_positions), cudaMemcpyHostToDevice));
			for (side = 0u; side < 2u; ++side)
			{
				PROBE_CUDA(cudaMemset(logprobs[side], 0x5a, sizeof(host_logprobs[side])));
				ProbeLaunch(side == 0u ? 1u : PROBE_THREADS, vocab, rows[side]);
				PROBE_CUDA(cudaMemcpy(host_tokens[side], tokens[side], sizeof(host_tokens[side]), cudaMemcpyDeviceToHost));
				PROBE_CUDA(cudaMemcpy(host_picked[side], picked[side], sizeof(host_picked[side]), cudaMemcpyDeviceToHost));
				PROBE_CUDA(cudaMemcpy(host_logprobs[side], logprobs[side], sizeof(host_logprobs[side]), cudaMemcpyDeviceToHost));
			}
			for (row = 0u; row < PROBE_ROWS; ++row)
			{
				uint32_t mismatch = host_tokens[0][row] != host_tokens[1][row] || memcmp(&host_picked[0][row], &host_picked[1][row], sizeof(float)) != 0 ? 1u : 0u;
				for (index = 0u; index < probe_rules[rule].logprobs; ++index)
				{
					const SparkSamplingLogprob *reference = &host_logprobs[0][row * SPARK_SAMPLING_MAX_LOGPROBS + index];
					const SparkSamplingLogprob *wide = &host_logprobs[1][row * SPARK_SAMPLING_MAX_LOGPROBS + index];
					if ( reference->token != wide->token )
						mismatch = 1u;
					else if ( wide->token == SPARK_SAMPLING_NO_TOKEN ? wide->logprob != 0.0f :
						wide->token >= PROBE_VOCAB || fabs((double)wide->logprob - ((double)host_logits[ProbeIndex(row, wide->token)] - lse[row])) > 1e-4 )
						mismatch = 1u;
				}
				checks++;
				if ( mismatch != 0u )
				{
					failures++;
					if ( failures <= 8u )
						printf("FAIL %s, seed %u, logit shape %u: one thread chose %u, %u threads chose %u\n", probe_rules[rule].name, seed, row,
							host_tokens[0][row], PROBE_THREADS, host_tokens[1][row]);
				}
			}
		}
	if ( failures != 0u )
	{
		printf("FAIL sampling threads: %u of %u rows differ between one thread and %u threads or miss the double-precision log-softmax\n", failures, checks, PROBE_THREADS);
		return 1;
	}
	printf("PASS sampling threads: %u rows over %u logit shapes (ties, flat, bf16 levels, -inf, twin peaks, wide, one spike) and %u rules choose the same token, "
		"logit and logprob tokens with %u threads as with one thread, and the logprobs match a double-precision log-softmax within 1e-4\n", checks, PROBE_ROWS, (uint32_t)(sizeof(probe_rules) / sizeof(probe_rules[0])), PROBE_THREADS);
	return 0;
}
