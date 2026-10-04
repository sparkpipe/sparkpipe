#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/sample.cuh"

#define ROWS 3u
#define VOCAB 300u
#define SHARDS 4u
#define SHARD_TOKENS 80u
#define SAMPLES 4000u

static float logits_flat[ROWS * VOCAB];
static float logits_sharded[SHARDS * ROWS * SHARD_TOKENS];
static uint32_t seed_state = 24680u;

static float NextRandom(void)
{
	seed_state = seed_state * 1664525u + 1013904223u;
	return (float)((seed_state >> 8) & 0xffffu) / 65536.0f;
}

static void Run(const float *logits, uint32_t shards, const SparkRowSampling *rules, const uint32_t *positions,
	uint32_t *tokens, float *picked, SparkSamplingLogprob *logprobs, const uint32_t *greedy_tokens = 0, const uint32_t *logprob_rows = 0)
{
	LmSampleVocab vocab;
	LmSampleRows rows;
	vocab.logits = logits;
	vocab.shard_tokens = shards == 1u ? VOCAB : SHARD_TOKENS;
	vocab.row_stride = vocab.shard_tokens;
	vocab.shard_stride = (uint64_t)ROWS * vocab.shard_tokens;
	vocab.vocabulary = VOCAB;
	rows.rules = rules;
	rows.positions = positions;
	rows.source_rows = 0;
	rows.greedy_tokens = greedy_tokens;
	rows.logprob_rows = logprob_rows;
	rows.token_out = tokens;
	rows.logit_out = picked;
	rows.logprobs_out = logprobs;
	rows.rows = ROWS;
	LM_LAUNCH((LmSampleRowsKernel<1u>), dim3(ROWS), 1u, 0, 0, vocab, rows);
}

static void Case(const char *name, float temperature, float top_p, uint32_t top_k, uint32_t logprobs)
{
	static uint32_t counts[ROWS][VOCAB];
	SparkRowSampling rules[ROWS];
	uint32_t positions[ROWS] = { 7u, 11u, 13u }, tokens[ROWS], tokens_flat[ROWS], row, sample, index, equal = 1u;
	float picked[ROWS], picked_flat[ROWS];
	SparkSamplingLogprob lp[ROWS * SPARK_SAMPLING_MAX_LOGPROBS], lp_flat[ROWS * SPARK_SAMPLING_MAX_LOGPROBS];
	memset(counts, 0, sizeof(counts));
	for (sample = 0u; sample < SAMPLES; ++sample)
	{
		for (row = 0u; row < ROWS; ++row)
		{
			rules[row] = SparkSamplingRuleWith(temperature, 1000003ull * (sample + 1u) + row, top_p, top_k, logprobs);
			if ( SparkSamplingRuleValid(&rules[row]) == 0u )
			{
				printf("FAIL invalid rule %s\n", name);
				return;
			}
		}
		memset(lp, 0xff, sizeof(lp));
		memset(lp_flat, 0xff, sizeof(lp_flat));
		Run(logits_sharded, SHARDS, rules, positions, tokens, picked, lp);
		Run(logits_flat, 1u, rules, positions, tokens_flat, picked_flat, lp_flat);
		for (row = 0u; row < ROWS; ++row)
		{
			if ( tokens[row] != tokens_flat[row] || picked[row] != picked_flat[row] )
				equal = 0u;
			if ( tokens[row] < VOCAB )
				counts[row][tokens[row]]++;
		}
		if ( logprobs != 0u && memcmp(lp, lp_flat, sizeof(SparkSamplingLogprob) * ROWS * SPARK_SAMPLING_MAX_LOGPROBS) != 0 )
			equal = 0u;
		if ( sample == 0u && logprobs != 0u )
			for (row = 0u; row < ROWS; ++row)
				for (index = 0u; index < logprobs; ++index)
					printf("LP %s %u %u %u %.9g\n", name, row, index, lp[row * SPARK_SAMPLING_MAX_LOGPROBS + index].token, (double)lp[row * SPARK_SAMPLING_MAX_LOGPROBS + index].logprob);
		if ( temperature == 0.0f )
			break;
	}
	printf("CASE %s %.9g %.9g %u %u %u\n", name, (double)temperature, (double)top_p, top_k, logprobs, temperature == 0.0f ? 1u : SAMPLES);
	for (row = 0u; row < ROWS; ++row)
		for (index = 0u; index < VOCAB; ++index)
			if ( counts[row][index] != 0u )
				printf("COUNT %s %u %u %u\n", name, row, index, counts[row][index]);
	printf("SHARD_EQUAL %s %u\n", name, equal);
}

static void ForcedGreedy(void)
{
	SparkRowSampling rules[ROWS];
	uint32_t positions[ROWS] = { 3u, 5u, 9u }, forced[ROWS] = { 17u, 50u, 299u }, remap[ROWS] = { 2u, 0u, 1u }, tokens[ROWS], row, index;
	float picked[ROWS];
	SparkSamplingLogprob lp[ROWS * SPARK_SAMPLING_MAX_LOGPROBS];
	for (row = 0u; row < ROWS; ++row)
		rules[row] = SparkSamplingRuleWith(0.0f, 0u, 1.0f, 0u, 4u);
	rules[2] = SparkSamplingRuleWith(0.9f, 77u, 1.0f, 5u, 4u);
	memset(lp, 0xff, sizeof(lp));
	Run(logits_sharded, SHARDS, rules, positions, tokens, picked, lp, forced, remap);
	for (row = 0u; row < ROWS; ++row)
	{
		printf("FORCED %u %u %u\n", row, forced[row], tokens[row]);
		for (index = 0u; index < 4u; ++index)
			printf("FORCED_LP %u %u %u %.9g\n", row, index, lp[remap[row] * SPARK_SAMPLING_MAX_LOGPROBS + index].token, (double)lp[remap[row] * SPARK_SAMPLING_MAX_LOGPROBS + index].logprob);
	}
}

int main(void)
{
	uint32_t row, token, shard;
	float value;
	for (row = 0u; row < ROWS; ++row)
		for (token = 0u; token < VOCAB; ++token)
		{
			value = row == 0u ? 6.0f * NextRandom() : row == 1u ? 12.0f * NextRandom() * NextRandom() : 2.0f * NextRandom();
			logits_flat[row * VOCAB + token] = value;
		}
	logits_flat[1u * VOCAB + 50u] = 13.0f;
	logits_flat[1u * VOCAB + 250u] = 13.0f;
	for (shard = 0u; shard < SHARDS; ++shard)
		for (row = 0u; row < ROWS; ++row)
			for (token = 0u; token < SHARD_TOKENS; ++token)
				logits_sharded[(shard * ROWS + row) * SHARD_TOKENS + token] =
					shard * SHARD_TOKENS + token < VOCAB ? logits_flat[row * VOCAB + shard * SHARD_TOKENS + token] : -INFINITY;
	for (row = 0u; row < ROWS; ++row)
		for (token = 0u; token < VOCAB; ++token)
			printf("LOGIT %u %u %.9g\n", row, token, (double)logits_flat[row * VOCAB + token]);
	Case("greedy_logprobs", 0.0f, 1.0f, 0u, 6u);
	Case("temperature", 0.8f, 1.0f, 0u, 0u);
	Case("top_k", 1.0f, 1.0f, 7u, 0u);
	Case("top_p", 0.7f, 0.6f, 0u, 0u);
	Case("top_k_top_p", 1.3f, 0.8f, 40u, 3u);
	Case("logprobs_max", 1.0f, 1.0f, 0u, SPARK_SAMPLING_MAX_LOGPROBS);
	ForcedGreedy();
	printf("DONE\n");
	return 0;
}
