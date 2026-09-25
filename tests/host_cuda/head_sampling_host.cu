#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1
#include "inference/kernels/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/head.cuh"
#define SPARK_FAMILY(name) HostSampling##name
#include "sparkpipe/family/glm/spark_glm_head_maxloc.cuh"

#define HOST_HIDDEN 64u
#define HOST_VOCABULARY 1000u
#define HOST_SHARD 500u
#define HOST_TILE 128u
#define HOST_MAX_ROWS 20u
#define HOST_THREADS 256u
#define HOST_ROWS_PER_BLOCK 16u
#define HOST_DRAWS 20000u
#define HOST_DISTRIBUTION_TOKENS 6u
#define HOST_CHI_SQUARE_LIMIT 30.0

typedef struct HostHead
{
	uint16_t normed[HOST_MAX_ROWS * HOST_HIDDEN],weight[HOST_VOCABULARY * HOST_HIDDEN];
	SparkRowSampling rules[HOST_MAX_ROWS];
	uint32_t positions[HOST_MAX_ROWS];
}
HostHead;

static uint32_t host_state = 90210u;

static float HostSigned(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return (float)((int32_t)(host_state >> 20u) - 2048) / 2048.0f;
}

static void HostFail(const char *what, uint32_t rows, uint32_t row, uint32_t a, uint32_t b)
{
	fprintf(stderr,"FAIL %s rows=%u row=%u token=%u/%u\n",what,rows,row,a,b);
	exit(1);
}

static void HostRun(uint32_t kernel, const HostHead *head, uint32_t rows, uint32_t first_token, uint32_t vocabulary, uint32_t *token, float *score)
{
	static float candidate_score[HOST_MAX_ROWS * HOST_VOCABULARY];
	static uint32_t candidate_token[HOST_MAX_ROWS * HOST_VOCABULARY];
	const uint16_t *weight = head->weight + (uint64_t)first_token * HOST_HIDDEN;
	const LmHeadSampling sampling = {head->rules, head->positions, rows, first_token};
	const uint32_t tiles = (vocabulary + HOST_TILE - 1u) / HOST_TILE;
	if ( kernel == 0u )
		LM_LAUNCH((LmHeadCandidateKernel<HOST_THREADS,HOST_TILE>),dim3(tiles,rows),HOST_THREADS,0,0,head->normed,weight,(const uint32_t *)0,candidate_score,candidate_token,rows,HOST_HIDDEN,vocabulary);
	else if ( kernel == 1u )
		LM_LAUNCH((LmHeadCandidateRowsKernel<HOST_THREADS,HOST_TILE,HOST_ROWS_PER_BLOCK>),dim3(tiles,(rows + HOST_ROWS_PER_BLOCK - 1u) / HOST_ROWS_PER_BLOCK),HOST_THREADS,0,0,head->normed,weight,(const uint32_t *)0,candidate_score,candidate_token,rows,HOST_HIDDEN,vocabulary);
	else if ( kernel == 2u )
		LM_LAUNCH((LmHeadSampledCandidateKernel<HOST_THREADS,HOST_TILE>),dim3(tiles,rows),HOST_THREADS,0,0,head->normed,weight,candidate_score,candidate_token,HOST_HIDDEN,vocabulary,sampling);
	else
		LM_LAUNCH((LmHeadSampledCandidateRowsKernel<HOST_THREADS,HOST_TILE,HOST_ROWS_PER_BLOCK>),dim3(tiles,(rows + HOST_ROWS_PER_BLOCK - 1u) / HOST_ROWS_PER_BLOCK),HOST_THREADS,0,0,head->normed,weight,candidate_score,candidate_token,rows,HOST_HIDDEN,vocabulary,sampling);
	LM_LAUNCH((LmHeadCommitKernel<HOST_THREADS>),dim3(rows),HOST_THREADS,0,0,candidate_score,candidate_token,tiles,token,score,rows);
}

static void HostFill(HostHead *head, uint32_t rows, uint32_t sampled)
{
	uint32_t index,row;
	for (index=0u; index<rows * HOST_HIDDEN; index++)
		head->normed[index] = LmFloatToBf16(HostSigned());
	for (index=0u; index<HOST_VOCABULARY * HOST_HIDDEN; index++)
		head->weight[index] = LmFloatToBf16(HostSigned() * 0.25f);
	for (row=0u; row<rows; row++)
	{
		head->rules[row] = SparkSamplingRule(sampled != 0u && row % 3u != 1u ? 0.5f + 0.25f * (float)(row % 4u) : 0.0f,1000u + row / 2u);
		head->positions[row] = 17u + row * 5u;
	}
}

static uint32_t HostSame(const uint32_t *token_a, const float *score_a, const uint32_t *token_b, const float *score_b, uint32_t rows)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
		if ( token_a[row] != token_b[row] || memcmp(&score_a[row],&score_b[row],sizeof(float)) != 0 )
			return(row);
	return(UINT32_MAX);
}

static void HostGreedyEquivalence(uint32_t rows)
{
	static HostHead head;
	uint32_t token_a[HOST_MAX_ROWS],token_b[HOST_MAX_ROWS],kernel,row;
	float score_a[HOST_MAX_ROWS],score_b[HOST_MAX_ROWS];
	HostFill(&head,rows,0u);
	for (kernel=0u; kernel<2u; kernel++)
	{
		HostRun(kernel,&head,rows,0u,HOST_VOCABULARY,token_a,score_a);
		HostRun(kernel + 2u,&head,rows,0u,HOST_VOCABULARY,token_b,score_b);
		row = HostSame(token_a,score_a,token_b,score_b,rows);
		if ( row != UINT32_MAX )
			HostFail("greedy rules change the greedy result",rows,row,token_a[row],token_b[row]);
	}
}

static uint32_t HostReference(const HostHead *head, uint32_t row)
{
	uint32_t token,element,best_token = 0u;
	float total,value,best = -INFINITY;
	for (token=0u; token<HOST_VOCABULARY; token++)
	{
		total = 0.0f;
		for (element=0u; element<HOST_HIDDEN; element++)
			total += LmBf16ToFloat(head->normed[row * HOST_HIDDEN + element]) * LmBf16ToFloat(head->weight[token * HOST_HIDDEN + element]);
		value = head->rules[row].inverse_temperature == 0.0f ? total : fmaf(total,head->rules[row].inverse_temperature,LmGumbelNoise(head->rules[row].seed,head->positions[row],token));
		if ( value > best )
			best = value, best_token = token;
	}
	return(best_token);
}

static uint64_t HostKey(float score, uint32_t token)
{
	return(((uint64_t)HostSamplingOrderedHeadScore(score) << 32u) | (UINT32_MAX - token));
}

static void HostSampledInvariance(uint32_t rows)
{
	static HostHead head;
	uint32_t single[HOST_MAX_ROWS],batched[HOST_MAX_ROWS],again[HOST_MAX_ROWS],low[HOST_MAX_ROWS],high[HOST_MAX_ROWS],greedy[HOST_MAX_ROWS],row,moved = 0u;
	float single_score[HOST_MAX_ROWS],batched_score[HOST_MAX_ROWS],again_score[HOST_MAX_ROWS],low_score[HOST_MAX_ROWS],high_score[HOST_MAX_ROWS],greedy_score[HOST_MAX_ROWS];
	HostFill(&head,rows,1u);
	HostRun(2u,&head,rows,0u,HOST_VOCABULARY,single,single_score);
	HostRun(3u,&head,rows,0u,HOST_VOCABULARY,batched,batched_score);
	HostRun(3u,&head,rows,0u,HOST_VOCABULARY,again,again_score);
	HostRun(3u,&head,rows,0u,HOST_SHARD,low,low_score);
	HostRun(3u,&head,rows,HOST_SHARD,HOST_VOCABULARY - HOST_SHARD,high,high_score);
	HostRun(1u,&head,rows,0u,HOST_VOCABULARY,greedy,greedy_score);
	for (row=0u; row<rows; row++)
	{
		if ( single[row] != batched[row] || memcmp(&single_score[row],&batched_score[row],sizeof(float)) != 0 )
			HostFail("single-row and rows kernels disagree",rows,row,single[row],batched[row]);
		if ( batched[row] != again[row] )
			HostFail("sampling is not deterministic",rows,row,batched[row],again[row]);
		if ( batched[row] != HostReference(&head,row) )
			HostFail("kernel sample differs from argmax(logit / T + gumbel(seed, position, token))",rows,row,batched[row],HostReference(&head,row));
		if ( (HostKey(low_score[row],low[row]) > HostKey(high_score[row],high[row] + HOST_SHARD) ? low[row] : high[row] + HOST_SHARD) != batched[row] )
			HostFail("vocabulary sharding changes the sample",rows,row,batched[row],low[row]);
		if ( head.rules[row].inverse_temperature == 0.0f && (batched[row] != greedy[row] || memcmp(&batched_score[row],&greedy_score[row],sizeof(float)) != 0) )
			HostFail("a greedy row inside a sampled batch moved",rows,row,batched[row],greedy[row]);
		moved += head.rules[row].inverse_temperature != 0.0f && batched[row] != greedy[row] ? 1u : 0u;
	}
	if ( rows >= 8u && moved == 0u )
		HostFail("no sampled row left the argmax",rows,0u,0u,0u);
}

static void HostDistribution(float temperature, uint64_t seed)
{
	static const float logits[HOST_DISTRIBUTION_TOKENS] = {2.0f,1.25f,0.5f,0.0f,-0.75f,-1.5f};
	const float inverse_temperature = 1.0f / temperature;
	uint32_t counts[HOST_DISTRIBUTION_TOKENS] = {0},draw,token,best_token;
	double expected,total = 0.0,chi_square = 0.0,probability[HOST_DISTRIBUTION_TOKENS];
	float best,value;
	for (token=0u; token<HOST_DISTRIBUTION_TOKENS; token++)
		total += probability[token] = exp((double)logits[token] * inverse_temperature);
	for (draw=0u; draw<HOST_DRAWS; draw++)
	{
		best = -INFINITY;
		best_token = 0u;
		for (token=0u; token<HOST_DISTRIBUTION_TOKENS; token++)
		{
			value = fmaf(logits[token],inverse_temperature,LmGumbelNoise(seed,draw,token));
			if ( value > best )
				best = value, best_token = token;
		}
		counts[best_token]++;
	}
	for (token=0u; token<HOST_DISTRIBUTION_TOKENS; token++)
	{
		expected = HOST_DRAWS * probability[token] / total;
		chi_square += (counts[token] - expected) * (counts[token] - expected) / expected;
	}
	printf("head sampling distribution T=%.2f seed=%llu chi_square=%.2f counts=%u,%u,%u,%u,%u,%u\n",temperature,(unsigned long long)seed,chi_square,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5]);
	if ( !(chi_square < HOST_CHI_SQUARE_LIMIT) )
		HostFail("sample frequencies do not follow softmax(logits / T)",0u,0u,0u,0u);
}

int main(void)
{
	HostGreedyEquivalence(1u);
	HostGreedyEquivalence(9u);
	HostSampledInvariance(1u);
	HostSampledInvariance(5u);
	HostSampledInvariance(HOST_MAX_ROWS);
	HostDistribution(1.0f,7u);
	HostDistribution(0.5f,8u);
	HostDistribution(2.0f,9u);
	puts("head sampling host: greedy rules bitwise greedy; single-row, rows and sharded runs agree; frequencies follow softmax");
	return(0);
}
