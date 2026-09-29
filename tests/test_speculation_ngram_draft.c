#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_ngram_draft.h"

#define TEST_CAPACITY 4096u
#define TEST_RANDOM_LENGTH 1500u
#define TEST_VOCAB 24u
#define TEST_MIN 2u
#define TEST_MAX 6u
#define TEST_SCAN 16u
#define TEST_DEPTH 7u

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static SparkStatus Draft(SparkSpeculationNgramDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t anchor,uint32_t depth,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest request;
	memset(&request,0,sizeof(request));
	memset(result,0,sizeof(*result));
	request.requested_token_count = depth;
	request.active_sequence_index = lane;
	request.sequence_id = sequence_id;
	request.sequence_position = anchor;
	return(SparkSpeculationNgramDraftTokens(draft,&request,result));
}

static uint32_t ReferenceNext(const uint32_t *history,uint32_t length,const uint32_t *suffix,uint32_t suffix_count,uint32_t *order_out)
{
	uint32_t seen[TEST_SCAN],counts[TEST_SCAN],order,end,scanned,distinct,index,best;
	for (order=suffix_count < TEST_MAX ? suffix_count : TEST_MAX; order>=TEST_MIN; order--)
	{
		scanned = 0u;
		distinct = 0u;
		for (end=length - 1u; end-- > order - 1u && scanned < TEST_SCAN; )
		{
			if ( memcmp(history + end + 1u - order,suffix + suffix_count - order,order * sizeof(uint32_t)) != 0 )
				continue;
			for (index=0u; index<distinct && seen[index]!=history[end + 1u]; index++)
				;
			if ( index == distinct )
			{
				seen[distinct] = history[end + 1u];
				counts[distinct++] = 0u;
			}
			counts[index]++;
			scanned++;
		}
		if ( scanned == 0u )
			continue;
		best = 0u;
		for (index=1u; index<distinct; index++)
			if ( counts[index] > counts[best] )
				best = index;
		*order_out = order;
		return(seen[best]);
	}
	*order_out = 0u;
	return(UINT32_MAX);
}

static void TestBounds(void)
{
	SparkSpeculationNgramDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[3] = {1u,2u,3u};
	Require(SparkSpeculationNgramDraftInitialize(&draft,0u,16u,2u,4u,8u) == SPARK_STATUS_INVALID_ARGUMENT,"zero lanes rejected");
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,16u,0u,4u,8u) == SPARK_STATUS_INVALID_ARGUMENT,"zero minimum rejected");
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,16u,5u,4u,8u) == SPARK_STATUS_INVALID_ARGUMENT,"minimum above maximum rejected");
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,16u,2u,SPARK_SPECULATION_NGRAM_MATCH_LIMIT + 1u,8u) == SPARK_STATUS_INVALID_ARGUMENT,"order above limit rejected");
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,16u,2u,4u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"zero scan limit rejected");
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,4u,2u,4u,8u) == SPARK_STATUS_OK,"initialize");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,7u,1u,tokens,1u) == SPARK_STATUS_INVALID_ARGUMENT,"new sequence must start at position 0");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,7u,0u,tokens,3u) == SPARK_STATUS_OK,"observe");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,7u,2u,tokens,1u) == SPARK_STATUS_INVALID_ARGUMENT,"rewrite refused");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,7u,3u,tokens,2u) == SPARK_STATUS_CAPACITY_EXCEEDED,"capacity enforced");
	Require(Draft(&draft,0u,8u,2u,4u,&result) == SPARK_STATUS_NOT_FOUND && result.token_count == 0u,"other sequence drafts nothing");
	Require(Draft(&draft,0u,7u,1u,4u,&result) == SPARK_STATUS_NOT_FOUND,"stale anchor drafts nothing");
	SparkSpeculationNgramDraftDestroy(&draft);
}

static void TestFrequencyBeatsRecency(void)
{
	SparkSpeculationNgramDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[] = {5u,6u,1u, 5u,6u,1u, 5u,6u,2u, 9u, 5u,6u};
	uint32_t count = (uint32_t)(sizeof(tokens) / sizeof(tokens[0]));
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,64u,2u,4u,8u) == SPARK_STATUS_OK,"initialize");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,3u,0u,tokens,count) == SPARK_STATUS_OK,"observe");
	Require(Draft(&draft,0u,3u,count - 1u,1u,&result) == SPARK_STATUS_OK && result.token_count == 1u,"draft");
	Require(result.token_ids[0] == 1u,"most frequent continuation wins over the most recent");
	Require(result.confidence_milli[0] == 666u,"confidence is count over scanned");
	SparkSpeculationNgramDraftDestroy(&draft);
}

static void TestTieTakesMostRecent(void)
{
	SparkSpeculationNgramDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[] = {5u,6u,1u, 5u,6u,2u, 8u, 5u,6u};
	uint32_t count = (uint32_t)(sizeof(tokens) / sizeof(tokens[0]));
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,64u,2u,4u,8u) == SPARK_STATUS_OK,"initialize");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,3u,0u,tokens,count) == SPARK_STATUS_OK,"observe");
	Require(Draft(&draft,0u,3u,count - 1u,1u,&result) == SPARK_STATUS_OK && result.token_ids[0] == 2u,"tie takes the most recent");
	SparkSpeculationNgramDraftDestroy(&draft);
}

static void TestChainExtendsWithDrafts(void)
{
	SparkSpeculationNgramDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[] = {1u,2u,3u,4u,5u,6u,7u, 9u, 1u,2u};
	uint32_t count = (uint32_t)(sizeof(tokens) / sizeof(tokens[0]));
	Require(SparkSpeculationNgramDraftInitialize(&draft,1u,64u,2u,4u,8u) == SPARK_STATUS_OK,"initialize");
	Require(SparkSpeculationNgramDraftObserve(&draft,0u,4u,0u,tokens,count) == SPARK_STATUS_OK,"observe");
	Require(Draft(&draft,0u,4u,count - 1u,7u,&result) == SPARK_STATUS_OK,"draft");
	Require(result.token_count == 7u,"chain extends through its own drafts");
	Require(result.token_ids[0] == 3u && result.token_ids[4] == 7u && result.token_ids[5] == 9u && result.token_ids[6] == 1u,"chain copies the continuation");
	SparkSpeculationNgramDraftDestroy(&draft);
}

static void TestMatchesReferenceOnRandomStream(void)
{
	SparkSpeculationNgramDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t history[TEST_RANDOM_LENGTH],suffix[TEST_MAX + TEST_DEPTH],take,anchor,depth,order,want,seed = 12345u;
	uint64_t compared = 0u;
	Require(SparkSpeculationNgramDraftInitialize(&draft,2u,TEST_CAPACITY,TEST_MIN,TEST_MAX,TEST_SCAN) == SPARK_STATUS_OK,"initialize");
	for (anchor=0u; anchor<TEST_RANDOM_LENGTH; anchor++)
	{
		seed = seed * 1664525u + 1013904223u;
		history[anchor] = (anchor % 97u) < 40u && anchor >= 50u ? history[anchor - 50u] : (seed >> 8) % TEST_VOCAB;
		Require(SparkSpeculationNgramDraftObserve(&draft,1u,11u,anchor,history + anchor,1u) == SPARK_STATUS_OK,"observe");
		if ( Draft(&draft,1u,11u,anchor,TEST_DEPTH,&result) != SPARK_STATUS_OK )
			result.token_count = 0u;
		take = anchor + 1u < TEST_MAX ? anchor + 1u : TEST_MAX;
		memcpy(suffix,history + anchor + 1u - take,take * sizeof(uint32_t));
		for (depth=0u; depth<TEST_DEPTH; depth++)
		{
			want = ReferenceNext(history,anchor + 1u,suffix + depth,take,&order);
			if ( want == UINT32_MAX )
				break;
			Require(depth < result.token_count && result.token_ids[depth] == want,"draft equals the brute-force reference");
			suffix[take + depth] = want;
			compared++;
		}
		Require(depth == result.token_count,"draft length equals the reference");
	}
	Require(compared > 1000u,"random stream exercises more than 1000 draft tokens");
	SparkSpeculationNgramDraftDestroy(&draft);
}

int main(void)
{
	TestBounds();
	TestFrequencyBeatsRecency();
	TestTieTakesMostRecent();
	TestChainExtendsWithDrafts();
	TestMatchesReferenceOnRandomStream();
	printf("PASS test_speculation_ngram_draft\n");
	return(0);
}
