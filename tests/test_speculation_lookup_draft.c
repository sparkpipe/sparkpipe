#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_lookup_draft.h"

#define TEST_VOCAB 512u
#define TEST_CAPACITY 4096u
#define TEST_LANES 3u
#define TEST_PROMPT 96u
#define TEST_GENERATE 600u

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static SparkStatus Draft(SparkSpeculationLookupDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t anchor,uint32_t depth,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest request;
	memset(&request,0,sizeof(request));
	memset(result,0,sizeof(*result));
	request.requested_token_count = depth;
	request.active_sequence_index = lane;
	request.sequence_id = sequence_id;
	request.sequence_position = anchor;
	return(SparkSpeculationLookupDraftTokens(draft,&request,result));
}

static void TestBounds(void)
{
	SparkSpeculationLookupDraft draft;
	SparkSpeculationPolicyDraftResult result;
	uint32_t tokens[4] = {1u,2u,3u,SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN};
	Require(SparkSpeculationLookupDraftInitialize(&draft,0u,16u,1u,4u) == SPARK_STATUS_INVALID_ARGUMENT,"zero lanes rejected");
	Require(SparkSpeculationLookupDraftInitialize(&draft,1u,16u,0u,4u) == SPARK_STATUS_INVALID_ARGUMENT,"zero minimum match rejected");
	Require(SparkSpeculationLookupDraftInitialize(&draft,1u,16u,5u,4u) == SPARK_STATUS_INVALID_ARGUMENT,"minimum above maximum rejected");
	Require(SparkSpeculationLookupDraftInitialize(&draft,1u,16u,1u,65u) == SPARK_STATUS_INVALID_ARGUMENT,"maximum above the match limit rejected");
	Require(SparkSpeculationLookupDraftInitialize(&draft,2u,16u,1u,4u) == SPARK_STATUS_OK,"valid drafter initializes");
	Require(SparkSpeculationLookupDraftObserve(&draft,2u,1u,0u,tokens,3u) == SPARK_STATUS_INVALID_ARGUMENT,"lane out of range rejected");
	Require(SparkSpeculationLookupDraftObserve(&draft,0u,1u,14u,tokens,3u) == SPARK_STATUS_CAPACITY_EXCEEDED,"history past capacity rejected");
	Require(SparkSpeculationLookupDraftObserve(&draft,0u,1u,0u,tokens,4u) == SPARK_STATUS_INVALID_ARGUMENT,"the unknown marker is not a token");
	Require(SparkSpeculationLookupDraftObserve(&draft,0u,1u,0u,tokens,3u) == SPARK_STATUS_OK,"history observed");
	Require(Draft(&draft,0u,1u,3u,4u,&result) == SPARK_STATUS_NOT_FOUND,"anchor past the history is not found");
	Require(Draft(&draft,0u,2u,2u,4u,&result) == SPARK_STATUS_NOT_FOUND,"another sequence on the lane is not found");
	Require(Draft(&draft,1u,1u,0u,4u,&result) == SPARK_STATUS_NOT_FOUND,"an unobserved lane is not found");
	Require(Draft(&draft,0u,1u,2u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u,"no repeat means no draft");
	SparkSpeculationLookupDraftDestroy(&draft);
}

static void TestMatches(void)
{
	SparkSpeculationLookupDraft draft;
	SparkSpeculationPolicyDraftResult result;
	SparkSpeculationLookupMatch match;
	static const uint32_t text[] = {10u,11u,12u,13u,14u,15u,50u,11u,12u,13u,20u,21u,22u,60u,11u,12u,13u};
	static const uint32_t rewrite[] = {12u,13u,99u};
	static const uint32_t later[] = {7u,8u};
	uint32_t length = (uint32_t)(sizeof(text) / sizeof(text[0]));
	Require(SparkSpeculationLookupDraftInitialize(&draft,2u,64u,2u,3u) == SPARK_STATUS_OK,"drafter initializes");
	Require(SparkSpeculationLookupDraftObserve(&draft,1u,9u,0u,text,length) == SPARK_STATUS_OK,"prompt observed");
	Require(SparkSpeculationLookupDraftFind(&draft,1u,length - 1u,&match) == SPARK_STATUS_OK && match.match_length == 3u && match.source_end == 9u,"the most recent longest suffix match wins");
	Require(Draft(&draft,1u,9u,length - 1u,4u,&result) == SPARK_STATUS_OK && result.token_count == 4u,"four tokens follow the match");
	Require(result.token_ids[0] == 20u && result.token_ids[1] == 21u && result.token_ids[2] == 22u && result.token_ids[3] == 60u,"continuation of the most recent match is drafted");
	Require(Draft(&draft,1u,9u,length - 1u,32u,&result) == SPARK_STATUS_OK && result.token_count == 7u,"continuation stops at the anchor");
	Require(Draft(&draft,1u,9u,5u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u,"a suffix without an earlier occurrence drafts nothing");
	Require(SparkSpeculationLookupDraftObserve(&draft,1u,9u,15u,rewrite,3u) == SPARK_STATUS_OK,"rewritten tail observed");
	Require(draft.lengths[1] == 18u,"a rewrite extends the history to its last position");
	Require(Draft(&draft,1u,9u,17u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u,"a new suffix does not match");
	Require(SparkSpeculationLookupDraftObserve(&draft,1u,9u,16u,rewrite + 1u,1u) == SPARK_STATUS_OK && draft.lengths[1] == 17u,"a rewrite at an earlier position truncates the history");
	Require(Draft(&draft,1u,9u,17u,4u,&result) == SPARK_STATUS_NOT_FOUND,"a truncated position is unknown");
	Require(Draft(&draft,1u,9u,15u,2u,&result) == SPARK_STATUS_OK && result.token_count == 2u && result.token_ids[0] == 13u && result.token_ids[1] == 20u,"an earlier anchor still drafts from its own prefix");
	Require(SparkSpeculationLookupDraftObserve(&draft,1u,9u,30u,later,2u) == SPARK_STATUS_OK,"a gap in the history is allowed");
	Require(Draft(&draft,1u,9u,25u,4u,&result) == SPARK_STATUS_NOT_FOUND,"an anchor inside the gap is unknown");
	Require(SparkSpeculationLookupDraftObserve(&draft,1u,10u,4u,later,2u) == SPARK_STATUS_OK,"a new sequence on the lane is observed");
	Require(Draft(&draft,1u,9u,5u,4u,&result) == SPARK_STATUS_NOT_FOUND,"the old sequence is gone");
	Require(Draft(&draft,1u,10u,5u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u,"the new sequence history never matches the old one");
	Require(Draft(&draft,0u,0u,0u,4u,&result) == SPARK_STATUS_NOT_FOUND,"lanes are independent");
	SparkSpeculationLookupDraftDestroy(&draft);
}

static uint32_t Target(const uint32_t *history,uint32_t length)
{
	if ( length % 29u == 0u )
		return((history[length - 1u] * 131u + length) % TEST_VOCAB);
	return(history[length - 17u]);
}

static void Greedy(const uint32_t *prompt,uint32_t *output,uint32_t count)
{
	uint32_t history[TEST_PROMPT + TEST_GENERATE],index;
	memcpy(history,prompt,TEST_PROMPT * sizeof(uint32_t));
	for (index=0u; index<count; index++)
	{
		output[index] = Target(history,TEST_PROMPT + index);
		history[TEST_PROMPT + index] = output[index];
	}
}

static void Speculate(const uint32_t *prompt,const uint32_t *expected,uint32_t depth,uint32_t *accepted_out,uint32_t *rounds_out)
{
	SparkSpeculationLookupDraft draft;
	SparkSpeculationPolicyDraftResult result;
	SparkSpeculationPolicyVerifyResult verify;
	uint32_t history[TEST_PROMPT + TEST_GENERATE + 16u],verifier[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT + 1u];
	uint32_t length,row,count,limit,emitted = 0u,accepted = 0u,rounds = 0u;
	Require(SparkSpeculationLookupDraftInitialize(&draft,TEST_LANES,TEST_CAPACITY,2u,6u) == SPARK_STATUS_OK,"drafter initializes");
	memcpy(history,prompt,TEST_PROMPT * sizeof(uint32_t));
	Require(SparkSpeculationLookupDraftObserve(&draft,2u,77u,0u,prompt,TEST_PROMPT) == SPARK_STATUS_OK,"prompt observed");
	while ( emitted < TEST_GENERATE )
	{
		length = TEST_PROMPT + emitted;
		limit = TEST_GENERATE - emitted - 1u < depth ? TEST_GENERATE - emitted - 1u : depth;
		result.token_count = 0u;
		if ( limit != 0u )
			Require(Draft(&draft,2u,77u,length - 1u,limit,&result) == SPARK_STATUS_OK,"lookup drafts");
		count = result.token_count;
		Require(count <= limit,"no more drafts than requested");
		for (row=0u; row<=count; row++)
		{
			verifier[row] = Target(history,length + row);
			if ( row < count )
				history[length + row] = result.token_ids[row];
		}
		memset(&verify,0,sizeof(verify));
		verify.committed_token_count = 1u;
		if ( count != 0u )
			Require(SparkSpeculationPolicyResolveVerifierTokens(result.token_ids,count,verifier,count + 1u,TEST_VOCAB,&verify) == SPARK_STATUS_OK,"verify resolves");
		for (row=0u; row<verify.committed_token_count; row++)
		{
			history[TEST_PROMPT + emitted] = verifier[row];
			Require(verifier[row] == expected[emitted],"speculative output equals the greedy stream");
			Require(SparkSpeculationLookupDraftObserve(&draft,2u,77u,TEST_PROMPT + emitted,&verifier[row],1u) == SPARK_STATUS_OK,"committed token observed");
			emitted++;
		}
		accepted += verify.accepted_draft_token_count;
		rounds++;
	}
	*accepted_out = accepted;
	*rounds_out = rounds;
	SparkSpeculationLookupDraftDestroy(&draft);
}

static void TestIndexMatchesLinearScan(uint32_t bucket_count,uint32_t alphabet,uint32_t seed)
{
	enum { LENGTH = 600u, GRAM = 3u, MAXIMUM = 8u };
	SparkSpeculationLookupDraft draft;
	SparkSpeculationLookupIndex index;
	SparkSpeculationLookupMatch linear,indexed;
	uint32_t history[LENGTH],heads[64],chain[LENGTH],length,state = seed;
	SparkStatus linear_status,indexed_status;
	uint32_t found = 0u;
	Require(SparkSpeculationLookupDraftInitialize(&draft,1u,LENGTH,GRAM,MAXIMUM) == SPARK_STATUS_OK,"linear drafter initializes");
	Require(SparkSpeculationLookupIndexBind(&index,heads,bucket_count,chain,LENGTH,GRAM) == SPARK_STATUS_OK,"index binds");
	for (length=1u; length<=LENGTH; length++)
	{
		state = state * 1103515245u + 12345u;
		history[length - 1u] = (state >> 16) % alphabet;
		Require(SparkSpeculationLookupDraftObserve(&draft,0u,9u,length - 1u,&history[length - 1u],1u) == SPARK_STATUS_OK,"linear drafter observes");
		Require(SparkSpeculationLookupIndexAppend(&index,history,length) == SPARK_STATUS_OK,"index appends");
		linear_status = SparkSpeculationLookupDraftFind(&draft,0u,length - 1u,&linear);
		indexed_status = SparkSpeculationLookupIndexFind(&index,history,MAXIMUM,UINT32_MAX,&indexed);
		Require(linear_status == SPARK_STATUS_OK,"linear scan runs");
		if ( linear.match_length == 0u )
		{
			Require(indexed_status == SPARK_STATUS_NOT_FOUND && indexed.match_length == 0u,"index finds nothing where the scan finds nothing");
			continue;
		}
		found++;
		Require(indexed_status == SPARK_STATUS_OK && indexed.match_length == linear.match_length && indexed.source_end == linear.source_end,"index picks the scan's most recent longest match");
	}
	Require(found > 50u,"the sequence repeats often enough to test matches");
	SparkSpeculationLookupDraftDestroy(&draft);
}

static void TestIndexBounds(void)
{
	SparkSpeculationLookupIndex index;
	SparkSpeculationLookupMatch match;
	uint32_t heads[4],chain[16],history[16] = {1u,2u,3u,9u,1u,2u,3u,8u,1u,2u,3u,7u,1u,2u,3u,0u};
	Require(SparkSpeculationLookupIndexBind(&index,heads,3u,chain,16u,3u) == SPARK_STATUS_INVALID_ARGUMENT,"bucket count must be a power of two");
	Require(SparkSpeculationLookupIndexBind(&index,heads,4u,chain,16u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"gram must be positive");
	Require(SparkSpeculationLookupIndexBind(&index,heads,4u,chain,16u,3u) == SPARK_STATUS_OK,"index binds");
	Require(SparkSpeculationLookupIndexAppend(&index,history,17u) == SPARK_STATUS_CAPACITY_EXCEEDED,"history beyond the capacity is refused");
	Require(SparkSpeculationLookupIndexAppend(&index,history,3u) == SPARK_STATUS_OK,"short history appends");
	Require(SparkSpeculationLookupIndexFind(&index,history,8u,8u,&match) == SPARK_STATUS_NOT_FOUND,"a history no longer than the gram has no earlier match");
	Require(SparkSpeculationLookupIndexAppend(&index,history,2u) == SPARK_STATUS_INVALID_ARGUMENT,"history cannot shrink without a reset");
	Require(SparkSpeculationLookupIndexAppend(&index,history,15u) == SPARK_STATUS_OK,"history appends");
	Require(SparkSpeculationLookupIndexFind(&index,history,2u,8u,&match) == SPARK_STATUS_INVALID_ARGUMENT,"maximum match below the gram is refused");
	Require(SparkSpeculationLookupIndexFind(&index,history,8u,8u,&match) == SPARK_STATUS_OK && match.match_length == 3u && match.source_end == 10u,"the most recent earlier occurrence wins a tie");
	Require(SparkSpeculationLookupIndexFind(&index,history,8u,1u,&match) == SPARK_STATUS_NOT_FOUND,"a candidate limit of one only reaches the anchor itself");
	Require(SparkSpeculationLookupIndexFind(&index,history,8u,2u,&match) == SPARK_STATUS_OK && match.source_end == 10u,"the second candidate is the most recent earlier occurrence");
	SparkSpeculationLookupIndexReset(&index);
	Require(index.length == 0u && SparkSpeculationLookupIndexFind(&index,history,8u,8u,&match) == SPARK_STATUS_NOT_FOUND,"reset forgets the history");
}

int main(void)
{
	uint32_t prompt[TEST_PROMPT],expected[TEST_GENERATE],index,depth,accepted,rounds;
	TestBounds();
	TestMatches();
	TestIndexBounds();
	TestIndexMatchesLinearScan(64u,4u,1u);
	TestIndexMatchesLinearScan(1u,3u,7u);
	TestIndexMatchesLinearScan(16u,6u,11u);
	for (index=0u; index<TEST_PROMPT; index++)
		prompt[index] = (index * 37u + 5u) % 61u;
	Greedy(prompt,expected,TEST_GENERATE);
	for (depth=1u; depth<=7u; depth++)
	{
		Speculate(prompt,expected,depth,&accepted,&rounds);
		Require(accepted > 0u && rounds + accepted == TEST_GENERATE,"repetitive output accepts lookup drafts and every token is accounted");
		printf("depth %u: %u rounds for %u tokens, %u drafts accepted\n",depth,rounds,TEST_GENERATE,accepted);
	}
	puts("PASS lookup drafts come from the most recent longest suffix match, the n-gram index finds the same match, and speculative greedy output equals the greedy stream");
	return(0);
}
