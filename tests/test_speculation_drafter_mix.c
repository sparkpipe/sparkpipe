#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_drafter_mix.h"
#include "sparkpipe/spark_speculation_lookup_draft.h"

#define TEST_VOCAB 97u
#define TEST_CAPACITY 2048u
#define TEST_PROMPT 40u
#define TEST_GENERATE 900u
#define TEST_CAP_MAX 7u
#define TEST_MIN_TOKENS 2u

typedef struct TestScripted
{
	uint32_t status;
	uint32_t count;
	uint32_t calls;
	uint32_t last_requested;
	uint32_t tokens[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
} TestScripted;

typedef struct TestNoisy
{
	const uint32_t *truth;
	uint32_t calls;
	uint32_t length;
} TestNoisy;

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static SparkStatus Scripted(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	TestScripted *scripted = (TestScripted *)context;
	uint32_t index,count;
	scripted->calls++;
	scripted->last_requested = request->requested_token_count;
	if ( scripted->status != SPARK_STATUS_OK )
		return((SparkStatus)scripted->status);
	count = scripted->count < request->requested_token_count ? scripted->count : request->requested_token_count;
	for (index=0u; index<count; index++)
		result->token_ids[index] = scripted->tokens[index];
	result->token_count = count;
	return(SPARK_STATUS_OK);
}

static uint32_t Target(const uint32_t *history,uint32_t length)
{
	uint32_t a = history[length - 1u],b = history[length - 2u];
	if ( (length / 97u) % 3u == 1u )
		return(history[length - 11u]);
	return((a * 31u + b * 7u + length / 5u) % TEST_VOCAB);
}

static SparkStatus Noisy(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	TestNoisy *noisy = (TestNoisy *)context;
	uint32_t index,anchor = (uint32_t)request->sequence_position,wrong;
	noisy->calls++;
	wrong = 1u + (anchor * 13u) % 6u;
	for (index=0u; index<request->requested_token_count && anchor + 1u + index < noisy->length; index++)
		result->token_ids[index] = index == wrong ? (noisy->truth[anchor + 1u + index] + 1u) % TEST_VOCAB : noisy->truth[anchor + 1u + index];
	result->token_count = index;
	return(index == 0u ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_OK);
}

static SparkStatus Ask(SparkSpeculationDrafterMix *mix,uint32_t lane,uint64_t sequence,uint64_t anchor,uint32_t depth,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest request;
	memset(&request,0,sizeof(request));
	memset(result,0,sizeof(*result));
	request.requested_token_count = depth;
	request.active_sequence_index = lane;
	request.sequence_id = sequence;
	request.sequence_position = anchor;
	return(SparkSpeculationDrafterMixTokens(mix,&request,result));
}

static void TestBounds(void)
{
	SparkSpeculationDrafterMix mix;
	TestScripted primary,fallback;
	SparkSpeculationPolicyDraftResult result;
	memset(&primary,0,sizeof(primary));
	memset(&fallback,0,sizeof(fallback));
	Require(SparkSpeculationDrafterMixInitialize(&mix,0,&primary,Scripted,&fallback,2u,2u,7u) == SPARK_STATUS_INVALID_ARGUMENT,"missing primary rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,0,&fallback,2u,2u,7u) == SPARK_STATUS_INVALID_ARGUMENT,"missing fallback rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,0u,2u,7u) == SPARK_STATUS_INVALID_ARGUMENT,"zero primary minimum rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,8u,2u,7u) == SPARK_STATUS_INVALID_ARGUMENT,"primary minimum above the cap rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,2u,0u,7u) == SPARK_STATUS_INVALID_ARGUMENT,"zero lanes rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,2u,2u,SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT + 1u) == SPARK_STATUS_INVALID_ARGUMENT,"cap above the draft width rejected");
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,2u,2u,7u) == SPARK_STATUS_OK,"valid mix initializes");
	Require(Ask(&mix,2u,1u,5u,3u,&result) == SPARK_STATUS_INVALID_ARGUMENT,"lane out of range rejected");
	Require(Ask(&mix,0u,1u,5u,0u,&result) == SPARK_STATUS_INVALID_ARGUMENT,"zero depth rejected");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,1u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"observe without a round rejected");
	SparkSpeculationDrafterMixDestroy(&mix);
}

static void TestRouting(void)
{
	SparkSpeculationDrafterMix mix;
	TestScripted primary,fallback;
	SparkSpeculationPolicyDraftResult result;
	uint32_t index;
	memset(&primary,0,sizeof(primary));
	memset(&fallback,0,sizeof(fallback));
	for (index=0u; index<8u; index++)
	{
		primary.tokens[index] = 10u + index;
		fallback.tokens[index] = 50u + index;
	}
	Require(SparkSpeculationDrafterMixInitialize(&mix,Scripted,&primary,Scripted,&fallback,TEST_MIN_TOKENS,2u,TEST_CAP_MAX) == SPARK_STATUS_OK,"mix initializes");
	primary.count = 5u;
	fallback.count = 7u;
	Require(Ask(&mix,0u,9u,20u,7u,&result) == SPARK_STATUS_OK && result.token_count == 5u && result.token_ids[0] == 10u,"a long primary match is used as is");
	Require(fallback.calls == 0u && SparkSpeculationDrafterMixLastSource(&mix,0u) == SPARK_SPECULATION_DRAFTER_MIX_PRIMARY,"the fallback is not asked when the primary drafts");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,4u,1u) == SPARK_STATUS_INVALID_ARGUMENT,"observe must name the proposed count");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,5u,1u) == SPARK_STATUS_OK,"primary round observed");
	Require(mix.caps[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] == 2u && mix.counters[0].rounds == 1u && mix.counters[0].accepted == 1u,"a rejection sets the primary cap to accepted plus one");
	primary.count = 1u;
	Require(Ask(&mix,0u,9u,26u,7u,&result) == SPARK_STATUS_OK && result.token_count == 7u && result.token_ids[0] == 50u,"a short primary match defers to the fallback");
	Require(primary.last_requested == 2u && fallback.last_requested == 7u && mix.counters[0].declined == 1u,"each source is asked at its own cap");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,7u,0u) == SPARK_STATUS_OK && mix.caps[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] == 1u,"a full fallback miss drops its cap to one");
	primary.status = SPARK_STATUS_NOT_FOUND;
	Require(Ask(&mix,0u,9u,30u,7u,&result) == SPARK_STATUS_OK && result.token_count == 1u && fallback.last_requested == 1u,"the fallback depth follows its acceptance");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,1u,1u) == SPARK_STATUS_OK && mix.caps[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] == 2u,"a full accept doubles the fallback cap");
	primary.status = SPARK_STATUS_OK;
	primary.count = 7u;
	Require(Ask(&mix,0u,9u,33u,7u,&result) == SPARK_STATUS_OK && result.token_count == 2u && primary.last_requested == 2u,"the primary cap never falls below its minimum");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,2u,0u) == SPARK_STATUS_OK && mix.caps[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] == TEST_MIN_TOKENS,"primary cap floor holds after a miss");
	Require(Ask(&mix,0u,10u,0u,7u,&result) == SPARK_STATUS_OK && primary.last_requested == TEST_CAP_MAX && mix.caps[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] == TEST_CAP_MAX,"a new sequence on the lane restores both caps");
	Require(SparkSpeculationDrafterMixObserve(&mix,0u,7u,7u) == SPARK_STATUS_OK,"new sequence round observed");
	Require(Ask(&mix,1u,11u,3u,3u,&result) == SPARK_STATUS_OK && primary.last_requested == 3u,"lanes keep separate caps");
	Require(SparkSpeculationDrafterMixObserve(&mix,1u,3u,3u) == SPARK_STATUS_OK,"second lane observed");
	primary.status = SPARK_STATUS_NOT_FOUND;
	fallback.status = SPARK_STATUS_NOT_FOUND;
	Require(Ask(&mix,1u,11u,6u,3u,&result) == SPARK_STATUS_NOT_FOUND && result.token_count == 0u,"no source means no draft");
	Require(SparkSpeculationDrafterMixLastSource(&mix,1u) == SPARK_SPECULATION_DRAFTER_MIX_NONE && SparkSpeculationDrafterMixObserve(&mix,1u,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"a round without a draft cannot be observed");
	fallback.status = SPARK_STATUS_IO_ERROR;
	Require(Ask(&mix,1u,11u,6u,3u,&result) == SPARK_STATUS_IO_ERROR,"a failing fallback fails the draft");
	primary.status = SPARK_STATUS_INTERNAL_ERROR;
	Require(Ask(&mix,1u,11u,6u,3u,&result) == SPARK_STATUS_INTERNAL_ERROR,"a failing primary fails the draft");
	SparkSpeculationDrafterMixDestroy(&mix);
	Require(mix.caps == 0,"destroy clears the mix");
}

static void TestGreedyLoop(void)
{
	static uint32_t truth[TEST_PROMPT + TEST_GENERATE + 16u],output[TEST_PROMPT + TEST_GENERATE + 16u];
	SparkSpeculationLookupDraft lookup;
	SparkSpeculationDrafterMix mix;
	SparkSpeculationPolicyDraftResult result;
	SparkSpeculationPolicyVerifyResult verify;
	TestNoisy noisy;
	uint32_t verifier[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT + 1u],history[TEST_PROMPT + TEST_GENERATE + 16u];
	uint32_t length,index,rounds = 0u,plain = 0u,source;
	uint64_t committed_total = 0u;
	for (index=0u; index<TEST_PROMPT; index++)
		truth[index] = (index * 37u + 5u) % TEST_VOCAB;
	for (length=TEST_PROMPT; length<TEST_PROMPT + TEST_GENERATE + 16u; length++)
		truth[length] = Target(truth,length);
	memset(&noisy,0,sizeof(noisy));
	noisy.truth = truth;
	noisy.length = TEST_PROMPT + TEST_GENERATE + 16u;
	Require(SparkSpeculationLookupDraftInitialize(&lookup,1u,TEST_CAPACITY,3u,8u) == SPARK_STATUS_OK,"lookup initializes");
	Require(SparkSpeculationDrafterMixInitialize(&mix,SparkSpeculationLookupDraftTokens,&lookup,Noisy,&noisy,TEST_MIN_TOKENS,1u,TEST_CAP_MAX) == SPARK_STATUS_OK,"mix initializes");
	memcpy(output,truth,TEST_PROMPT * sizeof(uint32_t));
	Require(SparkSpeculationLookupDraftObserve(&lookup,0u,5u,0u,output,TEST_PROMPT) == SPARK_STATUS_OK,"prompt observed");
	length = TEST_PROMPT;
	while ( length < TEST_PROMPT + TEST_GENERATE )
	{
		SparkStatus status = Ask(&mix,0u,5u,length - 1u,TEST_CAP_MAX,&result);
		Require(status == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND,"draft status");
		if ( status == SPARK_STATUS_NOT_FOUND )
		{
			output[length] = Target(output,length);
			Require(SparkSpeculationLookupDraftObserve(&lookup,0u,5u,length,&output[length],1u) == SPARK_STATUS_OK,"plain step observed");
			length++;
			plain++;
			continue;
		}
		source = SparkSpeculationDrafterMixLastSource(&mix,0u);
		Require(source == SPARK_SPECULATION_DRAFTER_MIX_PRIMARY ? result.token_count >= TEST_MIN_TOKENS : source == SPARK_SPECULATION_DRAFTER_MIX_FALLBACK,"the round names its source");
		memcpy(history,output,(uint64_t)length * sizeof(uint32_t));
		for (index=0u; index<=result.token_count; index++)
		{
			verifier[index] = Target(history,length + index);
			if ( index < result.token_count )
				history[length + index] = result.token_ids[index];
		}
		Require(SparkSpeculationPolicyResolveVerifierTokens(result.token_ids,result.token_count,verifier,result.token_count + 1u,TEST_VOCAB,&verify) == SPARK_STATUS_OK,"resolve");
		Require(SparkSpeculationDrafterMixObserve(&mix,0u,result.token_count,verify.accepted_draft_token_count) == SPARK_STATUS_OK,"round observed");
		for (index=0u; index<verify.committed_token_count; index++)
			output[length + index] = verifier[index];
		Require(SparkSpeculationLookupDraftObserve(&lookup,0u,5u,length,&output[length],verify.committed_token_count) == SPARK_STATUS_OK,"round observed by lookup");
		length += verify.committed_token_count;
		committed_total += verify.committed_token_count;
		rounds++;
	}
	for (index=0u; index<length; index++)
		Require(output[index] == truth[index],"mixed speculation equals the greedy stream");
	Require(mix.counters[0].rounds != 0u && mix.counters[1].rounds != 0u,"both sources drafted");
	Require(mix.counters[0].rounds + mix.counters[1].rounds == rounds,"every round is attributed to one source");
	Require(mix.counters[0].accepted + mix.counters[1].accepted + rounds == committed_total,"attributed acceptance plus bonus tokens equals the committed tokens");
	Require(mix.counters[0].accepted * 2u > mix.counters[0].proposed,"lookup rounds on the repeating stretches mostly accept");
	printf("mixed loop: %u tokens, %u rounds (lookup %llu proposed %llu accepted %llu, fallback %llu proposed %llu accepted %llu), %u plain steps, noisy calls %u\n",
		TEST_GENERATE,rounds,(unsigned long long)mix.counters[0].rounds,(unsigned long long)mix.counters[0].proposed,(unsigned long long)mix.counters[0].accepted,
		(unsigned long long)mix.counters[1].rounds,(unsigned long long)mix.counters[1].proposed,(unsigned long long)mix.counters[1].accepted,plain,noisy.calls);
	SparkSpeculationDrafterMixDestroy(&mix);
	SparkSpeculationLookupDraftDestroy(&lookup);
}

int main(void)
{
	TestBounds();
	TestRouting();
	TestGreedyLoop();
	printf("PASS test_speculation_drafter_mix\n");
	return(0);
}
