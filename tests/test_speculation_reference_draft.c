#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_reference_draft.h"

#define TEST_VOCAB 1000u
#define TEST_TOKENS 97u
#define TEST_FIRST 40u
#define TEST_SEQUENCE 7u
#define TEST_JUNK 999u

typedef struct TestRun
{
	uint32_t output[TEST_TOKENS];
	uint32_t emitted;
	uint32_t rounds;
	uint32_t accepted;
} TestRun;

static uint32_t reference[TEST_TOKENS];

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static void TestContract(SparkSpeculationModelContract *contract,uint32_t depth)
{
	memset(contract,0,sizeof(*contract));
	contract->abi_version = SPARK_SPECULATION_ABI_VERSION;
	contract->descriptor_bytes = SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES;
	contract->verifier_hidden_dtype = SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16;
	contract->draft_dtype = SPARK_SPECULATION_DRAFT_DTYPE_BF16;
	contract->draft_layer_count = 1u;
	contract->block_size = depth;
	contract->hidden_dimension = 64u;
	contract->attention_head_count = 4u;
	contract->kv_head_count = 4u;
	contract->head_dimension = 16u;
	contract->vocab_size = TEST_VOCAB;
	contract->draft_vocab_size = TEST_VOCAB;
	contract->maximum_speculative_token_count = depth;
	contract->verifier_accept_k = depth;
}

static void TestSpeculator(SparkSpeculationSpeculator *speculator,SparkSpeculationSequenceState *states,SparkSpeculationModelContract *contract,SparkSpeculationReferenceDraft *draft,uint32_t depth)
{
	SparkSpeculationConfiguration configuration;
	TestContract(contract,depth);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_SPECULATION_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_SPECULATION_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.sequence_state_count = 1u;
	configuration.sequence_states = states;
	configuration.default_speculative_token_count = depth;
	configuration.draft_function = SparkSpeculationReferenceDraftTokens;
	configuration.draft_context = draft;
	configuration.model_contract = contract;
	Require(SparkSpeculationPolicyInitialize(speculator,&configuration) == SPARK_STATUS_OK,"policy accepts the reference drafter");
}

static uint32_t TargetToken(uint64_t position,const uint32_t *fed,uint32_t fed_count)
{
	uint32_t index;
	for (index=0u; index<fed_count; index++)
		if ( fed[index] != reference[position - fed_count + 1u + index - TEST_FIRST] )
			return(TEST_JUNK);
	return(reference[position + 1u - TEST_FIRST]);
}

static uint32_t TestDraft(SparkSpeculationSpeculator *speculator,uint64_t position,uint32_t depth,SparkSpeculationPolicyDraftResult *draft)
{
	SparkSpeculationPolicyDraftRequest request;
	uint64_t generation;
	Require(SparkSpeculationPolicyMarkVerifierTapsReady(speculator,1u,TEST_SEQUENCE,position,&generation) == SPARK_STATUS_OK,"taps ready");
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_SPECULATION_ABI_VERSION;
	request.descriptor_bytes = SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count = depth;
	request.request_id = 1u;
	request.sequence_id = TEST_SEQUENCE;
	request.sequence_position = position;
	request.tap_generation = generation;
	Require(SparkSpeculationPolicyEnsureDraft(speculator,&request) == SPARK_STATUS_OK,"draft produced inside the recording");
	Require(SparkSpeculationPolicyGetDraft(speculator,TEST_SEQUENCE,draft) == SPARK_STATUS_OK,"draft readable");
	Require(draft->token_count == depth,"full-depth draft inside the recording");
	return(draft->token_count);
}

static void TestRound(SparkSpeculationSpeculator *speculator,TestRun *run,uint32_t depth)
{
	SparkSpeculationPolicyDraftResult draft;
	SparkSpeculationPolicyVerifyResult verify;
	uint32_t verifier[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT + 1u],count,row;
	uint64_t position;
	position = TEST_FIRST - 1u + run->emitted;
	count = TestDraft(speculator,position,depth,&draft);
	for (row=0u; row<=count; row++)
		verifier[row] = TargetToken(position + row,draft.token_ids,row);
	Require(SparkSpeculationPolicyResolveVerifierTokens(draft.token_ids,count,verifier,count + 1u,TEST_VOCAB,&verify) == SPARK_STATUS_OK,"verify resolves");
	Require(verify.committed_token_count == verify.accepted_draft_token_count + 1u,"one bonus token per round");
	memcpy(run->output + run->emitted,draft.token_ids,verify.accepted_draft_token_count * sizeof(uint32_t));
	run->output[run->emitted + verify.accepted_draft_token_count] = verify.fallback_token_id;
	run->emitted += verify.committed_token_count;
	run->accepted += verify.accepted_draft_token_count;
	run->rounds++;
	Require(SparkSpeculationPolicyCompleteVerify(speculator,TEST_SEQUENCE,&verify) == SPARK_STATUS_OK,"verify accounted");
}

static void TestDecode(SparkSpeculationReferenceMode mode,uint32_t depth,TestRun *run)
{
	SparkSpeculationSpeculator speculator;
	SparkSpeculationSequenceState states[1];
	SparkSpeculationModelContract contract;
	SparkSpeculationReferenceDraft draft;
	memset(run,0,sizeof(*run));
	memset(states,0,sizeof(states));
	Require(SparkSpeculationReferenceDraftInitialize(&draft,mode,TEST_VOCAB,TEST_FIRST,reference,TEST_TOKENS) == SPARK_STATUS_OK,"reference drafter initializes");
	TestSpeculator(&speculator,states,&contract,&draft,depth);
	while ( TEST_TOKENS - run->emitted > depth )
		TestRound(&speculator,run,depth);
	while ( run->emitted < TEST_TOKENS )
	{
		run->output[run->emitted] = TargetToken(TEST_FIRST - 1u + run->emitted,0,0u);
		run->emitted++;
		run->rounds++;
	}
	Require(memcmp(run->output,reference,sizeof(reference)) == 0,"speculative output equals the target's greedy stream");
	Require(speculator.accepted_draft_token_count == run->accepted,"policy counters match the rounds");
}

static void TestBounds(void)
{
	SparkSpeculationReferenceDraft draft;
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	uint32_t bad[2] = {1u,TEST_VOCAB};
	Require(SparkSpeculationReferenceDraftInitialize(&draft,(SparkSpeculationReferenceMode)3,TEST_VOCAB,TEST_FIRST,reference,TEST_TOKENS) == SPARK_STATUS_INVALID_ARGUMENT,"unknown mode rejected");
	Require(SparkSpeculationReferenceDraftInitialize(&draft,SPARK_SPECULATION_REFERENCE_ORACLE,TEST_VOCAB,TEST_FIRST,bad,2u) == SPARK_STATUS_INVALID_ARGUMENT,"out-of-vocabulary recording rejected");
	Require(SparkSpeculationReferenceDraftInitialize(&draft,SPARK_SPECULATION_REFERENCE_ORACLE,TEST_VOCAB,TEST_FIRST,reference,0u) == SPARK_STATUS_INVALID_ARGUMENT,"empty recording rejected");
	Require(SparkSpeculationReferenceDraftInitialize(&draft,SPARK_SPECULATION_REFERENCE_ORACLE,TEST_VOCAB,TEST_FIRST,reference,TEST_TOKENS) == SPARK_STATUS_OK,"oracle initializes");
	memset(&request,0,sizeof(request));
	memset(&result,0,sizeof(result));
	request.requested_token_count = 8u;
	request.sequence_position = TEST_FIRST - 2u;
	Require(SparkSpeculationReferenceDraftTokens(&draft,&request,&result) == SPARK_STATUS_NOT_FOUND,"no draft before the recording");
	request.sequence_position = TEST_FIRST - 1u + TEST_TOKENS;
	Require(SparkSpeculationReferenceDraftTokens(&draft,&request,&result) == SPARK_STATUS_NOT_FOUND,"no draft past the recording");
	request.sequence_position = TEST_FIRST + TEST_TOKENS - 4u;
	Require(SparkSpeculationReferenceDraftTokens(&draft,&request,&result) == SPARK_STATUS_OK && result.token_count == 3u,"draft truncated at the end of the recording");
	Require(memcmp(result.token_ids,reference + TEST_TOKENS - 3u,3u * sizeof(uint32_t)) == 0,"truncated oracle draft is the recording tail");
}

int main(void)
{
	TestRun run;
	uint32_t index,depth;
	for (index=0u; index<TEST_TOKENS; index++)
		reference[index] = (index * 7919u + 13u) % (TEST_VOCAB - 1u);
	TestBounds();
	for (depth=1u; depth<=8u; depth++)
	{
		TestDecode(SPARK_SPECULATION_REFERENCE_ORACLE,depth,&run);
		Require(run.accepted == run.emitted - run.rounds && run.rounds == TEST_TOKENS / (depth + 1u) + TEST_TOKENS % (depth + 1u),"oracle accepts every draft");
		TestDecode(SPARK_SPECULATION_REFERENCE_ADVERSARY,depth,&run);
		Require(run.accepted == 0u && run.rounds == TEST_TOKENS,"adversary is rejected at depth one every round");
	}
	puts("PASS oracle drafts are all accepted, adversary drafts all rejected, and both reproduce the target greedy stream");
	return(0);
}
