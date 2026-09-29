#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_lookup_draft.h"
#include "sparkpipe/spark_speculation_relay_draft.h"

#define TEST_VOCAB 1000u
#define TEST_PROMPT 64u
#define TEST_GENERATE 400u
#define TEST_DEPTH 7u
#define TEST_SEQUENCE 31u

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static SparkSpeculationRelayFrame DraftFrame(void)
{
	SparkSpeculationRelayFrame frame;
	memset(&frame,0,sizeof(frame));
	frame.kind = SPARK_SPECULATION_RELAY_KIND_DRAFT;
	frame.engine_generation = 0x1122334455667788u;
	frame.round_id = 9u;
	frame.sequence_id = 5u;
	frame.anchor_position = 700u;
	frame.anchor_token = 17u;
	frame.requested_token_count = 4u;
	frame.token_count = 3u;
	frame.token_ids[0] = 1u;
	frame.token_ids[1] = 2u;
	frame.token_ids[2] = 999u;
	return(frame);
}

static void TestCodec(void)
{
	SparkSpeculationRelayFrame frame = DraftFrame(),decoded;
	uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES],copy[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	Require(SparkSpeculationRelayEncode(&frame,TEST_VOCAB,bytes) == SPARK_STATUS_OK,"draft frame encodes");
	Require(bytes[0] == 0x53u && bytes[1] == 0x50u && bytes[2] == 0x52u && bytes[3] == 0x31u,"magic is SPR1 little-endian");
	Require(bytes[4] == 1u && bytes[5] == 0u && bytes[6] == SPARK_SPECULATION_RELAY_KIND_DRAFT && bytes[7] == 0u,"version then kind");
	Require(bytes[8] == 0x88u && bytes[15] == 0x11u,"engine generation is little-endian at offset 8");
	Require(bytes[32] == 0xbcu && bytes[33] == 0x02u && bytes[44] == 3u && bytes[46] == 4u,"anchor position, token count and requested count at fixed offsets");
	Require(bytes[56] == 0xe7u && bytes[57] == 0x03u && bytes[60] == 0u && SPARK_SPECULATION_RELAY_FRAME_BYTES == 112u,"tokens follow the 48-byte header and unused slots are zero");
	Require(SparkSpeculationRelayDecode(bytes,sizeof(bytes),TEST_VOCAB,&decoded) == SPARK_STATUS_OK && memcmp(&decoded,&frame,sizeof(frame)) == 0,"decode inverts encode");
	Require(SparkSpeculationRelayDecode(bytes,sizeof(bytes) - 1u,TEST_VOCAB,&decoded) == SPARK_STATUS_PARSE_ERROR,"short frame rejected");
	memcpy(copy,bytes,sizeof(bytes));
	copy[0] ^= 1u;
	Require(SparkSpeculationRelayDecode(copy,sizeof(copy),TEST_VOCAB,&decoded) == SPARK_STATUS_PARSE_ERROR,"bad magic rejected");
	memcpy(copy,bytes,sizeof(bytes));
	copy[4] = 2u;
	Require(SparkSpeculationRelayDecode(copy,sizeof(copy),TEST_VOCAB,&decoded) == SPARK_STATUS_ABI_MISMATCH,"other version rejected");
	memcpy(copy,bytes,sizeof(bytes));
	copy[6] = 3u;
	Require(SparkSpeculationRelayDecode(copy,sizeof(copy),TEST_VOCAB,&decoded) == SPARK_STATUS_SCHEMA_ERROR,"unknown kind rejected");
	memcpy(copy,bytes,sizeof(bytes));
	copy[64] = 1u;
	Require(SparkSpeculationRelayDecode(copy,sizeof(copy),TEST_VOCAB,&decoded) == SPARK_STATUS_PARSE_ERROR,"bytes past the token count rejected");
	Require(SparkSpeculationRelayDecode(bytes,sizeof(bytes),999u,&decoded) == SPARK_STATUS_SCHEMA_ERROR,"token outside the vocabulary rejected");
	frame.token_count = 5u;
	Require(SparkSpeculationRelayEncode(&frame,TEST_VOCAB,bytes) == SPARK_STATUS_SCHEMA_ERROR,"more drafts than requested rejected");
	frame = DraftFrame();
	frame.kind = SPARK_SPECULATION_RELAY_KIND_REQUEST;
	Require(SparkSpeculationRelayEncode(&frame,TEST_VOCAB,bytes) == SPARK_STATUS_SCHEMA_ERROR,"a request must end with its anchor token");
	frame.token_ids[2] = 17u;
	Require(SparkSpeculationRelayEncode(&frame,TEST_VOCAB,bytes) == SPARK_STATUS_OK,"a request carrying the committed tail encodes");
}

static void Answer(const uint8_t *request_bytes,const uint32_t *drafts,uint32_t count,uint64_t generation,uint8_t *bytes)
{
	SparkSpeculationRelayFrame request,draft;
	Require(SparkSpeculationRelayDecode(request_bytes,SPARK_SPECULATION_RELAY_FRAME_BYTES,TEST_VOCAB,&request) == SPARK_STATUS_OK && request.kind == SPARK_SPECULATION_RELAY_KIND_REQUEST,"draftd decodes the request");
	draft = request;
	draft.kind = SPARK_SPECULATION_RELAY_KIND_DRAFT;
	draft.engine_generation = generation;
	draft.token_count = count;
	memset(draft.token_ids,0,sizeof(draft.token_ids));
	memcpy(draft.token_ids,drafts,(size_t)count * sizeof(uint32_t));
	Require(SparkSpeculationRelayEncode(&draft,TEST_VOCAB,bytes) == SPARK_STATUS_OK,"draftd encodes the draft");
}

static SparkStatus Take(SparkSpeculationRelayDraft *relay,uint64_t sequence,uint64_t anchor,uint32_t depth,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest request;
	memset(&request,0,sizeof(request));
	request.requested_token_count = depth;
	request.sequence_id = sequence;
	request.sequence_position = anchor;
	return(SparkSpeculationRelayDraftTokens(relay,&request,result));
}

static void TestMailbox(void)
{
	SparkSpeculationRelayDraft relay;
	SparkSpeculationPolicyDraftResult result;
	uint8_t request[SPARK_SPECULATION_RELAY_FRAME_BYTES],draft[SPARK_SPECULATION_RELAY_FRAME_BYTES],old[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	uint32_t tail[2] = {40u,41u},drafts[4] = {42u,43u,44u,45u};
	Require(SparkSpeculationRelayDraftInitialize(&relay,0u,TEST_VOCAB) == SPARK_STATUS_INVALID_ARGUMENT,"generation zero is not an engine");
	Require(SparkSpeculationRelayDraftInitialize(&relay,7u,TEST_VOCAB) == SPARK_STATUS_OK,"relay initializes");
	Require(SparkSpeculationRelayDraftIssue(&relay,3u,100u,41u,tail,2u,4u,request) == SPARK_STATUS_OK && relay.round_id == 1u,"round 1 issued");
	Answer(request,drafts,4u,7u,draft);
	Require(SparkSpeculationRelayDraftDeliver(&relay,draft,sizeof(draft)) == SPARK_STATUS_OK && relay.delivered == 1u,"matching draft delivered");
	Require(Take(&relay,3u,100u,3u,&result) == SPARK_STATUS_OK && result.token_count == 3u && result.token_ids[2] == 44u && relay.used == 1u,"the round takes at most its depth");
	Require(Take(&relay,3u,100u,3u,&result) == SPARK_STATUS_OK && result.token_count == 0u && relay.misses == 1u,"a draft is used once");
	memcpy(old,draft,sizeof(draft));
	Require(SparkSpeculationRelayDraftIssue(&relay,3u,103u,44u,tail,1u,4u,request) == SPARK_STATUS_SCHEMA_ERROR && relay.round_id == 1u,"a request whose tail does not end at its anchor is refused");
	tail[0] = 44u;
	Require(SparkSpeculationRelayDraftIssue(&relay,3u,103u,44u,tail,1u,4u,request) == SPARK_STATUS_OK && relay.round_id == 2u,"round 2 issued");
	Require(SparkSpeculationRelayDraftDeliver(&relay,old,sizeof(old)) == SPARK_STATUS_OK && relay.stale == 1u && relay.ready_valid == 0u,"a draft for an old round is dropped and counted");
	Answer(request,drafts,2u,8u,draft);
	Require(SparkSpeculationRelayDraftDeliver(&relay,draft,sizeof(draft)) == SPARK_STATUS_OK && relay.stale == 2u,"a draft from another engine generation is dropped and counted");
	Require(Take(&relay,3u,103u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u && relay.misses == 2u,"no draft by the deadline is a counted k=0 round");
	Require(SparkSpeculationRelayDraftDeliver(&relay,request,sizeof(request)) == SPARK_STATUS_SCHEMA_ERROR,"a request is not a draft");
	Answer(request,drafts,2u,7u,draft);
	Require(SparkSpeculationRelayDraftDeliver(&relay,draft,sizeof(draft)) == SPARK_STATUS_OK && relay.delivered == 2u,"late but current draft delivered");
	Require(Take(&relay,4u,103u,4u,&result) == SPARK_STATUS_OK && result.token_count == 0u && relay.misses == 3u,"a draft for another sequence is never used");
}

static uint32_t Target(const uint32_t *history,uint32_t length)
{
	if ( length % 19u == 0u )
		return((history[length - 1u] * 13u + length) % TEST_VOCAB);
	return(history[length - 23u]);
}

static void TestExactLoop(void)
{
	SparkSpeculationRelayDraft relay;
	SparkSpeculationLookupDraft remote;
	SparkSpeculationPolicyDraftRequest lookup;
	SparkSpeculationPolicyDraftResult proposal,result;
	SparkSpeculationPolicyVerifyResult verify;
	SparkSpeculationRelayFrame decoded;
	uint8_t request[SPARK_SPECULATION_RELAY_FRAME_BYTES],draft[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	uint32_t history[TEST_PROMPT + TEST_GENERATE + 16u],expected[TEST_PROMPT + TEST_GENERATE + 16u],verifier[TEST_DEPTH + 1u];
	uint32_t length,row,round = 0u,accepted = 0u,committed_tail = 1u,index;
	for (index=0u; index<TEST_PROMPT; index++)
		history[index] = (index * 53u + 11u) % 97u;
	memcpy(expected,history,sizeof(uint32_t) * TEST_PROMPT);
	for (length=TEST_PROMPT; length<TEST_PROMPT + TEST_GENERATE + 16u; length++)
		expected[length] = Target(expected,length);
	Require(SparkSpeculationRelayDraftInitialize(&relay,77u,TEST_VOCAB) == SPARK_STATUS_OK,"relay initializes");
	Require(SparkSpeculationLookupDraftInitialize(&remote,1u,4096u,3u,8u) == SPARK_STATUS_OK,"draftd lookup initializes");
	Require(SparkSpeculationLookupDraftObserve(&remote,0u,TEST_SEQUENCE,0u,history,TEST_PROMPT - 1u) == SPARK_STATUS_OK,"prefill history reaches draftd");
	length = TEST_PROMPT;
	while ( length < TEST_PROMPT + TEST_GENERATE )
	{
		Require(SparkSpeculationRelayDraftIssue(&relay,TEST_SEQUENCE,length - 1u,history[length - 1u],history + length - committed_tail,committed_tail,TEST_DEPTH,request) == SPARK_STATUS_OK,"round issued");
		Require(SparkSpeculationRelayDecode(request,sizeof(request),TEST_VOCAB,&decoded) == SPARK_STATUS_OK,"draftd receives the round");
		Require(SparkSpeculationLookupDraftObserve(&remote,0u,decoded.sequence_id,decoded.anchor_position + 1u - decoded.token_count,decoded.token_ids,decoded.token_count) == SPARK_STATUS_OK,"draftd appends the committed tail");
		memset(&lookup,0,sizeof(lookup));
		lookup.requested_token_count = decoded.requested_token_count;
		lookup.sequence_id = decoded.sequence_id;
		lookup.sequence_position = decoded.anchor_position;
		Require(SparkSpeculationLookupDraftTokens(&remote,&lookup,&proposal) == SPARK_STATUS_OK,"draftd drafts");
		Answer(request,proposal.token_ids,proposal.token_count,77u,draft);
		if ( round % 5u != 4u )
			Require(SparkSpeculationRelayDraftDeliver(&relay,draft,sizeof(draft)) == SPARK_STATUS_OK,"draft delivered before the deadline");
		Require(Take(&relay,TEST_SEQUENCE,length - 1u,TEST_DEPTH,&result) == SPARK_STATUS_OK,"round takes its draft");
		if ( round % 5u == 4u )
			Require(SparkSpeculationRelayDraftDeliver(&relay,draft,sizeof(draft)) == SPARK_STATUS_OK && relay.ready_valid == 1u,"a late draft waits for nobody");
		for (row=0u; row<=result.token_count; row++)
			verifier[row] = expected[length + row];
		for (row=0u; row<result.token_count && result.token_ids[row] == expected[length + row]; row++)
			;
		verify.accepted_draft_token_count = row;
		verify.committed_token_count = row + 1u;
		if ( result.token_count != 0u )
			Require(SparkSpeculationPolicyResolveVerifierTokens(result.token_ids,result.token_count,verifier,result.token_count + 1u,TEST_VOCAB,&verify) == SPARK_STATUS_OK && verify.accepted_draft_token_count == row,"verify resolves the same prefix");
		memcpy(history + length,verifier,(size_t)verify.committed_token_count * sizeof(uint32_t));
		committed_tail = verify.committed_token_count;
		length += verify.committed_token_count;
		accepted += verify.accepted_draft_token_count;
		round++;
	}
	Require(memcmp(history,expected,sizeof(uint32_t) * length) == 0,"relayed speculation reproduces the greedy stream");
	Require(accepted > TEST_GENERATE / 2u && relay.misses == round / 5u && relay.used + relay.misses == round,"late rounds are counted misses and the rest use their drafts");
	printf("relay loop: %u rounds for %u tokens, %u drafts accepted, %llu misses\n",round,length - TEST_PROMPT,accepted,(unsigned long long)relay.misses);
	SparkSpeculationLookupDraftDestroy(&remote);
}

int main(void)
{
	TestCodec();
	TestMailbox();
	TestExactLoop();
	puts("PASS relay frames are fixed little-endian, stale and late drafts are counted and dropped, and relayed lookup drafts keep greedy output exact");
	return(0);
}
