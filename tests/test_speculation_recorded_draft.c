#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_speculation_recorded_draft.h"

#define TEST_VOCAB 500u
#define TEST_DEPTH 4u
#define TEST_ENTRIES 3u
#define TEST_ENTRY_BYTES (SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 4u * TEST_DEPTH)
#define TEST_BYTES (SPARK_SPECULATION_RECORDED_HEADER_BYTES + TEST_ENTRIES * TEST_ENTRY_BYTES)

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

static void Put32(uint8_t *bytes,uint32_t value)
{
	bytes[0] = (uint8_t)value;
	bytes[1] = (uint8_t)(value >> 8);
	bytes[2] = (uint8_t)(value >> 16);
	bytes[3] = (uint8_t)(value >> 24);
}

static void Put64(uint8_t *bytes,uint64_t value)
{
	Put32(bytes,(uint32_t)value);
	Put32(bytes + 4,(uint32_t)(value >> 32));
}

static uint8_t *Entry(uint8_t *table,uint32_t index)
{
	return(table + SPARK_SPECULATION_RECORDED_HEADER_BYTES + index * TEST_ENTRY_BYTES);
}

static void PutEntry(uint8_t *table,uint32_t index,uint64_t sequence,uint64_t position,uint32_t count,const uint32_t *tokens)
{
	uint8_t *entry = Entry(table,index);
	uint32_t slot;
	Put64(entry,sequence);
	Put64(entry + 8,position);
	Put32(entry + 16,count);
	Put32(entry + 20,0u);
	for (slot=0u; slot<TEST_DEPTH; slot++)
		Put32(entry + SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 4u * slot,slot < count ? tokens[slot] : 0u);
}

static void Build(uint8_t *table)
{
	static const uint32_t a[] = {11u,12u,13u,14u},b[] = {21u,22u},c[] = {31u,32u,33u};
	memset(table,0,TEST_BYTES);
	Put32(table,SPARK_SPECULATION_RECORDED_MAGIC);
	Put32(table + 4,SPARK_SPECULATION_RECORDED_VERSION);
	Put32(table + 8,TEST_DEPTH);
	Put32(table + 12,TEST_VOCAB);
	Put64(table + 16,TEST_ENTRIES);
	PutEntry(table,0u,3u,40u,4u,a);
	PutEntry(table,1u,3u,41u,2u,b);
	PutEntry(table,2u,9u,5u,3u,c);
}

static SparkSpeculationPolicyDraftRequest Request(uint64_t sequence,uint64_t position,uint32_t count)
{
	SparkSpeculationPolicyDraftRequest request;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_SPECULATION_ABI_VERSION;
	request.requested_token_count = count;
	request.sequence_id = sequence;
	request.sequence_position = position;
	return(request);
}

static void TestLookups(void)
{
	uint8_t table[TEST_BYTES];
	SparkSpeculationRecordedDraft draft;
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	Build(table);
	Require(SparkSpeculationRecordedDraftInitialize(&draft,table,TEST_BYTES,TEST_VOCAB) == SPARK_STATUS_OK,"valid table");
	memset(&result,0,sizeof(result));
	request = Request(3u,40u,7u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_OK && result.token_count == 4u && result.token_ids[0] == 11u && result.token_ids[3] == 14u && result.confidence_milli[3] == SPARK_SPECULATION_CONFIDENCE_MILLI_ONE,"hit returns the recorded chain");
	request = Request(3u,40u,2u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_OK && result.token_count == 2u && result.token_ids[1] == 12u,"hit is cut to the requested depth");
	request = Request(3u,41u,4u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_OK && result.token_count == 2u && result.token_ids[0] == 21u,"shorter recorded chain");
	request = Request(9u,5u,4u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_OK && result.token_count == 3u && result.token_ids[2] == 33u,"last entry");
	request = Request(3u,42u,4u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_NOT_FOUND,"missing position");
	request = Request(4u,40u,4u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_NOT_FOUND,"missing sequence");
	request = Request(9u,4u,4u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_NOT_FOUND,"position below the first entry");
	Require(draft.hits == 4u && draft.misses == 3u,"hit and miss counters");
	request = Request(3u,40u,0u);
	Require(SparkSpeculationRecordedDraftTokens(&draft,&request,&result) == SPARK_STATUS_INVALID_ARGUMENT,"zero requested tokens");
}

static void RequireRejected(uint8_t *table,uint64_t bytes,uint32_t vocab,const char *what)
{
	SparkSpeculationRecordedDraft draft;
	Require(SparkSpeculationRecordedDraftInitialize(&draft,table,bytes,vocab) != SPARK_STATUS_OK,what);
}

static void TestRejects(void)
{
	uint8_t table[TEST_BYTES];
	Build(table);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB + 1u,"vocab mismatch");
	RequireRejected(table,TEST_BYTES - 1u,TEST_VOCAB,"truncated table");
	RequireRejected(table,SPARK_SPECULATION_RECORDED_HEADER_BYTES - 1u,TEST_VOCAB,"short header");
	Build(table);
	Put32(table,0u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"bad magic");
	Build(table);
	Put32(table + 4,2u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"bad version");
	Build(table);
	Put32(table + 8,0u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"zero depth");
	Build(table);
	Put64(table + 24,1u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"reserved header word");
	Build(table);
	Put64(table + 16,TEST_ENTRIES + 1u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"entry count past the end");
	Build(table);
	Put64(Entry(table,1u) + 8,40u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"duplicate key");
	Build(table);
	Put64(Entry(table,2u),2u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"unsorted sequence");
	Build(table);
	Put32(Entry(table,0u) + 16,0u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"empty chain");
	Build(table);
	Put32(Entry(table,0u) + 16,TEST_DEPTH + 1u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"chain longer than the depth");
	Build(table);
	Put32(Entry(table,1u) + SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES,TEST_VOCAB);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"token outside the vocabulary");
	Build(table);
	Put32(Entry(table,1u) + SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 12u,5u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"nonzero padding past the chain");
	Build(table);
	Put32(Entry(table,1u) + 20,1u);
	RequireRejected(table,TEST_BYTES,TEST_VOCAB,"reserved entry word");
}

int main(void)
{
	TestLookups();
	TestRejects();
	printf("PASS speculation recorded draft: lookups, depth cut, misses, schema rejects\n");
	return(0);
}
