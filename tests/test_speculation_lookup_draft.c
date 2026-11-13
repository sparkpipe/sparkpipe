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

static void TestIndexTail(void)
{
	SparkSpeculationLookupIndex index;
	SparkSpeculationLookupMatch match,plain;
	uint32_t heads[8],chain[16],history[10] = {1u,2u,3u,4u,5u,6u,9u,1u,2u,3u},tail[2] = {4u,5u},miss[2] = {4u,7u};
	Require(SparkSpeculationLookupIndexBind(&index,heads,8u,chain,16u,3u) == SPARK_STATUS_OK,"tail index binds");
	Require(SparkSpeculationLookupIndexAppend(&index,history,10u) == SPARK_STATUS_OK,"tail history appends");
	Require(SparkSpeculationLookupIndexFindTail(&index,history,tail,2u,8u,8u,&match) == SPARK_STATUS_OK && match.match_length == 5u && match.source_end == 4u &&
		history[match.source_end + 1u] == 6u,"drafted tokens complete an earlier occurrence and its continuation follows in the history");
	Require(SparkSpeculationLookupIndexFindTail(&index,history,miss,2u,8u,8u,&match) == SPARK_STATUS_NOT_FOUND,"a tail that never occurred has no match");
	Require(SparkSpeculationLookupIndexFindTail(&index,history,0,0u,8u,8u,&match) == SparkSpeculationLookupIndexFind(&index,history,8u,8u,&plain) &&
		match.match_length == plain.match_length && match.source_end == plain.source_end,"an empty tail finds what the plain lookup finds");
	Require(SparkSpeculationLookupIndexFindTail(&index,history,0,2u,8u,8u,&match) == SPARK_STATUS_INVALID_ARGUMENT,"a tail length needs tail tokens");
}

static void TestIndexTailMatchesLinearScan(uint32_t alphabet,uint32_t seed)
{
	enum { LENGTH = 400u, GRAM = 3u, MAXIMUM = 12u };
	SparkSpeculationLookupIndex index;
	SparkSpeculationLookupMatch match;
	uint32_t sequence[LENGTH],heads[64],chain[LENGTH],length,state = seed,found = 0u;
	for (length=0u; length<LENGTH; length++)
	{
		state = state * 1103515245u + 12345u;
		sequence[length] = (state >> 16) % alphabet;
	}
	for (length=GRAM + 2u; length<=LENGTH; length++)
	{
		const uint32_t tail_length = 1u + length % 6u < length - GRAM ? 1u + length % 6u : 1u;
		const uint32_t history_length = length - tail_length;
		uint32_t end,best_length = 0u,best_end = 0u;
		SparkStatus status;
		Require(SparkSpeculationLookupIndexBind(&index,heads,64u,chain,LENGTH,GRAM) == SPARK_STATUS_OK,"tail scan index binds");
		Require(SparkSpeculationLookupIndexAppend(&index,sequence,history_length) == SPARK_STATUS_OK,"tail scan history appends");
		for (end=history_length >= 2u ? history_length - 2u : 0u; end + 1u >= GRAM && history_length >= 2u; end--)
		{
			uint32_t run = 0u;
			while ( run < MAXIMUM && run <= end && run < length && sequence[end - run] == sequence[length - 1u - run] )
				run++;
			if ( run >= GRAM && run > best_length )
			{
				best_length = run;
				best_end = end;
			}
			if ( end == 0u )
				break;
		}
		status = SparkSpeculationLookupIndexFindTail(&index,sequence,sequence + history_length,tail_length,MAXIMUM,UINT32_MAX,&match);
		if ( best_length == 0u )
		{
			Require(status == SPARK_STATUS_NOT_FOUND,"tail lookup finds nothing where the scan finds nothing");
			continue;
		}
		found++;
		Require(status == SPARK_STATUS_OK && match.match_length == best_length && match.source_end == best_end,"tail lookup picks the scan's most recent longest match with a continuation");
	}
	Require(found > 50u,"the tail sequence repeats often enough to test matches");
}

static void TestIndexRanked(void)
{
	uint32_t heads[64],chain[64],history[64],index_count,place;
	SparkSpeculationLookupIndex index;
	SparkSpeculationLookupMatch best,ranked[4];
	const uint32_t passage[6] = { 7u, 8u, 9u, 10u, 11u, 12u };
	for (place=0u; place<6u; place++)
	{
		history[place] = passage[place];
		history[8u + place] = passage[place];
		history[16u + place] = passage[place];
	}
	history[6] = 100u;
	history[7] = 101u;
	history[12] = 40u;
	history[14] = 102u;
	history[15] = 103u;
	history[20] = 41u;
	history[22] = 7u;
	history[23] = 8u;
	history[24] = 9u;
	Require(SparkSpeculationLookupIndexBind(&index,heads,64u,chain,64u,3u) == SPARK_STATUS_OK,"ranked: index binds");
	Require(SparkSpeculationLookupIndexAppend(&index,history,25u) == SPARK_STATUS_OK,"ranked: history appends");
	Require(SparkSpeculationLookupIndexFind(&index,history,8u,64u,&best) == SPARK_STATUS_OK,"ranked: a best match exists");
	Require(SparkSpeculationLookupIndexFindRanked(&index,history,8u,64u,ranked,4u,&index_count) == SPARK_STATUS_OK && index_count == 3u,"ranked: every earlier occurrence is returned");
	Require(ranked[0].match_length == best.match_length && ranked[0].source_end == best.source_end,"ranked: the first entry is the single best match");
	Require(ranked[0].source_end == 18u && ranked[1].source_end == 10u && ranked[2].source_end == 2u,"ranked: the most recent occurrence ranks first on a tie and older ones follow");
	Require(history[ranked[0].source_end + 1u] == 10u && history[ranked[1].source_end + 1u] == 10u && history[ranked[0].source_end + 2u] == 41u && history[ranked[1].source_end + 2u] == 40u,
		"ranked: the two most recent continuations diverge at the second token");
	Require(SparkSpeculationLookupIndexFindRanked(&index,history,8u,64u,ranked,1u,&index_count) == SPARK_STATUS_OK && index_count == 1u && ranked[0].source_end == 18u,
		"ranked: a capacity of one keeps the best match");
	Require(SparkSpeculationLookupIndexFindRanked(&index,history,8u,64u,ranked,0u,&index_count) == SPARK_STATUS_INVALID_ARGUMENT,"ranked: zero capacity is refused");
}

int main(void)
{
	TestIndexBounds();
	TestIndexRanked();
	TestIndexMatchesLinearScan(64u,4u,1u);
	TestIndexMatchesLinearScan(1u,3u,7u);
	TestIndexMatchesLinearScan(16u,6u,11u);
	TestIndexTail();
	TestIndexTailMatchesLinearScan(3u,5u);
	TestIndexTailMatchesLinearScan(5u,13u);
	puts("PASS the n-gram index finds the most recent longest suffix match, ranks its candidates and continues a tail, as a linear scan does");
	return(0);
}
