#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "runtime/launch.h"

struct HostTrap {};
static void __trap(void) { throw HostTrap(); }

LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;

#include "inference/kernels/expert_cover.cuh"
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next
#include "sparkpipe/family/spark_family.h"
#include "sparkpipe/family/glm/spark_glm_head_maxloc.cuh"
#include "glm5_next_head_unpack.inc"

#define HOST_EXPERTS 288u
#define HOST_STRIDE 9u
#define HOST_LAYERS 8u
#define HOST_PACK 512u
#define HOST_CANARY 0xa5a5a5a5u
#define HOST_RANKS 16u
#define HOST_ROWS 3u
#define HOST_VOCAB_PER_RANK 1000u

static uint32_t failures,host_seed = 12345u;

static void Expect(uint32_t condition,const char *label)
{
	printf("%s %s\n",condition != 0u ? "ok" : "FAIL",label);
	failures += condition != 0u ? 0u : 1u;
}

static uint32_t HostRandom(void)
{
	host_seed = host_seed * 1664525u + 1013904223u;
	return(host_seed);
}

static void CoverSet(uint32_t *cover,uint32_t layer,uint32_t expert)
{
	cover[layer * HOST_STRIDE + expert / 32u] |= 1u << (expert % 32u);
}

static uint32_t CoverRun(uint32_t *routes,uint32_t *log,uint32_t *ring,const uint32_t *cover,uint32_t layer,uint32_t rows,uint32_t capacity)
{
	try
	{
		LM_LAUNCH((LmExpertCoverKernel),rows,1u,0,0,routes,cover,HOST_STRIDE,HOST_EXPERTS,layer,HOST_PACK,capacity,rows,log,ring);
	}
	catch (HostTrap &)
	{
		return(1u);
	}
	return(0u);
}

static void CheckCoverHits(uint32_t *cover)
{
	uint32_t routes[4] = {3u,200u,287u,3u},log[4] = {0u},ring[2u + 8u] = {0u};
	uint32_t trapped = CoverRun(routes,log,ring,cover,4u,4u,8u);
	Expect(trapped == 0u && routes[0] == 3u && routes[1] == 200u && routes[2] == 287u && routes[3] == 3u,"covered routes stay bitwise untouched");
	Expect(ring[SPARK_STEP_MISS_FLAG] == 0u && ring[SPARK_STEP_MISS_COUNT] == 0u && ring[SPARK_STEP_MISS_ENTRIES] == 0u,"a full hit writes nothing to the miss ring");
	Expect(log[0] == 3u && log[1] == 200u && log[2] == 287u && log[3] == 3u,"the route log keeps every original expert");
}

static void CheckCoverMisses(uint32_t *cover)
{
	uint32_t routes[4] = {5u,288u,0xffffffffu,200u},log[4] = {0u},ring[2u + 8u] = {0u};
	uint32_t trapped = CoverRun(routes,log,ring,cover,4u,4u,8u);
	Expect(trapped == 0u && routes[0] == 3u && routes[1] == 3u && routes[2] == 3u && routes[3] == 200u,"misses and out-of-range routes take the first covered expert of the layer");
	Expect(ring[SPARK_STEP_MISS_FLAG] == 1u && ring[SPARK_STEP_MISS_COUNT] == 3u,"three misses raise the flag and count three");
	Expect(ring[2] == 4u * HOST_PACK + 5u && ring[3] == 4u * HOST_PACK + HOST_EXPERTS && ring[4] == 4u * HOST_PACK + HOST_EXPERTS,"ring entries pack layer and expert; out-of-range routes record the expert-count marker");
	Expect(log[1] == 288u && log[2] == 0xffffffffu,"the route log keeps out-of-range originals");
}

static void CheckNoExpertZero(uint32_t *cover)
{
	uint32_t routes[2] = {0u,7u},ring[2u + 8u] = {0u};
	uint32_t trapped = CoverRun(routes,0,ring,cover,5u,2u,8u);
	Expect(trapped == 0u && routes[0] == 200u && routes[1] == 200u,"an uncovered expert 0 is substituted by a held expert, never read as a fallback");
}

static void CheckRingBound(uint32_t *cover,uint32_t capacity,uint32_t misses)
{
	static uint32_t routes[1100],ring[2u + 1024u + 64u];
	uint32_t index,ordered = 1u,canary = 1u;
	for (index=0u; index<misses; index++)
		routes[index] = 1u + index % 2u;
	for (index=0u; index<2u + capacity + 64u; index++)
		ring[index] = index < 2u ? 0u : HOST_CANARY;
	(void)CoverRun(routes,0,ring,cover,4u,misses,capacity);
	for (index=0u; index<capacity; index++)
		ordered &= ring[2u + index] == 4u * HOST_PACK + 1u + index % 2u ? 1u : 0u;
	for (index=capacity; index<capacity + 64u; index++)
		canary &= ring[2u + index] == HOST_CANARY ? 1u : 0u;
	Expect(ring[SPARK_STEP_MISS_COUNT] == misses && ring[SPARK_STEP_MISS_FLAG] == 1u,"the miss count keeps counting past ring capacity");
	Expect(ordered,"the earliest misses stay in the ring; nothing wraps over them");
	Expect(canary,"entries past capacity are not written");
}

static void CheckEmptyLayerTraps(uint32_t *cover)
{
	uint32_t routes[1] = {7u},ring[2u + 8u] = {0u};
	uint32_t trapped = CoverRun(routes,0,ring,cover,6u,1u,8u);
	Expect(trapped == 1u && routes[0] == 7u && ring[SPARK_STEP_MISS_FLAG] == 0u && ring[SPARK_STEP_MISS_COUNT] == 0u,"a layer with no held expert traps before any substitution");
}

static void CheckPoisonKernel(void)
{
	uint64_t maxloc[4] = {1u,2u,3u,4u};
	uint32_t ring[2] = {0u,0u};
	LM_LAUNCH((LmHeadMissPoisonKernel),4u,1u,0,0,ring,maxloc,3u);
	Expect(maxloc[0] == 1u && maxloc[2] == 3u,"no local miss leaves the head maxloc untouched");
	ring[SPARK_STEP_MISS_FLAG] = 1u;
	LM_LAUNCH((LmHeadMissPoisonKernel),4u,1u,0,0,ring,maxloc,3u);
	Expect(maxloc[0] == UINT64_MAX && maxloc[1] == UINT64_MAX && maxloc[2] == UINT64_MAX && maxloc[3] == 4u,"a local miss poisons exactly the wave rows");
}

static void CheckOrderedScoreSweep(void)
{
	uint64_t bits;
	uint32_t worst = 0u,score;
	for (bits=0u; bits<=UINT32_MAX; bits++)
	{
		score = SparkGlm5NextOrderedHeadScore(__uint_as_float((uint32_t)bits));
		worst = score > worst ? score : worst;
	}
	Expect(worst == 0xff800000u,"every float bit pattern orders at or below +inf, so a packed maxloc never reaches the poison value");
}

static float HostScore(void)
{
	uint32_t pick = HostRandom() % 16u;
	if ( pick == 0u )
		return(__uint_as_float(0x7fc00000u));
	if ( pick == 1u )
		return(__uint_as_float(0xff800000u));
	return((float)(int32_t)(HostRandom() % 64u) - 32.0f);
}

static void FoldRanks(uint64_t *folded,float scores[HOST_RANKS][HOST_ROWS],uint32_t tokens[HOST_RANKS][HOST_ROWS],uint32_t poisoned_rank)
{
	uint64_t maxloc[HOST_ROWS];
	uint32_t rank,row,ring[2] = {0u,0u};
	for (row=0u; row<HOST_ROWS; row++)
		folded[row] = 0u;
	for (rank=0u; rank<HOST_RANKS; rank++)
	{
		ring[SPARK_STEP_MISS_FLAG] = rank == poisoned_rank ? 1u : 0u;
		LM_LAUNCH((SparkGlm5NextHeadMaxlocPackKernel),HOST_ROWS,1u,0,0,scores[rank],tokens[rank],maxloc,HOST_ROWS,rank * HOST_VOCAB_PER_RANK);
		LM_LAUNCH((LmHeadMissPoisonKernel),HOST_ROWS,1u,0,0,ring,maxloc,HOST_ROWS);
		for (row=0u; row<HOST_ROWS; row++)
			folded[row] = maxloc[row] > folded[row] ? maxloc[row] : folded[row];
	}
}

static uint32_t ReferenceToken(float scores[HOST_RANKS][HOST_ROWS],uint32_t tokens[HOST_RANKS][HOST_ROWS],uint32_t row)
{
	uint32_t rank,score,best_score = 0u,token,best_token = UINT32_MAX;
	for (rank=0u; rank<HOST_RANKS; rank++)
	{
		score = SparkGlm5NextOrderedHeadScore(scores[rank][row]);
		token = tokens[rank][row] + rank * HOST_VOCAB_PER_RANK;
		if ( score > best_score || (score == best_score && token < best_token) )
		{
			best_score = score;
			best_token = token;
		}
	}
	return(best_token);
}

static void CheckRankFold(void)
{
	static float scores[HOST_RANKS][HOST_ROWS];
	static uint32_t tokens[HOST_RANKS][HOST_ROWS];
	uint64_t folded[HOST_ROWS];
	uint32_t trial,rank,row,out[HOST_ROWS],clean = 1u,poison = 1u;
	for (trial=0u; trial<2000u; trial++)
	{
		for (rank=0u; rank<HOST_RANKS; rank++)
			for (row=0u; row<HOST_ROWS; row++)
			{
				scores[rank][row] = HostScore();
				tokens[rank][row] = HostRandom() % HOST_VOCAB_PER_RANK;
			}
		FoldRanks(folded,scores,tokens,HOST_RANKS);
		LM_LAUNCH((SparkGlm5NextHeadMaxlocUnpackKernel),HOST_ROWS,1u,0,0,folded,out,HOST_ROWS);
		for (row=0u; row<HOST_ROWS; row++)
			clean &= out[row] == ReferenceToken(scores,tokens,row) ? 1u : 0u;
		clean &= SparkStepVerdictClassify(out,HOST_ROWS,0u) == SPARK_STEP_VERDICT_COMMIT ? 1u : 0u;
		FoldRanks(folded,scores,tokens,trial % HOST_RANKS);
		LM_LAUNCH((SparkGlm5NextHeadMaxlocUnpackKernel),HOST_ROWS,1u,0,0,folded,out,HOST_ROWS);
		poison &= SparkStepVerdictClassify(out,HOST_ROWS,1u) == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
		poison &= SparkStepVerdictClassify(out,HOST_ROWS,0u) == SPARK_STEP_VERDICT_ROLLBACK_REMOTE ? 1u : 0u;
	}
	Expect(clean,"without a miss the 16-rank MAX fold unpacks the exact argmax and commits");
	Expect(poison,"one poisoned rank reaches every row on every rank through the MAX fold");
}

static void CheckVerdictTable(void)
{
	uint32_t clean[3] = {5u,6u,7u},poisoned[3] = {UINT32_MAX,UINT32_MAX,UINT32_MAX},mixed[3] = {5u,UINT32_MAX,7u};
	Expect(SparkStepVerdictClassify(clean,3u,0u) == SPARK_STEP_VERDICT_COMMIT,"clean rows without a local miss commit");
	Expect(SparkStepVerdictClassify(poisoned,3u,1u) == SPARK_STEP_VERDICT_ROLLBACK_LOCAL,"poisoned rows with a local miss roll back as local");
	Expect(SparkStepVerdictClassify(poisoned,3u,0u) == SPARK_STEP_VERDICT_ROLLBACK_REMOTE,"poisoned rows without a local miss roll back as remote");
	Expect(SparkStepVerdictClassify(mixed,3u,0u) == SPARK_STEP_VERDICT_MIXED && SparkStepVerdictClassify(mixed,3u,1u) == SPARK_STEP_VERDICT_MIXED,"mixed rows are an internal error");
	Expect(SparkStepVerdictClassify(clean,3u,1u) == SPARK_STEP_VERDICT_POISON_LOST,"a local miss without poisoned rows is a lost poison, never a commit");
	Expect(SparkStepVerdictClassify(clean,0u,0u) == SPARK_STEP_VERDICT_MIXED,"an empty wave never commits");
}

int main(void)
{
	static uint32_t cover[HOST_LAYERS * HOST_STRIDE];
	CoverSet(cover,4u,3u);
	CoverSet(cover,4u,200u);
	CoverSet(cover,4u,287u);
	CoverSet(cover,5u,200u);
	CheckCoverHits(cover);
	CheckCoverMisses(cover);
	CheckNoExpertZero(cover);
	CheckRingBound(cover,8u,11u);
	CheckRingBound(cover,1024u,1100u);
	CheckEmptyLayerTraps(cover);
	CheckPoisonKernel();
	CheckOrderedScoreSweep();
	CheckRankFold();
	CheckVerdictTable();
	printf("%s (%u failures)\n",failures == 0u ? "PASS" : "FAIL",failures);
	return(failures == 0u ? 0 : 1);
}
