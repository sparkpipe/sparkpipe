#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_mtp_tree.h"

typedef struct LegacyMtpTreeResolution
{
	uint32_t path_id;
	uint32_t accepted_token_count;
	uint32_t committed_token_count;
	uint32_t fallback_row_index;
} LegacyMtpTreeResolution;

static uint32_t LegacyVerifierPositionOffset(uint32_t row_index)
{
	if (row_index == SPARK_MODEL_MTP_TREE_VERIFIER_INPUT_ROW)
		return 0u;
	if (row_index == SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW)
		return 1u;
	if (row_index == SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW ||
		row_index == SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW)
		return 2u;
	if (row_index == SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW ||
		row_index == SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW)
		return 3u;
	return UINT32_MAX;
}

static uint32_t LegacyAcceptedTokenCount(uint32_t path_id)
{
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH1)
		return 1u;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY ||
		path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return 2u;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY ||
		path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return 3u;
	return 0u;
}

static uint32_t LegacyFallbackRowIndex(uint32_t path_id)
{
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY)
		return SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW;
	return LegacyAcceptedTokenCount(path_id);
}

static uint32_t LegacyTailCandidateIndex(uint32_t path_id)
{
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY)
		return SPARK_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY)
		return SPARK_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX;
	return SPARK_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX;
}

static uint32_t LegacyTailParentRowIndex(uint32_t path_id)
{
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY ||
		path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW;
	if (path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY ||
		path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW;
	return SPARK_MODEL_MTP_TREE_VERIFIER_INPUT_ROW;
}

static uint32_t LegacyTailBasePositionOffset(uint32_t path_id)
{
	uint32_t accepted_token_count;
	accepted_token_count = LegacyAcceptedTokenCount(path_id);
	return accepted_token_count == 0u ? 0u : accepted_token_count - 1u;
}

static uint32_t LegacyResolutionIsValid(
	uint32_t proposed_token_count,
	uint32_t accepted_token_count,
	uint32_t path_id)
{
	if (accepted_token_count > proposed_token_count)
		return 0u;
	if (proposed_token_count == 0u)
		return accepted_token_count == 0u &&
			path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (proposed_token_count != SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return path_id == SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (path_id >= SPARK_MODEL_MTP_TREE_RESOLUTION_COUNT)
		return 0u;
	return accepted_token_count == LegacyAcceptedTokenCount(path_id);
}

static SparkStatus LegacyMtpTreeResolve(
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids,
	LegacyMtpTreeResolution *resolution)
{
	uint32_t path_id,token_index;
	if (candidate_token_ids == 0 || verifier_token_ids == 0 ||
		resolution == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index=0u;
		 token_index<SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
		 token_index++)
	{
		if (candidate_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	for (token_index=0u;
		 token_index<SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
		 token_index++)
	{
		if (verifier_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (verifier_token_ids[
			SPARK_MODEL_MTP_TREE_VERIFIER_INPUT_ROW] ==
		candidate_token_ids[SPARK_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX])
	{
		path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH1;
		if (verifier_token_ids[
				SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] ==
			candidate_token_ids[SPARK_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX])
		{
			path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY;
			if (verifier_token_ids[
					SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] ==
				candidate_token_ids[
					SPARK_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX])
				path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY;
			else if (verifier_token_ids[
					SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] ==
				candidate_token_ids[
					SPARK_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX])
				path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE;
		}
		else if (verifier_token_ids[
				SPARK_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] ==
			candidate_token_ids[SPARK_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX])
			path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE;
	}
	resolution->path_id = path_id;
	resolution->accepted_token_count = LegacyAcceptedTokenCount(path_id);
	resolution->committed_token_count = resolution->accepted_token_count + 1u;
	resolution->fallback_row_index = LegacyFallbackRowIndex(path_id);
	return SPARK_STATUS_OK;
}

static void SparkTestMtpTreeCompareResolve(
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids)
{
	SparkMtpTreeResolution table_resolution;
	LegacyMtpTreeResolution legacy_resolution;
	SparkStatus table_status,legacy_status;
	memset(&table_resolution,0xa5,sizeof(table_resolution));
	memset(&legacy_resolution,0x5a,sizeof(legacy_resolution));
	table_status = SparkMtpTreeResolve(
		candidate_token_ids,verifier_token_ids,&table_resolution);
	legacy_status = LegacyMtpTreeResolve(
		candidate_token_ids,verifier_token_ids,&legacy_resolution);
	assert(table_status == legacy_status);
	if (table_status != SPARK_STATUS_OK)
		return;
	assert(table_resolution.path_id == legacy_resolution.path_id);
	assert(table_resolution.accepted_token_count ==
		legacy_resolution.accepted_token_count);
	assert(table_resolution.committed_token_count ==
		legacy_resolution.committed_token_count);
	assert(table_resolution.fallback_row_index ==
		legacy_resolution.fallback_row_index);
}

static void SparkTestMtpTreeTopology(void)
{
	assert(SparkMtpTreeTopologyIsValid() == 1u);
}

static void SparkTestMtpTreeHelperEquivalence(void)
{
	uint32_t path_id,row_index,proposed,accepted;
	for (path_id = 0u; path_id < 9u; ++path_id)
	{
		assert(SparkMtpTreeAcceptedTokenCount(path_id) ==
			LegacyAcceptedTokenCount(path_id));
		assert(SparkMtpTreeFallbackRowIndex(path_id) ==
			LegacyFallbackRowIndex(path_id));
		assert(SparkMtpTreeTailCandidateIndex(path_id) ==
			LegacyTailCandidateIndex(path_id));
		assert(SparkMtpTreeTailParentRowIndex(path_id) ==
			LegacyTailParentRowIndex(path_id));
		assert(SparkMtpTreeTailBasePositionOffset(path_id) ==
			LegacyTailBasePositionOffset(path_id));
	}
	for (row_index = 0u; row_index < 9u; ++row_index)
		assert(SparkMtpTreeVerifierPositionOffset(row_index) ==
			LegacyVerifierPositionOffset(row_index));
	for (proposed = 0u; proposed < 7u; ++proposed)
		for (accepted = 0u; accepted < 7u; ++accepted)
			for (path_id = 0u; path_id < 9u; ++path_id)
				assert(SparkMtpTreeResolutionIsValid(
						proposed,accepted,path_id) ==
					LegacyResolutionIsValid(proposed,accepted,path_id));
}

static void SparkTestMtpTreeResolveExhaustiveAlphabet3(void)
{
	uint32_t tokens[11];
	uint32_t combo_index,slot_index,value;
	uint64_t combo_count,combination;
	combo_count = 1u;
	for (slot_index = 0u; slot_index < 11u; ++slot_index)
		combo_count *= 3u;
	for (combination = 0u; combination < combo_count; ++combination)
	{
		value = (uint32_t)combination;
		for (slot_index = 0u; slot_index < 11u; ++slot_index)
		{
			tokens[slot_index] = value % 3u;
			value /= 3u;
		}
		SparkTestMtpTreeCompareResolve(&tokens[0u],&tokens[5u]);
		combo_index = 0u;
		(void)combo_index;
	}
}

static void SparkTestMtpTreeResolveRandomized(void)
{
	uint32_t tokens[11];
	uint32_t trial_index,slot_index;
	srand(20260718u);
	for (trial_index = 0u; trial_index < 200000u; ++trial_index)
	{
		for (slot_index = 0u; slot_index < 11u; ++slot_index)
			tokens[slot_index] = (uint32_t)rand() % 8u;
		SparkTestMtpTreeCompareResolve(&tokens[0u],&tokens[5u]);
	}
}

static void SparkTestMtpTreeResolveRejectsInvalid(void)
{
	uint32_t candidates[SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT];
	uint32_t verifiers[SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];
	SparkMtpTreeResolution resolution;
	memset(candidates,0,sizeof(candidates));
	memset(verifiers,0,sizeof(verifiers));
	assert(SparkMtpTreeResolve(0,verifiers,&resolution) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkMtpTreeResolve(candidates,0,&resolution) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkMtpTreeResolve(candidates,verifiers,0) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	candidates[2u] = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	assert(SparkMtpTreeResolve(candidates,verifiers,&resolution) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	candidates[2u] = 0u;
	verifiers[5u] = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	assert(SparkMtpTreeResolve(candidates,verifiers,&resolution) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
	SparkTestMtpTreeTopology();
	SparkTestMtpTreeHelperEquivalence();
	SparkTestMtpTreeResolveExhaustiveAlphabet3();
	SparkTestMtpTreeResolveRandomized();
	SparkTestMtpTreeResolveRejectsInvalid();
	printf("test_glm52_mtp_tree PASS\n");
	return 0;
}
