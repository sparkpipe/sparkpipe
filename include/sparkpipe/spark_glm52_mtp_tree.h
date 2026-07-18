#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#define SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT 5u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT 6u
#define SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT 3u
#define SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT 4u
#define SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION 3u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX 2u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX 3u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX 4u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW 0u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW 1u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW 2u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW 3u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW 4u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW 5u
#define SPARK_GLM52_MODEL_MTP_TREE_BRANCH_ROW_COUNT 4u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT 2u
#define SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT \
	SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT
#define SPARK_GLM52_MODEL_MTP_TREE_ANCESTOR_COPY_COUNT 6u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_POSITION_COUNT 3u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH1_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH2_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH3_INDEX 2u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE 0u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1 1u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY 2u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE 3u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY 4u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE 5u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT 6u

typedef struct SparkGlm52MtpTreeResolution
{
	uint32_t path_id;
	uint32_t accepted_token_count;
	uint32_t committed_token_count;
	uint32_t fallback_row_index;
} SparkGlm52MtpTreeResolution;

typedef struct SparkGlm52MtpTreeNode
{
	uint8_t parent_row;
	uint8_t depth;
	uint8_t candidate_index;
	uint8_t child_row_base;
	uint8_t child_count;
} SparkGlm52MtpTreeNode;

static inline const SparkGlm52MtpTreeNode *SparkGlm52MtpTreeNodeAt(
	uint32_t row_index)
{
	static const SparkGlm52MtpTreeNode
		Nodes[SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT] =
	{
		{0u,0u,0u,1u,1u},
		{0u,1u,0u,2u,2u},
		{1u,2u,1u,4u,2u},
		{1u,2u,2u,0u,0u},
		{2u,3u,3u,0u,0u},
		{2u,3u,4u,0u,0u}
	};
	if (row_index >= SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT)
		return 0;
	return &Nodes[row_index];
}

static inline uint32_t SparkGlm52MtpTreeVerifierPositionOffset(
	uint32_t row_index)
{
	const SparkGlm52MtpTreeNode *node;
	node = SparkGlm52MtpTreeNodeAt(row_index);
	return node != 0 ? node->depth : UINT32_MAX;
}

static inline uint32_t SparkGlm52MtpTreeAcceptedTokenCount(uint32_t path_id)
{
	const SparkGlm52MtpTreeNode *node;
	node = SparkGlm52MtpTreeNodeAt(path_id);
	return node != 0 ? node->depth : 0u;
}

static inline uint32_t SparkGlm52MtpTreeFallbackRowIndex(uint32_t path_id)
{
	return SparkGlm52MtpTreeNodeAt(path_id) != 0 ? path_id : 0u;
}

static inline uint32_t SparkGlm52MtpTreeTailCandidateIndex(uint32_t path_id)
{
	const SparkGlm52MtpTreeNode *node;
	node = SparkGlm52MtpTreeNodeAt(path_id);
	return node != 0 ? node->candidate_index : 0u;
}

static inline uint32_t SparkGlm52MtpTreeTailParentRowIndex(uint32_t path_id)
{
	const SparkGlm52MtpTreeNode *node;
	node = SparkGlm52MtpTreeNodeAt(path_id);
	return node != 0 ? node->parent_row : 0u;
}

static inline uint32_t SparkGlm52MtpTreeTailBasePositionOffset(
	uint32_t path_id)
{
	uint32_t accepted_token_count;
	accepted_token_count = SparkGlm52MtpTreeAcceptedTokenCount(path_id);
	return accepted_token_count == 0u ? 0u : accepted_token_count - 1u;
}

static inline uint32_t SparkGlm52MtpTreeResolutionIsValid(
	uint32_t proposed_token_count,
	uint32_t accepted_token_count,
	uint32_t path_id)
{
	if (accepted_token_count > proposed_token_count)
		return 0u;
	if (proposed_token_count == 0u)
		return accepted_token_count == 0u &&
			path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (proposed_token_count != SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (path_id >= SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT)
		return 0u;
	return accepted_token_count ==
		SparkGlm52MtpTreeAcceptedTokenCount(path_id);
}

static inline uint32_t SparkGlm52MtpTreeTopologyIsValid(void)
{
	uint32_t row_index,child_offset,candidate_seen_mask,max_depth;
	const SparkGlm52MtpTreeNode *node,*parent,*child;
	node = SparkGlm52MtpTreeNodeAt(0u);
	if (node == 0 || node->depth != 0u || node->parent_row != 0u)
		return 0u;
	candidate_seen_mask = 0u;
	max_depth = 0u;
	for (row_index = 1u;
		 row_index < SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
		 ++row_index)
	{
		node = SparkGlm52MtpTreeNodeAt(row_index);
		parent = SparkGlm52MtpTreeNodeAt(node->parent_row);
		if (parent == 0 || node->parent_row >= row_index ||
			node->depth != parent->depth + 1u ||
			node->candidate_index >=
				SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
			(candidate_seen_mask & (1u << node->candidate_index)) != 0u)
			return 0u;
		candidate_seen_mask |= 1u << node->candidate_index;
		if (node->depth > max_depth)
			max_depth = node->depth;
	}
	if (candidate_seen_mask !=
			(1u << SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) - 1u ||
		max_depth != SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION ||
		max_depth + 1u != SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT)
		return 0u;
	for (row_index = 0u;
		 row_index < SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
		 ++row_index)
	{
		node = SparkGlm52MtpTreeNodeAt(row_index);
		if (node->child_count == 0u)
			continue;
		if (node->child_row_base <= row_index ||
			(uint32_t)node->child_row_base + node->child_count >
				SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT)
			return 0u;
		for (child_offset = 0u; child_offset < node->child_count;
			 ++child_offset)
		{
			child = SparkGlm52MtpTreeNodeAt(
				node->child_row_base + child_offset);
			if (child->parent_row != row_index)
				return 0u;
		}
	}
	return 1u;
}

static inline SparkStatus SparkGlm52MtpTreeResolve(
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids,
	SparkGlm52MtpTreeResolution *resolution)
{
	const SparkGlm52MtpTreeNode *node,*child;
	uint32_t current_row,path_id,token_index,child_offset,matched;
	if (candidate_token_ids == 0 || verifier_token_ids == 0 ||
		resolution == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index=0u;
		 token_index<SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
		 token_index++)
	{
		if (candidate_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	for (token_index=0u;
		 token_index<SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
		 token_index++)
	{
		if (verifier_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	current_row = SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW;
	path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	matched = 1u;
	while (matched != 0u)
	{
		matched = 0u;
		node = SparkGlm52MtpTreeNodeAt(current_row);
		for (child_offset = 0u; child_offset < node->child_count;
			 ++child_offset)
		{
			child = SparkGlm52MtpTreeNodeAt(
				node->child_row_base + child_offset);
			if (verifier_token_ids[current_row] !=
				candidate_token_ids[child->candidate_index])
				continue;
			current_row = node->child_row_base + child_offset;
			path_id = current_row;
			matched = 1u;
			break;
		}
	}
	resolution->path_id = path_id;
	resolution->accepted_token_count =
		SparkGlm52MtpTreeAcceptedTokenCount(path_id);
	resolution->committed_token_count = resolution->accepted_token_count + 1u;
	resolution->fallback_row_index =
		SparkGlm52MtpTreeFallbackRowIndex(path_id);
	return SPARK_STATUS_OK;
}
