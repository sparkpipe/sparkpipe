#pragma once

#include <stdint.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

#define SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE 0.05f
#define SPARK_DRAFT_TREE_MAX_BRANCH_LENGTH 8u

typedef struct SparkDraftTreeBranch
{
	uint32_t fork_step;
	uint32_t length;
	uint32_t tokens[SPARK_DRAFT_TREE_MAX_BRANCH_LENGTH];
	float scores[SPARK_DRAFT_TREE_MAX_BRANCH_LENGTH];
} SparkDraftTreeBranch;

static inline uint32_t SparkDraftTreePackBranches(
	uint32_t anchor_token_id,
	uint32_t source_bit,
	const uint32_t *tokens,
	const float *scores,
	const SparkDraftTreeBranch *branches,
	uint32_t branch_count,
	uint32_t trunk,
	uint32_t requested_depth,
	uint32_t node_limit,
	float min_score,
	SparkDraftBridgeNode *nodes)
{
	uint32_t count, gate, used = 0u, emitted = 0u, ranked[SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES];
	uint32_t ranked_len[SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES];
	float ranked_expected[SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES];
	if ( node_limit == 0u )
		return 0u;
	if ( node_limit > SPARK_SPECULATION_TREE_FRAME_MAX_ROWS )
		node_limit = SPARK_SPECULATION_TREE_FRAME_MAX_ROWS;
	if ( trunk > node_limit - 1u )
		trunk = node_limit - 1u;
	nodes[0].token_id = anchor_token_id;
	nodes[0].parent_index = SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX;
	nodes[0].depth = 0u;
	nodes[0].source_bit = source_bit;
	nodes[0].score = 1.0f;
	for ( uint32_t step = 0u; step < trunk; step++ )
	{
		nodes[step + 1u].token_id = tokens[step];
		nodes[step + 1u].parent_index = step;
		nodes[step + 1u].depth = step + 1u;
		nodes[step + 1u].source_bit = source_bit;
		nodes[step + 1u].score = scores[step];
	}
	count = trunk + 1u;
	gate = requested_depth > trunk ? requested_depth : trunk;
	if ( branches == 0 || branch_count == 0u || node_limit <= gate + 1u )
		return count;
	for ( uint32_t index = 0u; index < branch_count; index++ )
	{
		const SparkDraftTreeBranch *branch = &branches[index];
		uint32_t max_len = branch->length, usable = 0u, pos = used;
		float path = 1.0f, expected = 0.0f;
		if ( branch->fork_step >= trunk || max_len == 0u || branch->tokens[0] == tokens[branch->fork_step] || !(branch->scores[0] >= min_score) )
			continue;
		if ( max_len > SPARK_DRAFT_TREE_MAX_BRANCH_LENGTH )
			max_len = SPARK_DRAFT_TREE_MAX_BRANCH_LENGTH;
		if ( max_len > gate - branch->fork_step )
			max_len = gate - branch->fork_step;
		for ( uint32_t k = 0u; k < max_len; k++ )
		{
			const float score = branch->scores[k];
			if ( !(score >= min_score) )
				break;
			path *= score < 1.0f ? score : 1.0f;
			expected += path;
			usable++;
		}
		if ( usable == 0u )
			continue;
		while ( pos > 0u && (expected > ranked_expected[pos - 1u] ||
			(expected == ranked_expected[pos - 1u] && branch->fork_step < branches[ranked[pos - 1u]].fork_step)) )
		{
			if ( pos < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
			{
				ranked[pos] = ranked[pos - 1u];
				ranked_len[pos] = ranked_len[pos - 1u];
				ranked_expected[pos] = ranked_expected[pos - 1u];
			}
			pos--;
		}
		if ( pos < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
		{
			ranked[pos] = index;
			ranked_len[pos] = usable;
			ranked_expected[pos] = expected;
			if ( used < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
				used++;
		}
	}
	for ( uint32_t index = 0u; index < used && count < node_limit && emitted < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES; index++ )
	{
		const SparkDraftTreeBranch *branch = &branches[ranked[index]];
		uint32_t parent = branch->fork_step, duplicate = 0u;
		for ( uint32_t prior = trunk + 1u; prior < count; prior++ )
			if ( nodes[prior].parent_index == parent && nodes[prior].token_id == branch->tokens[0] )
			{
				duplicate = 1u;
				break;
			}
		if ( duplicate != 0u )
			continue;
		for ( uint32_t k = 0u; k < ranked_len[index] && count < node_limit; k++ )
		{
			nodes[count].token_id = branch->tokens[k];
			nodes[count].parent_index = parent;
			nodes[count].depth = branch->fork_step + 1u + k;
			nodes[count].source_bit = source_bit;
			nodes[count].score = branch->scores[k];
			parent = count;
			count++;
		}
		emitted++;
	}
	return count;
}

static inline uint32_t SparkDraftTreePack(
	uint32_t anchor_token_id,
	uint32_t source_bit,
	const uint32_t *tokens,
	const float *scores,
	const uint32_t *alt_tokens,
	const float *alt_scores,
	uint32_t trunk,
	uint32_t requested_depth,
	uint32_t node_limit,
	float min_score,
	SparkDraftBridgeNode *nodes)
{
	SparkDraftTreeBranch candidates[SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES];
	uint32_t used = 0u, capped_trunk = trunk, capped_limit = node_limit;
	if ( capped_limit == 0u )
		return 0u;
	if ( capped_limit > SPARK_SPECULATION_TREE_FRAME_MAX_ROWS )
		capped_limit = SPARK_SPECULATION_TREE_FRAME_MAX_ROWS;
	if ( capped_trunk > capped_limit - 1u )
		capped_trunk = capped_limit - 1u;
	if ( alt_tokens != 0 && alt_scores != 0 && capped_limit > (requested_depth > capped_trunk ? requested_depth : capped_trunk) + 1u )
		for ( uint32_t step = 0u; step < capped_trunk; step++ )
		{
			const float score = alt_scores[step];
			uint32_t pos = used;
			if ( alt_tokens[step] == tokens[step] || !(score >= min_score) )
				continue;
			while ( pos > 0u && score > candidates[pos - 1u].scores[0] )
			{
				if ( pos < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
					candidates[pos] = candidates[pos - 1u];
				pos--;
			}
			if ( pos < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
			{
				candidates[pos].fork_step = step;
				candidates[pos].length = 1u;
				candidates[pos].tokens[0] = alt_tokens[step];
				candidates[pos].scores[0] = score;
				if ( used < SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
					used++;
			}
		}
	return SparkDraftTreePackBranches(anchor_token_id, source_bit, tokens, scores, candidates, used, trunk, requested_depth, node_limit, min_score, nodes);
}
