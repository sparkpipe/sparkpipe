#pragma once

#include <stdint.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

#define SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE 0.05f

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
	uint32_t count, gate, branch_limit, used = 0u, ranked[SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES];
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
	if ( alt_tokens == 0 || alt_scores == 0 || node_limit <= gate + 1u )
		return count;
	branch_limit = node_limit - count;
	if ( branch_limit > SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
		branch_limit = SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES;
	for ( uint32_t step = 0u; step < trunk; step++ )
	{
		const float score = alt_scores[step];
		uint32_t pos = used;
		if ( alt_tokens[step] == tokens[step] || !(score >= min_score) )
			continue;
		while ( pos > 0u && score > alt_scores[ranked[pos - 1u]] )
		{
			if ( pos < branch_limit )
				ranked[pos] = ranked[pos - 1u];
			pos--;
		}
		if ( pos < branch_limit )
		{
			ranked[pos] = step;
			if ( used < branch_limit )
				used++;
		}
	}
	for ( uint32_t index = 0u; index < used; index++ )
	{
		const uint32_t step = ranked[index];
		nodes[count].token_id = alt_tokens[step];
		nodes[count].parent_index = step;
		nodes[count].depth = step + 1u;
		nodes[count].source_bit = source_bit;
		nodes[count].score = alt_scores[step];
		count++;
	}
	return count;
}
