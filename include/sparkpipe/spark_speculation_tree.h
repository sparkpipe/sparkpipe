#pragma once

/* Neutral speculative-decode tree machinery: node topology, resolution
 * paths, and the longest-prefix resolve walk. The tree SHAPE is per-model:
 * the includer defines SPARK_SPECULATION_TREE_* shape constants, the
 * vocabulary bound, and SPARK_SPECULATION_TREE_NODE_ROWS (the topology
 * table) before including this header; the machinery here is identical for
 * every drafter depth/candidate layout. */

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifndef SPARK_SPECULATION_TREE_CANDIDATE_COUNT
#error "define SPARK_SPECULATION_TREE_CANDIDATE_COUNT before including spark_speculation_tree.h"
#endif
#ifndef SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT
#error "define SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT before including spark_speculation_tree.h"
#endif
#ifndef SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT
#error "define SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT before including spark_speculation_tree.h"
#endif
#ifndef SPARK_SPECULATION_TREE_CONTEXT_EXTENSION
#error "define SPARK_SPECULATION_TREE_CONTEXT_EXTENSION before including spark_speculation_tree.h"
#endif
#ifndef SPARK_SPECULATION_TREE_VOCAB_COUNT
#error "define SPARK_SPECULATION_TREE_VOCAB_COUNT before including spark_speculation_tree.h"
#endif
#ifndef SPARK_SPECULATION_TREE_NODE_ROWS
#error "define SPARK_SPECULATION_TREE_NODE_ROWS before including spark_speculation_tree.h"
#endif

typedef struct SparkSpeculationTreeResolution
{
	uint32_t path_id;
	uint32_t accepted_token_count;
	uint32_t committed_token_count;
	uint32_t fallback_row_index;
} SparkSpeculationTreeResolution;

typedef struct SparkSpeculationTreeNode
{
	uint8_t parent_row;
	uint8_t depth;
	uint8_t candidate_index;
	uint8_t child_row_base;
	uint8_t child_count;
} SparkSpeculationTreeNode;

/* The RUNTIME plan form (the composition program's step 1): the five
 * shape constants and the topology table become fields, so a host-side
 * compositor can build a plan that no includer predicted. The macro API
 * below delegates to the plan-taking functions via one static plan, so
 * every existing consumer compiles unchanged and resolves identically
 * (the bit-identity gate: tests/test_speculation_tree_oracle.c). */
#define SPARK_SPECULATION_TREE_MAX_PLAN_ROWS 64u

typedef struct SparkSpeculationTreePlan
{
	uint32_t candidate_count;
	uint32_t verifier_row_count;
	uint32_t max_committed_token_count;
	uint32_t context_extension;
	uint32_t vocab_count;
	uint32_t row_count;
	SparkSpeculationTreeNode rows[SPARK_SPECULATION_TREE_MAX_PLAN_ROWS];
} SparkSpeculationTreePlan;

static inline void SparkSpeculationTreePlanFromMacros(
	SparkSpeculationTreePlan *plan)
{
	uint32_t row_index;
	static const SparkSpeculationTreeNode rows[SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT] =
		SPARK_SPECULATION_TREE_NODE_ROWS;
	plan->candidate_count = SPARK_SPECULATION_TREE_CANDIDATE_COUNT;
	plan->verifier_row_count = SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
	plan->max_committed_token_count =
		SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT;
	plan->context_extension = SPARK_SPECULATION_TREE_CONTEXT_EXTENSION;
	plan->vocab_count = SPARK_SPECULATION_TREE_VOCAB_COUNT;
	plan->row_count = SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
	for (row_index = 0u; row_index < SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
		++row_index)
		plan->rows[row_index] = rows[row_index];
}

static inline const SparkSpeculationTreeNode *SparkSpeculationTreeNodeAt(
	uint32_t row_index)
{
	static const SparkSpeculationTreeNode
		Nodes[SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT] =
		SPARK_SPECULATION_TREE_NODE_ROWS;
	if (row_index >= SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT)
		return 0;
	return &Nodes[row_index];
}

static inline uint32_t SparkSpeculationTreeVerifierPositionOffset(
	uint32_t row_index)
{
	const SparkSpeculationTreeNode *node;
	node = SparkSpeculationTreeNodeAt(row_index);
	return node != 0 ? node->depth : UINT32_MAX;
}

static inline uint32_t SparkSpeculationTreeAcceptedTokenCount(uint32_t path_id)
{
	const SparkSpeculationTreeNode *node;
	node = SparkSpeculationTreeNodeAt(path_id);
	return node != 0 ? node->depth : 0u;
}

static inline uint32_t SparkSpeculationTreeFallbackRowIndex(uint32_t path_id)
{
	return SparkSpeculationTreeNodeAt(path_id) != 0 ? path_id : 0u;
}

static inline uint32_t SparkSpeculationTreeTailCandidateIndex(uint32_t path_id)
{
	const SparkSpeculationTreeNode *node;
	node = SparkSpeculationTreeNodeAt(path_id);
	return node != 0 ? node->candidate_index : 0u;
}

static inline uint32_t SparkSpeculationTreeTailParentRowIndex(uint32_t path_id)
{
	const SparkSpeculationTreeNode *node;
	node = SparkSpeculationTreeNodeAt(path_id);
	return node != 0 ? node->parent_row : 0u;
}

static inline uint32_t SparkSpeculationTreeTailBasePositionOffset(
	uint32_t path_id)
{
	uint32_t accepted_token_count;
	accepted_token_count = SparkSpeculationTreeAcceptedTokenCount(path_id);
	return accepted_token_count == 0u ? 0u : accepted_token_count - 1u;
}

static inline uint32_t SparkSpeculationTreeResolutionIsValid(
	uint32_t proposed_token_count,
	uint32_t accepted_token_count,
	uint32_t path_id)
{
	if (accepted_token_count > proposed_token_count)
		return 0u;
	if (proposed_token_count == 0u)
		return accepted_token_count == 0u &&
			path_id == SPARK_SPECULATION_TREE_RESOLUTION_NONE;
	if (proposed_token_count != SPARK_SPECULATION_TREE_CANDIDATE_COUNT)
		return path_id == SPARK_SPECULATION_TREE_RESOLUTION_NONE;
	if (path_id >= SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT)
		return 0u;
	return accepted_token_count ==
		SparkSpeculationTreeAcceptedTokenCount(path_id);
}

static inline uint32_t SparkSpeculationTreeTopologyIsValid(void)
{
	uint32_t row_index,child_offset,candidate_seen_mask,max_depth;
	const SparkSpeculationTreeNode *node,*parent,*child;
	node = SparkSpeculationTreeNodeAt(0u);
	if (node == 0 || node->depth != 0u || node->parent_row != 0u)
		return 0u;
	candidate_seen_mask = 0u;
	max_depth = 0u;
	for (row_index = 1u;
		 row_index < SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
		 ++row_index)
	{
		node = SparkSpeculationTreeNodeAt(row_index);
		parent = SparkSpeculationTreeNodeAt(node->parent_row);
		if (parent == 0 || node->parent_row >= row_index ||
			node->depth != parent->depth + 1u ||
			node->candidate_index >=
				SPARK_SPECULATION_TREE_CANDIDATE_COUNT ||
			(candidate_seen_mask & (1u << node->candidate_index)) != 0u)
			return 0u;
		candidate_seen_mask |= 1u << node->candidate_index;
		if (node->depth > max_depth)
			max_depth = node->depth;
	}
	if (candidate_seen_mask !=
			(1u << SPARK_SPECULATION_TREE_CANDIDATE_COUNT) - 1u ||
		max_depth != SPARK_SPECULATION_TREE_CONTEXT_EXTENSION ||
		max_depth + 1u != SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT)
		return 0u;
	for (row_index = 0u;
		 row_index < SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
		 ++row_index)
	{
		node = SparkSpeculationTreeNodeAt(row_index);
		if (node->child_count == 0u)
			continue;
		if (node->child_row_base <= row_index ||
			(uint32_t)node->child_row_base + node->child_count >
				SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT)
			return 0u;
		for (child_offset = 0u; child_offset < node->child_count;
			 ++child_offset)
		{
			child = SparkSpeculationTreeNodeAt(
				node->child_row_base + child_offset);
			if (child->parent_row != row_index)
				return 0u;
		}
	}
	return 1u;
}

static inline SparkStatus SparkSpeculationTreeResolve(
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids,
	SparkSpeculationTreeResolution *resolution)
{
	const SparkSpeculationTreeNode *node,*child;
	uint32_t current_row,path_id,token_index,child_offset,matched;
	if (candidate_token_ids == 0 || verifier_token_ids == 0 ||
		resolution == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index=0u;
		 token_index<SPARK_SPECULATION_TREE_CANDIDATE_COUNT;
		 token_index++)
	{
		if (candidate_token_ids[token_index] >=
			SPARK_SPECULATION_TREE_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	for (token_index=0u;
		 token_index<SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT;
		 token_index++)
	{
		if (verifier_token_ids[token_index] >=
			SPARK_SPECULATION_TREE_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	current_row = SPARK_SPECULATION_TREE_VERIFIER_INPUT_ROW;
	path_id = SPARK_SPECULATION_TREE_RESOLUTION_NONE;
	matched = 1u;
	while (matched != 0u)
	{
		matched = 0u;
		node = SparkSpeculationTreeNodeAt(current_row);
		for (child_offset = 0u; child_offset < node->child_count;
			 ++child_offset)
		{
			child = SparkSpeculationTreeNodeAt(
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
		SparkSpeculationTreeAcceptedTokenCount(path_id);
	resolution->committed_token_count = resolution->accepted_token_count + 1u;
	resolution->fallback_row_index =
		SparkSpeculationTreeFallbackRowIndex(path_id);
	return SPARK_STATUS_OK;
}

/* ===== The runtime-plan form (composition step 1): same semantics,
 * parameterized on a SparkSpeculationTreePlan instead of the includer's
 * macros. The macro API above stays; the oracle pin
 * (tests/test_speculation_tree_oracle.c) proves the two forms resolve
 * identically on the includer's topology. ===== */

static inline const SparkSpeculationTreeNode *SparkSpeculationTreePlanNodeAt(
	const SparkSpeculationTreePlan *plan, uint32_t row_index)
{
	if (plan == 0 || row_index >= plan->row_count ||
		plan->row_count > SPARK_SPECULATION_TREE_MAX_PLAN_ROWS)
		return 0;
	return &plan->rows[row_index];
}

static inline uint32_t SparkSpeculationTreePlanTopologyIsValid(
	const SparkSpeculationTreePlan *plan)
{
	uint32_t row_index,child_offset,candidate_seen_mask,max_depth;
	const SparkSpeculationTreeNode *node,*parent,*child;
	if (plan == 0 || plan->row_count == 0u ||
		plan->row_count > SPARK_SPECULATION_TREE_MAX_PLAN_ROWS)
		return 0u;
	node = SparkSpeculationTreePlanNodeAt(plan, 0u);
	if (node == 0 || node->depth != 0u || node->parent_row != 0u)
		return 0u;
	candidate_seen_mask = 0u;
	max_depth = 0u;
	for (row_index = 1u; row_index < plan->row_count; ++row_index)
	{
		node = SparkSpeculationTreePlanNodeAt(plan, row_index);
		parent = SparkSpeculationTreePlanNodeAt(plan, node->parent_row);
		if (parent == 0 || node->parent_row >= row_index ||
			node->depth != parent->depth + 1u ||
			node->candidate_index >= plan->candidate_count ||
			(candidate_seen_mask & (1u << node->candidate_index)) != 0u)
			return 0u;
		candidate_seen_mask |= 1u << node->candidate_index;
		if (node->depth > max_depth)
			max_depth = node->depth;
	}
	if (candidate_seen_mask != (1u << plan->candidate_count) - 1u ||
		max_depth != plan->context_extension ||
		max_depth + 1u != plan->max_committed_token_count)
		return 0u;
	for (row_index = 0u; row_index < plan->row_count; ++row_index)
	{
		node = SparkSpeculationTreePlanNodeAt(plan, row_index);
		if (node->child_count == 0u)
			continue;
		if (node->child_row_base <= row_index ||
			(uint32_t)node->child_row_base + node->child_count >
				plan->row_count)
			return 0u;
		for (child_offset = 0u; child_offset < node->child_count;
			++child_offset)
		{
			child = SparkSpeculationTreePlanNodeAt(plan,
				node->child_row_base + child_offset);
			if (child->parent_row != row_index)
				return 0u;
		}
	}
	return 1u;
}

static inline SparkStatus SparkSpeculationTreePlanResolve(
	const SparkSpeculationTreePlan *plan,
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids,
	SparkSpeculationTreeResolution *resolution)
{
	const SparkSpeculationTreeNode *node,*child;
	uint32_t current_row,path_id,token_index,child_offset,matched;
	if (plan == 0 || candidate_token_ids == 0 || verifier_token_ids == 0 ||
		resolution == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index = 0u; token_index < plan->candidate_count;
		++token_index)
		if (candidate_token_ids[token_index] >= plan->vocab_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index = 0u; token_index < plan->row_count; ++token_index)
		if (verifier_token_ids[token_index] >= plan->vocab_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
	current_row = 0u;
	path_id = 0u;
	matched = 1u;
	while (matched != 0u)
	{
		matched = 0u;
		node = SparkSpeculationTreePlanNodeAt(plan, current_row);
		for (child_offset = 0u; child_offset < node->child_count;
			++child_offset)
		{
			child = SparkSpeculationTreePlanNodeAt(plan,
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
	resolution->accepted_token_count = path_id < plan->row_count ?
		plan->rows[path_id].depth : 0u;
	resolution->committed_token_count =
		resolution->accepted_token_count + 1u;
	resolution->fallback_row_index = path_id;
	return SPARK_STATUS_OK;
}
