#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_speculation_frame.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

#define NONE SPARK_SPECULATION_TREE_FRAME_NO_ROW

static void TestChainLayoutIsPositions(void)
{
	const uint32_t parents[5] = { NONE, 0u, 1u, 2u, 3u };
	const uint32_t tokens[5] = { 5u, 6u, 7u, 8u, 9u };
	uint64_t positions[5];
	uint32_t slots[5],row;
	SparkSpeculationTreeLayout layout;
	assert(SparkSpeculationTreeLayoutBuild(parents,tokens,5u,100u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == 0u && layout.max_depth == 4u && layout.trunk_depth == 4u && layout.stride == 16u && layout.context == 105u);
	for (row=0u; row<5u; row++)
	{
		assert(layout.row[row].depth == row && layout.row[row].branch == 0u && layout.row[row].slot == 100u + row && layout.row[row].slot_offset == 0u);
		assert(layout.row[row].fork_position == 100u + row + 1u);
		positions[row] = 100u + row;
		slots[row] = 100u + row;
	}
	assert(SparkSpeculationTreeLayoutMatches(&layout,positions,slots,5u) == SPARK_STATUS_OK);
	assert(SparkSpeculationTreeIsChain(parents,5u) == 1u);
	slots[3] = 200u;
	assert(SparkSpeculationTreeLayoutMatches(&layout,positions,slots,5u) != SPARK_STATUS_OK);
}

static void TestBranchLayoutSlotsAndForks(void)
{
	const uint32_t parents[8] = { NONE, 0u, 1u, 2u, 1u, 4u, 0u, 2u };
	const uint32_t tokens[8] = { 1u, 2u, 3u, 4u, 30u, 31u, 20u, 40u };
	SparkSpeculationTreeLayout layout;
	assert(SparkSpeculationTreeLayoutBuild(parents,tokens,8u,1000u,1u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == 3u && layout.max_depth == 3u && layout.trunk_depth == 3u && layout.stride == 4u);
	assert(layout.row[3].depth == 3u && layout.row[3].branch == 0u && layout.row[3].slot == 1003u);
	assert(layout.row[4].depth == 2u && layout.row[4].branch == 1u && layout.row[4].slot == 1002u + 4u && layout.row[4].fork_position == 1002u && layout.row[4].slot_offset == 4u);
	assert(layout.row[5].depth == 3u && layout.row[5].branch == 1u && layout.row[5].slot == 1003u + 4u && layout.row[5].fork_position == 1002u);
	assert(layout.row[6].depth == 1u && layout.row[6].branch == 2u && layout.row[6].slot == 1001u + 8u && layout.row[6].fork_position == 1001u);
	assert(layout.row[7].depth == 3u && layout.row[7].branch == 3u && layout.row[7].slot == 1003u + 12u && layout.row[7].fork_position == 1003u);
	assert(layout.context == 1003u + 12u + 1u);
	assert(SparkSpeculationTreeIsChain(parents,8u) == 0u);
	assert(SparkSpeculationTreeLayoutBuild(parents,tokens,8u,1000u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.stride == 16u && layout.row[7].slot == 1003u + 48u && layout.context == 1003u + 48u + 1u);
	assert(layout.row[7].slot % 16u == 1003u % 16u && layout.row[4].slot % 16u == 1002u % 16u);
}

static void TestLayoutRefusals(void)
{
	const uint32_t tokens[5] = { 1u, 2u, 3u, 4u, 5u };
	uint32_t parents[5] = { NONE, 0u, 1u, 1u, 3u };
	uint32_t nested[5] = { NONE, 0u, 0u, 2u, 2u };
	uint32_t duplicate_tokens[5] = { 1u, 2u, 3u, 3u, 5u };
	uint32_t forward[5] = { NONE, 2u, 1u, 2u, 3u };
	uint32_t interleaved[5] = { NONE, 0u, 1u, 0u, 2u };
	SparkSpeculationTreeLayout layout;
	assert(SparkSpeculationTreeLayoutBuild(parents,tokens,5u,10u,1u,&layout) == SPARK_STATUS_OK);
	assert(SparkSpeculationTreeLayoutBuild(interleaved,tokens,5u,10u,1u,&layout) != SPARK_STATUS_OK);
	assert(SparkSpeculationTreeLayoutBuild(nested,tokens,5u,10u,1u,&layout) != SPARK_STATUS_OK);
	assert(SparkSpeculationTreeLayoutBuild(parents,duplicate_tokens,5u,10u,1u,&layout) != SPARK_STATUS_OK);
	assert(SparkSpeculationTreeLayoutBuild(forward,tokens,5u,10u,1u,&layout) != SPARK_STATUS_OK);
	parents[0] = 0u;
	assert(SparkSpeculationTreeLayoutBuild(parents,tokens,5u,10u,1u,&layout) != SPARK_STATUS_OK);
}

static void TestResolvePathPicksDeepestAcceptedBranch(void)
{
	const uint32_t parents[7] = { NONE, 0u, 1u, 2u, 1u, 4u, 0u };
	const uint32_t inputs[7] = { 9u, 2u, 3u, 4u, 30u, 31u, 20u };
	uint32_t outputs[7] = { 2u, 30u, 4u, 5u, 31u, 77u, 21u };
	uint32_t path[7],length;
	length = SparkSpeculationTreeResolvePath(parents,inputs,outputs,7u,path);
	assert(length == 4u && path[0] == 0u && path[1] == 1u && path[2] == 4u && path[3] == 5u);
	outputs[1] = 3u;
	length = SparkSpeculationTreeResolvePath(parents,inputs,outputs,7u,path);
	assert(length == 4u && path[0] == 0u && path[1] == 1u && path[2] == 2u && path[3] == 3u);
	outputs[0] = 20u;
	length = SparkSpeculationTreeResolvePath(parents,inputs,outputs,7u,path);
	assert(length == 2u && path[0] == 0u && path[1] == 6u);
	outputs[0] = 55u;
	length = SparkSpeculationTreeResolvePath(parents,inputs,outputs,7u,path);
	assert(length == 1u && path[0] == 0u);
}

static void TestResolvePathChainMatchesPrefixRule(void)
{
	const uint32_t parents[4] = { NONE, 0u, 1u, 2u };
	const uint32_t inputs[4] = { 1u, 2u, 3u, 4u };
	const uint32_t outputs[4] = { 2u, 3u, 9u, 5u };
	uint32_t path[4];
	assert(SparkSpeculationTreeResolvePath(parents,inputs,outputs,4u,path) == 3u);
	assert(path[0] == 0u && path[1] == 1u && path[2] == 2u);
}

static void TestHazardNodeObservations(void)
{
	SparkSpeculationHazard chain,tree;
	memset(&chain,0,sizeof(chain));
	memset(&tree,0,sizeof(tree));
	SparkSpeculationHazardObserve(&chain,4u,2u,0.5f);
	SparkSpeculationHazardDecay(&tree,0.5f);
	SparkSpeculationHazardObserveNode(&tree,1u,0u);
	SparkSpeculationHazardObserveNode(&tree,2u,0u);
	SparkSpeculationHazardObserveNode(&tree,3u,1u);
	assert(memcmp(&chain,&tree,sizeof(chain)) == 0);
	SparkSpeculationHazardObserveNode(&tree,0u,1u);
	assert(memcmp(&chain,&tree,sizeof(chain)) == 0);
}

static void TestDraftTreePackContract(void)
{
	const uint32_t tokens[15] = { 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u };
	const float scores[15] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
	const uint32_t alts[15] = { 10u, 101u, 102u, 103u, 104u, 105u, 106u, 107u, 108u, 109u, 110u, 111u, 112u, 113u, 114u };
	const float alt_scores[15] = { 0.45f, 0.12f, 0.35f, 0.04f, 0.35f, NAN, 0.28f, 0.19f, 0.22f, 0.15f, 0.41f, 0.31f, 0.26f, 0.18f, 0.11f };
	SparkDraftBridgeNode nodes[32];
	uint32_t parents[32], node_tokens[32], count, index;
	SparkSpeculationTreeLayout layout;
	count = SparkDraftTreePack(7u, 2u, tokens, scores, alts, alt_scores, 5u, 5u, 6u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 6u);
	assert(nodes[0].token_id == 7u && nodes[0].parent_index == SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX && nodes[0].depth == 0u && nodes[0].score == 1.0f);
	for (index=0u; index<5u; index++)
		assert(nodes[index + 1u].token_id == tokens[index] && nodes[index + 1u].parent_index == index && nodes[index + 1u].depth == index + 1u);
	count = SparkDraftTreePack(7u, 2u, tokens, scores, alts, alt_scores, 5u, 8u, 9u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 6u);
	count = SparkDraftTreePack(7u, 2u, tokens, scores, alts, alt_scores, 5u, 5u, 10u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 9u);
	assert(nodes[6].token_id == 102u && nodes[6].parent_index == 2u && nodes[6].depth == 3u && fabsf(nodes[6].score - 0.35f) < 1e-6f);
	assert(nodes[7].token_id == 104u && nodes[7].parent_index == 4u && nodes[7].depth == 5u && fabsf(nodes[7].score - 0.35f) < 1e-6f);
	assert(nodes[8].token_id == 101u && nodes[8].parent_index == 1u && nodes[8].depth == 2u && fabsf(nodes[8].score - 0.12f) < 1e-6f);
	for (index=0u; index<count; index++)
	{
		parents[index] = nodes[index].parent_index;
		node_tokens[index] = nodes[index].token_id;
	}
	assert(SparkSpeculationTreeLayoutBuild(parents,node_tokens,count,100u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == 3u && layout.trunk_depth == 5u);
	count = SparkDraftTreePack(7u, 4u, tokens, scores, alts, alt_scores, 15u, 15u, SPARK_SPECULATION_TREE_FRAME_MAX_ROWS, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 16u + SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES);
	assert(nodes[16].token_id == 110u && nodes[16].parent_index == 10u && nodes[16].depth == 11u);
	assert(nodes[17].token_id == 102u && nodes[17].parent_index == 2u && nodes[17].depth == 3u);
	assert(nodes[18].token_id == 104u && nodes[18].parent_index == 4u && nodes[18].depth == 5u);
	for (index=0u; index<count; index++)
	{
		parents[index] = nodes[index].parent_index;
		node_tokens[index] = nodes[index].token_id;
	}
	assert(SparkSpeculationTreeLayoutBuild(parents,node_tokens,count,500u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES && layout.trunk_depth == 15u);
}

static void TestDraftTreePackBranchesContract(void)
{
	const uint32_t tokens[7] = { 10u, 11u, 12u, 13u, 14u, 15u, 16u };
	const float scores[7] = { 0.9f, 0.85f, 0.8f, 0.75f, 0.7f, 0.65f, 0.6f };
	SparkDraftTreeBranch candidates[8];
	SparkDraftBridgeNode nodes[64];
	uint32_t parents[64], node_tokens[64], outputs[64], path[64], trunk_nodes[8];
	uint32_t count, index, depth, walked = 0u, total_walked_nodes = 0u;
	SparkSpeculationTreeLayout layout;
	memset(candidates, 0, sizeof(candidates));
	candidates[0].fork_step = 4u;
	candidates[0].length = 1u;
	candidates[0].tokens[0] = 400u;
	candidates[0].scores[0] = 0.40f;
	candidates[1].fork_step = 1u;
	candidates[1].length = 4u;
	candidates[1].tokens[0] = 200u;
	candidates[1].tokens[1] = 201u;
	candidates[1].tokens[2] = 202u;
	candidates[1].tokens[3] = 203u;
	candidates[1].scores[0] = 0.25f;
	candidates[1].scores[1] = 0.90f;
	candidates[1].scores[2] = 0.85f;
	candidates[1].scores[3] = 0.80f;
	candidates[2].fork_step = 3u;
	candidates[2].length = 3u;
	candidates[2].tokens[0] = 300u;
	candidates[2].tokens[1] = 301u;
	candidates[2].tokens[2] = 302u;
	candidates[2].scores[0] = 0.20f;
	candidates[2].scores[1] = 0.50f;
	candidates[2].scores[2] = 0.02f;
	candidates[3].fork_step = 0u;
	candidates[3].length = 2u;
	candidates[3].tokens[0] = 100u;
	candidates[3].tokens[1] = 101u;
	candidates[3].scores[0] = 0.20f;
	candidates[3].scores[1] = 0.50f;
	candidates[4].fork_step = 0u;
	candidates[4].length = 2u;
	candidates[4].tokens[0] = 10u;
	candidates[4].tokens[1] = 109u;
	candidates[4].scores[0] = 0.99f;
	candidates[4].scores[1] = 0.99f;
	candidates[5].fork_step = 2u;
	candidates[5].length = 2u;
	candidates[5].tokens[0] = 250u;
	candidates[5].tokens[1] = 251u;
	candidates[5].scores[0] = NAN;
	candidates[5].scores[1] = 0.90f;
	candidates[6].fork_step = 1u;
	candidates[6].length = 2u;
	candidates[6].tokens[0] = 200u;
	candidates[6].tokens[1] = 299u;
	candidates[6].scores[0] = 0.20f;
	candidates[6].scores[1] = 0.90f;
	candidates[7].fork_step = 5u;
	candidates[7].length = 4u;
	candidates[7].tokens[0] = 500u;
	candidates[7].tokens[1] = 501u;
	candidates[7].tokens[2] = 502u;
	candidates[7].tokens[3] = 503u;
	candidates[7].scores[0] = 0.15f;
	candidates[7].scores[1] = 0.80f;
	candidates[7].scores[2] = 0.80f;
	candidates[7].scores[3] = 0.80f;
	count = SparkDraftTreePackBranches(7u, 1u, tokens, scores, candidates, 8u, 7u, 7u, 8u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 8u);
	count = SparkDraftTreePackBranches(7u, 1u, tokens, scores, candidates, 8u, 7u, 10u, 11u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 8u);
	count = SparkDraftTreePackBranches(7u, 1u, tokens, scores, candidates, 8u, 7u, 7u, 32u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 19u);
	assert(nodes[8].token_id == 200u && nodes[8].parent_index == 1u && nodes[8].depth == 2u);
	assert(nodes[9].token_id == 201u && nodes[9].parent_index == 8u && nodes[9].depth == 3u);
	assert(nodes[10].token_id == 202u && nodes[10].parent_index == 9u && nodes[10].depth == 4u);
	assert(nodes[11].token_id == 203u && nodes[11].parent_index == 10u && nodes[11].depth == 5u);
	assert(nodes[12].token_id == 400u && nodes[12].parent_index == 4u && nodes[12].depth == 5u);
	assert(nodes[13].token_id == 100u && nodes[13].parent_index == 0u && nodes[13].depth == 1u);
	assert(nodes[14].token_id == 101u && nodes[14].parent_index == 13u && nodes[14].depth == 2u);
	assert(nodes[15].token_id == 300u && nodes[15].parent_index == 3u && nodes[15].depth == 4u);
	assert(nodes[16].token_id == 301u && nodes[16].parent_index == 15u && nodes[16].depth == 5u);
	assert(nodes[17].token_id == 500u && nodes[17].parent_index == 5u && nodes[17].depth == 6u);
	assert(nodes[18].token_id == 501u && nodes[18].parent_index == 17u && nodes[18].depth == 7u);
	for (index=0u; index<count; index++)
	{
		parents[index] = nodes[index].parent_index;
		node_tokens[index] = nodes[index].token_id;
		outputs[index] = 999u;
	}
	assert(SparkSpeculationTreeLayoutBuild(parents,node_tokens,count,200u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == 5u && layout.trunk_depth == 7u && layout.max_depth == 7u);
	outputs[0] = 10u;
	outputs[1] = 200u;
	outputs[8] = 201u;
	outputs[9] = 202u;
	outputs[10] = 203u;
	assert(SparkSpeculationTreeResolvePath(parents,node_tokens,outputs,count,path) == 6u);
	assert(path[0] == 0u && path[1] == 1u && path[2] == 8u && path[3] == 9u && path[4] == 10u && path[5] == 11u);
	for (index=0u; index<=7u; index++)
		trunk_nodes[index] = index;
	for (depth=0u; depth<7u; depth++)
	{
		const uint32_t trunk_node = trunk_nodes[depth], trunk_child = trunk_nodes[depth + 1u];
		uint32_t node;
		for (node=trunk_node + 1u; node<count; node++)
		{
			uint32_t current = node, usable = 0u;
			if ( nodes[node].parent_index != trunk_node || nodes[node].depth != depth + 1u || node == trunk_child )
				continue;
			while ( depth + 1u + usable <= 7u )
			{
				uint32_t child;
				usable++;
				for (child=current + 1u; child<count && (nodes[child].parent_index != current || nodes[child].depth != depth + 1u + usable); child++)
					;
				if ( child >= count )
					break;
				current = child;
			}
			walked++;
			total_walked_nodes += usable;
		}
	}
	assert(walked == 5u && total_walked_nodes == 11u);
	count = SparkDraftTreePackBranches(7u, 1u, tokens, scores, candidates, 8u, 7u, 7u, 10u, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	assert(count == 10u);
	assert(nodes[8].token_id == 200u && nodes[8].parent_index == 1u && nodes[8].depth == 2u);
	assert(nodes[9].token_id == 201u && nodes[9].parent_index == 8u && nodes[9].depth == 3u);
	for (index=0u; index<count; index++)
	{
		parents[index] = nodes[index].parent_index;
		node_tokens[index] = nodes[index].token_id;
	}
	assert(SparkSpeculationTreeLayoutBuild(parents,node_tokens,count,200u,16u,&layout) == SPARK_STATUS_OK);
	assert(layout.branches == 1u && layout.trunk_depth == 7u && layout.max_depth == 7u);
}

int main(void)
{
	TestChainLayoutIsPositions();
	TestBranchLayoutSlotsAndForks();
	TestLayoutRefusals();
	TestResolvePathPicksDeepestAcceptedBranch();
	TestResolvePathChainMatchesPrefixRule();
	TestHazardNodeObservations();
	TestDraftTreePackContract();
	TestDraftTreePackBranchesContract();
	return(0);
}
