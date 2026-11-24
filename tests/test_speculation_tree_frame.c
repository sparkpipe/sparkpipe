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

int main(void)
{
	TestChainLayoutIsPositions();
	TestBranchLayoutSlotsAndForks();
	TestLayoutRefusals();
	TestResolvePathPicksDeepestAcceptedBranch();
	TestResolvePathChainMatchesPrefixRule();
	TestHazardNodeObservations();
	TestDraftTreePackContract();
	return(0);
}
