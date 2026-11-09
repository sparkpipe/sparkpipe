#include "sparkpipe/spark_speculation_tree_frame.h"

#include <string.h>

#include "sparkpipe/spark_error_site.h"

static uint32_t SparkSpeculationTreeRoundUp(uint32_t value,uint32_t grain)
{
	return(grain > 1u ? (value + grain - 1u) / grain * grain : value);
}

SparkStatus SparkSpeculationTreeLayoutBuild(const uint32_t *parent_rows,const uint32_t *token_ids,uint32_t rows,uint32_t anchor_position,uint32_t slot_grain,
	SparkSpeculationTreeLayout *layout)
{
	uint32_t first_child[SPARK_SPECULATION_TREE_FRAME_MAX_ROWS];
	uint32_t row,parent,sibling,extent;
	if ( parent_rows == 0 || token_ids == 0 || layout == 0 || rows == 0u || rows > SPARK_SPECULATION_TREE_FRAME_MAX_ROWS || parent_rows[0] != SPARK_SPECULATION_TREE_FRAME_NO_ROW )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(layout,0,sizeof(*layout));
	layout->rows = rows;
	layout->anchor_position = anchor_position;
	for (row=0u; row<rows; row++)
		first_child[row] = SPARK_SPECULATION_TREE_FRAME_NO_ROW;
	for (row=1u; row<rows; row++)
	{
		parent = parent_rows[row];
		if ( parent >= row )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for (sibling=parent + 1u; sibling<row; sibling++)
			if ( parent_rows[sibling] == parent && token_ids[sibling] == token_ids[row] )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		layout->row[row].depth = layout->row[parent].depth + 1u;
		if ( first_child[parent] == SPARK_SPECULATION_TREE_FRAME_NO_ROW )
		{
			first_child[parent] = row;
			layout->row[row].branch = layout->row[parent].branch;
			layout->row[row].fork_position = layout->row[parent].fork_position;
		}
		else
		{
			if ( layout->row[parent].branch != 0u || layout->branches >= SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			layout->branches++;
			layout->row[row].branch = layout->branches;
			layout->row[row].fork_position = anchor_position + layout->row[row].depth;
		}
		if ( layout->row[row].depth > layout->max_depth )
			layout->max_depth = layout->row[row].depth;
		if ( layout->row[row].branch == 0u )
			layout->trunk_depth = layout->row[row].depth;
	}
	if ( anchor_position > UINT32_MAX - (layout->max_depth + 1u) * (SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES + 2u) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout->stride = SparkSpeculationTreeRoundUp(layout->max_depth + 1u,slot_grain);
	extent = 0u;
	for (row=0u; row<rows; row++)
	{
		SparkSpeculationTreeRow *entry = &layout->row[row];
		entry->slot_offset = entry->branch * layout->stride;
		entry->slot = anchor_position + entry->depth + entry->slot_offset;
		if ( entry->branch == 0u )
			entry->fork_position = anchor_position + entry->depth + 1u;
		if ( entry->depth + entry->slot_offset > extent )
			extent = entry->depth + entry->slot_offset;
	}
	layout->context = anchor_position + extent + 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationTreeLayoutMatches(const SparkSpeculationTreeLayout *layout,const uint64_t *positions,const uint32_t *slots,uint32_t rows)
{
	uint32_t row;
	if ( layout == 0 || positions == 0 || slots == 0 || rows != layout->rows )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<rows; row++)
		if ( positions[row] != (uint64_t)layout->anchor_position + layout->row[row].depth || slots[row] != layout->row[row].slot )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

uint32_t SparkSpeculationTreeIsChain(const uint32_t *parent_rows,uint32_t rows)
{
	uint32_t row;
	if ( parent_rows == 0 || rows == 0u || parent_rows[0] != SPARK_SPECULATION_TREE_FRAME_NO_ROW )
		return(0u);
	for (row=1u; row<rows; row++)
		if ( parent_rows[row] != row - 1u )
			return(0u);
	return(1u);
}

uint32_t SparkSpeculationTreeResolvePath(const uint32_t *parent_rows,const uint32_t *input_token_ids,const uint32_t *output_token_ids,uint32_t rows,
	uint32_t *path_rows)
{
	uint32_t depth[SPARK_SPECULATION_TREE_FRAME_MAX_ROWS];
	uint8_t accepted[SPARK_SPECULATION_TREE_FRAME_MAX_ROWS];
	uint32_t row,parent,deepest = 0u,length;
	if ( parent_rows == 0 || input_token_ids == 0 || output_token_ids == 0 || path_rows == 0 || rows == 0u || rows > SPARK_SPECULATION_TREE_FRAME_MAX_ROWS ||
		parent_rows[0] != SPARK_SPECULATION_TREE_FRAME_NO_ROW )
		return(0u);
	depth[0] = 0u;
	accepted[0] = 1u;
	for (row=1u; row<rows; row++)
	{
		parent = parent_rows[row];
		if ( parent >= row )
			return(0u);
		depth[row] = depth[parent] + 1u;
		accepted[row] = accepted[parent] != 0u && output_token_ids[parent] == input_token_ids[row] ? 1u : 0u;
		if ( accepted[row] != 0u && depth[row] > depth[deepest] )
			deepest = row;
	}
	length = depth[deepest] + 1u;
	for (row=deepest; length != 0u; row=parent_rows[row])
		path_rows[--length] = row;
	return(depth[deepest] + 1u);
}
