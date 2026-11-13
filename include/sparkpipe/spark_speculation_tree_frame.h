#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_TREE_FRAME_MAX_ROWS 256u
#define SPARK_SPECULATION_TREE_FRAME_MAX_BRANCHES 8u
#define SPARK_SPECULATION_TREE_FRAME_NO_ROW UINT32_MAX

typedef struct SparkSpeculationTreeRow
{
	uint32_t depth;
	uint32_t branch;
	uint32_t slot;
	uint32_t fork_position;
	uint32_t slot_offset;
} SparkSpeculationTreeRow;

typedef struct SparkSpeculationTreeLayout
{
	uint32_t rows;
	uint32_t anchor_position;
	uint32_t stride;
	uint32_t branches;
	uint32_t max_depth;
	uint32_t trunk_depth;
	uint32_t context;
	SparkSpeculationTreeRow row[SPARK_SPECULATION_TREE_FRAME_MAX_ROWS];
} SparkSpeculationTreeLayout;

SparkStatus SparkSpeculationTreeLayoutBuild(const uint32_t *parent_rows,const uint32_t *token_ids,uint32_t rows,uint32_t anchor_position,uint32_t slot_grain,
	SparkSpeculationTreeLayout *layout);
SparkStatus SparkSpeculationTreeLayoutMatches(const SparkSpeculationTreeLayout *layout,const uint64_t *positions,const uint32_t *slots,uint32_t rows);
uint32_t SparkSpeculationTreeIsChain(const uint32_t *parent_rows,uint32_t rows);
uint32_t SparkSpeculationTreeResolvePath(const uint32_t *parent_rows,const uint32_t *input_token_ids,const uint32_t *output_token_ids,uint32_t rows,
	uint32_t *path_rows);

#ifdef __cplusplus
}
#endif
