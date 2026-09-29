#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_graph_regime.h"
#include "sparkpipe/spark_status.h"

#define SPARK_GLM5_NEXT_VERIFY_ROWS_MIN 2u
#define SPARK_GLM5_NEXT_VERIFY_ROWS_MAX 8u
#define SPARK_GLM5_NEXT_VERIFY_TABLE_COUNT (SPARK_GLM5_NEXT_VERIFY_ROWS_MAX - SPARK_GLM5_NEXT_VERIFY_ROWS_MIN + 1u)
#define SPARK_GLM5_NEXT_VERIFY_ROWS_ENV "SPARK_GLM5_NEXT_VERIFY_ROWS"
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_ENV "SPARK_GLM5_NEXT_VERIFY_DRAFTER"
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE 0u
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP 1u
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE 2u
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY 3u
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP 4u
#define SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP 5u
#define SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV "SPARK_GLM5_NEXT_VERIFY_MTP_DIR"
#define SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS 2u
#define SPARK_GLM5_NEXT_VERIFY_MTP_PACK_PREFIX "glm5_next_mtp.tp"
#define SPARK_GLM5_NEXT_VERIFY_ORACLE_PREFIX "oracle:"
#define SPARK_GLM5_NEXT_VERIFY_ADVERSARY_PREFIX "adversary:"
#define SPARK_GLM5_NEXT_VERIFY_LOOKUP_MIN_MATCH 3u
#define SPARK_GLM5_NEXT_VERIFY_LOOKUP_MAX_MATCH 8u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_ELIGIBLE 0u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SHAPE 1u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SAMPLED 2u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD 3u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_NO_DRAFT 4u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_RANK_LOCAL 5u
#define SPARK_GLM5_NEXT_VERIFY_FRAME_CLASS_COUNT 6u

static inline SparkStatus SparkGlm5NextVerifyRowsParse(const char *text,uint32_t *rows_out)
{
	uint32_t rows;
	*rows_out = 0u;
	if ( text == 0 )
		return(SPARK_STATUS_OK);
	if ( text[0] < '0' || text[0] > '9' || text[1] != '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows = (uint32_t)(text[0] - '0');
	if ( rows != 0u && (rows < SPARK_GLM5_NEXT_VERIFY_ROWS_MIN || rows > SPARK_GLM5_NEXT_VERIFY_ROWS_MAX) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*rows_out = rows;
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkGlm5NextVerifyDrafterParse(const char *text,uint32_t rows,uint32_t *kind_out,const char **path_out)
{
	*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE;
	*path_out = 0;
	if ( rows == 0u )
		return(text == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( text == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( strcmp(text,"lookup") == 0 )
	{
		*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP;
		return(SPARK_STATUS_OK);
	}
	if ( strcmp(text,"mtp") == 0 )
	{
		*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP;
		return(SPARK_STATUS_OK);
	}
	if ( strcmp(text,"mtp+lookup") == 0 )
	{
		*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP;
		return(SPARK_STATUS_OK);
	}
	if ( strncmp(text,SPARK_GLM5_NEXT_VERIFY_ORACLE_PREFIX,sizeof(SPARK_GLM5_NEXT_VERIFY_ORACLE_PREFIX) - 1u) == 0 && text[sizeof(SPARK_GLM5_NEXT_VERIFY_ORACLE_PREFIX) - 1u] != '\0' )
	{
		*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE;
		*path_out = text + sizeof(SPARK_GLM5_NEXT_VERIFY_ORACLE_PREFIX) - 1u;
		return(SPARK_STATUS_OK);
	}
	if ( strncmp(text,SPARK_GLM5_NEXT_VERIFY_ADVERSARY_PREFIX,sizeof(SPARK_GLM5_NEXT_VERIFY_ADVERSARY_PREFIX) - 1u) == 0 && text[sizeof(SPARK_GLM5_NEXT_VERIFY_ADVERSARY_PREFIX) - 1u] != '\0' )
	{
		*kind_out = SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY;
		*path_out = text + sizeof(SPARK_GLM5_NEXT_VERIFY_ADVERSARY_PREFIX) - 1u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static inline uint32_t SparkGlm5NextVerifyDrafterUsesMtp(uint32_t kind)
{
	return(kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP || kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextVerifyDrafterUsesLookup(uint32_t kind)
{
	return(kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP || kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP ? 1u : 0u);
}

static inline SparkStatus SparkGlm5NextMtpPackPath(const char *directory,uint32_t tp_degree,uint32_t tp_rank,char *path,uint32_t path_bytes)
{
	int written;
	if ( directory == 0 || directory[0] == '\0' || path == 0 || path_bytes == 0u || tp_degree == 0u || tp_rank >= tp_degree )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	written = snprintf(path,path_bytes,"%s/" SPARK_GLM5_NEXT_VERIFY_MTP_PACK_PREFIX "%u.rank%u.g5nsp",directory,tp_degree,tp_rank);
	if ( written < 0 || (uint32_t)written >= path_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkGlm5NextVerifyFrameClass(uint32_t decode_single_greedy_shape,uint32_t sampled,uint32_t experts_warm,uint32_t graph_path_enabled,uint32_t graph_disabled,uint32_t b1_capture_failed,uint32_t verify_captured)
{
	if ( decode_single_greedy_shape == 0u )
		return(SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SHAPE);
	if ( sampled != 0u )
		return(SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SAMPLED);
	if ( experts_warm == 0u )
		return(SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD);
	if ( graph_path_enabled == 0u || graph_disabled != 0u || b1_capture_failed != 0u )
		return(SPARK_GLM5_NEXT_VERIFY_FRAME_RANK_LOCAL);
	if ( verify_captured == 0u )
		return(SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD);
	return(SPARK_GLM5_NEXT_VERIFY_FRAME_ELIGIBLE);
}

static inline uint32_t SparkGlm5NextVerifyTableIndex(uint32_t rows)
{
	return(rows - SPARK_GLM5_NEXT_VERIFY_ROWS_MIN);
}

static inline uint32_t SparkGlm5NextVerifyRowsFit(uint32_t position,uint32_t rows,uint32_t split_threshold,uint32_t max_positions)
{
	uint32_t regime,fit;
	if ( rows == 0u || position >= max_positions )
		return(0u);
	regime = SparkGlm5NextGraphRegime(position + 1u,split_threshold);
	fit = 1u;
	while ( fit < rows && position + fit < max_positions && SparkGlm5NextGraphRegime(position + fit + 1u,split_threshold) == regime )
		fit++;
	return(fit);
}

static inline SparkStatus SparkGlm5NextVerifyRowsAgree(uint32_t rows,const uint32_t *resident_slots,const uint32_t *positions)
{
	uint32_t row;
	for (row=1u; row<rows; row++)
		if ( resident_slots[row] != resident_slots[0] || positions[row] != positions[0] + row )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkGlm5NextVerifyDepth(uint32_t budget,uint32_t produced,uint32_t rows_max,uint32_t position,uint32_t split_threshold,uint32_t max_positions)
{
	uint32_t depth,fit;
	if ( rows_max < SPARK_GLM5_NEXT_VERIFY_ROWS_MIN || produced >= budget || budget - produced < 2u )
		return(0u);
	depth = budget - produced - 1u;
	if ( depth > rows_max - 1u )
		depth = rows_max - 1u;
	fit = SparkGlm5NextVerifyRowsFit(position,depth + 1u,split_threshold,max_positions);
	return(fit == 0u ? 0u : fit - 1u);
}

static inline SparkStatus SparkGlm5NextVerifyWaveCheck(uint32_t rows,uint32_t rows_max,uint32_t first_row,uint32_t active_sequences,uint32_t sampled,const uint32_t *resident_slots,const uint32_t *positions,uint32_t split_threshold,uint32_t max_positions)
{
	if ( rows_max == 0u || sampled != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( rows < SPARK_GLM5_NEXT_VERIFY_ROWS_MIN || rows > rows_max || first_row != 0u || active_sequences != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkGlm5NextVerifyRowsAgree(rows,resident_slots,positions) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkGlm5NextVerifyRowsFit(positions[0],rows,split_threshold,max_positions) != rows )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkGlm5NextVerifyCaptureBound(uint32_t regime,uint32_t rows,uint32_t context,uint32_t split_threshold,uint32_t max_positions)
{
	uint32_t bound;
	if ( regime != SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED )
		context = regime == SPARK_GLM5_NEXT_GRAPH_REGIME_UNSPLIT ? 1u : split_threshold;
	if ( context == 0u || context > max_positions || SparkGlm5NextGraphRegime(context,split_threshold) != regime )
		return(0u);
	bound = SparkGlm5NextGraphBound(context,split_threshold,max_positions);
	if ( bound < rows || SparkGlm5NextVerifyRowsFit(bound - rows,rows,split_threshold,max_positions) != rows )
		return(0u);
	return(bound);
}
