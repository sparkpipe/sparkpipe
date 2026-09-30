#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

#define SPARK_SPECULATION_VERIFY_ROWS_MIN 2u
#define SPARK_SPECULATION_VERIFY_DRAFTER_NONE 0u
#define SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP 1u
#define SPARK_SPECULATION_VERIFY_DRAFTER_ORACLE 2u
#define SPARK_SPECULATION_VERIFY_DRAFTER_ADVERSARY 3u
#define SPARK_SPECULATION_VERIFY_DRAFTER_RECORDED 4u
#define SPARK_SPECULATION_VERIFY_ORACLE_PREFIX "oracle:"
#define SPARK_SPECULATION_VERIFY_ADVERSARY_PREFIX "adversary:"
#define SPARK_SPECULATION_VERIFY_RECORDED_PREFIX "recorded:"

static inline SparkStatus SparkSpeculationVerifyRowsParse(const char *text,uint32_t rows_limit,uint32_t *rows_out)
{
	uint32_t rows;
	*rows_out = 0u;
	if ( text == 0 )
		return(SPARK_STATUS_OK);
	if ( text[0] < '0' || text[0] > '9' || text[1] != '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows = (uint32_t)(text[0] - '0');
	if ( rows != 0u && (rows < SPARK_SPECULATION_VERIFY_ROWS_MIN || rows > rows_limit) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*rows_out = rows;
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkSpeculationVerifyPrefixed(const char *text,const char *prefix,const char **path_out)
{
	size_t length = strlen(prefix);
	if ( strncmp(text,prefix,length) != 0 || text[length] == '\0' )
		return(0u);
	*path_out = text + length;
	return(1u);
}

static inline SparkStatus SparkSpeculationVerifyDrafterParse(const char *text,uint32_t rows,uint32_t *kind_out,const char **path_out)
{
	*kind_out = SPARK_SPECULATION_VERIFY_DRAFTER_NONE;
	*path_out = 0;
	if ( rows == 0u )
		return(text == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( text == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( strcmp(text,"lookup") == 0 )
		*kind_out = SPARK_SPECULATION_VERIFY_DRAFTER_LOOKUP;
	else if ( SparkSpeculationVerifyPrefixed(text,SPARK_SPECULATION_VERIFY_ORACLE_PREFIX,path_out) != 0u )
		*kind_out = SPARK_SPECULATION_VERIFY_DRAFTER_ORACLE;
	else if ( SparkSpeculationVerifyPrefixed(text,SPARK_SPECULATION_VERIFY_ADVERSARY_PREFIX,path_out) != 0u )
		*kind_out = SPARK_SPECULATION_VERIFY_DRAFTER_ADVERSARY;
	else if ( SparkSpeculationVerifyPrefixed(text,SPARK_SPECULATION_VERIFY_RECORDED_PREFIX,path_out) != 0u )
		*kind_out = SPARK_SPECULATION_VERIFY_DRAFTER_RECORDED;
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkSpeculationVerifyDepth(uint32_t budget,uint32_t produced,uint32_t rows_max,uint32_t fit_rows)
{
	uint32_t depth;
	if ( rows_max < SPARK_SPECULATION_VERIFY_ROWS_MIN || produced >= budget || budget - produced < 2u || fit_rows < 2u )
		return(0u);
	depth = budget - produced - 1u;
	if ( depth > rows_max - 1u )
		depth = rows_max - 1u;
	if ( depth > fit_rows - 1u )
		depth = fit_rows - 1u;
	return(depth);
}
