#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WORKING_STATE_DIRECTIVE_VERSION 1u
#define SPARK_WORKING_STATE_DIRECTIVE_BYTES_MAX 256u
#define SPARK_WORKING_STATE_PATH_BYTES 1024u
#define SPARK_WORKING_STATE_SECONDS_MAX UINT64_C(1099511627776)
#define SPARK_WORKING_STATE_ROOT "sparkdata/working-state"
#define SPARK_WORKING_STATE_DIRECTIVE_NAME "directive"
#define SPARK_WORKING_STATE_SCRATCH_NAME "scratch"
#define SPARK_WORKING_STATE_MIRROR_NAME "mirror"
#define SPARK_WORKING_STATE_LAST_USE_NAME ".last-use"
#define SPARK_WORKING_STATE_SCRATCH_DEPTH_MAX 128u

typedef struct SparkWorkingStateDirective
{
	uint64_t wiped_seconds;
	uint64_t held_until_seconds;
} SparkWorkingStateDirective;

typedef struct SparkWorkingStatePurgeResult
{
	uint64_t deleted_entries;
	uint64_t deleted_bytes;
	uint32_t areas;
	uint32_t emptied_areas;
	uint32_t failed_unlinks;
	uint32_t reserved0;
} SparkWorkingStatePurgeResult;

SparkStatus SparkWorkingStateOwnerRoot(char *path,uint32_t capacity);
SparkStatus SparkWorkingStateDirectivePath(const char *home,char *path,uint32_t capacity);
SparkStatus SparkWorkingStateDirectiveParse(const char *text,uint32_t bytes,SparkWorkingStateDirective *directive);
SparkStatus SparkWorkingStateDirectiveRead(const char *path,SparkWorkingStateDirective *directive);
uint32_t SparkWorkingStateDirectiveEqual(const SparkWorkingStateDirective *left,const SparkWorkingStateDirective *right);
uint32_t SparkWorkingStateExpired(const SparkWorkingStateDirective *directive,uint64_t used_seconds,uint64_t now_seconds);
SparkStatus SparkWorkingStatePurgeScratchArea(const char *area,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkWorkingStatePurgeResult *result);
SparkStatus SparkWorkingStatePurgeScratch(const char *home,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkWorkingStatePurgeResult *result);

#ifdef __cplusplus
}
#endif
