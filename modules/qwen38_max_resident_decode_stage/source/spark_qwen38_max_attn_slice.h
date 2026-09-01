/* Host-only staging for the head-split attention projection slices; see
 * the .c for the cut semantics. Split from the module load path so the
 * slice math unit-tests without CUDA. */
#ifndef SPARK_QWEN38_MAX_ATTN_SLICE_H
#define SPARK_QWEN38_MAX_ATTN_SLICE_H

#include <stdint.h>
#include <stdio.h>
#include "sparkpipe/spark_status.h"
#include "spark_qwen38_max_stagepack_format.h"

/* Stage the rank's slice of a full-width attention projection entry from
 * `file` into a malloc'd host buffer. On OK, *staging_out holds the bytes,
 * *staging_bytes_out their length, and *sliced the rank-local entry shape
 * (rows/columns/payload_bytes; scale cleared). The caller uploads and
 * frees. Kinds: ATTN_KEY, ATTN_VALUE (row cut), ATTN_OUTPUT (column cut). */
SparkStatus SparkQwen38MaxModuleStageAttnProjectionSlice(
	FILE *file,
	const SparkQwen38MaxStagePackEntry *entry,
	uint32_t tp_degree,
	uint32_t tp_rank,
	uint16_t **staging_out,
	SparkQwen38MaxStagePackEntry *sliced,
	uint64_t *staging_bytes_out);

#endif
