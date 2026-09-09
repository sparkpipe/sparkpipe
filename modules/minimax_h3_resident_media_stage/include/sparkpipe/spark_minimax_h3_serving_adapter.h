#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_minimax_h3_resident_media_stage_firmware.h"

#define SPARK_MINIMAX_H3_MEDIA_JOB_ID_BYTES 48u
#define SPARK_MINIMAX_H3_MEDIA_PROMPT_BYTES 4096u
#define SPARK_MINIMAX_H3_MEDIA_MAX_STEPS \
	SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_MAX_STEPS

#define SPARK_MINIMAX_H3_MEDIA_JOB_QUEUED 0u
#define SPARK_MINIMAX_H3_MEDIA_JOB_RUNNING 1u
#define SPARK_MINIMAX_H3_MEDIA_JOB_COMPLETED 2u
#define SPARK_MINIMAX_H3_MEDIA_JOB_FAILED 3u
#define SPARK_MINIMAX_H3_MEDIA_JOB_CANCELLED 4u

#define SPARK_MINIMAX_H3_MEDIA_ARTIFACT_RAW_TAR 0u
#define SPARK_MINIMAX_H3_MEDIA_ARTIFACT_MP4 1u

#define SPARK_MINIMAX_H3_MEDIA_RECEIPT_MAGIC 0x53503348u
#define SPARK_MINIMAX_H3_MEDIA_RECEIPT_VERSION 1u
#define SPARK_MINIMAX_H3_MEDIA_RECEIPT_BYTES \
	(uint32_t)(4u + 4u + 4u + 4u + 8u + 32u + 8u + 8u + 4u + 4u + 4u + 4u + 4u + 4u + 4u)

typedef struct SparkMinimaxH3MediaReceipt
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t state;
	uint32_t artifact_kind;
	uint64_t job_id_u64;
	uint8_t artifact_sha256[32];
	uint64_t artifact_bytes;
	uint64_t step_compute_ns;
	uint32_t frames;
	uint32_t width;
	uint32_t height;
	uint32_t sample_rate_hz;
	uint32_t step_count;
	uint32_t step_index;
	uint32_t reserved;
} SparkMinimaxH3MediaReceipt;

typedef struct SparkMinimaxH3AbiSeamSample
{
	uint64_t submission_to_dispatch_ns;
	uint64_t dispatch_to_completion_ns;
	uint64_t step_compute_ns;
	uint64_t receipt_write_ns;
	uint64_t spool_bytes_written;
	uint32_t admission_payload_bytes;
	uint32_t reserved;
} SparkMinimaxH3AbiSeamSample;

typedef struct SparkMinimaxH3MediaJobSpec
{
	uint64_t job_id_u64;
	int32_t seed;
	uint32_t duration_seconds;
	uint32_t fps;
	uint32_t short_edge;
	uint32_t width;
	uint32_t height;
	uint32_t step_count;
	uint32_t prompt_token_count;
	int32_t prompt_token_ids[SPARK_MINIMAX_H3_RESIDENT_MEDIA_STAGE_MAX_PROMPT_TOKENS];
} SparkMinimaxH3MediaJobSpec;

_Static_assert(sizeof(SparkMinimaxH3MediaReceipt) <=
	SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,
	"h3 media receipt must ride the adapter ABI model extension slot");
