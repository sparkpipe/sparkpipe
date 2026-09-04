#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_SEAM_ABI_VERSION 1u

#define SPARK_SPECULATION_SEAM_SOURCE_MTP 0x00000001u
#define SPARK_SPECULATION_SEAM_SOURCE_DSPARK 0x00000002u
#define SPARK_SPECULATION_SEAM_SOURCE_DFLASH2 0x00000004u
#define SPARK_SPECULATION_SEAM_SOURCE_NGRAM 0x00000008u
#define SPARK_SPECULATION_SEAM_SOURCE_SUFFIX 0x00000010u
#define SPARK_SPECULATION_SEAM_SOURCE_NGRAM3 0x00000020u

#define SPARK_SPECULATION_SEAM_REMOTE_TAP_FREE_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_NGRAM | \
	 SPARK_SPECULATION_SEAM_SOURCE_SUFFIX | \
	 SPARK_SPECULATION_SEAM_SOURCE_NGRAM3)
#define SPARK_SPECULATION_SEAM_REMOTE_TAP_SOURCES \
	SPARK_SPECULATION_SEAM_SOURCE_DFLASH2
#define SPARK_SPECULATION_SEAM_REMOTE_SOURCES \
	(SPARK_SPECULATION_SEAM_REMOTE_TAP_FREE_SOURCES | \
	 SPARK_SPECULATION_SEAM_REMOTE_TAP_SOURCES)
#define SPARK_SPECULATION_SEAM_KNOWN_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_MTP | \
	 SPARK_SPECULATION_SEAM_SOURCE_DSPARK | \
	 SPARK_SPECULATION_SEAM_REMOTE_SOURCES)

#define SPARK_SPECULATION_SEAM_TARGET_MODEL_BYTES 32u
#define SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkSpeculationSeamConfiguration))

typedef struct SparkSpeculationSeam SparkSpeculationSeam;

typedef struct SparkSpeculationSeamConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t available_source_mask;
	uint32_t default_speculative_token_count;
	uint32_t lane_count;
	uint32_t max_committed_token_count;
	uint32_t max_tap_row_count;
	uint32_t draft_time_budget_ms;
	uint32_t draft_max_depth;
	uint32_t draft_max_node_count;
	uint32_t connect_timeout_ms;
	uint32_t io_timeout_ms;
	const char *control_value;
	const char *bridge_host;
	uint32_t bridge_port;
	char target_model[SPARK_SPECULATION_SEAM_TARGET_MODEL_BYTES];
	SparkSpeculationModelContract model_contract;
} SparkSpeculationSeamConfiguration;

SparkStatus SparkSpeculationSeamParseControl(
	const char *control_value,
	uint32_t available_source_mask,
	uint32_t *enabled_source_mask_out);

SparkStatus SparkSpeculationSeamInitialize(
	const SparkSpeculationSeamConfiguration *configuration,
	SparkSpeculationSeam **seam_out);
void SparkSpeculationSeamDestroy(
	SparkSpeculationSeam *seam);

uint32_t SparkSpeculationSeamEnabledSources(
	const SparkSpeculationSeam *seam);
SparkSpeculationSpeculator *SparkSpeculationSeamSpeculator(
	SparkSpeculationSeam *seam);

SparkStatus SparkSpeculationSeamDraftRemoteChain(
	SparkSpeculationSeam *seam,
	uint64_t request_id,
	uint64_t sequence_id,
	uint64_t position,
	uint32_t anchor_token_id,
	const uint32_t *committed_token_ids,
	uint32_t committed_token_count,
	const void *tap_rows,
	uint32_t tap_row_count,
	uint32_t requested_token_count,
	uint32_t *draft_token_ids_out,
	uint32_t draft_token_id_capacity,
	uint32_t *draft_token_count_out);

SparkStatus SparkSpeculationSeamAcceptChain(
	SparkSpeculationSeam *seam,
	uint64_t sequence_id,
	const uint32_t *verifier_token_ids,
	uint32_t verifier_token_count,
	SparkSpeculationPolicyVerifyResult *verify_result_out);

SparkStatus SparkSpeculationSeamCancelSequence(
	SparkSpeculationSeam *seam,
	uint64_t sequence_id);

SparkStatus SparkSpeculationSeamStageLocalDraft(
	SparkSpeculationSeam *seam,
	uint64_t request_id,
	uint64_t sequence_id,
	uint64_t position,
	const uint32_t *draft_token_ids,
	uint32_t draft_token_count);

#ifdef __cplusplus
}
#endif
