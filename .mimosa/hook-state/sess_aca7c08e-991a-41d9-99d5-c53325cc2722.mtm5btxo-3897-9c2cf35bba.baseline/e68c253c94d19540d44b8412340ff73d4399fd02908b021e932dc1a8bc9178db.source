#pragma once


#include <stdint.h>

#include "sparkpipe/spark_nvme_tier.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TOPOLOGY_SWITCH_ABI_VERSION 1u
#define SPARK_TOPOLOGY_SWITCH_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkTopologySwitchConfiguration))

typedef enum SparkTopologyStrategy
{
	SPARK_TOPOLOGY_STRATEGY_TENSOR_PARALLEL = 0,
	SPARK_TOPOLOGY_STRATEGY_PIPELINE_PARALLEL = 1
}
SparkTopologyStrategy;

typedef struct SparkTopologyRecipe
{
	uint64_t recipe_id;
	uint64_t weight_pack_bytes;
	uint32_t strategy;
	uint32_t reserved0;
}
SparkTopologyRecipe;

typedef enum SparkTopologySwitchState
{
	SPARK_TOPOLOGY_SWITCH_STEADY = 0,
	SPARK_TOPOLOGY_SWITCH_QUIESCE,
	SPARK_TOPOLOGY_SWITCH_CHECKPOINT,
	SPARK_TOPOLOGY_SWITCH_SWAP,
	SPARK_TOPOLOGY_SWITCH_RESUME
}
SparkTopologySwitchState;

typedef enum SparkTopologySwitchResumeClass
{
	SPARK_TOPOLOGY_SWITCH_RESUME_PENDING = 0,
	SPARK_TOPOLOGY_SWITCH_RESUME_WARM,
	SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE
}
SparkTopologySwitchResumeClass;

typedef SparkStatus (*SparkTopologySwitchBeginSwap)(
	void *context,
	const SparkTopologyRecipe *from,
	const SparkTopologyRecipe *target);
typedef SparkStatus (*SparkTopologySwitchPollSwap)(void *context);

typedef struct SparkTopologySwitchSwapDevice
{
	void *context;
	SparkTopologySwitchBeginSwap begin_swap;
	SparkTopologySwitchPollSwap poll_swap;
}
SparkTopologySwitchSwapDevice;

typedef SparkStatus (*SparkTopologySwitchWriteBlock)(
	void *context,
	uint64_t device_offset,
	const void *payload,
	uint32_t bytes);

typedef struct SparkTopologySwitchConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t kv_namespace;
	uint32_t max_sequences;
	uint32_t max_blocks_per_sequence;
	uint32_t step_time_microseconds;
	uint32_t manifest_block_bytes;
	uint64_t nvme_read_bytes_per_second;
	uint64_t nvme_write_bytes_per_second;
	uint64_t swap_fixed_microseconds;
	SparkNvmeTier *tier;
	SparkTopologyRecipe initial_recipe;
}
SparkTopologySwitchConfiguration;

typedef struct SparkTopologySwitchBudget
{
	uint64_t quiesce_us;
	uint64_t checkpoint_us;
	uint64_t swap_fixed_us;
	uint64_t swap_stream_us;
	uint64_t resume_warm_us;
	uint64_t total_us;
}
SparkTopologySwitchBudget;

typedef struct SparkTopologySwitchStatistics
{
	uint64_t switches_completed;
	uint64_t sequences_checkpointed;
	uint64_t sequences_resumed_warm;
	uint64_t sequences_resumed_recompute;
	uint64_t sequences_completed_mid_switch;
	uint64_t blocks_pinned;
	uint64_t manifest_writes;
	uint64_t manifest_bytes;
	uint64_t tier_blocks_found;
	uint64_t tier_blocks_absent;
	uint64_t swaps_started;
}
SparkTopologySwitchStatistics;

typedef struct SparkTopologySwitch SparkTopologySwitch;
struct SparkTopologySwitch
{
	SparkTopologySwitchConfiguration configuration;
	SparkTopologySwitchSwapDevice swap_device;
	SparkTopologySwitchWriteBlock write_block;
	void *write_block_context;
	void *sequences;
	uint64_t *block_keys;
	uint8_t *manifest_buffer;
	SparkTopologyRecipe current_recipe;
	SparkTopologyRecipe target_recipe;
	uint32_t state;
	uint32_t sequence_count;
	uint32_t phase_cursor;
	uint32_t swap_started;
	SparkStatus last_error;
	SparkTopologySwitchStatistics statistics;
};

uint64_t SparkTopologySwitchTableBytes(
	const SparkTopologySwitchConfiguration *configuration);

SparkStatus SparkTopologySwitchInitialize(
	SparkTopologySwitch *sw,
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologySwitchSwapDevice *swap_device,
	SparkTopologySwitchWriteBlock write_block,
	void *write_block_context,
	void *tables);

uint64_t SparkTopologySwitchKvKey(
	uint64_t kv_namespace,
	uint64_t content_hash);

uint32_t SparkTopologySwitchAdmissionsOpen(
	const SparkTopologySwitch *sw);

SparkStatus SparkTopologySwitchTrackSequence(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint64_t recipe_id);

SparkStatus SparkTopologySwitchSetSequenceKv(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint32_t position_tokens,
	const uint64_t *content_hashes,
	uint32_t hash_count);

SparkStatus SparkTopologySwitchSequenceAtBoundary(
	SparkTopologySwitch *sw,
	uint64_t sequence_id);

SparkStatus SparkTopologySwitchSequenceComplete(
	SparkTopologySwitch *sw,
	uint64_t sequence_id);

SparkStatus SparkTopologySwitchBegin(
	SparkTopologySwitch *sw,
	const SparkTopologyRecipe *target);

SparkTopologySwitchState SparkTopologySwitchAdvance(
	SparkTopologySwitch *sw,
	uint32_t step_now);

SparkTopologySwitchState SparkTopologySwitchStateOf(
	const SparkTopologySwitch *sw);

const SparkTopologyRecipe *SparkTopologySwitchCurrentRecipe(
	const SparkTopologySwitch *sw);

SparkTopologySwitchResumeClass SparkTopologySwitchResumeClassOf(
	const SparkTopologySwitch *sw,
	uint64_t sequence_id);

SparkStatus SparkTopologySwitchEstimateBudget(
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologyRecipe *target,
	uint32_t active_sequence_count,
	uint64_t warm_kv_bytes,
	SparkTopologySwitchBudget *budget_out);

void SparkTopologySwitchGetStatistics(
	const SparkTopologySwitch *sw,
	SparkTopologySwitchStatistics *statistics_out);

SparkStatus SparkTopologySwitchLastError(const SparkTopologySwitch *sw);

#ifdef __cplusplus
}
#endif
