#pragma once


#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVME_TIER_ABI_VERSION 3u
#define SPARK_NVME_TIER_DIGEST_BYTES SPARK_SHA256_DIGEST_BYTES
#define SPARK_NVME_TIER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkNvmeTierConfiguration))
#define SPARK_NVME_TIER_DEFAULT_BUDGET_BYTES (1099511627776ULL)
#define SPARK_NVME_TIER_DEFAULT_DEVICE_BYTES_PER_SECOND (5000000000ULL)
#define SPARK_NVME_TIER_DEFAULT_STEP_TIME_MICROSECONDS 67700u
#define SPARK_NVME_TIER_NO_SLOT 0xffffffffu
#define SPARK_NVME_TIER_MAX_STAGING_BUFFERS 16u
#define SPARK_NVME_TIER_DEFAULT_PENDING_CAPACITY 256u
#define SPARK_NVME_TIER_IO_ALIGNMENT_BYTES 4096u

typedef struct SparkNvmeTierNeed
{
	uint64_t content_hash;
	uint32_t need_by_step;
	uint32_t reserved0;
	uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES];
}
SparkNvmeTierNeed;

typedef struct SparkNvmeTierConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t budget_bytes;
	uint64_t base_offset;
	uint32_t block_bytes;
	uint32_t hash_bucket_count;
	uint32_t staging_buffer_count;
	uint32_t demand_reserve_buffers;
	uint32_t pending_capacity;
	uint64_t device_bytes_per_second;
	uint32_t step_time_microseconds;
	uint32_t reserved0;
}
SparkNvmeTierConfiguration;

typedef SparkStatus (*SparkNvmeTierSubmitRead)(
	void *context, uint64_t device_offset, void *destination,
	uint32_t bytes, uint64_t *ticket_out);
typedef SparkStatus (*SparkNvmeTierPollRead)(void *context, uint64_t ticket);
typedef SparkStatus (*SparkNvmeTierCancelRead)(void *context, uint64_t ticket);

typedef struct SparkNvmeTierDevice
{
	void *context;
	SparkNvmeTierSubmitRead submit_read;
	SparkNvmeTierPollRead poll_read;
	SparkNvmeTierCancelRead cancel_read;
}
SparkNvmeTierDevice;

typedef enum SparkNvmeTierDemandState
{
	SPARK_NVME_TIER_DEMAND_READY = 0,
	SPARK_NVME_TIER_DEMAND_IN_FLIGHT,
	SPARK_NVME_TIER_DEMAND_STARTED,
	SPARK_NVME_TIER_DEMAND_MISS
}
SparkNvmeTierDemandState;


typedef struct SparkNvmeTierWriteReservation
{
	uint64_t content_hash;
	uint64_t device_offset;
	uint32_t slot_index;
	uint32_t generation;
	uint32_t already_present;
	uint32_t reserved0;
	uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES];
}
SparkNvmeTierWriteReservation;

typedef struct SparkNvmeTierDemandResult
{
	SparkNvmeTierDemandState state;
	uint32_t ordered;
	void *staging_pointer;
}
SparkNvmeTierDemandResult;

typedef enum SparkNvmeTierConfidence
{
	SPARK_NVME_TIER_CONFIDENCE_ALL = 0,
	SPARK_NVME_TIER_CONFIDENCE_PARTIAL,
	SPARK_NVME_TIER_CONFIDENCE_NONE
}
SparkNvmeTierConfidence;

typedef struct SparkNvmeTierResidencyAssessment
{
	SparkNvmeTierConfidence confidence;
	uint32_t ready_count;
	uint32_t inflight_confident_count;
	uint32_t planned_confident_count;
	uint32_t late_count;
	uint32_t absent_count;
	uint32_t reserved0;
}
SparkNvmeTierResidencyAssessment;

typedef struct SparkNvmeTierStatistics
{
	uint64_t write_reservations;
	uint64_t publishes;
	uint64_t write_aborts;
	uint64_t cancel_pending_count;
	uint64_t evictions;
	uint64_t pinned_eviction_skips;
	uint64_t demand_hits;
	uint64_t demand_joins;
	uint64_t demand_loads;
	uint64_t demand_misses;
	uint64_t demand_stalls;
	uint64_t prefetch_issues;
	uint64_t prefetch_landings;
	uint64_t prefetch_late_landings;
	uint64_t prefetch_preemptions;
	uint64_t prefetch_dropped;
	uint64_t stale_completions;
	uint64_t io_errors;
	uint64_t read_bytes;
	uint64_t slot_count;
	uint64_t slots_in_use;
	uint64_t digest_verifications;
	uint64_t digest_mismatches;
	uint64_t demand_deadline_orders;
}
SparkNvmeTierStatistics;

typedef struct SparkNvmeTier SparkNvmeTier;
struct SparkNvmeTier
{
	SparkNvmeTierConfiguration configuration;
	SparkNvmeTierDevice device;
	void *slots;
	uint32_t *buckets;
	void *pending;
	uint8_t *staging;
	void *staging_state;
	uint32_t slot_count;
	uint32_t free_head;
	uint32_t clock_hand;
	uint32_t slots_in_use;
	uint64_t tick;
	uint32_t bytes_per_step;
	uint32_t transfer_steps;
	SparkNvmeTierStatistics statistics;
};

uint64_t SparkNvmeTierTableBytes(
	const SparkNvmeTierConfiguration *configuration);

SparkStatus SparkNvmeTierInitialize(
	SparkNvmeTier *tier,
	const SparkNvmeTierConfiguration *configuration,
	const SparkNvmeTierDevice *device,
	void *tables,
	uint64_t tables_bytes,
	void *staging,
	uint64_t staging_bytes);

SparkStatus SparkNvmeTierReserveWrite(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	SparkNvmeTierWriteReservation *reservation_out);

SparkStatus SparkNvmeTierCommitWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation);

SparkStatus SparkNvmeTierAbortWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation);

SparkStatus SparkNvmeTierOffsetOf(
	const SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	uint64_t *device_offset_out);

typedef struct SparkNvmeTierPlanReport
{
	uint32_t already_ready_count;
	uint32_t already_inflight_count;
	uint32_t queued_count;
	uint32_t absent_count;
	uint32_t queue_full_count;
	uint32_t late_risk_count;
}
SparkNvmeTierPlanReport;

SparkStatus SparkNvmeTierPlanLookahead(
	SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	SparkNvmeTierPlanReport *report_out);

SparkStatus SparkNvmeTierRequestDemand(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	uint32_t step_now,
	SparkNvmeTierDemandResult *result_out);

SparkStatus SparkNvmeTierRequestDemandDeadline(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	uint32_t step_now,
	uint32_t deadline_step,
	SparkNvmeTierDemandResult *result_out);

SparkStatus SparkNvmeTierPump(SparkNvmeTier *tier, uint32_t step_now);

SparkStatus SparkNvmeTierConsume(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES]);

SparkStatus SparkNvmeTierPin(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	int32_t pin);

SparkStatus SparkNvmeTierWillBeResidentBy(
	const SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	uint32_t step_deadline,
	SparkNvmeTierResidencyAssessment *assessment_out);

void SparkNvmeTierGetStatistics(
	const SparkNvmeTier *tier,
	SparkNvmeTierStatistics *statistics_out);

#ifdef __cplusplus
}
#endif
