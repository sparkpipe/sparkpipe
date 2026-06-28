#ifndef SPARKPIPE_SPARK_EVENT_PLAN_H
#define SPARKPIPE_SPARK_EVENT_PLAN_H

#include "sparkpipe/spark_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkStageStreamKind
{
    SPARK_STAGE_STREAM_RECV = 1,
    SPARK_STAGE_STREAM_KV = 2,
    SPARK_STAGE_STREAM_COMPUTE = 3,
    SPARK_STAGE_STREAM_SEND = 4,
    SPARK_STAGE_STREAM_MAINTENANCE = 5
} SparkStageStreamKind;

typedef enum SparkStageEventKind
{
    SPARK_STAGE_EVENT_RECV_READY = 1,
    SPARK_STAGE_EVENT_KV_READY = 2,
    SPARK_STAGE_EVENT_COMPUTE_DONE = 3,
    SPARK_STAGE_EVENT_SEND_READY = 4
} SparkStageEventKind;

typedef struct SparkStageEventRecord
{
    bool initialized;
    bool signaled;
    SparkStageEventKind event_kind;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint64_t event_id;
    uint64_t payload_checksum;
    uint64_t record_checksum;
} SparkStageEventRecord;

typedef struct SparkStageEventRing
{
    bool initialized;
    uint32_t stage_id;
    uint32_t next_record_index;
    uint64_t signal_count;
    uint64_t ring_checksum;
    SparkStageEventRecord records[SPARKPIPE_STAGE_EVENT_RING_CAPACITY];
} SparkStageEventRing;

typedef struct SparkStageEventDependencyReport
{
    bool initialized;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint32_t recv_stream_id;
    uint32_t kv_stream_id;
    uint32_t compute_stream_id;
    uint32_t send_stream_id;
    uint32_t maintenance_stream_id;
    bool requires_recv_ready;
    bool requires_kv_ready;
    bool recv_ready_satisfied;
    bool kv_ready_satisfied;
    bool compute_done_emitted;
    bool send_ready_emitted;
    uint64_t recv_ready_event_id;
    uint64_t kv_ready_event_id;
    uint64_t compute_done_event_id;
    uint64_t send_ready_event_id;
    uint32_t dependency_count;
    uint32_t event_count;
    uint32_t missing_recv_ready_count;
    uint32_t missing_kv_ready_count;
    uint32_t stale_event_violation_count;
    uint32_t dependency_violation_count;
    uint64_t event_ring_checksum_before;
    uint64_t event_ring_checksum_after;
    uint64_t event_dependency_checksum;
} SparkStageEventDependencyReport;

uint64_t SparkBuildStageStreamId(SparkStageStreamKind stream_kind, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation);
uint64_t SparkBuildStageEventId(SparkStageEventKind event_kind, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, uint64_t fabric_tick, uint64_t payload_checksum);
uint64_t SparkComputeStageEventRecordChecksum(const SparkStageEventRecord *event_record);
uint64_t SparkComputeStageEventRingChecksum(const SparkStageEventRing *event_ring);
uint64_t SparkComputeStageEventDependencyReportChecksum(const SparkStageEventDependencyReport *dependency_report);
SparkStatus SparkInitializeStageEventRing(SparkStageEventRing *event_ring, uint32_t stage_id);
SparkStatus SparkSignalStageEvent(SparkStageEventRing *event_ring, SparkStageEventKind event_kind, uint64_t fabric_tick, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, uint64_t payload_checksum, uint64_t *event_id_out);
SparkStatus SparkBuildStageEventDependencyReport(SparkStageEventDependencyReport *dependency_report, SparkStageEventRing *event_ring, uint64_t fabric_tick, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, bool requires_recv_ready, bool requires_kv_ready, uint64_t recv_payload_checksum, uint64_t kv_payload_checksum);
SparkStatus SparkValidateStageEventDependencyReport(const SparkStageEventDependencyReport *dependency_report);

#ifdef __cplusplus
}
#endif

#endif
