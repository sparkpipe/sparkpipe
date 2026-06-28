#ifndef SPARKPIPE_SPARK_COMMON_H
#define SPARKPIPE_SPARK_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_MAX_STAGES 32
#define SPARKPIPE_MAX_ACTIVATION_RINGS (SPARKPIPE_MAX_STAGES + 1)
#define SPARKPIPE_ACTIVATION_RING_CAPACITY 8
#define SPARKPIPE_MAX_MODEL_LANES 8
#define SPARKPIPE_MODEL_LANE_COUNT 3
#define SPARKPIPE_MAX_PHYSICAL_SLOTS 128
#define SPARKPIPE_PHYSICAL_PROFILE_COUNT 12
#define SPARKPIPE_MAX_READY_REQUESTS 4096
#define SPARKPIPE_MAX_KV_LEASES_PER_STAGE SPARKPIPE_MAX_PHYSICAL_SLOTS
#define SPARKPIPE_MAX_KV_LEASES (SPARKPIPE_MAX_STAGES * SPARKPIPE_MAX_PHYSICAL_SLOTS)
#define SPARKPIPE_MAX_KV_LOOKAHEAD_TICKETS SPARKPIPE_MAX_STAGES
#define SPARKPIPE_MAX_COMPLETED_SLOTS_PER_COMMIT SPARKPIPE_MAX_PHYSICAL_SLOTS
#define SPARKPIPE_MAX_CONSTRAINT_TOKENS 128u
#define SPARKPIPE_STAGE_PACK_MAGIC 0x53504B32u
#define SPARKPIPE_STAGE_PACK_VERSION 3u
#define SPARKPIPE_STAGE_PACK_HARDWARE_TARGET_SIMULATED 1u
#define SPARKPIPE_STAGE_PACK_MAX_PAYLOAD_ENTRIES 16u
#define SPARKPIPE_STAGE_PACK_PAYLOAD_ALIGNMENT_BYTES 4096u
#define SPARKPIPE_INITIAL_ACTIVATION_CHECKSUM 0x535041524B504950ull
#define SPARKPIPE_CONSTRAINED_CHOICE_DEFAULT_TOKENS 4u
#define SPARKPIPE_CONSTRAINED_TRIE_DEFAULT_EDGES 32u
#define SPARKPIPE_DEFAULT_KV_STAGE_HOP_TICKS 1u
#define SPARKPIPE_DEFAULT_KV_WARM_BUDGET_MULTIPLIER 2u
#define SPARKPIPE_ACTIVE_MASK_WORD_BITS 64u
#define SPARKPIPE_ACTIVE_MASK_WORD_COUNT ((SPARKPIPE_MAX_PHYSICAL_SLOTS + SPARKPIPE_ACTIVE_MASK_WORD_BITS - 1u) / SPARKPIPE_ACTIVE_MASK_WORD_BITS)
#define SPARKPIPE_ACTIVATION_ALIGNMENT_BYTES 128u
#define SPARKPIPE_DEFAULT_HIDDEN_SIZE 8192u
#define SPARKPIPE_ACTIVATION_ELEMENT_BYTES_BF16 2u
#define SPARKPIPE_SIMULATED_ACTIVATION_BUFFER_BYTES (8u * 1024u)
#define SPARKPIPE_KV_PAGE_BYTES 16384u
#define SPARKPIPE_MAX_KV_PAGE_TABLE_ENTRIES_PER_STAGE 16384u
#define SPARKPIPE_WORKSPACE_ALIGNMENT_BYTES 256u
#define SPARKPIPE_CUDA_STAGE_LAUNCH_DESCRIPTOR_VERSION 4u
#define SPARKPIPE_CUDA_STAGE_BACKEND_C_DUMMY 1u
#define SPARKPIPE_CUDA_STAGE_BACKEND_OPTIONAL_CUDA_DUMMY 2u
#define SPARKPIPE_CUDA_STAGE_BACKEND_PRODUCTION_ADAPTER 3u
#define SPARKPIPE_STAGE_ACTIVATION_POOL_BUFFER_COUNT 2u
#define SPARKPIPE_STAGE_EVENT_RING_CAPACITY 16u
#define SPARKPIPE_STAGE_STREAM_COUNT 5u
#define SPARKPIPE_ACTIVATION_TRANSPORT_SLOT_COUNT 4u
#define SPARKPIPE_ACTIVATION_TRANSPORT_CHANNEL_COUNT SPARKPIPE_MAX_ACTIVATION_RINGS
#define SPARKPIPE_WIRE_PROTOCOL_MAGIC 0x53505752u
#define SPARKPIPE_WIRE_PROTOCOL_VERSION 1u
#define SPARKPIPE_WIRE_MESSAGE_HEADER_BYTES 48u
#define SPARKPIPE_WIRE_MAX_PAYLOAD_BYTES 2048u
#define SPARKPIPE_WIRE_LOOPBACK_QUEUE_CAPACITY 128u

typedef enum SparkModelLaneKind
{
    SPARK_MODEL_LANE_FRONTIER = 0,
    SPARK_MODEL_LANE_NEAR_FRONTIER = 1,
    SPARK_MODEL_LANE_AGENT = 2
} SparkModelLaneKind;

typedef enum SparkRequestMode
{
    SPARK_REQUEST_MODE_FREE_FORM = 0,
    SPARK_REQUEST_MODE_CONSTRAINED_CHOICE = 1,
    SPARK_REQUEST_MODE_CONSTRAINED_TRIE = 2
} SparkRequestMode;

typedef enum SparkSchedulingPolicy
{
    SPARK_SCHEDULING_POLICY_LATENCY_FIRST = 0,
    SPARK_SCHEDULING_POLICY_THROUGHPUT_FIRST = 1,
    SPARK_SCHEDULING_POLICY_CONSTRAINED_FAST = 2
} SparkSchedulingPolicy;

typedef enum SparkPhysicalProfileId
{
    SPARK_PROFILE_DECODE_B1_LONG = 0,
    SPARK_PROFILE_DECODE_B8_TOOL = 1,
    SPARK_PROFILE_DECODE_B64_AGENT = 2,
    SPARK_PROFILE_DECODE_B96_AGENT = 3,
    SPARK_PROFILE_DECODE_B128_THROUGHPUT = 4,
    SPARK_PROFILE_PREFILL_2048 = 5,
    SPARK_PROFILE_PREFILL_8192 = 6,
    SPARK_PROFILE_PREFILL_65536 = 7,
    SPARK_PROFILE_CONSTRAINED_B1_K4 = 8,
    SPARK_PROFILE_CONSTRAINED_B96_K8 = 9,
    SPARK_PROFILE_CONSTRAINED_B96_K32 = 10,
    SPARK_PROFILE_CONSTRAINED_B96_TRIE = 11
} SparkPhysicalProfileId;

typedef struct SparkActiveSlotMask
{
    uint32_t physical_slot_count;
    uint32_t active_slot_count;
    uint64_t generation;
    uint64_t words[SPARKPIPE_ACTIVE_MASK_WORD_COUNT];
} SparkActiveSlotMask;

typedef struct SparkPhysicalSlotMapping
{
    uint32_t physical_slot_count;
    uint32_t mapped_slot_count;
    uint64_t generation;
    uint32_t slot_ids[SPARKPIPE_MAX_PHYSICAL_SLOTS];
} SparkPhysicalSlotMapping;

typedef struct SparkPhysicalProfile
{
    SparkPhysicalProfileId profile_id;
    const char *profile_name;
    uint32_t physical_slot_count;
    uint32_t context_band_pages;
    uint32_t kv_hot_window_pages_per_slot;
    bool constrained_output;
} SparkPhysicalProfile;

typedef struct SparkRequestSpec
{
    uint64_t request_id;
    SparkModelLaneKind model_lane;
    SparkRequestMode request_mode;
    uint32_t target_output_tokens;
    uint32_t context_pages;
    uint32_t priority;
    uint32_t constraint_candidate_count;
    uint32_t constraint_trie_depth;
} SparkRequestSpec;

typedef struct SparkRunCounters
{
    uint64_t fabric_ticks;
    uint64_t graph_launches;
    uint64_t committed_tokens;
    uint64_t completed_requests;
    uint64_t constrained_completions;
    uint64_t constrained_tokens_committed;
    uint64_t slot_refills;
    uint64_t kv_pages_loaded;
    uint64_t kv_pages_released;
    uint64_t kv_pages_promoted;
    uint64_t kv_pages_evicted;
    uint64_t kv_cancelled_pages;
    uint64_t kv_prepare_failures;
    uint64_t kv_leases_created;
    uint64_t kv_leases_reused;
    uint64_t kv_leases_canceled;
    uint64_t graph_failures;
    uint64_t profile_downshifts;
    uint64_t profile_upshifts;
} SparkRunCounters;

const SparkPhysicalProfile *SparkGetPhysicalProfile(SparkPhysicalProfileId profile_id);
const char *SparkModelLaneKindToString(SparkModelLaneKind model_lane);
const char *SparkRequestModeToString(SparkRequestMode request_mode);
const char *SparkSchedulingPolicyToString(SparkSchedulingPolicy scheduling_policy);
uint32_t SparkGetModelLaneIndex(SparkModelLaneKind model_lane);
bool SparkIsValidModelLane(SparkModelLaneKind model_lane);
bool SparkPhysicalProfileIsSmallerThan(const SparkPhysicalProfile *left_profile, const SparkPhysicalProfile *right_profile);
uint64_t SparkAlignUpU64(uint64_t value, uint64_t alignment);
uint64_t SparkMixU64(uint64_t checksum, uint64_t value);
SparkPhysicalProfileId SparkSelectEffectiveProfile(SparkPhysicalProfileId lane_profile_id, SparkSchedulingPolicy scheduling_policy, uint32_t active_slots, bool constrained_work_present);
SparkPhysicalProfileId SparkSelectHysteresisProfile(SparkPhysicalProfileId configured_profile_id, SparkPhysicalProfileId previous_profile_id, SparkSchedulingPolicy scheduling_policy, uint32_t active_slots, bool constrained_work_present);

#ifdef __cplusplus
}
#endif

#endif
