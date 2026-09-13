#ifndef SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_PROBE_H
#define SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_PROBE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION 1u
#define SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_SYMBOL "SparkHardwareTopologyProbeGetInterface"
#define SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES 96u
#define SPARK_HARDWARE_TOPOLOGY_PROBE_SHA256_BYTES 65u

#define SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_EXACT_PRODUCTION_PATH 0x00000001u
#define SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_ALL_RANK_COMMIT 0x00000002u
#define SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_FINAL_EVENT_ACK 0x00000004u
#define SPARK_HARDWARE_TOPOLOGY_PROVIDER_KNOWN_FLAGS \
    (SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_EXACT_PRODUCTION_PATH | \
     SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_ALL_RANK_COMMIT | \
     SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_FINAL_EVENT_ACK)

#define SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_EXACT_PRODUCTION_PATH 0x00000001u
#define SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_ALL_RANKS_PARTICIPATED 0x00000002u
#define SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_GLOBAL_COMMIT_VERIFIED 0x00000004u
#define SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_FINAL_EVENT_ACK_VERIFIED 0x00000008u
#define SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_INTEGRITY_VERIFIED 0x00000010u
#define SPARK_HARDWARE_TOPOLOGY_RESULT_KNOWN_FLAGS \
    (SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_EXACT_PRODUCTION_PATH | \
     SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_ALL_RANKS_PARTICIPATED | \
     SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_GLOBAL_COMMIT_VERIFIED | \
     SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_FINAL_EVENT_ACK_VERIFIED | \
     SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_INTEGRITY_VERIFIED)

typedef struct SparkHardwareTopologyProbeRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t batch_size;
    uint32_t context_tokens;
    uint32_t pipeline_degree;
    uint32_t window_depth;
    uint32_t iterations;
    uint32_t reserved_u32;
    char question_id[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES];
    char topology[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES];
    char node[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES];
    char model_id[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES];
    char candidate[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES];
} SparkHardwareTopologyProbeRequest;

typedef struct SparkHardwareTopologyProbeResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkStatus status;
    uint32_t result_flags;
    char topology_artifact_sha256[SPARK_HARDWARE_TOPOLOGY_PROBE_SHA256_BYTES];
    uint64_t sample_count;
    uint64_t latency_min_ns;
    uint64_t latency_p50_ns;
    uint64_t latency_p95_ns;
    uint64_t latency_p99_ns;
    uint64_t latency_max_ns;
    double tokens_per_second;
    double rows_per_second;
    double network_gb_s;
    double cpu_percent;
    uint64_t retry_count;
    uint64_t duplicate_count;
    uint64_t integrity_error_count;
    uint32_t rank_count;
    uint32_t completion_count;
    uint32_t numerical_pass;
    uint32_t integrity_pass;
    uint32_t reserved_u32[4];
} SparkHardwareTopologyProbeResult;

typedef SparkStatus (*SparkHardwareTopologyProbeRunFunction)(
    const SparkHardwareTopologyProbeRequest *request,
    SparkHardwareTopologyProbeResult *result);

typedef struct SparkHardwareTopologyProbeInterface
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t reserved_u32;
    const char *provider_id;
    const char *provider_build_identity;
    SparkHardwareTopologyProbeRunFunction run;
    uint64_t reserved_u64[4];
} SparkHardwareTopologyProbeInterface;

typedef const SparkHardwareTopologyProbeInterface *
    (*SparkHardwareTopologyProbeGetInterfaceFunction)(void);

#define SPARK_HARDWARE_TOPOLOGY_PROBE_REQUEST_BYTES ((uint32_t)sizeof(SparkHardwareTopologyProbeRequest))
#define SPARK_HARDWARE_TOPOLOGY_PROBE_RESULT_BYTES ((uint32_t)sizeof(SparkHardwareTopologyProbeResult))
#define SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_BYTES ((uint32_t)sizeof(SparkHardwareTopologyProbeInterface))

#ifdef __cplusplus
}
#endif

#endif
