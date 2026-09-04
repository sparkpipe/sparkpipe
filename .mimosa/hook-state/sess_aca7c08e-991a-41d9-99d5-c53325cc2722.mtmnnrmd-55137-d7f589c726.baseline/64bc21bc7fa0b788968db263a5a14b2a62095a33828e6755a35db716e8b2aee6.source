#ifndef SPARKPIPE_SPARK_HARDWARE_TRANSPORT_PROBE_H
#define SPARKPIPE_SPARK_HARDWARE_TRANSPORT_PROBE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION 4u
#define SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_SYMBOL \
    "SparkHardwareTransportProbeGetInterface"
#define SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES 96u
#define SPARK_HARDWARE_TRANSPORT_PROBE_SHA256_BYTES 65u
#define SPARK_HARDWARE_TRANSPORT_PROBE_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkHardwareTransportProbeRequest))
#define SPARK_HARDWARE_TRANSPORT_PROBE_RESULT_BYTES \
    ((uint32_t)sizeof(SparkHardwareTransportProbeResult))
#define SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_BYTES \
    ((uint32_t)sizeof(SparkHardwareTransportProbeInterface))

#define SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_EXACT_PRODUCTION_TRANSPORT 0x00000001u
#define SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_VERBS_PATH 0x00000002u
#define SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REMOTE_COMPLETION 0x00000004u
#define SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_TCP_PATH 0x00000008u
#define SPARK_HARDWARE_TRANSPORT_PROVIDER_KNOWN_FLAGS \
    (SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_EXACT_PRODUCTION_TRANSPORT | \
     SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_VERBS_PATH | \
     SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REMOTE_COMPLETION | \
     SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_TCP_PATH)

#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH 0x00000001u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR 0x00000002u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY 0x00000004u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR 0x00000008u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_INTEGRITY_VERIFIED 0x00000010u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REMOTE_COMPLETION_VERIFIED 0x00000020u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_EXACT_PRODUCTION_TRANSPORT 0x00000040u
#define SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH 0x00000080u
#define SPARK_HARDWARE_TRANSPORT_RESULT_KNOWN_FLAGS \
    (SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_INTEGRITY_VERIFIED | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REMOTE_COMPLETION_VERIFIED | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_EXACT_PRODUCTION_TRANSPORT | \
     SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH)

typedef struct SparkHardwareTransportProbeRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t iterations;
    uint32_t payload_bytes;
    uint32_t lane_count;
    uint32_t window_depth;
    uint32_t cq_batch;
    uint32_t registered_region_count;
    uint32_t local_rank;
    uint32_t peer_rank;
    uint32_t reserved_u32[2];
    char question_id[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
    char candidate[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
    char progress_mode[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
    char node[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
    char peer[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
    char topology[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES];
} SparkHardwareTransportProbeRequest;

typedef struct SparkHardwareTransportProbeResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkStatus status;
    uint32_t result_flags;
    char local_transport_artifact_sha256[SPARK_HARDWARE_TRANSPORT_PROBE_SHA256_BYTES];
    char peer_transport_artifact_sha256[SPARK_HARDWARE_TRANSPORT_PROBE_SHA256_BYTES];
    uint64_t sample_count;
    uint64_t transferred_bytes;
    uint64_t latency_min_ns;
    uint64_t latency_p50_ns;
    uint64_t latency_p95_ns;
    uint64_t latency_p99_ns;
    uint64_t latency_max_ns;
    uint64_t setup_time_ns;
    uint64_t registration_time_ns;
    uint64_t deregistration_time_ns;
    double throughput_gb_s;
    double message_rate_per_second;
    double cpu_percent;
    uint64_t cq_poll_count;
    uint64_t cq_wakeup_count;
    uint64_t mr_registration_count;
    uint64_t mr_cache_hit_count;
    uint64_t mr_eviction_count;
    uint64_t retry_count;
    uint64_t integrity_error_count;
    uint64_t source_fingerprint;
    uint64_t destination_fingerprint;
    uint32_t numerical_pass;
    uint32_t integrity_pass;
    uint32_t reserved_u32[6];
} SparkHardwareTransportProbeResult;

typedef SparkStatus (*SparkHardwareTransportProbeRunFunction)(
    const SparkHardwareTransportProbeRequest *request,
    SparkHardwareTransportProbeResult *result);

typedef struct SparkHardwareTransportProbeInterface
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t reserved;
    const char *provider_id;
    const char *provider_build_identity;
    SparkHardwareTransportProbeRunFunction run;
    uint64_t reserved_u64[4];
} SparkHardwareTransportProbeInterface;

typedef const SparkHardwareTransportProbeInterface *
    (*SparkHardwareTransportProbeGetInterfaceFunction)(void);

#ifdef __cplusplus
}
#endif

#endif
