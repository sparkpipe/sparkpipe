#ifndef SPARKPIPE_SPARK_HARDWARE_KERNEL_PROBE_H
#define SPARKPIPE_SPARK_HARDWARE_KERNEL_PROBE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION 3u
#define SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_SYMBOL \
    "SparkHardwareKernelProbeGetInterface"
#define SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES 96u
#define SPARK_HARDWARE_KERNEL_PROBE_SHA256_BYTES 65u
#define SPARK_HARDWARE_KERNEL_PROBE_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkHardwareKernelProbeRequest))
#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_BYTES \
    ((uint32_t)sizeof(SparkHardwareKernelProbeResult))
#define SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_BYTES \
    ((uint32_t)sizeof(SparkHardwareKernelProbeInterface))

#define SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_NUMERICAL_REFERENCE 0x00000001u
#define SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_EXACT_PRODUCTION_KERNEL 0x00000002u
#define SPARK_HARDWARE_KERNEL_PROBE_FLAG_COLLECT_RESOURCE_METRICS 0x00000004u
#define SPARK_HARDWARE_KERNEL_PROBE_KNOWN_FLAGS \
    (SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_NUMERICAL_REFERENCE | \
     SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_EXACT_PRODUCTION_KERNEL | \
     SPARK_HARDWARE_KERNEL_PROBE_FLAG_COLLECT_RESOURCE_METRICS)

#define SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_EXACT_PRODUCTION_PATH 0x00000001u
#define SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_INDEPENDENT_REFERENCE 0x00000002u
#define SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_RESOURCE_RECEIPTS 0x00000004u
#define SPARK_HARDWARE_KERNEL_PROVIDER_KNOWN_FLAGS \
    (SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_EXACT_PRODUCTION_PATH | \
     SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_INDEPENDENT_REFERENCE | \
     SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_RESOURCE_RECEIPTS)

#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_EXACT_PRODUCTION_KERNEL 0x00000001u
#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_NATIVE_TENSOR_CORE 0x00000002u
#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_PTXAS_RECEIPT_BOUND 0x00000004u
#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_REFERENCE_INDEPENDENT 0x00000008u
#define SPARK_HARDWARE_KERNEL_PROBE_RESULT_KNOWN_FLAGS \
    (SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_EXACT_PRODUCTION_KERNEL | \
     SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_NATIVE_TENSOR_CORE | \
     SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_PTXAS_RECEIPT_BOUND | \
     SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_REFERENCE_INDEPENDENT)

typedef struct SparkHardwareKernelProbeRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t iterations;
    uint32_t batch_size;
    uint32_t context_tokens;
    uint32_t reserved_u32[2];
    char question_id[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
    char model_id[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
    char role[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
    char candidate[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
    char kernel_class[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
    char route_distribution[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES];
} SparkHardwareKernelProbeRequest;

typedef struct SparkHardwareKernelProbeResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkStatus status;
    uint32_t result_flags;
    char kernel_artifact_sha256[SPARK_HARDWARE_KERNEL_PROBE_SHA256_BYTES];
    char ptxas_receipt_sha256[SPARK_HARDWARE_KERNEL_PROBE_SHA256_BYTES];
    char reference_artifact_sha256[SPARK_HARDWARE_KERNEL_PROBE_SHA256_BYTES];
    uint64_t sample_count;
    uint64_t latency_p50_ns;
    uint64_t latency_p95_ns;
    uint64_t latency_p99_ns;
    double throughput_gb_s;
    double tokens_per_second;
    double achieved_tflops;
    double maximum_absolute_error;
    double maximum_relative_error;
    uint32_t register_count;
    uint32_t static_shared_bytes;
    uint32_t dynamic_shared_bytes;
    uint32_t local_bytes_per_thread;
    uint32_t spill_load_bytes_per_thread;
    uint32_t spill_store_bytes_per_thread;
    uint32_t active_blocks_per_sm;
    uint32_t active_warps_per_sm;
    double theoretical_occupancy;
    double achieved_occupancy;
    uint32_t numerical_pass;
    uint32_t integrity_pass;
    uint32_t reserved_u32[6];
} SparkHardwareKernelProbeResult;

typedef SparkStatus (*SparkHardwareKernelProbeRunFunction)(
    const SparkHardwareKernelProbeRequest *request,
    SparkHardwareKernelProbeResult *result);

typedef struct SparkHardwareKernelProbeInterface
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t reserved;
    const char *provider_id;
    const char *provider_build_identity;
    SparkHardwareKernelProbeRunFunction run;
    uint64_t reserved_u64[4];
} SparkHardwareKernelProbeInterface;

typedef const SparkHardwareKernelProbeInterface *
    (*SparkHardwareKernelProbeGetInterfaceFunction)(void);

#ifdef __cplusplus
}
#endif

#endif
