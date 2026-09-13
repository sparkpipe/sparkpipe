#include <string.h>

#include "sparkpipe/spark_hardware_kernel_probe.h"

static SparkStatus SparkFakeModelKernelRun(
    const SparkHardwareKernelProbeRequest *request,
    SparkHardwareKernelProbeResult *result)
{
    (void)request;
    result->status = SPARK_STATUS_OK;
    result->result_flags =
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_EXACT_PRODUCTION_KERNEL |
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_NATIVE_TENSOR_CORE |
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_PTXAS_RECEIPT_BOUND |
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_REFERENCE_INDEPENDENT;
    (void)strcpy(result->kernel_artifact_sha256,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    (void)strcpy(result->ptxas_receipt_sha256,
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
    (void)strcpy(result->reference_artifact_sha256,
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    result->sample_count = request->iterations;
    result->latency_p50_ns = 1000u;
    result->latency_p95_ns = 1500u;
    result->latency_p99_ns = 2000u;
    result->throughput_gb_s = 100.0;
    result->tokens_per_second = 1000.0;
    result->achieved_tflops = 10.0;
    result->maximum_absolute_error = 0.001;
    result->maximum_relative_error = 0.002;
    result->register_count = 64u;
    result->static_shared_bytes = 1024u;
    result->dynamic_shared_bytes = 2048u;
    result->active_blocks_per_sm = 2u;
    result->active_warps_per_sm = 16u;
    result->theoretical_occupancy = 0.5;
    result->achieved_occupancy = 0.45;
    result->numerical_pass = 1u;
    result->integrity_pass = 1u;
    return SPARK_STATUS_OK;
}

static const SparkHardwareKernelProbeInterface SparkFakeModelKernelInterface =
{
    SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION,
    SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_BYTES,
    SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_INDEPENDENT_REFERENCE |
        SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_RESOURCE_RECEIPTS,
    0u,
    "fake_model_kernel_provider",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    SparkFakeModelKernelRun,
    {0u, 0u, 0u, 0u}
};

const SparkHardwareKernelProbeInterface *SparkHardwareKernelProbeGetInterface(void)
{
    return &SparkFakeModelKernelInterface;
}
