#include <string.h>

#include "sparkpipe/spark_hardware_transport_probe.h"

static SparkStatus SparkFakeTransportRun(
    const SparkHardwareTransportProbeRequest *request,
    SparkHardwareTransportProbeResult *result)
{
    result->status = SPARK_STATUS_OK;
    result->result_flags =
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_INTEGRITY_VERIFIED |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REMOTE_COMPLETION_VERIFIED |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_EXACT_PRODUCTION_TRANSPORT;
    (void)strcpy(result->local_transport_artifact_sha256,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    (void)strcpy(result->peer_transport_artifact_sha256,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    result->sample_count = request->iterations;
    result->transferred_bytes = (uint64_t)request->payload_bytes * request->iterations;
    result->latency_min_ns = 500u;
    result->latency_p50_ns = 1000u;
    result->latency_p95_ns = 1500u;
    result->latency_p99_ns = 2000u;
    result->latency_max_ns = 2500u;
    result->setup_time_ns = 10000u;
    result->registration_time_ns = 3000u;
    result->deregistration_time_ns = 2000u;
    result->throughput_gb_s = 12.0;
    result->message_rate_per_second = 1000000.0;
    result->cpu_percent = 5.0;
    result->cq_poll_count = request->iterations;
    result->cq_wakeup_count = 1u;
    result->mr_registration_count = 1u;
    result->mr_cache_hit_count = request->iterations - 1u;
    result->source_fingerprint = 0x123456789abcdef0ULL;
    result->destination_fingerprint = result->source_fingerprint;
    result->numerical_pass = 1u;
    result->integrity_pass = 1u;
    return SPARK_STATUS_OK;
}

static const SparkHardwareTransportProbeInterface SparkFakeTransportInterface =
{
    SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION,
    SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_BYTES,
    SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_EXACT_PRODUCTION_TRANSPORT |
        SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_VERBS_PATH |
        SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REMOTE_COMPLETION,
    0u,
    "fake_transport_provider",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    SparkFakeTransportRun,
    {0u, 0u, 0u, 0u}
};

const SparkHardwareTransportProbeInterface *SparkHardwareTransportProbeGetInterface(void)
{
    return &SparkFakeTransportInterface;
}
