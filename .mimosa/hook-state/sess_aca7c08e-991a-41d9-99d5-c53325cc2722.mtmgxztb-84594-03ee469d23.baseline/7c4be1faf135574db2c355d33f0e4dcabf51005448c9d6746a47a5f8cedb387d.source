#include <string.h>

#include "sparkpipe/spark_hardware_topology_probe.h"

static SparkStatus SparkFakeTopologyRun(
    const SparkHardwareTopologyProbeRequest *request,
    SparkHardwareTopologyProbeResult *result)
{
    result->status = SPARK_STATUS_OK;
    result->result_flags =
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_ALL_RANKS_PARTICIPATED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_GLOBAL_COMMIT_VERIFIED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_FINAL_EVENT_ACK_VERIFIED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_INTEGRITY_VERIFIED;
    (void)strcpy(result->topology_artifact_sha256,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    result->sample_count = request->iterations;
    result->latency_min_ns = 100000u;
    result->latency_p50_ns = 200000u;
    result->latency_p95_ns = 250000u;
    result->latency_p99_ns = 300000u;
    result->latency_max_ns = 350000u;
    result->tokens_per_second = 500.0;
    result->rows_per_second = 5000.0;
    result->network_gb_s = 8.0;
    result->cpu_percent = 10.0;
    result->rank_count = request->pipeline_degree;
    result->completion_count = request->iterations;
    result->numerical_pass = 1u;
    result->integrity_pass = 1u;
    return SPARK_STATUS_OK;
}

static const SparkHardwareTopologyProbeInterface SparkFakeTopologyInterface =
{
    SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION,
    SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_BYTES,
    SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_ALL_RANK_COMMIT |
        SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_FINAL_EVENT_ACK,
    0u,
    "fake_topology_provider",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    SparkFakeTopologyRun,
    {0u, 0u, 0u, 0u}
};

const SparkHardwareTopologyProbeInterface *SparkHardwareTopologyProbeGetInterface(void)
{
    return &SparkFakeTopologyInterface;
}
