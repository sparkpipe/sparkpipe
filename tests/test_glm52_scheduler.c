#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_glm52_scheduler.h"

static void SparkTestInitializeSchedulerConfiguration(
    SparkGlm52SchedulerConfiguration *configuration,
    uint32_t quantization_mode)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    configuration->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration->spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    configuration->queue_depth_per_spark = 1u;
    configuration->measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    configuration->quantization_mode = quantization_mode;
}

static void SparkTestInitializeDecodeRequest(
    SparkGlm52SchedulerRequest *request,
    uint32_t active_sequence_count)
{
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    request->active_sequence_count = active_sequence_count;
    request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE;
}

static void SparkTestInitializePrefillRequest(
    SparkGlm52SchedulerRequest *request,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count)
{
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    request->active_sequence_count = active_sequence_count;
    request->prompt_token_count = prompt_token_count;
    request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL;
}

static void SparkTestGlm52SchedulerAdmitsCurrentSparkPp13Decode(void)
{
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t stage_index;

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);
    assert(decision.stage_count == SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT);
    assert(decision.estimated_critical_path_ns != 0u);
    assert(decision.stage_plan.stages[0].first_layer_index == 0u);
    assert(decision.stage_plan.stages[0].layer_count == 6u);
    assert(decision.stage_plan.stages[1].first_layer_index == 6u);
    assert(decision.stage_plan.stages[1].layer_count == 6u);
    assert(decision.stage_plan.stages[12].first_layer_index == 72u);
    assert(decision.stage_plan.stages[12].layer_count == 6u);
    for (stage_index = 0u; stage_index < decision.stage_count; ++stage_index)
    {
        assert(decision.dispatch_stages[stage_index].spark_index == stage_index);
        assert(decision.dispatch_stages[stage_index].estimated_service_time_ns !=
            0u);
        assert(scheduler.spark_inflight_counts[stage_index] == 1u);
    }

    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);
    assert(decision.rejected_status == SPARK_STATUS_BUSY);
    assert(scheduler.rejected_count == 1u);

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);

    SparkTestInitializeDecodeRequest(&request, 64u);
    configuration.queue_depth_per_spark = 2u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    for (stage_index = 0u; stage_index < decision.stage_count; ++stage_index)
    {
        assert(scheduler.spark_inflight_counts[stage_index] == 0u);
    }
}

static void SparkTestGlm52SchedulerSupportsFp8AndPrefill(void)
{
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decode_decision;
    SparkGlm52SchedulerDecision prefill_decision;

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    configuration.queue_depth_per_spark = 2u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializeDecodeRequest(&request, 16u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decode_decision) == SPARK_STATUS_OK);
    assert(decode_decision.accepted == 1u);
    assert(decode_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(decode_decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);

    SparkTestInitializePrefillRequest(&request, 33u, 32u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(prefill_decision.accepted == 1u);
    assert(prefill_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(prefill_decision.estimated_critical_path_ns >=
        decode_decision.estimated_critical_path_ns);

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decode_decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &prefill_decision) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerRejectsInvalidInputs(void)
{
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        99u);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    assert(scheduler.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);

    SparkTestInitializeDecodeRequest(&request, 65u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);
    assert(decision.rejected_status == SPARK_STATUS_CAPACITY_EXCEEDED);
}

int main(void)
{
    SparkTestGlm52SchedulerAdmitsCurrentSparkPp13Decode();
    SparkTestGlm52SchedulerSupportsFp8AndPrefill();
    SparkTestGlm52SchedulerRejectsInvalidInputs();
    return 0;
}
