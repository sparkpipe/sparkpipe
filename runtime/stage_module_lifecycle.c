/*
 * Shared SparkFirmwareModule lifecycle implementation. See
 * include/sparkpipe/spark_stage_module_lifecycle.h for the contract and the
 * paste this replaces (one lifecycle per family, five copies at extraction
 * time). The per-family hooks own everything model-specific; this file owns
 * the ABI shell around them and nothing else.
 */

#include <stdlib.h>

#include "sparkpipe/spark_stage_module_lifecycle.h"

static void SparkStageModuleLifecycleInitializeCounters(
    SparkStageModuleLifecycle *lifecycle)
{
    atomic_init(lifecycle->submitted_count, 0u);
    atomic_init(lifecycle->completed_count, 0u);
    atomic_init(lifecycle->rejected_count, 0u);
    atomic_init(lifecycle->failed_count, 0u);
    atomic_init(lifecycle->tokens_emitted, 0u);
}

SparkStatus SparkStageModuleLifecycleInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state,
    const SparkStageModuleLifecycleOps *ops)
{
    SparkStageModuleLifecycle lifecycle;
    void *state;
    SparkStatus status;

    status = SparkFirmwareModuleValidateInitialization(
        configuration,
        host_services,
        module_state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (ops->initialize_gate != 0)
    {
        status = ops->initialize_gate();
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    state = calloc(1u, (size_t)ops->state_bytes);
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    ops->describe(state, &lifecycle);
    lifecycle.ledger->module_tag = lifecycle.module_tag;
    SparkStageModuleLifecycleInitializeCounters(&lifecycle);
    status = ops->state_prepare(state, host_services);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleLifecycleDestroy(state, ops);
        return status;
    }
    if (ops->state_report_ready != 0)
    {
        ops->state_report_ready(state);
    }
    *module_state = state;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleLifecycleExecute(
    void *module_state,
    SparkModelDriverFrame *frame,
    const SparkStageModuleLifecycleOps *ops)
{
    if (module_state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return ops->execute(module_state, frame);
}

SparkStatus SparkStageModuleLifecycleAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision,
    const SparkStageModuleLifecycleOps *ops)
{
    if (module_state == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return ops->admit(module_state, request, decision);
}

SparkStatus SparkStageModuleLifecycleSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot,
    const SparkStageModuleLifecycleOps *ops)
{
    SparkStageModuleLifecycle lifecycle;

    if (module_state == 0 || snapshot == 0 || program_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    ops->describe(module_state, &lifecycle);
    SparkStageModuleRuntimeSnapshotInitialize(
        snapshot,
        program_id,
        lifecycle.slot_states,
        lifecycle.pipeline_slot_count);
    snapshot->submitted_count = atomic_load_explicit(
        lifecycle.submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        lifecycle.completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        lifecycle.rejected_count,
        memory_order_relaxed);
    if (ops->snapshot_extend != 0)
    {
        ops->snapshot_extend(module_state, snapshot);
    }
    return SPARK_STATUS_OK;
}

void SparkStageModuleLifecycleDestroy(
    void *module_state,
    const SparkStageModuleLifecycleOps *ops)
{
    SparkStageModuleLifecycle lifecycle;

    if (module_state == 0)
    {
        return;
    }
    ops->describe(module_state, &lifecycle);
    if (SparkStageModuleWaitForSlots(
            lifecycle.module_tag,
            lifecycle.slot_states,
            lifecycle.pipeline_slot_count,
            SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK)
    {
        return;
    }
    if (ops->state_destroy != 0)
    {
        ops->state_destroy(module_state);
    }
    SparkStageModuleLedgerRelease(lifecycle.ledger);
    free(module_state);
}
