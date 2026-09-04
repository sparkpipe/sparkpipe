#pragma once

#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_stage_module_common.h"


typedef struct SparkStageModuleLifecycle
{
    const char *module_tag;
    SparkStageModuleLedger *ledger;
    atomic_uint *slot_states;
    uint32_t pipeline_slot_count;
    atomic_ullong *submitted_count;
    atomic_ullong *completed_count;
    atomic_ullong *rejected_count;
    atomic_ullong *failed_count;
    atomic_ullong *tokens_emitted;
} SparkStageModuleLifecycle;

typedef struct SparkStageModuleLifecycleOps
{
    uint32_t state_bytes;
    SparkStatus (*initialize_gate)(void);
    void (*describe)(void *state, SparkStageModuleLifecycle *lifecycle);
    SparkStatus (*state_prepare)(void *state,
        const SparkFirmwareModuleConfiguration *configuration,
        const SparkFirmwareModuleHostServices *host_services);
    void (*state_report_ready)(void *state);
    void (*state_destroy)(void *state);
    SparkStatus (*execute)(void *state, SparkModelDriverFrame *frame);
    SparkStatus (*admit)(void *state,
        const SparkModelDriverAdmissionRequest *request,
        SparkModelDriverAdmissionDecision *decision);
    void (*snapshot_extend)(void *state,
        SparkModelDriverRuntimeSnapshot *snapshot);
} SparkStageModuleLifecycleOps;

SparkStatus SparkStageModuleLifecycleInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state,
    const SparkStageModuleLifecycleOps *ops);

SparkStatus SparkStageModuleLifecycleExecute(
    void *module_state,
    SparkModelDriverFrame *frame,
    const SparkStageModuleLifecycleOps *ops);

SparkStatus SparkStageModuleLifecycleAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision,
    const SparkStageModuleLifecycleOps *ops);

SparkStatus SparkStageModuleLifecycleSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot,
    const SparkStageModuleLifecycleOps *ops);

void SparkStageModuleLifecycleDestroy(
    void *module_state,
    const SparkStageModuleLifecycleOps *ops);

#define SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(entry_prefix, ops_address) \
    SparkStatus entry_prefix##Initialize( \
        const SparkFirmwareModuleConfiguration *configuration, \
        const SparkFirmwareModuleHostServices *host_services, \
        void **module_state) \
    { \
        return SparkStageModuleLifecycleInitialize( \
            configuration, host_services, module_state, (ops_address)); \
    } \
    SparkStatus entry_prefix##Execute( \
        void *module_state, \
        SparkModelDriverFrame *frame) \
    { \
        return SparkStageModuleLifecycleExecute( \
            module_state, frame, (ops_address)); \
    } \
    SparkStatus entry_prefix##Admit( \
        void *module_state, \
        const SparkModelDriverAdmissionRequest *request, \
        SparkModelDriverAdmissionDecision *decision) \
    { \
        return SparkStageModuleLifecycleAdmit( \
            module_state, request, decision, (ops_address)); \
    } \
    SparkStatus entry_prefix##Snapshot( \
        void *module_state, \
        uint32_t program_id, \
        SparkModelDriverRuntimeSnapshot *snapshot) \
    { \
        return SparkStageModuleLifecycleSnapshot( \
            module_state, program_id, snapshot, (ops_address)); \
    } \
    void entry_prefix##Destroy(void *module_state) \
    { \
        SparkStageModuleLifecycleDestroy(module_state, (ops_address)); \
    }
