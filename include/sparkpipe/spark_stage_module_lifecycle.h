#pragma once

#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_stage_module_common.h"

/*
 * Shared SparkFirmwareModule lifecycle for resident decode stage families.
 *
 * Every stage module publishes the same five ABI entry points and every one
 * of them wrapped the same plumbing: ABI validation, the environment gate, a
 * zeroed state allocation, ledger tagging, the submission counters, the
 * destroy-on-failed-prepare path, the quiesce-then-teardown destroy, and the
 * snapshot's slot/counter base. That shell was pasted per family (five
 * copies) and the paste was where the admission-default-reject bug class was
 * born twice; this header is the single implementation. A family keeps only
 * its genuinely family-specific work - configuration parsing, pack loading,
 * pool allocation, the frame walk, admission policy - as hooks on the ops
 * table, and publishes its entry points through
 * SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS so a new family cannot paste the
 * lifecycle even if it tries.
 *
 * The family state layout is untouched: the shared code never embeds into
 * the state struct, it asks the family to DESCRIBE the common pieces (ledger,
 * slot array, counters) for the state it was handed. Describe is called
 * fresh in every entry point, so values that configuration fills in (the
 * pipeline slot count) are read as they are at that moment - including the
 * all-zero state of a failed prepare, where the quiesce wait trivially
 * passes over zeroed (free) slots.
 */

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
    /* sizeof the family state struct; the lifecycle allocates and frees it. */
    uint32_t state_bytes;
    /* Optional environment gate run BEFORE any allocation (the
     * allow-unqualified-execution refusal). Non-OK aborts initialization. */
    SparkStatus (*initialize_gate)(void);
    /* Fill the lifecycle view for THIS state as it is right now. */
    void (*describe)(void *state, SparkStageModuleLifecycle *lifecycle);
    /* Configuration, pack load, pools, slots: everything between the state
     * allocation and readiness; families that derive configuration from the
     * host services (transport stream, kv backing) receive them here, and
     * the module configuration is forwarded because some families
     * cross-check it against the node context. A non-OK status routes to
     * the full destroy path (quiesce over zeroed slots, family teardown,
     * ledger release). */
    SparkStatus (*state_prepare)(void *state,
        const SparkFirmwareModuleConfiguration *configuration,
        const SparkFirmwareModuleHostServices *host_services);
    /* Optional readiness banner (stderr, family format). */
    void (*state_report_ready)(void *state);
    /* Family teardown AFTER the quiesce wait; the lifecycle releases the
     * ledger and frees the state afterwards. */
    void (*state_destroy)(void *state);
    /* The frame walk. The lifecycle has already rejected null state/frame. */
    SparkStatus (*execute)(void *state, SparkModelDriverFrame *frame);
    /* Admission policy. The lifecycle has already rejected null arguments. */
    SparkStatus (*admit)(void *state,
        const SparkModelDriverAdmissionRequest *request,
        SparkModelDriverAdmissionDecision *decision);
    /* Optional snapshot extension (kv capacity, resident sequence counts);
     * the lifecycle has already written the slot and counter base. */
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

/*
 * Publish the five firmware-module entry points (the names the module
 * library resolves) as delegations to the shared lifecycle. entry_prefix is
 * the family's MODULE_ENTRY_PREFIX (e.g. the family's entry symbol)
 * and ops_address is the address of the family's static ops table.
 */
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
