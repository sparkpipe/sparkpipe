#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_FIRMWARE_MODULE_ABI_VERSION 4u
#define SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION 4u

typedef struct SparkFirmwareModuleConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t operation_index;
    uint32_t reserved0;
    const char *model_id;
    const char *model_revision;
    const char *stage_name;
    const char *program_name;
    const char *operation_name;
    const char *configuration_json;
    uint32_t configuration_json_bytes;
    uint32_t reserved1;
} SparkFirmwareModuleConfiguration;

typedef struct SparkFirmwareModuleHostServices
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    SparkModelDriverWakeFunction wake_function;
    void *wake_context;
    const char *node_id;
    const char *node_target;
    void *node_context;
    uint32_t kv_logical_page_capacity;
    uint32_t kv_physical_page_capacity;
    const char *kv_backing_directory;
    uint64_t kv_backing_maximum_bytes;
    void *execution_stream;
    uint64_t reserved[1];
} SparkFirmwareModuleHostServices;

static inline SparkStatus SparkFirmwareModuleValidateInitialization(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    if (configuration == 0 || host_services == 0 || module_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *module_state = 0;
    if (configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION ||
        configuration->descriptor_bytes < sizeof(*configuration) ||
        configuration->reserved0 != 0u ||
        configuration->reserved1 != 0u ||
        host_services->abi_version != SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION ||
        host_services->descriptor_bytes < sizeof(*host_services) ||
        host_services->reserved[0] != 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (((host_services->kv_logical_page_capacity == 0u) !=
         (host_services->kv_physical_page_capacity == 0u)) ||
        host_services->kv_physical_page_capacity >
            host_services->kv_logical_page_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

typedef SparkStatus (*SparkFirmwareModuleInitializeFunction)(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

typedef SparkStatus (*SparkFirmwareModuleExecuteFunction)(
    void *module_state,
    SparkModelDriverFrame *frame);

typedef SparkStatus (*SparkFirmwareModuleAdmitFunction)(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

typedef SparkStatus (*SparkFirmwareModuleSnapshotFunction)(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

typedef void (*SparkFirmwareModuleDestroyFunction)(void *module_state);

#ifdef __cplusplus
}

#endif
