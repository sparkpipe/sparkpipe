#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

SparkStatus SparkTestAddOneInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);
SparkStatus SparkTestAddOneExecute(void *module_state, SparkModelDriverFrame *frame);
void SparkTestAddOneDestroy(void *module_state);

SparkStatus SparkTestAddOneInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    SparkStatus status;

    status = SparkFirmwareModuleValidateInitialization(
        configuration,
        host_services,
        module_state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *module_state = (void *)(uintptr_t)1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTestAddOneExecute(void *module_state, SparkModelDriverFrame *frame)
{
    uint64_t increment;

    if (module_state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    increment = (uint64_t)(uintptr_t)module_state;
    frame->scalar[0] += increment;
    return SPARK_STATUS_OK;
}

void SparkTestAddOneDestroy(void *module_state)
{
    (void)module_state;
}
