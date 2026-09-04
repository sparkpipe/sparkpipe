#ifndef SPARKPIPE_SPARK_DRIVER_LOADER_H
#define SPARKPIPE_SPARK_DRIVER_LOADER_H

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkLoadedModelDriver
{
    void *dynamic_library;
    const SparkModelDriverInterface *interface;
} SparkLoadedModelDriver;

void SparkLoadedModelDriverReset(SparkLoadedModelDriver *driver);
SparkStatus SparkLoadModelDriver(const char *driver_path, const char *expected_target, SparkLoadedModelDriver *driver, char *error_buffer, uint32_t error_buffer_bytes);
void SparkUnloadModelDriver(SparkLoadedModelDriver *driver);
const SparkModelDriverProgramDescriptor *SparkFindLoadedModelDriverProgram(const SparkLoadedModelDriver *driver, const char *program_name);

#ifdef __cplusplus
}
#endif

#endif
