#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_driver_loader.h"

static void SparkPrintUsage(const char *program_name)
{
    fprintf(stderr, "usage: %s DRIVER_SO [EXPECTED_TARGET]\n", program_name);
}

int main(int argument_count, char **arguments)
{
    SparkLoadedModelDriver driver;
    const SparkModelDriverDescriptor *descriptor;
    const char *expected_target;
    char error_buffer[1024];
    uint32_t program_index;
    SparkStatus status;

    if (argument_count < 2 || argument_count > 3)
    {
        SparkPrintUsage(arguments[0]);
        return 2;
    }
    expected_target = argument_count == 3 ? arguments[2] : 0;
    SparkLoadedModelDriverReset(&driver);
    status = SparkLoadModelDriver(arguments[1], expected_target, &driver, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "driver load failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        return 1;
    }
    descriptor = driver.interface->descriptor;
    printf(
        "model=%s revision=%s stage=%s target=%s programs=%u modules=%u model_sha256=%s compiled_program_sha256=%s\n",
        descriptor->model_id,
        descriptor->model_revision,
        descriptor->stage_name,
        descriptor->target,
        descriptor->program_count,
        descriptor->module_instance_count,
        descriptor->model_description_sha256,
        descriptor->compiled_program_sha256);
    for (program_index = 0u; program_index < descriptor->program_count; ++program_index)
    {
        const SparkModelDriverProgramDescriptor *program;
        const SparkModelDriverProgramProfile *profile;

        program = &descriptor->programs[program_index];
        profile = program->profile;
        printf(
            "program id=%u name=%s max_inflight=%u completion=%s flags=0x%08x max_active_slots=%u max_new_tokens=%u resident_sequences=%u max_sequence_tokens=%llu target_latency_ns=%llu validated_latency_ns=%llu device_memcpy_ceiling=%llu host_staging_ceiling=%llu private_queues=%u\n",
            program->program_id,
            program->name,
            program->max_inflight,
            (program->flags & SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION) != 0u ? "external" : "submit_return",
            program->flags,
            profile != 0 ? profile->max_active_slots : 0u,
            profile != 0 ? profile->max_new_tokens : 0u,
            profile != 0 ? profile->max_resident_sequences : 0u,
            profile != 0 ? (unsigned long long)profile->max_sequence_tokens : 0ull,
            profile != 0 ? (unsigned long long)profile->target_latency_ns : 0ull,
            profile != 0 ? (unsigned long long)profile->validated_latency_ns : 0ull,
            profile != 0 ? (unsigned long long)profile->device_memcpy_bytes_per_submit_ceiling : 0ull,
            profile != 0 ? (unsigned long long)profile->host_staging_bytes_per_submit_ceiling : 0ull,
            profile != 0 ? profile->private_queue_count : 0u);
    }
    SparkUnloadModelDriver(&driver);
    return 0;
}
