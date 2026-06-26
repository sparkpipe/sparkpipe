#include "sparkpipe/spark_driver_loader.h"

#include <dlfcn.h>
#include <string.h>

#include "spark_filesystem.h"

void SparkLoadedModelDriverReset(SparkLoadedModelDriver *driver)
{
    if (driver == 0)
    {
        return;
    }
    driver->dynamic_library = 0;
    driver->interface = 0;
}

static SparkStatus SparkValidateLoadedModelDriverInterface(
    const SparkModelDriverInterface *interface,
    const char *expected_target,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    const SparkModelDriverDescriptor *descriptor;
    uint32_t program_index;

    if (interface == 0 || interface->abi_version != SPARK_MODEL_DRIVER_ABI_VERSION ||
        interface->interface_bytes < sizeof(*interface) || interface->descriptor == 0 ||
        interface->create == 0 || interface->destroy == 0 || interface->admit == 0 ||
        interface->snapshot == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "model driver interface ABI is invalid");
        return SPARK_STATUS_ABI_MISMATCH;
    }
    descriptor = interface->descriptor;
    if (descriptor->abi_version != SPARK_MODEL_DRIVER_ABI_VERSION ||
        descriptor->descriptor_bytes < sizeof(*descriptor) || descriptor->model_id == 0 ||
        descriptor->model_revision == 0 || descriptor->stage_name == 0 || descriptor->target == 0 ||
        descriptor->model_description_sha256 == 0 || descriptor->compiled_program_sha256 == 0 ||
        descriptor->program_count == 0u || descriptor->programs == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "model driver descriptor ABI is invalid");
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (expected_target != 0 && expected_target[0] != '\0' && strcmp(descriptor->target, expected_target) != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver target '%s' does not match node target '%s'", descriptor->target, expected_target);
        return SPARK_STATUS_TARGET_MISMATCH;
    }
    for (program_index = 0u; program_index < descriptor->program_count; ++program_index)
    {
        const SparkModelDriverProgramDescriptor *program;
        uint32_t previous_program_index;

        program = &descriptor->programs[program_index];
        if (program->program_id == 0u || program->name == 0 || program->name[0] == '\0' ||
            program->submit == 0 || program->max_inflight == 0u || program->profile == 0 ||
            program->profile->descriptor_bytes < sizeof(*program->profile) ||
            program->profile->max_inflight != program->max_inflight)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "driver program descriptor %u is invalid", program_index);
            return SPARK_STATUS_ABI_MISMATCH;
        }
        for (previous_program_index = 0u; previous_program_index < program_index; ++previous_program_index)
        {
            const SparkModelDriverProgramDescriptor *previous_program;

            previous_program = &descriptor->programs[previous_program_index];
            if (previous_program->program_id == program->program_id || strcmp(previous_program->name, program->name) == 0)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "driver contains duplicate program descriptors");
                return SPARK_STATUS_ABI_MISMATCH;
            }
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkLoadModelDriver(
    const char *driver_path,
    const char *expected_target,
    SparkLoadedModelDriver *driver,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    void *dynamic_library;
    SparkModelDriverGetInterfaceFunction get_interface;
    const SparkModelDriverInterface *interface;
    const char *dynamic_error;
    SparkStatus status;

    if (driver_path == 0 || driver == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    SparkUnloadModelDriver(driver);

    dlerror();
    dynamic_library = dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    if (dynamic_library == 0)
    {
        dynamic_error = dlerror();
        SparkSetError(error_buffer, error_buffer_bytes, "cannot load model driver '%s': %s", driver_path, dynamic_error != 0 ? dynamic_error : "unknown loader error");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }

    dlerror();
    *(void **)(&get_interface) = dlsym(dynamic_library, SPARK_MODEL_DRIVER_INTERFACE_SYMBOL);
    dynamic_error = dlerror();
    if (dynamic_error != 0 || get_interface == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver '%s' does not export %s", driver_path, SPARK_MODEL_DRIVER_INTERFACE_SYMBOL);
        dlclose(dynamic_library);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    interface = get_interface();
    status = SparkValidateLoadedModelDriverInterface(interface, expected_target, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        dlclose(dynamic_library);
        return status;
    }

    driver->dynamic_library = dynamic_library;
    driver->interface = interface;
    return SPARK_STATUS_OK;
}

void SparkUnloadModelDriver(SparkLoadedModelDriver *driver)
{
    if (driver == 0)
    {
        return;
    }
    if (driver->dynamic_library != 0)
    {
        dlclose(driver->dynamic_library);
    }
    SparkLoadedModelDriverReset(driver);
}

const SparkModelDriverProgramDescriptor *SparkFindLoadedModelDriverProgram(
    const SparkLoadedModelDriver *driver,
    const char *program_name)
{
    uint32_t program_index;

    if (driver == 0 || driver->interface == 0 || driver->interface->descriptor == 0 || program_name == 0)
    {
        return 0;
    }
    for (program_index = 0u; program_index < driver->interface->descriptor->program_count; ++program_index)
    {
        const SparkModelDriverProgramDescriptor *program;

        program = &driver->interface->descriptor->programs[program_index];
        if (strcmp(program->name, program_name) == 0)
        {
            return program;
        }
    }
    return 0;
}
