#include "sparkpipe/spark_status.h"

const char *SparkStatusToString(SparkStatus status)
{
    switch (status)
    {
        case SPARK_STATUS_OK:
        {
            return "ok";
        }
        case SPARK_STATUS_INVALID_ARGUMENT:
        {
            return "invalid_argument";
        }
        case SPARK_STATUS_CAPACITY_EXCEEDED:
        {
            return "capacity_exceeded";
        }
        case SPARK_STATUS_NOT_FOUND:
        {
            return "not_found";
        }
        case SPARK_STATUS_IO_ERROR:
        {
            return "io_error";
        }
        case SPARK_STATUS_PARSE_ERROR:
        {
            return "parse_error";
        }
        case SPARK_STATUS_SCHEMA_ERROR:
        {
            return "schema_error";
        }
        case SPARK_STATUS_HASH_MISMATCH:
        {
            return "hash_mismatch";
        }
        case SPARK_STATUS_MODULE_NOT_VALIDATED:
        {
            return "module_not_validated";
        }
        case SPARK_STATUS_VALIDATION_FAILED:
        {
            return "validation_failed";
        }
        case SPARK_STATUS_ABI_MISMATCH:
        {
            return "abi_mismatch";
        }
        case SPARK_STATUS_TARGET_MISMATCH:
        {
            return "target_mismatch";
        }
        case SPARK_STATUS_COMPILER_ERROR:
        {
            return "compiler_error";
        }
        case SPARK_STATUS_DRIVER_LOAD_ERROR:
        {
            return "driver_load_error";
        }
        case SPARK_STATUS_ROUTE_NOT_FOUND:
        {
            return "route_not_found";
        }
        case SPARK_STATUS_BUSY:
        {
            return "busy";
        }
        case SPARK_STATUS_DUPLICATE:
        {
            return "duplicate";
        }
        case SPARK_STATUS_INTERNAL_ERROR:
        {
            return "internal_error";
        }
        default:
        {
            return "unknown_status";
        }
    }
}
