// spark_hw_rocm_status_string.c - SparkHwStatusToString for the
// rocm.gfx950.mi350p archive.
//
// hwiface_v1.md section 4 provides SparkHwStatusToString in the neutral
// header; section 6 assigns every spark_hw_* symbol to the per-target
// archive, so this definition lives here (the future CUDA archive ships
// its own - exactly one target links per deployment, rule R2/R6).

#include "spark_hw_rocm_internal.h"

const char *SparkHwStatusToString(SparkHwStatus status)
{
    switch (status)
    {
    case SPARK_HW_OK:
        return "SPARK_HW_OK";
    case SPARK_HW_NOT_READY:
        return "SPARK_HW_NOT_READY";
    case SPARK_HW_INVALID:
        return "SPARK_HW_INVALID";
    case SPARK_HW_UNSUPPORTED:
        return "SPARK_HW_UNSUPPORTED";
    case SPARK_HW_EXHAUSTED:
        return "SPARK_HW_EXHAUSTED";
    case SPARK_HW_LOST:
        return "SPARK_HW_LOST";
    default:
        return "SPARK_HW_UNKNOWN"; /* out-of-range enum: never trusted */
    }
}
