#pragma once

#include <stddef.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_SHORT_NAME_BYTES (SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u)
#define SPARK_NVRM_SHORT_DIGEST_DIGITS 12u

void SparkNvrmShortName(const char *name, char short_name[SPARK_NVRM_SHORT_NAME_BYTES]);

#ifdef __cplusplus
}
#endif
