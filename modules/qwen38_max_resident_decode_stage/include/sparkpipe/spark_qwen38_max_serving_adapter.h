#pragma once

#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN38_MAX_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION 3u

const SparkModelServingAdapterInterface *
SparkModelServingAdapterGetInterface(void);

#ifdef __cplusplus
}
#endif
