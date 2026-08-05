#pragma once

#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION 1u

const SparkModelServingAdapterInterface *
SparkModelServingAdapterGetInterface(void);

#ifdef __cplusplus
}
#endif
