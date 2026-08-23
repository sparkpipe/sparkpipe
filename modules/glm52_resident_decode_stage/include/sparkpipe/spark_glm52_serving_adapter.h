#pragma once

#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* v4: adds the DFlash2 speculator stanza (speculation_enabled,
 * speculation_draft_count, dspark_pack_path). */
#define SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION 4u

const SparkModelServingAdapterInterface *
SparkModelServingAdapterGetInterface(void);

#ifdef __cplusplus
}
#endif
