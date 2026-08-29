#include "sparkpipe/spark_qwen38_max_work_control.h"

/* The family-prefixed KV work-control API implemented by the shared body
 * (spark_work_control_common.h): this file is the family configuration. */
#define SPARK_WORK_CONTROL_FN(name) SparkQwen38MaxWorkControl##name
#define SPARK_WORK_CONTROL_TYPE(name) SparkQwen38MaxWorkControl##name
#define SPARK_WORK_CONTROL_CONST(name) SPARK_QWEN38_MAX_WORK_CONTROL_##name

#include "sparkpipe/spark_work_control_common.h"
