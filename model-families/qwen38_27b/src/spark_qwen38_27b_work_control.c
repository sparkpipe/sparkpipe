#include "sparkpipe/spark_qwen38_27b_work_control.h"

#define SPARK_WORK_CONTROL_FN(name) SparkQwen38_27bWorkControl##name
#define SPARK_WORK_CONTROL_TYPE(name) SparkQwen38_27bWorkControl##name
#define SPARK_WORK_CONTROL_CONST(name) SPARK_QWEN38_27B_WORK_CONTROL_##name

#include "sparkpipe/spark_work_control_common.h"
