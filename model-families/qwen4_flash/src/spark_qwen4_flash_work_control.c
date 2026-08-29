#include "sparkpipe/spark_qwen4_flash_work_control.h"

/* The family-prefixed KV work-control API implemented by the shared body
 * (spark_work_control_common.h): this file is the family configuration. */
#define SPARK_WORK_CONTROL_FN(name) SparkQwen4FlashWorkControl##name
#define SPARK_WORK_CONTROL_TYPE(name) SparkQwen4FlashWorkControl##name
#define SPARK_WORK_CONTROL_CONST(name) SPARK_QWEN4_FLASH_WORK_CONTROL_##name

#include "sparkpipe/spark_work_control_common.h"
