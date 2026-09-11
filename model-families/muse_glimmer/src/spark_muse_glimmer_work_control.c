#include "sparkpipe/spark_muse_glimmer_work_control.h"

#define SPARK_WORK_CONTROL_FN(name) SparkMuseGlimmerWorkControl##name
#define SPARK_WORK_CONTROL_TYPE(name) SparkMuseGlimmerWorkControl##name
#define SPARK_WORK_CONTROL_CONST(name) SPARK_MUSE_GLIMMER_WORK_CONTROL_##name

#include "sparkpipe/spark_work_control_common.h"
