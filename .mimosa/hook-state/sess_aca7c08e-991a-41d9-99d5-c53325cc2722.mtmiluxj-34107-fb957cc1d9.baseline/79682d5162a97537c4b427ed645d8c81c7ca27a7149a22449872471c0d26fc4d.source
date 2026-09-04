#include "sparkpipe/spark_module_abi.h"

SparkStatus SparkTestDoubleExecute(void *module_state, SparkModelDriverFrame *frame);

SparkStatus SparkTestDoubleExecute(void *module_state, SparkModelDriverFrame *frame)
{
    (void)module_state;

    if (frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    frame->scalar[0] *= 2u;
    return SPARK_STATUS_OK;
}
