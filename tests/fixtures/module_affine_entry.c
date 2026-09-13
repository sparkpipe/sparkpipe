#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

uint64_t SparkTestAffineApply(uint64_t value, uint64_t multiplier, uint64_t increment);
SparkStatus SparkTestAffineArchiveExecute(void *module_state, SparkModelDriverFrame *frame);

SparkStatus SparkTestAffineArchiveExecute(void *module_state, SparkModelDriverFrame *frame)
{
    (void)module_state;

    if (frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    frame->scalar[0] = SparkTestAffineApply(frame->scalar[0], 3u, 5u);
    return SPARK_STATUS_OK;
}
