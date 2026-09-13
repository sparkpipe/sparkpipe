#include "spark_dsv4_rocm_islands.h"
#include "spark_dsv4_rocm_islands.h"
int spark_dsv4_rocm_islands_header_c_driver(void)
{
    return SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16 == 0u &&
           SPARK_DSV4_ROCM_WEIGHT_FORMAT_FP8_E4M3 == 4u ? 0 : 1;
}
