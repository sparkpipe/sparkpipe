#pragma once
#include "sparkpipe/spark_weightd.h"
#ifdef __cplusplus
extern "C" {
#endif
void SparkTestMeshWaitInitialize(void *region,uint32_t degree);
void SparkTestMeshWaitPoll(uint64_t now_ns);
#ifdef __cplusplus
}
#endif
