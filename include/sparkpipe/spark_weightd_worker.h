#pragma once
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdWorker SparkWeightdWorker;
typedef void (*SparkWeightdWorkFunction)(void *context);
#define SPARK_WEIGHTD_WORK_QUEUE_CAPACITY 64u

SparkStatus SparkWeightdWorkerCreate(SparkWeightdWorker **out);
SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context);
SparkStatus SparkWeightdWorkerWaitIdle(SparkWeightdWorker *worker,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdWorkerDestroy(SparkWeightdWorker *worker);

#ifdef __cplusplus
}
#endif
