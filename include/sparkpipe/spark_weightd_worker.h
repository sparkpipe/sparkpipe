#pragma once
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdWorker SparkWeightdWorker;
typedef void (*SparkWeightdWorkFunction)(void *context);
#define SPARK_WEIGHTD_WORK_QUEUE_CAPACITY 64u

// Create on an initialized CUDA context. It must outlive the worker. Functions
// execute serially on that context, outside CUDA/collective callbacks. Functions
// own their operation deadlines, completion notification and lease cleanup.
SparkStatus SparkWeightdWorkerCreate(SparkWeightdWorker **out);
// Nonblocking bounded admission; OK transfers context lifetime until return of
// function. No allocation on submit. Caller must serialize Destroy with Submit.
SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context);
// Quiescence observation with a monotonic timeout. Caller must stop producers
// before relying on idle for teardown. Never wait from the worker itself.
SparkStatus SparkWeightdWorkerWaitIdle(SparkWeightdWorker *worker,uint64_t timeout_nanoseconds);
// BUSY retains the worker while queued or executing work exists. Never call on
// the worker thread. Success joins the idle worker; no forced cancellation.
SparkStatus SparkWeightdWorkerDestroy(SparkWeightdWorker *worker);

#ifdef __cplusplus
}
#endif
