#pragma once
/* Minimal HIP API surface for host-side syntax verification only (no ROCm
 * on this box). Mirrors the subset spark_dsv4_rocm_* uses; NOT for device
 * compilation. */
#include <stdint.h>
#include <stddef.h>
typedef int hipError_t;
typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef struct { int dummy; } hipDeviceProp_t;
#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipStreamDefault 0
#define hipEventDisableTiming 2
hipError_t hipGetLastError(void);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned);

#define hipStreamNonBlocking 1
#define hipHostMallocDefault 0
#define hipHostMallocPortable 1
#define hipHostMallocMapped 2
#define hipErrorNotReady 600
