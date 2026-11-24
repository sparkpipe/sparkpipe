#ifndef NVGPUTYPES_INCLUDED
#define NVGPUTYPES_INCLUDED
#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"

typedef volatile struct NvNotificationRec {
 struct {
  NvU32 nanoseconds[2];
 } timeStamp;
 NvV32 info32;
 NvV16 info16;
 NvV16 status;
} NvNotification;

typedef volatile struct NvGpuSemaphoreRec {
 NvV32 data[2];
 struct {
  NvV32 nanoseconds[2];
 } timeStamp;
} NvGpuSemaphore;

typedef volatile struct NVGetReportRec {
    struct  {
        NvU32 nanoseconds[2];
    } timeStamp;
    NvU32 value;
    NvU32 zero;
} NvGetReport;

typedef struct NvRcNotificationRec {
    struct {
        NvU32 nanoseconds[2];
    } timeStamp;
    NvU32 exceptLevel;
    NvU32 exceptType;
} NvRcNotification;

typedef struct NvSyncPointFenceRec {
    NvU32   syncPointID;
    NvU32   value;
} NvSyncPointFence;

#define NV_INVALID_SYNCPOINT_ID ((NvU32)-1)

typedef NvU64           NvOffset;

#define NvOffset_HI32(n)  ((NvU32)(((NvU64)(n)) >> 32))
#define NvOffset_LO32(n)  ((NvU32)((NvU64)(n)))

#define NV_GPU_UUID_SHA1_LEN            (16)
#define NV_GPU_UUID_SHA256_LEN          (32)
#define NV_GPU_UUID_LEN                 NV_GPU_UUID_SHA1_LEN

#ifdef __cplusplus
};
#endif

#endif
