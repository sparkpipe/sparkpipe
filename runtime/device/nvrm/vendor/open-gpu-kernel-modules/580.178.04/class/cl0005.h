#pragma once

#include <nvtypes.h>

#include "cl0005_notification.h"

#define NV01_EVENT (0x5U)

#define NV0005_ALLOC_PARAMETERS_MESSAGE_ID (0x0005U)

typedef struct NV0005_ALLOC_PARAMETERS {
    NvHandle hParentClient;
    NvHandle hSrcResource;

    NvV32    hClass;
    NvV32    notifyIndex;
    NV_DECLARE_ALIGNED(NvP64 data, 8);
} NV0005_ALLOC_PARAMETERS;

#define NV0005_NOTIFY_INDEX_INDEX     15:0
#define NV0005_NOTIFY_INDEX_SUBDEVICE 23:16
#define NV0005_NOTIFY_INDEX_FLAGS     31:24
