#pragma once

#include <nvtypes.h>

#include "nvlimits.h"
#include "cl0000_notification.h"

#define NV01_NULL_OBJECT (0x0)

#define NV1_NULL_OBJECT  NV01_NULL_OBJECT

#define NV01_ROOT        (0x0U)

#define NV0000_ALLOC_PARAMETERS_MESSAGE_ID (0x0000U)

typedef struct NV0000_ALLOC_PARAMETERS {
    NvHandle hClient;
    NvU32    processID;
    char     processName[NV_PROC_NAME_MAX_LENGTH];
    NV_DECLARE_ALIGNED(NvP64 pOsPidInfo, 8);
} NV0000_ALLOC_PARAMETERS;
