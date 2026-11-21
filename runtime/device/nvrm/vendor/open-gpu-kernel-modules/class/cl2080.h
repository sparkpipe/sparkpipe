#pragma once

#include <nvtypes.h>

#include "nvlimits.h"
#include "cl2080_notification.h"

#define NV20_SUBDEVICE_0      (0x2080U)

#define NV2080_MAX_SUBDEVICES NV_MAX_SUBDEVICES

#define NV2080_ALLOC_PARAMETERS_MESSAGE_ID (0x2080U)

typedef struct NV2080_ALLOC_PARAMETERS {
    NvU32 subDeviceId;
} NV2080_ALLOC_PARAMETERS;
