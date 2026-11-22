#pragma once

#include <nvtypes.h>

#include "nvlimits.h"
#include "cl0080_notification.h"

#define NV01_DEVICE_0      (0x80U)

#define NV0080_MAX_DEVICES NV_MAX_DEVICES

#define NV0080_ALLOC_PARAMETERS_MESSAGE_ID (0x0080U)

typedef struct NV0080_ALLOC_PARAMETERS {
    NvU32    deviceId;
    NvHandle hClientShare;
    NvHandle hTargetClient;
    NvHandle hTargetDevice;
    NvV32    flags;
    NV_DECLARE_ALIGNED(NvU64 vaSpaceSize, 8);
    NV_DECLARE_ALIGNED(NvU64 vaStartInternal, 8);
    NV_DECLARE_ALIGNED(NvU64 vaLimitInternal, 8);
    NvV32    vaMode;
} NV0080_ALLOC_PARAMETERS;
