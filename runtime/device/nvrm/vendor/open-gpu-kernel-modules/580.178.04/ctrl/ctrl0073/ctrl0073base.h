#pragma once

#include "ctrl/ctrlxxxx.h"

#define NV0073_CTRL_CMD(cat,idx)                NVXXXX_CTRL_CMD(0x0073, NV0073_CTRL_##cat, idx)

#define NV0073_CTRL_RESERVED (0x00U)
#define NV0073_CTRL_SYSTEM   (0x01U)
#define NV0073_CTRL_SPECIFIC (0x02U)
#define NV0073_CTRL_EVENT    (0x03U)
#define NV0073_CTRL_INTERNAL (0x04U)
#define NV0073_CTRL_COMMON   (0x05U)
#define NV0073_CTRL_DFP      (0x11U)
#define NV0073_CTRL_DP       (0x13U)
#define NV0073_CTRL_SVP      (0x14U)
#define NV0073_CTRL_DPU      (0x15U)
#define NV0073_CTRL_PSR      (0x16U)
#define NV0073_CTRL_STEREO   (0x17U)

#define NV0073_CTRL_CMD_NULL (0x730000U)
