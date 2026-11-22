#pragma once

#include <nvtypes.h>

#include "nvtypes.h"

#define NVXXXX_CTRL_CMD_CLASS                                             31:16

#define NVXXXX_CTRL_CMD_CATEGORY                                           13:8
#define NVXXXX_CTRL_CMD_INDEX                                               7:0

#  define NVXXXX_CTRL_CMD(cls,cat,idx)     \
                               (((cls) << 16) | ((0) << 15) | ((0) << 14) \
                               | ((cat) << 8) | ((idx) & 0xFF))

#define NVXXXX_CTRL_CMD_NULL              (0x00000000)

#define NVxxxx_CTRL_LEGACY_PRIVILEGED     (0xC0)
#define NVxxxx_CTRL_LEGACY_NON_PRIVILEGED (0x80)

typedef struct NVXXXX_CTRL_XXX_INFO {
    NvU32 index;
    NvU32 data;
} NVXXXX_CTRL_XXX_INFO;
