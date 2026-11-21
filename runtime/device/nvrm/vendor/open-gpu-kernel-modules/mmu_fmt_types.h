#pragma once

#include <nvtypes.h>

#include "nvtypes.h"

typedef struct MMU_FMT_LEVEL {

    NvU8   virtAddrBitLo;

    NvU8   virtAddrBitHi;

    NvU8   entrySize;

    NvBool bPageTable;

    NvU8   numSubLevels;

    NvU32  pageLevelIdTag;

    NV_DECLARE_ALIGNED(struct MMU_FMT_LEVEL *subLevels, 8);
} MMU_FMT_LEVEL;

#define MMU_FMT_MAX_SUB_LEVELS 2

#define MMU_FMT_PT_SURF_ID_PD0           0
#define MMU_FMT_PT_SURF_ID_PD1           1
#define MMU_FMT_PT_SURF_ID_PD2           2
#define MMU_FMT_PT_SURF_ID_PD3           3
#define MMU_FMT_PT_SURF_ID_PD4           4
#define MMU_FMT_PT_SURF_ID_PT_4K         5
#define MMU_FMT_PT_SURF_ID_PT_BIG        6
