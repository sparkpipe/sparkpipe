#pragma once

#include "ctrl/ctrlxxxx.h"

#define NV0080_CTRL_CMD(cat,idx)                NVXXXX_CTRL_CMD(0x0080, NV0080_CTRL_##cat, idx)

#define NV0080_CTRL_RESERVED                   (0x00)
#define NV0080_CTRL_BIF                        (0x01)
#define NV0080_CTRL_GPU                        (0x02)
#define NV0080_CTRL_CLK                        (0x10)
#define NV0080_CTRL_GR                         (0x11)
#define NV0080_CTRL_CIPHER                     (0x12)
#define NV0080_CTRL_FB                         (0x13)
#define NV0080_CTRL_HOST                       (0x14)
#define NV0080_CTRL_VIDEO                      (0x15)
#define NV0080_CTRL_FIFO                       (0x17)
#define NV0080_CTRL_DMA                        (0x18)
#define NV0080_CTRL_PERF                       (0x19)
#define NV0080_CTRL_PERF_LEGACY_NON_PRIVILEGED (0x99)
#define NV0080_CTRL_MSENC                      (0x1B)
#define NV0080_CTRL_BSP                        (0x1C)
#define NV0080_CTRL_RC                         (0x1D)
#define NV0080_CTRL_OS_UNIX                    (0x1E)
#define NV0080_CTRL_NVJPG                      (0x1F)
#define NV0080_CTRL_INTERNAL                   (0x20)
#define NV0080_CTRL_NVLINK                     (0x21)

#define NV0080_CTRL_CMD_NULL                   (0x800000)

#define NV0080_CTRL_GET_CAP(cat,tbl,c)    \
    NV0080_CTRL_##cat##_GET_CAP(tbl, NV0080_CTRL_##cat##_CAPS_##c)
