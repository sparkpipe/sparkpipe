#pragma once

#include "ctrl/ctrlxxxx.h"

#define NV0000_CTRL_CMD(cat,idx)          NVXXXX_CTRL_CMD(0x0000,NV0000_CTRL_##cat,idx)

#define NV0000_CTRL_RESERVED       (0x00)
#define NV0000_CTRL_SYSTEM         (0x01)
#define NV0000_CTRL_GPU            (0x02)
#define NV0000_CTRL_GSYNC          (0x03)
#define NV0000_CTRL_DIAG           (0x04)
#define NV0000_CTRL_EVENT          (0x05)
#define NV0000_CTRL_NVD            (0x06)
#define NV0000_CTRL_SWINSTR        (0x07)
#define NV0000_CTRL_PROC           (0x09)
#define NV0000_CTRL_SYNC_GPU_BOOST (0x0A)
#define NV0000_CTRL_GPUACCT        (0x0B)
#define NV0000_CTRL_VGPU           (0x0C)
#define NV0000_CTRL_CLIENT         (0x0D)

#define NV0000_CTRL_OS_WINDOWS     (0x3F)
#define NV0000_CTRL_OS_MACOS       (0x3E)
#define NV0000_CTRL_OS_UNIX        (0x3D)

#define NV0000_CTRL_CMD_NULL       (0x0)
