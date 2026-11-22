#pragma once

#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl906f.h"

#define NVA06F_CTRL_CMD(cat,idx)  \
    NVXXXX_CTRL_CMD(0xA06F, NVA06F_CTRL_##cat, idx)

#define NVA06F_CTRL_RESERVED (0x00)
#define NVA06F_CTRL_GPFIFO   (0x01)
#define NVA06F_CTRL_INTERNAL (0x03)

#define NVA06F_CTRL_CMD_NULL (0xa06f0000)
