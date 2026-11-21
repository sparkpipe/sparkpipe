#ifndef _UVM_LINUX_IOCTL_H
#define _UVM_LINUX_IOCTL_H

#include "uvm_ioctl.h"

#define UVM_INITIALIZE                                                0x30000001

typedef struct
{
    NvU64     flags     NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_INITIALIZE_PARAMS;

#define UVM_DEINITIALIZE                                              0x30000002

#endif
