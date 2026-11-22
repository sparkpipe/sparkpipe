#ifndef _NV_UNIX_NVOS_PARAMS_WRAPPERS_H_
#define _NV_UNIX_NVOS_PARAMS_WRAPPERS_H_

#include <nvos.h>

typedef struct
{
    NVOS02_PARAMETERS params;
    int fd;
} nv_ioctl_nvos02_parameters_with_fd;

typedef struct
{
    NVOS33_PARAMETERS params;
    int fd;
} nv_ioctl_nvos33_parameters_with_fd;

#endif
