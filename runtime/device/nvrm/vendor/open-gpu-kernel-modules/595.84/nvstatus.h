#ifndef SDK_NVSTATUS_H
#define SDK_NVSTATUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"

typedef NvU32 NV_STATUS;

#define NV_STATUS_CODE( name, code, string ) name = (code),

enum
{
    #include "nvstatuscodes.h"
};

#undef NV_STATUS_CODE

#define NV_STATUS_LEVEL_OK              0

#define NV_STATUS_LEVEL_WARN            1

#define NV_STATUS_LEVEL_ERR             3

#define NV_STATUS_LEVEL(_S)                                               \
    ((_S) == NV_OK?                               NV_STATUS_LEVEL_OK:     \
    ((_S) != NV_ERR_GENERIC && (_S) & 0x00010000? NV_STATUS_LEVEL_WARN:   \
                                                  NV_STATUS_LEVEL_ERR))

#define NV_STATUS_LEVEL_CHAR(_S)                           \
    ((_S) == NV_OK?                                '0':    \
    ((_S) != NV_ERR_GENERIC && (_S) & 0x00010000?  'W':    \
                                                   'E'))

const char *nvstatusToString(NV_STATUS nvStatusIn);

#ifdef __cplusplus
}
#endif

#endif
