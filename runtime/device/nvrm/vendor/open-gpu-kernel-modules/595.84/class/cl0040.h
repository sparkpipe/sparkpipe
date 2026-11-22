#ifndef _cl0040_h_
#define _cl0040_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"

#define  NV01_MEMORY_LOCAL_USER                                    (0x00000040)

#define NV040_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

typedef volatile struct _cl0040_tag0 {
 NvV32 Reserved00[0x7c0];
} Nv040Typedef, Nv01MemoryLocalUser;
#define NV040_TYPEDEF                                       Nv01MemoryLocalUser

#define NV01_MEMORY_USER                                           (0x00000040)
#define NV1_MEMORY_USER                                            (0x00000040)
#define Nv01MemoryUser                                      Nv01MemoryLocalUser
#define nv01MemoryUser                                      Nv01MemoryLocalUser
#define Nv1MemoryUser                                       Nv01MemoryLocalUser
#define nv1MemoryUser                                       Nv01MemoryLocalUser
#define nv01MemoryLocalUser                                 Nv01MemoryLocalUser

#ifdef __cplusplus
};
#endif

#endif
