#ifndef _cl003e_h_
#define _cl003e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"

#define   NV01_CONTEXT_ERROR_TO_MEMORY                             (0x0000003E)
#define   NV01_MEMORY_SYSTEM                                       (0x0000003E)

#define NV03E_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

typedef volatile struct _cl003e_tag0 {
 NvV32 Reserved00[0x7c0];
} Nv03eTypedef, Nv01ContextErrorToMemory;
#define  NV03E_TYPEDEF                                 Nv01ContextErrorToMemory

#define  NV1_CONTEXT_ERROR_TO_MEMORY                               (0x0000003E)
#define  Nv1ContextErrorToMemory                       Nv01ContextErrorToMemory
#define  nv1ContextErrorToMemory                       Nv01ContextErrorToMemory
#define  nv01ContextErrorToMemory                      Nv01ContextErrorToMemory

#ifdef __cplusplus
};
#endif

#endif
