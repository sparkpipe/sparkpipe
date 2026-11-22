#ifndef _cl0080_notification_h_
#define _cl0080_notification_h_

#ifdef __cplusplus
extern "C" {
#endif

#define NV080_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

typedef volatile struct _cl0080_tag0 {
 NvV32 Reserved00[0x7c0];
} Nv080Typedef, Nv01Device0;

#define  NV080_TYPEDEF                                             Nv01Device0

#ifdef __cplusplus
};
#endif

#endif
