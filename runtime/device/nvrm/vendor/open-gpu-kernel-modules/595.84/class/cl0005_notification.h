#ifndef _cl0005_notification_h_
#define _cl0005_notification_h_

#ifdef __cplusplus
extern "C" {
#endif

#define NV003_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

typedef volatile struct _cl0005_tag0 {
    NvV32 Reserved00[0x7c0];
} Nv005Typedef, Nv01Event;

#define NV005_TYPEDEF                                              Nv01Event

#define NV1_TIMER                                                  (0x00000004)
#define Nv1Event                                                   Nv01Event
#define nv1Event                                                   Nv01Event
#define nv01Event                                                  Nv01Event

#ifdef __cplusplus
};
#endif

#endif
