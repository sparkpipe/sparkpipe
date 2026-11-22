#ifndef _cl0000_notification_h_
#define _cl0000_notification_h_

#ifdef __cplusplus
extern "C" {
#endif

#define NV0000_NOTIFIERS_STATUS_ACPI_DISPLAY_DEVICE_CYCLE          (0)

#define NV000_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

typedef volatile struct _cl0000_tag0 {
    NvV32 Reserved00[0x7c0];
} Nv000Typedef, Nv01Root;

#define NV000_TYPEDEF                                              Nv01Root
#define Nv1Root                                                    Nv01Root
#define nv1Root                                                    Nv01Root
#define nv01Root                                                   Nv01Root

#define NV0000_NOTIFIERS_ENABLE_CPU_UTIL_CTRL                      (1)

#ifdef __cplusplus
};
#endif

#endif
