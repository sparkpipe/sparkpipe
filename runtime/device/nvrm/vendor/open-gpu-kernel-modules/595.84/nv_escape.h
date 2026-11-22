#ifndef NV_ESCAPE_H_INCLUDED
#define NV_ESCAPE_H_INCLUDED

#define NV_ESC_RM_ALLOC_MEMORY                      0x27
#define NV_ESC_RM_ALLOC_OBJECT                      0x28
#define NV_ESC_RM_FREE                              0x29
#define NV_ESC_RM_CONTROL                           0x2A
#define NV_ESC_RM_ALLOC                             0x2B
#define NV_ESC_RM_DUP_OBJECT                        0x34
#define NV_ESC_RM_SHARE                             0x35
#define NV_ESC_RM_I2C_ACCESS                        0x39
#define NV_ESC_RM_IDLE_CHANNELS                     0x41
#define NV_ESC_RM_VID_HEAP_CONTROL                  0x4A
#define NV_ESC_RM_ACCESS_REGISTRY                   0x4D
#define NV_ESC_RM_MAP_MEMORY                        0x4E
#define NV_ESC_RM_UNMAP_MEMORY                      0x4F
#define NV_ESC_RM_GET_EVENT_DATA                    0x52
#define NV_ESC_RM_ALLOC_CONTEXT_DMA2                0x54
#define NV_ESC_RM_ADD_VBLANK_CALLBACK               0x56
#define NV_ESC_RM_MAP_MEMORY_DMA                    0x57
#define NV_ESC_RM_UNMAP_MEMORY_DMA                  0x58
#define NV_ESC_RM_BIND_CONTEXT_DMA                  0x59
#define NV_ESC_RM_EXPORT_OBJECT_TO_FD               0x5C
#define NV_ESC_RM_IMPORT_OBJECT_FROM_FD             0x5D
#define NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO        0x5E
#define NV_ESC_RM_LOCKLESS_DIAGNOSTIC               0x5F

#endif
