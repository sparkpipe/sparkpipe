#ifndef NVOS_INCLUDED
#define NVOS_INCLUDED
#ifdef __cplusplus
extern "C" {
#endif

#include "nvstatus.h"

#include "nvgputypes.h"
#include "rs_access.h"
#include "nvcfg_sdk.h"

#include "alloc/alloc_channel.h"

#define FILE_DEVICE_NV      0x00008000
#define NV_IOCTL_FCT_BASE   0x00000800

#define NVOS_MAX_SUBDEVICES 8

#define UNIFIED_NV_STATUS 1

#define NVOS_STATUS NV_STATUS

#define NVOS_STATUS_SUCCESS                                     NV_OK
#define NVOS_STATUS_ERROR_CARD_NOT_PRESENT                      NV_ERR_CARD_NOT_PRESENT
#define NVOS_STATUS_ERROR_DUAL_LINK_INUSE                       NV_ERR_DUAL_LINK_INUSE
#define NVOS_STATUS_ERROR_GENERIC                               NV_ERR_GENERIC
#define NVOS_STATUS_ERROR_GPU_NOT_FULL_POWER                    NV_ERR_GPU_NOT_FULL_POWER
#define NVOS_STATUS_ERROR_ILLEGAL_ACTION                        NV_ERR_ILLEGAL_ACTION
#define NVOS_STATUS_ERROR_IN_USE                                NV_ERR_STATE_IN_USE
#define NVOS_STATUS_ERROR_INSUFFICIENT_RESOURCES                NV_ERR_INSUFFICIENT_RESOURCES
#define NVOS_STATUS_ERROR_INSUFFICIENT_ZBC_ENTRY                NV_ERR_INSUFFICIENT_ZBC_ENTRY
#define NVOS_STATUS_ERROR_INVALID_ACCESS_TYPE                   NV_ERR_INVALID_ACCESS_TYPE
#define NVOS_STATUS_ERROR_INVALID_ARGUMENT                      NV_ERR_INVALID_ARGUMENT
#define NVOS_STATUS_ERROR_INVALID_BASE                          NV_ERR_INVALID_BASE
#define NVOS_STATUS_ERROR_INVALID_CHANNEL                       NV_ERR_INVALID_CHANNEL
#define NVOS_STATUS_ERROR_INVALID_CLASS                         NV_ERR_INVALID_CLASS
#define NVOS_STATUS_ERROR_INVALID_CLIENT                        NV_ERR_INVALID_CLIENT
#define NVOS_STATUS_ERROR_INVALID_COMMAND                       NV_ERR_INVALID_COMMAND
#define NVOS_STATUS_ERROR_INVALID_DATA                          NV_ERR_INVALID_DATA
#define NVOS_STATUS_ERROR_INVALID_DEVICE                        NV_ERR_INVALID_DEVICE
#define NVOS_STATUS_ERROR_INVALID_DMA_SPECIFIER                 NV_ERR_INVALID_DMA_SPECIFIER
#define NVOS_STATUS_ERROR_INVALID_EVENT                         NV_ERR_INVALID_EVENT
#define NVOS_STATUS_ERROR_INVALID_FLAGS                         NV_ERR_INVALID_FLAGS
#define NVOS_STATUS_ERROR_INVALID_FUNCTION                      NV_ERR_INVALID_FUNCTION
#define NVOS_STATUS_ERROR_INVALID_HEAP                          NV_ERR_INVALID_HEAP
#define NVOS_STATUS_ERROR_INVALID_INDEX                         NV_ERR_INVALID_INDEX
#define NVOS_STATUS_ERROR_INVALID_LIMIT                         NV_ERR_INVALID_LIMIT
#define NVOS_STATUS_ERROR_INVALID_METHOD                        NV_ERR_INVALID_METHOD
#define NVOS_STATUS_ERROR_INVALID_OBJECT_BUFFER                 NV_ERR_BUFFER_TOO_SMALL
#define NVOS_STATUS_ERROR_INVALID_OBJECT_ERROR                  NV_ERR_INVALID_OBJECT
#define NVOS_STATUS_ERROR_INVALID_OBJECT_HANDLE                 NV_ERR_INVALID_OBJECT_HANDLE
#define NVOS_STATUS_ERROR_INVALID_OBJECT_NEW                    NV_ERR_INVALID_OBJECT_NEW
#define NVOS_STATUS_ERROR_INVALID_OBJECT_OLD                    NV_ERR_INVALID_OBJECT_OLD
#define NVOS_STATUS_ERROR_INVALID_OBJECT_PARENT                 NV_ERR_INVALID_OBJECT_PARENT
#define NVOS_STATUS_ERROR_INVALID_OFFSET                        NV_ERR_INVALID_OFFSET
#define NVOS_STATUS_ERROR_INVALID_OWNER                         NV_ERR_INVALID_OWNER
#define NVOS_STATUS_ERROR_INVALID_PARAM_STRUCT                  NV_ERR_INVALID_PARAM_STRUCT
#define NVOS_STATUS_ERROR_INVALID_PARAMETER                     NV_ERR_INVALID_PARAMETER
#define NVOS_STATUS_ERROR_INVALID_POINTER                       NV_ERR_INVALID_POINTER
#define NVOS_STATUS_ERROR_INVALID_REGISTRY_KEY                  NV_ERR_INVALID_REGISTRY_KEY
#define NVOS_STATUS_ERROR_INVALID_STATE                         NV_ERR_INVALID_STATE
#define NVOS_STATUS_ERROR_INVALID_STRING_LENGTH                 NV_ERR_INVALID_STRING_LENGTH
#define NVOS_STATUS_ERROR_INVALID_XLATE                         NV_ERR_INVALID_XLATE
#define NVOS_STATUS_ERROR_IRQ_NOT_FIRING                        NV_ERR_IRQ_NOT_FIRING
#define NVOS_STATUS_ERROR_MULTIPLE_MEMORY_TYPES                 NV_ERR_MULTIPLE_MEMORY_TYPES
#define NVOS_STATUS_ERROR_NOT_SUPPORTED                         NV_ERR_NOT_SUPPORTED
#define NVOS_STATUS_ERROR_OPERATING_SYSTEM                      NV_ERR_OPERATING_SYSTEM
#define NVOS_STATUS_ERROR_LIB_RM_VERSION_MISMATCH               NV_ERR_LIB_RM_VERSION_MISMATCH
#define NVOS_STATUS_ERROR_PROTECTION_FAULT                      NV_ERR_PROTECTION_FAULT
#define NVOS_STATUS_ERROR_TIMEOUT                               NV_ERR_TIMEOUT
#define NVOS_STATUS_ERROR_TOO_MANY_PRIMARIES                    NV_ERR_TOO_MANY_PRIMARIES
#define NVOS_STATUS_ERROR_IRQ_EDGE_TRIGGERED                    NV_ERR_IRQ_EDGE_TRIGGERED
#define NVOS_STATUS_ERROR_INVALID_OPERATION                     NV_ERR_INVALID_OPERATION
#define NVOS_STATUS_ERROR_NOT_COMPATIBLE                        NV_ERR_NOT_COMPATIBLE
#define NVOS_STATUS_ERROR_MORE_PROCESSING_REQUIRED              NV_WARN_MORE_PROCESSING_REQUIRED
#define NVOS_STATUS_ERROR_INSUFFICIENT_PERMISSIONS              NV_ERR_INSUFFICIENT_PERMISSIONS
#define NVOS_STATUS_ERROR_TIMEOUT_RETRY                         NV_ERR_TIMEOUT_RETRY
#define NVOS_STATUS_ERROR_NOT_READY                             NV_ERR_NOT_READY
#define NVOS_STATUS_ERROR_GPU_IS_LOST                           NV_ERR_GPU_IS_LOST
#define NVOS_STATUS_ERROR_IN_FULLCHIP_RESET                     NV_ERR_GPU_IN_FULLCHIP_RESET
#define NVOS_STATUS_ERROR_INVALID_LOCK_STATE                    NV_ERR_INVALID_LOCK_STATE
#define NVOS_STATUS_ERROR_INVALID_ADDRESS                       NV_ERR_INVALID_ADDRESS
#define NVOS_STATUS_ERROR_INVALID_IRQ_LEVEL                     NV_ERR_INVALID_IRQ_LEVEL
#define NVOS_STATUS_ERROR_MEMORY_TRAINING_FAILED                NV_ERR_MEMORY_TRAINING_FAILED
#define NVOS_STATUS_ERROR_BUSY_RETRY                            NV_ERR_BUSY_RETRY
#define NVOS_STATUS_ERROR_INSUFFICIENT_POWER                    NV_ERR_INSUFFICIENT_POWER
#define NVOS_STATUS_ERROR_OBJECT_NOT_FOUND                      NV_ERR_OBJECT_NOT_FOUND
#define NVOS_STATUS_ERROR_RESOURCE_LOST                         NV_ERR_RESOURCE_LOST
#define NVOS_STATUS_ERROR_BUFFER_TOO_SMALL                      NV_ERR_BUFFER_TOO_SMALL
#define NVOS_STATUS_ERROR_RESET_REQUIRED                        NV_ERR_RESET_REQUIRED
#define NVOS_STATUS_ERROR_INVALID_REQUEST                       NV_ERR_INVALID_REQUEST

#define NVOS_STATUS_ERROR_PRIV_SEC_VIOLATION                    NV_ERR_PRIV_SEC_VIOLATION
#define NVOS_STATUS_ERROR_GPU_IN_DEBUG_MODE                     NV_ERR_GPU_IN_DEBUG_MODE
#define NVOS_STATUS_ERROR_ALREADY_SIGNALLED                     NV_ERR_ALREADY_SIGNALLED

#define  NV01_FREE                                                 (0x00000000)

typedef struct
{
  NvHandle  hRoot;
  NvHandle  hObjectParent;
  NvHandle  hObjectOld;
  NvV32     status;
} NVOS00_PARAMETERS;

#define  NV01_ROOT                                                 (0x0U)

#define  NV01_ROOT_NON_PRIV                                        (0x00000001)

#define  NV01_ROOT_USER                                            NV01_ROOT_CLIENT

#define  NV01_ROOT_CLIENT                                          (0x00000041)

#define  NV01_ALLOC_MEMORY                                         (0x00000002)

#define NVOS02_FLAGS_PHYSICALITY                                   7:4
#define NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS                        (0x00000000)
#define NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS                     (0x00000001)
#define NVOS02_FLAGS_LOCATION                                      11:8
#define NVOS02_FLAGS_LOCATION_PCI                                  (0x00000000)
#define NVOS02_FLAGS_LOCATION_VIDMEM                               (0x00000002)
#define NVOS02_FLAGS_COHERENCY                                     15:12
#define NVOS02_FLAGS_COHERENCY_UNCACHED                            (0x00000000)
#define NVOS02_FLAGS_COHERENCY_CACHED                              (0x00000001)
#define NVOS02_FLAGS_COHERENCY_WRITE_COMBINE                       (0x00000002)
#define NVOS02_FLAGS_COHERENCY_WRITE_THROUGH                       (0x00000003)
#define NVOS02_FLAGS_COHERENCY_WRITE_PROTECT                       (0x00000004)
#define NVOS02_FLAGS_COHERENCY_WRITE_BACK                          (0x00000005)
#define NVOS02_FLAGS_ALLOC                                         17:16
#define NVOS02_FLAGS_ALLOC_NONE                                    (0x00000001)
#define NVOS02_FLAGS_GPU_CACHEABLE                                 18:18
#define NVOS02_FLAGS_GPU_CACHEABLE_NO                              (0x00000000)
#define NVOS02_FLAGS_GPU_CACHEABLE_YES                             (0x00000001)

#define NVOS02_FLAGS_KERNEL_MAPPING                                19:19
#define NVOS02_FLAGS_KERNEL_MAPPING_NO_MAP                         (0x00000000)
#define NVOS02_FLAGS_KERNEL_MAPPING_MAP                            (0x00000001)
#define NVOS02_FLAGS_ALLOC_NISO_DISPLAY                            20:20
#define NVOS02_FLAGS_ALLOC_NISO_DISPLAY_NO                         (0x00000000)
#define NVOS02_FLAGS_ALLOC_NISO_DISPLAY_YES                        (0x00000001)

#define NVOS02_FLAGS_ALLOC_USER_READ_ONLY                          21:21
#define NVOS02_FLAGS_ALLOC_USER_READ_ONLY_NO                       (0x00000000)
#define NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES                      (0x00000001)

#define NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY                        22:22
#define NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_NO                     (0x00000000)
#define NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_YES                    (0x00000001)

#define NVOS02_FLAGS_PEER_MAP_OVERRIDE                             23:23
#define NVOS02_FLAGS_PEER_MAP_OVERRIDE_DEFAULT                     (0x00000000)
#define NVOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED                    (0x00000001)

#define NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT                          24:24
#define NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT_APERTURE                 (0x00000001)

#define NVOS02_FLAGS_MEMORY_PROTECTION                             26:25
#define NVOS02_FLAGS_MEMORY_PROTECTION_DEFAULT                     (0x00000000)
#define NVOS02_FLAGS_MEMORY_PROTECTION_PROTECTED                   (0x00000001)
#define NVOS02_FLAGS_MEMORY_PROTECTION_UNPROTECTED                 (0x00000002)

#define NVOS02_FLAGS_REGISTER_MEMDESC_TO_PHYS_RM                   27:27
#define NVOS02_FLAGS_REGISTER_MEMDESC_TO_PHYS_RM_FALSE             (0x00000000)
#define NVOS02_FLAGS_REGISTER_MEMDESC_TO_PHYS_RM_TRUE              (0x00000001)

#define NVOS02_FLAGS_MAPPING                                       31:30
#define NVOS02_FLAGS_MAPPING_DEFAULT                               (0x00000000)
#define NVOS02_FLAGS_MAPPING_NO_MAP                                (0x00000001)
#define NVOS02_FLAGS_MAPPING_NEVER_MAP                             (0x00000002)

typedef struct
{
    NvHandle    hRoot;
    NvHandle    hObjectParent;
    NvHandle    hObjectNew;
    NvV32       hClass;
    NvV32       flags;
    NvP64       pMemory NV_ALIGN_BYTES(8);
    NvU64       limit NV_ALIGN_BYTES(8);
    NvV32       status;
} NVOS02_PARAMETERS;

#define NVOS03_FLAGS_ACCESS                                        1:0
#define NVOS03_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define NVOS03_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define NVOS03_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

#define NVOS03_FLAGS_PREALLOCATE                                   2:2
#define NVOS03_FLAGS_PREALLOCATE_DISABLE                           (0x00000000)
#define NVOS03_FLAGS_PREALLOCATE_ENABLE                            (0x00000001)

#define NVOS03_FLAGS_GPU_MAPPABLE                                  15:15
#define NVOS03_FLAGS_GPU_MAPPABLE_DISABLE                          (0x00000000)
#define NVOS03_FLAGS_GPU_MAPPABLE_ENABLE                           (0x00000001)

#define NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE                          16:16
#define NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_FALSE                    (0x00000000)
#define NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_TRUE                     (0x00000001)

#define NVOS03_FLAGS_PTE_KIND                                      17:16
#define NVOS03_FLAGS_PTE_KIND_NONE                                 (0x00000000)
#define NVOS03_FLAGS_PTE_KIND_BL                                   (0x00000001)
#define NVOS03_FLAGS_PTE_KIND_PITCH                                (0x00000002)

#define NVOS03_FLAGS_TYPE                                          23:20
#define NVOS03_FLAGS_TYPE_NOTIFIER                                 (0x00000001)

#define NVOS03_FLAGS_MAPPING                                       20:20
#define NVOS03_FLAGS_MAPPING_NONE                                  (0x00000000)
#define NVOS03_FLAGS_MAPPING_KERNEL                                (0x00000001)

#define NVOS03_FLAGS_CACHE_SNOOP                                   28:28
#define NVOS03_FLAGS_CACHE_SNOOP_ENABLE                            (0x00000000)
#define NVOS03_FLAGS_CACHE_SNOOP_DISABLE                           (0x00000001)

#define NVOS03_FLAGS_HASH_TABLE                                    29:29
#define NVOS03_FLAGS_HASH_TABLE_ENABLE                             (0x00000000)
#define NVOS03_FLAGS_HASH_TABLE_DISABLE                            (0x00000001)

#define  NV01_ALLOC_OBJECT                                         (0x00000005)

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvV32    status;
} NVOS05_PARAMETERS;

#define  NV01_EVENT_KERNEL_CALLBACK                                (0x00000078)
#define  NV01_EVENT_OS_EVENT                                       (0x00000079)
#define  NV01_EVENT_WIN32_EVENT                             NV01_EVENT_OS_EVENT
#define  NV01_EVENT_KERNEL_CALLBACK_EX                             (0x0000007E)

typedef void (*Callback1ArgVoidReturn)(void *arg);
typedef void (*Callback5ArgVoidReturn)(void *arg1, void *arg2, NvHandle hEvent, NvU32 data, NvU32 status);

typedef struct
{
    Callback1ArgVoidReturn  func;
    void                   *arg;
} NVOS10_EVENT_KERNEL_CALLBACK;

typedef struct
{
    Callback5ArgVoidReturn  func;
    void                   *arg;
} NVOS10_EVENT_KERNEL_CALLBACK_EX;

#define NV01_EVENT_BROADCAST                                       (0x80000000)

#define NV01_EVENT_PERMIT_NON_ROOT_EVENT_KERNEL_CALLBACK_CREATION  (0x40000000)

#define NV01_EVENT_SUBDEVICE_SPECIFIC                              (0x20000000)

#define NV01_EVENT_WITHOUT_EVENT_DATA                              (0x10000000)

#define NV01_EVENT_NONSTALL_INTR                                   (0x08000000)

#define NV01_EVENT_CLIENT_RM                                       (0x04000000)

#define  NV04_I2C_ACCESS                                           (0x00000013)

#define NVOS_I2C_ACCESS_MAX_BUFFER_SIZE  2048

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvU32    paramSize;
    NvP64    paramStructPtr NV_ALIGN_BYTES(8);
    NvV32    status;
} NVOS_I2C_ACCESS_PARAMS;

#define NVOS20_COMMAND_unused0001                  0x0001
#define NVOS20_COMMAND_unused0002                  0x0002
#define NVOS20_COMMAND_STRING_PRINT                0x0003

#define  NV04_ALLOC                                                (0x00000015)

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvP64    pAllocParms NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS21_PARAMETERS;

#define NVOS64_FLAGS_NONE             (0x00000000)
#define NVOS64_FLAGS_FINN_SERIALIZED  (0x00000001)

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvP64    pAllocParms NV_ALIGN_BYTES(8);
    NvP64    pRightsRequested NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvU32    flags;
    NvV32    status;
} NVOS64_PARAMETERS;

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvU32    paramSize;
    NvV32    status;
} NVOS62_PARAMETERS;

#define NVOS65_PARAMETERS_VERSION_MAGIC 0x77FEF81E

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvU32    paramSize;
    NvU32    versionMagic;
    NvU32    maskSize;
    NvV32    status;
} NVOS65_PARAMETERS;

#define NV04_IDLE_CHANNELS                                         (0x0000001E)

#define NVOS30_FLAGS_BEHAVIOR                                      3:0
#define NVOS30_FLAGS_BEHAVIOR_SPIN                                 (0x00000000)
#define NVOS30_FLAGS_BEHAVIOR_SLEEP                                (0x00000001)
#define NVOS30_FLAGS_BEHAVIOR_QUERY                                (0x00000002)
#define NVOS30_FLAGS_BEHAVIOR_FORCE_BUSY_CHECK                     (0x00000003)
#define NVOS30_FLAGS_CHANNEL                                       7:4
#define NVOS30_FLAGS_CHANNEL_LIST                                  (0x00000000)
#define NVOS30_FLAGS_CHANNEL_SINGLE                                (0x00000001)
#define NVOS30_FLAGS_IDLE                                          30:8
#define NVOS30_FLAGS_IDLE_PUSH_BUFFER                              (0x00000001)
#define NVOS30_FLAGS_IDLE_CACHE1                                   (0x00000002)
#define NVOS30_FLAGS_IDLE_GRAPHICS                                 (0x00000004)
#define NVOS30_FLAGS_IDLE_MPEG                                     (0x00000008)
#define NVOS30_FLAGS_IDLE_MOTION_ESTIMATION                        (0x00000010)
#define NVOS30_FLAGS_IDLE_VIDEO_PROCESSOR                          (0x00000020)
#define NVOS30_FLAGS_IDLE_MSPDEC                                   (0x00000020)
#define NVOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR                      (0x00000040)
#define NVOS30_FLAGS_IDLE_MSVLD                                    (0x00000040)
#define NVOS30_FLAGS_IDLE_NVDEC0                                   NVOS30_FLAGS_IDLE_MSVLD
#define NVOS30_FLAGS_IDLE_CIPHER_DMA                               (0x00000080)
#define NVOS30_FLAGS_IDLE_SEC                                      (0x00000080)
#define NVOS30_FLAGS_IDLE_CALLBACKS                                (0x00000100)
#define NVOS30_FLAGS_IDLE_MSPPP                                    (0x00000200)
#define NVOS30_FLAGS_IDLE_CE0                                      (0x00000400)
#define NVOS30_FLAGS_IDLE_CE1                                      (0x00000800)
#define NVOS30_FLAGS_IDLE_CE2                                      (0x00001000)
#define NVOS30_FLAGS_IDLE_CE3                                      (0x00002000)
#define NVOS30_FLAGS_IDLE_CE4                                      (0x00004000)
#define NVOS30_FLAGS_IDLE_CE5                                      (0x00008000)
#define NVOS30_FLAGS_IDLE_VIC                                      (0x00010000)
#define NVOS30_FLAGS_IDLE_MSENC                                    (0x00020000)
#define NVOS30_FLAGS_IDLE_NVENC0                                    NVOS30_FLAGS_IDLE_MSENC
#define NVOS30_FLAGS_IDLE_NVENC1                                   (0x00040000)
#define NVOS30_FLAGS_IDLE_NVENC2                                   (0x00080000)
#define NVOS30_FLAGS_IDLE_NVJPG                                    (0x00100000)
#define NVOS30_FLAGS_IDLE_NVDEC1                                   (0x00200000)
#define NVOS30_FLAGS_IDLE_NVDEC2                                   (0x00400000)
#define NVOS30_FLAGS_IDLE_ACTIVECHANNELS                           (0x00800000)
#define NVOS30_FLAGS_IDLE_ALL_ENGINES (NVOS30_FLAGS_IDLE_GRAPHICS | \
                                       NVOS30_FLAGS_IDLE_MPEG | \
                                       NVOS30_FLAGS_IDLE_MOTION_ESTIMATION | \
                                       NVOS30_FLAGS_IDLE_VIDEO_PROCESSOR | \
                                       NVOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR |  \
                                       NVOS30_FLAGS_IDLE_CIPHER_DMA  | \
                                       NVOS30_FLAGS_IDLE_MSPDEC      | \
                                       NVOS30_FLAGS_IDLE_NVDEC0      | \
                                       NVOS30_FLAGS_IDLE_SEC         | \
                                       NVOS30_FLAGS_IDLE_MSPPP       | \
                                       NVOS30_FLAGS_IDLE_CE0         | \
                                       NVOS30_FLAGS_IDLE_CE1         | \
                                       NVOS30_FLAGS_IDLE_CE2         | \
                                       NVOS30_FLAGS_IDLE_CE3         | \
                                       NVOS30_FLAGS_IDLE_CE4         | \
                                       NVOS30_FLAGS_IDLE_CE5         | \
                                       NVOS30_FLAGS_IDLE_NVENC0      | \
                                       NVOS30_FLAGS_IDLE_NVENC1      | \
                                       NVOS30_FLAGS_IDLE_NVENC2      | \
                                       NVOS30_FLAGS_IDLE_VIC         | \
                                       NVOS30_FLAGS_IDLE_NVJPG       | \
                                       NVOS30_FLAGS_IDLE_NVDEC1      | \
                                       NVOS30_FLAGS_IDLE_NVDEC2)
#define NVOS30_FLAGS_WAIT_FOR_ELPG_ON                              31:31
#define NVOS30_FLAGS_WAIT_FOR_ELPG_ON_NO                           (0x00000000)
#define NVOS30_FLAGS_WAIT_FOR_ELPG_ON_YES                          (0x00000001)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hChannel;
    NvV32    numChannels;

    NvP64    phClients NV_ALIGN_BYTES(8);
    NvP64    phDevices NV_ALIGN_BYTES(8);
    NvP64    phChannels NV_ALIGN_BYTES(8);

    NvV32    flags;
    NvV32    timeout;
    NvV32    status;
} NVOS30_PARAMETERS;

typedef void (*BindResultFunc)(void * pVoid, NvU32 gpuMask, NvU32 bState, NvU32 bResult);

#define NV04_VID_HEAP_CONTROL                                      (0x00000020)

#define NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS          0
#define NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY            1
#define NVOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY             2
#define NVOS32_DESCRIPTOR_TYPE_OS_PHYS_ADDR             3
#define NVOS32_DESCRIPTOR_TYPE_OS_FILE_HANDLE           4
#define NVOS32_DESCRIPTOR_TYPE_OS_DMA_BUF_PTR           5
#define NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR               6
#define NVOS32_DESCRIPTOR_TYPE_KERNEL_VIRTUAL_ADDRESS   7

#define NVOS32_FUNCTION_ALLOC_SIZE                      2
#define NVOS32_FUNCTION_FREE                            3
#define NVOS32_FUNCTION_INFO                            5
#define NVOS32_FUNCTION_ALLOC_TILED_PITCH_HEIGHT        6
#define NVOS32_FUNCTION_ALLOC_SIZE_RANGE                14
#define NVOS32_FUNCTION_REACQUIRE_COMPR                 15
#define NVOS32_FUNCTION_RELEASE_COMPR                   16
#define NVOS32_FUNCTION_GET_MEM_ALIGNMENT               18
#define NVOS32_FUNCTION_HW_ALLOC                        19
#define NVOS32_FUNCTION_HW_FREE                         20
#define NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR             27

typedef struct
{
    NvP64 sgt NV_ALIGN_BYTES(8);
    NvP64 gem NV_ALIGN_BYTES(8);
} NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS;

#define NVOS32_IVC_HEAP_NUMBER_DONT_ALLOCATE_ON_IVC_HEAP 0

typedef struct
{
  NvHandle  hRoot;
  NvHandle  hObjectParent;
  NvU32     function;
  NvHandle  hVASpace;
  NvS16     ivcHeapNumber;

  NvV32     status;
  NvU64     total NV_ALIGN_BYTES(8);
  NvU64     free  NV_ALIGN_BYTES(8);

  union
  {

      struct
      {
          NvU32     owner;
          NvHandle  hMemory;
          NvU32     type;
          NvU32     flags;
          NvU32     attr;
          NvU32     format;
          NvU32     comprCovg;
          NvU32     zcullCovg;
          NvU32     partitionStride;
          NvU32     width;
          NvU32     height;
          NvU64     size      NV_ALIGN_BYTES(8);
          NvU64     alignment NV_ALIGN_BYTES(8);
          NvU64     offset    NV_ALIGN_BYTES(8);
          NvU64     limit     NV_ALIGN_BYTES(8);
          NvP64     address NV_ALIGN_BYTES(8);
          NvU64     rangeBegin NV_ALIGN_BYTES(8);
          NvU64     rangeEnd   NV_ALIGN_BYTES(8);
          NvU32     attr2;
          NvU32     ctagOffset;
          NvS32     numaNode;
      } AllocSize;

      struct
      {
          NvU32     owner;
          NvHandle  hMemory;
          NvU32     type;
          NvU32     flags;
          NvU32     height;
          NvS32     pitch;
          NvU32     attr;
          NvU32     width;
          NvU32     format;
          NvU32     comprCovg;
          NvU32     zcullCovg;
          NvU32     partitionStride;
          NvU64     size      NV_ALIGN_BYTES(8);
          NvU64     alignment NV_ALIGN_BYTES(8);
          NvU64     offset    NV_ALIGN_BYTES(8);
          NvU64     limit     NV_ALIGN_BYTES(8);
          NvP64     address NV_ALIGN_BYTES(8);
          NvU64     rangeBegin NV_ALIGN_BYTES(8);
          NvU64     rangeEnd   NV_ALIGN_BYTES(8);
          NvU32     attr2;
          NvU32     ctagOffset;
          NvS32     numaNode;
      } AllocTiledPitchHeight;

      struct
      {
          NvU32     owner;
          NvHandle  hMemory;
          NvU32     flags;
      } Free;

      struct
      {
          NvU32     owner;
          NvU32     flags;
          NvHandle  hMemory;
      } ReleaseCompr;

      struct
      {
          NvU32     owner;
          NvU32     flags;
          NvHandle  hMemory;
      } ReacquireCompr;

      struct
      {
          NvU32 attr;
          NvU64 offset NV_ALIGN_BYTES(8);
          NvU64 size   NV_ALIGN_BYTES(8);
          NvU64 base   NV_ALIGN_BYTES(8);
      } Info;

      struct
      {
          NvU32     owner;
          NvHandle  hMemory;
          NvU32     type;
          NvU32     flags;
          NvU32     attr;
          NvU32     format;
          NvU32     comprCovg;
          NvU32     zcullCovg;
          NvU32     partitionStride;
          NvU64     size      NV_ALIGN_BYTES(8);
          NvU64     alignment NV_ALIGN_BYTES(8);
          NvU64     offset    NV_ALIGN_BYTES(8);
          NvU64     limit     NV_ALIGN_BYTES(8);
          NvU64     rangeBegin NV_ALIGN_BYTES(8);
          NvU64     rangeEnd   NV_ALIGN_BYTES(8);
          NvP64     address NV_ALIGN_BYTES(8);
          NvU32     attr2;
          NvU32     ctagOffset;
          NvS32     numaNode;
      } AllocSizeRange;

      struct
      {
          NvU32 alignType;
          NvU32 alignAttr;
          NvU32 alignInputFlags;
          NvU64 alignSize NV_ALIGN_BYTES(8);
          NvU32 alignHeight;
          NvU32 alignWidth;
          NvU32 alignPitch;
          NvU32 alignPad;
          NvU32 alignMask;
          NvU32 alignKind;
          NvU32 alignAdjust;
          NvU32 alignAttr2;
      } AllocHintAlignment;

      struct
      {
          NvU32     allocOwner;
          NvHandle  allochMemory;
          NvU32     flags;
          NvU32     allocType;
          NvU32     allocAttr;
          NvU32     allocInputFlags;
          NvU64     allocSize       NV_ALIGN_BYTES(8);
          NvU32     allocHeight;
          NvU32     allocWidth;
          NvU32     allocPitch;
          NvU32     allocMask;
          NvU32     allocComprCovg;
          NvU32     allocZcullCovg;
          NvP64     bindResultFunc  NV_ALIGN_BYTES(8);
          NvP64     pHandle         NV_ALIGN_BYTES(8);
          NvHandle  hResourceHandle;
          NvU32     retAttr;
          NvU32     kind;
          NvU64     osDeviceHandle  NV_ALIGN_BYTES(8);
          NvU32     allocAttr2;
          NvU32     retAttr2;
          NvU64     allocAddr       NV_ALIGN_BYTES(8);

          struct
          {
              NvU32 compPageShift;
              NvU32 compressedKind;
              NvU32 compTagLineMin;
              NvU32 compPageIndexLo;
              NvU32 compPageIndexHi;
              NvU32 compTagLineMultiplier;
          } comprInfo;

          NvU32  uncompressedKind;
      } HwAlloc;

      struct
      {
          NvHandle  hResourceHandle;
          NvU32     flags;
      } HwFree;

#define NV_RM_OS32_ALLOC_OS_DESCRIPTOR_WITH_OS32_ATTR   1

      struct
      {
          NvHandle  hMemory;
          NvU32     type;
          NvU32     flags;
          NvU32     attr;
          NvU32     attr2;
          NvP64     descriptor NV_ALIGN_BYTES(8);
          NvU64     limit      NV_ALIGN_BYTES(8);
          NvU32     descriptorType;
      } AllocOsDesc;

  } data;
} NVOS32_PARAMETERS;

#define NVOS32_DELETE_RESOURCES_ALL                     0

#define NVOS32_TYPE_IMAGE                               0
#define NVOS32_TYPE_DEPTH                               1
#define NVOS32_TYPE_TEXTURE                             2
#define NVOS32_TYPE_VIDEO                               3
#define NVOS32_TYPE_FONT                                4
#define NVOS32_TYPE_CURSOR                              5
#define NVOS32_TYPE_DMA                                 6
#define NVOS32_TYPE_INSTANCE                            7
#define NVOS32_TYPE_PRIMARY                             8
#define NVOS32_TYPE_ZCULL                               9
#define NVOS32_TYPE_UNUSED                              10
#define NVOS32_TYPE_SHADER_PROGRAM                      11
#define NVOS32_TYPE_OWNER_RM                            12
#define NVOS32_TYPE_NOTIFIER                            13
#define NVOS32_TYPE_RESERVED                            14
#define NVOS32_TYPE_PMA                                 15
#define NVOS32_TYPE_STENCIL                             16
#define NVOS32_TYPE_SYNCPOINT                           17
#define NVOS32_NUM_MEM_TYPES                            18

#define NVOS32_ATTR_NONE                                0x00000000

#define NVOS32_ATTR_DEPTH                                      2:0
#define NVOS32_ATTR_DEPTH_UNKNOWN                       0x00000000
#define NVOS32_ATTR_DEPTH_8                             0x00000001
#define NVOS32_ATTR_DEPTH_16                            0x00000002
#define NVOS32_ATTR_DEPTH_24                            0x00000003
#define NVOS32_ATTR_DEPTH_32                            0x00000004
#define NVOS32_ATTR_DEPTH_64                            0x00000005
#define NVOS32_ATTR_DEPTH_128                           0x00000006

#define NVOS32_ATTR_COMPR_COVG                                 3:3
#define NVOS32_ATTR_COMPR_COVG_DEFAULT                  0x00000000
#define NVOS32_ATTR_COMPR_COVG_PROVIDED                 0x00000001

#define NVOS32_ATTR_AA_SAMPLES                                 7:4
#define NVOS32_ATTR_AA_SAMPLES_1                        0x00000000
#define NVOS32_ATTR_AA_SAMPLES_2                        0x00000001
#define NVOS32_ATTR_AA_SAMPLES_4                        0x00000002
#define NVOS32_ATTR_AA_SAMPLES_4_ROTATED                0x00000003
#define NVOS32_ATTR_AA_SAMPLES_6                        0x00000004
#define NVOS32_ATTR_AA_SAMPLES_8                        0x00000005
#define NVOS32_ATTR_AA_SAMPLES_16                       0x00000006
#define NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_8              0x00000007
#define NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_16             0x00000008
#define NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_16             0x00000009
#define NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_32             0x0000000A

#define NVOS32_ATTR_GPU_CACHE_SNOOPABLE                        9:8
#define NVOS32_ATTR_GPU_CACHE_SNOOPABLE_MAPPING         0x00000000
#define NVOS32_ATTR_GPU_CACHE_SNOOPABLE_OFF             0x00000001
#define NVOS32_ATTR_GPU_CACHE_SNOOPABLE_ON              0x00000002
#define NVOS32_ATTR_GPU_CACHE_SNOOPABLE_INVALID         0x00000003

#define NVOS32_ATTR_ZCULL                                    11:10
#define NVOS32_ATTR_ZCULL_NONE                          0x00000000
#define NVOS32_ATTR_ZCULL_REQUIRED                      0x00000001
#define NVOS32_ATTR_ZCULL_ANY                           0x00000002
#define NVOS32_ATTR_ZCULL_SHARED                        0x00000003

#define NVOS32_ATTR_COMPR                                    13:12
#define NVOS32_ATTR_COMPR_NONE                          0x00000000
#define NVOS32_ATTR_COMPR_REQUIRED                      0x00000001
#define NVOS32_ATTR_COMPR_ANY                           0x00000002
#define NVOS32_ATTR_COMPR_PLC_REQUIRED                  NVOS32_ATTR_COMPR_REQUIRED
#define NVOS32_ATTR_COMPR_PLC_ANY                       NVOS32_ATTR_COMPR_ANY
#define NVOS32_ATTR_COMPR_DISABLE_PLC_ANY               0x00000003

#define NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP              14:14
#define NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_NO      0x00000000
#define NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_YES     0x00000001

#define NVOS32_ATTR_FORMAT                                   17:16

#define NVOS32_ATTR_FORMAT_LOW_FIELD                            16
#define NVOS32_ATTR_FORMAT_HIGH_FIELD                           17
#define NVOS32_ATTR_FORMAT_PITCH                        0x00000000
#define NVOS32_ATTR_FORMAT_SWIZZLED                     0x00000001
#define NVOS32_ATTR_FORMAT_BLOCK_LINEAR                 0x00000002

#define NVOS32_ATTR_Z_TYPE                                   18:18
#define NVOS32_ATTR_Z_TYPE_FIXED                        0x00000000
#define NVOS32_ATTR_Z_TYPE_FLOAT                        0x00000001

#define NVOS32_ATTR_ZS_PACKING                               21:19
#define NVOS32_ATTR_ZS_PACKING_S8                       0x00000000
#define NVOS32_ATTR_ZS_PACKING_Z24S8                    0x00000000
#define NVOS32_ATTR_ZS_PACKING_S8Z24                    0x00000001
#define NVOS32_ATTR_ZS_PACKING_Z32                      0x00000002
#define NVOS32_ATTR_ZS_PACKING_Z24X8                    0x00000003
#define NVOS32_ATTR_ZS_PACKING_X8Z24                    0x00000004
#define NVOS32_ATTR_ZS_PACKING_Z32_X24S8                0x00000005
#define NVOS32_ATTR_ZS_PACKING_X8Z24_X24S8              0x00000006
#define NVOS32_ATTR_ZS_PACKING_Z16                      0x00000007

#define NVOS32_ATTR_COLOR_PACKING                       NVOS32_ATTR_ZS_PACKING
#define NVOS32_ATTR_COLOR_PACKING_A8R8G8B8              0x00000000
#define NVOS32_ATTR_COLOR_PACKING_X8R8G8B8              0x00000001

#define NVOS32_ATTR_PAGE_SIZE                                24:23
#define NVOS32_ATTR_PAGE_SIZE_DEFAULT                   0x00000000
#define NVOS32_ATTR_PAGE_SIZE_4KB                       0x00000001
#define NVOS32_ATTR_PAGE_SIZE_BIG                       0x00000002
#define NVOS32_ATTR_PAGE_SIZE_HUGE                      0x00000003

#define NVOS32_ATTR_LOCATION                                 26:25
#define NVOS32_ATTR_LOCATION_VIDMEM                     0x00000000
#define NVOS32_ATTR_LOCATION_PCI                        0x00000001
#define NVOS32_ATTR_LOCATION_ANY                        0x00000003

#define NVOS32_ATTR_PHYSICALITY                              28:27
#define NVOS32_ATTR_PHYSICALITY_DEFAULT                 0x00000000
#define NVOS32_ATTR_PHYSICALITY_NONCONTIGUOUS           0x00000001
#define NVOS32_ATTR_PHYSICALITY_CONTIGUOUS              0x00000002
#define NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS     0x00000003

#define NVOS32_ATTR_COHERENCY                                31:29
#define NVOS32_ATTR_COHERENCY_UNCACHED                  0x00000000
#define NVOS32_ATTR_COHERENCY_CACHED                    0x00000001
#define NVOS32_ATTR_COHERENCY_WRITE_COMBINE             0x00000002
#define NVOS32_ATTR_COHERENCY_WRITE_THROUGH             0x00000003
#define NVOS32_ATTR_COHERENCY_WRITE_PROTECT             0x00000004
#define NVOS32_ATTR_COHERENCY_WRITE_BACK                0x00000005

#define NVOS32_ATTR2_NONE                               0x00000000

#define NVOS32_ATTR2_ZBC                                       1:0
#define NVOS32_ATTR2_ZBC_DEFAULT                        0x00000000
#define NVOS32_ATTR2_ZBC_PREFER_NO_ZBC                  0x00000001
#define NVOS32_ATTR2_ZBC_PREFER_ZBC                     0x00000002
#define NVOS32_ATTR2_ZBC_REQUIRE_ONLY_ZBC               0x00000003
#define NVOS32_ATTR2_ZBC_INVALID                        0x00000003

#define NVOS32_ATTR2_GPU_CACHEABLE                             3:2
#define NVOS32_ATTR2_GPU_CACHEABLE_DEFAULT              0x00000000
#define NVOS32_ATTR2_GPU_CACHEABLE_YES                  0x00000001
#define NVOS32_ATTR2_GPU_CACHEABLE_NO                   0x00000002
#define NVOS32_ATTR2_GPU_CACHEABLE_INVALID              0x00000003

#define NVOS32_ATTR2_P2P_GPU_CACHEABLE                         5:4
#define NVOS32_ATTR2_P2P_GPU_CACHEABLE_DEFAULT          0x00000000
#define NVOS32_ATTR2_P2P_GPU_CACHEABLE_YES              0x00000001
#define NVOS32_ATTR2_P2P_GPU_CACHEABLE_NO               0x00000002

#define NVOS32_ATTR2_32BIT_POINTER                             6:6
#define NVOS32_ATTR2_32BIT_POINTER_DISABLE              0x00000000
#define NVOS32_ATTR2_32BIT_POINTER_ENABLE               0x00000001

#define NVOS32_ATTR2_FIXED_NUMA_NODE_ID                        7:7
#define NVOS32_ATTR2_FIXED_NUMA_NODE_ID_NO              0x00000000
#define NVOS32_ATTR2_FIXED_NUMA_NODE_ID_YES             0x00000001

#define NVOS32_ATTR2_SMMU_ON_GPU                               9:8
#define NVOS32_ATTR2_SMMU_ON_GPU_DEFAULT                 0x00000000
#define NVOS32_ATTR2_SMMU_ON_GPU_DISABLE                 0x00000001
#define NVOS32_ATTR2_SMMU_ON_GPU_ENABLE                  0x00000002

#define NVOS32_ATTR2_USE_SCANOUT_CARVEOUT                    10:10
#define NVOS32_ATTR2_USE_SCANOUT_CARVEOUT_FALSE              0x00000000
#define NVOS32_ATTR2_USE_SCANOUT_CARVEOUT_TRUE               0x00000001

#define NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN                11:11
#define NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF              0x0
#define NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_ON               0x1
#define NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_DEFAULT            \
                                   NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF

#define NVOS32_ATTR2_PRIORITY                                 13:12
#define NVOS32_ATTR2_PRIORITY_DEFAULT                           0x0
#define NVOS32_ATTR2_PRIORITY_HIGH                              0x1
#define NVOS32_ATTR2_PRIORITY_LOW                               0x2

#define NVOS32_ATTR2_INTERNAL                                 14:14
#define NVOS32_ATTR2_INTERNAL_NO                                0x0
#define NVOS32_ATTR2_INTERNAL_YES                               0x1

#define NVOS32_ATTR2_PREFER_2C                                15:15
#define NVOS32_ATTR2_PREFER_2C_NO                        0x00000000
#define NVOS32_ATTR2_PREFER_2C_YES                       0x00000001

#define NVOS32_ATTR2_NISO_DISPLAY                             16:16
#define NVOS32_ATTR2_NISO_DISPLAY_NO                     0x00000000
#define NVOS32_ATTR2_NISO_DISPLAY_YES                    0x00000001

#define NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT                     17:17
#define NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_NO             0x00000000
#define NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_YES            0x00000001

#define NVOS32_ATTR2_ISO                                      18:18
#define NVOS32_ATTR2_ISO_NO                              0x00000000
#define NVOS32_ATTR2_ISO_YES                             0x00000001

#define NVOS32_ATTR2_BLACKLIST                                19:19
#define NVOS32_ATTR2_BLACKLIST_ON                        0x00000000
#define NVOS32_ATTR2_BLACKLIST_OFF                       0x00000001
#define NVOS32_ATTR2_PAGE_OFFLINING                           19:19
#define NVOS32_ATTR2_PAGE_OFFLINING_ON                   0x00000000
#define NVOS32_ATTR2_PAGE_OFFLINING_OFF                  0x00000001

#define NVOS32_ATTR2_PAGE_SIZE_HUGE                           21:20
#define NVOS32_ATTR2_PAGE_SIZE_HUGE_DEFAULT              0x00000000
#define NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB                  0x00000001
#define NVOS32_ATTR2_PAGE_SIZE_HUGE_512MB                0x00000002
#define NVOS32_ATTR2_PAGE_SIZE_HUGE_256GB                0x00000003

#define NVOS32_ATTR2_PROTECTION_USER                          22:22
#define NVOS32_ATTR2_PROTECTION_USER_READ_WRITE          0x00000000
#define NVOS32_ATTR2_PROTECTION_USER_READ_ONLY           0x00000001

#define NVOS32_ATTR2_PROTECTION_DEVICE                        23:23
#define NVOS32_ATTR2_PROTECTION_DEVICE_READ_WRITE        0x00000000
#define NVOS32_ATTR2_PROTECTION_DEVICE_READ_ONLY         0x00000001

#define NVOS32_ATTR2_USE_EGM                                 24:24
#define NVOS32_ATTR2_USE_EGM_FALSE                      0x00000000
#define NVOS32_ATTR2_USE_EGM_TRUE                       0x00000001

#define NVOS32_ATTR2_MEMORY_PROTECTION                       26:25
#define NVOS32_ATTR2_MEMORY_PROTECTION_DEFAULT          0x00000000
#define NVOS32_ATTR2_MEMORY_PROTECTION_PROTECTED        0x00000001
#define NVOS32_ATTR2_MEMORY_PROTECTION_UNPROTECTED      0x00000002

#define NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP                   27:27
#define NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_NO           0x00000000
#define NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_YES          0x00000001

#define NVOS32_ATTR2_ENABLE_LOCALIZED_MEMORY            30:29
#define NVOS32_ATTR2_ENABLE_LOCALIZED_MEMORY_DEFAULT    0x00000000
#define NVOS32_ATTR2_ENABLE_LOCALIZED_MEMORY_UGPU0      0x00000001
#define NVOS32_ATTR2_ENABLE_LOCALIZED_MEMORY_UGPU1      0x00000002

#define NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM            31:31
#define NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_FALSE      0x00000000
#define NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_TRUE       0x00000001

#define NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT        0x00000001
#define NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_UP           0x00000002
#define NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_DOWN         0x00000004
#define NVOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE        0x00000008
#define NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE       0x00000010
#define NVOS32_ALLOC_FLAGS_BANK_HINT                    0x00000020
#define NVOS32_ALLOC_FLAGS_BANK_FORCE                   0x00000040
#define NVOS32_ALLOC_FLAGS_ALIGNMENT_HINT               0x00000080
#define NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE              0x00000100
#define NVOS32_ALLOC_FLAGS_BANK_GROW_UP                 0x00000000
#define NVOS32_ALLOC_FLAGS_BANK_GROW_DOWN               0x00000200
#define NVOS32_ALLOC_FLAGS_LAZY                         0x00000400
#define NVOS32_ALLOC_FLAGS_FORCE_REVERSE_ALLOC          0x00000800
#define NVOS32_ALLOC_FLAGS_NO_SCANOUT                   0x00001000
#define NVOS32_ALLOC_FLAGS_PITCH_FORCE                  0x00002000
#define NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED       0x00004000
#define NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED             0x00008000
#define NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM            0x00010000
#define NVOS32_ALLOC_FLAGS_USE_BEGIN_END                0x00020000
#define NVOS32_ALLOC_FLAGS_TURBO_CIPHER_ENCRYPTED       0x00040000
#define NVOS32_ALLOC_FLAGS_VIRTUAL                      0x00080000
#define NVOS32_ALLOC_FLAGS_FORCE_INTERNAL_INDEX         0x00100000
#define NVOS32_ALLOC_FLAGS_ZCULL_COVG_SPECIFIED         0x00200000
#define NVOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED           0x00400000
#define NVOS32_ALLOC_FLAGS_FORCE_DEDICATED_PDE          0x00800000
#define NVOS32_ALLOC_FLAGS_PROTECTED                    0x01000000
#define NVOS32_ALLOC_FLAGS_KERNEL_MAPPING_MAP           0x02000000
#define NVOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE       0x02000000
#define NVOS32_ALLOC_FLAGS_SPARSE                       0x04000000
#define NVOS32_ALLOC_FLAGS_USER_READ_ONLY               0x04000000
#define NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY             0x08000000
#define NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED   0x08000000
#define NVOS32_ALLOC_FLAGS_SKIP_RESOURCE_ALLOC          0x10000000
#define NVOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY     0x20000000
#define NVOS32_ALLOC_FLAGS_SKIP_ALIGN_PAD               0x40000000
#define NVOS32_ALLOC_FLAGS_WPR1                         0x40000000
#define NVOS32_ALLOC_FLAGS_ZCULL_DONT_ALLOCATE_SHARED_1X 0x80000000
#define NVOS32_ALLOC_FLAGS_WPR2                         0x80000000

#define NVOS32_ALLOC_INTERNAL_FLAGS_CLIENTALLOC         0x00000001
#define NVOS32_ALLOC_INTERNAL_FLAGS_SKIP_SCRUB          0x00000004
#define NVOS32_ALLOC_FLAGS_MAXIMIZE_4GB_ADDRESS_SPACE NVOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE

#define NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY         ( \
    NVOS32_ALLOC_FLAGS_VIRTUAL                  | \
    NVOS32_ALLOC_FLAGS_LAZY                     | \
    NVOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED       | \
    NVOS32_ALLOC_FLAGS_SPARSE                   | \
    NVOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE   | \
    NVOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY )

#define NVOS32_ALLOC_COMPR_COVG_SCALE                           10
#define NVOS32_ALLOC_COMPR_COVG_BITS                           1:0
#define NVOS32_ALLOC_COMPR_COVG_BITS_DEFAULT            0x00000000
#define NVOS32_ALLOC_COMPR_COVG_BITS_1                  0x00000001
#define NVOS32_ALLOC_COMPR_COVG_BITS_2                  0x00000002
#define NVOS32_ALLOC_COMPR_COVG_BITS_4                  0x00000003
#define NVOS32_ALLOC_COMPR_COVG_MAX                           11:2
#define NVOS32_ALLOC_COMPR_COVG_MIN                          21:12
#define NVOS32_ALLOC_COMPR_COVG_START                        31:22

#define NVOS32_ALLOC_ZCULL_COVG_FORMAT                         3:0
#define NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_Z        0x00000000
#define NVOS32_ALLOC_ZCULL_COVG_FORMAT_HIGH_RES_Z       0x00000002
#define NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_ZS       0x00000003
#define NVOS32_ALLOC_ZCULL_COVG_FALLBACK                       4:4
#define NVOS32_ALLOC_ZCULL_COVG_FALLBACK_DISALLOW       0x00000000
#define NVOS32_ALLOC_ZCULL_COVG_FALLBACK_ALLOW          0x00000001

#define NVOS32_ALLOC_COMPTAG_OFFSET_START                     19:0
#define NVOS32_ALLOC_COMPTAG_OFFSET_START_DEFAULT       0x00000000
#define NVOS32_ALLOC_COMPTAG_OFFSET_USAGE                    31:30
#define NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_DEFAULT       0x00000000
#define NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_OFF           0x00000000
#define NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_FIXED         0x00000001
#define NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_MIN           0x00000002

#define NVOS32_REALLOC_FLAGS_GROW_ALLOCATION            0x00000000
#define NVOS32_REALLOC_FLAGS_SHRINK_ALLOCATION          0x00000001
#define NVOS32_REALLOC_FLAGS_REALLOC_UP                 0x00000000
#define NVOS32_REALLOC_FLAGS_REALLOC_DOWN               0x00000002

#define NVOS32_RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED   0x000000001

#define NVOS32_REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED 0x000000001

#define NVOS32_FREE_FLAGS_MEMORY_HANDLE_PROVIDED        0x00000001

#define NVOS32_DUMP_FLAGS_TYPE                                 1:0
#define NVOS32_DUMP_FLAGS_TYPE_FB                       0x00000000
#define NVOS32_DUMP_FLAGS_TYPE_CLIENT_PD                0x00000001
#define NVOS32_DUMP_FLAGS_TYPE_CLIENT_VA                0x00000002
#define NVOS32_DUMP_FLAGS_TYPE_CLIENT_VAPTE             0x00000003

#define NVOS32_BLOCK_TYPE_FREE                          0xFFFFFFFF
#define NVOS32_INVALID_BLOCK_FREE_OFFSET                0xFFFFFFFF

#define NVOS32_MEM_TAG_NONE                             0x00000000

typedef struct
{
    NvHandle hSubDevice;
    NvV32    flags;
    NvHandle hMemory;
    NvU64    offset NV_ALIGN_BYTES(8);
    NvU64    limit NV_ALIGN_BYTES(8);
} NV_CONTEXT_DMA_ALLOCATION_PARAMS;

typedef struct
{
    NvU32     owner;
    NvU32     type;
    NvU32     flags;

    NvU32     width;
    NvU32     height;
    NvS32     pitch;

    NvU32     attr;
    NvU32     attr2;

    NvU32     format;
    NvU32     comprCovg;
    NvU32     zcullCovg;

    NvU64     rangeLo   NV_ALIGN_BYTES(8);
    NvU64     rangeHi   NV_ALIGN_BYTES(8);

    NvU64     size      NV_ALIGN_BYTES(8);
    NvU64     alignment NV_ALIGN_BYTES(8);
    NvU64     offset    NV_ALIGN_BYTES(8);
    NvU64     limit     NV_ALIGN_BYTES(8);
    NvP64     address   NV_ALIGN_BYTES(8);

    NvU32     ctagOffset;
    NvHandle  hVASpace;

    NvU32     internalflags;

    NvU32     tag;

    NvS32     numaNode;
} NV_MEMORY_ALLOCATION_PARAMS;

typedef struct
{
    NvU32     type;
    NvU32     flags;
    NvU32     attr;
    NvU32     attr2;
    NvP64     descriptor NV_ALIGN_BYTES(8);
    NvU64     limit      NV_ALIGN_BYTES(8);
    NvU32     descriptorType;
    NvU32     tag;
} NV_OS_DESC_MEMORY_ALLOCATION_PARAMS;

typedef struct
{
    NvU32     flags;
    NvU64     physAddr   NV_ALIGN_BYTES(8);
    NvU64     size       NV_ALIGN_BYTES(8);
    NvU32     tag;
    NvBool    bGuestAllocated;
} NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS;

typedef struct
{
    NvU32     owner;
    NvU32     flags;
    NvU32     type;

    NvU32     attr;
    NvU32     attr2;

    NvU32     height;
    NvU32     width;
    NvU32     pitch;
    NvU32     alignment;
    NvU32     comprCovg;
    NvU32     zcullCovg;

    NvU32     kind;

    NvP64     bindResultFunc  NV_ALIGN_BYTES(8);
    NvP64     pHandle         NV_ALIGN_BYTES(8);
    NvU64     osDeviceHandle  NV_ALIGN_BYTES(8);
    NvU64     size            NV_ALIGN_BYTES(8);
    NvU64     allocAddr       NV_ALIGN_BYTES(8);

    NvU32 compPageShift;
    NvU32 compressedKind;
    NvU32 compTagLineMin;
    NvU32 compPageIndexLo;
    NvU32 compPageIndexHi;
    NvU32 compTagLineMultiplier;

    NvU32  uncompressedKind;

    NvU32     tag;
} NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS;

#define NV04_MAP_MEMORY                                 (0x00000021)

#define NV04_MAP_MEMORY_FLAGS_NONE                      (0x00000000)
#define NV04_MAP_MEMORY_FLAGS_USER                      (0x00004000)

#define NVOS33_FLAGS_ACCESS                                        1:0
#define NVOS33_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define NVOS33_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define NVOS33_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

#define NVOS33_FLAGS_PERSISTENT                                    4:4
#define NVOS33_FLAGS_PERSISTENT_DISABLE                            (0x00000000)
#define NVOS33_FLAGS_PERSISTENT_ENABLE                             (0x00000001)

#define NVOS33_FLAGS_SKIP_SIZE_CHECK                               8:8
#define NVOS33_FLAGS_SKIP_SIZE_CHECK_DISABLE                       (0x00000000)
#define NVOS33_FLAGS_SKIP_SIZE_CHECK_ENABLE                        (0x00000001)

#define NVOS33_FLAGS_MEM_SPACE                                     14:14
#define NVOS33_FLAGS_MEM_SPACE_CLIENT                              (0x00000000)
#define NVOS33_FLAGS_MEM_SPACE_USER                                (0x00000001)

#define NVOS33_FLAGS_MAPPING                                       16:15
#define NVOS33_FLAGS_MAPPING_DEFAULT                               (0x00000000)
#define NVOS33_FLAGS_MAPPING_DIRECT                                (0x00000001)
#define NVOS33_FLAGS_MAPPING_REFLECTED                             (0x00000002)

#define NVOS33_FLAGS_FIFO_MAPPING                                  17:17
#define NVOS33_FLAGS_FIFO_MAPPING_DEFAULT                          (0x00000000)
#define NVOS33_FLAGS_FIFO_MAPPING_ENABLE                           (0x00000001)

#define NVOS33_FLAGS_MAP_FIXED                                     18:18
#define NVOS33_FLAGS_MAP_FIXED_DISABLE                             (0x00000000)
#define NVOS33_FLAGS_MAP_FIXED_ENABLE                              (0x00000001)

#define NVOS33_FLAGS_RESERVE_ON_UNMAP                              19:19
#define NVOS33_FLAGS_RESERVE_ON_UNMAP_DISABLE                      (0x00000000)
#define NVOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE                       (0x00000001)

#define NVOS33_FLAGS_OS_DESCRIPTOR                                 22:22
#define NVOS33_FLAGS_OS_DESCRIPTOR_DISABLE                         (0x00000000)
#define NVOS33_FLAGS_OS_DESCRIPTOR_ENABLE                          (0x00000001)

#define NVOS33_FLAGS_CACHING_TYPE                                  25:23
#define NVOS33_FLAGS_CACHING_TYPE_CACHED                           0
#define NVOS33_FLAGS_CACHING_TYPE_UNCACHED                         1
#define NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED                    2
#define NVOS33_FLAGS_CACHING_TYPE_WRITEBACK                        5
#define NVOS33_FLAGS_CACHING_TYPE_DEFAULT                          6
#define NVOS33_FLAGS_CACHING_TYPE_UNCACHED_WEAK                    7

#define NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC                     26:26
#define NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_NO                  (0x00000000)
#define NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_YES                 (0x00000001)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hMemory;
    NvU64    offset NV_ALIGN_BYTES(8);
    NvU64    length NV_ALIGN_BYTES(8);
    NvP64    pLinearAddress NV_ALIGN_BYTES(8);
    NvU32    status;
    NvU32    flags;
} NVOS33_PARAMETERS;

#define NV04_UNMAP_MEMORY                                          (0x00000022)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hMemory;
    NvP64    pLinearAddress NV_ALIGN_BYTES(8);
    NvU32    status;
    NvU32    flags;
} NVOS34_PARAMETERS;

#define NV04_ACCESS_REGISTRY                                       (0x00000026)

#define NVOS38_ACCESS_TYPE_READ_DWORD                                        1
#define NVOS38_ACCESS_TYPE_WRITE_DWORD                                       2
#define NVOS38_ACCESS_TYPE_READ_BINARY                                       6
#define NVOS38_ACCESS_TYPE_WRITE_BINARY                                      7

#define NVOS38_MAX_REGISTRY_STRING_LENGTH                                  256
#define NVOS38_MAX_REGISTRY_BINARY_LENGTH                                  256

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    AccessType;

    NvV32    DevNodeLength;
    NvP64    pDevNode NV_ALIGN_BYTES(8);

    NvV32    ParmStrLength;
    NvP64    pParmStr NV_ALIGN_BYTES(8);

    NvV32    BinaryDataLength;
    NvP64    pBinaryData NV_ALIGN_BYTES(8);

    NvV32    Data;
    NvV32    Entry;
    NvV32    status;
} NVOS38_PARAMETERS;

#define  NV04_ALLOC_CONTEXT_DMA                                    (0x00000027)

typedef struct
{
    NvHandle hObjectParent;
    NvHandle hSubDevice;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvV32    flags;
    NvU32    selector;
    NvHandle hMemory;
    NvU64    offset NV_ALIGN_BYTES(8);
    NvU64    limit NV_ALIGN_BYTES(8);
    NvV32    status;
} NVOS39_PARAMETERS;

#define NV04_GET_EVENT_DATA                                        (0x00000028)

typedef struct
{
    NvHandle hObject;
    NvV32    NotifyIndex;

    NvV32    info32;
    NvU16    info16;
} NvUnixEvent;

typedef struct
{
    NvP64 pEvent NV_ALIGN_BYTES(8);
    NvV32 MoreEvents;
    NvV32 status;
} NVOS41_PARAMETERS;

#define  NVSIM01_BUS_XACT                                          (0x0000002C)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvU32    offset;
    NvU32    bar;
    NvU32    bytes;
    NvU32    write;
    NvU32    data;
    NvU32    status;
} NVOS2C_PARAMETERS;

#define NV04_MAP_MEMORY_DMA                                        (0x0000002E)

#define NVOS46_FLAGS_ACCESS                                        1:0
#define NVOS46_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define NVOS46_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define NVOS46_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

#define NVOS46_FLAGS_32BIT_POINTER                                 2:2
#define NVOS46_FLAGS_32BIT_POINTER_DISABLE                         (0x00000000)
#define NVOS46_FLAGS_32BIT_POINTER_ENABLE                          (0x00000001)

#define NVOS46_FLAGS_PAGE_KIND                                     3:3
#define NVOS46_FLAGS_PAGE_KIND_PHYSICAL                            (0x00000000)
#define NVOS46_FLAGS_PAGE_KIND_VIRTUAL                             (0x00000001)

#define NVOS46_FLAGS_CACHE_SNOOP                                   4:4
#define NVOS46_FLAGS_CACHE_SNOOP_DISABLE                           (0x00000000)
#define NVOS46_FLAGS_CACHE_SNOOP_ENABLE                            (0x00000001)

#define NVOS46_FLAGS_KERNEL_MAPPING                                5:5
#define NVOS46_FLAGS_KERNEL_MAPPING_NONE                           (0x00000000)
#define NVOS46_FLAGS_KERNEL_MAPPING_ENABLE                         (0x00000001)

#define NVOS46_FLAGS_SHADER_ACCESS                                 7:6
#define NVOS46_FLAGS_SHADER_ACCESS_DEFAULT                         (0x00000000)
#define NVOS46_FLAGS_SHADER_ACCESS_READ_ONLY                       (0x00000001)
#define NVOS46_FLAGS_SHADER_ACCESS_WRITE_ONLY                      (0x00000002)
#define NVOS46_FLAGS_SHADER_ACCESS_READ_WRITE                      (0x00000003)

#define NVOS46_FLAGS_PAGE_SIZE                                     11:8
#define NVOS46_FLAGS_PAGE_SIZE_DEFAULT                             (0x00000000)
#define NVOS46_FLAGS_PAGE_SIZE_4KB                                 (0x00000001)
#define NVOS46_FLAGS_PAGE_SIZE_BIG                                 (0x00000002)
#define NVOS46_FLAGS_PAGE_SIZE_BOTH                                (0x00000003)
#define NVOS46_FLAGS_PAGE_SIZE_HUGE                                (0x00000004)
#define NVOS46_FLAGS_PAGE_SIZE_512M                                (0x00000005)

#define NVOS46_FLAGS_SYSTEM_L3_ALLOC                               13:13
#define NVOS46_FLAGS_SYSTEM_L3_ALLOC_DEFAULT                       (0x00000000)
#define NVOS46_FLAGS_SYSTEM_L3_ALLOC_ENABLE_HINT                   (0x00000001)

#define NVOS46_FLAGS_DMA_OFFSET_GROWS                              14:14
#define NVOS46_FLAGS_DMA_OFFSET_GROWS_UP                           (0x00000000)
#define NVOS46_FLAGS_DMA_OFFSET_GROWS_DOWN                         (0x00000001)

#define NVOS46_FLAGS_DMA_OFFSET_FIXED                              15:15
#define NVOS46_FLAGS_DMA_OFFSET_FIXED_FALSE                        (0x00000000)
#define NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE                         (0x00000001)

#define NVOS46_FLAGS_DISABLE_ENCRYPTION                            16:16
#define NVOS46_FLAGS_DISABLE_ENCRYPTION_FALSE                      (0x00000000)
#define NVOS46_FLAGS_DISABLE_ENCRYPTION_TRUE                       (0x00000001)

#define NVOS46_FLAGS_GPU_CACHEABLE                                 18:17
#define NVOS46_FLAGS_GPU_CACHEABLE_DEFAULT                         (0x00000000)
#define NVOS46_FLAGS_GPU_CACHEABLE_YES                             (0x00000001)
#define NVOS46_FLAGS_GPU_CACHEABLE_NO                              (0x00000002)
#define NVOS46_FLAGS_GPU_CACHEABLE_INVALID                         (0x00000003)

#define NVOS46_FLAGS_PAGE_KIND_OVERRIDE                            19:19
#define NVOS46_FLAGS_PAGE_KIND_OVERRIDE_NO                         (0x00000000)
#define NVOS46_FLAGS_PAGE_KIND_OVERRIDE_YES                        (0x00000001)

#define NVOS46_FLAGS_P2P                                           27:20

#define NVOS46_FLAGS_P2P_ENABLE                                    21:20
#define NVOS46_FLAGS_P2P_ENABLE_NO                                 (0x00000000)
#define NVOS46_FLAGS_P2P_ENABLE_YES                                (0x00000001)
#define NVOS46_FLAGS_P2P_ENABLE_NONE                               NVOS46_FLAGS_P2P_ENABLE_NO
#define NVOS46_FLAGS_P2P_ENABLE_SLI                                NVOS46_FLAGS_P2P_ENABLE_YES
#define NVOS46_FLAGS_P2P_ENABLE_NOSLI                              (0x00000002)

#define NVOS46_FLAGS_P2P_SUBDEVICE_ID                              24:22
#define NVOS46_FLAGS_P2P_SUBDEV_ID_SRC                             NVOS46_FLAGS_P2P_SUBDEVICE_ID
#define NVOS46_FLAGS_P2P_SUBDEV_ID_TGT                             27:25
#define NVOS46_FLAGS_TLB_LOCK                                      28:28
#define NVOS46_FLAGS_TLB_LOCK_DISABLE                              (0x00000000)
#define NVOS46_FLAGS_TLB_LOCK_ENABLE                               (0x00000001)
#define NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC                       29:29
#define NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_FALSE                 (0x00000000)
#define NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_TRUE                  (0x00000001)

#define NVOS46_FLAGS_ENABLE_FORCE_COMPRESSED_MAP                   30:30
#define NVOS46_FLAGS_ENABLE_FORCE_COMPRESSED_MAP_FALSE             (0x00000000)
#define NVOS46_FLAGS_ENABLE_FORCE_COMPRESSED_MAP_TRUE              (0x00000001)

#define NVOS46_FLAGS_DEFER_TLB_INVALIDATION                        31:31
#define NVOS46_FLAGS_DEFER_TLB_INVALIDATION_FALSE                  (0x00000000)
#define NVOS46_FLAGS_DEFER_TLB_INVALIDATION_TRUE                   (0x00000001)

#define NVOS46_FLAGS2_GPU_CACHE_SNOOP                          1:0
#define NVOS46_FLAGS2_GPU_CACHE_SNOOP_DEFAULT                  (0x00000000)
#define NVOS46_FLAGS2_GPU_CACHE_SNOOP_ENABLE                   (0x00000001)
#define NVOS46_FLAGS2_GPU_CACHE_SNOOP_DISABLE                  (0x00000002)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hDma;
    NvHandle hMemory;
    NvU64    offset NV_ALIGN_BYTES(8);
    NvU64    length NV_ALIGN_BYTES(8);
    NvV32    flags;
    NvV32    flags2;
    NvV32    kindOverride;
    NvU64    dmaOffset NV_ALIGN_BYTES(8);

    NvV32    status;
} NVOS46_PARAMETERS;

typedef NVOS46_PARAMETERS NV_MAP_MEMORY_DMA_PARAMETERS;

#define NV04_UNMAP_MEMORY_DMA                                      (0x0000002F)

#define NVOS47_FLAGS_DEFER_TLB_INVALIDATION                        0:0
#define NVOS47_FLAGS_DEFER_TLB_INVALIDATION_FALSE                  (0x00000000)
#define NVOS47_FLAGS_DEFER_TLB_INVALIDATION_TRUE                   (0x00000001)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hDma;
    NvHandle hMemory;
    NvV32    flags;
    NvU64    dmaOffset NV_ALIGN_BYTES(8);
    NvU64    size NV_ALIGN_BYTES(8);
    NvV32    status;
} NVOS47_PARAMETERS;

typedef NVOS47_PARAMETERS NV_UNMAP_MEMORY_DMA_PARAMETERS;

#define NV04_BIND_CONTEXT_DMA                                      (0x00000031)

typedef struct
{
    NvHandle    hClient;
    NvHandle    hChannel;
    NvHandle    hCtxDma;
    NvV32       status;
} NVOS49_PARAMETERS;

#define NV04_CONTROL                                               (0x00000036)

#define NVOS54_FLAGS_NONE                                          (0x00000000)
#define NVOS54_FLAGS_IRQL_RAISED                                   (0x00000001)
#define NVOS54_FLAGS_LOCK_BYPASS                                   (0x00000002)
#define NVOS54_FLAGS_FINN_SERIALIZED                               (0x00000004)

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvP64    params NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    paramsSize;
    NvV32    status;
} NVOS63_PARAMETERS;

#define NV04_DUP_OBJECT                                             (0x00000037)

typedef struct
{
  NvHandle  hClient;
  NvHandle  hParent;
  NvHandle  hObject;
  NvHandle  hClientSrc;
  NvHandle  hObjectSrc;
  NvU32     flags;
  NvU32     status;
} NVOS55_PARAMETERS;

#define NV04_DUP_HANDLE_FLAGS_NONE                                  (0x00000000)
#define NV04_DUP_HANDLE_FLAGS_REJECT_KERNEL_DUP_PRIVILEGE           (0x00000001)

#define NV04_UPDATE_DEVICE_MAPPING_INFO                             (0x00000038)

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hMemory;
    NvP64    pOldCpuAddress NV_ALIGN_BYTES(8);
    NvP64    pNewCpuAddress NV_ALIGN_BYTES(8);
    NvV32    status;
} NVOS56_PARAMETERS;

#define NV04_SHARE                                             (0x0000003E)

typedef struct
{
  NvHandle        hClient;
  NvHandle        hObject;
  RS_SHARE_POLICY sharePolicy;
  NvU32           status;
} NVOS57_PARAMETERS;

typedef NvU32 NVPOWERSTATE_STAGE;
#define NVPOWERSTATE_STAGE_NONE                                                     0U
#define NVPOWERSTATE_STAGE_WAIT_FOR_GFW_BOOT_OK                                     1U
#define NVPOWERSTATE_STAGE_INIT_LIBOS_LOGGING_STRUCTURES                            2U
#define NVPOWERSTATE_STAGE_GSP_PREPARE_FOR_BOOTSTRAP                                3U
#define NVPOWERSTATE_STAGE_GSP_BOOTSTRAP                                            4U
#define NVPOWERSTATE_STAGE_BOOT_GSP_RM_PROXY                                        5U
#define NVPOWERSTATE_STAGE_VBIOS_HANDLE_SECURE_BOOT                                 6U
#define NVPOWERSTATE_STAGE_RESTORE_PCIE_CONFIG_REGISTERS                            7U
#define NVPOWERSTATE_STAGE_GCX_BOOT_TIMER_CB_SCHEDULE                               8U
#define NVPOWERSTATE_STAGE_POWER_MANAGEMENT_RESUME_PRE_LOAD_PHYSICAL_UNATTACHED     9U
#define NVPOWERSTATE_STAGE_PMS_EXPECTED_CHECKPOINT_DONE                             10U
#define NVPOWERSTATE_STAGE_POLL_FOR_NVLINK_READY                                    11U
#define NVPOWERSTATE_STAGE_CE_STATE_PRE_LOAD                                        12U
#define NVPOWERSTATE_STAGE_LOAD_PROXY_UCODE_EARLY_INIT                              13U
#define NVPOWERSTATE_STAGE_RESTORE_NON_WPR_REGION                                   14U
#define NVPOWERSTATE_STAGE_PMU_20_OS_BOOTSTRAP                                      15U
#define NVPOWERSTATE_STAGE_STATE_PRE_LOAD_ENGINE                                    16U
#define NVPOWERSTATE_STAGE_STATE_PRE_LOAD_UNKNOWN                                   17U
#define NVPOWERSTATE_STAGE_STATE_LOAD_ENGINE                                        18U
#define NVPOWERSTATE_STAGE_STATE_LOAD_UNKNOWN                                       19U
#define NVPOWERSTATE_STAGE_STATE_LOAD_PHYSICAL                                      20U
#define NVPOWERSTATE_STAGE_STATE_POST_LOAD_ENGINE                                   21U
#define NVPOWERSTATE_STAGE_STATE_POST_LOAD_UNKNOWN                                  22U
#define NVPOWERSTATE_STAGE_GSP_PREPARE_SUSPEND_RESUME_DATA                          23U
#define NVPOWERSTATE_STAGE_MC_POINTER_NULL                                          24U
#define NVPOWERSTATE_STAGE_SAVE_PCIE_CONFIG_REGISTERS                               25U
#define NVPOWERSTATE_STAGE_STATE_PRE_UNLOAD_ENGINE                                  26U
#define NVPOWERSTATE_STAGE_STATE_PRE_UNLOAD_UNKNOWN                                 27U
#define NVPOWERSTATE_STAGE_STATE_UNLOAD_ENGINE                                      28U
#define NVPOWERSTATE_STAGE_STATE_UNLOAD_UNKNOWN                                     29U
#define NVPOWERSTATE_STAGE_STATE_POST_UNLOAD_ENGINE                                 30U
#define NVPOWERSTATE_STAGE_STATE_POST_UNLOAD_UNKNOWN                                31U
#define NVPOWERSTATE_STAGE_GSP_UNLOAD_RM                                            32U
#define NVPOWERSTATE_STAGE_MONITOR_STATE_0                                          33U
#define NVPOWERSTATE_STAGE_MONITOR_STATE_1                                          34U
#define NVPOWERSTATE_STAGE_MONITOR_STATE_HIBERNATE                                  35U
#define NVPOWERSTATE_STAGE_SET_POWER_STATE_UNKNOWN                                  36U
#define NVPOWERSTATE_STAGE__COUNT                                                   37U

typedef struct
{

    NvU32 classId;

    NvU64 cumulativeTimeus;

    NvU32 engineMaxTimeus;

    NvU32 engineMaxTimeClassId;
} NVPOWERSTATE_FAILURE_ENGINE_TRANSITION;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_WAIT_FOR_GFW_BOOT_OK;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_INIT_LIBOS_LOGGING_STRUCTURES;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_GSP_PREPARE_FOR_BOOTSTRAP;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_GSP_BOOTSTRAP;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_BOOT_GSP_RM_PROXY;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_VBIOS_HANDLE_SECURE_BOOT;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_RESTORE_PCIE_CONFIG_REGISTERS;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_GCX_BOOT_TIMER_CB_SCHEDULE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_POWER_MANAGEMENT_RESUME_PRE_LOAD_PHYSICAL_UNATTACHED;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_PMS_EXPECTED_CHECKPOINT_DONE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_POLL_FOR_NVLINK_READY;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_CE_STATE_PRE_LOAD;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_LOAD_PROXY_UCODE_EARLY_INIT;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_RESTORE_NON_WPR_REGION;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_PMU_20_OS_BOOTSTRAP;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_PRE_LOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_PRE_LOAD_UNKNOWN;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_LOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_LOAD_UNKNOWN;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_POST_LOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_POST_LOAD_UNKNOWN;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_LOAD_PHYSICAL;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_GSP_PREPARE_SUSPEND_RESUME_DATA;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_MC_POINTER_NULL;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_SAVE_PCIE_CONFIG_REGISTERS;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_PRE_UNLOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_PRE_UNLOAD_UNKNOWN;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_UNLOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_UNLOAD_UNKNOWN;

typedef struct
{
    NVPOWERSTATE_FAILURE_ENGINE_TRANSITION engineTransition;
} NVPOWERSTATE_FAILURE_STATE_POST_UNLOAD_ENGINE;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_STATE_POST_UNLOAD_UNKNOWN;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_GSP_UNLOAD_RM;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_MONITOR_STATE_0;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_MONITOR_STATE_1;

typedef struct
{
    NvU8 rsvd;
} NVPOWERSTATE_FAILURE_MONITOR_STATE_HIBERNATE;

typedef struct
{

    NvU32 state;
} NVPOWERSTATE_FAILURE_SET_POWER_STATE_UNKNOWN;

typedef union
{
    NVPOWERSTATE_FAILURE_WAIT_FOR_GFW_BOOT_OK waitForGfwBootOk;
    NVPOWERSTATE_FAILURE_INIT_LIBOS_LOGGING_STRUCTURES initLibosLoggingStructures;
    NVPOWERSTATE_FAILURE_GSP_PREPARE_FOR_BOOTSTRAP gspPrepareForBootstrap;
    NVPOWERSTATE_FAILURE_GSP_BOOTSTRAP gspBootstrap;
    NVPOWERSTATE_FAILURE_BOOT_GSP_RM_PROXY bootGspRmProxy;
    NVPOWERSTATE_FAILURE_VBIOS_HANDLE_SECURE_BOOT vbiosHandleSecureBoot;
    NVPOWERSTATE_FAILURE_RESTORE_PCIE_CONFIG_REGISTERS restorePcieConfigRegisters;
    NVPOWERSTATE_FAILURE_GCX_BOOT_TIMER_CB_SCHEDULE gcxBootTimerCbSchedule;
    NVPOWERSTATE_FAILURE_POWER_MANAGEMENT_RESUME_PRE_LOAD_PHYSICAL_UNATTACHED powerManagementResumePreLoadPhysicalUnattached;
    NVPOWERSTATE_FAILURE_PMS_EXPECTED_CHECKPOINT_DONE pmsExpectedCheckpointDone;
    NVPOWERSTATE_FAILURE_POLL_FOR_NVLINK_READY pollForNvlinkReady;
    NVPOWERSTATE_FAILURE_CE_STATE_PRE_LOAD ceStatePreLoad;
    NVPOWERSTATE_FAILURE_LOAD_PROXY_UCODE_EARLY_INIT loadProxyUcodeEarlyInit;
    NVPOWERSTATE_FAILURE_RESTORE_NON_WPR_REGION restoreNonWprRegion;
    NVPOWERSTATE_FAILURE_PMU_20_OS_BOOTSTRAP pmu20OsBootstrap;
    NVPOWERSTATE_FAILURE_STATE_PRE_LOAD_ENGINE statePreLoadEngine;
    NVPOWERSTATE_FAILURE_STATE_PRE_LOAD_UNKNOWN statePreLoadUnknown;
    NVPOWERSTATE_FAILURE_STATE_LOAD_ENGINE stateLoadEngine;
    NVPOWERSTATE_FAILURE_STATE_LOAD_UNKNOWN stateLoadUnknown;
    NVPOWERSTATE_FAILURE_STATE_POST_LOAD_ENGINE statePostLoadEngine;
    NVPOWERSTATE_FAILURE_STATE_POST_LOAD_UNKNOWN statePostLoadUnknown;
    NVPOWERSTATE_FAILURE_STATE_LOAD_PHYSICAL stateLoadPhysical;
    NVPOWERSTATE_FAILURE_GSP_PREPARE_SUSPEND_RESUME_DATA gspPrepareSuspendResumeData;
    NVPOWERSTATE_FAILURE_MC_POINTER_NULL mcPointerNull;
    NVPOWERSTATE_FAILURE_SAVE_PCIE_CONFIG_REGISTERS savePcieConfigRegisters;
    NVPOWERSTATE_FAILURE_STATE_PRE_UNLOAD_ENGINE statePreUnloadEngine;
    NVPOWERSTATE_FAILURE_STATE_PRE_UNLOAD_UNKNOWN statePreUnloadUnknown;
    NVPOWERSTATE_FAILURE_STATE_UNLOAD_ENGINE stateUnloadEngine;
    NVPOWERSTATE_FAILURE_STATE_UNLOAD_UNKNOWN stateUnloadUnknown;
    NVPOWERSTATE_FAILURE_STATE_POST_UNLOAD_ENGINE statePostUnloadEngine;
    NVPOWERSTATE_FAILURE_STATE_POST_UNLOAD_UNKNOWN statePostUnloadUnknown;
    NVPOWERSTATE_FAILURE_GSP_UNLOAD_RM gspUnloadRm;
    NVPOWERSTATE_FAILURE_MONITOR_STATE_0 monitorState0;
    NVPOWERSTATE_FAILURE_MONITOR_STATE_1 monitorState1;
    NVPOWERSTATE_FAILURE_MONITOR_STATE_HIBERNATE monitorStateHibernate;
    NVPOWERSTATE_FAILURE_SET_POWER_STATE_UNKNOWN setPowerStateUnknown;
} NVPOWERSTATE_FAILURE_DATA;

typedef struct
{

    NVPOWERSTATE_STAGE stage;

    NV_STATUS status;

    NVPOWERSTATE_FAILURE_DATA data;
} NVPOWERSTATE_FAILURE;

static NV_FORCEINLINE void
nvPowerStateFailureInit
(
    NVPOWERSTATE_FAILURE *pFailure
)
{
    (void)NVMISC_MEMSET(
        pFailure,
        0x0,
        sizeof(*pFailure));
}

static NV_FORCEINLINE NvBool
nvPowerStateFailureIsPopulated
(
    const NVPOWERSTATE_FAILURE *pFailure
)
{
    return pFailure->stage != NVPOWERSTATE_STAGE_NONE;
}

typedef struct
{
    NvU32 deviceReference;
    NvU32 head;
    NvU32 state;
    NvU8  forceMonitorState;
    NvU8  bForcePerfBiosLevel;
    NvU8  bIsD3HotTransition;
    NvU8  bForcePowerStateFail;
    NvU32 errorStatus;
    NvU32 fastBootPowerState;
    NvU8  bGC8Transition;
    NvU8  bGC8InputRailCutOff;
    NVPOWERSTATE_FAILURE failure;
} NVPOWERSTATE_PARAMETERS, *PNVPOWERSTATE_PARAMETERS;

typedef struct {
    NvU32   version;
    NvU32   flags;
    NvU32   size;
    NvU32   caps;
} NV_GR_ALLOCATION_PARAMETERS;

#define NV_DEVICE_ALLOCATION_SZNAME_MAXLEN    128
#define NV_DEVICE_ALLOCATION_FLAGS_NONE                            (0x00000000)
#define NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE_GLOBALLY                (0x00000001)
#define NV_DEVICE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE          (0x00000002)
#define NV_DEVICE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS          (0x00000004)
#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SIZE                    (0x00000008)
#define NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE                         (0x00000010)
#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_TARGET               (0x00000020)
#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SHARED_MANAGEMENT       (0x00000100)
#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_64k       (0x00000200)
#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_128k      (0x00000400)
#define NV_DEVICE_ALLOCATION_FLAGS_RESTRICT_RESERVED_VALIMITS      (0x00000800)

#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_MIRRORED             (0x00000040)

#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_PTABLE_PMA_MANAGED      (0x00001000)

#define NV_DEVICE_ALLOCATION_FLAGS_HOST_VGPU_DEVICE                (0x00002000)

#define NV_DEVICE_ALLOCATION_FLAGS_PLUGIN_CONTEXT                  (0x00004000)

#define NV_DEVICE_ALLOCATION_FLAGS_VASPACE_REQUIRE_FIXED_OFFSET    (0x00008000)

#define NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES     (0x00000000)
#define NV_DEVICE_ALLOCATION_VAMODE_SINGLE_VASPACE                 (0x00000001)
#define NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES              (0x00000002)

#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR                0x00000000
#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN    0x00000001
#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE_KEY_ROTATION_STATUS  0x00000002
#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1              3
#define NV_CHANNELGPFIFO_NOTIFICATION_STATUS_VALUE              14:0
#define NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS        15:15
#define NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_TRUE   0x1
#define NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_FALSE  0x0

typedef enum
{
    PB_SIZE_4KB = 0,
    PB_SIZE_8KB,
    PB_SIZE_16KB,
    PB_SIZE_32KB,
    PB_SIZE_64KB
} ChannelPBSize;

typedef struct
{
    NvV32         channelInstance;

    NvHandle      hObjectBuffer;
    NvHandle      hObjectNotify;
    NvU32         offset;
    NvP64         pControl NV_ALIGN_BYTES(8);

    NvU32         flags;
    ChannelPBSize channelPBSize;
#define NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB                1:1
#define NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_YES            0x00000000
#define NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_NO             0x00000001

    NvU32    subDeviceId;
} NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS;

typedef struct
{
    NvV32    channelInstance;

    NvHandle hObjectNotify;
    NvP64    pControl NV_ALIGN_BYTES(8);
} NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS;

typedef struct
{
    NvHandle hObjectError;
    NvHandle hObjectEccError;
    NvHandle hVASpace;
    NvU32    engineType;
    NvBool   bIsCallingContextVgpuPlugin;
} NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32    engineId;
    NvU32    maxTSGs;
    NvU32    qosIntrEnableMask;
} NV_SWRUNLIST_ALLOCATION_PARAMS;

#define NV_SWRUNLIST_QOS_INTR_NONE                                   0x00000000
#define NV_SWRUNLIST_QOS_INTR_RUNLIST_AND_ENG_IDLE_ENABLE            NVBIT32(0)
#define NV_SWRUNLIST_QOS_INTR_RUNLIST_IDLE_ENABLE                    NVBIT32(1)
#define NV_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_ENABLE                 NVBIT32(2)
#define NV_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_AND_ENG_IDLE_ENABLE    NVBIT32(3)

typedef struct
{
    NvU32 size;
    NvU32 caps;
} NV_ME_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
    NvU32 engineInstance;
} NV_BSP_ALLOCATION_PARAMETERS;

#define  NV_VP_ALLOCATION_FLAGS_STANDARD_UCODE                               (0x00000000)
#define  NV_VP_ALLOCATION_FLAGS_STATIC_UCODE                                 (0x00000001)
#define  NV_VP_ALLOCATION_FLAGS_DYNAMIC_UCODE                                (0x00000002)

#define  NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_VIDEO            (0x00000000)
#define  NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_AUDIO            (0x00000001)

typedef struct
{
    NvU32       size;
    NvU32       caps;
    NvU32       flags;
    NvU32       altUcode;
    NvP64       rawUcode            NV_ALIGN_BYTES(8);
    NvU32       rawUcodeSize;
    NvU32       numSubClasses;
    NvU32       numSubSets;
    NvP64       subClasses          NV_ALIGN_BYTES(8);
    NvU32       prohibitMultipleInstances;
    NvP64       pControl            NV_ALIGN_BYTES(8);
    NvHandle    hMemoryCmdBuffer    NV_ALIGN_BYTES(8);
    NvU64       offset              NV_ALIGN_BYTES(8);

} NV_VP_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
} NV_PPP_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
    NvU32 engineInstance;
} NV_MSENC_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
} NV_SEC2_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
    NvU32 engineInstance;
} NV_NVJPG_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32 size;
    NvU32 prohibitMultipleInstances;
    NvU32 engineInstance;
} NV_OFA_ALLOCATION_PARAMETERS;

#define NV04_ADD_VBLANK_CALLBACK                          (0x0000003D)

#include "class/cl9010.h"

typedef struct
{
    NvHandle             hClient;
    NvHandle             hDevice;
    NvHandle             hVblank;
    OSVBLANKCALLBACKPROC pProc;

    NvV32                LogicalHead;
    void                *pParm1;
    void                *pParm2;
    NvU32                bAdd;
    NvV32                status;
} NVOS61_PARAMETERS;

typedef struct
{
    NvU32   index;
    NvV32   flags;
    NvU64   vaSize NV_ALIGN_BYTES(8);
    NvU64   vaStartInternal NV_ALIGN_BYTES(8);
    NvU64   vaLimitInternal NV_ALIGN_BYTES(8);
    NvU32   bigPageSize;
    NvU64   vaBase NV_ALIGN_BYTES(8);
    NvU32   pasid;
} NV_VASPACE_ALLOCATION_PARAMETERS;

#define NV_VASPACE_ALLOCATION_FLAGS_NONE                            (0x00000000)
#define NV_VASPACE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE                BIT(0)
#define NV_VASPACE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS                BIT(1)
#define NV_VASPACE_ALLOCATION_FLAGS_SHARED_MANAGEMENT                     BIT(2)
#define NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED                   BIT(3)
#define NV_VASPACE_ALLOCATION_FLAGS_ENABLE_NVLINK_ATS                     BIT(4)
#define NV_VASPACE_ALLOCATION_FLAGS_IS_MIRRORED                           BIT(5)
#define NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING                  BIT(6)
#define NV_VASPACE_ALLOCATION_FLAGS_VA_INTERNAL_LIMIT                     BIT(7)
#define NV_VASPACE_ALLOCATION_FLAGS_ALLOW_ZERO_ADDRESS                    BIT(8)
#define NV_VASPACE_ALLOCATION_FLAGS_IS_FLA                                BIT(9)
#define NV_VASPACE_ALLOCATION_FLAGS_SKIP_SCRUB_MEMPOOL                    BIT(10)
#define NV_VASPACE_ALLOCATION_FLAGS_OPTIMIZE_PTETABLE_MEMPOOL_USAGE       BIT(11)
#define NV_VASPACE_ALLOCATION_FLAGS_REQUIRE_FIXED_OFFSET                  BIT(12)
#define NV_VASPACE_ALLOCATION_FLAGS_PTETABLE_HEAP_MANAGED                 BIT(13)

#define NV_VASPACE_ALLOCATION_FLAGS_ENABLE_NVLINK_ATS_TEST                BIT(14)

#define NV_VASPACE_ALLOCATION_INDEX_GPU_NEW                                 0x00
#define NV_VASPACE_ALLOCATION_INDEX_GPU_HOST                                0x01
#define NV_VASPACE_ALLOCATION_INDEX_GPU_GLOBAL                              0x02
#define NV_VASPACE_ALLOCATION_INDEX_GPU_DEVICE                              0x03
#define NV_VASPACE_ALLOCATION_INDEX_GPU_FLA                                 0x04
#define NV_VASPACE_ALLOCATION_INDEX_GPU_MAX                                 0x05

#define NV_VASPACE_BIG_PAGE_SIZE_64K                                (64 * 1024)
#define NV_VASPACE_BIG_PAGE_SIZE_128K                               (128 * 1024)

typedef struct
{
    NvHandle hVASpace;
    NvU32    flags;
    NvU32    subctxId;
} NV_CTXSHARE_ALLOCATION_PARAMETERS;

#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT                      1:0
#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC                 (0x00000000)
#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC                (0x00000001)
#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SPECIFIED            (0x00000002)
#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC_PREFER_LOWER   (0x00000003)

#define NV_CTXSHARE_ALLOCATION_SUBCTXID_SUBCTXID                                 30:0
#define NV_CTXSHARE_ALLOCATION_SUBCTXID_ASYNC_PREFER_LOWER_ALLOCATION            31:31
#define NV_CTXSHARE_ALLOCATION_SUBCTXID_ASYNC_PREFER_LOWER_ALLOCATION_SUCCESS    (0x00000001)
#define NV_CTXSHARE_ALLOCATION_SUBCTXID_ASYNC_PREFER_LOWER_ALLOCATION_FAIL       (0x00000000)

typedef struct
{
    NvU32 cmd;
    NvU32 timeoutInMs;
    NvU32 deviceInstance;
} NV_TIMEOUT_CONTROL_PARAMETERS;

#define NV_TIMEOUT_CONTROL_CMD_SET_DEVICE_TIMEOUT                   (0x00000002)
#define NV_TIMEOUT_CONTROL_CMD_RESET_DEVICE_TIMEOUT                 (0x00000003)

typedef enum
{
    NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT = 0,
    NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH,
    NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH,
    NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID
} NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE;

typedef struct
{

    NvBool bDirtyTracking;

    NvU32 granularity;

    NV_DECLARE_ALIGNED(NvU64 accessBitMask[64], 8);

    NvU32 noOfEntries;

    NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE addrSpace;
} NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS;

typedef struct
{

    NvBool bBar1Mapping;

    NvBool bPriv;
} NV_HOPPER_USERMODE_A_PARAMS;

#ifdef __cplusplus
};
#endif
#endif
