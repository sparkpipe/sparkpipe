#ifndef _UVM_IOCTL_H
#define _UVM_IOCTL_H

#include "uvm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

#define UVM_RESERVE_VA                                                UVM_IOCTL_BASE(1)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_RESERVE_VA_PARAMS;

#define UVM_RELEASE_VA                                                UVM_IOCTL_BASE(2)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_RELEASE_VA_PARAMS;

#define UVM_REGION_COMMIT                                             UVM_IOCTL_BASE(3)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8);
    NvU64           length        NV_ALIGN_BYTES(8);
    UvmStream       streamId      NV_ALIGN_BYTES(8);
    NvProcessorUuid gpuUuid;
    NV_STATUS       rmStatus;
} UVM_REGION_COMMIT_PARAMS;

#define UVM_REGION_DECOMMIT                                           UVM_IOCTL_BASE(4)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_REGION_DECOMMIT_PARAMS;

#define UVM_REGION_SET_STREAM                                         UVM_IOCTL_BASE(5)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8);
    NvU64           length        NV_ALIGN_BYTES(8);
    UvmStream       newStreamId   NV_ALIGN_BYTES(8);
    NvProcessorUuid gpuUuid;
    NV_STATUS       rmStatus;
} UVM_REGION_SET_STREAM_PARAMS;

#define UVM_SET_STREAM_RUNNING                                        UVM_IOCTL_BASE(6)

typedef struct
{
    UvmStream  streamId NV_ALIGN_BYTES(8);
    NV_STATUS  rmStatus;
} UVM_SET_STREAM_RUNNING_PARAMS;

#define UVM_MAX_STREAMS_PER_IOCTL_CALL 32

#define UVM_SET_STREAM_STOPPED                                        UVM_IOCTL_BASE(7)

typedef struct
{
    UvmStream streamIdArray[UVM_MAX_STREAMS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8);
    NvU64     nStreams                                      NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_SET_STREAM_STOPPED_PARAMS;

#define UVM_ADD_SESSION                                               UVM_IOCTL_BASE(10)

typedef struct
{
    NvU32        pidTarget;
    NV_STATUS    rmStatus;
} UVM_ADD_SESSION_PARAMS;

#define UVM_MAX_COUNTERS_PER_IOCTL_CALL 32

#define UVM_ENABLE_COUNTERS                                           UVM_IOCTL_BASE(12)

typedef struct
{
    UvmCounterConfig config[UVM_MAX_COUNTERS_PER_IOCTL_CALL];
    NvU32            count;
    NV_STATUS        rmStatus;
} UVM_ENABLE_COUNTERS_PARAMS;

#define UVM_MAP_COUNTER                                               UVM_IOCTL_BASE(13)

typedef struct
{
    NvU32           scope;
    NvU32           counterName;
    NvProcessorUuid gpuUuid;
    NvP64           addr         NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_MAP_COUNTER_PARAMS;

#define UVM_CREATE_EVENT_QUEUE                                        UVM_IOCTL_BASE(14)

typedef struct
{
    NvU32                 eventQueueIndex;
    NvU64                 queueSize          NV_ALIGN_BYTES(8);
    NvU64                 notificationCount  NV_ALIGN_BYTES(8);
    NvU64                 notificationHandle NV_ALIGN_BYTES(8);
    NvU32                 timeStampType;
    NV_STATUS             rmStatus;
} UVM_CREATE_EVENT_QUEUE_PARAMS;

#define UVM_REMOVE_EVENT_QUEUE                                        UVM_IOCTL_BASE(15)

typedef struct
{
    NvU32         eventQueueIndex;
    NV_STATUS     rmStatus;
} UVM_REMOVE_EVENT_QUEUE_PARAMS;

#define UVM_MAP_EVENT_QUEUE                                           UVM_IOCTL_BASE(16)

typedef struct
{
    NvU32         eventQueueIndex;
    NvP64         userRODataAddr   NV_ALIGN_BYTES(8);
    NvP64         userRWDataAddr   NV_ALIGN_BYTES(8);
    NvP64         readIndexAddr    NV_ALIGN_BYTES(8);
    NvP64         writeIndexAddr   NV_ALIGN_BYTES(8);
    NvP64         queueBufferAddr  NV_ALIGN_BYTES(8);
    NV_STATUS     rmStatus;
} UVM_MAP_EVENT_QUEUE_PARAMS;

#define UVM_EVENT_CTRL                                                UVM_IOCTL_BASE(17)

typedef struct
{
    NvU32        eventQueueIndex;
    NvS32        eventType;
    NvU32        enable;
    NV_STATUS    rmStatus;
} UVM_EVENT_CTRL_PARAMS;

#define UVM_GET_GPU_UUID_TABLE                                        UVM_IOCTL_BASE(20)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1];
    NvU32           validCount;
    NV_STATUS       rmStatus;
} UVM_GET_GPU_UUID_TABLE_PARAMS;

#if defined(WIN32) || defined(WIN64)

#define UVM_REGION_SET_BACKING                                        UVM_IOCTL_BASE(21)

typedef struct
{
    NvProcessorUuid gpuUuid;
    NvU32           hAllocation;
    NvP64           vaAddr       NV_ALIGN_BYTES(8);
    NvU64           regionLength NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_REGION_SET_BACKING_PARAMS;

#define UVM_REGION_UNSET_BACKING                                      UVM_IOCTL_BASE(22)

typedef struct
{
    NvP64     vaAddr       NV_ALIGN_BYTES(8);
    NvU64     regionLength NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_REGION_UNSET_BACKING_PARAMS;

#endif

#define UVM_CREATE_RANGE_GROUP                                        UVM_IOCTL_BASE(23)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_CREATE_RANGE_GROUP_PARAMS;

#define UVM_DESTROY_RANGE_GROUP                                       UVM_IOCTL_BASE(24)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_DESTROY_RANGE_GROUP_PARAMS;

#define UVM_REGISTER_GPU_VASPACE                                      UVM_IOCTL_BASE(25)

typedef struct
{
    NvProcessorUuid gpuUuid;
    NvS32           rmCtrlFd;
    NvHandle        hClient;
    NvHandle        hVaSpace;
    NV_STATUS       rmStatus;
} UVM_REGISTER_GPU_VASPACE_PARAMS;

#define UVM_UNREGISTER_GPU_VASPACE                                    UVM_IOCTL_BASE(26)

typedef struct
{
    NvProcessorUuid gpuUuid;
    NV_STATUS       rmStatus;
} UVM_UNREGISTER_GPU_VASPACE_PARAMS;

#define UVM_REGISTER_CHANNEL                                          UVM_IOCTL_BASE(27)

typedef struct
{
    NvProcessorUuid gpuUuid;
    NvS32           rmCtrlFd;
    NvHandle        hClient;
    NvHandle        hChannel;
    NvU64           base      NV_ALIGN_BYTES(8);
    NvU64           length    NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_REGISTER_CHANNEL_PARAMS;

#define UVM_UNREGISTER_CHANNEL                                        UVM_IOCTL_BASE(28)

typedef struct
{
    NvHandle        hClient;
    NvHandle        hChannel;
    NV_STATUS       rmStatus;
} UVM_UNREGISTER_CHANNEL_PARAMS;

#define UVM_ENABLE_PEER_ACCESS                                       UVM_IOCTL_BASE(29)

typedef struct
{
    NvProcessorUuid gpuUuidA;
    NvProcessorUuid gpuUuidB;
    NV_STATUS  rmStatus;
} UVM_ENABLE_PEER_ACCESS_PARAMS;

#define UVM_DISABLE_PEER_ACCESS                                      UVM_IOCTL_BASE(30)

typedef struct
{
    NvProcessorUuid gpuUuidA;
    NvProcessorUuid gpuUuidB;
    NV_STATUS  rmStatus;
} UVM_DISABLE_PEER_ACCESS_PARAMS;

#define UVM_SET_RANGE_GROUP                                           UVM_IOCTL_BASE(31)

typedef struct
{
    NvU64     rangeGroupId  NV_ALIGN_BYTES(8);
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_SET_RANGE_GROUP_PARAMS;

#define UVM_MAP_EXTERNAL_ALLOCATION                                   UVM_IOCTL_BASE(33)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    NvU64                   offset                          NV_ALIGN_BYTES(8);
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8);
    NvS32                   rmCtrlFd;
    NvU32                   hClient;
    NvU32                   hMemory;

    NV_STATUS               rmStatus;
} UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

#define UVM_FREE                                                      UVM_IOCTL_BASE(34)
typedef struct
{
    NvU64     base      NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_FREE_PARAMS;

#define UVM_MEM_MAP                                                   UVM_IOCTL_BASE(35)

typedef struct
{
    NvP64     regionBase   NV_ALIGN_BYTES(8);
    NvU64     regionLength NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_MEM_MAP_PARAMS;

#define UVM_REGISTER_GPU                                              UVM_IOCTL_BASE(37)

typedef struct
{
    NvProcessorUuid gpu_uuid;
    NvBool          numaEnabled;
    NvS32           numaNodeId;
    NvS32           rmCtrlFd;
    NvHandle        hClient;
    NvHandle        hSmcPartRef;
    NV_STATUS       rmStatus;
} UVM_REGISTER_GPU_PARAMS;

#define UVM_UNREGISTER_GPU                                            UVM_IOCTL_BASE(38)

typedef struct
{
    NvProcessorUuid gpu_uuid;
    NV_STATUS       rmStatus;
} UVM_UNREGISTER_GPU_PARAMS;

#define UVM_PAGEABLE_MEM_ACCESS                                       UVM_IOCTL_BASE(39)

typedef struct
{
    NvBool    pageableMemAccess;
    NV_STATUS rmStatus;
} UVM_PAGEABLE_MEM_ACCESS_PARAMS;

#define UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL 32

#define UVM_PREVENT_MIGRATION_RANGE_GROUPS                            UVM_IOCTL_BASE(40)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8);
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS;

#define UVM_ALLOW_MIGRATION_RANGE_GROUPS                              UVM_IOCTL_BASE(41)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8);
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS;

#define UVM_SET_PREFERRED_LOCATION                                    UVM_IOCTL_BASE(42)

typedef struct
{
    NvU64           requestedBase      NV_ALIGN_BYTES(8);
    NvU64           length             NV_ALIGN_BYTES(8);
    NvProcessorUuid preferredLocation;
    NvS32           preferredCpuNumaNode;
    NV_STATUS       rmStatus;
} UVM_SET_PREFERRED_LOCATION_PARAMS;

#define UVM_UNSET_PREFERRED_LOCATION                                  UVM_IOCTL_BASE(43)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_UNSET_PREFERRED_LOCATION_PARAMS;

#define UVM_ENABLE_READ_DUPLICATION                                   UVM_IOCTL_BASE(44)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_ENABLE_READ_DUPLICATION_PARAMS;

#define UVM_DISABLE_READ_DUPLICATION                                  UVM_IOCTL_BASE(45)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8);
    NvU64     length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_DISABLE_READ_DUPLICATION_PARAMS;

#define UVM_SET_ACCESSED_BY                                           UVM_IOCTL_BASE(46)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8);
    NvU64           length          NV_ALIGN_BYTES(8);
    NvProcessorUuid accessedByUuid;
    NV_STATUS       rmStatus;
} UVM_SET_ACCESSED_BY_PARAMS;

#define UVM_UNSET_ACCESSED_BY                                         UVM_IOCTL_BASE(47)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8);
    NvU64           length          NV_ALIGN_BYTES(8);
    NvProcessorUuid accessedByUuid;
    NV_STATUS       rmStatus;
} UVM_UNSET_ACCESSED_BY_PARAMS;

#define UVM_MIGRATE_FLAG_ASYNC              0x00000001

#define UVM_MIGRATE_FLAG_SKIP_CPU_MAP       0x00000002

#define UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE    0x00000004

#define UVM_MIGRATE_FLAGS_TEST_ALL              (UVM_MIGRATE_FLAG_SKIP_CPU_MAP      | \
                                                 UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)

#define UVM_MIGRATE_FLAGS_ALL                   (UVM_MIGRATE_FLAG_ASYNC | \
                                                 UVM_MIGRATE_FLAGS_TEST_ALL)

#define UVM_MIGRATE                                                   UVM_IOCTL_BASE(51)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8);
    NvU64           length             NV_ALIGN_BYTES(8);
    NvProcessorUuid destinationUuid;
    NvU32           flags;
    NvU64           semaphoreAddress   NV_ALIGN_BYTES(8);
    NvU32           semaphorePayload;
    NvS32           cpuNumaNode;
    NvU64           userSpaceStart     NV_ALIGN_BYTES(8);
    NvU64           userSpaceLength    NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_MIGRATE_PARAMS;

#define UVM_MIGRATE_RANGE_GROUP                                       UVM_IOCTL_BASE(53)
typedef struct
{
    NvU64           rangeGroupId       NV_ALIGN_BYTES(8);
    NvProcessorUuid destinationUuid;
    NV_STATUS       rmStatus;
} UVM_MIGRATE_RANGE_GROUP_PARAMS;

#define UVM_TOOLS_INIT_EVENT_TRACKER                                  UVM_IOCTL_BASE(56)
typedef struct
{
    NvU64           queueBuffer        NV_ALIGN_BYTES(8);
    NvU64           queueBufferSize    NV_ALIGN_BYTES(8);
    NvU64           controlBuffer      NV_ALIGN_BYTES(8);
    NvProcessorUuid processor;
    NvU32           allProcessors;
    NvU32           uvmFd;
    NV_STATUS       rmStatus;
} UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS;

#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD                          UVM_IOCTL_BASE(57)
typedef struct
{
    NvU32     notificationThreshold;
    NV_STATUS rmStatus;
} UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS;

#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS                           UVM_IOCTL_BASE(58)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS;

#define UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS                          UVM_IOCTL_BASE(59)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS;

#define UVM_TOOLS_ENABLE_COUNTERS                                     UVM_IOCTL_BASE(60)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_ENABLE_COUNTERS_PARAMS;

#define UVM_TOOLS_DISABLE_COUNTERS                                    UVM_IOCTL_BASE(61)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_DISABLE_COUNTERS_PARAMS;

#define UVM_TOOLS_READ_PROCESS_MEMORY                                 UVM_IOCTL_BASE(62)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8);
    NvU64     size                      NV_ALIGN_BYTES(8);
    NvU64     targetVa                  NV_ALIGN_BYTES(8);
    NvU64     bytesRead                 NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS;

#define UVM_TOOLS_WRITE_PROCESS_MEMORY                                UVM_IOCTL_BASE(63)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8);
    NvU64     size                      NV_ALIGN_BYTES(8);
    NvU64     targetVa                  NV_ALIGN_BYTES(8);
    NvU64     bytesWritten              NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS;

#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE                            UVM_IOCTL_BASE(64)
typedef struct
{
    NvU64     tablePtr                 NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS;

#define UVM_MAP_DYNAMIC_PARALLELISM_REGION                            UVM_IOCTL_BASE(65)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    NvProcessorUuid         gpuUuid;
    NV_STATUS               rmStatus;
} UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS;

#define UVM_UNMAP_EXTERNAL                                            UVM_IOCTL_BASE(66)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    NvProcessorUuid         gpuUuid;
    NV_STATUS               rmStatus;
} UVM_UNMAP_EXTERNAL_PARAMS;

#define UVM_TOOLS_FLUSH_EVENTS                                        UVM_IOCTL_BASE(67)
typedef struct
{
    NV_STATUS rmStatus;
} UVM_TOOLS_FLUSH_EVENTS_PARAMS;

#define UVM_ALLOC_SEMAPHORE_POOL                                     UVM_IOCTL_BASE(68)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8);
    NV_STATUS               rmStatus;
} UVM_ALLOC_SEMAPHORE_POOL_PARAMS;

#define UVM_CLEAN_UP_ZOMBIE_RESOURCES                                 UVM_IOCTL_BASE(69)
typedef struct
{
    NV_STATUS rmStatus;
} UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS;

#define UVM_PAGEABLE_MEM_ACCESS_ON_GPU                                UVM_IOCTL_BASE(70)

typedef struct
{
    NvProcessorUuid gpu_uuid;
    NvBool          pageableMemAccess;
    NV_STATUS       rmStatus;
} UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS;

#define UVM_POPULATE_PAGEABLE                                         UVM_IOCTL_BASE(71)

#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED              0x00000001

#define UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK            0x00000002

#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL              0x00000004

#define UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL    UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL

#define UVM_POPULATE_PAGEABLE_FLAGS_TEST        (UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED | \
                                                 UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK)

#define UVM_POPULATE_PAGEABLE_FLAGS_ALL         (UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL | \
                                                 UVM_POPULATE_PAGEABLE_FLAGS_TEST)

typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8);
    NvU64           length    NV_ALIGN_BYTES(8);
    NvU32           flags;
    NV_STATUS       rmStatus;
} UVM_POPULATE_PAGEABLE_PARAMS;

#define UVM_VALIDATE_VA_RANGE                                         UVM_IOCTL_BASE(72)
typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8);
    NvU64           length    NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_VALIDATE_VA_RANGE_PARAMS;

#define UVM_CREATE_EXTERNAL_RANGE                                     UVM_IOCTL_BASE(73)
typedef struct
{
    NvU64                  base                             NV_ALIGN_BYTES(8);
    NvU64                  length                           NV_ALIGN_BYTES(8);
    NV_STATUS              rmStatus;
} UVM_CREATE_EXTERNAL_RANGE_PARAMS;

#define UVM_MAP_EXTERNAL_SPARSE                                       UVM_IOCTL_BASE(74)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    NvProcessorUuid         gpuUuid;
    NV_STATUS               rmStatus;
} UVM_MAP_EXTERNAL_SPARSE_PARAMS;

#define UVM_MM_INITIALIZE                                             UVM_IOCTL_BASE(75)
typedef struct
{
    NvS32                   uvmFd;
    NV_STATUS               rmStatus;
} UVM_MM_INITIALIZE_PARAMS;

#define UVM_TOOLS_INIT_EVENT_TRACKER_V2                               UVM_IOCTL_BASE(76)
typedef UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS;

#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2                         UVM_IOCTL_BASE(77)
typedef UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2_PARAMS;

#define UVM_ALLOC_DEVICE_P2P                                          UVM_IOCTL_BASE(78)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8);
    NvU64                   length                          NV_ALIGN_BYTES(8);
    NvU64                   offset                          NV_ALIGN_BYTES(8);
    NvProcessorUuid         gpuUuid;
    NvS32                   rmCtrlFd;
    NvU32                   hClient;
    NvU32                   hMemory;

    NV_STATUS               rmStatus;
} UVM_ALLOC_DEVICE_P2P_PARAMS;

#define UVM_CLEAR_ALL_ACCESS_COUNTERS                                 UVM_IOCTL_BASE(79)

typedef struct
{
    NV_STATUS       rmStatus;
} UVM_CLEAR_ALL_ACCESS_COUNTERS_PARAMS;

#define UVM_DISCARD                                                   UVM_IOCTL_BASE(80)
typedef struct
{
    NvU64           base                                    NV_ALIGN_BYTES(8);
    NvU64           length                                  NV_ALIGN_BYTES(8);
    NvU64           flags                                   NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_DISCARD_PARAMS;

#define UVM_IS_8_SUPPORTED                                            UVM_IOCTL_BASE(2047)

typedef struct
{
    NvU32     is8Supported;
    NV_STATUS rmStatus;
} UVM_IS_8_SUPPORTED_PARAMS;

#ifdef __cplusplus
}
#endif

#endif
