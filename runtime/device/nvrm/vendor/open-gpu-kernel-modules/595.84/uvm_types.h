#ifndef _UVM_TYPES_H_
#define _UVM_TYPES_H_

#include "nvlimits.h"
#include "nvtypes.h"
#include "nvstatus.h"
#include "nvCpuUuid.h"
#include "nv_uvm_user_types.h"

#define UVM_STREAM_INVALID  ((UvmStream)0ULL)
#define UVM_STREAM_ALL      ((UvmStream)2ULL)
#define UVM_STREAM_NONE     ((UvmStream)3ULL)

typedef unsigned long long UvmStream;

#define UVM_PARENT_ID_MAX_SUB_PROCESSORS 8

#define UVM_MAX_GPUS_V1       NV_MAX_DEVICES
#define UVM_MAX_PROCESSORS_V1 (UVM_MAX_GPUS_V1 + 1)
#define UVM_MAX_GPUS          (NV_MAX_DEVICES * UVM_PARENT_ID_MAX_SUB_PROCESSORS)
#define UVM_MAX_PROCESSORS    (UVM_MAX_GPUS + 1)

#define UVM_PROCESSOR_MASK_SIZE ((UVM_MAX_PROCESSORS + (sizeof(NvU64) * 8) - 1) / (sizeof(NvU64) * 8))

#define UVM_INIT_FLAGS_DISABLE_HMM                       ((NvU64)0x1)
#define UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE        ((NvU64)0x2)
#define UVM_INIT_FLAGS_MASK                              ((NvU64)0x3)

#define UVM_RANGE_GROUP_ID_NONE        ((NvU64)0)

#define UVM_DISCARD_FLAGS_UNMAP ((NvU64)1)

typedef struct
{

    NvProcessorUuid gpuUuid;
    NvU32           gpuMappingType;
    NvU32           gpuCachingType;
    NvU32           gpuFormatType;
    NvU32           gpuElementBits;
    NvU32           gpuCompressionType;
} UvmGpuMappingAttributes;

typedef struct UvmGlobalState_tag UvmGlobalState;

typedef union
{
    struct {

        int ctrlFd;

        NvHandle hClient;

        NvHandle hSmcPartRef;
    } rm_linux;
} UvmGpuPlatformParams;

typedef union
{
    struct {

        int ctrlFd;

        NvHandle hClient;

        NvHandle hVaSpace;
    } rm_linux;
    struct {

        NvHandle hClient;

        NvHandle hVaSpace;
    } rm_windows;
} UvmGpuVaSpacePlatformParams;

typedef union
{
    struct {

        int ctrlFd;

        NvHandle hClient;

        NvHandle hChannel;
    } rm_linux;
} UvmChannelPlatformParams;

typedef union
{
    struct {

        int ctrlFd;

        NvHandle hClient;

        NvHandle hMemory;
    } rm_linux;
} UvmAllocationPlatformParams;

typedef union
{
    struct {

        int ctrlFd;

        NvHandle hClient;

        NvHandle hMemory;
    } rm_linux;
} UvmDeviceP2PPlatformParams;

typedef NvUPtr UvmDebugSession;

typedef enum
{
    UvmCounterScopeProcessSingleGpu = 0,
    UvmCounterScopeProcessAllGpu = 1,
    UvmCounterScopeGlobalSingleGpu = 2,
    UvmCounterScopeSize
} UvmCounterScope;

typedef enum
{
    UvmCounterNameBytesXferHtD = 0,
    UvmCounterNameBytesXferDtH = 1,
    UvmCounterNameCpuPageFaultCount = 2,
#ifdef __windows__
    UvmCounterNameWddmBytesXferBtH = 3,
    UvmCounterNameWddmBytesXferHtB = 4,

    UvmCounterNameWddmBytesXferDtB = 5,

    UvmCounterNameWddmBytesXferBtD = 6,
#endif

    UvmCounterNamePrefetchBytesXferHtD = 7,

    UvmCounterNamePrefetchBytesXferDtH = 8,

    UvmCounterNameGpuPageFaultCount = 9,
    UVM_TOTAL_COUNTERS
} UvmCounterName;

#define UVM_COUNTER_NAME_FLAG_BYTES_XFER_HTD 0x1
#define UVM_COUNTER_NAME_FLAG_BYTES_XFER_DTH 0x2
#define UVM_COUNTER_NAME_FLAG_CPU_PAGE_FAULT_COUNT 0x4
#define UVM_COUNTER_NAME_FLAG_WDDM_BYTES_XFER_BTH 0x8
#define UVM_COUNTER_NAME_FLAG_WDDM_BYTES_XFER_HTB 0x10
#define UVM_COUNTER_NAME_FLAG_BYTES_XFER_DTB 0x20
#define UVM_COUNTER_NAME_FLAG_BYTES_XFER_BTD 0x40
#define UVM_COUNTER_NAME_FLAG_PREFETCH_BYTES_XFER_HTD 0x80
#define UVM_COUNTER_NAME_FLAG_PREFETCH_BYTES_XFER_DTH 0x100
#define UVM_COUNTER_NAME_FLAG_GPU_PAGE_FAULT_COUNT 0x200

typedef struct
{
    NvU32           scope;
    NvU32           name;
    NvProcessorUuid gpuid;
    NvU32           state;
} UvmCounterConfig;

#define UVM_COUNTER_CONFIG_STATE_DISABLE_REQUESTED  0
#define UVM_COUNTER_CONFIG_STATE_ENABLE_REQUESTED   1

typedef enum
{
    UvmEventMemoryAccessTypeInvalid        = 0,
    UvmEventMemoryAccessTypeRead           = 1,
    UvmEventMemoryAccessTypeWrite          = 2,
    UvmEventMemoryAccessTypeAtomic         = 3,
    UvmEventMemoryAccessTypePrefetch       = 4,

    UvmEventNumMemoryAccessTypes
} UvmEventMemoryAccessType;

typedef enum
{
    UvmEventTypeInvalid                    = 0,

    UvmEventTypeMemoryViolation            = 1,
    UvmEventTypeCpuFault                   = UvmEventTypeMemoryViolation,
    UvmEventTypeMigration                  = 2,
    UvmEventTypeGpuFault                   = 3,
    UvmEventTypeGpuFaultReplay             = 4,
    UvmEventTypeFaultBufferOverflow        = 5,
    UvmEventTypeFatalFault                 = 6,
    UvmEventTypeReadDuplicate              = 7,
    UvmEventTypeReadDuplicateInvalidate    = 8,
    UvmEventTypePageSizeChange             = 9,
    UvmEventTypeThrashingDetected          = 10,
    UvmEventTypeThrottlingStart            = 11,
    UvmEventTypeThrottlingEnd              = 12,
    UvmEventTypeMapRemote                  = 13,
    UvmEventTypeEviction                   = 14,

    UvmEventNumTypes,

    UvmEventTestTypesFirst                 = 62,

    UvmEventTypeTestHmmSplitInvalidate     = UvmEventTestTypesFirst,
    UvmEventTypeTestAccessCounter          = UvmEventTestTypesFirst + 1,

    UvmEventTestTypesLast                  = UvmEventTypeTestAccessCounter,

    UvmEventNumTypesAll
} UvmEventType;

#define UVM_EVENT_ENABLE_MEMORY_VIOLATION             ((NvU64)1 << UvmEventTypeMemoryViolation)
#define UVM_EVENT_ENABLE_CPU_FAULT                    ((NvU64)1 << UvmEventTypeCpuFault)
#define UVM_EVENT_ENABLE_MIGRATION                    ((NvU64)1 << UvmEventTypeMigration)
#define UVM_EVENT_ENABLE_GPU_FAULT                    ((NvU64)1 << UvmEventTypeGpuFault)
#define UVM_EVENT_ENABLE_GPU_FAULT_REPLAY             ((NvU64)1 << UvmEventTypeGpuFaultReplay)
#define UVM_EVENT_ENABLE_FAULT_BUFFER_OVERFLOW        ((NvU64)1 << UvmEventTypeFaultBufferOverflow)
#define UVM_EVENT_ENABLE_FATAL_FAULT                  ((NvU64)1 << UvmEventTypeFatalFault)
#define UVM_EVENT_ENABLE_READ_DUPLICATE               ((NvU64)1 << UvmEventTypeReadDuplicate)
#define UVM_EVENT_ENABLE_READ_DUPLICATE_INVALIDATE    ((NvU64)1 << UvmEventTypeReadDuplicateInvalidate)
#define UVM_EVENT_ENABLE_PAGE_SIZE_CHANGE             ((NvU64)1 << UvmEventTypePageSizeChange)
#define UVM_EVENT_ENABLE_THRASHING_DETECTED           ((NvU64)1 << UvmEventTypeThrashingDetected)
#define UVM_EVENT_ENABLE_THROTTLING_START             ((NvU64)1 << UvmEventTypeThrottlingStart)
#define UVM_EVENT_ENABLE_THROTTLING_END               ((NvU64)1 << UvmEventTypeThrottlingEnd)
#define UVM_EVENT_ENABLE_MAP_REMOTE                   ((NvU64)1 << UvmEventTypeMapRemote)
#define UVM_EVENT_ENABLE_EVICTION                     ((NvU64)1 << UvmEventTypeEviction)
#define UVM_EVENT_ENABLE_TEST_ACCESS_COUNTER          ((NvU64)1 << UvmEventTypeTestAccessCounter)
#define UVM_EVENT_ENABLE_TEST_HMM_SPLIT_INVALIDATE    ((NvU64)1 << UvmEventTypeTestHmmSplitInvalidate)

typedef struct
{

    NvU8 eventType;
    NvU8 accessType;

    NvU16 padding16Bits;
    NvU32 cpuId;
    NvU64 address;
    NvU64 timeStamp;
    NvU32 pid;
    NvU32 threadId;
    NvU64 pc;
} UvmEventCpuFaultInfo;

typedef enum
{
    UvmEventMigrationDirectionInvalid = 0,
    UvmEventMigrationDirectionCpuToGpu = 1,
    UvmEventMigrationDirectionGpuToCpu = 2,

    UvmEventNumMigrationDirections
} UvmEventMigrationDirection;

typedef struct
{

    NvU8 eventType;

    NvU8 direction;

    NvU8 srcIndex;
    NvU8 dstIndex;

    NvU32 padding32Bits;
    NvU64 address;
    NvU64 migratedBytes;
    NvU64 beginTimeStamp;

    NvU64 endTimeStamp;

    NvU64 streamId;
} UvmEventMigrationInfo_Lite;

typedef enum
{

    UvmFaultTypeInvalid                                                        = 0,
    UvmFaultTypeInvalidPde                                                     = 1,
    UvmFaultTypeInvalidPte                                                     = 2,
    UvmFaultTypeWrite                                                          = 3,
    UvmFaultTypeAtomic                                                         = 4,

    UvmFaultTypeFatal                                                          = 5,
    UvmFaultTypeInvalidPdeSize                                                 = UvmFaultTypeFatal,
    UvmFaultTypeLimitViolation                                                 = 6,
    UvmFaultTypeUnboundInstBlock                                               = 7,
    UvmFaultTypePrivViolation                                                  = 8,
    UvmFaultTypePitchMaskViolation                                             = 9,
    UvmFaultTypeWorkCreation                                                   = 10,
    UvmFaultTypeUnsupportedAperture                                            = 11,
    UvmFaultTypeCompressionFailure                                             = 12,
    UvmFaultTypeUnsupportedKind                                                = 13,
    UvmFaultTypeRegionViolation                                                = 14,
    UvmFaultTypePoison                                                         = 15,
    UvmFaultTypeCcViolation                                                    = 16,

    UvmEventNumFaultTypes
} UvmEventFaultType;

typedef enum
{
    UvmEventFatalReasonInvalid            = 0,
    UvmEventFatalReasonInvalidAddress     = 1,
    UvmEventFatalReasonInvalidPermissions = 2,
    UvmEventFatalReasonInvalidFaultType   = 3,
    UvmEventFatalReasonOutOfMemory        = 4,
    UvmEventFatalReasonInternalError      = 5,

    UvmEventFatalReasonInvalidOperation   = 6,

    UvmEventNumFatalReasons
} UvmEventFatalReason;

typedef enum
{
    UvmEventMigrationCauseInvalid        = 0,

    UvmEventMigrationCauseUser           = 1,

    UvmEventMigrationCauseCoherence      = 2,

    UvmEventMigrationCausePrefetch       = 3,

    UvmEventMigrationCauseEviction       = 4,

    UvmEventMigrationCauseAccessCounters = 5,

    UvmEventNumMigrationCauses
} UvmEventMigrationCause;

typedef struct
{

    NvU8 eventType;

    NvU8 migrationCause;

    NvU8 srcIndex;
    NvU8 dstIndex;

    NvU32 padding32Bits;
    NvU64 address;
    NvU64 migratedBytes;
    NvU64 beginTimeStamp;

    NvU64 endTimeStamp;

    NvU64 rangeGroupId;
    NvU64 beginTimeStampGpu;

    NvU64 endTimeStampGpu;

} UvmEventMigrationInfo;

typedef struct
{

    NvU8 eventType;

    NvU8 migrationCause;

    NvU16 padding16Bits;

    NvU16 srcIndex;
    NvU16 dstIndex;
    NvS32 srcNid;
    NvS32 dstNid;
    NvU64 address;
    NvU64 migratedBytes;
    NvU64 beginTimeStamp;

    NvU64 endTimeStamp;

    NvU64 rangeGroupId;
    NvU64 beginTimeStampGpu;

    NvU64 endTimeStampGpu;

} UvmEventMigrationInfo_V2;

typedef enum
{
    UvmEventFaultClientTypeInvalid            = 0,
    UvmEventFaultClientTypeGpc                = 1,
    UvmEventFaultClientTypeHub                = 2,

    UvmEventNumFaultClientTypes
} UvmEventFaultClientType;

typedef struct
{

    NvU8 eventType;
    NvU8 faultType;
    NvU8 accessType;
    NvU8 gpuIndex;
    union
    {
        NvU16 gpcId;

        NvU16 channelId;

    };
    NvU16 clientId;

    NvU64 address;
    NvU64 timeStamp;

    NvU64 timeStampGpu;

    NvU32 batchId;

    NvU8 clientType;

    NvU8 padding8Bits;
    NvU16 padding16Bits;
} UvmEventGpuFaultInfo;

typedef struct
{

    NvU8 eventType;
    NvU8 faultType;
    NvU16 gpuIndex;
    union
    {
        NvU16 gpcId;

        NvU16 channelId;

    };
    NvU16 clientId;

    NvU64 address;
    NvU64 timeStamp;

    NvU64 timeStampGpu;

    NvU32 batchId;

    NvU8 clientType;

    NvU8 accessType;

    NvU16 padding16bits;
} UvmEventGpuFaultInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 gpuIndex;
    NvU8 clientType;

    NvU8 padding8bits;
    NvU32 batchId;

    NvU64 timeStamp;

    NvU64 timeStampGpu;

} UvmEventGpuFaultReplayInfo;

typedef struct
{

    NvU8 eventType;
    NvU8 clientType;
    NvU16 gpuIndex;
    NvU32 batchId;

    NvU64 timeStamp;

    NvU64 timeStampGpu;

} UvmEventGpuFaultReplayInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 faultType;

    NvU8 accessType;
    NvU8 processorIndex;
    NvU8 reason;

    NvU8 padding8bits;
    NvU16 padding16bits;
    NvU64 address;
    NvU64 timeStamp;
} UvmEventFatalFaultInfo;

typedef struct
{

    NvU8 eventType;
    NvU8 faultType;

    NvU8 accessType;
    NvU8 reason;

    NvU16 processorIndex;
    NvU16 padding16bits;
    NvU64 address;
    NvU64 timeStamp;
} UvmEventFatalFaultInfo_V2;

typedef struct
{

    NvU8 eventType;

    NvU8  padding8bits;
    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 processors;

    NvU64 address;

    NvU64 size;

    NvU64 timeStamp;

} UvmEventReadDuplicateInfo;

typedef struct
{

    NvU8 eventType;

    NvU8  padding8bits;
    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 size;

    NvU64 timeStamp;

    NvU64 processors[UVM_PROCESSOR_MASK_SIZE];

} UvmEventReadDuplicateInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 residentIndex;

    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 size;

    NvU64 timeStamp;

} UvmEventReadDuplicateInvalidateInfo;

typedef struct
{

    NvU8 eventType;
    NvU8 padding8bits;
    NvU16 residentIndex;

    NvU32 padding32bits;
    NvU64 address;

    NvU64 size;

    NvU64 timeStamp;

} UvmEventReadDuplicateInvalidateInfo_V2;

typedef struct
{

    NvU8 eventType;

    NvU8 processorIndex;

    NvU16 padding16bits;
    NvU32 size;
    NvU64 address;

    NvU64 timeStamp;

} UvmEventPageSizeChangeInfo;

typedef struct
{

    NvU8 eventType;

    NvU8 padding8bits;
    NvU16 processorIndex;

    NvU32 size;
    NvU64 address;

    NvU64 timeStamp;

} UvmEventPageSizeChangeInfo_V2;

typedef struct
{

    NvU8 eventType;

    NvU8 padding8bits;
    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 processors;

    NvU64 address;

    NvU64 size;
    NvU64 timeStamp;
} UvmEventThrashingDetectedInfo;

typedef struct
{

    NvU8 eventType;

    NvU8 padding8bits;
    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 size;
    NvU64 timeStamp;
    NvU64 processors[UVM_PROCESSOR_MASK_SIZE];

} UvmEventThrashingDetectedInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 processorIndex;

    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 timeStamp;
} UvmEventThrottlingStartInfo;

typedef struct
{

    NvU8 eventType;

    NvU8  padding8bits;
    NvU16 processorIndex;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 timeStamp;
} UvmEventThrottlingStartInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 processorIndex;

    NvU16 padding16bits;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 timeStamp;
} UvmEventThrottlingEndInfo;

typedef struct
{

    NvU8 eventType;

    NvU8  padding8bits;
    NvU16 processorIndex;
    NvU32 padding32bits;
    NvU64 address;

    NvU64 timeStamp;
} UvmEventThrottlingEndInfo_V2;

typedef enum
{
    UvmEventMapRemoteCauseInvalid     = 0,

    UvmEventMapRemoteCauseCoherence   = 1,

    UvmEventMapRemoteCauseThrashing   = 2,

    UvmEventMapRemoteCausePolicy      = 3,

    UvmEventMapRemoteCauseOutOfMemory = 4,

    UvmEventMapRemoteCauseEviction    = 5,
} UvmEventMapRemoteCause;

typedef struct
{

    NvU8 eventType;
    NvU8 srcIndex;
    NvU8 dstIndex;

    NvU8 mapRemoteCause;

    NvU32 padding32bits;
    NvU64 address;

    NvU64 size;
    NvU64 timeStamp;

    NvU64 timeStampGpu;

} UvmEventMapRemoteInfo;

typedef struct
{

    NvU8 eventType;
    NvU8 mapRemoteCause;

    NvU16 padding16bits;
    NvU16 srcIndex;
    NvU16 dstIndex;

    NvU64 address;

    NvU64 size;
    NvU64 timeStamp;

    NvU64 timeStampGpu;

} UvmEventMapRemoteInfo_V2;

typedef struct
{

    NvU8 eventType;
    NvU8 srcIndex;

    NvU8 dstIndex;

    NvU8  padding8bits;
    NvU32 padding32bits;
    NvU64 addressOut;

    NvU64 addressIn;
    NvU64 size;
    NvU64 timeStamp;
} UvmEventEvictionInfo;

typedef struct
{

    NvU8 eventType;

    NvU8  padding8bits;
    NvU16 padding16bits;
    NvU16 srcIndex;

    NvU16 dstIndex;

    NvU64 addressOut;

    NvU64 addressIn;
    NvU64 size;
    NvU64 timeStamp;
} UvmEventEvictionInfo_V2;

typedef enum
{
    UvmEventAperturePeer0   = 1,
    UvmEventAperturePeer1   = 2,
    UvmEventAperturePeer2   = 3,
    UvmEventAperturePeer3   = 4,
    UvmEventAperturePeer4   = 5,
    UvmEventAperturePeer5   = 6,
    UvmEventAperturePeer6   = 7,
    UvmEventAperturePeer7   = 8,
    UvmEventAperturePeerMax = UvmEventAperturePeer7,
    UvmEventApertureSys     = 9,
    UvmEventApertureVid     = 10,
} UvmEventApertureType;

typedef struct
{

    NvU8 eventType;
    NvU8 srcIndex;

    NvU8 aperture;
    NvU8 instancePtrAperture;

    NvU8 padding8bits;

    NvU8 veId;

    NvU16 padding16bits;

    NvU32 value;
    NvU32 subGranularity;
    NvU32 tag;
    NvU32 bank;
    NvU64 address;
    NvU64 instancePtr;
} UvmEventTestAccessCounterInfo;

typedef struct
{

    NvU8 eventType;

    NvU8 aperture;
    NvU8 instancePtrAperture;
    NvU8 veId;

    NvU16 padding16bits;

    NvU16 srcIndex;

    NvU32 value;
    NvU32 subGranularity;
    NvU32 tag;
    NvU32 bank;
    NvU32 padding32bits;
    NvU64 address;
    NvU64 instancePtr;
} UvmEventTestAccessCounterInfo_V2;

typedef struct
{
    NvU8 eventType;
} UvmEventTestSplitInvalidateInfo;

typedef struct
{
    union
    {
        union
        {
            NvU8 eventType;
            UvmEventMigrationInfo_Lite migration_Lite;

            UvmEventCpuFaultInfo cpuFault;
            UvmEventMigrationInfo migration;
            UvmEventGpuFaultInfo gpuFault;
            UvmEventGpuFaultReplayInfo gpuFaultReplay;
            UvmEventFatalFaultInfo fatalFault;
            UvmEventReadDuplicateInfo readDuplicate;
            UvmEventReadDuplicateInvalidateInfo readDuplicateInvalidate;
            UvmEventPageSizeChangeInfo pageSizeChange;
            UvmEventThrashingDetectedInfo thrashing;
            UvmEventThrottlingStartInfo throttlingStart;
            UvmEventThrottlingEndInfo throttlingEnd;
            UvmEventMapRemoteInfo mapRemote;
            UvmEventEvictionInfo eviction;
        } eventData;

        union
        {
            NvU8 eventType;

            UvmEventTestAccessCounterInfo accessCounter;
            UvmEventTestSplitInvalidateInfo splitInvalidate;
        } testEventData;
    };
} UvmEventEntry;

typedef struct
{
    union
    {
        union
        {
            NvU8 eventType;
            UvmEventMigrationInfo_Lite migration_Lite;

            UvmEventCpuFaultInfo cpuFault;
            UvmEventMigrationInfo_V2 migration;
            UvmEventGpuFaultInfo_V2 gpuFault;
            UvmEventGpuFaultReplayInfo_V2 gpuFaultReplay;
            UvmEventFatalFaultInfo_V2 fatalFault;
            UvmEventReadDuplicateInfo_V2 readDuplicate;
            UvmEventReadDuplicateInvalidateInfo_V2 readDuplicateInvalidate;
            UvmEventPageSizeChangeInfo_V2 pageSizeChange;
            UvmEventThrashingDetectedInfo_V2 thrashing;
            UvmEventThrottlingStartInfo_V2 throttlingStart;
            UvmEventThrottlingEndInfo_V2 throttlingEnd;
            UvmEventMapRemoteInfo_V2 mapRemote;
            UvmEventEvictionInfo_V2 eviction;
        } eventData;

        union
        {
            NvU8 eventType;

            UvmEventTestAccessCounterInfo_V2 accessCounter;
            UvmEventTestSplitInvalidateInfo splitInvalidate;
        } testEventData;
    };
} UvmEventEntry_V2;

typedef enum
{
    UvmEventTimeStampTypeInvalid = 0,
    UvmEventTimeStampTypeWin32QPC = 1,
    UvmEventTimeStampTypePosixClockGetTime = 2,
    UvmEventTimeStampTypeAuto = 3,

    UvmEventNumTimeStampTypes
} UvmEventTimeStampType;

typedef NvUPtr UvmEventQueueHandle;

typedef struct UvmEventControlData_tag {

    volatile NvU32 get_ahead;
    volatile NvU32 get_behind;

    volatile NvU32 put_ahead;
    volatile NvU32 put_behind;

    NvU64 dropped[UvmEventNumTypesAll];
} UvmToolsEventControlData;

struct UvmToolsSession_tag;
struct UvmToolsEventQueue_tag;
struct UvmToolsCounters_tag;

typedef struct UvmToolsSession_tag UvmToolsSession;
typedef struct UvmToolsEventQueue_tag UvmToolsEventQueue;
typedef struct UvmToolsCounters_tag UvmToolsCounters;

typedef UvmToolsSession *UvmToolsSessionHandle;
typedef UvmToolsEventQueue *UvmToolsEventQueueHandle;
typedef UvmToolsCounters *UvmToolsCountersHandle;

#endif
