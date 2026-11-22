#ifndef _NV_UVM_USER_TYPES_H_
#define _NV_UVM_USER_TYPES_H_

#include "nvtypes.h"

#define UVM_PAGE_SIZE_DEFAULT    0x0ULL
#define UVM_PAGE_SIZE_4K         0x1000ULL
#define UVM_PAGE_SIZE_64K        0x10000ULL
#define UVM_PAGE_SIZE_128K       0x20000ULL
#define UVM_PAGE_SIZE_2M         0x200000ULL
#define UVM_PAGE_SIZE_512M       0x20000000ULL
#define UVM_PAGE_SIZE_256G       0x4000000000ULL

typedef enum
{
    UVM_VIRT_MODE_NONE = 0,
    UVM_VIRT_MODE_LEGACY = 1,
    UVM_VIRT_MODE_SRIOV_HEAVY = 2,
    UVM_VIRT_MODE_SRIOV_STANDARD = 3,
    UVM_VIRT_MODE_COUNT = 4,
} UVM_VIRT_MODE;

typedef enum
{
    UvmGpuMappingTypeDefault = 0,
    UvmGpuMappingTypeReadWriteAtomic = 1,
    UvmGpuMappingTypeReadWrite = 2,
    UvmGpuMappingTypeReadOnly = 3,
    UvmGpuMappingTypeCount = 4
} UvmGpuMappingType;

typedef enum
{
    UvmGpuCachingTypeDefault = 0,
    UvmGpuCachingTypeForceUncached = 1,
    UvmGpuCachingTypeForceCached = 2,
    UvmGpuCachingTypeCount = 3
} UvmGpuCachingType;

typedef enum {
   UvmGpuFormatTypeDefault = 0,
   UvmGpuFormatTypeBlockLinear = 1,
   UvmGpuFormatTypeCount = 2
} UvmGpuFormatType;

typedef enum {
   UvmGpuFormatElementBitsDefault = 0,
   UvmGpuFormatElementBits8 = 1,
   UvmGpuFormatElementBits16 = 2,

   UvmGpuFormatElementBits32 = 4,
   UvmGpuFormatElementBits64 = 5,
   UvmGpuFormatElementBits128 = 6,
   UvmGpuFormatElementBitsCount = 7
} UvmGpuFormatElementBits;

typedef enum {
    UvmGpuCompressionTypeDefault = 0,
    UvmGpuCompressionTypeEnabledNoPlc = 1,
    UvmGpuCompressionTypeCount = 2
} UvmGpuCompressionType;

#define UVM_PMA_MAX_LOCALIZED_REGION_COUNT    2

typedef struct UvmPmaStatistics_tag
{
    volatile NvU64 numPages2m;
    volatile NvU64 numFreePages64k;
    volatile NvU64 numFreePages2m;
    volatile NvU64 numPages2mProtected;
    volatile NvU64 numFreePages64kProtected;
    volatile NvU64 numFreePages2mProtected;
    volatile NvU64 numPages2mLocalizable[UVM_PMA_MAX_LOCALIZED_REGION_COUNT];
    volatile NvU64 numFreePages64kLocalizable[UVM_PMA_MAX_LOCALIZED_REGION_COUNT];
    volatile NvU64 numFreePages2mLocalizable[UVM_PMA_MAX_LOCALIZED_REGION_COUNT];
} UvmPmaStatistics;

typedef enum
{
    UVM_ACCESS_COUNTER_GRANULARITY_64K = 1,
    UVM_ACCESS_COUNTER_GRANULARITY_2M  = 2,
    UVM_ACCESS_COUNTER_GRANULARITY_16M = 3,
    UVM_ACCESS_COUNTER_GRANULARITY_16G = 4,
} UVM_ACCESS_COUNTER_GRANULARITY;

#endif
