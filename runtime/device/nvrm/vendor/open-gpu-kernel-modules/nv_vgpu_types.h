#pragma once

#include <nvtypes.h>

#include "nvtypes.h"

#define VM_UUID_SIZE            16
#define INVALID_VGPU_DEV_INST   0xFFFFFFFFU
#define MAX_VGPU_DEVICES_PER_VM 16U

typedef enum GUEST_VM_INFO_STATE {
    GUEST_VM_INFO_STATE_UNINITIALIZED = 0,
    GUEST_VM_INFO_STATE_INITIALIZED = 1,
} GUEST_VM_INFO_STATE;

typedef enum VM_ID_TYPE {
    VM_ID_DOMAIN_ID = 0,
    VM_ID_UUID = 1,
} VM_ID_TYPE;

typedef union VM_ID {
    NvU8 vmUuid[VM_UUID_SIZE];
    NV_DECLARE_ALIGNED(NvU64 vmId, 8);
} VM_ID;
