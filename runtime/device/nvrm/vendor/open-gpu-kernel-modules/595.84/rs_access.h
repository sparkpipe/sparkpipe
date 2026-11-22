#pragma once

#include <nvtypes.h>

#include "nvtypes.h"
#include "nvmisc.h"

#define RS_ACCESS_DUP_OBJECT 0U
#define RS_ACCESS_NICE       1U
#define RS_ACCESS_DEBUG      2U
#define RS_ACCESS_PERFMON    3U
#define RS_ACCESS_COUNT      4U

typedef NvU16 RsAccessRight;

typedef NvU32 RsAccessLimb;
#define SDK_RS_ACCESS_LIMB_BITS 32

#define SDK_RS_ACCESS_MAX_LIMBS 1

#define SDK_RS_ACCESS_MAX_COUNT (0x20)

typedef struct RS_ACCESS_MASK {
    RsAccessLimb limbs[SDK_RS_ACCESS_MAX_LIMBS];
} RS_ACCESS_MASK;

typedef struct RS_ACCESS_INFO {
    NvU32 flags;
} RS_ACCESS_INFO;

#define SDK_RS_ACCESS_LIMB_INDEX(index) ((index) / SDK_RS_ACCESS_LIMB_BITS)
#define SDK_RS_ACCESS_LIMB_POS(index)   ((index) % SDK_RS_ACCESS_LIMB_BITS)

#define SDK_RS_ACCESS_LIMB_ELT(pAccessMask, index) \
    ((pAccessMask)->limbs[SDK_RS_ACCESS_LIMB_INDEX(index)])
#define SDK_RS_ACCESS_OFFSET_MASK(index) \
    NVBIT_TYPE(SDK_RS_ACCESS_LIMB_POS(index), RsAccessLimb)

#define RS_ACCESS_BOUNDS_CHECK(accessRight) \
    (accessRight < RS_ACCESS_COUNT)

#define RS_ACCESS_MASK_TEST(pAccessMask, index) \
    (RS_ACCESS_BOUNDS_CHECK(index) && \
        (SDK_RS_ACCESS_LIMB_ELT(pAccessMask, index) & SDK_RS_ACCESS_OFFSET_MASK(index)) != 0)

#define RS_ACCESS_MASK_ADD(pAccessMask, index) \
    do \
    { \
        if (RS_ACCESS_BOUNDS_CHECK(index)) { \
            SDK_RS_ACCESS_LIMB_ELT(pAccessMask, index) |= SDK_RS_ACCESS_OFFSET_MASK(index); \
        } \
    } while (NV_FALSE)

#define RS_ACCESS_MASK_REMOVE(pAccessMask, index) \
    do \
    { \
        if (RS_ACCESS_BOUNDS_CHECK(index)) { \
            SDK_RS_ACCESS_LIMB_ELT(pAccessMask, index) &= ~SDK_RS_ACCESS_OFFSET_MASK(index); \
        } \
    } while (NV_FALSE)

#define RS_ACCESS_MASK_UNION(pMaskOut, pMaskIn) \
    do \
    { \
        NvLength limb; \
        for (limb = 0; limb < SDK_RS_ACCESS_MAX_LIMBS; limb++) \
        { \
            SDK_RS_ACCESS_LIMB_ELT(pMaskOut, limb) |= SDK_RS_ACCESS_LIMB_ELT(pMaskIn, limb); \
        } \
    } while (NV_FALSE)

#define RS_ACCESS_MASK_SUBTRACT(pMaskOut, pMaskIn) \
    do \
    { \
        NvLength limb; \
        for (limb = 0; limb < SDK_RS_ACCESS_MAX_LIMBS; limb++) \
        { \
            SDK_RS_ACCESS_LIMB_ELT(pMaskOut, limb) &= ~SDK_RS_ACCESS_LIMB_ELT(pMaskIn, limb); \
        } \
    } while (NV_FALSE)

#define RS_ACCESS_MASK_CLEAR(pAccessMask) \
    do \
    { \
        portMemSet(pAccessMask, 0, sizeof(*pAccessMask)); \
    } while (NV_FALSE)

#define RS_ACCESS_MASK_FILL(pAccessMask) \
    do \
    { \
        portMemSet(pAccessMask, 0xff, sizeof(*pAccessMask)); \
    } while (NV_FALSE)

#define RS_SHARE_TYPE_NONE              (0U)
#define RS_SHARE_TYPE_ALL               (1U)
#define RS_SHARE_TYPE_OS_SECURITY_TOKEN (2U)
#define RS_SHARE_TYPE_CLIENT            (3U)
#define RS_SHARE_TYPE_PID               (4U)
#define RS_SHARE_TYPE_SMC_PARTITION     (5U)
#define RS_SHARE_TYPE_GPU               (6U)
#define RS_SHARE_TYPE_FM_CLIENT         (7U)

#define RS_SHARE_TYPE_MAX               (8U)

#define RS_SHARE_ACTION_FLAG_REVOKE      NVBIT(0)
#define RS_SHARE_ACTION_FLAG_REQUIRE     NVBIT(1)
#define RS_SHARE_ACTION_FLAG_COMPOSE     NVBIT(2)

typedef struct RS_SHARE_POLICY {
    NvU32          target;
    RS_ACCESS_MASK accessMask;
    NvU16          type;
    NvU8           action;
} RS_SHARE_POLICY;
