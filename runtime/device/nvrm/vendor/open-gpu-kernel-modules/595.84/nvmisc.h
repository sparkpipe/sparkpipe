#ifndef __NV_MISC_H
#define __NV_MISC_H

#if !defined(NVRISCV_LIBFSP_BUILD) || !NVRISCV_LIBFSP_BUILD

#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"

#ifndef NVBIT
#define NVBIT(b)         (1U<<(b))
#endif
#ifndef NVBIT_TYPE
#define NVBIT_TYPE(b, t) (((t)1U)<<(b))
#endif
#ifndef NVBIT32
#define NVBIT32(b)       NVBIT_TYPE(b, NvU32)
#endif
#ifndef NVBIT64
#define NVBIT64(b)       NVBIT_TYPE(b, NvU64)
#endif

#define NV_CONCAT_32_TO_64(hi, lo) ((((NvU64)hi) << 32) | ((NvU64)lo))

#define NV_BITMASK32_ELEMENT_SIZE            (sizeof(NvU32) << 3)
#define NV_BITMASK32_IDX(chId)               (((chId) & ~(0x1F)) >> 5)
#define NV_BITMASK32_OFFSET(chId)            ((chId) & (0x1F))
#define NV_BITMASK32_SET(pChannelMask, chId) \
        (pChannelMask)[NV_BITMASK32_IDX(chId)] |= NVBIT(NV_BITMASK32_OFFSET(chId))
#define NV_BITMASK32_GET(pChannelMask, chId) \
        ((pChannelMask)[NV_BITMASK32_IDX(chId)] & NVBIT(NV_BITMASK32_OFFSET(chId)))

#define BIT_IDX_32(n)                            \
   (((((n) & 0xFFFF0000U) != 0U) ? 0x10U: 0U) |  \
    ((((n) & 0xFF00FF00U) != 0U) ? 0x08U: 0U) |  \
    ((((n) & 0xF0F0F0F0U) != 0U) ? 0x04U: 0U) |  \
    ((((n) & 0xCCCCCCCCU) != 0U) ? 0x02U: 0U) |  \
    ((((n) & 0xAAAAAAAAU) != 0U) ? 0x01U: 0U) )

#define BIT_IDX_64(n)                                       \
   (((((n) & 0xFFFFFFFF00000000ULL) != 0U) ? 0x20U: 0U) |   \
    ((((n) & 0xFFFF0000FFFF0000ULL) != 0U) ? 0x10U: 0U) |   \
    ((((n) & 0xFF00FF00FF00FF00ULL) != 0U) ? 0x08U: 0U) |   \
    ((((n) & 0xF0F0F0F0F0F0F0F0ULL) != 0U) ? 0x04U: 0U) |   \
    ((((n) & 0xCCCCCCCCCCCCCCCCULL) != 0U) ? 0x02U: 0U) |   \
    ((((n) & 0xAAAAAAAAAAAAAAAAULL) != 0U) ? 0x01U: 0U) )

#ifndef _NVMISC_MACROS_H

#define DRF_ISBIT(bitval,drf)                \
        (   \
          (bitval != 0) ? drf )
#define DEVICE_BASE(d)          (0?d)
#define DEVICE_EXTENT(d)        (1?d)
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
#ifdef MISRA_14_3
#define DRF_BASE(drf)           (drf##_LOW_FIELD)
#define DRF_EXTENT(drf)         (drf##_HIGH_FIELD)
#define DRF_SHIFT(drf)          ((drf##_LOW_FIELD) % 32U)
#define DRF_SHIFT_RT(drf)       ((drf##_HIGH_FIELD) % 32U)
#define DRF_SIZE(drf)           ((drf##_HIGH_FIELD)-(drf##_LOW_FIELD)+1U)
#define DRF_MASK(drf)           (0xFFFFFFFFU >> (31U - ((drf##_HIGH_FIELD) % 32U) + ((drf##_LOW_FIELD) % 32U)))
#else
#define DRF_BASE(drf)           (NV_FALSE?drf)
#define DRF_EXTENT(drf)         (NV_TRUE?drf)
#define DRF_SHIFT(drf)          (((NvU32)DRF_BASE(drf)) % 32U)
#define DRF_SHIFT_RT(drf)       (((NvU32)DRF_EXTENT(drf)) % 32U)
#define DRF_SIZE(drf)           (DRF_EXTENT(drf)-DRF_BASE(drf)+1U)
#define DRF_MASK(drf)           (0xFFFFFFFFU>>(31U - DRF_SHIFT_RT(drf) + DRF_SHIFT(drf)))
#endif
#define DRF_DEF(d,r,f,c)        (((NvU32)(NV ## d ## r ## f ## c))<<DRF_SHIFT(NV ## d ## r ## f))
#define DRF_NUM(d,r,f,n)        ((((NvU32)(n))&DRF_MASK(NV ## d ## r ## f))<<DRF_SHIFT(NV ## d ## r ## f))
#else
#define DRF_BASE(drf)           (0?drf)
#define DRF_EXTENT(drf)         (1?drf)
#define DRF_SHIFT(drf)          ((DRF_ISBIT(0,drf)) % 32)
#define DRF_SHIFT_RT(drf)       ((DRF_ISBIT(1,drf)) % 32)
#define DRF_SIZE(drf)           (DRF_EXTENT(drf)-DRF_BASE(drf)+1U)
#define DRF_MASK(drf)           (0xFFFFFFFFU>>(31-((DRF_ISBIT(1,drf)) % 32)+((DRF_ISBIT(0,drf)) % 32)))
#define DRF_DEF(d,r,f,c)        ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV ## d ## r ## f))
#define DRF_NUM(d,r,f,n)        (((n)&DRF_MASK(NV ## d ## r ## f))<<DRF_SHIFT(NV ## d ## r ## f))
#endif
#define DRF_SHIFTMASK(drf)      (DRF_MASK(drf)<<(DRF_SHIFT(drf)))

#define DRF_VAL(d,r,f,v)        (((v)>>DRF_SHIFT(NV ## d ## r ## f))&DRF_MASK(NV ## d ## r ## f))
#endif

#define DRF_VAL_SIGNED(d,r,f,v)         (((DRF_VAL(d,r,f,(v)) ^ (NVBIT(DRF_SIZE(NV ## d ## r ## f)-1U)))) - (NVBIT(DRF_SIZE(NV ## d ## r ## f)-1U)))
#define DRF_IDX_DEF(d,r,f,i,c)          ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_DEF(d,r,f,i,o,c) ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV##d##r##f(i,o)))
#define DRF_IDX_NUM(d,r,f,i,n)          (((n)&DRF_MASK(NV##d##r##f(i)))<<DRF_SHIFT(NV##d##r##f(i)))
#define DRF_IDX_VAL(d,r,f,i,v)          (((v)>>DRF_SHIFT(NV##d##r##f(i)))&DRF_MASK(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_VAL(d,r,f,i,o,v) (((v)>>DRF_SHIFT(NV##d##r##f(i,o)))&DRF_MASK(NV##d##r##f(i,o)))

#define DRF_VAL_FRAC(d,r,x,y,v,z)       ((DRF_VAL(d,r,x,(v))*z) + ((DRF_VAL(d,r,y,v)*z) / (1<<DRF_SIZE(NV##d##r##y))))

#define DRF_SHIFT64(drf)                ((DRF_ISBIT(0,drf)) % 64)
#define DRF_MASK64(drf)                 (NV_U64_MAX>>(63-((DRF_ISBIT(1,drf)) % 64)+((DRF_ISBIT(0,drf)) % 64)))
#define DRF_SHIFTMASK64(drf)            (DRF_MASK64(drf)<<(DRF_SHIFT64(drf)))

#define DRF_DEF64(d,r,f,c)              (((NvU64)(NV ## d ## r ## f ## c))<<DRF_SHIFT64(NV ## d ## r ## f))
#define DRF_NUM64(d,r,f,n)              ((((NvU64)(n))&DRF_MASK64(NV ## d ## r ## f))<<DRF_SHIFT64(NV ## d ## r ## f))
#define DRF_VAL64(d,r,f,v)              ((((NvU64)(v))>>DRF_SHIFT64(NV ## d ## r ## f))&DRF_MASK64(NV ## d ## r ## f))

#define DRF_VAL_SIGNED64(d,r,f,v)       (((DRF_VAL64(d,r,f,(v)) ^ (NVBIT64(DRF_SIZE(NV ## d ## r ## f)-1)))) - (NVBIT64(DRF_SIZE(NV ## d ## r ## f)-1)))
#define DRF_IDX_DEF64(d,r,f,i,c)        (((NvU64)(NV ## d ## r ## f ## c))<<DRF_SHIFT64(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_DEF64(d,r,f,i,o,c) ((NvU64)(NV ## d ## r ## f ## c)<<DRF_SHIFT64(NV##d##r##f(i,o)))
#define DRF_IDX_NUM64(d,r,f,i,n)        ((((NvU64)(n))&DRF_MASK64(NV##d##r##f(i)))<<DRF_SHIFT64(NV##d##r##f(i)))
#define DRF_IDX_VAL64(d,r,f,i,v)        ((((NvU64)(v))>>DRF_SHIFT64(NV##d##r##f(i)))&DRF_MASK64(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_VAL64(d,r,f,i,o,v) (((NvU64)(v)>>DRF_SHIFT64(NV##d##r##f(i,o)))&DRF_MASK64(NV##d##r##f(i,o)))

#define FLD_SET_DRF64(d,r,f,c,v)        (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_DEF64(d,r,f,c))
#define FLD_SET_DRF_NUM64(d,r,f,n,v)    ((((NvU64)(v)) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_NUM64(d,r,f,n))
#define FLD_IDX_SET_DRF64(d,r,f,i,c,v)  (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_DEF64(d,r,f,i,c))
#define FLD_IDX_OFFSET_SET_DRF64(d,r,f,i,o,c,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i,o))) | DRF_IDX_OFFSET_DEF64(d,r,f,i,o,c))
#define FLD_IDX_SET_DRF_DEF64(d,r,f,i,c,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_DEF64(d,r,f,i,c))
#define FLD_IDX_SET_DRF_NUM64(d,r,f,i,n,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_NUM64(d,r,f,i,n))
#define FLD_SET_DRF_IDX64(d,r,f,c,i,v)  (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_DEF64(d,r,f,c(i)))

#define FLD_TEST_DRF64(d,r,f,c,v)       (DRF_VAL64(d, r, f, (v)) == NV##d##r##f##c)
#define FLD_TEST_DRF_AND64(d,r,f,c,v)   (DRF_VAL64(d, r, f, (v)) & NV##d##r##f##c)
#define FLD_TEST_DRF_NUM64(d,r,f,n,v)   (DRF_VAL64(d, r, f, (v)) == (n))
#define FLD_IDX_TEST_DRF64(d,r,f,i,c,v) (DRF_IDX_VAL64(d, r, f, i, (v)) == NV##d##r##f##c)
#define FLD_IDX_OFFSET_TEST_DRF64(d,r,f,i,o,c,v) (DRF_IDX_OFFSET_VAL64(d, r, f, i, o, (v)) == NV##d##r##f##c)

#define REF_DEF64(drf,d)            (((drf ## d)&DRF_MASK64(drf))<<DRF_SHIFT64(drf))
#define REF_VAL64(drf,v)            (((NvU64)(v)>>DRF_SHIFT64(drf))&DRF_MASK64(drf))
#if defined(NV_MISRA_COMPLIANCE_REQUIRED) && defined(MISRA_14_3)
#define REF_NUM64(drf,n)            (((NvU64)(n)&(0xFFFFFFFFFFFFFFFFU>>(63U-((drf##_HIGH_FIELD) % 63U)+((drf##_LOW_FIELD) % 63U)))) << ((drf##_LOW_FIELD) % 63U))
#else
#define REF_NUM64(drf,n)            (((NvU64)(n)&DRF_MASK64(drf))<<DRF_SHIFT64(drf))
#endif
#define FLD_TEST_REF64(drf,c,v)     (REF_VAL64(drf, v) == drf##c)
#define FLD_TEST_REF_AND64(drf,c,v) (REF_VAL64(drf, v) & drf##c)
#define FLD_SET_REF_NUM64(drf,n,v)  (((NvU64)(v) & ~DRF_SHIFTMASK64(drf)) | REF_NUM64(drf,n))

#ifdef NV_MISRA_COMPLIANCE_REQUIRED
#define FLD_SET_DRF(d,r,f,c,v)                  (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c))
#define FLD_SET_DRF_NUM(d,r,f,n,v)              (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_NUM(d,r,f,n))
#define FLD_IDX_SET_DRF(d,r,f,i,c,v)            (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_OFFSET_SET_DRF(d,r,f,i,o,c,v)   (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f(i,o))) | DRF_IDX_OFFSET_DEF(d,r,f,i,o,c))
#define FLD_IDX_SET_DRF_DEF(d,r,f,i,c,v)        (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_SET_DRF_NUM(d,r,f,i,n,v)        (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_NUM(d,r,f,i,n))
#define FLD_SET_DRF_IDX(d,r,f,c,i,v)            (((NvU32)(v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c(i)))

#define FLD_TEST_DRF(d,r,f,c,v)                 ((DRF_VAL(d, r, f, (v)) == (NvU32)(NV##d##r##f##c)))
#define FLD_TEST_DRF_AND(d,r,f,c,v)             ((DRF_VAL(d, r, f, (v)) & (NvU32)(NV##d##r##f##c)) != 0U)
#define FLD_TEST_DRF_NUM(d,r,f,n,v)             ((DRF_VAL(d, r, f, (v)) == (NvU32)(n)))
#define FLD_IDX_TEST_DRF(d,r,f,i,c,v)           ((DRF_IDX_VAL(d, r, f, i, (v)) == (NvU32)(NV##d##r##f##c)))
#define FLD_IDX_OFFSET_TEST_DRF(d,r,f,i,o,c,v)  ((DRF_IDX_OFFSET_VAL(d, r, f, i, o, (v)) == (NvU32)(NV##d##r##f##c)))
#else
#define FLD_SET_DRF(d,r,f,c,v)                  (((v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c))
#define FLD_SET_DRF_NUM(d,r,f,n,v)              (((v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_NUM(d,r,f,n))
#define FLD_IDX_SET_DRF(d,r,f,i,c,v)            (((v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_OFFSET_SET_DRF(d,r,f,i,o,c,v)   (((v) & ~DRF_SHIFTMASK(NV##d##r##f(i,o))) | DRF_IDX_OFFSET_DEF(d,r,f,i,o,c))
#define FLD_IDX_SET_DRF_DEF(d,r,f,i,c,v)        (((v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_SET_DRF_NUM(d,r,f,i,n,v)        (((v) & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_NUM(d,r,f,i,n))
#define FLD_SET_DRF_IDX(d,r,f,c,i,v)            (((v) & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c(i)))

#define FLD_TEST_DRF(d,r,f,c,v)                 ((DRF_VAL(d, r, f, (v)) == NV##d##r##f##c))
#define FLD_TEST_DRF_AND(d,r,f,c,v)             ((DRF_VAL(d, r, f, (v)) & NV##d##r##f##c))
#define FLD_TEST_DRF_NUM(d,r,f,n,v)             ((DRF_VAL(d, r, f, (v)) == (n)))
#define FLD_IDX_TEST_DRF(d,r,f,i,c,v)           ((DRF_IDX_VAL(d, r, f, i, (v)) == NV##d##r##f##c))
#define FLD_IDX_OFFSET_TEST_DRF(d,r,f,i,o,c,v)  ((DRF_IDX_OFFSET_VAL(d, r, f, i, o, (v)) == NV##d##r##f##c))
#endif

#define REF_DEF(drf,d)            (((drf ## d)&DRF_MASK(drf))<<DRF_SHIFT(drf))
#define REF_VAL(drf,v)            (((v)>>DRF_SHIFT(drf))&DRF_MASK(drf))
#if defined(NV_MISRA_COMPLIANCE_REQUIRED) && defined(MISRA_14_3)
#define REF_NUM(drf,n)            (((n)&(0xFFFFFFFFU>>(31U-((drf##_HIGH_FIELD) % 32U)+((drf##_LOW_FIELD) % 32U)))) << ((drf##_LOW_FIELD) % 32U))
#else
#define REF_NUM(drf,n)            (((n)&DRF_MASK(drf))<<DRF_SHIFT(drf))
#endif
#define FLD_TEST_REF(drf,c,v)     (REF_VAL(drf, (v)) == drf##c)
#define FLD_TEST_REF_AND(drf,c,v) (REF_VAL(drf, (v)) &  drf##c)
#define FLD_SET_REF_NUM(drf,n,v)  (((v) & ~DRF_SHIFTMASK(drf)) | REF_NUM(drf,n))

#define CR_DRF_DEF(d,r,f,c)     ((CR ## d ## r ## f ## c)<<DRF_SHIFT(CR ## d ## r ## f))
#define CR_DRF_NUM(d,r,f,n)     (((n)&DRF_MASK(CR ## d ## r ## f))<<DRF_SHIFT(CR ## d ## r ## f))
#define CR_DRF_VAL(d,r,f,v)     (((v)>>DRF_SHIFT(CR ## d ## r ## f))&DRF_MASK(CR ## d ## r ## f))

#define DRF_EXPAND_MW(drf)         drf
#define DRF_PICK_MW(drf,v)         ((v)? DRF_EXPAND_##drf)
#define DRF_WORD_MW(drf)           (DRF_PICK_MW(drf,0)/32)
#define DRF_BASE_MW(drf)           (DRF_PICK_MW(drf,0)%32)
#define DRF_EXTENT_MW(drf)         (DRF_PICK_MW(drf,1)%32)
#define DRF_SHIFT_MW(drf)          (DRF_PICK_MW(drf,0)%32)
#define DRF_MASK_MW(drf)           (0xFFFFFFFFU>>((31-(DRF_EXTENT_MW(drf))+(DRF_BASE_MW(drf)))%32))
#define DRF_SHIFTMASK_MW(drf)      ((DRF_MASK_MW(drf))<<(DRF_SHIFT_MW(drf)))
#define DRF_SIZE_MW(drf)           (DRF_EXTENT_MW(drf)-DRF_BASE_MW(drf)+1)

#define DRF_DEF_MW(d,r,f,c)        ((NV##d##r##f##c) << DRF_SHIFT_MW(NV##d##r##f))
#define DRF_NUM_MW(d,r,f,n)        (((n)&DRF_MASK_MW(NV##d##r##f))<<DRF_SHIFT_MW(NV##d##r##f))

#define DRF_VAL_MW_1WORD(d,r,f,v)       ((((v)[DRF_WORD_MW(NV##d##r##f)])>>DRF_SHIFT_MW(NV##d##r##f))&DRF_MASK_MW(NV##d##r##f))
#define DRF_SPANS(drf)                  ((DRF_PICK_MW(drf,0)/32) != (DRF_PICK_MW(drf,1)/32))
#define DRF_WORD_MW_LOW(drf)            (DRF_PICK_MW(drf,0)/32)
#define DRF_WORD_MW_HIGH(drf)           (DRF_PICK_MW(drf,1)/32)
#define DRF_MASK_MW_LOW(drf)            (0xFFFFFFFFU)
#define DRF_MASK_MW_HIGH(drf)           (0xFFFFFFFFU>>(31-(DRF_EXTENT_MW(drf))))
#define DRF_SHIFT_MW_LOW(drf)           (DRF_PICK_MW(drf,0)%32)
#define DRF_SHIFT_MW_HIGH(drf)          (0)
#define DRF_MERGE_SHIFT(drf)            ((32-((DRF_PICK_MW(drf,0)%32)))%32)
#define DRF_VAL_MW_2WORD(d,r,f,v)       (((((v)[DRF_WORD_MW_LOW(NV##d##r##f)])>>DRF_SHIFT_MW_LOW(NV##d##r##f))&DRF_MASK_MW_LOW(NV##d##r##f)) | \
    (((((v)[DRF_WORD_MW_HIGH(NV##d##r##f)])>>DRF_SHIFT_MW_HIGH(NV##d##r##f))&DRF_MASK_MW_HIGH(NV##d##r##f)) << DRF_MERGE_SHIFT(NV##d##r##f)))
#define DRF_VAL_MW(d,r,f,v)             ( DRF_SPANS(NV##d##r##f) ? DRF_VAL_MW_2WORD(d,r,f,v) : DRF_VAL_MW_1WORD(d,r,f,v) )

#define DRF_IDX_DEF_MW(d,r,f,i,c)  ((NV##d##r##f##c)<<DRF_SHIFT_MW(NV##d##r##f(i)))
#define DRF_IDX_NUM_MW(d,r,f,i,n)  (((n)&DRF_MASK_MW(NV##d##r##f(i)))<<DRF_SHIFT_MW(NV##d##r##f(i)))
#define DRF_IDX_VAL_MW(d,r,f,i,v)  ((((v)[DRF_WORD_MW(NV##d##r##f(i))])>>DRF_SHIFT_MW(NV##d##r##f(i)))&DRF_MASK_MW(NV##d##r##f(i)))

#define FLD_IDX_OR_DRF_DEF(d,r,f,c,s,v)                 \
do                                                      \
{   NvU32 idx;                                          \
    for (idx = 0; idx < (NV ## d ## r ## f ## s); ++idx)\
    {                                                   \
        v |= DRF_IDX_DEF(d,r,f,idx,c);                  \
    }                                                   \
} while(0)

#define FLD_MERGE_MW(drf,n,v)               (((v)[DRF_WORD_MW(drf)] & ~DRF_SHIFTMASK_MW(drf)) | n)
#define FLD_ASSIGN_MW(drf,n,v)              ((v)[DRF_WORD_MW(drf)] = FLD_MERGE_MW(drf, n, v))
#define FLD_IDX_MERGE_MW(drf,i,n,v)         (((v)[DRF_WORD_MW(drf(i))] & ~DRF_SHIFTMASK_MW(drf(i))) | n)
#define FLD_IDX_ASSIGN_MW(drf,i,n,v)        ((v)[DRF_WORD_MW(drf(i))] = FLD_MERGE_MW(drf(i), n, v))

#define FLD_SET_DRF_MW(d,r,f,c,v)              FLD_MERGE_MW(NV##d##r##f, DRF_DEF_MW(d,r,f,c), v)
#define FLD_SET_DRF_NUM_MW(d,r,f,n,v)          FLD_ASSIGN_MW(NV##d##r##f, DRF_NUM_MW(d,r,f,n), v)
#define FLD_SET_DRF_DEF_MW(d,r,f,c,v)          FLD_ASSIGN_MW(NV##d##r##f, DRF_DEF_MW(d,r,f,c), v)
#define FLD_IDX_SET_DRF_MW(d,r,f,i,c,v)        FLD_IDX_MERGE_MW(NV##d##r##f, i, DRF_IDX_DEF_MW(d,r,f,i,c), v)
#define FLD_IDX_SET_DRF_DEF_MW(d,r,f,i,c,v)    FLD_IDX_MERGE_MW(NV##d##r##f, i, DRF_IDX_DEF_MW(d,r,f,i,c), v)
#define FLD_IDX_SET_DRF_NUM_MW(d,r,f,i,n,v)    FLD_IDX_ASSIGN_MW(NV##d##r##f, i, DRF_IDX_NUM_MW(d,r,f,i,n), v)

#define FLD_TEST_DRF_MW(d,r,f,c,v)          ((DRF_VAL_MW(d, r, f, (v)) == NV##d##r##f##c))
#define FLD_TEST_DRF_NUM_MW(d,r,f,n,v)      ((DRF_VAL_MW(d, r, f, (v)) == n))
#define FLD_IDX_TEST_DRF_MW(d,r,f,i,c,v)    ((DRF_IDX_VAL_MW(d, r, f, i, (v)) == NV##d##r##f##c))

#define DRF_VAL_BS(d,r,f,v)                 ( DRF_SPANS(NV##d##r##f) ? DRF_VAL_BS_2WORD(d,r,f,(v)) : DRF_VAL_BS_1WORD(d,r,f,(v)) )

#define ENG_RD_REG(g,o,d,r)             GPU_REG_RD32(g, ENG_REG##d(o,d,r))
#define ENG_WR_REG(g,o,d,r,v)           GPU_REG_WR32(g, ENG_REG##d(o,d,r), (v))
#define ENG_RD_DRF(g,o,d,r,f)           ((GPU_REG_RD32(g, ENG_REG##d(o,d,r))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))
#define ENG_WR_DRF_DEF(g,o,d,r,f,c)     GPU_REG_WR32(g, ENG_REG##d(o,d,r),(GPU_REG_RD32(g,ENG_REG##d(o,d,r))&~(GPU_DRF_MASK(NV##d##r##f)<<GPU_DRF_SHIFT(NV##d##r##f)))|GPU_DRF_DEF(d,r,f,c))
#define ENG_WR_DRF_NUM(g,o,d,r,f,n)     GPU_REG_WR32(g, ENG_REG##d(o,d,r),(GPU_REG_RD32(g,ENG_REG##d(o,d,r))&~(GPU_DRF_MASK(NV##d##r##f)<<GPU_DRF_SHIFT(NV##d##r##f)))|GPU_DRF_NUM(d,r,f,n))
#define ENG_TEST_DRF_DEF(g,o,d,r,f,c)   (ENG_RD_DRF(g, o, d, r, f) == NV##d##r##f##c)

#define ENG_RD_IDX_DRF(g,o,d,r,f,i)     ((GPU_REG_RD32(g, ENG_REG##d(o,d,r(i)))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))
#define ENG_TEST_IDX_DRF_DEF(g,o,d,r,f,c,i) (ENG_RD_IDX_DRF(g, o, d, r, f, (i)) == NV##d##r##f##c)

#define ENG_IDX_RD_REG(g,o,d,i,r)       GPU_REG_RD32(g, ENG_IDX_REG##d(o,d,i,r))
#define ENG_IDX_WR_REG(g,o,d,i,r,v)     GPU_REG_WR32(g, ENG_IDX_REG##d(o,d,i,r), (v))

#define ENG_IDX_RD_DRF(g,o,d,i,r,f)     ((GPU_REG_RD32(g, ENG_IDX_REG##d(o,d,i,r))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))

#define DRF_VAL_BS_1WORD(d,r,f,v)           ((DRF_READ_1WORD_BS(d,r,f,v)>>DRF_SHIFT_MW(NV##d##r##f))&DRF_MASK_MW(NV##d##r##f))
#define DRF_VAL_BS_2WORD(d,r,f,v)           (((DRF_READ_4BYTE_BS(NV##d##r##f,v)>>DRF_SHIFT_MW_LOW(NV##d##r##f))&DRF_MASK_MW_LOW(NV##d##r##f)) | \
    (((DRF_READ_1WORD_BS_HIGH(d,r,f,v)>>DRF_SHIFT_MW_HIGH(NV##d##r##f))&DRF_MASK_MW_HIGH(NV##d##r##f)) << DRF_MERGE_SHIFT(NV##d##r##f)))

#define DRF_READ_1BYTE_BS(drf,v)            ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4]))
#define DRF_READ_2BYTE_BS(drf,v)            (DRF_READ_1BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+1])<<8))
#define DRF_READ_3BYTE_BS(drf,v)            (DRF_READ_2BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+2])<<16))
#define DRF_READ_4BYTE_BS(drf,v)            (DRF_READ_3BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+3])<<24))

#define DRF_READ_1BYTE_BS_HIGH(drf,v)       ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4]))
#define DRF_READ_2BYTE_BS_HIGH(drf,v)       (DRF_READ_1BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+1])<<8))
#define DRF_READ_3BYTE_BS_HIGH(drf,v)       (DRF_READ_2BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+2])<<16))
#define DRF_READ_4BYTE_BS_HIGH(drf,v)       (DRF_READ_3BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+3])<<24))

#define NV_TWO_N_MINUS_ONE(n) (((1ULL<<(n/2))<<((n+1)/2))-1)

#define NV_BITMASK64(n) ((n<1) ? 0ULL : (NV_U64_MAX>>((n>64) ? 0 : (64-n))))

#define DRF_READ_1WORD_BS(d,r,f,v) \
    ((DRF_EXTENT_MW(NV##d##r##f)<8)?DRF_READ_1BYTE_BS(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<16)?DRF_READ_2BYTE_BS(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<24)?DRF_READ_3BYTE_BS(NV##d##r##f,(v)): \
    DRF_READ_4BYTE_BS(NV##d##r##f,(v)))))

#define DRF_READ_1WORD_BS_HIGH(d,r,f,v) \
    ((DRF_EXTENT_MW(NV##d##r##f)<8)?DRF_READ_1BYTE_BS_HIGH(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<16)?DRF_READ_2BYTE_BS_HIGH(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<24)?DRF_READ_3BYTE_BS_HIGH(NV##d##r##f,(v)): \
    DRF_READ_4BYTE_BS_HIGH(NV##d##r##f,(v)))))

#define LOWESTBIT(x)            ( (x) &  (((x) - 1U) ^ (x)) )

#define HIGHESTBIT(n32)     \
{                           \
    HIGHESTBITIDX_32(n32);  \
    n32 = NVBIT(n32);       \
}
#define ONEBITSET(x)            ( ((x) != 0U) && (((x) & ((x) - 1U)) == 0U) )

#define NUMSETBITS_32(n32)                                         \
{                                                                  \
    n32 = n32 - ((n32 >> 1) & 0x55555555);                         \
    n32 = (n32 & 0x33333333) + ((n32 >> 2) & 0x33333333);          \
    n32 = (((n32 + (n32 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;  \
}

static NV_FORCEINLINE NvU32
nvPopCount32(const NvU32 x)
{
    NvU32 temp = x;
    temp = temp - ((temp >> 1) & 0x55555555U);
    temp = (temp & 0x33333333U) + ((temp >> 2) & 0x33333333U);
    temp = (((temp + (temp >> 4)) & 0x0F0F0F0FU) * 0x01010101U) >> 24;
    return temp;
}

static NV_FORCEINLINE NvU32
nvPopCount64(const NvU64 x)
{
    NvU64 temp = x;
    temp = temp - ((temp >> 1) & 0x5555555555555555ULL);
    temp = (temp & 0x3333333333333333ULL) + ((temp >> 2) & 0x3333333333333333ULL);
    temp = (temp + (temp >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    temp = (temp * 0x0101010101010101ULL) >> 56;
    return (NvU32)temp;
}

static NV_FORCEINLINE NvU32
nvMaskPos32(const NvU32 mask, const NvU32 bitIdx)
{
    return nvPopCount32(mask & (NVBIT32(bitIdx) - 1U));
}

#define LOWESTBITIDX_32(n32)         \
{                                    \
    n32 = BIT_IDX_32(LOWESTBIT(n32));\
}

#define LOWESTBITIDX_64(n64)         \
{                                    \
    n64 = BIT_IDX_64(LOWESTBIT(n64));\
}

#define HIGHESTBITIDX_32(n32)   \
{                               \
    NvU32 count = 0;            \
    while (n32 >>= 1)           \
    {                           \
        count++;                \
    }                           \
    n32 = count;                \
}

#define HIGHESTBITIDX_64(n64)   \
{                               \
    NvU64 count = 0;            \
    while (n64 >>= 1)           \
    {                           \
        count++;                \
    }                           \
    n64 = count;                \
}

#define ROUNDUP_POW2(n32) \
{                         \
    n32--;                \
    n32 |= n32 >> 1;      \
    n32 |= n32 >> 2;      \
    n32 |= n32 >> 4;      \
    n32 |= n32 >> 8;      \
    n32 |= n32 >> 16;     \
    n32++;                \
}

static NV_FORCEINLINE NvU32
nvNextPow2_U32(const NvU32 x)
{
    NvU32 y = x;
    y--;
    y |= y >> 1;
    y |= y >> 2;
    y |= y >> 4;
    y |= y >> 8;
    y |= y >> 16;
    y++;
    return y;
}

static NV_FORCEINLINE NvU32
nvPrevPow2_U32(const NvU32 x )
{
    NvU32 y = x;
    y |= (y >> 1);
    y |= (y >> 2);
    y |= (y >> 4);
    y |= (y >> 8);
    y |= (y >> 16);
    return y - (y >> 1);
}

static NV_FORCEINLINE NvU64
nvPrevPow2_U64(const NvU64 x )
{
    NvU64 y = x;
    y |= (y >> 1);
    y |= (y >> 2);
    y |= (y >> 4);
    y |= (y >> 8);
    y |= (y >> 16);
    y |= (y >> 32);
    return y - (y >> 1);
}

#define ROUNDUP_POW2_U64(n64) \
{                         \
    n64--;                \
    n64 |= n64 >> 1;      \
    n64 |= n64 >> 2;      \
    n64 |= n64 >> 4;      \
    n64 |= n64 >> 8;      \
    n64 |= n64 >> 16;     \
    n64 |= n64 >> 32;     \
    n64++;                \
}

#define NV_SWAP_U8(a,b) \
{                       \
    NvU8 temp;          \
    temp = a;           \
    a = b;              \
    b = temp;           \
}

#define NV_SWAP_U32(a,b)    \
{                           \
    NvU32 temp;             \
    temp = a;               \
    a = b;                  \
    b = temp;               \
}

#define FOR_EACH_INDEX_IN_MASK(maskWidth,index,mask)        \
{                                                           \
    NvU##maskWidth lclMsk = (NvU##maskWidth)(mask);         \
    for ((index) = 0U; lclMsk != 0U; (index)++, lclMsk >>= 1U)\
    {                                                       \
        if (((NvU##maskWidth)NVBIT64(0) & lclMsk) == 0U)    \
        {                                                   \
            continue;                                       \
        }
#define FOR_EACH_INDEX_IN_MASK_END                          \
    }                                                       \
}

static NV_FORCEINLINE NvS32
nvGetNthSetBitIndex32(NvU32 mask, NvU32 n)
{
    NvU32 seenSetBitsCount = 0;
    NvS32 index;
    FOR_EACH_INDEX_IN_MASK(32, index, mask)
    {
        if (seenSetBitsCount == n)
        {
            return index;
        }
        ++seenSetBitsCount;
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return -1;
}

#define NV_ANYSIZE_ARRAY                                                      1

#define NV_CEIL(a,b) (((a)+(b)-1)/(b))

#ifndef NV_DIV_AND_CEIL
#define NV_DIV_AND_CEIL(a, b) NV_CEIL(a,b)
#endif

#ifndef NV_MIN
#define NV_MIN(a, b)        (((a) < (b)) ? (a) : (b))
#endif

#ifndef NV_MAX
#define NV_MAX(a, b)        (((a) > (b)) ? (a) : (b))
#endif

#define NV_ABS(a) ((a)>=0?(a):(-(a)))

#define NV_SIGN(s) ((NvS8)(((s) > 0) - ((s) < 0)))

#define NV_ZERO_SIGN(s) ((NvS8)((((s) >= 0) * 2) - 1))

#ifndef NV_OFFSETOF
    #if defined(__GNUC__) && (__GNUC__ > 3)
        #define NV_OFFSETOF(type, member)   ((NvUPtr) __builtin_offsetof(type, member))
    #else
        #define NV_OFFSETOF(type, member)    ((NvUPtr) &(((type *)0)->member))
    #endif
#endif

#define NV_CONTAINEROF(ptr, type, member) ((type *) (((NvUPtr) ptr) - NV_OFFSETOF(type, member)))

#define NV_UNSIGNED_ROUNDED_DIV(a,b)    (((a) + ((b) / 2U)) / (b))

#define NV_UNSIGNED_DIV_CEIL(a, b)      (((a) + (b - 1)) / (b))

#define NV_SUBTRACT_NO_UNDERFLOW(a, b) ((a)>(b) ? (a)-(b) : 0)

#define NV_RIGHT_SHIFT_ROUNDED(a, shift)                                       \
    (((a) >> (shift)) + !!((NVBIT((shift) - 1) & (a)) == NVBIT((shift) - 1)))

#ifndef NV_ALIGN_DOWN

#define NV_ALIGN_DOWN(v, gran)      ((v) & ~((v) - (v) + (gran) - 1))
#endif

#ifndef NV_ALIGN_UP

#define NV_ALIGN_UP(v, gran)        (((v) + ((gran) - 1)) & ~((v) - (v) + (gran) - 1))
#endif

#ifndef NV_ALIGN_DOWN64
#define NV_ALIGN_DOWN64(v, gran)      ((v) & ~(((NvU64)gran) - 1))
#endif

#ifndef NV_ALIGN_UP64
#define NV_ALIGN_UP64(v, gran)        (((v) + ((gran) - 1)) & ~(((NvU64)gran)-1))
#endif

#ifndef NV_IS_ALIGNED
#define NV_IS_ALIGNED(v, gran)      (0U == ((v) & ((gran) - 1U)))
#endif

#ifndef NV_IS_ALIGNED64
#define NV_IS_ALIGNED64(v, gran)      (0U == ((v) & (((NvU64)gran) - 1U)))
#endif

#ifndef NVMISC_MEMSET
static NV_FORCEINLINE void *NVMISC_MEMSET(void *s, NvU8 c, NvLength n)
{
    NvU8 *b = (NvU8 *) s;
    NvLength i;

    for (i = 0; i < n; i++)
    {
        b[i] = c;
    }

    return s;
}
#endif

#ifndef NVMISC_MEMCPY
static NV_FORCEINLINE void *NVMISC_MEMCPY(void *dest, const void *src, NvLength n)
{
    NvU8 *destByte = (NvU8 *) dest;
    const NvU8 *srcByte = (const NvU8 *) src;
    NvLength i;

    for (i = 0; i < n; i++)
    {
        destByte[i] = srcByte[i];
    }

    return dest;
}
#endif

static NV_FORCEINLINE char *NVMISC_STRNCPY(char *dest, const char *src, NvLength n)
{
    NvLength i;

    for (i = 0; i < n; i++)
    {
        dest[i] = src[i];
        if (src[i] == '\0')
        {
            break;
        }
    }

    for (; i < n; i++)
    {
        dest[i] = '\0';
    }

    return dest;
}

static NV_FORCEINLINE NvUPtr NV_PTR_TO_NVUPTR(void *ptr)
{
    union
    {
        NvUPtr v;
        void *p;
    } uAddr;

    uAddr.p = ptr;
    return uAddr.v;
}

static NV_FORCEINLINE void *NV_NVUPTR_TO_PTR(NvUPtr address)
{
    union
    {
        NvUPtr v;
        void *p;
    } uAddr;

    uAddr.v = address;
    return uAddr.p;
}

#define NV_BIT_GET(k, x)                       (((x) >> (k)) & 1)

#define NV_BIT_GET_128(n, lo, hi)              (((n) < 64) ? NV_BIT_GET((n), (lo)) : NV_BIT_GET((n) - 64, (hi)))

#define NV_BIT_SET_128(b, lo, hi)              { nvAssert( (b) < 128 ); if ( (b) < 64 ) (lo) |= NVBIT64(b); else (hi) |= NVBIT64( b & 0x3F ); }

#define NV_BIT_CLEAR_128(b, lo, hi)            { nvAssert( (b) < 128 ); if ( (b) < 64 ) (lo) &= ~NVBIT64(b); else (hi) &= ~NVBIT64( b & 0x3F ); }

#define NV_ARRAY_ELEMENTS(x)                   ((sizeof(x)/sizeof((x)[0])))

#if !defined(NVIDIA_UNDEF_LEGACY_BIT_MACROS)

#ifndef BIT
#define BIT(b)   (1U<<(b))
#endif
#ifndef BIT32
#define BIT32(b) ((NvU32)1U<<(b))
#endif
#ifndef BIT64
#define BIT64(b) ((NvU64)1U<<(b))
#endif
#endif

#define NV_BITFIELD_SIZE_ADD_ONE(x, bits) \
    ((!!(((NvU64)(x)) >> (bits))) ? 1 : 0)

#define NV_BITFIELD_SIZE_64(x)                         \
    (((x) <= 0) ? -1 :                                 \
                  (NV_BITFIELD_SIZE_ADD_ONE((x), 0) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 1) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 2) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 3) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 4) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 5) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 6) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 7) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 8) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 9) +  \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 10) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 11) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 12) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 13) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 14) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 15) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 16) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 17) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 18) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 19) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 20) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 21) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 22) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 23) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 24) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 25) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 26) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 27) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 28) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 29) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 30) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 31) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 32) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 33) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 34) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 35) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 36) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 37) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 38) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 39) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 40) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 41) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 42) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 43) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 44) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 45) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 46) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 47) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 48) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 49) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 50) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 51) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 52) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 53) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 54) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 55) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 56) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 57) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 58) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 59) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 60) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 61) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 62) + \
                   NV_BITFIELD_SIZE_ADD_ONE((x), 63)))

#ifndef NVDEC_1_0

static NV_FORCEINLINE NvU64 nvFindGcdU64(NvU64 a, NvU64 b)
{
    while (1)
    {
        NvU64 temp;

        if (a == 0)
            return b;
        temp = a;
        a = b % a;
        b = temp;
    }
}
#endif

#ifdef __cplusplus
}
#endif
#else
#include <misc/nvmisc_drf.h>
#include <misc/bitops.h>
#endif
#endif
