#ifndef NVTYPES_INCLUDED
#define NVTYPES_INCLUDED

#include "cpuopsys.h"

#define NVTYPES_USE_STDINT 0

#if NVTYPES_USE_STDINT
#ifdef __cplusplus
#include <cstdint>
#include <cinttypes>
#else
#include <stdint.h>
#include <inttypes.h>
#endif
#endif

#ifndef __cplusplus

#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MAKE_NV64TYPES_8BYTES_ALIGNED) && defined(__i386__)

#define OPTIONAL_ALIGN8_ATTR __attribute__((aligned(8)))
#else

#define OPTIONAL_ALIGN8_ATTR
#endif

#ifdef NV_MISRA_COMPLIANCE_REQUIRED

typedef unsigned long long   UInt64;
typedef   signed long long    Int64;
typedef unsigned int         UInt32;
typedef   signed int          Int32;
typedef unsigned short       UInt16;
typedef   signed short        Int16;
typedef unsigned char        UInt8 ;
typedef   signed char         Int8 ;

typedef          void          Void;
typedef          float    float32_t;
typedef          double   float64_t;
#endif

#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef float32_t          NvF32;
typedef float64_t          NvF64 OPTIONAL_ALIGN8_ATTR;
#else
typedef float              NvF32;
typedef double             NvF64 OPTIONAL_ALIGN8_ATTR;
#endif

#if NVTYPES_USE_STDINT
typedef uint8_t            NvV8;
typedef uint8_t            NvU8;
typedef  int8_t            NvS8;
#else
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef UInt8              NvV8;
typedef UInt8              NvU8;
typedef  Int8              NvS8;
#else
typedef unsigned char      NvV8;
typedef unsigned char      NvU8;
typedef   signed char      NvS8;
#endif
#endif

#if NVTYPES_USE_STDINT
typedef uint16_t           NvV16;
typedef uint16_t           NvU16;
typedef  int16_t           NvS16;
#else

#ifdef __INT16_TYPE__
typedef unsigned __INT16_TYPE__ NvV16;
typedef unsigned __INT16_TYPE__ NvU16;
typedef   signed __INT16_TYPE__ NvS16;

#else
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef UInt16             NvV16;
typedef UInt16             NvU16;
typedef  Int16             NvS16;
#else
typedef unsigned short     NvV16;
typedef unsigned short     NvU16;
typedef   signed short     NvS16;
#endif
#endif
#endif

#if defined(NV_HAS_WCHAR_T_TYPEDEF)
 typedef wchar_t NvWchar;
#else
 typedef NvV16   NvWchar;
#endif

#define NvU32_BUILD(a, b, c, d) \
    ((NvU32)( \
     (((NvU32)(a) & 0xff) << 24) | \
     (((NvU32)(b) & 0xff) << 16) | \
     (((NvU32)(c) & 0xff) << 8)  | \
     (((NvU32)(d) & 0xff))))

#define NvU64_BUILD(a, b) \
    ((NvU64)( \
     (((NvU64)(a) & ~0U) << 32) | \
     (((NvU64)(b) & ~0U))))

#if NVTYPES_USE_STDINT
typedef uint32_t           NvV32;
typedef uint32_t           NvU32;
typedef  int32_t           NvS32;
#else

#ifdef __INT32_TYPE__
typedef unsigned __INT32_TYPE__ NvV32;
typedef unsigned __INT32_TYPE__ NvU32;
typedef   signed __INT32_TYPE__ NvS32;

#else

#if (defined(NV_UNIX) || defined(vxworks) || defined(NV_WINDOWS_CE) ||  \
     defined(__arm) || defined(__IAR_SYSTEMS_ICC__) || defined(NV_QNX) || \
     defined(NV_INTEGRITY) || defined(NV_MODS) || \
     defined(__GNUC__) || defined(__clang__) || defined(NV_MACINTOSH_64)) && \
    (!defined(NV_MACINTOSH) || defined(NV_MACINTOSH_64))
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef UInt32             NvV32;
typedef UInt32             NvU32;
#else
typedef unsigned int       NvV32;
typedef unsigned int       NvU32;
#endif

#else
typedef unsigned long      NvV32;
typedef unsigned long      NvU32;
#endif

#if defined(NV_MACINTOSH) && !defined(NV_MACINTOSH_64)
typedef   signed long      NvS32;
#else
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef   Int32            NvS32;
#else
typedef   signed int       NvS32;
#endif
#endif
#endif
#endif

#if NVTYPES_USE_STDINT
typedef uint64_t           NvU64 OPTIONAL_ALIGN8_ATTR;
typedef  int64_t           NvS64 OPTIONAL_ALIGN8_ATTR;

#define NvU64_fmtX PRIX64
#define NvU64_fmtx PRIx64
#define NvU64_fmtu PRIu64
#define NvU64_fmto PRIo64
#define NvS64_fmtd PRId64
#define NvS64_fmti PRIi64
#else

#if defined(__GNUC__) || defined(__clang__) || defined(__arm) || \
    defined(__IAR_SYSTEMS_ICC__) || defined(__ghs__) || defined(_WIN64) || \
    defined(__SUNPRO_C) || defined(__SUNPRO_CC) || defined (__xlC__)
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
typedef UInt64             NvU64 OPTIONAL_ALIGN8_ATTR;
typedef  Int64             NvS64 OPTIONAL_ALIGN8_ATTR;
#else
typedef unsigned long long NvU64 OPTIONAL_ALIGN8_ATTR;
typedef          long long NvS64 OPTIONAL_ALIGN8_ATTR;
#endif

#define NvU64_fmtX "llX"
#define NvU64_fmtx "llx"
#define NvU64_fmtu "llu"
#define NvU64_fmto "llo"
#define NvS64_fmtd "lld"
#define NvS64_fmti "lli"

#else
typedef unsigned __int64   NvU64 OPTIONAL_ALIGN8_ATTR;
typedef          __int64   NvS64 OPTIONAL_ALIGN8_ATTR;

#define NvU64_fmtX "I64X"
#define NvU64_fmtx "I64x"
#define NvU64_fmtu "I64u"
#define NvU64_fmto "I64o"
#define NvS64_fmtd "I64d"
#define NvS64_fmti "I64i"

#endif
#endif

#ifdef NV_TYPESAFE_HANDLES

typedef struct
{
    NvU32 val;
} NvHandle;
#else

typedef NvU32 NvHandle;
#endif

typedef NvU8 NvBool;
#define NV_TRUE           ((NvBool)(0 == 0))
#define NV_FALSE          ((NvBool)(0 != 0))

typedef NvU8 NvTristate;
#define NV_TRISTATE_FALSE           ((NvTristate) 0)
#define NV_TRISTATE_TRUE            ((NvTristate) 1)
#define NV_TRISTATE_INDETERMINATE   ((NvTristate) 2)

#ifdef NV_MISRA_COMPLIANCE_REQUIRED
#define NvU64_HI32(n)     ((NvU32)((((NvU64)(n)) >> 32) & 0xffffffffU))
#define NvU64_LO32(n)     ((NvU32)(( (NvU64)(n))        & 0xffffffffU))
#else
#define NvU64_HI32(n)     ((NvU32)((((NvU64)(n)) >> 32) & 0xffffffff))
#define NvU64_LO32(n)     ((NvU32)(( (NvU64)(n))        & 0xffffffff))
#endif
#define NvU40_HI32(n)     ((NvU32)((((NvU64)(n)) >>  8) & 0xffffffffU))
#define NvU40_HI24of32(n) ((NvU32)(  (NvU64)(n)         & 0xffffff00U))

#define NvU32_HI16(n)     ((NvU16)((((NvU32)(n)) >> 16) & 0xffffU))
#define NvU32_LO16(n)     ((NvU16)(( (NvU32)(n))        & 0xffffU))

#if defined(NV_64_BITS)

typedef void*              NvP64;
typedef NvU64             NvUPtr;
typedef NvS64             NvSPtr;
typedef NvU64           NvLength;

#define NvP64_VALUE(n)        (n)
#define NvP64_fmt "%p"

#define KERNEL_POINTER_FROM_NvP64(p,v) ((p)(v))
#define NvP64_PLUS_OFFSET(p,o) (NvP64)((NvU64)(p) + (NvU64)(o))

#define NvUPtr_fmtX NvU64_fmtX
#define NvUPtr_fmtx NvU64_fmtx
#define NvUPtr_fmtu NvU64_fmtu
#define NvUPtr_fmto NvU64_fmto
#define NvSPtr_fmtd NvS64_fmtd
#define NvSPtr_fmti NvS64_fmti

#else

typedef NvU64              NvP64;
typedef NvU32             NvUPtr;
typedef NvS32             NvSPtr;
typedef NvU32           NvLength;

#define NvP64_VALUE(n)        ((void *)(NvUPtr)(n))
#define NvP64_fmt "0x%llx"

#define KERNEL_POINTER_FROM_NvP64(p,v) ((p)(NvUPtr)(v))
#define NvP64_PLUS_OFFSET(p,o) ((p) + (NvU64)(o))

#define NvUPtr_fmtX "X"
#define NvUPtr_fmtx "x"
#define NvUPtr_fmtu "u"
#define NvUPtr_fmto "o"
#define NvSPtr_fmtd "d"
#define NvSPtr_fmti "i"

#endif

#define NvP64_NULL       (NvP64)0

#define NvU64_ALIGN32_PACK(pDst, pSrc)                                         \
do {                                                                           \
    (pDst)->lo = NvU64_LO32(*(pSrc));                                          \
    (pDst)->hi = NvU64_HI32(*(pSrc));                                          \
} while (NV_FALSE)

#define NvU64_ALIGN32_UNPACK(pDst, pSrc)                                       \
do {                                                                           \
    (*(pDst)) = NvU64_ALIGN32_VAL(pSrc);                                       \
} while (NV_FALSE)

#define NvU64_ALIGN32_VAL(pSrc)                                                \
    ((NvU64) ((NvU64)((pSrc)->lo) | (((NvU64)(pSrc)->hi) << 32U)))

#define NvU64_ALIGN32_IS_ZERO(_pU64)                                          \
    (((_pU64)->lo == 0U) && ((_pU64)->hi == 0U))

#define NvU64_ALIGN32_ADD(pDst, pSrc1, pSrc2)                                 \
do {                                                                          \
    NvU64 __dst, __src1, __scr2;                                              \
                                                                              \
    NvU64_ALIGN32_UNPACK(&__src1, (pSrc1));                                   \
    NvU64_ALIGN32_UNPACK(&__scr2, (pSrc2));                                   \
    __dst = __src1 + __scr2;                                                  \
    NvU64_ALIGN32_PACK((pDst), &__dst);                                       \
} while (NV_FALSE)

#define NvU64_ALIGN32_SUB(pDst, pSrc1, pSrc2)                                  \
do {                                                                           \
    NvU64 __dst, __src1, __scr2;                                               \
                                                                               \
    NvU64_ALIGN32_UNPACK(&__src1, (pSrc1));                                    \
    NvU64_ALIGN32_UNPACK(&__scr2, (pSrc2));                                    \
    __dst = __src1 - __scr2;                                                   \
    NvU64_ALIGN32_PACK((pDst), &__dst);                                        \
} while (NV_FALSE)

typedef struct
{

    NvU32 lo;

    NvU32 hi;
} NvU64_ALIGN32;

#define NV_PTR_TO_NvP64(n) (NvP64)(NvUPtr)(n)
#define NV_SIGN_EXT_PTR_TO_NvP64(p) ((NvP64)(NvS64)(NvSPtr)(p))
#define KERNEL_POINTER_TO_NvP64(p) ((NvP64)(uintptr_t)(p))

#define NV_S8_MIN       (-128)
#define NV_S8_MAX       (+127)
#define NV_U8_MIN       (0U)
#define NV_U8_MAX       (+255U)
#define NV_S16_MIN      (-32768)
#define NV_S16_MAX      (+32767)
#define NV_U16_MIN      (0U)
#define NV_U16_MAX      (+65535U)
#define NV_S32_MIN      (-2147483647 - 1)
#define NV_S32_MAX      (+2147483647)
#define NV_U32_MIN      (0U)
#define NV_U32_MAX      (+4294967295U)
#define NV_S64_MIN      (-9223372036854775807LL - 1LL)
#define NV_S64_MAX      (+9223372036854775807LL)
#define NV_U64_MIN      (0ULL)
#define NV_U64_MAX      (+18446744073709551615ULL)

#if defined(__GNUC__) || defined(__clang__) || defined(NV_QNX)
#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))
#elif defined(__arm)
#define NV_ALIGN_BYTES(size) __align(ALIGN)
#else

#define NV_ALIGN_BYTES(size)
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(NV_QNX)
#define NV_DECLARE_ALIGNED(TYPE_VAR, ALIGN) TYPE_VAR __attribute__ ((aligned (ALIGN)))
#elif defined(__arm)
#define NV_DECLARE_ALIGNED(TYPE_VAR, ALIGN) __align(ALIGN) TYPE_VAR
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    #define NV_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
    #define NV_ATTRIBUTE_UNUSED
#endif

    #if defined(__GNUC__)
        #if (__GNUC__ > 3) || \
            ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) && (__GNUC_PATCHLEVEL__ >= 1))
            #define NV_NOINLINE __attribute__((__noinline__))
        #endif
    #elif defined(__clang__)
        #if __has_attribute(noinline)
        #define NV_NOINLINE __attribute__((__noinline__))
        #endif
    #elif defined(__arm) && (__ARMCC_VERSION >= 300000)
        #define NV_NOINLINE __attribute__((__noinline__))
    #elif (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590)) ||\
            (defined(__SUNPRO_CC) && (__SUNPRO_CC >= 0x590))
        #define NV_NOINLINE __attribute__((__noinline__))
    #elif defined (__INTEL_COMPILER)
        #define NV_NOINLINE __attribute__((__noinline__))
    #endif

    #if !defined(NV_NOINLINE)
    #define NV_NOINLINE
    #endif

    #if defined(__ghs__)
    #define NV_INLINE inline
    #elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    #define NV_INLINE __inline__
    #elif defined (macintosh) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)
    #define NV_INLINE inline
    #elif defined(__arm)
    #define NV_INLINE __inline
    #else
    #define NV_INLINE
    #endif

    #if !defined(DEBUG)

        #if defined(__ghs__)
            #define NV_FORCEINLINE inline
        #elif defined(__GNUC__)

            #if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1))
            #define NV_FORCEINLINE __attribute__((__always_inline__)) __inline__
            #else
            #define NV_FORCEINLINE __inline__
            #endif
        #elif defined(__clang__)
            #if __has_attribute(always_inline)
            #define NV_FORCEINLINE __attribute__((__always_inline__)) __inline__
            #else
            #define NV_FORCEINLINE __inline__
            #endif
        #elif defined(__arm) && (__ARMCC_VERSION >= 220000)

            #define NV_FORCEINLINE __forceinline
        #else
            #define NV_FORCEINLINE NV_INLINE
        #endif
    #else
        #define NV_FORCEINLINE NV_INLINE
    #endif

    #define NV_APIENTRY
    #define NV_FASTCALL
    #define NV_CDECLCALL
    #define NV_STDCALL

    #if defined(__GNUC__) && defined(NV_UNIX)
        #if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
        #define NV_FORCERESULTCHECK __attribute__((__warn_unused_result__))
        #else
        #define NV_FORCERESULTCHECK
        #endif
    #elif defined(__clang__)
        #if __has_attribute(warn_unused_result)
        #define NV_FORCERESULTCHECK __attribute__((__warn_unused_result__))
        #else
        #define NV_FORCERESULTCHECK
        #endif
    #else
        #define NV_FORCERESULTCHECK
    #endif

    #if defined(__GNUC__)
        #define NV_FORMAT_PRINTF(_f, _a) __attribute__((format(printf, _f, _a)))
    #else
        #define NV_FORMAT_PRINTF(_f, _a)
    #endif

#ifdef __cplusplus
}
#endif

#endif
