#ifndef CPUOPSYS_H
#define CPUOPSYS_H

#if defined(_WIN32) || defined(_WIN16)
#   define NV_WINDOWS

#   if defined(_WIN32_WINNT)
#      define NV_WINDOWS_NT
#   elif defined(_WIN32_WCE)
#      define NV_WINDOWS_CE
#   else
#      define NV_WINDOWS_9X
#   endif
#endif

#if defined(__linux__) && !defined(NV_LINUX) && !defined(NV_VMWARE)
#   define NV_LINUX
#endif

#if defined(__VMWARE__) && !defined(NV_VMWARE)
#   define NV_VMWARE
#endif

#if defined(__sun__) && defined(__svr4__) && !defined(NV_SUNOS)
#   define NV_SUNOS
#endif

#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#   define NV_SUNPRO_C
#   define NV_SUNOS
#endif

#if defined(__FreeBSD__) && !defined(NV_BSD)
#   define NV_BSD
#endif

#if (defined(__unix__) || defined(__unix) || defined(__INTEGRITY) ) && !defined(nvmacosx) && !defined(vxworks) && !defined(NV_UNIX) && !defined(__QNX__) && !defined(__QNXNTO__)
#   define NV_UNIX
#endif

#if (defined(__QNX__) || defined(__QNXNTO__)) && !defined(NV_QNX)
#   define NV_QNX
#endif

#if (defined(__ANDROID__) || defined(ANDROID)) && !defined(NV_ANDROID)
#   define NV_ANDROID
#endif

#if defined(DceCore) && !defined(NV_DCECORE)
#    define NV_DCECORE
#endif

#if defined(macintosh) || defined(__APPLE__)
#   define NV_MACINTOSH
#   if defined(__MACH__)
#      define NV_MACINTOSH_OSX
#   else
#      define NV_MACINTOSH_OS9
#   endif
#   if defined(__LP64__)
#      define NV_MACINTOSH_64
#   endif
#endif

#if defined(__vxworks) || defined(vxworks)
#   define NV_VXWORKS
#endif

#if defined(__INTEGRITY)
#  if !defined(NV_INTEGRITY)
#    define NV_INTEGRITY
#  endif
#endif

#if ((defined(_M_IX86) || defined(__i386__) || defined(__i386)) && !defined(NVCPU_X86))

#   define NVCPU_X86
#   define NVCPU_MIN_PAGE_SHIFT 12
#endif

#if defined(NV_LINUX) && defined(__ia64__)
#   define NVCPU_IA64_LINUX
#endif
#if defined(NVCPU_IA64_WINDOWS) || defined(NVCPU_IA64_LINUX) || defined(IA64)
#   define NVCPU_IA64
#endif

#if (defined(NV_MACINTOSH) && !(defined(__i386__) || defined(__x86_64__)))  || defined(__PPC__) || defined(__ppc)
#    if defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
#       ifndef NVCPU_PPC64LE
#           define NVCPU_PPC64LE
#       endif
#    else
#       ifndef NVCPU_PPC
#           define NVCPU_PPC
#       endif
#       ifndef NV_BIG_ENDIAN
#           define NV_BIG_ENDIAN
#       endif
#    endif
#    define NVCPU_FAMILY_PPC
#endif

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#    define NVCPU_X86_64
#endif

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
#    define NVCPU_FAMILY_X86
#endif

#if defined(__riscv) && (__riscv_xlen==64)
#    define NVCPU_RISCV64
#    if defined(__nvriscv)
#       define NVCPU_NVRISCV64
#    endif
#endif

#if defined(__arm__) || defined(_M_ARM)

#   define NVCPU_ARM
#   define NVCPU_MIN_PAGE_SHIFT 12
#endif

#if defined(__aarch64__) || defined(__ARM64__) || defined(_M_ARM64)
#   define NVCPU_AARCH64
#   define NVCPU_MIN_PAGE_SHIFT 12
#endif

#if defined(NVCPU_ARM) || defined(NVCPU_AARCH64)
#   define NVCPU_FAMILY_ARM
#endif

#if defined(__SH4__)
#   ifndef NVCPU_SH4
#   define NVCPU_SH4
#   endif
#   if   defined NV_WINDOWS_CE
#       define NVCPU_MIN_PAGE_SHIFT 12
#   endif
#endif

#if defined(__XTENSA__)
# define NVCPU_XTENSA
# if defined(__XTENSA_EB__)
#  define NV_BIG_ENDIAN
# endif
#endif

#if defined(NVCPU_MIN_PAGE_SHIFT)
#define NVCPU_MIN_PAGE_SIZE (1 << NVCPU_MIN_PAGE_SHIFT)
#endif

#if defined(NVCPU_IA64) || defined(NVCPU_X86_64) || \
    defined(NV_MACINTOSH_64) || defined(NVCPU_AARCH64) || \
    defined(NVCPU_PPC64LE) || defined(NVCPU_RISCV64)
#   define NV_64_BITS
#else

#endif

#if defined(ENABLE_VERIF_FEATURES)
#define NV_VERIF_FEATURES
#endif

#define NV_IS_MODS 0

#define NVOS_IS_WINDOWS 0
#if defined(NV_WINDOWS_CE)
#define NVOS_IS_WINDOWS_CE 1
#else
#define NVOS_IS_WINDOWS_CE 0
#endif
#if defined(NV_LINUX)
#define NVOS_IS_LINUX 1
#else
#define NVOS_IS_LINUX 0
#endif
#if defined(NV_UNIX)
#define NVOS_IS_UNIX 1
#else
#define NVOS_IS_UNIX 0
#endif
#if defined(NV_BSD)
#define NVOS_IS_FREEBSD 1
#else
#define NVOS_IS_FREEBSD 0
#endif
#if defined(NV_SUNOS)
#define NVOS_IS_SOLARIS 1
#else
#define NVOS_IS_SOLARIS 0
#endif
#define NVOS_IS_VMWARE 0
#if defined(NV_QNX)
#define NVOS_IS_QNX 1
#else
#define NVOS_IS_QNX 0
#endif
#if defined(NV_ANDROID)
#define NVOS_IS_ANDROID 1
#else
#define NVOS_IS_ANDROID 0
#endif
#if defined(NV_MACINTOSH)
#define NVOS_IS_MACINTOSH 1
#else
#define NVOS_IS_MACINTOSH 0
#endif
#if defined(NV_VXWORKS)
#define NVOS_IS_VXWORKS 1
#else
#define NVOS_IS_VXWORKS 0
#endif
#if defined(NV_LIBOS)
#define NVOS_IS_LIBOS 1
#else
#define NVOS_IS_LIBOS 0
#endif
#if defined(NV_INTEGRITY)
#define NVOS_IS_INTEGRITY 1
#else
#define NVOS_IS_INTEGRITY 0
#endif

#if defined(NVCPU_X86)
#define NVCPU_IS_X86 1
#else
#define NVCPU_IS_X86 0
#endif
#if defined(NVCPU_RISCV64)
#define NVCPU_IS_RISCV64 1
#else
#define NVCPU_IS_RISCV64 0
#endif
#if defined(NVCPU_NVRISCV64)
#define NVCPU_IS_NVRISCV64 1
#else
#define NVCPU_IS_NVRISCV64 0
#endif
#if defined(NVCPU_IA64)
#define NVCPU_IS_IA64 1
#else
#define NVCPU_IS_IA64 0
#endif
#if defined(NVCPU_X86_64)
#define NVCPU_IS_X86_64 1
#else
#define NVCPU_IS_X86_64 0
#endif
#if defined(NVCPU_FAMILY_X86)
#define NVCPU_IS_FAMILY_X86 1
#else
#define NVCPU_IS_FAMILY_X86 0
#endif
#if defined(NVCPU_PPC)
#define NVCPU_IS_PPC 1
#else
#define NVCPU_IS_PPC 0
#endif
#if defined(NVCPU_PPC64LE)
#define NVCPU_IS_PPC64LE 1
#else
#define NVCPU_IS_PPC64LE 0
#endif
#if defined(NVCPU_FAMILY_PPC)
#define NVCPU_IS_FAMILY_PPC 1
#else
#define NVCPU_IS_FAMILY_PPC 0
#endif
#if defined(NVCPU_ARM)
#define NVCPU_IS_ARM 1
#else
#define NVCPU_IS_ARM 0
#endif
#if defined(NVCPU_AARCH64)
#define NVCPU_IS_AARCH64 1
#else
#define NVCPU_IS_AARCH64 0
#endif
#if defined(NVCPU_FAMILY_ARM)
#define NVCPU_IS_FAMILY_ARM 1
#else
#define NVCPU_IS_FAMILY_ARM 0
#endif
#if defined(NVCPU_SH4)
#define NVCPU_IS_SH4 1
#else
#define NVCPU_IS_SH4 0
#endif
#if defined(NVCPU_XTENSA)
#define NVCPU_IS_XTENSA 1
#else
#define NVCPU_IS_XTENSA 0
#endif
#if defined(NV_BIG_ENDIAN)
#define NVCPU_IS_BIG_ENDIAN 1
#else
#define NVCPU_IS_BIG_ENDIAN 0
#endif
#if defined(NV_64_BITS)
#define NVCPU_IS_64_BITS 1
#else
#define NVCPU_IS_64_BITS 0
#endif
#if defined(NVCPU_FAMILY_ARM)
#define NVCPU_IS_PCIE_CACHE_COHERENT 0
#else
#define NVCPU_IS_PCIE_CACHE_COHERENT 1
#endif
#if defined(NV_DCECORE)
#define NVOS_IS_DCECORE 1
#else
#define NVOS_IS_DCECORE 0
#endif

#endif
