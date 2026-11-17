#ifndef NV_IOCTL_NUMA_H
#define NV_IOCTL_NUMA_H

#include <nv-ioctl-numbers.h>

#if defined(NV_KERNEL_INTERFACE_LAYER) && defined(NV_LINUX)
#include <linux/types.h>
#elif defined (NV_KERNEL_INTERFACE_LAYER) && defined(NV_BSD)
#include <sys/stdint.h>
#else

#include <stdint.h>

#if !defined(__aligned)
#define __aligned(n) __attribute__((aligned(n)))
#endif

#endif

#define NV_ESC_NUMA_INFO         (NV_IOCTL_BASE + 15)
#define NV_ESC_SET_NUMA_STATUS   (NV_IOCTL_BASE + 16)

#define NV_IOCTL_NUMA_INFO_MAX_OFFLINE_ADDRESSES 64
typedef struct offline_addresses
{
    uint64_t addresses[NV_IOCTL_NUMA_INFO_MAX_OFFLINE_ADDRESSES] __aligned(8);
    uint32_t numEntries;
} nv_offline_addresses_t;

typedef struct nv_ioctl_numa_info
{
    int32_t  nid;
    int32_t  status;
    uint64_t memblock_size __aligned(8);
    uint64_t numa_mem_addr __aligned(8);
    uint64_t numa_mem_size __aligned(8);
    uint8_t  use_auto_online;
    nv_offline_addresses_t offline_addresses __aligned(8);
} nv_ioctl_numa_info_t;

typedef struct nv_ioctl_set_numa_status
{
    int32_t status;
} nv_ioctl_set_numa_status_t;

#define NV_IOCTL_NUMA_STATUS_DISABLED               0
#define NV_IOCTL_NUMA_STATUS_OFFLINE                1
#define NV_IOCTL_NUMA_STATUS_ONLINE_IN_PROGRESS     2
#define NV_IOCTL_NUMA_STATUS_ONLINE                 3
#define NV_IOCTL_NUMA_STATUS_ONLINE_FAILED          4
#define NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS    5
#define NV_IOCTL_NUMA_STATUS_OFFLINE_FAILED         6

#endif
