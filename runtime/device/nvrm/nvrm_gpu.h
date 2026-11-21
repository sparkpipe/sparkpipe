#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_rm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_PAGE_BYTES 4096u
#define SPARK_NVRM_HUGE_PAGE_BYTES (2u << 20)
#define SPARK_NVRM_CHANNEL_ENTRIES 1024u
#define SPARK_NVRM_PUSH_BYTES (1u << 20)

typedef enum SparkNvrmMemoryKind
{
	SPARK_NVRM_MEMORY_HOST = 1,
	SPARK_NVRM_MEMORY_SYSTEM = 2,
	SPARK_NVRM_MEMORY_SYSTEM_UNCACHED = 3
} SparkNvrmMemoryKind;

typedef struct SparkNvrmMemory
{
	SparkNvrmMemoryKind kind;
	uint32_t handle;
	uint64_t address;
	uint8_t *cpu;
	uint64_t bytes;
	uint32_t page_bytes;
	uint32_t host_mapped;
	uint32_t dma_mapped;
	uint32_t uvm_range;
	int map_fd;
} SparkNvrmMemory;

typedef struct SparkNvrmChannel
{
	uint32_t handle;
	uint32_t engine;
	uint32_t token;
	uint32_t entries;
	uint64_t put;
	SparkNvrmMemory control;
	SparkNvrmMemory notifier;
	SparkNvrmMemory push;
	uint64_t push_used;
	SparkNvrmMemory fence;
	uint64_t fence_value;
	uint32_t uvm_registered;
} SparkNvrmChannel;

typedef struct SparkNvrmGpu
{
	SparkNvrmRm rm;
	int device_fd;
	int uvm_mm_fd;
	int usermode_fd;
	uint32_t minor;
	uint32_t gpu_id;
	uint32_t instance;
	uint32_t device;
	uint32_t subdevice;
	uint32_t virtual_memory;
	uint32_t vaspace;
	uint32_t channel_group;
	uint32_t context_share;
	uint32_t usermode;
	volatile uint32_t *usermode_page;
	uint8_t uuid[16];
	uint32_t compute_class;
	uint32_t gpfifo_class;
	uint32_t copy_class;
	uint32_t usermode_class;
	uint32_t architecture;
	uint32_t implementation;
	uint32_t gpcs;
	uint32_t tpcs_per_gpc;
	uint32_t sms_per_tpc;
	uint32_t warps_per_sm;
	uint32_t sm_version;
	uint32_t sass_version;
	uint64_t low_next;
	uint64_t low_end;
	uint32_t host_handle_next;
	uint32_t uvm_registered;
	uint32_t vaspace_registered;
	SparkNvrmChannel compute;
	SparkNvrmMemory local_memory;
	uint32_t local_bytes_per_thread;
	uint64_t shared_window;
	uint64_t local_window;
} SparkNvrmGpu;

SparkStatus SparkNvrmGpuOpen(SparkNvrmGpu *gpu, uint32_t ordinal, SparkNvrmMemoryKind control_kind, const char *owner);
SparkStatus SparkNvrmVirtual(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t align, uint64_t *address, const char *owner);
SparkStatus SparkNvrmChannelOpen(SparkNvrmGpu *gpu, SparkNvrmMemoryKind control_kind, const char *owner);
void SparkNvrmChannelClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmChannelSync(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t timeout_ns, const char *owner);
void SparkNvrmGpuClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmMemoryAllocate(SparkNvrmGpu *gpu, uint64_t bytes, SparkNvrmMemoryKind kind, SparkNvrmMemory *memory, const char *owner);
void SparkNvrmMemoryRelease(SparkNvrmGpu *gpu, SparkNvrmMemory *memory);
SparkStatus SparkNvrmPush(SparkNvrmChannel *channel, uint32_t subchannel, uint32_t method, const uint32_t *values, uint32_t count, const char *owner);
SparkStatus SparkNvrmSubmit(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t first_byte, const char *owner);
SparkStatus SparkNvrmSemaphoreRelease(SparkNvrmChannel *channel, uint64_t address, uint64_t value, uint32_t timestamp, const char *owner);
SparkStatus SparkNvrmEnsureLocalMemory(SparkNvrmGpu *gpu, uint32_t bytes_per_thread, const char *owner);
SparkStatus SparkNvrmWaitValue(const volatile uint64_t *semaphore, uint64_t value, uint64_t timeout_ns, const char *operation, const char *owner);

#ifdef __cplusplus
}
#endif
