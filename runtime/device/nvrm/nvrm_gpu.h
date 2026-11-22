#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_rm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_PAGE_BYTES 4096u
#define SPARK_NVRM_HUGE_PAGE_BYTES (2u << 20)
#define SPARK_NVRM_BIG_PAGE_BYTES (64u << 10)
#define SPARK_NVRM_HUGE_OBJECT_LIMIT (4ull << 30)
#define SPARK_NVRM_MODULE_PARAMETERS "/proc/driver/nvidia/params"
#define SPARK_NVRM_CHANNEL_ENTRIES 1024u
#define SPARK_NVRM_PUSH_BYTES (1u << 20)
#define SPARK_NVRM_LOCAL_RETIRED 16u
#define SPARK_NVRM_WAIT_TIMEOUT_NS 10000000000ull
#define SPARK_NVRM_MARK_BYTES 32u
#define SPARK_NVRM_MARK_TIME_OFFSET 16u

typedef enum SparkNvrmMemoryKind
{
	SPARK_NVRM_MEMORY_HOST = 1,
	SPARK_NVRM_MEMORY_SYSTEM = 2,
	SPARK_NVRM_MEMORY_SYSTEM_UNCACHED = 3,
	SPARK_NVRM_MEMORY_USER = 4,
	SPARK_NVRM_MEMORY_USER_READ_ONLY = 5
} SparkNvrmMemoryKind;

typedef enum SparkNvrmEngine
{
	SPARK_NVRM_ENGINE_NONE = 0,
	SPARK_NVRM_ENGINE_HOST = 1,
	SPARK_NVRM_ENGINE_COMPUTE = 2,
	SPARK_NVRM_ENGINE_COPY = 3
} SparkNvrmEngine;

typedef struct SparkNvrmRange
{
	uint64_t base;
	uint64_t bytes;
} SparkNvrmRange;

typedef struct SparkNvrmSegment
{
	uint64_t start;
	uint64_t end;
	uint64_t value;
} SparkNvrmSegment;

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
	uint32_t virtual_reserved;
	int map_fd;
} SparkNvrmMemory;

typedef struct SparkNvrmChannel
{
	uint32_t handle;
	uint32_t group;
	uint32_t compute_object;
	uint32_t copy_object;
	uint32_t token;
	uint32_t entries;
	uint64_t put;
	SparkNvrmMemory control;
	SparkNvrmMemory notifier;
	SparkNvrmMemory push;
	uint64_t push_used;
	uint64_t push_submitted;
	SparkNvrmMemory fence;
	uint64_t fence_value;
	uint32_t segment_fenced;
	SparkNvrmEngine last_engine;
	uint32_t latch;
	uint32_t latch_run;
	uint64_t after_address;
	uint64_t after_value;
	uint64_t uvm_base;
	uint32_t uvm_registered;
	uint64_t local_address;
	uint32_t segment_head;
	uint32_t segment_count;
	SparkNvrmSegment segments[SPARK_NVRM_CHANNEL_ENTRIES];
} SparkNvrmChannel;

typedef struct SparkNvrmFault
{
	uint32_t code;
	uint32_t engine;
	uint32_t located;
	uint64_t address;
	char kind[32];
} SparkNvrmFault;

typedef struct SparkNvrmGpu
{
	SparkNvrmRm rm;
	int device_fd;
	int uvm_mm_fd;
	int usermode_fd;
	uint32_t minor;
	uint32_t gpu_id;
	uint32_t instance;
	uint32_t integrated;
	uint32_t device;
	uint32_t subdevice;
	uint32_t virtual_memory;
	uint32_t vaspace;
	uint32_t channel_group;
	uint32_t channel_group_scheduled;
	uint32_t context_share;
	uint32_t copy_group;
	uint32_t copy_group_scheduled;
	uint32_t copy_share;
	uint32_t copy_engine;
	SparkNvrmChannel copy;
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
	uint64_t reserved_base;
	uint64_t reserved_bytes;
	SparkNvrmRange *free_ranges;
	uint32_t free_count;
	uint32_t free_capacity;
	uint32_t virtual_live;
	uint32_t host_handle_next;
	uint32_t uvm_registered;
	uint32_t vaspace_registered;
	SparkNvrmChannel compute;
	SparkNvrmMemory local_memory;
	uint64_t local_bytes_per_tpc;
	uint32_t local_bytes_per_thread;
	SparkNvrmMemory local_retired[SPARK_NVRM_LOCAL_RETIRED];
	uint32_t local_retired_count;
	uint64_t shared_window;
	uint64_t local_window;
} SparkNvrmGpu;

uint32_t SparkNvrmGpuIntegrated(uint32_t gpu_flags);
SparkStatus SparkNvrmGpuOpen(SparkNvrmGpu *gpu, uint32_t ordinal, SparkNvrmMemoryKind control_kind, const char *owner);
SparkStatus SparkNvrmAddressOpen(SparkNvrmGpu *gpu, uint64_t base, uint64_t bytes, const char *owner);
void SparkNvrmAddressClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmVirtual(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t align, uint64_t *address, const char *owner);
void SparkNvrmVirtualRelease(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes);
void SparkNvrmReserveCpu(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes);
SparkStatus SparkNvrmChannelOpen(SparkNvrmGpu *gpu, SparkNvrmMemoryKind control_kind, const char *owner);
void SparkNvrmChannelClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmChannelCreate(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmMemoryKind control_kind, const char *owner);
void SparkNvrmChannelDestroy(SparkNvrmGpu *gpu, SparkNvrmChannel *channel);
void SparkNvrmChannelAfter(SparkNvrmChannel *channel, const SparkNvrmChannel *other);
SparkStatus SparkNvrmCopyEngineAsync(SparkNvrmGpu *gpu, uint32_t *engine, const char *owner);
SparkStatus SparkNvrmCopyEngines(SparkNvrmGpu *gpu, uint32_t *engines, uint32_t capacity, uint32_t *count, const char *owner);
SparkStatus SparkNvrmCopyGroupOpen(SparkNvrmGpu *gpu, uint32_t engine, const char *owner);
void SparkNvrmCopyGroupClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmCopyChannelCreate(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner);
SparkStatus SparkNvrmChannelSync(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t timeout_ns, const char *owner);
void SparkNvrmGpuClose(SparkNvrmGpu *gpu);
SparkStatus SparkNvrmMemoryAllocate(SparkNvrmGpu *gpu, uint64_t bytes, SparkNvrmMemoryKind kind, SparkNvrmMemory *memory, const char *owner);
SparkStatus SparkNvrmMemoryAttach(SparkNvrmGpu *gpu, void *host, uint64_t bytes, SparkNvrmMemoryKind kind, SparkNvrmMemory *memory, const char *owner);
void SparkNvrmMemoryRelease(SparkNvrmGpu *gpu, SparkNvrmMemory *memory);
uint32_t SparkNvrmSystemPageBytes(uint64_t bytes);
int SparkNvrmModuleZeroesMemory(const char *parameters_path);
SparkStatus SparkNvrmSystemAllocate(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t page_bytes, uint32_t uncached, uint32_t *handle, const char *owner);
SparkStatus SparkNvrmGpuMap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle, uint64_t offset, uint32_t read_only, const char *owner);
void SparkNvrmGpuUnmap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle);
SparkStatus SparkNvrmShareExport(SparkNvrmGpu *gpu, uint32_t handle, int *fd, const char *owner);
SparkStatus SparkNvrmShareImport(SparkNvrmGpu *gpu, int fd, uint32_t *handle, uint64_t *bytes, const char *owner);
SparkStatus SparkNvrmPush(SparkNvrmChannel *channel, uint32_t subchannel, uint32_t method, const uint32_t *values, uint32_t count, const char *owner);
SparkStatus SparkNvrmSubmit(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner);
SparkStatus SparkNvrmSemaphoreRelease(SparkNvrmChannel *channel, uint64_t address, uint64_t value, uint32_t timestamp, const char *owner);
SparkStatus SparkNvrmSemaphoreAcquire(SparkNvrmChannel *channel, uint64_t address, uint64_t value, const char *owner);
SparkStatus SparkNvrmSemaphoreMax(SparkNvrmChannel *channel, uint64_t address, uint64_t value, const char *owner);
SparkStatus SparkNvrmEnsureLocalMemory(SparkNvrmGpu *gpu, uint32_t bytes_per_thread, const char *owner);
SparkStatus SparkNvrmBindLocalMemory(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner);
SparkStatus SparkNvrmWaitValue(const volatile uint64_t *semaphore, uint64_t value, uint64_t timeout_ns, const char *operation, const char *owner);
SparkStatus SparkNvrmWaitFence(const SparkNvrmChannel *channel, uint64_t value, uint64_t timeout_ns, const char *operation, const char *owner);
uint64_t SparkNvrmChannelCompleted(const SparkNvrmChannel *channel);
uint32_t SparkNvrmChannelFault(const SparkNvrmChannel *channel, SparkNvrmFault *fault);
void SparkNvrmFaultLocate(SparkNvrmGpu *gpu, const SparkNvrmChannel *channel, SparkNvrmFault *fault);
const char *SparkNvrmFaultName(uint32_t code);
void SparkNvrmFaultText(const SparkNvrmFault *fault, char *text, uint32_t capacity);
SparkStatus SparkNvrmOpBegin(SparkNvrmChannel *channel, SparkNvrmEngine engine, const char *owner);
SparkStatus SparkNvrmOpBeginLatched(SparkNvrmChannel *channel, const char *owner);
SparkStatus SparkNvrmOpEnd(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmEngine engine, uint64_t *value, const char *owner);
void SparkNvrmOpAbandon(SparkNvrmChannel *channel);
SparkStatus SparkNvrmOpFinishHost(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t *value, const char *owner);
SparkStatus SparkNvrmMark(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t address, uint64_t payload, uint64_t *value, const char *owner);
SparkStatus SparkNvrmAwait(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t address, uint64_t payload, uint64_t *value, const char *owner);
SparkStatus SparkNvrmCopy(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t destination, uint64_t source, uint64_t bytes, uint64_t *value,
	const char *owner);
SparkStatus SparkNvrmFill(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t destination, uint8_t byte, uint64_t bytes, uint64_t *value,
	const char *owner);

#ifdef __cplusplus
}
#endif
