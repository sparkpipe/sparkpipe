#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "nvos.h"
#include "uvm_linux_ioctl.h"
#include "class/cl2080_notification.h"
#include "class/cl9067.h"
#include "class/cla06c.h"
#include "class/clc361.h"
#include "class/clc56f.h"
#include "class/clc96f.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrlc36f.h"
#include "classes/compute/clcec0.h"

#define CHANNEL_USERD_BYTES 4096u
#define CHANNEL_NOTIFIER_BYTES (2u << 20)
#define CHANNEL_FENCE_BYTES 4096u
#define CHANNEL_UVM_BYTES 0x4000000ull
#define CHANNEL_SYNC_TIMEOUT_NS 10000000000ull
#define CHANNEL_COMPUTE_SUBCHANNEL 1u
#define CHANNEL_HOST_SUBCHANNEL 0u
#define CHANNEL_METHOD_COUNT_MAX 8191u
#define CHANNEL_ENTRY_LEVEL_BIT 41u
#define CHANNEL_ENTRY_LENGTH_SHIFT 42u
#define CHANNEL_LOCAL_THREAD_ALIGN 32u
#define CHANNEL_LOCAL_WARP_ALIGN 0x200u
#define CHANNEL_LOCAL_TPC_ALIGN 0x8000u
#define CHANNEL_LOCAL_TOTAL_ALIGN 0x20000u
#define CHANNEL_LOCAL_SM_MASK 0xffu
#define CHANNEL_SPIN_POLLS 4096u

#define CHANNEL_TRY(call) \
	do { \
		const SparkStatus channel_status = (call); \
		if ( channel_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(channel_status); \
	} while (0)

static void ChannelStoreFence(void)
{
#if defined(__aarch64__)
	__asm__ volatile("dsb st" ::: "memory");
#else
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static uint64_t ChannelRound(uint64_t value, uint64_t unit)
{
	return((value + unit - 1u) / unit * unit);
}

SparkStatus SparkNvrmPush(SparkNvrmChannel *channel, uint32_t subchannel, uint32_t method, const uint32_t *values, uint32_t count, const char *owner)
{
	uint32_t *at;
	const uint64_t bytes = sizeof(uint32_t) * ((uint64_t)count + 1u);
	if ( count == 0u || count > CHANNEL_METHOD_COUNT_MAX || (method & 3u) != 0u || subchannel > 7u )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"submit",owner,0,"a method needs 1 to 8191 words, a word-aligned address and a subchannel below 8");
		SPARK_FAIL(status);
	}
	if ( channel->push_used + bytes > channel->push.bytes )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"submit",owner,0,"the channel's push buffer is full");
		SPARK_FAIL(status);
	}
	at = (uint32_t *)(void *)(channel->push.cpu + channel->push_used);
	at[0] = (uint32_t)(SPARK_NVRM_VALUE(NVC56F_DMA_SEC_OP,NVC56F_DMA_SEC_OP_INC_METHOD) | SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_COUNT,count) |
		SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_SUBCHANNEL,subchannel) | SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_ADDRESS,method >> 2));
	memcpy(at + 1,values,(size_t)count * sizeof(*values));
	channel->push_used += bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmSubmit(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	const uint64_t first_byte = channel->push_submitted;
	volatile uint64_t *ring = (volatile uint64_t *)(void *)channel->control.cpu;
	volatile uint32_t *put = (volatile uint32_t *)(void *)(channel->control.cpu + (uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE + offsetof(Nvc96fControl,GPPut));
	const uint64_t words = (channel->push_used - first_byte) / sizeof(uint32_t);
	const uint64_t address = channel->push.address + first_byte;
	if ( words == 0u || first_byte > channel->push_used || address >> 40 != 0u )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"submit",owner,0,"a submission needs unsubmitted methods in a push buffer below 2^40");
		SPARK_FAIL(status);
	}
	channel->push_submitted = channel->push_used;
	ring[channel->put % channel->entries] = address | (words << CHANNEL_ENTRY_LENGTH_SHIFT) | (1ull << CHANNEL_ENTRY_LEVEL_BIT);
	channel->put++;
	ChannelStoreFence();
	*put = (uint32_t)(channel->put % channel->entries);
	ChannelStoreFence();
	gpu->usermode_page[NVC361_NOTIFY_CHANNEL_PENDING / 4u] = channel->token;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmSemaphoreRelease(SparkNvrmChannel *channel, uint64_t address, uint64_t value, uint32_t timestamp, const char *owner)
{
	uint32_t words[5];
	if ( (address & (timestamp != 0u ? 0xfu : 0x7u)) != 0u )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"submit",owner,0,"a semaphore is not aligned to its payload");
		SPARK_FAIL(status);
	}
	words[0] = (uint32_t)address;
	words[1] = (uint32_t)(address >> 32);
	words[2] = (uint32_t)value;
	words[3] = (uint32_t)(value >> 32);
	words[4] = (uint32_t)(SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_OPERATION,NVC56F_SEM_EXECUTE_OPERATION_RELEASE) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_RELEASE_WFI,NVC56F_SEM_EXECUTE_RELEASE_WFI_EN) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_PAYLOAD_SIZE,NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT) |
		(timestamp != 0u ? SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP,NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN) : 0u));
	return(SparkNvrmPush(channel,CHANNEL_HOST_SUBCHANNEL,NVC56F_SEM_ADDR_LO,words,5u,owner));
}

SparkStatus SparkNvrmWaitValue(const volatile uint64_t *semaphore, uint64_t value, uint64_t timeout_ns, const char *operation, const char *owner)
{
	const uint64_t deadline = SparkMonotonicNs() + timeout_ns;
	uint32_t polls = 0u;
	while ( __atomic_load_n(semaphore,__ATOMIC_ACQUIRE) < value )
	{
		struct timespec pause = {0, 20000};
		if ( ++polls < CHANNEL_SPIN_POLLS )
			continue;
		if ( SparkMonotonicNs() >= deadline )
		{
			char text[96];
			snprintf(text,sizeof(text),"the GPU did not reach %llu (at %llu) in time",(unsigned long long)value,
				(unsigned long long)__atomic_load_n(semaphore,__ATOMIC_ACQUIRE));
			const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,owner,0,text);
			SPARK_FAIL(status);
		}
		nanosleep(&pause,0);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmPushRewind(SparkNvrmChannel *channel, const char *owner)
{
	if ( channel->push_submitted != channel->push_used )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"submit",owner,0,"the push buffer holds methods not yet submitted");
		SPARK_FAIL(status);
	}
	channel->push_used = 0u;
	channel->push_submitted = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmChannelSync(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t timeout_ns, const char *owner)
{
	const uint64_t value = ++channel->fence_value;
	CHANNEL_TRY(SparkNvrmSemaphoreRelease(channel,channel->fence.address,value,0u,owner));
	CHANNEL_TRY(SparkNvrmSubmit(gpu,channel,owner));
	return(SparkNvrmWaitValue((const volatile uint64_t *)(void *)channel->fence.cpu,value,timeout_ns,"synchronize",owner));
}

static SparkStatus ChannelObjects(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	NV_CHANNEL_ALLOC_PARAMS made;
	NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token;
	UVM_REGISTER_CHANNEL_PARAMS registration;
	uint64_t base = 0u;
	memset(&made,0,sizeof(made));
	made.hObjectError = channel->notifier.handle;
	made.hObjectBuffer = channel->control.handle;
	made.gpFifoOffset = channel->control.address;
	made.gpFifoEntries = channel->entries;
	made.hContextShare = gpu->context_share;
	made.hUserdMemory[0] = channel->control.handle;
	made.userdOffset[0] = (uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->channel_group,gpu->gpfifo_class,&made,(uint32_t)sizeof(made),&channel->handle,owner));
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,channel->handle,gpu->compute_class,0,0u,&channel->engine,owner));
	memset(&token,0,sizeof(token));
	token.workSubmitToken = UINT32_MAX;
	CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,channel->handle,NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,&token,(uint32_t)sizeof(token),owner));
	channel->token = token.workSubmitToken;
	memset(&registration,0,sizeof(registration));
	memcpy(registration.gpuUuid.uuid,gpu->uuid,sizeof(gpu->uuid));
	registration.rmCtrlFd = gpu->rm.control;
	registration.hClient = gpu->rm.client;
	registration.hChannel = channel->handle;
	registration.length = CHANNEL_UVM_BYTES;
	CHANNEL_TRY(SparkNvrmVirtual(gpu,CHANNEL_UVM_BYTES,SPARK_NVRM_HUGE_PAGE_BYTES,&base,owner));
	registration.base = base;
	CHANNEL_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_REGISTER_CHANNEL,&registration,(uint32_t)sizeof(registration),&registration.rmStatus,"initialize",owner));
	channel->uvm_registered = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus ChannelWindows(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	uint32_t words[2];
	words[0] = (uint32_t)(gpu->local_window >> 32);
	words[1] = (uint32_t)gpu->local_window;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,words,2u,owner));
	words[0] = (uint32_t)(gpu->shared_window >> 32);
	words[1] = (uint32_t)gpu->shared_window;
	return(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_A,words,2u,owner));
}

static SparkStatus ChannelBind(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	uint32_t object = gpu->compute_class;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_OBJECT,&object,1u,owner));
	CHANNEL_TRY(ChannelWindows(gpu,channel,owner));
	return(SparkNvrmChannelSync(gpu,channel,CHANNEL_SYNC_TIMEOUT_NS,owner));
}

SparkStatus SparkNvrmChannelOpen(SparkNvrmGpu *gpu, SparkNvrmMemoryKind control_kind, const char *owner)
{
	NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS group;
	NV_CTXSHARE_ALLOCATION_PARAMETERS share;
	NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS schedule;
	SparkNvrmChannel *channel = &gpu->compute;
	memset(&group,0,sizeof(group));
	group.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,KEPLER_CHANNEL_GROUP_A,&group,(uint32_t)sizeof(group),&gpu->channel_group,owner));
	memset(&share,0,sizeof(share));
	share.hVASpace = gpu->vaspace;
	share.flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->channel_group,FERMI_CONTEXT_SHARE_A,&share,(uint32_t)sizeof(share),&gpu->context_share,owner));
	channel->entries = SPARK_NVRM_CHANNEL_ENTRIES;
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,(uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE + CHANNEL_USERD_BYTES,control_kind,&channel->control,owner));
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,CHANNEL_NOTIFIER_BYTES,SPARK_NVRM_MEMORY_SYSTEM_UNCACHED,&channel->notifier,owner));
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,SPARK_NVRM_PUSH_BYTES,SPARK_NVRM_MEMORY_HOST,&channel->push,owner));
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,CHANNEL_FENCE_BYTES,SPARK_NVRM_MEMORY_HOST,&channel->fence,owner));
	CHANNEL_TRY(ChannelObjects(gpu,channel,owner));
	memset(&schedule,0,sizeof(schedule));
	schedule.bEnable = 1u;
	CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,gpu->channel_group,NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,&schedule,(uint32_t)sizeof(schedule),owner));
	return(ChannelBind(gpu,channel,owner));
}

void SparkNvrmChannelClose(SparkNvrmGpu *gpu)
{
	SparkNvrmChannel *channel = &gpu->compute;
	if ( channel->engine != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,channel->handle,channel->engine,"free");
	if ( channel->handle != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->channel_group,channel->handle,"free");
	if ( gpu->context_share != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->channel_group,gpu->context_share,"free");
	if ( gpu->channel_group != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->device,gpu->channel_group,"free");
	if ( channel->fence.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->fence);
	if ( channel->push.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->push);
	if ( channel->notifier.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->notifier);
	if ( channel->control.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->control);
	memset(channel,0,sizeof(*channel));
	gpu->context_share = 0u;
	gpu->channel_group = 0u;
}

SparkStatus SparkNvrmEnsureLocalMemory(SparkNvrmGpu *gpu, uint32_t bytes_per_thread, const char *owner)
{
	SparkNvrmMemory grown,previous;
	uint32_t words[3];
	uint64_t per_thread,per_tpc,total;
	if ( gpu->local_bytes_per_thread >= bytes_per_thread && gpu->local_memory.bytes != 0u )
		return(SPARK_STATUS_OK);
	per_thread = ChannelRound(bytes_per_thread,CHANNEL_LOCAL_THREAD_ALIGN);
	per_tpc = ChannelRound(ChannelRound(per_thread * 32u,CHANNEL_LOCAL_WARP_ALIGN) * gpu->warps_per_sm * gpu->sms_per_tpc,CHANNEL_LOCAL_TPC_ALIGN);
	total = ChannelRound(per_tpc * gpu->tpcs_per_gpc * gpu->gpcs,CHANNEL_LOCAL_TOTAL_ALIGN);
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,total,SPARK_NVRM_MEMORY_SYSTEM,&grown,owner));
	words[0] = (uint32_t)(grown.address >> 32);
	words[1] = (uint32_t)grown.address;
	CHANNEL_TRY(SparkNvrmPush(&gpu->compute,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_A,words,2u,owner));
	words[0] = (uint32_t)(per_tpc >> 32);
	words[1] = (uint32_t)per_tpc;
	words[2] = CHANNEL_LOCAL_SM_MASK;
	CHANNEL_TRY(SparkNvrmPush(&gpu->compute,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,words,3u,owner));
	CHANNEL_TRY(SparkNvrmChannelSync(gpu,&gpu->compute,CHANNEL_SYNC_TIMEOUT_NS,owner));
	previous = gpu->local_memory;
	gpu->local_memory = grown;
	gpu->local_bytes_per_thread = (uint32_t)per_thread;
	if ( previous.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&previous);
	return(SPARK_STATUS_OK);
}
