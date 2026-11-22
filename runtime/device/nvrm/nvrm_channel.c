#include <stddef.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "nvos.h"
#include "nvrm_release.h"
#include "class/cl2080_notification.h"
#include "class/cl9067.h"
#include "class/cla06c.h"
#include "class/clc96f.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrl2080/ctrl2080ce.h"
#include "ctrl/ctrlc36f.h"
#include "classes/compute/clcec0.h"

#define CHANNEL_USERD_BYTES 4096u
#define CHANNEL_NOTIFIER_BYTES (2u << 20)
#define CHANNEL_FENCE_BYTES 4096u
#define CHANNEL_UVM_BYTES 0x4000000ull
#define CHANNEL_COMPUTE_SUBCHANNEL 1u
#define CHANNEL_COPY_SUBCHANNEL 4u
#define CHANNEL_LOCAL_THREAD_ALIGN 32u
#define CHANNEL_LOCAL_WARP_ALIGN 0x200u
#define CHANNEL_LOCAL_TPC_ALIGN 0x8000u
#define CHANNEL_LOCAL_TOTAL_ALIGN 0x20000u
#define CHANNEL_LOCAL_SM_MASK 0xffu

#define CHANNEL_TRY(call) \
	do { \
		const SparkStatus channel_status = (call); \
		if ( channel_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(channel_status); \
	} while (0)

static uint64_t ChannelRound(uint64_t value, uint64_t unit)
{
	return((value + unit - 1u) / unit * unit);
}

static SparkStatus ChannelObjects(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint32_t group, uint32_t share, uint32_t compute, const char *owner)
{
	NV_CHANNEL_ALLOC_PARAMS made;
	NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token;
	UVM_REGISTER_CHANNEL_PARAMS registration;
	memset(&made,0,sizeof(made));
	made.hObjectError = channel->notifier.handle;
	made.hObjectBuffer = channel->control.handle;
	made.gpFifoOffset = channel->control.address;
	made.gpFifoEntries = channel->entries;
	made.hContextShare = share;
	made.hUserdMemory[0] = channel->control.handle;
	made.userdOffset[0] = (uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,group,gpu->gpfifo_class,&made,(uint32_t)sizeof(made),&channel->handle,owner));
	channel->group = group;
	if ( compute != 0u )
		CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,channel->handle,gpu->compute_class,0,0u,&channel->compute_object,owner));
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,channel->handle,gpu->copy_class,0,0u,&channel->copy_object,owner));
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
	CHANNEL_TRY(SparkNvrmVirtual(gpu,CHANNEL_UVM_BYTES,SPARK_NVRM_HUGE_PAGE_BYTES,&channel->uvm_base,owner));
	registration.base = channel->uvm_base;
	CHANNEL_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_REGISTER_CHANNEL,&registration,(uint32_t)sizeof(registration),&registration.rmStatus,"initialize",owner));
	channel->uvm_registered = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus ChannelCopyBind(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	uint32_t words[1];
	words[0] = gpu->copy_class;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COPY_SUBCHANNEL,NVCEC0_SET_OBJECT,words,1u,owner));
	return(SparkNvrmChannelSync(gpu,channel,SPARK_NVRM_WAIT_TIMEOUT_NS,owner));
}

static SparkStatus ChannelBind(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	uint32_t words[2];
	words[0] = gpu->compute_class;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_OBJECT,words,1u,owner));
	words[0] = gpu->copy_class;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COPY_SUBCHANNEL,NVCEC0_SET_OBJECT,words,1u,owner));
	words[0] = (uint32_t)(gpu->local_window >> 32);
	words[1] = (uint32_t)gpu->local_window;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,words,2u,owner));
	words[0] = (uint32_t)(gpu->shared_window >> 32);
	words[1] = (uint32_t)gpu->shared_window;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_A,words,2u,owner));
	return(SparkNvrmChannelSync(gpu,channel,SPARK_NVRM_WAIT_TIMEOUT_NS,owner));
}

static SparkStatus ChannelCreateIn(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmMemoryKind control_kind, uint32_t compute, const char *owner)
{
	SparkStatus status;
	memset(channel,0,sizeof(*channel));
	channel->control.map_fd = -1;
	channel->notifier.map_fd = -1;
	channel->push.map_fd = -1;
	channel->fence.map_fd = -1;
	channel->entries = SPARK_NVRM_CHANNEL_ENTRIES;
	status = SparkNvrmMemoryAllocate(gpu,(uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE + CHANNEL_USERD_BYTES,control_kind,&channel->control,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmMemoryAllocate(gpu,CHANNEL_NOTIFIER_BYTES,SPARK_NVRM_MEMORY_SYSTEM_UNCACHED,&channel->notifier,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmMemoryAllocate(gpu,SPARK_NVRM_PUSH_BYTES,SPARK_NVRM_MEMORY_HOST,&channel->push,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmMemoryAllocate(gpu,CHANNEL_FENCE_BYTES,SPARK_NVRM_MEMORY_HOST,&channel->fence,owner);
	if ( status == SPARK_STATUS_OK )
		status = ChannelObjects(gpu,channel,compute != 0u ? gpu->channel_group : gpu->copy_group,compute != 0u ? gpu->context_share : gpu->copy_share,compute,owner);
	if ( status == SPARK_STATUS_OK && compute != 0u && gpu->channel_group_scheduled != 0u )
		status = ChannelBind(gpu,channel,owner);
	if ( status == SPARK_STATUS_OK && compute == 0u && gpu->copy_group_scheduled != 0u )
		status = ChannelCopyBind(gpu,channel,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmChannelDestroy(gpu,channel);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmChannelCreate(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmMemoryKind control_kind, const char *owner)
{
	return(ChannelCreateIn(gpu,channel,control_kind,1u,owner));
}

SparkStatus SparkNvrmCopyEngines(SparkNvrmGpu *gpu, uint32_t *engines, uint32_t capacity, uint32_t *count, const char *owner)
{
	NV2080_CTRL_GPU_GET_ENGINES_V2_PARAMS list;
	uint32_t index;
	memset(&list,0,sizeof(list));
	CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,gpu->subdevice,NV2080_CTRL_CMD_GPU_GET_ENGINES_V2,&list,(uint32_t)sizeof(list),owner));
	*count = 0u;
	for (index = 0u; index < list.engineCount && index < NV2080_GPU_MAX_ENGINES_LIST_SIZE; index++)
		if ( NV2080_ENGINE_TYPE_IS_COPY(list.engineList[index]) && *count < capacity )
			engines[(*count)++] = list.engineList[index];
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmCopyEngineAsync(SparkNvrmGpu *gpu, uint32_t *engine, const char *owner)
{
	uint32_t engines[NV2080_ENGINE_TYPE_COPY_SIZE],count = 0u,index;
	CHANNEL_TRY(SparkNvrmCopyEngines(gpu,engines,NV2080_ENGINE_TYPE_COPY_SIZE,&count,owner));
	for (index = 0u; index < count; index++)
	{
		NV2080_CTRL_CE_GET_CAPS_V2_PARAMS caps;
		memset(&caps,0,sizeof(caps));
		caps.ceEngineType = engines[index];
		CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,gpu->subdevice,NV2080_CTRL_CMD_CE_GET_CAPS_V2,&caps,(uint32_t)sizeof(caps),owner));
		if ( NV2080_CTRL_CE_GET_CAP(caps.capsTbl,NV2080_CTRL_CE_CAPS_CE_GRCE) == 0u )
		{
			*engine = engines[index];
			return(SPARK_STATUS_OK);
		}
	}
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"initialize",owner,0,
			"the GPU has no copy engine outside the graphics engine, so copies would wait behind other processes' grids");
		SPARK_FAIL(status);
	}
}

SparkStatus SparkNvrmCopyGroupOpen(SparkNvrmGpu *gpu, uint32_t engine, const char *owner)
{
	NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS group;
	memset(&group,0,sizeof(group));
	group.engineType = engine;
	group.hVASpace = gpu->vaspace;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,KEPLER_CHANNEL_GROUP_A,&group,(uint32_t)sizeof(group),&gpu->copy_group,owner));
	gpu->copy_engine = engine;
	return(SparkNvrmCopyChannelCreate(gpu,&gpu->copy,owner));
}

SparkStatus SparkNvrmCopyChannelCreate(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS schedule;
	CHANNEL_TRY(ChannelCreateIn(gpu,channel,SPARK_NVRM_MEMORY_HOST,0u,owner));
	if ( gpu->copy_group_scheduled != 0u )
		return(SPARK_STATUS_OK);
	memset(&schedule,0,sizeof(schedule));
	schedule.bEnable = 1u;
	CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,gpu->copy_group,NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,&schedule,(uint32_t)sizeof(schedule),owner));
	gpu->copy_group_scheduled = 1u;
	return(ChannelCopyBind(gpu,channel,owner));
}

void SparkNvrmCopyGroupClose(SparkNvrmGpu *gpu)
{
	if ( gpu->copy.handle != 0u || gpu->copy.control.bytes != 0u )
		SparkNvrmChannelDestroy(gpu,&gpu->copy);
	if ( gpu->copy_share != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->copy_group,gpu->copy_share,"free");
	if ( gpu->copy_group != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->device,gpu->copy_group,"free");
	gpu->copy_share = 0u;
	gpu->copy_group = 0u;
	gpu->copy_group_scheduled = 0u;
	gpu->copy_engine = 0u;
}

void SparkNvrmChannelDestroy(SparkNvrmGpu *gpu, SparkNvrmChannel *channel)
{
	if ( channel->fence.cpu != 0 && channel->handle != 0u )
		(void)SparkNvrmWaitValue((const volatile uint64_t *)(void *)channel->fence.cpu,channel->fence_value,SPARK_NVRM_WAIT_TIMEOUT_NS,"queue_destroy","free");
	if ( channel->uvm_registered != 0u )
	{
		UVM_UNREGISTER_CHANNEL_PARAMS registration;
		memset(&registration,0,sizeof(registration));
		SPARK_NVRM_UVM_UNREGISTER_GPU(registration,gpu->uuid);
		registration.hClient = gpu->rm.client;
		registration.hChannel = channel->handle;
		(void)SparkNvrmRmUvm(&gpu->rm,-1,UVM_UNREGISTER_CHANNEL,&registration,(uint32_t)sizeof(registration),&registration.rmStatus,"free","free");
	}
	if ( channel->copy_object != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,channel->handle,channel->copy_object,"free");
	if ( channel->compute_object != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,channel->handle,channel->compute_object,"free");
	if ( channel->handle != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,channel->group,channel->handle,"free");
	if ( channel->uvm_base != 0u )
		SparkNvrmVirtualRelease(gpu,channel->uvm_base,CHANNEL_UVM_BYTES);
	if ( channel->fence.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->fence);
	if ( channel->push.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->push);
	if ( channel->notifier.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->notifier);
	if ( channel->control.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&channel->control);
	memset(channel,0,sizeof(*channel));
}

SparkStatus SparkNvrmChannelOpen(SparkNvrmGpu *gpu, SparkNvrmMemoryKind control_kind, const char *owner)
{
	NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS group;
	NV_CTXSHARE_ALLOCATION_PARAMETERS share;
	NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS schedule;
	memset(&group,0,sizeof(group));
	group.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,KEPLER_CHANNEL_GROUP_A,&group,(uint32_t)sizeof(group),&gpu->channel_group,owner));
	memset(&share,0,sizeof(share));
	share.hVASpace = gpu->vaspace;
	share.flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
	CHANNEL_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->channel_group,FERMI_CONTEXT_SHARE_A,&share,(uint32_t)sizeof(share),&gpu->context_share,owner));
	CHANNEL_TRY(SparkNvrmChannelCreate(gpu,&gpu->compute,control_kind,owner));
	memset(&schedule,0,sizeof(schedule));
	schedule.bEnable = 1u;
	CHANNEL_TRY(SparkNvrmRmControl(&gpu->rm,gpu->channel_group,NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,&schedule,(uint32_t)sizeof(schedule),owner));
	gpu->channel_group_scheduled = 1u;
	return(ChannelBind(gpu,&gpu->compute,owner));
}

void SparkNvrmChannelClose(SparkNvrmGpu *gpu)
{
	SparkNvrmChannelDestroy(gpu,&gpu->compute);
	if ( gpu->context_share != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->channel_group,gpu->context_share,"free");
	if ( gpu->channel_group != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->device,gpu->channel_group,"free");
	gpu->context_share = 0u;
	gpu->channel_group = 0u;
	gpu->channel_group_scheduled = 0u;
}

SparkStatus SparkNvrmEnsureLocalMemory(SparkNvrmGpu *gpu, uint32_t bytes_per_thread, const char *owner)
{
	SparkNvrmMemory grown;
	uint64_t per_thread,per_tpc,total;
	if ( gpu->local_bytes_per_thread >= bytes_per_thread && gpu->local_memory.bytes != 0u )
		return(SPARK_STATUS_OK);
	if ( gpu->local_memory.bytes != 0u && gpu->local_retired_count == SPARK_NVRM_LOCAL_RETIRED )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"library_load",owner,0,
			"per-thread local memory grew more times than the device keeps retired allocations for");
		SPARK_FAIL(status);
	}
	per_thread = ChannelRound(bytes_per_thread > 2u * gpu->local_bytes_per_thread ? bytes_per_thread : 2u * gpu->local_bytes_per_thread,
		CHANNEL_LOCAL_THREAD_ALIGN);
	per_tpc = ChannelRound(ChannelRound(per_thread * 32u,CHANNEL_LOCAL_WARP_ALIGN) * gpu->warps_per_sm * gpu->sms_per_tpc,CHANNEL_LOCAL_TPC_ALIGN);
	total = ChannelRound(per_tpc * gpu->tpcs_per_gpc * gpu->gpcs,CHANNEL_LOCAL_TOTAL_ALIGN);
	CHANNEL_TRY(SparkNvrmMemoryAllocate(gpu,total,SPARK_NVRM_MEMORY_SYSTEM,&grown,owner));
	if ( gpu->local_memory.bytes != 0u )
		gpu->local_retired[gpu->local_retired_count++] = gpu->local_memory;
	gpu->local_memory = grown;
	gpu->local_bytes_per_tpc = per_tpc;
	gpu->local_bytes_per_thread = (uint32_t)per_thread;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmBindLocalMemory(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	uint32_t words[3];
	if ( gpu->local_memory.bytes == 0u || channel->local_address == gpu->local_memory.address )
		return(SPARK_STATUS_OK);
	words[0] = (uint32_t)(gpu->local_memory.address >> 32);
	words[1] = (uint32_t)gpu->local_memory.address;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_A,words,2u,owner));
	words[0] = (uint32_t)(gpu->local_bytes_per_tpc >> 32);
	words[1] = (uint32_t)gpu->local_bytes_per_tpc;
	words[2] = CHANNEL_LOCAL_SM_MASK;
	CHANNEL_TRY(SparkNvrmPush(channel,CHANNEL_COMPUTE_SUBCHANNEL,NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,words,3u,owner));
	channel->local_address = gpu->local_memory.address;
	return(SPARK_STATUS_OK);
}
