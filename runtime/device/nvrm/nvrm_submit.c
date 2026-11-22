#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_wait.h"
#include "class/clc361.h"
#include "class/clc56f.h"
#include "class/clc96f.h"

#define SUBMIT_HOST_SUBCHANNEL 0u
#define SUBMIT_METHOD_COUNT_MAX 8191u
#define SUBMIT_ENTRY_LEVEL_BIT 41u
#define SUBMIT_ENTRY_LENGTH_SHIFT 42u
#define SUBMIT_SEMAPHORE_WORDS 5u

#define SUBMIT_TRY(call) \
	do { \
		const SparkStatus submit_status = (call); \
		if ( submit_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(submit_status); \
	} while (0)

#define SUBMIT_REFUSE(code, text) \
	do { \
		const SparkStatus submit_status = SparkDeviceRecordError((code),"submit",owner,0,(text)); \
		SPARK_FAIL(submit_status); \
	} while (0)

static void SubmitStoreFence(void)
{
#if defined(__aarch64__)
	__asm__ volatile("dsb st" ::: "memory");
#else
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

uint64_t SparkNvrmChannelCompleted(const SparkNvrmChannel *channel)
{
	return(__atomic_load_n((const volatile uint64_t *)(const void *)channel->fence.cpu,__ATOMIC_ACQUIRE));
}

SparkStatus SparkNvrmWaitValue(const volatile uint64_t *semaphore, uint64_t value, uint64_t timeout_ns, const char *operation, const char *owner)
{
	SparkNvrmWait wait;
	int expired = 0;
	SparkNvrmWaitBegin(&wait);
	while ( __atomic_load_n(semaphore,__ATOMIC_ACQUIRE) < value && (expired = SparkNvrmWaitPace(&wait,timeout_ns)) == 0 )
		;
	SparkNvrmWaitEnd(&wait);
	if ( expired != 0 )
	{
		char text[96];
		snprintf(text,sizeof(text),"the GPU did not reach %llu (at %llu) in time",(unsigned long long)value,
			(unsigned long long)__atomic_load_n(semaphore,__ATOMIC_ACQUIRE));
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,owner,0,text);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static void SubmitRetire(SparkNvrmChannel *channel)
{
	const uint64_t completed = SparkNvrmChannelCompleted(channel);
	while ( channel->segment_count != 0u && channel->segments[channel->segment_head].value <= completed )
	{
		channel->segment_head = (channel->segment_head + 1u) % SPARK_NVRM_CHANNEL_ENTRIES;
		channel->segment_count--;
	}
}

static SparkStatus SubmitRetireOldest(SparkNvrmChannel *channel, const char *owner)
{
	const SparkNvrmSegment *oldest = &channel->segments[channel->segment_head];
	SUBMIT_TRY(SparkNvrmWaitValue((const volatile uint64_t *)(void *)channel->fence.cpu,oldest->value,SPARK_NVRM_WAIT_TIMEOUT_NS,"submit",owner));
	channel->segment_head = (channel->segment_head + 1u) % SPARK_NVRM_CHANNEL_ENTRIES;
	channel->segment_count--;
	return(SPARK_STATUS_OK);
}

static uint32_t SubmitFree(const SparkNvrmChannel *channel, uint64_t start, uint64_t end)
{
	uint64_t tail;
	if ( channel->segment_count == 0u )
		return(1u);
	tail = channel->segments[channel->segment_head].start;
	if ( tail < channel->push_submitted )
		return(start >= channel->push_submitted || end <= tail ? 1u : 0u);
	if ( tail > channel->push_submitted )
		return(start >= channel->push_submitted && end <= tail ? 1u : 0u);
	return(0u);
}

static SparkStatus SubmitRoom(SparkNvrmChannel *channel, uint64_t start, uint64_t end, const char *owner)
{
	SubmitRetire(channel);
	while ( SubmitFree(channel,start,end) == 0u )
		SUBMIT_TRY(SubmitRetireOldest(channel,owner));
	return(SPARK_STATUS_OK);
}

static SparkStatus SubmitSpace(SparkNvrmChannel *channel, uint64_t bytes, const char *owner)
{
	const uint64_t open = channel->push_used - channel->push_submitted;
	if ( open + bytes > channel->push.bytes )
		SUBMIT_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,"the methods since the last submission do not fit the push buffer");
	if ( channel->push_used + bytes <= channel->push.bytes )
		return(SubmitRoom(channel,channel->push_used,channel->push_used + bytes,owner));
	SUBMIT_TRY(SubmitRoom(channel,0u,open + bytes,owner));
	memmove(channel->push.cpu,channel->push.cpu + channel->push_submitted,(size_t)open);
	channel->push_submitted = 0u;
	channel->push_used = open;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmPush(SparkNvrmChannel *channel, uint32_t subchannel, uint32_t method, const uint32_t *values, uint32_t count, const char *owner)
{
	uint32_t *at;
	const uint64_t bytes = sizeof(uint32_t) * ((uint64_t)count + 1u);
	if ( count == 0u || count > SUBMIT_METHOD_COUNT_MAX || (method & 3u) != 0u || subchannel > 7u )
		SUBMIT_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a method needs 1 to 8191 words, a word-aligned address and a subchannel below 8");
	SUBMIT_TRY(SubmitSpace(channel,bytes,owner));
	at = (uint32_t *)(void *)(channel->push.cpu + channel->push_used);
	at[0] = (uint32_t)(SPARK_NVRM_VALUE(NVC56F_DMA_SEC_OP,NVC56F_DMA_SEC_OP_INC_METHOD) | SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_COUNT,count) |
		SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_SUBCHANNEL,subchannel) | SPARK_NVRM_VALUE(NVC56F_DMA_METHOD_ADDRESS,method >> 2));
	memcpy(at + 1,values,(size_t)count * sizeof(*values));
	channel->push_used += bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SubmitSemaphore(SparkNvrmChannel *channel, uint64_t address, uint64_t value, uint32_t execute, uint64_t align, const char *owner)
{
	uint32_t words[SUBMIT_SEMAPHORE_WORDS];
	if ( (address & (align - 1u)) != 0u )
		SUBMIT_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a semaphore is not aligned to its payload");
	words[0] = (uint32_t)address;
	words[1] = (uint32_t)(address >> 32);
	words[2] = (uint32_t)value;
	words[3] = (uint32_t)(value >> 32);
	words[4] = execute | (uint32_t)SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_PAYLOAD_SIZE,NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT);
	return(SparkNvrmPush(channel,SUBMIT_HOST_SUBCHANNEL,NVC56F_SEM_ADDR_LO,words,SUBMIT_SEMAPHORE_WORDS,owner));
}

SparkStatus SparkNvrmSemaphoreAcquire(SparkNvrmChannel *channel, uint64_t address, uint64_t value, const char *owner)
{
	const uint32_t acquire = (uint32_t)SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_OPERATION,NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ);
	return(SubmitSemaphore(channel,address,value,acquire,8u,owner));
}

SparkStatus SparkNvrmSemaphoreMax(SparkNvrmChannel *channel, uint64_t address, uint64_t value, const char *owner)
{
	const uint32_t execute = (uint32_t)(SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_OPERATION,NVC56F_SEM_EXECUTE_OPERATION_REDUCTION) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_RELEASE_WFI,NVC56F_SEM_EXECUTE_RELEASE_WFI_EN) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_REDUCTION,NVC56F_SEM_EXECUTE_REDUCTION_IMAX) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_REDUCTION_FORMAT,NVC56F_SEM_EXECUTE_REDUCTION_FORMAT_UNSIGNED));
	return(SubmitSemaphore(channel,address,value,execute,8u,owner));
}

SparkStatus SparkNvrmSemaphoreRelease(SparkNvrmChannel *channel, uint64_t address, uint64_t value, uint32_t timestamp, const char *owner)
{
	const uint32_t execute = (uint32_t)(SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_OPERATION,NVC56F_SEM_EXECUTE_OPERATION_RELEASE) |
		SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_RELEASE_WFI,NVC56F_SEM_EXECUTE_RELEASE_WFI_EN) |
		(timestamp != 0u ? SPARK_NVRM_VALUE(NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP,NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN) : 0u));
	return(SubmitSemaphore(channel,address,value,execute,timestamp != 0u ? 16u : 8u,owner));
}

static uint32_t SubmitOrdered(SparkNvrmEngine previous, SparkNvrmEngine next)
{
	if ( previous == SPARK_NVRM_ENGINE_NONE )
		return(1u);
	return(previous == next && (next == SPARK_NVRM_ENGINE_HOST || next == SPARK_NVRM_ENGINE_COPY) ? 1u : 0u);
}

void SparkNvrmOpAbandon(SparkNvrmChannel *channel)
{
	channel->push_used = channel->push_submitted;
}

static SparkStatus SubmitOrder(SparkNvrmChannel *channel, SparkNvrmEngine engine, const char *owner)
{
	if ( SubmitOrdered(channel->last_engine,engine) != 0u )
		return(SPARK_STATUS_OK);
	return(SparkNvrmSemaphoreAcquire(channel,channel->fence.address,channel->fence_value,owner));
}

void SparkNvrmChannelAfter(SparkNvrmChannel *channel, const SparkNvrmChannel *other)
{
	if ( SparkNvrmChannelCompleted(other) >= other->fence_value )
		return;
	channel->after_address = other->fence.address;
	channel->after_value = other->fence_value;
}

SparkStatus SparkNvrmOpBegin(SparkNvrmChannel *channel, SparkNvrmEngine engine, const char *owner)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( channel->push_used != channel->push_submitted )
		SUBMIT_REFUSE(SPARK_STATUS_BUSY,"an ordered operation starts after the previous submission");
	if ( channel->after_value != 0u )
		status = SparkNvrmSemaphoreAcquire(channel,channel->after_address,channel->after_value,owner);
	if ( status == SPARK_STATUS_OK )
		status = SubmitOrder(channel,engine,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SubmitHostFence(SparkNvrmChannel *channel, const char *owner)
{
	SUBMIT_TRY(SubmitOrder(channel,SPARK_NVRM_ENGINE_HOST,owner));
	SUBMIT_TRY(SparkNvrmSemaphoreRelease(channel,channel->fence.address,channel->fence_value + 1u,0u,owner));
	channel->fence_value++;
	channel->segment_fenced = 1u;
	channel->last_engine = SPARK_NVRM_ENGINE_HOST;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmSubmit(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const char *owner)
{
	volatile uint64_t *ring = (volatile uint64_t *)(void *)channel->control.cpu;
	volatile uint32_t *put = (volatile uint32_t *)(void *)(channel->control.cpu + (uint64_t)channel->entries * NVC96F_GP_ENTRY__SIZE + offsetof(Nvc96fControl,GPPut));
	SparkNvrmSegment *segment;
	uint64_t words,address;
	if ( channel->push_used == channel->push_submitted )
		SUBMIT_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a submission needs methods pushed since the previous one");
	if ( (channel->push.address + channel->push.bytes - 1u) >> 40 != 0u )
		SUBMIT_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a push buffer must lie below 2^40");
	if ( channel->segment_fenced == 0u )
		SUBMIT_TRY(SubmitHostFence(channel,owner));
	while ( channel->segment_count + 1u >= channel->entries || channel->segment_count + 1u >= SPARK_NVRM_CHANNEL_ENTRIES )
		SUBMIT_TRY(SubmitRetireOldest(channel,owner));
	words = (channel->push_used - channel->push_submitted) / sizeof(uint32_t);
	address = channel->push.address + channel->push_submitted;
	segment = &channel->segments[(channel->segment_head + channel->segment_count) % SPARK_NVRM_CHANNEL_ENTRIES];
	segment->start = channel->push_submitted;
	segment->end = channel->push_used;
	segment->value = channel->fence_value;
	channel->segment_count++;
	channel->push_submitted = channel->push_used;
	channel->segment_fenced = 0u;
	ring[channel->put % channel->entries] = address | (words << SUBMIT_ENTRY_LENGTH_SHIFT) | (1ull << SUBMIT_ENTRY_LEVEL_BIT);
	channel->put++;
	SubmitStoreFence();
	*put = (uint32_t)(channel->put % channel->entries);
	SubmitStoreFence();
	gpu->usermode_page[NVC361_NOTIFY_CHANNEL_PENDING / 4u] = channel->token;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmOpEnd(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmEngine engine, uint64_t *value, const char *owner)
{
	SparkStatus status;
	channel->fence_value++;
	channel->segment_fenced = 1u;
	status = SparkNvrmSubmit(gpu,channel,owner);
	if ( status != SPARK_STATUS_OK )
	{
		channel->fence_value--;
		channel->segment_fenced = 0u;
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	channel->last_engine = engine;
	channel->after_address = 0u;
	channel->after_value = 0u;
	if ( value != 0 )
		*value = channel->fence_value;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmChannelSync(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t timeout_ns, const char *owner)
{
	SUBMIT_TRY(SubmitHostFence(channel,owner));
	SUBMIT_TRY(SparkNvrmSubmit(gpu,channel,owner));
	return(SparkNvrmWaitValue((const volatile uint64_t *)(void *)channel->fence.cpu,channel->fence_value,timeout_ns,"synchronize",owner));
}

SparkStatus SparkNvrmOpFinishHost(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t *value, const char *owner)
{
	SparkStatus status = SparkNvrmSemaphoreRelease(channel,channel->fence.address,channel->fence_value + 1u,0u,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmOpEnd(gpu,channel,SPARK_NVRM_ENGINE_HOST,value,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmMark(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t address, uint64_t payload, uint64_t *value, const char *owner)
{
	SparkStatus status;
	SUBMIT_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_HOST,owner));
	status = SparkNvrmSemaphoreRelease(channel,address + SPARK_NVRM_MARK_TIME_OFFSET,payload,1u,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmSemaphoreRelease(channel,channel->fence.address,channel->fence_value + 1u,0u,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmSemaphoreMax(channel,address,payload,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmOpEnd(gpu,channel,SPARK_NVRM_ENGINE_HOST,value,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmAwait(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t address, uint64_t payload, uint64_t *value, const char *owner)
{
	SparkStatus status;
	SUBMIT_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_HOST,owner));
	status = SparkNvrmSemaphoreAcquire(channel,address,payload,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SparkNvrmOpFinishHost(gpu,channel,value,owner));
}
