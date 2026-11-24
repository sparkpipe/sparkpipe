#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

#define ADDRESS_RANGES 64u
#define ADDRESS_RESERVE_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)
#if defined(__linux__)
#define ADDRESS_FIRST_FLAGS (ADDRESS_RESERVE_FLAGS | MAP_FIXED_NOREPLACE)
#else
#define ADDRESS_FIRST_FLAGS ADDRESS_RESERVE_FLAGS
#endif

#define ADDRESS_REFUSE(code, owner, text) \
	do { \
		const SparkStatus address_status = SparkDeviceRecordError((code),"allocate",(owner),0,(text)); \
		SPARK_FAIL(address_status); \
	} while (0)

#define ADDRESS_SERVER_BASE 0x100000000ull
#define ADDRESS_SERVER_BYTES 0x20000000ull
#define ADDRESS_WINDOW_BYTES 0x100000000ull
#define ADDRESS_GPU_LIMIT 0x1fffffb001000ull

static pthread_mutex_t address_lock = PTHREAD_MUTEX_INITIALIZER;

static void AddressKeepOut(SparkNvrmGpu *gpu, uint32_t index, uint64_t base, uint64_t bytes, const char *name)
{
	SparkNvrmKeepOut *keep = &gpu->keep_outs[index];
	void *reserved;
	if ( base == 0u || bytes == 0u )
		return;
	reserved = mmap((void *)(uintptr_t)base,(size_t)bytes,PROT_NONE,ADDRESS_FIRST_FLAGS,-1,0);
	keep->base = base;
	keep->bytes = bytes;
	keep->name = name;
	keep->reserved = reserved == (void *)(uintptr_t)base ? 1u : 0u;
	if ( reserved != MAP_FAILED && reserved != (void *)(uintptr_t)base )
		munmap(reserved,(size_t)bytes);
	if ( keep->reserved == 0u )
		fprintf(stderr,"nvrm: %s [0x%llx, 0x%llx) is already mapped in this process; registrations that overlap it are refused\n",
			name,(unsigned long long)base,(unsigned long long)(base + bytes));
}

static SparkStatus AddressOverlapRefusal(uint64_t base, uint64_t bytes, uint64_t other_base, uint64_t other_bytes, const char *other,
	const char *advice, const char *owner)
{
	char text[SPARK_DEVICE_ERROR_TEXT_BYTES];
	snprintf(text,sizeof(text),"the host range [0x%llx, 0x%llx) overlaps %s [0x%llx, 0x%llx); %s",(unsigned long long)base,
		(unsigned long long)(base + bytes),other,(unsigned long long)other_base,(unsigned long long)(other_base + other_bytes),advice);
	return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"register",owner,0,text));
}

static uint32_t AddressOverlaps(uint64_t base, uint64_t bytes, uint64_t other_base, uint64_t other_bytes)
{
	return(other_bytes != 0u && base < other_base + other_bytes && other_base < base + bytes ? 1u : 0u);
}

SparkStatus SparkNvrmIdentityTake(SparkNvrmGpu *gpu, uint64_t base, uint64_t bytes, const char *owner)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	pthread_mutex_lock(&address_lock);
	if ( bytes == 0u || base + bytes < base || base < 0x1000u || base + bytes > ADDRESS_GPU_LIMIT )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"register",owner,0,
			"a registered range is mapped at its own address, so it must lie inside the GPU address space");
	else if ( AddressOverlaps(base,bytes,gpu->reserved_base,gpu->reserved_bytes) != 0u )
		status = AddressOverlapRefusal(base,bytes,gpu->reserved_base,gpu->reserved_bytes,"the backend's device range",
			"host memory there is not the device's to map",owner);
	for (index = 0u; status == SPARK_STATUS_OK && index < SPARK_NVRM_KEEP_OUTS; index++)
		if ( AddressOverlaps(base,bytes,gpu->keep_outs[index].base,gpu->keep_outs[index].bytes) != 0u )
			status = AddressOverlapRefusal(base,bytes,gpu->keep_outs[index].base,gpu->keep_outs[index].bytes,gpu->keep_outs[index].name,
				"a registered range is mapped at its own address and cannot share GPU addresses with it",owner);
	for (index = 0u; status == SPARK_STATUS_OK && index < gpu->identity_count; index++)
		if ( AddressOverlaps(base,bytes,gpu->identity_ranges[index].base,gpu->identity_ranges[index].bytes) != 0u )
			status = AddressOverlapRefusal(base,bytes,gpu->identity_ranges[index].base,gpu->identity_ranges[index].bytes,"the registered range",
				"unregister it before registering memory there again",owner);
	if ( status == SPARK_STATUS_OK && gpu->identity_count == gpu->identity_capacity )
	{
		const uint32_t capacity = gpu->identity_capacity != 0u ? 2u * gpu->identity_capacity : ADDRESS_RANGES;
		SparkNvrmRange *grown = (SparkNvrmRange *)realloc(gpu->identity_ranges,(size_t)capacity * sizeof(*grown));
		if ( grown == 0 )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"register",owner,0,"out of host memory for the registered ranges");
		else
		{
			gpu->identity_ranges = grown;
			gpu->identity_capacity = capacity;
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		gpu->identity_ranges[gpu->identity_count].base = base;
		gpu->identity_ranges[gpu->identity_count].bytes = bytes;
		gpu->identity_count++;
	}
	pthread_mutex_unlock(&address_lock);
	return(status);
}

void SparkNvrmIdentityGive(SparkNvrmGpu *gpu, uint64_t base, uint64_t bytes)
{
	uint32_t index;
	pthread_mutex_lock(&address_lock);
	for (index = 0u; index < gpu->identity_count; index++)
		if ( gpu->identity_ranges[index].base == base && gpu->identity_ranges[index].bytes == bytes )
		{
			gpu->identity_ranges[index] = gpu->identity_ranges[gpu->identity_count - 1u];
			gpu->identity_count--;
			break;
		}
	pthread_mutex_unlock(&address_lock);
}

static SparkStatus AddressRoom(SparkNvrmGpu *gpu, uint32_t needed, const char *owner)
{
	SparkNvrmRange *grown;
	uint32_t capacity = gpu->free_capacity != 0u ? gpu->free_capacity : ADDRESS_RANGES;
	if ( needed <= gpu->free_capacity )
		return(SPARK_STATUS_OK);
	while ( capacity < needed )
		capacity *= 2u;
	grown = (SparkNvrmRange *)realloc(gpu->free_ranges,(size_t)capacity * sizeof(*grown));
	if ( grown == 0 )
		ADDRESS_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,owner,"out of host memory for the device address map");
	gpu->free_ranges = grown;
	gpu->free_capacity = capacity;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmAddressOpen(SparkNvrmGpu *gpu, uint64_t base, uint64_t bytes, const char *owner)
{
	void *reserved = mmap((void *)(uintptr_t)base,(size_t)bytes,PROT_NONE,ADDRESS_FIRST_FLAGS,-1,0);
	SparkStatus status;
	if ( reserved != MAP_FAILED && reserved != (void *)(uintptr_t)base )
		munmap(reserved,(size_t)bytes);
	if ( reserved == MAP_FAILED || reserved != (void *)(uintptr_t)base )
	{
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"initialize",owner,errno,
			"cannot reserve the device address range in the process: another mapping holds part of it");
		SPARK_FAIL(status);
	}
	gpu->reserved_base = base;
	gpu->reserved_bytes = bytes;
	status = AddressRoom(gpu,ADDRESS_RANGES,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmAddressClose(gpu);
		SPARK_FAIL(status);
	}
	gpu->free_ranges[0].base = base;
	gpu->free_ranges[0].bytes = bytes;
	gpu->free_count = 1u;
	gpu->virtual_live = 0u;
	AddressKeepOut(gpu,0u,ADDRESS_SERVER_BASE,ADDRESS_SERVER_BYTES,"GSP's split address range");
	AddressKeepOut(gpu,1u,gpu->local_window,ADDRESS_WINDOW_BYTES,"the shader local-memory window");
	AddressKeepOut(gpu,2u,gpu->shared_window,ADDRESS_WINDOW_BYTES,"the shader shared-memory window");
	return(SPARK_STATUS_OK);
}

void SparkNvrmAddressClose(SparkNvrmGpu *gpu)
{
	uint32_t index;
	if ( gpu->reserved_bytes != 0u )
		munmap((void *)(uintptr_t)gpu->reserved_base,(size_t)gpu->reserved_bytes);
	for (index = 0u; index < SPARK_NVRM_KEEP_OUTS; index++)
		if ( gpu->keep_outs[index].reserved != 0u )
			munmap((void *)(uintptr_t)gpu->keep_outs[index].base,(size_t)gpu->keep_outs[index].bytes);
	memset(gpu->keep_outs,0,sizeof(gpu->keep_outs));
	free(gpu->identity_ranges);
	gpu->identity_ranges = 0;
	gpu->identity_count = 0u;
	gpu->identity_capacity = 0u;
	free(gpu->free_ranges);
	gpu->free_ranges = 0;
	gpu->free_count = 0u;
	gpu->free_capacity = 0u;
	gpu->virtual_live = 0u;
	gpu->reserved_base = 0u;
	gpu->reserved_bytes = 0u;
}

static void AddressInsert(SparkNvrmGpu *gpu, uint32_t index, uint64_t base, uint64_t bytes)
{
	memmove(gpu->free_ranges + index + 1u,gpu->free_ranges + index,(size_t)(gpu->free_count - index) * sizeof(gpu->free_ranges[0]));
	gpu->free_ranges[index].base = base;
	gpu->free_ranges[index].bytes = bytes;
	gpu->free_count++;
}

static void AddressRemove(SparkNvrmGpu *gpu, uint32_t index)
{
	memmove(gpu->free_ranges + index,gpu->free_ranges + index + 1u,(size_t)(gpu->free_count - index - 1u) * sizeof(gpu->free_ranges[0]));
	gpu->free_count--;
}

static void AddressCarve(SparkNvrmGpu *gpu, uint32_t index, uint64_t start, uint64_t bytes)
{
	SparkNvrmRange *range = &gpu->free_ranges[index];
	const uint64_t head = start - range->base,tail = range->base + range->bytes - start - bytes;
	if ( head != 0u && tail != 0u )
	{
		range->bytes = head;
		AddressInsert(gpu,index + 1u,start + bytes,tail);
	}
	else if ( head != 0u )
		range->bytes = head;
	else if ( tail != 0u )
	{
		range->base = start + bytes;
		range->bytes = tail;
	}
	else
		AddressRemove(gpu,index);
}

static SparkStatus AddressVirtual(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t align, uint64_t *address, const char *owner)
{
	SparkStatus room;
	uint32_t index;
	if ( bytes == 0u || align == 0u || (align & (align - 1u)) != 0u )
		ADDRESS_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,owner,"a device address range needs a size and a power-of-two alignment");
	room = AddressRoom(gpu,gpu->free_count + gpu->virtual_live + 2u,owner);
	if ( room != SPARK_STATUS_OK )
		SPARK_FAIL(room);
	for (index = 0u; index < gpu->free_count; index++)
	{
		const SparkNvrmRange *range = &gpu->free_ranges[index];
		const uint64_t start = (range->base + align - 1u) & ~(align - 1u);
		if ( start < range->base || start - range->base > range->bytes || range->bytes - (start - range->base) < bytes )
			continue;
		AddressCarve(gpu,index,start,bytes);
		gpu->virtual_live++;
		*address = start;
		return(SPARK_STATUS_OK);
	}
	ADDRESS_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,owner,"the device address range has no free span of that size");
}

SparkStatus SparkNvrmVirtual(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t align, uint64_t *address, const char *owner)
{
	SparkStatus status;
	pthread_mutex_lock(&address_lock);
	status = AddressVirtual(gpu,bytes,align,address,owner);
	pthread_mutex_unlock(&address_lock);
	return(status);
}

static void AddressRelease(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
{
	uint32_t index = 0u;
	SparkNvrmRange *before,*after;
	while ( index < gpu->free_count && gpu->free_ranges[index].base < address )
		index++;
	before = index != 0u ? &gpu->free_ranges[index - 1u] : 0;
	after = index < gpu->free_count ? &gpu->free_ranges[index] : 0;
	if ( gpu->virtual_live != 0u )
		gpu->virtual_live--;
	if ( before != 0 && before->base + before->bytes == address )
	{
		before->bytes += bytes;
		if ( after != 0 && address + bytes == after->base )
		{
			before->bytes += after->bytes;
			AddressRemove(gpu,index);
		}
	}
	else if ( after != 0 && address + bytes == after->base )
	{
		after->base = address;
		after->bytes += bytes;
	}
	else
		AddressInsert(gpu,index,address,bytes);
}

void SparkNvrmVirtualRelease(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
{
	pthread_mutex_lock(&address_lock);
	AddressRelease(gpu,address,bytes);
	pthread_mutex_unlock(&address_lock);
}

void SparkNvrmReserveCpu(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
{
	(void)gpu;
	(void)mmap((void *)(uintptr_t)address,(size_t)bytes,PROT_NONE,ADDRESS_RESERVE_FLAGS | MAP_FIXED,-1,0);
}
