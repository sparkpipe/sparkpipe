#include <errno.h>
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
	return(SPARK_STATUS_OK);
}

void SparkNvrmAddressClose(SparkNvrmGpu *gpu)
{
	if ( gpu->reserved_bytes != 0u )
		munmap((void *)(uintptr_t)gpu->reserved_base,(size_t)gpu->reserved_bytes);
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

SparkStatus SparkNvrmVirtual(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t align, uint64_t *address, const char *owner)
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

void SparkNvrmVirtualRelease(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
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

void SparkNvrmReserveCpu(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
{
	(void)gpu;
	(void)mmap((void *)(uintptr_t)address,(size_t)bytes,PROT_NONE,ADDRESS_RESERVE_FLAGS | MAP_FIXED,-1,0);
}
