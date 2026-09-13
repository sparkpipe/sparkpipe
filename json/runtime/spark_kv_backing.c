#include "sparkpipe/spark_kv_backing.h"

#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define SPARK_KV_BACKING_TAG "kv_backing"

typedef struct SparkKvBackingHeader
{
	char magic[8];
	uint64_t slot_bytes;
	uint32_t slot_count;
	uint32_t free_hint;
	uint32_t reserved[8];
} SparkKvBackingHeader;

static SparkStatus spark_kv_backing_write_header(SparkKvBacking *backing)
{
	SparkKvBackingHeader header;
	ssize_t written;
	memset(&header,0,sizeof(header));
	memcpy(header.magic,SPARK_KV_BACKING_MAGIC,sizeof(header.magic));
	header.slot_bytes = backing->slot_bytes;
	header.slot_count = backing->slot_count;
	header.free_hint = backing->free_hint;
	written = pwrite(backing->file_descriptor,&header,sizeof(header),0);
	if ( written != (ssize_t)sizeof(header) )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvBackingOpen(const SparkKvBackingConfiguration *configuration,
	SparkKvBacking *backing)
{
	SparkKvBackingHeader header;
	uint64_t capacity_bytes;
	uint32_t index;
	int descriptor;
	int enabled = 1;
	if ( configuration == 0 || backing == 0 || configuration->path == 0 ||
		configuration->path[0] == '\0' || configuration->maximum_bytes <
		SPARK_KV_BACKING_SLOT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(backing,0,sizeof(*backing));
	backing->slot_bytes = SPARK_KV_BACKING_SLOT_BYTES;
	backing->slot_count = (uint32_t)(configuration->maximum_bytes /
		backing->slot_bytes);
	if ( backing->slot_count == 0u || backing->slot_count >
		SPARK_KV_BACKING_MAX_SLOT_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	descriptor = open(configuration->path,O_RDWR | O_CREAT,0644);
	if ( descriptor < 0 )
		return(SPARK_STATUS_IO_ERROR);
	(void)posix_fadvise(descriptor,0,0,POSIX_FADV_RANDOM);
	if ( ftruncate(descriptor,SPARK_KV_BACKING_HEADER_BYTES +
		(uint64_t)backing->slot_count * backing->slot_bytes) != 0 )
	{
		close(descriptor);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	backing->file_descriptor = descriptor;
	backing->slot_live = (uint8_t *)calloc((size_t)backing->slot_count,1u);
	if ( backing->slot_live == 0 )
	{
		close(descriptor);
		backing->file_descriptor = -1;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	/* fresh files carry a header; an existing file with a mismatched
	 * geometry is a configuration error (refuse, never mis-map slots) */
	if ( pread(descriptor,&header,sizeof(header),0) == (ssize_t)sizeof(header) )
	{
		/* a freshly created sparse file reads back all-zero (the hole);
		 * only a NONZERO wrong geometry is a mis-map risk */
		uint32_t nonzero = 0u,i;
		for ( i = 0u; i < sizeof(header.magic); i++ )
			nonzero |= (uint32_t)(unsigned char)header.magic[i];
		if ( nonzero != 0u && (memcmp(header.magic,SPARK_KV_BACKING_MAGIC,
			sizeof(header.magic)) != 0 || header.slot_bytes != backing->slot_bytes) )
		{
			free(backing->slot_live);
			backing->slot_live = 0;
			close(descriptor);
			backing->file_descriptor = -1;
			return(SPARK_STATUS_TARGET_MISMATCH);
		}
	}
	backing->free_hint = 0u;
	(void)enabled;
	if ( spark_kv_backing_write_header(backing) != SPARK_STATUS_OK )
	{
		free(backing->slot_live);
		backing->slot_live = 0;
		close(descriptor);
		backing->file_descriptor = -1;
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)index;
	(void)capacity_bytes;
	return(SPARK_STATUS_OK);
}

void SparkKvBackingClose(SparkKvBacking *backing)
{
	if ( backing == 0 )
		return;
	if ( backing->file_descriptor >= 0 )
		close(backing->file_descriptor);
	free(backing->slot_live);
	memset(backing,0,sizeof(*backing));
}

int64_t SparkKvBackingAllocate(SparkKvBacking *backing)
{
	uint32_t slot;
	if ( backing == 0 || backing->slot_live == 0 )
		return(-1);
	for ( slot = 0u; slot < backing->slot_count; slot++ )
	{
		uint32_t candidate = (backing->free_hint + slot) % backing->slot_count;
		if ( backing->slot_live[candidate] == 0u )
		{
			backing->slot_live[candidate] = 1u;
			backing->live_count++;
			backing->free_hint = (candidate + 1u) % backing->slot_count;
			return((int64_t)candidate);
		}
	}
	return(-1); /* horizon full: backpressure, never thrash */
}

void SparkKvBackingRelease(SparkKvBacking *backing, uint32_t slot)
{
	if ( backing == 0 || backing->slot_live == 0 || slot >= backing->slot_count )
		return;
	if ( backing->slot_live[slot] != 0u )
	{
		backing->slot_live[slot] = 0u;
		backing->live_count--;
	}
}

static SparkStatus spark_kv_backing_transfer(SparkKvBacking *backing, uint32_t slot,
	void *buffer, int reading)
{
	uint64_t offset;
	size_t remaining = (size_t)backing->slot_bytes;
	uint8_t *cursor = (uint8_t *)buffer;
	if ( backing == 0 || backing->file_descriptor < 0 || buffer == 0 ||
		slot >= backing->slot_count || backing->slot_live[slot] == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	offset = SPARK_KV_BACKING_HEADER_BYTES + (uint64_t)slot * backing->slot_bytes;
	while ( remaining > 0u )
	{
		size_t chunk = remaining > (64u * 1024u * 1024u) ?
			(64u * 1024u * 1024u) : remaining;
		ssize_t moved = reading ?
			pread(backing->file_descriptor,cursor,chunk,(off_t)offset) :
			pwrite(backing->file_descriptor,cursor,chunk,(off_t)offset);
		if ( moved <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		cursor += (size_t)moved;
		offset += (uint64_t)moved;
		remaining -= (size_t)moved;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvBackingWriteBlock(SparkKvBacking *backing, uint32_t slot,
	const void *host_buffer)
{
	return(spark_kv_backing_transfer(backing,slot,(void *)(size_t)host_buffer,0));
}

SparkStatus SparkKvBackingReadBlock(SparkKvBacking *backing, uint32_t slot,
	void *host_buffer)
{
	return(spark_kv_backing_transfer(backing,slot,host_buffer,1));
}

uint64_t SparkKvBackingLiveBytes(const SparkKvBacking *backing)
{
	if ( backing == 0 )
		return(0ull);
	return((uint64_t)backing->live_count * backing->slot_bytes);
}
