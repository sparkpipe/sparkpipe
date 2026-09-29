#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "sparkpipe/spark_weightd_direct.h"
#include "sparkpipe/spark_error_site.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DIRECT_SLOT_EMPTY 0u
#define DIRECT_SLOT_FULL 1u

typedef struct DirectSlot
{
	uint8_t *buffer;
	uint64_t block;
	uint64_t length;
	uint32_t state;
	SparkStatus status;
} DirectSlot;

typedef struct DirectBlock
{
	uint64_t start;
	uint64_t end;
} DirectBlock;

struct SparkWeightdDirect
{
	uint64_t block_bytes;
	uint64_t capacity;
	uint32_t readers;
	uint32_t slot_count;
	DirectSlot slots[2u * SPARK_WEIGHTD_DIRECT_READERS_MAX];
};

typedef struct DirectRun
{
	SparkWeightdDirect *direct;
	const DirectBlock *blocks;
	uint64_t block_count;
	int32_t fd;
	uint32_t is_direct;
	uint32_t abort;
	pthread_mutex_t lock;
	pthread_cond_t changed;
} DirectRun;

typedef struct DirectReader
{
	DirectRun *run;
	uint32_t index;
} DirectReader;

static uint64_t direct_now_ns(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

SparkStatus SparkWeightdDirectCreate(uint64_t block_bytes,uint32_t readers,SparkWeightdDirect **out)
{
	SparkWeightdDirect *direct;
	uint32_t index;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( readers == 0u || readers > SPARK_WEIGHTD_DIRECT_READERS_MAX || block_bytes < SPARK_WEIGHTD_DIRECT_ALIGNMENT || (block_bytes % SPARK_WEIGHTD_DIRECT_ALIGNMENT) != 0u || block_bytes > (UINT64_C(1) << 32) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	direct = calloc(1u,sizeof(*direct));
	if ( direct == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	direct->block_bytes = block_bytes;
	direct->capacity = block_bytes + 2u * SPARK_WEIGHTD_DIRECT_ALIGNMENT;
	direct->readers = readers;
	direct->slot_count = 2u * readers;
	for (index = 0u; index < direct->slot_count; index++)
	{
		void *buffer = 0;
		if ( posix_memalign(&buffer,(size_t)SPARK_WEIGHTD_DIRECT_ALIGNMENT,(size_t)direct->capacity) != 0 )
		{
			SparkWeightdDirectDestroy(direct);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		direct->slots[index].buffer = buffer;
	}
	*out = direct;
	return(SPARK_STATUS_OK);
}

void SparkWeightdDirectDestroy(SparkWeightdDirect *direct)
{
	uint32_t index;
	if ( direct == 0 )
		return;
	for (index = 0u; index < 2u * SPARK_WEIGHTD_DIRECT_READERS_MAX; index++)
		free(direct->slots[index].buffer);
	free(direct);
}

int32_t SparkWeightdDirectOpen(const char *path,uint32_t *is_direct)
{
	int32_t fd = -1;
	if ( is_direct != 0 )
		*is_direct = 0u;
	if ( path == 0 )
		return(-1);
#ifdef O_DIRECT
	fd = open(path,O_RDONLY | O_DIRECT | O_CLOEXEC);
	if ( fd >= 0 )
	{
		if ( is_direct != 0 )
			*is_direct = 1u;
		return(fd);
	}
	if ( errno != EINVAL )
		return(-1);
#endif
	fd = open(path,O_RDONLY | O_CLOEXEC);
	return(fd);
}

int32_t SparkWeightdDirectReopen(int32_t fd,uint32_t *is_direct)
{
	if ( is_direct != 0 )
		*is_direct = 0u;
	if ( fd < 0 )
		return(-1);
#ifdef O_DIRECT
	{
		char path[64];
		struct stat before,after;
		int32_t reopened;
		(void)snprintf(path,sizeof(path),"/proc/self/fd/%d",fd);
		reopened = open(path,O_RDONLY | O_DIRECT | O_CLOEXEC);
		if ( reopened >= 0 )
		{
			if ( fstat(fd,&before) == 0 && fstat(reopened,&after) == 0 && before.st_dev == after.st_dev && before.st_ino == after.st_ino )
			{
				if ( is_direct != 0 )
					*is_direct = 1u;
				return(reopened);
			}
			(void)close(reopened);
		}
	}
#endif
	return(dup(fd));
}

static SparkStatus direct_plan(const SparkWeightdDirectSpan *spans,uint32_t span_count,uint64_t block_bytes,DirectBlock **blocks_out,uint64_t *count_out)
{
	DirectBlock *blocks;
	uint64_t capacity = span_count,total = 0u,count = 0u,position = 0u,start,end,span_end;
	uint32_t index;
	for (index = 0u; index < span_count; index++)
	{
		if ( spans[index].bytes == 0u || spans[index].offset > UINT64_MAX - spans[index].bytes )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( index > 0u && spans[index].offset < spans[index - 1u].offset + spans[index - 1u].bytes )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		total += spans[index].bytes;
	}
	capacity += total / block_bytes + 1u;
	blocks = calloc((size_t)capacity,sizeof(*blocks));
	if ( blocks == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	index = 0u;
	while ( index < span_count )
	{
		start = spans[index].offset > position ? spans[index].offset : position;
		span_end = spans[index].offset + spans[index].bytes;
		end = (span_end - start) > block_bytes ? start + block_bytes : span_end;
		if ( end == span_end )
		{
			index++;
			while ( index < span_count && spans[index].offset - end <= SPARK_WEIGHTD_DIRECT_GAP_BYTES && spans[index].offset + spans[index].bytes - start <= block_bytes )
			{
				end = spans[index].offset + spans[index].bytes;
				index++;
			}
		}
		if ( count >= capacity )
		{
			free(blocks);
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		blocks[count].start = start;
		blocks[count].end = end;
		count++;
		position = end;
	}
	*blocks_out = blocks;
	*count_out = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus direct_read_block(DirectRun *run,const DirectBlock *block,DirectSlot *slot)
{
	uint64_t mask = SPARK_WEIGHTD_DIRECT_ALIGNMENT - 1u;
	uint64_t first = run->is_direct != 0u ? (block->start & ~mask) : block->start;
	uint64_t last = run->is_direct != 0u ? ((block->end + mask) & ~mask) : block->end;
	uint64_t want = last - first,got = 0u;
	ssize_t moved;
	if ( want > run->direct->capacity )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	while ( got < want )
	{
		moved = pread(run->fd,slot->buffer + got,(size_t)(want - got),(off_t)(first + got));
		if ( moved < 0 && errno == EINTR )
			continue;
		if ( moved < 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( moved == 0 )
			break;
		got += (uint64_t)moved;
		if ( run->is_direct != 0u && ((uint64_t)moved & mask) != 0u )
			break;
	}
	if ( got < block->end - first )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	slot->length = got;
	return(SPARK_STATUS_OK);
}

static void *direct_reader_main(void *argument)
{
	DirectReader *reader = argument;
	DirectRun *run = reader->run;
	DirectSlot *slot;
	SparkStatus status;
	uint64_t block;
	for (block = reader->index; block < run->block_count; block += run->direct->readers)
	{
		slot = &run->direct->slots[block % run->direct->slot_count];
		pthread_mutex_lock(&run->lock);
		while ( slot->state != DIRECT_SLOT_EMPTY && run->abort == 0u )
			pthread_cond_wait(&run->changed,&run->lock);
		if ( run->abort != 0u )
		{
			pthread_mutex_unlock(&run->lock);
			break;
		}
		pthread_mutex_unlock(&run->lock);
		status = direct_read_block(run,&run->blocks[block],slot);
		pthread_mutex_lock(&run->lock);
		slot->status = status;
		slot->block = block;
		slot->state = DIRECT_SLOT_FULL;
		pthread_cond_broadcast(&run->changed);
		pthread_mutex_unlock(&run->lock);
		if ( status != SPARK_STATUS_OK )
			break;
	}
	return(0);
}

static SparkStatus direct_deliver(const SparkWeightdDirectSpan *spans,uint32_t span_count,uint32_t *cursor,const DirectBlock *block,const uint8_t *data,SparkWeightdDirectSink sink,void *context)
{
	uint32_t index;
	uint64_t span_end,piece_start,piece_end;
	SparkStatus status;
	for (index = *cursor; index < span_count; index++)
	{
		if ( spans[index].offset >= block->end )
			break;
		span_end = spans[index].offset + spans[index].bytes;
		piece_start = spans[index].offset > block->start ? spans[index].offset : block->start;
		piece_end = span_end < block->end ? span_end : block->end;
		if ( piece_end > piece_start )
		{
			status = sink(context,index,piece_start - spans[index].offset,data + (piece_start - block->start),piece_end - piece_start);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		if ( span_end > block->end )
			break;
		*cursor = index + 1u;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdDirectStream(SparkWeightdDirect *direct,int32_t fd,uint32_t is_direct,const SparkWeightdDirectSpan *spans,uint32_t span_count,SparkWeightdDirectSink sink,void *context,SparkWeightdDirectStats *stats)
{
	DirectRun run;
	DirectReader readers[SPARK_WEIGHTD_DIRECT_READERS_MAX];
	pthread_t threads[SPARK_WEIGHTD_DIRECT_READERS_MAX];
	DirectBlock *blocks = 0;
	DirectSlot *slot;
	uint64_t block_count = 0u,block,started_ns,sink_ns = 0u,bytes_read = 0u,mark;
	uint32_t index,launched = 0u,cursor = 0u;
	SparkStatus status;
	if ( direct == 0 || fd < 0 || sink == 0 || (spans == 0 && span_count != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	started_ns = direct_now_ns();
	status = direct_plan(spans,span_count,direct->block_bytes,&blocks,&block_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&run,0,sizeof(run));
	run.direct = direct;
	run.blocks = blocks;
	run.block_count = block_count;
	run.fd = fd;
	run.is_direct = is_direct;
	pthread_mutex_init(&run.lock,0);
	pthread_cond_init(&run.changed,0);
	for (index = 0u; index < direct->slot_count; index++)
	{
		direct->slots[index].state = DIRECT_SLOT_EMPTY;
		direct->slots[index].status = SPARK_STATUS_OK;
	}
	for (index = 0u; index < direct->readers && index < block_count; index++)
	{
		readers[index].run = &run;
		readers[index].index = index;
		if ( pthread_create(&threads[index],0,direct_reader_main,&readers[index]) != 0 )
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
			break;
		}
		launched++;
	}
	for (block = 0u; status == SPARK_STATUS_OK && block < block_count; block++)
	{
		slot = &direct->slots[block % direct->slot_count];
		pthread_mutex_lock(&run.lock);
		while ( !(slot->state == DIRECT_SLOT_FULL && slot->block == block) )
			pthread_cond_wait(&run.changed,&run.lock);
		pthread_mutex_unlock(&run.lock);
		status = slot->status;
		if ( status == SPARK_STATUS_OK )
		{
			uint64_t first = is_direct != 0u ? (blocks[block].start & ~(SPARK_WEIGHTD_DIRECT_ALIGNMENT - 1u)) : blocks[block].start;
			mark = direct_now_ns();
			status = direct_deliver(spans,span_count,&cursor,&blocks[block],slot->buffer + (blocks[block].start - first),sink,context);
			sink_ns += direct_now_ns() - mark;
			bytes_read += slot->length;
		}
		pthread_mutex_lock(&run.lock);
		slot->state = DIRECT_SLOT_EMPTY;
		if ( status != SPARK_STATUS_OK )
			run.abort = 1u;
		pthread_cond_broadcast(&run.changed);
		pthread_mutex_unlock(&run.lock);
	}
	pthread_mutex_lock(&run.lock);
	run.abort = 1u;
	pthread_cond_broadcast(&run.changed);
	pthread_mutex_unlock(&run.lock);
	for (index = 0u; index < launched; index++)
		pthread_join(threads[index],0);
	pthread_cond_destroy(&run.changed);
	pthread_mutex_destroy(&run.lock);
	free(blocks);
	if ( stats != 0 )
	{
		stats->bytes_read = bytes_read;
		stats->blocks = block_count;
		stats->wall_ns = direct_now_ns() - started_ns;
		stats->sink_ns = sink_ns;
		stats->direct = is_direct;
		stats->readers = direct->readers;
	}
	SPARK_RETURN(status);
}
