#include "sparkpipe/spark_weightd_direct.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct Check
{
	const uint8_t *file;
	const SparkWeightdDirectSpan *spans;
	uint32_t span_count;
	uint64_t *received;
	uint32_t last_span;
	uint32_t fail_at_call;
	uint32_t calls;
} Check;

static uint64_t next_random(uint64_t *state)
{
	*state ^= *state << 13;
	*state ^= *state >> 7;
	*state ^= *state << 17;
	return(*state);
}

static SparkStatus check_sink(void *context,uint32_t span_index,uint64_t span_offset,const uint8_t *data,uint64_t bytes)
{
	Check *check = context;
	check->calls++;
	if ( check->fail_at_call != 0u && check->calls == check->fail_at_call )
		return(SPARK_STATUS_HASH_MISMATCH);
	assert(span_index < check->span_count);
	assert(span_index >= check->last_span);
	check->last_span = span_index;
	assert(span_offset == check->received[span_index]);
	assert(bytes > 0u && span_offset + bytes <= check->spans[span_index].bytes);
	assert(memcmp(data,check->file + check->spans[span_index].offset + span_offset,(size_t)bytes) == 0);
	check->received[span_index] += bytes;
	return(SPARK_STATUS_OK);
}

static uint32_t random_spans(uint64_t *state,uint64_t file_bytes,SparkWeightdDirectSpan *spans,uint32_t capacity)
{
	uint64_t cursor = next_random(state) % 9000u;
	uint32_t count = 0u;
	while ( count < capacity )
	{
		uint64_t bytes = 1u + next_random(state) % ((next_random(state) & 3u) == 0u ? 3000000u : 70000u);
		if ( cursor + bytes > file_bytes )
			break;
		spans[count].offset = cursor;
		spans[count].bytes = bytes;
		count++;
		cursor += bytes;
		if ( (next_random(state) & 1u) != 0u )
			cursor += next_random(state) % ((next_random(state) & 7u) == 0u ? 2500000u : 5000u);
	}
	return(count);
}

int main(void)
{
	char path[] = "/var/tmp/sparkpipe-direct-XXXXXX";
	uint64_t file_bytes = UINT64_C(23) * 1024u * 1024u + 1237u,state = UINT64_C(0x9e3779b97f4a7c15),index;
	uint8_t *file = malloc((size_t)file_bytes);
	SparkWeightdDirectSpan spans[512];
	uint64_t received[512];
	int32_t fd = mkstemp(path);
	uint32_t trial;
	assert(file != 0 && fd >= 0);
	for (index = 0u; index < file_bytes; index++)
		file[index] = (uint8_t)(next_random(&state) >> 24);
	assert(write(fd,file,(size_t)file_bytes) == (ssize_t)file_bytes);
	assert(fsync(fd) == 0);
	close(fd);
	{
		SparkWeightdDirect *direct = 0;
		assert(SparkWeightdDirectCreate(1000u,1u,&direct) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(SparkWeightdDirectCreate(4096u,0u,&direct) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(SparkWeightdDirectCreate(4096u,SPARK_WEIGHTD_DIRECT_READERS_MAX + 1u,&direct) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	for (trial = 0u; trial < 96u; trial++)
	{
		SparkWeightdDirect *direct = 0;
		SparkWeightdDirectStats stats;
		Check check;
		uint32_t is_direct = 0u,count,readers = 1u + trial % 4u;
		uint64_t block_bytes = SPARK_WEIGHTD_DIRECT_ALIGNMENT * (1u + next_random(&state) % 1024u);
		int32_t stream_fd = trial % 3u == 2u ? open(path,O_RDONLY) : SparkWeightdDirectOpen(path,&is_direct);
		assert(stream_fd >= 0);
		count = random_spans(&state,file_bytes,spans,trial % 7u == 0u ? 1u : 512u);
		memset(received,0,sizeof(received));
		memset(&check,0,sizeof(check));
		check.file = file;
		check.spans = spans;
		check.span_count = count;
		check.received = received;
		check.fail_at_call = trial % 11u == 5u ? 1u + (uint32_t)(next_random(&state) % (count + 1u)) : 0u;
		assert(SparkWeightdDirectCreate(block_bytes,readers,&direct) == SPARK_STATUS_OK);
		{
			SparkStatus status = SparkWeightdDirectStream(direct,stream_fd,is_direct,spans,count,check_sink,&check,&stats);
			if ( check.fail_at_call != 0u && check.fail_at_call <= check.calls )
				assert(status == SPARK_STATUS_HASH_MISMATCH);
			else
			{
				assert(status == SPARK_STATUS_OK);
				for (index = 0u; index < count; index++)
					assert(received[index] == spans[index].bytes);
				assert(stats.direct == is_direct && stats.readers == readers);
			}
		}
		SparkWeightdDirectDestroy(direct);
		close(stream_fd);
	}
	{
		SparkWeightdDirect *direct = 0;
		SparkWeightdDirectSpan bad[2] = {{100u,50u},{120u,10u}};
		SparkWeightdDirectSpan past[1] = {{0u,0u}};
		Check check;
		uint32_t is_direct = 0u;
		int32_t stream_fd = SparkWeightdDirectOpen(path,&is_direct);
		memset(&check,0,sizeof(check));
		assert(SparkWeightdDirectCreate(65536u,2u,&direct) == SPARK_STATUS_OK);
		assert(SparkWeightdDirectStream(direct,stream_fd,is_direct,bad,2u,check_sink,&check,0) == SPARK_STATUS_INVALID_ARGUMENT);
		past[0].offset = file_bytes - 10u;
		past[0].bytes = 20u;
		memset(received,0,sizeof(received));
		check.file = file;
		check.spans = past;
		check.span_count = 1u;
		check.received = received;
		assert(SparkWeightdDirectStream(direct,stream_fd,is_direct,past,1u,check_sink,&check,0) == SPARK_STATUS_IO_ERROR);
		assert(check.calls == 0u);
		SparkWeightdDirectDestroy(direct);
		close(stream_fd);
	}
	unlink(path);
	free(file);
	printf("test_weightd_direct: 96 streams delivered every span byte in order\n");
	return(0);
}
