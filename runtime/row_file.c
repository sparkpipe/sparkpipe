#include "sparkpipe/spark_row_file.h"
#include "sparkpipe/spark_error_site.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SPARK_ROW_FILE_BATCH_SPANS 16u
#define SPARK_ROW_FILE_SPAN_BYTES 65536u
#define SPARK_ROW_FILE_RADIX_BITS 11u

typedef struct SparkRowFileMiss
{
	uint64_t index;
	uint64_t row;
} SparkRowFileMiss;

typedef struct SparkRowFileSpan
{
	uint64_t first;
	uint64_t bytes;
	uint64_t miss_begin;
	uint64_t miss_end;
} SparkRowFileSpan;

struct SparkRowFile
{
	int descriptor;
	uint64_t offset, rows, file_bytes;
	uint32_t row_bytes, cache_rows, workers, started;
	uint64_t *tags;
	uint8_t *cache;
	pthread_t threads[SPARK_ROW_FILE_MAX_WORKERS];
	uint8_t *blocks[SPARK_ROW_FILE_MAX_WORKERS];
	pthread_mutex_t fetch_lock, job_lock;
	pthread_cond_t job_ready, job_done;
	SparkRowFileMiss *misses, *sorted;
	uint64_t miss_capacity;
	uint64_t *radix_counts;
	SparkRowFileSpan *spans;
	uint64_t span_capacity, span_count, span_next, span_left;
	uint8_t *out;
	uint64_t out_stride;
	uint32_t stop, failed;
	SparkRowFileCounters counters;
};

typedef struct SparkRowFileWorker
{
	SparkRowFile *file;
	uint32_t index;
} SparkRowFileWorker;

static int32_t SparkRowFileReadSpan(SparkRowFile *file, uint8_t *block, const SparkRowFileSpan *span, uint8_t *out, uint64_t stride)
{
	uint64_t done = 0u;
	if ( span->first + span->bytes > file->file_bytes &&
		file->offset + (file->sorted[span->miss_end - 1u].row + 1u) * file->row_bytes > file->file_bytes )
		return(-1);
	while ( done < span->bytes )
	{
		const ssize_t got = pread(file->descriptor, block + done, (size_t)(span->bytes - done), (off_t)(span->first + done));
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got < 0 )
			return(-2);
		if ( got == 0 )
			break;
		done += (uint64_t)got;
	}
	for ( uint64_t at = span->miss_begin; at < span->miss_end; ++at )
	{
		const SparkRowFileMiss *miss = &file->sorted[at];
		const uint64_t start = file->offset + miss->row * file->row_bytes - span->first;
		if ( start + file->row_bytes > done )
			return(-3);
		memcpy(out + miss->index * stride, block + start, file->row_bytes);
	}
	return(0);
}

static void *SparkRowFileWork(void *argument)
{
	SparkRowFileWorker *worker = (SparkRowFileWorker *)argument;
	SparkRowFile *file = worker->file;
	uint8_t *block = file->blocks[worker->index];
	free(worker);
	pthread_mutex_lock(&file->job_lock);
	for ( ;; )
	{
		while ( file->stop == 0u && file->span_next >= file->span_count )
			pthread_cond_wait(&file->job_ready, &file->job_lock);
		if ( file->stop != 0u )
			break;
		const SparkRowFileSpan *spans = file->spans + file->span_next;
		uint8_t *out = file->out;
		const uint64_t stride = file->out_stride;
		uint64_t take = (file->span_count - file->span_next) / (4u * file->workers);
		uint32_t failed = 0u;
		take = take == 0u ? 1u : take > SPARK_ROW_FILE_BATCH_SPANS ? SPARK_ROW_FILE_BATCH_SPANS : take;
		file->span_next += take;
		pthread_mutex_unlock(&file->job_lock);
		for ( uint64_t index = 0u; index < take; ++index )
			if ( SparkRowFileReadSpan(file, block, &spans[index], out, stride) != 0 )
				failed = 1u;
		pthread_mutex_lock(&file->job_lock);
		if ( failed != 0u )
			file->failed = 1u;
		file->span_left -= take;
		if ( file->span_left == 0u )
			pthread_cond_signal(&file->job_done);
	}
	pthread_mutex_unlock(&file->job_lock);
	return(0);
}

static void SparkRowFileSortMisses(SparkRowFile *file, uint64_t count)
{
	const uint64_t buckets = 1ull << SPARK_ROW_FILE_RADIX_BITS;
	uint32_t bits = 1u;
	SparkRowFileMiss *from = file->misses, *to = file->sorted;
	while ( bits < 64u && (file->rows - 1u) >> bits != 0u )
		++bits;
	for ( uint32_t shift = 0u; shift < bits; shift += SPARK_ROW_FILE_RADIX_BITS )
	{
		uint64_t total = 0u;
		memset(file->radix_counts, 0, (size_t)buckets * sizeof(uint64_t));
		for ( uint64_t at = 0u; at < count; ++at )
			++file->radix_counts[(from[at].row >> shift) & (buckets - 1u)];
		for ( uint64_t bucket = 0u; bucket < buckets; ++bucket )
		{
			const uint64_t here = file->radix_counts[bucket];
			file->radix_counts[bucket] = total;
			total += here;
		}
		for ( uint64_t at = 0u; at < count; ++at )
			to[file->radix_counts[(from[at].row >> shift) & (buckets - 1u)]++] = from[at];
		SparkRowFileMiss *swap = from;
		from = to;
		to = swap;
	}
	if ( from != file->sorted )
		memcpy(file->sorted, from, (size_t)count * sizeof(*from));
}

static uint64_t SparkRowFilePlanSpans(SparkRowFile *file, uint64_t count)
{
	uint64_t spans = 0u;
	for ( uint64_t at = 0u; at < count; ++at )
	{
		const uint64_t begin = file->offset + file->sorted[at].row * file->row_bytes, end = begin + file->row_bytes;
		const uint64_t first = begin - begin % SPARK_ROW_FILE_BLOCK_BYTES;
		const uint64_t last = (end + SPARK_ROW_FILE_BLOCK_BYTES - 1u) / SPARK_ROW_FILE_BLOCK_BYTES * SPARK_ROW_FILE_BLOCK_BYTES;
		SparkRowFileSpan *span = spans != 0u ? &file->spans[spans - 1u] : 0;
		if ( span != 0 && first < span->first + span->bytes )
		{
			if ( last > span->first + span->bytes )
				span->bytes = last - span->first;
			span->miss_end = at + 1u;
			continue;
		}
		if ( span != 0 && first == span->first + span->bytes && last - span->first <= SPARK_ROW_FILE_SPAN_BYTES )
		{
			span->bytes = last - span->first;
			span->miss_end = at + 1u;
			continue;
		}
		file->spans[spans].first = first;
		file->spans[spans].bytes = last - first;
		file->spans[spans].miss_begin = at;
		file->spans[spans].miss_end = at + 1u;
		++spans;
	}
	return(spans);
}

void SparkRowFileClose(SparkRowFile *file)
{
	if ( file == 0 )
		return;
	pthread_mutex_lock(&file->job_lock);
	file->stop = 1u;
	pthread_cond_broadcast(&file->job_ready);
	pthread_mutex_unlock(&file->job_lock);
	for ( uint32_t index = 0u; index < file->started; ++index )
		pthread_join(file->threads[index], 0);
	for ( uint32_t index = 0u; index < file->workers; ++index )
		free(file->blocks[index]);
	if ( file->descriptor >= 0 )
		close(file->descriptor);
	pthread_cond_destroy(&file->job_done);
	pthread_cond_destroy(&file->job_ready);
	pthread_mutex_destroy(&file->job_lock);
	pthread_mutex_destroy(&file->fetch_lock);
	free(file->misses);
	free(file->sorted);
	free(file->spans);
	free(file->radix_counts);
	free(file->cache);
	free(file->tags);
	free(file);
}

SparkStatus SparkRowFileOpen(const char *path, const SparkRowFileOptions *options, SparkRowFile **out)
{
	SparkRowFile *file;
	struct stat status;
	if ( path == 0 || options == 0 || out == 0 || options->rows == 0u || options->row_bytes == 0u ||
		options->row_bytes > SPARK_ROW_FILE_BLOCK_BYTES || options->workers == 0u || options->workers > SPARK_ROW_FILE_MAX_WORKERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	file = (SparkRowFile *)calloc(1u, sizeof(*file));
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	pthread_mutex_init(&file->fetch_lock, 0);
	pthread_mutex_init(&file->job_lock, 0);
	pthread_cond_init(&file->job_ready, 0);
	pthread_cond_init(&file->job_done, 0);
	file->offset = options->offset;
	file->rows = options->rows;
	file->row_bytes = options->row_bytes;
	file->cache_rows = options->cache_rows;
	file->workers = options->workers;
#if defined(O_DIRECT)
	file->descriptor = open(path, O_RDONLY | O_CLOEXEC | (options->buffered != 0u ? 0 : O_DIRECT));
#else
	file->descriptor = open(path, O_RDONLY | O_CLOEXEC);
#if defined(F_NOCACHE)
	if ( file->descriptor >= 0 && options->buffered == 0u )
		(void)fcntl(file->descriptor, F_NOCACHE, 1);
#endif
#endif
	if ( file->descriptor < 0 || fstat(file->descriptor, &status) != 0 )
	{
		SparkRowFileClose(file);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	file->file_bytes = (uint64_t)status.st_size;
	if ( file->offset > file->file_bytes || file->rows > (file->file_bytes - file->offset) / file->row_bytes )
	{
		SparkRowFileClose(file);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( file->cache_rows != 0u )
	{
		file->tags = (uint64_t *)calloc(file->cache_rows, sizeof(uint64_t));
		file->cache = (uint8_t *)malloc((size_t)file->cache_rows * file->row_bytes);
		if ( file->tags == 0 || file->cache == 0 )
		{
			SparkRowFileClose(file);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
	}
	file->radix_counts = (uint64_t *)malloc(((size_t)1u << SPARK_ROW_FILE_RADIX_BITS) * sizeof(uint64_t));
	if ( file->radix_counts == 0 )
	{
		SparkRowFileClose(file);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for ( uint32_t index = 0u; index < file->workers; ++index )
		if ( posix_memalign((void **)&file->blocks[index], SPARK_ROW_FILE_BLOCK_BYTES, SPARK_ROW_FILE_SPAN_BYTES + SPARK_ROW_FILE_BLOCK_BYTES) != 0 )
		{
			file->blocks[index] = 0;
			SparkRowFileClose(file);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
	for ( uint32_t index = 0u; index < file->workers; ++index )
	{
		SparkRowFileWorker *worker = (SparkRowFileWorker *)malloc(sizeof(*worker));
		if ( worker == 0 )
		{
			SparkRowFileClose(file);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		worker->file = file;
		worker->index = index;
		if ( pthread_create(&file->threads[index], 0, SparkRowFileWork, worker) != 0 )
		{
			free(worker);
			SparkRowFileClose(file);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		file->started = index + 1u;
	}
	*out = file;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkRowFileFetch(SparkRowFile *file, const int64_t *ids, uint64_t count, int64_t first_id, uint8_t *out, uint64_t out_stride)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( file == 0 || (count != 0u && (ids == 0 || out == 0)) || out_stride < file->row_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	pthread_mutex_lock(&file->fetch_lock);
	if ( count > file->miss_capacity )
	{
		SparkRowFileMiss *grown = (SparkRowFileMiss *)realloc(file->misses, (size_t)count * sizeof(*grown));
		SparkRowFileMiss *grown_sorted = grown != 0 ? (SparkRowFileMiss *)realloc(file->sorted, (size_t)count * sizeof(*grown_sorted)) : 0;
		SparkRowFileSpan *grown_spans = grown_sorted != 0 ? (SparkRowFileSpan *)realloc(file->spans, (size_t)count * sizeof(*grown_spans)) : 0;
		if ( grown != 0 )
			file->misses = grown;
		if ( grown_sorted != 0 )
			file->sorted = grown_sorted;
		if ( grown_spans != 0 )
			file->spans = grown_spans;
		if ( grown == 0 || grown_sorted == 0 || grown_spans == 0 )
		{
			pthread_mutex_unlock(&file->fetch_lock);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		file->miss_capacity = count;
		file->span_capacity = count;
	}
	uint64_t missing = 0u, wanted = 0u;
	for ( uint64_t index = 0u; index < count; ++index )
	{
		if ( ids[index] < first_id || (uint64_t)(ids[index] - first_id) >= file->rows )
			continue;
		const uint64_t row = (uint64_t)(ids[index] - first_id);
		++wanted;
		if ( file->cache_rows != 0u && file->tags[row % file->cache_rows] == row + 1u )
		{
			memcpy(out + index * out_stride, file->cache + (row % file->cache_rows) * file->row_bytes, file->row_bytes);
			continue;
		}
		file->misses[missing].index = index;
		file->misses[missing].row = row;
		++missing;
	}
	if ( missing != 0u )
	{
		SparkRowFileSortMisses(file, missing);
		const uint64_t spans = SparkRowFilePlanSpans(file, missing);
		pthread_mutex_lock(&file->job_lock);
		file->out = out;
		file->out_stride = out_stride;
		file->span_count = spans;
		file->span_next = 0u;
		file->span_left = spans;
		file->failed = 0u;
		if ( spans >= file->workers )
			pthread_cond_broadcast(&file->job_ready);
		else
			for ( uint64_t wake = 0u; wake < spans; ++wake )
				pthread_cond_signal(&file->job_ready);
		while ( file->span_left != 0u )
			pthread_cond_wait(&file->job_done, &file->job_lock);
		if ( file->failed != 0u )
			status = SPARK_STATUS_IO_ERROR;
		file->span_count = 0u;
		pthread_mutex_unlock(&file->job_lock);
		if ( status == SPARK_STATUS_OK && file->cache_rows != 0u )
			for ( uint64_t index = 0u; index < missing; ++index )
			{
				const uint64_t row = file->sorted[index].row, slot = row % file->cache_rows;
				memcpy(file->cache + slot * file->row_bytes, out + file->sorted[index].index * out_stride, file->row_bytes);
				file->tags[slot] = row + 1u;
			}
	}
	file->counters.requested += wanted;
	file->counters.cached += wanted - missing;
	file->counters.read += missing;
	pthread_mutex_unlock(&file->fetch_lock);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

void SparkRowFileCount(SparkRowFile *file, SparkRowFileCounters *counters)
{
	if ( file == 0 || counters == 0 )
		return;
	pthread_mutex_lock(&file->fetch_lock);
	*counters = file->counters;
	pthread_mutex_unlock(&file->fetch_lock);
}
