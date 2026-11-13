#include "sparkpipe/spark_row_file.h"
#include "sparkpipe/spark_error_site.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SPARK_ROW_FILE_BATCH_ROWS 64u

typedef struct SparkRowFileMiss
{
	uint64_t index;
	uint64_t row;
} SparkRowFileMiss;

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
	SparkRowFileMiss *misses;
	uint64_t miss_capacity, miss_count, miss_next, miss_left;
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

static int32_t SparkRowFileRead(SparkRowFile *file, uint8_t *block, uint64_t row, uint8_t *target)
{
	const uint64_t at = file->offset + row * file->row_bytes;
	const uint64_t first = at - at % SPARK_ROW_FILE_BLOCK_BYTES;
	const uint64_t end = at + file->row_bytes;
	uint64_t span = ((end + SPARK_ROW_FILE_BLOCK_BYTES - 1u) / SPARK_ROW_FILE_BLOCK_BYTES) * SPARK_ROW_FILE_BLOCK_BYTES - first;
	uint64_t done = 0u;
	if ( end > file->file_bytes )
		return(-1);
	while ( done < end - first )
	{
		const ssize_t got = pread(file->descriptor, block + done, (size_t)(span - done), (off_t)(first + done));
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got <= 0 )
			return(-2);
		done += (uint64_t)got;
	}
	memcpy(target, block + (at - first), file->row_bytes);
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
		while ( file->stop == 0u && file->miss_next >= file->miss_count )
			pthread_cond_wait(&file->job_ready, &file->job_lock);
		if ( file->stop != 0u )
			break;
		const SparkRowFileMiss *misses = file->misses + file->miss_next;
		uint8_t *out = file->out;
		const uint64_t stride = file->out_stride;
		uint64_t take = (file->miss_count - file->miss_next) / (4u * file->workers);
		uint32_t failed = 0u;
		take = take == 0u ? 1u : take > SPARK_ROW_FILE_BATCH_ROWS ? SPARK_ROW_FILE_BATCH_ROWS : take;
		file->miss_next += take;
		pthread_mutex_unlock(&file->job_lock);
		for ( uint64_t index = 0u; index < take; ++index )
			if ( SparkRowFileRead(file, block, misses[index].row, out + misses[index].index * stride) != 0 )
				failed = 1u;
		pthread_mutex_lock(&file->job_lock);
		if ( failed != 0u )
			file->failed = 1u;
		file->miss_left -= take;
		if ( file->miss_left == 0u )
			pthread_cond_signal(&file->job_done);
	}
	pthread_mutex_unlock(&file->job_lock);
	return(0);
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
	for ( uint32_t index = 0u; index < file->workers; ++index )
		if ( posix_memalign((void **)&file->blocks[index], SPARK_ROW_FILE_BLOCK_BYTES, 2u * SPARK_ROW_FILE_BLOCK_BYTES) != 0 )
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
		if ( grown == 0 )
		{
			pthread_mutex_unlock(&file->fetch_lock);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		file->misses = grown;
		file->miss_capacity = count;
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
		pthread_mutex_lock(&file->job_lock);
		file->out = out;
		file->out_stride = out_stride;
		file->miss_count = missing;
		file->miss_next = 0u;
		file->miss_left = missing;
		file->failed = 0u;
		if ( missing >= file->workers )
			pthread_cond_broadcast(&file->job_ready);
		else
			for ( uint64_t wake = 0u; wake < missing; ++wake )
				pthread_cond_signal(&file->job_ready);
		while ( file->miss_left != 0u )
			pthread_cond_wait(&file->job_done, &file->job_lock);
		if ( file->failed != 0u )
			status = SPARK_STATUS_IO_ERROR;
		file->miss_count = 0u;
		pthread_mutex_unlock(&file->job_lock);
		if ( status == SPARK_STATUS_OK && file->cache_rows != 0u )
			for ( uint64_t index = 0u; index < missing; ++index )
			{
				const uint64_t row = file->misses[index].row, slot = row % file->cache_rows;
				memcpy(file->cache + slot * file->row_bytes, out + file->misses[index].index * out_stride, file->row_bytes);
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
