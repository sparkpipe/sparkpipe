#include "sparkpipe/spark_prefix_index_file.h"
#include "sparkpipe/spark_error_site.h"

#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint64_t SparkPrefixIndexNowNs(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static void SparkPrefixIndexDigestText(SparkSha256Context *context,const char *text)
{
	uint32_t length = text != 0 ? (uint32_t)strlen(text) : 0u;
	SparkSha256Update(context,&length,sizeof(length));
	if ( length != 0u )
		SparkSha256Update(context,text,length);
}

void SparkPrefixIndexModelDigest(const char *adapter_id,const char *model_id,const char *model_revision,const char *artifact_sha256,uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	SparkSha256Context context;
	SparkSha256Initialize(&context);
	SparkPrefixIndexDigestText(&context,adapter_id);
	SparkPrefixIndexDigestText(&context,model_id);
	SparkPrefixIndexDigestText(&context,model_revision);
	SparkPrefixIndexDigestText(&context,artifact_sha256);
	SparkSha256Finalize(&context,digest);
}

static void SparkPrefixIndexHeaderDigest(const SparkPrefixIndexFileHeader *header,uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	SparkPrefixIndexFileHeader copy = *header;
	SparkSha256Context context;
	memset(copy.header_sha256,0,sizeof(copy.header_sha256));
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,&copy,sizeof(copy));
	SparkSha256Finalize(&context,digest);
}

static SparkStatus SparkPrefixIndexReadAll(int descriptor,void *data,size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		ssize_t got = read(descriptor,(uint8_t *)data + done,bytes - done);
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)got;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkPrefixIndexWriteAll(int descriptor,const void *data,size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		ssize_t put = write(descriptor,(const uint8_t *)data + done,bytes - done);
		if ( put < 0 && errno == EINTR )
			continue;
		if ( put <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)put;
	}
	return(SPARK_STATUS_OK);
}

static const char *SparkPrefixIndexCheckHeader(const SparkPrefixIndexFileHeader *header,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,uint32_t record_capacity,uint64_t file_bytes)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	if ( header->magic != SPARK_PREFIX_INDEX_FILE_MAGIC )
		return("magic");
	if ( header->version != SPARK_PREFIX_INDEX_FILE_VERSION || header->header_bytes != sizeof(*header) || header->record_bytes != sizeof(SparkPrefixCacheCommittedRecord) || header->reserved0 != 0u )
		return("version");
	SparkPrefixIndexHeaderDigest(header,digest);
	if ( memcmp(digest,header->header_sha256,sizeof(digest)) != 0 )
		return("digest");
	if ( header->block_token_count != block_token_count )
		return("block_tokens");
	if ( memcmp(header->model_sha256,model_sha256,SPARK_SHA256_DIGEST_BYTES) != 0 )
		return("model");
	if ( header->record_count > record_capacity || file_bytes != sizeof(*header) + (uint64_t)header->record_count * sizeof(SparkPrefixCacheCommittedRecord) )
		return("size");
	return(0);
}

SparkStatus SparkPrefixIndexFileRead(const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,SparkPrefixCacheCommittedRecord *records,uint32_t record_capacity,uint32_t *record_count_out,const char **reason_out)
{
	SparkPrefixIndexFileHeader header;
	SparkSha256Context context;
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	struct stat info;
	const char *reason;
	SparkStatus status;
	int descriptor;
	if ( path == 0 || model_sha256 == 0 || record_count_out == 0 || reason_out == 0 || (record_capacity != 0u && records == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*record_count_out = 0u;
	*reason_out = 0;
	descriptor = open(path,O_RDONLY | O_CLOEXEC);
	if ( descriptor < 0 )
	{
		if ( errno == ENOENT )
			return(SPARK_STATUS_NOT_FOUND);
		*reason_out = "io";
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = fstat(descriptor,&info) == 0 && (uint64_t)info.st_size >= sizeof(header) ? SparkPrefixIndexReadAll(descriptor,&header,sizeof(header)) : SPARK_STATUS_VALIDATION_FAILED;
	reason = status == SPARK_STATUS_OK ? SparkPrefixIndexCheckHeader(&header,model_sha256,block_token_count,record_capacity,(uint64_t)info.st_size) : (status == SPARK_STATUS_VALIDATION_FAILED ? "size" : "io");
	if ( reason == 0 && header.record_count != 0u )
		status = SparkPrefixIndexReadAll(descriptor,records,(size_t)header.record_count * sizeof(records[0]));
	(void)close(descriptor);
	if ( reason == 0 && status != SPARK_STATUS_OK )
		reason = "io";
	if ( reason == 0 )
	{
		SparkSha256Initialize(&context);
		SparkSha256Update(&context,records,(size_t)header.record_count * sizeof(records[0]));
		SparkSha256Finalize(&context,digest);
		if ( memcmp(digest,header.records_sha256,sizeof(digest)) != 0 )
			reason = "digest";
	}
	if ( reason != 0 )
	{
		*reason_out = reason;
		SPARK_FAIL(strcmp(reason,"io") == 0 ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_HASH_MISMATCH);
	}
	*record_count_out = header.record_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkPrefixIndexSyncDirectory(const char *path)
{
	char copy[SPARK_PREFIX_INDEX_FILE_PATH_BYTES];
	int descriptor;
	SparkStatus status;
	if ( strlen(path) >= sizeof(copy) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	strcpy(copy,path);
	descriptor = open(dirname(copy),O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if ( descriptor < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = fsync(descriptor) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	(void)close(descriptor);
	SPARK_RETURN(status);
}

SparkStatus SparkPrefixIndexFileWrite(const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,const SparkPrefixCacheCommittedRecord *records,uint32_t record_count)
{
	SparkPrefixIndexFileHeader header;
	SparkSha256Context context;
	char temporary[SPARK_PREFIX_INDEX_FILE_PATH_BYTES + 32u];
	SparkStatus status;
	int descriptor,written;
	if ( path == 0 || model_sha256 == 0 || (record_count != 0u && records == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	written = snprintf(temporary,sizeof(temporary),"%s.tmp-%d",path,(int)getpid());
	if ( written < 0 || (size_t)written >= sizeof(temporary) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&header,0,sizeof(header));
	header.magic = SPARK_PREFIX_INDEX_FILE_MAGIC;
	header.version = SPARK_PREFIX_INDEX_FILE_VERSION;
	header.header_bytes = sizeof(header);
	header.block_token_count = block_token_count;
	header.record_count = record_count;
	header.record_bytes = sizeof(SparkPrefixCacheCommittedRecord);
	memcpy(header.model_sha256,model_sha256,SPARK_SHA256_DIGEST_BYTES);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,records,(size_t)record_count * sizeof(records[0]));
	SparkSha256Finalize(&context,header.records_sha256);
	SparkPrefixIndexHeaderDigest(&header,header.header_sha256);
	descriptor = open(temporary,O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,0600);
	if ( descriptor < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkPrefixIndexWriteAll(descriptor,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK && record_count != 0u )
		status = SparkPrefixIndexWriteAll(descriptor,records,(size_t)record_count * sizeof(records[0]));
	if ( status == SPARK_STATUS_OK && fsync(descriptor) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( close(descriptor) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && rename(temporary,path) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		(void)unlink(temporary);
		SPARK_RETURN(status);
	}
	return(SparkPrefixIndexSyncDirectory(path));
}

static void *SparkPrefixIndexWriterMain(void *context)
{
	SparkPrefixIndexWriter *writer = (SparkPrefixIndexWriter *)context;
	uint64_t start,elapsed;
	SparkStatus status;
	(void)pthread_mutex_lock(&writer->mutex);
	for (;;)
	{
		while ( writer->pending == 0u && writer->stop == 0u )
			(void)pthread_cond_wait(&writer->condition,&writer->mutex);
		if ( writer->pending == 0u )
			break;
		writer->pending = 0u;
		writer->busy = 1u;
		(void)pthread_mutex_unlock(&writer->mutex);
		start = SparkPrefixIndexNowNs();
		status = SparkPrefixIndexFileWrite(writer->path,writer->model_sha256,writer->block_token_count,writer->records,writer->record_count);
		elapsed = SparkPrefixIndexNowNs() - start;
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"batch engine prefix index save path=%s status=%s\n",writer->path,SparkStatusToString(status));
		(void)pthread_mutex_lock(&writer->mutex);
		if ( status == SPARK_STATUS_OK )
			writer->save_count++;
		else
			writer->save_failure_count++;
		writer->write_ns_total += elapsed;
		if ( elapsed > writer->write_ns_maximum )
			writer->write_ns_maximum = elapsed;
		writer->busy = 0u;
		(void)pthread_cond_broadcast(&writer->condition);
	}
	(void)pthread_mutex_unlock(&writer->mutex);
	return(0);
}

SparkStatus SparkPrefixIndexWriterStart(SparkPrefixIndexWriter *writer,const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,uint32_t record_capacity)
{
	if ( writer == 0 || path == 0 || model_sha256 == 0 || strlen(path) >= sizeof(writer->path) || record_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(writer,0,sizeof(*writer));
	strcpy(writer->path,path);
	memcpy(writer->model_sha256,model_sha256,SPARK_SHA256_DIGEST_BYTES);
	writer->block_token_count = block_token_count;
	writer->record_capacity = record_capacity;
	writer->records = (SparkPrefixCacheCommittedRecord *)calloc(record_capacity,sizeof(writer->records[0]));
	if ( writer->records == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_mutex_init(&writer->mutex,0) != 0 || pthread_cond_init(&writer->condition,0) != 0 || pthread_create(&writer->thread,0,SparkPrefixIndexWriterMain,writer) != 0 )
	{
		free(writer->records);
		writer->records = 0;
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	writer->started = 1u;
	return(SPARK_STATUS_OK);
}

uint32_t SparkPrefixIndexWriterIdle(SparkPrefixIndexWriter *writer)
{
	uint32_t idle;
	if ( writer == 0 || writer->started == 0u )
		return(0u);
	(void)pthread_mutex_lock(&writer->mutex);
	idle = writer->pending == 0u && writer->busy == 0u ? 1u : 0u;
	(void)pthread_mutex_unlock(&writer->mutex);
	return(idle);
}

void SparkPrefixIndexWriterWaitIdle(SparkPrefixIndexWriter *writer)
{
	if ( writer == 0 || writer->started == 0u )
		return;
	(void)pthread_mutex_lock(&writer->mutex);
	while ( writer->pending != 0u || writer->busy != 0u )
		(void)pthread_cond_wait(&writer->condition,&writer->mutex);
	(void)pthread_mutex_unlock(&writer->mutex);
}

SparkPrefixCacheCommittedRecord *SparkPrefixIndexWriterBuffer(SparkPrefixIndexWriter *writer)
{
	return(writer != 0 ? writer->records : 0);
}

void SparkPrefixIndexWriterSubmit(SparkPrefixIndexWriter *writer,uint32_t record_count)
{
	if ( writer == 0 || writer->started == 0u )
		return;
	(void)pthread_mutex_lock(&writer->mutex);
	writer->record_count = record_count;
	writer->pending = 1u;
	(void)pthread_cond_broadcast(&writer->condition);
	(void)pthread_mutex_unlock(&writer->mutex);
}

void SparkPrefixIndexWriterStop(SparkPrefixIndexWriter *writer)
{
	if ( writer == 0 || writer->started == 0u )
		return;
	(void)pthread_mutex_lock(&writer->mutex);
	writer->stop = 1u;
	(void)pthread_cond_broadcast(&writer->condition);
	(void)pthread_mutex_unlock(&writer->mutex);
	(void)pthread_join(writer->thread,0);
	(void)pthread_cond_destroy(&writer->condition);
	(void)pthread_mutex_destroy(&writer->mutex);
	free(writer->records);
	writer->records = 0;
	writer->started = 0u;
}

void SparkPrefixIndexWriterSample(SparkPrefixIndexWriter *writer,uint64_t *save_count,uint64_t *save_failure_count,uint64_t *write_ns_total,uint64_t *write_ns_maximum)
{
	if ( writer == 0 || writer->started == 0u )
	{
		*save_count = *save_failure_count = *write_ns_total = *write_ns_maximum = 0u;
		return;
	}
	(void)pthread_mutex_lock(&writer->mutex);
	*save_count = writer->save_count;
	*save_failure_count = writer->save_failure_count;
	*write_ns_total = writer->write_ns_total;
	*write_ns_maximum = writer->write_ns_maximum;
	(void)pthread_mutex_unlock(&writer->mutex);
}
