#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _DARWIN_C_SOURCE
#include "sparkpipe/spark_weightd_receipt.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define RECEIPT_HEADER "sparkpipe-pack-receipt 1\n"
#define RECEIPT_SEAL "seal "

uint64_t SparkWeightdPackMtimeNs(const struct stat *status)
{
#if defined(__APPLE__)
	return((uint64_t)status->st_mtimespec.tv_sec * UINT64_C(1000000000) + (uint64_t)status->st_mtimespec.tv_nsec);
#else
	return((uint64_t)status->st_mtim.tv_sec * UINT64_C(1000000000) + (uint64_t)status->st_mtim.tv_nsec);
#endif
}

uint64_t SparkWeightdPackCtimeNs(const struct stat *status)
{
#if defined(__APPLE__)
	return((uint64_t)status->st_ctimespec.tv_sec * UINT64_C(1000000000) + (uint64_t)status->st_ctimespec.tv_nsec);
#else
	return((uint64_t)status->st_ctim.tv_sec * UINT64_C(1000000000) + (uint64_t)status->st_ctim.tv_nsec);
#endif
}

int32_t SparkWeightdPackStatSame(const struct stat *before,const struct stat *after)
{
	return(before->st_dev == after->st_dev && before->st_ino == after->st_ino && before->st_size == after->st_size &&
		SparkWeightdPackMtimeNs(before) == SparkWeightdPackMtimeNs(after) && SparkWeightdPackCtimeNs(before) == SparkWeightdPackCtimeNs(after));
}

static uint64_t receipt_now(clockid_t clock)
{
	struct timespec now;
	if ( clock_gettime(clock,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static int32_t receipt_hex_valid(const char *hex,size_t length)
{
	size_t i;
	if ( hex == 0 || strlen(hex) != length )
		return(0);
	for (i=0u; i<length; i++)
		if ( (hex[i] < '0' || hex[i] > '9') && (hex[i] < 'a' || hex[i] > 'f') )
			return(0);
	return(1);
}

static int32_t receipt_digest_field(const char *field,size_t length)
{
	return(strcmp(field,"-") == 0 || receipt_hex_valid(field,length) != 0);
}

static SparkStatus receipt_state_dir(char *out,size_t bytes)
{
	const char *state = getenv("XDG_STATE_HOME");
	const char *home = getenv("HOME");
	int written;
	if ( state != 0 && state[0] == '/' )
		written = snprintf(out,bytes,"%s/sparkpipe/verified",state);
	else if ( home != 0 && home[0] == '/' )
		written = snprintf(out,bytes,"%s/.local/state/sparkpipe/verified",home);
	else
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( written <= 0 || (size_t)written >= bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdPackDigestRead(const char *pack_path,char hex[65])
{
	char path[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	char text[80];
	size_t count;
	FILE *file;
	int written;
	if ( pack_path == 0 || hex == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	written = snprintf(path,sizeof(path),"%s.sha256",pack_path);
	if ( written <= 0 || (size_t)written >= sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	file = fopen(path,"rb");
	if ( file == 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	count = fread(text,1u,65u,file);
	(void)fclose(file);
	if ( count < 64u || (count == 65u && text[64] != ' ' && text[64] != '\t' && text[64] != '\n') )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	text[64] = 0;
	if ( receipt_hex_valid(text,64u) == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	memcpy(hex,text,65u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdReceiptLocate(const char *pack_path,const struct stat *pack,char primary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],char fallback[SPARK_WEIGHTD_RECEIPT_PATH_BYTES])
{
	char resolved[PATH_MAX];
	char state[SPARK_WEIGHTD_RECEIPT_PATH_BYTES - 64u];
	int written;
	if ( pack_path == 0 || pack == 0 || primary == 0 || fallback == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	primary[0] = 0;
	fallback[0] = 0;
	if ( realpath(pack_path,resolved) == 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	written = snprintf(primary,SPARK_WEIGHTD_RECEIPT_PATH_BYTES,"%s%s",resolved,SPARK_WEIGHTD_RECEIPT_SUFFIX);
	if ( written <= 0 || (size_t)written >= SPARK_WEIGHTD_RECEIPT_PATH_BYTES )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( receipt_state_dir(state,sizeof(state)) == SPARK_STATUS_OK )
		(void)snprintf(fallback,SPARK_WEIGHTD_RECEIPT_PATH_BYTES,"%s/%llu-%llu%s",state,(unsigned long long)pack->st_dev,(unsigned long long)pack->st_ino,SPARK_WEIGHTD_RECEIPT_SUFFIX);
	return(SPARK_STATUS_OK);
}

static SparkStatus receipt_render(const SparkWeightdReceipt *receipt,char *text,size_t capacity,size_t *bytes)
{
	char seal[SPARK_SHA256_HEX_BYTES];
	int body,sealed;
	body = snprintf(text,capacity,RECEIPT_HEADER "device %llu\ninode %llu\nsize %llu\nmtime_ns %llu\nctime_ns %llu\nsha256 %s\nck128 %s\nverifier %s\nverified_unix_ns %llu\n",
		(unsigned long long)receipt->device,(unsigned long long)receipt->inode,(unsigned long long)receipt->size,
		(unsigned long long)receipt->mtime_ns,(unsigned long long)receipt->ctime_ns,
		receipt->sha256[0] != 0 ? receipt->sha256 : "-",receipt->ck128[0] != 0 ? receipt->ck128 : "-",
		receipt->verifier,(unsigned long long)receipt->verified_unix_ns);
	if ( body <= 0 || (size_t)body >= capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkSha256Bytes(text,(size_t)body,seal) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	sealed = snprintf(text + body,capacity - (size_t)body,RECEIPT_SEAL "%s\n",seal);
	if ( sealed <= 0 || (size_t)sealed >= capacity - (size_t)body )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*bytes = (size_t)body + (size_t)sealed;
	return(SPARK_STATUS_OK);
}

static SparkStatus receipt_parse(const char *text,size_t bytes,SparkWeightdReceipt *out)
{
	char canonical[SPARK_WEIGHTD_RECEIPT_BYTES_MAX + 1u];
	char sha[80],ck[80];
	unsigned long long device,inode,size,mtime_ns,ctime_ns,verified;
	size_t rendered;
	int consumed = 0;
	memset(out,0,sizeof(*out));
	if ( bytes == 0u || bytes > SPARK_WEIGHTD_RECEIPT_BYTES_MAX || memchr(text,0,bytes) != 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( sscanf(text,RECEIPT_HEADER "device %llu\ninode %llu\nsize %llu\nmtime_ns %llu\nctime_ns %llu\nsha256 %79s\nck128 %79s\nverifier %95[^\n]\nverified_unix_ns %llu\n%n",
		&device,&inode,&size,&mtime_ns,&ctime_ns,sha,ck,out->verifier,&verified,&consumed) != 9 || consumed <= 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( receipt_digest_field(sha,64u) == 0 || receipt_digest_field(ck,32u) == 0 || (sha[0] == '-' && ck[0] == '-') )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	out->device = device;
	out->inode = inode;
	out->size = size;
	out->mtime_ns = mtime_ns;
	out->ctime_ns = ctime_ns;
	out->verified_unix_ns = verified;
	if ( sha[0] != '-' )
		memcpy(out->sha256,sha,65u);
	if ( ck[0] != '-' )
		memcpy(out->ck128,ck,33u);
	if ( receipt_render(out,canonical,sizeof(canonical),&rendered) != SPARK_STATUS_OK || rendered != bytes || memcmp(canonical,text,bytes) != 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdReceiptLoad(const char *receipt_path,SparkWeightdReceipt *out)
{
	char text[SPARK_WEIGHTD_RECEIPT_BYTES_MAX + 1u];
	struct stat info;
	ssize_t count;
	size_t done = 0u;
	int32_t fd;
	if ( receipt_path == 0 || receipt_path[0] == 0 || out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	fd = open(receipt_path,O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
	if ( fd < 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	if ( fstat(fd,&info) != 0 || S_ISREG(info.st_mode) == 0 || (info.st_uid != geteuid() && info.st_uid != 0) || (info.st_mode & (S_IWGRP | S_IWOTH)) != 0 )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( info.st_size <= 0 || (uint64_t)info.st_size > SPARK_WEIGHTD_RECEIPT_BYTES_MAX )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	while ( done < sizeof(text) )
	{
		count = read(fd,text + done,sizeof(text) - done);
		if ( count < 0 && errno == EINTR )
			continue;
		if ( count <= 0 )
			break;
		done += (size_t)count;
	}
	(void)close(fd);
	if ( done != (size_t)info.st_size )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return(receipt_parse(text,done,out));
}

static SparkStatus receipt_check_one(const char *path,const struct stat *pack,const char *sha256_hex,const char *ck128_hex,const char **reason)
{
	SparkWeightdReceipt receipt;
	SparkStatus status = SparkWeightdReceiptLoad(path,&receipt);
	if ( status == SPARK_STATUS_NOT_FOUND )
	{
		*reason = "absent";
		return(status);
	}
	if ( status != SPARK_STATUS_OK )
	{
		*reason = status == SPARK_STATUS_VALIDATION_FAILED ? "untrusted owner or mode" : "unreadable or malformed";
		return(status);
	}
	if ( receipt.device != (uint64_t)pack->st_dev || receipt.inode != (uint64_t)pack->st_ino || receipt.size != (uint64_t)pack->st_size ||
		receipt.mtime_ns != SparkWeightdPackMtimeNs(pack) || receipt.ctime_ns != SparkWeightdPackCtimeNs(pack) )
	{
		*reason = "stale: the pack changed since it was verified";
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( (sha256_hex != 0 && receipt.sha256[0] != 0 && strcmp(receipt.sha256,sha256_hex) == 0) ||
		(ck128_hex != 0 && receipt.ck128[0] != 0 && strcmp(receipt.ck128,ck128_hex) == 0) )
	{
		*reason = "valid";
		return(SPARK_STATUS_OK);
	}
	*reason = "digest differs from the expected digest";
	return(SPARK_STATUS_HASH_MISMATCH);
}

SparkStatus SparkWeightdReceiptCheck(const char *pack_path,int32_t fd,const char *sha256_hex,const char *ck128_hex,const char **reason)
{
	char primary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],fallback[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	const char *first = "absent",*second = "absent",*ignored;
	struct stat pack;
	SparkStatus status,other;
	if ( reason == 0 )
		reason = &ignored;
	*reason = "invalid request";
	if ( pack_path == 0 || fd < 0 || (sha256_hex == 0 && ck128_hex == 0) ||
		(sha256_hex != 0 && receipt_hex_valid(sha256_hex,64u) == 0) || (ck128_hex != 0 && receipt_hex_valid(ck128_hex,32u) == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*reason = "pack stat failed";
	if ( fstat(fd,&pack) != 0 || S_ISREG(pack.st_mode) == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdReceiptLocate(pack_path,&pack,primary,fallback);
	if ( status != SPARK_STATUS_OK )
	{
		*reason = "receipt path unresolved";
		return(status);
	}
	status = receipt_check_one(primary,&pack,sha256_hex,ck128_hex,&first);
	if ( status == SPARK_STATUS_OK || fallback[0] == 0 )
	{
		*reason = first;
		return(status);
	}
	other = receipt_check_one(fallback,&pack,sha256_hex,ck128_hex,&second);
	if ( other == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND )
	{
		*reason = second;
		return(other);
	}
	*reason = first;
	return(status);
}

static SparkStatus receipt_write_atomic(const char *path,const char *text,size_t bytes)
{
	char temporary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES + 16u];
	size_t done = 0u;
	ssize_t count;
	int32_t fd;
	int written = snprintf(temporary,sizeof(temporary),"%s.tmp.XXXXXX",path);
	if ( written <= 0 || (size_t)written >= sizeof(temporary) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	fd = mkstemp(temporary);
	if ( fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	while ( done < bytes )
	{
		count = write(fd,text + done,bytes - done);
		if ( count < 0 && errno == EINTR )
			continue;
		if ( count <= 0 )
			break;
		done += (size_t)count;
	}
	if ( done != bytes || fchmod(fd,0644) != 0 || fsync(fd) != 0 )
		done = 0u;
	if ( close(fd) != 0 || done != bytes || rename(temporary,path) != 0 )
	{
		(void)unlink(temporary);
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus receipt_make_state_dir(const char *fallback)
{
	char path[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	char *cursor;
	size_t length = strlen(fallback);
	if ( length >= sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(path,fallback,length + 1u);
	cursor = strrchr(path,'/');
	if ( cursor == 0 || cursor == path )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*cursor = 0;
	for (cursor = path + 1; *cursor != 0; cursor++)
	{
		if ( *cursor != '/' )
			continue;
		*cursor = 0;
		if ( mkdir(path,0700) != 0 && errno != EEXIST )
			return(SPARK_STATUS_IO_ERROR);
		*cursor = '/';
	}
	if ( mkdir(path,0700) != 0 && errno != EEXIST )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static void receipt_verifier(char *out,const char *role)
{
	char host[64];
	size_t i;
	if ( gethostname(host,sizeof(host)) != 0 )
		memcpy(host,"unknown",8u);
	host[sizeof(host) - 1u] = 0;
	(void)snprintf(out,SPARK_WEIGHTD_RECEIPT_VERIFIER_BYTES,"%s host=%s pid=%ld",role,host,(long)getpid());
	for (i=0u; out[i] != 0; i++)
		if ( out[i] < 0x20 || out[i] > 0x7e )
			out[i] = '_';
}

SparkStatus SparkWeightdReceiptRecord(const char *pack_path,int32_t fd,const struct stat *verified,const char *sha256_hex,const char *ck128_hex,const char *role)
{
	char primary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],fallback[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	char text[SPARK_WEIGHTD_RECEIPT_BYTES_MAX + 1u];
	SparkWeightdReceipt receipt;
	struct stat current;
	SparkStatus status;
	uint64_t now;
	size_t bytes;
	if ( pack_path == 0 || fd < 0 || verified == 0 || role == 0 || (sha256_hex == 0 && ck128_hex == 0) ||
		(sha256_hex != 0 && receipt_hex_valid(sha256_hex,64u) == 0) || (ck128_hex != 0 && receipt_hex_valid(ck128_hex,32u) == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fstat(fd,&current) != 0 || S_ISREG(current.st_mode) == 0 || SparkWeightdPackStatSame(verified,&current) == 0 )
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	now = receipt_now(CLOCK_REALTIME);
	if ( now < SPARK_WEIGHTD_RECEIPT_SETTLE_NS || SparkWeightdPackMtimeNs(&current) > now - SPARK_WEIGHTD_RECEIPT_SETTLE_NS ||
		SparkWeightdPackCtimeNs(&current) > now - SPARK_WEIGHTD_RECEIPT_SETTLE_NS )
		return(SPARK_STATUS_BUSY);
	memset(&receipt,0,sizeof(receipt));
	receipt.device = (uint64_t)current.st_dev;
	receipt.inode = (uint64_t)current.st_ino;
	receipt.size = (uint64_t)current.st_size;
	receipt.mtime_ns = SparkWeightdPackMtimeNs(&current);
	receipt.ctime_ns = SparkWeightdPackCtimeNs(&current);
	receipt.verified_unix_ns = now;
	if ( sha256_hex != 0 )
		memcpy(receipt.sha256,sha256_hex,65u);
	if ( ck128_hex != 0 )
		memcpy(receipt.ck128,ck128_hex,33u);
	receipt_verifier(receipt.verifier,role);
	status = receipt_render(&receipt,text,sizeof(text),&bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdReceiptLocate(pack_path,&current,primary,fallback);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( receipt_write_atomic(primary,text,bytes) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	if ( fallback[0] == 0 || receipt_make_state_dir(fallback) != SPARK_STATUS_OK )
		return(SPARK_STATUS_IO_ERROR);
	return(receipt_write_atomic(fallback,text,bytes));
}

typedef struct StreamState
{
	pthread_mutex_t lock;
	pthread_cond_t changed;
	uint8_t *buffers[SPARK_WEIGHTD_STREAM_DEPTH];
	uint64_t lengths[SPARK_WEIGHTD_STREAM_DEPTH];
	uint64_t bytes;
	uint64_t produced;
	uint64_t consumed;
	uint64_t chunks;
	int32_t fd;
	int32_t direct;
	int32_t stop;
	SparkStatus status;
} StreamState;

static SparkStatus stream_read(StreamState *state,uint8_t *buffer,uint64_t offset,uint64_t want)
{
	uint64_t request = want,done = 0u;
	ssize_t count;
	if ( state->direct != 0 )
		request = (want + SPARK_WEIGHTD_STREAM_ALIGN - 1u) & ~(uint64_t)(SPARK_WEIGHTD_STREAM_ALIGN - 1u);
	while ( done < want )
	{
		count = pread(state->fd,buffer + done,(size_t)(request - done),(off_t)(offset + done));
		if ( count < 0 && errno == EINTR )
			continue;
		if ( count <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (uint64_t)count;
	}
	return(SPARK_STATUS_OK);
}

static void *stream_reader(void *argument)
{
	StreamState *state = (StreamState *)argument;
	uint64_t index,offset,want;
	SparkStatus status = SPARK_STATUS_OK;
	for (index=0u; index<state->chunks && status == SPARK_STATUS_OK; index++)
	{
		(void)pthread_mutex_lock(&state->lock);
		while ( state->stop == 0 && (state->produced - state->consumed) >= SPARK_WEIGHTD_STREAM_DEPTH )
			(void)pthread_cond_wait(&state->changed,&state->lock);
		if ( state->stop != 0 )
		{
			(void)pthread_mutex_unlock(&state->lock);
			return(0);
		}
		(void)pthread_mutex_unlock(&state->lock);
		offset = index * SPARK_WEIGHTD_STREAM_CHUNK_BYTES;
		want = state->bytes - offset < SPARK_WEIGHTD_STREAM_CHUNK_BYTES ? state->bytes - offset : SPARK_WEIGHTD_STREAM_CHUNK_BYTES;
		status = stream_read(state,state->buffers[index % SPARK_WEIGHTD_STREAM_DEPTH],offset,want);
		(void)pthread_mutex_lock(&state->lock);
		if ( status == SPARK_STATUS_OK )
		{
			state->lengths[index % SPARK_WEIGHTD_STREAM_DEPTH] = want;
			state->produced++;
		}
		else
		{
			state->status = status;
			state->stop = 1;
		}
		(void)pthread_cond_broadcast(&state->changed);
		(void)pthread_mutex_unlock(&state->lock);
	}
	return(0);
}

static SparkStatus stream_consume(StreamState *state,SparkWeightdStreamSink sink,void *context,SparkWeightdStreamStats *stats)
{
	uint64_t index,mark;
	SparkStatus status = SPARK_STATUS_OK;
	for (index=0u; index<state->chunks && status == SPARK_STATUS_OK; index++)
	{
		mark = receipt_now(CLOCK_MONOTONIC);
		(void)pthread_mutex_lock(&state->lock);
		while ( state->stop == 0 && state->produced <= index )
			(void)pthread_cond_wait(&state->changed,&state->lock);
		if ( state->produced <= index )
			status = state->status != SPARK_STATUS_OK ? state->status : SPARK_STATUS_IO_ERROR;
		(void)pthread_mutex_unlock(&state->lock);
		stats->wait_ns += receipt_now(CLOCK_MONOTONIC) - mark;
		if ( status != SPARK_STATUS_OK )
			break;
		mark = receipt_now(CLOCK_MONOTONIC);
		status = sink(context,state->buffers[index % SPARK_WEIGHTD_STREAM_DEPTH],index * SPARK_WEIGHTD_STREAM_CHUNK_BYTES,state->lengths[index % SPARK_WEIGHTD_STREAM_DEPTH]);
		stats->sink_ns += receipt_now(CLOCK_MONOTONIC) - mark;
		(void)pthread_mutex_lock(&state->lock);
		if ( status == SPARK_STATUS_OK )
		{
			state->consumed++;
			stats->bytes += state->lengths[index % SPARK_WEIGHTD_STREAM_DEPTH];
		}
		else
			state->stop = 1;
		(void)pthread_cond_broadcast(&state->changed);
		(void)pthread_mutex_unlock(&state->lock);
	}
	return(status);
}

static SparkStatus stream_open(StreamState *state,const char *pack_path,int32_t fd)
{
	struct stat shared,direct;
	state->fd = fd;
	state->direct = 0;
#if defined(O_DIRECT)
	{
		int32_t direct_fd = open(pack_path,O_RDONLY | O_DIRECT | O_CLOEXEC);
		if ( direct_fd < 0 )
			return(errno == EINVAL ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
		if ( fstat(fd,&shared) != 0 || fstat(direct_fd,&direct) != 0 || shared.st_dev != direct.st_dev || shared.st_ino != direct.st_ino )
		{
			(void)close(direct_fd);
			SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
		}
		state->fd = direct_fd;
		state->direct = 1;
	}
#else
	(void)pack_path;
	(void)shared;
	(void)direct;
#endif
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdPackStream(const char *pack_path,int32_t fd,uint64_t bytes,SparkWeightdStreamSink sink,void *context,SparkWeightdStreamStats *stats)
{
	StreamState state;
	pthread_t reader;
	SparkStatus status;
	uint64_t start;
	uint32_t i,allocated = 0u;
	if ( pack_path == 0 || fd < 0 || bytes == 0u || sink == 0 || stats == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(stats,0,sizeof(*stats));
	memset(&state,0,sizeof(state));
	state.bytes = bytes;
	state.chunks = (bytes + SPARK_WEIGHTD_STREAM_CHUNK_BYTES - 1u) / SPARK_WEIGHTD_STREAM_CHUNK_BYTES;
	start = receipt_now(CLOCK_MONOTONIC);
	status = stream_open(&state,pack_path,fd);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	stats->direct = (uint32_t)state.direct;
	for (i=0u; i<SPARK_WEIGHTD_STREAM_DEPTH && status == SPARK_STATUS_OK; i++)
	{
		if ( posix_memalign((void **)&state.buffers[i],SPARK_WEIGHTD_STREAM_ALIGN,(size_t)SPARK_WEIGHTD_STREAM_CHUNK_BYTES) != 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		else
			allocated++;
	}
	if ( status == SPARK_STATUS_OK && (pthread_mutex_init(&state.lock,0) != 0 || pthread_cond_init(&state.changed,0) != 0) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		if ( pthread_create(&reader,0,stream_reader,&state) != 0 )
			status = SPARK_STATUS_INTERNAL_ERROR;
		else
		{
			status = stream_consume(&state,sink,context,stats);
			(void)pthread_join(reader,0);
		}
		(void)pthread_cond_destroy(&state.changed);
		(void)pthread_mutex_destroy(&state.lock);
	}
	for (i=0u; i<allocated; i++)
		free(state.buffers[i]);
	if ( state.direct != 0 )
		(void)close(state.fd);
	stats->elapsed_ns = receipt_now(CLOCK_MONOTONIC) - start;
	if ( status == SPARK_STATUS_OK && stats->bytes != bytes )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}
