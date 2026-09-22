#define _DARWIN_C_SOURCE
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t spine_stat_mtime_ns(const struct stat *st)
{
#if defined(__APPLE__)
	return (uint64_t)st->st_mtimespec.tv_sec * 1000000000ull + (uint64_t)st->st_mtimespec.tv_nsec;
#else
	return (uint64_t)st->st_mtim.tv_sec * 1000000000ull + (uint64_t)st->st_mtim.tv_nsec;
#endif
}

static uint64_t spine_stat_ctime_ns(const struct stat *st)
{
#if defined(__APPLE__)
	return (uint64_t)st->st_ctimespec.tv_sec * 1000000000ull + (uint64_t)st->st_ctimespec.tv_nsec;
#else
	return (uint64_t)st->st_ctim.tv_sec * 1000000000ull + (uint64_t)st->st_ctim.tv_nsec;
#endif
}

static SparkStatus spine_read(int32_t fd,uint8_t *buffer,uint64_t offset,uint32_t bytes)
{
	uint32_t done = 0u;
	ssize_t count;
	while ( done < bytes )
	{
		count = pread(fd,buffer + done,bytes - done,(off_t)(offset + done));
		if ( count < 0 && errno == EINTR )
			continue;
		if ( count <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (uint32_t)count;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus spine_copy(const SparkWeightdManifest *manifest,uint32_t *index,const uint8_t *buffer,uint64_t offset,uint32_t bytes,uint8_t *destination)
{
	const SparkWeightdSpan *span;
	uint64_t start,end,limit = (offset + bytes);
	while ( *index < manifest->spine_count )
	{
		span = &manifest->spine[*index];
		if ( span->offset >= limit )
			break;
		start = span->offset > offset ? span->offset : offset;
		end = (span->offset + span->bytes);
		if ( end > limit )
			end = limit;
		if ( end > start && cudaMemcpy(destination + span->compact_offset + (start - span->offset),buffer + (start - offset),(size_t)(end - start),cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( (span->offset + span->bytes) > limit )
			break;
		(*index)++;
	}
	return(SPARK_STATUS_OK);
}

typedef struct SpineReceipt
{
	uint64_t magic;
	uint64_t size;
	uint64_t mtime_ns;
	uint64_t ctime_ns;
	uint8_t sha[32];
	uint8_t ck[16];
	uint64_t proof;
} SpineReceipt;

#define SPINE_RECEIPT_MAGIC UINT64_C(0x5350494e45524531)
/* Trust chain (receipt proof basis). CLIENT_SHA: the client spine loader
 * hashed the whole pack image and matched SHA256 against the expected
 * digest. DAEMON_SHA: the weightd daemon hashed the whole pack image at
 * arena materialization, matched SHA256 against the SAME expected digest
 * with the file stat stable across the read (spark_weightd.c), and
 * recorded this receipt. The fast path accepts exactly these two - they
 * prove the same predicate over the same bytes. Anything else (including
 * the pre-proof 80-byte receipts, which fail the exact-size read) is
 * treated as absent and re-proven by a full client hash. The daemon
 * writes NO receipt in ck128-sidecar mode: there the client full hash is
 * the only SHA256 proof, so this skip is structurally narrowed to the
 * strictly-redundant case (trust-chain document, dedicated PR). */
#define SPINE_RECEIPT_PROOF_CLIENT_SHA UINT64_C(0)
#define SPINE_RECEIPT_PROOF_DAEMON_SHA UINT64_C(1)

static void spine_receipt_path(char *out,size_t out_bytes,int32_t fd,
	const char *expected)
{
	struct stat st;
	SparkSha256Context hash;
	uint8_t digest[32];
	char hex[SPARK_SHA256_HEX_BYTES];
	if ( fstat(fd,&st) != 0 )
	{
		snprintf(out,out_bytes,"/tmp/spine-receipt-invalid");
		return;
	}
	SparkSha256Initialize(&hash);
	SparkSha256Update(&hash,expected,strlen(expected));
	SparkSha256Finalize(&hash,digest);
	SparkSha256DigestToHex(digest,hex);
	snprintf(out,out_bytes,"/tmp/spark-weightd-spine/%s-%llu-%llu.receipt",
		hex,(unsigned long long)st.st_size,
		(unsigned long long)st.st_ino);
}

static SparkStatus spine_stream(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *expected,uint8_t *destination)
{
	SparkSha256Context hash;
	SparkCk128Context quick;
	static _Thread_local uint8_t buffer[1048576];
	uint8_t digest[32];
	char hex[SPARK_SHA256_HEX_BYTES];
	char receipt_path[192];
	SpineReceipt receipt;
	uint64_t offset = 0u;
	uint32_t bytes,index = 0u;
	SparkStatus status;
	int have_receipt = 0;
	int receipt_fd;
	mkdir("/tmp/spark-weightd-spine",0755);
	spine_receipt_path(receipt_path,sizeof(receipt_path),fd,expected);
	receipt_fd = open(receipt_path,O_RDONLY);
	if ( receipt_fd >= 0 )
	{
		struct stat st;
		if ( fstat(fd,&st) == 0 &&
			read(receipt_fd,&receipt,sizeof(receipt)) == (ssize_t)sizeof(receipt) &&
			receipt.magic == SPINE_RECEIPT_MAGIC &&
			receipt.size == (uint64_t)st.st_size &&
			receipt.mtime_ns == spine_stat_mtime_ns(&st) &&
			receipt.ctime_ns == spine_stat_ctime_ns(&st) &&
			(receipt.proof == SPINE_RECEIPT_PROOF_CLIENT_SHA ||
			 receipt.proof == SPINE_RECEIPT_PROOF_DAEMON_SHA) )
			have_receipt = 1;
		close(receipt_fd);
	}
	if ( have_receipt != 0 )
	{
		uint32_t copy_index = 0u;
		for (index = 0u; index < manifest->spine_count; index++)
		{
			const SparkWeightdSpan *span = &manifest->spine[index];
			uint64_t span_offset = span->offset;
			while ( span_offset < span->offset + span->bytes )
			{
				uint64_t remain = span->offset + span->bytes - span_offset;
				bytes = (uint32_t)(remain < sizeof(buffer) ? remain : sizeof(buffer));
				status = spine_read(fd,buffer,span_offset,bytes);
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
				status = spine_copy(manifest,&copy_index,buffer,span_offset,bytes,destination);
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
				span_offset += bytes;
			}
		}
		return(SPARK_STATUS_OK);
	}
	SparkCk128Initialize(&quick);
	SparkSha256Initialize(&hash);
	while ( offset < pack_bytes )
	{
		bytes = (uint32_t)((pack_bytes - offset) < sizeof(buffer) ? (pack_bytes - offset) : sizeof(buffer));
		status = spine_read(fd,buffer,offset,bytes);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkCk128Update(&quick,buffer,bytes);
		SparkSha256Update(&hash,buffer,bytes);
		status = spine_copy(manifest,&index,buffer,offset,bytes,destination);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		offset += bytes;
	}
	SparkCk128Finalize(&quick,digest);
	{
		uint8_t ck[16];
		memcpy(ck,digest,16u);
		SparkSha256Finalize(&hash,digest);
		SparkSha256DigestToHex(digest,hex);
		if ( strcmp(hex,expected) != 0 )
			return(SPARK_STATUS_HASH_MISMATCH);
		{
			struct stat st;
			receipt.magic = SPINE_RECEIPT_MAGIC;
			if ( fstat(fd,&st) == 0 )
			{
				receipt.size = (uint64_t)st.st_size;
				receipt.mtime_ns = spine_stat_mtime_ns(&st);
				receipt.ctime_ns = spine_stat_ctime_ns(&st);
				memcpy(receipt.sha,digest,32u);
				memcpy(receipt.ck,ck,16u);
				receipt.proof = SPINE_RECEIPT_PROOF_CLIENT_SHA;
				receipt_fd = open(receipt_path,O_WRONLY | O_CREAT | O_TRUNC,0644);
				if ( receipt_fd >= 0 )
				{
					ssize_t written = write(receipt_fd,&receipt,sizeof(receipt));
					(void)written;
					close(receipt_fd);
				}
			}
		}
		return(SPARK_STATUS_OK);
	}
}

static int32_t spine_unchanged(const struct stat *before,const struct stat *after)
{
	if ( before->st_dev != after->st_dev || before->st_ino != after->st_ino || before->st_size != after->st_size )
		return(0);
#if defined(__APPLE__)
	return(before->st_mtimespec.tv_sec == after->st_mtimespec.tv_sec && before->st_mtimespec.tv_nsec == after->st_mtimespec.tv_nsec && before->st_ctimespec.tv_sec == after->st_ctimespec.tv_sec && before->st_ctimespec.tv_nsec == after->st_ctimespec.tv_nsec);
#else
	return(before->st_mtim.tv_sec == after->st_mtim.tv_sec && before->st_mtim.tv_nsec == after->st_mtim.tv_nsec && before->st_ctim.tv_sec == after->st_ctim.tv_sec && before->st_ctim.tv_nsec == after->st_ctim.tv_nsec);
#endif
}

SparkStatus SparkWeightdSpineLoad(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity)
{
	struct stat before,after;
	SparkStatus status;
	if ( fd < 0 || manifest == 0 || sha256 == 0 || SparkSha256HexIsValid(sha256) == 0 || pack_bytes == 0u || pack_bytes > INT64_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( manifest->spine_allocation_bytes > capacity || capacity > SIZE_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( manifest->spine_allocation_bytes != 0u && (destination == 0 || ((uintptr_t)destination & 255u) != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fstat(fd,&before) != 0 || S_ISREG(before.st_mode) == 0 || before.st_size < 0 || (uint64_t)before.st_size != pack_bytes )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = spine_stream(fd,manifest,pack_bytes,sha256,destination);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( fstat(fd,&after) != 0 || spine_unchanged(&before,&after) == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdSpineReceiptRecordDaemon(int32_t fd,const char *expected,
	const uint8_t sha_digest[32])
{
	char receipt_path[192];
	SpineReceipt receipt;
	struct stat st;
	int receipt_fd;
	if ( fd < 0 || expected == 0 || sha_digest == 0 ||
		SparkSha256HexIsValid(expected) == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fstat(fd,&st) != 0 || S_ISREG(st.st_mode) == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	memset(&receipt,0,sizeof(receipt));
	receipt.magic = SPINE_RECEIPT_MAGIC;
	receipt.size = (uint64_t)st.st_size;
	receipt.mtime_ns = spine_stat_mtime_ns(&st);
	receipt.ctime_ns = spine_stat_ctime_ns(&st);
	memcpy(receipt.sha,sha_digest,32u);
	receipt.proof = SPINE_RECEIPT_PROOF_DAEMON_SHA;
	spine_receipt_path(receipt_path,sizeof(receipt_path),fd,expected);
	receipt_fd = open(receipt_path,O_WRONLY | O_CREAT | O_TRUNC,0644);
	if ( receipt_fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( write(receipt_fd,&receipt,sizeof(receipt)) != (ssize_t)sizeof(receipt) )
	{
		close(receipt_fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	close(receipt_fd);
	return(SPARK_STATUS_OK);
}
