#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd_receipt.h"

#define PACK_BYTES (SPARK_WEIGHTD_STREAM_CHUNK_BYTES * 2u + 4097u)
#define THREADS 8u
#define ROUNDS 16u

typedef struct HashSink
{
	SparkSha256Context sha;
	SparkCk128Context ck;
	uint64_t next;
	uint32_t fail_at;
	uint32_t calls;
} HashSink;

typedef struct Racer
{
	const char *pack;
	const char *sha;
	const char *ck;
	uint32_t failures;
} Racer;

static char root[] = "/tmp/weightd-receipt-XXXXXX";
static char pack[256],receipt[280],sha[SPARK_SHA256_HEX_BYTES],ck[SPARK_CK128_HEX_BYTES];

static SparkStatus hash_sink(void *context,const uint8_t *data,uint64_t offset,uint64_t bytes)
{
	HashSink *sink = (HashSink *)context;
	sink->calls++;
	if ( sink->fail_at != 0u && sink->calls == sink->fail_at )
		return(SPARK_STATUS_IO_ERROR);
	assert(offset == sink->next);
	sink->next += bytes;
	SparkSha256Update(&sink->sha,data,(size_t)bytes);
	SparkCk128Update(&sink->ck,data,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static void write_pack(const char *path,uint32_t seed)
{
	static uint8_t block[1u << 20];
	uint64_t done = 0u,step;
	uint32_t state = seed,i;
	FILE *file = fopen(path,"wb");
	assert(file != 0);
	while ( done < PACK_BYTES )
	{
		for (i=0u; i<sizeof(block); i++)
		{
			state = state * 1664525u + 1013904223u;
			block[i] = (uint8_t)(state >> 24);
		}
		step = PACK_BYTES - done < sizeof(block) ? PACK_BYTES - done : sizeof(block);
		assert(fwrite(block,1u,(size_t)step,file) == step);
		done += step;
	}
	assert(fclose(file) == 0);
}

static int32_t open_pack(struct stat *info)
{
	int32_t fd = open(pack,O_RDONLY);
	assert(fd >= 0 && fstat(fd,info) == 0);
	return(fd);
}

static SparkStatus check(const char *path,const char *sha_hex,const char *ck_hex,const char **reason)
{
	struct stat info;
	int32_t fd = open(path,O_RDONLY);
	SparkStatus status;
	assert(fd >= 0 && fstat(fd,&info) == 0);
	status = SparkWeightdReceiptCheck(path,fd,sha_hex,ck_hex,reason);
	assert(close(fd) == 0);
	return(status);
}

static void record(void)
{
	struct stat info;
	int32_t fd = open_pack(&info);
	assert(SparkWeightdReceiptRecord(pack,fd,&info,sha,ck,"test") == SPARK_STATUS_OK);
	assert(close(fd) == 0);
}

static void settle(void)
{
	usleep(1100000);
}

static void write_text(const char *path,const char *text,size_t bytes,mode_t mode)
{
	int32_t fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0600);
	assert(fd >= 0 && write(fd,text,bytes) == (ssize_t)bytes && fchmod(fd,mode) == 0 && close(fd) == 0);
}

static void check_stream(void)
{
	SparkWeightdStreamStats stats;
	HashSink sink;
	uint8_t digest[32];
	char hex[SPARK_SHA256_HEX_BYTES];
	struct stat info;
	int32_t fd = open_pack(&info);
	memset(&sink,0,sizeof(sink));
	SparkSha256Initialize(&sink.sha);
	SparkCk128Initialize(&sink.ck);
	assert(SparkWeightdPackStream(pack,fd,PACK_BYTES,hash_sink,&sink,&stats) == SPARK_STATUS_OK);
	assert(stats.bytes == PACK_BYTES && sink.next == PACK_BYTES && sink.calls == 3u);
	SparkSha256Finalize(&sink.sha,digest);
	SparkSha256DigestToHex(digest,hex);
	assert(strcmp(hex,sha) == 0);
	SparkCk128Finalize(&sink.ck,digest);
	SparkCk128DigestToHex(digest,ck);
	memset(&sink,0,sizeof(sink));
	sink.fail_at = 2u;
	assert(SparkWeightdPackStream(pack,fd,PACK_BYTES,hash_sink,&sink,&stats) == SPARK_STATUS_IO_ERROR);
	memset(&sink,0,sizeof(sink));
	assert(SparkWeightdPackStream(pack,fd,PACK_BYTES + 1u,hash_sink,&sink,&stats) == SPARK_STATUS_IO_ERROR);
	assert(close(fd) == 0);
	puts("PASS stream: three pipelined chunks hash to the file digest; sink and short-read failures propagate");
}

static void check_record_rules(void)
{
	struct stat info,changed;
	const char *reason = 0;
	int32_t fd = open_pack(&info);
	assert(SparkWeightdReceiptRecord(pack,fd,&info,sha,ck,"test") == SPARK_STATUS_BUSY);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_NOT_FOUND && strcmp(reason,"absent") == 0);
	settle();
	changed = info;
	changed.st_size++;
	assert(SparkWeightdReceiptRecord(pack,fd,&changed,sha,ck,"test") == SPARK_STATUS_HASH_MISMATCH);
	assert(SparkWeightdReceiptRecord(pack,fd,&info,0,0,"test") == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdReceiptRecord(pack,fd,&info,"ABC",0,"test") == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdReceiptRecord(pack,fd,&info,sha,ck,"test") == SPARK_STATUS_OK);
	assert(close(fd) == 0);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_OK && strcmp(reason,"valid") == 0);
	assert(check(pack,0,ck,&reason) == SPARK_STATUS_OK);
	assert(check(pack,"1111111111111111111111111111111111111111111111111111111111111111",0,&reason) == SPARK_STATUS_HASH_MISMATCH);
	assert(strcmp(reason,"digest differs from the expected digest") == 0);
	assert(check(pack,0,"22222222222222222222222222222222",&reason) == SPARK_STATUS_HASH_MISMATCH);
	{
		SparkWeightdReceipt loaded;
		assert(SparkWeightdReceiptLoad(receipt,&loaded) == SPARK_STATUS_OK);
		assert(strcmp(loaded.sha256,sha) == 0 && strcmp(loaded.ck128,ck) == 0 && strncmp(loaded.verifier,"test host=",10u) == 0);
		assert(loaded.size == PACK_BYTES && loaded.inode == (uint64_t)info.st_ino && loaded.device == (uint64_t)info.st_dev);
		assert(loaded.mtime_ns == SparkWeightdPackMtimeNs(&info) && loaded.ctime_ns == SparkWeightdPackCtimeNs(&info));
	}
	puts("PASS record: racy stamps refused, changed stat refused, digests bound, fields persisted");
}

static void check_malformed(void)
{
	char text[SPARK_WEIGHTD_RECEIPT_BYTES_MAX + 1u];
	const char *reason = 0;
	ssize_t bytes;
	char original;
	int32_t fd = open(receipt,O_RDONLY);
	assert(fd >= 0);
	bytes = read(fd,text,sizeof(text));
	assert(bytes > 0 && close(fd) == 0);
	write_text(receipt,text,(size_t)bytes / 2u,0644);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_PARSE_ERROR && strcmp(reason,"unreadable or malformed") == 0);
	write_text(receipt,"",0u,0644);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_PARSE_ERROR);
	write_text(receipt,"garbage\n\x01\x02\x03",11u,0644);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_PARSE_ERROR);
	original = text[40];
	text[40] = original == '9' ? '8' : '9';
	write_text(receipt,text,(size_t)bytes,0644);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_PARSE_ERROR);
	text[40] = original;
	write_text(receipt,text,(size_t)bytes,0666);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_VALIDATION_FAILED && strcmp(reason,"untrusted owner or mode") == 0);
	write_text(receipt,text,(size_t)bytes,0644);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_OK);
	puts("PASS malformed: truncated, empty, garbage, edited-without-seal and group-writable receipts are not trusted");
}

static void check_invalidation(void)
{
	char copy[300];
	struct stat info;
	struct timespec times[2];
	const char *reason = 0;
	uint8_t byte;
	int32_t fd = open_pack(&info);
	times[0].tv_sec = 0;
	times[0].tv_nsec = UTIME_OMIT;
#if defined(__APPLE__)
	times[1] = info.st_mtimespec;
#else
	times[1] = info.st_mtim;
#endif
	assert(futimens(fd,times) == 0 && close(fd) == 0);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_HASH_MISMATCH && strncmp(reason,"stale",5u) == 0);
	settle();
	record();
	assert(chmod(pack,0640) == 0);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_HASH_MISMATCH && strncmp(reason,"stale",5u) == 0);
	settle();
	record();
	fd = open(pack,O_RDWR);
	assert(fd >= 0 && pread(fd,&byte,1u,77u) == 1 && pwrite(fd,&byte,1u,77u) == 1 && close(fd) == 0);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_HASH_MISMATCH && strncmp(reason,"stale",5u) == 0);
	settle();
	record();
	assert(snprintf(copy,sizeof(copy),"%s.copy",pack) > 0);
	{
		static uint8_t block[1u << 20];
		FILE *in = fopen(pack,"rb"),*out = fopen(copy,"wb");
		size_t count;
		assert(in != 0 && out != 0);
		while ( (count = fread(block,1u,sizeof(block),in)) != 0u )
			assert(fwrite(block,1u,count,out) == count);
		assert(fclose(in) == 0 && fclose(out) == 0);
	}
	assert(check(copy,sha,0,&reason) == SPARK_STATUS_NOT_FOUND);
	assert(rename(copy,pack) == 0);
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_HASH_MISMATCH && strncmp(reason,"stale",5u) == 0);
	settle();
	record();
	assert(check(pack,sha,0,&reason) == SPARK_STATUS_OK);
	puts("PASS invalidation: restamp, chmod, in-place write and copy-then-rename all make the receipt stale");
}

static void check_symlink_and_fallback(void)
{
	char link_path[300],directory[300],state[300],fallback_path[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],primary_path[SPARK_WEIGHTD_RECEIPT_PATH_BYTES];
	char state_root[300];
	struct stat info;
	const char *reason = 0;
	int32_t fd;
	assert(snprintf(link_path,sizeof(link_path),"%s/link.sp",root) > 0);
	assert(symlink(pack,link_path) == 0);
	assert(check(link_path,sha,0,&reason) == SPARK_STATUS_OK);
	assert(unlink(link_path) == 0);
	if ( geteuid() == 0 )
	{
		puts("SKIP fallback: root ignores directory write permission");
		return;
	}
	assert(snprintf(directory,sizeof(directory),"%s/readonly",root) > 0);
	assert(snprintf(state_root,sizeof(state_root),"%s/state",root) > 0);
	assert(mkdir(directory,0755) == 0);
	assert(snprintf(state,sizeof(state),"%s/pack.sp",directory) > 0);
	assert(rename(pack,state) == 0);
	assert(unlink(receipt) == 0);
	assert(chmod(directory,0555) == 0);
	assert(setenv("XDG_STATE_HOME",state_root,1) == 0);
	settle();
	fd = open(state,O_RDONLY);
	assert(fd >= 0 && fstat(fd,&info) == 0);
	assert(SparkWeightdReceiptCheck(state,fd,sha,0,&reason) == SPARK_STATUS_NOT_FOUND);
	assert(SparkWeightdReceiptRecord(state,fd,&info,sha,ck,"test") == SPARK_STATUS_OK);
	assert(SparkWeightdReceiptCheck(state,fd,sha,0,&reason) == SPARK_STATUS_OK);
	assert(SparkWeightdReceiptLocate(state,&info,primary_path,fallback_path) == SPARK_STATUS_OK);
	assert(access(primary_path,F_OK) != 0 && access(fallback_path,F_OK) == 0);
	assert(strncmp(fallback_path,state_root,strlen(state_root)) == 0);
	assert(close(fd) == 0);
	assert(unlink(fallback_path) == 0);
	assert(chmod(directory,0755) == 0);
	assert(rename(state,pack) == 0);
	assert(rmdir(directory) == 0);
	{
		char path[400];
		assert(snprintf(path,sizeof(path),"%s/sparkpipe/verified",state_root) > 0 && rmdir(path) == 0);
		assert(snprintf(path,sizeof(path),"%s/sparkpipe",state_root) > 0 && rmdir(path) == 0);
		assert(rmdir(state_root) == 0);
	}
	assert(unsetenv("XDG_STATE_HOME") == 0);
	settle();
	puts("PASS location: a symlinked pack uses the real pack's receipt; a read-only pack dir falls back to the state dir");
}

static void *race(void *argument)
{
	Racer *racer = (Racer *)argument;
	struct stat info;
	const char *reason;
	uint32_t i;
	int32_t fd = open(racer->pack,O_RDONLY);
	assert(fd >= 0 && fstat(fd,&info) == 0);
	for (i=0u; i<ROUNDS; i++)
	{
		if ( SparkWeightdReceiptRecord(racer->pack,fd,&info,racer->sha,racer->ck,"race") != SPARK_STATUS_OK )
			racer->failures++;
		if ( SparkWeightdReceiptCheck(racer->pack,fd,racer->sha,0,&reason) != SPARK_STATUS_OK )
			racer->failures++;
	}
	assert(close(fd) == 0);
	return(0);
}

static void check_concurrent(void)
{
	pthread_t threads[THREADS];
	Racer racers[THREADS];
	struct dirent *entry;
	DIR *listing;
	uint32_t i;
	for (i=0u; i<THREADS; i++)
	{
		racers[i].pack = pack;
		racers[i].sha = sha;
		racers[i].ck = ck;
		racers[i].failures = 0u;
		assert(pthread_create(&threads[i],0,race,&racers[i]) == 0);
	}
	for (i=0u; i<THREADS; i++)
	{
		assert(pthread_join(threads[i],0) == 0);
		assert(racers[i].failures == 0u);
	}
	listing = opendir(root);
	assert(listing != 0);
	while ( (entry = readdir(listing)) != 0 )
		assert(strstr(entry->d_name,".tmp.") == 0);
	assert(closedir(listing) == 0);
	puts("PASS concurrent: 8 writers x 16 atomic replacements while reading; every read saw a whole valid receipt, no temp files left");
}

int main(void)
{
	assert(mkdtemp(root) != 0);
	assert(snprintf(pack,sizeof(pack),"%s/pack.sp",root) > 0);
	assert(snprintf(receipt,sizeof(receipt),"%s/pack.sp%s",root,SPARK_WEIGHTD_RECEIPT_SUFFIX) > 0);
	write_pack(pack,7u);
	assert(SparkSha256File(pack,sha) == SPARK_STATUS_OK);
	{
		char resolved[4096];
		assert(realpath(root,resolved) != 0);
		assert(snprintf(pack,sizeof(pack),"%s/pack.sp",resolved) > 0);
		assert(snprintf(receipt,sizeof(receipt),"%s/pack.sp%s",resolved,SPARK_WEIGHTD_RECEIPT_SUFFIX) > 0);
	}
	check_stream();
	check_record_rules();
	check_malformed();
	check_invalidation();
	check_symlink_and_fallback();
	check_concurrent();
	assert(unlink(receipt) == 0 && unlink(pack) == 0 && rmdir(root) == 0);
	puts("PASS weightd receipt: verify-once receipts are atomic, stat-bound, digest-bound and fail closed");
	return(0);
}
