#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sys/types.h>
#ifdef __APPLE__
typedef off_t off64_t;
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef ssize_t (*pread_fn)(int, void *, size_t, off_t);
typedef int (*memcpy_fn)(void *, const void *, size_t, int);
typedef ssize_t (*pread64_fn)(int, void *, size_t, off64_t);
typedef ssize_t (*pread64_chk_fn)(int, void *, size_t, off64_t, size_t);

typedef struct SpineInstrumentSite
{
	uint64_t calls;
	uint64_t bytes;
	uint64_t nanoseconds;
} SpineInstrumentSite;

static pread_fn real_pread;
static pread64_fn real_pread64;
static pread64_chk_fn real_pread64_chk;
static memcpy_fn real_memcpy;
static SpineInstrumentSite pread_site;
static SpineInstrumentSite memcpy_site;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int reporting;

static uint64_t now_ns(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void dump(const char *tag, const SpineInstrumentSite *site)
{
	double megabytes, seconds;
	if ( site->calls == 0u )
		return;
	megabytes = (double)site->bytes / (1024.0 * 1024.0);
	seconds = (double)site->nanoseconds / 1e9;
	fprintf(stderr,
		"spine_instrument %s calls=%llu bytes=%llu (%.1f MiB) seconds=%.3f throughput=%.1f MB/s\n",
		tag, (unsigned long long)site->calls, (unsigned long long)site->bytes,
		megabytes, seconds, seconds > 0.0 ? megabytes / seconds : 0.0);
}

static void spine_instrument_report(void)
{
	pthread_mutex_lock(&lock);
	if ( reporting != 0 )
	{
		pthread_mutex_unlock(&lock);
		return;
	}
	reporting = 1;
	pthread_mutex_unlock(&lock);
	dump("pread", &pread_site);
	dump("cudaMemcpy", &memcpy_site);
}

static void spine_instrument_init(void)
{
	real_pread = (pread_fn)dlsym(RTLD_NEXT, "pread");
	real_pread64 = (pread64_fn)dlsym(RTLD_NEXT, "pread64");
	real_pread64_chk = (pread64_chk_fn)dlsym(RTLD_NEXT, "__pread64_chk");
	real_memcpy = (memcpy_fn)dlsym(RTLD_NEXT, "cudaMemcpy");
	atexit(spine_instrument_report);
}

ssize_t pread(int fd, void *buffer, size_t count, off_t offset)
{
	uint64_t begin, end;
	ssize_t result;
	if ( real_pread == 0 )
		spine_instrument_init();
	begin = now_ns();
	result = real_pread(fd, buffer, count, offset);
	end = now_ns();
	pthread_mutex_lock(&lock);
	pread_site.calls += 1u;
	pread_site.bytes += (uint64_t)(result > 0 ? result : 0);
	pread_site.nanoseconds += end - begin;
	if ( (pread_site.calls % 65536ull) == 0ull )
	{
		dump("pread", &pread_site);
		dump("cudaMemcpy", &memcpy_site);
	}
	pthread_mutex_unlock(&lock);
	return(result);
}

ssize_t pread64(int fd, void *buffer, size_t count, off64_t offset)
{
	uint64_t begin, end;
	ssize_t result;
	if ( real_pread64 == 0 )
		spine_instrument_init();
	begin = now_ns();
	result = real_pread64(fd, buffer, count, offset);
	end = now_ns();
	pthread_mutex_lock(&lock);
	pread_site.calls += 1u;
	pread_site.bytes += (uint64_t)(result > 0 ? result : 0);
	pread_site.nanoseconds += end - begin;
	if ( (pread_site.calls % 65536ull) == 0ull )
	{
		dump("pread", &pread_site);
		dump("cudaMemcpy", &memcpy_site);
	}
	pthread_mutex_unlock(&lock);
	return(result);
}

ssize_t __pread64_chk(int fd, void *buffer, size_t count, off64_t offset, size_t bufsize)
{
	uint64_t begin, end;
	ssize_t result;
	if ( real_pread64_chk == 0 )
		spine_instrument_init();
	begin = now_ns();
	result = real_pread64_chk(fd, buffer, count, offset, bufsize);
	end = now_ns();
	pthread_mutex_lock(&lock);
	pread_site.calls += 1u;
	pread_site.bytes += (uint64_t)(result > 0 ? result : 0);
	pread_site.nanoseconds += end - begin;
	if ( (pread_site.calls % 65536ull) == 0ull )
	{
		dump("pread", &pread_site);
		dump("cudaMemcpy", &memcpy_site);
	}
	pthread_mutex_unlock(&lock);
	return(result);
}

int cudaMemcpy(void *destination, const void *source, size_t count, int kind)
{
	uint64_t begin, end;
	int result;
	if ( real_memcpy == 0 )
		spine_instrument_init();
	begin = now_ns();
	result = real_memcpy(destination, source, count, kind);
	end = now_ns();
	if ( kind == 1 )
	{
		pthread_mutex_lock(&lock);
		memcpy_site.calls += 1u;
		memcpy_site.bytes += (uint64_t)count;
		memcpy_site.nanoseconds += end - begin;
		pthread_mutex_unlock(&lock);
	}
	return(result);
}

__attribute__((constructor)) static void spine_instrument_constructor(void)
{
	spine_instrument_init();
}
