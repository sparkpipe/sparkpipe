#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROBE_BUFFER_BYTES (1ull << 30)
#define PROBE_THREAD_COUNT 20
#define PROBE_WARM_ROUNDS 2
#define PROBE_ROUNDS 12

typedef struct ProbeContext
{
	uint64_t *data;
	uint64_t elements;
	uint64_t sink;
} ProbeContext;

static double probe_now_seconds(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t probe_sum(const uint64_t *data, uint64_t elements)
{
	uint64_t total_a = 0;
	uint64_t total_b = 0;
	uint64_t i;

	for (i = 0; i + 2u <= elements; i += 2u)
	{
		total_a += data[i];
		total_b += data[i + 1u];
	}
	return total_a + total_b;
}

static void probe_fill(uint64_t *data, uint64_t elements, uint64_t seed)
{
	uint64_t i;

	for (i = 0; i < elements; i++)
		data[i] = seed + i;
}

static void *probe_read_worker(void *raw_context)
{
	ProbeContext *context = raw_context;

	context->sink = probe_sum(context->data, context->elements);
	return context;
}

static void *probe_write_worker(void *raw_context)
{
	ProbeContext *context = raw_context;

	probe_fill(context->data, context->elements, 0x9e3779b97f4a7c15ull + context->sink);
	return context;
}

static double probe_run(void *(*worker)(void *), uint64_t *data, uint64_t elements_per_thread)
{
	ProbeContext contexts[PROBE_THREAD_COUNT];
	pthread_t threads[PROBE_THREAD_COUNT];
	double start;
	double seconds;
	uint32_t index;

	for (index = 0; index < PROBE_THREAD_COUNT; index++)
	{
		contexts[index].data = data + (uint64_t)index * elements_per_thread;
		contexts[index].elements = elements_per_thread;
		contexts[index].sink = (uint64_t)index;
	}
	start = probe_now_seconds();
	for (index = 0; index < PROBE_THREAD_COUNT; index++)
		(void)pthread_create(&threads[index], NULL, worker, &contexts[index]);
	for (index = 0; index < PROBE_THREAD_COUNT; index++)
		(void)pthread_join(threads[index], NULL);
	seconds = probe_now_seconds() - start;
	return seconds;
}

int32_t main(void)
{
	uint64_t elements_per_thread = PROBE_BUFFER_BYTES / sizeof(uint64_t) / PROBE_THREAD_COUNT;
	uint64_t *data = NULL;
	uint64_t sink = 0;
	double read_seconds;
	double write_seconds;
	double best_read = 0.0;
	double best_write = 0.0;
	uint32_t round;

	if (posix_memalign((void **)&data, 4096, PROBE_BUFFER_BYTES) != 0)
	{
		(void)fprintf(stderr, "alloc failed\n");
		return -1;
	}
	memset(data, 1, PROBE_BUFFER_BYTES);
	for (round = 0; round < PROBE_WARM_ROUNDS + PROBE_ROUNDS; round++)
	{
		read_seconds = probe_run(probe_read_worker, data, elements_per_thread);
		write_seconds = probe_run(probe_write_worker, data, elements_per_thread);
		if (round < PROBE_WARM_ROUNDS)
			continue;
		if (read_seconds > 0.0 && (PROBE_BUFFER_BYTES / read_seconds) > best_read)
			best_read = PROBE_BUFFER_BYTES / read_seconds;
		if (write_seconds > 0.0 && (PROBE_BUFFER_BYTES / write_seconds) > best_write)
			best_write = PROBE_BUFFER_BYTES / write_seconds;
		sink += data[0];
	}
	(void)printf("sink=%llu read_gb_per_s=%.2f write_gb_per_s=%.2f buffer_gib=%.0f threads=%u rounds=%u\n",
		(unsigned long long)sink,
		best_read / 1e9,
		best_write / 1e9,
		(double)PROBE_BUFFER_BYTES / (double)(1ull << 30),
		PROBE_THREAD_COUNT,
		PROBE_ROUNDS);
	free(data);
	return 0;
}
