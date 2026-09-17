#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include "sparkpipe/spark_weightd.h"

#define SPARK_MESH_TEST_OPS 64u
#define SPARK_MESH_TEST_PAYLOAD_BYTES 4096u
#define SPARK_MESH_TEST_FIXED_DEADLINE_NS (10ull * 1000ull * 1000ull * 1000ull)
#define SPARK_MESH_TEST_LEGACY_DEADLINE_NS (2ull * 1000ull * 1000ull * 1000ull)
#define SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES (64u * 1024u)
#define SPARK_MESH_TEST_ACKS_OFFSET \
	(SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + \
	 SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * \
	 SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES)

static uint64_t now_ns(void)
{
	struct timespec ts;
	if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0 )
		return 0ull;
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int backing_open(uint64_t bytes)
{
	int fd;
#if defined(__linux__)
	fd = (int)syscall(SYS_memfd_create,"spark-mesh-doorbell-test",0u);
	if ( fd >= 0 )
	{
		if ( ftruncate(fd,(off_t)bytes) == 0 )
			return fd;
		(void)close(fd);
	}
#endif
	{
		char path[] = "/tmp/spark_mesh_doorbell_XXXXXX";
		fd = mkstemp(path);
		if ( fd < 0 )
			return -1;
		(void)unlink(path);
		if ( ftruncate(fd,(off_t)bytes) != 0 )
		{
			(void)close(fd);
			return -1;
		}
	}
	return fd;
}

static void doorbell_child(int fd)
{
	volatile uint8_t *map;
	volatile uint64_t *entry;
	volatile uint64_t *acks;
	uint64_t seq;
	uint64_t bytes;
	uint64_t slot;
	uint64_t posted;
	uint64_t acked;
	uint64_t offset;
	uint64_t word;
	uint64_t end_word;
	uint32_t tries;
	uint32_t stable;
	map = (volatile uint8_t *)mmap(0,SPARK_WEIGHTD_MESH_REGION_BYTES,
		PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
	if ( map == MAP_FAILED )
		_exit(2);
	alarm(30);
	entry = (volatile uint64_t *)(map +
		SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0,0));
	acks = (volatile uint64_t *)(map + SPARK_MESH_TEST_ACKS_OFFSET);
	posted = 0ull;
	acked = 0ull;
	for (;;)
	{
		stable = 0u;
		for ( tries = 0u; tries < 64u && stable == 0u; tries++ )
		{
			seq = entry[0];
			bytes = entry[1];
			slot = entry[2];
			__sync_synchronize();
			if ( seq == entry[0] && bytes == entry[1] &&
				slot == entry[2] )
				stable = 1u;
		}
		if ( stable == 0u || seq == 0ull || bytes == 0ull ||
			seq == posted )
			continue;
		if ( slot >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND ||
			bytes + 8ull > SPARK_WEIGHTD_MESH_SLOT_BYTES )
			_exit(3);
		for ( offset = 0ull; offset + 8ull <= bytes; offset += 8ull )
		{
			memcpy(&word,(const void *)(map +
				slot * SPARK_WEIGHTD_MESH_SLOT_BYTES + offset),8);
			if ( word != (seq * 2654435761ull + offset) )
				_exit(4);
		}
		memcpy(&end_word,(const void *)(map +
			slot * SPARK_WEIGHTD_MESH_SLOT_BYTES + bytes),8);
		if ( end_word != seq )
			_exit(5);
		acks[seq - 1ull] = seq;
		posted = seq;
		acked++;
		if ( acked == (uint64_t)SPARK_MESH_TEST_OPS )
			_exit(0);
	}
}

static void publish_op(volatile uint8_t *base, uint64_t seq)
{
	volatile uint64_t *entry;
	volatile uint8_t *slot;
	uint64_t slot_index;
	uint64_t offset;
	uint64_t word;
	slot_index = (seq - 1ull) % SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
	slot = base + slot_index * SPARK_WEIGHTD_MESH_SLOT_BYTES;
	entry = (volatile uint64_t *)(base +
		SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0,0));
	for ( offset = 0ull;
		offset + 8ull <= SPARK_MESH_TEST_PAYLOAD_BYTES; offset += 8ull )
	{
		word = seq * 2654435761ull + offset;
		memcpy((void *)(slot + offset),&word,8);
	}
	memcpy((void *)(slot + SPARK_MESH_TEST_PAYLOAD_BYTES),&seq,8);
	__sync_synchronize();
	entry[2] = slot_index;
	entry[1] = SPARK_MESH_TEST_PAYLOAD_BYTES;
	entry[0] = seq;
	__sync_synchronize();
}

static uint8_t *legacy_client_map(int fd, uint64_t *map_bytes_out)
{
	volatile uint8_t *raw;
	uintptr_t aligned;
	uintptr_t delta;
	uint64_t map_bytes;
	uint32_t try_index;
	map_bytes = SPARK_WEIGHTD_MESH_REGION_BYTES +
		SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES;
	*map_bytes_out = map_bytes;
	for ( try_index = 0u; try_index < 64u; try_index++ )
	{
		raw = (volatile uint8_t *)mmap(0,map_bytes,
			PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
		if ( raw == MAP_FAILED )
			return 0;
		aligned = ((uintptr_t)raw +
			SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES - 1u) &
			~(uintptr_t)(SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES - 1u);
		delta = aligned - (uintptr_t)raw;
		if ( delta != 0u )
			return (uint8_t *)aligned;
		(void)munmap((void *)raw,map_bytes);
	}
	*map_bytes_out = map_bytes + SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES;
	raw = (volatile uint8_t *)mmap(0,*map_bytes_out,
		PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
	if ( raw == MAP_FAILED )
	{
		*map_bytes_out = 0ull;
		return 0;
	}
	return (uint8_t *)((uintptr_t)raw +
		SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES);
}

static int wait_ack(volatile uint8_t *base, uint64_t seq, uint64_t deadline_ns)
{
	volatile uint64_t *acks;
	uint64_t deadline;
	acks = (volatile uint64_t *)(base + SPARK_MESH_TEST_ACKS_OFFSET);
	deadline = now_ns() + deadline_ns;
	while ( acks[seq - 1ull] != seq )
	{
		if ( now_ns() >= deadline )
			return 0;
	}
	return 1;
}

static int run_leg(uint32_t fixed_mode, uint64_t *delivered_out)
{
	int fd;
	int leg_ok;
	int child_status;
	pid_t child;
	uint64_t deadline_ns;
	uint64_t map_bytes;
	uint8_t *base;
	uint8_t *raw;
	uint64_t seq;
	fd = backing_open(SPARK_WEIGHTD_MESH_REGION_BYTES +
		SPARK_MESH_TEST_LEGACY_ROUND_UP_BYTES);
	if ( fd < 0 )
		return 0;
	child = fork();
	if ( child < 0 )
	{
		(void)close(fd);
		return 0;
	}
	if ( child == 0 )
		doorbell_child(fd);
	leg_ok = 0;
	raw = 0;
	base = 0;
	if ( fixed_mode != 0u )
	{
		map_bytes = SPARK_WEIGHTD_MESH_REGION_BYTES;
		raw = mmap(0,map_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
		base = raw != (uint8_t *)MAP_FAILED ? raw : 0;
		deadline_ns = SPARK_MESH_TEST_FIXED_DEADLINE_NS;
	}
	else
	{
		base = legacy_client_map(fd,&map_bytes);
		deadline_ns = SPARK_MESH_TEST_LEGACY_DEADLINE_NS;
	}
	if ( base != 0 )
	{
		uint64_t acked_ops;
		acked_ops = 0ull;
		for ( seq = 1ull; seq <= (uint64_t)SPARK_MESH_TEST_OPS; seq++ )
		{
			publish_op(base,seq);
			if ( wait_ack(base,seq,deadline_ns) == 0 )
				break;
			acked_ops = seq;
		}
		*delivered_out = acked_ops;
		if ( fixed_mode != 0u )
			leg_ok = acked_ops == (uint64_t)SPARK_MESH_TEST_OPS;
		else
			leg_ok = acked_ops == 0ull;
	}
	if ( fixed_mode != 0u && leg_ok != 0 )
		(void)waitpid(child,&child_status,0);
	else
	{
		(void)kill(child,SIGKILL);
		(void)waitpid(child,&child_status,0);
	}
	if ( fixed_mode != 0u && leg_ok != 0 )
		leg_ok = WIFEXITED(child_status) != 0 &&
			WEXITSTATUS(child_status) == 0;
	if ( fixed_mode != 0u )
	{
		if ( raw != 0 && raw != (uint8_t *)MAP_FAILED )
			(void)munmap(raw,map_bytes);
	}
	else if ( base != 0 )
		(void)munmap(base,map_bytes);
	(void)close(fd);
	return leg_ok;
}

int main(void)
{
	uint64_t delivered_fixed;
	uint64_t delivered_legacy;
	int fixed_ok;
	int legacy_ok;
	delivered_fixed = 0ull;
	delivered_legacy = 0ull;
	fixed_ok = run_leg(1u,&delivered_fixed);
	legacy_ok = run_leg(0u,&delivered_legacy);
	printf("mesh-doorbell fixed-mode delivered=%llu/%u verify=%s\n",
		(unsigned long long)delivered_fixed,SPARK_MESH_TEST_OPS,
		fixed_ok != 0 ? "pass" : "FAIL");
	printf("mesh-doorbell legacy-align delivered=%llu/%u negative-control=%s\n",
		(unsigned long long)delivered_legacy,SPARK_MESH_TEST_OPS,
		legacy_ok != 0 ? "pass" : "FAIL");
	if ( fixed_ok == 0 || legacy_ok == 0 )
		return 1;
	return 0;
}
