#define _POSIX_C_SOURCE 200112L
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SIM_HOP_COUNT 13u
#define SIM_FRAME_BYTES 12800u
#define SIM_TOKEN_COUNT 200u
#define SIM_PORT_BASE 42400u

typedef struct SimHop
{
	pthread_t sender_thread;
	pthread_t receiver_thread;
	pthread_mutex_t lock;
	pthread_cond_t signal;
	int32_t out_fd;
	int32_t in_fd;
	uint32_t hop_index;
	uint32_t pending;
	uint8_t frame[SIM_FRAME_BYTES];
	struct SimHop *next;
	uint64_t deliver_ns[SIM_TOKEN_COUNT];
	volatile uint32_t delivered;
} SimHop;

static uint64_t SimMonotonicNs(void)
{
	struct timespec ts;
	(void)clock_gettime(CLOCK_MONOTONIC,&ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int32_t SimWriteAll(int32_t fd,const uint8_t *data,uint64_t bytes)
{
	uint64_t sent;
	sent = 0u;
	while (sent < bytes)
	{
		int64_t rc;
		rc = write(fd,data + sent,(size_t)(bytes - sent));
		if (rc <= 0)
			return -1;
		sent += (uint64_t)rc;
	}
	return 0;
}

static int32_t SimReadAll(int32_t fd,uint8_t *data,uint64_t bytes)
{
	uint64_t got;
	got = 0u;
	while (got < bytes)
	{
		int64_t rc;
		rc = read(fd,data + got,(size_t)(bytes - got));
		if (rc <= 0)
			return -2;
		got += (uint64_t)rc;
	}
	return 0;
}

static void *SimSenderMain(void *context)
{
	SimHop *hop;
	hop = (SimHop *)context;
	for (;;)
	{
		pthread_mutex_lock(&hop->lock);
		while (hop->pending == 0u)
			pthread_cond_wait(&hop->signal,&hop->lock);
		hop->pending -= 1u;
		pthread_mutex_unlock(&hop->lock);
		if (SimWriteAll(hop->out_fd,hop->frame,SIM_FRAME_BYTES) != 0)
			return 0;
	}
	return 0;
}

static void SimPost(SimHop *hop)
{
	pthread_mutex_lock(&hop->lock);
	hop->pending += 1u;
	pthread_cond_signal(&hop->signal);
	pthread_mutex_unlock(&hop->lock);
}

static void *SimReceiverMain(void *context)
{
	SimHop *hop;
	hop = (SimHop *)context;
	for (;;)
	{
		if (SimReadAll(hop->in_fd,hop->frame,SIM_FRAME_BYTES) != 0)
			return 0;
		if (hop->delivered < SIM_TOKEN_COUNT)
			hop->deliver_ns[hop->delivered] = SimMonotonicNs();
		hop->delivered += 1u;
		if (hop->next != 0)
			SimPost(hop->next);
	}
	return 0;
}

static int32_t SimListenAccept(uint16_t port,int32_t *accepted_out,int32_t *listen_out)
{
	struct sockaddr_in address;
	int32_t listen_fd;
	int32_t option;
	listen_fd = socket(AF_INET,SOCK_STREAM,0);
	if (listen_fd < 0)
		return -3;
	option = 1;
	(void)setsockopt(listen_fd,SOL_SOCKET,SO_REUSEADDR,&option,sizeof(option));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	address.sin_port = htons(port);
	if (bind(listen_fd,(struct sockaddr *)&address,sizeof(address)) != 0 ||
		listen(listen_fd,1) != 0)
		return -4;
	*listen_out = listen_fd;
	*accepted_out = -1;
	return 0;
}

static int32_t SimConnect(uint16_t port)
{
	struct sockaddr_in address;
	int32_t fd;
	int32_t option;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if (fd < 0)
		return -5;
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	address.sin_port = htons(port);
	if (connect(fd,(struct sockaddr *)&address,sizeof(address)) != 0)
		return -6;
	option = 1;
	(void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&option,sizeof(option));
	return fd;
}

int main(void)
{
	static SimHop hops[SIM_HOP_COUNT];
	int32_t listen_fds[SIM_HOP_COUNT];
	uint64_t start_ns[SIM_TOKEN_COUNT];
	uint64_t total_ns;
	uint64_t hop_sum_ns;
	uint32_t hop_index;
	uint32_t token_index;

	for (hop_index = 0u; hop_index < SIM_HOP_COUNT; ++hop_index)
	{
		int32_t accepted;
		if (SimListenAccept((uint16_t)(SIM_PORT_BASE + hop_index),&accepted,&listen_fds[hop_index]) != 0)
			return 1;
	}
	for (hop_index = 0u; hop_index < SIM_HOP_COUNT; ++hop_index)
	{
		hops[hop_index].hop_index = hop_index;
		hops[hop_index].pending = 0u;
		hops[hop_index].delivered = 0u;
		pthread_mutex_init(&hops[hop_index].lock,0);
		pthread_cond_init(&hops[hop_index].signal,0);
		hops[hop_index].out_fd = SimConnect((uint16_t)(SIM_PORT_BASE + hop_index));
		if (hops[hop_index].out_fd < 0)
			return 2;
		hops[hop_index].in_fd = accept(listen_fds[hop_index],0,0);
		if (hops[hop_index].in_fd < 0)
			return 3;
		{
			int32_t option;
			option = 1;
			(void)setsockopt(hops[hop_index].in_fd,IPPROTO_TCP,TCP_NODELAY,&option,sizeof(option));
		}
	}
	for (hop_index = 0u; hop_index < SIM_HOP_COUNT; ++hop_index)
	{
		hops[hop_index].next = hop_index + 1u < SIM_HOP_COUNT ? &hops[hop_index + 1u] : 0;
		pthread_create(&hops[hop_index].receiver_thread,0,SimReceiverMain,&hops[hop_index]);
		pthread_create(&hops[hop_index].sender_thread,0,SimSenderMain,&hops[hop_index]);
	}
	{
		struct timespec warm;
		warm.tv_sec = 0;
		warm.tv_nsec = 50000000;
		(void)nanosleep(&warm,0);
	}
	for (token_index = 0u; token_index < SIM_TOKEN_COUNT; ++token_index)
	{
		start_ns[token_index] = SimMonotonicNs();
		SimPost(&hops[0]);
		while (hops[SIM_HOP_COUNT - 1u].delivered <= token_index)
			(void)0;
	}
	total_ns = 0u;
	for (token_index = SIM_TOKEN_COUNT / 2u; token_index < SIM_TOKEN_COUNT; ++token_index)
		total_ns += hops[SIM_HOP_COUNT - 1u].deliver_ns[token_index] - start_ns[token_index];
	hop_sum_ns = total_ns / (uint64_t)(SIM_TOKEN_COUNT / 2u);
	printf("sim_hops=%u frame_bytes=%u tokens=%u\n",SIM_HOP_COUNT,SIM_FRAME_BYTES,SIM_TOKEN_COUNT);
	printf("chain_latency_us_avg=%llu per_hop_us_avg=%llu\n",
		(unsigned long long)(hop_sum_ns / 1000ull),
		(unsigned long long)(hop_sum_ns / 1000ull / SIM_HOP_COUNT));
	return 0;
}
