#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"

static void Completion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	uint32_t *done = (uint32_t *)context;
	(void)completion;
	*done = 1u;
}

static SparkStatus CombineBf16(void *c, void *d, const void *s, uint32_t n, uint32_t dim, void *st) { (void)c;(void)d;(void)s;(void)n;(void)dim;(void)st; return SPARK_STATUS_OK; }
static SparkStatus CombineRelay(void *c, void *d, const void *s, void *r, uint32_t n, uint32_t dim, void *st) { (void)c;(void)d;(void)s;(void)r;(void)n;(void)dim;(void)st; return SPARK_STATUS_OK; }
static SparkStatus CombineTp4(void *c, void *d, const void *const *rd, uint32_t rank, uint32_t n, uint32_t dim, void *st) { (void)c;(void)d;(void)rd;(void)rank;(void)n;(void)dim;(void)st; return SPARK_STATUS_OK; }
static SparkStatus CombineU64(void *c, uint64_t *d, const uint64_t *s, uint32_t n, void *st) { (void)c;(void)d;(void)s;(void)n;(void)st; return SPARK_STATUS_OK; }

int main(int argc, char **argv)
{
	SparkTpDeviceCollectiveConfig config;
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveCreditBinding bindings[64];
	SparkTpDeviceCollectiveSubmission submission;
	static const char *rail[2][16] = {
		{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13", 0,0,0,0,0,0,0,0,0,0,0,0 },
		{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13", 0,0,0,0,0,0,0,0,0,0,0,0 } };
	cudaStream_t stream;
	void *send_buf, *recv_buf, *partial;
	void *send_device, *recv_device, *send_transport, *recv_transport;
	void *host_send, *host_recv;
	uint32_t rank, route, credit, route_count, memory_mode, iterations, i;
	uint64_t start, elapsed;
	struct timespec ts;
	SparkStatus status;
	if ( argc != 3 ) return(2);
	rank = (uint32_t)strtoul(argv[1], 0, 0);
	iterations = (uint32_t)strtoul(argv[2], 0, 0);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cudaMalloc(&partial, 5120u * 2u);
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config.tp_degree = 4u;
	config.tp_rank = rank;
	config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	config.credit_count = 4u;
	config.local_hidden_dimension = 5120u;
	config.max_active_sequence_count = 8u;
	config.connect_timeout_milli = 60000u;
	config.operation_timeout_milli = 60000u;
	config.control_port_base = 59700u;
	config.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING | SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL | SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
	config.rail_count = 2u;
	config.direct_all_to_all_max_payload_bytes = 81920u;
	config.split_ring_min_payload_bytes = 655360u;
	config.step_rail_indices[0] = 0u;
	config.step_rail_indices[1] = 1u;
	config.step_rail_indices[2] = 1u;
	config.collective_identifier = 0x513630545033ull;
	config.backend_module_path = "libhidden_transport.so";
	config.local_host = rail[0][rank];
	for (i = 0u; i < 4u; i++) config.rank_hosts[i] = rail[0][i];
	for (i = 0u; i < 2u; i++) memcpy(config.rail_rank_hosts[i], rail[i], sizeof(rail[i]));
	config.registration_cuda_stream = stream;
	config.combine_bf16_function = CombineBf16;
	config.combine_relay_bf16_function = CombineRelay;
	config.combine_tp4_bf16_function = CombineTp4;
	config.combine_u64_max_function = CombineU64;
	status = SparkTpDeviceCollectiveProbeMemoryMode(config.backend_kind, config.backend_module_path, &memory_mode);
	if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "probe %d\n", (int)status); return(1); }
	host_send = 0;
	host_recv = 0;
	if ( memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		if ( cudaHostAlloc(&host_send, 131072u * 64u, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
			cudaHostAlloc(&host_recv, 131072u * 64u, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
			return(1);
		if ( cudaHostGetDevicePointer(&send_device, host_send, 0u) != cudaSuccess ||
			cudaHostGetDevicePointer(&recv_device, host_recv, 0u) != cudaSuccess )
			return(1);
		send_buf = send_device;
		recv_buf = recv_device;
		send_transport = host_send;
		recv_transport = host_recv;
	}
	else
	{
		cudaMalloc(&send_buf, 131072u * 64u);
		cudaMalloc(&recv_buf, 131072u * 64u);
		send_transport = send_buf;
		recv_transport = recv_buf;
	}
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&config, &route_count);
	for (route = 0u; route < route_count; route++)
		for (credit = 0u; credit < config.credit_count; credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *b = &bindings[route * config.credit_count + credit];
			memset(b, 0, sizeof(*b));
			b->step_index = route;
			b->credit_index = credit;
			b->send_device = (uint8_t *)send_buf + (route * config.credit_count + credit) * 131072u;
			b->receive_device = (uint8_t *)recv_buf + (route * config.credit_count + credit) * 131072u;
			b->send_transport = (uint8_t *)send_transport + (route * config.credit_count + credit) * 131072u;
			b->receive_transport = (uint8_t *)recv_transport + (route * config.credit_count + credit) * 131072u;
			b->flags = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
		}
	config.credit_bindings = bindings;
	config.credit_binding_count = route_count * config.credit_count;
	status = SparkTpDeviceCollectiveCreate(&config, &collective);
	if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "create %d\n", (int)status); return(1); }
	printf("rank=%u backend ready mode=%u, warming up...\n", rank, memory_mode);
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = 1u;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.local_device = partial;
	submission.full_device = partial;
	submission.cuda_stream = stream;
	submission.completion_function = Completion;
	for (i = 0u; i < 20u; i++)
	{
		uint32_t done = 0u;
		submission.ordinal = i;
		submission.completion_context = &done;
		if ( SparkTpDeviceCollectiveSubmitBf16(&collective, &submission) != SPARK_STATUS_OK )
		{ fprintf(stderr, "warmup submit failed %u\n", i); return(1); }
		while ( done == 0u ) sched_yield();
	}
	if ( cudaStreamSynchronize(stream) != cudaSuccess ) return(1);
	clock_gettime(CLOCK_MONOTONIC, &ts);
	start = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
	for (i = 0u; i < iterations; i++)
	{
		uint32_t done = 0u;
		submission.ordinal = 20u + i;
		submission.completion_context = &done;
		if ( SparkTpDeviceCollectiveSubmitBf16(&collective, &submission) != SPARK_STATUS_OK )
		{ fprintf(stderr, "submit failed %u\n", i); return(1); }
		while ( done == 0u ) sched_yield();
	}
	if ( cudaStreamSynchronize(stream) != cudaSuccess ) return(1);
	clock_gettime(CLOCK_MONOTONIC, &ts);
	elapsed = ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec) - start;
	printf("rank=%u iterations=%u total_us=%llu per_op_us=%.1f\n", rank, iterations, (unsigned long long)(elapsed / 1000ull), (double)elapsed / 1000.0 / (double)iterations);
	SparkTpDeviceCollectiveDestroy(&collective);
	return(0);
}
