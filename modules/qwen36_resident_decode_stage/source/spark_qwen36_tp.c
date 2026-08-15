#include "spark_qwen36_tp.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen36_model.h"

#define SPARK_QWEN36_TP_TAG "qwen36_tp"

/* Transport-collective configuration. The control port base is the TP4
 * band's transport block; the identifier is unique to this model band. The
 * library resolves through the process LD_LIBRARY_PATH (bundled with the
 * runtime); an env override names a specific file. */
#define SPARK_QWEN36_TP_DEFAULT_TRANSPORT_PORT_BASE 58700u
#define SPARK_QWEN36_TP_DEFAULT_NCCL_PORT_BASE 61620u
#define SPARK_QWEN36_TP_IDENTIFIER 0x513630545031ull
#define SPARK_QWEN36_TP_DEFAULT_TRANSPORT_LIBRARY "libhidden_transport.so"
#define SPARK_QWEN36_TP_DEFAULT_NCCL_LIBRARY "libnccl.so.2"
/* Two rails satisfy the direct-all-to-all topology contract; both map to
 * the 100G switch rail addresses because the pairwise 200.x links are
 * reserved for the split ring, which serving payloads never select. */
static const char *SparkQwen36TpRailHosts[2][SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE] =
{
	{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
	  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
	  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

extern cudaError_t SparkQwen36LaunchAccumAdd(cudaStream_t stream, void *destination, const void *source, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen36LaunchAccumAddRelay(cudaStream_t stream, void *destination, const void *source, void *relay, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen36LaunchAccumAddTp4(cudaStream_t stream, void *destination, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen36LaunchAccumU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);

/* Combine functions fold the staged reduction into the consumer buffer on
 * the operation stream; the backend orders the consumer's stream after
 * them, so the next layer reads the full reduction. */
static SparkStatus SparkQwen36TpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen36LaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen36TpCombineRelayBf16(void *combine_context, void *destination_device, const void *source_device, void *relay_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen36LaunchAccumAddRelay((cudaStream_t)cuda_stream,destination_device,source_device,relay_device,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen36TpCombineTp4Bf16(void *combine_context, void *destination_device, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen36LaunchAccumAddTp4((cudaStream_t)cuda_stream,destination_device,rank_devices,tp_rank,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen36TpCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen36LaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

/* The completion callback stores the operation status and releases the
 * waiting submitter. */
static void SparkQwen36TpPendingCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkQwen36TpPending *pending = (SparkQwen36TpPending *)context;
	if ( pending == 0 || completion == 0 )
		return;
	pending->status = (uint32_t)completion->status;
	atomic_store_explicit(&pending->done,1u,memory_order_release);
}

static SparkStatus SparkQwen36TpSubmit(SparkQwen36TpState *tp, void *buffer, uint32_t count, void *cuda_stream, uint32_t u64_max)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkQwen36TpPending pending;
	SparkStatus status;
	uint32_t spin;
	atomic_init(&pending.done,0u);
	pending.status = (uint32_t)SPARK_STATUS_OK;
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.flags =
		SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = tp->next_ordinal++;
	submission.local_device = buffer;
	submission.full_device = buffer;
	submission.cuda_stream = cuda_stream;
	submission.completion_function = SparkQwen36TpPendingCompletion;
	submission.completion_context = &pending;
	status = u64_max != 0u ? SparkTpDeviceCollectiveSubmitU64Max(&tp->collective,&submission) : SparkTpDeviceCollectiveSubmitBf16(&tp->collective,&submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	/* Wait for the stream-ordered completion: the NCCL backend publishes at
	 * submit time; the transport backend publishes once the reduction is
	 * folded and the consumer stream is ordered after it. */
	spin = 0u;
	while ( atomic_load_explicit(&pending.done,memory_order_acquire) == 0u )
	{
		if ( (spin & 0x3Fu) == 0u )
			sched_yield();
		spin++;
	}
	return (SparkStatus)pending.status;
}

SparkStatus SparkQwen36TpInitialize(
	SparkQwen36TpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	uint32_t pipeline_slot_count,
	void *registration_cuda_stream)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveCreditBinding *bindings;
	const char *backend_name;
	uint32_t transport_backend;
	uint32_t credit,route,route_count,hidden,credit_count,memory_mode;
	uint64_t credit_bytes,offset,total_bytes;
	void *mapped_receive,*mapped_send;
	SparkStatus status;
	uint32_t index;

	if ( tp == 0 || degree == 0u || rank >= degree ||
		max_active_sequence_count == 0u || registration_cuda_stream == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(tp, 0, sizeof(*tp));
	tp->degree = degree;
	tp->rank = rank;
	tp->cuda_stream = registration_cuda_stream;
	if ( (SPARK_QWEN36_MODEL_GDN_QK_DIMENSION % degree) != 0u ||
		(SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION % degree) != 0u ||
		(SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION % degree) != 0u ||
		(SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT % degree) != 0u )
	{
		fprintf(stderr, "%s geometry_undividable degree=%u\n",
			SPARK_QWEN36_TP_TAG, degree);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	tp->gdn_qk_channels = SPARK_QWEN36_MODEL_GDN_QK_DIMENSION / degree;
	tp->gdn_value_channels = SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION / degree;
	tp->gdn_conv_channels =
		2u * tp->gdn_qk_channels + tp->gdn_value_channels;
	tp->gdn_key_heads = SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT / degree;
	tp->gdn_value_heads = SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT / degree;
	tp->attn_query_heads = SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT / degree;
	tp->attn_kv_heads = SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT / degree;
	tp->ffn_intermediate =
		SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION / degree;
	tp->head_rows = SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT / degree;
	if ( degree == 1u )
		return SPARK_STATUS_OK;
	/* GPU-validator escape hatch: one rank validates the whole-stack TP4
	 * path without a peer group. The collective is skipped and the reduces
	 * become no-ops. */
	if ( getenv("SPARK_QWEN36_TP_STANDALONE") != 0 )
	{
		fprintf(stderr, "%s standalone degree=%u rank=%u (collective skipped)\n",
			SPARK_QWEN36_TP_TAG, degree, rank);
		return SPARK_STATUS_OK;
	}
	backend_name = getenv("SPARK_QWEN36_TP_BACKEND");
	transport_backend = backend_name == 0 || strcmp(backend_name,"nccl") != 0 ? 1u : 0u;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.tp_degree = degree;
	configuration.tp_rank = rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.local_hidden_dimension =
		SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = max_active_sequence_count;
	configuration.connect_timeout_milli = 120000u;
	configuration.operation_timeout_milli = 120000u;
	configuration.collective_identifier = SPARK_QWEN36_TP_IDENTIFIER;
	configuration.registration_cuda_stream = registration_cuda_stream;
	configuration.local_host = SparkQwen36TpRailHosts[0][rank];
	if ( transport_backend != 0u )
	{
		configuration.backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
		configuration.backend_module_path = getenv("SPARK_QWEN36_TP_TRANSPORT_LIBRARY") != 0 ? getenv("SPARK_QWEN36_TP_TRANSPORT_LIBRARY") : SPARK_QWEN36_TP_DEFAULT_TRANSPORT_LIBRARY;
		configuration.control_port_base =
			SPARK_QWEN36_TP_DEFAULT_TRANSPORT_PORT_BASE;
		configuration.algorithm_mask =
			SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING |
			SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL |
			SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
		configuration.rail_count = 2u;
		configuration.direct_all_to_all_max_payload_bytes = 81920u;
		configuration.split_ring_min_payload_bytes = 655360u;
		configuration.step_rail_indices[0] = 0u;
		configuration.step_rail_indices[1] = 1u;
		configuration.step_rail_indices[2] = 1u;
		configuration.combine_bf16_function = SparkQwen36TpCombineBf16;
		configuration.combine_relay_bf16_function = SparkQwen36TpCombineRelayBf16;
		configuration.combine_tp4_bf16_function = SparkQwen36TpCombineTp4Bf16;
		configuration.combine_u64_max_function = SparkQwen36TpCombineU64Max;
		configuration.combine_context = tp;
		for (index = 0u; index < 2u; index++)
			memcpy(configuration.rail_rank_hosts[index],SparkQwen36TpRailHosts[index],sizeof(SparkQwen36TpRailHosts[index]));
		for (index = 0u; index < degree; index++)
			configuration.rank_hosts[index] = SparkQwen36TpRailHosts[0][index];
		credit_count = pipeline_slot_count * 2u;
		if ( credit_count == 0u || credit_count > SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT )
			credit_count = 2u;
		configuration.credit_count = credit_count;
	}
	else
	{
		configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
		configuration.backend_module_path = getenv("SPARK_QWEN36_TP_NCCL_LIBRARY") != 0 ? getenv("SPARK_QWEN36_TP_NCCL_LIBRARY") : SPARK_QWEN36_TP_DEFAULT_NCCL_LIBRARY;
		configuration.control_port_base = SPARK_QWEN36_TP_DEFAULT_NCCL_PORT_BASE;
		configuration.credit_count = 1u;
		for (index = 0u; index < degree; index++)
			configuration.rank_hosts[index] = SparkQwen36TpRailHosts[0][index];
	}
	if ( transport_backend != 0u )
	{
		status = SparkTpDeviceCollectiveProbeMemoryMode(
			configuration.backend_kind,configuration.backend_module_path,
			&memory_mode);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "%s probe_memory_mode status=%d\n",
				SPARK_QWEN36_TP_TAG, (int)status);
			return status;
		}
		status = SparkTpDeviceCollectiveCreditBindingRouteCount(
			&configuration,&route_count);
		if ( status != SPARK_STATUS_OK )
			return status;
		total_bytes = 0u;
		for (route = 0u; route < route_count; route++)
		{
			hidden = configuration.local_hidden_dimension;
			credit_bytes = (uint64_t)configuration.max_active_sequence_count *
				hidden * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
			if ( credit_bytes == 0u || total_bytes > UINT64_MAX -
				credit_bytes * configuration.credit_count )
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			total_bytes += credit_bytes * configuration.credit_count;
		}
		status = SPARK_STATUS_OK;
		if ( total_bytes != 0u && cudaMalloc(&tp->credit_send_bf16,(size_t)total_bytes) != cudaSuccess )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status == SPARK_STATUS_OK && total_bytes != 0u && cudaMalloc(&tp->credit_receive_bf16,(size_t)total_bytes) != cudaSuccess )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( total_bytes != 0u && memory_mode ==
			SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
		{
			mapped_receive = 0;
			mapped_send = 0;
			if ( cudaHostAlloc(&tp->host_credit_send_bf16,(size_t)total_bytes,cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess )
				(void)cudaHostAlloc(&tp->host_credit_receive_bf16,(size_t)total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
			if ( tp->host_credit_send_bf16 == 0 || tp->host_credit_receive_bf16 == 0 )
			{
				fprintf(stderr, "%s credit_host_alloc_failed\n", SPARK_QWEN36_TP_TAG);
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			}
			if ( cudaHostGetDevicePointer(&mapped_send,tp->host_credit_send_bf16,0u) == cudaSuccess &&
				cudaHostGetDevicePointer(&mapped_receive,tp->host_credit_receive_bf16,0u) == cudaSuccess )
			{
				cudaFree(tp->credit_send_bf16);
				cudaFree(tp->credit_receive_bf16);
				tp->credit_send_bf16 = mapped_send;
				tp->credit_receive_bf16 = mapped_receive;
			}
		}
		bindings = (SparkTpDeviceCollectiveCreditBinding *)calloc(
			(uint64_t)route_count * configuration.credit_count,sizeof(*bindings));
		if ( bindings == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		offset = 0u;
		configuration.credit_binding_count = 0u;
		for (route = 0u; route < route_count; route++)
		{
			hidden = configuration.local_hidden_dimension;
			credit_bytes = (uint64_t)configuration.max_active_sequence_count *
				hidden * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
			for (credit = 0u; credit < configuration.credit_count; credit++)
			{
				SparkTpDeviceCollectiveCreditBinding *binding =
					&bindings[configuration.credit_binding_count++];
				binding->step_index = route;
				binding->credit_index = credit;
				binding->send_device = (uint8_t *)tp->credit_send_bf16 + offset;
				binding->receive_device = (uint8_t *)tp->credit_receive_bf16 + offset;
				binding->send_transport = memory_mode ==
					SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
					(uint8_t *)tp->host_credit_send_bf16 + offset :
					binding->send_device;
				binding->receive_transport = memory_mode ==
					SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
					(uint8_t *)tp->host_credit_receive_bf16 + offset :
					binding->receive_device;
				binding->flags = memory_mode ==
					SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
					SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
				offset += credit_bytes;
			}
		}
		configuration.credit_bindings = bindings;
	}
	fprintf(stderr, "%s config degree=%u rank=%u backend=%s credits=%u mask=0x%x rails=%u port=%u direct_max=%u split_min=%u step=[%u,%u,%u] local=%s host0=%s bindings=%u\n",
		SPARK_QWEN36_TP_TAG, degree, rank,
		transport_backend != 0u ? "transport" : "nccl",
		configuration.credit_count, configuration.algorithm_mask,
		configuration.rail_count, configuration.control_port_base,
		configuration.direct_all_to_all_max_payload_bytes,
		configuration.split_ring_min_payload_bytes,
		configuration.step_rail_indices[0], configuration.step_rail_indices[1],
		configuration.step_rail_indices[2], configuration.local_host,
		configuration.rank_hosts[0], configuration.credit_binding_count);
	status = SparkTpDeviceCollectiveCreate(&configuration, &tp->collective);
	free((void *)configuration.credit_bindings);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s create_failed status=%d degree=%u rank=%u backend=%s\n",
			SPARK_QWEN36_TP_TAG, (int)status, degree, rank,
			transport_backend != 0u ? "transport" : "nccl");
		return status;
	}
	tp->initialized = 1u;
	tp->next_ordinal = 0u;
	fprintf(stderr, "%s ready degree=%u rank=%u backend=%s credits=%u\n",
		SPARK_QWEN36_TP_TAG, degree, rank,
		transport_backend != 0u ? "transport" : "nccl",
		tp->collective.credit_count);
	return SPARK_STATUS_OK;
}

void SparkQwen36TpDestroy(SparkQwen36TpState *tp)
{
	if ( tp == 0 )
		return;
	if ( tp->initialized != 0u )
		(void)SparkTpDeviceCollectiveDestroy(&tp->collective);
	if ( tp->credit_send_bf16 != 0 && tp->host_credit_send_bf16 == 0 )
		(void)cudaFree(tp->credit_send_bf16);
	if ( tp->credit_receive_bf16 != 0 && tp->host_credit_receive_bf16 == 0 )
		(void)cudaFree(tp->credit_receive_bf16);
	if ( tp->host_credit_send_bf16 != 0 )
		(void)cudaFreeHost(tp->host_credit_send_bf16);
	if ( tp->host_credit_receive_bf16 != 0 )
		(void)cudaFreeHost(tp->host_credit_receive_bf16);
	memset(tp, 0, sizeof(*tp));
}

SparkStatus SparkQwen36TpReduceHidden(
	SparkQwen36TpState *tp,
	void *buffer,
	uint32_t rows,
	void *cuda_stream)
{
	if ( tp == 0 || buffer == 0 || rows == 0u || cuda_stream == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	if ( rows > tp->collective.max_active_sequence_count )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkQwen36TpSubmit(tp,buffer,rows,cuda_stream,0u);
}

SparkStatus SparkQwen36TpReduceU64Max(
	SparkQwen36TpState *tp,
	uint64_t *buffer,
	uint32_t count,
	void *cuda_stream)
{
	if ( tp == 0 || buffer == 0 || count == 0u || cuda_stream == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	return SparkQwen36TpSubmit(tp,buffer,count,cuda_stream,1u);
}
