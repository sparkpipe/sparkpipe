#include "spark_qwen38_27b_tp.h"
#include "sparkpipe/spark_error_site.h"

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen38_27b_model.h"

#define SPARK_QWEN38_27B_TP_TAG "qwen38_27b_tp"

#define SPARK_QWEN38_27B_TP_DEFAULT_TRANSPORT_PORT_BASE 58700u
#define SPARK_QWEN38_27B_TP_IDENTIFIER 0x513630545031ull
#define SPARK_QWEN38_27B_TP_DEFAULT_TRANSPORT_LIBRARY "libhidden_transport.so"
static const char *SparkQwen38_27bTpRailHosts[2][SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE] =
{
	{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
	  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ "10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
	  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

extern cudaError_t SparkQwen38_27bLaunchAccumAdd(cudaStream_t stream, void *destination, const void *source, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38_27bLaunchAccumAddRelay(cudaStream_t stream, void *destination, const void *source, void *relay, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38_27bLaunchAccumAddTp4(cudaStream_t stream, void *destination, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38_27bLaunchAccumU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);

static SparkStatus SparkQwen38_27bTpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen38_27bLaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen38_27bTpCombineRelayBf16(void *combine_context, void *destination_device, const void *source_device, void *relay_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen38_27bLaunchAccumAddRelay((cudaStream_t)cuda_stream,destination_device,source_device,relay_device,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen38_27bTpCombineTp4Bf16(void *combine_context, void *destination_device, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen38_27bLaunchAccumAddTp4((cudaStream_t)cuda_stream,destination_device,rank_devices,tp_rank,active_sequence_count,hidden_dimension) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkQwen38_27bTpCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context;
	return SparkQwen38_27bLaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static void SparkQwen38_27bTpPendingCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkQwen38_27bTpPending *pending = (SparkQwen38_27bTpPending *)context;
	if ( pending == 0 || completion == 0 )
		return;
	pending->status = (uint32_t)completion->status;
	atomic_store_explicit(&pending->done,1u,memory_order_release);
}

static SparkStatus SparkQwen38_27bTpSubmit(SparkQwen38_27bTpState *tp, void *buffer, uint32_t count, uint32_t logical_count, void *cuda_stream, uint32_t u64_max)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkQwen38_27bTpPending pending;
	SparkStatus status;
	uint32_t spin;
	atomic_init(&pending.done,0u);
	pending.status = (uint32_t)SPARK_STATUS_OK;
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.logical_sequence_count = logical_count;
	submission.flags =
		SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = tp->next_ordinal++;
	submission.local_device = buffer;
	submission.full_device = buffer;
	submission.cuda_stream = cuda_stream;
	submission.completion_function = SparkQwen38_27bTpPendingCompletion;
	submission.completion_context = &pending;
	status = u64_max != 0u ? SparkTpDeviceCollectiveSubmitU64Max(&tp->collective,&submission) : SparkTpDeviceCollectiveSubmitBf16(&tp->collective,&submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s submit_failed status=%d rows=%u ordinal=%llu\n",
			SPARK_QWEN38_27B_TP_TAG, (int)status, count,
			(unsigned long long)(tp->next_ordinal - 1u));
		return status;
	}
	spin = 0u;
	while ( atomic_load_explicit(&pending.done,memory_order_acquire) == 0u )
	{
		if ( (spin & 0x3Fu) == 0u )
			sched_yield();
		spin++;
	}
	if ( pending.status != SPARK_STATUS_OK )
		fprintf(stderr, "%s completion_failed status=%d rows=%u ordinal=%llu\n",
			SPARK_QWEN38_27B_TP_TAG, (int)pending.status, count,
			(unsigned long long)(tp->next_ordinal - 1u));
	return (SparkStatus)pending.status;
}

SparkStatus SparkQwen38_27bTpInitialize(
	SparkQwen38_27bTpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	uint32_t pipeline_slot_count,
	void *registration_cuda_stream)
{
	(void)pipeline_slot_count;
	SparkTpDeviceCollectiveConfig configuration;
	const char *library_path;
	SparkStatus status;
	uint32_t index;

	if ( tp == 0 || degree == 0u || rank >= degree ||
		max_active_sequence_count == 0u || registration_cuda_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(tp, 0, sizeof(*tp));
	tp->degree = degree;
	tp->rank = rank;
	tp->cuda_stream = registration_cuda_stream;
	if ( (SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION % degree) != 0u ||
		(SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT % degree) != 0u )
	{
		fprintf(stderr, "%s geometry_undividable degree=%u\n",
			SPARK_QWEN38_27B_TP_TAG, degree);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	tp->gdn_qk_channels = SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION / degree;
	tp->gdn_value_channels = SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION / degree;
	tp->gdn_conv_channels =
		2u * tp->gdn_qk_channels + tp->gdn_value_channels;
	tp->gdn_key_heads = SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT / degree;
	tp->gdn_value_heads = SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT / degree;
	tp->attn_query_heads = SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT / degree;
	tp->attn_kv_heads = SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT / degree;
	tp->ffn_intermediate =
		SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION / degree;
	tp->head_rows = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT / degree;
	if ( degree == 1u )
		return SPARK_STATUS_OK;
	if ( getenv("SPARK_QWEN38_27B_TP_STANDALONE") != 0 )
	{
		fprintf(stderr, "%s standalone degree=%u rank=%u (collective skipped)\n",
			SPARK_QWEN38_27B_TP_TAG, degree, rank);
		return SPARK_STATUS_OK;
	}
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = degree;
	configuration.tp_rank = rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.local_hidden_dimension =
		SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = max_active_sequence_count;
	configuration.connect_timeout_milli = 120000u;
	configuration.operation_timeout_milli = 120000u;
	configuration.collective_identifier = SPARK_QWEN38_27B_TP_IDENTIFIER;
	configuration.registration_cuda_stream = registration_cuda_stream;
	configuration.local_host = SparkQwen38_27bTpRailHosts[0][rank];
	library_path = getenv("SPARK_QWEN38_27B_TP_TRANSPORT_LIBRARY");
	configuration.backend_module_path = library_path != 0 ? library_path : SPARK_QWEN38_27B_TP_DEFAULT_TRANSPORT_LIBRARY;
	configuration.control_port_base =
		SPARK_QWEN38_27B_TP_DEFAULT_TRANSPORT_PORT_BASE;
	configuration.algorithm_mask =
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
	configuration.rail_count = 0u;
	configuration.direct_all_to_all_max_payload_bytes = 0u;
	configuration.split_ring_min_payload_bytes = 0u;
	configuration.combine_bf16_function = SparkQwen38_27bTpCombineBf16;
	configuration.combine_relay_bf16_function = SparkQwen38_27bTpCombineRelayBf16;
	configuration.combine_tp4_bf16_function = SparkQwen38_27bTpCombineTp4Bf16;
	configuration.combine_u64_max_function = SparkQwen38_27bTpCombineU64Max;
	configuration.combine_context = tp;
	for (index = 0u; index < 2u; index++)
		memcpy(configuration.rail_rank_hosts[index],SparkQwen38_27bTpRailHosts[index],sizeof(SparkQwen38_27bTpRailHosts[index]));
	for (index = 0u; index < degree; index++)
		configuration.rank_hosts[index] = SparkQwen38_27bTpRailHosts[0][index];
	configuration.credit_count = 8u;
	{
		const char *session_ports_text = getenv("SPARK_QWEN38_27B_TP_SESSION_PORTS");
		const char *cell_scan;
		uint32_t row_index,column_index;
		unsigned long cell_value;
		if ( session_ports_text == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		cell_scan = session_ports_text;
		for (row_index = 0u; row_index < degree; row_index++)
			for (column_index = 0u; column_index < degree; column_index++)
			{
				char *cell_end;
				errno = 0;
				cell_value = strtoul(cell_scan,&cell_end,10);
				if ( cell_end == cell_scan || errno != 0 ||
					cell_value > 65535u ||
					(row_index == column_index ? cell_value != 0u : cell_value == 0u) )
					return SPARK_STATUS_INVALID_ARGUMENT;
				configuration.session_ports[row_index][column_index] = (uint16_t)cell_value;
				cell_scan = cell_end;
				while ( *cell_scan == ',' )
					cell_scan++;
			}
	}
	fprintf(stderr, "%s config degree=%u rank=%u port=%u local=%s host0=%s\n",
		SPARK_QWEN38_27B_TP_TAG, degree, rank,
		configuration.control_port_base, configuration.local_host,
		configuration.rank_hosts[0]);
	status = SparkTpDeviceCollectiveCreate(&configuration, &tp->collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s create_failed status=%d degree=%u rank=%u\n",
			SPARK_QWEN38_27B_TP_TAG, (int)status, degree, rank);
		SparkQwen38_27bTpDestroy(tp);
		return status;
	}
	tp->initialized = 1u;
	tp->next_ordinal = 0u;
	fprintf(stderr, "%s ready degree=%u rank=%u\n",
		SPARK_QWEN38_27B_TP_TAG, degree, rank);
	return SPARK_STATUS_OK;
}

void SparkQwen38_27bTpDestroy(SparkQwen38_27bTpState *tp)
{
	if ( tp == 0 )
		return;
	if ( tp->initialized != 0u )
		(void)SparkTpDeviceCollectiveDestroy(&tp->collective);
	memset(tp, 0, sizeof(*tp));
}

SparkStatus SparkQwen38_27bTpReduceHidden(
	SparkQwen38_27bTpState *tp,
	void *buffer,
	uint32_t rows,
	uint32_t logical_count,
	void *cuda_stream)
{
	if ( tp == 0 || buffer == 0 || rows == 0u || cuda_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	if ( rows > tp->collective.max_active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return SparkQwen38_27bTpSubmit(tp,buffer,rows,logical_count,cuda_stream,0u);
}

SparkStatus SparkQwen38_27bTpReduceU64Max(
	SparkQwen38_27bTpState *tp,
	uint64_t *buffer,
	uint32_t count,
	uint32_t logical_count,
	void *cuda_stream)
{
	if ( tp == 0 || buffer == 0 || count == 0u || cuda_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	return SparkQwen38_27bTpSubmit(tp,buffer,count,logical_count,cuda_stream,1u);
}
