#include "spark_qwen36_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen36_model.h"

#define SPARK_QWEN36_TP_TAG "qwen36_tp"

/* v1 NCCL configuration. The control port base and the collective
 * identifier come from the environment so the band deployment owns its
 * port block; the bootstrap hosts are the TP4 band's 100G rail addresses.
 * The NCCL library is bundled with the runtime (lib/runtime libs pattern)
 * and resolved through the process LD_LIBRARY_PATH; the env override names
 * a specific file for out-of-band layouts. */
#define SPARK_QWEN36_TP_DEFAULT_CONTROL_PORT_BASE 61620u
#define SPARK_QWEN36_TP_IDENTIFIER 0x513630545031ull
#define SPARK_QWEN36_TP_DEFAULT_NCCL_LIBRARY "libnccl.so.2"

static const char *SparkQwen36TpDefaultHosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE] =
{
	"10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* The NCCL backend requires a completion callback on every submission and
 * invokes it synchronously right after enqueue (a stream-order
 * continuation); the v1 synchronous caller waits on the stream instead. */
static void SparkQwen36TpCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

SparkStatus SparkQwen36TpInitialize(
	SparkQwen36TpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	void *registration_cuda_stream)
{
	SparkTpDeviceCollectiveConfig configuration;
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
	 * path without a peer group. The collective is skipped and the
	 * reduces become no-ops, so the validator checks internal consistency
	 * and determinism; cross-rank numerics are gated by the band E2E run. */
	if ( getenv("SPARK_QWEN36_TP_STANDALONE") != 0 )
	{
		fprintf(stderr, "%s standalone degree=%u rank=%u (collective skipped)\n",
			SPARK_QWEN36_TP_TAG, degree, rank);
		return SPARK_STATUS_OK;
	}

	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	configuration.tp_degree = degree;
	configuration.tp_rank = rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 1u;
	configuration.local_hidden_dimension =
		SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = max_active_sequence_count;
	configuration.connect_timeout_milli = 120000u;
	configuration.operation_timeout_milli = 120000u;
	configuration.control_port_base = SPARK_QWEN36_TP_DEFAULT_CONTROL_PORT_BASE;
	configuration.collective_identifier = SPARK_QWEN36_TP_IDENTIFIER;
	configuration.backend_module_path =
		getenv("SPARK_QWEN36_TP_NCCL_LIBRARY") != 0
			? getenv("SPARK_QWEN36_TP_NCCL_LIBRARY")
			: SPARK_QWEN36_TP_DEFAULT_NCCL_LIBRARY;
	configuration.local_host = SparkQwen36TpDefaultHosts[rank];
	for (index = 0u; index < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; index++)
		configuration.rank_hosts[index] = SparkQwen36TpDefaultHosts[index];
	configuration.registration_cuda_stream = registration_cuda_stream;

	status = SparkTpDeviceCollectiveCreate(&configuration, &tp->collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s create_failed status=%d degree=%u rank=%u\n",
			SPARK_QWEN36_TP_TAG, (int)status, degree, rank);
		return status;
	}
	tp->initialized = 1u;
	/* The backend's own ordinal counter starts at zero. */
	tp->next_ordinal = 0u;
	fprintf(stderr, "%s ready degree=%u rank=%u\n",
		SPARK_QWEN36_TP_TAG, degree, rank);
	return SPARK_STATUS_OK;
}

void SparkQwen36TpDestroy(SparkQwen36TpState *tp)
{
	if ( tp == 0 )
		return;
	if ( tp->initialized != 0u )
		(void)SparkTpDeviceCollectiveDestroy(&tp->collective);
	memset(tp, 0, sizeof(*tp));
}

SparkStatus SparkQwen36TpReduceHidden(
	SparkQwen36TpState *tp,
	void *buffer,
	uint32_t rows)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus status;

	if ( tp == 0 || buffer == 0 || rows == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	if ( rows > tp->collective.max_active_sequence_count )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = rows;
	submission.ordinal = tp->next_ordinal++;
	submission.local_device = buffer;
	submission.full_device = buffer;
	submission.cuda_stream = tp->cuda_stream;
	submission.completion_function = SparkQwen36TpCompletion;
	submission.completion_context = tp;
	status = SparkTpDeviceCollectiveSubmitBf16(
		&tp->collective, &submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	/* v1 synchronous: the NCCL all-reduce is stream-ordered; the caller's
	 * next sharded GEMM reads the result, so the stream must drain here. */
	return SparkQwen36TpStreamSynchronize(tp);
}

SparkStatus SparkQwen36TpReduceU64Max(
	SparkQwen36TpState *tp,
	uint64_t *buffer,
	uint32_t count)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus status;

	if ( tp == 0 || buffer == 0 || count == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( tp->degree <= 1u || tp->initialized == 0u )
		return SPARK_STATUS_OK;
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.ordinal = tp->next_ordinal++;
	submission.local_device = buffer;
	submission.full_device = buffer;
	submission.cuda_stream = tp->cuda_stream;
	submission.completion_function = SparkQwen36TpCompletion;
	submission.completion_context = tp;
	status = SparkTpDeviceCollectiveSubmitU64Max(
		&tp->collective, &submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkQwen36TpStreamSynchronize(tp);
}
