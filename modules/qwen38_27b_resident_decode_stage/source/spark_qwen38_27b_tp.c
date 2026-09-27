#include "spark_qwen38_27b_tp.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_tp_mesh_register.h"

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen38_27b_model.h"

#define SPARK_QWEN38_27B_TP_TAG "qwen38_27b_tp"

#define SPARK_QWEN38_27B_TP_OPERATION_TIMEOUT_MILLI 120000u

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

static SparkStatus SparkQwen38_27bTpOpen(SparkQwen38_27bTpState *tp,uint32_t max_active_sequence_count)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkStatus status;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = tp->degree;
	configuration.tp_rank = tp->rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.local_hidden_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = max_active_sequence_count;
	configuration.operation_timeout_milli = SPARK_QWEN38_27B_TP_OPERATION_TIMEOUT_MILLI;
	SparkTpMeshRegisterCommonCombines(&configuration);
	configuration.combine_context = tp;
	status = SparkTpDeviceCollectiveCreate(&configuration,&tp->collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s create_failed status=%d degree=%u rank=%u\n",SPARK_QWEN38_27B_TP_TAG,(int)status,tp->degree,tp->rank);
		return(status);
	}
	tp->initialized = 1u;
	return(SparkTpDeviceCollectiveAttach(&tp->collective,0));
}

SparkStatus SparkQwen38_27bTpInitialize(
	SparkQwen38_27bTpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	uint32_t pipeline_slot_count,
	void *registration_cuda_stream)
{
	uint32_t standalone;
	SparkStatus status;
	(void)pipeline_slot_count;
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
		return(SPARK_STATUS_OK);
	standalone = 0u;
	status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_QWEN38_27B_TP_TAG,"SPARK_QWEN38_27B_TP_STANDALONE",0u,1u,0u,&standalone);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( standalone != 0u )
	{
		fprintf(stderr,"%s standalone degree=%u rank=%u (collective skipped)\n",SPARK_QWEN38_27B_TP_TAG,degree,rank);
		return(SPARK_STATUS_OK);
	}
	status = SparkQwen38_27bTpOpen(tp,max_active_sequence_count);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen38_27bTpDestroy(tp);
		return(status);
	}
	fprintf(stderr,"%s ready degree=%u rank=%u\n",SPARK_QWEN38_27B_TP_TAG,degree,rank);
	return(SPARK_STATUS_OK);
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
