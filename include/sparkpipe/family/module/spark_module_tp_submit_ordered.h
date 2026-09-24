#pragma once

static SparkStatus SPARK_FAMILY(ModuleTpCombineU64Max)(void *combine_context,uint64_t *destination_device,const uint64_t *source_device,uint32_t count,void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),SPARK_FAMILY(LaunchTpCombineU64Max)((cudaStream_t)cuda_stream,destination_device,source_device,count),"tp_combine_u64_max"));
}

static SparkStatus SPARK_FAMILY(ModuleTpSubmitOrdered)(SPARK_FAMILY(ModuleState) *state,void *device_buffer,uint32_t count,SPARK_FAMILY(ModuleSlot) *slot,uint32_t u64_max)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec pause;
	uint32_t polls,flag;
	SparkStatus status;
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	if ( state->tp_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&state->tp_completion_flag,0u,memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.logical_sequence_count = slot->logical_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = device_buffer;
	submission.full_device = device_buffer;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkStageModuleTpCompletionFlag;
	submission.completion_context = &state->tp_completion_flag;
	status = u64_max != 0u
		? SparkTpDeviceCollectiveSubmitU64Max(&state->tp_device_collective,&submission)
		: SparkTpDeviceCollectiveSubmitBf16(&state->tp_device_collective,&submission);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pause.tv_sec = 0u;
	pause.tv_nsec = 100000;
	for (polls = 0u; polls < 100000u; polls++)
	{
		flag = atomic_load_explicit(&state->tp_completion_flag,memory_order_acquire);
		if ( flag == 1u )
			return(SPARK_STATUS_OK);
		if ( flag == 2u )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		nanosleep(&pause,0);
	}
	fprintf(stderr,"%s tp_all_reduce_stall\n",SPARK_FAMILY_CONST(MODULE_TAG));
	SPARK_FAIL(SPARK_STATUS_IO_ERROR);
}
