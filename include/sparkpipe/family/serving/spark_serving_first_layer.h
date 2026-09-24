#pragma once

static uint32_t SPARK_FAMILY(ServingFirstLayer)(uint32_t stage_index)
{
	uint32_t index,first_layer;
#if SPARK_FAMILY_CONST(SERVING_TP)
	(void)stage_index;
	return(0u);
#endif
	first_layer = 0u;
	for (index=0u; index<stage_index; index++)
		first_layer += SPARK_FAMILY(ServingDescriptor).stage_layer_counts[index];
	return(first_layer);
}

static SparkStatus SPARK_FAMILY(ServingValidateSubmission)(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SPARK_FAMILY(ServingState) *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SPARK_FAMILY(ServingState) *)adapter_state;
	status = SPARK_FAMILY(ServingValidateSubmissionBase)(state,submission);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}
