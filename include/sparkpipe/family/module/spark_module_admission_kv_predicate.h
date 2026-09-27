#pragma once

static SparkStatus SPARK_FAMILY(AdmissionKvPredicate)(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u && SparkModelDriverRangeFitsWithinCapacity(request->sequence_position,request->new_token_count,SPARK_FAMILY_CONST(MODEL_MAXIMUM_CONTEXT_TOKENS)) == 0u )
		SparkModelDriverRejectAdmission(decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY,decision->available_dispatch_slot_count);
	else
	{
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	}
	return(SPARK_STATUS_OK);
}
