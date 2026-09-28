#pragma once

#if !defined(SPARK_MODULE_ADMIT_POLICY_FLAGS) || !defined(SPARK_MODULE_ADMIT_MAX_INPUT_ROWS)
#error "a module including the shared admission shape names its policy flags and input-row cap"
#endif

static SparkStatus SPARK_FAMILY(ModuleAdmit)(void *module_state,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	SPARK_FAMILY(ModuleState) *state;
	SparkAdmissionPolicyTable table;
	uint32_t available_slot_count;
	SparkStatus status;
	state = (SPARK_FAMILY(ModuleState) *)module_state;
#ifdef SPARK_MODULE_ADMIT_RESET
	if ( request != 0 && decision != 0 && request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SPARK_MODULE_ADMIT_RESET(state,request);
		if ( status == SPARK_STATUS_OK )
		{
			decision->accepted = 1u;
			decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		}
		SPARK_RETURN(status);
	}
#endif
	available_slot_count = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->max_active_sequence_count;
	table.max_input_row_count = SPARK_MODULE_ADMIT_MAX_INPUT_ROWS(state);
	table.max_sequence_positions = SPARK_FAMILY_CONST(MODEL_MAXIMUM_CONTEXT_TOKENS);
	table.flags = SPARK_MODULE_ADMIT_POLICY_FLAGS;
	table.predicate = SPARK_FAMILY(AdmissionKvPredicate);
	table.predicate_context = state;
	table.cost = SPARK_FAMILY(AdmissionCost);
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available_slot_count,request,decision);
	if (status != SPARK_STATUS_OK)
		SPARK_RETURN(status);
	if (decision->accepted == 0u)
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}
