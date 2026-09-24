#pragma once

static void SPARK_FAMILY(ModuleDescribe)(void *module_state, SparkStageModuleLifecycle *lifecycle)
{
	SPARK_FAMILY(ModuleState) *state = (SPARK_FAMILY(ModuleState) *)module_state;
	lifecycle->module_tag = SPARK_FAMILY_CONST(MODULE_TAG);
	lifecycle->ledger = &state->ledger;
	lifecycle->slot_states = state->slot_states;
	lifecycle->pipeline_slot_count = state->pipeline_slot_count;
	lifecycle->submitted_count = &state->submitted_count;
	lifecycle->completed_count = &state->completed_count;
	lifecycle->rejected_count = &state->rejected_count;
	lifecycle->failed_count = &state->failed_count;
	lifecycle->tokens_emitted = &state->tokens_emitted;
}
