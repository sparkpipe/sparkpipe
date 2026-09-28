#pragma once

static void SPARK_FAMILY(ModuleReportReady)(void *module_state)
{
	SPARK_FAMILY(ModuleState) *state = (SPARK_FAMILY(ModuleState) *)module_state;
	fprintf(stderr,"%s initialize ok slice=%u+%u gdn=%u attn=%u owns_embedding=%u owns_head=%u\n",SPARK_FAMILY_CONST(MODULE_TAG),state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->owns_embedding,state->owns_final_head);
}
