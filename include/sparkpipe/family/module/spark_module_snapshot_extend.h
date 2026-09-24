#pragma once

static void SPARK_FAMILY(ModuleSnapshotExtend)(
	void *module_state,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SPARK_FAMILY(ModuleState) *state = (SPARK_FAMILY(ModuleState) *)module_state;
	snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
}
