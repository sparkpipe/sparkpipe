#pragma once

static const LmKvFrameOps SPARK_FAMILY(ModuleKvFrameOps) =
{
	SPARK_FAMILY_CONST(MODULE_TAG),
	SPARK_FAMILY(ModuleKvFrameBuildRestoreBatch),
	SPARK_FAMILY(ModuleKvFrameBuildEvictBatch),
	SPARK_FAMILY(ModuleKvFrameSubmit),
	SPARK_FAMILY(ModuleKvFrameProgress),
	SPARK_FAMILY(ModuleKvFrameAcknowledge)
};

static SparkStatus SPARK_FAMILY(ModuleOpenKvTier)(SPARK_FAMILY(ModuleState) *state, const SparkFirmwareModuleHostServices *host_services)
{
	SPARK_FAMILY_CONST(MODULE_KV_GEOMETRY) geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3],block_record_bytes,staging_bytes;
	uint32_t workers = 0u,block_record_elements,index;
	SparkStatus status;
	static const char *none = "none";
	state->kv.tier_active = 0u;
	state->kv_logical_page_capacity = host_services->kv_logical_page_capacity;
	state->kv_physical_page_capacity = host_services->kv_physical_page_capacity;
	state->kv_backing_maximum_bytes = host_services->kv_backing_maximum_bytes;
	if ( host_services->kv_backing_directory == 0 && host_services->kv_backing_maximum_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	provider = getenv("SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_KV_STORE");
	if ( provider == 0 )
		provider = none;
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv.client,SPARK_FAMILY_CONST(MODULE_TAG),provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	SPARK_FAMILY_CONST(MODULE_KV_EXPECTED_GEOMETRY)(&geometry,state);
	model_fp = SparkStageModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	block_record_elements = SPARK_FAMILY_CONST(MODULE_KV_BLOCK_RECORD_ELEMENTS)(state);
	layout_bits[0] = block_record_elements;
	layout_bits[1] = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	layout_bits[2] = state->kv.block_count;
	layout_fp = SparkStageModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	block_record_bytes = (uint64_t)block_record_elements * SPARK_FAMILY_CONST(MODEL_BF16_ELEMENT_BYTES);
	staging_bytes = block_record_bytes * SPARK_LLM_KV_STAGING_RECORDS;
	if ( state->kv_physical_page_capacity != 0u && state->kv.block_count > state->kv_physical_page_capacity )
		state->kv.block_count = state->kv_physical_page_capacity;
	if ( state->kv.block_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->kv.plan.model_fingerprint = model_fp;
	state->kv.plan.cache_layout_fingerprint = layout_fp;
	state->kv.plan.rank_index = state->stage_index;
	state->kv.plan.block_record_bytes = (uint32_t)block_record_bytes;
	state->kv.plan.gdn_record_bytes = SPARK_FAMILY_CONST(MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES);
	state->kv.plan.lookahead_packet_count = 3u;
	state->kv.plan.physical_block_capacity = state->kv.block_count;
	state->kv.plan.allocated_physical_block_count = 0u;
	state->kv.plan.staging_block_capacity = SPARK_LLM_KV_STAGING_RECORDS;
	status = SparkStageKvClientOpen(&state->kv.client,SPARK_FAMILY_CONST(MODULE_TAG),provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->kv.slot_lane = (uint32_t *)malloc((size_t)state->kv.block_count * sizeof(uint32_t));
	state->kv.slot_logical = (uint32_t *)malloc((size_t)state->kv.block_count * sizeof(uint32_t));
	state->kv.slot_sequence = (uint64_t *)malloc((size_t)state->kv.block_count * sizeof(uint64_t));
	state->kv.slot_dirty = (uint8_t *)calloc((size_t)state->kv.block_count,sizeof(uint8_t));
	state->kv.slot_pinned = (uint8_t *)calloc((size_t)state->kv.block_count,sizeof(uint8_t));
	state->kv.slot_free_stack = (uint32_t *)malloc((size_t)state->kv.block_count * sizeof(uint32_t));
	state->kv.block_staging = malloc((size_t)staging_bytes);
	state->kv.gdn_staging = malloc(SPARK_FAMILY_CONST(MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES));
	if ( cudaMalloc((void **)&state->kv.table_indices_device,(size_t)SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT) * SPARK_LLM_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t)) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc((void **)&state->kv.table_counts_device,(size_t)SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT) * sizeof(uint32_t)) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv.table_indices_host = (uint32_t *)malloc((size_t)SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT) * SPARK_LLM_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t));
	if ( state->kv.slot_lane == 0 || state->kv.slot_logical == 0 || state->kv.slot_sequence == 0 || state->kv.slot_dirty == 0 || state->kv.slot_pinned == 0 || state->kv.slot_free_stack == 0 || state->kv.block_staging == 0 || state->kv.gdn_staging == 0 || state->kv.table_indices_host == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index = 0u; index < state->kv.block_count; index++)
		state->kv.slot_free_stack[index] = index;
	state->kv.slot_free_count = state->kv.block_count;
	state->kv.evict_cursor = 0u;
	state->kv.tier_active = 1u;
	state->kv.ops = SPARK_FAMILY(ModuleKvFrameOps);
	fprintf(stderr,"%s kv_tier_open provider=%s window=%u logical=%u physical=%u backing_bytes=%llu\n",SPARK_FAMILY_CONST(MODULE_TAG),provider,state->kv.block_count,state->kv_logical_page_capacity,state->kv_physical_page_capacity,(unsigned long long)state->kv_backing_maximum_bytes);
	return(SPARK_STATUS_OK);
}
