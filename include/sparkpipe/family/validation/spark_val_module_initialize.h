#pragma once

static int SPARK_FAMILY(ValModuleInitialize)(SPARK_FAMILY(ValModule) *module)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkStatus status;
	const char *stage_count_text;
	uint32_t lane;
	cudaError_t error;
	memset(module,0,sizeof(*module));
	stage_count_text = getenv("SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_STAGE_COUNT");
	module->head_stage = stage_count_text != 0 && strcmp(stage_count_text,"1") == 0 ? 1u : 0u;
	for (lane = 0u; lane < SPARK_FAMILY_CONST(VALIDATION_KV_LANES); lane++)
	{
		module->host_blocks[lane] = lane;
		module->host_counts[lane] = 1u;
	}
	error = cudaMalloc((void **)&module->device_blocks,sizeof(module->host_blocks));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_blocks,module->host_blocks,sizeof(module->host_blocks),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMalloc((void **)&module->device_counts,sizeof(module->host_counts));
	if (error == cudaSuccess) error = cudaMemcpy(module->device_counts,module->host_counts,sizeof(module->host_counts),cudaMemcpyHostToDevice);
	if (SPARK_FAMILY(ValCuda)(error,"module_table_alloc") != 0)
		return(1);
	module->table.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION);
	module->table.descriptor_bytes = sizeof(module->table);
	module->table.block_token_count = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	module->table.lane_count = SPARK_FAMILY_CONST(VALIDATION_KV_LANES);
	module->table.lane_stride = 1u;
	module->table.lane_capacity = SPARK_FAMILY_CONST(VALIDATION_KV_LANES);
	module->table.physical_block_indices = module->device_blocks;
	module->table.lane_physical_block_counts = module->device_counts;
	module->table.host_physical_block_indices = module->host_blocks;
	module->table.host_lane_physical_block_counts = module->host_counts;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "Qwen/Qwen3.8-27B";
	configuration.model_revision = "validation";
	configuration.stage_name = SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_resident_decode_stage";
	configuration.program_name = "resident_decode";
	configuration.operation_name = SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_resident_decode_stage";
	configuration.configuration_json = "{}";
	configuration.configuration_json_bytes = 2u;
	memset(&host_services,0,sizeof(host_services));
	host_services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services.descriptor_bytes = sizeof(host_services);
	host_services.node_id = "spark-" SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "-validator";
	host_services.node_target = "cuda.sm121." SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) ".resident_decode_stage.bf16";
	host_services.execution_stream = (void *)cudaStreamPerThread;
	status = SPARK_FAMILY(ResidentDecodeStageInitialize)(&configuration,&host_services,&module->state);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation failure=module_initialize status=%d\n",(int)status);
		return(1);
	}
	return(0);
}
