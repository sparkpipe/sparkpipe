#pragma once

static SparkStatus SPARK_FAMILY(LazyOpen)(SPARK_FAMILY(ModuleState) *state,const char *path,uint64_t bytes,const SPARK_FAMILY(StagePackEntry) *entries,uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SPARK_FAMILY(ManifestContext) context = {entries,count};
	SparkStatus status;
	const char *digest;
	uint64_t spine_budget;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,sizeof(request.identity.pack_sha256));
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_FAMILY_CONST(MODULE_TAG));
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkStageModuleEnvironmentUnsigned64(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX,&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,&spine_budget);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SPARK_FAMILY(ManifestCheck),&context,&state->lazy_pack);
	return(status);
}
