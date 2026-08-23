/* Shared link-only recorders for the glm52 white-box gates: the harness
 * includes the module TU, whose speculator paths reference the DFlash2
 * drafter backend entry points. None of these gates launch engine work -
 * drafter numerics gate on GB10 via the epoch-3 validator - so the symbols
 * resolve as recorders that refuse if ever reached. */
#ifndef GLM52_DSPARK_BACKEND_STUBS_H
#define GLM52_DSPARK_BACKEND_STUBS_H

#include <cuda_runtime.h>

#include "sparkpipe/spark_glm52_dspark_draft_backend.h"

#include "sparkpipe/spark_glm52_dspark_draft_backend.h"
SparkStatus SparkGlm52DsparkDraftBackendInitialize(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
	(void)backend;(void)configuration;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendTeardown(SparkGlm52DsparkDraftBackend *backend)
{
	(void)backend;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkGlm52DsparkDraftBackendModelContract(const SparkGlm52DsparkDraftBackend *backend,SparkGlm52DsparkModelContract *contract_out)
{
	(void)backend;(void)contract_out;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(SparkGlm52DsparkDraftBackend *backend,uint32_t lane_index,void *tap_output_bf16[SPARK_DSPARK_AUX_LAYER_COUNT],uint64_t *lane_stride_bytes_out)
{
	(void)backend;(void)lane_index;(void)tap_output_bf16;(void)lane_stride_bytes_out;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendStageBatch(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftBackendStage *stages,uint32_t stage_count)
{
	(void)backend;(void)stages;(void)stage_count;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendLaunchDraftBatch(SparkGlm52DsparkDraftBackend *backend,const SparkGlm52DsparkDraftRequest *requests,uint32_t lane_count)
{
	(void)backend;(void)requests;(void)lane_count;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendTakeBatchResults(SparkGlm52DsparkDraftBackend *backend,SparkGlm52DsparkDraftResult *results,uint32_t result_capacity,uint32_t *result_count)
{
	(void)backend;(void)results;(void)result_capacity;(void)result_count;
	return(1);
}
SparkStatus SparkGlm52DsparkDraftBackendResetLanes(SparkGlm52DsparkDraftBackend *backend,const uint32_t *lane_indices,uint32_t lane_count)
{
	(void)backend;(void)lane_indices;(void)lane_count;
	return(SPARK_STATUS_OK);
}

#endif
