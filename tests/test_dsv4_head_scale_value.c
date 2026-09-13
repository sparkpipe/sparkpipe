/* P0-A release-blocker regression guard: hc_head_scale_value must be
 * non-zero after pack load.
 *
 * The blocker was a declared-but-never-assigned hc_head_scale_value: every
 * HcHeadReduce launch then ran with scale=0.0f and every token emitted at
 * zero gate. The fix seeds the scalar by value from the pack's 1x1
 * HC_HEAD_SCALE tensor through one device-safe readback inside the global
 * bind step, and readiness refuses packs whose bound scale is zero or
 * non-finite. This test executes that exact seam on the host against the
 * real module translation unit (compiled in below so the static
 * bind/coverage functions are reachable):
 *
 *   1. binding SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE seeds the known
 *      non-zero pack scalar into state->hc_head_scale_value;
 *   2. rebinding a different payload updates the mirror - the field is a
 *      live readback, not an accidentally non-zero constant;
 *   3. SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE seeds
 *      mtp_hc_head_scale_value the same way for the DSpark draft head;
 *   4. coverage refuses zero/non-finite scales and accepts the seeded one,
 *      so a regression to "declared but never assigned" fails on BOTH sides:
 *      the bind assertions see the calloc'd zero, and readiness would refuse
 *      every pack in production.
 *
 * No device work is launched here: cudaMemcpy runs through
 * tests/cuda_stub/cuda_runtime_stub.c and any kernel-launcher symbols left
 * over are satisfied by never-called link stubs (see the Python driver).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "spark_dsv4_resident_decode_stage_module.c"

static uint32_t failure_count;

static void Require(int condition,const char *label)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",label);
		failure_count++;
		return;
	}
	printf("PASS %s\n",label);
	fflush(stdout);
}

int main(void)
{
	SparkDsv4ModuleState state;
	SparkDsv4StagePackEntry entry;
	static char sink;
	float pack_scale = 1.375f;
	float rebound_scale = 0.8125f;
	float mtp_pack_scale = 2.5f;
	void *global_payload = (void *)0;
	void *rebound_payload = (void *)0;
	void *mtp_payload = (void *)0;
	SparkStatus status;
	uint32_t tensor,stage;

	/* 1: the pack load seeds the launch scalar from the 1x1 tensor. */
	cudaMalloc(&global_payload,sizeof(float));
	cudaMemcpy(global_payload,&pack_scale,sizeof(float),cudaMemcpyHostToDevice);
	memset(&state,0,sizeof(state));
	memset(&entry,0,sizeof(entry));
	entry.tensor_kind = SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE;
	entry.layer_index = SPARK_DSV4_STAGEPACK_GLOBAL_LAYER;
	entry.rows = 1u;
	entry.columns = 1u;
	status = SparkDsv4ModuleBindGlobal(&state,&entry,global_payload,(void *)0);
	Require(status == SPARK_STATUS_OK,"bind_hc_head_scale_ok");
	Require(state.hc_head_scale_f32 == (const float *)global_payload,
		"bind_keeps_device_pointer");
	Require(state.hc_head_scale_value != 0.0f,
		"head_scale_nonzero_after_load");
	Require(state.hc_head_scale_value == pack_scale,
		"head_scale_matches_pack_tensor");
	Require(isfinite(state.hc_head_scale_value) != 0,
		"head_scale_finite_after_load");

	/* 2: rebinding another pack updates the mirror by value. */
	cudaMalloc(&rebound_payload,sizeof(float));
	cudaMemcpy(rebound_payload,&rebound_scale,sizeof(float),
		cudaMemcpyHostToDevice);
	status = SparkDsv4ModuleBindGlobal(&state,&entry,rebound_payload,(void *)0);
	Require(status == SPARK_STATUS_OK,"bind_rebound_hc_head_scale_ok");
	Require(state.hc_head_scale_f32 == (const float *)rebound_payload,
		"rebind_updates_device_pointer");
	Require(state.hc_head_scale_value == rebound_scale,
		"rebind_updates_seeded_value");

	/* 3: the draft head's own 1x1 scale seeds the same way. */
	entry.tensor_kind = SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE;
	entry.layer_index = SPARK_DSV4_STAGEPACK_MTP_LAYER(0);
	cudaMalloc(&mtp_payload,sizeof(float));
	cudaMemcpy(mtp_payload,&mtp_pack_scale,sizeof(float),
		cudaMemcpyHostToDevice);
	status = SparkDsv4ModuleBindGlobal(&state,&entry,mtp_payload,(void *)0);
	Require(status == SPARK_STATUS_OK,"bind_mtp_hc_head_scale_ok");
	Require(state.mtp.hc_head_scale_f32 == (const float *)mtp_payload,
		"mtp_bind_keeps_device_pointer");
	Require(state.mtp_hc_head_scale_value != 0.0f,
		"mtp_head_scale_nonzero_after_load");
	Require(state.mtp_hc_head_scale_value == mtp_pack_scale,
		"mtp_head_scale_matches_pack_tensor");

	/* 4: readiness refuses the pre-fix world (bound pointer, zero value)
	 * and accepts the seeded one, for both scalars. */
	memset(&state,0,sizeof(state));
	state.participates_final_head = 1u;
	for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM;
		tensor <= SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE; tensor++)
		state.global_seen_bits |= 1ull << tensor;
	for (stage = 0u; stage < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; stage++)
		state.mtp_seen_bits[stage] =
			SparkDsv4ModuleExpectedLayerBits(SPARK_DSV4_STAGEPACK_MTP_LAYER(stage));
	state.hc_head_fn_f32 = (const float *)&sink;
	state.hc_head_base_f32 = (const float *)&sink;
	state.hc_head_scale_f32 = (const float *)global_payload;
	state.mtp.main_proj.payload = (const void *)&sink;
	state.mtp.main_proj.scale_data = (const void *)&sink;
	state.mtp.main_norm_weight_bf16 = (const void *)&sink;
	state.mtp.final_norm_weight_bf16 = (const void *)&sink;
	state.mtp.hc_head_fn_f32 = (const float *)&sink;
	state.mtp.hc_head_base_f32 = (const float *)&sink;
	state.mtp.hc_head_scale_f32 = (const float *)mtp_payload;
	state.mtp.markov_w1.payload = (const void *)&sink;
	state.mtp.markov_w2.payload = (const void *)&sink;
	state.mtp.confidence_proj.payload = (const void *)&sink;
	state.hc_head_scale_value = pack_scale;
	state.mtp_hc_head_scale_value = mtp_pack_scale;

	/* A final-head stage with the DSpark draft present also expects the
	 * shared embedding bit even when this stage does not own it. */
	state.global_seen_bits |= 1ull << SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING;
	for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ;
		tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ; tensor++)
		state.global_seen_bits |= 1ull << tensor;

	/* Baseline first: a fully seeded pack must pass, so each single-field
	 * corruption below is refused by exactly the head-scale gate. */
	status = SparkDsv4ModuleVerifyCoverage(&state);
	Require(status == SPARK_STATUS_OK,
		"coverage_accepts_seeded_head_scales");

	state.hc_head_scale_value = 0.0f;
	status = SparkDsv4ModuleVerifyCoverage(&state);
	Require(status == SPARK_STATUS_VALIDATION_FAILED,
		"coverage_refuses_zero_head_scale");

	state.hc_head_scale_value = NAN;
	status = SparkDsv4ModuleVerifyCoverage(&state);
	Require(status == SPARK_STATUS_VALIDATION_FAILED,
		"coverage_refuses_nan_head_scale");

	state.hc_head_scale_value = pack_scale;
	state.mtp_hc_head_scale_value = 0.0f;
	status = SparkDsv4ModuleVerifyCoverage(&state);
	Require(status == SPARK_STATUS_VALIDATION_FAILED,
		"coverage_refuses_zero_mtp_head_scale");

	if ( failure_count != 0u )
	{
		fprintf(stderr,"FAIL dsv4 hc_head_scale_value pack-load unit: %u failures\n",
			failure_count);
		return(1);
	}
	printf("PASS dsv4 hc_head_scale_value nonzero after pack load\n");
	return(0);
}
