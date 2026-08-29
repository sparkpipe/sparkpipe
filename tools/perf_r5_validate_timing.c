/* PERF PROGRAM 2 - R5 RECEIPT (perf-r1 lane, 2026-08-28).
 *
 * Submit-path validation fanout: a single submission crossing the
 * pipeline client -> residentd -> serving adapter chain used to re-run
 * SparkModelServingAdapterValidateRuntimeSubmission at EVERY layer, and
 * that function begins by re-validating the (descriptor, limits) pair —
 * the 12-check DESCRIPTOR_CHECKS array plus the limits chain — which is
 * immutable for a component's lifetime. The R5 hoist validates the pair
 * once at configure/connect and switches the per-submission path to
 * SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated (same
 * checks minus the immutable part, same statuses).
 *
 * This harness times the REAL functions on the DSV4 decode bucket shape
 * (8 lanes / 8 rows) so the per-layer saving has a number:
 *
 *   cc -O2 -Iinclude tools/perf_r5_validate_timing.c \
 *      runtime/model_serving_adapter.c -o /tmp/perf_r5_validate_timing -ldl
 *
 * Fail-loud contract: the harness ALSO asserts both functions return
 * identical statuses on the valid case, a field-invalid case, and a
 * capacity-exceeded case.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_model_serving_adapter.h"

#define BUCKET 8u
#define ITERATIONS 200000u

static void BuildDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	static const uint32_t layer_counts[13] = {3u,3u,3u,3u,3u,3u,3u,4u,4u,4u,4u,4u,2u};
	uint32_t index;
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV;
	descriptor->stage_count = 13u;
	descriptor->layer_count = 43u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 16384u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 128u;
	descriptor->max_input_row_count = 256u;
	descriptor->max_resident_sequence_count = 512u;
	descriptor->max_output_token_count = BUCKET; /* one token per decode lane */
	descriptor->resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO;
	descriptor->minimum_efficient_submission_row_count = 16u;
	descriptor->adapter_id = "spark.dsv4.flash.serving.v1";
	descriptor->model_id = "deepseek-ai/DeepSeek-V4-Flash-0731";
	descriptor->model_revision = "fixture";
	descriptor->driver_program_name = "resident_decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	for (index=0u; index<13u; index++)
		descriptor->stage_layer_counts[index] = layer_counts[index];
}

static double NowSeconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
	return((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

int main(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	SparkModelServingLane lanes[BUCKET];
	uint32_t token_ids[BUCKET],row_lanes[BUCKET];
	uint64_t row_positions[BUCKET],row_sequences[BUCKET];
	SparkModelServingSubmission submission;
	double start,full_ns,pre_ns;
	uint32_t iteration;
	SparkStatus status_full,status_pre;
	int failures = 0;

	BuildDescriptor(&descriptor);
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 4u;
	limits.max_active_sequence_count = 128u;
	limits.max_input_row_count = 256u;
	limits.resident_sequence_capacity = 512u;

	memset(lanes,0,sizeof(lanes));
	for (uint32_t lane=0u; lane<BUCKET; lane++)
	{
		lanes[lane].request_id = 10u + lane;
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = 1u;
		lanes[lane].sequence_id = 100u + lane;
		lanes[lane].resident_sequence_slot = lane * 3u;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		token_ids[lane] = 20u + lane;
		row_lanes[lane] = lane;
		row_positions[lane] = 128u + lane;
		row_sequences[lane] = 100u + lane;
	}
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = 10u;
	submission.request_id = 10u;
	submission.sequence_id = 100u;
	submission.control_generation = 1u;
	submission.transaction_id = 2u;
	submission.dispatch_generation = 3u;
	submission.request_generation = 1u;
	submission.step_generation = 4u;
	submission.active_sequence_count = BUCKET;
	submission.new_token_count = BUCKET;
	submission.lane_count = BUCKET;
	submission.row_count = BUCKET;
	submission.token_count = BUCKET;
	submission.lanes = lanes;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequences;

	/* STATUS PARITY: same accept/reject decisions, same statuses. */
	status_full = SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission);
	status_pre = SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated(&descriptor,&limits,&submission);
	printf("parity valid          : full=%d pre=%d %s\n",(int)status_full,(int)status_pre,status_full == status_pre && status_full == SPARK_STATUS_OK ? "OK" : "FAIL");
	if (status_full != status_pre || status_full != SPARK_STATUS_OK)
		failures++;
	limits.resident_sequence_capacity = 128u; /* a VALID limits pair (>= max_active_sequence_count); the per-submission slot check trips */
	lanes[7].resident_sequence_slot = 200u; /* >= capacity */
	status_full = SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission);
	status_pre = SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated(&descriptor,&limits,&submission);
	printf("parity capacity       : full=%d pre=%d %s\n",(int)status_full,(int)status_pre,status_full == status_pre && status_full == SPARK_STATUS_CAPACITY_EXCEEDED ? "OK" : "FAIL");
	if (status_full != status_pre || status_full != SPARK_STATUS_CAPACITY_EXCEEDED)
		failures++;
	limits.resident_sequence_capacity = 512u;
	lanes[7].resident_sequence_slot = 21u;
	lanes[3].flags = 0u; /* row order violation */
	status_full = SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission);
	status_pre = SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated(&descriptor,&limits,&submission);
	printf("parity field-invalid  : full=%d pre=%d %s\n",(int)status_full,(int)status_pre,status_full == status_pre && status_full == SPARK_STATUS_INVALID_ARGUMENT ? "OK" : "FAIL");
	if (status_full != status_pre || status_full != SPARK_STATUS_INVALID_ARGUMENT)
		failures++;
	lanes[3].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;

	/* TIMING: valid decode bucket, both validators. */
	start = NowSeconds();
	for (iteration=0u; iteration<ITERATIONS; iteration++)
		(void)SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission);
	full_ns = (NowSeconds() - start) * 1e9 / ITERATIONS;
	start = NowSeconds();
	for (iteration=0u; iteration<ITERATIONS; iteration++)
		(void)SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated(&descriptor,&limits,&submission);
	pre_ns = (NowSeconds() - start) * 1e9 / ITERATIONS;
	printf("timing full    %8.1f ns / submission\n",full_ns);
	printf("timing pre     %8.1f ns / submission\n",pre_ns);
	printf("hoisted per call      %8.1f ns (%.2fx)\n",full_ns - pre_ns,full_ns / pre_ns);
	/* The chain re-validates the same submission ~5x (pipeline client,
	 * resident client per rank x2 paths, residentd x2, adapter), so the
	 * per-submission hoist across the chain is ~5x this delta today. */
	printf("chain estimate (~5 re-validations/submit): %8.1f ns saved\n",(full_ns - pre_ns) * 5.0);
	printf(failures == 0 ? "ALL PARITY OK\n" : "PARITY FAILURES: %d\n",failures);
	return failures == 0 ? 0 : 1;
}
