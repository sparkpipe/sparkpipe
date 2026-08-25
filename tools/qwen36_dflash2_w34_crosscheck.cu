// qwen36_dflash2_w34_crosscheck.cu - INDEPENDENT cross-check of the LANDED
// DFlash2 W4/W3 kernels (SparkQwen36DsparkHeadTopK*/Selector*) against a
// second, independently generated oracle case set
// (tools/qwen36_dflash2_w34_oracle.py).
//
// Why a second harness: the landed validator and this one were written from
// the same contract but make DIFFERENT coverage choices, and coverage is the
// part a parity harness cannot self-check. This one adds two cases whose only
// purpose is to make a rule observable:
//   w3_tie          - an exact tie at the max of EVERY walk row, so the walk's
//                     first-max rule is distinguishable from last-max
//   w3_exact_trunc  - the context gate's BF16 truncation rounds 2509/3584
//                     entries, so a kernel that skipped it cannot pass
// Both are mutation-proven (see the report). Launch configuration below is
// copied verbatim from the landed wrappers in
// spark_qwen36_resident_decode_stage_cuda.cu so this drives the same shapes.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "spark_qwen38_27b_dspark_cuda.cuh"
#include "qwen38_27b_dflash2_parity_io.h"

static void RunW4(const char *directory, const char *test_case, uint64_t *mismatches)
{
	SparkParityDims dims = SparkParityLoadDims(directory, test_case);
	uint32_t rows = (uint32_t)SparkParityDim(&dims, "rows");
	uint32_t candidates = (uint32_t)SparkParityDim(&dims, "candidates");
	uint32_t hidden = (uint32_t)SparkParityDim(&dims, "hidden");
	uint32_t top_k = (uint32_t)SparkParityDim(&dims, "top_k");
	int exact = (int)SparkParityDim(&dims, "exact");
	uint64_t hidden_elements = (uint64_t)rows * hidden;
	uint64_t head_elements = (uint64_t)candidates * hidden;
	uint64_t out_elements = (uint64_t)rows * top_k;
	uint64_t key_elements = (uint64_t)rows * SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT * top_k;
	uint16_t *host_hidden = (uint16_t *)SparkParityLoad(directory, test_case, "hidden.bf16", hidden_elements * 2u);
	uint16_t *host_head = (uint16_t *)SparkParityLoad(directory, test_case, "head.bf16", head_elements * 2u);
	uint32_t *expect_ids = (uint32_t *)SparkParityLoad(directory, test_case, "expect_ids.u32", out_elements * 4u);
	float *expect_scores = (float *)SparkParityLoad(directory, test_case, "expect_scores.f32", out_elements * 4u);
	uint32_t *out_ids = (uint32_t *)malloc(out_elements * 4u);
	float *out_scores = (float *)malloc(out_elements * 4u);
	void *device_hidden = 0, *device_head = 0;
	uint64_t *device_keys = 0;
	uint32_t *device_ids = 0;
	float *device_scores = 0;
	dim3 grid(rows, SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT);
	cudaEvent_t start, stop;
	float elapsed_ms = 0.0f;

	printf("== W4 %s: rows=%u candidates=%u hidden=%u top_k=%u exact=%d chunks=%u\n",
		test_case, rows, candidates, hidden, top_k, exact, SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT);
	SPARK_PARITY_CUDA(cudaMalloc(&device_hidden, hidden_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_head, head_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_keys, key_elements * 8u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_ids, out_elements * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_scores, out_elements * 4u));
	SPARK_PARITY_CUDA(cudaMemcpy(device_hidden, host_hidden, hidden_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_head, host_head, head_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaEventCreate(&start));
	SPARK_PARITY_CUDA(cudaEventCreate(&stop));
	/* Landed launch configuration, verbatim from SparkQwen36LaunchDsparkHeadTopK. */
	SparkQwen36DsparkHeadTopKChunkKernel<<<grid, SPARK_LM_CTA_THREADS, hidden * (uint32_t)sizeof(float)>>>(
		device_hidden, device_head, device_keys, rows, candidates, hidden, top_k);
	SparkQwen36DsparkHeadTopKMergeKernel<<<rows, SPARK_LM_CTA_THREADS>>>(
		device_keys, device_ids, device_scores, 0, rows, 0u, top_k);
	SPARK_PARITY_CUDA(cudaDeviceSynchronize());
	SPARK_PARITY_CUDA(cudaEventRecord(start, 0));
	SparkQwen36DsparkHeadTopKChunkKernel<<<grid, SPARK_LM_CTA_THREADS, hidden * (uint32_t)sizeof(float)>>>(
		device_hidden, device_head, device_keys, rows, candidates, hidden, top_k);
	SparkQwen36DsparkHeadTopKMergeKernel<<<rows, SPARK_LM_CTA_THREADS>>>(
		device_keys, device_ids, device_scores, 0, rows, 0u, top_k);
	SPARK_PARITY_CUDA(cudaEventRecord(stop, 0));
	SPARK_PARITY_CUDA(cudaDeviceSynchronize());
	SPARK_PARITY_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
	SPARK_PARITY_CUDA(cudaMemcpy(out_ids, device_ids, out_elements * 4u, cudaMemcpyDeviceToHost));
	SPARK_PARITY_CUDA(cudaMemcpy(out_scores, device_scores, out_elements * 4u, cudaMemcpyDeviceToHost));
	printf("landed top-16 latency: %.4f ms\n", (double)elapsed_ms);
	*mismatches += SparkParityCompareU32("top_ids", out_ids, expect_ids, out_elements);
	*mismatches += SparkParityCompareF32("top_scores", out_scores, expect_scores, out_elements, exact, 1.0e-6);
	SPARK_PARITY_CUDA(cudaFree(device_hidden));
	SPARK_PARITY_CUDA(cudaFree(device_head));
	SPARK_PARITY_CUDA(cudaFree(device_keys));
	SPARK_PARITY_CUDA(cudaFree(device_ids));
	SPARK_PARITY_CUDA(cudaFree(device_scores));
	free(host_hidden); free(host_head); free(expect_ids); free(expect_scores); free(out_ids); free(out_scores);
}

static void RunW3(const char *directory, const char *test_case, uint64_t *mismatches)
{
	SparkParityDims dims = SparkParityLoadDims(directory, test_case);
	uint32_t batch = (uint32_t)SparkParityDim(&dims, "batch");
	uint32_t slots = (uint32_t)SparkParityDim(&dims, "slots");
	uint32_t top_k = (uint32_t)SparkParityDim(&dims, "top_k");
	uint32_t rank = (uint32_t)SparkParityDim(&dims, "rank");
	uint32_t hidden = (uint32_t)SparkParityDim(&dims, "hidden");
	uint32_t vocab = (uint32_t)SparkParityDim(&dims, "vocab");
	int exact = (int)SparkParityDim(&dims, "exact");
	uint32_t pairs = batch * slots, index;
	uint64_t hidden_elements = (uint64_t)pairs * hidden;
	uint64_t table_elements = (uint64_t)vocab * rank;
	uint64_t proj_elements = (uint64_t)rank * hidden;
	uint64_t gate_elements = (uint64_t)pairs * rank;
	uint64_t candidate_elements = (uint64_t)pairs * top_k;
	uint64_t edge_elements = candidate_elements * top_k;
	uint16_t *host_hidden = (uint16_t *)SparkParityLoad(directory, test_case, "hidden.bf16", hidden_elements * 2u);
	uint16_t *host_proj = (uint16_t *)SparkParityLoad(directory, test_case, "proj.bf16", proj_elements * 2u);
	uint16_t *host_pred = (uint16_t *)SparkParityLoad(directory, test_case, "pred.bf16", table_elements * 2u);
	uint16_t *host_succ = (uint16_t *)SparkParityLoad(directory, test_case, "succ.bf16", table_elements * 2u);
	uint32_t *host_candidates = (uint32_t *)SparkParityLoad(directory, test_case, "candidates.u32", candidate_elements * 4u);
	float *host_unary = (float *)SparkParityLoad(directory, test_case, "unary.f32", candidate_elements * 4u);
	uint32_t *host_anchor = (uint32_t *)SparkParityLoad(directory, test_case, "anchor.u32", (uint64_t)batch * 4u);
	float *expect_gate = (float *)SparkParityLoad(directory, test_case, "expect_hproj.f32", gate_elements * 4u);
	float *expect_edges = (float *)SparkParityLoad(directory, test_case, "expect_edges.f32", edge_elements * 4u);
	uint32_t *expect_drafts = (uint32_t *)SparkParityLoad(directory, test_case, "expect_drafts.u32", (uint64_t)pairs * 4u);
	uint16_t *out_gate_bits = (uint16_t *)malloc(gate_elements * 2u);
	float *out_gate = (float *)malloc(gate_elements * 4u);
	float *out_edges = (float *)malloc(edge_elements * 4u);
	uint32_t *out_drafts = (uint32_t *)malloc((uint64_t)pairs * 4u);
	void *device_hidden = 0, *device_proj = 0, *device_pred = 0, *device_succ = 0, *device_gate = 0;
	uint32_t *device_candidates = 0, *device_anchor = 0, *device_drafts = 0, *device_slots = 0;
	float *device_unary = 0, *device_edges = 0;

	printf("== W3 %s: batch=%u slots=%u top_k=%u rank=%u hidden=%u vocab=%u exact=%d\n",
		test_case, batch, slots, top_k, rank, hidden, vocab, exact);
	SPARK_PARITY_CUDA(cudaMalloc(&device_hidden, hidden_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_proj, proj_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_pred, table_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_succ, table_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_gate, gate_elements * 2u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_candidates, candidate_elements * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_unary, candidate_elements * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_anchor, (uint64_t)batch * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_edges, edge_elements * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_drafts, (uint64_t)pairs * 4u));
	SPARK_PARITY_CUDA(cudaMalloc(&device_slots, (uint64_t)pairs * 4u));
	SPARK_PARITY_CUDA(cudaMemcpy(device_hidden, host_hidden, hidden_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_proj, host_proj, proj_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_pred, host_pred, table_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_succ, host_succ, table_elements * 2u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_candidates, host_candidates, candidate_elements * 4u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_unary, host_unary, candidate_elements * 4u, cudaMemcpyHostToDevice));
	SPARK_PARITY_CUDA(cudaMemcpy(device_anchor, host_anchor, (uint64_t)batch * 4u, cudaMemcpyHostToDevice));
	/* Landed launch configuration, verbatim from the three Selector wrappers. */
	SparkQwen36DsparkSelectorProjectKernel<<<pairs, SPARK_LM_CTA_THREADS, hidden * (uint32_t)sizeof(float)>>>(
		device_hidden, device_proj, device_gate, pairs, rank, hidden);
	SparkQwen36DsparkSelectorEdgeKernel<<<pairs, SPARK_LM_CTA_THREADS>>>(
		device_pred, device_succ, device_candidates, device_anchor, device_unary, device_gate,
		device_edges, batch, slots, top_k, rank);
	SparkQwen36DsparkSelectorWalkKernel<<<(batch + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS, SPARK_LM_CTA_THREADS>>>(
		device_edges, device_candidates, device_drafts, device_slots, batch, slots, top_k);
	SPARK_PARITY_CUDA(cudaDeviceSynchronize());
	SPARK_PARITY_CUDA(cudaMemcpy(out_gate_bits, device_gate, gate_elements * 2u, cudaMemcpyDeviceToHost));
	SPARK_PARITY_CUDA(cudaMemcpy(out_edges, device_edges, edge_elements * 4u, cudaMemcpyDeviceToHost));
	SPARK_PARITY_CUDA(cudaMemcpy(out_drafts, device_drafts, (uint64_t)pairs * 4u, cudaMemcpyDeviceToHost));
	/* The landed gate is BF16 STORAGE; the oracle's value is the BF16-rounded
	 * fp32, so widening the bits back is exact and the compare stays bitwise. */
	for (index = 0; index < (uint32_t)gate_elements; index++)
	{
		uint32_t widened = (uint32_t)out_gate_bits[index] << 16u;
		memcpy(&out_gate[index], &widened, sizeof(float));
	}
	*mismatches += SparkParityCompareF32("context_gate", out_gate, expect_gate, gate_elements, exact, 4.0e-3);
	*mismatches += SparkParityCompareF32("edges", out_edges, expect_edges, edge_elements, exact, 1.0e-5);
	*mismatches += SparkParityCompareU32("draft_ids", out_drafts, expect_drafts, (uint64_t)pairs);
	SPARK_PARITY_CUDA(cudaFree(device_hidden)); SPARK_PARITY_CUDA(cudaFree(device_proj));
	SPARK_PARITY_CUDA(cudaFree(device_pred)); SPARK_PARITY_CUDA(cudaFree(device_succ));
	SPARK_PARITY_CUDA(cudaFree(device_gate)); SPARK_PARITY_CUDA(cudaFree(device_candidates));
	SPARK_PARITY_CUDA(cudaFree(device_unary)); SPARK_PARITY_CUDA(cudaFree(device_anchor));
	SPARK_PARITY_CUDA(cudaFree(device_edges)); SPARK_PARITY_CUDA(cudaFree(device_drafts));
	SPARK_PARITY_CUDA(cudaFree(device_slots));
}

int main(int argc, char **argv)
{
	const char *directory = argc > 1 ? argv[1] : "/tmp/dflash2_w34";
	const char *test_case = argc > 2 ? argv[2] : "w3_tie";
	int exact_case;
	uint64_t mismatches = 0;
	SparkParityDims dims = SparkParityLoadDims(directory, test_case);
	exact_case = (int)SparkParityDim(&dims, "exact");
	if (test_case[1] == '4')
		RunW4(directory, test_case, &mismatches);
	else
		RunW3(directory, test_case, &mismatches);
	if (exact_case != 0)
		printf("CROSSCHECK %s case=%s (bitwise)\n", mismatches == 0 ? "PASS" : "FAIL", test_case);
	else
		printf("CROSSCHECK %s case=%s (tolerance, informational)\n", mismatches == 0 ? "PASS" : "DIFF", test_case);
	return (exact_case != 0 && mismatches != 0) ? 1 : 0;
}
