// THE FAIL-FRAME PROBE - device receipt for the kernel-crew lane's four
// correctness items (docs/BUG_LEDGER.md), run on a live sm_121a GPU.
//
// Each probe injects corrupt kernel inputs - a route map past the source
// row count, a sparse top-k index past the page table, an rANS chunk span
// past the fixed shared window - through the SHIPPING extern "C" launchers
// of three resident-decode-stage modules, plus the shared attention kernel
// the old code could trap. The old code answered several of these with
// `asm volatile("trap;")`: the whole CUDA context died, taking every
// resident model with it. The fixed kernels record a per-frame error
// (inference/kernels/frame_error.cuh), return bounded results, and the
// context keeps running - asserted here by launching and syncing work
// AFTER every injection.
//
// UE8M0 receipt (C-UE8M0): LmFloatToUe8m0 must round to nearest - the
// pinned table below encodes the upper half-octave UP one step where the
// retired cvt.rz spelling truncated it down. The values come from
// tests/test_ue8m0_encoder_oracle.py.
//
// Build (spark5): tools/kernel_crew/spark_frame_error_probe_build.sh
// PASS = "FRAME-ERROR-PROBE PASS" and exit 0.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/attn.cuh"
#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

// -- module launchers under test (their .cu files link alongside) ------------

extern "C" cudaError_t SparkQwen4FlashLaunchGroupedExpertLinear(
	cudaStream_t stream, const SparkQwen4FlashLinearView *view,
	const void *input_bf16, const uint32_t *source_row_map,
	const uint32_t *group_row_offset, const uint32_t *group_tile_prefix,
	void *output_bf16, uint32_t source_row_count,
	uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank,
	uint32_t route_group_base, const void *frame_error);

extern "C" cudaError_t SparkDsv4LaunchSparseAttn(
	cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16,
	uint64_t lane_stride_elements, const uint32_t *row_lane_indices,
	const uint32_t *row_page_table_indices, const uint32_t *physical_page_table,
	uint32_t page_table_stride, uint32_t page_table_rows,
	uint32_t compressed_entries_per_page, uint32_t pool_page_count,
	const int32_t *topk_idxs, const uint32_t *valid_topk_counts, uint32_t topk,
	const float *sink_f32, float scale, void *out_bf16, float *partials_f32,
	uint32_t partial_capacity, uint32_t multiprocessor_count,
	uint32_t row_count, uint32_t head_count, uint32_t head_dim,
	const void *frame_error);

extern "C" cudaError_t SparkQwen38_27bLaunchLinear(cudaStream_t stream,
	const SparkQwen38_27bLinearView *view, const void *input_bf16,
	void *output_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen38_27bFrameErrorClear(cudaStream_t stream);
extern "C" cudaError_t SparkQwen38_27bFrameErrorRead(void *destination, cudaStream_t stream);

// -- harness -----------------------------------------------------------------

static int32_t failures;
static cudaStream_t probe_stream;

static void Expect(int32_t condition, const char *label)
{
	if ( condition )
		printf("ok %s\n", label);
	else
	{
		printf("FAIL %s\n", label);
		++failures;
	}
}

// THE CONTEXT LIVES: work submitted after every injection must complete.
static float *probe_scratch;
static void ExpectContextAlive(const char *after)
{
	char label[160];
	cudaError_t error = cudaMemsetAsync(probe_scratch, 0,
		64u * sizeof(float), probe_stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(probe_stream);
	snprintf(label, sizeof(label),
		"the CUDA context survives the %s injection", after);
	Expect(error == cudaSuccess, label);
}

__global__ void LmProbeUe8m0Kernel(const float *values, uint8_t *encoded)
{
	uint32_t index = threadIdx.x;
	encoded[index] = LmFloatToUe8m0(values[index]);
}

__global__ void LmProbeScaleKernel(float *values, uint32_t count)
{
	for (uint32_t index = threadIdx.x; index < count; index += blockDim.x)
		values[index] *= 2.0f;
}

int main(void)
{
	int32_t device = 0;
	int32_t multiprocessors = 1;
	cudaError_t error = cudaSetDevice(0);
	if ( error == cudaSuccess )
		error = cudaGetDevice(&device);
	if ( error == cudaSuccess )
		error = cudaDeviceGetAttribute(&multiprocessors,
			cudaDevAttrMultiProcessorCount, device);
	if ( error != cudaSuccess )
	{
		printf("FAIL device probe needs a CUDA device: %s\n",
			cudaGetErrorString(error));
		return 1;
	}
	cudaStreamCreate(&probe_stream);
	error = cudaMalloc(&probe_scratch, 64u * sizeof(float));
	if ( error != cudaSuccess )
	{
		printf("FAIL probe scratch: %s\n", cudaGetErrorString(error));
		return 1;
	}

	// =====================================================================
	// PROBE 1 - K1/KV + K2 shared layer: a wild sparse position through the
	// real attention kernel records a frame failure instead of trapping on
	// an unmapped page.
	// =====================================================================
	{
		using ProbeKv = LmKvGeometry<32u, 2u, true>;
		static uint8_t *pool;
		static uint32_t *page_table;
		static LmKvAccessError *access_error;
		uint16_t *latent_query;
		uint16_t *latent_output;
		uint32_t *sequence;
		uint32_t *context_length;
		uint32_t *selected_positions;
		error = cudaMalloc(&pool, ProbeKv::kPageBytes);
		if ( error == cudaSuccess ) error = cudaMalloc(&page_table, sizeof(uint32_t));
		if ( error == cudaSuccess ) error = cudaMalloc(&access_error, sizeof(LmKvAccessError));
		if ( error == cudaSuccess ) error = cudaMalloc(&latent_query, 16u * sizeof(uint16_t));
		if ( error == cudaSuccess ) error = cudaMalloc(&latent_output, 12u * sizeof(uint16_t));
		if ( error == cudaSuccess ) error = cudaMalloc(&sequence, sizeof(uint32_t));
		if ( error == cudaSuccess ) error = cudaMalloc(&context_length, sizeof(uint32_t));
		if ( error == cudaSuccess ) error = cudaMalloc(&selected_positions, 2u * sizeof(uint32_t));
		const uint32_t unmapped = LM_KV_PAGE_UNMAPPED;
		const uint32_t zero = 0u, one = 1u, wild = 128u;
		error = cudaMemcpy(page_table, &unmapped, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(sequence, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(context_length, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(selected_positions, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(selected_positions + 1u, &wild, sizeof(uint32_t), cudaMemcpyHostToDevice);
		LmKvView host_view;
		memset(&host_view, 0, sizeof(host_view));
		host_view.pool = pool;
		host_view.page_table = page_table;
		host_view.page_table_stride = 1u;
		host_view.sequence_count = 1u;
		host_view.pool_page_count = 1u;
		host_view.access_error = access_error;
		error = cudaMemsetAsync(access_error, 0, sizeof(LmKvAccessError), probe_stream);
		if ( error == cudaSuccess )
		{
			LmAttentionDecodeKernel<ProbeKv, 64u, 12u, 4u>
				<<<dim3(1u, 1u), 64u, 0u, probe_stream>>>(
					latent_query, latent_query, host_view, sequence,
					context_length, selected_positions, 2u, 1u, 1.0f,
					latent_output, 0);
			error = cudaStreamSynchronize(probe_stream);
		}
		LmKvAccessError host_error;
		memset(&host_error, 0, sizeof(host_error));
		if ( error == cudaSuccess )
			error = cudaMemcpy(&host_error, access_error, sizeof(host_error), cudaMemcpyDeviceToHost);
		Expect(error == cudaSuccess && host_error.error_code == LM_KV_ACCESS_ERROR_PAGE_UNMAPPED,
			"wild sparse position records a frame failure, launch stays clean");
		ExpectContextAlive("shared-attention");
		cudaFree(pool); cudaFree(page_table); cudaFree(access_error);
		cudaFree(latent_query); cudaFree(latent_output);
		cudaFree(sequence); cudaFree(context_length); cudaFree(selected_positions);
	}

	// =====================================================================
	// PROBE 2 - K1/qwen4: corrupt route map through the shipping grouped
	// expert linear (E8M0 family-local kernel, rows < 8).
	// =====================================================================
	{
		uint32_t *frame_error;
		error = cudaMalloc(&frame_error, LM_FRAME_ERROR_WORDS * sizeof(uint32_t));
		error = cudaMemsetAsync(frame_error, 0, LM_FRAME_ERROR_WORDS * sizeof(uint32_t), probe_stream);
		const uint32_t experts =
			SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT;
		const uint32_t rows_per_expert = 128u;
		SparkQwen4FlashLinearView view;
		memset(&view, 0, sizeof(view));
		view.abi_version = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
		view.weight_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128;
		view.input_dimension = 128u;
		view.output_dimension = experts * rows_per_expert;
		error = cudaMalloc((void **)&view.weight_payload,
			(uint64_t)experts * rows_per_expert * view.input_dimension);
		error = cudaMalloc((void **)&view.weight_scale_e8m0,
			(uint64_t)experts * rows_per_expert * (view.input_dimension / 128u));
		if ( error == cudaSuccess )
			error = cudaMemsetAsync((void *)view.weight_payload, 0,
				(uint64_t)experts * rows_per_expert * view.input_dimension, probe_stream);
		uint32_t *group_row_offset;
		uint32_t *group_tile_prefix;
		uint32_t *source_row_map;
		void *input_bf16;
		void *output_bf16;
		uint32_t host_offsets[experts + 1u];
		uint32_t host_prefix[experts + 1u];
		for (uint32_t g = 0u; g <= experts; ++g)
		{
			// one packed row in group 0: offsets [0,1,1,...,1]
			host_offsets[g] = g == 0u ? 0u : 1u;
			host_prefix[g] = g < experts ? g * 8u : experts * 8u;
		}
		error = cudaMalloc(&group_row_offset, sizeof(host_offsets));
		error = cudaMalloc(&group_tile_prefix, sizeof(host_prefix));
		error = cudaMalloc(&source_row_map, sizeof(uint32_t));
		error = cudaMalloc(&input_bf16, 128u * sizeof(uint16_t));
		error = cudaMalloc(&output_bf16,
			(uint64_t)experts * rows_per_expert * sizeof(uint16_t));
		const uint32_t corrupt_source_row = 0x7fffffffu;
		error = cudaMemcpy(group_row_offset, host_offsets, sizeof(host_offsets), cudaMemcpyHostToDevice);
		error = cudaMemcpy(group_tile_prefix, host_prefix, sizeof(host_prefix), cudaMemcpyHostToDevice);
		error = cudaMemcpy(source_row_map, &corrupt_source_row, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = SparkQwen4FlashLaunchGroupedExpertLinear(probe_stream, &view,
			input_bf16, source_row_map, group_row_offset, group_tile_prefix,
			output_bf16, 1u, (uint32_t)multiprocessors, 1u, 0u, 0u, frame_error);
		uint32_t host_frame[LM_FRAME_ERROR_WORDS] = {0u};
		if ( error == cudaSuccess )
			error = cudaMemcpy(host_frame, frame_error, sizeof(host_frame), cudaMemcpyDeviceToHost);
		Expect(error == cudaSuccess &&
			host_frame[0] == LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
			"corrupt route map through qwen4 expert linear records "
			"ROUTE_MAP_OUT_OF_RANGE instead of trapping");
		ExpectContextAlive("qwen4 route-map");
		cudaFree(frame_error); cudaFree((void *)view.weight_payload);
		cudaFree((void *)view.weight_scale_e8m0);
		cudaFree(group_row_offset); cudaFree(group_tile_prefix);
		cudaFree(source_row_map); cudaFree(input_bf16); cudaFree(output_bf16);
	}

	// =====================================================================
	// PROBE 3 - K2/dsv4: a corrupt top-k index through the shipping sparse
	// attention launcher is bounds-checked at the index consumer.
	// =====================================================================
	{
		uint32_t *frame_error;
		error = cudaMalloc(&frame_error, LM_FRAME_ERROR_WORDS * sizeof(uint32_t));
		error = cudaMemsetAsync(frame_error, 0, LM_FRAME_ERROR_WORDS * sizeof(uint32_t), probe_stream);
		const uint32_t head_dim = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
		const uint32_t head_count = 8u;
		const uint32_t topk = 8u;
		void *q_bf16;
		void *kv_cache_bf16;
		void *out_bf16;
		float *partials_f32;
		float *sink_f32;
		uint32_t *row_lane_indices;
		uint32_t *row_page_table_indices;
		uint32_t *physical_page_table;
		int32_t *topk_idxs;
		uint32_t *valid_topk_counts;
		error = cudaMalloc(&q_bf16, (uint64_t)head_count * head_dim * sizeof(uint16_t));
		error = cudaMalloc(&kv_cache_bf16, 4096u * sizeof(uint16_t));
		error = cudaMalloc(&out_bf16, (uint64_t)head_count * head_dim * sizeof(uint16_t));
		error = cudaMalloc(&partials_f32, 65536u * sizeof(float));
		error = cudaMalloc(&sink_f32, sizeof(float));
		error = cudaMalloc(&row_lane_indices, sizeof(uint32_t));
		error = cudaMalloc(&row_page_table_indices, sizeof(uint32_t));
		error = cudaMalloc(&physical_page_table, sizeof(uint32_t));
		error = cudaMalloc(&topk_idxs, topk * sizeof(int32_t));
		error = cudaMalloc(&valid_topk_counts, sizeof(uint32_t));
		const uint32_t zero = 0u, one = 1u;
		const int32_t corrupt_index = 0x7fffffff;
		error = cudaMemcpy(row_lane_indices, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(row_page_table_indices, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(physical_page_table, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(topk_idxs, &corrupt_index, sizeof(int32_t), cudaMemcpyHostToDevice);
		error = cudaMemcpy(valid_topk_counts, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);
		error = SparkDsv4LaunchSparseAttn(probe_stream, q_bf16, kv_cache_bf16,
			4096u, row_lane_indices, row_page_table_indices,
			physical_page_table, 1u, 1u, 64u, 1u, topk_idxs,
			valid_topk_counts, topk, sink_f32, 0.125f, out_bf16,
			partials_f32, 65536u, (uint32_t)multiprocessors, 1u, head_count,
			head_dim, frame_error);
		uint32_t host_frame[LM_FRAME_ERROR_WORDS] = {0u};
		if ( error == cudaSuccess )
			error = cudaMemcpy(host_frame, frame_error, sizeof(host_frame), cudaMemcpyDeviceToHost);
		Expect(error == cudaSuccess &&
			host_frame[0] == LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE,
			"corrupt dsv4 top-k index records SPARSE_INDEX_OUT_OF_RANGE "
			"instead of reading a wild KV address");
		ExpectContextAlive("dsv4 sparse-attn");
		cudaFree(frame_error); cudaFree(q_bf16); cudaFree(kv_cache_bf16);
		cudaFree(out_bf16); cudaFree(partials_f32); cudaFree(sink_f32);
		cudaFree(row_lane_indices); cudaFree(row_page_table_indices);
		cudaFree(physical_page_table); cudaFree(topk_idxs);
		cudaFree(valid_topk_counts);
	}

	// =====================================================================
	// PROBE 4 - K4/qwen38: an rANS chunk span past the fixed 20000-byte
	// shared window is bounded and reported, not copied through shared
	// memory and past it.
	// =====================================================================
	{
		SparkQwen38_27bLinearView view;
		memset(&view, 0, sizeof(view));
		view.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
		view.weight_format = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS;
		view.input_dimension = 128u;
		view.output_dimension = 64u;
		/* Header: ndirect=0, id_bits=1, chunk_count=1, offsets[0]=4 - and
		 * the final chunk's end is payload_bytes, declared a gigabyte out.
		 * end - off overflows the fixed 20000-byte stage window. */
		uint8_t host_payload[64];
		memset(host_payload, 0, sizeof(host_payload));
		host_payload[24] = 1u;  // chunk_count = 1
		host_payload[28] = 4u;  // offsets[0] = 4
		error = cudaMalloc((void **)&view.weight_payload, sizeof(host_payload));
		error = cudaMemcpy((void *)view.weight_payload, host_payload,
			sizeof(host_payload), cudaMemcpyHostToDevice);
		view.weight_payload_bytes = 1ull << 30;
		void *input_bf16;
		void *output_bf16;
		error = cudaMalloc(&input_bf16, 128u * sizeof(uint16_t));
		error = cudaMalloc(&output_bf16, 64u * sizeof(uint16_t));
		error = SparkQwen38_27bFrameErrorClear(probe_stream);
		error = SparkQwen38_27bLaunchLinear(probe_stream, &view, input_bf16,
			output_bf16, 1u);
		uint32_t host_frame[LM_FRAME_ERROR_WORDS] = {0u};
		if ( error == cudaSuccess )
			error = SparkQwen38_27bFrameErrorRead(host_frame, probe_stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(probe_stream);
		Expect(error == cudaSuccess &&
			host_frame[0] == LM_FRAME_ERROR_PAYLOAD_WINDOW_OUT_OF_RANGE,
			"oversized rANS chunk records PAYLOAD_WINDOW_OUT_OF_RANGE "
			"instead of streaming past the shared window");
		ExpectContextAlive("qwen38 rANS");
		cudaFree((void *)view.weight_payload);
		cudaFree(input_bf16); cudaFree(output_bf16);
	}

	// =====================================================================
	// PROBE 5 - K3/UE8M0: the encoder rounds to nearest. The upper
	// half-octave moves one step where cvt.rz truncated it; the lower
	// half-octave - the certified receipts - does not move.
	// =====================================================================
	{
		/* From tests/test_ue8m0_encoder_oracle.py. Expected bytes:
		 * 1.7 -> 128 (2.0), 2.9 -> 129 (4.0), 1.0625 -> 127 (1.0),
		 * 0.6 -> 126 (0.5), 0.35 -> 125 (0.25), 5.2 -> 129 (4.0). */
		static const float host_values[6] = {
			1.7f, 2.9f, 1.0625f, 0.6f, 0.35f, 5.2f };
		static const uint8_t host_expected[6] = { 128u, 129u, 127u, 126u, 125u, 129u };
		float *values;
		uint8_t *encoded;
		uint8_t host_encoded[6];
		error = cudaMalloc(&values, sizeof(host_values));
		error = cudaMalloc(&encoded, sizeof(host_encoded));
		error = cudaMemcpy(values, host_values, sizeof(host_values), cudaMemcpyHostToDevice);
		LmProbeUe8m0Kernel<<<1u, 6u, 0u, probe_stream>>>(values, encoded);
		error = cudaMemcpy(host_encoded, encoded, sizeof(host_encoded), cudaMemcpyDeviceToHost);
		Expect(error == cudaSuccess &&
			memcmp(host_encoded, host_expected, sizeof(host_expected)) == 0,
			"LmFloatToUe8m0 rounds to nearest on device (upper half-octave "
			"up, lower half-octave unchanged)");
		// and the context still runs
		float *scratch;
		error = cudaMalloc(&scratch, 64u * sizeof(float));
		LmProbeScaleKernel<<<1u, 64u, 0u, probe_stream>>>(scratch, 64u);
		error = cudaStreamSynchronize(probe_stream);
		Expect(error == cudaSuccess, "the CUDA context survives every probe");
		cudaFree(values); cudaFree(encoded); cudaFree(scratch);
	}

	cudaFree(probe_scratch);
	cudaStreamDestroy(probe_stream);
	printf("%s (%d failures)\n", failures == 0 ? "FRAME-ERROR-PROBE PASS" : "FAIL", failures);
	return failures == 0 ? 0 : 1;
}
