// K3 resident decode stage runner: embed -> slice -> head over the dispatch,
// with the TP4 all-reduce wiring (see the header for the ownership layout).
//
// The slice's layer_collective hook fires after every layer and all-reduces
// the input-sharded projection destinations the layer wrote WITHOUT the
// partial fold (tp_sharded): attention_out (kda_out / mla_out), hidden
// (routed_up / dense_down) and shared_out (shared_w2). The attention fold
// keeps the fused path's rule - SET at a block boundary (the slice already
// set the partial to the LOCAL attention output there, and the summed value
// replaces it), ADD otherwise; the MLP folds are ALWAYS adds, because the
// boundary restart already happened with the attention output and the MoE
// half accumulates on top of it. The embedding (stage 0) and the head
// candidates (last stage) use the same ring collective; the head exchange
// is a slot-encoded SUM - each rank fills only its slot, so the sum over
// the rank slots IS the all-gather of the four local argmaxes and the
// winner reduces locally.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"

/* the kernels below take the state through opaque contexts; forward */
typedef struct SparkK3RunnerState SparkK3RunnerState;

// The dense GEMM's row offsets are [0, rows] per step; a device-side write
// keeps the hot path free of host traffic (the CUDA-graph capture contract).
__global__ static void K3RunnerDenseOffsetsKernel(uint32_t *offsets, uint32_t rows)
{
	if ( threadIdx.x == 0u )
	{
		offsets[0] = 0u;
		offsets[1] = rows;
	}
}

// The stream-ordered completion the device tier's submission carries: fold
// the summed fused segment(s) into the AttnRes partial with the
// fused-epilogue rule (SET at a boundary restart, ADD otherwise), then free
// the heap context. The folds land on the SUBMISSION's stream, so the order
// (NCCL kernels, then folds, then the next consumer) holds without the
// legacy default stream.
typedef struct SparkK3RunnerTpContext
{
	K3LayerBuffers *buffers;
	uint16_t *fused;
	cudaStream_t stream;
	uint32_t rows;
	uint32_t boundary;
	uint32_t segments;
	uint32_t phase;
} SparkK3RunnerTpContext;

static void K3RunnerTpCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkK3RunnerTpContext *tp = (SparkK3RunnerTpContext *)context;
	(void)completion;
	K3LayerBuffers *b = tp->buffers;
	uint32_t rows = tp->rows;
	uint32_t elements = rows * K3_HIDDEN;
	uint16_t *fused = tp->fused;
	if ( tp->phase == 0u )
	{
		/* the attention segment: restart at a boundary, accumulate otherwise */
		if ( tp->boundary != 0u )
			K3PartialSet(b, fused, rows, tp->stream);
		else
			K3PartialAdd(b, fused, rows, tp->stream);
	}
	else
	{
		/* the MLP segment(s): routed_up (the dense_down at layer 0) and the
		 * shared_w2 always accumulate on top of the restart */
		K3PartialAdd(b, fused, rows, tp->stream);
		if ( tp->segments == 2u )
			K3PartialAdd(b, fused + elements, rows, tp->stream);
	}
	delete tp;
}

// The per-phase pack: phase 0 stages attention_out, phase 1 stages hidden
// (routed_up / dense_down) and, for routed layers, shared_out after it.
__global__ static void K3RunnerFusedPackKernel(const uint16_t *attention,
	const uint16_t *hidden,const uint16_t *shared,uint16_t *fused,
	uint32_t rows,uint32_t phase,uint32_t segments)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * K3_HIDDEN;
	if ( i >= elements )
		return;
	if ( phase == 0u )
		fused[i] = attention[i];
	else
	{
		fused[i] = hidden[i];
		if ( segments == 2u )
			fused[elements + i] = shared[i];
	}
}

// The recursive TP4 BF16 tree the device collective's direct all-to-all
// contract specifies: round(0+1), round(2+3), round(local+remote), each
// widen-add-narrow so every rank lands the bit-identical BF16 sum.
__global__ static void K3RunnerCombineTp4TreeKernel(const uint16_t *const *rank_devices,
	uint16_t *destination,uint32_t tp_rank,uint32_t rows,uint32_t hidden_dimension)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * hidden_dimension;
	if ( i >= elements )
		return;
	const uint16_t *r0 = rank_devices[0];
	const uint16_t *r1 = rank_devices[1];
	const uint16_t *r2 = rank_devices[2];
	const uint16_t *r3 = rank_devices[3];
	float a = LmBf16ToFloat(r0[i]) + LmBf16ToFloat(r1[i]);
	float b = LmBf16ToFloat(r2[i]) + LmBf16ToFloat(r3[i]);
	destination[i] = LmFloatToBf16(a + b);
	(void)tp_rank;
}

// The two-rank BF16 combine (the collective's plain fallback path).
static SparkStatus K3RunnerCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	(void)combine_context;
	const uint16_t *pair[4] = { (const uint16_t *)destination_device,
		(const uint16_t *)source_device, 0, 0 };
	K3RunnerCombineTp4TreeKernel<<<(active_sequence_count * hidden_dimension + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		pair,(uint16_t *)destination_device,0u,active_sequence_count,hidden_dimension);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

// Host side of the combine contract (the device collective's
// CombineTp4Bf16 function pointer).
static SparkStatus K3RunnerCombineTp4Bf16(void *combine_context,
	void *destination_device,const void *const rank_devices[4],uint32_t tp_rank,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)combine_context;
	(void)state;
	K3RunnerCombineTp4TreeKernel<<<(active_sequence_count * hidden_dimension + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		(const uint16_t *const *)rank_devices,(uint16_t *)destination_device,
		tp_rank,active_sequence_count,hidden_dimension);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

typedef struct SparkK3RunnerState SparkK3RunnerState;

typedef struct SparkK3RunnerState
{
	SparkK3ModuleState module;
	SparkK3Dispatch dispatch;
	SparkTpCollective collective;
	int collective_created;
	SparkTpDeviceCollective device_collective;
	int device_collective_created;
	uint32_t tp_rank;
	uint16_t *fused_device;
	uint32_t fused_rows;
	uint64_t tp_next_ordinal;
	uint32_t rows;              /* the step in flight, for the hook */
	const uint16_t *embed_weight;
	const uint16_t *head_norm_weight;
	const uint16_t *head_weight;
	uint32_t vocab;
	uint32_t vocab_slice_rows;
	/* host staging for the BF16 collective: values + scratch */
	uint16_t *staging_values;
	uint16_t *staging_scratch;
	uint32_t staging_capacity;
	/* the fused per-layer collective packs attention_out | hidden |
	 * shared_out into one staging buffer (3 x rows x K3_HIDDEN) */
	uint32_t fused_capacity;
	/* head candidate slot exchange: rows x (2 * tp_degree) floats */
	float *head_slots_host;
	float *head_slots_device;
	uint32_t head_slots_capacity;
	/* per-step device arrays the dispatch step consumes */
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *head_candidate_token;
	float *head_candidate_score;
	uint32_t head_tiles;
	uint32_t *output_token;
	float *output_score;
	uint32_t *output_token_host;
	float *output_score_host;
	cudaStream_t stream;
	uint32_t max_rows;
	uint32_t max_context;
	uint32_t multiprocessors;
	uint64_t kv_page_bytes;
	/* captured slice graphs, keyed by rows (sequences = rows, commit = 1,
	 * packed_rows = rows * K3_TOP_K - all deterministic per shape) */
	struct
	{
		cudaGraphExec_t executable;
		uint32_t rows;
		uint32_t warm;
		uint32_t live;
	} graphs[4];
	uint32_t graph_capture_enabled;
	uint32_t graphs_broken;
} SparkK3RunnerState;

/* The dense-offset kernel + the whole slice launch form the capture unit. */
static int32_t K3RunnerLaunchSliceDirect(SparkK3RunnerState *state,
	SparkK3StepInput *in, uint32_t rows, uint32_t sequences,
	uint32_t packed_rows, cudaStream_t stream)
{
	K3RunnerDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, rows);
	return SparkK3DispatchStep(&state->dispatch, in, rows, sequences, 1u,
		packed_rows, state->max_context, state->multiprocessors, stream);
}

/* Capture-safe only with no collective (tp_degree 1: the layer folds its own
 * projections and the hook no-ops) or with the NCCL device tier (the host
 * tiers' syncs and host staging are not replayable), and never on the legacy
 * default stream. */
static uint32_t K3RunnerGraphsEligible(const SparkK3RunnerState *state,
	cudaStream_t stream)
{
	if ( state->graph_capture_enabled == 0u || state->graphs_broken != 0u ||
		stream == 0 )
		return 0u;
	if ( state->device_collective_created == 0 )
		return 1u;
	return state->device_collective.backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL ? 1u : 0u;
}

static int32_t K3RunnerLaunchSliceGraph(SparkK3RunnerState *state,
	SparkK3StepInput *in, uint32_t rows, uint32_t sequences,
	uint32_t packed_rows, cudaStream_t stream)
{
	SparkK3RunnerState *s = state;
	uint32_t i;
	for ( i = 0u; i < 4u; ++i )
	{
		if ( s->graphs[i].live != 0u && s->graphs[i].rows == rows )
		{
			if ( s->graphs[i].executable != 0 )
			{
				cudaError_t error = cudaGraphLaunch(s->graphs[i].executable,
					stream);
				return error == cudaSuccess ? SPARK_K3_DISPATCH_OK :
					SPARK_K3_DISPATCH_ERR_CUDA;
			}
			/* first sighting was the warm run; capture now */
			break;
		}
	}
	/* First sighting of this shape: run the warm step DIRECT (the shared-memory
	 * opt-ins and tensor-map encodes must precede capture), slot it, and
	 * capture on the next submit. */
	if ( i == 4u )
	{
		for ( i = 0u; i < 4u; ++i )
			if ( s->graphs[i].live == 0u )
			{
				s->graphs[i].rows = rows;
				s->graphs[i].warm = 1u;
				s->graphs[i].live = 1u;
				break;
			}
		return K3RunnerLaunchSliceDirect(s, in, rows, sequences,
			packed_rows, stream);
	}
	cudaGraph_t graph = 0;
	cudaError_t begin_error = cudaStreamBeginCapture(stream,
		cudaStreamCaptureModeRelaxed);
	if ( begin_error != cudaSuccess )
		{ s->graphs_broken = 1u; return K3RunnerLaunchSliceDirect(s, in, rows, sequences, packed_rows, stream); }
	int32_t status = K3RunnerLaunchSliceDirect(s, in, rows, sequences,
		packed_rows, stream);
	cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
	if ( end_error != cudaSuccess || status != SPARK_K3_DISPATCH_OK || graph == 0 )
	{
		/* An invalidated capture discards the recorded work - the step did
		 * NOT execute. Run it directly and disable the graph path. */
		if ( graph != 0 )
			(void)cudaGraphDestroy(graph);
		s->graphs_broken = 1u;
		return K3RunnerLaunchSliceDirect(s, in, rows, sequences, packed_rows, stream);
	}
	cudaGraphExec_t executable = 0;
	cudaError_t instantiate_error = cudaGraphInstantiate(&executable, graph, 0ull);
	(void)cudaGraphDestroy(graph);
	if ( instantiate_error != cudaSuccess )
	{
		s->graphs_broken = 1u;
		return K3RunnerLaunchSliceDirect(s, in, rows, sequences, packed_rows, stream);
	}
	/* store (reuse the warm slot when found, else the first free one) */
	for ( i = 0u; i < 4u; ++i )
		if ( s->graphs[i].live != 0u && s->graphs[i].rows == rows )
		{
			s->graphs[i].executable = executable;
			return cudaGraphLaunch(executable, stream) == cudaSuccess ?
				SPARK_K3_DISPATCH_OK : SPARK_K3_DISPATCH_ERR_CUDA;
		}
	for ( i = 0u; i < 4u; ++i )
		if ( s->graphs[i].live == 0u )
		{
			s->graphs[i].executable = executable;
			s->graphs[i].rows = rows;
			s->graphs[i].warm = 1u;
			s->graphs[i].live = 1u;
			return cudaGraphLaunch(executable, stream) == cudaSuccess ?
				SPARK_K3_DISPATCH_OK : SPARK_K3_DISPATCH_ERR_CUDA;
		}
	/* no slot: keep the executable orphaned-safe by launching, then leak-free
	 * teardown cannot reach it - destroy immediately and run direct */
	(void)cudaGraphExecDestroy(executable);
	s->graphs_broken = 1u;
	return K3RunnerLaunchSliceDirect(s, in, rows, sequences, packed_rows, stream);
}

static uint32_t K3RunnerFirstLayer(uint32_t stage_index)
{
	static const uint32_t first[4] = { 0u, 24u, 47u, 70u };
	return(first[stage_index % 4u]);
}

static uint32_t K3RunnerLayerCount(uint32_t stage_index)
{
	static const uint32_t count[4] = { 24u, 23u, 23u, 23u };
	return(count[stage_index % 4u]);
}

// One BF16 all-reduce of a rows x K3_HIDDEN device tensor: sync the stream,
// stage to the host, all-reduce in place, upload. The sync-per-projection is
// the host-collective tier's known cost; the device-direct tier replaces it
// without changing this file's contract.
static void K3RunnerEmbedCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

static SparkStatus K3RunnerReduceBf16(SparkK3RunnerState *state, cudaStream_t stream,
	const uint16_t *device_values, uint32_t rows)
{
	SparkStatus status;
	uint32_t elements = rows * K3_HIDDEN;
	if ( state->device_collective_created != 0 &&
		state->device_collective.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		/* THE DEVICE TIER: the embedding is slot-encoded (the out-of-slice
		 * rank contributes zero), so ONE stream-ordered all-reduce of the
		 * rows x K3_HIDDEN buffer IS the embedding exchange - no sync, no
		 * host staging. The buffer reduces in place; nothing folds. NCCL
		 * only: the hidden-transport tier cannot narrow its pre-registered
		 * frame. */
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission, 0, sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = rows;
		submission.flags =
			SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
		submission.ordinal = state->tp_next_ordinal++;
		submission.reserved0 = elements;
		submission.local_device = device_values;
		submission.full_device = (void *)device_values;
		submission.cuda_stream = stream;
		submission.completion_function = K3RunnerEmbedCompletion;
		submission.completion_context = 0;
		return SparkTpDeviceCollectiveSubmitBf16(&state->device_collective,
			&submission);
	}
	if ( state->collective_created == 0 )
		return SPARK_STATUS_OK;
	cudaError_t error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	error = cudaMemcpy(state->staging_values, device_values,
		(uint64_t)elements * 2u, cudaMemcpyDeviceToHost);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = SparkTpCollectiveAllReduceSumBf16(&state->collective,
		state->staging_values, elements, state->staging_scratch);
	if ( status != SPARK_STATUS_OK )
		return status;
	error = cudaMemcpy((void *)device_values, state->staging_values,
		(uint64_t)elements * 2u, cudaMemcpyHostToDevice);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

// The slice's per-layer hook, fired in two phases. Phase 0 all-reduces
// and folds the attention output (kda_out / mla_out) BEFORE the MLP-side
// retrieval, whose partial mix must contain the post-attention contribution;
// phase 1 does the MoE's routed_up (layer 0's dense_down) and, for routed
// layers, the shared_w2. At tp_degree 1 the layer folded its own projections
// (tp_sharded = 0) and the hook is a no-op.
/* ENV-GATED STATE DUMPS for the offline equivalence bisect: K3_HOOK_DUMP =
 * a path prefix; the hook writes the raw BF16 states per rank/layer/phase
 * (attention_out, hidden, shared_out, the AttnRes partial). The tp_degree 1
 * leg dumps at the hook entry (the layer's own values); the TP4 legs dump
 * again after the all-reduce and the fold - the two are directly
 * comparable. */
static const char *K3RunnerDumpPrefix(void)
{
	static const char *prefix = 0;
	static int checked = 0;
	if ( checked == 0 )
	{
		prefix = getenv("K3_HOOK_DUMP");
		checked = 1;
	}
	return prefix;
}

static void K3RunnerDumpTensor(const char *prefix, uint32_t rank,
	uint32_t layer, uint32_t phase, const char *stage, const char *name,
	const uint16_t *device, uint32_t elements)
{
	char path[320];
	FILE *file;
	snprintf(path, sizeof(path), "%s_r%u_l%u_p%u_%s_%s.bin",
		prefix, rank, layer, phase, stage, name);
	file = fopen(path, "wb");
	if ( file == 0 )
		return;
	{
		std::vector<uint16_t> host(elements);
		cudaMemcpy(host.data(), device, (uint64_t)elements * 2u,
			cudaMemcpyDeviceToHost);
		fwrite(host.data(), 2u, elements, file);
	}
	fclose(file);
}

static void K3RunnerHookDump(SparkK3RunnerState *state, uint32_t rank,
	uint32_t layer, uint32_t phase, const char *stage, uint32_t elements)
{
	const char *prefix = K3RunnerDumpPrefix();
	K3LayerBuffers *b;
	if ( prefix == 0 || prefix[0] == '\0' )
		return;
	b = state->dispatch.buffers;
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "attn_out",
		b->attention_out_bf16, elements);
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "hidden",
		b->hidden_bf16, elements);
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "shared",
		b->shared_out_bf16, elements);
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "partial",
		b->attnres_partial_bf16, elements);
	/* the MLA intermediates: the per-head query (rows x rank q dim) and the
	 * gated value (rows x rank v dim) */
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "query",
		b->query_bf16, state->rows * K3_RANK_DIM(b, mla_q_up_rows, K3_MLA_Q_DIM));
	K3RunnerDumpTensor(prefix, rank, layer, phase, stage, "value",
		b->value_bf16, state->rows * K3_RANK_DIM(b, mla_out_input, K3_MLA_OUT_DIM));
}

static void K3RunnerLayerCollective(void *context, void *stream_void,
	uint32_t layer, uint32_t phase)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)context;
	K3LayerBuffers *b = state->dispatch.buffers;
	cudaStream_t stream = (cudaStream_t)stream_void;
	uint32_t rows = state->rows;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u;
	uint32_t elements = rows * K3_HIDDEN;
	uint32_t segments = phase == 0u ? 1u : (layer == 0u ? 1u : 2u);
	/* THE KDA PHASE-0 SOURCE IS hidden_bf16, NOT attention_out: the o_proj
	 * writes its output there (the in-place GEMM fix in layer.cuh). The MLA
	 * o_proj still lands in attention_out. */
	uint16_t *phase0_source =
		(K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT)
			? b->hidden_bf16 : b->attention_out_bf16;
	K3RunnerHookDump(state, state->tp_rank, layer, phase, "pre", elements);
	if ( b->tp_sharded == 0u )
		return;
	if ( state->device_collective_created != 0 )
	{
		/* THE DEVICE TIER: one pack kernel + ONE stream-ordered combine
		 * submission per phase; the completion folds the summed segment(s)
		 * into the partial on the same stream. No sync, no host staging. */
		K3RunnerFusedPackKernel<<<(elements + 255u) / 256u, 256u, 0, stream>>>(
			phase0_source,b->hidden_bf16,b->shared_out_bf16,
			state->fused_device,rows,phase,segments);
		SparkK3RunnerTpContext *completion_context = new SparkK3RunnerTpContext;
		completion_context->fused = state->fused_device;
		completion_context->buffers = b;
		completion_context->stream = stream;
		completion_context->rows = rows;
		completion_context->boundary = boundary;
		completion_context->segments = segments;
		completion_context->phase = phase;
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission, 0, sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = rows;
		submission.flags =
			SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
		submission.ordinal = state->tp_next_ordinal++;
		/* the per-phase payload, not the 3-segment frame: phase 0 ships ONE
		 * segment (14 KB per row), phase 1 one or two - the fixed frame
		 * tripled phase 0's bytes on the wire */
		submission.reserved0 = elements * segments;
		submission.local_device = state->fused_device;
		submission.full_device = state->fused_device;
		submission.cuda_stream = stream;
		submission.completion_function = K3RunnerTpCompletion;
		submission.completion_context = completion_context;
		SparkTpDeviceCollectiveSubmitBf16(&state->device_collective, &submission);
		return;
	}
	if ( state->collective_created != 0 )
	{
		/* THE HOST TIER: per-phase stage + all-reduce + upload + fold. Two
		 * exchanges per layer instead of one - the ordering the MLP-side
		 * retrieval needs - the device tier replaces this whole block with
		 * two stream-ordered combines. */
		cudaStreamSynchronize(stream);
		if ( phase == 0u )
		{
			cudaMemcpy(state->staging_values, phase0_source,
				(uint64_t)elements * 2u, cudaMemcpyDeviceToHost);
			SparkTpCollectiveAllReduceSumBf16(&state->collective,
				state->staging_values, elements, state->staging_scratch);
			cudaMemcpy(phase0_source, state->staging_values,
				(uint64_t)elements * 2u, cudaMemcpyHostToDevice);
			if ( boundary != 0u )
				K3PartialSet(b, phase0_source, rows, stream);
			else
				K3PartialAdd(b, phase0_source, rows, stream);
		}
		else
		{
			cudaMemcpy(state->staging_values, b->hidden_bf16,
				(uint64_t)elements * 2u, cudaMemcpyDeviceToHost);
			if ( segments == 2u )
				cudaMemcpy(state->staging_values + elements, b->shared_out_bf16,
					(uint64_t)elements * 2u, cudaMemcpyDeviceToHost);
			SparkTpCollectiveAllReduceSumBf16(&state->collective,
				state->staging_values, (uint64_t)segments * elements,
				state->staging_scratch);
			cudaMemcpy(b->hidden_bf16, state->staging_values,
				(uint64_t)elements * 2u, cudaMemcpyHostToDevice);
			if ( segments == 2u )
				cudaMemcpy(b->shared_out_bf16, state->staging_values + elements,
					(uint64_t)elements * 2u, cudaMemcpyHostToDevice);
			K3PartialAdd(b, b->hidden_bf16, rows, stream);
			if ( segments == 2u )
				K3PartialAdd(b, b->shared_out_bf16, rows, stream);
		}
		K3RunnerHookDump(state, state->tp_rank, layer, phase, "post", elements);
	}
}
SparkStatus SparkK3StageRunnerInitialize(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerConfiguration *configuration)
{
	SparkK3RunnerState *state;
	SparkStatus status;
	SparkK3PackEntry entry;
	uint32_t routes;
	if ( runner == 0 || configuration == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( configuration->abi_version != SPARK_K3_STAGE_RUNNER_ABI_VERSION ||
		configuration->stage_index >= configuration->stage_count ||
		(configuration->stage_count != 1u &&
			configuration->stage_count != 4u) ||
		configuration->rank_pack_path == 0 ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_input_row_count == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(runner, 0, sizeof(*runner));
	state = new SparkK3RunnerState;
	memset(state, 0, sizeof(*state));
	runner->abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->descriptor_bytes = SPARK_K3_STAGE_RUNNER_BYTES;
	runner->flags = configuration->flags;
	runner->stage_index = configuration->stage_index;
	runner->stage_count = configuration->stage_count;
	runner->tp_degree = configuration->tp_degree;
	runner->tp_rank = configuration->tp_rank;
	runner->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
	runner->owns_final_head = configuration->stage_index + 1u == configuration->stage_count ? 1u : 0u;
	runner->private_state = state;
	state->tp_rank = configuration->tp_rank;
	state->stream = (cudaStream_t)configuration->execution_stream;
	state->max_rows = configuration->max_input_row_count;
	state->max_context = configuration->resident_sequence_capacity;
	state->multiprocessors = configuration->multiprocessors;
	state->graph_capture_enabled =
		(configuration->flags & SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS) != 0u ?
		1u : 0u;
	/* Zero kv_page_bytes means the K3 latent KV geometry (the adapter is
	 * CUDA-free and cannot see it). */
	if ( configuration->kv_page_bytes == 0u )
		state->kv_page_bytes = K3GlobalKv::kPageBytes;
	else
		state->kv_page_bytes = configuration->kv_page_bytes;
	runner->stats.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->stats.descriptor_bytes = (uint32_t)sizeof(SparkK3StageRunnerStats);
	/* Pack + bind + pools + device objects. The PP4 placement checks the
	 * pack slice against the stage tables; the PP1 placement (TP16) takes
	 * the slice bounds from the pack manifest itself. */
	{
		uint32_t first_layer = configuration->stage_count == 4u ?
			K3RunnerFirstLayer(configuration->stage_index) :
			SPARK_K3_MODULE_DERIVE_SLICE;
		uint32_t layer_count = configuration->stage_count == 4u ?
			K3RunnerLayerCount(configuration->stage_index) :
			SPARK_K3_MODULE_DERIVE_SLICE;
		status = SparkK3ModuleInitialize(&state->module,
			configuration->rank_pack_path, first_layer, layer_count);
	}
	if ( status != SPARK_STATUS_OK )
		{ delete state; return status; }
	if ( SparkK3DispatchCreate(&state->dispatch,&state->module.sizing,
		configuration->max_active_sequence_count,
		configuration->max_input_row_count,
		configuration->kv_pages_per_sequence,
		state->kv_page_bytes, 0) != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: dispatch create failed\n"); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	/* The pack mmap registers (in chunks) for UVA weight access: the tensor
	 * maps encode the registered addresses, so a failed registration is
	 * fatal - the unregistered path cannot launch the GEMMs. */
	if ( SparkK3DispatchRegisterPack(&state->module.pack) != SPARK_K3_DISPATCH_OK )
		{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	if ( SparkK3DispatchBindWeights(&state->dispatch,&state->module.pack,
			state->module.bound,state->module.bound_count) != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: weight bind failed\n"); SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	/* The page tables start all-zero (every position maps to physical page
	 * 0); the serving tier owns the real mappings and rewrites them before
	 * publishing a step, but an uninitialised table would be a wild read. */
	cudaMemset(state->dispatch.page_table, 0,
		(uint64_t)state->module.sizing.mla_layer_count *
		configuration->kv_pages_per_sequence * 4u);
	state->vocab = state->module.pack.config.vocab;
	/* The rank pack's embed/lm_head are ALREADY the rank's slice: the
	 * sharder row-split them by the PACK's tp degree, so the rank's rows
	 * come from the tensor's own shape, never vocab / run_degree (which
	 * double-divides for a TP4 pack run single-rank, or vice versa). */
	if ( SparkK3PackLoadEntry(&state->module.pack,"model.embed_tokens.weight",&entry) == 0 &&
		entry.shape_count >= 1u )
		state->vocab_slice_rows = entry.shape[0];
	else
		state->vocab_slice_rows = state->vocab;
	/* The TP contract: sharded ranks defer the partial epilogues to the hook. */
	state->dispatch.buffers->tp_sharded = configuration->tp_degree > 1u ? 1u : 0u;
	/* The hook registers unconditionally: at tp_degree 1 it no-ops (the
	 * layer folds its own projections) except for the env-gated state dumps
	 * the equivalence bisect reads. */
	state->dispatch.slice_state->layer_collective = K3RunnerLayerCollective;
	state->dispatch.slice_state->collective_context = state;
	if ( configuration->tp_degree > 1u )
	{
		if ( configuration->tp_collective == 0 )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return SPARK_STATUS_INVALID_ARGUMENT; }
		fprintf(stderr, "sparkpipe_k3: creating host collective tp=%u rank=%u port=%u\n",
			configuration->tp_degree, configuration->tp_rank,
			configuration->tp_collective->listen_port);
		status = SparkTpCollectiveCreate(configuration->tp_collective,&state->collective);
		fprintf(stderr, "sparkpipe_k3: host collective create -> %d\n", (int)status);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return status; }
		state->collective_created = 1;
	}
	/* The diagnostic override wins over the TP hook (tests use it at
	 * tp_degree 1 to observe the serving path layer by layer). */
	if ( configuration->layer_collective_override != 0 )
	{
		state->dispatch.slice_state->layer_collective =
			configuration->layer_collective_override;
		state->dispatch.slice_state->collective_context =
			configuration->layer_collective_context;
	}
	/* The device-direct tier: the fused buffer + the collective with the
	 * K3 combine kernels. */
	if ( configuration->device_collective != 0 )
	{
		state->fused_rows = configuration->max_input_row_count;
		cudaMalloc(&state->fused_device,
			(uint64_t)state->fused_rows * 3u * K3_HIDDEN * 2u);
		SparkTpDeviceCollectiveConfig device_config =
			*configuration->device_collective;
		/* The K3 combine kernels replace the transport's math for the
		 * hidden-transport backend; NCCL reduces in the library, and its
		 * config validation rejects non-null combine functions. */
		if ( device_config.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		{
			device_config.combine_bf16_function = K3RunnerCombineBf16;
			device_config.combine_tp4_bf16_function = K3RunnerCombineTp4Bf16;
			device_config.combine_context = state;
		}
		status = SparkTpDeviceCollectiveCreate(&device_config,
			&state->device_collective);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); delete state; return status; }
		state->device_collective_created = 1;
	}
	/* Stage 0 and the head stage need the model-level tensors. */
	if ( runner->owns_embedding != 0u &&
		SparkK3PackLoadEntry(&state->module.pack,"model.embed_tokens.weight",&entry) == 0 )
		state->embed_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
	if ( runner->owns_final_head != 0u )
	{
		if ( SparkK3PackLoadEntry(&state->module.pack,"model.norm.weight",&entry) == 0 )
			state->head_norm_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
		if ( SparkK3PackLoadEntry(&state->module.pack,"lm_head.weight",&entry) == 0 )
			state->head_weight = (const uint16_t *)SparkK3PackPayload(&state->module.pack,&entry);
	}
	/* Host staging + head slots + the per-step device arrays. */
	state->staging_capacity = configuration->max_input_row_count * 3u * K3_HIDDEN;
	state->staging_values = new uint16_t[state->staging_capacity];
	state->staging_scratch = new uint16_t[state->staging_capacity];
	state->fused_capacity = state->staging_capacity;
	state->head_slots_capacity = configuration->max_input_row_count * 2u * configuration->tp_degree;
	state->head_slots_host = new float[state->head_slots_capacity];
	cudaMalloc(&state->head_slots_device,(uint64_t)state->head_slots_capacity * 4u);
	routes = configuration->max_input_row_count * K3_TOP_K;
	cudaMalloc(&state->route_expert,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_packed_row,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_source_token,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_weight,(uint64_t)routes * 4u);
	cudaMalloc(&state->group_row_offset,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w1,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w2,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->dense_row_offset, 8u);
	cudaMalloc(&state->dense_tile_prefix, 8u);
	state->head_tiles = (state->vocab_slice_rows + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	cudaMalloc(&state->head_candidate_token,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->head_candidate_score,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->output_token,
		(uint64_t)configuration->max_input_row_count * 4u);
	cudaMalloc(&state->output_score,
		(uint64_t)configuration->max_input_row_count * 4u);
	state->output_token_host = new uint32_t[configuration->max_input_row_count];
	state->output_score_host = new float[configuration->max_input_row_count];
	return SPARK_STATUS_OK;
}

// The head's cross-rank argmax: the slot-encoded sum (see the header).
static SparkStatus K3RunnerHeadExchange(SparkK3RunnerState *state,
	uint32_t rows, uint32_t tp_degree, uint32_t tp_rank,
	uint32_t *output_token_ids, float *output_scores)
{
	uint32_t slots = rows * 2u * tp_degree;
	SparkStatus status;
	uint32_t row, rank_slot, best, r;
	if ( tp_degree <= 1u )
		return SPARK_STATUS_OK;
	memset(state->head_slots_host, 0, (uint64_t)slots * 4u);
	for ( row = 0u; row < rows; ++row )
	{
		rank_slot = tp_rank * 2u;
		state->head_slots_host[row * 2u * tp_degree + rank_slot] = output_scores[row];
		state->head_slots_host[row * 2u * tp_degree + rank_slot + 1u] = (float)output_token_ids[row];
	}
	/* The collective exchanges HOST buffers (the wire is TCP); the slot
	 * vector lives in host memory end to end. */
	status = SparkTpCollectiveAllReduceSumF32(&state->collective,
		state->head_slots_host, slots, (float *)state->staging_values);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( row = 0u; row < rows; ++row )
	{
		best = 0u;
		for ( r = 1u; r < tp_degree; ++r )
		{
			float bs = state->head_slots_host[row * 2u * tp_degree + best * 2u];
			float rs = state->head_slots_host[row * 2u * tp_degree + r * 2u];
			if ( rs > bs )
				best = r;
		}
		output_token_ids[row] = (uint32_t)state->head_slots_host[row * 2u * tp_degree + best * 2u + 1u];
		output_scores[row] = state->head_slots_host[row * 2u * tp_degree + best * 2u];
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerSubmit(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerDispatch *dispatch)
{
	SparkK3RunnerState *state;
	SparkK3StepInput in;
	K3LayerBuffers *b;
	cudaStream_t stream;
	uint32_t rows, sequences, packed_rows;
	uint32_t dense_offsets[2];
	int32_t status;
	SparkStatus exchange_status;
	SparkModelDriverCompletion completion;
	uint32_t *host_tokens;
	float *host_scores;
	uint32_t i;
	if ( runner == 0 || dispatch == 0 || runner->private_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	rows = dispatch->row_count;
	if ( rows == 0u || rows > state->max_rows ||
		(runner->owns_embedding != 0u && dispatch->token_ids == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	stream = state->stream;
	state->rows = rows;
	b = state->dispatch.buffers;
	sequences = rows;
	packed_rows = rows * K3_TOP_K;
	memset(&in, 0, sizeof(in));
	/* Stage 0 embeds; the others consume the transported hidden stream. */
	if ( runner->owns_embedding != 0u )
	{
		status = K3Embedding(state->embed_weight, dispatch->token_ids,
			b->hidden_bf16, rows, runner->tp_rank * state->vocab_slice_rows,
			state->vocab_slice_rows, stream);
		if ( status != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( K3RunnerReduceBf16(state, stream, b->hidden_bf16, rows) != SPARK_STATUS_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		K3RunnerHookDump(state, runner->tp_rank, 0u, 0u, "embed", rows * K3_HIDDEN);
	}
	else
	{
		if ( dispatch->hidden_input_bf16 == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		cudaMemcpy(b->hidden_bf16, dispatch->hidden_input_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	}
	in.hidden_in = b->hidden_bf16;
	in.positions = dispatch->positions;
	in.context_length = dispatch->context_length;
	in.sequence_of_row = dispatch->sequence_of_row;
	in.sequence_row_begin = 0; /* pure decode: row i is sequence i */
	in.kda_state_index = dispatch->kda_state_index;
	in.route_expert = state->route_expert;
	in.route_packed_row = state->route_packed_row;
	in.route_source_token = state->route_source_token;
	in.route_weight = state->route_weight;
	in.group_row_offset = state->group_row_offset;
	in.group_tile_prefix_w1 = state->group_tile_prefix_w1;
	in.group_tile_prefix_w2 = state->group_tile_prefix_w2;
	in.dense_row_offset = state->dense_row_offset;
	in.dense_tile_prefix = state->dense_tile_prefix;
	in.head_candidate_score = state->head_candidate_score;
	in.head_candidate_token = state->head_candidate_token;
	in.output_token = state->output_token;
	in.output_score = state->output_score;
	(void)dense_offsets;
	/* Device-side dense offsets: no host traffic on the hot path (the
	 * CUDA-graph capture contract; a per-step H2D would sync). The slice
	 * runs through the per-shape graph when capture is eligible. */
	if ( K3RunnerGraphsEligible(state, stream) != 0u )
		status = K3RunnerLaunchSliceGraph(state, &in, rows, sequences,
			packed_rows, stream);
	else
		status = K3RunnerLaunchSliceDirect(state, &in, rows, sequences,
			packed_rows, stream);
	if ( status != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: slice dispatch failed %d\n", status); return SPARK_STATUS_INTERNAL_ERROR; }
	/* The head stage commits the tokens - only when the pack actually
	 * carries the head weight (a PP1 slice pack lacks it, and the
	 * equivalence legs run such packs with stage_count 1). */
	if ( runner->owns_final_head != 0u && state->head_weight != 0 )
	{
		status = K3Head(b, state->head_norm_weight, state->head_weight, 0,
			state->vocab_slice_rows, rows, stream);
		if ( status != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		cudaStreamSynchronize(stream);
		cudaMemcpy(state->output_token_host, state->output_token,
			(uint64_t)rows * 4u, cudaMemcpyDeviceToHost);
		cudaMemcpy(state->output_score_host, state->output_score,
			(uint64_t)rows * 4u, cudaMemcpyDeviceToHost);
		exchange_status = K3RunnerHeadExchange(state, rows, runner->tp_degree,
			runner->tp_rank, state->output_token_host, state->output_score_host);
		if ( exchange_status != SPARK_STATUS_OK )
			return exchange_status;
		if ( dispatch->output_token_ids != 0 )
			cudaMemcpy(dispatch->output_token_ids, state->output_token_host,
				(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
		if ( dispatch->output_scores != 0 )
			cudaMemcpy(dispatch->output_scores, state->output_score_host,
				(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
	}
	else if ( dispatch->hidden_output_bf16 != 0 )
	{
		cudaMemcpy(dispatch->hidden_output_bf16, b->hidden_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	}
	runner->stats.submitted_count++;
	runner->stats.completed_count++;
	if ( dispatch->completion_function != 0 )
	{
		if ( runner->owns_final_head == 0u )
		{
			memset(state->output_token_host, 0, (uint64_t)rows * 4u);
		}
		memset(&completion, 0, sizeof(completion));
		completion.request_id = dispatch->request_id;
		completion.sequence_id = dispatch->sequence_id;
		completion.sequence_position = dispatch->sequence_position;
		completion.accepted_token_count = rows;
		completion.token_count = rows;
		completion.tokens_per_sequence = 1u;
		for ( i = 0u; i < rows && i < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY; ++i )
			completion.token_ids[i] = state->output_token_host[i];
		completion.status = SPARK_STATUS_OK;
		dispatch->completion_function(dispatch->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerGetStats(
	const SparkK3StageRunner *runner,
	SparkK3StageRunnerStats *stats_out)
{
	if ( runner == 0 || stats_out == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	*stats_out = runner->stats;
	return SPARK_STATUS_OK;
}

const void *SparkK3StageRunnerProbeBuffers(const SparkK3StageRunner *runner)
{
	if ( runner == 0 || runner->private_state == 0 )
		return 0;
	return ((SparkK3RunnerState *)runner->private_state)->dispatch.buffers;
}

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner)
{
	SparkK3RunnerState *state;
	if ( runner == 0 || runner->private_state == 0 )
		return;
	state = (SparkK3RunnerState *)runner->private_state;
	if ( state->collective_created != 0 )
		SparkTpCollectiveDestroy(&state->collective);
	if ( state->device_collective_created != 0 )
		SparkTpDeviceCollectiveDestroy(&state->device_collective);
	for ( uint32_t i = 0u; i < 4u; ++i )
		if ( state->graphs[i].live != 0u && state->graphs[i].executable != 0 )
			(void)cudaGraphExecDestroy(state->graphs[i].executable);
	cudaFree(state->fused_device);
	SparkK3DispatchDestroy(&state->dispatch);
	/* The dispatch registered the pack mmap for UVA weight access; a
	 * re-initialise remaps (often the same address) and registering an
	 * already-registered region fails, so the runner unregisters before the
	 * munmap. */
	if ( state->module.pack.mapping != 0 )
		SparkK3DispatchUnregisterPack(&state->module.pack);
	SparkK3ModuleDestroy(&state->module);
	delete[] state->staging_values;
	delete[] state->staging_scratch;
	delete[] state->head_slots_host;
	delete[] state->output_token_host;
	delete[] state->output_score_host;
	cudaFree(state->head_slots_device);
	cudaFree(state->route_expert);
	cudaFree(state->route_packed_row);
	cudaFree(state->route_source_token);
	cudaFree(state->route_weight);
	cudaFree(state->group_row_offset);
	cudaFree(state->group_tile_prefix_w1);
	cudaFree(state->group_tile_prefix_w2);
	cudaFree(state->dense_row_offset);
	cudaFree(state->dense_tile_prefix);
	cudaFree(state->head_candidate_token);
	cudaFree(state->head_candidate_score);
	cudaFree(state->output_token);
	cudaFree(state->output_score);
	delete state;
	runner->private_state = 0;
}
