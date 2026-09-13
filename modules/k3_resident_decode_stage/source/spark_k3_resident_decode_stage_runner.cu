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
// is slot-encoded on the host tier - each rank fills only its slot, so the
// f32 sum over the rank slots IS the all-gather of the local argmaxes and
// the winner reduces locally - and a packed u64 MAX on the device tier,
// which carries score and vocab-scale token id in ONE exact word (bf16
// cannot carry those ids; see K3RunnerHeadPackKernel).

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

/* THE GATE|UP PAYLOAD SIZE, SPOKEN ONCE (driver-audit D3). The w1 partial
 * is packed_rows x 2*intermediate BF16; every consumer - the phase-2
 * exchange, the fused device stage, both host staging buffers, and the
 * serial-TP replay copies - used to restate that product by hand.
 * Rows-form for per-row callers, packed-form for the route-packed ones. */
static uint64_t K3RunnerGateUpElementsForRows(uint32_t rows)
{
	return((uint64_t)rows * K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u));
}

static uint64_t K3RunnerGateUpBytesForPackedRows(uint32_t packed_rows)
{
	return((uint64_t)packed_rows * (K3_EXPERT_INTERMEDIATE * 2u) * 2ull);
}
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
	uint32_t gate_up_elements;
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
	if ( tp->phase == 2u )
	{
		/* the gate|up segment: write the SUMMED partial back into the scratch
		 * the w1 half left it in, so SiTU runs on the full-width gate|up */
		cudaMemcpyAsync(b->gate_up_bf16, fused,
			(uint64_t)tp->gate_up_elements * 2u,
			cudaMemcpyDeviceToDevice, tp->stream);
		delete tp;
		return;
	}
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

// The head's cross-rank argmax on the DEVICE tier. The wire cannot be bf16
// (a bf16 mantissa carries integers only to 256; the K3 vocab is 2^18-class),
// so each rank packs ONE u64 per row and the ranks exchange a u64 MAX:
//
//   [63:32] score under the IEEE-754 total-order transform - sign-clear
//           floats map to bits|0x80000000, sign-set to ~bits - which is
//           exactly monotonic in float order and involves NO arithmetic, so
//           every rank derives the identical winner bit-exactly;
//   [31:24] the INVERTED rank index: MAX therefore breaks score ties toward
//           the LOWEST rank, reproducing the host slot-sum path's first-max
//           scan rule byte for byte;
//   [23:0]  the token id itself (vocab < 2^24), carried through the reduce
//           so no second exchange is needed.
//
// A NaN score encodes as key 0 and loses to everything - the host scan's
// false-compare behaviour. The unpack inverts the transform with pure bit
// ops, so the emitted score bits are the winner's original bits.
__global__ static void K3RunnerHeadPackKernel(const uint32_t *tokens,
	const float *scores, unsigned long long *keys, uint32_t tp_rank,
	uint32_t rows)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= rows )
		return;
	float s = scores[i];
	if ( s != s )
		{ keys[i] = 0ull; return; }
	uint32_t bits = __float_as_uint(s);
	uint32_t ordered = (bits & 0x80000000u) != 0u ? ~bits :
		(bits | 0x80000000u);
	keys[i] = ((unsigned long long)ordered << 32) |
		((unsigned long long)(uint32_t)(~tp_rank & 0xFFu) << 24) |
		(unsigned long long)(tokens[i] & 0xFFFFFFu);
}

__global__ static void K3RunnerHeadUnpackKernel(
	const unsigned long long *keys, uint32_t *tokens, float *scores,
	uint32_t rows)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= rows )
		return;
	unsigned long long k = keys[i];
	uint32_t bits = (uint32_t)(k >> 32);
	bits = (bits & 0x80000000u) != 0u ? (bits ^ 0x80000000u) : ~bits;
	scores[i] = __uint_as_float(bits);
	tokens[i] = (uint32_t)(k & 0xFFFFFFull);
}

typedef struct SparkK3RunnerHeadContext
{
	unsigned long long *keys;
	uint32_t *tokens;
	float *scores;
	cudaStream_t stream;
	uint32_t rows;
} SparkK3RunnerHeadContext;

// Stream-ordered continuation of the head's u64-max submission: the
// all-reduce is enqueued first, so the unpack launched from this callback
// reads the reduced keys without any host sync on the submission path.
static void K3RunnerHeadCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkK3RunnerHeadContext *head = (SparkK3RunnerHeadContext *)context;
	(void)completion;
	K3RunnerHeadUnpackKernel<<<(head->rows + 255u) / 256u, 256u, 0,
		head->stream>>>(head->keys, head->tokens, head->scores, head->rows);
	delete head;
}

// The per-phase pack: phase 0 stages attention_out, phase 1 stages hidden
// (routed_up / dense_down) and, for routed layers, shared_out after it, phase 2
// stages the w1 gate|up partial (packed_rows x 2*intermediate wide).
__global__ static void K3RunnerFusedPackKernel(const uint16_t *attention,
	const uint16_t *hidden,const uint16_t *shared,const uint16_t *gate_up,
	uint16_t *fused,uint32_t rows,uint32_t phase,uint32_t segments,
	uint32_t gate_up_elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * K3_HIDDEN;
	if ( phase == 2u )
	{
		if ( i >= gate_up_elements )
			return;
		fused[i] = gate_up[i];
		return;
	}
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

// The two-rank BF16 combine (the collective's plain fallback path). It
// CANNOT reuse the TP4 tree kernel: that kernel dereferences all four rank
// lanes unconditionally, and this path has only two - lanes 2/3 were NULL,
// so the first two-rank operation would fault instead of reduce. One
// dedicated kernel reads exactly the lanes that exist.
__global__ static void K3RunnerCombinePairKernel(uint16_t *destination,
	const uint16_t *source,uint32_t elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= elements )
		return;
	destination[i] = LmFloatToBf16(LmBf16ToFloat(destination[i]) +
		LmBf16ToFloat(source[i]));
}

static SparkStatus K3RunnerCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	(void)combine_context;
	K3RunnerCombinePairKernel<<<(active_sequence_count * hidden_dimension + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		(uint16_t *)destination_device,(const uint16_t *)source_device,
		active_sequence_count * hidden_dimension);
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
	/* head candidate slot exchange: rows x (2 * tp_degree) floats. The
	 * slots live on the HOST end to end (the host tier's wire is TCP); the
	 * device tier exchanges the packed u64 keys instead, so no device slot
	 * mirror exists - the dead buffer that used to sit here is gone. */
	float *head_slots_host;
	uint32_t head_slots_capacity;
	/* the device tier's head exchange: one packed u64 argmax key per row */
	unsigned long long *head_keys_device;
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
	/* per-token device tensors for the serial-TP half step (single-token decode) */
	uint32_t *positions;        /* rows */
	uint32_t *context_length;   /* rows */
	uint32_t *sequence_of_row;  /* rows */
	uint32_t *kda_state_index;  /* sequences */
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

/* THE PP4 SLICE TABLE LIVES IN spark_k3_pool_sizing.H - one spelling
 * shared with the module's pack validation and the geometry macros
 * (driver-audit D2); these wrappers keep the historical call sites. */
static uint32_t K3RunnerFirstLayer(uint32_t stage_index)
{
	return(SparkK3StageFirstLayer(stage_index % 4u));
}

static uint32_t K3RunnerLayerCount(uint32_t stage_index)
{
	return(SparkK3StageLayerCount(stage_index % 4u));
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
	/* No tier reached: legitimate ONLY at tp_degree 1, where there is
	 * nothing to reduce. Sharded inits guarantee a usable tier - the init
	 * gate admits device-only configurations solely with the NCCL backend,
	 * which the branch above handled - so this fall-through can never mask
	 * a missing exchange on a sharded run. */
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
	/* THE w1 GATE|UP ALL-REDUCE. The input-split w1 emits a FULL-width gate|up
	 * partial (packed_rows x 2*intermediate), and SiTU is non-linear, so the
	 * partial must be summed BEFORE SiTU. This is the point the replay harness
	 * folds host-side; the serving tier does it here (NCCL honours reserved0;
	 * the hidden-transport tier keeps its pre-registered 7168 frame). */
	if ( phase == 2u )
	{
		const uint32_t gate_up_elements =
			(uint32_t)K3RunnerGateUpElementsForRows(rows);
		if ( state->device_collective_created != 0 )
		{
			K3RunnerFusedPackKernel<<<(gate_up_elements + 255u) / 256u,
				256u, 0, stream>>>(
				0, 0, 0, b->gate_up_bf16, state->fused_device,
				rows, 2u, 1u, gate_up_elements);
			SparkK3RunnerTpContext *completion_context = new SparkK3RunnerTpContext;
			completion_context->fused = state->fused_device;
			completion_context->buffers = b;
			completion_context->stream = stream;
			completion_context->rows = rows;
			completion_context->boundary = 0u;
			completion_context->segments = 1u;
			completion_context->phase = 2u;
			completion_context->gate_up_elements = gate_up_elements;
			SparkTpDeviceCollectiveSubmission submission;
			memset(&submission, 0, sizeof(submission));
			submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = 0u;
			submission.active_sequence_count = rows;
			submission.flags =
				SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
			submission.ordinal = state->tp_next_ordinal++;
			submission.reserved0 = gate_up_elements;
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
			cudaStreamSynchronize(stream);
			cudaMemcpy(state->staging_values, b->gate_up_bf16,
				(uint64_t)gate_up_elements * 2u, cudaMemcpyDeviceToHost);
			SparkTpCollectiveAllReduceSumBf16(&state->collective,
				state->staging_values, gate_up_elements, state->staging_scratch);
			cudaMemcpy(b->gate_up_bf16, state->staging_values,
				(uint64_t)gate_up_elements * 2u, cudaMemcpyHostToDevice);
		}
		return;
	}
	if ( state->device_collective_created != 0 )
	{
		/* THE DEVICE TIER: one pack kernel + ONE stream-ordered combine
		 * submission per phase; the completion folds the summed segment(s)
		 * into the partial on the same stream. No sync, no host staging. */
		K3RunnerFusedPackKernel<<<(elements + 255u) / 256u, 256u, 0, stream>>>(
			phase0_source,b->hidden_bf16,b->shared_out_bf16,b->gate_up_bf16,
			state->fused_device,rows,phase,segments,0u);
		SparkK3RunnerTpContext *completion_context = new SparkK3RunnerTpContext;
		completion_context->fused = state->fused_device;
		completion_context->buffers = b;
		completion_context->stream = stream;
		completion_context->rows = rows;
		completion_context->boundary = boundary;
		completion_context->segments = segments;
		completion_context->phase = phase;
		completion_context->gate_up_elements = 0u;
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

/* F4 teardown contract: private_state is armed before the first allocating
 * step, so EVERY init failure funnels through the ONE Destroy - safe on a
 * partially built state because each free is NULL-tolerant or flag-guarded
 * (and the module close is fd-hardened against a second pass) - so no error
 * path can leak a device buffer or hand Submit an unchecked null pointer. */
#define K3_INIT_FAIL(status_value) \
	do { SparkK3StageRunnerDestroy(runner); return (status_value); } while ( 0 )

#define K3_CUDA_ALLOC_OR_FAIL(device_pointer, byte_count) \
	do { if ( cudaMalloc((void **)&(device_pointer), \
		(uint64_t)(byte_count)) != cudaSuccess ) \
		K3_INIT_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); } while ( 0 )

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
		K3_INIT_FAIL(status);
	if ( SparkK3DispatchCreate(&state->dispatch,&state->module.sizing,
		configuration->max_active_sequence_count,
		configuration->max_input_row_count,
		configuration->kv_pages_per_sequence,
		state->kv_page_bytes, 0) != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: dispatch create failed\n"); K3_INIT_FAIL(SPARK_STATUS_INTERNAL_ERROR); }
	/* The pack mmap registers (in chunks) for UVA weight access: the tensor
	 * maps encode the registered addresses, so a failed registration is
	 * fatal - the unregistered path cannot launch the GEMMs. */
	if ( SparkK3DispatchRegisterPack(&state->module.pack) != SPARK_K3_DISPATCH_OK )
		K3_INIT_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( SparkK3DispatchBindWeights(&state->dispatch,&state->module.pack,
			state->module.bound,state->module.bound_count) != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: weight bind failed\n"); K3_INIT_FAIL(SPARK_STATUS_INTERNAL_ERROR); }
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
		/* F1: the host TCP tier caps at four ranks, so a TP16 deployment
		 * supplies ONLY the device collective - the host tier is optional
		 * whenever the device tier carries the exchanges. The one shape
		 * still rejected is sharded decode with NO tier at all. Only the
		 * NCCL device backend counts as self-sufficient here: it alone
		 * implements every exchange the runner makes (bf16 hidden sums,
		 * and the head's u64-max argmax), while the hidden-transport tier
		 * keeps its pre-registered frame and has no u64 combine, so a
		 * hidden-only init would silently no-op the embedding reduce. */
		int device_carries_tp =
			configuration->device_collective != 0 &&
			configuration->device_collective->backend_kind ==
				SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
		if ( configuration->tp_collective == 0 && device_carries_tp == 0 )
			K3_INIT_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( configuration->tp_collective != 0 )
		{
		fprintf(stderr, "sparkpipe_k3: creating host collective tp=%u rank=%u port=%u\n",
			configuration->tp_degree, configuration->tp_rank,
			configuration->tp_collective->listen_port);
		status = SparkTpCollectiveCreate(configuration->tp_collective,&state->collective);
		fprintf(stderr, "sparkpipe_k3: host collective create -> %d\n", (int)status);
		if ( status != SPARK_STATUS_OK )
			K3_INIT_FAIL(status);
		state->collective_created = 1;
		}
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
		/* The fused stage must hold the widest per-phase payload: the w1
		 * gate|up is packed_rows x 2*intermediate (24576 u16 at B1), wider
		 * than the 3*hidden frame the phase 0/1 hooks used. */
		K3_CUDA_ALLOC_OR_FAIL(state->fused_device,
			K3RunnerGateUpElementsForRows(state->fused_rows) *
			sizeof(uint16_t));
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
			K3_INIT_FAIL(status); /* also frees fused_device - the old path leaked it */
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
	/* Host staging + head slots + the per-step device arrays. The stage must
	 * hold the widest per-phase payload - the w1 gate|up (packed_rows x
	 * 2*intermediate = 24576 u16 at B1) is wider than the 3*hidden frame. */
	state->staging_capacity =
		(uint32_t)K3RunnerGateUpElementsForRows(
			configuration->max_input_row_count);
	state->staging_values = new uint16_t[state->staging_capacity];
	state->staging_scratch = new uint16_t[state->staging_capacity];
	state->fused_capacity = state->staging_capacity;
	state->head_slots_capacity = configuration->max_input_row_count * 2u * configuration->tp_degree;
	state->head_slots_host = new float[state->head_slots_capacity];
	K3_CUDA_ALLOC_OR_FAIL(state->head_keys_device,
		(uint64_t)configuration->max_input_row_count *
			sizeof(*state->head_keys_device));
	routes = configuration->max_input_row_count * K3_TOP_K;
	K3_CUDA_ALLOC_OR_FAIL(state->route_expert,(uint64_t)routes * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->route_packed_row,(uint64_t)routes * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->route_source_token,(uint64_t)routes * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->route_weight,(uint64_t)routes * sizeof(float));
	K3_CUDA_ALLOC_OR_FAIL(state->group_row_offset,(uint64_t)(K3_EXPERTS + 1u) * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->group_tile_prefix_w1,(uint64_t)(K3_EXPERTS + 1u) * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->group_tile_prefix_w2,(uint64_t)(K3_EXPERTS + 1u) * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->dense_row_offset, 8u);
	K3_CUDA_ALLOC_OR_FAIL(state->dense_tile_prefix, 8u);
	state->head_tiles = (state->vocab_slice_rows + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	K3_CUDA_ALLOC_OR_FAIL(state->head_candidate_token,
		(uint64_t)configuration->max_input_row_count * state->head_tiles *
			sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->head_candidate_score,
		(uint64_t)configuration->max_input_row_count * state->head_tiles *
			sizeof(float));
	K3_CUDA_ALLOC_OR_FAIL(state->output_token,
		(uint64_t)configuration->max_input_row_count * sizeof(uint32_t));
	K3_CUDA_ALLOC_OR_FAIL(state->output_score,
		(uint64_t)configuration->max_input_row_count * sizeof(float));
	/* per-token tensors for the serial-TP half step (single token, position 0,
	 * context length 1, sequence 0, state slot 0). */
	K3_CUDA_ALLOC_OR_FAIL(state->positions, 4u);
	K3_CUDA_ALLOC_OR_FAIL(state->context_length, 4u);
	K3_CUDA_ALLOC_OR_FAIL(state->sequence_of_row, 4u);
	K3_CUDA_ALLOC_OR_FAIL(state->kda_state_index, 4u);
	{
		uint32_t pos = 0u, ctx = 1u, seq = 0u, st = 0u;
		cudaMemcpy(state->positions, &pos, 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(state->context_length, &ctx, 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(state->sequence_of_row, &seq, 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(state->kda_state_index, &st, 4u, cudaMemcpyHostToDevice);
	}
	state->output_token_host = new uint32_t[configuration->max_input_row_count];
	state->output_score_host = new float[configuration->max_input_row_count];
	return SPARK_STATUS_OK;
}

// The head's cross-rank argmax. Two tiers, identical outcomes:
//   device tier (NCCL): pack one u64 argmax key per row, ONE stream-ordered
//     all-reduce-max, the completion unpacks the winner on the same stream;
//     a final sync materializes the winners into the host mirrors. The wire
//     is u64-exact by construction - bf16 cannot carry vocab-scale ids and
//     an f32 sum would need a second exchange to move them.
//   host tier: the slot-encoded SUM - each rank fills only its slot pair, so
//     the sum over rank slots IS the all-gather of the local argmaxes, then
//     the first-max scan (strict > keeps the LOWEST rank on ties).
// The device key layout reproduces that tie rule via the inverted-rank field,
// so both tiers emit the same token for the same head outputs.
static SparkStatus K3RunnerHeadExchange(SparkK3RunnerState *state,
	uint32_t rows, uint32_t tp_degree, uint32_t tp_rank, cudaStream_t stream,
	uint32_t *device_tokens, float *device_scores,
	uint32_t *host_tokens, float *host_scores)
{
	uint32_t slots = rows * 2u * tp_degree;
	SparkStatus status;
	cudaError_t error;
	uint32_t row, rank_slot, best, r;
	if ( tp_degree <= 1u )
	{
		error = cudaStreamSynchronize(stream);
		if ( error != cudaSuccess )
			return SPARK_STATUS_INTERNAL_ERROR;
		cudaMemcpy(host_tokens, device_tokens, (uint64_t)rows * 4u,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(host_scores, device_scores, (uint64_t)rows * 4u,
			cudaMemcpyDeviceToHost);
		return SPARK_STATUS_OK;
	}
	if ( state->device_collective_created != 0 &&
		state->device_collective.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		SparkK3RunnerHeadContext *context = new SparkK3RunnerHeadContext;
		context->keys = state->head_keys_device;
		context->tokens = device_tokens;
		context->scores = device_scores;
		context->stream = stream;
		context->rows = rows;
		K3RunnerHeadPackKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(
			device_tokens, device_scores, state->head_keys_device, tp_rank,
			rows);
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission, 0, sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.active_sequence_count = rows; /* the op's element count */
		submission.ordinal = state->tp_next_ordinal++;
		submission.local_device = state->head_keys_device;
		submission.full_device = state->head_keys_device;
		submission.cuda_stream = stream;
		submission.completion_function = K3RunnerHeadCompletion;
		submission.completion_context = context;
		status = SparkTpDeviceCollectiveSubmitU64Max(&state->device_collective,
			&submission);
		if ( status != SPARK_STATUS_OK )
			{ delete context; return status; }
		error = cudaStreamSynchronize(stream);
		if ( error != cudaSuccess )
			return SPARK_STATUS_INTERNAL_ERROR;
		cudaMemcpy(host_tokens, device_tokens, (uint64_t)rows * 4u,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(host_scores, device_scores, (uint64_t)rows * 4u,
			cudaMemcpyDeviceToHost);
		return SPARK_STATUS_OK;
	}
	if ( state->collective_created == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	cudaMemcpy(host_tokens, device_tokens, (uint64_t)rows * 4u,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(host_scores, device_scores, (uint64_t)rows * 4u,
		cudaMemcpyDeviceToHost);
	memset(state->head_slots_host, 0, (uint64_t)slots * 4u);
	for ( row = 0u; row < rows; ++row )
	{
		rank_slot = tp_rank * 2u;
		state->head_slots_host[row * 2u * tp_degree + rank_slot] =
			host_scores[row];
		state->head_slots_host[row * 2u * tp_degree + rank_slot + 1u] =
			(float)host_tokens[row];
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
		host_tokens[row] =
			(uint32_t)state->head_slots_host[row * 2u * tp_degree + best * 2u + 1u];
		host_scores[row] =
			state->head_slots_host[row * 2u * tp_degree + best * 2u];
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
		/* The exchange syncs the stream and leaves the winning tokens in the
		 * host mirrors (device tier: packed u64-max; host tier: f32 slot
		 * sum) - no raw copy precedes it. */
		exchange_status = K3RunnerHeadExchange(state, rows, runner->tp_degree,
			runner->tp_rank, stream, state->output_token,
			state->output_score, state->output_token_host,
			state->output_score_host);
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
		uint32_t copied = 0u;
		memset(&completion, 0, sizeof(completion));
		completion.request_id = dispatch->request_id;
		completion.sequence_id = dispatch->sequence_id;
		completion.sequence_position = dispatch->sequence_position;
		completion.tokens_per_sequence = 1u;
		/* F7: the counts report what the fixed token_ids array actually
		 * CARRIES - rows can exceed the completion capacity at wide batch
		 * configs, and an accepted count above the carried count tells the
		 * orchestrator tokens exist that it can never read. */
		for ( i = 0u; i < rows && i < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY; ++i )
		{
			completion.token_ids[i] = state->output_token_host[i];
			copied++;
		}
		completion.accepted_token_count = copied;
		completion.token_count = copied;
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
	cudaFree(state->head_keys_device);
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
	cudaFree(state->positions);
	cudaFree(state->context_length);
	cudaFree(state->sequence_of_row);
	cudaFree(state->kda_state_index);
	delete state;
	runner->private_state = 0;
}

/* bind.cu's serial-TP half-step ABI (docs/serial_tp_replay.md). */
extern "C" int32_t K3StageSliceHalf(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t layer, uint32_t phase, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream);

SparkStatus SparkK3StageRunnerStepHalf(SparkK3StageRunner *runner, uint32_t layer, uint32_t phase,
	const void *hidden_input_bf16, const void *partial_input_bf16, void *partial_output_bf16)
{
	SparkK3RunnerState *state;
	SparkK3Dispatch *d;
	K3LayerBuffers *b;
	cudaStream_t stream;
	uint32_t rows, sequences, packed_rows;
	int32_t status;
	if ( runner == 0 || runner->private_state == 0 || partial_output_bf16 == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	d = &state->dispatch;
	if ( layer < d->first_layer || layer >= d->first_layer + d->layer_count )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( phase == 0u && hidden_input_bf16 == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	b = d->buffers;
	stream = state->stream;
	rows = 1u;            /* the replay is single-token decode (B1) */
	sequences = 1u;
	packed_rows = rows * K3_TOP_K;
	if ( phase == 0u )
		cudaMemcpy(b->hidden_bf16, hidden_input_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	if ( phase == 2u )
	{
		/* MoE rest: feed the all-reduced gate|up back into the scratch the w1
		 * half left it in, so SiTU runs on the summed partial. */
		cudaMemcpy(b->gate_up_bf16, partial_input_bf16,
			K3RunnerGateUpBytesForPackedRows(packed_rows),
			cudaMemcpyDeviceToDevice);
	}
	else if ( partial_input_bf16 != 0 )
		cudaMemcpy(b->attnres_partial_bf16, partial_input_bf16,
			(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	/* Fill the per-step device tensors the normal dispatch step would set
	 * (the half step bypasses SparkK3DispatchStep). The routing arrays are
	 * consumed by the MoE path; the dense/group prefixes by every GEMM. */
	b->dense_row_offset = state->dense_row_offset;
	/* The non-grouped GEMMs derive their row extent from
	 * dense_row_offset[1]-[0] (the dispatch step's launch wrote it; the half
	 * step bypasses that launch), so write [0, rows] here or every dense
	 * projection sees zero rows and the AttnRes fold never happens. */
	K3RunnerDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, rows);
	b->dense_tile_prefix = state->dense_tile_prefix;
	b->group_row_offset = state->group_row_offset;
	b->group_tile_prefix_w1 = state->group_tile_prefix_w1;
	b->group_tile_prefix_w2 = state->group_tile_prefix_w2;
	b->route_expert = state->route_expert;
	b->route_packed_row = state->route_packed_row;
	b->route_source_token = state->route_source_token;
	b->route_weight = state->route_weight;
	b->sequence_row_begin = 0; /* identity: row i is sequence i */
	b->positions = state->positions;
	b->context_length = state->context_length;
	b->sequence_of_row = state->sequence_of_row;
	b->kda_state_index = state->kda_state_index;
	status = K3StageSliceHalf(d->weights + (layer - d->first_layer), d->slice_state, d->buffers,
		layer, phase, rows, sequences, 1u, packed_rows, rows, state->multiprocessors, stream);
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: half step layer %u phase %u -> %d\n",
			layer, phase, status);
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	/* Capture the rank's CONTRIBUTION (the un-folded input-sharded projection
	 * output), NOT the folded partial. Phase 1 on a MoE layer is the w1 half
	 * and captures the gate|up partial (packed_rows x 2*inter); every other
	 * phase captures hidden_bf16 (kda_out/dense_down/routed_up), with
	 * attention_out_bf16 for the MLA phase 0 and shared_out_bf16 added on the
	 * MoE rest (phase 2). The harness sums these across ranks and folds the
	 * sum; the replicated partial is only the retrieval's read. */
	if ( phase == 1u && layer >= K3_FIRST_ROUTED_LAYER )
	{
		cudaMemcpy(partial_output_bf16, b->gate_up_bf16,
			K3RunnerGateUpBytesForPackedRows(packed_rows),
			cudaMemcpyDeviceToDevice);
		return SPARK_STATUS_OK;
	}
	const uint16_t *phase0_source = (phase == 0u &&
		K3_LAYER_KIND(layer) == LM_LAYER_LATENT)
		? b->attention_out_bf16 : b->hidden_bf16;
	cudaMemcpy(partial_output_bf16, phase0_source,
		(uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice);
	if ( phase == 2u )
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
			dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
			K3_LAYER_THREADS, 0, stream,
			(uint16_t *)partial_output_bf16,b->shared_out_bf16,
			(uint16_t *)partial_output_bf16,rows,K3_HIDDEN);
	return SPARK_STATUS_OK;
}
