// Bind weights to the layer and run a rank's slice of Kimi K3.
//
// This file is now the format choice and the C ABI, nothing else. The slice
// loop, the weight table and the per-layer state binding live in slice.cuh so
// a host harness can execute them with a recorder format - this translation
// unit includes the 4-bit format headers, whose inline PTX assembles nowhere on
// a CPU, and that include is exactly what kept the loop out of every gate.
//
// Weights arrive as an explicit per-layer table for the same reason as the
// other drivers: no K3 pack format exists, and inventing one from the layer's
// requirements is the mistake glm5_2/bind.cu records its author stopping to
// avoid. What a packer produces is its business; this file needs the pointers.

#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/kimi_k3/slice.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	// THE FORMAT FOLLOWS THE CHECKPOINT'S RECIPE, NOT A GLOBAL CHOICE.
	//
	// K3's quantization_config quantises the routed experts to MXFP4 group 32
	// and its ignore list excludes attention, latent projections, shared
	// experts, routers and lm_head - and the report says the quantisation-aware
	// training ran from SFT onward, so the routed experts were trained INTO
	// that grid and nothing else was. Requantising attention to INT7 is
	// off-recipe in the same way that storing derived factors at MXFP4 would
	// be. The grid is not the protection; the training into the grid is.
	return(K3LaunchSlice<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,sequences,commit,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}

// The serial-TP half step (docs/serial_tp_replay.md): one layer's attention
// half (phase 0) or MLP half (phase 1). The harness replicates the FULL hidden
// and the FULL AttnRes partial before each half, runs every rank in turn, and
// host-sums the rank partials between halves.
extern "C" int32_t K3StageSliceHalf(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t layer, uint32_t phase, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(K3LaunchSliceHalf<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		layer,phase,rows,sequences,commit,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}

// Acceptance is a fold, not a rewind: replay the accepted rows from the
// verify slabs with the same kernels a committed run uses.
extern "C" int32_t K3StageFold(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, void *stream)
{
	return(K3FoldAccepted<LmMxfp4>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,sequences,verify_row_begin,accepted,slab_rows,
		multiprocessors,(cudaStream_t)stream));
}
