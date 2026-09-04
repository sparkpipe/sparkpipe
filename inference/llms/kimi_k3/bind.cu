
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/kimi_k3/slice.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(K3LaunchSlice<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,sequences,commit,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}

extern "C" int32_t K3StageSliceHalf(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t layer, uint32_t phase, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(K3LaunchSliceHalf<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		layer,phase,rows,sequences,commit,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}

extern "C" int32_t K3StageFold(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, void *stream)
{
	return(K3FoldAccepted<LmMxfp4>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,sequences,verify_row_begin,accepted,slab_rows,
		multiprocessors,(cudaStream_t)stream));
}
