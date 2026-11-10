
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/kernels/spec_verify.cuh"
#include "model-families/k3/source/slice.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(K3LaunchSlice<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,sequences,commit,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}

extern "C" int32_t K3StageSlicePair(const void *layer_weights, const void *slice_state, void *buffers_a, void *buffers_b, uint32_t first_layer, uint32_t layer_count, uint32_t rows_a, uint32_t rows_b, uint32_t commit, uint32_t multiprocessors, void *stream)
{
	return(K3LaunchSlicePair<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)buffers_a,
		(K3LayerBuffers *)buffers_b,
		first_layer,layer_count,rows_a,rows_b,commit,multiprocessors,
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

extern "C" int32_t K3StageFold(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *verify_row_indices, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, void *stream)
{
	return(K3FoldAccepted<LmMxfp4>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,sequences,verify_row_begin,verify_row_indices,accepted,slab_rows,
		multiprocessors,(cudaStream_t)stream));
}

extern "C" int32_t K3SpecVerifyAccept(const uint32_t *row_tokens, const uint32_t *sequence_row_begin, const uint32_t *draft_tokens, const uint32_t *draft_counts, uint32_t draft_stride, uint32_t sequences, uint32_t *accepted, uint32_t *bonus, void *stream)
{
	if ( row_tokens == 0 || sequence_row_begin == 0 || draft_tokens == 0
		|| draft_counts == 0 || accepted == 0 || bonus == 0
		|| sequences == 0u || draft_stride == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	LmSpecVerifyKernel<32u,7u><<<sequences,32u,0,(cudaStream_t)stream>>>(
		row_tokens,sequence_row_begin,draft_tokens,draft_counts,draft_stride,
		accepted,bonus);
	return(cudaGetLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

