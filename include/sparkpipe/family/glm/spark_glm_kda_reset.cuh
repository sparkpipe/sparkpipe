#pragma once

__global__ static void SPARK_FAMILY(KdaResetKernel)(
	uint8_t *state_pools, uint64_t state_layer_stride, uint64_t state_slot_bytes,
	uint8_t *q_windows, uint8_t *k_windows, uint8_t *v_windows,
	uint64_t window_layer_stride, uint64_t qk_window_slot_bytes, uint64_t v_window_slot_bytes,
	const uint32_t *state_index, const uint32_t *positions, uint32_t layer_count, uint32_t rows)
{
	uint32_t layer = blockIdx.x, row = blockIdx.y, i;
	uint64_t slot;
	uint8_t *base;
	if ( row >= rows || layer >= layer_count || positions[row] != 0u )
		return;
	slot = state_index[row];
	base = state_pools + (uint64_t)layer * state_layer_stride + slot * state_slot_bytes;
	for ( i = threadIdx.x; i < state_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = q_windows + (uint64_t)layer * window_layer_stride + slot * qk_window_slot_bytes;
	for ( i = threadIdx.x; i < qk_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = k_windows + (uint64_t)layer * window_layer_stride + slot * qk_window_slot_bytes;
	for ( i = threadIdx.x; i < qk_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
	base = v_windows + (uint64_t)layer * window_layer_stride + slot * v_window_slot_bytes;
	for ( i = threadIdx.x; i < v_window_slot_bytes; i += blockDim.x )
		base[i] = 0u;
}
