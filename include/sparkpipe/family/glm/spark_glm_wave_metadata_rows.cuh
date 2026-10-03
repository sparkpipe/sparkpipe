#pragma once

__global__ static void SPARK_FAMILY(WaveMetadataKernel)(
	const uint32_t *resident_slots,
	const uint32_t *positions,
	uint32_t *context_lengths,
	uint32_t *dense_row_offset,
	uint32_t row_count)
{
	uint32_t row;
	if ( blockIdx.x != 0u )
		return;
	for ( row = threadIdx.x; row < row_count; row += blockDim.x )
		context_lengths[resident_slots[row]] = 0u;
	__syncthreads();
	for ( row = threadIdx.x; row < row_count; row += blockDim.x )
		atomicMax(&context_lengths[resident_slots[row]],positions[row] + 1u);
	if ( threadIdx.x == 0u )
	{
		dense_row_offset[0] = 0u;
		dense_row_offset[1] = row_count;
	}
}

static int32_t SPARK_FAMILY(ValidateWaveShape)(const SPARK_FAMILY(CudaWave) *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || wave->row_count > wave->execution_row_capacity || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}
