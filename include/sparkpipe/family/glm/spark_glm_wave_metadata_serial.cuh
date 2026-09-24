#pragma once

__global__ static void SPARK_FAMILY(WaveMetadataKernel)(
	const uint32_t *resident_slots,
	const uint32_t *positions,
	uint32_t *context_lengths,
	uint32_t *dense_row_offset,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
	{
		for ( row = 0u; row < row_count; ++row )
		{
			uint32_t slot = resident_slots[row];
			uint32_t len = positions[row] + 1u;
			if ( len > context_lengths[slot] )
				context_lengths[slot] = len;
		}
		dense_row_offset[0] = 0u;
		dense_row_offset[1] = row_count;
	}
}

static int32_t SPARK_FAMILY(RunLayerMlpExperts)(const SPARK_FAMILY(CudaWave) *wave,uint32_t local_layer)
{
	SPARK_FAMILY_BARE(LayerBuffers) buffers;
	if ( (wave->first_layer_index + local_layer) < SPARK_FAMILY_BARE_CONST(FIRST_ROUTED_LAYER) )
		return(LM_LAUNCH_OK);
	SPARK_FAMILY(BindLayer)(wave,local_layer,&buffers);
	return(SPARK_FAMILY_BARE(LayerMoeExperts)<SPARK_FAMILY_BARE_CONST(EXPERT_WEIGHT_CODEC)>(&buffers,wave->row_count,wave->row_count * SPARK_FAMILY_BARE_CONST(TOP_K),wave->multiprocessor_count,(cudaStream_t)wave->slot->stream));
}

extern "C" cudaError_t SPARK_FAMILY(PollCudaLayerMlpRoute)(const SPARK_FAMILY(CudaWave) *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->route_ready_event == 0 || wave->slot->route_recorded == 0u )
		return(cudaErrorInvalidValue);
	return(cudaEventQuery((cudaEvent_t)wave->slot->route_ready_event));
}
