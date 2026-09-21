#include <stdint.h>

#include <cuda_runtime.h>

#include "modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_internal.h"

#ifndef LM_LAUNCH_OK
#define LM_LAUNCH_OK 0
#endif

int32_t SparkLagunaLaunchCudaWave(const SparkLagunaCudaWave *wave)
{
	(void)wave;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaWaveBegin(const SparkLagunaCudaWave *wave)
{
	(void)wave;
	return(LM_LAUNCH_OK);
}

SparkStatus SparkLagunaLaunchOpWait(cudaStream_t stream,void *flag_device,uint64_t wait_value)
{
	(void)stream;
	(void)flag_device;
	(void)wait_value;
	return(SPARK_STATUS_OK);
}

int32_t SparkLagunaLaunchCudaLayerAttention(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaLayerMlp(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaLayerMlpRoute(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

cudaError_t SparkLagunaPollCudaLayerMlpRoute(const SparkLagunaCudaWave *wave)
{
	(void)wave;
	return(cudaSuccess);
}

int32_t SparkLagunaLaunchCudaLayerMlpExperts(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaLayerAttentionPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaLayerMlpPost(const SparkLagunaCudaWave *wave,uint32_t local_layer)
{
	(void)wave;
	(void)local_layer;
	return(LM_LAUNCH_OK);
}

int32_t SparkLagunaLaunchCudaWaveHead(const SparkLagunaCudaWave *wave)
{
	(void)wave;
	return(LM_LAUNCH_OK);
}

cudaError_t SparkLagunaLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	(void)stream;
	(void)scores;
	(void)token_ids;
	(void)maxloc;
	(void)row_count;
	(void)rank_offset;
	return(cudaSuccess);
}

cudaError_t SparkLagunaLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	(void)stream;
	(void)maxloc;
	(void)token_ids;
	(void)row_count;
	return(cudaSuccess);
}

cudaError_t SparkLagunaLaunchDirectSum(cudaStream_t stream,void *destination,const void *const *rank_devices,uint32_t local_rank,uint32_t rows,uint32_t width)
{
	(void)stream;
	(void)destination;
	(void)rank_devices;
	(void)local_rank;
	(void)rows;
	(void)width;
	return(cudaSuccess);
}

int32_t SparkLagunaConfigureCudaModule(uint32_t *multiprocessor_count)
{
	if ( multiprocessor_count == 0 )
		return(-1);
	*multiprocessor_count = 1u;
	return(0);
}

SparkStatus SparkLagunaStageYarnTableUpload(float *device_inv_freq,void *stream)
{
	(void)device_inv_freq;
	(void)stream;
	return(SPARK_STATUS_OK);
}

cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;
	(void)destination_bf16;
	(void)source_bf16;
	(void)row_count;
	(void)width;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)source;
	(void)element_count;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchAddF32(cudaStream_t stream,float *destination,const float *source,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)source;
	(void)element_count;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchRoundF32(cudaStream_t stream,const float *source,float *destination,uint32_t element_count)
{
	(void)stream;
	(void)source;
	(void)destination;
	(void)element_count;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchSeedF32(cudaStream_t stream,float *destination,float value,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)value;
	(void)element_count;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchSumRanksF32(cudaStream_t stream,const float *const *rank_sources,float *destination,uint32_t rank_count,uint32_t local_rank,uint32_t element_count)
{
	(void)stream;
	(void)rank_sources;
	(void)destination;
	(void)rank_count;
	(void)local_rank;
	(void)element_count;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchMeshGuard(cudaStream_t stream,uint32_t *flags,uint32_t flag_count,uint32_t expected_generation)
{
	(void)stream;
	(void)flags;
	(void)flag_count;
	(void)expected_generation;
	return(cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchMeshCopyDown(cudaStream_t stream,const void *source,void *destination,uint32_t row_count,uint32_t width)
{
	(void)stream;
	(void)source;
	(void)destination;
	(void)row_count;
	(void)width;
	return(cudaSuccess);
}
