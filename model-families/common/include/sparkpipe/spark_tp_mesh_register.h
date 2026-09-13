#pragma once
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"

extern int SparkGlm5NextLaunchSumRanksF32(void *stream,void *destination,const void *const *sources,uint32_t source_count,uint32_t element_count);
extern int SparkGlm5NextLaunchSeedF32(void *stream,float *destination,const void *a,const void *b,uint32_t element_count);
extern int SparkGlm5NextLaunchAddF32(void *stream,float *destination,const void *b,uint32_t element_count);
extern int SparkGlm5NextLaunchRoundF32(void *stream,void *destination,const float *source,uint32_t element_count);
extern int SparkGlm5NextLaunchAccumAdd(void *stream,void *destination,const void *source,uint32_t row_count,uint32_t width);
extern int SparkGlm5NextLaunchAccumU64Max(void *stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);

static SparkStatus SparkTpMeshCombineFusedBf16(
	void *combine_context,
	void *destination_device,
	const void *const *source_devices,
	uint32_t source_count,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchSumRanksF32((void *)cuda_stream,
	    destination_device,source_devices,source_count,
	    (uint32_t)((uint64_t)active_sequence_count * hidden_dimension));
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpMeshCombineF32Seed(
	void *combine_context,
	void *destination_f32_device,
	const void *source_a_bf16_device,
	const void *source_b_bf16_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchSeedF32((void *)cuda_stream,
	    (float *)destination_f32_device,source_a_bf16_device,
	    source_b_bf16_device,element_count);
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpMeshCombineF32Add(
	void *combine_context,
	void *destination_f32_device,
	const void *source_bf16_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAddF32((void *)cuda_stream,
	    (float *)destination_f32_device,source_bf16_device,
	    element_count);
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpMeshRoundF32(
	void *combine_context,
	void *destination_bf16_device,
	const void *source_f32_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchRoundF32((void *)cuda_stream,
	    destination_bf16_device,(const float *)source_f32_device,
	    element_count);
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpMeshCombineBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAccumAdd((void *)cuda_stream,
	    destination_device,source_device,active_sequence_count,
	    hidden_dimension);
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpMeshCombineU64Max(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAccumU64Max((void *)cuda_stream,
	    destination_device,source_device,element_count);
	return(error == cudaSuccess ?
	    SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static inline void SparkTpMeshRegisterCommonCombines(
    SparkTpDeviceCollectiveConfig *configuration)
{
	configuration->combine_fused_bf16_function = SparkTpMeshCombineFusedBf16;
	configuration->combine_f32_seed_function = SparkTpMeshCombineF32Seed;
	configuration->combine_f32_add_function = SparkTpMeshCombineF32Add;
	configuration->round_f32_function = SparkTpMeshRoundF32;
	configuration->combine_bf16_function = SparkTpMeshCombineBf16;
	configuration->combine_u64_max_function = SparkTpMeshCombineU64Max;
}
