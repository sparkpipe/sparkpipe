#pragma once

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadShadowQuantize)(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadCertifiedFp8Quantize)(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, float *shadow_scale_f32, float *cert_norm_f32, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadCertifiedFp8Quantize(stream,head_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,candidate_count,hidden_dimension));
}
