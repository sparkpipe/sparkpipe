#ifndef SPARK_GLM52_SOTA_DECODE_COMMON_CUH
#define SPARK_GLM52_SOTA_DECODE_COMMON_CUH

#include "spark_glm52_sota_cuda_common.cuh"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t SparkGlm52SotaRmsNormSm121(
    const SparkGlm52SotaNormArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaResidualAddRmsNormSm121(
    const SparkGlm52SotaNormArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaRunLinearPlanSm121(
    const SparkGlm52SotaLinearPlan *plan,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaFusedRopeKvWriteSm121(
    const SparkGlm52SotaRopeKvArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaDenseMlpEpilogueSm121(
    const SparkGlm52SotaDenseMlpArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaRestrictedLogitsSamplerSm121(
    const SparkGlm52SotaRestrictedLogitsArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaMtpMxfp4VerifySm121(
    const SparkGlm52SotaMtpArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaDecodeGraphLaunchOrCaptureSm121(
    SparkGlm52SotaDecodeGraphPlan *plan,
    cudaStream_t stream,
    cudaError_t (*capture_body)(void *context, cudaStream_t stream),
    void *context);

#ifdef __cplusplus
}
#endif

#endif
