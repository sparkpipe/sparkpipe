/* Link stubs for the glm52 validator ORACLE SELFTEST build.
 *
 * The selftest entry (-DSPARK_GLM52_VALIDATOR_ORACLE_SELFTEST) executes only
 * the pure oracle/codec/selection/shaping functions, but the translation
 * unit still compiles the device-fixture tiers, whose (never-executed) code
 * references the module's CUDA launchers and the CUDA runtime. The runtime
 * side comes from tests/cuda_stub/cuda_runtime_stub.c; these four launcher
 * no-ops close the module side so the selftest links on any POSIX host
 * without a GPU or toolkit. A call would abort loudly - reaching one is a
 * selftest bug, not a supported path. */
#include "cuda_runtime.h"

#include "spark_glm52_resident_decode_stage_internal.h"

#include <stdlib.h>

extern "C" int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
{
    (void)multiprocessor_count;
    abort();
}

extern "C" int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave)
{
    (void)wave;
    abort();
}

extern "C" int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave, uint32_t local_layer)
{
    (void)wave;
    (void)local_layer;
    abort();
}

extern "C" int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave, uint32_t local_layer)
{
    (void)wave;
    (void)local_layer;
    abort();
}
