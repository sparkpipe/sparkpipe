#pragma once
#include <stdint.h>
#include "sparkpipe/spark_weightd_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

// Startup only. Manifest must be the validated immutable layout of this pack.
// Caller owns an aligned destination of capacity bytes, and must quarantine it
// until success. Failure may leave partial bytes; never publish them. Caller
// controls allocation budget and GPU-use lifetime. Reads preserve fd position.
// SHA256 validates the entire pack, so startup still reads expert bytes from
// disk, but only compact non-expert spans are copied to GPU memory.
SparkStatus SparkWeightdSpineLoad(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity);

#ifdef __cplusplus
}
#endif
