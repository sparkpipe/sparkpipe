#pragma once
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdMap SparkWeightdMap;

// Serialized calls on the creating CUDA context. The borrowed client and its
// lazy attach must outlive the map. Allocation and events are created here.
SparkStatus SparkWeightdMapCreate(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,SparkWeightdMap **out);

const void *SparkWeightdMapEpochDevice(const SparkWeightdMap *map);
// Once teardown starts, errors permit only a Destroy retry, never new work.
SparkStatus SparkWeightdMapDestroy(SparkWeightdMap *map);

// On error a nonzero identifier still requires Release; no GPU use is allowed.
// Acquire maps all chunks before success and exposes no address until BeginUse.
// One deadline covers request/import batches; elapsed time returns BUSY. CUDA
// calls cannot be preempted. Expiry retains a nonzero identifier for cleanup.
SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier,uint64_t timeout);
SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **address);
// The map's stable virtual base: a consumer address for any pack offset is
// base + offset while its covering chunk is committed. Constant after Create.
SparkStatus SparkWeightdMapBase(const SparkWeightdMap *map,void **address);
// Record after every stream using the lease has joined this stream. No further
// work may use this lease after recording completion.
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream);
// Unused leases can be cancelled. Begun leases require a completed event;
// BUSY retains mappings/pins. Unmap/release precedes the daemon RELEASE message.
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout);

// Startup-only spine copy straight from the daemon's verified arena image:
// with the pool mapped, every spine span is device-readable at base + file
// offset, so the compacted spine is assembled device-to-device instead of
// re-reading the pack file (no page-cache dependence). UNSUPPORTED when the
// pool is not mapped - callers fall back to SparkWeightdSpineLoad. The
// destination must stay quarantined until success; a fallback rewrites the
// same span bytes.
SparkStatus SparkWeightdMapSpineCopy(const SparkWeightdMap *map,
    const SparkWeightdManifest *manifest,void *destination,uint64_t capacity);

#ifdef __cplusplus
}
#endif
