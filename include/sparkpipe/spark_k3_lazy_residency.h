#ifndef SPARKPIPE_SPARK_K3_LAZY_RESIDENCY_H
#define SPARKPIPE_SPARK_K3_LAZY_RESIDENCY_H

/* The k3 side of the shared lazy-expert residency (runtime/spark_weightd*).
 * Compiled as C: the weightd headers are C11 and this unit is the only
 * k3 file that includes them. The runner (nvcc, C++) sees plain C
 * functions and opaque handles.
 *
 * Contract (fail-closed, mirrors docs/GLM_LAZY_DRIVER_INTEGRATION.md):
 *   - SparkK3LazyStartup is startup-only and strict: a missing daemon,
 *     missing/invalid manifest, identity mismatch or insufficient budget
 *     fails with the exact status. There is no eager fallback; callers
 *     that require lazy treat any failure as fatal.
 *   - SparkK3LazySlice exposes a non-expert pack range from the compact
 *     spine (host memory; readable by k3 kernels through coherence).
 *   - SparkK3LazyExpertAcquire leases one routed layer's working set
 *     (keyed by the completed host routing offsets), binds the layer's
 *     expert tensor pointers against the consumer-local lease address,
 *     and reports them. SparkK3LazyExpertRelease records completion on
 *     the submission stream and releases the lease; a failed release
 *     retains the lease for recovery and reports the status.
 */
#include <stdint.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_k3_pack_load.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkK3LazyResidency SparkK3LazyResidency;

/* Manifest checker context: validates the manifest against the pack's
 * own entries (every routed layer, both tensors, all experts, exact
 * spans and kinds). */
typedef struct SparkK3LazyResidencyStartup
{
	const char *socket_path;
	const char *pack_path;
	const char *pack_sha256_hex; /* 64 hex chars */
	const char *model;           /* identity model, e.g. "kimi-k3" */
	const char *revision;        /* identity revision */
	uint32_t tp_degree;
	uint64_t expert_pool_bytes;
	uint64_t spine_budget_bytes;
	uint64_t timeout_ns;
} SparkK3LazyResidencyStartup;

SparkStatus SparkK3LazyResidencyStartup(SparkK3LazyResidencyStartup startup,
	SparkK3Pack *pack, SparkK3LazyResidency **out);

/* Non-expert pack range from the compact spine (host pointer). */
SparkStatus SparkK3LazySlice(SparkK3LazyResidency *residency,
	uint64_t pack_offset, uint64_t bytes, const void **pointer);

/* Per-layer working set: keys from the completed host routing offsets,
 * acquisition, BeginUse, and the bound expert tensor pointers for this
 * layer. lease_out identifies the lease for the release call. */
SparkStatus SparkK3LazyExpertAcquire(SparkK3LazyResidency *residency,
	uint32_t layer, uint32_t packed_rows,
	const uint32_t *group_row_offset, cudaStream_t stream,
	uint64_t w1_offset, uint64_t w2_offset,
	const void **w1, const void **w2, uint64_t *lease);

/* Records completion on the submission stream and releases the lease.
 * A failed release retains the lease (recovery at destroy) and returns
 * the status. */
SparkStatus SparkK3LazyExpertRelease(SparkK3LazyResidency *residency,
	uint64_t lease, cudaStream_t stream);

void SparkK3LazyResidencyDestroy(SparkK3LazyResidency *residency);

#ifdef __cplusplus
}
#endif

#endif
