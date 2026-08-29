#pragma once

/* spark_weightd serving-side attach (docs/WEIGHTD_DESIGN.md W2b): the
 * consumer surface a module load path uses to request an arena BY IDENTITY
 * from a running weightd, with kill-switch parity to the W1 pipeline env-off
 * pattern and an unconditional fallback contract — every "not attached"
 * outcome (env off, socket unset, daemon absent, deadline, daemon refusal)
 * returns cleanly so the caller proceeds with its direct pack load. The
 * daemon stays the digest authority: it verifies the pack bytes before any
 * arena exists, so a stale or lying deployment identity can only fall back,
 * never serve wrong bytes.
 *
 * The identity is published by the deployment, not computed per process:
 * a consumer-side full-pack hash would re-read the bytes the daemon exists
 * to stop re-reading (the warm code-redeploy attach must stay sub-second).
 * SPARK_WEIGHTD_PACK_SHA256 carries the pack's content digest; without it
 * there is no identity, and the helper declines (direct load).
 *
 * Cross-process truth (W3 fd tier): a raw attach result's `device_handle`
 * names the daemon's pid-local VMM span. The consumer's OWN mapping is built
 * by SparkWeightdAttachImportMap: the daemon exports each physical chunk as
 * a CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR shareable fd (SCM_RIGHTS,
 * batched), the helper imports every fd, verifies the chunk set covers the
 * expected byte range, reserves the consumer's own virtual span, maps every
 * imported chunk at its pack offset and makes the map read-write with
 * cuMemSetAccess. Outcome.device_handle becomes that consumer-local base —
 * borrowed-slice binding needs no further change. Release unmaps WITHOUT
 * detaching: the daemon reaps the socket, keeps the refcount book honest,
 * and the arena stays warm. */

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The environment contract (kill-switch parity with
 * SPARK_STAGE_MODULE_LOAD_PIPELINE=0):
 * - SPARK_WEIGHTD_SOCKET set non-empty: attach is REQUESTED (absent/empty:
 *   not requested — today's direct-load behavior, the default).
 * - SPARK_WEIGHTD_ATTACH=0: attach is OFF even when the socket is set.
 * - SPARK_WEIGHTD_IDENTITY_MODEL / SPARK_WEIGHTD_IDENTITY_REVISION: override
 *   the caller's family default (the daemon keys arenas by these).
 * - SPARK_WEIGHTD_PACK_SHA256: the pack's content digest (64 lowercase hex);
 *   REQUIRED for an attach — without it the helper declines. */
#define SPARK_WEIGHTD_ATTACH_ENV_SOCKET "SPARK_WEIGHTD_SOCKET"
#define SPARK_WEIGHTD_ATTACH_ENV_SWITCH "SPARK_WEIGHTD_ATTACH"
#define SPARK_WEIGHTD_ATTACH_ENV_MODEL "SPARK_WEIGHTD_IDENTITY_MODEL"
#define SPARK_WEIGHTD_ATTACH_ENV_REVISION "SPARK_WEIGHTD_IDENTITY_REVISION"
#define SPARK_WEIGHTD_ATTACH_ENV_SHA256 "SPARK_WEIGHTD_PACK_SHA256"

/* Default attach deadline: an attach that reaches a live daemon may have to
 * wait out the daemon's cold load (verify + read + copy of the whole pack,
 * ~20 s per 100 GB of NVMe), so the deadline must exceed it. */
#define SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS 120000000000ull

/* Fixed fallback-reason buffer (SparkStatusToString names the daemon
 * refusals; the helper names its own gates). */
#define SPARK_WEIGHTD_ATTACH_REASON_BYTES 32u

typedef struct SparkWeightdPackSlice
{
    const char *model;    /* the family's name (the caller's default) */
    const char *revision; /* deployment revision; may be empty */
    uint32_t topology;    /* slice topology identity (e.g. the tp hash) */
    uint64_t geometry_fingerprint; /* fold of the pack header geometry */
    uint64_t pack_bytes;  /* the pack size claim (= identity arena_bytes) */
} SparkWeightdPackSlice;

typedef struct SparkWeightdAttachOutcome
{
    SparkWeightdClient *client; /* non-null iff attached; close at teardown */
    uint64_t device_handle;     /* after ImportMap: THE CONSUMER'S mapped
                                 * base; before it, the daemon-local span
                                 * (see the cross-process note above) */
    uint64_t arena_bytes;
    uint64_t arena_generation;
    uint32_t loaded_from_pack; /* 1 = cold load happened, 0 = warm hit */
    uint32_t refcount;
    /* W3 fd-tier consumer map (owned by Release; zero while unmapped) */
    void *map_base;             /* the consumer's reserved span base */
    uint64_t map_span_bytes;    /* chunk_bytes * chunk_count */
    uint64_t map_chunk_bytes;   /* one physical chunk (arena granularity) */
    void **map_handles;         /* imported physical handles, pack order */
    uint32_t map_chunk_count;   /* chunks covering the arena */
    uint32_t map_handle_count;  /* imported so far (== count when mapped) */
    uint32_t map_mapped_count;  /* currently mapped at map_base */
} SparkWeightdAttachOutcome;

/* OK when an attach should be attempted, BUSY when the kill switch or the
 * absent socket turns it off (the SparkStageModuleLoadPipelineRequested
 * convention). */
SparkStatus SparkWeightdAttachRequested(void);

/* One-call attach attempt. Returns OK with outcome->client == 0 whenever the
 * consumer should use its direct pack load — `reason` (bounded) names the
 * gate: "no_socket", "env_off", "no_identity", "identity", "no_daemon", or
 * the daemon refusal's status name (HASH_MISMATCH, ...). A non-OK RETURN is
 * a transport/argument fault, not a fallback. Always pair a non-null
 * outcome->client with SparkWeightdAttachRelease. */
SparkStatus SparkWeightdAttachPack(const SparkWeightdPackSlice *slice,
    const char *pack_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdAttachOutcome *outcome,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES]);

/* W3 fd tier: turn the attached arena into a CONSUMER-local mapping. Asks
 * the daemon for the arena's chunk shareable fds (batch by batch), verifies
 * the chunk set covers `expected_arena_bytes` — the CALLER's byte-range
 * claim (the pack size the identity pins), not the daemon's echo — BEFORE
 * anything is mapped, imports every fd, reserves the consumer's own virtual
 * span, maps each chunk at its pack offset and sets the map read-write with
 * cuMemSetAccess. On success outcome->device_handle == outcome->map_base is
 * valid for the process lifetime. Every "not mapped" outcome returns OK with
 * a reason and the attach RELEASED (client closed, nothing detached — the
 * daemon keeps the arena warm): "import_shape" (a frame whose chunk geometry
 * cannot be trusted), "import_short" (the chunk set does not cover the
 * expected byte range — the identity check), "import_handle" (an fd refused
 * by cuMemImportFromShareableHandle), "import_map" (reserve/map failed),
 * "import_access" (cuMemSetAccess failed), or the exchange fault's status
 * name. A non-OK RETURN is an argument fault (no attach to map, or already
 * mapped) and leaves the outcome untouched. */
SparkStatus SparkWeightdAttachImportMap(SparkWeightdAttachOutcome *outcome,
    uint64_t expected_arena_bytes,
    uint64_t timeout_nanoseconds,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES]);

/* Drop this process's reference: first unmaps + tears down the consumer's
 * imported map when one exists (unmap every chunk, release every imported
 * handle, free the reservation — nothing daemon-side changes), then closes
 * the connection WITHOUT detaching — the daemon reaps the dead socket and
 * the arena stays resident warm for the next attach (the sub-second
 * code-redeploy path). The arena bytes are daemon-owned; the consumer never
 * frees them. */
void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome);

#ifdef __cplusplus
}
#endif
