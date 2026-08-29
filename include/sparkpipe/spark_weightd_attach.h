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
 * Cross-process note: `device_handle` names the daemon's pid-local VMM
 * mapping until the POSIX-fd export + consumer import/map tier lands; the
 * identity/refcount/fallback semantics exercised here are exactly what the
 * fd tier rides on. */

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
    uint64_t device_handle;     /* daemon VMM mapping (see the note above) */
    uint64_t arena_bytes;
    uint64_t arena_generation;
    uint32_t loaded_from_pack; /* 1 = cold load happened, 0 = warm hit */
    uint32_t refcount;
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

/* Drop this process's reference: closes the connection WITHOUT detaching —
 * the daemon reaps the dead socket and the arena stays resident warm for
 * the next attach (the sub-second code-redeploy path). The arena bytes are
 * daemon-owned; the consumer never frees them. */
void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome);

#ifdef __cplusplus
}
#endif
