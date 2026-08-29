/* spark_weightd serving-side attach (docs/WEIGHTD_DESIGN.md W2b + W3 fd
 * tier): the consumer helper over the deadline-client. Every gate below
 * falls back — the direct pack load is the contract, the attach is the
 * optimization — and the daemon's digest verification stays the only bytes
 * authority. W3 adds the import/map leg: attached chunks arrive as POSIX
 * shareable fds, are verified against the caller's byte-range claim, and
 * are mapped at the consumer's own virtual span (cuMemImportFromShareable-
 * Handle + cuMemAddressReserve + cuMemMap + cuMemSetAccess RW) — the
 * device_handle stopgap replaced by a real cross-process mapping. */

#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_weightd_attach.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

static void SparkWeightdAttachSetReason(char *reason, const char *text)
{
    if (reason != 0)
    {
        memset(reason, 0, SPARK_WEIGHTD_ATTACH_REASON_BYTES);
        memcpy(reason, text,
            strlen(text) + 1u <= SPARK_WEIGHTD_ATTACH_REASON_BYTES
                ? strlen(text) + 1u
                : SPARK_WEIGHTD_ATTACH_REASON_BYTES);
    }
}

static int SparkWeightdAttachEnvIsOff(const char *name)
{
    const char *text = getenv(name);
    return text != 0 && text[0] == '0' && text[1] == '\0';
}

static const char *SparkWeightdAttachEnvText(const char *name)
{
    const char *text = getenv(name);
    return text != 0 && text[0] != '\0' ? text : 0;
}

SparkStatus SparkWeightdAttachRequested(void)
{
    const char *socket = SparkWeightdAttachEnvText(
        SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0 || SparkWeightdAttachEnvIsOff(
        SPARK_WEIGHTD_ATTACH_ENV_SWITCH) != 0)
    {
        return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdAttachPack(const SparkWeightdPackSlice *slice,
    const char *pack_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdAttachOutcome *outcome,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES])
{
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    const char *socket;
    const char *digest;
    SparkStatus status;

    if (outcome == 0 || slice == 0 || pack_path == 0 ||
        pack_path[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(outcome, 0, sizeof(*outcome));
    if (reason != 0)
    {
        reason[0] = '\0';
    }

    /* gate 1: the kill-switch parity (off wins over anything) */
    if (SparkWeightdAttachEnvIsOff(SPARK_WEIGHTD_ATTACH_ENV_SWITCH) != 0)
    {
        SparkWeightdAttachSetReason(reason, "env_off");
        return SPARK_STATUS_OK;
    }
    /* gate 2: no socket configured — the daemon is absent by deployment */
    socket = SparkWeightdAttachEnvText(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0)
    {
        SparkWeightdAttachSetReason(reason, "no_socket");
        return SPARK_STATUS_OK;
    }
    /* gate 3: the identity is published, never computed per process */
    digest = SparkWeightdAttachEnvText(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
    if (digest == 0)
    {
        SparkWeightdAttachSetReason(reason, "no_identity");
        return SPARK_STATUS_OK;
    }
    memset(&identity, 0, sizeof(identity));
    {
        const char *model = SparkWeightdAttachEnvText(
            SPARK_WEIGHTD_ATTACH_ENV_MODEL);
        const char *revision = SparkWeightdAttachEnvText(
            SPARK_WEIGHTD_ATTACH_ENV_REVISION);
        model = model != 0 ? model : slice->model;
        revision = revision != 0 ? revision : slice->revision;
        revision = revision != 0 ? revision : "";
        if (model == 0)
        {
            SparkWeightdAttachSetReason(reason, "no_identity");
            return SPARK_STATUS_OK;
        }
        memcpy(identity.model, model,
            strlen(model) + 1u <= SPARK_WEIGHTD_ID_BYTES
                ? strlen(model) + 1u
                : SPARK_WEIGHTD_ID_BYTES);
        memcpy(identity.revision, revision,
            strlen(revision) + 1u <= SPARK_WEIGHTD_REVISION_BYTES
                ? strlen(revision) + 1u
                : SPARK_WEIGHTD_REVISION_BYTES);
    }
    memcpy(identity.pack_sha256, digest,
        strlen(digest) + 1u <= SPARK_WEIGHTD_SHA256_HEX_BYTES
            ? strlen(digest) + 1u
            : SPARK_WEIGHTD_SHA256_HEX_BYTES);
    identity.topology = slice->topology;
    identity.geometry_fingerprint = slice->geometry_fingerprint;
    identity.arena_bytes = slice->pack_bytes;
    identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    if (SparkWeightdIdentityPrepare(&identity) != SPARK_STATUS_OK)
    {
        SparkWeightdAttachSetReason(reason, "identity");
        return SPARK_STATUS_OK;
    }

    /* the deadline-client: a refused/missing daemon fails the connect
     * closed at the front door — that IS the fallback signal */
    status = SparkWeightdClientConnect(socket, &outcome->client, 0);
    if (status != SPARK_STATUS_OK)
    {
        outcome->client = 0;
        SparkWeightdAttachSetReason(reason, "no_daemon");
        return SPARK_STATUS_OK;
    }
    memset(&request, 0, sizeof(request));
    request.identity = identity;
    memcpy(request.pack_path, pack_path,
        strlen(pack_path) + 1u <= SPARK_WEIGHTD_PATH_BYTES
            ? strlen(pack_path) + 1u
            : SPARK_WEIGHTD_PATH_BYTES);
    memset(&result, 0, sizeof(result));
    status = SparkWeightdClientAttach(outcome->client, &request, &result,
        timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        /* transport fault mid-exchange: deadline expiry (busy) or the
         * daemon dying under us (io_error) — fail closed to the fallback */
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, SparkStatusToString(status));
        return SPARK_STATUS_OK;
    }
    if (result.status != SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason,
            SparkStatusToString(result.status));
        return SPARK_STATUS_OK;
    }
    outcome->device_handle = result.device_handle;
    outcome->arena_bytes = result.arena_bytes;
    outcome->arena_generation = result.arena_generation;
    outcome->loaded_from_pack = result.loaded_from_pack;
    outcome->refcount = result.refcount;
    return SPARK_STATUS_OK;
}

/* ------------------------------ W3 fd tier: import + map ------------------------------ */

/* A received export batch's fds are the receiver's until each one's import
 * takes over — every exit path must drain exactly the fds still open. */
static void SparkWeightdAttachCloseBatchFds(SparkWeightdExportBatch *batch)
{
    uint32_t index;
    for (index = 0u; index < batch->batch_count; index++)
    {
        (void)close(batch->fds[index]);
    }
    batch->batch_count = 0u;
}

/* The map's access descriptor names the caller's current device, exactly as
 * the daemon names its own when it builds the arena. The cudaFree(0) first
 * is the standard lazy-context bootstrap: a fresh consumer process has NO
 * current context until the runtime makes its primary context, and the
 * driver-side import/map calls below need it current right now (a no-op
 * where a context already exists — including on the host stub). */
static int SparkWeightdAttachDeviceId(void)
{
    int device = 0;
    (void)cudaFree(0);
    (void)cudaGetDevice(&device);
    return device;
}

/* Tear down the consumer-side map state only (unmap mapped chunks, release
 * imported handles, free the reservation and the handle array). The driver
 * refuses out-of-order releases, so unmap-before-release is load-bearing.
 * Daemon-side state is untouched here — the client close is Release's job. */
static void SparkWeightdAttachMapUndo(SparkWeightdAttachOutcome *outcome)
{
    if (outcome->map_base != 0)
    {
        CUdeviceptr base = (CUdeviceptr)(uintptr_t)outcome->map_base;
        uint32_t index;
        for (index = 0u; index < outcome->map_mapped_count; index++)
        {
            (void)cuMemUnmap(base + (CUdeviceptr)index *
                    outcome->map_chunk_bytes,
                (size_t)outcome->map_chunk_bytes);
        }
        for (index = 0u; index < outcome->map_handle_count; index++)
        {
            (void)cuMemRelease(
                (CUmemGenericAllocationHandle)outcome->map_handles[index]);
        }
        (void)cuMemAddressFree(base, (size_t)outcome->map_span_bytes);
    }
    free(outcome->map_handles);
    outcome->map_handles = 0;
    outcome->map_base = 0;
    outcome->map_span_bytes = 0ull;
    outcome->map_chunk_bytes = 0ull;
    outcome->map_chunk_count = 0u;
    outcome->map_handle_count = 0u;
    outcome->map_mapped_count = 0u;
}

SparkStatus SparkWeightdAttachImportMap(SparkWeightdAttachOutcome *outcome,
    uint64_t expected_arena_bytes,
    uint64_t timeout_nanoseconds,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES])
{
    SparkWeightdExportBatch batch;
    uint64_t chunk_bytes = 0ull;
    uint64_t covered_bytes = 0ull;
    uint32_t chunk_count = 0u;
    uint32_t batch_offset = 0u;
    CUmemAccessDesc access;
    CUdeviceptr base = 0;

    if (outcome == 0 || reason == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (outcome->client == 0)
    {
        SparkWeightdAttachSetReason(reason, "not_attached");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (outcome->map_base != 0)
    {
        SparkWeightdAttachSetReason(reason, "already_mapped");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&batch, 0, sizeof(batch));

    /* batch 0 reveals the arena's chunk geometry; the daemon's verdict (e.g.
     * NOT_FOUND for a detached generation) is a completed exchange here */
    if (SparkWeightdClientExportBatch(outcome->client,
            outcome->arena_generation, 0u, &batch, timeout_nanoseconds) !=
        SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_exchange");
        return SPARK_STATUS_OK;
    }
    if (batch.status != SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, SparkStatusToString(batch.status));
        return SPARK_STATUS_OK;
    }

    /* THE IDENTITY CHECK, before anything is imported into a mapping: the
     * chunk set must COVER the caller's expected byte range (the pack size
     * the identity pins), and the geometry must be multipliable and inside
     * the map bound — anything else is a lying/broken frame, never mapped. */
    chunk_bytes = batch.chunk_bytes;
    chunk_count = batch.chunk_count;
    if (chunk_bytes == 0ull || chunk_count == 0u ||
        chunk_count > SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX ||
        chunk_bytes > UINT64_MAX / (uint64_t)chunk_count)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_shape");
        return SPARK_STATUS_OK;
    }
    covered_bytes = chunk_bytes * (uint64_t)chunk_count;
    if (covered_bytes < expected_arena_bytes)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_short");
        return SPARK_STATUS_OK;
    }

    outcome->map_handles = (void **)calloc(chunk_count, sizeof(void *));
    if (outcome->map_handles == 0)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_shape");
        return SPARK_STATUS_OK;
    }
    outcome->map_chunk_bytes = chunk_bytes;
    outcome->map_chunk_count = chunk_count;

    /* receive + import every batch; the fd set tiles [0, chunk_count)
     * exactly or the whole thing unwinds. Each fd is the caller's until its
     * import succeeds — closed immediately after (the import holds the
     * driver's own reference). */
    while (batch_offset < chunk_count)
    {
        uint32_t index;
        if (batch.batch_count == 0u)
        {
            /* first iteration reuses batch 0; later iterations fetch here */
            memset(&batch, 0, sizeof(batch));
            if (SparkWeightdClientExportBatch(outcome->client,
                    outcome->arena_generation, batch_offset, &batch,
                    timeout_nanoseconds) != SPARK_STATUS_OK)
            {
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_exchange");
                return SPARK_STATUS_OK;
            }
        }
        if (batch.status != SPARK_STATUS_OK ||
            batch.batch_offset != batch_offset ||
            batch.chunk_bytes != chunk_bytes ||
            batch.chunk_count != chunk_count ||
            batch.batch_count == 0u ||
            batch_offset > chunk_count - batch.batch_count)
        {
            SparkWeightdAttachCloseBatchFds(&batch);
            SparkWeightdAttachMapUndo(outcome);
            SparkWeightdAttachRelease(outcome);
            SparkWeightdAttachSetReason(reason, "import_shape");
            return SPARK_STATUS_OK;
        }
        for (index = 0u; index < batch.batch_count; index++)
        {
            CUmemGenericAllocationHandle handle = 0;
            CUresult import_rc;
            /* osHandle carries the POSIX fd BY VALUE (cuda.h: "Shareable
             * Handle representing the memory allocation") - the real driver
             * reads the pointer's integer value as the fd; passing the
             * fd's ADDRESS fed it a stack address as an fd number and every
             * import failed CUDA_ERROR_INVALID_HANDLE (GPU receipt,
             * cell-runner 2026-08-29: reason=import_handle on real HW). */
            import_rc = cuMemImportFromShareableHandle(&handle,
                    (void *)(uintptr_t)batch.fds[index],
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
            if (import_rc != CUDA_SUCCESS)
            {
                /* the reason code names the stage; this names the driver
                 * error (receipt instrumentation, SPARK_WEIGHTD_IMPORT_DIAG) */
                if (getenv("SPARK_WEIGHTD_IMPORT_DIAG") != 0)
                    fprintf(stderr,
                        "weightd import diag: fd=%d curesult=%d\n",
                        batch.fds[index], (int)import_rc);
                (void)close(batch.fds[index]);
                SparkWeightdAttachCloseBatchFds(&batch);
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_handle");
                return SPARK_STATUS_OK;
            }
            (void)close(batch.fds[index]);
            outcome->map_handles[outcome->map_handle_count] = (void *)handle;
            outcome->map_handle_count++;
        }
        batch_offset += batch.batch_count;
        memset(&batch, 0, sizeof(batch));
    }

    /* the verified chunk set is complete: now — and only now — the
     * consumer's own span. The VA is the consumer's process-lifetime
     * mapping of the daemon's stable arena (docs/WEIGHTD_DESIGN.md). */
    outcome->map_span_bytes = covered_bytes;
    if (cuMemAddressReserve(&base, (size_t)covered_bytes, 0u, 0ull, 0ull) !=
        CUDA_SUCCESS)
    {
        SparkWeightdAttachMapUndo(outcome);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_map");
        return SPARK_STATUS_OK;
    }
    outcome->map_base = (void *)(uintptr_t)base;
    {
        uint32_t index;
        for (index = 0u; index < chunk_count; index++)
        {
            if (cuMemMap(base + (CUdeviceptr)index * chunk_bytes,
                    (size_t)chunk_bytes, 0u,
                    (CUmemGenericAllocationHandle)outcome->map_handles[index],
                    0ull) != CUDA_SUCCESS)
            {
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_map");
                return SPARK_STATUS_OK;
            }
            outcome->map_mapped_count++;
        }
    }
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = SparkWeightdAttachDeviceId();
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(base, (size_t)covered_bytes, &access, 1u) !=
        CUDA_SUCCESS)
    {
        SparkWeightdAttachMapUndo(outcome);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_access");
        return SPARK_STATUS_OK;
    }

    outcome->device_handle = (uint64_t)(uintptr_t)base;
    SparkWeightdAttachSetReason(reason, "");
    return SPARK_STATUS_OK;
}

void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome)
{
    if (outcome == 0)
    {
        return;
    }
    if (outcome->map_base != 0 && outcome->map_handles != 0)
    {
        /* W3 teardown: free the CONSUMER-side map first (exact unmap pairs,
         * then every imported handle, then the reservation — the driver
         * refuses releases out of order, so this ordering is load-bearing),
         * all without touching the daemon: the attach reference rides the
         * client close below, and the arena itself stays warm. */
        CUdeviceptr base = (CUdeviceptr)(uintptr_t)outcome->map_base;
        uint32_t index;
        for (index = 0u; index < outcome->map_mapped_count; index++)
        {
            (void)cuMemUnmap(base + (CUdeviceptr)index *
                    outcome->map_chunk_bytes,
                (size_t)outcome->map_chunk_bytes);
        }
        for (index = 0u; index < outcome->map_handle_count; index++)
        {
            (void)cuMemRelease(
                (CUmemGenericAllocationHandle)outcome->map_handles[index]);
        }
        (void)cuMemAddressFree(base, (size_t)outcome->map_span_bytes);
    }
    free(outcome->map_handles);
    outcome->map_handles = 0;
    outcome->map_base = 0;
    outcome->map_span_bytes = 0ull;
    outcome->map_chunk_bytes = 0ull;
    outcome->map_chunk_count = 0u;
    outcome->map_handle_count = 0u;
    outcome->map_mapped_count = 0u;
    if (outcome->client != 0)
    {
        /* close WITHOUT detaching: the daemon reaps the dead socket, drops
         * this process's reference, and the arena stays warm — the crash
         * path and the clean teardown are the same code */
        SparkWeightdClientClose(outcome->client);
        outcome->client = 0;
    }
    outcome->device_handle = 0ull;
    outcome->arena_bytes = 0ull;
    outcome->arena_generation = 0ull;
    outcome->loaded_from_pack = 0u;
    outcome->refcount = 0u;
}
